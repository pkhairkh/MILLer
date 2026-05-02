//! Shard descriptor types and sharded pipeline construction.
//!
//! This module contains the shard-related types and functions for building
//! sharded linear pipelines: shard descriptors, per-shard MIR lowering,
//! per-shard bridge payloads, and the sharded PIR graph builder.

use crate::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirNodeId, MirOp};
use crate::pir::{
    FunctionEntry, Handoff, Package, PackageRole, PirGraph, ShardPartitionEntry, ShardRole,
    ShardTemplate, TensorSpec as PirTensorSpec,
};
use crate::sir::KvCacheLayout;
use crate::task_spec::{SyntheticTaskSpec, TaskOp};

use super::payload::{BRIDGE_VERSION, FunctionDescriptor, TensorDescriptor};

/// Description of a single shard within a sharded pipeline.
///
/// Each shard has a role (Entry, Interior, Exit), its own input/output
/// dimensions, a shard name used for the output mlpackage directory,
/// and the compute units appropriate for its role.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardDesc {
    /// Shard role: Entry, Interior, or Exit.
    pub role: ShardRole,
    /// Shard name (e.g., "entry_shard", "interior_shard", "exit_shard").
    pub shard_name: String,
    /// Input dimension for this shard's linear projection.
    pub input_dim: usize,
    /// Output dimension for this shard's linear projection.
    pub output_dim: usize,
    /// Compute units for this shard (ANE-targeted for decoder shards).
    pub compute_units: ComputeUnitHint,
}

/// Produce the shard descriptors for a ShardedLinearPipeline task.
///
/// The pipeline composes three shards:
/// - Entry:     [batch, input_dim]  -> [batch, hidden_dim]   (CPU_AND_NE)
/// - Interior:  [batch, hidden_dim] -> [batch, hidden_dim]   (CPU_AND_NE)
/// - Exit:      [batch, hidden_dim] -> [batch, output_dim]   (CPU_AND_NE)
///
/// This mirrors the Qwen3 three-shard decomposition at a micro scale.
pub fn sharded_pipeline_shards(spec: &SyntheticTaskSpec) -> Result<Vec<ShardDesc>, String> {
    let (input_dim, hidden_dim, output_dim, _batch_size, _dtype) = match &spec.op {
        TaskOp::ShardedLinearPipeline { input_dim, hidden_dim, output_dim, batch_size, dtype } => {
            (*input_dim, *hidden_dim, *output_dim, *batch_size, dtype.clone())
        }
        _ => return Err("Expected ShardedLinearPipeline task".into()),
    };

    Ok(vec![
        ShardDesc {
            role: ShardRole::Entry,
            shard_name: format!("{}_entry", spec.name),
            input_dim,
            output_dim: hidden_dim,
            compute_units: ShardRole::Entry.default_compute_units(),
        },
        ShardDesc {
            role: ShardRole::Interior,
            shard_name: format!("{}_interior", spec.name),
            input_dim: hidden_dim,
            output_dim: hidden_dim,
            compute_units: ShardRole::Interior.default_compute_units(),
        },
        ShardDesc {
            role: ShardRole::Exit,
            shard_name: format!("{}_exit", spec.name),
            input_dim: hidden_dim,
            output_dim,
            compute_units: ShardRole::Exit.default_compute_units(),
        },
    ])
}

/// Build a MIR graph for one shard of a sharded linear pipeline.
///
/// Each shard is a simple linear projection (matmul + bias add),
/// identical in structure to the single-shard path but with its
/// own dimensions and shard name.
pub fn lower_shard_to_mir(
    shard: &ShardDesc,
    batch_size: usize,
    dtype: &str,
) -> Result<MirGraph, String> {
    let mil_dtype = match dtype {
        "fp16" => MilDtype::Fp16,
        "fp32" => MilDtype::Fp32,
        _ => MilDtype::Fp16,
    };

    // Sprint 58 (S58.3): inline conversion removed — compute_units is now
    // ComputeUnitHint directly, so no conversion needed.
    let compute_hint = shard.compute_units.clone();

    let weight_id = MirNodeId("weight".into());
    let bias_id = MirNodeId("bias".into());
    let input_id = MirNodeId("input".into());
    let matmul_id = MirNodeId("matmul".into());
    let add_id = MirNodeId("add".into());

    let nodes = vec![
        MirNode {
            id: weight_id.clone(),
            op: MirOp::MILConst {
                name: "weight".into(),
                value_path: "weight.npy".into(),
                dtype: mil_dtype.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![shard.input_dim, shard.output_dim],
            compute_unit_hint: None,
            air_source: None,
        },
        MirNode {
            id: bias_id.clone(),
            op: MirOp::MILConst {
                name: "bias".into(),
                value_path: "bias.npy".into(),
                dtype: mil_dtype.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![shard.output_dim],
            compute_unit_hint: None,
            air_source: None,
        },
        MirNode {
            id: matmul_id.clone(),
            op: MirOp::MILMatMul {
                name: "matmul".into(),
                x: input_id.clone(),
                y: weight_id.clone(),
                transpose_y: false,
            },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, shard.output_dim],
            compute_unit_hint: Some(compute_hint.clone()),
            air_source: None,
        },
        MirNode {
            id: add_id.clone(),
            op: MirOp::MILAdd { name: "add".into(), x: matmul_id.clone(), y: bias_id.clone() },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, shard.output_dim],
            compute_unit_hint: Some(compute_hint),
            air_source: None,
        },
    ];

    Ok(MirGraph {
        nodes,
        inputs: vec![input_id],
        outputs: vec![add_id],
        opset_version: crate::DEFAULT_OPSET_VERSION.into(),
        shard_name: shard.shard_name.clone(),
        input_shapes: std::collections::HashMap::new(),
    })
}

/// Bridge payload for one shard of a sharded pipeline.
///
/// Each shard gets its own payload with role metadata, allowing
/// the Python emitter and downstream manifests to reflect the
/// shard's role semantics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardedShardPayload {
    pub bridge_version: u32,
    pub command: String,
    pub task_name: String,
    pub family: String,
    pub shard_name: String,
    /// Shard role: "Entry", "Interior", or "Exit".
    pub shard_role: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    pub functions: Vec<FunctionDescriptor>,
}

impl ShardedShardPayload {
    /// Build a bridge payload for one shard.
    pub fn from_shard(
        shard: &ShardDesc,
        task_name: &str,
        family: &str,
        batch_size: usize,
        dtype: &str,
        output_path: &str,
        seed: u64,
    ) -> Self {
        Self::from_shard_with_override(
            shard,
            task_name,
            family,
            batch_size,
            dtype,
            output_path,
            seed,
            None,
        )
    }

    /// Build a bridge payload for one shard with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This ensures precision adaptations
    /// propagate to the emitted mlpackage per shard.
    #[allow(clippy::too_many_arguments)]
    pub fn from_shard_with_override(
        shard: &ShardDesc,
        task_name: &str,
        family: &str,
        batch_size: usize,
        dtype: &str,
        output_path: &str,
        seed: u64,
        dtype_override: Option<&str>,
    ) -> Self {
        let effective_dtype = dtype_override.unwrap_or(dtype);
        let compute_units_str = shard.compute_units.to_coreml_string();
        Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_linear_projection".into(),
            task_name: task_name.into(),
            family: family.into(),
            shard_name: shard.shard_name.clone(),
            shard_role: shard.role.canonical_name().to_string(),
            input_dim: shard.input_dim,
            output_dim: shard.output_dim,
            batch_size,
            dtype: effective_dtype.into(),
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: compute_units_str.into(),
            output_path: output_path.into(),
            seed,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "x".into(),
                    shape: vec![batch_size, shard.input_dim],
                    dtype: effective_dtype.into(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, shard.output_dim],
                    dtype: effective_dtype.into(),
                }],
                stateful: false,
            }],
        }
    }

    /// Build a bridge payload for one decode-step shard with role-sensitive emission.
    ///
    /// This uses the `emit_shard_decode_step` bridge command instead of
    /// `emit_linear_projection`, ensuring that each shard role produces a
    /// structurally different MIL program (different dimensions, head counts,
    /// and KV cache state shapes). This closes the Sprint 37 gap where
    /// "shard emission is still too uniform until shard role materially
    /// changes emitted graphs and/or dimensions."
    ///
    /// The payload includes decode-step-specific dimensions (embed_dim,
    /// num_heads, head_dim, kv_len) and passes shard_role so the Python
    /// emitter can vary the program structure by role.
    #[allow(clippy::too_many_arguments)]
    pub fn from_shard_decode_step(
        shard: &ShardDesc,
        task_name: &str,
        family: &str,
        batch_size: usize,
        dtype: &str,
        output_path: &str,
        seed: u64,
        dtype_override: Option<&str>,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Self {
        let effective_dtype = dtype_override.unwrap_or(dtype);
        let compute_units_str = shard.compute_units.to_coreml_string();
        Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_shard_decode_step".into(),
            task_name: task_name.into(),
            family: family.into(),
            shard_name: shard.shard_name.clone(),
            shard_role: shard.role.canonical_name().to_string(),
            input_dim: shard.input_dim,
            output_dim: shard.output_dim,
            batch_size,
            dtype: effective_dtype.into(),
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: compute_units_str.into(),
            output_path: output_path.into(),
            seed,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![
                    TensorDescriptor {
                        name: "x".into(),
                        shape: vec![batch_size, embed_dim],
                        dtype: effective_dtype.into(),
                    },
                    TensorDescriptor {
                        name: "k_state".into(),
                        shape: vec![1, num_heads, kv_len, head_dim],
                        dtype: effective_dtype.into(),
                    },
                    TensorDescriptor {
                        name: "v_state".into(),
                        shape: vec![1, num_heads, kv_len, head_dim],
                        dtype: effective_dtype.into(),
                    },
                ],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, shard.output_dim],
                    dtype: effective_dtype.into(),
                }],
                stateful: true,
            }],
        }
    }
}

/// Build a PIR graph for a sharded linear pipeline.
///
/// The PIR captures the full deployment structure: three decoder shard
/// packages with Entry/Interior/Exit roles, inter-shard handoffs, and
/// a shard template reference.
pub fn build_sharded_pipeline_pir(spec: &SyntheticTaskSpec) -> Result<PirGraph, String> {
    let (_input_dim, hidden_dim, _output_dim, _batch_size, _dtype) = match &spec.op {
        TaskOp::ShardedLinearPipeline { input_dim, hidden_dim, output_dim, batch_size, dtype } => {
            (*input_dim, *hidden_dim, *output_dim, *batch_size, dtype.clone())
        }
        _ => return Err("Expected ShardedLinearPipeline task".into()),
    };

    let shards = sharded_pipeline_shards(spec)?;

    let packages: Vec<Package> = shards
        .iter()
        .map(|shard| Package {
            name: shard.shard_name.clone(),
            role: PackageRole::DecoderShard(shard.role.clone()),
            compute_units: shard.compute_units.clone(),
            mil_program_ref: shard.shard_name.clone(),
            functions: vec![FunctionEntry {
                name: "main".into(),
                inputs: vec![PirTensorSpec {
                    name: "x".into(),
                    shape: vec![1, shard.input_dim],
                    dtype: "fp16".into(),
                }],
                outputs: vec![PirTensorSpec {
                    name: "output".into(),
                    shape: vec![1, shard.output_dim],
                    dtype: "fp16".into(),
                }],
                stateful: false,
            }],
        })
        .collect();

    // Build handoffs: entry -> interior -> exit
    // Each handoff carries concrete runtime semantics:
    // - execution_order defines the pipeline sequence
    // - source_output_name/target_input_name link to function I/O
    // - handoff_kind captures the mechanism (direct pass-through)
    let handoffs = vec![
        Handoff {
            from_package: format!("{}_entry", spec.name),
            to_package: format!("{}_interior", spec.name),
            tensor_name: "output".into(),
            shape: vec![1, hidden_dim],
            dtype: "fp16".into(),
            handoff_kind: crate::pir::HandoffKind::TensorPassThrough,
            execution_order: 0,
            source_output_name: "output".into(),
            target_input_name: "x".into(),
        },
        Handoff {
            from_package: format!("{}_interior", spec.name),
            to_package: format!("{}_exit", spec.name),
            tensor_name: "output".into(),
            shape: vec![1, hidden_dim],
            dtype: "fp16".into(),
            handoff_kind: crate::pir::HandoffKind::TensorPassThrough,
            execution_order: 1,
            source_output_name: "output".into(),
            target_input_name: "x".into(),
        },
    ];

    // Shard template describing the three-shard decomposition
    let shard_template = ShardTemplate {
        template_id: format!("{}_3shard_template", spec.name),
        partition_spec: vec![
            ShardPartitionEntry {
                role: ShardRole::Entry,
                layer_start: 0,
                layer_end: 0, // synthetic task has no real layers
                compute_units: ComputeUnitHint::CPUAndNE,
            },
            ShardPartitionEntry {
                role: ShardRole::Interior,
                layer_start: 1,
                layer_end: 1,
                compute_units: ComputeUnitHint::CPUAndNE,
            },
            ShardPartitionEntry {
                role: ShardRole::Exit,
                layer_start: 2,
                layer_end: 2,
                compute_units: ComputeUnitHint::CPUAndNE,
            },
        ],
        io_compute_units: None,      // No IO model in this synthetic task
        sampler_compute_units: None, // No sampler in this synthetic task
        state_config: None,          // No state in linear projection
        context_length: 0,
    };

    Ok(PirGraph {
        packages,
        state_declarations: vec![],
        handoffs,
        shard_template: Some(shard_template),
        context_length: 0,
        opset_version: crate::DEFAULT_OPSET_VERSION.into(),
        minimum_deployment_target: crate::DEFAULT_OPSET_VERSION.into(),
        kv_cache_layout: KvCacheLayout::default(),
        sampler_spec: None,
        io_model_spec: None,
    })
}
