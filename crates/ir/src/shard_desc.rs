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

use super::payload::{FunctionDescriptor, TensorDescriptor, BRIDGE_VERSION};

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
    // V-011: Previously defaulted to Fp16 for unrecognized dtype strings, which
    // silently produced wrong precision for typos or unsupported types like "int8".
    // Now returns an explicit error listing valid options.
    let mil_dtype = match dtype {
        "fp16" => MilDtype::Fp16,
        "fp32" => MilDtype::Fp32,
        "int4" => MilDtype::Int4,
        "int8" => MilDtype::Int8,
        "uint4" => MilDtype::UInt4,
        "uint8" => MilDtype::UInt8,
        "e4m3" => MilDtype::E4M3,
        "e5m2" => MilDtype::E5M2,
        "uint16" => MilDtype::UInt16,
        other => {
            return Err(format!(
                "Unrecognized dtype string '{}'. Valid options: fp16, fp32, int4, int8, uint4, uint8, e4m3, e5m2, uint16",
                other
            ));
        }
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
            target_annotation: Default::default(),
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
            target_annotation: Default::default(),
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
            target_annotation: Default::default(),
        },
        MirNode {
            id: add_id.clone(),
            op: MirOp::MILAdd { name: "add".into(), x: matmul_id.clone(), y: bias_id.clone() },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, shard.output_dim],
            compute_unit_hint: Some(compute_hint),
            air_source: None,
            target_annotation: Default::default(),
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
        // T-115: Use DEFAULT_MINIMUM_DEPLOYMENT_TARGET instead of DEFAULT_OPSET_VERSION
        minimum_deployment_target: crate::DEFAULT_MINIMUM_DEPLOYMENT_TARGET.into(),
        kv_cache_layout: KvCacheLayout::default(),
        sampler_spec: None,
        io_model_spec: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::HandoffKind;
    use crate::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};

    // ─── Helpers ──────────────────────────────────────────────────────

    fn measurement() -> MeasurementConfig {
        MeasurementConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            metrics: vec!["Latency".into()],
        }
    }

    fn sharded_linear_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_sharded".into(),
            family: "ShardedLinearPipeline".into(),
            description: None,
            op: TaskOp::ShardedLinearPipeline {
                input_dim: 64,
                hidden_dim: 48,
                output_dim: 32,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: measurement(),
        }
    }

    fn entry_shard() -> ShardDesc {
        ShardDesc {
            role: ShardRole::Entry,
            shard_name: "test_entry".into(),
            input_dim: 64,
            output_dim: 48,
            compute_units: ComputeUnitHint::CPUAndNE,
        }
    }

    // ─── sharded_pipeline_shards ──────────────────────────────────────

    #[test]
    fn test_sharded_pipeline_shards_structure() {
        let spec = sharded_linear_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        assert_eq!(shards.len(), 3);

        // Entry shard
        assert_eq!(shards[0].role, ShardRole::Entry);
        assert_eq!(shards[0].shard_name, "test_sharded_entry");
        assert_eq!(shards[0].input_dim, 64);
        assert_eq!(shards[0].output_dim, 48); // hidden_dim

        // Interior shard
        assert_eq!(shards[1].role, ShardRole::Interior);
        assert_eq!(shards[1].shard_name, "test_sharded_interior");
        assert_eq!(shards[1].input_dim, 48); // hidden_dim
        assert_eq!(shards[1].output_dim, 48); // hidden_dim

        // Exit shard
        assert_eq!(shards[2].role, ShardRole::Exit);
        assert_eq!(shards[2].shard_name, "test_sharded_exit");
        assert_eq!(shards[2].input_dim, 48); // hidden_dim
        assert_eq!(shards[2].output_dim, 32); // output_dim
    }

    #[test]
    fn test_sharded_pipeline_shards_wrong_op_type() {
        let wrong_spec = SyntheticTaskSpec {
            name: "bad".into(),
            family: "LinearProjection".into(),
            description: None,
            op: TaskOp::LinearProjection {
                input_dim: 64,
                output_dim: 128,
                batch_size: 1,
                has_bias: true,
                dtype: "fp16".into(),
            },
            measurement: measurement(),
        };
        let result = sharded_pipeline_shards(&wrong_spec);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ShardedLinearPipeline"));
    }

    // ─── ShardDesc serialization ──────────────────────────────────────

    #[test]
    fn test_shard_desc_serialization() {
        let shard = entry_shard();
        let json = serde_json::to_string(&shard).unwrap();
        let de: ShardDesc = serde_json::from_str(&json).unwrap();
        assert_eq!(de.role, ShardRole::Entry);
        assert_eq!(de.shard_name, "test_entry");
        assert_eq!(de.input_dim, 64);
        assert_eq!(de.output_dim, 48);
        assert_eq!(de.compute_units, ComputeUnitHint::CPUAndNE);
    }

    // ─── lower_shard_to_mir ───────────────────────────────────────────

    #[test]
    fn test_lower_shard_to_mir_structure() {
        let shard = entry_shard();
        let mir = lower_shard_to_mir(&shard, 1, "fp16").unwrap();

        // Should have 4 nodes: weight, bias, matmul, add
        assert_eq!(mir.nodes.len(), 4);
        assert_eq!(mir.nodes[0].id.0, "weight");
        assert_eq!(mir.nodes[1].id.0, "bias");
        assert_eq!(mir.nodes[2].id.0, "matmul");
        assert_eq!(mir.nodes[3].id.0, "add");

        // Inputs reference "input" node (not in nodes list)
        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.inputs[0].0, "input");

        // Output is the "add" node
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.outputs[0].0, "add");

        // Shard name is preserved
        assert_eq!(mir.shard_name, "test_entry");
    }

    #[test]
    fn test_lower_shard_to_mir_dtypes() {
        let shard = entry_shard();

        // Test fp16
        let mir = lower_shard_to_mir(&shard, 1, "fp16").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::Fp16);

        // Test fp32
        let mir = lower_shard_to_mir(&shard, 1, "fp32").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::Fp32);

        // Test int4
        let mir = lower_shard_to_mir(&shard, 1, "int4").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::Int4);

        // Test e4m3
        let mir = lower_shard_to_mir(&shard, 1, "e4m3").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::E4M3);

        // Test e5m2
        let mir = lower_shard_to_mir(&shard, 1, "e5m2").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::E5M2);
    }

    #[test]
    fn test_lower_shard_to_mir_default_dtype() {
        let shard = entry_shard();
        // T-88 (V-011): Unknown dtype now produces an explicit error instead of
        // silently defaulting to Fp16.
        let result = lower_shard_to_mir(&shard, 1, "unknown_dtype");
        assert!(result.is_err(), "Expected error for unrecognized dtype");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Unrecognized dtype string"),
            "Error should mention unrecognized dtype, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("unknown_dtype"),
            "Error should include the invalid dtype string, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_lower_shard_to_mir_int8_dtype() {
        let shard = entry_shard();
        // T-88: Int8 is now a recognized dtype string
        let mir = lower_shard_to_mir(&shard, 1, "int8").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::Int8);
    }

    #[test]
    fn test_lower_shard_to_mir_uint8_dtype() {
        let shard = entry_shard();
        // T-88: UInt8 is now a recognized dtype string
        let mir = lower_shard_to_mir(&shard, 1, "uint8").unwrap();
        assert_eq!(mir.nodes[0].dtype, MilDtype::UInt8);
    }

    // ─── ShardedShardPayload ──────────────────────────────────────────

    #[test]
    fn test_sharded_shard_payload_from_shard() {
        let shard = entry_shard();
        let payload = ShardedShardPayload::from_shard(
            &shard,
            "task",
            "ShardedLinearPipeline",
            1,
            "fp16",
            "/out",
            42,
        );
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
        assert_eq!(payload.command, "emit_linear_projection");
        assert_eq!(payload.task_name, "task");
        assert_eq!(payload.family, "ShardedLinearPipeline");
        assert_eq!(payload.shard_name, "test_entry");
        assert_eq!(payload.shard_role, "Entry");
        assert_eq!(payload.input_dim, 64);
        assert_eq!(payload.output_dim, 48);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
        assert_eq!(payload.seed, 42);
        assert_eq!(payload.functions[0].stateful, false);
    }

    #[test]
    fn test_sharded_shard_payload_dtype_override() {
        let shard = entry_shard();
        let payload = ShardedShardPayload::from_shard_with_override(
            &shard,
            "task",
            "family",
            1,
            "fp16",
            "/out",
            42,
            Some("fp32"),
        );
        assert_eq!(payload.dtype, "fp32");
        assert_eq!(payload.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_sharded_shard_payload_decode_step() {
        let shard = entry_shard();
        let payload = ShardedShardPayload::from_shard_decode_step(
            &shard,
            "task",
            "ShardedDecodeStep",
            1,
            "fp16",
            "/out",
            42,
            None,
            128, // embed_dim
            4,   // num_heads
            32,  // head_dim
            64,  // kv_len
        );
        assert_eq!(payload.command, "emit_shard_decode_step");
        assert_eq!(payload.shard_role, "Entry");
        // Decode step has 3 inputs: x, k_state, v_state
        assert_eq!(payload.functions[0].inputs.len(), 3);
        assert_eq!(payload.functions[0].inputs[0].name, "x");
        assert_eq!(payload.functions[0].inputs[1].name, "k_state");
        assert_eq!(payload.functions[0].inputs[2].name, "v_state");
        // k_state shape: [1, num_heads, kv_len, head_dim]
        assert_eq!(payload.functions[0].inputs[1].shape, vec![1, 4, 64, 32]);
        // Decode step is stateful
        assert_eq!(payload.functions[0].stateful, true);
    }

    #[test]
    fn test_sharded_shard_payload_serialization() {
        let shard = entry_shard();
        let payload =
            ShardedShardPayload::from_shard(&shard, "task", "family", 1, "fp16", "/out", 42);
        let json = serde_json::to_string(&payload).unwrap();
        let de: ShardedShardPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(de.bridge_version, payload.bridge_version);
        assert_eq!(de.command, payload.command);
        assert_eq!(de.shard_name, payload.shard_name);
        assert_eq!(de.shard_role, payload.shard_role);
        assert_eq!(de.input_dim, payload.input_dim);
        assert_eq!(de.output_dim, payload.output_dim);
        assert_eq!(de.dtype, payload.dtype);
    }

    // ─── build_sharded_pipeline_pir ───────────────────────────────────

    #[test]
    fn test_build_sharded_pipeline_pir_structure() {
        let spec = sharded_linear_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        // 3 packages
        assert_eq!(pir.packages.len(), 3);
        assert_eq!(pir.packages[0].name, "test_sharded_entry");
        assert_eq!(pir.packages[1].name, "test_sharded_interior");
        assert_eq!(pir.packages[2].name, "test_sharded_exit");

        // Shard template is present
        assert!(pir.shard_template.is_some());
        let template = pir.shard_template.unwrap();
        assert_eq!(template.partition_spec.len(), 3);
    }

    #[test]
    fn test_build_sharded_pipeline_pir_wrong_op_type() {
        let wrong_spec = SyntheticTaskSpec {
            name: "bad".into(),
            family: "LinearProjection".into(),
            description: None,
            op: TaskOp::LinearProjection {
                input_dim: 64,
                output_dim: 128,
                batch_size: 1,
                has_bias: true,
                dtype: "fp16".into(),
            },
            measurement: measurement(),
        };
        let result = build_sharded_pipeline_pir(&wrong_spec);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ShardedLinearPipeline"));
    }

    #[test]
    fn test_build_sharded_pipeline_pir_handoffs() {
        let spec = sharded_linear_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        // 2 handoffs
        assert_eq!(pir.handoffs.len(), 2);

        // Entry → Interior
        assert_eq!(pir.handoffs[0].from_package, "test_sharded_entry");
        assert_eq!(pir.handoffs[0].to_package, "test_sharded_interior");
        assert_eq!(pir.handoffs[0].handoff_kind, HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[0].execution_order, 0);
        assert_eq!(pir.handoffs[0].source_output_name, "output");
        assert_eq!(pir.handoffs[0].target_input_name, "x");

        // Interior → Exit
        assert_eq!(pir.handoffs[1].from_package, "test_sharded_interior");
        assert_eq!(pir.handoffs[1].to_package, "test_sharded_exit");
        assert_eq!(pir.handoffs[1].handoff_kind, HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[1].execution_order, 1);
    }

    #[test]
    fn test_build_sharded_pipeline_pir_serialization() {
        let spec = sharded_linear_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();
        let bytes = crate::serialize::serialize_pir(&pir).unwrap();
        let de = crate::serialize::deserialize_pir(&bytes).unwrap();
        assert_eq!(de.packages.len(), pir.packages.len());
        assert_eq!(de.handoffs.len(), pir.handoffs.len());
        assert_eq!(de.packages[0].name, "test_sharded_entry");
    }
}
