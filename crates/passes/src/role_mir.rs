//! Role-Specific MIR Builder
//!
//! Produces genuinely different MIR graphs based on `ShardOpProfile`.
//! Before Sprint 43, all three decoder shards in a linear pipeline had
//! identical op structure (just a single MILLinear), differing only in
//! dimensions. This module produces role-appropriate op sequences:
//!
//! - **Entry**: Linear + optional Reshape for handoff preparation
//! - **Interior**: Linear + Activation (GELU)
//! - **Exit**: Linear + LayerNorm
//! - **QKV Projection**: Linear producing concatenated Q, K, V
//! - **Attention**: ScaledDotProductAttention + state ops
//! - **Output Projection**: Linear + optional LayerNorm
//!
//! This is the concrete proof that "sharding is real": two shards with
//! the same role but different op profiles produce genuinely different
//! MIR graphs, not just different-dimension clones of the same ops.

use ane_ir::common::ModelArchConfig;
use ane_ir::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirNodeId, MirOp};
use ane_ir::pir::{ActivationType, ShardOpProfile, ShardSpec};
use anyhow::Result;

// Sprint 58 (S58.3): compute_units_to_hint() removed.
// ComputeUnits and ComputeUnitHint are now the same type (ComputeUnitHint),
// so no conversion is needed.

/// Builder that produces role-specific MIR graphs from shard specifications.
///
/// Each `ShardOpProfile` variant maps to a distinct MIR op sequence.
/// This is the key difference from the pre-Sprint-43 behavior where all
/// decoder shards produced the same MIR structure.
///
/// Sprint 57: the builder now derives compute unit hints from
/// `ShardSpec.compute_units` instead of hardcoding `CPUAndNE`.
/// The `with_compute_hint()` method remains available for callers
/// that need to override the spec-derived value.
pub struct RoleMirBuilder {
    /// Default dtype for all nodes.
    default_dtype: MilDtype,
    /// Default compute unit hint (used when spec doesn't provide one,
    /// or as a fallback). In practice, `build_mir()` now reads the
    /// compute hint from the shard spec, so this is only used as a
    /// default by the `with_compute_hint()` builder method.
    default_compute_hint: ComputeUnitHint,
    /// Model architecture configuration for model-specific constants
    /// (vocab_size, embed_dim, head_dim, max_seq_len, architecture).
    /// When `None`, falls back to `ModelArchConfig::default()` (Qwen3-0.6B).
    /// T-36 (I-15): replaces hardcoded vocab_size=32000, embed_dim=128.
    arch_config: Option<ModelArchConfig>,
}

impl Default for RoleMirBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RoleMirBuilder {
    /// Create a new builder with fp16 dtype, CPU+NE compute hint,
    /// and no model architecture config (uses default Qwen3-0.6B values).
    pub fn new() -> Self {
        Self {
            default_dtype: MilDtype::Fp16,
            default_compute_hint: ComputeUnitHint::CPUAndNE,
            arch_config: None,
        }
    }

    /// Create a builder with a custom dtype.
    pub fn with_dtype(mut self, dtype: MilDtype) -> Self {
        self.default_dtype = dtype;
        self
    }

    /// Create a builder with a custom compute unit hint.
    ///
    /// Note: `build_mir()` now derives the compute hint from the shard
    /// spec's `compute_units` field. This builder method is kept for
    /// backward compatibility and for callers that want to override
    /// the spec-derived value.
    pub fn with_compute_hint(mut self, hint: ComputeUnitHint) -> Self {
        self.default_compute_hint = hint;
        self
    }

    /// Create a builder with a model architecture configuration.
    ///
    /// T-36 (I-15): The config provides model-specific constants
    /// (vocab_size, embed_dim, head_dim, max_seq_len, architecture)
    /// that were previously hardcoded. Without this, the builder
    /// uses Qwen3-0.6B defaults.
    pub fn with_arch_config(mut self, config: ModelArchConfig) -> Self {
        self.arch_config = Some(config);
        self
    }

    /// Returns the effective model architecture config, falling back
    /// to the default (Qwen3-0.6B) when none was provided.
    ///
    /// M-035: Warns when falling back to default values so callers know
    /// they are using model-specific defaults without having set an
    /// explicit config.
    fn arch_config(&self) -> &ModelArchConfig {
        match self.arch_config.as_ref() {
            Some(cfg) => cfg,
            None => {
                static DEFAULT: std::sync::OnceLock<ModelArchConfig> = std::sync::OnceLock::new();
                log::warn!(
                    "RoleMirBuilder: no arch_config provided — falling back to \
                     ModelArchConfig::default() (Qwen3-0.6B). Set one via \
                     with_arch_config() to avoid model-specific defaults."
                );
                DEFAULT.get_or_init(ModelArchConfig::default)
            }
        }
    }

    /// Build a MIR graph from a shard specification.
    ///
    /// The shard's `op_profile` determines the exact op sequence produced.
    /// Two shards with different op profiles produce structurally different
    /// MIR graphs, even if they have the same role.
    pub fn build_mir(&self, spec: &ShardSpec) -> Result<MirGraph> {
        let mut nodes = Vec::new();
        let input_id = MirNodeId(format!("{}_input", spec.shard_name));
        // T-101: Populate input_shapes from the spec so that mir_to_compat
        // can determine shapes/dtypes for input nodes that aren't in the graph's
        // node list. Previously, input_shapes was always empty, causing the
        // compat conversion to fail when the input node is not present as a MirNode.
        let input_shapes: std::collections::HashMap<MirNodeId, Vec<usize>> = {
            let mut map = std::collections::HashMap::new();
            for input_spec in &spec.input_specs {
                map.insert(
                    MirNodeId(format!("{}_input", spec.shard_name)),
                    input_spec.shape.clone(),
                );
            }
            map
        };

        // Sprint 57: derive compute hint from the shard spec's compute_units
        // instead of always using the builder's default (which was CPUAndNE).
        // This ensures knowledge-driven compute unit adaptation (e.g., from
        // ShardPlanPass) propagates through RoleMirBuilder to MIR nodes.
        let compute_hint = spec.compute_units.clone();

        match &spec.op_profile {
            ShardOpProfile::EntryLinear { needs_reshape, reshape_target } => {
                // Entry shard: Const (weight) → Linear → optional Reshape
                let weight_name = format!("{}_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                // T-125 (V-089): Replaced unwrap_or(64) with fail-closed error.
                // Missing shape specs produce incorrect weight tensors.
                let output_dim = spec
                    .output_specs
                    .first()
                    .map(|s| s.shape.iter().product::<usize>())
                    .ok_or_else(|| anyhow::anyhow!(
                        "EntryLinear shard '{}' has no output_specs — cannot determine output_dim",
                        spec.shard_name
                    ))?;
                let input_dim = spec
                    .input_specs
                    .first()
                    .map(|s| s.shape.iter().product::<usize>())
                    .ok_or_else(|| anyhow::anyhow!(
                        "EntryLinear shard '{}' has no input_specs — cannot determine input_dim",
                        spec.shard_name
                    ))?;

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![output_dim, input_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_linear_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                // Entry-specific: add reshape if needed for handoff
                let output_id = if *needs_reshape {
                    if let Some(target) = reshape_target {
                        let reshape_name = format!("{}_reshape", spec.shard_name);
                        let reshape_id = MirNodeId(format!("{}_reshape_out", spec.shard_name));
                        nodes.push(MirNode {
                            id: reshape_id.clone(),
                            op: MirOp::MILReshape {
                                name: reshape_name,
                                x: linear_id,
                                shape: target.clone(),
                            },
                            dtype: self.default_dtype.clone(),
                            shape: target.clone(),
                            compute_unit_hint: Some(compute_hint.clone()),
                            air_source: None,
                        });
                        reshape_id
                    } else {
                        linear_id
                    }
                } else {
                    linear_id
                };

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![output_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::InteriorLinear { activation } => {
                // Interior shard: Const → Linear → Activation
                let weight_name = format!("{}_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                // T-125 (V-089): Replaced unwrap_or(48) with fail-closed error.
                // Missing shape specs produce incorrect weight tensors.
                let hidden_dim = spec
                    .input_specs
                    .first()
                    .and_then(|s| s.shape.get(1).copied())
                    .ok_or_else(|| anyhow::anyhow!(
                        "InteriorLinear shard '{}' has no input_specs or shape[1] — cannot determine hidden_dim",
                        spec.shard_name
                    ))?;

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![hidden_dim, hidden_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_linear_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                // Interior-specific: activation function (the key difference from Entry/Exit)
                let output_id = match activation {
                    ActivationType::GeluTanh => {
                        let gelu_name = format!("{}_gelu", spec.shard_name);
                        let gelu_id = MirNodeId(format!("{}_gelu_out", spec.shard_name));
                        nodes.push(MirNode {
                            id: gelu_id.clone(),
                            op: MirOp::MILGelu {
                                name: gelu_name,
                                x: linear_id,
                                mode: "TANH_APPROXIMATION".into(),
                            },
                            dtype: self.default_dtype.clone(),
                            shape: spec
                                .output_specs
                                .first()
                                .map(|s| s.shape.clone())
                                .unwrap_or_default(),
                            compute_unit_hint: Some(compute_hint.clone()),
                            air_source: None,
                        });
                        gelu_id
                    }
                    ActivationType::Relu => {
                        let relu_name = format!("{}_relu", spec.shard_name);
                        let relu_id = MirNodeId(format!("{}_relu_out", spec.shard_name));
                        nodes.push(MirNode {
                            id: relu_id.clone(),
                            op: MirOp::MILRelu { name: relu_name, x: linear_id },
                            dtype: self.default_dtype.clone(),
                            shape: spec
                                .output_specs
                                .first()
                                .map(|s| s.shape.clone())
                                .unwrap_or_default(),
                            compute_unit_hint: Some(compute_hint.clone()),
                            air_source: None,
                        });
                        relu_id
                    }
                    ActivationType::None => linear_id,
                };

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![output_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::ExitLinear { ln_epsilon } => {
                // Exit shard: Const (weight) → Linear → LayerNorm
                let weight_name = format!("{}_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                // T-125 (V-089): Replaced unwrap_or(32)/unwrap_or(48) with fail-closed errors.
                // Missing shape specs produce incorrect weight tensors.
                let output_dim = spec
                    .output_specs
                    .first()
                    .and_then(|s| s.shape.get(1).copied())
                    .ok_or_else(|| anyhow::anyhow!(
                        "ExitLinear shard '{}' has no output_specs or shape[1] — cannot determine output_dim",
                        spec.shard_name
                    ))?;
                let input_dim = spec
                    .input_specs
                    .first()
                    .and_then(|s| s.shape.get(1).copied())
                    .ok_or_else(|| anyhow::anyhow!(
                        "ExitLinear shard '{}' has no input_specs or shape[1] — cannot determine input_dim",
                        spec.shard_name
                    ))?;

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![output_dim, input_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_linear_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                // Exit-specific: LayerNorm (the key difference from Entry/Interior)
                let ln_weight_name = format!("{}_ln_weight", spec.shard_name);
                let ln_name = format!("{}_layernorm", spec.shard_name);
                let ln_id = MirNodeId(format!("{}_layernorm_out", spec.shard_name));
                nodes.push(MirNode {
                    id: ln_id.clone(),
                    op: MirOp::MILLayerNorm {
                        name: ln_name,
                        x: linear_id,
                        weight: ln_weight_name,
                        bias: None,
                        epsilon: *ln_epsilon,
                        axes: vec![1],
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![ln_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::QkvProjection { num_heads, head_dim } => {
                // QKV projection: Linear producing [batch, 3 * embed_dim]
                let weight_name = format!("{}_qkv_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                let embed_dim = *num_heads * *head_dim;
                let qkv_dim = 3 * embed_dim;

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/qkv_weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![qkv_dim, embed_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_qkv_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_qkv_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, qkv_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                // Extract Q, K, V from QKV projection output using slice_by_index.
                // NOTE: We intentionally do NOT use MILSplit here because Core ML's
                // split op returns a *list* of tensors, which our IR cannot model.
                // SliceByIndex on the original tensor produces valid single-output MIL.
                let embed_dim_val = embed_dim; // size of each Q/K/V slice

                let q_id = MirNodeId(format!("{}_q", spec.shard_name));
                nodes.push(MirNode {
                    id: q_id.clone(),
                    op: MirOp::MILSliceByIndex {
                        name: format!("{}_q_extract", spec.shard_name),
                        x: linear_id.clone(),
                        begin: vec![0, 0],
                        end: vec![1, embed_dim_val as i64],
                        stride: vec![1, 1],
                        begin_mask: vec![false, false],
                        end_mask: vec![false, false],
                        squeeze_mask: vec![false, false],
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, embed_dim_val],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let k_id = MirNodeId(format!("{}_k", spec.shard_name));
                nodes.push(MirNode {
                    id: k_id.clone(),
                    op: MirOp::MILSliceByIndex {
                        name: format!("{}_k_extract", spec.shard_name),
                        x: linear_id.clone(),
                        begin: vec![0, embed_dim_val as i64],
                        end: vec![1, (2 * embed_dim_val) as i64],
                        stride: vec![1, 1],
                        begin_mask: vec![false, false],
                        end_mask: vec![false, false],
                        squeeze_mask: vec![false, false],
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, embed_dim_val],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let v_id = MirNodeId(format!("{}_v", spec.shard_name));
                nodes.push(MirNode {
                    id: v_id.clone(),
                    op: MirOp::MILSliceByIndex {
                        name: format!("{}_v_extract", spec.shard_name),
                        x: linear_id.clone(),
                        begin: vec![0, (2 * embed_dim_val) as i64],
                        end: vec![1, (3 * embed_dim_val) as i64],
                        stride: vec![1, 1],
                        begin_mask: vec![false, false],
                        end_mask: vec![false, false],
                        squeeze_mask: vec![false, false],
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, embed_dim_val],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![q_id, k_id, v_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::AttentionComputation {
                causal: _,
                stateful,
                num_heads,
                head_dim,
                context_length,
            } => {
                // Attention: ReadState (KV cache) → ScaledDotProductAttention → UpdateState
                // KV cache shape is derived from the shard spec dimensions:
                //   [batch, num_heads, context_length, head_dim]
                let kv_cache_shape = vec![1, *num_heads, *context_length, *head_dim];

                let q_id = MirNodeId(format!("{}_q", spec.shard_name));
                let k_id = MirNodeId(format!("{}_k", spec.shard_name));
                let v_id = MirNodeId(format!("{}_v", spec.shard_name));

                if *stateful {
                    // Read KV cache state
                    let read_k_name = format!("{}_read_k_cache", spec.shard_name);
                    let read_k_id = MirNodeId(format!("{}_k_cache", spec.shard_name));
                    nodes.push(MirNode {
                        id: read_k_id.clone(),
                        op: MirOp::MILReadState {
                            name: read_k_name,
                            state_id: format!("{}_kv_cache_k", spec.shard_name),
                            shape: kv_cache_shape.clone(),
                            dtype: self.default_dtype.clone(),
                        },
                        dtype: self.default_dtype.clone(),
                        shape: kv_cache_shape.clone(),
                        compute_unit_hint: Some(compute_hint.clone()),
                        air_source: None,
                    });

                    let read_v_name = format!("{}_read_v_cache", spec.shard_name);
                    let read_v_id = MirNodeId(format!("{}_v_cache", spec.shard_name));
                    nodes.push(MirNode {
                        id: read_v_id.clone(),
                        op: MirOp::MILReadState {
                            name: read_v_name,
                            state_id: format!("{}_kv_cache_v", spec.shard_name),
                            shape: kv_cache_shape.clone(),
                            dtype: self.default_dtype.clone(),
                        },
                        dtype: self.default_dtype.clone(),
                        shape: kv_cache_shape.clone(),
                        compute_unit_hint: Some(compute_hint.clone()),
                        air_source: None,
                    });
                }

                // ScaledDotProductAttention
                let attn_name = format!("{}_sdpa", spec.shard_name);
                let attn_id = MirNodeId(format!("{}_attn_out", spec.shard_name));
                nodes.push(MirNode {
                    id: attn_id.clone(),
                    op: MirOp::MILScaledDotProductAttention {
                        name: attn_name,
                        query: q_id,
                        key: k_id,
                        value: v_id,
                        attention_mask: None,
                        scale: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                if *stateful {
                    // Update KV cache state
                    let update_name = format!("{}_update_k_cache", spec.shard_name);
                    let update_id = MirNodeId(format!("{}_k_cache_update", spec.shard_name));
                    nodes.push(MirNode {
                        id: update_id.clone(),
                        op: MirOp::MILCoremlUpdateState {
                            name: update_name,
                            state_id: format!("{}_kv_cache_k", spec.shard_name),
                            value: attn_id.clone(),
                        },
                        dtype: self.default_dtype.clone(),
                        shape: vec![],
                        compute_unit_hint: Some(compute_hint.clone()),
                        air_source: None,
                    });
                }

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![attn_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::OutputProjection { with_norm, ln_epsilon } => {
                // Output projection: Linear + optional LayerNorm
                let weight_name = format!("{}_out_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                let embed_dim = spec
                    .input_specs
                    .first()
                    .map(|s| {
                        s.shape.get(1).copied().unwrap_or_else(|| self.arch_config().embed_dim)
                    })
                    .unwrap_or_else(|| self.arch_config().embed_dim);

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/out_weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![embed_dim, embed_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_out_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_out_linear_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let output_id = if *with_norm {
                    let ln_weight_name = format!("{}_out_ln_weight", spec.shard_name);
                    let ln_name = format!("{}_out_layernorm", spec.shard_name);
                    let ln_id = MirNodeId(format!("{}_out_ln_out", spec.shard_name));
                    nodes.push(MirNode {
                        id: ln_id.clone(),
                        op: MirOp::MILLayerNorm {
                            name: ln_name,
                            x: linear_id,
                            weight: ln_weight_name,
                            bias: None,
                            epsilon: ln_epsilon.unwrap_or(1e-5),
                            axes: vec![1],
                        },
                        dtype: self.default_dtype.clone(),
                        shape: spec
                            .output_specs
                            .first()
                            .map(|s| s.shape.clone())
                            .unwrap_or_default(),
                        compute_unit_hint: Some(compute_hint.clone()),
                        air_source: None,
                    });
                    ln_id
                } else {
                    linear_id
                };

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![output_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::IoEmbedding { with_lm_head: _ } => {
                // IO model: simplified embedding + projection
                // For now, a linear-only graph (embedding is a special gather)
                let weight_name = format!("{}_embed_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/embed_weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec
                        .output_specs
                        .first()
                        .map(|s| {
                            // Embedding weight shape = [vocab_size, embed_dim].
                            // Derive from output spec: if output is [batch, embed_dim],
                            // the weight shape is [vocab_size, embed_dim].
                            // T-36 (I-15): Use arch_config.vocab_size and arch_config.embed_dim
                            // instead of hardcoded 32000 and 128.
                            let embed_dim = s
                                .shape
                                .get(1)
                                .copied()
                                .unwrap_or_else(|| self.arch_config().embed_dim);
                            vec![self.arch_config().vocab_size, embed_dim]
                        })
                        .unwrap_or_else(|| {
                            vec![self.arch_config().vocab_size, self.arch_config().embed_dim]
                        }),
                    compute_unit_hint: Some(compute_hint.clone()), // IO compute hint from spec
                    air_source: None,
                });

                let gather_name = format!("{}_embed_gather", spec.shard_name);
                let gather_id = MirNodeId(format!("{}_embed_out", spec.shard_name));
                let indices_id = MirNodeId(format!("{}_token_ids", spec.shard_name));

                nodes.push(MirNode {
                    id: gather_id.clone(),
                    op: MirOp::MILGather {
                        name: gather_name,
                        x: weight_id,
                        indices: indices_id,
                        axis: 0,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![gather_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::SamplerTopk { k } => {
                // Sampler: Top-k + softmax
                let topk_name = format!("{}_topk", spec.shard_name);
                let topk_id = MirNodeId(format!("{}_topk_out", spec.shard_name));

                nodes.push(MirNode {
                    id: topk_id.clone(),
                    op: MirOp::MILTopk { name: topk_name, x: input_id.clone(), k: *k, axis: -1 },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, *k],
                    compute_unit_hint: Some(compute_hint.clone()), // Sampler compute hint from spec
                    air_source: None,
                });

                let softmax_name = format!("{}_softmax", spec.shard_name);
                let softmax_id = MirNodeId(format!("{}_softmax_out", spec.shard_name));

                nodes.push(MirNode {
                    id: softmax_id.clone(),
                    op: MirOp::MILSoftmax { name: softmax_name, x: topk_id, axis: -1 },
                    dtype: self.default_dtype.clone(),
                    shape: vec![1, *k],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![softmax_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }

            ShardOpProfile::LinearOnly => {
                // Backward-compatible: single linear, same as pre-Sprint-43 behavior
                let weight_name = format!("{}_weight", spec.shard_name);
                let weight_id = MirNodeId(weight_name.clone());
                // T-125 (V-089): Replaced unwrap_or(32)/unwrap_or(64) with fail-closed errors.
                // Missing shape specs produce incorrect weight tensors.
                let output_dim = spec
                    .output_specs
                    .first()
                    .and_then(|s| s.shape.get(1).copied())
                    .ok_or_else(|| anyhow::anyhow!(
                        "LinearOnly shard '{}' has no output_specs or shape[1] — cannot determine output_dim",
                        spec.shard_name
                    ))?;
                let input_dim = spec
                    .input_specs
                    .first()
                    .and_then(|s| s.shape.get(1).copied())
                    .ok_or_else(|| anyhow::anyhow!(
                        "LinearOnly shard '{}' has no input_specs or shape[1] — cannot determine input_dim",
                        spec.shard_name
                    ))?;

                nodes.push(MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: weight_name.clone(),
                        value_path: format!("weights/{}/weight.bin", spec.shard_name),
                        dtype: self.default_dtype.clone(),
                    },
                    dtype: self.default_dtype.clone(),
                    shape: vec![output_dim, input_dim],
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                let linear_name = format!("{}_linear", spec.shard_name);
                let linear_id = MirNodeId(format!("{}_linear_out", spec.shard_name));
                nodes.push(MirNode {
                    id: linear_id.clone(),
                    op: MirOp::MILLinear {
                        name: linear_name,
                        x: input_id.clone(),
                        weight: weight_name,
                        bias: None,
                    },
                    dtype: self.default_dtype.clone(),
                    shape: spec.output_specs.first().map(|s| s.shape.clone()).unwrap_or_default(),
                    compute_unit_hint: Some(compute_hint.clone()),
                    air_source: None,
                });

                Ok(MirGraph {
                    nodes,
                    inputs: vec![input_id],
                    outputs: vec![linear_id],
                    opset_version: ane_ir::DEFAULT_OPSET_VERSION.into(),
                    shard_name: spec.shard_name.clone(),
                    input_shapes: input_shapes.clone(),
                })
            }
        }
    }

    /// Classify the op types in a MIR graph for structural comparison.
    ///
    /// Returns a sorted list of op type names (e.g., "Linear", "Gelu", "LayerNorm").
    /// Two shards with genuinely different op structures will produce different
    /// op type signatures.
    pub fn op_type_signature(graph: &MirGraph) -> Vec<String> {
        let mut sig: Vec<String> = graph
            .nodes
            .iter()
            .map(|n| {
                match &n.op {
                    MirOp::MILConst { .. } => "Const",
                    MirOp::MILLinear { .. } => "Linear",
                    MirOp::MILMatMul { .. } => "MatMul",
                    MirOp::MILAdd { .. } => "Add",
                    MirOp::MILMul { .. } => "Mul",
                    MirOp::MILSub { .. } => "Sub",
                    MirOp::MILAbs { .. } => "Abs",
                    MirOp::MILMaximum { .. } => "Maximum",
                    MirOp::MILMinimum { .. } => "Minimum",
                    MirOp::MILReshape { .. } => "Reshape",
                    MirOp::MILTranspose { .. } => "Transpose",
                    MirOp::MILSplit { .. } => "Split",
                    MirOp::MILConcat { .. } => "Concat",
                    MirOp::MILSoftmax { .. } => "Softmax",
                    MirOp::MILGelu { .. } => "Gelu",
                    MirOp::MILScaledDotProductAttention { .. } => "SDPA",
                    MirOp::MILSliceByIndex { .. } => "SliceByIndex",
                    MirOp::MILReadState { .. } => "ReadState",
                    MirOp::MILCoremlUpdateState { .. } => "UpdateState",
                    MirOp::MILReduceMean { .. } => "ReduceMean",
                    MirOp::MILReduceSum { .. } => "ReduceSum",
                    MirOp::MILRsqrt { .. } => "Rsqrt",
                    MirOp::MILRealDiv { .. } => "RealDiv",
                    MirOp::MILLayerNorm { .. } => "LayerNorm",
                    MirOp::MILTopk { .. } => "Topk",
                    MirOp::MILGather { .. } => "Gather",
                    MirOp::MILCos { .. } => "Cos",
                    MirOp::MILSin { .. } => "Sin",
                    MirOp::MILCast { .. } => "Cast",
                    MirOp::MILConv { .. } => "Conv",
                    MirOp::MILStateWrite { .. } => "StateWrite",
                    MirOp::MILSliceUpdate { .. } => "SliceUpdate",
                    MirOp::MILExp { .. } => "Exp",
                    MirOp::MILSigmoid { .. } => "Sigmoid",
                    MirOp::MILTanh { .. } => "Tanh",
                    MirOp::MILRelu { .. } => "Relu",
                    MirOp::MILWhere { .. } => "Where",
                    _ => "Other",
                }
                .to_string()
            })
            .collect();
        sig.sort();
        sig.dedup();
        sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::pir::{ComputeUnitHint, ShardRole, TensorSpec};

    fn make_entry_spec() -> ShardSpec {
        ShardSpec {
            shard_name: "test_entry".into(),
            role: ShardRole::Entry,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 64],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::EntryLinear {
                needs_reshape: true,
                reshape_target: Some(vec![1, 48]),
            },
        }
    }

    fn make_interior_spec() -> ShardSpec {
        ShardSpec {
            shard_name: "test_interior".into(),
            role: ShardRole::Interior,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::InteriorLinear { activation: ActivationType::GeluTanh },
        }
    }

    fn make_exit_spec() -> ShardSpec {
        ShardSpec {
            shard_name: "test_exit".into(),
            role: ShardRole::Exit,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 32],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::ExitLinear { ln_epsilon: 1e-5 },
        }
    }

    /// Sprint 43 core test: Entry/Interior/Exit produce different op structures.
    ///
    /// Before Sprint 43, all three produced just [Const, Linear].
    /// After Sprint 43:
    /// - Entry produces [Const, Linear, Reshape]
    /// - Interior produces [Const, Linear, Gelu]
    /// - Exit produces [Const, Linear, LayerNorm]
    #[test]
    fn test_roles_produce_different_op_structures() {
        let builder = RoleMirBuilder::new();

        let entry_mir = builder.build_mir(&make_entry_spec()).unwrap();
        let interior_mir = builder.build_mir(&make_interior_spec()).unwrap();
        let exit_mir = builder.build_mir(&make_exit_spec()).unwrap();

        let entry_sig = RoleMirBuilder::op_type_signature(&entry_mir);
        let interior_sig = RoleMirBuilder::op_type_signature(&interior_mir);
        let exit_sig = RoleMirBuilder::op_type_signature(&exit_mir);

        // All three should differ from each other
        assert_ne!(entry_sig, interior_sig,
            "Entry and Interior must have different op type signatures. Entry: {:?}, Interior: {:?}",
            entry_sig, interior_sig);
        assert_ne!(
            entry_sig, exit_sig,
            "Entry and Exit must have different op type signatures. Entry: {:?}, Exit: {:?}",
            entry_sig, exit_sig
        );
        assert_ne!(
            interior_sig, exit_sig,
            "Interior and Exit must have different op type signatures. Interior: {:?}, Exit: {:?}",
            interior_sig, exit_sig
        );
    }

    /// Verify Entry shard has Reshape op (unique to Entry).
    #[test]
    fn test_entry_has_reshape() {
        let builder = RoleMirBuilder::new();
        let mir = builder.build_mir(&make_entry_spec()).unwrap();

        let has_reshape = mir.nodes.iter().any(|n| matches!(n.op, MirOp::MILReshape { .. }));
        assert!(has_reshape, "Entry shard must include a Reshape op");
    }

    /// Verify Interior shard has GELU activation (unique to Interior).
    #[test]
    fn test_interior_has_gelu() {
        let builder = RoleMirBuilder::new();
        let mir = builder.build_mir(&make_interior_spec()).unwrap();

        let has_gelu = mir.nodes.iter().any(|n| matches!(n.op, MirOp::MILGelu { .. }));
        assert!(has_gelu, "Interior shard must include a GELU activation op");
    }

    /// Verify Exit shard has LayerNorm (unique to Exit).
    #[test]
    fn test_exit_has_layernorm() {
        let builder = RoleMirBuilder::new();
        let mir = builder.build_mir(&make_exit_spec()).unwrap();

        let has_ln = mir.nodes.iter().any(|n| matches!(n.op, MirOp::MILLayerNorm { .. }));
        assert!(has_ln, "Exit shard must include a LayerNorm op");
    }

    /// Verify LinearOnly produces [Const, Linear] (backward compat).
    #[test]
    fn test_linear_only_backward_compat() {
        let builder = RoleMirBuilder::new();
        let spec = ShardSpec {
            shard_name: "legacy".into(),
            role: ShardRole::Interior,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::LinearOnly,
        };

        let mir = builder.build_mir(&spec).unwrap();
        let sig = RoleMirBuilder::op_type_signature(&mir);

        assert_eq!(
            sig,
            vec!["Const", "Linear"],
            "LinearOnly profile must produce exactly [Const, Linear]"
        );
    }

    /// Test decode-step pipeline: QKV, Attention, OutputProjection all differ.
    #[test]
    fn test_decode_step_roles_produce_different_structures() {
        let builder = RoleMirBuilder::new();

        let qkv_spec = ShardSpec {
            shard_name: "qkv".into(),
            role: ShardRole::Entry,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 128],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "qkv".into(),
                shape: vec![1, 384],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::QkvProjection { num_heads: 4, head_dim: 32 },
        };

        let attn_spec = ShardSpec {
            shard_name: "attn".into(),
            role: ShardRole::Interior,
            input_specs: vec![TensorSpec {
                name: "qkv".into(),
                shape: vec![1, 384],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "attn_out".into(),
                shape: vec![1, 128],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::AttentionComputation {
                causal: true,
                stateful: true,
                num_heads: 4,
                head_dim: 32,
                context_length: 64,
            },
        };

        let out_spec = ShardSpec {
            shard_name: "out_proj".into(),
            role: ShardRole::Exit,
            input_specs: vec![TensorSpec {
                name: "attn_out".into(),
                shape: vec![1, 128],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 128],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::OutputProjection {
                with_norm: true,
                ln_epsilon: Some(1e-5),
            },
        };

        let qkv_mir = builder.build_mir(&qkv_spec).unwrap();
        let attn_mir = builder.build_mir(&attn_spec).unwrap();
        let out_mir = builder.build_mir(&out_spec).unwrap();

        let qkv_sig = RoleMirBuilder::op_type_signature(&qkv_mir);
        let attn_sig = RoleMirBuilder::op_type_signature(&attn_mir);
        let out_sig = RoleMirBuilder::op_type_signature(&out_mir);

        assert_ne!(
            qkv_sig, attn_sig,
            "QKV and Attention must differ: {:?} vs {:?}",
            qkv_sig, attn_sig
        );
        assert_ne!(qkv_sig, out_sig, "QKV and Output must differ: {:?} vs {:?}", qkv_sig, out_sig);
        assert_ne!(
            attn_sig, out_sig,
            "Attention and Output must differ: {:?} vs {:?}",
            attn_sig, out_sig
        );

        // QKV should have SliceByIndex (not Split — Split is invalid MIL),
        // Attention should have SDPA + state ops
        assert!(
            qkv_sig.contains(&"SliceByIndex".to_string()),
            "QKV must include SliceByIndex (not Split)"
        );
        assert!(
            !qkv_sig.contains(&"Split".to_string()),
            "QKV must NOT include Split (invalid MIL)"
        );
        assert!(attn_sig.contains(&"SDPA".to_string()), "Attention must include SDPA");
        assert!(attn_sig.contains(&"ReadState".to_string()), "Attention must include ReadState");
        assert!(out_sig.contains(&"LayerNorm".to_string()), "Output must include LayerNorm");
    }

    /// Test IO and Sampler shards have correct compute unit hints.
    #[test]
    fn test_io_and_sampler_use_cpu_gpu() {
        let builder = RoleMirBuilder::new();

        let io_spec = ShardSpec {
            shard_name: "io".into(),
            role: ShardRole::Io,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 128],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndGPU,
            op_profile: ShardOpProfile::IoEmbedding { with_lm_head: false },
        };

        let sampler_spec = ShardSpec {
            shard_name: "sampler".into(),
            role: ShardRole::Sampler,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 32000],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 5],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndGPU,
            op_profile: ShardOpProfile::SamplerTopk { k: 5 },
        };

        let io_mir = builder.build_mir(&io_spec).unwrap();
        let sampler_mir = builder.build_mir(&sampler_spec).unwrap();

        // IO should have Gather, Sampler should have Topk + Softmax
        let io_sig = RoleMirBuilder::op_type_signature(&io_mir);
        let sampler_sig = RoleMirBuilder::op_type_signature(&sampler_mir);

        assert!(io_sig.contains(&"Gather".to_string()), "IO must include Gather");
        assert!(sampler_sig.contains(&"Topk".to_string()), "Sampler must include Topk");
        assert!(sampler_sig.contains(&"Softmax".to_string()), "Sampler must include Softmax");

        // Both should use CPU+GPU hints
        for node in &io_mir.nodes {
            assert_eq!(
                node.compute_unit_hint,
                Some(ComputeUnitHint::CPUAndGPU),
                "IO node {} must use CPUAndGPU",
                node.id.0
            );
        }
        for node in &sampler_mir.nodes {
            assert_eq!(
                node.compute_unit_hint,
                Some(ComputeUnitHint::CPUAndGPU),
                "Sampler node {} must use CPUAndGPU",
                node.id.0
            );
        }
    }

    /// Sprint 57: verify that ShardSpec.compute_units propagates to MIR nodes.
    /// Before Sprint 57, RoleMirBuilder always used CPUAndNE regardless of spec.
    #[test]
    fn test_compute_units_from_spec_propagates_to_mir_nodes() {
        let builder = RoleMirBuilder::new();
        let mut spec = make_entry_spec();
        spec.compute_units = ComputeUnitHint::CPUAndGPU;

        let mir = builder.build_mir(&spec).unwrap();

        for node in &mir.nodes {
            assert_eq!(
                node.compute_unit_hint,
                Some(ComputeUnitHint::CPUAndGPU),
                "All nodes should use CPUAndGPU from spec, but {} has {:?}",
                node.id.0,
                node.compute_unit_hint
            );
        }
    }

    // T-125 (V-089): EntryLinear with empty input_specs/output_specs must error.
    // Missing shape specs produce incorrect weight tensors; the builder
    // now returns an error instead of silently defaulting dimensions.
    #[test]
    fn test_t125_entry_linear_missing_specs_errors() {
        let spec = ShardSpec {
            shard_name: "test_entry_missing".into(),
            role: ShardRole::Entry,
            input_specs: vec![],
            output_specs: vec![],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::EntryLinear {
                needs_reshape: false,
                reshape_target: None,
            },
        };
        let builder = RoleMirBuilder::new();
        let result = builder.build_mir(&spec);
        assert!(result.is_err(), "EntryLinear with empty specs should return an error");
    }
}
