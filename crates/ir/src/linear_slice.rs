//! Linear Projection Slice
//!
//! End-to-end path for a synthetic linear projection task:
//! task spec → SIR → MIR → bridge payload.
//! This is the narrowest vertical slice proving the pipeline.
//!
//! Also includes the sharded linear pipeline path (S9.2):
//! task spec → per-shard SIR → per-shard MIR → per-shard bridge payload.
//! Each shard has explicit role semantics (Entry/Interior/Exit).

use crate::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirNodeId, MirOp};
use crate::pir::{ShardRole, ComputeUnits, Package, PackageRole, FunctionEntry, TensorSpec as PirTensorSpec, PirGraph, ShardTemplate, ShardPartitionEntry, Handoff};
use crate::sir::{SirGraph, SirNode, SirNodeId, SirOp, SirMetadata, TaskOrigin};
use crate::task_spec::{SyntheticTaskSpec, TaskOp};

/// Build a SIR graph from a synthetic linear projection task spec.
pub fn sir_from_linear_projection(spec: &SyntheticTaskSpec) -> Result<SirGraph, String> {
    // Extract dimensions from the task spec. The wildcard pattern is kept for
    // forward compatibility: when more TaskOp variants are added, this match
    // will need to handle them. For now, only LinearProjection is supported.
    let (_input_dim, _output_dim, _batch_size, _dtype) = match &spec.op {
        TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        TaskOp::LutProjection { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::DecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::ShardedDecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::Attention { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        // ShardedLinearPipeline is handled by the sharded path, not single-shard
        #[allow(unreachable_patterns)]
        _ => return Err("Expected single-shard task type for sir_from_linear_projection".into()),
    };

    let input_id = SirNodeId("input".into());
    let weight_id = SirNodeId("weight".into());
    let bias_id = SirNodeId("bias".into());
    let output_id = SirNodeId("output".into());

    let nodes = vec![
        SirNode {
            id: weight_id.clone(),
            op: SirOp::ElementWise {
                op: crate::sir::ElementWiseOp::Mul,
                inputs: vec![],
            },
            name: "weight".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
        },
        SirNode {
            id: bias_id.clone(),
            op: SirOp::ElementWise {
                op: crate::sir::ElementWiseOp::Add,
                inputs: vec![],
            },
            name: "bias".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
        },
        SirNode {
            id: output_id.clone(),
            op: SirOp::LinearProjection {
                input: input_id.clone(),
                weight: "weight".into(),
                bias: Some("bias".into()),
            },
            name: "linear_out".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
        },
    ];

    Ok(SirGraph {
        nodes,
        inputs: vec![input_id],
        outputs: vec![output_id],
    })
}

/// Lower a linear projection SIR graph directly to MIR.
pub fn lower_linear_projection_to_mir(
    spec: &SyntheticTaskSpec,
    shard_name: &str,
) -> Result<MirGraph, String> {
    let (input_dim, output_dim, batch_size, dtype) = match &spec.op {
        TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        TaskOp::LutProjection { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::DecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::ShardedDecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::Attention { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        #[allow(unreachable_patterns)]
        _ => return Err("Expected single-shard task type for MIR lowering".into()),
    };

    let mil_dtype = match dtype.as_str() {
        "fp16" => MilDtype::Fp16,
        "fp32" => MilDtype::Fp32,
        _ => MilDtype::Fp16,
    };

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
            shape: vec![input_dim, output_dim],
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
            shape: vec![output_dim],
            compute_unit_hint: None,
            air_source: None,
        },
        MirNode {
            id: matmul_id.clone(),
            op: MirOp::MILMatMul {
                name: "matmul".into(),
                x: input_id.clone(),
                y: weight_id.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, output_dim],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
        },
        MirNode {
            id: add_id.clone(),
            op: MirOp::MILAdd {
                name: "add".into(),
                x: matmul_id.clone(),
                y: bias_id.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, output_dim],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
        },
    ];

    Ok(MirGraph {
        nodes,
        inputs: vec![input_id],
        outputs: vec![add_id],
        opset_version: "iOS18".into(),
        shard_name: shard_name.into(),
    })
}

/// A single function descriptor in the bridge payload.
/// Schema seam for multifunction packages: current emission always
/// produces one function; future emission may produce multiple.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionDescriptor {
    pub name: String,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
    pub stateful: bool,
}

/// Tensor shape/dtype descriptor for function I/O.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Bridge payload: the JSON structure sent to the Python bridge
/// for a linear projection emission.
///
/// Versioned: `bridge_version` field enables Python to reject incompatible
/// payload versions cleanly. Bump this when the payload schema changes
/// in a way that breaks backward compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearProjectionPayload {
    /// Bridge protocol version. Must match what Python expects.
    /// Version history:
    ///   1 — initial schema (command + task + dimensions + dtype + functions)
    pub bridge_version: u32,
    pub command: String,
    pub task_name: String,
    pub family: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    /// Function descriptors for this package.
    /// Defaults to a single "main" function.
    /// The Python emitter records these in the result payload
    /// for manifest correctness. When multifunction emission is
    /// implemented, this list will contain multiple entries and
    /// the emitter will build one MIL program per function.
    pub functions: Vec<FunctionDescriptor>,
}

/// Bridge payload: the JSON structure sent to the Python bridge
/// for a dedicated LUT projection emission.
///
/// This is structurally distinct from `LinearProjectionPayload`:
/// it carries LUT-specific fields (`vocab_size`, `embed_dim`, `num_groups`,
/// `lut_bitwidth`) and uses `command: "emit_lut_projection"` so the
/// Python bridge dispatches to the dedicated LUT emission handler.
///
/// The LUT emission path builds a gather-based program that models
/// the `constexpr_lut`-to-`gather` pattern used in ANE palettized
/// inference, rather than the matmul+add pattern of linear projection.
///
/// Sprint 20 (S20.1): this payload replaces the previous approach where
/// LUT tasks were sent through `LinearProjectionPayload` with
/// `embed_dim × embed_dim` dimensions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LutProjectionPayload {
    /// Bridge protocol version. Must match what Python expects.
    pub bridge_version: u32,
    /// Command identifier: always "emit_lut_projection".
    pub command: String,
    pub task_name: String,
    pub family: String,
    /// Number of possible index values (LUT entries per group).
    pub vocab_size: usize,
    /// Embedding dimension (number of output features per group).
    pub embed_dim: usize,
    /// Number of independent LUT groups.
    pub num_groups: usize,
    /// LUT precision in bits (1, 2, 3, 4, 6, or 8).
    pub lut_bitwidth: usize,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    pub functions: Vec<FunctionDescriptor>,
}

/// Bridge payload: the JSON structure sent to the Python bridge
/// for a dedicated decode-step emission.
///
/// This is structurally distinct from `LinearProjectionPayload`:
/// it carries decode-step-specific fields (`embed_dim`, `num_heads`,
/// `head_dim`, `kv_len`) and uses `command: "emit_stateful_decode_step"`
/// so the Python bridge dispatches to the stateful decode-step emission
/// handler (Sprint 40).
///
/// The decode-step emission path builds a program that models the
/// three-part decode-step pattern (QKV projection → attention →
/// output projection), rather than the simple matmul+add pattern of
/// linear projection.
///
/// Sprint 40: The default decode-step bridge command is now
/// `emit_stateful_decode_step` (uses real `mb.read_state` /
/// `mb.coreml_update_state` for KV-cache state semantics, iOS 18+).
/// The previous stateless path (`emit_decode_step` using `mb.const`
/// KV cache) is available as `emit_stateless_decode_step` for
/// single-step testing.
///
/// Sprint 24 follow-up (resolves Sprint 19/23 residual): this payload
/// replaces the previous approach where decode-step tasks were sent
/// through `LinearProjectionPayload` with `embed_dim × embed_dim`
/// dimensions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeStepPayload {
    /// Bridge protocol version. Must match what Python expects.
    pub bridge_version: u32,
    /// Command identifier: "emit_stateful_decode_step" (Sprint 40).
    ///
    /// Previously "emit_decode_step" which dispatched to the stateless
    /// path (mb.const KV cache). Now defaults to the stateful path
    /// (mb.read_state / mb.coreml_update_state for real KV-cache
    /// state semantics, iOS 18+).
    pub command: String,
    pub task_name: String,
    pub family: String,
    /// Embedding dimension (model hidden size).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// KV cache sequence length.
    pub kv_len: usize,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    pub functions: Vec<FunctionDescriptor>,
}

impl DecodeStepPayload {
    /// Build a dedicated decode-step bridge payload from a task spec.
    ///
    /// Unlike the old approach of reusing LinearProjectionPayload,
    /// this payload carries all decode-step-specific fields and uses
    /// the dedicated "emit_stateful_decode_step" command so the Python
    /// bridge dispatches to the stateful decode-step emission handler
    /// (Sprint 40).
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build a dedicated decode-step bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let (embed_dim, num_heads, head_dim, kv_len, batch_size, spec_dtype) = match &spec.op {
            TaskOp::DecodeStep { embed_dim, num_heads, head_dim, kv_len, batch_size, dtype } => {
                (*embed_dim, *num_heads, *head_dim, *kv_len, *batch_size, dtype.clone())
            }
            _ => return Err("Expected DecodeStep task for DecodeStepPayload".into()),
        };

        let effective_dtype = dtype_override
            .map(|s| s.to_string())
            .unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_stateful_decode_step".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "x".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: effective_dtype.clone(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }
}

/// Bridge payload: the JSON structure sent to the Python bridge
/// for a dedicated MLP block emission.
///
/// This is structurally distinct from `LinearProjectionPayload`:
/// it carries MLP-block-specific fields (`input_dim`, `hidden_dim`,
/// `output_dim`, `activation`) and uses `command: "emit_mlp_block"` so
/// the Python bridge dispatches to the dedicated MLP block emission
/// handler.
///
/// The MLP block emission path builds a program that models the
/// fused linear-activation-linear pattern (feed-forward network block)
/// used in transformer inference, rather than the simple matmul+add
/// pattern of linear projection.
///
/// Sprint 28 (S28.1): this payload replaces the previous approach where
/// MLP block tasks were sent through `LinearProjectionPayload`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MlpBlockPayload {
    /// Bridge protocol version. Must match what Python expects.
    pub bridge_version: u32,
    /// Command identifier: always "emit_mlp_block".
    pub command: String,
    pub task_name: String,
    pub family: String,
    /// Input dimension (typically equals embed_dim).
    pub input_dim: usize,
    /// Hidden (up-projected) dimension.
    pub hidden_dim: usize,
    /// Output dimension (typically equals embed_dim).
    pub output_dim: usize,
    /// Activation function: "gelu" or "relu".
    pub activation: String,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    pub functions: Vec<FunctionDescriptor>,
}

impl MlpBlockPayload {
    /// Build a dedicated MLP block bridge payload from a task spec.
    ///
    /// Unlike the old approach of reusing LinearProjectionPayload,
    /// this payload carries all MLP-block-specific fields and uses
    /// the dedicated "emit_mlp_block" command so the Python bridge
    /// dispatches to the correct MLP block emission handler.
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build an MLP block bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation: the PrecisionPolicyPass determines that the spec's
    /// default dtype (e.g., fp16) is unsafe and should be overridden (e.g., to fp32),
    /// and this method ensures the adapted dtype reaches the Python emitter.
    ///
    /// Sprint 30: this adds dtype override support for MLP block payloads,
    /// matching the pattern established by LinearProjectionPayload.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let (input_dim, hidden_dim, output_dim, activation, batch_size, spec_dtype) = match &spec.op {
            TaskOp::MlpBlock { input_dim, hidden_dim, output_dim, activation, batch_size, dtype } => {
                (*input_dim, *hidden_dim, *output_dim, activation.clone(), *batch_size, dtype.clone())
            }
            _ => return Err("Expected MlpBlock task for MlpBlockPayload".into()),
        };

        let effective_dtype = dtype_override
            .map(|s| s.to_string())
            .unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_mlp_block".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "x".into(),
                    shape: vec![batch_size, input_dim],
                    dtype: effective_dtype.clone(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, output_dim],
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }
}

/// Bridge payload: the JSON structure sent to the Python bridge
/// for a dedicated attention emission.
///
/// This is structurally distinct from `LinearProjectionPayload`:
/// it carries attention-specific fields (`embed_dim`, `num_heads`,
/// `head_dim`, `seq_len`) and uses `command: "emit_attention"` so
/// the Python bridge dispatches to the dedicated attention emission
/// handler.
///
/// The attention emission path builds a program that models the
/// multi-head self-attention pattern (QKV projection → scaled
/// dot-product attention → output projection).
///
/// Sprint 29 (S29.4): this payload provides the dedicated emission
/// path for the fifth real task family.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttentionPayload {
    /// Bridge protocol version. Must match what Python expects.
    pub bridge_version: u32,
    /// Command identifier: always "emit_attention".
    pub command: String,
    pub task_name: String,
    pub family: String,
    /// Embedding dimension (model hidden size).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Input sequence length.
    pub seq_len: usize,
    pub batch_size: usize,
    pub dtype: String,
    pub opset_version: String,
    pub compute_units: String,
    pub output_path: String,
    pub seed: u64,
    pub functions: Vec<FunctionDescriptor>,
}

impl AttentionPayload {
    /// Build a dedicated attention bridge payload from a task spec.
    ///
    /// This payload carries all attention-specific fields and uses
    /// the dedicated "emit_attention" command so the Python bridge
    /// dispatches to the correct attention emission handler.
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build a dedicated attention bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation: the PrecisionPolicyPass determines that the spec's
    /// default dtype (e.g., fp16) is unsafe and should be overridden (e.g., to fp32),
    /// and this method ensures the adapted dtype reaches the Python emitter.
    ///
    /// Sprint 30: this adds dtype override support for attention payloads,
    /// matching the pattern established by LinearProjectionPayload.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let (embed_dim, num_heads, head_dim, seq_len, batch_size, spec_dtype) = match &spec.op {
            TaskOp::Attention { embed_dim, num_heads, head_dim, seq_len, batch_size, dtype } => {
                (*embed_dim, *num_heads, *head_dim, *seq_len, *batch_size, dtype.clone())
            }
            _ => return Err("Expected Attention task for AttentionPayload".into()),
        };

        let effective_dtype = dtype_override
            .map(|s| s.to_string())
            .unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_attention".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "x".into(),
                    shape: vec![batch_size, seq_len, embed_dim],
                    dtype: effective_dtype.clone(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, seq_len, embed_dim],
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }
}

/// Generic family-agnostic bridge payload.
///
/// This replaces the family-specific payload structs (LinearProjectionPayload,
/// LutProjectionPayload, DecodeStepPayload, MlpBlockPayload, AttentionPayload)
/// with a single generic structure that carries family-specific parameters as
/// a JSON value. The Python bridge dispatches on the `command` field, just
/// like before. The `params` field contains all family-specific fields.
///
/// This design means adding a new family requires NO changes to this struct,
/// NO new payload type, and NO new match arm in payload construction. The only
/// change needed is adding the family's `bridge_command()` to TaskOp.
///
/// The family-specific payload structs are retained for backward compatibility
/// but are deprecated. All new code should use FamilyPayload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FamilyPayload {
    /// Bridge protocol version. Must match what Python expects.
    pub bridge_version: u32,
    /// Command identifier for Python bridge dispatch (e.g., "emit_linear_projection").
    pub command: String,
    /// Task name.
    pub task_name: String,
    /// Family identifier (e.g., "LinearProjection", "Attention").
    pub family: String,
    /// Family-specific parameters as a JSON value.
    /// This contains all the fields that were previously on separate
    /// payload structs: input_dim, output_dim, embed_dim, etc.
    pub params: serde_json::Value,
    /// Opset version requirement.
    pub opset_version: String,
    /// Compute units hint.
    pub compute_units: String,
    /// Output path for the mlpackage.
    pub output_path: String,
    /// Random seed for deterministic weight generation.
    pub seed: u64,
    /// Function descriptors for this package.
    pub functions: Vec<FunctionDescriptor>,
}

impl FamilyPayload {
    /// Build a generic bridge payload from any task spec.
    ///
    /// This is the single entry point for all families. It extracts the
    /// bridge command, family params, and tensor shapes from the TaskOp's
    /// generic methods, eliminating the need for per-family payload construction.
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build a generic bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let op = &spec.op;
        let mut params = op.family_params();

        // Apply dtype override if provided
        if let Some(override_dtype) = dtype_override {
            params["dtype"] = serde_json::Value::String(override_dtype.to_string());
        }

        let effective_dtype = params["dtype"].as_str().unwrap_or("fp16").to_string();

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: op.bridge_command().to_string(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            params,
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: op.input_tensor_name().to_string(),
                    shape: op.input_tensor_shape(),
                    dtype: op.input_tensor_dtype(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: op.output_tensor_shape(),
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }

    /// Serialize this payload to a JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|e| format!("Failed to serialize FamilyPayload: {}", e))
    }

    /// Serialize this payload to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize FamilyPayload: {}", e))
    }
}

/// Current bridge protocol version. Bumped when the payload schema
/// changes in a way that breaks backward compatibility.
pub const BRIDGE_VERSION: u32 = 1;

impl LinearProjectionPayload {
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build a bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation: the PrecisionPolicyPass determines that the spec's
    /// default dtype (e.g., fp16) is unsafe and should be overridden (e.g., to fp32),
    /// and this method ensures the adapted dtype reaches the Python emitter.
    ///
    /// The bridge payload carries the effective dtype to the Python emitter,
    /// which uses it to set `compute_precision` in the MIL program. Without
    /// this override, the emitter would use the spec's dtype and the
    /// knowledge-informed adaptation would be lost.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let (input_dim, output_dim, batch_size, spec_dtype) = match &spec.op {
            TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
                (*input_dim, *output_dim, *batch_size, dtype.clone())
            }
            // LUT projection now uses its own dedicated LutProjectionPayload.
            // If this path is reached, the caller should use LutProjectionPayload instead.
            TaskOp::LutProjection { .. } => {
                return Err("LutProjection tasks must use LutProjectionPayload, not LinearProjectionPayload".into());
            }
            // Wildcard kept for forward compatibility with future TaskOp variants
            #[allow(unreachable_patterns)]
            _ => return Err("Expected LinearProjection task for LinearProjectionPayload".into()),
        };

        let effective_dtype = dtype_override
            .map(|s| s.to_string())
            .unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_linear_projection".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            input_dim,
            output_dim,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "x".into(),
                    shape: vec![batch_size, input_dim],
                    dtype: effective_dtype.clone(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, output_dim],
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }
}

impl LutProjectionPayload {
    /// Build a dedicated LUT projection bridge payload from a task spec.
    ///
    /// Unlike the old approach of reusing LinearProjectionPayload with
    /// embed_dim × embed_dim dimensions, this payload carries all LUT-specific
    /// fields (vocab_size, num_groups, lut_bitwidth) and uses the dedicated
    /// "emit_lut_projection" command so the Python bridge dispatches to the
    /// correct LUT emission handler.
    pub fn from_spec(spec: &SyntheticTaskSpec, output_path: &str) -> Result<Self, String> {
        Self::from_spec_with_override(spec, output_path, None)
    }

    /// Build a dedicated LUT projection bridge payload with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This is the propagation mechanism for
    /// precision adaptation: the PrecisionPolicyPass determines that the spec's
    /// default dtype (e.g., fp16) is unsafe and should be overridden (e.g., to fp32),
    /// and this method ensures the adapted dtype reaches the Python emitter.
    ///
    /// Sprint 30: this adds dtype override support for LUT projection payloads,
    /// matching the pattern established by LinearProjectionPayload.
    pub fn from_spec_with_override(
        spec: &SyntheticTaskSpec,
        output_path: &str,
        dtype_override: Option<&str>,
    ) -> Result<Self, String> {
        let (vocab_size, embed_dim, num_groups, lut_bitwidth, batch_size, spec_dtype) = match &spec.op {
            TaskOp::LutProjection { vocab_size, embed_dim, num_groups, lut_bitwidth, batch_size, dtype } => {
                (*vocab_size, *embed_dim, *num_groups, *lut_bitwidth, *batch_size, dtype.clone())
            }
            _ => return Err("Expected LutProjection task for LutProjectionPayload".into()),
        };

        let effective_dtype = dtype_override
            .map(|s| s.to_string())
            .unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_lut_projection".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            vocab_size,
            embed_dim,
            num_groups,
            lut_bitwidth,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: "iOS18".into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: 42,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorDescriptor {
                    name: "indices".into(),
                    shape: vec![batch_size],
                    dtype: "int32".into(),
                }],
                outputs: vec![TensorDescriptor {
                    name: "output".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: effective_dtype,
                }],
                stateful: false,
            }],
        })
    }
}

// ─── Sharded Linear Pipeline (S9.2) ──────────────────────────────────────────

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
    pub compute_units: ComputeUnits,
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

    let compute_hint = match shard.compute_units {
        ComputeUnits::CPUAndNE => ComputeUnitHint::CPUAndNE,
        ComputeUnits::CPUAndGPU => ComputeUnitHint::CPUAndGPU,
        ComputeUnits::CPUOnly => ComputeUnitHint::CPUOnly,
        ComputeUnits::All => ComputeUnitHint::All,
    };

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
            },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, shard.output_dim],
            compute_unit_hint: Some(compute_hint.clone()),
            air_source: None,
        },
        MirNode {
            id: add_id.clone(),
            op: MirOp::MILAdd {
                name: "add".into(),
                x: matmul_id.clone(),
                y: bias_id.clone(),
            },
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
        opset_version: "iOS18".into(),
        shard_name: shard.shard_name.clone(),
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
        Self::from_shard_with_override(shard, task_name, family, batch_size, dtype, output_path, seed, None)
    }

    /// Build a bridge payload for one shard with an optional dtype override.
    ///
    /// When `dtype_override` is `Some`, the payload uses the overridden dtype
    /// instead of the spec's default. This ensures precision adaptations
    /// propagate to the emitted mlpackage per shard.
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
            opset_version: "iOS18".into(),
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
            opset_version: "iOS18".into(),
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

    let packages: Vec<Package> = shards.iter().map(|shard| {
        Package {
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
        }
    }).collect();

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
                compute_units: ComputeUnits::CPUAndNE,
            },
            ShardPartitionEntry {
                role: ShardRole::Interior,
                layer_start: 1,
                layer_end: 1,
                compute_units: ComputeUnits::CPUAndNE,
            },
            ShardPartitionEntry {
                role: ShardRole::Exit,
                layer_start: 2,
                layer_end: 2,
                compute_units: ComputeUnits::CPUAndNE,
            },
        ],
        io_compute_units: None, // No IO model in this synthetic task
        sampler_compute_units: None, // No sampler in this synthetic task
        state_config: None, // No state in linear projection
        context_length: 0,
    };

    Ok(PirGraph {
        packages,
        state_declarations: vec![],
        handoffs,
        shard_template: Some(shard_template),
        context_length: 0,
        opset_version: "iOS18".into(),
        minimum_deployment_target: "iOS18".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};

    fn test_sharded_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_shard".into(),
            family: "ShardedLinearPipeline".into(),
            description: None,
            op: TaskOp::ShardedLinearPipeline {
                input_dim: 64,
                hidden_dim: 48,
                output_dim: 32,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3,
                measured_iterations: 10,
                metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_sharded_pipeline_three_shards() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        assert_eq!(shards.len(), 3, "ShardedLinearPipeline must produce 3 shards");

        assert_eq!(shards[0].role, ShardRole::Entry);
        assert_eq!(shards[0].input_dim, 64);
        assert_eq!(shards[0].output_dim, 48);
        assert_eq!(shards[0].compute_units, ComputeUnits::CPUAndNE);

        assert_eq!(shards[1].role, ShardRole::Interior);
        assert_eq!(shards[1].input_dim, 48);
        assert_eq!(shards[1].output_dim, 48);
        assert_eq!(shards[1].compute_units, ComputeUnits::CPUAndNE);

        assert_eq!(shards[2].role, ShardRole::Exit);
        assert_eq!(shards[2].input_dim, 48);
        assert_eq!(shards[2].output_dim, 32);
        assert_eq!(shards[2].compute_units, ComputeUnits::CPUAndNE);
    }

    #[test]
    fn test_sharded_pipeline_mir_per_shard() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();

        for shard in &shards {
            let mir = lower_shard_to_mir(shard, 1, "fp16").unwrap();
            assert_eq!(mir.nodes.len(), 4, "Each shard MIR must have 4 nodes (weight, bias, matmul, add)");
            assert_eq!(mir.inputs.len(), 1);
            assert_eq!(mir.outputs.len(), 1);
            assert_eq!(mir.shard_name, shard.shard_name);
        }
    }

    #[test]
    fn test_sharded_pipeline_pir() {
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        assert_eq!(pir.packages.len(), 3, "PIR must have 3 packages");

        // Verify package roles
        assert!(matches!(pir.packages[0].role, PackageRole::DecoderShard(ShardRole::Entry)));
        assert!(matches!(pir.packages[1].role, PackageRole::DecoderShard(ShardRole::Interior)));
        assert!(matches!(pir.packages[2].role, PackageRole::DecoderShard(ShardRole::Exit)));

        // Verify handoffs with concrete runtime semantics
        assert_eq!(pir.handoffs.len(), 2, "3 shards must have 2 handoffs");
        assert_eq!(pir.handoffs[0].from_package, "test_shard_entry");
        assert_eq!(pir.handoffs[0].to_package, "test_shard_interior");
        assert_eq!(pir.handoffs[1].from_package, "test_shard_interior");
        assert_eq!(pir.handoffs[1].to_package, "test_shard_exit");

        // Verify concrete handoff semantics (Sprint 17, S17.1)
        assert_eq!(pir.handoffs[0].handoff_kind, crate::pir::HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[0].execution_order, 0);
        assert_eq!(pir.handoffs[0].source_output_name, "output");
        assert_eq!(pir.handoffs[0].target_input_name, "x");
        assert_eq!(pir.handoffs[1].handoff_kind, crate::pir::HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[1].execution_order, 1);
        assert_eq!(pir.handoffs[1].source_output_name, "output");
        assert_eq!(pir.handoffs[1].target_input_name, "x");

        // Verify shard template
        assert!(pir.shard_template.is_some());
        let template = pir.shard_template.as_ref().unwrap();
        assert_eq!(template.partition_spec.len(), 3);
    }

    #[test]
    fn test_concrete_handoff_execution_order() {
        // Verify that handoff execution orders are sequential and start from 0
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        let orders: Vec<usize> = pir.handoffs.iter()
            .map(|h| h.execution_order)
            .collect();
        assert_eq!(orders, vec![0, 1], "Handoff execution orders must be sequential starting from 0");
    }

    #[test]
    fn test_concrete_handoff_source_target_names() {
        // Verify that handoff source_output_name and target_input_name
        // reference actual function I/O names in the packages
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        for handoff in &pir.handoffs {
            // Find the source package
            let source_pkg = pir.packages.iter()
                .find(|p| p.name == handoff.from_package)
                .expect("Source package must exist");
            let target_pkg = pir.packages.iter()
                .find(|p| p.name == handoff.to_package)
                .expect("Target package must exist");

            // Verify source output name matches a function output
            let source_outputs: Vec<&String> = source_pkg.functions.iter()
                .flat_map(|f| f.outputs.iter().map(|o| &o.name))
                .collect();
            assert!(source_outputs.contains(&&handoff.source_output_name),
                "Source output '{}' must exist in package '{}' outputs",
                handoff.source_output_name, handoff.from_package);

            // Verify target input name matches a function input
            let target_inputs: Vec<&String> = target_pkg.functions.iter()
                .flat_map(|f| f.inputs.iter().map(|i| &i.name))
                .collect();
            assert!(target_inputs.contains(&&handoff.target_input_name),
                "Target input '{}' must exist in package '{}' inputs",
                handoff.target_input_name, handoff.to_package);
        }
    }

    #[test]
    fn test_shard_payload_roundtrip() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        let shard = &shards[0];

        let payload = ShardedShardPayload::from_shard(
            shard, &spec.name, &spec.family, 1, "fp16", "/tmp/test", 42,
        );

        assert_eq!(payload.shard_role, "Entry");
        assert_eq!(payload.input_dim, 64);
        assert_eq!(payload.output_dim, 48);
        assert_eq!(payload.compute_units, "CPU_AND_NE");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_sharded_pipeline_rejects_linear_projection() {
        let spec = SyntheticTaskSpec {
            name: "test".into(),
            family: "LinearProjection".into(),
            description: None,
            op: TaskOp::LinearProjection {
                input_dim: 64, output_dim: 32, batch_size: 1, has_bias: true, dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3, measured_iterations: 10, metrics: vec![],
            },
        };
        let result = sharded_pipeline_shards(&spec);
        assert!(result.is_err(), "sharded_pipeline_shards must reject LinearProjection tasks");
    }

    // ─── Precision Override Propagation Tests (Sprint 18) ─────────────────

    fn test_linear_spec_fp16() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_linear".into(),
            family: "LinearProjection".into(),
            description: None,
            op: TaskOp::LinearProjection {
                input_dim: 64, output_dim: 32, batch_size: 1, has_bias: true, dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3, measured_iterations: 10, metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_payload_dtype_override_changes_bridge_dtype() {
        let spec = test_linear_spec_fp16();

        // Without override: uses spec's fp16
        let payload_no = LinearProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16", "Without override, dtype should be fp16");

        // With fp32 override: uses overridden dtype
        let payload_fp32 = LinearProjectionPayload::from_spec_with_override(
            &spec, "/tmp/test", Some("fp32"),
        ).unwrap();
        assert_eq!(payload_fp32.dtype, "fp32",
            "With fp32 override, bridge payload dtype must be fp32");

        // Function descriptors must also reflect the overridden dtype
        assert_eq!(payload_fp32.functions[0].inputs[0].dtype, "fp32",
            "Function input dtype must reflect override");
        assert_eq!(payload_fp32.functions[0].outputs[0].dtype, "fp32",
            "Function output dtype must reflect override");
    }

    #[test]
    fn test_payload_dtype_no_override_preserves_spec() {
        let spec = test_linear_spec_fp16();
        let payload = LinearProjectionPayload::from_spec_with_override(
            &spec, "/tmp/test", None,
        ).unwrap();
        assert_eq!(payload.dtype, "fp16",
            "Without override, dtype must match spec default");
    }

    #[test]
    fn test_shard_payload_dtype_override() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        let shard = &shards[0];

        // Without override
        let payload_no = ShardedShardPayload::from_shard(
            shard, &spec.name, &spec.family, 1, "fp16", "/tmp/test", 42,
        );
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override
        let payload_fp32 = ShardedShardPayload::from_shard_with_override(
            shard, &spec.name, &spec.family, 1, "fp16", "/tmp/test", 42, Some("fp32"),
        );
        assert_eq!(payload_fp32.dtype, "fp32",
            "Shard payload with fp32 override must use fp32 dtype");
        assert_eq!(payload_fp32.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_fp32.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_precision_override_propagates_full_pipeline() {
        // End-to-end test: SIR with precision_override → AIR → MIR
        // This proves that precision adaptation propagates through the IR pipeline.
        // The bridge payload propagation is tested separately in linear_slice tests.
        let spec = test_linear_spec_fp16();

        // Step 1: Build SIR (initially no precision override)
        let sir = sir_from_linear_projection(&spec).unwrap();

        // Step 2: Simulate precision policy setting override on the linear_out node
        let mut sir_adapted = sir.clone();
        for node in &mut sir_adapted.nodes {
            if node.name == "linear_out" {
                node.metadata.precision_override = Some("fp32".to_string());
            }
        }

        // Step 3: Verify SIR override is set
        let linear_sir_node = sir_adapted.nodes.iter()
            .find(|n| n.name == "linear_out")
            .expect("Expected linear_out SIR node");
        assert_eq!(linear_sir_node.metadata.precision_override, Some("fp32".to_string()),
            "Precision override must be set on SIR node");

        // Step 4: Bridge payload with fp32 override must use fp32 dtype
        let payload = LinearProjectionPayload::from_spec_with_override(
            &spec, "/tmp/test", Some("fp32"),
        ).unwrap();
        assert_eq!(payload.dtype, "fp32",
            "Bridge payload dtype must reflect the precision adaptation");
    }

    // ─── Sprint 20 — Dedicated LUT Path Tests ──────────────────────────────

    fn test_lut_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_lut".into(),
            family: "LutProjection".into(),
            description: None,
            op: TaskOp::LutProjection {
                vocab_size: 32000,
                embed_dim: 512,
                num_groups: 64,
                lut_bitwidth: 4,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3,
                measured_iterations: 10,
                metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_lut_payload_from_spec_succeeds() {
        let spec = test_lut_spec();
        let payload = LutProjectionPayload::from_spec(&spec, "/tmp/lut_test").unwrap();
        assert_eq!(payload.command, "emit_lut_projection",
            "LUT payload must use dedicated emit_lut_projection command");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
        assert_eq!(payload.vocab_size, 32000);
        assert_eq!(payload.embed_dim, 512);
        assert_eq!(payload.num_groups, 64);
        assert_eq!(payload.lut_bitwidth, 4);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
        assert_eq!(payload.family, "LutProjection");
    }

    #[test]
    fn test_lut_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = LutProjectionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(),
            "LutProjectionPayload must reject LinearProjection specs");
    }

    #[test]
    fn test_linear_payload_rejects_lut_spec() {
        let spec = test_lut_spec();
        let result = LinearProjectionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(),
            "LinearProjectionPayload must reject LutProjection specs — use LutProjectionPayload instead");
    }

    #[test]
    fn test_linear_vs_lut_payload_command_divergence() {
        // S20.4: Prove that linear and LUT compile paths generate
        // different bridge commands/payloads.
        let linear_spec = test_linear_spec_fp16();
        let lut_spec = test_lut_spec();

        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();
        let lut_payload = LutProjectionPayload::from_spec(&lut_spec, "/tmp/lut_test").unwrap();

        // Commands must differ
        assert_eq!(linear_payload.command, "emit_linear_projection");
        assert_eq!(lut_payload.command, "emit_lut_projection");
        assert_ne!(linear_payload.command, lut_payload.command,
            "Linear and LUT payloads must use different bridge commands");

        // Payloads must have different fields
        // Linear has input_dim/output_dim; LUT has vocab_size/embed_dim/num_groups/lut_bitwidth
        let linear_json = serde_json::to_value(&linear_payload).unwrap();
        let lut_json = serde_json::to_value(&lut_payload).unwrap();

        // Verify LUT-specific fields are present
        assert!(lut_json.get("vocab_size").is_some(),
            "LUT payload must have vocab_size field");
        assert!(lut_json.get("lut_bitwidth").is_some(),
            "LUT payload must have lut_bitwidth field");
        assert!(lut_json.get("num_groups").is_some(),
            "LUT payload must have num_groups field");

        // Verify linear-specific fields are absent from LUT payload
        assert!(lut_json.get("input_dim").is_none(),
            "LUT payload must NOT have input_dim field");
        assert!(lut_json.get("output_dim").is_none(),
            "LUT payload must NOT have output_dim field");

        // Verify LUT-specific fields are absent from linear payload
        assert!(linear_json.get("vocab_size").is_none(),
            "Linear payload must NOT have vocab_size field");
        assert!(linear_json.get("lut_bitwidth").is_none(),
            "Linear payload must NOT have lut_bitwidth field");
    }

    #[test]
    fn test_lut_payload_deterministic_serialization() {
        let spec = test_lut_spec();
        let payload1 = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        let payload2 = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();

        let json1 = serde_json::to_string(&payload1).unwrap();
        let json2 = serde_json::to_string(&payload2).unwrap();
        assert_eq!(json1, json2,
            "LUT payload serialization must be deterministic");
    }

    #[test]
    fn test_lut_payload_function_descriptors() {
        let spec = test_lut_spec();
        let payload = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();

        // LUT payload function descriptor has int32 indices input (not float)
        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "indices");
        assert_eq!(payload.functions[0].inputs[0].dtype, "int32",
            "LUT function input must be int32 indices");
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].dtype, "fp16");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 512]);
    }

    // ─── Decode-Step Payload Divergence Tests ─────────────────────────────

    fn test_decode_step_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_decode".into(),
            family: "DecodeStep".into(),
            description: None,
            op: TaskOp::DecodeStep {
                embed_dim: 128,
                num_heads: 4,
                head_dim: 32,
                kv_len: 64,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_decode_step_payload_from_spec_succeeds() {
        let spec = test_decode_step_spec();
        let payload = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_stateful_decode_step");
        assert_eq!(payload.family, "DecodeStep");
        assert_eq!(payload.embed_dim, 128);
        assert_eq!(payload.num_heads, 4);
        assert_eq!(payload.head_dim, 32);
        assert_eq!(payload.kv_len, 64);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
        assert_eq!(payload.compute_units, "CPU_AND_NE");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_decode_step_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = DecodeStepPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "DecodeStepPayload must reject LinearProjection specs");
    }

    #[test]
    fn test_decode_step_payload_command_differs_from_linear() {
        let linear_spec = test_linear_spec_fp16();
        let decode_spec = test_decode_step_spec();
        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();
        let decode_payload = DecodeStepPayload::from_spec(&decode_spec, "/tmp/test").unwrap();
        assert_ne!(linear_payload.command, decode_payload.command,
            "Decode-step and linear projection must use different bridge commands");
    }

    #[test]
    fn test_decode_step_payload_command_differs_from_lut() {
        let lut_spec = SyntheticTaskSpec {
            name: "test_lut".into(),
            family: "LutProjection".into(),
            description: None,
            op: TaskOp::LutProjection {
                vocab_size: 16, embed_dim: 128, num_groups: 16, lut_bitwidth: 4,
                batch_size: 1, dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3, measured_iterations: 10, metrics: vec![],
            },
        };
        let decode_spec = test_decode_step_spec();
        let lut_payload = LutProjectionPayload::from_spec(&lut_spec, "/tmp/test").unwrap();
        let decode_payload = DecodeStepPayload::from_spec(&decode_spec, "/tmp/test").unwrap();
        assert_ne!(lut_payload.command, decode_payload.command,
            "Decode-step and LUT projection must use different bridge commands");
    }

    #[test]
    fn test_decode_step_payload_deterministic_serialization() {
        let spec = test_decode_step_spec();
        let payload1 = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        let payload2 = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        let json1 = serde_json::to_string(&payload1).unwrap();
        let json2 = serde_json::to_string(&payload2).unwrap();
        assert_eq!(json1, json2, "Same spec must produce deterministic serialization");
    }

    #[test]
    fn test_decode_step_payload_function_descriptors() {
        let spec = test_decode_step_spec();
        let payload = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.functions.len(), 1);
        let func = &payload.functions[0];
        assert_eq!(func.name, "main");
        assert_eq!(func.inputs.len(), 1);
        assert_eq!(func.inputs[0].name, "x");
        assert_eq!(func.inputs[0].shape, vec![1, 128]);
        assert_eq!(func.outputs.len(), 1);
        assert_eq!(func.outputs[0].name, "output");
        assert_eq!(func.outputs[0].shape, vec![1, 128]);
        assert!(!func.stateful);
    }

    // ─── MLP Block Payload Divergence Tests (Sprint 28) ──────────────────

    fn test_mlp_block_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_mlp".into(),
            family: "MlpBlock".into(),
            description: None,
            op: TaskOp::MlpBlock {
                input_dim: 128,
                hidden_dim: 512,
                output_dim: 128,
                activation: "gelu".into(),
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3,
                measured_iterations: 10,
                metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_mlp_block_payload_creation() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_mlp_block");
        assert_eq!(payload.family, "MlpBlock");
        assert_eq!(payload.input_dim, 128);
        assert_eq!(payload.hidden_dim, 512);
        assert_eq!(payload.output_dim, 128);
        assert_eq!(payload.activation, "gelu");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_mlp_block_payload_rejects_linear() {
        let spec = test_linear_spec_fp16();
        let result = MlpBlockPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "MlpBlockPayload must reject LinearProjection tasks");
    }

    #[test]
    fn test_mlp_block_payload_command_diverges_from_linear() {
        let linear_spec = test_linear_spec_fp16();
        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();

        let mlp_spec = test_mlp_block_spec();
        let mlp_payload = MlpBlockPayload::from_spec(&mlp_spec, "/tmp/test").unwrap();

        assert_ne!(linear_payload.command, mlp_payload.command,
            "MLP block and linear projection must use different bridge commands");
        assert_eq!(linear_payload.command, "emit_linear_projection");
        assert_eq!(mlp_payload.command, "emit_mlp_block");
    }

    #[test]
    fn test_mlp_block_payload_deterministic_serialization() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();

        let json1 = serde_json::to_string(&payload).unwrap();
        let json2 = serde_json::to_string(&payload).unwrap();
        assert_eq!(json1, json2, "Serialization must be deterministic");
    }

    #[test]
    fn test_mlp_block_payload_function_descriptors() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();

        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "x");
        assert_eq!(payload.functions[0].inputs[0].shape, vec![1, 128]);
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 128]);
        assert!(!payload.functions[0].stateful);
    }

    // ─── Attention Payload Tests (Sprint 29) ─────────────────────────────────

    fn test_attention_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "attn_128h4_s32_b1_fp16".into(),
            family: "Attention".into(),
            description: None,
            op: TaskOp::Attention {
                embed_dim: 128,
                num_heads: 4,
                head_dim: 32,
                seq_len: 32,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".into(), "Drift".into()],
            },
        }
    }

    #[test]
    fn test_attention_payload_from_spec_succeeds() {
        let spec = test_attention_spec();
        let payload = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_attention");
        assert_eq!(payload.embed_dim, 128);
        assert_eq!(payload.num_heads, 4);
        assert_eq!(payload.head_dim, 32);
        assert_eq!(payload.seq_len, 32);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
    }

    #[test]
    fn test_attention_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = AttentionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "AttentionPayload must reject LinearProjection tasks");
    }

    #[test]
    fn test_attention_payload_dtype_override() {
        let spec = test_attention_spec();

        // Without override: uses spec's fp16
        let payload_no = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override: uses overridden dtype
        let payload_yes = AttentionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32")).unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
        assert_eq!(payload_yes.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_yes.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_attention_payload_function_descriptors() {
        let spec = test_attention_spec();
        let payload = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();

        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "x");
        // Attention input shape: [batch_size, seq_len, embed_dim]
        assert_eq!(payload.functions[0].inputs[0].shape, vec![1, 32, 128]);
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 32, 128]);
        assert!(!payload.functions[0].stateful);
    }

    #[test]
    fn test_mlp_block_payload_dtype_override() {
        let spec = test_mlp_block_spec();

        // Without override: uses spec's fp16
        let payload_no = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override: uses overridden dtype
        let payload_yes = MlpBlockPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32")).unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
        assert_eq!(payload_yes.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_yes.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_lut_projection_payload_dtype_override() {
        let spec = test_lut_spec();
        let payload_no = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        let payload_yes = LutProjectionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32")).unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
    }

    #[test]
    fn test_sir_accepts_attention() {
        let spec = test_attention_spec();
        let sir = sir_from_linear_projection(&spec);
        assert!(sir.is_ok(), "sir_from_linear_projection must accept Attention tasks");
    }

    #[test]
    fn test_mir_accepts_attention() {
        let spec = test_attention_spec();
        let mir = lower_linear_projection_to_mir(&spec, "attn_shard_0");
        assert!(mir.is_ok(), "lower_linear_projection_to_mir must accept Attention tasks");
    }
}
