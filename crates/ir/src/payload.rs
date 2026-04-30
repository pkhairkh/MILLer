//! Bridge payload types for the Python emission bridge.
//!
//! These JSON structures are sent to the Python bridge for emission.
//! Each payload type corresponds to a family of synthetic tasks.
//! The generic `FamilyPayload` replaces the family-specific payloads;
//! the family-specific ones are retained for backward compatibility
//! but are deprecated.

use crate::task_spec::{SyntheticTaskSpec, TaskOp};

/// Default random seed for deterministic weight generation.
pub const DEFAULT_SEED: u64 = 42;

/// Current bridge protocol version. Bumped when the payload schema
/// changes in a way that breaks backward compatibility.
pub const BRIDGE_VERSION: u32 = 1;

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

        let effective_dtype = dtype_override.map(|s| s.to_string()).unwrap_or(spec_dtype);

        Ok(Self {
            bridge_version: BRIDGE_VERSION,
            command: "emit_linear_projection".into(),
            task_name: spec.name.clone(),
            family: spec.family.clone(),
            input_dim,
            output_dim,
            batch_size,
            dtype: effective_dtype.clone(),
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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
        let (vocab_size, embed_dim, num_groups, lut_bitwidth, batch_size, spec_dtype) = match &spec
            .op
        {
            TaskOp::LutProjection {
                vocab_size,
                embed_dim,
                num_groups,
                lut_bitwidth,
                batch_size,
                dtype,
            } => (*vocab_size, *embed_dim, *num_groups, *lut_bitwidth, *batch_size, dtype.clone()),
            _ => return Err("Expected LutProjection task for LutProjectionPayload".into()),
        };

        let effective_dtype = dtype_override.map(|s| s.to_string()).unwrap_or(spec_dtype);

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
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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

        let effective_dtype = dtype_override.map(|s| s.to_string()).unwrap_or(spec_dtype);

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
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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
                stateful: true, // DecodeStep manages KV cache state
            }],
        })
    }
}

/// MLP block payload for the Python bridge.
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
        let (input_dim, hidden_dim, output_dim, activation, batch_size, spec_dtype) = match &spec.op
        {
            TaskOp::MlpBlock {
                input_dim,
                hidden_dim,
                output_dim,
                activation,
                batch_size,
                dtype,
            } => (
                *input_dim,
                *hidden_dim,
                *output_dim,
                activation.clone(),
                *batch_size,
                dtype.clone(),
            ),
            _ => return Err("Expected MlpBlock task for MlpBlockPayload".into()),
        };

        let effective_dtype = dtype_override.map(|s| s.to_string()).unwrap_or(spec_dtype);

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
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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

        let effective_dtype = dtype_override.map(|s| s.to_string()).unwrap_or(spec_dtype);

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
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            compute_units: "CPU_AND_NE".into(),
            output_path: output_path.into(),
            seed: DEFAULT_SEED,
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
        serde_json::to_string(self).map_err(|e| format!("Failed to serialize FamilyPayload: {}", e))
    }

    /// Serialize this payload to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize FamilyPayload: {}", e))
    }
}
