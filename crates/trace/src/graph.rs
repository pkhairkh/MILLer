//! Traced computation graph representation.
//!
//! A `TracedGraph` is the intermediate representation produced by
//! torch.fx tracing and consumed by the SIR construction pipeline.
//! It captures the full computational structure of a transformers model
//! in a format that can be validated against ANE constraints before
//! being lowered into MILLer's IR stack.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default RoPE theta value (standard transformer: 10,000).
fn default_rope_theta() -> f64 {
    10_000.0
}

/// A complete traced computation graph from a transformers model.
///
/// This is the JSON-serializable format produced by the Python tracing
/// module (`python/trace_model.py`) and consumed by the Rust-side
/// `build_sir_from_trace()` function.
///
/// # Fully Dynamic Tracing
///
/// All feature detection (norm type, RoPE usage, GQA config) is derived
/// from the model's actual structure at runtime — no model_type heuristics
/// or hardcoded model lists are used. The `discovered_features` field
/// records how each feature was detected, providing an audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedGraph {
    /// Model identifier (HuggingFace ID or local path).
    pub model_id: String,

    /// Model architecture type (e.g., "GPT2LMHeadModel", "LlamaForCausalLM").
    pub architecture: String,

    /// Transformers library version used during tracing.
    pub transformers_version: String,

    /// Torch version used during tracing.
    pub torch_version: String,

    /// Configuration of the traced model.
    pub model_config: ModelConfig,

    /// Features discovered dynamically during tracing.
    ///
    /// This records how each feature (norm type, RoPE, GQA) was detected,
    /// providing an audit trail and validation signal. Features are detected
    /// by (in order of reliability):
    /// 1. Module type inspection (isinstance checks on actual nn.Module objects)
    /// 2. Config field presence (rms_norm_eps, rope_theta, etc.)
    /// 3. Structural detection (weight-without-bias patterns for RMSNorm)
    #[serde(default)]
    pub discovered_features: DiscoveredFeatures,

    /// Ordered list of computation nodes.
    pub nodes: Vec<TracedNode>,

    /// Named weight tensors (parameter name → shape + dtype).
    pub weights: HashMap<String, WeightInfo>,

    /// Mapping from fx node name to HuggingFace parameter names.
    /// Key: torch.fx node name (e.g., "linear1").
    /// Value: {"module_path": "model.layers.0.self_attn.q_proj",
    ///        "weight": "model.layers.0.self_attn.q_proj.weight",
    ///        "bias": "model.layers.0.self_attn.q_proj.bias" | null}
    #[serde(default)]
    pub weight_name_map: HashMap<String, WeightNameMapEntry>,

    /// Path to the HuggingFace model cache snapshot directory containing safetensors.
    #[serde(default)]
    pub model_cache_dir: Option<String>,

    /// Paths to safetensors files in the cache directory.
    #[serde(default)]
    pub safetensors_files: Vec<String>,

    /// Input tensor specifications.
    pub inputs: Vec<TensorSpec>,

    /// Output tensor specifications.
    pub outputs: Vec<TensorSpec>,

    /// KV-cache state declarations (if with_kv_cache was enabled).
    pub state_declarations: Vec<StateDeclaration>,

    /// Metadata about the tracing process.
    pub trace_metadata: TraceMetadata,
}

/// Model configuration extracted from the transformers config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_key_value_heads: Option<usize>,
    /// Number of hidden layers (transformer blocks).
    pub num_hidden_layers: usize,
    /// Intermediate size (MLP hidden dimension).
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Layer normalization epsilon.
    pub layer_norm_epsilon: f64,
    /// Activation function used in MLP ("gelu", "silu", "relu", etc.).
    pub hidden_act: String,
    /// Whether the model uses RoPE (Rotary Position Embeddings).
    pub uses_rope: bool,
    /// RoPE base frequency (theta). Default: 10,000 (standard transformer).
    /// Models like Qwen3 use 1,000,000; Llama uses 500,000.
    /// Read from HuggingFace config `rope_theta` field.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Whether the model uses QK normalization (per-head RMSNorm on Q and K).
    /// Models like Qwen3 use QK norm; most Llama/GPT-2 models do not.
    /// Detected by the presence of `q_norm`/`k_norm` weights in the model.
    #[serde(default)]
    pub has_qk_norm: bool,
    /// Whether the model uses RMSNorm (vs LayerNorm).
    pub uses_rms_norm: bool,
    /// Whether the model uses GQA (Grouped Query Attention).
    pub uses_gqa: bool,
    /// Model type identifier from HuggingFace config.
    pub model_type: String,
    /// Which Auto class was used to load the model.
    /// "causal_lm" = AutoModelForCausalLM, "seq2seq_lm" = AutoModelForSeq2SeqLM,
    /// "decoder_only" = extracted decoder from multimodal model.
    #[serde(default)]
    pub model_class: String,
    /// Whether the original model is encoder-decoder architecture.
    /// For seq2seq models, the traced graph represents the decoder path
    /// (the autoregressive generation path that runs on ANE).
    #[serde(default)]
    pub is_encoder_decoder: bool,
    /// Dimension per attention head. When `None`, derived as
    /// `hidden_size / num_attention_heads` (the common case for most models).
    /// Some models (Qwen3, etc.) have head_dim != hidden_size / num_heads
    /// because the q/k/v projection output dims are num_heads * head_dim
    /// which may differ from hidden_size.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

/// A single node in the traced computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedNode {
    /// Unique node identifier (from torch.fx).
    pub id: String,

    /// The operation performed by this node.
    pub op: TracedOp,

    /// Human-readable name (from torch.fx node name).
    pub name: String,

    /// Input node IDs.
    pub inputs: Vec<String>,

    /// Output tensor shape (concrete, after tracing).
    pub output_shape: TensorShape,

    /// Whether this node represents a weight/parameter load.
    pub is_parameter: bool,

    /// Optional: the original PyTorch module path (e.g., "model.layers.0.self_attn.q_proj").
    pub module_path: Option<String>,
}

/// Operations in the traced graph.
///
/// These are the canonical set of operations that can appear in a
/// traced transformers model. They are mapped 1:1 or N:1 to SIR ops
/// during `build_sir_from_trace()`.
///
/// # Serialization Format
///
/// The Python tracer (`trace_model.py`) uses internally-tagged JSON:
/// `{"type": "Linear", "in_features": 1024, ...}`. The `#[serde(tag = "type")`
/// attribute matches this format so Rust can deserialize directly.
///
/// Extra fields emitted by Python (e.g., `_module_path`, `_module_type`,
/// `_detection_method`) are silently ignored by `#[serde(deny_unknown_fields)]`
/// being absent — serde's default is to ignore unknown fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TracedOp {
    // ─── High-Level Transformer Ops (decompose_at_trace = false) ───
    /// Full attention block (QKV projection + attention + output projection).
    AttentionBlock {
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        #[serde(default)]
        use_sdpa: bool,
        #[serde(default)]
        has_qk_norm: bool,
    },
    /// MLP / feed-forward block.
    MlpBlock { input_dim: usize, hidden_dim: usize, output_dim: usize, activation: String },
    /// RoPE (Rotary Position Embedding) application.
    RopeTransform { head_dim: usize, max_seq_len: usize },
    /// RMSNorm layer.
    RmsNorm { hidden_size: usize, epsilon: f64 },

    // ─── Primitive Ops (after decomposition) ───────────────────────
    /// Linear projection: y = x @ W^T + b
    Linear {
        #[serde(default)]
        in_features: usize,
        #[serde(default)]
        out_features: usize,
        #[serde(default)]
        has_bias: bool,
    },
    /// Matrix multiplication: C = A @ B
    MatMul {
        #[serde(default)]
        a_shape: TensorShape,
        #[serde(default)]
        b_shape: TensorShape,
    },
    /// Embedding lookup (vocab_size × embed_dim).
    Embedding { vocab_size: usize, embed_dim: usize },
    /// Layer normalization.
    LayerNorm { normalized_shape: Vec<usize>, epsilon: f64 },
    /// Scaled dot-product attention.
    ScaledDotProductAttention { scale: f64 },
    /// Softmax along an axis.
    Softmax { axis: isize },
    /// GELU activation.
    Gelu { approximate: String },
    /// SiLU (Swish) activation: x * sigmoid(x).
    Silu,
    /// ReLU activation.
    Relu,
    /// Identity / no-op (contiguous, size query, etc.).
    /// Python tracer emits `"type": "Identity"` for no-ops.
    Identity,
    /// Reshape operation.
    Reshape { target_shape: Vec<usize> },
    /// Transpose / permute dimensions.
    Transpose { perm: Vec<usize> },
    /// Concatenate tensors along an axis.
    Concat { axis: usize },
    /// Split tensor along an axis.
    Split {
        /// Axis to split on. Python may emit negative values (e.g., -1 for last axis).
        axis: isize,
        num_splits: usize,
    },
    /// Slice operation.
    Slice { begin: Vec<i64>, end: Vec<i64>, stride: Vec<i64> },
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Reciprocal square root (1/sqrt(x)).
    Rsqrt,
    /// Cast to a different dtype.
    Cast { target_dtype: String },
    /// Tanh activation.
    Tanh,
    /// Sigmoid activation.
    Sigmoid,
    /// Exp (e^x).
    Exp,
    /// Cosine.
    Cos,
    /// Sine.
    Sin,
    /// Gather operation.
    Gather { axis: isize },
    /// Index select operation.
    IndexSelect { axis: isize },
    /// Where (conditional select).
    Where,
    /// Expand dimensions (unsqueeze). Python emits `"type": "ExpandDims"`.
    ExpandDims {
        #[serde(default)]
        axis: Vec<isize>,
    },
    /// Squeeze dimensions. Python emits `"type": "Squeeze"`.
    Squeeze {
        #[serde(default)]
        axis: Vec<isize>,
    },
    /// KV-cache read.
    KvCacheRead { layer_idx: usize, head_dim: usize, num_heads: usize },
    /// KV-cache write.
    KvCacheWrite { layer_idx: usize, head_dim: usize, num_heads: usize },
    /// Placeholder (model input).
    Placeholder,
    /// Output node.
    Output,
    /// Get item from a tuple/list.
    GetItem { index: usize },
    /// An operation not yet mapped to a specific TracedOp variant.
    Unknown {
        #[serde(default)]
        op_name: String,
        #[serde(default)]
        target: String,
    },
}

/// Tensor shape with dtype information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub dtype: String,
}

impl TensorShape {
    /// The rank (number of dimensions) of this tensor.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Whether this shape is compatible with ANE requirements (rank ≤ 5).
    pub fn ane_compatible(&self) -> bool {
        self.rank() <= 5
    }
}

/// Weight tensor metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightInfo {
    /// Shape of the weight tensor.
    pub shape: Vec<usize>,
    /// Data type of the weight.
    pub dtype: String,
    /// Optional: file path to the weight data (for large models).
    pub data_path: Option<String>,
    /// Optional: whether this is a quantized weight.
    pub quantized: Option<QuantizedWeightInfo>,
}

/// Entry in the weight_name_map: maps a torch.fx node name to HuggingFace parameter names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightNameMapEntry {
    /// PyTorch module path (e.g., "model.layers.0.self_attn.q_proj").
    pub module_path: String,
    /// HuggingFace weight parameter name (e.g., "model.layers.0.self_attn.q_proj.weight").
    /// `None` for modules that have parameters but no `.weight` attribute
    /// (e.g., norm layers with only a single parameter not named "weight").
    pub weight: Option<String>,
    /// HuggingFace bias parameter name, if the module has a bias.
    pub bias: Option<String>,
}

/// Quantized weight metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedWeightInfo {
    /// Quantization scheme ("palettized", "linear_quantized").
    pub scheme: String,
    /// Bit width (1, 2, 3, 4, 6, or 8 for palettized).
    pub bit_width: usize,
}

/// Input/output tensor specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: TensorShape,
}

/// KV-cache state declaration for stateful models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDeclaration {
    /// State identifier (e.g., "kv_cache_layer_0_key").
    pub state_id: String,
    /// Shape of the state tensor.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: String,
    /// Which layer owns this state.
    pub layer_idx: usize,
    /// Whether this is a key or value cache.
    pub is_key: bool,
}

/// Features discovered dynamically during model tracing.
///
/// This struct records what features were found and how they were detected,
/// providing an audit trail that validates the fully-dynamic tracing approach.
/// No model_type string matching is used — features are discovered from the
/// model's actual structure at runtime.
///
/// Detection methods (in order of reliability):
/// 1. `module_type_inspection` — isinstance checks on actual nn.Module objects
/// 2. `config_field_presence` — config fields like rms_norm_eps, rope_theta
/// 3. `structural_detection` — patterns like weight-without-bias for RMSNorm
/// 4. `config_field_comparison` — structural comparisons like num_kv_heads < num_heads
/// 5. `function_call_inspection` — torch.nn.functional.rms_norm detected in call_function
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveredFeatures {
    /// Norm types encountered during module walk (e.g., ["RMSNorm", "LayerNorm"]).
    #[serde(default)]
    pub norm_types_encountered: Vec<String>,

    /// Whether a Rotary Embedding module was found.
    #[serde(default)]
    pub has_rope_module: bool,

    /// Attention module class names found (e.g., ["SdpaAttention", "LlamaAttention"]).
    #[serde(default)]
    pub attention_module_types: Vec<String>,

    /// MLP module class names found (e.g., ["LlamaMLP", "Qwen2MLP"]).
    #[serde(default)]
    pub mlp_module_types: Vec<String>,

    /// Number of nn.Linear modules found.
    #[serde(default)]
    pub linear_count: usize,

    /// Number of nn.Embedding modules found.
    #[serde(default)]
    pub embedding_count: usize,

    /// Whether the model uses Grouped Query Attention.
    #[serde(default)]
    pub uses_gqa: bool,

    /// How each feature was detected (feature_name → detection_method).
    #[serde(default)]
    pub detection_methods: HashMap<String, String>,
}

/// Metadata about the tracing process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    /// Timestamp of the trace.
    pub timestamp: String,
    /// Duration of the trace in seconds.
    pub trace_duration_secs: f64,
    /// Number of nodes in the traced graph.
    pub num_nodes: usize,
    /// Number of parameters in the model.
    pub num_parameters: usize,
    /// Total parameter size in bytes.
    pub parameter_bytes: usize,
    /// Whether decomposition was applied during tracing.
    pub decomposed: bool,
    /// Any warnings produced during tracing.
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── TensorShape ─────────────────────────────────────────────────

    #[test]
    fn test_tensor_shape_rank() {
        let s0 = TensorShape { dims: vec![], dtype: "fp16".to_string() };
        let s1 = TensorShape { dims: vec![128], dtype: "fp16".to_string() };
        let s2 = TensorShape { dims: vec![1, 512], dtype: "fp16".to_string() };
        let s3 = TensorShape { dims: vec![4, 64, 128], dtype: "fp32".to_string() };
        let s5 = TensorShape { dims: vec![1, 2, 3, 4, 5], dtype: "fp16".to_string() };
        let s6 = TensorShape { dims: vec![1, 2, 3, 4, 5, 6], dtype: "fp16".to_string() };

        assert_eq!(s0.rank(), 0);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s2.rank(), 2);
        assert_eq!(s3.rank(), 3);
        assert_eq!(s5.rank(), 5);
        assert_eq!(s6.rank(), 6);
    }

    #[test]
    fn test_tensor_shape_num_elements() {
        let s1d = TensorShape { dims: vec![100], dtype: "fp16".to_string() };
        let s2d = TensorShape { dims: vec![4, 64], dtype: "fp16".to_string() };
        let s3d = TensorShape { dims: vec![2, 3, 4], dtype: "fp32".to_string() };

        assert_eq!(s1d.num_elements(), 100);
        assert_eq!(s2d.num_elements(), 256);
        assert_eq!(s3d.num_elements(), 24);
    }

    #[test]
    fn test_tensor_shape_ane_compatible() {
        let s0 = TensorShape { dims: vec![], dtype: "fp16".to_string() };
        let s5 = TensorShape { dims: vec![1, 2, 3, 4, 5], dtype: "fp16".to_string() };
        let s6 = TensorShape { dims: vec![1, 2, 3, 4, 5, 6], dtype: "fp16".to_string() };

        assert!(s0.ane_compatible()); // rank 0 <= 5
        assert!(s5.ane_compatible()); // rank 5 <= 5
        assert!(!s6.ane_compatible()); // rank 6 > 5
    }

    #[test]
    fn test_tensor_shape_default() {
        let s = TensorShape::default();
        assert!(s.dims.is_empty());
        assert!(s.dtype.is_empty());
    }

    // ─── TracedOp serialization ──────────────────────────────────────

    #[test]
    fn test_traced_op_serialization() {
        // Test each major TracedOp variant via JSON roundtrip
        let ops: Vec<TracedOp> = vec![
            TracedOp::Linear { in_features: 1024, out_features: 512, has_bias: true },
            TracedOp::AttentionBlock {
                embed_dim: 768,
                num_heads: 12,
                head_dim: 64,
                use_sdpa: true,
                has_qk_norm: false,
            },
            TracedOp::MlpBlock {
                input_dim: 768,
                hidden_dim: 3072,
                output_dim: 768,
                activation: "silu".to_string(),
            },
            TracedOp::RmsNorm { hidden_size: 768, epsilon: 1e-6 },
            TracedOp::RopeTransform { head_dim: 64, max_seq_len: 2048 },
            TracedOp::Embedding { vocab_size: 50257, embed_dim: 768 },
            TracedOp::MatMul {
                a_shape: TensorShape { dims: vec![1, 64, 128], dtype: "fp16".to_string() },
                b_shape: TensorShape { dims: vec![128, 256], dtype: "fp16".to_string() },
            },
            TracedOp::Softmax { axis: -1 },
            TracedOp::Silu,
            TracedOp::Identity,
            TracedOp::Reshape { target_shape: vec![1, 64, 128] },
            TracedOp::Transpose { perm: vec![0, 2, 1] },
            TracedOp::Concat { axis: 1 },
            TracedOp::Split { axis: -1, num_splits: 2 },
            TracedOp::Slice { begin: vec![0, 0], end: vec![1, 128], stride: vec![1, 1] },
            TracedOp::Add,
            TracedOp::Mul,
            TracedOp::Div,
            TracedOp::Rsqrt,
            TracedOp::Cast { target_dtype: "fp32".to_string() },
            TracedOp::Tanh,
            TracedOp::Sigmoid,
            TracedOp::Exp,
            TracedOp::Cos,
            TracedOp::Sin,
            TracedOp::Gather { axis: 0 },
            TracedOp::IndexSelect { axis: 1 },
            TracedOp::Where,
            TracedOp::ExpandDims { axis: vec![1] },
            TracedOp::Squeeze { axis: vec![1] },
            TracedOp::KvCacheRead { layer_idx: 0, head_dim: 64, num_heads: 12 },
            TracedOp::KvCacheWrite { layer_idx: 0, head_dim: 64, num_heads: 12 },
            TracedOp::Placeholder,
            TracedOp::Output,
            TracedOp::GetItem { index: 0 },
            TracedOp::Unknown { op_name: "custom_op".to_string(), target: "my_fn".to_string() },
        ];

        for op in &ops {
            let json = serde_json::to_string(op).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            // Verify the "type" tag is present
            assert!(parsed.get("type").is_some(), "Missing 'type' tag for {:?}", op);

            // Roundtrip
            let deserialized: TracedOp = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2, "Roundtrip mismatch for {:?}", op);
        }
    }

    // ─── ModelConfig ─────────────────────────────────────────────────

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            num_hidden_layers: 24,
            intermediate_size: 2048,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            layer_norm_epsilon: 1e-6,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            rope_theta: 1000000.0,
            has_qk_norm: true,
            uses_rms_norm: true,
            uses_gqa: true,
            model_type: "qwen3".to_string(),
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
            head_dim: Some(128),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.hidden_size, 1024);
        assert_eq!(deserialized.num_attention_heads, 16);
        assert_eq!(deserialized.num_key_value_heads, Some(8));
        assert_eq!(deserialized.rope_theta, 1000000.0);
        assert_eq!(deserialized.has_qk_norm, true);
        assert_eq!(deserialized.head_dim, Some(128));
    }

    #[test]
    fn test_model_config_rope_theta_default() {
        // When rope_theta is omitted, it should default to 10_000.0
        let json = r#"{
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "layer_norm_epsilon": 1e-5,
            "hidden_act": "gelu",
            "uses_rope": false,
            "uses_rms_norm": false,
            "uses_gqa": false,
            "model_type": "gpt2"
        }"#;

        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert!((config.rope_theta - 10_000.0).abs() < f64::EPSILON);
    }

    // ─── DiscoveredFeatures ──────────────────────────────────────────

    #[test]
    fn test_discovered_features_default() {
        let df = DiscoveredFeatures::default();
        assert!(df.norm_types_encountered.is_empty());
        assert!(!df.has_rope_module);
        assert!(df.attention_module_types.is_empty());
        assert!(df.mlp_module_types.is_empty());
        assert_eq!(df.linear_count, 0);
        assert_eq!(df.embedding_count, 0);
        assert!(!df.uses_gqa);
        assert!(df.detection_methods.is_empty());
    }

    // ─── WeightInfo ──────────────────────────────────────────────────

    #[test]
    fn test_weight_info_serialization() {
        // Without quantized field
        let wi = WeightInfo {
            shape: vec![1024, 512],
            dtype: "fp16".to_string(),
            data_path: Some("/path/to/weight.bin".to_string()),
            quantized: None,
        };
        let json = serde_json::to_string(&wi).unwrap();
        let deserialized: WeightInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.shape, vec![1024, 512]);
        assert!(deserialized.quantized.is_none());

        // With quantized field
        let wi_q = WeightInfo {
            shape: vec![1024, 512],
            dtype: "fp16".to_string(),
            data_path: None,
            quantized: Some(QuantizedWeightInfo { scheme: "palettized".to_string(), bit_width: 4 }),
        };
        let json_q = serde_json::to_string(&wi_q).unwrap();
        let deserialized_q: WeightInfo = serde_json::from_str(&json_q).unwrap();
        assert!(deserialized_q.quantized.is_some());
        let q = deserialized_q.quantized.unwrap();
        assert_eq!(q.scheme, "palettized");
        assert_eq!(q.bit_width, 4);
    }

    // ─── WeightNameMapEntry ──────────────────────────────────────────

    #[test]
    fn test_weight_name_map_entry_serialization() {
        let entry = WeightNameMapEntry {
            module_path: "model.layers.0.self_attn.q_proj".to_string(),
            weight: Some("model.layers.0.self_attn.q_proj.weight".to_string()),
            bias: Some("model.layers.0.self_attn.q_proj.bias".to_string()),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: WeightNameMapEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.module_path, "model.layers.0.self_attn.q_proj");
        assert_eq!(deserialized.weight, Some("model.layers.0.self_attn.q_proj.weight".to_string()));
        assert_eq!(deserialized.bias, Some("model.layers.0.self_attn.q_proj.bias".to_string()));
    }

    // ─── StateDeclaration ────────────────────────────────────────────

    #[test]
    fn test_state_declaration_serialization() {
        let sd = StateDeclaration {
            state_id: "kv_cache_layer_0_key".to_string(),
            shape: vec![1, 32, 128],
            dtype: "fp16".to_string(),
            layer_idx: 0,
            is_key: true,
        };
        let json = serde_json::to_string(&sd).unwrap();
        let deserialized: StateDeclaration = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.state_id, "kv_cache_layer_0_key");
        assert_eq!(deserialized.shape, vec![1, 32, 128]);
        assert_eq!(deserialized.dtype, "fp16");
        assert_eq!(deserialized.layer_idx, 0);
        assert!(deserialized.is_key);
    }

    // ─── TracedGraph minimal deserialization ─────────────────────────

    #[test]
    fn test_traced_graph_minimal_deserialization() {
        let json = r#"{
            "model_id": "test-model",
            "architecture": "LlamaForCausalLM",
            "transformers_version": "4.40.0",
            "torch_version": "2.3.0",
            "model_config": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "intermediate_size": 3072,
                "vocab_size": 50257,
                "max_position_embeddings": 1024,
                "layer_norm_epsilon": 1e-5,
                "hidden_act": "gelu",
                "uses_rope": false,
                "uses_rms_norm": false,
                "uses_gqa": false,
                "model_type": "gpt2"
            },
            "nodes": [],
            "weights": {},
            "inputs": [],
            "outputs": [],
            "state_declarations": [],
            "trace_metadata": {
                "timestamp": "2025-01-01T00:00:00Z",
                "trace_duration_secs": 1.5,
                "num_nodes": 0,
                "num_parameters": 0,
                "parameter_bytes": 0,
                "decomposed": false,
                "warnings": []
            }
        }"#;

        let graph: TracedGraph = serde_json::from_str(json).unwrap();
        assert_eq!(graph.model_id, "test-model");
        assert_eq!(graph.architecture, "LlamaForCausalLM");
        assert!(graph.nodes.is_empty());
        assert!(graph.discovered_features.norm_types_encountered.is_empty());
    }

    // ─── TraceMetadata ───────────────────────────────────────────────

    #[test]
    fn test_trace_metadata_serialization() {
        let meta = TraceMetadata {
            timestamp: "2025-06-15T12:00:00Z".to_string(),
            trace_duration_secs: 3.1415,
            num_nodes: 42,
            num_parameters: 1_000_000,
            parameter_bytes: 2_000_000,
            decomposed: true,
            warnings: vec!["test warning".to_string()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: TraceMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.timestamp, "2025-06-15T12:00:00Z");
        assert!((deserialized.trace_duration_secs - 3.1415).abs() < f64::EPSILON);
        assert_eq!(deserialized.num_nodes, 42);
        assert_eq!(deserialized.warnings, vec!["test warning"]);
    }

    // ─── TracedNode ──────────────────────────────────────────────────

    #[test]
    fn test_traced_node_serialization() {
        let node = TracedNode {
            id: "node_0".to_string(),
            op: TracedOp::Linear { in_features: 768, out_features: 768, has_bias: true },
            name: "q_proj".to_string(),
            inputs: vec!["input_0".to_string()],
            output_shape: TensorShape { dims: vec![1, 512, 768], dtype: "fp16".to_string() },
            is_parameter: false,
            module_path: Some("model.layers.0.self_attn.q_proj".to_string()),
        };
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: TracedNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "node_0");
        assert!(matches!(deserialized.op, TracedOp::Linear { .. }));
        assert_eq!(deserialized.name, "q_proj");
        assert_eq!(deserialized.inputs, vec!["input_0"]);
        assert_eq!(deserialized.output_shape.dims, vec![1, 512, 768]);
        assert!(!deserialized.is_parameter);
        assert_eq!(deserialized.module_path, Some("model.layers.0.self_attn.q_proj".to_string()));
    }

    // ─── QuantizedWeightInfo ─────────────────────────────────────────

    #[test]
    fn test_quantized_weight_info_serialization() {
        let qi = QuantizedWeightInfo { scheme: "palettized".to_string(), bit_width: 4 };
        let json = serde_json::to_string(&qi).unwrap();
        let deserialized: QuantizedWeightInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.scheme, "palettized");
        assert_eq!(deserialized.bit_width, 4);
    }

    // ─── TensorSpec ──────────────────────────────────────────────────

    #[test]
    fn test_tensor_spec_serialization() {
        let spec = TensorSpec {
            name: "input_ids".to_string(),
            shape: TensorShape { dims: vec![1, 512], dtype: "int32".to_string() },
        };
        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: TensorSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "input_ids");
        assert_eq!(deserialized.shape.dims, vec![1, 512]);
    }
}
