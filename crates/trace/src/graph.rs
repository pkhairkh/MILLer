//! Traced computation graph representation.
//!
//! A `TracedGraph` is the intermediate representation produced by
//! torch.fx tracing and consumed by the SIR construction pipeline.
//! It captures the full computational structure of a transformers model
//! in a format that can be validated against ANE constraints before
//! being lowered into MILLer's IR stack.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete traced computation graph from a transformers model.
///
/// This is the JSON-serializable format produced by the Python tracing
/// module (`python/trace_model.py`) and consumed by the Rust-side
/// `build_sir_from_trace()` function.
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

    /// Ordered list of computation nodes.
    pub nodes: Vec<TracedNode>,

    /// Named weight tensors (parameter name → shape + dtype).
    pub weights: HashMap<String, WeightInfo>,

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
    /// Whether the model uses RMSNorm (vs LayerNorm).
    pub uses_rms_norm: bool,
    /// Whether the model uses GQA (Grouped Query Attention).
    pub uses_gqa: bool,
    /// Model type identifier from HuggingFace config.
    pub model_type: String,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracedOp {
    // ─── High-Level Transformer Ops (decompose_at_trace = false) ───
    /// Full attention block (QKV projection + attention + output projection).
    AttentionBlock {
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        use_sdpa: bool,
    },
    /// MLP / feed-forward block.
    MlpBlock {
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: String,
    },
    /// RoPE (Rotary Position Embedding) application.
    RopeTransform {
        head_dim: usize,
        max_seq_len: usize,
    },
    /// RMSNorm layer.
    RmsNorm {
        hidden_size: usize,
        epsilon: f64,
    },

    // ─── Primitive Ops (after decomposition) ───────────────────────
    /// Linear projection: y = x @ W^T + b
    Linear {
        in_features: usize,
        out_features: usize,
        has_bias: bool,
    },
    /// Matrix multiplication: C = A @ B
    MatMul {
        a_shape: TensorShape,
        b_shape: TensorShape,
    },
    /// Embedding lookup (vocab_size × embed_dim).
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },
    /// Layer normalization.
    LayerNorm {
        normalized_shape: Vec<usize>,
        epsilon: f64,
    },
    /// Scaled dot-product attention.
    ScaledDotProductAttention {
        scale: f64,
    },
    /// Softmax along an axis.
    Softmax { axis: isize },
    /// GELU activation.
    Gelu { approximate: String },
    /// SiLU (Swish) activation: x * sigmoid(x).
    Silu,
    /// ReLU activation.
    Relu,
    /// Reshape operation.
    Reshape { target_shape: Vec<usize> },
    /// Transpose / permute dimensions.
    Transpose { perm: Vec<usize> },
    /// Concatenate tensors along an axis.
    Concat { axis: usize },
    /// Split tensor along an axis.
    Split { axis: usize, num_splits: usize },
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
    /// KV-cache read.
    KvCacheRead {
        layer_idx: usize,
        head_dim: usize,
        num_heads: usize,
    },
    /// KV-cache write.
    KvCacheWrite {
        layer_idx: usize,
        head_dim: usize,
        num_heads: usize,
    },
    /// Placeholder (model input).
    Placeholder,
    /// Output node.
    Output,
    /// Get item from a tuple/list.
    GetItem { index: usize },
    /// An operation not yet mapped to a specific TracedOp variant.
    Unknown { op_name: String, target: String },
}

/// Tensor shape with dtype information.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
