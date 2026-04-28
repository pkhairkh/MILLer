//! Tracing configuration types.
//!
//! Controls how a transformers model is traced and what constraints
//! are applied during compilation.

use ane_ir::ane_target::AneFamily;
use serde::{Deserialize, Serialize};

/// Configuration for a tracing run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// The model to trace (HuggingFace model ID or local path).
    pub target: TraceTarget,

    /// Target ANE family for constraint-aware compilation.
    /// Defaults to A16 (the first family with reliable SDPA support).
    pub target_family: AneFamily,

    /// Whether to enforce ANE-only compilation (reject CPU-fallback ops).
    /// When false, ops that cannot run on ANE are flagged but not rejected.
    /// When true, the compilation fails if any op requires CPU fallback.
    pub ane_only: bool,

    /// Whether to decompose composite ops into ANE-faithful primitives
    /// during tracing (rather than during later passes).
    /// E.g., decompose AttentionBlock into QKV projection + SDPA + output
    /// projection at trace time, ensuring each sub-op is ANE-plannable.
    pub decompose_at_trace: bool,

    /// Input shapes for the model (batch_size, seq_len).
    /// Required for torch.fx tracing (symbolic shapes are not supported).
    pub input_shapes: Vec<InputShape>,

    /// Whether to include KV-cache state in the traced graph.
    /// Required for decode-step (autoregressive) compilation.
    pub with_kv_cache: bool,

    /// Maximum sequence length for static shape resolution.
    /// The ANE requires static shapes; this sets the upper bound.
    pub max_seq_len: usize,

    /// Dtype for the model weights and activations.
    /// "fp16" is the primary ANE compute format.
    pub dtype: String,

    /// Path to the Python tracing script.
    /// Defaults to "python/trace_model.py".
    pub trace_script: String,

    /// Path to the Python interpreter.
    pub python_path: String,

    /// Which Auto class to use for loading the model.
    /// "auto" = auto-detect from config (default)
    /// "causal_lm" = AutoModelForCausalLM
    /// "seq2seq_lm" = AutoModelForSeq2SeqLM
    /// "decoder_only" = extract decoder from multimodal model
    pub model_class: String,

    /// Additional tracing options forwarded to torch.fx.
    pub fx_options: FxTraceOptions,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            target: TraceTarget::HuggingFaceId("gpt2".to_string()),
            target_family: AneFamily::A16,
            ane_only: true,
            decompose_at_trace: true,
            input_shapes: vec![InputShape { batch_size: 1, seq_len: 32 }],
            with_kv_cache: false,
            max_seq_len: 2048,
            dtype: "fp16".to_string(),
            trace_script: "python/trace_model.py".to_string(),
            python_path: "python3".to_string(),
            model_class: "auto".to_string(),
            fx_options: FxTraceOptions::default(),
        }
    }
}

/// Target specification for tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceTarget {
    /// A HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf").
    HuggingFaceId(String),
    /// A local directory containing a transformers model.
    LocalPath(String),
    /// A pre-traced JSON graph file (skip the tracing step).
    PreTraced(String),
}

/// Input shape specification for tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputShape {
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Options for torch.fx tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxTraceOptions {
    /// Whether to trace with concrete args (symbolic vs concrete tracing).
    pub concrete_args: bool,

    /// Whether to flatten the traced graph (merge submodules).
    pub flatten: bool,

    /// Custom leaf modules that should not be traced through.
    pub leaf_modules: Vec<String>,

    /// Whether to suppress shape assertions during tracing.
    pub suppress_shape_assertions: bool,
}

impl Default for FxTraceOptions {
    fn default() -> Self {
        Self {
            concrete_args: true,
            flatten: true,
            leaf_modules: vec![
                "torch.nn.functional.embedding".to_string(),
                "torch.nn.functional.scaled_dot_product_attention".to_string(),
            ],
            suppress_shape_assertions: false,
        }
    }
}
