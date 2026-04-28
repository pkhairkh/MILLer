//! Transformers Model Tracing and ANE-Faithful Graph Compilation
//!
//! This crate extends MILLer to trace HuggingFace transformers models via
//! torch.fx and compile them into ANE-faithful computation graphs with
//! per-family constraint enforcement.
//!
//! # Architecture
//!
//! The tracing pipeline flows:
//! ```text
//! HuggingFace Model → torch.fx trace → JSON export → TracedGraph
//!   → SIR construction → AIR (ANE-legal) → MIR → Core ML emission
//! ```
//!
//! # Ad-Hoc Tracing (No Registry Required)
//!
//! The SIR construction is driven entirely by the model's `AutoConfig`
//! extracted during tracing. The config flags (`uses_rms_norm`, `uses_gqa`,
//! `uses_rope`, `hidden_act`) determine how composite ops decompose into
//! ANE-faithful primitives. This means **any** HuggingFace model works
//! ad-hoc without a hardcoded registry — Qwen3, Qwen3.5, Llama-4, or
//! any future architecture that follows the standard transformer pattern.
//!
//! The `ModelRegistry` is deprecated — it was never required by the build
//! pipeline and is kept only for backward compatibility. All tracing now
//! works fully ad-hoc via `AutoConfig`/`AutoModel`.
//!
//! # ANE-Faithful Compilation
//!
//! Unlike naive conversion (which may produce graphs that the ANE rejects at
//! runtime), ANE-faithful compilation enforces constraints *during* lowering:
//! - Per-family operation support checks (A11 through A18)
//! - Tensor dimension validation against hal_params
//! - Version-aware decomposition (e.g., SDPA only on A16+)
//! - Dynamic shape elimination (ANE requires static shapes)
//!
//! # Strategy-Driven Optimization
//!
//! The compilation pipeline uses strategy-driven optimization to adapt
//! to the target hardware and model characteristics:
//!
//! - **Normalization stabilization**: Pre-scale insertion for fp16-safe RMSNorm
//! - **Static tables**: Pre-computed RoPE (sin/cos), causal mask, and identity
//!   tables embedded as fp16 constants in the Core ML graph
//! - **Masked-blend KV cache**: Active context in contiguous suffix,
//!   new K/V written by masked blending instead of scatter
//! - **Mixed quantization**: Different bit-widths for different weight types
//!   (conservative for Q/K, aggressive for MLP, 1-bit for masks)
//! - **On-device sampler**: Dedicated MLProgram for temperature/min-p/top-k
//!   sampling, keeping the decode loop fully on-device
//! - **Conditional IO model**: Shared embedding/LM-head weights with a mode
//!   switch, halving memory for tied-weight models

pub mod config;
pub mod graph;
pub mod discovery;
pub mod sir_build;
pub mod versioned;
pub mod subprocess;

pub use config::{TraceConfig, TraceTarget};
pub use graph::{TracedGraph, TracedNode, TracedOp, TensorShape, ModelConfig};
pub use sir_build::build_sir_from_trace;
pub use versioned::{VersionedCompiler, VersionedCompileResult, AnceFaithfulnessReport};
