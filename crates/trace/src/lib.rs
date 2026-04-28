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
//! # Config-Driven Decomposition (No Registry Required)
//!
//! The SIR construction is driven entirely by the model's `AutoConfig`
//! extracted during tracing. The config flags (`uses_rms_norm`, `uses_gqa`,
//! `uses_rope`, `hidden_act`) determine how composite ops decompose into
//! ANE-faithful primitives. This means **any** HuggingFace model works
//! ad-hoc without a hardcoded registry — Qwen3, Qwen3.5, Llama-4, or
//! any future architecture that follows the standard transformer pattern.
//!
//! The `ModelRegistry` is available as an optional override mechanism for
//! edge cases, but it is NOT required for the trace pipeline to work.
//!
//! # ANE-Faithful Compilation
//!
//! Unlike naive conversion (which may produce graphs that the ANE rejects at
//! runtime), ANE-faithful compilation enforces constraints *during* lowering:
//! - Per-family operation support checks (A11 through A18)
//! - Tensor dimension validation against hal_params
//! - Version-aware decomposition (e.g., SDPA only on A16+)
//! - Dynamic shape elimination (ANE requires static shapes)

pub mod config;
pub mod graph;
pub mod registry;
pub mod sir_build;
pub mod versioned;
pub mod subprocess;

pub use config::{TraceConfig, TraceTarget};
pub use graph::{TracedGraph, TracedNode, TracedOp, TensorShape, ModelConfig};
pub use registry::{ModelRegistry, ModelPattern, TransformerLayerKind};
pub use sir_build::build_sir_from_trace;
pub use versioned::{VersionedCompiler, VersionedCompileResult, AnceFaithfulnessReport};
