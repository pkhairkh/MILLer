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
//! # ANE-Faithful Compilation
//!
//! Unlike naive conversion (which may produce graphs that the ANE rejects at
//! runtime), ANE-faithful compilation enforces constraints *during* lowering:
//! - Per-family operation support checks (A11 through A18)
//! - Tensor dimension validation against hal_params
//! - Version-aware decomposition (e.g., SDPA only on A16+)
//! - Dynamic shape elimination (ANE requires static shapes)
//!
//! # Supported Model Architectures
//!
//! The model registry maps HuggingFace model types to known ANE-faithful
//! decomposition patterns:
//! - GPT-2 family (causal LM)
//! - LLaMA / Qwen family (RoPE-based)
//! - BERT family (encoder-only)
//! - Phi family (small form factor)
//!
//! Custom architectures can be registered via `ModelRegistry::register()`.

pub mod config;
pub mod graph;
pub mod registry;
pub mod sir_build;
pub mod versioned;
pub mod subprocess;

pub use config::{TraceConfig, TraceTarget};
pub use graph::{TracedGraph, TracedNode, TracedOp, TensorShape};
pub use registry::{ModelRegistry, ModelPattern, TransformerLayerKind};
pub use sir_build::build_sir_from_trace;
pub use versioned::{VersionedCompiler, VersionedCompileResult, AnceFaithfulnessReport};
