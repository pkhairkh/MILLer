//! ANE Compiler Pass Pipeline
//!
//! Collection of compilation passes that transform the IR
//! through the various levels of abstraction.
//!
//! ## Passes Derived from pkhairkh/qwen3-coreml-palettized
//!
//! The following passes implement techniques from the qwen3-coreml-palettized
//! deployment stack, adapted into MILLer's IR pass pipeline:
//!
//! - **slanc_scales**: SLaNC pre-scale computation for fp16 numerical stabilization
//! - **static_tables**: Pre-compute RoPE, causal mask, and identity tables as constants
//! - **kv_cache_rewrite**: Transform naive KV cache to reverse ring-buffer layout
//! - **palettize_weights**: Annotate weight tensors with mixed quantization strategies

pub mod canonicalize;
pub mod cpu_only_ops;
pub mod dtype_constraints;
pub mod knowledge_query;
pub mod kv_cache_rewrite;
pub mod legality_rewrite;
pub mod mil_lower;
pub mod op_constraints;
pub mod palettize_weights;
pub mod placement_validate;
pub mod precision_policy;
pub mod risk_annotate;
pub mod role_mir;
pub mod shard_plan;
pub mod slanc_scales;
pub mod state_topology;
pub mod staticize;
pub mod static_tables;
