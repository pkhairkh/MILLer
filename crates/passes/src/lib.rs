//! ANE Compiler Pass Pipeline
//!
//! Collection of compilation passes that transform the IR
//! through the various levels of abstraction.
//!
//! ## Strategy-Driven Optimization Passes
//!
//! The following passes implement strategy-driven optimizations that
//! adapt to the target hardware and model characteristics:
//!
//! - **slanc_scales**: Normalization stabilization via pre-scale insertion for fp16 safety
//! - **static_tables**: Pre-compute RoPE, causal mask, and identity tables as constants
//! - **kv_cache_rewrite**: ~~Transform naive KV cache to masked-blend layout~~ DEPRECATED — generates ANE-illegal `Where` ops; KV masking is now handled by arithmetic masks in `legality_rewrite`
//! - **palettize_weights**: Annotate weight tensors with mixed quantization strategies

pub mod canonicalize;
pub mod cpu_only_ops;
pub mod dtype_constraints;
pub mod knowledge_query;
#[deprecated(
    note = "Generates ANE-illegal Where ops. KV masking is now handled by arithmetic masks in legality_rewrite."
)]
pub(crate) mod kv_cache_rewrite;
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
pub mod static_tables;
pub mod staticize;

#[cfg(test)]
pub mod test_utils;
