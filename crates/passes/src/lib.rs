//! ANE Compiler Pass Pipeline
//!
//! Collection of compilation passes that transform the IR
//! through the various levels of abstraction.

pub mod canonicalize;
pub mod cpu_only_ops;
pub mod dtype_constraints;
pub mod knowledge_query;
pub mod legality_rewrite;
pub mod mil_lower;
pub mod op_constraints;
pub mod placement_validate;
pub mod precision_policy;
pub mod risk_annotate;
pub mod role_mir;
pub mod shard_plan;
pub mod state_topology;
pub mod staticize;
