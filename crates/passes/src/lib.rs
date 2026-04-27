//! ANE Compiler Pass Pipeline
//!
//! Collection of compilation passes that transform the IR
//! through the various levels of abstraction.

pub mod canonicalize;
pub mod staticize;
pub mod state_topology;
pub mod shard_plan;
pub mod precision_policy;
pub mod legality_rewrite;
pub mod risk_annotate;
pub mod mil_lower;
pub mod knowledge_query;
pub mod role_mir;
