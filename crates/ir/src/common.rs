//! Shared type definitions across SIR/AIR/MIR.
//!
//! This module centralizes types that are used by multiple IR levels,
//! reducing duplication and ensuring consistency.

use serde::{Deserialize, Serialize};

// ─── Data Types ───────────────────────────────────────────────────

/// MIL data type enum shared across all IR levels.
///
/// Sprint 58 (S58.2): MilDtypeRepr was removed. All IR levels now
/// use this single unified type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MilDtype {
    Fp16,
    Fp32,
    Int32,
    UInt8,
    Bool,
    Fp64,
    Int8,
    Int16,
}

// ─── Compute Unit Hints ──────────────────────────────────────────

/// Compute unit hint for MIR nodes and PIR packages.
///
/// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs
/// to become the unified compute unit representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeUnitHint {
    CPUAndNE,
    CPUAndGPU,
    CPUOnly,
    All,
}

impl ComputeUnitHint {
    /// Parse compute unit hint from the string representation used in
    /// task specs, bridge payloads, and shard template seeds.
    ///
    /// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs.
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s {
            "CPU_AND_NE" | "CPUAndNE" => Some(ComputeUnitHint::CPUAndNE),
            "CPU_AND_GPU" | "CPUAndGPU" => Some(ComputeUnitHint::CPUAndGPU),
            "CPU_ONLY" | "CPUOnly" => Some(ComputeUnitHint::CPUOnly),
            "ALL" | "All" => Some(ComputeUnitHint::All),
            _ => None,
        }
    }

    /// Returns the Core ML compatible string for this compute unit setting.
    ///
    /// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs.
    pub fn to_coreml_string(&self) -> &'static str {
        match self {
            ComputeUnitHint::CPUAndNE => "CPU_AND_NE",
            ComputeUnitHint::CPUAndGPU => "CPU_AND_GPU",
            ComputeUnitHint::CPUOnly => "CPU_ONLY",
            ComputeUnitHint::All => "ALL",
        }
    }
}

// ─── Shared Traits ───────────────────────────────────────────────

/// Trait for IR node identifiers.
///
/// All IR levels (SIR, AIR, MIR) use string-based node IDs.
/// This trait provides a uniform interface for common operations
/// on node IDs across the IR stack.
pub trait IrNodeId:
    std::fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Serialize + serde::de::DeserializeOwned
{
    /// Returns the string representation of this node ID.
    fn as_str(&self) -> &str;

    /// Construct a node ID from a string.
    fn from_string(s: String) -> Self;
}

/// Minimal trait for common graph operations across IR levels.
///
/// Each IR level (SIR, AIR, MIR) implements this trait to provide
/// a uniform interface for graph traversal and analysis.
pub trait IrGraph {
    /// The node ID type used by this IR level.
    type NodeId: IrNodeId;

    /// Returns the input node IDs of this graph.
    fn inputs(&self) -> &[Self::NodeId];

    /// Returns the output node IDs of this graph.
    fn outputs(&self) -> &[Self::NodeId];

    /// Returns the total number of nodes in this graph.
    fn node_count(&self) -> usize;
}
