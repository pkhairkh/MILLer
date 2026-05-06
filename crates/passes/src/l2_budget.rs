//! L2 Memory Budget Modeling and Legalization (T-P6-06)
//!
//! This module implements L2 memory budget modeling per AneFamily and
//! detects when individually legal ops collectively exceed the L2 budget.
//!
//! ## Background
//!
//! The ANE's L2 cache is a finite shared resource. Each op that executes
//! on the ANE consumes L2 cache for its input/output tensors. While any
//! single op may fit within the L2 budget, a sequence of ops (especially
//! in a fusion group) may collectively exceed the budget, causing ANEC
//! compile-time failures or runtime stalls.
//!
//! ANEC's own `L2Legalizer` pass and `L2FootprintCalc` perform similar
//! analysis. This module provides MILLer's equivalent, enabling early
//! detection at placement time rather than deferring to ANEC.
//!
//! ## Architecture
//!
//! 1. **L2 Footprint Estimation**: Each op's L2 footprint is computed
//!    from its tensor shapes and data type.
//! 2. **Budget Accumulation**: Footprints are summed across ops in a
//!    fusion group or sequential placement.
//! 3. **Overflow Detection**: When the total exceeds `AneHwLimits::total_l2_budget()`,
//!    a violation is reported.
//! 4. **Legalization** (future): Ops can be split or reordered to fit
//!    within the budget. This is a follow-up — ANEC uses `SpatialSplitter`,
//!    `BatchOrChannelSplitter`, and `L2Legalizer` for this purpose.

use ane_ir::ane_hw_limits::AneHwLimits;
use ane_ir::ane_target::AneRevision;
use ane_ir::mir::{MilDtype, MirGraph, MirOp};
use ane_ir::toproto::ToProto;
use std::fmt;

/// L2 memory budget violation.
#[derive(Debug, Clone)]
pub struct L2BudgetViolation {
    /// Total L2 footprint in bytes that was estimated.
    pub total_footprint: u64,
    /// L2 budget in bytes for this revision.
    pub budget: u64,
    /// Number of ops that contributed to the footprint.
    pub op_count: usize,
    /// The ANE revision being targeted.
    pub revision: AneRevision,
}

impl fmt::Display for L2BudgetViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "L2 budget exceeded: {} bytes estimated across {} ops, \
             but budget is {} bytes (revision {:?}). \
             Consider splitting the model or reducing tensor dimensions.",
            self.total_footprint, self.op_count, self.budget, self.revision
        )
    }
}

/// Result of an L2 budget check.
#[derive(Debug, Clone)]
pub struct L2BudgetCheckResult {
    /// Total estimated L2 footprint in bytes.
    pub total_footprint: u64,
    /// L2 budget in bytes for this revision.
    pub budget: u64,
    /// Per-op L2 footprint estimates (op name, estimated bytes).
    pub per_op_footprint: Vec<(String, u64)>,
    /// Whether the budget was exceeded.
    pub exceeded: bool,
}

impl L2BudgetCheckResult {
    /// Returns the L2 budget violation if the budget was exceeded.
    pub fn violation(&self, revision: AneRevision) -> Option<L2BudgetViolation> {
        if self.exceeded {
            Some(L2BudgetViolation {
                total_footprint: self.total_footprint,
                budget: self.budget,
                op_count: self.per_op_footprint.len(),
                revision,
            })
        } else {
            None
        }
    }
}

/// Estimate the L2 footprint (in bytes) for a single MirOp.
///
/// The L2 footprint is the sum of all tensor sizes (inputs + outputs)
/// that the op needs to be resident in L2 cache simultaneously.
/// This is a conservative estimate — it assumes all tensors are in L2
/// at the same time, which is the worst case.
///
/// For weight constants (MILConst), the footprint is the weight data size.
/// For compute ops, the footprint is the sum of all input and output tensor sizes.
pub fn estimate_op_l2_footprint(
    op: &MirOp,
    node_shapes: &std::collections::HashMap<String, Vec<usize>>,
) -> u64 {
    let bytes_per_element = |dtype: &MilDtype| -> u64 {
        match dtype {
            MilDtype::Fp16 => 2,
            MilDtype::Fp32 => 4,
            MilDtype::UInt8 => 1,
            MilDtype::Int32 => 4,
            MilDtype::Bool => 1,
            // T-P6-01: Conservative defaults for less common dtypes
            _ => 2, // Default to FP16 size (2 bytes)
        }
    };

    // Default element size (fp16 is the most common for ANE ops)
    let default_bpe: u64 = 2;

    match op {
        MirOp::MILConst { dtype, .. } => {
            // Weight constant: size = product of shape * bytes_per_element
            // If shape is unknown, estimate conservatively
            let bpe = bytes_per_element(dtype);
            let name = op.proto_output_name().to_string();
            let _ = &name; // suppress unused warning
            if let Some(shape) = node_shapes.get(&name) {
                let elements: u64 = shape.iter().product::<usize>() as u64;
                elements * bpe
            } else {
                // Unknown shape — skip (returns 0, underestimates)
                0
            }
        }
        _ => {
            // For compute ops, estimate footprint as sum of all input
            // and output tensor sizes.
            let mut total: u64 = 0;

            // Input references
            for input_ref in op.proto_input_refs() {
                if let Some(shape) = node_shapes.get(&input_ref) {
                    let elements: u64 = shape.iter().product::<usize>() as u64;
                    total += elements * default_bpe;
                }
            }

            // Output tensor
            let output_name = op.proto_output_name().to_string();
            if let Some(shape) = node_shapes.get(&output_name) {
                let elements: u64 = shape.iter().product::<usize>() as u64;
                total += elements * default_bpe;
            }

            total
        }
    }
}

/// Check the L2 budget for an entire MIR graph.
///
/// Computes the estimated L2 footprint for each op and checks whether
/// the total exceeds the budget for the given revision.
///
/// Note: This is a conservative check — it sums ALL op footprints,
/// which overestimates actual L2 usage since not all tensors are in
/// L2 simultaneously. A more accurate analysis would consider tensor
/// lifetimes and fusion groups, but this is sufficient for detecting
/// gross overages.
pub fn check_l2_budget(
    graph: &MirGraph,
    hw_limits: &AneHwLimits,
    node_shapes: &std::collections::HashMap<String, Vec<usize>>,
) -> L2BudgetCheckResult {
    let budget = hw_limits.total_l2_budget();
    let mut per_op_footprint: Vec<(String, u64)> = Vec::new();
    let mut total_footprint: u64 = 0;

    for node in &graph.nodes {
        let footprint = estimate_op_l2_footprint(&node.op, node_shapes);
        let op_name = node.op.proto_output_name().to_string();
        total_footprint += footprint;
        per_op_footprint.push((op_name, footprint));
    }

    let exceeded = total_footprint > budget;

    L2BudgetCheckResult { total_footprint, budget, per_op_footprint, exceeded }
}

/// Check the L2 budget for a single op (lightweight check).
///
/// Returns the estimated L2 footprint for the op and whether adding
/// it to the given cumulative footprint would exceed the budget.
pub fn check_op_l2_fit(
    op: &MirOp,
    cumulative_footprint: u64,
    hw_limits: &AneHwLimits,
    node_shapes: &std::collections::HashMap<String, Vec<usize>>,
) -> (u64, bool) {
    let op_footprint = estimate_op_l2_footprint(op, node_shapes);
    let budget = hw_limits.total_l2_budget();
    let would_exceed = cumulative_footprint + op_footprint > budget;
    (op_footprint, would_exceed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::ane_hw_limits::AneHwLimits;
    use ane_ir::ane_target::AneRevision;
    use ane_ir::mir::{MilDtype, MirNodeId, MirOp};
    use std::collections::HashMap;

    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    #[test]
    fn test_l2_budget_a11_is_small() {
        let hw_limits = AneHwLimits::for_revision(AneRevision::V4);
        // A11 has 1 NE with 32KB L2 = 32768 bytes total
        assert_eq!(hw_limits.total_l2_budget(), 32768);
    }

    #[test]
    fn test_l2_budget_a16_is_large() {
        let hw_limits = AneHwLimits::for_revision(AneRevision::V10);
        // A16 has 4 NEs with 256KB each = 1MB total
        assert_eq!(hw_limits.total_l2_budget(), 262144 * 4);
    }

    #[test]
    fn test_l2_budget_increases_with_revision() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        let a18 = AneHwLimits::for_revision(AneRevision::V19);
        assert!(a14.total_l2_budget() > a11.total_l2_budget());
        assert!(a16.total_l2_budget() > a14.total_l2_budget());
        assert!(a18.total_l2_budget() >= a16.total_l2_budget());
    }

    #[test]
    fn test_estimate_const_footprint() {
        let op = MirOp::MILConst {
            name: "weight".into(),
            value_path: "w.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let mut shapes = HashMap::new();
        shapes.insert("weight".to_string(), vec![128, 64]); // 8192 elements * 2 bytes = 16384
        let footprint = estimate_op_l2_footprint(&op, &shapes);
        assert_eq!(footprint, 16384);
    }

    #[test]
    fn test_estimate_const_footprint_int32() {
        let op = MirOp::MILConst {
            name: "embed".into(),
            value_path: "e.bin".into(),
            dtype: MilDtype::Int32,
        };
        let mut shapes = HashMap::new();
        shapes.insert("embed".to_string(), vec![1000, 1024]); // 1024000 elements * 4 bytes
        let footprint = estimate_op_l2_footprint(&op, &shapes);
        assert_eq!(footprint, 1000 * 1024 * 4);
    }

    #[test]
    fn test_estimate_linear_footprint() {
        let op = MirOp::MILLinear {
            name: "linear".into(),
            x: nid("input"),
            weight: "weight".into(),
            bias: None,
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![1, 128]);
        shapes.insert("linear".to_string(), vec![1, 64]);
        let footprint = estimate_op_l2_footprint(&op, &shapes);
        // input: 128 * 2 = 256, output: 64 * 2 = 128, total = 384
        assert_eq!(footprint, 384);
    }

    #[test]
    fn test_check_op_l2_fit_within_budget() {
        let hw_limits = AneHwLimits::for_revision(AneRevision::V4);
        let op = MirOp::MILConst {
            name: "big_weight".into(),
            value_path: "bw.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let mut shapes = HashMap::new();
        shapes.insert("big_weight".to_string(), vec![1, 16384]); // 32768 bytes

        // With 0 cumulative, should fit in 32768 budget
        let (footprint, exceeds) = check_op_l2_fit(&op, 0, &hw_limits, &shapes);
        assert!(!exceeds);
        assert_eq!(footprint, 32768);

        // With 1 byte cumulative, should exceed
        let (_, exceeds) = check_op_l2_fit(&op, 1, &hw_limits, &shapes);
        assert!(exceeds);
    }

    #[test]
    fn test_l2_budget_violation_display() {
        let violation = L2BudgetViolation {
            total_footprint: 100000,
            budget: 32768,
            op_count: 5,
            revision: AneRevision::V4,
        };
        let msg = format!("{}", violation);
        assert!(msg.contains("100000"));
        assert!(msg.contains("32768"));
        assert!(msg.contains("5 ops"));
    }

    #[test]
    fn test_l2_cache_size_per_ne_values() {
        // A11: 32768 (32KB per NE, 1 NE)
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        assert_eq!(a11.l2_cache_size_per_ne, 32768);

        // A14: 131072 (128KB per NE, 2 NEs = 256KB total)
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(a14.l2_cache_size_per_ne, 131072);
        assert_eq!(a14.total_l2_budget(), 131072 * 2);

        // A16: 262144 (256KB per NE, 4 NEs = 1MB total)
        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        assert_eq!(a16.l2_cache_size_per_ne, 262144);
        assert_eq!(a16.total_l2_budget(), 262144 * 4);
    }

    #[test]
    fn test_estimate_unknown_shape_returns_zero() {
        let op = MirOp::MILConst {
            name: "mystery".into(),
            value_path: "m.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let shapes = HashMap::new(); // No shapes — footprint is 0
        let footprint = estimate_op_l2_footprint(&op, &shapes);
        assert_eq!(footprint, 0);
    }
}
