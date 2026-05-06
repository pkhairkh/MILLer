//! ANE op-to-engine placement logic.
//!
//! Extracted from `mir.rs` to separate the placement decision from the
//! IR type definition. The `engine_for_op()` function is the single
//! source of truth for determining which ANE engine (NE, PE,
//! TransposeEngine, or None/CPU) handles each MIR operation.
//!
//! T-P5-07: The `base_engine()` method on `MirOp` is deprecated in favor
//! of this module. The static engine mapping is being migrated here so
//! that engine assignment is parameterized by target rather than being
//! a fixed property of the op. Eventually, `base_engine()` will be
//! removed entirely and all placement logic will live in this module.

use crate::ane_engine::AneEngine;
use crate::ane_target::AneRevision;
use crate::mir::MirOp;

/// Determine the ANE engine assignment for a MIR op given a target revision.
///
/// This is the revision-aware placement function that considers both the
/// static per-op engine mapping and family-specific capability overrides.
///
/// When `revision` is `None`, returns the base engine assignment without
/// family-specific overrides (backward-compatible behavior).
///
/// T-P5-07: This function is the canonical entry point for engine placement.
/// Prefer this over `MirOp::base_engine()` or `MirOp::default_engine()`,
/// which are deprecated.
pub fn engine_for_op(
    op: &MirOp,
    revision: Option<AneRevision>,
) -> Option<AneEngine> {
    #[allow(deprecated)] // T-P5-07: Still delegates to base_engine() during migration
    let base = op.base_engine();

    let family = match revision {
        Some(rev) => rev.family(),
        None => return base,
    };

    // Apply family-specific overrides
    match op {
        MirOp::MILReduceArgmax { .. } | MirOp::MILReduceArgmin { .. }
            if !family.supports_argminmax() =>
        {
            return None;
        }
        MirOp::MILReduceL2Norm { .. } if family.uses_a14minus_converters() => {
            return None;
        }
        MirOp::MILSquare { .. } if family.uses_a14minus_converters() => {
            return None;
        }
        MirOp::MILScaledDotProductAttention { .. } if !family.supports_sdpa() => {
            return None;
        }
        MirOp::MILLayerNorm { .. } if !family.supports_layernorm() => {
            return None;
        }
        _ => {}
    }

    base
}

/// T-P5-07: Query whether an op is CPU-only for a given revision.
///
/// This is equivalent to `engine_for_op(op, Some(rev)).is_none()` but
/// is provided as a named function for clarity at call sites.
pub fn is_cpu_only_for_revision(op: &MirOp, revision: AneRevision) -> bool {
    engine_for_op(op, Some(revision)).is_none()
}

/// T-P5-07: Query whether an op is CPU-only (revision-agnostic).
///
/// This is equivalent to `engine_for_op(op, None).is_none()` but
/// is provided as a named function for clarity at call sites.
pub fn is_cpu_only(op: &MirOp) -> bool {
    engine_for_op(op, None).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MirNodeId;

    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    // ─── T-P5-07: engine_for_op tests ─────────────────────────────────

    #[test]
    fn test_engine_for_op_conv_is_ne() {
        let op = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1, 1],
            pad_amounts: vec![0, 0],
            dilations: vec![1, 1],
        };
        assert_eq!(engine_for_op(&op, None), Some(AneEngine::NE));
    }

    #[test]
    fn test_engine_for_op_relu_is_pe() {
        let op = MirOp::MILRelu { name: "relu".into(), x: nid("x") };
        assert_eq!(engine_for_op(&op, None), Some(AneEngine::PE));
    }

    #[test]
    fn test_engine_for_op_transpose_is_transpose_engine() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![0, 2, 1] };
        assert_eq!(engine_for_op(&op, None), Some(AneEngine::TransposeEngine));
    }

    #[test]
    fn test_engine_for_op_const_is_cpu_only() {
        let op = MirOp::MILConst { name: "c".into(), value_path: "w.bin".into(), dtype: crate::common::MilDtype::Fp16 };
        assert_eq!(engine_for_op(&op, None), None);
    }

    #[test]
    fn test_engine_for_op_argmax_on_a18_is_none() {
        let op = MirOp::MILReduceArgmax { name: "am".into(), x: nid("x"), axis: 1, keep_dims: false };
        // A18 does NOT support ArgMinMax (no LSE_7 converter)
        assert_eq!(engine_for_op(&op, Some(AneRevision::V19)), None);
        // A17 DOES support ArgMinMax
        assert_eq!(engine_for_op(&op, Some(AneRevision::V11)), Some(AneEngine::PE));
    }

    #[test]
    fn test_engine_for_op_sdpa_on_a14_is_none() {
        let op = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: None,
        };
        // A14 does NOT support SDPA
        assert_eq!(engine_for_op(&op, Some(AneRevision::V7)), None);
        // A16 DOES support SDPA
        assert_eq!(engine_for_op(&op, Some(AneRevision::V10)), Some(AneEngine::NE));
    }

    #[test]
    fn test_engine_for_op_layernorm_on_a14_is_none() {
        let op = MirOp::MILLayerNorm {
            name: "ln".into(),
            x: nid("x"),
            weight: "ln_weight".into(),
            bias: None,
            epsilon: 1e-5,
            axes: vec![2],
        };
        // A14 does NOT support LayerNorm
        assert_eq!(engine_for_op(&op, Some(AneRevision::V7)), None);
        // A15 DOES support LayerNorm
        assert_eq!(engine_for_op(&op, Some(AneRevision::V8)), Some(AneEngine::PE));
    }

    #[test]
    fn test_is_cpu_only() {
        let relu = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        assert!(!is_cpu_only(&relu));

        let const_op = MirOp::MILConst { name: "c".into(), value_path: "w.bin".into(), dtype: crate::common::MilDtype::Fp16 };
        assert!(is_cpu_only(&const_op));
    }

    #[test]
    fn test_is_cpu_only_for_revision() {
        let argmax = MirOp::MILReduceArgmax { name: "am".into(), x: nid("x"), axis: 1, keep_dims: false };
        // ArgMax is CPU-only on A18, but ANE-legal on A17
        assert!(is_cpu_only_for_revision(&argmax, AneRevision::V19));
        assert!(!is_cpu_only_for_revision(&argmax, AneRevision::V11));
    }

    #[test]
    fn test_engine_for_op_none_revision_returns_base() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        // With None revision, should return base engine (PE for Relu)
        assert_eq!(engine_for_op(&op, None), Some(AneEngine::PE));
    }
}
