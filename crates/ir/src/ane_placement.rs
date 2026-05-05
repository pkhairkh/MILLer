//! ANE op-to-engine placement logic.
//!
//! Extracted from `mir.rs` to separate the placement decision from the
//! IR type definition. The `engine_for_op()` function is the single
//! source of truth for determining which ANE engine (NE, PE,
//! TransposeEngine, or None/CPU) handles each MIR operation.

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
pub fn engine_for_op(
    op: &MirOp,
    revision: Option<AneRevision>,
) -> Option<AneEngine> {
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
