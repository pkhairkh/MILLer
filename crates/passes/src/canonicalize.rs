//! Canonicalize pass.
//!
//! Normalizes SIR into a canonical form: fusing patterns,
//! eliminating redundancies, and standardizing naming.
//!
//! Current implementation: pass-through for the linear projection vertical slice.
//! The single-op graph is already canonical, so no transformation is needed.
//! Future work: pattern fusion (e.g., linear + bias → fused linear),
//! redundant identity elimination, naming standardization.

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// Canonicalize pass implementation.
pub struct CanonicalizePass {
    // No configuration needed for pass-through
}

impl CanonicalizePass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the canonicalize pass.
    ///
    /// For the current linear projection vertical slice, the SIR is already
    /// canonical (single linear projection op), so this is a pass-through.
    /// When more complex SIR graphs are supported, this pass will:
    /// - Fuse linear + bias into a single LinearProjection op
    /// - Eliminate identity operations
    /// - Standardize node naming conventions
    /// - Merge consecutive elementwise operations
    pub fn run(&self, input: SirGraph) -> Result<SirGraph> {
        // Pass-through: return the input unchanged.
        // The linear projection SIR produced by sir_from_linear_projection
        // is already in canonical form.
        Ok(input)
    }
}
