//! Staticize pass.
//!
//! Resolves dynamic constructs in SIR into static equivalents
//! based on profiling knowledge and shape inference.
//!
//! Current implementation: pass-through for the linear projection vertical slice.
//! The linear projection SIR from sir_from_linear_projection already has all
//! shapes concrete and no dynamic constructs, so staticization is a no-op.

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// Staticize pass implementation.
pub struct StaticizePass {
    // No configuration needed for pass-through
}

impl Default for StaticizePass {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticizePass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the staticize pass.
    ///
    /// For the current linear projection vertical slice, all shapes in the
    /// SIR are already concrete (specified in the task spec), and there are
    /// no dynamic constructs (no runtime-computed indices, no variable-length
    /// sequences, no data-dependent branching). This is a pass-through.
    ///
    /// When more complex SIR graphs are supported (e.g., attention with
    /// variable sequence lengths, decode steps with dynamic KV cache),
    /// this pass will:
    /// - Replace symbolic dimensions with concrete values from the task spec
    /// - Replace runtime-computed indices with static lookup tables
    /// - Resolve variable-length sequences to fixed lengths
    /// - Record staticization decisions in the SIR metadata
    pub fn run(&self, input: SirGraph) -> Result<SirGraph> {
        // Pass-through: the linear projection SIR is already fully static.
        Ok(input)
    }
}
