//! Staticize pass — **REMOVED** (T-P4-03, originally T-107).
//!
//! This pass was originally intended to resolve dynamic constructs in SIR
//! into static equivalents based on profiling knowledge and shape inference:
//! - Replace symbolic dimensions with concrete values from the task spec
//! - Replace runtime-computed indices with static lookup tables
//! - Resolve variable-length sequences to fixed lengths
//! - Record staticization decisions in SIR metadata
//!
//! However, none of these features were ever implemented. The pass was a
//! pure pass-through (`Ok(input)`) that consumed a pipeline step while doing
//! nothing, wasting developer trust and obscuring the actual pipeline.
//!
//! **Removal rationale** (T-107/T-P4-03): A phantom pass that claims capabilities it
//! doesn't have is worse than no pass at all — it misleads developers into
//! thinking dynamic SIR constructs are being resolved when they are not.
//! The pass has been removed from the compile pipeline in `main.rs`. If
//! staticization is needed in the future, it should be implemented as a new
//! pass with clear scope and tests before being wired into the pipeline.
//!
//! The module is preserved as documentation only. No `StaticizePass` struct
//! or `run()` method exists anymore.

// No struct, no impl, no run() — the phantom pass is gone.
// If you need staticization in the future, implement a new pass from scratch
// with real functionality and comprehensive tests.

#[cfg(test)]
mod tests {
    // T-P4-03: The StaticizePass struct has been removed.
    // The only test needed is to confirm that the module compiles without it.
    #[test]
    fn staticize_pass_removed() {
        // This test exists to confirm the module is present and the
        // StaticizePass has been removed. If you need to re-implement
        // staticization, create a new pass with a different name.
        assert!(true, "StaticizePass has been removed per T-P4-03");
    }
}
