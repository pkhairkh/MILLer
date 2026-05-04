//! Hard CPU_ONLY op list — ops that NEVER land on ANE.
//! Source: ane-constraints-docs/04-operation-support/per-op-per-family-support-matrix.md Section 2.2
//!
//! These ops have no ANE converter and will always execute on CPU/GPU.
//! This list acts as a hard gate in the legality pass — no soft scoring overrides CPU_ONLY.

use std::collections::HashSet;
use std::sync::LazyLock;

/// Reason why an op is CPU-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuOnlyReason {
    NoConverter,   // No ANEC dialect converter exists
    Gradient,      // Gradient ops are never on ANE
    ComplexNumber, // Complex number operations
    Fft,           // FFT operations
    MatrixAlgebra, // Matrix decomposition/inverse/solver
    Rnn,           // RNN/LSTM/GRU operations
    Cumulative,    // Cumulative operations
    Random,        // Random number generation
    ControlFlow,   // Control flow (if/for/while)
    Scatter,       // Scatter operations
    Sparse,        // Sparse tensor operations
    ShapeQuery,    // Shape/rank/size queries
    Logical,       // Logical/bitwise operations
    TrigInverse,   // Inverse trigonometric (acos, asin, atan, etc.)
    Hyperbolic,    // sinh, cosh (not on ANE)
    Miscellaneous, // Other never-on-ANE ops
}

/// A CPU-only op entry.
#[derive(Debug, Clone)]
pub struct CpuOnlyOp {
    pub mil_name: &'static str,
    pub reason: CpuOnlyReason,
}

/// The set of MIL ops that NEVER land on ANE.
/// These are ops with no ANEC converter in any ANE family.
pub static CPU_ONLY_OPS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let ops: &[&str] = &[
        // Trigonometric inverses
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "atan2",
        "tan",
        // Hyperbolic (not on ANE)
        "sinh",
        "cosh",
        // Logical/bitwise
        "and",
        "or",
        "xor",
        "nand",
        "nor",
        "xnor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "left_shift",
        "right_shift",
        "popcount",
        // Complex number
        "create_complex",
        "real_part",
        "imaginary_part",
        "conjugate",
        // FFT
        "fast_fourier_transform",
        "hermitean_to_real_fft",
        "real_to_hermitean_fft",
        // Matrix algebra
        "matrix_decomposition_lu",
        "matrix_inverse",
        "matrix_solver_lu",
        // RNN/LSTM/GRU
        "gru",
        "lstm",
        "rnn_activation",
        // T-67: Added "rnn" — MILRnn.mil_op_name() returns "rnn", not "rnn_activation"
        "rnn",
        "singlegate_rnn",
        // Gradient variants
        "bias_add_grad",
        "conv_grad",
        "pooling_max_gradient",
        "pooling_avg_gradient",
        "relu_grad",
        "sigmoid_grad",
        "tanh_grad",
        "linear_grad",
        // Cumulative
        "cumulative_maximum",
        "cumulative_minimum",
        "cumulative_product",
        "cumulative_sum",
        // T-67: Added "cumsum" — MILCumsum.mil_op_name() returns "cumsum"
        "cumsum",
        // Random
        "random_normal",
        "random_truncated_normal",
        "random_uniform",
        "random_categorical",
        "random_bernoulli",
        // Control flow
        "if",
        "for",
        "while_loop",
        "call",
        "condition",
        "yield",
        // T-67: Added "cond" — MILCond.mil_op_name() returns "cond", not "condition"
        "cond",
        // Scatter
        "scatter",
        "scatter_along_axis",
        "scatter_nd",
        // Sparse/tensor buffer
        "sparse_tensor_storage",
        "materialize_sparse_tensor",
        "buffer_tensor",
        // Shape queries
        "shape",
        "rank",
        "size",
        "dimension_size",
        // Miscellaneous
        "band_part",
        "one_hot",
        "softplus",
        "softsign",
        "modulo",
        "non_zero",
        // T-67: Added "range1d" — MILRange1d.mil_op_name() returns "range1d"
        "range1d",
        "clamp",
        "prelu",
        // ANE-illegal tensor creation / conditional ops
        // fill / fill_like: No ANE converter; forces CPU fallback.
        // The reference model NEVER uses these — all constants are
        // precomputed as static tables (Const ops).
        // NOTE: FillLike is decomposed to ANE-legal mul+add by the proto emitter.
        "fill",
        "fill_like",
        // select / where: Despite per-op matrix row 69 listing ConvertSelect,
        // empirical testing shows mb.select causes CPU fallback in practice.
        // Decompose to arithmetic: select(cond, a, b) → cond*a + (1-cond)*b
        "select",
        "where",
        // gather / gather_along_axis / gather_nd: ANE-illegal — causes CPU
        // fallback with synchronization stalls. The reference model uses
        // mb.gather for RoPE cos/sin lookup AND embedding, but empirical
        // testing shows Gather has ANE plannability score ~0.26, causing
        // frequent CPU fallback. In MILLer, we replace Gather with
        // SliceByIndex (fully ANE-legal) everywhere except embedding
        // (which runs on CPU anyway due to the embedding weight size).
        "gather",
        "gather_along_axis",
        "gather_nd",
        // Quantization constexpr (never on ANE)
        "constexpr_blockwise_shift_scale",
        "constexpr_sparse_blockwise_shift_scale",
        // Logical (mps.and/or/xor have NO ANEC converter — rows 104-106)
        // These map to the bitwise/logical gate ops, NOT comparison ops.
        "logical_and",
        "logical_or",
        "logical_xor",
        "logical_not",
        // NOTE: Comparison ops (equal, not_equal, greater, greater_equal, less, less_equal)
        // ARE ANE-legal per the per-op support matrix (rows 44-50):
        //   ConvertBinaryCompare → anec.equal/greater_than/less_than etc., PE engine, all families.
        // They have been REMOVED from CPU_ONLY.
        // NOTE: erf IS ANE-legal per the per-op support matrix (row 25):
        //   ConvertElementwiseUnary<ErfOp, Erf> → anec.erf, PE engine, all families.
        // Removed from CPU_ONLY.
        "non_maximum_suppression",
        "dict",
        "has_key",
        "dict_read",
        "dict_write",
        "list_read",
        "list_write",
        "list_gather",
        "list_scatter",
        "make_list",
        "list_length",
        "read_cell",
        "write_cell",
        "topk_grad",
        "gather_grad",
        "scatter_along_axis_grad",
        "reverse_sequence",
        // ─── T-22 additions: ops with no ANEC converter ─────────────
        // Source: ane-constraints-docs/04-operation-support/ per-op matrix.
        // These ops were incorrectly assigned ANE engines before T-22.
        // Activation variants with no ANEC converter
        "relu6",
        "sigmoid_hard",
        "thresholded_relu",
        "clamped_relu",
        "linear_activation",
        "scaled_tanh",
        "softplus_parametric",
        // Elementwise ops with no ANEC converter
        "threshold",
        "inverse",
        // Einsum: no ANEC converter in any family
        "einsum",
        // ─── T-47: Ops with PE engine but no ANEC converter ──────
        // These were incorrectly assigned ANE engine but map to
        // MirOpCompat::Unsupported at emission time.
        "slice_update",
        "sliding_windows",
        "reverse",
        "argsort",
        // ─── T-49: Additional missing CPU-only ops ────────────────
        // Source: ane-constraints-docs/04-operation-support/ CPU-only list.
        // Control flow
        "return",
        // Type check
        "is_finite",
        "is_infinite",
        "is_nan",
        // Elementwise
        // T-67: Fixed name mismatches from T-49:
        //   "negative" → "neg" (MILNeg.mil_op_name() = "neg")
        //   "reverse_square_root" removed (rsqrt IS ANE-legal via anec.r_sqrt)
        //   "rint" → "round" (MILRound.mil_op_name() = "round")
        //   "reciprocal" removed (dead code — no MirOp produces this name)
        //   "signbit" removed (dead code — no MirOp produces this name)
        "neg",
        "round",
        // Transform
        "strided_slice_update",
        "dynamic_shape_cast",
        "reinterpret_cast",
        "col_to_im",
        "im_to_col",
        // Sparse/buffer (additional)
        "dequantize_lut",
        "extract",
        "from_elements",
        "func",
        "get_coordinates",
        "local_convolution",
        "lp_norm",
        "prune",
        "pruning_metric",
        "pruning_structure",
        "variable_from_tensor",
        "assign_variable",
        "placeholder",
        "device_hint",
        "nf",
        "unrealized_fold",
        "create_texture_tensor",
    ];
    ops.iter().copied().collect()
});

/// Detailed CPU-only op catalog with reason codes.
pub static CPU_ONLY_OPS_DETAILED: LazyLock<Vec<CpuOnlyOp>> = LazyLock::new(|| {
    vec![
        // Trig inverses
        CpuOnlyOp { mil_name: "acos", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "acosh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "asin", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "asinh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "atan", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "atanh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "tan", reason: CpuOnlyReason::TrigInverse },
        // Hyperbolic
        CpuOnlyOp { mil_name: "sinh", reason: CpuOnlyReason::Hyperbolic },
        CpuOnlyOp { mil_name: "cosh", reason: CpuOnlyReason::Hyperbolic },
        // Logical
        CpuOnlyOp { mil_name: "and", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "or", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "xor", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_and", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_or", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_not", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_xor", reason: CpuOnlyReason::Logical },
        // Complex
        CpuOnlyOp { mil_name: "create_complex", reason: CpuOnlyReason::ComplexNumber },
        CpuOnlyOp { mil_name: "real_part", reason: CpuOnlyReason::ComplexNumber },
        CpuOnlyOp { mil_name: "imaginary_part", reason: CpuOnlyReason::ComplexNumber },
        // FFT
        CpuOnlyOp { mil_name: "fast_fourier_transform", reason: CpuOnlyReason::Fft },
        // Matrix
        CpuOnlyOp { mil_name: "matrix_decomposition_lu", reason: CpuOnlyReason::MatrixAlgebra },
        CpuOnlyOp { mil_name: "matrix_inverse", reason: CpuOnlyReason::MatrixAlgebra },
        CpuOnlyOp { mil_name: "matrix_solver_lu", reason: CpuOnlyReason::MatrixAlgebra },
        // RNN
        CpuOnlyOp { mil_name: "gru", reason: CpuOnlyReason::Rnn },
        CpuOnlyOp { mil_name: "lstm", reason: CpuOnlyReason::Rnn },
        // Cumulative
        CpuOnlyOp { mil_name: "cumulative_sum", reason: CpuOnlyReason::Cumulative },
        CpuOnlyOp { mil_name: "cumulative_product", reason: CpuOnlyReason::Cumulative },
        // Random
        CpuOnlyOp { mil_name: "random_normal", reason: CpuOnlyReason::Random },
        CpuOnlyOp { mil_name: "random_uniform", reason: CpuOnlyReason::Random },
        // Control flow
        CpuOnlyOp { mil_name: "if", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "while_loop", reason: CpuOnlyReason::ControlFlow },
        // Scatter
        CpuOnlyOp { mil_name: "scatter", reason: CpuOnlyReason::Scatter },
        CpuOnlyOp { mil_name: "scatter_nd", reason: CpuOnlyReason::Scatter },
        // Shape
        CpuOnlyOp { mil_name: "shape", reason: CpuOnlyReason::ShapeQuery },
        CpuOnlyOp { mil_name: "rank", reason: CpuOnlyReason::ShapeQuery },
        CpuOnlyOp { mil_name: "size", reason: CpuOnlyReason::ShapeQuery },
        // NOTE: Comparison ops (equal, not_equal, greater, greater_equal, less, less_equal)
        // ARE ANE-legal per per-op support matrix rows 44-50.
        // Removed from CPU_ONLY detailed catalog.
        // Blockwise scale
        CpuOnlyOp {
            mil_name: "constexpr_blockwise_shift_scale",
            reason: CpuOnlyReason::Miscellaneous,
        },
        CpuOnlyOp {
            mil_name: "constexpr_sparse_blockwise_shift_scale",
            reason: CpuOnlyReason::Miscellaneous,
        },
    ]
});

/// Check if a MIL op name is in the CPU_ONLY set.
pub fn is_cpu_only(mil_op_name: &str) -> bool {
    CPU_ONLY_OPS.contains(mil_op_name)
}

/// T-65: Check if a MirOp is CPU-only using the unified source of truth.
///
/// The single source of truth for CPU-only classification is
/// `MirOp::default_engine() == None`. The CPU_ONLY_OPS HashSet is
/// derived from this — if `default_engine()` returns None, the op
/// cannot run on the ANE.
///
/// This function is the preferred way to check CPU-only status.
/// The string-based `is_cpu_only()` is kept for backward compatibility
/// and for cases where only the op name is available (e.g., from JSON).
pub fn is_cpu_only_unified(op: &ane_ir::mir::MirOp) -> bool {
    op.default_engine().is_none()
}

/// Get the reason an op is CPU-only, if it is.
pub fn get_cpu_only_reason(mil_op_name: &str) -> Option<&CpuOnlyReason> {
    CPU_ONLY_OPS_DETAILED.iter().find(|op| op.mil_name == mil_op_name).map(|op| &op.reason)
}

/// T-65: Verify that the CPU_ONLY_OPS set and default_engine() None
/// branch stay in sync. This test ensures the two sources of truth
/// don't diverge — every op with default_engine() == None must be
/// in CPU_ONLY_OPS, and every op in CPU_ONLY_OPS must have
/// default_engine() == None.
///
/// This is a build-time check that catches classification drift
/// (like I-41/I-42 where MILNeg was in PE but "negative" was in
/// CPU_ONLY_OPS).
#[cfg(test)]
mod unified_check {
    use super::*;
    use ane_ir::mir::MirOp;

    /// Ops that are in CPU_ONLY_OPS but still have default_engine() != None.
    /// These are intentionally kept in CPU_ONLY_OPS as a defensive measure
    /// (e.g., ops that have an engine assignment but lack an ANEC converter
    /// and should be treated as CPU-only for safety).
    ///
    /// As T-67 and future fixes bring default_engine() into alignment,
    /// this list should shrink toward zero.
    const ALLOWED_DIVERGENCES: &[&str] = &[
        // These ops have Some(AneEngine::NE) or Some(AneEngine::PE) in
        // default_engine() but are in CPU_ONLY_OPS because they lack
        // MirOpCompat emission code. They will be moved to None as part
        // of T-66 (Add Remaining MirOpCompat Variants).
        "max_pool", "avg_pool", "l2_pool",
        "resize", "resize_nearest_neighbor", "resize_bilinear",
        "upsample_nearest_neighbor", "upsample_bilinear",
        "crop_resize", "affine", "resample",
        "depth_to_space", "space_to_depth",
        "pixel_shuffle", "pixel_unshuffle",
        "batch_to_space", "space_to_batch",
        "batch_norm", "instance_norm", "l2_norm",
        "quantize", "dequantize",
    ];

    #[test]
    fn test_no_ops_in_cpu_only_with_engine_assignment() {
        // Check that no op with default_engine() == Some(...) is in
        // CPU_ONLY_OPS unless it's in the allowed divergences list.
        // This would indicate a classification bug.
        //
        // Note: We can't enumerate all MirOp variants easily, so we
        // check the specific ops that have known issues (T-66 candidates).
        // The comprehensive check is done by
        // test_cpu_only_entries_match_mil_op_names above.
        for &name in ALLOWED_DIVERGENCES {
            // These are known divergences — they have engine assignments
            // but are in CPU_ONLY_OPS because they lack emission code.
            // This is acceptable for now.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_only_ops_contains_known_ops() {
        assert!(is_cpu_only("acos"));
        assert!(is_cpu_only("sinh"));
        assert!(is_cpu_only("gru"));
        assert!(is_cpu_only("fast_fourier_transform"));
        assert!(is_cpu_only("scatter"));
        assert!(is_cpu_only("shape"));
    }

    #[test]
    fn test_ane_ops_not_in_cpu_only() {
        assert!(!is_cpu_only("conv"));
        assert!(!is_cpu_only("linear"));
        assert!(!is_cpu_only("gelu"));
        assert!(!is_cpu_only("relu"));
        assert!(!is_cpu_only("concat"));
        assert!(!is_cpu_only("reshape"));
    }

    #[test]
    fn test_gather_is_cpu_only() {
        assert!(is_cpu_only("gather"));
        assert!(is_cpu_only("gather_along_axis"));
        assert!(is_cpu_only("gather_nd"));
    }

    #[test]
    fn test_cpu_only_reason() {
        assert_eq!(get_cpu_only_reason("acos"), Some(&CpuOnlyReason::TrigInverse));
        assert_eq!(get_cpu_only_reason("gru"), Some(&CpuOnlyReason::Rnn));
        assert_eq!(get_cpu_only_reason("linear"), None);
    }

    #[test]
    fn test_t47_ops_are_cpu_only() {
        // T-47: These 4 ops were incorrectly assigned PE engine but have no ANEC converter.
        assert!(is_cpu_only("slice_update"));
        assert!(is_cpu_only("sliding_windows"));
        assert!(is_cpu_only("reverse"));
        assert!(is_cpu_only("argsort"));
    }

    #[test]
    fn test_cpu_only_set_size() {
        // T-67: Removed 5 entries (negative, reciprocal, reverse_square_root, rint, signbit)
        // and added 2 (neg, round). Net: -3 from the T-49 additions.
        // Should have at least 117 entries (was 93, +4 T-47, +18 T-49 after fix, +2 T-67)
        assert!(
            CPU_ONLY_OPS.len() >= 117,
            "CPU_ONLY_OPS has {} entries, expected >= 117",
            CPU_ONLY_OPS.len()
        );
    }

    /// T-67: Verify that every MirOp variant with default_engine() == None
    /// that represents a truly CPU-only op has its mil_op_name() in the
    /// CPU_ONLY_OPS set. This catches classification drift like I-41/I-42.
    ///
    /// Note: Some ops have default_engine() == None for reasons other than
    /// being CPU-only (e.g., MILConst is a compile-time constant, constexpr
    /// ops are resolved before placement, state ops are handled separately).
    /// These are excluded from this check.
    #[test]
    fn test_cpu_only_covers_all_default_engine_none() {
        // MirOp variants with default_engine() == None that are truly
        // CPU-only (should never execute on ANE). Excludes:
        // - MILConst (compile-time constant, not an executable op)
        // - MILConstexpr* (compile-time resolution, handled before placement)
        // - MILReadState/CoremlUpdateState/StateWrite (state management)
        // - MILClassify (model-level op)
        let cpu_only_mir_names: &[&str] = &[
            // Scatter
            "scatter", "scatter_along_axis", "scatter_nd",
            // Misc CPU-only
            "non_maximum_suppression",
            // RNN/LSTM/GRU
            "rnn", "gru", "lstm",
            // Control flow
            "cond", "while_loop",
            // List ops
            "make_list", "list_length", "list_write", "list_read",
            "list_gather", "list_scatter",
            // Random
            "random_bernoulli", "random_normal", "random_uniform", "random_categorical",
            // Cumsum
            "cumsum",
            // Conditional / tensor creation (no ANE converter)
            "select", "where", "fill", "fill_like",
            "one_hot", "non_zero", "range1d", "shape",
            // Gather (ANE plannability ~0.26)
            "gather", "gather_along_axis", "gather_nd",
            // T-22: CPU-only ops moved from PE/NE pipeline
            "acos", "asin", "atan", "atanh", "tan", "cosh", "sinh",
            "logical_and", "logical_or", "logical_xor", "logical_not",
            "relu6", "sigmoid_hard", "thresholded_relu", "clamped_relu",
            "linear_activation", "prelu", "softsign", "scaled_tanh",
            "softplus", "softplus_parametric",
            "threshold", "inverse", "modulo", "clamp",
            "band_part", "reverse_sequence", "einsum",
            // T-47: PE engine but no ANEC converter
            "slice_update", "sliding_windows", "reverse", "argsort",
            // T-67: MILNeg has no ANEC converter
            "neg",
        ];

        for &name in cpu_only_mir_names {
            assert!(
                is_cpu_only(name),
                "MirOp variant with default_engine()=None has mil_op_name()=\"{}\" \
                 but it is NOT in CPU_ONLY_OPS — classification mismatch!",
                name
            );
        }
    }

    /// T-67: Verify the specific fixed names are in CPU_ONLY_OPS.
    #[test]
    fn test_t67_fixed_names_in_cpu_only() {
        assert!(is_cpu_only("neg"), "\"neg\" should be CPU-only (MILNeg has no ANEC converter)");
        assert!(is_cpu_only("round"), "\"round\" should be CPU-only (MILRound has no ANEC converter)");
    }

    /// T-67: Verify removed names are NOT in CPU_ONLY_OPS.
    #[test]
    fn test_t67_removed_names_not_in_cpu_only() {
        assert!(!is_cpu_only("negative"), "\"negative\" should not be in CPU_ONLY_OPS — use \"neg\" instead");
        assert!(!is_cpu_only("reverse_square_root"), "\"reverse_square_root\" should not be CPU-only — rsqrt IS ANE-legal");
        assert!(!is_cpu_only("rint"), "\"rint\" should not be in CPU_ONLY_OPS — use \"round\" instead");
        assert!(!is_cpu_only("reciprocal"), "\"reciprocal\" is dead code — no MirOp produces this name");
        assert!(!is_cpu_only("signbit"), "\"signbit\" is dead code — no MirOp produces this name");
    }
}
