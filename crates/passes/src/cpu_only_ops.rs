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

/// Get the reason an op is CPU-only, if it is.
pub fn get_cpu_only_reason(mil_op_name: &str) -> Option<&CpuOnlyReason> {
    CPU_ONLY_OPS_DETAILED.iter().find(|op| op.mil_name == mil_op_name).map(|op| &op.reason)
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
    fn test_cpu_only_reason() {
        assert_eq!(get_cpu_only_reason("acos"), Some(&CpuOnlyReason::TrigInverse));
        assert_eq!(get_cpu_only_reason("gru"), Some(&CpuOnlyReason::Rnn));
        assert_eq!(get_cpu_only_reason("linear"), None);
    }

    #[test]
    fn test_cpu_only_set_size() {
        // Should have at least 80 entries
        assert!(
            CPU_ONLY_OPS.len() >= 80,
            "CPU_ONLY_OPS has {} entries, expected >= 80",
            CPU_ONLY_OPS.len()
        );
    }
}
