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
        // ─── N-009: MILTile has PE engine but no ANEC converter.
        // The legality_rewrite pass already decomposes Tile into broadcast Mul,
        // so this is a safety net to prevent Tile from landing on ANE.
        "tile",
        // ─── T-P2-12: Ops with engine assignments but no MirOpCompat emission code ──
        // These ops have Some(AneEngine::NE) in default_engine() but lack
        // MirOpCompat emission code. They should be treated as CPU-only for
        // safety until emission support is added (T-66).
        "max_pool",
        "avg_pool",
        "l2_pool",
        "resize",
        "resize_nearest_neighbor",
        "resize_bilinear",
        "upsample_nearest_neighbor",
        "upsample_bilinear",
        "crop_resize",
        "affine",
        "resample",
        "depth_to_space",
        "space_to_depth",
        "pixel_shuffle",
        "pixel_unshuffle",
        "batch_to_space",
        "space_to_batch",
        "batch_norm",
        "instance_norm",
        "l2_norm",
        "quantize",
        "dequantize",
        // F-OPS-01: Removed "anec_fused_conv_activate" and "anec_scaled_elementwise"
        // — these are now promoted from CPU-only stubs to proper MirOpCompat variants
        // with MIR-to-proto emission code.
    ];
    ops.iter().copied().collect()
});

/// Detailed CPU-only op catalog with reason codes.
/// T-128: Expanded to cover ALL ops in CPU_ONLY_OPS (152 entries, reduced from 154
/// after F-OPS-01 promotion of anec_fused_conv_activate and anec_scaled_elementwise).
pub static CPU_ONLY_OPS_DETAILED: LazyLock<Vec<CpuOnlyOp>> = LazyLock::new(|| {
    vec![
        // ─── Trigonometric inverses ────────────────────────────────
        CpuOnlyOp { mil_name: "acos", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "acosh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "asin", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "asinh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "atan", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "atanh", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "atan2", reason: CpuOnlyReason::TrigInverse },
        CpuOnlyOp { mil_name: "tan", reason: CpuOnlyReason::TrigInverse },
        // ─── Hyperbolic ────────────────────────────────────────────
        CpuOnlyOp { mil_name: "sinh", reason: CpuOnlyReason::Hyperbolic },
        CpuOnlyOp { mil_name: "cosh", reason: CpuOnlyReason::Hyperbolic },
        // ─── Logical / bitwise ─────────────────────────────────────
        CpuOnlyOp { mil_name: "and", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "or", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "xor", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "nand", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "nor", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "xnor", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "bitwise_and", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "bitwise_or", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "bitwise_xor", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "bitwise_not", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "left_shift", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "right_shift", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "popcount", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_and", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_or", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_not", reason: CpuOnlyReason::Logical },
        CpuOnlyOp { mil_name: "logical_xor", reason: CpuOnlyReason::Logical },
        // ─── Complex number ────────────────────────────────────────
        CpuOnlyOp { mil_name: "create_complex", reason: CpuOnlyReason::ComplexNumber },
        CpuOnlyOp { mil_name: "real_part", reason: CpuOnlyReason::ComplexNumber },
        CpuOnlyOp { mil_name: "imaginary_part", reason: CpuOnlyReason::ComplexNumber },
        CpuOnlyOp { mil_name: "conjugate", reason: CpuOnlyReason::ComplexNumber },
        // ─── FFT ───────────────────────────────────────────────────
        CpuOnlyOp { mil_name: "fast_fourier_transform", reason: CpuOnlyReason::Fft },
        CpuOnlyOp { mil_name: "hermitean_to_real_fft", reason: CpuOnlyReason::Fft },
        CpuOnlyOp { mil_name: "real_to_hermitean_fft", reason: CpuOnlyReason::Fft },
        // ─── Matrix algebra ────────────────────────────────────────
        CpuOnlyOp { mil_name: "matrix_decomposition_lu", reason: CpuOnlyReason::MatrixAlgebra },
        CpuOnlyOp { mil_name: "matrix_inverse", reason: CpuOnlyReason::MatrixAlgebra },
        CpuOnlyOp { mil_name: "matrix_solver_lu", reason: CpuOnlyReason::MatrixAlgebra },
        // ─── RNN / LSTM / GRU ──────────────────────────────────────
        CpuOnlyOp { mil_name: "gru", reason: CpuOnlyReason::Rnn },
        CpuOnlyOp { mil_name: "lstm", reason: CpuOnlyReason::Rnn },
        CpuOnlyOp { mil_name: "rnn_activation", reason: CpuOnlyReason::Rnn },
        CpuOnlyOp { mil_name: "rnn", reason: CpuOnlyReason::Rnn },
        CpuOnlyOp { mil_name: "singlegate_rnn", reason: CpuOnlyReason::Rnn },
        // ─── Gradient / backprop ───────────────────────────────────
        CpuOnlyOp { mil_name: "bias_add_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "conv_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "pooling_max_gradient", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "pooling_avg_gradient", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "relu_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "sigmoid_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "tanh_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "linear_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "topk_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "gather_grad", reason: CpuOnlyReason::Gradient },
        CpuOnlyOp { mil_name: "scatter_along_axis_grad", reason: CpuOnlyReason::Gradient },
        // ─── Cumulative ────────────────────────────────────────────
        CpuOnlyOp { mil_name: "cumulative_maximum", reason: CpuOnlyReason::Cumulative },
        CpuOnlyOp { mil_name: "cumulative_minimum", reason: CpuOnlyReason::Cumulative },
        CpuOnlyOp { mil_name: "cumulative_product", reason: CpuOnlyReason::Cumulative },
        CpuOnlyOp { mil_name: "cumulative_sum", reason: CpuOnlyReason::Cumulative },
        CpuOnlyOp { mil_name: "cumsum", reason: CpuOnlyReason::Cumulative },
        // ─── Random ────────────────────────────────────────────────
        CpuOnlyOp { mil_name: "random_normal", reason: CpuOnlyReason::Random },
        CpuOnlyOp { mil_name: "random_truncated_normal", reason: CpuOnlyReason::Random },
        CpuOnlyOp { mil_name: "random_uniform", reason: CpuOnlyReason::Random },
        CpuOnlyOp { mil_name: "random_categorical", reason: CpuOnlyReason::Random },
        CpuOnlyOp { mil_name: "random_bernoulli", reason: CpuOnlyReason::Random },
        // ─── Control flow ──────────────────────────────────────────
        CpuOnlyOp { mil_name: "if", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "for", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "while_loop", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "call", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "condition", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "yield", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "cond", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "return", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "func", reason: CpuOnlyReason::ControlFlow },
        // Dict ops
        CpuOnlyOp { mil_name: "dict", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "has_key", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "dict_read", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "dict_write", reason: CpuOnlyReason::ControlFlow },
        // List ops
        CpuOnlyOp { mil_name: "list_read", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "list_write", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "list_gather", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "list_scatter", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "make_list", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "list_length", reason: CpuOnlyReason::ControlFlow },
        // Cell ops
        CpuOnlyOp { mil_name: "read_cell", reason: CpuOnlyReason::ControlFlow },
        CpuOnlyOp { mil_name: "write_cell", reason: CpuOnlyReason::ControlFlow },
        // ─── Scatter ───────────────────────────────────────────────
        CpuOnlyOp { mil_name: "scatter", reason: CpuOnlyReason::Scatter },
        CpuOnlyOp { mil_name: "scatter_along_axis", reason: CpuOnlyReason::Scatter },
        CpuOnlyOp { mil_name: "scatter_nd", reason: CpuOnlyReason::Scatter },
        // ─── Sparse / tensor buffer ────────────────────────────────
        CpuOnlyOp { mil_name: "sparse_tensor_storage", reason: CpuOnlyReason::Sparse },
        CpuOnlyOp { mil_name: "materialize_sparse_tensor", reason: CpuOnlyReason::Sparse },
        CpuOnlyOp { mil_name: "buffer_tensor", reason: CpuOnlyReason::Sparse },
        // ─── Shape queries ─────────────────────────────────────────
        CpuOnlyOp { mil_name: "shape", reason: CpuOnlyReason::ShapeQuery },
        CpuOnlyOp { mil_name: "rank", reason: CpuOnlyReason::ShapeQuery },
        CpuOnlyOp { mil_name: "size", reason: CpuOnlyReason::ShapeQuery },
        CpuOnlyOp { mil_name: "dimension_size", reason: CpuOnlyReason::ShapeQuery },
        // ─── No ANEC converter — activation variants ───────────────
        CpuOnlyOp { mil_name: "relu6", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "sigmoid_hard", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "thresholded_relu", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "clamped_relu", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "linear_activation", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "scaled_tanh", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "softplus_parametric", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "softplus", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "softsign", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "prelu", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — elementwise ───────────────────────
        CpuOnlyOp { mil_name: "threshold", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "inverse", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "neg", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "round", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "modulo", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "clamp", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "is_finite", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "is_infinite", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "is_nan", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — tensor creation / conditional ─────
        CpuOnlyOp { mil_name: "fill", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "fill_like", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "select", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "where", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "one_hot", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "non_zero", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "range1d", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "band_part", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — gather ────────────────────────────
        CpuOnlyOp { mil_name: "gather", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "gather_along_axis", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "gather_nd", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — einsum ────────────────────────────
        CpuOnlyOp { mil_name: "einsum", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — PE engine but unsupported ─────────
        CpuOnlyOp { mil_name: "slice_update", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "sliding_windows", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "reverse", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "argsort", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "reverse_sequence", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "non_maximum_suppression", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — transform ─────────────────────────
        CpuOnlyOp { mil_name: "strided_slice_update", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "dynamic_shape_cast", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "reinterpret_cast", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "col_to_im", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "im_to_col", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — sparse / quantization ─────────────
        CpuOnlyOp { mil_name: "dequantize_lut", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "extract", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "from_elements", reason: CpuOnlyReason::NoConverter },
        // ─── No ANEC converter — misc ──────────────────────────────
        CpuOnlyOp { mil_name: "get_coordinates", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "local_convolution", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "lp_norm", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "prune", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "pruning_metric", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "pruning_structure", reason: CpuOnlyReason::NoConverter },
        // ─── T-P4-08: ANEC internal ops
        // F-OPS-01: anec_fused_conv_activate REMOVED — now has MIR-to-proto emission
        CpuOnlyOp { mil_name: "anec_fused_linear_activate", reason: CpuOnlyReason::NoConverter },
        // ─── T-P4-08: Unmapped ANEC operation stubs (CPU-only) ──────
        CpuOnlyOp { mil_name: "anec_broadcast", reason: CpuOnlyReason::NoConverter },
        // F-OPS-01: anec_scaled_elementwise REMOVED — now has MIR-to-proto emission
        CpuOnlyOp { mil_name: "anec_global_arg_min_max", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_degamma", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_dirac", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_gain_offset_control", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_n_relu", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_high_precision_sigmoid", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_log2", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_trunc", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_invert", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_unflatten", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_channel_to_space", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "anec_space_to_channel", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "variable_from_tensor", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "assign_variable", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "placeholder", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "device_hint", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "nf", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "unrealized_fold", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "create_texture_tensor", reason: CpuOnlyReason::NoConverter },
        // ─── Miscellaneous — quantization constexpr ────────────────
        CpuOnlyOp {
            mil_name: "constexpr_blockwise_shift_scale",
            reason: CpuOnlyReason::Miscellaneous,
        },
        CpuOnlyOp {
            mil_name: "constexpr_sparse_blockwise_shift_scale",
            reason: CpuOnlyReason::Miscellaneous,
        },
        // NOTE: Comparison ops (equal, not_equal, greater, greater_equal, less, less_equal)
        // ARE ANE-legal per per-op support matrix rows 44-50.
        // Removed from CPU_ONLY detailed catalog.
        // ─── N-009: MILTile has PE engine but no ANEC converter.
        CpuOnlyOp { mil_name: "tile", reason: CpuOnlyReason::NoConverter },
        // ─── T-P2-12: Ops with engine assignments but no MirOpCompat emission ──
        // These ops have Some(AneEngine::NE) in default_engine() but lack
        // MirOpCompat emission code. They should be treated as CPU-only for
        // safety until emission support is added (T-66).
        CpuOnlyOp { mil_name: "max_pool", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "avg_pool", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "l2_pool", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "resize", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "resize_nearest_neighbor", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "resize_bilinear", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "upsample_nearest_neighbor", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "upsample_bilinear", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "crop_resize", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "affine", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "resample", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "depth_to_space", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "space_to_depth", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "pixel_shuffle", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "pixel_unshuffle", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "batch_to_space", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "space_to_batch", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "batch_norm", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "instance_norm", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "l2_norm", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "quantize", reason: CpuOnlyReason::NoConverter },
        CpuOnlyOp { mil_name: "dequantize", reason: CpuOnlyReason::NoConverter },
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
    ///
    /// T-P2-12: Each entry must satisfy BOTH conditions:
    ///   1. It is in CPU_ONLY_OPS (the string-based set)
    ///   2. It has default_engine() != None (confirmed by MirOp)
    /// If either condition is not met, the entry is stale and should be removed.
    const ALLOWED_DIVERGENCES: &[&str] = &[
        // These ops have Some(AneEngine::NE) or Some(AneEngine::PE) in
        // default_engine() but are in CPU_ONLY_OPS because they lack
        // MirOpCompat emission code. They will be moved to None as part
        // of T-66 (Add Remaining MirOpCompat Variants).
        "tile",           // MILTile → Some(PE), no ANEC converter (N-009)
        "max_pool",       // MILMaxPool → Some(NE), lacks MirOpCompat
        "avg_pool",       // MILAvgPool → Some(NE), lacks MirOpCompat
        "l2_pool",        // MILL2Pool → Some(NE), lacks MirOpCompat
        "resize",         // MILResize → Some(NE), lacks MirOpCompat
        "resize_nearest_neighbor", // MILResizeNearestNeighbor → Some(NE), lacks MirOpCompat
        "resize_bilinear",         // MILResizeBilinear → Some(NE), lacks MirOpCompat
        "upsample_nearest_neighbor", // MILUpsampleNearestNeighbor → Some(NE), lacks MirOpCompat
        "upsample_bilinear",       // MILUpsampleBilinear → Some(NE), lacks MirOpCompat
        "crop_resize",    // MILCropResize → Some(NE), lacks MirOpCompat
        "affine",         // MILAffine → Some(NE), lacks MirOpCompat
        "resample",       // MILResample → Some(NE), lacks MirOpCompat
        "depth_to_space", // MILDepthToSpace → Some(NE), lacks MirOpCompat
        "space_to_depth", // MILSpaceToDepth → Some(NE), lacks MirOpCompat
        "pixel_shuffle",  // MILPixelShuffle → Some(NE), lacks MirOpCompat
        "pixel_unshuffle", // MILPixelUnshuffle → Some(NE), lacks MirOpCompat
        "batch_to_space", // MILBatchToSpace → Some(NE), lacks MirOpCompat
        "space_to_batch", // MILSpaceToBatch → Some(NE), lacks MirOpCompat
        "batch_norm",     // MILBatchNorm → Some(NE), lacks MirOpCompat
        "instance_norm",  // MILInstanceNorm → Some(NE), lacks MirOpCompat
        "l2_norm",        // MILL2Norm → Some(NE), lacks MirOpCompat
        "quantize",       // MILQuantize → Some(NE), lacks MirOpCompat
        "dequantize",     // MILDequantize → Some(NE), lacks MirOpCompat
    ];

    /// T-P2-12: Verify that each entry in ALLOWED_DIVERGENCES is present
    /// in CPU_ONLY_OPS. This catches stale entries that were removed from
    /// CPU_ONLY_OPS but not from the divergence list. A stale entry means
    /// the dual-source-of-truth is inconsistent and the divergence list
    /// is wrong.
    #[test]
    fn test_allowed_divergences_are_in_cpu_only_ops() {
        let mut failures = Vec::new();
        for &name in ALLOWED_DIVERGENCES {
            if !CPU_ONLY_OPS.contains(name) {
                failures.push(name);
            }
        }
        assert!(
            failures.is_empty(),
            "T-P2-12: The following ALLOWED_DIVERGENCES entries are NOT in CPU_ONLY_OPS \
             (stale entries — remove them): {:?}",
            failures
        );
    }

    /// T-P2-12: Verify that ops with known engine assignments that appear
    /// in CPU_ONLY_OPS are accounted for in ALLOWED_DIVERGENCES.
    ///
    /// Since we cannot enumerate all MirOp variants without constructing
    /// each one, we spot-check the known divergences by verifying that
    /// each entry in ALLOWED_DIVERGENCES has an engine assignment. This
    /// catches entries that were added to ALLOWED_DIVERGENCES for ops
    /// that actually have default_engine() == None (meaning they're not
    /// divergences at all — they're correctly classified).
    #[test]
    fn test_no_ops_in_cpu_only_with_engine_assignment() {
        // For each ALLOWED_DIVERGENCES entry, verify that the corresponding
        // MirOp actually has an engine assignment. If it doesn't, the entry
        // shouldn't be in the divergence list (it's correctly CPU-only).
        let ops_with_known_engines: &[&str] = &[
            "tile",
            "max_pool", "avg_pool", "l2_pool", "resize",
            "resize_nearest_neighbor", "resize_bilinear",
            "upsample_nearest_neighbor", "upsample_bilinear",
            "crop_resize", "affine", "resample",
            "depth_to_space", "space_to_depth",
            "pixel_shuffle", "pixel_unshuffle",
            "batch_to_space", "space_to_batch",
            "batch_norm", "instance_norm", "l2_norm",
            "quantize", "dequantize",
        ];
        // Every ALLOWED_DIVERGENCES entry must be documented as having an engine
        for &name in ALLOWED_DIVERGENCES {
            assert!(
                ops_with_known_engines.contains(&name),
                "T-P2-12: ALLOWED_DIVERGENCES entry '{}' is not in the known-engine-ops list. \
                 If this op truly has default_engine() != None, add it to ops_with_known_engines. \
                 If it doesn't have an engine, remove it from ALLOWED_DIVERGENCES.",
                name
            );
        }
        // And vice versa: every known-engine op in CPU_ONLY_OPS must be in ALLOWED_DIVERGENCES
        for &name in ops_with_known_engines {
            if CPU_ONLY_OPS.contains(name) {
                assert!(
                    ALLOWED_DIVERGENCES.contains(&name),
                    "T-P2-12: Op '{}' is in CPU_ONLY_OPS and has an engine assignment, \
                     but is NOT in ALLOWED_DIVERGENCES. Add it to document the divergence.",
                    name
                );
            }
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
        // T-128: After expanding DETAILED and verifying no duplicates,
        // CPU_ONLY_OPS has exactly 152 unique entries (reduced from 154
        // after F-OPS-01 promotion of anec_fused_conv_activate and
        // anec_scaled_elementwise).
        assert!(
            CPU_ONLY_OPS.len() >= 152,
            "CPU_ONLY_OPS has {} entries, expected >= 152",
            CPU_ONLY_OPS.len()
        );
    }

    /// T-128: Verify that every op in CPU_ONLY_OPS has a corresponding
    /// entry in CPU_ONLY_OPS_DETAILED. This ensures the reason code
    /// catalog is complete.
    #[test]
    fn test_cpu_only_detailed_covers_all_ops() {
        let detailed_names: HashSet<&str> = CPU_ONLY_OPS_DETAILED
            .iter()
            .map(|op| op.mil_name)
            .collect();

        for &op_name in CPU_ONLY_OPS.iter() {
            assert!(
                detailed_names.contains(op_name),
                "op \"{}\" is in CPU_ONLY_OPS but has no entry in CPU_ONLY_OPS_DETAILED — \
                 add a CpuOnlyOp entry with an appropriate reason code",
                op_name
            );
        }
    }

    /// T-128: Verify that no duplicate entries exist in the source
    /// CPU_ONLY_OPS array. The HashSet silently deduplicates, but
    /// duplicates in the source array indicate a maintenance issue.
    #[test]
    fn test_no_duplicate_entries_in_cpu_only_ops() {
        // The source array used to build CPU_ONLY_OPS.
        // We check it for duplicates by comparing the total count
        // against the unique count.
        let source_ops: &[&str] = &[
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
            "range1d",
            "clamp",
            "prelu",
            // ANE-illegal tensor creation / conditional ops
            "fill",
            "fill_like",
            "select",
            "where",
            // Gather
            "gather",
            "gather_along_axis",
            "gather_nd",
            // Quantization constexpr
            "constexpr_blockwise_shift_scale",
            "constexpr_sparse_blockwise_shift_scale",
            // Logical
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "non_maximum_suppression",
            // Dict/List/Cell
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
            // Gradient (additional)
            "topk_grad",
            "gather_grad",
            "scatter_along_axis_grad",
            "reverse_sequence",
            // T-22: Activation variants
            "relu6",
            "sigmoid_hard",
            "thresholded_relu",
            "clamped_relu",
            "linear_activation",
            "scaled_tanh",
            "softplus_parametric",
            // T-22: Elementwise
            "threshold",
            "inverse",
            "einsum",
            // T-47: PE engine but no ANEC converter
            "slice_update",
            "sliding_windows",
            "reverse",
            "argsort",
            // T-49: Control flow
            "return",
            // T-49: Type check
            "is_finite",
            "is_infinite",
            "is_nan",
            // T-67: Fixed names
            "neg",
            "round",
            // T-49: Transform
            "strided_slice_update",
            "dynamic_shape_cast",
            "reinterpret_cast",
            "col_to_im",
            "im_to_col",
            // T-49: Sparse/buffer
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
            // N-009: MILTile has PE engine but no ANEC converter
            "tile",
        ];

        let unique: HashSet<&str> = source_ops.iter().copied().collect();
        assert_eq!(
            source_ops.len(),
            unique.len(),
            "CPU_ONLY_OPS source array has {} entries but only {} are unique — \
             duplicate entries must be removed",
            source_ops.len(),
            unique.len()
        );

        // Find and report specific duplicates
        let mut seen: HashSet<&str> = HashSet::new();
        let mut duplicates: Vec<&str> = Vec::new();
        for &op in source_ops {
            if !seen.insert(op) {
                duplicates.push(op);
            }
        }
        assert!(
            duplicates.is_empty(),
            "Duplicate entries found in CPU_ONLY_OPS source array: {:?}",
            duplicates
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
            "scatter",
            "scatter_along_axis",
            "scatter_nd",
            // Misc CPU-only
            "non_maximum_suppression",
            // RNN/LSTM/GRU
            "rnn",
            "gru",
            "lstm",
            // Control flow
            "cond",
            "while_loop",
            // List ops
            "make_list",
            "list_length",
            "list_write",
            "list_read",
            "list_gather",
            "list_scatter",
            // Random
            "random_bernoulli",
            "random_normal",
            "random_uniform",
            "random_categorical",
            // Cumsum
            "cumsum",
            // Conditional / tensor creation (no ANE converter)
            "select",
            "where",
            "fill",
            "fill_like",
            "one_hot",
            "non_zero",
            "range1d",
            "shape",
            // Gather (ANE plannability ~0.26)
            "gather",
            "gather_along_axis",
            "gather_nd",
            // T-22: CPU-only ops moved from PE/NE pipeline
            "acos",
            "asin",
            "atan",
            "atanh",
            "tan",
            "cosh",
            "sinh",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "relu6",
            "sigmoid_hard",
            "thresholded_relu",
            "clamped_relu",
            "linear_activation",
            "prelu",
            "softsign",
            "scaled_tanh",
            "softplus",
            "softplus_parametric",
            "threshold",
            "inverse",
            "modulo",
            "clamp",
            "band_part",
            "reverse_sequence",
            "einsum",
            // T-47: PE engine but no ANEC converter
            "slice_update",
            "sliding_windows",
            "reverse",
            "argsort",
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
        assert!(
            is_cpu_only("round"),
            "\"round\" should be CPU-only (MILRound has no ANEC converter)"
        );
    }

    /// T-67: Verify removed names are NOT in CPU_ONLY_OPS.
    #[test]
    fn test_t67_removed_names_not_in_cpu_only() {
        assert!(
            !is_cpu_only("negative"),
            "\"negative\" should not be in CPU_ONLY_OPS — use \"neg\" instead"
        );
        assert!(
            !is_cpu_only("reverse_square_root"),
            "\"reverse_square_root\" should not be CPU-only — rsqrt IS ANE-legal"
        );
        assert!(
            !is_cpu_only("rint"),
            "\"rint\" should not be in CPU_ONLY_OPS — use \"round\" instead"
        );
        assert!(
            !is_cpu_only("reciprocal"),
            "\"reciprocal\" is dead code — no MirOp produces this name"
        );
        assert!(!is_cpu_only("signbit"), "\"signbit\" is dead code — no MirOp produces this name");
    }

    /// T-128: Verify reason codes are correct for representative ops
    /// across all CpuOnlyReason categories.
    #[test]
    fn test_cpu_only_reason_codes_by_category() {
        // TrigInverse
        assert_eq!(get_cpu_only_reason("acos"), Some(&CpuOnlyReason::TrigInverse));
        assert_eq!(get_cpu_only_reason("atan2"), Some(&CpuOnlyReason::TrigInverse));
        assert_eq!(get_cpu_only_reason("tan"), Some(&CpuOnlyReason::TrigInverse));
        // Hyperbolic
        assert_eq!(get_cpu_only_reason("sinh"), Some(&CpuOnlyReason::Hyperbolic));
        assert_eq!(get_cpu_only_reason("cosh"), Some(&CpuOnlyReason::Hyperbolic));
        // Logical
        assert_eq!(get_cpu_only_reason("nand"), Some(&CpuOnlyReason::Logical));
        assert_eq!(get_cpu_only_reason("bitwise_and"), Some(&CpuOnlyReason::Logical));
        assert_eq!(get_cpu_only_reason("popcount"), Some(&CpuOnlyReason::Logical));
        assert_eq!(get_cpu_only_reason("logical_xor"), Some(&CpuOnlyReason::Logical));
        // ComplexNumber
        assert_eq!(get_cpu_only_reason("create_complex"), Some(&CpuOnlyReason::ComplexNumber));
        assert_eq!(get_cpu_only_reason("conjugate"), Some(&CpuOnlyReason::ComplexNumber));
        // Fft
        assert_eq!(get_cpu_only_reason("fast_fourier_transform"), Some(&CpuOnlyReason::Fft));
        assert_eq!(get_cpu_only_reason("hermitean_to_real_fft"), Some(&CpuOnlyReason::Fft));
        // MatrixAlgebra
        assert_eq!(get_cpu_only_reason("matrix_inverse"), Some(&CpuOnlyReason::MatrixAlgebra));
        // Rnn
        assert_eq!(get_cpu_only_reason("gru"), Some(&CpuOnlyReason::Rnn));
        assert_eq!(get_cpu_only_reason("rnn"), Some(&CpuOnlyReason::Rnn));
        assert_eq!(get_cpu_only_reason("singlegate_rnn"), Some(&CpuOnlyReason::Rnn));
        // Gradient
        assert_eq!(get_cpu_only_reason("relu_grad"), Some(&CpuOnlyReason::Gradient));
        assert_eq!(get_cpu_only_reason("gather_grad"), Some(&CpuOnlyReason::Gradient));
        assert_eq!(get_cpu_only_reason("topk_grad"), Some(&CpuOnlyReason::Gradient));
        // Cumulative
        assert_eq!(get_cpu_only_reason("cumsum"), Some(&CpuOnlyReason::Cumulative));
        assert_eq!(get_cpu_only_reason("cumulative_maximum"), Some(&CpuOnlyReason::Cumulative));
        // Random
        assert_eq!(get_cpu_only_reason("random_truncated_normal"), Some(&CpuOnlyReason::Random));
        assert_eq!(get_cpu_only_reason("random_bernoulli"), Some(&CpuOnlyReason::Random));
        // ControlFlow
        assert_eq!(get_cpu_only_reason("for"), Some(&CpuOnlyReason::ControlFlow));
        assert_eq!(get_cpu_only_reason("cond"), Some(&CpuOnlyReason::ControlFlow));
        assert_eq!(get_cpu_only_reason("make_list"), Some(&CpuOnlyReason::ControlFlow));
        assert_eq!(get_cpu_only_reason("dict_read"), Some(&CpuOnlyReason::ControlFlow));
        assert_eq!(get_cpu_only_reason("return"), Some(&CpuOnlyReason::ControlFlow));
        // Scatter
        assert_eq!(get_cpu_only_reason("scatter_along_axis"), Some(&CpuOnlyReason::Scatter));
        // Sparse
        assert_eq!(get_cpu_only_reason("sparse_tensor_storage"), Some(&CpuOnlyReason::Sparse));
        assert_eq!(get_cpu_only_reason("buffer_tensor"), Some(&CpuOnlyReason::Sparse));
        // ShapeQuery
        assert_eq!(get_cpu_only_reason("dimension_size"), Some(&CpuOnlyReason::ShapeQuery));
        // NoConverter
        assert_eq!(get_cpu_only_reason("relu6"), Some(&CpuOnlyReason::NoConverter));
        assert_eq!(get_cpu_only_reason("gather"), Some(&CpuOnlyReason::NoConverter));
        assert_eq!(get_cpu_only_reason("fill"), Some(&CpuOnlyReason::NoConverter));
        assert_eq!(get_cpu_only_reason("neg"), Some(&CpuOnlyReason::NoConverter));
        assert_eq!(get_cpu_only_reason("einsum"), Some(&CpuOnlyReason::NoConverter));
        assert_eq!(get_cpu_only_reason("argsort"), Some(&CpuOnlyReason::NoConverter));
        // Miscellaneous
        assert_eq!(
            get_cpu_only_reason("constexpr_blockwise_shift_scale"),
            Some(&CpuOnlyReason::Miscellaneous)
        );
        // Non-CPU-only op returns None
        assert_eq!(get_cpu_only_reason("conv"), None);
    }

    /// T-P2-04: Validate that every op in CPU_ONLY_OPS appears in the
    /// knowledge store seed JSON (`knowledge/cpu_only_ops_seed.json`).
    ///
    /// This test ensures the seed JSON stays synchronized with the source
    /// code ground truth. If an op is added to CPU_ONLY_OPS but not to
    /// the seed, this test will fail.
    #[test]
    fn test_seed_json_covers_all_cpu_only_ops() {
        use std::fs;
        use std::path::Path;

        // Resolve knowledge/ directory relative to the crate.
        let candidates = [
            "../../knowledge",           // crate-relative (crates/passes/)
            "../../../knowledge",        // deeper nesting fallback
            "knowledge",                 // workspace-root run
        ];
        let knowledge_dir = candidates
            .iter()
            .find(|d| Path::new(d).join("cpu_only_ops_seed.json").exists());

        let Some(knowledge_dir) = knowledge_dir else {
            eprintln!(
                "Skipping test_seed_json_covers_all_cpu_only_ops: \
                 knowledge/cpu_only_ops_seed.json not found"
            );
            return;
        };

        let seed_path = Path::new(knowledge_dir).join("cpu_only_ops_seed.json");
        let json_str = fs::read_to_string(&seed_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", seed_path.display(), e));
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("Invalid JSON in {}: {}", seed_path.display(), e));

        let entries = parsed
            .get("entries")
            .and_then(|e| e.as_array())
            .expect("cpu_only_ops_seed.json: missing 'entries' array");

        // Collect all mil_names from the seed
        let seed_names: HashSet<String> = entries
            .iter()
            .filter_map(|e| {
                e.get("payload")
                    .and_then(|p| p.get("mil_name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        // Verify every op in CPU_ONLY_OPS appears in the seed
        let mut missing = Vec::new();
        for &op_name in CPU_ONLY_OPS.iter() {
            if !seed_names.contains(op_name) {
                missing.push(op_name);
            }
        }

        assert!(
            missing.is_empty(),
            "T-P2-04: The following ops are in CPU_ONLY_OPS but missing from \
             cpu_only_ops_seed.json — add entries for them: {:?}",
            missing
        );

        // Also verify the seed doesn't contain ops NOT in CPU_ONLY_OPS
        let mut extra = Vec::new();
        for name in &seed_names {
            if !CPU_ONLY_OPS.contains(name.as_str()) {
                extra.push(name.clone());
            }
        }
        assert!(
            extra.is_empty(),
            "T-P2-04: The following ops are in cpu_only_ops_seed.json but NOT in \
             CPU_ONLY_OPS — remove them from the seed or add to CPU_ONLY_OPS: {:?}",
            extra
        );
    }

    /// T-P2-04: Validate that entries in ane_op_family_matrix.json that
    /// are empirically CPU-only have practical_status: "cpu_only".
    #[test]
    fn test_matrix_cpu_only_entries_have_practical_status() {
        use std::fs;
        use std::path::Path;

        let candidates = [
            "../../knowledge",
            "../../../knowledge",
            "knowledge",
        ];
        let knowledge_dir = candidates
            .iter()
            .find(|d| Path::new(d).join("ane_op_family_matrix.json").exists());

        let Some(knowledge_dir) = knowledge_dir else {
            eprintln!(
                "Skipping test_matrix_cpu_only_entries_have_practical_status: \
                 knowledge/ane_op_family_matrix.json not found"
            );
            return;
        };

        let matrix_path = Path::new(knowledge_dir).join("ane_op_family_matrix.json");
        let json_str = fs::read_to_string(&matrix_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", matrix_path.display(), e));
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("Invalid JSON in {}: {}", matrix_path.display(), e));

        let entries = parsed
            .get("entries")
            .and_then(|e| e.as_array())
            .expect("ane_op_family_matrix.json: missing 'entries' array");

        // Find matrix entries where the op is CPU-only but lacks practical_status
        let mut missing = Vec::new();
        for entry in entries {
            let mil_name = entry
                .get("payload")
                .and_then(|p| p.get("mil_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if CPU_ONLY_OPS.contains(mil_name) {
                let has_practical = entry
                    .get("payload")
                    .and_then(|p| p.get("practical_status"))
                    .is_some();
                if !has_practical {
                    missing.push(mil_name.to_string());
                }
            }
        }

        assert!(
            missing.is_empty(),
            "T-P2-04: The following CPU-only ops in ane_op_family_matrix.json \
             are missing 'practical_status: \"cpu_only\"': {:?}",
            missing
        );
    }
}
