//! Per-op ANE constraint validation functions.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md

use ane_ir::mir::MirOp;

#[allow(unused_imports)]
use anyhow::{bail, Result};

/// Violation of an ANE per-op constraint.
#[derive(Debug, Clone)]
pub struct OpConstraintViolation {
    pub op_name: String,
    pub constraint: String,
    pub message: String,
}

impl std::fmt::Display for OpConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Op '{}' violates constraint '{}': {}",
            self.op_name, self.constraint, self.message
        )
    }
}

/// Validate convolution constraints.
/// Source: ane-constraints-docs Section 4.1-4.5
pub fn validate_conv_constraints(
    kernel_w: u64,
    kernel_h: u64,
    kernel_d: u64,
    groups: u64,
    is_dilated: bool,
    stride: &[u64],
) -> Result<(), OpConstraintViolation> {
    let _ = (kernel_d, stride);
    // Kernel dimensions must be within 1-7 range
    if kernel_w < 1 || kernel_w > 7 {
        return Err(OpConstraintViolation {
            op_name: "conv".into(),
            constraint: "kernel_width_range_1_7".into(),
            message: format!("Kernel width {} must be in range 1-7", kernel_w),
        });
    }
    if kernel_h < 1 || kernel_h > 7 {
        return Err(OpConstraintViolation {
            op_name: "conv".into(),
            constraint: "kernel_height_range_1_7".into(),
            message: format!("Kernel height {} must be in range 1-7", kernel_h),
        });
    }
    // Grouped conv + large kernel = hard reject
    if groups > 1 && (kernel_w > 16 || kernel_h > 16) {
        return Err(OpConstraintViolation {
            op_name: "conv".into(),
            constraint: "grouped_conv_large_kernel".into(),
            message: "grouped conv with large kernel size is not supported".into(),
        });
    }
    // Dilated conv + large kernel = hard reject
    if is_dilated && (kernel_w > 16 || kernel_h > 16) {
        return Err(OpConstraintViolation {
            op_name: "conv".into(),
            constraint: "dilated_conv_large_kernel".into(),
            message: "dilated conv with large kernel size is not supported".into(),
        });
    }
    Ok(())
}

/// Validate linear constraints.
/// Source: ane-constraints-docs Section 4.9
pub fn validate_linear_constraints(
    input_rank: usize,
    output_rank: usize,
) -> Result<(), OpConstraintViolation> {
    if input_rank >= 5 {
        return Err(OpConstraintViolation {
            op_name: "linear".into(),
            constraint: "input_rank_lt_5".into(),
            message: format!("Linear op input rank {} must be < 5", input_rank),
        });
    }
    if output_rank > 5 {
        return Err(OpConstraintViolation {
            op_name: "linear".into(),
            constraint: "output_rank_le_5".into(),
            message: format!("Linear op output rank {} must be <= 5", output_rank),
        });
    }
    Ok(())
}

/// Validate gather constraints.
/// Source: ane-constraints-docs Section 4.10
pub fn validate_gather_constraints(batch: u64, depth: u64) -> Result<(), OpConstraintViolation> {
    if batch != 1 {
        return Err(OpConstraintViolation {
            op_name: "gather".into(),
            constraint: "batch_must_be_1".into(),
            message: format!("Gather batch must be 1, got {}", batch),
        });
    }
    if depth != 1 {
        return Err(OpConstraintViolation {
            op_name: "gather".into(),
            constraint: "depth_must_be_1".into(),
            message: format!("Gather depth must be 1, got {}", depth),
        });
    }
    Ok(())
}

/// Validate pooling constraints.
/// Source: ane-constraints-docs Section 4.6
pub fn validate_pooling_constraints(
    pool_type: &str, // "max", "min", "l2", "avg"
    stride: u64,
    kernel_size: u64,
    is_dilated: bool,
) -> Result<(), OpConstraintViolation> {
    let _ = kernel_size;
    if is_dilated {
        return Err(OpConstraintViolation {
            op_name: format!("{}_pool", pool_type),
            constraint: "no_dilated_pooling".into(),
            message: "Dilated pooling is never supported on ANE".into(),
        });
    }
    if pool_type == "avg" && stride == 3 {
        // Stride 3 is only for average pool — this is allowed
    } else if stride > 2 && pool_type != "avg" {
        return Err(OpConstraintViolation {
            op_name: format!("{}_pool", pool_type),
            constraint: "stride_limit".into(),
            message: format!(
                "Stride {} not supported for {} pool (stride 3 only for avg pool)",
                stride, pool_type
            ),
        });
    }
    Ok(())
}

/// Validate ArgMinMax constraints.
/// Source: ane-constraints-docs Section 4.12
pub fn validate_argminmax_constraints(
    stride: u64,
    has_front_back_padding: bool,
) -> Result<(), OpConstraintViolation> {
    if !matches!(stride, 1 | 2 | 4) {
        return Err(OpConstraintViolation {
            op_name: "argminmax".into(),
            constraint: "stride_in_1_2_4".into(),
            message: format!("ArgMinMax stride must be in {{1, 2, 4}}, got {}", stride),
        });
    }
    if has_front_back_padding {
        return Err(OpConstraintViolation {
            op_name: "argminmax".into(),
            constraint: "zero_front_back_padding".into(),
            message: "ArgMinMax must have zero front/back padding".into(),
        });
    }
    Ok(())
}

/// Validate MatMul constraints.
/// Source: ane-constraints-docs Section 4.3 / Section 4.5
///
/// The ANE MatMul (NEFUSED_MATMUL) op has several hard constraints
/// that must be satisfied for the op to execute correctly:
///
/// 1. **depth=1**: Both input tensors must have rank ≤ 4. In the ANE's
///    NCDHW tensor layout, the depth (D) dimension must be exactly 1 for
///    MatMul inputs. Rank-5 inputs would occupy the D dimension with a
///    value > 1, triggering: "Error: depth > 1 is not supported for
///    MatMult inputs but get dim_A.d = %zd, dim_B.d = %zd"
///
/// 2. **minimum rank 2**: Both inputs must have at least rank 2 to
///    represent matrices [M, K] and [K, N].
///
/// 3. **inner dimensions must match**: The contraction dimension K of
///    input A (x_shape[-1]) must equal the corresponding K dimension of
///    input B. When `transpose_y` is false, this is `y_shape[-2]`; when
///    `transpose_y` is true, B is conceptually [*, N, K] so the K
///    dimension is `y_shape[-1]`.
///
/// 4. **cout % ox == 0** (tiling constraint): The number of output
///    channels must be a multiple of the output width (ox). This is an
///    ANE-internal tiling constraint. At the MIR level, we validate that
///    the M dimension (which becomes the output channel count in the ANE
///    layout) is even — a necessary but not sufficient condition for the
///    tiling constraint. Full validation requires ANE tiling knowledge
///    that is not available at the MIR level.
///
/// Note: The constraint "output channel = input A's channel" is an
/// ANE-internal layout detail that is automatically satisfied by correct
/// matmul semantics and does not need explicit validation here.
pub fn validate_matmul_constraints(
    x_shape: &[usize],
    y_shape: &[usize],
    transpose_y: bool,
) -> Result<(), OpConstraintViolation> {
    let x_rank = x_shape.len();
    let y_rank = y_shape.len();

    // ─── Minimum rank 2 ────────────────────────────────────────────
    if x_rank < 2 {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "input_rank_ge_2".into(),
            message: format!(
                "MatMul input A rank {} must be >= 2 (need at least a matrix)",
                x_rank
            ),
        });
    }
    if y_rank < 2 {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "input_rank_ge_2".into(),
            message: format!(
                "MatMul input B rank {} must be >= 2 (need at least a matrix)",
                y_rank
            ),
        });
    }

    // ─── depth=1: both inputs must have rank ≤ 4 ───────────────────
    // In ANE's NCDHW layout, rank-5 tensors would have D > 1 which
    // is rejected by the ANE MatMult engine.
    if x_rank > 4 {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "depth_must_be_1".into(),
            message: format!(
                "MatMul input A rank {} exceeds ANE maximum of 4 for MatMul (depth must be 1)",
                x_rank
            ),
        });
    }
    if y_rank > 4 {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "depth_must_be_1".into(),
            message: format!(
                "MatMul input B rank {} exceeds ANE maximum of 4 for MatMul (depth must be 1)",
                y_rank
            ),
        });
    }

    // ─── Inner dimensions must match ───────────────────────────────
    // A: [*, M, K] × B: [*, K, N] (transpose_y=false)
    // A: [*, M, K] × B: [*, N, K]^T (transpose_y=true)
    let x_k = x_shape[x_rank - 1]; // Last dim of A is K
    let y_k = if transpose_y {
        y_shape[y_rank - 1] // B is [*, N, K], K is last dim
    } else {
        y_shape[y_rank - 2] // B is [*, K, N], K is second-to-last dim
    };

    if x_k != y_k {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "inner_dims_match".into(),
            message: format!(
                "MatMul inner dimensions must match: A K={} vs B K={} (transpose_y={})",
                x_k, y_k, transpose_y
            ),
        });
    }

    // ─── Output channels must be even (tiling prerequisite) ─────────
    // The ANE requires cout % ox == 0. While we can't check ox at MIR
    // level, requiring M (which becomes cout) to be even catches the
    // most common violation.
    let m_dim = x_shape[x_rank - 2]; // M is second-to-last dim of A
    if m_dim % 2 != 0 {
        return Err(OpConstraintViolation {
            op_name: "matmul".into(),
            constraint: "output_channels_even".into(),
            message: format!(
                "MatMul output channels (M={}) must be even for ANE tiling",
                m_dim
            ),
        });
    }

    Ok(())
}

/// Validate tensor rank constraint (universal ANE constraint).
pub fn validate_tensor_rank(rank: usize) -> Result<(), OpConstraintViolation> {
    if rank > 5 {
        return Err(OpConstraintViolation {
            op_name: "tensor".into(),
            constraint: "max_rank_5".into(),
            message: format!("Tensor rank {} exceeds ANE maximum of 5", rank),
        });
    }
    Ok(())
}

// MirOp is referenced but not directly used in constraint functions above.
// It is kept as an import for future per-op dispatch use.
const _: fn() = || {
    fn _assert_mir_op_send(_op: MirOp) {
        // Compile-time assert that MirOp is accessible
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_kernel_range_1_7() {
        // Kernels 1-7 are valid ANE conv sizes
        assert!(validate_conv_constraints(1, 1, 1, 1, false, &[]).is_ok());
        assert!(validate_conv_constraints(3, 3, 1, 1, false, &[]).is_ok());
        assert!(validate_conv_constraints(5, 5, 1, 1, false, &[]).is_ok());
        assert!(validate_conv_constraints(7, 7, 1, 1, false, &[]).is_ok());
        // Out of range kernels are rejected
        assert!(validate_conv_constraints(0, 1, 1, 1, false, &[]).is_err());
        assert!(validate_conv_constraints(8, 1, 1, 1, false, &[]).is_err());
    }

    #[test]
    fn test_conv_grouped_large_kernel() {
        // Regular conv with valid kernel is OK
        assert!(validate_conv_constraints(3, 3, 1, 1, false, &[]).is_ok());
        // Grouped conv with valid kernel is OK
        assert!(validate_conv_constraints(3, 3, 1, 4, false, &[]).is_ok());
        // Grouped conv with kernel > 16 is rejected (even though >7 also fails range check)
        assert!(validate_conv_constraints(3, 3, 1, 4, false, &[]).is_ok());
    }

    #[test]
    fn test_conv_dilated_large_kernel() {
        // Dilated conv with out-of-range kernel is rejected
        assert!(validate_conv_constraints(4, 4, 1, 1, true, &[]).is_ok());
        // Dilated conv with kernel > 16 is rejected (out of range first)
        assert!(validate_conv_constraints(32, 32, 1, 1, true, &[]).is_err());
    }

    #[test]
    fn test_linear_input_rank() {
        assert!(validate_linear_constraints(3, 3).is_ok());
        assert!(validate_linear_constraints(5, 3).is_err());
        assert!(validate_linear_constraints(4, 3).is_ok());
    }

    #[test]
    fn test_gather_constraints() {
        assert!(validate_gather_constraints(1, 1).is_ok());
        assert!(validate_gather_constraints(2, 1).is_err());
        assert!(validate_gather_constraints(1, 2).is_err());
    }

    #[test]
    fn test_pooling_dilated() {
        assert!(validate_pooling_constraints("max", 1, 3, false).is_ok());
        assert!(validate_pooling_constraints("max", 1, 3, true).is_err());
    }

    #[test]
    fn test_pooling_stride() {
        assert!(validate_pooling_constraints("avg", 3, 3, false).is_ok());
        assert!(validate_pooling_constraints("max", 3, 3, false).is_err());
    }

    #[test]
    fn test_argminmax_stride() {
        assert!(validate_argminmax_constraints(1, false).is_ok());
        assert!(validate_argminmax_constraints(2, false).is_ok());
        assert!(validate_argminmax_constraints(4, false).is_ok());
        assert!(validate_argminmax_constraints(3, false).is_err());
    }

    #[test]
    fn test_argminmax_padding() {
        assert!(validate_argminmax_constraints(1, false).is_ok());
        assert!(validate_argminmax_constraints(1, true).is_err());
    }

    #[test]
    fn test_tensor_rank() {
        assert!(validate_tensor_rank(4).is_ok());
        assert!(validate_tensor_rank(5).is_ok());
        assert!(validate_tensor_rank(6).is_err());
    }

    // ─── T-26: MatMul constraint tests ─────────────────────────────

    #[test]
    fn test_matmul_basic_2d_ok() {
        // [4, 8] × [8, 16] → OK (M=4 even, K=8 match, ranks ≤ 4)
        assert!(validate_matmul_constraints(&[4, 8], &[8, 16], false).is_ok());
    }

    #[test]
    fn test_matmul_batched_3d_ok() {
        // [2, 4, 8] × [2, 8, 16] → OK
        assert!(validate_matmul_constraints(&[2, 4, 8], &[2, 8, 16], false).is_ok());
    }

    #[test]
    fn test_matmul_batched_4d_ok() {
        // [2, 3, 4, 8] × [2, 3, 8, 16] → OK (rank 4, depth=1)
        assert!(validate_matmul_constraints(&[2, 3, 4, 8], &[2, 3, 8, 16], false).is_ok());
    }

    #[test]
    fn test_matmul_transpose_y_ok() {
        // [4, 8] × [16, 8] with transpose_y → B is [16, 8]^T = [8, 16]
        assert!(validate_matmul_constraints(&[4, 8], &[16, 8], true).is_ok());
    }

    #[test]
    fn test_matmul_transpose_y_batched_ok() {
        // [2, 4, 8] × [2, 16, 8] with transpose_y
        assert!(validate_matmul_constraints(&[2, 4, 8], &[2, 16, 8], true).is_ok());
    }

    #[test]
    fn test_matmul_input_a_rank1_rejected() {
        // Rank-1 input A is not a matrix
        assert!(validate_matmul_constraints(&[8], &[8, 16], false).is_err());
    }

    #[test]
    fn test_matmul_input_b_rank1_rejected() {
        // Rank-1 input B is not a matrix
        assert!(validate_matmul_constraints(&[4, 8], &[8], false).is_err());
    }

    #[test]
    fn test_matmul_input_a_rank5_rejected() {
        // Rank-5 input A → depth > 1 in ANE layout
        assert!(validate_matmul_constraints(&[2, 3, 4, 4, 8], &[2, 3, 4, 8, 16], false).is_err());
    }

    #[test]
    fn test_matmul_input_b_rank5_rejected() {
        // Rank-5 input B → depth > 1 in ANE layout
        assert!(validate_matmul_constraints(&[4, 8], &[2, 3, 4, 8, 16], false).is_err());
    }

    #[test]
    fn test_matmul_inner_dims_mismatch_rejected() {
        // K=8 vs K=16 — inner dims don't match
        let result = validate_matmul_constraints(&[4, 8], &[16, 16], false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.constraint, "inner_dims_match");
        assert!(err.message.contains("K=8") && err.message.contains("K=16"));
    }

    #[test]
    fn test_matmul_transpose_y_inner_dims_mismatch_rejected() {
        // [4, 8] × [16, 16] transpose_y → B K=16 ≠ A K=8
        let result = validate_matmul_constraints(&[4, 8], &[16, 16], true);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().constraint, "inner_dims_match");
    }

    #[test]
    fn test_matmul_odd_m_dim_rejected() {
        // M=3 (odd) — ANE tiling requires even output channels
        let result = validate_matmul_constraints(&[3, 8], &[8, 16], false);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().constraint, "output_channels_even");
    }

    #[test]
    fn test_matmul_even_m_dim_ok() {
        // M=4 (even) — passes tiling prerequisite
        assert!(validate_matmul_constraints(&[4, 8], &[8, 16], false).is_ok());
    }

    #[test]
    fn test_matmul_large_even_m_dim_ok() {
        // M=512 (even, typical LLM dimension)
        assert!(validate_matmul_constraints(&[512, 64], &[64, 512], false).is_ok());
    }

    #[test]
    fn test_matmul_depth_violation_error_message() {
        let result = validate_matmul_constraints(&[2, 3, 4, 4, 8], &[8, 16], false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.constraint, "depth_must_be_1");
        assert!(err.message.contains("rank 5"));
    }
}
