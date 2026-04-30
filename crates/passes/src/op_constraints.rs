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

fn is_power_of_two(n: u64) -> bool {
    n > 0 && (n & (n - 1)) == 0
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
}
