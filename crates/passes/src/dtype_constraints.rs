//! Data type and format constraint validation for ANE placement.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md Section 5
//!
//! T-35 (I-14): Expanded with Int4, UInt4, E4M3, E5M2, UInt16 dtype
//! variants and their ANE constraint enforcement.

use ane_ir::ane_target::AneFamily;
use ane_ir::mir::MilDtype;

/// Errors from dtype constraint validation.
#[derive(Debug, Clone)]
pub enum DtypeConstraintError {
    /// This dtype is rejected on ANE for the given family.
    RejectedDtype { dtype: String, family: String, message: String },
    /// Quantization format violation.
    QuantFormatViolation { message: String },
    /// Version-gated dtype not available on this family.
    VersionGatedDtype { dtype: String, min_family: String, actual_family: String },
    /// Int4-specific constraint violation.
    Int4ConstraintViolation { message: String },
    /// Float8 (E4M3/E5M2) constraint violation.
    Float8ConstraintViolation { message: String },
}

impl std::fmt::Display for DtypeConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RejectedDtype { dtype, family, message } => {
                write!(f, "Dtype '{}' rejected on family '{}': {}", dtype, family, message)
            }
            Self::QuantFormatViolation { message } => {
                write!(f, "Quantization format violation: {}", message)
            }
            Self::VersionGatedDtype { dtype, min_family, actual_family } => write!(
                f,
                "Dtype '{}' requires family {}+, got {}",
                dtype, min_family, actual_family
            ),
            Self::Int4ConstraintViolation { message } => {
                write!(f, "Int4 constraint violation: {}", message)
            }
            Self::Float8ConstraintViolation { message } => {
                write!(f, "Float8 constraint violation: {}", message)
            }
        }
    }
}

/// Check if a dtype is legal for ANE compute on the given family.
///
/// Based on the ANE constraint canon (§5.1):
/// - FP16: always legal (primary ANE dtype)
/// - FP32: legal for some ops but not compute
/// - Int8/UInt8: legal for quantized paths
/// - Int16: limited support for some ops
/// - Int4: constrained — must use interleave factor 8
/// - UInt4: constrained — must use interleave factor 8
/// - E4M3: architecture-dependent — NOT supported on most families;
///   limited support on A17/A18 only
/// - E5M2: NOT supported on any ANE family
/// - UInt16: limited support
/// - Int32: NOT supported for compute
/// - FP64: NOT supported on ANE
/// - Bool: limited support
pub fn is_dtype_ane_legal(
    dtype: &MilDtype,
    family: &AneFamily,
) -> Result<(), DtypeConstraintError> {
    match dtype {
        // FP16 is the primary ANE dtype — always legal
        MilDtype::Fp16 => Ok(()),
        // FP32 is allowed for some ops but not compute — conditional
        MilDtype::Fp32 => Ok(()), // allowed for input/output but may be downcast
        // Int8/UInt8 — legal for quantized paths
        MilDtype::Int8 | MilDtype::UInt8 => Ok(()),
        // Int16 — limited support
        MilDtype::Int16 => Ok(()), // supported for some ops
        // Int4 — constrained: legal but requires interleave factor 8
        MilDtype::Int4 => Ok(()), // constrained: caller must also check interleave==8
        // UInt4 — constrained: legal but requires interleave factor 8
        MilDtype::UInt4 => Ok(()), // constrained: caller must also check interleave==8
        // E4M3 (FP8) — architecture-dependent
        // NOT supported on most families; limited support on A17/A18 only.
        // Per the per-op support matrix: E4M3/E5M2 row shows ❌ for A11-A16,
        // ⚠️ for A17/A18.
        MilDtype::E4M3 => {
            if family.supports_e4m3() {
                Ok(())
            } else {
                Err(DtypeConstraintError::VersionGatedDtype {
                    dtype: "E4M3".into(),
                    min_family: "A17".into(),
                    actual_family: format!("{:?}", family),
                })
            }
        }
        // E5M2 (FP8) — NOT supported on any ANE family
        // Error message from ANE: "E4M3 or E5M2 format not supported"
        MilDtype::E5M2 => Err(DtypeConstraintError::RejectedDtype {
            dtype: "E5M2".into(),
            family: format!("{:?}", family),
            message: "E4M3 or E5M2 format not supported on ANE".into(),
        }),
        // UInt16 — limited support
        MilDtype::UInt16 => Ok(()),
        // Int32 — NOT supported for compute on ANE
        MilDtype::Int32 => Err(DtypeConstraintError::RejectedDtype {
            dtype: "Int32".into(),
            family: format!("{:?}", family),
            message: "32 bit format not supported for ANE compute".into(),
        }),
        // Bool — limited support
        MilDtype::Bool => Ok(()),
        // FP64 — NOT supported on ANE
        MilDtype::Fp64 => Err(DtypeConstraintError::RejectedDtype {
            dtype: "Fp64".into(),
            family: format!("{:?}", family),
            message: "64-bit floating point not supported on ANE".into(),
        }),
    }
}

/// Check if Int4 per-output-channel (per-cout) dequantization is supported.
///
/// ANE constraint: "Int4 Per-Cout Dequant is not supported"
/// Int4 can dequant, but NOT using per-output-channel scale.
pub fn is_int4_per_cout_dequant_supported() -> bool {
    false
}

/// Check if E4M3 quantization supports zero point.
///
/// ANE constraint: "Zero point is not supported for quant with E4M3 output format"
pub fn is_e4m3_zero_point_supported() -> bool {
    false
}

/// Validate that Int4 uses the correct interleave factor.
///
/// ANE constraint: "Tensor with the int4 format must have an interleave factor of 8"
pub fn validate_int4_interleave(interleave: usize) -> Result<(), DtypeConstraintError> {
    if interleave != 8 {
        return Err(DtypeConstraintError::Int4ConstraintViolation {
            message: format!(
                "Int4 format requires interleave factor 8, got {}",
                interleave
            ),
        });
    }
    Ok(())
}

/// Validate that UInt4 uses the correct interleave factor.
///
/// UInt4 follows the same interleave=8 constraint as Int4 for 4-bit packed
/// tensors on the ANE.
pub fn validate_uint4_interleave(interleave: usize) -> Result<(), DtypeConstraintError> {
    if interleave != 8 {
        return Err(DtypeConstraintError::Int4ConstraintViolation {
            message: format!(
                "UInt4 format requires interleave factor 8, got {}",
                interleave
            ),
        });
    }
    Ok(())
}

/// Validate quantization format constraints.
///
/// Per the ANE constraint canon (§5.2):
/// - Quantize input must be fp16 or fp32
/// - Quantize output must be int8, uint8, e4m3, or e5m2
pub fn validate_quantization_constraints(
    quant_input_dtype: &MilDtype,
    quant_output_dtype: &MilDtype,
) -> Result<(), DtypeConstraintError> {
    // Quantize input must be fp16 or fp32
    if !matches!(quant_input_dtype, MilDtype::Fp16 | MilDtype::Fp32) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!("Quantize input must be fp16 or fp32, got {:?}", quant_input_dtype),
        });
    }
    // Quantize output must be int8, uint8, e4m3, or e5m2
    // Per ANE canon: "Quant layer must have int8, uint8, e4m3 or e5m2 output format"
    if !matches!(
        quant_output_dtype,
        MilDtype::Int8 | MilDtype::UInt8 | MilDtype::E4M3 | MilDtype::E5M2
    ) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!(
                "Quantize output must be int8, uint8, e4m3, or e5m2, got {:?}",
                quant_output_dtype
            ),
        });
    }
    Ok(())
}

/// Validate dequantization format constraints.
///
/// Per the ANE constraint canon (§5.2):
/// - Dequantize input must be int8, uint8, int4, or e4m3
/// - Dequantize output must be fp16
/// - Int4 per-cout dequant is NOT supported
pub fn validate_dequantization_constraints(
    dequant_input_dtype: &MilDtype,
    dequant_output_dtype: &MilDtype,
    dequant_scale_type: Option<&DequantScaleType>,
) -> Result<(), DtypeConstraintError> {
    // Dequantize input must be int8, uint8, int4, or e4m3
    // Per ANE canon: "Dequant layer must have int8, uint8, int4 or e4m3 input format"
    if !matches!(
        dequant_input_dtype,
        MilDtype::Int8 | MilDtype::UInt8 | MilDtype::Int4 | MilDtype::E4M3
    ) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!(
                "Dequantize input must be int8, uint8, int4, or e4m3, got {:?}",
                dequant_input_dtype
            ),
        });
    }
    // Int4 per-cout dequant is NOT supported
    if matches!(dequant_input_dtype, MilDtype::Int4) {
        if let Some(DequantScaleType::PerOutputChannel) = dequant_scale_type {
            return Err(DtypeConstraintError::Int4ConstraintViolation {
                message: "Int4 Per-Cout Dequant is not supported".into(),
            });
        }
    }
    // Dequantize output must be fp16
    if !matches!(dequant_output_dtype, MilDtype::Fp16) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!("Dequantize output must be fp16, got {:?}", dequant_output_dtype),
        });
    }
    Ok(())
}

/// Scale type used in dequantization, for Int4 per-cout constraint checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantScaleType {
    /// Per-tensor (scalar) scale — always legal.
    PerTensor,
    /// Per-output-channel scale — NOT legal for Int4.
    PerOutputChannel,
}

/// Check if asymmetric quantization is supported (it's not on ANE).
pub fn is_asymmetric_quantization_supported() -> bool {
    false // "Asym quantization is not supported"
}

/// Check if blockwise scale is supported (it's not on ANE).
pub fn is_blockwise_scale_supported() -> bool {
    false // "ANE doesn't support blockwise scale"
}

/// Check if broadcast dtype is legal for the given family.
pub fn is_broadcast_dtype_legal(
    dtype: &MilDtype,
    family: &AneFamily,
) -> Result<(), DtypeConstraintError> {
    if family.broadcast_fp16_only() && *dtype != MilDtype::Fp16 {
        return Err(DtypeConstraintError::RejectedDtype {
            dtype: format!("{:?}", dtype),
            family: format!("{:?}", family),
            message: "Only fp16 is supported for A11/A12 Broadcasts".into(),
        });
    }
    Ok(())
}

/// Check if a dtype requires interleave factor 8 on ANE.
///
/// Int4 and UInt4 packed tensors must use interleave factor 8 per the ANE
/// constraint: "Tensor with the int4 format must have an interleave factor of 8".
pub fn dtype_requires_interleave_8(dtype: &MilDtype) -> bool {
    matches!(dtype, MilDtype::Int4 | MilDtype::UInt4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_always_legal() {
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18] {
            assert!(is_dtype_ane_legal(&MilDtype::Fp16, &family).is_ok());
        }
    }

    #[test]
    fn test_int32_rejected() {
        assert!(is_dtype_ane_legal(&MilDtype::Int32, &AneFamily::A18).is_err());
    }

    #[test]
    fn test_fp64_rejected() {
        assert!(is_dtype_ane_legal(&MilDtype::Fp64, &AneFamily::A18).is_err());
    }

    #[test]
    fn test_quantization_input_must_be_fp() {
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::Int8).is_ok());
        assert!(validate_quantization_constraints(&MilDtype::Fp32, &MilDtype::UInt8).is_ok());
        assert!(validate_quantization_constraints(&MilDtype::Int8, &MilDtype::Int8).is_err());
    }

    #[test]
    fn test_quantization_output_must_be_int() {
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::Fp16).is_err());
    }

    #[test]
    fn test_dequantization_constraints() {
        assert!(validate_dequantization_constraints(&MilDtype::Int8, &MilDtype::Fp16, None).is_ok());
        assert!(validate_dequantization_constraints(&MilDtype::UInt8, &MilDtype::Fp16, None).is_ok());
        assert!(validate_dequantization_constraints(&MilDtype::Fp16, &MilDtype::Fp16, None).is_err());
    }

    #[test]
    fn test_asymmetric_quant_not_supported() {
        assert!(!is_asymmetric_quantization_supported());
    }

    #[test]
    fn test_blockwise_scale_not_supported() {
        assert!(!is_blockwise_scale_supported());
    }

    #[test]
    fn test_broadcast_fp16_only_a11_a12() {
        // A11/A12: FP16-only broadcast
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp16, &AneFamily::A12).is_ok());
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp32, &AneFamily::A12).is_err());
        // A13+: full dtype broadcast
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp32, &AneFamily::A13).is_ok());
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp32, &AneFamily::A14).is_ok());
    }

    // ─── T-35: New dtype constraint tests ──────────────────────────

    #[test]
    fn test_int4_legal_on_all_families() {
        // Int4 is legal on ANE but requires interleave=8
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18] {
            assert!(is_dtype_ane_legal(&MilDtype::Int4, &family).is_ok(),
                "Int4 should be legal on {:?}", family);
        }
    }

    #[test]
    fn test_uint4_legal_on_all_families() {
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18] {
            assert!(is_dtype_ane_legal(&MilDtype::UInt4, &family).is_ok(),
                "UInt4 should be legal on {:?}", family);
        }
    }

    #[test]
    fn test_e4m3_rejected_on_pre_a17() {
        // E4M3 is NOT supported on A11 through A16
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A11Legacy).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A12).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A13).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A14).is_err());
    }

    #[test]
    fn test_e4m3_legal_on_a18() {
        // E4M3 has limited support on A18
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A18).is_ok());
    }

    #[test]
    fn test_e5m2_rejected_on_all_families() {
        // E5M2 is NOT supported on any ANE family
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18] {
            assert!(is_dtype_ane_legal(&MilDtype::E5M2, &family).is_err(),
                "E5M2 should be rejected on {:?}", family);
        }
    }

    #[test]
    fn test_uint16_legal_on_all_families() {
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18] {
            assert!(is_dtype_ane_legal(&MilDtype::UInt16, &family).is_ok(),
                "UInt16 should be legal on {:?}", family);
        }
    }

    #[test]
    fn test_int4_interleave_must_be_8() {
        assert!(validate_int4_interleave(8).is_ok());
        assert!(validate_int4_interleave(1).is_err());
        assert!(validate_int4_interleave(4).is_err());
    }

    #[test]
    fn test_uint4_interleave_must_be_8() {
        assert!(validate_uint4_interleave(8).is_ok());
        assert!(validate_uint4_interleave(1).is_err());
        assert!(validate_uint4_interleave(4).is_err());
    }

    #[test]
    fn test_int4_per_cout_dequant_not_supported() {
        assert!(!is_int4_per_cout_dequant_supported());
    }

    #[test]
    fn test_e4m3_zero_point_not_supported() {
        assert!(!is_e4m3_zero_point_supported());
    }

    #[test]
    fn test_quantize_e4m3_output() {
        // Quantize output can be E4M3 per ANE canon
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::E4M3).is_ok());
    }

    #[test]
    fn test_quantize_e5m2_output() {
        // Quantize output can be E5M2 per ANE canon
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::E5M2).is_ok());
    }

    #[test]
    fn test_quantize_int4_output_rejected() {
        // Int4 is NOT a valid quantize output dtype
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::Int4).is_err());
    }

    #[test]
    fn test_quantize_uint4_output_rejected() {
        // UInt4 is NOT a valid quantize output dtype
        assert!(validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::UInt4).is_err());
    }

    #[test]
    fn test_dequantize_int4_input() {
        // Dequantize input can be Int4 per ANE canon
        assert!(validate_dequantization_constraints(&MilDtype::Int4, &MilDtype::Fp16, None).is_ok());
    }

    #[test]
    fn test_dequantize_e4m3_input() {
        // Dequantize input can be E4M3 per ANE canon
        assert!(validate_dequantization_constraints(&MilDtype::E4M3, &MilDtype::Fp16, None).is_ok());
    }

    #[test]
    fn test_dequantize_int4_per_cout_rejected() {
        // Int4 per-cout dequant is NOT supported
        assert!(validate_dequantization_constraints(
            &MilDtype::Int4,
            &MilDtype::Fp16,
            Some(&DequantScaleType::PerOutputChannel),
        ).is_err());
    }

    #[test]
    fn test_dequantize_int4_per_tensor_ok() {
        // Int4 per-tensor dequant is supported
        assert!(validate_dequantization_constraints(
            &MilDtype::Int4,
            &MilDtype::Fp16,
            Some(&DequantScaleType::PerTensor),
        ).is_ok());
    }

    #[test]
    fn test_dequantize_e5m2_input_rejected() {
        // E5M2 is NOT a valid dequantize input dtype
        assert!(validate_dequantization_constraints(&MilDtype::E5M2, &MilDtype::Fp16, None).is_err());
    }

    #[test]
    fn test_dequantize_uint4_input_rejected() {
        // UInt4 is NOT listed as valid dequantize input in the ANE canon
        assert!(validate_dequantization_constraints(&MilDtype::UInt4, &MilDtype::Fp16, None).is_err());
    }

    #[test]
    fn test_dtype_requires_interleave_8() {
        assert!(dtype_requires_interleave_8(&MilDtype::Int4));
        assert!(dtype_requires_interleave_8(&MilDtype::UInt4));
        assert!(!dtype_requires_interleave_8(&MilDtype::Int8));
        assert!(!dtype_requires_interleave_8(&MilDtype::Fp16));
        assert!(!dtype_requires_interleave_8(&MilDtype::E4M3));
        assert!(!dtype_requires_interleave_8(&MilDtype::E5M2));
    }

    #[test]
    fn test_e5m2_error_message_content() {
        let err = is_dtype_ane_legal(&MilDtype::E5M2, &AneFamily::A14);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("E4M3 or E5M2 format not supported"), "Error message should match ANE error: {}", msg);
    }

    #[test]
    fn test_e4m3_version_gated_error() {
        let err = is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A14);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("VersionGatedDtype") || msg.contains("A17") || msg.contains("E4M3"),
            "Error should mention version gating: {}", msg);
    }

    #[test]
    fn test_int4_interleave_error_message() {
        let err = validate_int4_interleave(4);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("interleave factor 8"), "Error should mention interleave=8: {}", msg);
    }
}
