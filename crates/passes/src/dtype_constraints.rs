//! Data type and format constraint validation for ANE placement.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md Section 5

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
        }
    }
}

/// Check if a dtype is legal for ANE compute on the given family.
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

/// Validate quantization format constraints.
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
    // Quantize output must be int8, uint8
    if !matches!(quant_output_dtype, MilDtype::Int8 | MilDtype::UInt8) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!("Quantize output must be int8 or uint8, got {:?}", quant_output_dtype),
        });
    }
    Ok(())
}

/// Validate dequantization format constraints.
pub fn validate_dequantization_constraints(
    dequant_input_dtype: &MilDtype,
    dequant_output_dtype: &MilDtype,
) -> Result<(), DtypeConstraintError> {
    // Dequantize input must be int8, uint8
    if !matches!(dequant_input_dtype, MilDtype::Int8 | MilDtype::UInt8) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!(
                "Dequantize input must be int8 or uint8, got {:?}",
                dequant_input_dtype
            ),
        });
    }
    // Dequantize output must be fp16
    if !matches!(dequant_output_dtype, MilDtype::Fp16) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!("Dequantize output must be fp16, got {:?}", dequant_output_dtype),
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_always_legal() {
        for family in [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A14, AneFamily::A18] {
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
        assert!(validate_dequantization_constraints(&MilDtype::Int8, &MilDtype::Fp16).is_ok());
        assert!(validate_dequantization_constraints(&MilDtype::UInt8, &MilDtype::Fp16).is_ok());
        assert!(validate_dequantization_constraints(&MilDtype::Fp16, &MilDtype::Fp16).is_err());
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
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp16, &AneFamily::A12).is_ok());
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp32, &AneFamily::A12).is_err());
        assert!(is_broadcast_dtype_legal(&MilDtype::Fp32, &AneFamily::A14).is_ok());
    }
}
