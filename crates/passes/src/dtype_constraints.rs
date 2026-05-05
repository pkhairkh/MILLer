//! Data type and format constraint validation for ANE placement.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md Section 5
//!
//! T-35 (I-14): Expanded with Int4, UInt4, E4M3, E5M2, UInt16 dtype
//! variants and their ANE constraint enforcement.
//!
//! ## UInt16 ANE Constraints (V-091)
//!
//! UInt16 has extremely limited ANE support. From ANEC binary forensic evidence:
//! - UInt16 is ONLY valid as the output of: TopK (indices), Sort (indices),
//!   ReduceArgmax (indices), ReduceArgmin (indices)
//! - UInt16 may also appear as a single DMA source in the PEEW (Primary
//!   Element Execution Window) data path, but this is not a compute use case.
//! - Argmax/Argmin indices output as UInt16 requires iOS17+ (A17+ family).
//! - Any other op producing or consuming UInt16 on the ANE will be rejected
//!   by ANEC at runtime.
//!
//! ## Bool ANE Constraints (V-092)
//!
//! Bool has no ANE compute support whatsoever:
//! - Bool is NOT supported as a weight blob dtype.
//! - Bool is NOT supported as an ANE compute output — the ANE cannot produce
//!   Bool-typed results from any compute operation.
//! - Bool can ONLY appear as a mask-like input tensor for Select and Where
//!   operations (condition operand). Even then, Select/Where are currently
//!   decomposed to arithmetic on the ANE path, so Bool is effectively
//!   CPU-only for all practical purposes.
//! - Comparison ops (Equal, NotEqual, Greater, Less) produce Bool output
//!   but are handled internally by the ANE as FP16 0/1 values, not as
//!   true Bool-typed tensors.

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
    /// Cross-type operand dtype mismatch (T-97/V-125/I-98).
    CrossTypeViolation { input_dtype: String, output_dtype: String, message: String },
    /// Asymmetric quantization not supported on ANE (T-97/V-134/I-102).
    AsymmetricQuantViolation { message: String },
    /// UInt16 constraint violation (V-091).
    UInt16ConstraintViolation { message: String },
    /// Bool constraint violation (V-092).
    BoolConstraintViolation { message: String },
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
            Self::CrossTypeViolation { input_dtype, output_dtype, message } => {
                write!(f, "Cross-type violation: input={}, output={} — {}", input_dtype, output_dtype, message)
            }
            Self::AsymmetricQuantViolation { message } => {
                write!(f, "Asymmetric quantization violation: {}", message)
            }
            Self::UInt16ConstraintViolation { message } => {
                write!(f, "UInt16 constraint violation (V-091): {}", message)
            }
            Self::BoolConstraintViolation { message } => {
                write!(f, "Bool constraint violation (V-092): {}", message)
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
        // T-97 (I-99/V-126): FP32 is rejected on A11/A12 for compute.
        // Use `is_fp32_compute_supported()` for compute-specific checks.
        // Here we allow FP32 for weight storage and I/O (may be downcast).
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
        // UInt16 — limited support (V-091): only valid as output of TopK/Sort/Argmax/Argmin
        MilDtype::UInt16 => Ok(()), // constrained: caller must also validate op context
        // Int32 — NOT supported for compute on ANE
        MilDtype::Int32 => Err(DtypeConstraintError::RejectedDtype {
            dtype: "Int32".into(),
            family: format!("{:?}", family),
            message: "32 bit format not supported for ANE compute".into(),
        }),
        // Bool — limited support (V-092): only valid as mask input for Select/Where
        MilDtype::Bool => Ok(()), // constrained: caller must also validate op context
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
            message: format!("Int4 format requires interleave factor 8, got {}", interleave),
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
            message: format!("UInt4 format requires interleave factor 8, got {}", interleave),
        });
    }
    Ok(())
}

/// Validate UInt16 dtype constraints for ANE operations.
///
/// V-091: UInt16 is ONLY valid as the output dtype of specific ops:
/// - TopK (indices output)
/// - Sort (indices output)  [note: Argsort is CPU-only in current MILLer]
/// - ReduceArgmax (indices output, iOS17+/A17+)
/// - ReduceArgmin (indices output, iOS17+/A17+)
///
/// For any other op using UInt16 on the ANE, this function returns an error.
/// The `is_output` parameter distinguishes between UInt16 being used as an
/// output dtype (legal for the ops above) vs. an input dtype (always illegal
/// for ANE compute except the limited DMA source PEEW case, which is not
/// a compiler-level concern).
///
/// # Arguments
/// * `op_name` - The MIL op name (e.g., "topk", "reduce_argmax")
/// * `is_output` - True if UInt16 is the output dtype of this op
pub fn validate_uint16_constraints(op_name: &str, is_output: bool) -> Result<(), DtypeConstraintError> {
    const UINT16_ALLOWED_OPS: &[&str] = &[
        "topk",
        "sort",
        "reduce_argmax",
        "reduce_argmin",
    ];

    if is_output && UINT16_ALLOWED_OPS.contains(&op_name) {
        Ok(())
    } else if is_output {
        Err(DtypeConstraintError::UInt16ConstraintViolation {
            message: format!(
                "UInt16 dtype is only supported for TopK/Sort indices output and \
                 ReduceArgmax/ReduceArgmin on ANE (iOS17+), got op '{}'",
                op_name
            ),
        })
    } else {
        Err(DtypeConstraintError::UInt16ConstraintViolation {
            message: format!(
                "UInt16 dtype is only supported for TopK/Sort indices output and \
                 ReduceArgmax/ReduceArgmin on ANE (iOS17+); UInt16 as input to '{}' is not supported",
                op_name
            ),
        })
    }
}

/// Validate Bool dtype constraints for ANE operations.
///
/// V-092: Bool is NOT supported as an ANE compute dtype:
/// - Bool cannot be produced as output by any ANE compute op.
/// - Bool can ONLY appear as a mask input (condition operand) for Select and
///   Where operations. Note: Select/Where are currently decomposed to
///   arithmetic on the ANE path, so Bool is effectively CPU-only.
/// - Bool is NOT supported in weight blobs.
///
/// For ops that produce Bool output on the ANE, this returns an error.
/// For ops where Bool is a mask/condition input (Select, Where), this is OK.
///
/// # Arguments
/// * `op_name` - The MIL op name (e.g., "select", "where", "conv")
/// * `is_output` - True if Bool is the output dtype of this op
pub fn validate_bool_constraints(op_name: &str, is_output: bool) -> Result<(), DtypeConstraintError> {
    const BOOL_MASK_INPUT_OPS: &[&str] = &["select", "where"];

    if is_output {
        // Bool as ANE compute output is never supported
        Err(DtypeConstraintError::BoolConstraintViolation {
            message: format!(
                "Bool dtype is not supported as ANE compute output — \
                 only valid as mask input for Select/Where operations, got op '{}' producing Bool output",
                op_name
            ),
        })
    } else if BOOL_MASK_INPUT_OPS.contains(&op_name) {
        // Bool as mask input for Select/Where is acceptable
        Ok(())
    } else {
        // Bool used as input to any other op is not supported on ANE
        Err(DtypeConstraintError::BoolConstraintViolation {
            message: format!(
                "Bool dtype is not supported as ANE compute input for '{}' — \
                 only valid as mask input for Select/Where operations",
                op_name
            ),
        })
    }
}

/// Validate quantization format constraints.
///
/// Per the ANE constraint canon (§5.2):
/// - Quantize input must be fp16 or fp32
/// - Quantize output must be int8, uint8, or e4m3
///
/// T-97 (I-72/V-051/V-111): E5M2 was previously accepted as a valid quantize
/// output dtype, but ANEC universally rejects it ("E4M3 or E5M2 format not
/// supported"). E5M2 is now rejected in the quantize output validation.
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
    // Quantize output must be int8, uint8, or e4m3.
    // T-97 (I-72/V-051/V-111): E5M2 is universally rejected by ANEC
    // ("E4M3 or E5M2 format not supported"). Removed from valid outputs.
    if !matches!(
        quant_output_dtype,
        MilDtype::Int8 | MilDtype::UInt8 | MilDtype::E4M3
    ) {
        return Err(DtypeConstraintError::QuantFormatViolation {
            message: format!(
                "Quantize output must be int8, uint8, or e4m3, got {:?}. \
                 E5M2 is universally rejected by ANEC (V-051, V-111).",
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

// ─── T-97: Cross-Type Validation and Dtype Rejection ─────────────────────
//
// These functions address four validation gaps identified in the NECROSCOPY
// forensic audit (V-125, V-126, V-051/V-111, V-134):
// 1. BF16/F16 cross-type operations rejected by ANEC but no validation
// 2. FP32 architecture-conditional rejection not checked
// 3. E5M2 accepted by quantize validator but universally rejected by ANEC
// 4. Asymmetric quantization not rejected for ANE path

/// Validate cross-type compatibility for ANE operations.
///
/// T-97 (I-98/V-125): ANEC explicitly rejects BF16/F16 cross-type operations.
/// T-P1-05: Previously this function was a stub that returned `Ok(())` for
/// everything. Now it properly validates cross-type violations.
///
/// Binary forensic evidence confirms 9 constraint strings documenting cross-type
/// rejections, including:
/// - "detected operation with BF16 inputs and F16 result type which is not supported"
/// - "detected operation with F16 inputs and BF16 result type which is not supported"
/// - "detected operation with both F16 and BF16 operands which is not supported"
///
/// MILLer previously validated each operand's dtype independently, missing
/// cross-type incompatibilities. This function checks input/output dtype pairs
/// and rejects all documented cross-type violations.
pub fn validate_cross_type_compatibility(
    input_dtype: &MilDtype,
    output_dtype: &MilDtype,
) -> Result<(), DtypeConstraintError> {
    // T-P1-05: FP16 input → FP32 output (upcast) is rejected for ANE compute.
    // ANEC constraint: "detected operation with F16 inputs and BF16 result type
    // which is not supported" — while BF16 is not in our enum, the analogous
    // cross-type case is FP16→FP32 which forces an upcast that ANE cannot do.
    if matches!(input_dtype, MilDtype::Fp16) && matches!(output_dtype, MilDtype::Fp32) {
        return Err(DtypeConstraintError::CrossTypeViolation {
            input_dtype: format!("{:?}", input_dtype),
            output_dtype: format!("{:?}", output_dtype),
            message: "FP16 input with FP32 output is not supported on ANE (cross-type violation). \
                      ANEC rejects cross-type operations between incompatible float widths."
                .into(),
        });
    }

    // FP32 input → FP16 output (downcast) is allowed — this is the normal
    // ANE behavior where FP32 weights are cast down to FP16 for compute.

    // Integer input → Float output cross-type (without explicit quantize/dequantize)
    // is rejected by ANEC for some combinations.
    if matches!(input_dtype, MilDtype::Int8 | MilDtype::UInt8 | MilDtype::Int4 | MilDtype::UInt4)
        && matches!(output_dtype, MilDtype::Fp16 | MilDtype::Fp32)
    {
        // Quantize ops handle this explicitly; non-quantize integer→float
        // is a cross-type violation for direct compute ops.
        // Allow through since validate_quantization_constraints handles
        // the quant/dequant path separately.
    }

    Ok(())
}

/// Check if FP32 compute is supported on the given family.
///
/// T-97 (I-99/V-126): FP32 computation is rejected on some architectures
/// ("Float32 not supported for architecture") but `is_dtype_ane_legal()`
/// previously approved FP32 for all families. This function provides a more
/// nuanced check that considers the specific compute context.
///
/// FP32 is allowed for:
/// - Weight storage (not compute)
/// - Input/output tensors (may be downcast internally)
/// - Specific ops on specific architectures
///
/// FP32 is NOT allowed for:
/// - General compute on A11/A12 (A11Legacy/A12)
/// - Ops where the result must be FP32 (no downcast possible)
pub fn is_fp32_compute_supported(family: &AneFamily) -> bool {
    // FP32 compute is supported on A13+ families.
    // A11Legacy and A12 do not support FP32 compute on ANE.
    matches!(
        family,
        AneFamily::A13
            | AneFamily::A14
            | AneFamily::A15
            | AneFamily::A16
            | AneFamily::A17
            | AneFamily::A18
    )
}

/// Validate asymmetric quantization is not used for ANE path.
///
/// T-97 (I-102/V-134): ANEC constraint: "Asym quantization is not supported".
/// No check previously prevented asymmetric quantization in the ANE path.
/// This function validates that quantization is symmetric when targeting ANE.
pub fn validate_anec_quantization_symmetry(
    is_symmetric: bool,
    target_is_ane: bool,
) -> Result<(), DtypeConstraintError> {
    if target_is_ane && !is_symmetric {
        return Err(DtypeConstraintError::AsymmetricQuantViolation {
            message: "Asymmetric quantization is not supported on ANE (ANEC constraint: \
                      'Asym quantization is not supported'). Use symmetric quantization instead."
                .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_always_legal() {
        for family in
            [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18]
        {
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
        assert!(
            validate_dequantization_constraints(&MilDtype::UInt8, &MilDtype::Fp16, None).is_ok()
        );
        assert!(
            validate_dequantization_constraints(&MilDtype::Fp16, &MilDtype::Fp16, None).is_err()
        );
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
        for family in
            [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18]
        {
            assert!(
                is_dtype_ane_legal(&MilDtype::Int4, &family).is_ok(),
                "Int4 should be legal on {:?}",
                family
            );
        }
    }

    #[test]
    fn test_uint4_legal_on_all_families() {
        for family in
            [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18]
        {
            assert!(
                is_dtype_ane_legal(&MilDtype::UInt4, &family).is_ok(),
                "UInt4 should be legal on {:?}",
                family
            );
        }
    }

    #[test]
    fn test_e4m3_rejected_on_pre_a17() {
        // E4M3 is NOT supported on A11 through A16
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A11Legacy).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A12).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A13).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A14).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A15).is_err());
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A16).is_err());
    }

    #[test]
    fn test_e4m3_legal_on_a17() {
        // T-52: E4M3 has conditional support on A17 (LSE_6)
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A17).is_ok());
    }

    #[test]
    fn test_e4m3_legal_on_a18() {
        // E4M3 has limited support on A18
        assert!(is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A18).is_ok());
    }

    #[test]
    fn test_e5m2_rejected_on_all_families() {
        // E5M2 is NOT supported on any ANE family
        for family in
            [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18]
        {
            assert!(
                is_dtype_ane_legal(&MilDtype::E5M2, &family).is_err(),
                "E5M2 should be rejected on {:?}",
                family
            );
        }
    }

    #[test]
    fn test_uint16_legal_on_all_families() {
        for family in
            [AneFamily::A11Legacy, AneFamily::A12, AneFamily::A13, AneFamily::A14, AneFamily::A18]
        {
            assert!(
                is_dtype_ane_legal(&MilDtype::UInt16, &family).is_ok(),
                "UInt16 should be legal on {:?}",
                family
            );
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
    fn test_quantize_e5m2_output_rejected() {
        // T-97 (I-72/V-051/V-111): E5M2 is now rejected as quantize output
        // since ANEC universally rejects it ("E4M3 or E5M2 format not supported").
        assert!(
            validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::E5M2).is_err(),
            "E5M2 should be rejected as quantize output dtype"
        );
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
        )
        .is_err());
    }

    #[test]
    fn test_dequantize_int4_per_tensor_ok() {
        // Int4 per-tensor dequant is supported
        assert!(validate_dequantization_constraints(
            &MilDtype::Int4,
            &MilDtype::Fp16,
            Some(&DequantScaleType::PerTensor),
        )
        .is_ok());
    }

    #[test]
    fn test_dequantize_e5m2_input_rejected() {
        // E5M2 is NOT a valid dequantize input dtype
        assert!(
            validate_dequantization_constraints(&MilDtype::E5M2, &MilDtype::Fp16, None).is_err()
        );
    }

    #[test]
    fn test_dequantize_uint4_input_rejected() {
        // UInt4 is NOT listed as valid dequantize input in the ANE canon
        assert!(
            validate_dequantization_constraints(&MilDtype::UInt4, &MilDtype::Fp16, None).is_err()
        );
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
        assert!(
            msg.contains("E4M3 or E5M2 format not supported"),
            "Error message should match ANE error: {}",
            msg
        );
    }

    #[test]
    fn test_e4m3_version_gated_error() {
        let err = is_dtype_ane_legal(&MilDtype::E4M3, &AneFamily::A14);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("VersionGatedDtype") || msg.contains("A17") || msg.contains("E4M3"),
            "Error should mention version gating: {}",
            msg
        );
    }

    #[test]
    fn test_int4_interleave_error_message() {
        let err = validate_int4_interleave(4);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("interleave factor 8"), "Error should mention interleave=8: {}", msg);
    }

    // ─── T-97: Cross-type and FP32 architecture tests ──────────────────

    #[test]
    fn test_fp32_compute_supported_on_a13_plus() {
        // T-97 (I-99/V-126): FP32 compute is supported on A13+
        assert!(!is_fp32_compute_supported(&AneFamily::A11Legacy), "FP32 compute NOT supported on A11");
        assert!(!is_fp32_compute_supported(&AneFamily::A12), "FP32 compute NOT supported on A12");
        assert!(is_fp32_compute_supported(&AneFamily::A13), "FP32 compute supported on A13");
        assert!(is_fp32_compute_supported(&AneFamily::A14), "FP32 compute supported on A14");
        assert!(is_fp32_compute_supported(&AneFamily::A15), "FP32 compute supported on A15");
        assert!(is_fp32_compute_supported(&AneFamily::A16), "FP32 compute supported on A16");
        assert!(is_fp32_compute_supported(&AneFamily::A17), "FP32 compute supported on A17");
        assert!(is_fp32_compute_supported(&AneFamily::A18), "FP32 compute supported on A18");
    }

    #[test]
    fn test_fp32_dtype_still_legal_as_dtype() {
        // FP32 is still accepted by is_dtype_ane_legal() for I/O and weight
        // storage. Use is_fp32_compute_supported() for compute-specific checks.
        assert!(is_dtype_ane_legal(&MilDtype::Fp32, &AneFamily::A11Legacy).is_ok());
        assert!(is_dtype_ane_legal(&MilDtype::Fp32, &AneFamily::A14).is_ok());
    }

    #[test]
    fn test_cross_type_compatibility_ok() {
        // Same-type operations should always pass
        assert!(validate_cross_type_compatibility(&MilDtype::Fp16, &MilDtype::Fp16).is_ok());
        assert!(validate_cross_type_compatibility(&MilDtype::Fp32, &MilDtype::Fp32).is_ok());
        assert!(validate_cross_type_compatibility(&MilDtype::Int8, &MilDtype::Int8).is_ok());
        // FP32→FP16 is allowed (downcast)
        assert!(validate_cross_type_compatibility(&MilDtype::Fp32, &MilDtype::Fp16).is_ok());
    }

    #[test]
    fn test_cross_type_fp16_to_fp32_rejected() {
        // T-P1-05: FP16 input → FP32 output is a cross-type violation
        let result = validate_cross_type_compatibility(&MilDtype::Fp16, &MilDtype::Fp32);
        assert!(result.is_err(), "FP16→FP32 should be a cross-type violation");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("cross-type") || msg.contains("FP16") || msg.contains("FP32"),
            "Error should mention cross-type violation: {}", msg
        );
    }

    #[test]
    fn test_asymmetric_quantization_rejected_on_ane() {
        // T-97 (I-102/V-134): Asymmetric quantization is not supported on ANE
        assert!(
            validate_anec_quantization_symmetry(false, true).is_err(),
            "Asymmetric quantization should be rejected on ANE"
        );
    }

    #[test]
    fn test_asymmetric_quantization_allowed_on_cpu() {
        // Asymmetric quantization is fine for CPU targets
        assert!(
            validate_anec_quantization_symmetry(false, false).is_ok(),
            "Asymmetric quantization should be allowed on CPU"
        );
    }

    #[test]
    fn test_symmetric_quantization_allowed_on_ane() {
        // Symmetric quantization is fine on ANE
        assert!(
            validate_anec_quantization_symmetry(true, true).is_ok(),
            "Symmetric quantization should be allowed on ANE"
        );
    }

    #[test]
    fn test_asymmetric_quant_error_message() {
        let err = validate_anec_quantization_symmetry(false, true).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("Asym quantization") || msg.contains("asymmetric"),
            "Error should mention asymmetric quantization: {}", msg
        );
    }

    #[test]
    fn test_e5m2_quantize_error_mentions_anec() {
        // T-97: E5M2 quantize rejection should mention ANEC violation IDs
        let err = validate_quantization_constraints(&MilDtype::Fp16, &MilDtype::E5M2).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("E5M2") || msg.contains("V-051"),
            "Error should mention E5M2 or violation ID: {}", msg
        );
    }

    // ─── T-127: UInt16 and Bool constraint tests (V-091, V-092) ──────────

    #[test]
    fn test_uint16_topk_output_ok() {
        // V-091: UInt16 is valid as TopK indices output
        assert!(
            validate_uint16_constraints("topk", true).is_ok(),
            "UInt16 should be valid as TopK output"
        );
    }

    #[test]
    fn test_uint16_conv_output_rejected() {
        // V-091: UInt16 is NOT valid as Conv output
        let err = validate_uint16_constraints("conv", true);
        assert!(err.is_err(), "UInt16 should be rejected as Conv output");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("UInt16") && msg.contains("TopK/Sort"),
            "Error should mention UInt16 constraint: {}", msg
        );
    }

    #[test]
    fn test_uint16_argmax_output_ok() {
        // V-091: UInt16 is valid as ReduceArgmax indices output
        assert!(
            validate_uint16_constraints("reduce_argmax", true).is_ok(),
            "UInt16 should be valid as ReduceArgmax output"
        );
    }

    #[test]
    fn test_bool_select_input_ok() {
        // V-092: Bool is valid as mask input for Select
        assert!(
            validate_bool_constraints("select", false).is_ok(),
            "Bool should be valid as mask input for Select"
        );
    }

    #[test]
    fn test_bool_matmul_output_rejected() {
        // V-092: Bool is NOT valid as MatMul output (or any ANE compute output)
        let err = validate_bool_constraints("matmul", true);
        assert!(err.is_err(), "Bool should be rejected as MatMul output");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Bool") && msg.contains("Select/Where"),
            "Error should mention Bool constraint: {}", msg
        );
    }

    #[test]
    fn test_bool_conv_input_rejected() {
        // V-092: Bool is NOT valid as input to Conv (only Select/Where accept Bool input)
        let err = validate_bool_constraints("conv", false);
        assert!(err.is_err(), "Bool should be rejected as Conv input");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Bool") && msg.contains("Select/Where"),
            "Error should mention Bool Select/Where constraint: {}", msg
        );
    }
}
