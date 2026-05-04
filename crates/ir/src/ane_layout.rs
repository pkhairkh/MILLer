//! ANE memory layout, interleave constraint types, and palette bit-width validation.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md Section 6
//!
//! ## Palette Bit-Width Validation
//!
//! The ANE only supports palette bit-widths in the set {1, 2, 3, 4, 6, 8}.
//! Bit-widths 5 and 7 are **invalid** and will cause ANE runtime errors.
//! [`validate_palette_bits()`] provides centralized validation used by
//! `ane-passes`, `ane-lab`, and `ane-ir` (T-64 / I-38).

use serde::{Deserialize, Serialize};

/// Valid ANE interleave factors.
/// Source: "invalid input interleave factor; should be 1, 2, 3, 4, or 8"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneInterleave {
    Factor1 = 1,
    Factor2 = 2,
    Factor3 = 3,
    Factor4 = 4,
    Factor8 = 8,
}

impl AneInterleave {
    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::Factor1),
            2 => Some(Self::Factor2),
            3 => Some(Self::Factor3),
            4 => Some(Self::Factor4),
            8 => Some(Self::Factor8),
            _ => None,
        }
    }

    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Validate interleave for constant tensors (must be 1).
    pub fn is_valid_for_const(&self) -> bool {
        *self == AneInterleave::Factor1
    }

    /// Validate interleave for int4 tensors (must be 8).
    pub fn is_valid_for_int4(&self) -> bool {
        *self == AneInterleave::Factor8
    }

    /// Check if input channel is divisible by interleave factor.
    pub fn is_channel_divisible(&self, channels: u64) -> bool {
        channels.is_multiple_of(self.value() as u64)
    }
}

/// ANE memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneLayout {
    /// Channel-first (default) — NCHW or equivalent
    ChannelFirst,
    /// Channel-last (NHWC) — only for depthwise/channel-wise convolutions
    ChannelLast,
}

/// Interleave/layout constraint violation.
#[derive(Debug, Clone)]
pub struct LayoutConstraintViolation {
    pub constraint: String,
    pub message: String,
}

/// Validate interleave constraints.
pub fn validate_interleave_constraints(
    interleave: AneInterleave,
    is_const: bool,
    is_int4: bool,
    channels: u64,
) -> Result<(), LayoutConstraintViolation> {
    // Const tensors must have interleave 1
    if is_const && !interleave.is_valid_for_const() {
        return Err(LayoutConstraintViolation {
            constraint: "const_interleave_1".into(),
            message: format!("Const tensor interleave must be 1, got {}", interleave.value()),
        });
    }
    // Int4 tensors require interleave 8
    if is_int4 && !interleave.is_valid_for_int4() {
        return Err(LayoutConstraintViolation {
            constraint: "int4_interleave_8".into(),
            message: format!(
                "Tensor with int4 format must have interleave 8, got {}",
                interleave.value()
            ),
        });
    }
    // Input channel must be divisible by interleave factor
    if !interleave.is_channel_divisible(channels) {
        return Err(LayoutConstraintViolation {
            constraint: "channel_divisible_by_interleave".into(),
            message: format!(
                "Input channel {} must be divisible by interleave factor {}",
                channels,
                interleave.value()
            ),
        });
    }
    Ok(())
}

/// Validate ChannelLast constraints.
pub fn validate_channellast_constraints(
    layout: AneLayout,
    is_depthwise_conv: bool,
    interleave: AneInterleave,
) -> Result<(), LayoutConstraintViolation> {
    if layout == AneLayout::ChannelLast {
        // ChannelLast only supported for depthwise convolutions
        if !is_depthwise_conv {
            return Err(LayoutConstraintViolation {
                constraint: "channellast_depthwise_only".into(),
                message: "ChannelLast currently only supported for channel wise convolutions"
                    .into(),
            });
        }
        // ChannelLast does not support non-one interleave
        if interleave != AneInterleave::Factor1 {
            return Err(LayoutConstraintViolation {
                constraint: "channellast_interleave_1".into(),
                message: "ChannelLast does not support non-one interleave".into(),
            });
        }
    }
    Ok(())
}

/// Valid palette bit-widths supported by the ANE hardware.
///
/// The ANE supports LUT-based palettization at these specific bit-widths.
/// Values 5 and 7 are NOT supported and will cause runtime errors.
///
/// T-64 (I-38): Centralized definition — all palette bit-width validation
/// sites should reference this constant instead of duplicating the set.
/// Previously, {1, 2, 3, 4, 6, 8} was duplicated in `palettize_weights.rs`,
/// `lut_projection.rs`, and `task_spec.rs`.
pub const VALID_PALETTE_BITS: &[usize] = &[1, 2, 3, 4, 6, 8];

/// Validate that a palette bit-width is in the ANE-supported set.
///
/// Returns `Ok(())` if valid, or a descriptive error message if not.
///
/// # Examples
///
/// ```
/// use ane_ir::ane_layout::validate_palette_bits;
/// assert!(validate_palette_bits(4).is_ok());
/// assert!(validate_palette_bits(5).is_err());
/// ```
///
/// T-64 (I-38): Centralized validation — previously duplicated in 3 places.
pub fn validate_palette_bits(bits: usize) -> Result<(), String> {
    if VALID_PALETTE_BITS.contains(&bits) {
        Ok(())
    } else {
        Err(format!(
            "Invalid palette bit-width {}: must be one of {:?}. \
             ANE hardware does not support {}-bit palettization.",
            bits, VALID_PALETTE_BITS, bits
        ))
    }
}

/// Clamp a bit-width to the nearest valid ANE palette bit-width.
///
/// For bit-widths between valid values, rounds down to the nearest
/// supported bit-width (e.g., 5 → 4, 7 → 6). This preserves
/// quantization benefit while ensuring ANE compatibility.
///
/// # Examples
///
/// ```
/// use ane_ir::ane_layout::clamp_to_valid_palette_bits;
/// assert_eq!(clamp_to_valid_palette_bits(5), 4);
/// assert_eq!(clamp_to_valid_palette_bits(7), 6);
/// assert_eq!(clamp_to_valid_palette_bits(4), 4);
/// ```
pub fn clamp_to_valid_palette_bits(bits: usize) -> usize {
    if VALID_PALETTE_BITS.contains(&bits) {
        return bits;
    }
    // Round down to nearest valid bit-width
    *VALID_PALETTE_BITS.iter().filter(|&&b| b <= bits).last().unwrap_or(&1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_palette_bits_valid() {
        for &bits in VALID_PALETTE_BITS {
            assert!(validate_palette_bits(bits).is_ok(), "{} should be valid", bits);
        }
    }

    #[test]
    fn test_validate_palette_bits_invalid() {
        assert!(validate_palette_bits(5).is_err(), "5-bit is not ANE-supported");
        assert!(validate_palette_bits(7).is_err(), "7-bit is not ANE-supported");
        assert!(validate_palette_bits(9).is_err(), "9-bit is not ANE-supported");
        assert!(validate_palette_bits(0).is_err(), "0-bit is not ANE-supported");
    }

    #[test]
    fn test_clamp_to_valid_palette_bits() {
        assert_eq!(clamp_to_valid_palette_bits(1), 1);
        assert_eq!(clamp_to_valid_palette_bits(4), 4);
        assert_eq!(clamp_to_valid_palette_bits(5), 4); // 5 → 4 (round down)
        assert_eq!(clamp_to_valid_palette_bits(6), 6);
        assert_eq!(clamp_to_valid_palette_bits(7), 6); // 7 → 6 (round down)
        assert_eq!(clamp_to_valid_palette_bits(8), 8);
        assert_eq!(clamp_to_valid_palette_bits(10), 8); // 10 → 8 (round down)
        assert_eq!(clamp_to_valid_palette_bits(0), 1); // 0 → 1 (minimum)
    }

    #[test]
    fn test_interleave_from_u8() {
        assert_eq!(AneInterleave::from_u8(1), Some(AneInterleave::Factor1));
        assert_eq!(AneInterleave::from_u8(8), Some(AneInterleave::Factor8));
        assert_eq!(AneInterleave::from_u8(5), None);
        assert_eq!(AneInterleave::from_u8(6), None);
    }

    #[test]
    fn test_const_interleave_must_be_1() {
        assert!(validate_interleave_constraints(AneInterleave::Factor1, true, false, 64).is_ok());
        assert!(validate_interleave_constraints(AneInterleave::Factor2, true, false, 64).is_err());
    }

    #[test]
    fn test_int4_interleave_must_be_8() {
        assert!(validate_interleave_constraints(AneInterleave::Factor8, false, true, 64).is_ok());
        assert!(validate_interleave_constraints(AneInterleave::Factor4, false, true, 64).is_err());
    }

    #[test]
    fn test_channel_divisible_by_interleave() {
        assert!(validate_interleave_constraints(AneInterleave::Factor4, false, false, 64).is_ok());
        assert!(validate_interleave_constraints(AneInterleave::Factor4, false, false, 63).is_err());
    }

    #[test]
    fn test_channellast_depthwise_only() {
        assert!(validate_channellast_constraints(
            AneLayout::ChannelLast,
            true,
            AneInterleave::Factor1
        )
        .is_ok());
        assert!(validate_channellast_constraints(
            AneLayout::ChannelLast,
            false,
            AneInterleave::Factor1
        )
        .is_err());
    }

    #[test]
    fn test_channellast_interleave_1() {
        assert!(validate_channellast_constraints(
            AneLayout::ChannelLast,
            true,
            AneInterleave::Factor1
        )
        .is_ok());
        assert!(validate_channellast_constraints(
            AneLayout::ChannelLast,
            true,
            AneInterleave::Factor2
        )
        .is_err());
    }

    #[test]
    fn test_channel_first_always_ok() {
        assert!(validate_channellast_constraints(
            AneLayout::ChannelFirst,
            false,
            AneInterleave::Factor4
        )
        .is_ok());
    }
}
