//! ANE memory layout and interleave constraint types.
//! Source: ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md Section 6

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

#[cfg(test)]
mod tests {
    use super::*;

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
