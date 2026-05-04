//! ANE hardware limit parameters (hal_params) per revision.
//! Source: ane-constraints-docs/02-hardware-and-limits/hardware-versions-limits-and-op-support.md

use crate::ane_target::AneRevision;
use serde::{Deserialize, Serialize};

/// Per-revision ANE hardware limits.
/// Key hardware limit parameters that govern ANE op placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneHwLimits {
    pub revision: AneRevision,
    pub max_tensor_width: u64,
    pub max_tensor_height: u64,
    pub max_tensor_depth: u64,
    pub max_tensor_channels: u64,
    pub max_tensor_rank: u32,
    pub max_8b_conv_kernel_dim_x: u64,
    pub max_f16_conv_kernel_dim_x: u64,
    pub max_conv_kernel_dim_y: u64,
    pub max_pooling_kernel_dim: u64,
    pub pe_max_tile_height: u64,
    pub pe_reduction_cout_limit: u64,
    pub num_nes: u32,
    pub ne_transpose_c_max: u64,
}

impl AneHwLimits {
    /// Get hardware limits for a specific ANE revision.
    pub fn for_revision(rev: AneRevision) -> Self {
        match rev {
            AneRevision::V4 => Self::a11_legacy(),
            AneRevision::V5 => Self::a12(),
            AneRevision::V6 => Self::a13(),
            AneRevision::V7 => Self::a14(),
            AneRevision::V8 => Self::a15(),
            AneRevision::V10 => Self::a16(),
            AneRevision::V11 => Self::a17(),
            // T-40: V17 is M1 (Mac), not A18. M1 has A14-class ANE but
            // Mac-specific hardware limits (more NEs, larger tensors).
            AneRevision::V17 => Self::m1(),
            AneRevision::V19 => Self::a18_pro(),
            AneRevision::V20 => Self::a18_max(),
            AneRevision::V26 => Self::future(),
        }
    }

    fn a11_legacy() -> Self {
        Self {
            revision: AneRevision::V4,
            max_tensor_width: 16384,
            max_tensor_height: 4096,
            max_tensor_depth: 2048,
            max_tensor_channels: 65536,
            max_tensor_rank: 5,
            max_8b_conv_kernel_dim_x: 7,
            max_f16_conv_kernel_dim_x: 7,
            max_conv_kernel_dim_y: 7,
            max_pooling_kernel_dim: 27,
            pe_max_tile_height: 2048,
            pe_reduction_cout_limit: 16384,
            num_nes: 1,
            ne_transpose_c_max: 16384,
        }
    }

    /// A12 Bionic (ANE V5) hardware limits.
    ///
    /// **WARNING**: These limits are estimated/approximate. They are copied from
    /// the A11 legacy values and have NOT been independently verified on actual
    /// A12 hardware. The A12 ANE may differ from A11 in NE count, bandwidth,
    /// tensor dimension limits, and other parameters. Use these values with caution
    /// and verify against real hardware when possible.
    ///
    /// A runtime warning is emitted when A12 limits are selected to remind
    /// users that these are approximate.
    fn a12() -> Self {
        // T-84 (I-59): Replaced eprintln! with log::warn! — library code
        // should use structured logging, not stderr writes.
        log::warn!(
            "A12 Bionic (ANE V5) hardware limits are approximate — copied from A11 values \
             and not yet verified on real A12 hardware. Results may be inaccurate."
        );
        Self { revision: AneRevision::V5, ..Self::a11_legacy() }
    }

    /// A13 Bionic (ANE V6) hardware limits.
    /// A13 has doubled tensor width/height compared to A11/A12 but
    /// retains the A14Minus elementwise/reduction converter family.
    fn a13() -> Self {
        Self {
            revision: AneRevision::V6,
            max_tensor_width: 32768,
            max_tensor_height: 8192,
            ..Self::a11_legacy()
        }
    }

    fn a14() -> Self {
        Self { revision: AneRevision::V7, max_tensor_width: 65536, num_nes: 2, ..Self::a13() }
    }

    fn a15() -> Self {
        Self { revision: AneRevision::V8, num_nes: 2, ..Self::a14() }
    }

    fn a16() -> Self {
        Self {
            revision: AneRevision::V10,
            max_tensor_width: 131072,
            max_tensor_height: 16384,
            num_nes: 4,
            ..Self::a15()
        }
    }

    fn a17() -> Self {
        Self { revision: AneRevision::V11, num_nes: 4, ..Self::a16() }
    }

    /// M1 (Mac, ANE V17) hardware limits.
    ///
    /// M1 uses A14-class ANE (same constraint profile, converters, op support)
    /// but has Mac-specific hardware limits: 6 NEs (vs A14 Bionic's 2),
    /// 262144 max tensor width (vs A14's 65536). The family is A14 — M1
    /// does NOT get A18's SDPA or LayerNorm support.
    fn m1() -> Self {
        Self { revision: AneRevision::V17, max_tensor_width: 262144, num_nes: 6, ..Self::a17() }
    }

    /// A18 Bionic (iPhone 16, ANE V19) hardware limits.
    ///
    /// A18 uses A18-family ANE with SDPA, LayerNorm, and E4M3 support.
    /// Has 6 NEs and 262144 max tensor width.
    fn a18() -> Self {
        Self { revision: AneRevision::V19, max_tensor_width: 262144, num_nes: 6, ..Self::a17() }
    }

    fn a18_pro() -> Self {
        Self { revision: AneRevision::V19, num_nes: 8, ..Self::a18() }
    }

    fn a18_max() -> Self {
        Self { revision: AneRevision::V20, num_nes: 16, ..Self::a18_pro() }
    }

    fn future() -> Self {
        Self { revision: AneRevision::V26, num_nes: 16, ..Self::a18_max() }
    }

    /// Validate that tensor dimensions are within hardware limits.
    pub fn validate_tensor_dims(
        &self,
        width: u64,
        height: u64,
        depth: u64,
        channels: u64,
        rank: u32,
    ) -> Result<(), HwLimitViolation> {
        if width > self.max_tensor_width {
            return Err(HwLimitViolation {
                param: "max_tensor_width".into(),
                value: width,
                limit: self.max_tensor_width,
            });
        }
        if height > self.max_tensor_height {
            return Err(HwLimitViolation {
                param: "max_tensor_height".into(),
                value: height,
                limit: self.max_tensor_height,
            });
        }
        if depth > self.max_tensor_depth {
            return Err(HwLimitViolation {
                param: "max_tensor_depth".into(),
                value: depth,
                limit: self.max_tensor_depth,
            });
        }
        if channels > self.max_tensor_channels {
            return Err(HwLimitViolation {
                param: "max_tensor_channels".into(),
                value: channels,
                limit: self.max_tensor_channels,
            });
        }
        if rank > self.max_tensor_rank {
            return Err(HwLimitViolation {
                param: "max_tensor_rank".into(),
                value: rank as u64,
                limit: self.max_tensor_rank as u64,
            });
        }
        Ok(())
    }
}

/// Hardware limit violation.
#[derive(Debug, Clone)]
pub struct HwLimitViolation {
    pub param: String,
    pub value: u64,
    pub limit: u64,
}

impl std::fmt::Display for HwLimitViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hardware limit violation: {} = {} exceeds limit {}",
            self.param, self.value, self.limit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane_target::AneFamily;

    #[test]
    fn test_a12_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V5);
        assert_eq!(limits.max_tensor_rank, 5);
        assert_eq!(limits.num_nes, 1);
    }

    // T-40: M1 (V17) has A14-family constraints but Mac-specific hardware limits.
    #[test]
    fn test_m1_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V17);
        assert_eq!(limits.revision, AneRevision::V17);
        assert!(limits.max_tensor_width >= 262144);
        assert!(limits.num_nes >= 6);
        // M1 is A14-class family — no SDPA, no LayerNorm
        assert_eq!(AneRevision::V17.family(), AneFamily::A14);
        assert!(!AneRevision::V17.family().supports_sdpa());
        assert!(!AneRevision::V17.family().supports_layernorm());
    }

    #[test]
    fn test_a18_limits() {
        // T-40: V19 is the A18 revision (iPhone 16), not V17.
        let limits = AneHwLimits::for_revision(AneRevision::V19);
        assert_eq!(limits.revision, AneRevision::V19);
        assert!(limits.max_tensor_width >= 262144);
        assert!(limits.num_nes >= 8); // A18 Pro has 8 NEs
    }

    #[test]
    fn test_tensor_dims_within_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V5);
        assert!(limits.validate_tensor_dims(1024, 1024, 512, 256, 4).is_ok());
    }

    #[test]
    fn test_tensor_dims_exceed_width() {
        let limits = AneHwLimits::for_revision(AneRevision::V5);
        assert!(limits.validate_tensor_dims(999999, 1024, 512, 256, 4).is_err());
    }

    #[test]
    fn test_tensor_rank_exceeds_limit() {
        let limits = AneHwLimits::for_revision(AneRevision::V5);
        assert!(limits.validate_tensor_dims(1024, 1024, 512, 256, 6).is_err());
    }

    #[test]
    fn test_revision_ne_count_increases() {
        let a12 = AneHwLimits::for_revision(AneRevision::V5);
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        let a18 = AneHwLimits::for_revision(AneRevision::V19);
        assert!(a18.num_nes > a14.num_nes);
        assert!(a14.num_nes > a12.num_nes);
    }
}
