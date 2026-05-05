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
    /// T-99 (Orion #16): Conv-specific channel limit (32768), lower than
    /// general max_tensor_channels (65536). Convolutions with channel
    /// counts between 32768 and 65536 pass general validation but fail
    /// at ANEC compile time.
    pub max_conv_channels: u64,
    pub pe_max_tile_height: u64,
    pub pe_reduction_cout_limit: u64,
    pub num_nes: u32,
    pub ne_transpose_c_max: u64,
    /// Whether these hardware limits have been verified on real hardware.
    /// `false` for approximate/inherited/speculative limits (A12, A13, V26).
    pub verified: bool,
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
            // T-99 (Orion #16): Conv-specific channel limit is 32K
            max_conv_channels: 32768,
            pe_max_tile_height: 2048,
            pe_reduction_cout_limit: 16384,
            num_nes: 1,
            ne_transpose_c_max: 16384,
            verified: true,
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
        Self { revision: AneRevision::V5, verified: false, ..Self::a11_legacy() }
    }

    /// A13 Bionic (ANE V6) hardware limits.
    /// A13 has doubled tensor width/height compared to A11/A12 but
    /// retains the A14Minus elementwise/reduction converter family.
    fn a13() -> Self {
        Self {
            revision: AneRevision::V6,
            max_tensor_width: 32768,
            max_tensor_height: 8192,
            verified: false,
            ..Self::a11_legacy()
        }
    }

    fn a14() -> Self {
        Self { revision: AneRevision::V7, max_tensor_width: 65536, num_nes: 2, verified: true, ..Self::a13() }
    }

    fn a15() -> Self {
        Self { revision: AneRevision::V8, num_nes: 2, verified: true, ..Self::a14() }
    }

    fn a16() -> Self {
        Self {
            revision: AneRevision::V10,
            max_tensor_width: 131072,
            max_tensor_height: 16384,
            num_nes: 4,
            verified: true,
            ..Self::a15()
        }
    }

    fn a17() -> Self {
        Self { revision: AneRevision::V11, num_nes: 4, verified: true, ..Self::a16() }
    }

    /// M1 (Mac, ANE V17) hardware limits.
    ///
    /// M1 uses A14-class ANE (same constraint profile, converters, op support)
    /// but has Mac-specific hardware limits: 6 NEs (vs A14 Bionic's 2),
    /// 262144 max tensor width (vs A14's 65536). The family is A14 — M1
    /// does NOT get A18's SDPA or LayerNorm support.
    fn m1() -> Self {
        Self { revision: AneRevision::V17, max_tensor_width: 262144, num_nes: 6, verified: true, ..Self::a17() }
    }

    /// A18 Bionic (iPhone 16, ANE V19) hardware limits.
    ///
    /// A18 uses A18-family ANE with SDPA, LayerNorm, and E4M3 support.
    /// Has 6 NEs and 262144 max tensor width.
    fn a18() -> Self {
        Self { revision: AneRevision::V19, max_tensor_width: 262144, num_nes: 6, verified: true, ..Self::a17() }
    }

    fn a18_pro() -> Self {
        Self { revision: AneRevision::V19, num_nes: 8, verified: true, ..Self::a18() }
    }

    fn a18_max() -> Self {
        Self { revision: AneRevision::V20, num_nes: 16, verified: true, ..Self::a18_pro() }
    }

    /// T-124 (V-031/V-088): V26 is a speculative/future revision.
    /// These limits are fabricated (inherited from A18_max with num_nes=16)
    /// and have NOT been verified on any real hardware. Any compilation
    /// targeting V26 should be treated as speculative — the model may not
    /// work correctly or at all on actual V26 hardware when it becomes
    /// available.
    fn future() -> Self {
        log::warn!(
            "V26 (future) hardware limits are speculative — inherited from A18_max values \
             with num_nes=16. These have NOT been verified on real hardware. \
             Models compiled for V26 may not function correctly on actual hardware."
        );
        Self { revision: AneRevision::V26, num_nes: 16, verified: false, ..Self::a18_max() }
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

    /// T-99 (Orion #16): Validate conv-specific channel limits.
    /// Conv has a lower channel limit (32768) than the general
    /// max_tensor_channels (65536). This catches convolutions with
    /// channel counts between 32768 and 65536 that pass general
    /// validation but fail at ANEC compile time.
    pub fn validate_conv_channels(&self, channels: u64) -> Result<(), HwLimitViolation> {
        if channels > self.max_conv_channels {
            return Err(HwLimitViolation {
                param: "max_conv_channels".into(),
                value: channels,
                limit: self.max_conv_channels,
            });
        }
        Ok(())
    }

    /// Validate convolution kernel dimensions against hardware limits.
    pub fn validate_conv_dims(
        &self,
        kernel_x: u64,
        kernel_y: u64,
        is_8bit: bool,
    ) -> Result<(), HwLimitViolation> {
        let max_x = if is_8bit { self.max_8b_conv_kernel_dim_x } else { self.max_f16_conv_kernel_dim_x };
        if kernel_x > max_x {
            return Err(HwLimitViolation {
                param: if is_8bit { "max_8b_conv_kernel_dim_x" } else { "max_f16_conv_kernel_dim_x" }.into(),
                value: kernel_x,
                limit: max_x,
            });
        }
        if kernel_y > self.max_conv_kernel_dim_y {
            return Err(HwLimitViolation {
                param: "max_conv_kernel_dim_y".into(),
                value: kernel_y,
                limit: self.max_conv_kernel_dim_y,
            });
        }
        Ok(())
    }

    /// Validate that the channel dimension for a transpose operation
    /// does not exceed the NE transpose C maximum.
    pub fn validate_transpose_c_max(&self, channels: u64) -> Result<(), HwLimitViolation> {
        if channels > self.ne_transpose_c_max {
            return Err(HwLimitViolation {
                param: "ne_transpose_c_max".into(),
                value: channels,
                limit: self.ne_transpose_c_max,
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

    // T-99 (Orion #16): Conv-specific 32K channel limit
    #[test]
    fn test_conv_channels_at_limit_ok() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // 32768 channels should pass conv channel check
        assert!(limits.validate_conv_channels(32768).is_ok());
    }

    #[test]
    fn test_conv_channels_over_limit_rejected() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // 32769 channels should be rejected
        let result = limits.validate_conv_channels(32769);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.param, "max_conv_channels");
        assert_eq!(err.limit, 32768);
    }

    #[test]
    fn test_conv_channels_lower_than_general_channels() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Conv limit (32K) must be lower than general tensor channel limit
        assert!(limits.max_conv_channels < limits.max_tensor_channels);
    }

    // T-124: V26 (future) revision returns correct limits.
    // V26 is speculative but must produce valid struct values:
    // revision == V26, num_nes == 16 (inherited from A18_max).
    #[test]
    fn test_v26_future_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V26);
        assert_eq!(limits.revision, AneRevision::V26);
        assert_eq!(limits.num_nes, 16);
    }

    // ─── T-P3-04: validate_conv_dims() tests ─────────────────────────

    #[test]
    fn test_conv_dims_within_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Kernel 7x7 is at the limit for both fp16 and 8-bit
        assert!(limits.validate_conv_dims(7, 7, false).is_ok());
        assert!(limits.validate_conv_dims(7, 7, true).is_ok());
    }

    #[test]
    fn test_conv_dims_f16_exceeds_x() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // fp16 conv kernel x=8 exceeds max_f16_conv_kernel_dim_x=7
        let result = limits.validate_conv_dims(8, 5, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.param, "max_f16_conv_kernel_dim_x");
        assert_eq!(err.value, 8);
    }

    #[test]
    fn test_conv_dims_8bit_exceeds_x() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // 8-bit conv kernel x=8 exceeds max_8b_conv_kernel_dim_x=7
        let result = limits.validate_conv_dims(8, 5, true);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.param, "max_8b_conv_kernel_dim_x");
        assert_eq!(err.value, 8);
    }

    #[test]
    fn test_conv_dims_exceeds_y() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Kernel y=8 exceeds max_conv_kernel_dim_y=7
        let result = limits.validate_conv_dims(5, 8, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.param, "max_conv_kernel_dim_y");
        assert_eq!(err.value, 8);
    }

    #[test]
    fn test_conv_dims_small_kernel_ok() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Small kernels should always pass
        assert!(limits.validate_conv_dims(1, 1, false).is_ok());
        assert!(limits.validate_conv_dims(3, 3, true).is_ok());
        assert!(limits.validate_conv_dims(5, 5, false).is_ok());
    }

    // ─── T-P3-08: validate_transpose_c_max() tests ────────────────────

    #[test]
    fn test_transpose_c_max_within_limit() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Channels at the limit should pass
        assert!(limits.validate_transpose_c_max(limits.ne_transpose_c_max).is_ok());
    }

    #[test]
    fn test_transpose_c_max_exceeds_limit() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Channels over the limit should fail
        let result = limits.validate_transpose_c_max(limits.ne_transpose_c_max + 1);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.param, "ne_transpose_c_max");
        assert_eq!(err.limit, limits.ne_transpose_c_max);
    }

    #[test]
    fn test_transpose_c_max_small_channels_ok() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        assert!(limits.validate_transpose_c_max(256).is_ok());
        assert!(limits.validate_transpose_c_max(1024).is_ok());
    }
}
