//! ANE hardware limit parameters (hal_params) per revision.
//! Source: ane-constraints-docs/02-hardware-and-limits/hardware-versions-limits-and-op-support.md

use crate::ane_target::AneRevision;
use serde::{Deserialize, Serialize};

/// Per-revision ANE hardware limits.
/// These are the key hal_params that ANECompiler validates every op against.
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
            AneRevision::V17 => Self::a18(),
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

    fn a12() -> Self {
        Self { revision: AneRevision::V5, ..Self::a11_legacy() }
    }

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

    fn a18() -> Self {
        Self { revision: AneRevision::V17, max_tensor_width: 262144, num_nes: 6, ..Self::a17() }
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

    #[test]
    fn test_a12_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V5);
        assert_eq!(limits.max_tensor_rank, 5);
        assert_eq!(limits.num_nes, 1);
    }

    #[test]
    fn test_a18_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V17);
        assert!(limits.max_tensor_width >= 262144);
        assert!(limits.num_nes >= 6);
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
        let a18 = AneHwLimits::for_revision(AneRevision::V17);
        assert!(a18.num_nes > a14.num_nes);
        assert!(a14.num_nes > a12.num_nes);
    }
}
