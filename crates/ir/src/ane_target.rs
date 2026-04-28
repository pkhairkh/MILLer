//! ANE target device types — family and revision identifiers.
//!
//! Sprint 59 dependency: these types are used by constraint validation,
//! dtype legality checks, and hardware limit enforcement.

use serde::{Deserialize, Serialize};

/// ANE silicon family — groups revisions with similar constraint profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneFamily {
    /// A11 (M1) — legacy ANE, V4 hardware revision.
    A11Legacy,
    /// A12 (M2) — improved ANE.
    A12,
    /// A14 (M3) — significant expansion.
    A14,
    /// A15 — adds LayerNorm support.
    A15,
    /// A16 — adds reliable SDPA.
    A16,
    /// A18 (M4) — latest generation.
    A18,
}

impl AneFamily {
    /// Whether this family restricts broadcast operations to fp16 only.
    /// A11/A12 require fp16 for broadcasts; A14+ lifts this restriction.
    pub fn broadcast_fp16_only(&self) -> bool {
        matches!(self, AneFamily::A11Legacy | AneFamily::A12)
    }

    /// Whether this family supports scaled dot-product attention (SDPA) reliably.
    /// SDPA is only reliable starting from A16.
    pub fn supports_sdpa(&self) -> bool {
        matches!(self, AneFamily::A16 | AneFamily::A18)
    }

    /// Whether this family supports LayerNorm on ANE.
    /// LayerNorm is supported starting from A15.
    pub fn supports_layernorm(&self) -> bool {
        matches!(self, AneFamily::A15 | AneFamily::A16 | AneFamily::A18)
    }
}

/// ANE hardware revision — corresponds to specific silicon versions.
/// Maps to hardware revision numbers consistent with ANE behavior across chip generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneRevision {
    V4,  // A11 (M1)
    V5,  // A12 (M2)
    V6,  // A13
    V7,  // A14 (M3)
    V8,  // A15
    V10, // A16
    V11, // A17
    V17, // A18 (M4)
    V19, // A18 Pro
    V20, // A18 Max
    V26, // Future
}

impl AneRevision {
    /// Get the family for this revision.
    pub fn family(&self) -> AneFamily {
        match self {
            AneRevision::V4 => AneFamily::A11Legacy,
            AneRevision::V5 => AneFamily::A12,
            AneRevision::V6 | AneRevision::V7 => AneFamily::A14,
            AneRevision::V8 => AneFamily::A15,
            AneRevision::V10 | AneRevision::V11 => AneFamily::A16,
            AneRevision::V17 | AneRevision::V19 | AneRevision::V20 | AneRevision::V26 => {
                AneFamily::A18
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_fp16_only() {
        assert!(AneFamily::A11Legacy.broadcast_fp16_only());
        assert!(AneFamily::A12.broadcast_fp16_only());
        assert!(!AneFamily::A14.broadcast_fp16_only());
        assert!(!AneFamily::A18.broadcast_fp16_only());
    }

    #[test]
    fn test_revision_to_family() {
        assert_eq!(AneRevision::V4.family(), AneFamily::A11Legacy);
        assert_eq!(AneRevision::V5.family(), AneFamily::A12);
        assert_eq!(AneRevision::V7.family(), AneFamily::A14);
        assert_eq!(AneRevision::V8.family(), AneFamily::A15);
        assert_eq!(AneRevision::V10.family(), AneFamily::A16);
        assert_eq!(AneRevision::V17.family(), AneFamily::A18);
    }

    #[test]
    fn test_supports_sdpa() {
        assert!(!AneFamily::A14.supports_sdpa());
        assert!(AneFamily::A16.supports_sdpa());
        assert!(AneFamily::A18.supports_sdpa());
    }

    #[test]
    fn test_supports_layernorm() {
        assert!(!AneFamily::A14.supports_layernorm());
        assert!(AneFamily::A15.supports_layernorm());
        assert!(AneFamily::A18.supports_layernorm());
    }
}
