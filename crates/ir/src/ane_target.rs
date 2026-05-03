//! ANE target device types — family and revision identifiers.
//!
//! Sprint 59 dependency: these types are used by constraint validation,
//! dtype legality checks, and hardware limit enforcement.

use serde::{Deserialize, Serialize};

/// ANE silicon family — groups revisions with similar constraint profiles.
///
/// Each family represents a distinct constraint profile for the ANE.
/// Families are ordered by capability level (see `family_level()`).
///
/// | Family  | Broadcast     | SDPA   | LayerNorm | Elementwise | ReduceMin | ArgMinMax |
/// |---------|---------------|--------|-----------|-------------|-----------|-----------|
/// | A11Legacy | FP16-only   | No     | No        | A14Minus    | FP only   | Yes       |
/// | A12     | FP16-only     | No     | No        | A14Minus    | FP only   | Yes       |
/// | A13     | Full dtype    | No     | No        | A14Minus    | FP only   | Yes       |
/// | A14     | Full dtype    | No     | No        | A14Plus     | All types | Yes       |
/// | A15     | Full dtype    | No     | Yes       | A14Plus     | All types | Yes       |
/// | A16     | Full dtype    | Yes    | Yes       | A14Plus     | All types | Yes       |
/// | A18     | Full dtype    | Yes    | Yes       | A14Plus     | All types | **No**    |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneFamily {
    /// A11 Bionic — legacy ANE, V4 hardware revision.
    A11Legacy,
    /// A12 Bionic — improved ANE, FP16-only broadcast, A14Minus converters.
    A12,
    /// A13 Bionic — lifts FP16 broadcast restriction but retains A14Minus converters.
    A13,
    /// A14 Bionic — significant expansion, A14Plus elementwise/reduction converters.
    A14,
    /// A15 Bionic — adds LayerNorm support.
    A15,
    /// A16 Bionic — adds reliable SDPA.
    A16,
    /// A18 Bionic (M4) — latest generation.
    A18,
}

impl AneFamily {
    /// Whether this family restricts broadcast operations to fp16 only.
    /// A11/A12 require fp16 for broadcasts; A13+ lifts this restriction.
    pub fn broadcast_fp16_only(&self) -> bool {
        matches!(self, AneFamily::A11Legacy | AneFamily::A12)
    }

    /// Whether this family uses A14Minus elementwise/reduction converters.
    /// A14Minus converters have narrower dtype support and use older
    /// lowering paths (e.g., ConvertSquareA13Minus, ConvertReductionA14Minus).
    /// A13+ retains A14Minus; A14+ uses A14Plus converters.
    pub fn uses_a14minus_converters(&self) -> bool {
        matches!(self, AneFamily::A11Legacy | AneFamily::A12 | AneFamily::A13)
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

    /// Whether this family supports ReduceMin for non-FP types.
    /// A11/A12/A13 only support FP ReduceMin; A14+ supports all types.
    pub fn supports_reducemin_all_dtypes(&self) -> bool {
        matches!(self, AneFamily::A14 | AneFamily::A15 | AneFamily::A16 | AneFamily::A18)
    }

    /// Whether this family supports ArgMin/ArgMax (reduce_argmax, reduce_argmin).
    /// The ANEC has `ConvertReductionArg` converters for LSE_0 through LSE_6
    /// (A11Legacy through A16), but there is **no LSE_7 converter** for A18/M4.
    /// ArgMinMax ops that pass placement validation on A18 will silently fail
    /// at emission time because no ANEC converter exists.
    pub fn supports_argminmax(&self) -> bool {
        !matches!(self, AneFamily::A18)
    }

    /// Whether this family supports E4M3 (FP8) data type on ANE.
    ///
    /// Per the per-op support matrix, E4M3/E5M2 is ❌ on A11 through A16,
    /// and ⚠️ (conditionally supported) on A17/A18.
    /// The ANE error message is: "E4M3 is not supported" on older architectures.
    ///
    /// T-35 (I-14): E4M3 dtype constraint enforcement.
    pub fn supports_e4m3(&self) -> bool {
        matches!(self, AneFamily::A18)
    }
}

/// ANE hardware revision — corresponds to specific silicon versions.
/// Maps to hardware revision numbers consistent with ANE behavior across chip generations.
///
/// Note: The revision numbers (V4, V5, V6...) are ANE coprocessor revision IDs
/// and do NOT correspond to the HWTraits version numbers used in Apple's MLIR
/// compiler (HWTraits<6> = A12, HWTraits<7> = A13, etc.). The mapping between
/// ANE revision and chip generation is based on observed ANEC behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneRevision {
    V4,  // A11 Bionic (iPhone 8/X)
    V5,  // A12 Bionic (iPhone XS/XR)
    V6,  // A13 Bionic (iPhone 11)
    V7,  // A14 Bionic (iPhone 12)
    V8,  // A15 Bionic (iPhone 13)
    V10, // A16 Bionic (iPhone 14 Pro)
    V11, // A17 Pro (iPhone 15 Pro)
    V17, // M1 (Mac)
    V19, // A18/A18 Pro (iPhone 16)
    V20, // M4 (Mac)
    V26, // Future
}

impl AneRevision {
    /// Get the family for this revision.
    ///
    /// A13 (V6) is mapped to its own family because it has a distinct
    /// constraint profile: it lifts the FP16-only broadcast restriction
    /// (unlike A12) but retains A14Minus elementwise/reduction converters
    /// and FP-only ReduceMin (unlike A14).
    pub fn family(&self) -> AneFamily {
        match self {
            AneRevision::V4 => AneFamily::A11Legacy,
            AneRevision::V5 => AneFamily::A12,
            AneRevision::V6 => AneFamily::A13,
            AneRevision::V7 => AneFamily::A14,
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
        assert!(!AneFamily::A13.broadcast_fp16_only());
        assert!(!AneFamily::A14.broadcast_fp16_only());
        assert!(!AneFamily::A18.broadcast_fp16_only());
    }

    #[test]
    fn test_a13_constraint_profile() {
        // A13 lifts FP16-only broadcast but retains A14Minus converters
        assert!(!AneFamily::A13.broadcast_fp16_only());
        assert!(AneFamily::A13.uses_a14minus_converters());
        assert!(!AneFamily::A13.supports_sdpa());
        assert!(!AneFamily::A13.supports_layernorm());
        assert!(!AneFamily::A13.supports_reducemin_all_dtypes());
    }

    #[test]
    fn test_revision_to_family() {
        assert_eq!(AneRevision::V4.family(), AneFamily::A11Legacy);
        assert_eq!(AneRevision::V5.family(), AneFamily::A12);
        assert_eq!(AneRevision::V6.family(), AneFamily::A13);
        assert_eq!(AneRevision::V7.family(), AneFamily::A14);
        assert_eq!(AneRevision::V8.family(), AneFamily::A15);
        assert_eq!(AneRevision::V10.family(), AneFamily::A16);
        assert_eq!(AneRevision::V17.family(), AneFamily::A18);
    }

    #[test]
    fn test_supports_sdpa() {
        assert!(!AneFamily::A13.supports_sdpa());
        assert!(!AneFamily::A14.supports_sdpa());
        assert!(AneFamily::A16.supports_sdpa());
        assert!(AneFamily::A18.supports_sdpa());
    }

    #[test]
    fn test_supports_layernorm() {
        assert!(!AneFamily::A13.supports_layernorm());
        assert!(!AneFamily::A14.supports_layernorm());
        assert!(AneFamily::A15.supports_layernorm());
        assert!(AneFamily::A18.supports_layernorm());
    }

    #[test]
    fn test_uses_a14minus_converters() {
        assert!(AneFamily::A11Legacy.uses_a14minus_converters());
        assert!(AneFamily::A12.uses_a14minus_converters());
        assert!(AneFamily::A13.uses_a14minus_converters());
        assert!(!AneFamily::A14.uses_a14minus_converters());
        assert!(!AneFamily::A18.uses_a14minus_converters());
    }

    #[test]
    fn test_supports_reducemin_all_dtypes() {
        assert!(!AneFamily::A12.supports_reducemin_all_dtypes());
        assert!(!AneFamily::A13.supports_reducemin_all_dtypes());
        assert!(AneFamily::A14.supports_reducemin_all_dtypes());
        assert!(AneFamily::A18.supports_reducemin_all_dtypes());
    }

    #[test]
    fn test_supports_argminmax() {
        // ArgMinMax has ANEC converters for LSE_0-6 (all families through A16).
        // A18 (LSE_7) has no converter — this is the unique case where a newer
        // family drops support for an op that older families have.
        assert!(AneFamily::A11Legacy.supports_argminmax());
        assert!(AneFamily::A12.supports_argminmax());
        assert!(AneFamily::A13.supports_argminmax());
        assert!(AneFamily::A14.supports_argminmax());
        assert!(AneFamily::A15.supports_argminmax());
        assert!(AneFamily::A16.supports_argminmax());
        assert!(!AneFamily::A18.supports_argminmax());
    }

    #[test]
    fn test_supports_e4m3() {
        // E4M3 (FP8) is NOT supported on A11-A16; only A18+ has limited support
        assert!(!AneFamily::A11Legacy.supports_e4m3());
        assert!(!AneFamily::A12.supports_e4m3());
        assert!(!AneFamily::A13.supports_e4m3());
        assert!(!AneFamily::A14.supports_e4m3());
        assert!(!AneFamily::A15.supports_e4m3());
        assert!(!AneFamily::A16.supports_e4m3());
        assert!(AneFamily::A18.supports_e4m3());
    }
}
