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
/// | A17     | Full dtype    | Yes    | Yes       | A14Plus     | All types | Yes       |
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
    /// A17 Pro — adds E4M3 (FP8) conditional support (LSE_6).
    /// Retains SDPA, LayerNorm, and ArgMinMax from A16.
    A17,
    /// A18 Bionic (M4) — latest generation. Drops ArgMinMax (no LSE_7 converter).
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
        matches!(self, AneFamily::A16 | AneFamily::A17 | AneFamily::A18)
    }

    /// Whether this family supports LayerNorm on ANE.
    /// LayerNorm is supported starting from A15.
    pub fn supports_layernorm(&self) -> bool {
        matches!(self, AneFamily::A15 | AneFamily::A16 | AneFamily::A17 | AneFamily::A18)
    }

    /// Whether this family supports ReduceMin for non-FP types.
    /// A11/A12/A13 only support FP ReduceMin; A14+ supports all types.
    pub fn supports_reducemin_all_dtypes(&self) -> bool {
        matches!(
            self,
            AneFamily::A14 | AneFamily::A15 | AneFamily::A16 | AneFamily::A17 | AneFamily::A18
        )
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
    /// ⚠️ (conditionally supported) on A17+ (LSE_6 and later).
    /// A17 Pro (LSE_6) adds conditional E4M3 support; A18 (LSE_7) continues it.
    /// The ANE error message is: "E4M3 is not supported" on older architectures.
    ///
    /// T-35 (I-14): E4M3 dtype constraint enforcement.
    /// T-52 (I-26): Added A17 family for E4M3 support on V11 (A17 Pro).
    pub fn supports_e4m3(&self) -> bool {
        matches!(self, AneFamily::A17 | AneFamily::A18)
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
            // T-52: V11 (A17 Pro) is its own family (LSE_6) with E4M3 support.
            // V10 (A16 Bionic) remains in A16 family without E4M3.
            AneRevision::V10 => AneFamily::A16,
            AneRevision::V11 => AneFamily::A17,
            // V17 is Apple M1 (Mac), which uses A14-class ANE.
            // M1 has A14's constraint profile: no SDPA, no LayerNorm,
            // A14Plus elementwise/reduction converters, full-dtype broadcast.
            // Do NOT map V17 to A18 — M1 does not have A18's SDPA/LayerNorm.
            AneRevision::V17 => AneFamily::A14,
            AneRevision::V19 | AneRevision::V20 | AneRevision::V26 => AneFamily::A18,
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
        // T-52: V11 (A17 Pro) is its own family with E4M3 support.
        assert_eq!(AneRevision::V11.family(), AneFamily::A17);
        // T-40: V17 (M1) is A14-class, NOT A18.
        assert_eq!(AneRevision::V17.family(), AneFamily::A14);
        assert_eq!(AneRevision::V19.family(), AneFamily::A18);
    }

    #[test]
    fn test_supports_sdpa() {
        assert!(!AneFamily::A13.supports_sdpa());
        assert!(!AneFamily::A14.supports_sdpa());
        assert!(AneFamily::A16.supports_sdpa());
        assert!(AneFamily::A17.supports_sdpa());
        assert!(AneFamily::A18.supports_sdpa());
    }

    #[test]
    fn test_supports_layernorm() {
        assert!(!AneFamily::A13.supports_layernorm());
        assert!(!AneFamily::A14.supports_layernorm());
        assert!(AneFamily::A15.supports_layernorm());
        assert!(AneFamily::A16.supports_layernorm());
        assert!(AneFamily::A17.supports_layernorm());
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
        assert!(AneFamily::A16.supports_reducemin_all_dtypes());
        assert!(AneFamily::A17.supports_reducemin_all_dtypes());
        assert!(AneFamily::A18.supports_reducemin_all_dtypes());
    }

    #[test]
    fn test_supports_argminmax() {
        // ArgMinMax has ANEC converters for LSE_0-6 (all families through A17).
        // A18 (LSE_7) has no converter — this is the unique case where a newer
        // family drops support for an op that older families have.
        assert!(AneFamily::A11Legacy.supports_argminmax());
        assert!(AneFamily::A12.supports_argminmax());
        assert!(AneFamily::A13.supports_argminmax());
        assert!(AneFamily::A14.supports_argminmax());
        assert!(AneFamily::A15.supports_argminmax());
        assert!(AneFamily::A16.supports_argminmax());
        assert!(AneFamily::A17.supports_argminmax()); // LSE_6 has ConvertReductionArg
        assert!(!AneFamily::A18.supports_argminmax());
    }

    #[test]
    fn test_supports_e4m3() {
        // E4M3 (FP8) is NOT supported on A11-A16; A17+ has conditional support (LSE_6+)
        assert!(!AneFamily::A11Legacy.supports_e4m3());
        assert!(!AneFamily::A12.supports_e4m3());
        assert!(!AneFamily::A13.supports_e4m3());
        assert!(!AneFamily::A14.supports_e4m3());
        assert!(!AneFamily::A15.supports_e4m3());
        assert!(!AneFamily::A16.supports_e4m3());
        assert!(AneFamily::A17.supports_e4m3()); // LSE_6 adds conditional E4M3
        assert!(AneFamily::A18.supports_e4m3());
    }

    // ─── T-40: V17 (M1) → A14 Family Mapping Tests ────────────────

    #[test]
    fn test_m1_v17_is_a14_family() {
        // T-40: V17 is M1 (Mac), which has A14-class ANE.
        // Previously mapped to A18, which incorrectly gave M1
        // A18's SDPA and LayerNorm support.
        assert_eq!(AneRevision::V17.family(), AneFamily::A14);
    }

    #[test]
    fn test_m1_no_sdpa() {
        // M1 (V17, A14-class) does NOT support SDPA.
        // A18 does — this was the key misclassification bug.
        let m1_family = AneRevision::V17.family();
        assert!(
            !m1_family.supports_sdpa(),
            "M1 (V17) should NOT support SDPA — it has A14-class ANE"
        );
    }

    #[test]
    fn test_m1_no_layernorm() {
        // M1 (V17, A14-class) does NOT support LayerNorm on ANE.
        // A15+ supports LayerNorm; M1 is A14-class.
        let m1_family = AneRevision::V17.family();
        assert!(
            !m1_family.supports_layernorm(),
            "M1 (V17) should NOT support LayerNorm — it has A14-class ANE"
        );
    }

    #[test]
    fn test_m1_full_dtype_broadcast() {
        // M1 (V17, A14-class) supports full-dtype broadcast
        // (unlike A11/A12 which are FP16-only).
        let m1_family = AneRevision::V17.family();
        assert!(
            !m1_family.broadcast_fp16_only(),
            "M1 (V17) should support full-dtype broadcast — A14-class lifts this restriction"
        );
    }

    #[test]
    fn test_m1_a14plus_converters() {
        // M1 (V17, A14-class) uses A14Plus elementwise/reduction converters
        // (not A14Minus which is only A11/A12/A13).
        let m1_family = AneRevision::V17.family();
        assert!(
            !m1_family.uses_a14minus_converters(),
            "M1 (V17) should use A14Plus converters — it is A14-class"
        );
    }

    #[test]
    fn test_m1_supports_argminmax() {
        // M1 (V17, A14-class) DOES support ArgMinMax.
        // Only A18 (LSE_7) lacks the ConvertReductionArg converter.
        let m1_family = AneRevision::V17.family();
        assert!(
            m1_family.supports_argminmax(),
            "M1 (V17) should support ArgMinMax — A14 has LSE_3 converter"
        );
    }

    #[test]
    fn test_m1_no_e4m3() {
        // M1 (V17, A14-class) does NOT support E4M3 (FP8).
        // Only A18+ has limited E4M3 support.
        let m1_family = AneRevision::V17.family();
        assert!(
            !m1_family.supports_e4m3(),
            "M1 (V17) should NOT support E4M3 — only A18+ has limited support"
        );
    }

    #[test]
    fn test_m1_reducemin_all_dtypes() {
        // M1 (V17, A14-class) supports ReduceMin for all dtypes.
        // A14+ lifts the FP-only ReduceMin restriction.
        let m1_family = AneRevision::V17.family();
        assert!(
            m1_family.supports_reducemin_all_dtypes(),
            "M1 (V17) should support ReduceMin for all dtypes — A14+ feature"
        );
    }

    #[test]
    fn test_a18_v19_not_v17() {
        // A18 family's canonical revision is V19 (iPhone 16), NOT V17 (M1).
        // V17 is M1 which is A14-class.
        assert_ne!(AneRevision::V17.family(), AneFamily::A18, "V17 (M1) must NOT be in A18 family");
        assert_eq!(AneRevision::V19.family(), AneFamily::A18, "V19 is the correct A18 revision");
    }

    #[test]
    fn test_v17_and_v7_same_family() {
        // V7 (A14 Bionic) and V17 (M1) should be in the same family.
        // Both are A14-class ANE with identical constraint profiles.
        assert_eq!(
            AneRevision::V7.family(),
            AneRevision::V17.family(),
            "V7 (A14) and V17 (M1) should share A14 family"
        );
    }

    #[test]
    fn test_a18_constraints_differ_from_m1() {
        // A18 has capabilities that M1 (A14-class) does not:
        // SDPA, LayerNorm, E4M3. These must NOT be available on M1.
        let m1_family = AneRevision::V17.family();
        let a18_family = AneRevision::V19.family();

        assert!(a18_family.supports_sdpa());
        assert!(!m1_family.supports_sdpa());
        assert!(a18_family.supports_layernorm());
        assert!(!m1_family.supports_layernorm());
        assert!(a18_family.supports_e4m3());
        assert!(!m1_family.supports_e4m3());

        // But A18 drops ArgMinMax support (no LSE_7 converter)
        assert!(!a18_family.supports_argminmax());
        assert!(m1_family.supports_argminmax());
    }

    // ─── T-52: A17 Pro (V11) Family Tests ──────────────────────────

    #[test]
    fn test_a17_v11_family_mapping() {
        // T-52: V11 (A17 Pro) is A17 family, NOT A16.
        assert_eq!(AneRevision::V11.family(), AneFamily::A17);
        // V10 (A16 Bionic) remains A16
        assert_eq!(AneRevision::V10.family(), AneFamily::A16);
    }

    #[test]
    fn test_a17_e4m3_support() {
        // T-52: A17 (LSE_6) supports E4M3 — the core fix for I-26.
        assert!(AneFamily::A17.supports_e4m3());
        // A16 still does NOT support E4M3
        assert!(!AneFamily::A16.supports_e4m3());
    }

    #[test]
    fn test_a17_sdpa_support() {
        // A17 Pro has SDPA (same as A16)
        assert!(AneFamily::A17.supports_sdpa());
    }

    #[test]
    fn test_a17_layernorm_support() {
        // A17 Pro has LayerNorm (same as A16)
        assert!(AneFamily::A17.supports_layernorm());
    }

    #[test]
    fn test_a17_argminmax_support() {
        // A17 uses LSE_6 which has ConvertReductionArg converter
        assert!(AneFamily::A17.supports_argminmax());
        // A18 (LSE_7) drops it
        assert!(!AneFamily::A18.supports_argminmax());
    }

    #[test]
    fn test_a17_full_dtype_broadcast() {
        // A17 supports full-dtype broadcast (like A16)
        assert!(!AneFamily::A17.broadcast_fp16_only());
    }

    #[test]
    fn test_a17_a14plus_converters() {
        // A17 uses A14Plus converters
        assert!(!AneFamily::A17.uses_a14minus_converters());
    }

    #[test]
    fn test_a17_reducemin_all_dtypes() {
        // A17 supports ReduceMin for all dtypes (A14+ feature)
        assert!(AneFamily::A17.supports_reducemin_all_dtypes());
    }

    #[test]
    fn test_a17_vs_a16_capabilities() {
        // A17 differs from A16 in exactly one way: E4M3 support
        assert!(!AneFamily::A16.supports_e4m3());
        assert!(AneFamily::A17.supports_e4m3());
        // All other capabilities are identical
        assert_eq!(AneFamily::A16.broadcast_fp16_only(), AneFamily::A17.broadcast_fp16_only());
        assert_eq!(
            AneFamily::A16.uses_a14minus_converters(),
            AneFamily::A17.uses_a14minus_converters()
        );
        assert_eq!(AneFamily::A16.supports_sdpa(), AneFamily::A17.supports_sdpa());
        assert_eq!(AneFamily::A16.supports_layernorm(), AneFamily::A17.supports_layernorm());
        assert_eq!(
            AneFamily::A16.supports_reducemin_all_dtypes(),
            AneFamily::A17.supports_reducemin_all_dtypes()
        );
        assert_eq!(AneFamily::A16.supports_argminmax(), AneFamily::A17.supports_argminmax());
    }
}
