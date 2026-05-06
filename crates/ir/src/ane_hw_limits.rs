//! ANE hardware limit parameters (hal_params) per revision.
//! Source: ane-constraints-docs/02-hardware-and-limits/hardware-versions-limits-and-op-support.md

use crate::ane_target::AneRevision;
use serde::{Deserialize, Serialize};

/// Sub-variant within an ANE family for fine-grained differentiation.
///
/// Some revisions share the same AneFamily but have different hardware
/// characteristics (e.g., A18 Pro vs A18 Max have different NE counts).
/// Additionally, within each family there are chip-level sub-variants
/// (denoted by their HAL identifier, e.g., H14c, H14g) that may have
/// subtle constraint differences.
///
/// T-P4-07: Added chip-level HAL sub-variants. The naming convention
/// follows the ANEC binary's HAL identifiers:
/// - `H<family_number><suffix>` where suffix indicates the chip variant
/// - `c` = compact/budget (e.g., A14 Bionic = H14c)
/// - `g` = standard (e.g., A14 GPU variant = H14g)
/// - `s` = performance (e.g., A16 Pro = H16s)
/// - `a` = application processor (e.g., A17 Pro = H17a)
///
/// F-HAL-01 (T-P7-09): Sub-variant-specific constraint differences are
/// now modeled for verified sub-variants (H14c, H15c, H16c, H16s).
/// Compact variants (`c`) have fewer NEs than their standard (`g`)
/// counterparts. The performance variant H16s has expanded PE reduction
/// and hw_wa limits. H13g remains unverified (no reliable compact data
/// for A13). Canonical variants (H14g, H15g, H16g, H17a) inherit from
/// their parent family's verified limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum AneSubVariant {
    /// Standard variant (default).
    #[default]
    Standard,
    /// Pro variant with more NEs (e.g., A18 Pro with 8 NEs).
    Pro,
    /// Max variant with maximum NEs (e.g., A18 Max with 16 NEs).
    Max,
    /// Mac variant (e.g., M1 with Mac-specific limits).
    Mac,

    // ─── T-P4-07: Chip-level HAL sub-variants ──────────────────────
    // These represent specific chip SKUs within each ANE family.
    // F-HAL-01: Verified sub-variants now have differentiated limits.
    /// A13 Bionic — H13g (standard A13 SKU).
    /// Unverified: no reliable compact data for A13 family.
    H13g,
    /// A14 Bionic — H14c (compact/budget SKU, e.g., iPhone 12 mini).
    /// Verified: 1 NE (vs standard A14's 2 NEs).
    H14c,
    /// A14 Bionic — H14g (standard SKU, e.g., iPhone 12).
    /// Canonical A14 variant — inherits verified a14() limits.
    H14g,
    /// A15 Bionic — H15c (compact/budget SKU, e.g., iPhone 13 mini).
    /// Verified: 1 NE (vs standard A15's 2 NEs).
    H15c,
    /// A15 Bionic — H15g (standard SKU, e.g., iPhone 13).
    /// Canonical A15 variant — inherits verified a15() limits.
    H15g,
    /// A16 Bionic — H16c (compact/budget SKU, e.g., iPhone 15).
    /// Verified: 2 NEs (vs standard A16's 4 NEs).
    H16c,
    /// A16 Bionic — H16g (standard SKU, e.g., iPhone 14 Pro).
    /// Canonical A16 variant — inherits verified a16() limits.
    H16g,
    /// A16 Bionic — H16s (performance SKU, e.g., iPhone 15 Pro with A16).
    /// Verified: expanded pe_reduction_cout_limit and hw_wa limits.
    H16s,
    /// A17 Pro — H17a (application processor SKU, e.g., iPhone 15 Pro).
    /// Canonical A17 variant — inherits verified a17() limits.
    H17a,
}

/// Per-revision ANE hardware limits.
/// Key hardware limit parameters that govern ANE op placement.
///
/// T-P6-01: Extended with 35+ missing hal_params from ANEC binary research.
/// Many of these are unverified (marked via the `verified` field on the struct)
/// and use conservative values derived from forensic analysis of the ANEC binary.
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
    /// Kernel size threshold above which special handling is needed.
    /// A11-A13: 7, A14+: 9.
    pub large_kernel_threshold: u64,
    /// T-P4-01: Threshold above which ANE activates "large kernel" mode
    /// with additional constraints (W/H must be multiple of 8, no groups,
    /// no dilation, no depth, stride ≤ 2). Value is 16 for all revisions
    /// based on binary forensic evidence.
    pub large_kernel_mode_threshold: u64,
    /// Sub-variant within an ANE family for fine-grained differentiation.
    pub sub_variant: AneSubVariant,
    /// T-P6-04: Whether this is a uANE (unified ANE) variant found on
    /// Apple Silicon Macs.
    pub is_uane: bool,

    // ─── T-P6-01: Missing hal_params from ANEC binary research ──────
    // Source: ane-constraints-docs/02-hardware-and-limits/ §4
    // These were identified as missing during forensic analysis of the
    // ANEC binary. Many values are derived from binary research and
    // marked verified=false until hardware testing confirms them.

    // ─── Convolution Kernel Depth Limits ────────────────────────────
    /// T-P6-01: Max kernel depth (z-dimension) for convolutions.
    /// All revisions: 1 (convolutions are 2D only on ANE).
    pub max_conv_kernel_dim_z: u64,
    /// T-P6-01: Max kernel depth for large-kernel convolutions.
    /// All revisions: 1 (large kernel mode is still 2D).
    pub max_large_conv_kernel_dim_z: u64,

    // ─── Large Kernel Dimension Limits ──────────────────────────────
    /// T-P6-01: Max kernel width for 8-bit large conv.
    /// A11-A13: 7, A14+: 27 (large kernel mode extends range).
    pub max_8b_large_conv_kernel_dim_x: u64,
    /// T-P6-01: Max kernel width for FP16 large conv.
    /// A11-A13: 7, A14+: 27.
    pub max_f16_large_conv_kernel_dim_x: u64,
    /// T-P6-01: Max kernel height for large conv.
    /// All revisions: 27.
    pub max_large_conv_kernel_dim_y: u64,
    /// T-P6-01: Min kernel height for large conv.
    /// All revisions: 1.
    pub min_large_conv_kernel_dim_y: u64,
    /// T-P6-01: Min kernel width for 8-bit large conv.
    /// All revisions: 1.
    pub min_8b_large_conv_kernel_dim_x: u64,
    /// T-P6-01: Min kernel width for FP16 large conv.
    /// All revisions: 1.
    pub min_f16_large_conv_kernel_dim_x: u64,

    // ─── Pooling Kernel Limits ──────────────────────────────────────
    /// T-P6-01: Max pooling kernel height for PE.
    /// All revisions: 27.
    pub pe_max_pooling_kh: u64,
    /// T-P6-01: Max pooling kernel width for PE.
    /// All revisions: 27.
    pub pe_max_pooling_kw: u64,
    /// T-P6-01: MaxPool z-dim size when input z-size is 1.
    /// All revisions: 1.
    pub max_maxpool_kernel_dim_z_sz_1: u64,
    /// T-P6-01: MaxPool z-dim size when input z-size is 2.
    /// All revisions: 1.
    pub max_maxpool_kernel_dim_z_sz_2: u64,

    // ─── Convolution Padding Limits ─────────────────────────────────
    /// T-P6-01: Max padding in x-direction for conv.
    /// All revisions: 7.
    pub max_conv_pad_x: u64,
    /// T-P6-01: Max padding in y-direction for conv.
    /// All revisions: 7.
    pub max_conv_pad_y: u64,
    /// T-P6-01: Max padding in z-direction for conv.
    /// All revisions: 1 (2D conv only).
    pub max_conv_pad_z: u64,

    // ─── PE Limits ──────────────────────────────────────────────────
    /// T-P6-01: Max patch width+height sum (log2) for PE operations.
    /// Constrains the spatial extent of PE tile operations.
    /// A11-A13: 14, A14+: 15.
    pub pe_max_patch_width_height_sum_log2: u64,
    /// T-P6-01: Min patch width (log2) for PE operations.
    /// All revisions: 0.
    pub pe_min_patch_width_log2: u64,
    /// T-P6-01: Max input channels for PE W-to-C transpose.
    /// All revisions: 16384.
    pub pe_max_transpose_wtoc_cin: u64,
    /// T-P6-01: Max output channels for PE C-to-W transpose.
    /// All revisions: 16384.
    pub pe_max_transpose_ctow_cout: u64,
    /// T-P6-01: Feature flag for PE patch size constraint.
    /// A11-A13: false, A14+: true.
    pub has_pe_max_patch_width_height_sum: bool,

    // ─── NE Limits ──────────────────────────────────────────────────
    /// T-P6-01: Max width for NE transpose operations.
    /// All revisions: 16384.
    pub ne_transpose_w_max: u64,
    /// T-P6-01: Feature flag for NE RCAS (Row-Column Address Shuffling)
    /// support. A11-A13: false, A14+: true.
    pub ne_supports_rcas: bool,
    /// T-P6-01 (N-010): LUT size in bytes for NE palette operations.
    /// Critical for LUT overflow detection — palettized ops with
    /// large LUT entries may exceed this hardware limit.
    /// A11-A14: 256, A15+: 512.
    pub ne_palette_lut_size_in_bytes: u64,

    // ─── Elementwise Alignment Limits ───────────────────────────────
    /// T-P6-01: 64-byte alignment boundary for elementwise ops.
    /// All revisions: 64.
    pub ew_limit_64: u64,
    /// T-P6-01: 128-byte alignment boundary for elementwise ops.
    /// All revisions: 128.
    pub ew_limit_128: u64,
    /// T-P6-01: 256-byte alignment boundary for elementwise ops.
    /// All revisions: 256.
    pub ew_limit_256: u64,

    // ─── Small Source Mode Limits ───────────────────────────────────
    /// T-P6-01: Max source width for NP2-6 small source mode (inclusive).
    /// All revisions: 6.
    pub np2_6_max_src_width_inclusive: u64,
    /// T-P6-01: Min destination width for NP2-6 small source mode (exclusive).
    /// All revisions: 7.
    pub np2_6_min_dst_width_exclusive: u64,
    /// T-P6-01: Max source width for NP2-10 small source mode (inclusive).
    /// All revisions: 10.
    pub np2_10_max_src_width_inclusive: u64,
    /// T-P6-01: Min destination width for NP2-10 small source mode (exclusive).
    /// All revisions: 11.
    pub np2_10_min_dst_width_exclusive: u64,
    /// T-P6-01: Max source width for half-WU NP2-6 mode (inclusive).
    /// All revisions: 6.
    pub half_wu_np2_6_max_src_width_inclusive: u64,
    /// T-P6-01: Min destination width for half-WU NP2-6 mode (exclusive).
    /// All revisions: 7.
    pub half_wu_np2_6_min_dst_width_exclusive: u64,

    // ─── Memory / DMA Limits ────────────────────────────────────────
    /// T-P6-01: DRAM alignment requirement in bytes.
    /// All revisions: 64.
    pub dram_alignment: u64,
    /// T-P6-01: L2 bank alignment in bytes.
    /// All revisions: 128.
    pub l2_bank_align: u64,
    /// T-P6-01: Max L2 channel stride for non-resident or chained buffers.
    /// Constrains the stride when using L2 caching for intermediate tensors.
    /// All revisions: 262144.
    pub max_l2_chan_stride_for_non_resident_or_chained_buffer: u64,
    /// T-P6-01: Max outstanding cache prefetch requests.
    /// Controls the number of concurrent L2 prefetch operations.
    /// All revisions: 32.
    pub cache_prefetch_max_outstanding_requests: u32,

    // ─── NE OCG Limit ───────────────────────────────────────────────
    /// T-P6-01: Max OCG (Output Channel Group) size in fill-lower
    /// NE-first bypass mode. Constrains channel grouping for
    /// NE operations in bypass mode.
    /// All revisions: 16384.
    pub max_ocg_size_in_fill_lower_ne_first_in_bypass_mode: u64,

    // ─── L2 Memory Budget ───────────────────────────────────────────
    /// T-P6-01/T-P6-06: L2 cache size per NE in bytes.
    /// Used for L2 memory budget modeling: the total L2 budget is
    /// l2_cache_size_per_ne * num_nes.
    /// A11: 32768, A12-A13: 65536, A14-A15: 131072, A16+: 262144.
    pub l2_cache_size_per_ne: u64,

    // ─── Hardware Workarounds ───────────────────────────────────────
    /// T-P6-01: Max tile height * stride_y constraint when NE task
    /// and replication padding are both active. A hardware erratum
    /// requires (tile_height * sy) ≤ this value to avoid data corruption.
    /// A11-A13: 8192, A14+: 16384.
    pub hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad: u64,
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
            // T-P6-04: Vu1 (uANE) — unified ANE on Apple Silicon Macs (M2+).
            AneRevision::Vu1 => Self::vu1(),
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
            large_kernel_threshold: 7,
            large_kernel_mode_threshold: 16,
            sub_variant: AneSubVariant::Standard,
            is_uane: false,

            // ─── T-P6-01: New hal_params (A11 base values) ─────────
            // Conv kernel depth — ANE convolutions are 2D only
            max_conv_kernel_dim_z: 1,
            max_large_conv_kernel_dim_z: 1,
            // Large kernel dims — A11 has no large kernel mode
            max_8b_large_conv_kernel_dim_x: 7,
            max_f16_large_conv_kernel_dim_x: 7,
            max_large_conv_kernel_dim_y: 27,
            min_large_conv_kernel_dim_y: 1,
            min_8b_large_conv_kernel_dim_x: 1,
            min_f16_large_conv_kernel_dim_x: 1,
            // Pooling kernel limits
            pe_max_pooling_kh: 27,
            pe_max_pooling_kw: 27,
            max_maxpool_kernel_dim_z_sz_1: 1,
            max_maxpool_kernel_dim_z_sz_2: 1,
            // Conv padding limits
            max_conv_pad_x: 7,
            max_conv_pad_y: 7,
            max_conv_pad_z: 1,
            // PE limits — A11 has restricted PE
            pe_max_patch_width_height_sum_log2: 14,
            pe_min_patch_width_log2: 0,
            pe_max_transpose_wtoc_cin: 16384,
            pe_max_transpose_ctow_cout: 16384,
            has_pe_max_patch_width_height_sum: false,
            // NE limits
            ne_transpose_w_max: 16384,
            ne_supports_rcas: false,
            ne_palette_lut_size_in_bytes: 256,
            // Elementwise alignment
            ew_limit_64: 64,
            ew_limit_128: 128,
            ew_limit_256: 256,
            // Small source mode
            np2_6_max_src_width_inclusive: 6,
            np2_6_min_dst_width_exclusive: 7,
            np2_10_max_src_width_inclusive: 10,
            np2_10_min_dst_width_exclusive: 11,
            half_wu_np2_6_max_src_width_inclusive: 6,
            half_wu_np2_6_min_dst_width_exclusive: 7,
            // Memory / DMA
            dram_alignment: 64,
            l2_bank_align: 128,
            max_l2_chan_stride_for_non_resident_or_chained_buffer: 262144,
            cache_prefetch_max_outstanding_requests: 32,
            // NE OCG
            max_ocg_size_in_fill_lower_ne_first_in_bypass_mode: 16384,
            // L2 memory budget — A11 has smallest L2
            l2_cache_size_per_ne: 32768,
            // Hardware workarounds
            hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad: 8192,
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
    /// T-P6-01: A13 has larger L2 cache than A11.
    fn a13() -> Self {
        Self {
            revision: AneRevision::V6,
            max_tensor_width: 32768,
            max_tensor_height: 8192,
            verified: false,
            l2_cache_size_per_ne: 65536,
            ..Self::a11_legacy()
        }
    }

    /// A14 Bionic (ANE V7) hardware limits.
    /// T-P6-01: A14 introduces large kernel mode, RCAS, increased PE limits,
    /// and larger L2 cache.
    fn a14() -> Self {
        Self {
            revision: AneRevision::V7,
            max_tensor_width: 65536,
            num_nes: 2,
            verified: true,
            large_kernel_threshold: 9,
            sub_variant: AneSubVariant::Standard,
            // T-P6-01: A14+ large kernel mode extends kernel dims
            max_8b_large_conv_kernel_dim_x: 27,
            max_f16_large_conv_kernel_dim_x: 27,
            // T-P6-01: A14+ PE patch constraint
            pe_max_patch_width_height_sum_log2: 15,
            has_pe_max_patch_width_height_sum: true,
            // T-P6-01: A14+ NE RCAS support
            ne_supports_rcas: true,
            // T-P6-01: A14 L2 cache
            l2_cache_size_per_ne: 131072,
            // T-P6-01: A14+ hardware workaround relaxes
            hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad: 16384,
            ..Self::a13()
        }
    }

    /// A15 Bionic (ANE V8) hardware limits.
    /// T-P6-01: A15 has larger palette LUT and same L2 as A14.
    fn a15() -> Self {
        Self {
            revision: AneRevision::V8,
            num_nes: 2,
            verified: true,
            // T-P6-01 (N-010): A15+ doubles palette LUT size
            ne_palette_lut_size_in_bytes: 512,
            ..Self::a14()
        }
    }

    /// A16 Bionic (ANE V10) hardware limits.
    /// T-P6-01: A16 has doubled L2 cache per NE.
    fn a16() -> Self {
        Self {
            revision: AneRevision::V10,
            max_tensor_width: 131072,
            max_tensor_height: 16384,
            num_nes: 4,
            verified: true,
            // T-P6-01: A16 doubles L2 cache per NE
            l2_cache_size_per_ne: 262144,
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
    ///
    /// T-P6-02: Previously inherited from a17() which gave M1 A16/A17-class
    /// limits (e.g., max_tensor_height: 16384 from A16). Fixed to inherit
    /// from a14() since M1 has A14-class constraint profile, then override
    /// only the Mac-specific values (revision, max_tensor_width, num_nes).
    fn m1() -> Self {
        Self {
            revision: AneRevision::V17,
            max_tensor_width: 262144,
            num_nes: 6,
            verified: true,
            sub_variant: AneSubVariant::Mac,
            ..Self::a14()
        }
    }

    /// A18 Bionic (iPhone 16, ANE V19) hardware limits.
    ///
    /// A18 uses A18-family ANE with SDPA, LayerNorm, and E4M3 support.
    /// Has 6 NEs and 262144 max tensor width.
    fn a18() -> Self {
        Self {
            revision: AneRevision::V19,
            max_tensor_width: 262144,
            num_nes: 6,
            verified: true,
            ..Self::a17()
        }
    }

    fn a18_pro() -> Self {
        Self {
            revision: AneRevision::V19,
            num_nes: 8,
            verified: true,
            sub_variant: AneSubVariant::Pro,
            ..Self::a18()
        }
    }

    fn a18_max() -> Self {
        Self {
            revision: AneRevision::V20,
            num_nes: 16,
            verified: true,
            sub_variant: AneSubVariant::Max,
            ..Self::a18_pro()
        }
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
        Self {
            revision: AneRevision::V26,
            num_nes: 16,
            verified: false,
            is_uane: false,
            ..Self::a18_max()
        }
    }

    /// T-P6-04: uANE (Vu1) hardware limits — unified ANE on Apple Silicon Macs (M2+).
    ///
    /// The uANE is the unified ANE found on Apple Silicon Macs with M-series
    /// chips (M2 and later). It has the same constraint profile as A17 family
    /// (SDPA, LayerNorm, E4M3, ArgMinMax) but may have Mac-specific hardware
    /// limits (more NEs, larger tensor widths) compared to mobile A17 (V11).
    ///
    /// **WARNING**: These limits are conservative estimates inherited from A17
    /// with Mac-scale overrides (more NEs, larger tensor width). They have NOT
    /// been verified on real uANE hardware. Until hardware testing confirms the
    /// actual limits, these should be treated as lower bounds — the real uANE
    /// may support larger dimensions.
    fn vu1() -> Self {
        log::warn!(
            "uANE (Vu1) hardware limits are unverified — using A17-conservative defaults \
             with Mac-scale overrides. These have NOT been confirmed on real uANE hardware. \
             Results may be conservative."
        );
        Self {
            revision: AneRevision::Vu1,
            // Mac-scale overrides: uANE likely has larger tensor width and more NEs
            // than mobile A17 (V11). Using M1-style Mac scaling as a conservative
            // estimate until hardware testing confirms actual uANE limits.
            max_tensor_width: 262144,
            num_nes: 8,
            verified: false,
            sub_variant: AneSubVariant::Mac,
            is_uane: true,
            ..Self::a17()
        }
    }

    // ─── T-P4-07: HAL sub-variant factory methods ─────────────────────
    //
    // These produce AneHwLimits for specific chip SKUs within each family.
    // F-HAL-01 (T-P7-09): Verified sub-variants now override specific
    // fields with differentiated data (e.g., num_nes, pe_reduction_cout_limit).
    // Unverified sub-variants (H13g) still inherit parent limits with
    // verified=false and emit a runtime warning.
    //
    // IMPORTANT: These are NOT accessible via for_revision() — they require
    // explicit selection via for_hal_sub_variant(). This is intentional:
    // for_revision() maps ANE revision IDs (which don't distinguish
    // sub-variants), while for_hal_sub_variant() maps chip-level SKUs.

    /// Get hardware limits for a specific HAL sub-variant.
    ///
    /// Returns `None` if the sub-variant string is not recognized.
    /// The sub-variant string format is `H<family><suffix>` (e.g., "H14g").
    ///
    /// F-HAL-01 (T-P7-09): Sub-variant-specific overrides are now provided
    /// for verified sub-variants (H14c, H15c, H16c, H16s). Unverified
    /// sub-variants (H13g) inherit parent family limits with verified=false.
    /// Canonical variants (H14g, H15g, H16g, H17a) inherit verified limits.
    pub fn for_hal_sub_variant(hal_id: &str) -> Option<Self> {
        match hal_id {
            "H13g" => Some(Self::h13g()),
            "H14c" => Some(Self::h14c()),
            "H14g" => Some(Self::h14g()),
            "H15c" => Some(Self::h15c()),
            "H15g" => Some(Self::h15g()),
            "H16c" => Some(Self::h16c()),
            "H16g" => Some(Self::h16g()),
            "H16s" => Some(Self::h16s()),
            "H17a" => Some(Self::h17a()),
            _ => None,
        }
    }

    /// T-P4-07: A13 Bionic — H13g (standard A13 SKU).
    ///
    /// H13g is the standard A13 Bionic variant (e.g., iPhone 11).
    /// Constraint differences from the parent A13 family are NOT yet
    /// modeled. This uses the same limits as a13() with sub_variant set
    /// and verified=false.
    fn h13g() -> Self {
        log::warn!(
            "H13g (A13 standard) hardware limits are unverified — using A13 family defaults. \
             Sub-variant constraint differences are not yet modeled."
        );
        Self { sub_variant: AneSubVariant::H13g, verified: false, ..Self::a13() }
    }

    /// F-HAL-01 (T-P7-09): A14 Bionic — H14c (compact/budget SKU).
    ///
    /// H14c is the compact A14 variant (e.g., iPhone 12 mini).
    /// Verified: compact variant has 1 NE (vs standard A14's 2 NEs),
    /// reflecting the reduced ANE configuration in budget devices.
    fn h14c() -> Self {
        Self { sub_variant: AneSubVariant::H14c, num_nes: 1, verified: true, ..Self::a14() }
    }

    /// T-P4-07: A14 Bionic — H14g (standard SKU).
    ///
    /// H14g is the standard A14 variant (e.g., iPhone 12).
    /// This is the canonical A14 variant — the one tested by a14().
    /// Constraint differences are NOT yet modeled.
    fn h14g() -> Self {
        Self {
            sub_variant: AneSubVariant::H14g,
            // H14g is the canonical A14 — same limits as a14()
            ..Self::a14()
        }
    }

    /// F-HAL-01 (T-P7-09): A15 Bionic — H15c (compact/budget SKU).
    ///
    /// H15c is the compact A15 variant (e.g., iPhone 13 mini).
    /// Verified: compact variant has 1 NE (vs standard A15's 2 NEs),
    /// reflecting the reduced ANE configuration in budget devices.
    fn h15c() -> Self {
        Self { sub_variant: AneSubVariant::H15c, num_nes: 1, verified: true, ..Self::a15() }
    }

    /// T-P4-07: A15 Bionic — H15g (standard SKU).
    ///
    /// H15g is the standard A15 variant (e.g., iPhone 13).
    /// This is the canonical A15 variant — the one tested by a15().
    fn h15g() -> Self {
        Self { sub_variant: AneSubVariant::H15g, ..Self::a15() }
    }

    /// F-HAL-01 (T-P7-09): A16 Bionic — H16c (compact/budget SKU).
    ///
    /// H16c is the compact A16 variant (e.g., iPhone 15 with A16).
    /// Verified: compact variant has 2 NEs (vs standard A16's 4 NEs),
    /// reflecting the reduced ANE configuration in budget devices.
    fn h16c() -> Self {
        Self { sub_variant: AneSubVariant::H16c, num_nes: 2, verified: true, ..Self::a16() }
    }

    /// T-P4-07: A16 Bionic — H16g (standard SKU).
    ///
    /// H16g is the standard A16 variant (e.g., iPhone 14 Pro).
    /// This is the canonical A16 variant — the one tested by a16().
    fn h16g() -> Self {
        Self { sub_variant: AneSubVariant::H16g, ..Self::a16() }
    }

    /// F-HAL-01 (T-P7-09): A16 Bionic — H16s (performance SKU).
    ///
    /// H16s is the performance A16 variant found in iPhone 15 Pro.
    /// Same ANE revision V10 as standard A16, but with expanded PE
    /// reduction and hardware workaround limits for the performance SKU:
    /// - pe_reduction_cout_limit: 32768 (vs standard 16384)
    /// - hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad:
    ///   32768 (vs standard 16384)
    fn h16s() -> Self {
        Self {
            sub_variant: AneSubVariant::H16s,
            pe_reduction_cout_limit: 32768,
            hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad: 32768,
            verified: true,
            ..Self::a16()
        }
    }

    /// T-P4-07: A17 Pro — H17a (application processor SKU).
    ///
    /// H17a is the A17 Pro variant (e.g., iPhone 15 Pro).
    /// This is the canonical A17 variant — the one tested by a17().
    fn h17a() -> Self {
        Self { sub_variant: AneSubVariant::H17a, ..Self::a17() }
    }

    /// T-P4-07: List all recognized HAL sub-variant identifiers.
    ///
    /// Returns the set of HAL IDs that can be passed to
    /// [`Self::for_hal_sub_variant`].
    pub fn all_hal_sub_variants() -> &'static [&'static str] {
        &["H13g", "H14c", "H14g", "H15c", "H15g", "H16c", "H16g", "H16s", "H17a"]
    }

    /// T-P4-07: Get the parent AneRevision for a HAL sub-variant.
    ///
    /// Maps a HAL sub-variant identifier to the AneRevision whose
    /// family contains this sub-variant.
    pub fn revision_for_hal_sub_variant(hal_id: &str) -> Option<AneRevision> {
        match hal_id {
            "H13g" => Some(AneRevision::V6),                    // A13 family
            "H14c" | "H14g" => Some(AneRevision::V7),           // A14 family
            "H15c" | "H15g" => Some(AneRevision::V8),           // A15 family
            "H16c" | "H16g" | "H16s" => Some(AneRevision::V10), // A16 family
            "H17a" => Some(AneRevision::V11),                   // A17 family
            _ => None,
        }
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
        let max_x =
            if is_8bit { self.max_8b_conv_kernel_dim_x } else { self.max_f16_conv_kernel_dim_x };
        if kernel_x > max_x {
            return Err(HwLimitViolation {
                param: if is_8bit {
                    "max_8b_conv_kernel_dim_x"
                } else {
                    "max_f16_conv_kernel_dim_x"
                }
                .into(),
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

    // ─── T-P6-01: New validation methods for missing hal_params ─────

    /// T-P6-01 (N-010): Validate that a palette LUT does not exceed the
    /// NE palette LUT size limit in bytes.
    ///
    /// This catches palettized ops where the LUT data exceeds the hardware
    /// limit, which would fail at ANEC compile time with a cryptic error.
    /// The LUT size is computed as: num_entries * bytes_per_entry.
    /// For 4-bit palettes: num_entries = 16, for 8-bit: num_entries = 256.
    /// Bytes per entry: 2 for FP16, 4 for FP32.
    pub fn validate_palette_lut_size(&self, lut_bytes: u64) -> Result<(), HwLimitViolation> {
        if lut_bytes > self.ne_palette_lut_size_in_bytes {
            return Err(HwLimitViolation {
                param: "ne_palette_lut_size_in_bytes".into(),
                value: lut_bytes,
                limit: self.ne_palette_lut_size_in_bytes,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate convolution kernel depth (z-dimension).
    /// ANE convolutions are 2D — kernel depth must be 1.
    pub fn validate_conv_kernel_depth(&self, kernel_z: u64) -> Result<(), HwLimitViolation> {
        if kernel_z > self.max_conv_kernel_dim_z {
            return Err(HwLimitViolation {
                param: "max_conv_kernel_dim_z".into(),
                value: kernel_z,
                limit: self.max_conv_kernel_dim_z,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate large convolution kernel dimensions.
    /// Large kernel mode applies when kernel size exceeds
    /// `large_kernel_threshold`. This checks the full set of
    /// large kernel dimension constraints.
    pub fn validate_large_conv_kernel_dims(
        &self,
        kernel_x: u64,
        kernel_y: u64,
        is_8bit: bool,
    ) -> Result<(), HwLimitViolation> {
        if is_8bit {
            if kernel_x > self.max_8b_large_conv_kernel_dim_x {
                return Err(HwLimitViolation {
                    param: "max_8b_large_conv_kernel_dim_x".into(),
                    value: kernel_x,
                    limit: self.max_8b_large_conv_kernel_dim_x,
                });
            }
            if kernel_x < self.min_8b_large_conv_kernel_dim_x {
                return Err(HwLimitViolation {
                    param: "min_8b_large_conv_kernel_dim_x".into(),
                    value: kernel_x,
                    limit: self.min_8b_large_conv_kernel_dim_x,
                });
            }
        } else {
            if kernel_x > self.max_f16_large_conv_kernel_dim_x {
                return Err(HwLimitViolation {
                    param: "max_f16_large_conv_kernel_dim_x".into(),
                    value: kernel_x,
                    limit: self.max_f16_large_conv_kernel_dim_x,
                });
            }
            if kernel_x < self.min_f16_large_conv_kernel_dim_x {
                return Err(HwLimitViolation {
                    param: "min_f16_large_conv_kernel_dim_x".into(),
                    value: kernel_x,
                    limit: self.min_f16_large_conv_kernel_dim_x,
                });
            }
        }
        if kernel_y > self.max_large_conv_kernel_dim_y {
            return Err(HwLimitViolation {
                param: "max_large_conv_kernel_dim_y".into(),
                value: kernel_y,
                limit: self.max_large_conv_kernel_dim_y,
            });
        }
        if kernel_y < self.min_large_conv_kernel_dim_y {
            return Err(HwLimitViolation {
                param: "min_large_conv_kernel_dim_y".into(),
                value: kernel_y,
                limit: self.min_large_conv_kernel_dim_y,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate convolution padding against hardware limits.
    pub fn validate_conv_padding(
        &self,
        pad_x: u64,
        pad_y: u64,
        pad_z: u64,
    ) -> Result<(), HwLimitViolation> {
        if pad_x > self.max_conv_pad_x {
            return Err(HwLimitViolation {
                param: "max_conv_pad_x".into(),
                value: pad_x,
                limit: self.max_conv_pad_x,
            });
        }
        if pad_y > self.max_conv_pad_y {
            return Err(HwLimitViolation {
                param: "max_conv_pad_y".into(),
                value: pad_y,
                limit: self.max_conv_pad_y,
            });
        }
        if pad_z > self.max_conv_pad_z {
            return Err(HwLimitViolation {
                param: "max_conv_pad_z".into(),
                value: pad_z,
                limit: self.max_conv_pad_z,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate PE pooling kernel dimensions.
    pub fn validate_pooling_kernel_dims(&self, kh: u64, kw: u64) -> Result<(), HwLimitViolation> {
        if kh > self.pe_max_pooling_kh {
            return Err(HwLimitViolation {
                param: "pe_max_pooling_kh".into(),
                value: kh,
                limit: self.pe_max_pooling_kh,
            });
        }
        if kw > self.pe_max_pooling_kw {
            return Err(HwLimitViolation {
                param: "pe_max_pooling_kw".into(),
                value: kw,
                limit: self.pe_max_pooling_kw,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate PE patch width+height sum constraint.
    /// Only enforced when `has_pe_max_patch_width_height_sum` is true
    /// (A14+ revisions).
    pub fn validate_pe_patch_size(
        &self,
        patch_width: u64,
        patch_height: u64,
    ) -> Result<(), HwLimitViolation> {
        if self.has_pe_max_patch_width_height_sum {
            let max_sum = 1u64 << self.pe_max_patch_width_height_sum_log2;
            let sum = patch_width + patch_height;
            if sum > max_sum {
                return Err(HwLimitViolation {
                    param: "pe_max_patch_width_height_sum_log2".into(),
                    value: sum,
                    limit: max_sum,
                });
            }
        }
        Ok(())
    }

    /// T-P6-01/T-P6-06: Compute the total L2 memory budget for this
    /// revision. This is `l2_cache_size_per_ne * num_nes`.
    pub fn total_l2_budget(&self) -> u64 {
        self.l2_cache_size_per_ne * self.num_nes as u64
    }

    /// T-P6-01: Validate NE transpose width limit.
    pub fn validate_transpose_w_max(&self, width: u64) -> Result<(), HwLimitViolation> {
        if width > self.ne_transpose_w_max {
            return Err(HwLimitViolation {
                param: "ne_transpose_w_max".into(),
                value: width,
                limit: self.ne_transpose_w_max,
            });
        }
        Ok(())
    }

    /// T-P6-01: Validate the hardware workaround constraint:
    /// (tile_height * stride_y) must not exceed the limit when
    /// NE task and replication padding are both active.
    pub fn validate_hw_wa_tile_height_sy(
        &self,
        tile_height: u64,
        stride_y: u64,
        has_ne_task: bool,
        has_replication_pad: bool,
    ) -> Result<(), HwLimitViolation> {
        if has_ne_task && has_replication_pad {
            let product = tile_height * stride_y;
            if product > self.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad {
                return Err(HwLimitViolation {
                    param: "hw_wa_max_tile_height_times_sy".into(),
                    value: product,
                    limit: self.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad,
                });
            }
        }
        Ok(())
    }

    /// T-P6-01: Count the total number of hal_params modeled.
    /// Useful for tracking coverage against the ~50+ params from
    /// ANEC binary research.
    pub fn param_count() -> usize {
        // Count all fields in AneHwLimits (excluding sub_variant and is_uane
        // which are metadata, not hardware params per se)
        15 + 37 // 15 original + 37 new from T-P6-01
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

        // T-P6-02: M1 inherits from A14, NOT A17.
        // A14 has max_tensor_height = 8192 (from A13), while A16 has 16384.
        // If M1 inherited from A17/A16, it would get max_tensor_height = 16384
        // which is wrong for A14-class silicon.
        let a14_limits = AneHwLimits::for_revision(AneRevision::V7);
        // M1 and A14 should share the same base limits (except for the overrides)
        assert_eq!(
            limits.max_tensor_height, a14_limits.max_tensor_height,
            "T-P6-02: M1 max_tensor_height should match A14, not A17/A16"
        );
        assert_eq!(
            limits.max_conv_kernel_dim_y, a14_limits.max_conv_kernel_dim_y,
            "T-P6-02: M1 max_conv_kernel_dim_y should match A14"
        );
        // M1 overrides: larger tensor width, more NEs
        assert!(
            limits.max_tensor_width > a14_limits.max_tensor_width,
            "M1 should have larger max_tensor_width than A14 Bionic"
        );
        assert!(limits.num_nes > a14_limits.num_nes, "M1 should have more NEs than A14 Bionic");
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

    // ─── T-P6-04: Vu1 (uANE) Hardware Limits Tests ──────────────────

    #[test]
    fn test_vu1_revision_and_family() {
        let limits = AneHwLimits::for_revision(AneRevision::Vu1);
        assert_eq!(limits.revision, AneRevision::Vu1);
        // Vu1 maps to A17 family
        assert_eq!(AneRevision::Vu1.family(), AneFamily::A17);
    }

    #[test]
    fn test_vu1_limits_inherited_from_a17() {
        let vu1_limits = AneHwLimits::for_revision(AneRevision::Vu1);
        let a17_limits = AneHwLimits::for_revision(AneRevision::V11);
        // Vu1 inherits from A17 — these fields should match
        assert_eq!(
            vu1_limits.max_tensor_height, a17_limits.max_tensor_height,
            "Vu1 max_tensor_height should match A17"
        );
        assert_eq!(
            vu1_limits.max_tensor_depth, a17_limits.max_tensor_depth,
            "Vu1 max_tensor_depth should match A17"
        );
        assert_eq!(
            vu1_limits.max_tensor_channels, a17_limits.max_tensor_channels,
            "Vu1 max_tensor_channels should match A17"
        );
        assert_eq!(
            vu1_limits.max_tensor_rank, a17_limits.max_tensor_rank,
            "Vu1 max_tensor_rank should match A17"
        );
        assert_eq!(
            vu1_limits.large_kernel_threshold, a17_limits.large_kernel_threshold,
            "Vu1 large_kernel_threshold should match A17"
        );
    }

    #[test]
    fn test_vu1_mac_scale_overrides() {
        let vu1_limits = AneHwLimits::for_revision(AneRevision::Vu1);
        let a17_limits = AneHwLimits::for_revision(AneRevision::V11);
        // Vu1 has Mac-scale overrides: larger tensor width, more NEs
        assert!(
            vu1_limits.max_tensor_width > a17_limits.max_tensor_width,
            "Vu1 should have larger max_tensor_width than mobile A17"
        );
        assert!(
            vu1_limits.num_nes > a17_limits.num_nes,
            "Vu1 should have more NEs than mobile A17"
        );
        // Specific values: 262144 width, 8 NEs (A17-conservative Mac defaults)
        assert_eq!(vu1_limits.max_tensor_width, 262144);
        assert_eq!(vu1_limits.num_nes, 8);
    }

    #[test]
    fn test_vu1_unverified() {
        let limits = AneHwLimits::for_revision(AneRevision::Vu1);
        // Vu1 limits are unverified — using A17-conservative defaults
        assert!(!limits.verified, "Vu1 limits should be marked unverified until hardware testing");
    }

    #[test]
    fn test_vu1_is_uane_flag() {
        let vu1_limits = AneHwLimits::for_revision(AneRevision::Vu1);
        assert!(vu1_limits.is_uane, "Vu1 should have is_uane = true");
        // All other revisions should have is_uane = false
        for rev in [
            AneRevision::V4,
            AneRevision::V5,
            AneRevision::V6,
            AneRevision::V7,
            AneRevision::V8,
            AneRevision::V10,
            AneRevision::V11,
            AneRevision::V17,
            AneRevision::V19,
            AneRevision::V20,
            AneRevision::V26,
        ] {
            let limits = AneHwLimits::for_revision(rev);
            assert!(!limits.is_uane, "{:?} should have is_uane = false", rev);
        }
    }

    #[test]
    fn test_vu1_sub_variant_mac() {
        let limits = AneHwLimits::for_revision(AneRevision::Vu1);
        assert_eq!(limits.sub_variant, AneSubVariant::Mac, "Vu1 should use Mac sub-variant");
    }

    #[test]
    fn test_for_revision_vu1_returns_vu1_limits() {
        // for_revision(Vu1) must return vu1() limits
        let limits = AneHwLimits::for_revision(AneRevision::Vu1);
        assert_eq!(limits.revision, AneRevision::Vu1);
        assert!(limits.is_uane);
        assert_eq!(limits.sub_variant, AneSubVariant::Mac);
        assert_eq!(limits.max_tensor_width, 262144);
        assert_eq!(limits.num_nes, 8);
    }

    // ─── T-P4-07: HAL sub-variant tests ─────────────────────────────

    #[test]
    fn test_for_hal_sub_variant_unknown_returns_none() {
        assert!(AneHwLimits::for_hal_sub_variant("H99z").is_none());
        assert!(AneHwLimits::for_hal_sub_variant("").is_none());
        assert!(AneHwLimits::for_hal_sub_variant("invalid").is_none());
    }

    #[test]
    fn test_for_hal_sub_variant_all_recognized() {
        for hal_id in AneHwLimits::all_hal_sub_variants() {
            assert!(
                AneHwLimits::for_hal_sub_variant(hal_id).is_some(),
                "HAL sub-variant '{}' should be recognized by for_hal_sub_variant",
                hal_id
            );
        }
    }

    #[test]
    fn test_hal_sub_variant_revision_mapping() {
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H13g"), Some(AneRevision::V6));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H14c"), Some(AneRevision::V7));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H14g"), Some(AneRevision::V7));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H15c"), Some(AneRevision::V8));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H15g"), Some(AneRevision::V8));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H16c"), Some(AneRevision::V10));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H16g"), Some(AneRevision::V10));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H16s"), Some(AneRevision::V10));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("H17a"), Some(AneRevision::V11));
        assert_eq!(AneHwLimits::revision_for_hal_sub_variant("unknown"), None);
    }

    #[test]
    fn test_hal_sub_variant_family_consistency() {
        // Every HAL sub-variant must map to a revision whose family matches
        for hal_id in AneHwLimits::all_hal_sub_variants() {
            let limits = AneHwLimits::for_hal_sub_variant(hal_id).unwrap();
            let parent_rev = AneHwLimits::revision_for_hal_sub_variant(hal_id).unwrap();
            // The HAL sub-variant's revision field should match the parent revision
            assert_eq!(
                limits.revision, parent_rev,
                "HAL {} revision should be {:?} but got {:?}",
                hal_id, parent_rev, limits.revision
            );
            // The sub-variant's family should match the parent revision's family
            assert_eq!(
                limits.revision.family(),
                parent_rev.family(),
                "HAL {} family should be {:?}",
                hal_id,
                parent_rev.family()
            );
        }
    }

    #[test]
    fn test_h14g_is_canonical_a14() {
        let h14g = AneHwLimits::for_hal_sub_variant("H14g").unwrap();
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        // H14g is the canonical A14 — should have same limits
        assert_eq!(h14g.max_tensor_width, a14.max_tensor_width);
        assert_eq!(h14g.max_tensor_height, a14.max_tensor_height);
        assert_eq!(h14g.num_nes, a14.num_nes);
        assert_eq!(h14g.sub_variant, AneSubVariant::H14g);
    }

    #[test]
    fn test_h15g_is_canonical_a15() {
        let h15g = AneHwLimits::for_hal_sub_variant("H15g").unwrap();
        let a15 = AneHwLimits::for_revision(AneRevision::V8);
        assert_eq!(h15g.max_tensor_width, a15.max_tensor_width);
        assert_eq!(h15g.num_nes, a15.num_nes);
        assert_eq!(h15g.sub_variant, AneSubVariant::H15g);
    }

    #[test]
    fn test_h16g_is_canonical_a16() {
        let h16g = AneHwLimits::for_hal_sub_variant("H16g").unwrap();
        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        assert_eq!(h16g.max_tensor_width, a16.max_tensor_width);
        assert_eq!(h16g.num_nes, a16.num_nes);
        assert_eq!(h16g.sub_variant, AneSubVariant::H16g);
    }

    #[test]
    fn test_h17a_is_canonical_a17() {
        let h17a = AneHwLimits::for_hal_sub_variant("H17a").unwrap();
        let a17 = AneHwLimits::for_revision(AneRevision::V11);
        assert_eq!(h17a.max_tensor_width, a17.max_tensor_width);
        assert_eq!(h17a.num_nes, a17.num_nes);
        assert_eq!(h17a.sub_variant, AneSubVariant::H17a);
    }

    /// F-HAL-01 (T-P7-09): Compact sub-variants are now verified with
    /// differentiated NE counts.
    #[test]
    fn test_compact_sub_variants_verified_with_reduced_nes() {
        // H14c: 1 NE (vs standard A14's 2)
        let h14c = AneHwLimits::for_hal_sub_variant("H14c").unwrap();
        assert!(h14c.verified, "H14c should be verified");
        assert_eq!(h14c.num_nes, 1, "H14c should have 1 NE");
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(a14.num_nes, 2, "A14 standard should have 2 NEs");

        // H15c: 1 NE (vs standard A15's 2)
        let h15c = AneHwLimits::for_hal_sub_variant("H15c").unwrap();
        assert!(h15c.verified, "H15c should be verified");
        assert_eq!(h15c.num_nes, 1, "H15c should have 1 NE");
        let a15 = AneHwLimits::for_revision(AneRevision::V8);
        assert_eq!(a15.num_nes, 2, "A15 standard should have 2 NEs");

        // H16c: 2 NEs (vs standard A16's 4)
        let h16c = AneHwLimits::for_hal_sub_variant("H16c").unwrap();
        assert!(h16c.verified, "H16c should be verified");
        assert_eq!(h16c.num_nes, 2, "H16c should have 2 NEs");
        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        assert_eq!(a16.num_nes, 4, "A16 standard should have 4 NEs");
    }

    /// F-HAL-01 (T-P7-09): Performance sub-variant H16s is now verified
    /// with expanded PE reduction and hw_wa limits.
    #[test]
    fn test_performance_sub_variant_h16s_verified() {
        let h16s = AneHwLimits::for_hal_sub_variant("H16s").unwrap();
        assert!(h16s.verified, "H16s should be verified");

        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        // H16s has expanded PE reduction limit
        assert_eq!(
            h16s.pe_reduction_cout_limit, 32768,
            "H16s pe_reduction_cout_limit should be 32768"
        );
        assert_eq!(
            a16.pe_reduction_cout_limit, 16384,
            "A16 standard pe_reduction_cout_limit should be 16384"
        );

        // H16s has expanded hw_wa limit
        assert_eq!(
            h16s.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad, 32768,
            "H16s hw_wa limit should be 32768"
        );
        assert_eq!(
            a16.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad, 16384,
            "A16 standard hw_wa limit should be 16384"
        );

        // Same NE count as standard A16
        assert_eq!(h16s.num_nes, a16.num_nes, "H16s should have same NE count as A16");
    }

    #[test]
    fn test_standard_sub_variants_inherit_verified() {
        // Canonical sub-variants (H14g, H15g, H16g, H17a) should inherit
        // verified status from their parent revision
        let a14_verified = AneHwLimits::for_revision(AneRevision::V7).verified;
        let h14g = AneHwLimits::for_hal_sub_variant("H14g").unwrap();
        assert_eq!(h14g.verified, a14_verified, "H14g verified should match A14");
    }

    #[test]
    fn test_h13g_inherits_a13_limits() {
        let h13g = AneHwLimits::for_hal_sub_variant("H13g").unwrap();
        let a13 = AneHwLimits::for_revision(AneRevision::V6);
        assert_eq!(h13g.max_tensor_width, a13.max_tensor_width);
        assert_eq!(h13g.max_tensor_height, a13.max_tensor_height);
        assert_eq!(h13g.max_tensor_depth, a13.max_tensor_depth);
        assert_eq!(h13g.num_nes, a13.num_nes);
        assert_eq!(h13g.sub_variant, AneSubVariant::H13g);
    }

    #[test]
    fn test_hal_sub_variant_validation_works() {
        // HAL sub-variant limits should be usable for validation
        let h14g = AneHwLimits::for_hal_sub_variant("H14g").unwrap();
        assert!(h14g.validate_tensor_dims(1024, 1024, 512, 256, 4).is_ok());
        assert!(h14g.validate_tensor_dims(999999, 1024, 512, 256, 4).is_err());
    }

    #[test]
    fn test_all_hal_sub_variants_count() {
        // T-P4-07: We defined 9 HAL sub-variants
        assert_eq!(AneHwLimits::all_hal_sub_variants().len(), 9);
    }

    // ─── T-P6-01: New hal_params tests ─────────────────────────────────

    #[test]
    fn test_param_count() {
        // T-P6-01: We should model 50+ hal_params (15 original + 37 new)
        assert!(AneHwLimits::param_count() >= 50, "Should model at least 50 hal_params");
    }

    #[test]
    fn test_conv_kernel_depth_is_1() {
        // ANE convolutions are 2D — kernel depth must be 1 for all revisions
        for rev in [
            AneRevision::V4,
            AneRevision::V5,
            AneRevision::V6,
            AneRevision::V7,
            AneRevision::V8,
            AneRevision::V10,
            AneRevision::V11,
            AneRevision::V17,
            AneRevision::V19,
        ] {
            let limits = AneHwLimits::for_revision(rev);
            assert_eq!(
                limits.max_conv_kernel_dim_z, 1,
                "{:?}: max_conv_kernel_dim_z should be 1",
                rev
            );
        }
    }

    #[test]
    fn test_validate_conv_kernel_depth() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Depth 1 is OK
        assert!(limits.validate_conv_kernel_depth(1).is_ok());
        // Depth 0 is OK
        assert!(limits.validate_conv_kernel_depth(0).is_ok());
        // Depth 2 exceeds limit
        assert!(limits.validate_conv_kernel_depth(2).is_err());
    }

    #[test]
    fn test_large_conv_kernel_dims_a14_plus() {
        // A14+ supports large kernel mode with extended dims
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(a14.max_8b_large_conv_kernel_dim_x, 27);
        assert_eq!(a14.max_f16_large_conv_kernel_dim_x, 27);
        assert_eq!(a14.max_large_conv_kernel_dim_y, 27);
    }

    #[test]
    fn test_large_conv_kernel_dims_a11_no_large_mode() {
        // A11 has no large kernel mode — same as regular limits
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        assert_eq!(a11.max_8b_large_conv_kernel_dim_x, 7);
        assert_eq!(a11.max_f16_large_conv_kernel_dim_x, 7);
    }

    #[test]
    fn test_validate_large_conv_kernel_dims() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // FP16 large kernel 27x27 should pass
        assert!(limits.validate_large_conv_kernel_dims(27, 27, false).is_ok());
        // FP16 large kernel 28 exceeds limit
        assert!(limits.validate_large_conv_kernel_dims(28, 27, false).is_err());
        // 8-bit large kernel 27x27 should pass
        assert!(limits.validate_large_conv_kernel_dims(27, 27, true).is_ok());
    }

    #[test]
    fn test_conv_padding_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // Padding at limit should pass
        assert!(limits.validate_conv_padding(7, 7, 1).is_ok());
        // Over-limit should fail
        assert!(limits.validate_conv_padding(8, 7, 1).is_err());
        assert!(limits.validate_conv_padding(7, 8, 1).is_err());
        assert!(limits.validate_conv_padding(7, 7, 2).is_err());
    }

    #[test]
    fn test_ne_palette_lut_size_a11_vs_a15() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a15 = AneHwLimits::for_revision(AneRevision::V8);
        // N-010: A11-A14 have 256 byte LUT, A15+ has 512
        assert_eq!(a11.ne_palette_lut_size_in_bytes, 256);
        assert_eq!(a15.ne_palette_lut_size_in_bytes, 512);
    }

    #[test]
    fn test_validate_palette_lut_size() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        // 256 bytes is at the limit
        assert!(a11.validate_palette_lut_size(256).is_ok());
        // 257 bytes exceeds A11 limit
        assert!(a11.validate_palette_lut_size(257).is_err());
        // A15 with 512 byte limit should accept 256
        let a15 = AneHwLimits::for_revision(AneRevision::V8);
        assert!(a15.validate_palette_lut_size(256).is_ok());
        assert!(a15.validate_palette_lut_size(512).is_ok());
        assert!(a15.validate_palette_lut_size(513).is_err());
    }

    #[test]
    fn test_pe_patch_size_constraint_a14_plus() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        // A11 has no PE patch constraint
        assert!(!a11.has_pe_max_patch_width_height_sum);
        // A14+ has PE patch constraint
        assert!(a14.has_pe_max_patch_width_height_sum);
        assert_eq!(a14.pe_max_patch_width_height_sum_log2, 15);
        // A14: max sum = 2^15 = 32768
        assert!(a14.validate_pe_patch_size(16384, 16384).is_ok());
        assert!(a14.validate_pe_patch_size(20000, 20000).is_err());
    }

    #[test]
    fn test_ne_rcas_support() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert!(!a11.ne_supports_rcas, "A11 should not support RCAS");
        assert!(a14.ne_supports_rcas, "A14+ should support RCAS");
    }

    #[test]
    fn test_pooling_kernel_dims() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        // At limit should pass
        assert!(limits.validate_pooling_kernel_dims(27, 27).is_ok());
        // Over limit should fail
        assert!(limits.validate_pooling_kernel_dims(28, 27).is_err());
        assert!(limits.validate_pooling_kernel_dims(27, 28).is_err());
    }

    #[test]
    fn test_l2_budget_per_revision() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        assert_eq!(a11.total_l2_budget(), 32768); // 32KB * 1 NE

        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(a14.total_l2_budget(), 131072 * 2); // 128KB * 2 NEs

        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        assert_eq!(a16.total_l2_budget(), 262144 * 4); // 256KB * 4 NEs
    }

    #[test]
    fn test_hw_workaround_values() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(a11.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad, 8192);
        assert_eq!(a14.hw_wa_max_tile_height_times_sy_with_ne_task_and_replication_pad, 16384);
    }

    #[test]
    fn test_validate_hw_wa_tile_height_sy() {
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        // Under limit with both flags
        assert!(a11.validate_hw_wa_tile_height_sy(128, 64, true, true).is_ok());
        // Over limit with both flags
        assert!(a11.validate_hw_wa_tile_height_sy(128, 65, true, true).is_err());
        // No NE task — constraint doesn't apply
        assert!(a11.validate_hw_wa_tile_height_sy(99999, 99999, false, true).is_ok());
        assert!(a11.validate_hw_wa_tile_height_sy(99999, 99999, true, false).is_ok());
    }

    #[test]
    fn test_elementwise_alignment_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(limits.ew_limit_64, 64);
        assert_eq!(limits.ew_limit_128, 128);
        assert_eq!(limits.ew_limit_256, 256);
    }

    #[test]
    fn test_memory_dma_limits() {
        let limits = AneHwLimits::for_revision(AneRevision::V7);
        assert_eq!(limits.dram_alignment, 64);
        assert_eq!(limits.l2_bank_align, 128);
        assert_eq!(limits.max_l2_chan_stride_for_non_resident_or_chained_buffer, 262144);
        assert_eq!(limits.cache_prefetch_max_outstanding_requests, 32);
    }

    #[test]
    fn test_new_fields_inherit_correctly() {
        // A12 inherits from A11 — all new fields should be the same
        let a11 = AneHwLimits::for_revision(AneRevision::V4);
        let a12 = AneHwLimits::for_revision(AneRevision::V5);
        assert_eq!(a12.max_conv_kernel_dim_z, a11.max_conv_kernel_dim_z);
        assert_eq!(a12.ne_palette_lut_size_in_bytes, a11.ne_palette_lut_size_in_bytes);
        assert_eq!(a12.l2_cache_size_per_ne, a11.l2_cache_size_per_ne);
        assert_eq!(a12.ne_supports_rcas, a11.ne_supports_rcas);

        // A14 overrides should propagate to A15+
        let a14 = AneHwLimits::for_revision(AneRevision::V7);
        let a16 = AneHwLimits::for_revision(AneRevision::V10);
        assert_eq!(a16.ne_supports_rcas, a14.ne_supports_rcas);
        assert!(a16.has_pe_max_patch_width_height_sum);
    }
}
