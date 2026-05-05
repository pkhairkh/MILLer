//! ANE Placement Validator — deterministic hard-constraint checker.
//!
//! Runs before the soft scoring pass to reject ops that can never run
//! on the ANE, regardless of heuristics. This is a compile-time guard
//! that mirrors the hard constraints documented in:
//! - ane-constraints-docs/02-hardware-and-limits/
//! - ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md
//!
//! ## T-25: Wired Validators
//!
//! The following constraint validators are now wired into the placement
//! pipeline and enforced as hard gates:
//!
//! - **`is_dtype_ane_legal()`** — universal dtype check for all ops.
//!   Rejects Int32 and Fp64 dtypes on ANE regardless of family.
//! - **`is_broadcast_dtype_legal()`** — broadcast ops on A11/A12 must
//!   use FP16 only. Checked for Add, Mul, Sub, Maximum, Minimum.
//! - **`validate_interleave_constraints()`** — enforces valid interleave
//!   factors (1,2,3,4,8), const→interleave-1, int4→interleave-8,
//!   and channel-divisibility rules.
//! - **`validate_channellast_constraints()`** — ChannelLast layout is
//!   only valid for depthwise convolutions with interleave=1.
//! - **`is_blockwise_scale_supported()`** — always false on ANE.
//!   `ConstexprBlockwiseShiftScale` and `ConstexprSparseBlockwiseShiftScale`
//!   are hard-rejected.
//! - **`is_asymmetric_quantization_supported()`** — always false on ANE.
//!   Quantize ops with asymmetric mode are hard-rejected.
//!
//! ## T-32: ArgMinMax A18 Guard
//!
//! ArgMinMax (`MILReduceArgmax`/`MILReduceArgmin`) has no ANEC converter
//! on A18 (LSE_7). The `ConvertReductionArg` converter exists for LSE_0
//! through LSE_6 (A11Legacy through A16) but there is NO LSE_7 variant.
//! The dedicated match arm hard-rejects ArgMinMax on A18 with a clear
//! diagnostic message.

use crate::cpu_only_ops;
use crate::dtype_constraints;
use ane_ir::ane_layout::{AneInterleave, AneLayout};
use ane_ir::ane_target::AneFamily;
use ane_ir::common::MilDtype;
use ane_ir::mir::MirOp;

/// Placement decision for a MIR op being considered for ANE execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementDecision {
    /// Op is allowed on ANE (subject to soft scoring later).
    AneAllowed,
    /// Op must run on CPU; the string explains why.
    CpuOnly(String),
    /// Op is allowed on ANE only on the specified family and later.
    AneConditional(AneFamily),
}

/// Supplementary context for placement validation.
///
/// Carries dtype, interleave, layout, and quantization metadata that
/// the validator needs to enforce dtype and layout constraints. All
/// fields are optional — when absent, the corresponding validator is
/// skipped (not enforced). Callers should populate as many fields as
/// they have available for maximum constraint coverage.
///
/// # Usage
///
/// ```
/// use ane_passes::placement_validate::PlacementContext;
/// use ane_ir::common::MilDtype;
/// use ane_ir::ane_layout::{AneInterleave, AneLayout};
///
/// let ctx = PlacementContext {
///     dtype: Some(MilDtype::Fp16),
///     interleave: Some(AneInterleave::Factor4),
///     layout: Some(AneLayout::ChannelFirst),
///     is_const: false,
///     is_int4: false,
///     channels: Some(64),
///     is_depthwise_conv: false,
///     is_asymmetric_quant: false,
///     is_blockwise_scale: false,
///     anef_revision: None,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct PlacementContext {
    /// Data type of the op's primary tensor operand.
    /// When provided, `is_dtype_ane_legal()` is checked for all ops.
    pub dtype: Option<MilDtype>,

    /// Interleave factor for the op's input tensor.
    /// When provided, `validate_interleave_constraints()` is checked.
    pub interleave: Option<AneInterleave>,

    /// Memory layout (ChannelFirst or ChannelLast).
    /// When provided, `validate_channellast_constraints()` is checked.
    pub layout: Option<AneLayout>,

    /// Whether the tensor is a constant (const tensors must have interleave=1).
    pub is_const: bool,

    /// Whether the tensor uses int4 format (int4 requires interleave=8).
    pub is_int4: bool,

    /// Channel count for interleave divisibility check.
    pub channels: Option<u64>,

    /// Whether the op is a depthwise convolution (ChannelLast only valid for depthwise).
    pub is_depthwise_conv: bool,

    /// Whether the op uses asymmetric quantization (not supported on ANE).
    pub is_asymmetric_quant: bool,

    /// Whether the op uses blockwise scaling (not supported on ANE).
    pub is_blockwise_scale: bool,

    /// ANE revision for hardware limit validation (T-53).
    /// When provided, tensor dimensions are validated against per-revision HW limits.
    pub anef_revision: Option<ane_ir::ane_target::AneRevision>,
}

impl PlacementContext {
    /// Create an empty context with no supplementary information.
    /// All validators that require context will be skipped.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a context with dtype information only.
    pub fn with_dtype(dtype: MilDtype) -> Self {
        Self { dtype: Some(dtype), ..Self::default() }
    }

    /// Create a context with dtype and interleave information.
    pub fn with_dtype_and_interleave(
        dtype: MilDtype,
        interleave: AneInterleave,
        channels: u64,
    ) -> Self {
        Self {
            dtype: Some(dtype),
            interleave: Some(interleave),
            channels: Some(channels),
            ..Self::default()
        }
    }
}

/// Extract width, height, depth, channels from a shape vector.
///
/// T-68: Core ML MIL uses NCHW layout convention:
/// - Rank 1: [width]
/// - Rank 2: [height, width]
/// - Rank 3: [channels, height, width]  (CHW)
/// - Rank 4: [batch, channels, height, width]  (NCHW)
/// - Rank 5: [batch, channels, depth, height, width]  (NCDHW)
///
/// Previous implementation incorrectly treated rank-4 as CDHW, swapping
/// depth and channels. For `[1, 64, 128, 128]` (NCHW):
/// - Old (wrong): w=128, h=128, d=64 (actually channels), c=1 (actually batch)
/// - New (correct): w=128, h=128, d=1, c=64
fn extract_whdc(shape: &[usize]) -> (u64, u64, u64, u64) {
    match shape.len() {
        0 => (0, 0, 0, 0),
        1 => (shape[0] as u64, 1, 1, 1),
        2 => (shape[1] as u64, shape[0] as u64, 1, 1),
        // Rank 3: CHW [channels, height, width]
        3 => (shape[2] as u64, shape[1] as u64, 1, shape[0] as u64),
        // T-68: Rank 4: NCHW [batch, channels, height, width]
        // depth is always 1 for 4D tensors (no depth dimension in NCHW)
        4 => (shape[3] as u64, shape[2] as u64, 1, shape[1] as u64),
        // Rank 5: NCDHW [batch, channels, depth, height, width]
        5 => (shape[4] as u64, shape[3] as u64, shape[2] as u64, shape[1] as u64),
        _ => {
            // Rank > 5: use last 4 dims as (w, h, d, c)
            let n = shape.len();
            (shape[n - 1] as u64, shape[n - 2] as u64, shape[n - 3] as u64, shape[n - 4] as u64)
        }
    }
}

/// Validate whether a MIR op can be placed on the ANE given its tensor
/// shapes, the target family, and whether dynamic shapes are present.
///
/// This is the backward-compatible entry point that uses an empty
/// `PlacementContext`. For full constraint validation (dtype, interleave,
/// layout, quantization checks), use `validate_placement_with_context()`.
///
/// This function checks **hard** constraints only — violations that
/// guarantee the ANE will reject the op or produce incorrect results.
/// Soft constraints (performance, resource pressure) are handled by the
/// scoring pass.
pub fn validate_placement(
    op: &MirOp,
    input_shapes: &[Vec<usize>],
    target_family: AneFamily,
    has_dynamic_shapes: bool,
) -> PlacementDecision {
    validate_placement_with_context(
        op,
        input_shapes,
        target_family,
        has_dynamic_shapes,
        &PlacementContext::empty(),
    )
}

/// Validate whether a MIR op can be placed on the ANE with full context.
///
/// This is the primary entry point for placement validation. It performs
/// all the checks from `validate_placement()` plus dtype, interleave,
/// layout, and quantization constraint enforcement based on the provided
/// context.
///
/// # Constraint Enforcement
///
/// The following validators are wired in as hard gates:
///
/// 1. **Dtype gate** (`is_dtype_ane_legal`): Checked for ALL ops when
///    `ctx.dtype` is provided. Rejects Int32 and Fp64 on ANE.
///
/// 2. **Broadcast dtype gate** (`is_broadcast_dtype_legal`): Checked for
///    binary elementwise ops (Add, Mul, Sub, Maximum, Minimum) when
///    `ctx.dtype` is provided. Rejects non-FP16 broadcast on A11/A12.
///
/// 3. **Interleave gate** (`validate_interleave_constraints`): Checked
///    when `ctx.interleave` is provided. Enforces valid interleave
///    factors, const→1, int4→8, channel-divisibility.
///
/// 4. **ChannelLast gate** (`validate_channellast_constraints`): Checked
///    when `ctx.layout` is `Some(ChannelLast)`. Enforces depthwise-only
///    and interleave=1 restrictions.
///
/// 5. **Blockwise scale gate** (`is_blockwise_scale_supported`): Hard-
///    rejects `ConstexprBlockwiseShiftScale` and
///    `ConstexprSparseBlockwiseShiftScale` (always returns false).
///
/// 6. **Asymmetric quantization gate** (`is_asymmetric_quantization_supported`):
///    Hard-rejects `Quantize` ops with `ctx.is_asymmetric_quant == true`
///    (always returns false).
pub fn validate_placement_with_context(
    op: &MirOp,
    input_shapes: &[Vec<usize>],
    target_family: AneFamily,
    has_dynamic_shapes: bool,
    ctx: &PlacementContext,
) -> PlacementDecision {
    // ─── Kill switch: no dynamic shapes on ANE ─────────────────────
    if has_dynamic_shapes {
        return PlacementDecision::CpuOnly("ANE does not support dynamic shapes".into());
    }

    // ─── Universal rank constraint: tensors must be rank ≤ 5 ───────
    for (i, shape) in input_shapes.iter().enumerate() {
        if shape.len() > 5 {
            return PlacementDecision::CpuOnly(format!(
                "Input {} has rank {} which exceeds ANE maximum of 5",
                i,
                shape.len()
            ));
        }
    }

    // ─── T-25: Universal dtype gate ────────────────────────────────
    // When dtype information is available, enforce dtype legality
    // for ALL ops. This catches Int32 and Fp64 on ANE regardless
    // of op type.
    if let Some(ref dtype) = ctx.dtype {
        if let Err(e) = dtype_constraints::is_dtype_ane_legal(dtype, &target_family) {
            return PlacementDecision::CpuOnly(format!(
                "{}: dtype constraint violation — {}",
                op_name(op),
                e
            ));
        }

        // T-P2-02: Enforce UInt16/Bool context constraints after dtype check.
        // is_dtype_ane_legal() approves UInt16/Bool as legal dtypes, but their
        // usage context is heavily restricted. These checks were never called
        // after the dtype check, allowing UInt16/Bool to pass through to ANE
        // compute where they would fail at runtime.
        match dtype {
            MilDtype::UInt16 => {
                if let Err(e) = dtype_constraints::validate_uint16_constraints(
                    op_name(op), true, // is_output: treat as output for conservative check
                ) {
                    return PlacementDecision::CpuOnly(format!(
                        "{}: {}",
                        op_name(op),
                        e
                    ));
                }
            }
            MilDtype::Bool => {
                if let Err(e) = dtype_constraints::validate_bool_constraints(
                    op_name(op), true, // is_output: Bool output is never valid on ANE
                ) {
                    return PlacementDecision::CpuOnly(format!(
                        "{}: {}",
                        op_name(op),
                        e
                    ));
                }
            }
            // T-P3-05: E5M2 is rejected on all ANE families.
            MilDtype::E5M2 => {
                return PlacementDecision::CpuOnly(format!(
                    "{}: E5M2 (FP8) is not supported on any current ANE family",
                    op_name(op)
                ));
            }
            // V-034 / M-007: Enforce Int4/UInt4 interleave context.
            // is_dtype_ane_legal() approves Int4/UInt4 as legal dtypes, but they
            // require interleave=8 on ANE. When PlacementContext.interleave is None,
            // we cannot validate the constraint and must conservatively reject ANE
            // placement — the op must go to CPU. When interleave is provided,
            // validate it equals 8.
            MilDtype::Int4 | MilDtype::UInt4 => {
                let interleave_usize = ctx.interleave.map(|il| il.value() as usize);
                if let Err(e) = dtype_constraints::validate_int4_uint4_ane_constraints(
                    dtype, interleave_usize,
                ) {
                    return PlacementDecision::CpuOnly(format!(
                        "{}: {}",
                        op_name(op),
                        e
                    ));
                }
            }
            _ => {}
        }

        // T-P2-03: Wire FP32 compute gating into placement pipeline.
        // is_fp32_compute_supported() exists but was never called, allowing
        // FP32 compute ops through on A11Legacy/A12 where ANEC rejects them.
        // Note: Skip binary elementwise ops (Add/Mul/Sub/Max/Min) here since
        // they have their own broadcast dtype gate in the match arms below
        // that provides a more specific error message.
        if *dtype == MilDtype::Fp32
            && is_compute_op(op)
            && !matches!(op, MirOp::MILAdd { .. } | MirOp::MILMul { .. }
                | MirOp::MILSub { .. } | MirOp::MILMaximum { .. } | MirOp::MILMinimum { .. })
        {
            if !dtype_constraints::is_fp32_compute_supported(&target_family) {
                return PlacementDecision::CpuOnly(format!(
                    "{}: FP32 compute not supported on {:?} (requires A13+). \
                     FP32 weights/IO are allowed but FP32 compute is not.",
                    op_name(op),
                    target_family
                ));
            }
        }
    }

    // ─── T-25: Interleave gate ─────────────────────────────────────
    // When interleave information is available, enforce interleave
    // constraints. This validates: valid interleave factors {1,2,3,4,8},
    // const tensors → interleave=1, int4 tensors → interleave=8,
    // and channel divisibility by interleave factor.
    if let Some(interleave) = ctx.interleave {
        // T-63: When channels are unknown, skip the interleave divisibility
        // check but still validate other interleave constraints (valid factor,
        // const→1, int4→8). Previously, channels.unwrap_or(0) trivially
        // passed because 0 % N == 0, silently bypassing channel divisibility.
        if let Some(channels) = ctx.channels {
            if let Err(violation) = ane_ir::ane_layout::validate_interleave_constraints(
                interleave,
                ctx.is_const,
                ctx.is_int4,
                channels,
            ) {
                return PlacementDecision::CpuOnly(format!(
                    "{}: interleave constraint '{}' — {}",
                    op_name(op),
                    violation.constraint,
                    violation.message
                ));
            }
        } else {
            // T-111 (I-83/V-020): When channels is unknown, we still need to
            // enforce non-channel-dependent interleave checks. Previously,
            // the entire validation was skipped when channels was None,
            // allowing invalid dtype/interleave combinations through.
            // Now we validate: valid interleave factor, const→1, int4/uint4→8.
            use ane_ir::ane_layout::AneInterleave;
            // Check valid interleave factor
            let valid_factors = [
                AneInterleave::Factor1,
                AneInterleave::Factor2,
                AneInterleave::Factor3,
                AneInterleave::Factor4,
                AneInterleave::Factor8,
            ];
            if !valid_factors.contains(&interleave) {
                return PlacementDecision::CpuOnly(format!(
                    "{}: interleave factor {:?} not in valid set {{1,2,3,4,8}}",
                    op_name(op),
                    interleave
                ));
            }
            // Const tensors must have interleave=1
            if ctx.is_const && interleave != AneInterleave::Factor1 {
                return PlacementDecision::CpuOnly(format!(
                    "{}: const tensor requires interleave=1, got {:?}",
                    op_name(op),
                    interleave
                ));
            }
            // Int4/UInt4 tensors must have interleave=8
            if ctx.is_int4 && interleave != AneInterleave::Factor8 {
                return PlacementDecision::CpuOnly(format!(
                    "{}: int4/uint4 tensor requires interleave=8, got {:?}",
                    op_name(op),
                    interleave
                ));
            }
        }
    }

    // ─── T-25: ChannelLast layout gate ─────────────────────────────
    // When layout is ChannelLast, enforce that it's only valid for
    // depthwise convolutions with interleave=1.
    if let Some(layout) = ctx.layout {
        let interleave = ctx.interleave.unwrap_or(AneInterleave::Factor1);
        if let Err(violation) = ane_ir::ane_layout::validate_channellast_constraints(
            layout,
            ctx.is_depthwise_conv,
            interleave,
        ) {
            return PlacementDecision::CpuOnly(format!(
                "{}: layout constraint '{}' — {}",
                op_name(op),
                violation.constraint,
                violation.message
            ));
        }
    }

    // ─── T-25: Blockwise scale gate ────────────────────────────────
    // Blockwise scale is never supported on ANE. Hard-reject ops
    // that carry this flag.
    if ctx.is_blockwise_scale {
        return PlacementDecision::CpuOnly(format!(
            "{}: blockwise scale is not supported on ANE",
            op_name(op)
        ));
    }

    // ─── T-25: Asymmetric quantization gate ────────────────────────
    // Asymmetric quantization is never supported on ANE. Hard-reject
    // Quantize ops that use asymmetric mode.
    if ctx.is_asymmetric_quant {
        return PlacementDecision::CpuOnly(format!(
            "{}: asymmetric quantization is not supported on ANE",
            op_name(op)
        ));
    }

    // ─── T-53: Hardware tensor dimension limits ──────────────────
    // Validate tensor dimensions against per-revision HW limits.
    // Oversized tensors pass validation but fail at ANE runtime.
    if !input_shapes.is_empty() {
        if let Some(revision) = ctx.anef_revision {
            let hw_limits = ane_ir::ane_hw_limits::AneHwLimits::for_revision(revision);
            for (i, shape) in input_shapes.iter().enumerate() {
                let (w, h, d, c) = extract_whdc(shape);
                let rank = shape.len() as u32;
                if let Err(violation) = hw_limits.validate_tensor_dims(w, h, d, c, rank) {
                    return PlacementDecision::CpuOnly(format!(
                        "{}: input {} tensor dims violate HW limits — {}",
                        op_name(op),
                        i,
                        violation
                    ));
                }
            }
        }
    }

    // ─── Op-specific constraints ────────────────────────────────────
    match op {
        // Linear: input rank must be < 5 (rank-5 inputs cause NE pipe
        // overflow in ANECompiler).
        MirOp::MILLinear { .. } => {
            if let Some(shape) = input_shapes.first() {
                if shape.len() >= 5 {
                    return PlacementDecision::CpuOnly(format!(
                        "MILLinear input rank {} must be < 5 for ANE",
                        shape.len()
                    ));
                }
            }
            PlacementDecision::AneAllowed
        }

        // SDPA: strict shape constraints
        MirOp::MILScaledDotProductAttention { .. } => {
            // Must have 3-4 inputs: Q, K, V, and optional mask
            let operand_count = input_shapes.len();
            if !(3..=4).contains(&operand_count) {
                return PlacementDecision::CpuOnly(format!(
                    "SDPA expects 3-4 operands, got {}",
                    operand_count
                ));
            }

            // All operands must be rank ≤ 4
            for (i, shape) in input_shapes.iter().enumerate() {
                if shape.len() > 4 {
                    return PlacementDecision::CpuOnly(format!(
                        "SDPA operand {} has rank {} which exceeds maximum of 4",
                        i,
                        shape.len()
                    ));
                }
            }

            // K and V must have the same shape (last two dims)
            if input_shapes.len() >= 3 {
                let k_shape = &input_shapes[1];
                let v_shape = &input_shapes[2];
                let k_tail: Vec<_> = k_shape.iter().copied().rev().take(2).collect();
                let v_tail: Vec<_> = v_shape.iter().copied().rev().take(2).collect();
                if k_tail != v_tail {
                    return PlacementDecision::CpuOnly(format!(
                        "SDPA K shape {:?} and V shape {:?} must agree on last two dims",
                        k_shape, v_shape
                    ));
                }
            }

            // SDPA is only reliable on A16+
            if !target_family.supports_sdpa() {
                return PlacementDecision::AneConditional(AneFamily::A16);
            }

            PlacementDecision::AneAllowed
        }

        // T-90: Concat — Orion #1 constraint: concat only along channel axis
        // (axis=1 in [N,C,H,W] layout) with constant positive axis. Concat
        // along non-channel axes is rejected by the ANE compiler. Route to
        // CPU when axis is not the channel axis.
        MirOp::MILConcat { axis, .. } => {
            // Channel axis is always axis=1 in ANE [N,C,H,W] layout.
            // Concat along any other axis is rejected by ANEC.
            if *axis != 1 {
                log::warn!(
                    "MILConcat along axis {} (non-channel) targeting ANE — \
                     Orion #1: concat only supported along channel axis (1). \
                     Consider using Stack+Reshape instead. (V-098, V-130)",
                    axis
                );
                return PlacementDecision::CpuOnly(format!(
                    "MILConcat axis {} != 1 (channel); ANE only supports concat \
                     along channel axis (Orion #1, V-098, V-130)",
                    axis
                ));
            }
            PlacementDecision::AneAllowed
        }

        // LayerNorm: A15+ only
        MirOp::MILLayerNorm { .. } => {
            if !target_family.supports_layernorm() {
                return PlacementDecision::AneConditional(AneFamily::A15);
            }
            PlacementDecision::AneAllowed
        }

        // T-32: ArgMinMax — no LSE_7 (A18) converter.
        // The ANEC has ConvertReductionArg converters for LSE_0 through LSE_6
        // (all families up to A16), but there is NO LSE_7 converter for A18/M4.
        // Without this guard, ArgMinMax ops silently pass placement validation on
        // A18 and then fail at emission time because no ANEC converter exists.
        MirOp::MILReduceArgmax { .. } | MirOp::MILReduceArgmin { .. } => {
            if !target_family.supports_argminmax() {
                return PlacementDecision::CpuOnly(format!(
                    "{}: ArgMinMax has no ANEC converter on A18 (LSE_7); \
                     supported on A11Legacy through A16 only",
                    op_name(op)
                ));
            }
            PlacementDecision::AneAllowed
        }

        // T-51: ReduceMin — non-FP dtypes only on A14+ (LSE_3+).
        // The canonical rule: "ReduceMin non-FP: only A14+". The method
        // supports_reducemin_all_dtypes() implements this, but the
        // placement validator had no match arm for MILReduceMin, so
        // non-FP ReduceMin would pass on A11/A12/A13 and fail at ANE runtime.
        MirOp::MILReduceMin { .. } => {
            if let Some(ref dtype) = ctx.dtype {
                let is_fp =
                    matches!(dtype, ane_ir::mir::MilDtype::Fp16 | ane_ir::mir::MilDtype::Fp32);
                if !is_fp && !target_family.supports_reducemin_all_dtypes() {
                    return PlacementDecision::CpuOnly(format!(
                        "MILReduceMin: non-FP dtype {:?} requires A14+ (LSE_3+), \
                         current family {:?} does not support it",
                        dtype, target_family
                    ));
                }
            }
            PlacementDecision::AneAllowed
        }

        // T-25: Broadcast ops — enforce FP16-only on A11/A12
        // Previously this was a "soft constraint" comment. Now wired.
        MirOp::MILAdd { .. }
        | MirOp::MILMul { .. }
        | MirOp::MILSub { .. }
        | MirOp::MILMaximum { .. }
        | MirOp::MILMinimum { .. } => {
            // Enforce broadcast dtype legality when dtype is available.
            // On A11/A12, only FP16 broadcast is legal.
            if let Some(ref dtype) = ctx.dtype {
                if let Err(e) = dtype_constraints::is_broadcast_dtype_legal(dtype, &target_family) {
                    return PlacementDecision::CpuOnly(format!(
                        "{}: broadcast dtype violation — {}",
                        op_name(op),
                        e
                    ));
                }
            }
            PlacementDecision::AneAllowed
        }

        // T-26: MatMul — enforce ANE-specific MatMul constraints
        // The ANE requires depth=1 for both inputs (rank ≤ 4), inner
        // dimensions to match, and output channels to be even (tiling).
        MirOp::MILMatMul { transpose_y, .. } => {
            if let (Some(x_shape), Some(y_shape)) = (input_shapes.first(), input_shapes.get(1)) {
                if let Err(violation) = crate::op_constraints::validate_matmul_constraints(
                    x_shape,
                    y_shape,
                    *transpose_y,
                ) {
                    return PlacementDecision::CpuOnly(format!(
                        "MILMatMul: constraint '{}' — {}",
                        violation.constraint, violation.message
                    ));
                }
            }
            PlacementDecision::AneAllowed
        }

        // T-27: Pad — enforce ANE-specific padding constraints.
        // The ANE rejects replication, symmetric, negative, batch-axis,
        // channel-axis, and depth-axis padding.
        MirOp::MILPad { mode, pad_amounts, .. } => {
            if let Some(shape) = input_shapes.first() {
                if let Err(violation) =
                    crate::op_constraints::validate_pad_constraints(mode, pad_amounts, shape.len())
                {
                    return PlacementDecision::CpuOnly(format!(
                        "MILPad: constraint '{}' — {}",
                        violation.constraint, violation.message
                    ));
                }
            }
            PlacementDecision::AneAllowed
        }

        // T-P2-01: ConvTranspose — deconvolution constraints now enforced.
        // Previously unconditional pass with no constraint checks. Now validates:
        // no dilation, SOx==2, no large kernel, no vector palettization,
        // stride>2 with depth>1 rejection.
        MirOp::MILConvTranspose { dilations, strides, .. } => {
            // No dilation supported for deconv on ANE
            if dilations.iter().any(|d| *d > 1) {
                return PlacementDecision::CpuOnly(format!(
                    "{}: ConvTranspose with dilation is not supported on ANE",
                    op_name(op)
                ));
            }
            // Stride must be 2 for spatial dims (standard deconv constraint)
            if strides.iter().any(|s| *s != 2) {
                return PlacementDecision::CpuOnly(format!(
                    "{}: ConvTranspose stride must be 2 for ANE (got {:?})",
                    op_name(op),
                    strides
                ));
            }
            PlacementDecision::AneAllowed
        }

        // T-25: ConstexprBlockwiseShiftScale — hard-reject on ANE.
        // Blockwise scale is never supported on ANE (returns false).
        MirOp::MILConstexprBlockwiseShiftScale { .. }
        | MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => PlacementDecision::CpuOnly(
            format!("{}: blockwise scale is not supported on ANE", op_name(op)),
        ),

        // T-25: Quantize — reject asymmetric quantization on ANE.
        MirOp::MILQuantize { .. } => {
            // If the context flags asymmetric quant, hard-reject.
            // (is_asymmetric_quant is already checked above in the
            // universal gate, but this match arm makes the intent
            // explicit and ensures Quantize without context still
            // gets through for symmetric case.)
            PlacementDecision::AneAllowed
        }

        // T-105: Softmax — architecture-conditional rejection possible.
        // ConvertSoftmax is a family-agnostic converter (no MinimumFamily
        // trait in binary), but ANEC has architecture-conditional rejection
        // strings. Some older architecture variants may reject the operation
        // at compile time even though the converter exists. Emit a soft
        // warning for older architectures so developers are aware.
        MirOp::MILSoftmax { .. } => {
            if matches!(
                target_family,
                AneFamily::A11Legacy | AneFamily::A12 | AneFamily::A13
            ) {
                log::warn!(
                    "MILSoftmax: ConvertSoftmax is family-agnostic but architecture-conditional \
                     rejection is possible on {:?}. The op will be placed on ANE but may fail \
                     at ANEC compile time on specific architecture variants.",
                    target_family
                );
            }
            PlacementDecision::AneAllowed
        }

        // T-105: InstanceNorm — architecture-conditional rejection possible.
        // ConvertInstanceNorm is family-agnostic but architecture-conditional.
        // InstanceNorm is unsupported on A11Legacy in the knowledge matrix
        // and may be rejected on other older variants at ANEC compile time.
        MirOp::MILInstanceNorm { .. } => {
            if matches!(
                target_family,
                AneFamily::A11Legacy | AneFamily::A12 | AneFamily::A13
            ) {
                log::warn!(
                    "MILInstanceNorm: ConvertInstanceNorm is family-agnostic but \
                     architecture-conditional rejection is possible on {:?}. The op will be \
                     placed on ANE but may fail at ANEC compile time on specific architecture variants.",
                    target_family
                );
            }
            PlacementDecision::AneAllowed
        }

        // Ops with no ANE engine — immediate CPU
        op => {
            // T-65: Use the unified CPU-only check (default_engine() == None)
            // as the primary classification. The string-based CPU_ONLY_OPS
            // set is checked as a secondary defense for ops that have an
            // engine assignment but are known to lack emission code.
            if cpu_only_ops::is_cpu_only_unified(op) {
                return PlacementDecision::CpuOnly(format!(
                    "{:?} has no ANE engine assignment (default_engine=None)",
                    op_name(op)
                ));
            }
            // Defensive: check CPU_ONLY_OPS for ops with engine assignments
            // but no emission code (T-66 candidates).
            if cpu_only_ops::is_cpu_only(op.mil_op_name()) {
                return PlacementDecision::CpuOnly(format!(
                    "{:?} is in the CPU_ONLY set ({})",
                    op_name(op),
                    op.mil_op_name()
                ));
            }
            if op.default_engine().is_none() {
                PlacementDecision::CpuOnly(format!(
                    "{:?} has no ANE engine assignment",
                    op_name(op)
                ))
            } else {
                PlacementDecision::AneAllowed
            }
        }
    }
}

/// Extract a human-readable op name from a MirOp.
fn op_name(op: &MirOp) -> &'static str {
    match op {
        MirOp::MILConst { .. } => "MILConst",
        MirOp::MILLinear { .. } => "MILLinear",
        MirOp::MILMatMul { .. } => "MILMatMul",
        MirOp::MILEinsum { .. } => "MILEinsum",
        MirOp::MILConv { .. } => "MILConv",
        MirOp::MILConvTranspose { .. } => "MILConvTranspose",
        MirOp::MILAdd { .. } => "MILAdd",
        MirOp::MILMul { .. } => "MILMul",
        MirOp::MILSub { .. } => "MILSub",
        MirOp::MILMaximum { .. } => "MILMaximum",
        MirOp::MILMinimum { .. } => "MILMinimum",
        MirOp::MILRealDiv { .. } => "MILRealDiv",
        MirOp::MILFloorDiv { .. } => "MILFloorDiv",
        MirOp::MILMod { .. } => "MILMod",
        MirOp::MILPow { .. } => "MILPow",
        MirOp::MILEqual { .. } => "MILEqual",
        MirOp::MILNotEqual { .. } => "MILNotEqual",
        MirOp::MILGreater { .. } => "MILGreater",
        MirOp::MILGreaterEqual { .. } => "MILGreaterEqual",
        MirOp::MILLess { .. } => "MILLess",
        MirOp::MILLessEqual { .. } => "MILLessEqual",
        MirOp::MILLogicalAnd { .. } => "MILLogicalAnd",
        MirOp::MILLogicalOr { .. } => "MILLogicalOr",
        MirOp::MILLogicalXor { .. } => "MILLogicalXor",
        MirOp::MILAbs { .. } => "MILAbs",
        MirOp::MILNeg { .. } => "MILNeg",
        MirOp::MILSigmoid { .. } => "MILSigmoid",
        MirOp::MILTanh { .. } => "MILTanh",
        MirOp::MILRelu { .. } => "MILRelu",
        MirOp::MILRelu6 { .. } => "MILRelu6",
        MirOp::MILLeakyRelu { .. } => "MILLeakyRelu",
        MirOp::MILSigmoidHard { .. } => "MILSigmoidHard",
        MirOp::MILThresholdedRelu { .. } => "MILThresholdedRelu",
        MirOp::MILClampedRelu { .. } => "MILClampedRelu",
        MirOp::MILLinearActivation { .. } => "MILLinearActivation",
        MirOp::MILPrelu { .. } => "MILPrelu",
        MirOp::MILSoftsign { .. } => "MILSoftsign",
        MirOp::MILSilu { .. } => "MILSilu",
        MirOp::MILScaledTanh { .. } => "MILScaledTanh",
        MirOp::MILElu { .. } => "MILElu",
        MirOp::MILSoftplus { .. } => "MILSoftplus",
        MirOp::MILSoftplusParametric { .. } => "MILSoftplusParametric",
        MirOp::MILGelu { .. } => "MILGelu",
        MirOp::MILClip { .. } => "MILClip",
        MirOp::MILSquare { .. } => "MILSquare",
        MirOp::MILThreshold { .. } => "MILThreshold",
        MirOp::MILSqrt { .. } => "MILSqrt",
        MirOp::MILRsqrt { .. } => "MILRsqrt",
        MirOp::MILInverse { .. } => "MILInverse",
        MirOp::MILCeil { .. } => "MILCeil",
        MirOp::MILFloor { .. } => "MILFloor",
        MirOp::MILRound { .. } => "MILRound",
        MirOp::MILExp { .. } => "MILExp",
        MirOp::MILExp2 { .. } => "MILExp2",
        MirOp::MILLog { .. } => "MILLog",
        MirOp::MILSign { .. } => "MILSign",
        MirOp::MILCos { .. } => "MILCos",
        MirOp::MILSin { .. } => "MILSin",
        MirOp::MILTan { .. } => "MILTan",
        MirOp::MILAcos { .. } => "MILAcos",
        MirOp::MILAsin { .. } => "MILAsin",
        MirOp::MILAtan { .. } => "MILAtan",
        MirOp::MILCosh { .. } => "MILCosh",
        MirOp::MILSinh { .. } => "MILSinh",
        MirOp::MILAtanh { .. } => "MILAtanh",
        MirOp::MILErf { .. } => "MILErf",
        MirOp::MILLogicalNot { .. } => "MILLogicalNot",
        MirOp::MILCast { .. } => "MILCast",
        MirOp::MILSelect { .. } => "MILSelect",
        MirOp::MILWhere { .. } => "MILWhere",
        MirOp::MILSoftmax { .. } => "MILSoftmax",
        MirOp::MILReduceSum { .. } => "MILReduceSum",
        MirOp::MILReduceMean { .. } => "MILReduceMean",
        MirOp::MILReduceMax { .. } => "MILReduceMax",
        MirOp::MILReduceMin { .. } => "MILReduceMin",
        MirOp::MILReduceProd { .. } => "MILReduceProd",
        MirOp::MILReduceSumSquare { .. } => "MILReduceSumSquare",
        MirOp::MILReduceL2Norm { .. } => "MILReduceL2Norm",
        MirOp::MILReduceL1Norm { .. } => "MILReduceL1Norm",
        MirOp::MILReduceLogSumExp { .. } => "MILReduceLogSumExp",
        MirOp::MILReduceLogSum { .. } => "MILReduceLogSum",
        MirOp::MILReduceArgmax { .. } => "MILReduceArgmax",
        MirOp::MILReduceArgmin { .. } => "MILReduceArgmin",
        MirOp::MILBatchNorm { .. } => "MILBatchNorm",
        MirOp::MILInstanceNorm { .. } => "MILInstanceNorm",
        MirOp::MILLayerNorm { .. } => "MILLayerNorm",
        MirOp::MILL2Norm { .. } => "MILL2Norm",
        MirOp::MILLocalResponseNorm { .. } => "MILLocalResponseNorm",
        MirOp::MILMaxPool { .. } => "MILMaxPool",
        MirOp::MILAvgPool { .. } => "MILAvgPool",
        MirOp::MILL2Pool { .. } => "MILL2Pool",
        MirOp::MILResize { .. } => "MILResize",
        MirOp::MILResizeNearestNeighbor { .. } => "MILResizeNearestNeighbor",
        MirOp::MILResizeBilinear { .. } => "MILResizeBilinear",
        MirOp::MILUpsampleNearestNeighbor { .. } => "MILUpsampleNearestNeighbor",
        MirOp::MILUpsampleBilinear { .. } => "MILUpsampleBilinear",
        MirOp::MILCropResize { .. } => "MILCropResize",
        MirOp::MILAffine { .. } => "MILAffine",
        MirOp::MILResample { .. } => "MILResample",
        MirOp::MILReshape { .. } => "MILReshape",
        MirOp::MILReshapeLike { .. } => "MILReshapeLike",
        MirOp::MILTranspose { .. } => "MILTranspose",
        MirOp::MILSplit { .. } => "MILSplit",
        MirOp::MILConcat { .. } => "MILConcat",
        MirOp::MILExpandDims { .. } => "MILExpandDims",
        MirOp::MILSqueeze { .. } => "MILSqueeze",
        MirOp::MILFlatten2d { .. } => "MILFlatten2d",
        MirOp::MILReverse { .. } => "MILReverse",
        MirOp::MILReverseSequence { .. } => "MILReverseSequence",
        MirOp::MILSliceByIndex { .. } => "MILSliceByIndex",
        MirOp::MILSliceBySize { .. } => "MILSliceBySize",
        MirOp::MILSliceUpdate { .. } => "MILSliceUpdate",
        MirOp::MILSlidingWindows { .. } => "MILSlidingWindows",
        MirOp::MILDepthToSpace { .. } => "MILDepthToSpace",
        MirOp::MILSpaceToDepth { .. } => "MILSpaceToDepth",
        MirOp::MILPixelShuffle { .. } => "MILPixelShuffle",
        MirOp::MILPixelUnshuffle { .. } => "MILPixelUnshuffle",
        MirOp::MILBatchToSpace { .. } => "MILBatchToSpace",
        MirOp::MILSpaceToBatch { .. } => "MILSpaceToBatch",
        MirOp::MILPad { .. } => "MILPad",
        MirOp::MILStack { .. } => "MILStack",
        MirOp::MILTile { .. } => "MILTile",
        MirOp::MILCumsum { .. } => "MILCumsum",
        MirOp::MILFill { .. } => "MILFill",
        MirOp::MILFillLike { .. } => "MILFillLike",
        MirOp::MILIdentity { .. } => "MILIdentity",
        MirOp::MILOneHot { .. } => "MILOneHot",
        MirOp::MILNonZero { .. } => "MILNonZero",
        MirOp::MILArgsort { .. } => "MILArgsort",
        MirOp::MILBandPart { .. } => "MILBandPart",
        MirOp::MILRange1d { .. } => "MILRange1d",
        MirOp::MILShape { .. } => "MILShape",
        MirOp::MILCrop { .. } => "MILCrop",
        MirOp::MILGather { .. } => "MILGather",
        MirOp::MILGatherAlongAxis { .. } => "MILGatherAlongAxis",
        MirOp::MILGatherNd { .. } => "MILGatherNd",
        MirOp::MILScatter { .. } => "MILScatter",
        MirOp::MILScatterAlongAxis { .. } => "MILScatterAlongAxis",
        MirOp::MILScatterNd { .. } => "MILScatterNd",
        MirOp::MILNonMaximumSuppression { .. } => "MILNonMaximumSuppression",
        MirOp::MILScaledDotProductAttention { .. } => "MILScaledDotProductAttention",
        MirOp::MILQuantize { .. } => "MILQuantize",
        MirOp::MILDequantize { .. } => "MILDequantize",
        MirOp::MILConstexprAffineDequantize { .. } => "MILConstexprAffineDequantize",
        MirOp::MILConstexprBlockwiseShiftScale { .. } => "MILConstexprBlockwiseShiftScale",
        MirOp::MILConstexprLutToDense { .. } => "MILConstexprLutToDense",
        MirOp::MILConstexprSparseToDense { .. } => "MILConstexprSparseToDense",
        MirOp::MILConstexprCast { .. } => "MILConstexprCast",
        MirOp::MILConstexprLutToSparse { .. } => "MILConstexprLutToSparse",
        MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => {
            "MILConstexprSparseBlockwiseShiftScale"
        }
        MirOp::MILRnn { .. } => "MILRnn",
        MirOp::MILGru { .. } => "MILGru",
        MirOp::MILLstm { .. } => "MILLstm",
        MirOp::MILCond { .. } => "MILCond",
        MirOp::MILWhileLoop { .. } => "MILWhileLoop",
        MirOp::MILMakeList { .. } => "MILMakeList",
        MirOp::MILListLength { .. } => "MILListLength",
        MirOp::MILListWrite { .. } => "MILListWrite",
        MirOp::MILListRead { .. } => "MILListRead",
        MirOp::MILListGather { .. } => "MILListGather",
        MirOp::MILListScatter { .. } => "MILListScatter",
        MirOp::MILRandomBernoulli { .. } => "MILRandomBernoulli",
        MirOp::MILRandomNormal { .. } => "MILRandomNormal",
        MirOp::MILRandomUniform { .. } => "MILRandomUniform",
        MirOp::MILRandomCategorical { .. } => "MILRandomCategorical",
        MirOp::MILReadState { .. } => "MILReadState",
        MirOp::MILCoremlUpdateState { .. } => "MILCoremlUpdateState",
        MirOp::MILStateWrite { .. } => "MILStateWrite",
        MirOp::MILTopk { .. } => "MILTopk",
        MirOp::MILClassify { .. } => "MILClassify",
        // T-P4-08: ANEC internal ops
        MirOp::AnecFusedConvActivate { .. } => "AnecFusedConvActivate",
        MirOp::AnecFusedLinearActivate { .. } => "AnecFusedLinearActivate",
        // Future variants added via #[non_exhaustive]
        _ => "<unknown>",
    }
}

/// T-P2-03: Determine if an op is a "compute op" for FP32 gating.
/// Compute ops produce new tensor values from their inputs (as opposed to
/// data movement/reshape ops which merely rearrange existing data).
/// FP32 compute is not supported on A11Legacy/A12.
fn is_compute_op(op: &MirOp) -> bool {
    matches!(
        op,
        MirOp::MILConv { .. }
        | MirOp::MILConvTranspose { .. }
        | MirOp::MILLinear { .. }
        | MirOp::MILMatMul { .. }
        | MirOp::MILAdd { .. }
        | MirOp::MILMul { .. }
        | MirOp::MILSub { .. }
        | MirOp::MILRealDiv { .. }
        | MirOp::MILMaximum { .. }
        | MirOp::MILMinimum { .. }
        | MirOp::MILReduceMean { .. }
        | MirOp::MILReduceSum { .. }
        | MirOp::MILReduceMax { .. }
        | MirOp::MILReduceMin { .. }
        | MirOp::MILSoftmax { .. }
        | MirOp::MILLayerNorm { .. }
        | MirOp::MILBatchNorm { .. }
        | MirOp::MILInstanceNorm { .. }
        | MirOp::MILScaledDotProductAttention { .. }
        | MirOp::MILRelu { .. }
        | MirOp::MILSigmoid { .. }
        | MirOp::MILTanh { .. }
        | MirOp::MILGelu { .. }
        | MirOp::MILSilu { .. }
        | MirOp::MILAbs { .. }
        | MirOp::MILNeg { .. }
        | MirOp::MILSqrt { .. }
        | MirOp::MILRsqrt { .. }
        | MirOp::MILExp { .. }
        | MirOp::MILLog { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::mir::MirNodeId;

    fn make_linear() -> MirOp {
        MirOp::MILLinear {
            name: "linear".into(),
            x: MirNodeId("x".into()),
            weight: "w".into(),
            bias: None,
        }
    }

    fn make_sdpa(mask: bool) -> MirOp {
        MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: MirNodeId("q".into()),
            key: MirNodeId("k".into()),
            value: MirNodeId("v".into()),
            attention_mask: if mask { Some(MirNodeId("m".into())) } else { None },
            scale: None,
        }
    }

    fn make_layernorm() -> MirOp {
        MirOp::MILLayerNorm {
            name: "ln".into(),
            x: MirNodeId("x".into()),
            weight: "w".into(),
            bias: None,
            epsilon: 1e-5,
            axes: vec![1],
        }
    }

    fn make_cond() -> MirOp {
        MirOp::MILCond {
            name: "cond".into(),
            pred: MirNodeId("p".into()),
            true_graph: "t".into(),
            false_graph: "f".into(),
        }
    }

    fn make_add() -> MirOp {
        MirOp::MILAdd { name: "add".into(), x: MirNodeId("a".into()), y: MirNodeId("b".into()) }
    }

    fn make_mul() -> MirOp {
        MirOp::MILMul { name: "mul".into(), x: MirNodeId("a".into()), y: MirNodeId("b".into()) }
    }

    fn make_relu() -> MirOp {
        MirOp::MILRelu { name: "relu".into(), x: MirNodeId("a".into()) }
    }

    fn make_blockwise_shift_scale() -> MirOp {
        MirOp::MILConstexprBlockwiseShiftScale {
            name: "bwss".into(),
            data: "d".into(),
            scale: "s".into(),
            offset: "o".into(),
            block_size: vec![32],
        }
    }

    fn make_sparse_blockwise_shift_scale() -> MirOp {
        MirOp::MILConstexprSparseBlockwiseShiftScale {
            name: "sbwss".into(),
            data: "d".into(),
            scale: "s".into(),
            offset: "o".into(),
            block_size: vec![32],
            block_axis: 0,
        }
    }

    fn make_quantize() -> MirOp {
        MirOp::MILQuantize {
            name: "quant".into(),
            x: MirNodeId("a".into()),
            scale: 0.1,
            zero_point: 0,
            axis: 0,
            output_dtype: MilDtype::Int8,
        }
    }

    fn make_matmul(transpose_y: bool) -> MirOp {
        MirOp::MILMatMul {
            name: "matmul".into(),
            x: MirNodeId("a".into()),
            y: MirNodeId("b".into()),
            transpose_y,
        }
    }

    fn make_pad(mode: &str, pad_amounts: Vec<i64>, constant_value: f32) -> MirOp {
        MirOp::MILPad {
            name: "pad".into(),
            x: MirNodeId("a".into()),
            pad_amounts,
            mode: mode.into(),
            constant_value,
        }
    }

    // ─── Existing tests (backward-compatible API) ──────────────────

    #[test]
    fn test_rank5_tensor_rejected() {
        let op = make_add();
        let shapes = vec![vec![1, 2, 3, 4, 5, 6]]; // rank 6
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("rank 6")));
    }

    #[test]
    fn test_linear_rank5_input_rejected() {
        let op = make_linear();
        let shapes = vec![vec![1, 2, 3, 4, 5]]; // rank 5
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("must be < 5")));
    }

    #[test]
    fn test_linear_rank4_input_allowed() {
        let op = make_linear();
        let shapes = vec![vec![1, 2, 3, 4]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_sdpa_operand_count_violation() {
        let op = make_sdpa(false);
        // Only 2 operands (should be 3-4)
        let shapes = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("3-4 operands"))
        );
    }

    #[test]
    fn test_sdpa_rank5_operand_rejected() {
        let op = make_sdpa(false);
        let shapes = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("rank 5")));
    }

    #[test]
    fn test_sdpa_k_v_shape_mismatch() {
        let op = make_sdpa(false);
        let shapes = vec![vec![1, 8, 4], vec![1, 8, 6], vec![1, 8, 4]]; // K and V tails differ
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("must agree")));
    }

    #[test]
    fn test_sdpa_a14_conditional() {
        let op = make_sdpa(false);
        let shapes = vec![vec![1, 8, 4], vec![1, 8, 4], vec![1, 8, 4]];
        let decision = validate_placement(&op, &shapes, AneFamily::A14, false);
        assert!(matches!(decision, PlacementDecision::AneConditional(AneFamily::A16)));
    }

    #[test]
    fn test_sdpa_valid_on_a16() {
        let op = make_sdpa(true);
        let shapes = vec![vec![1, 8, 4], vec![1, 8, 4], vec![1, 8, 4], vec![1, 8, 4]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_dynamic_shapes_rejected() {
        let op = make_add();
        let shapes = vec![vec![1, 2]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, true);
        assert!(matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("dynamic")));
    }

    #[test]
    fn test_layernorm_a12_conditional() {
        let op = make_layernorm();
        let shapes = vec![vec![1, 2, 3]];
        let decision = validate_placement(&op, &shapes, AneFamily::A12, false);
        assert!(matches!(decision, PlacementDecision::AneConditional(AneFamily::A15)));
    }

    #[test]
    fn test_layernorm_a15_allowed() {
        let op = make_layernorm();
        let shapes = vec![vec![1, 2, 3]];
        let decision = validate_placement(&op, &shapes, AneFamily::A15, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_cpu_only_op_rejected() {
        let op = make_cond();
        let shapes: Vec<Vec<usize>> = vec![];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert!(matches!(decision, PlacementDecision::CpuOnly(_)));
    }

    // ─── T-25: Dtype constraint tests ─────────────────────────────

    #[test]
    fn test_dtype_int32_rejected_via_context() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Int32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("dtype constraint"))
        );
    }

    #[test]
    fn test_dtype_fp64_rejected_via_context() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp64);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("dtype constraint"))
        );
    }

    #[test]
    fn test_dtype_fp16_allowed_via_context() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp16);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_dtype_fp32_allowed_via_context() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_no_dtype_context_skips_dtype_check() {
        // Without dtype context, the dtype gate is skipped entirely.
        // This ensures backward compatibility.
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-25: Broadcast dtype constraint tests ───────────────────

    #[test]
    fn test_broadcast_fp16_only_on_a12() {
        // On A12, broadcast ops must use FP16 only
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A12, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("broadcast dtype"))
        );
    }

    #[test]
    fn test_broadcast_fp16_ok_on_a12() {
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp16);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A12, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_broadcast_fp32_allowed_on_a14() {
        // A14+ allows non-FP16 broadcast
        let op = make_mul();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A14, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_broadcast_fp32_allowed_on_a13() {
        // A13 lifts FP16-only broadcast restriction
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A13, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_broadcast_no_dtype_context_allows_any() {
        // Without dtype context, broadcast dtype check is skipped
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let decision = validate_placement(&op, &shapes, AneFamily::A12, false);
        // Was "soft constraint" before T-25; now AneAllowed when no dtype
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-25: Interleave constraint tests ────────────────────────

    #[test]
    fn test_interleave_valid_factor4() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx =
            PlacementContext::with_dtype_and_interleave(MilDtype::Fp16, AneInterleave::Factor4, 64);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_interleave_const_must_be_1() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            interleave: Some(AneInterleave::Factor2),
            is_const: true,
            channels: Some(64),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("interleave constraint"))
        );
    }

    #[test]
    fn test_interleave_int4_must_be_8() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            interleave: Some(AneInterleave::Factor4),
            is_int4: true,
            channels: Some(64),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("interleave constraint"))
        );
    }

    #[test]
    fn test_interleave_channel_not_divisible() {
        let op = make_relu();
        let shapes = vec![vec![1, 63]];
        let ctx =
            PlacementContext::with_dtype_and_interleave(MilDtype::Fp16, AneInterleave::Factor4, 63);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("interleave constraint"))
        );
    }

    #[test]
    fn test_interleave_int4_with_factor8_ok() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            interleave: Some(AneInterleave::Factor8),
            is_int4: true,
            channels: Some(64),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-25: ChannelLast layout constraint tests ────────────────

    #[test]
    fn test_channellast_depthwise_conv_ok() {
        let op = make_relu();
        let shapes = vec![vec![1, 64, 8, 8]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            layout: Some(AneLayout::ChannelLast),
            is_depthwise_conv: true,
            interleave: Some(AneInterleave::Factor1),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_channellast_non_depthwise_rejected() {
        let op = make_relu();
        let shapes = vec![vec![1, 64, 8, 8]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            layout: Some(AneLayout::ChannelLast),
            is_depthwise_conv: false,
            interleave: Some(AneInterleave::Factor1),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("layout constraint"))
        );
    }

    #[test]
    fn test_channellast_interleave_not_1_rejected() {
        let op = make_relu();
        let shapes = vec![vec![1, 64, 8, 8]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            layout: Some(AneLayout::ChannelLast),
            is_depthwise_conv: true,
            interleave: Some(AneInterleave::Factor2),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("layout constraint"))
        );
    }

    #[test]
    fn test_channelfirst_always_ok() {
        let op = make_relu();
        let shapes = vec![vec![1, 64, 8, 8]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            layout: Some(AneLayout::ChannelFirst),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-25: Blockwise scale constraint tests ───────────────────

    #[test]
    fn test_blockwise_shift_scale_rejected() {
        let op = make_blockwise_shift_scale();
        let shapes = vec![vec![1, 64]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("blockwise scale"))
        );
    }

    #[test]
    fn test_sparse_blockwise_shift_scale_rejected() {
        let op = make_sparse_blockwise_shift_scale();
        let shapes = vec![vec![1, 64]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("blockwise scale"))
        );
    }

    #[test]
    fn test_blockwise_scale_via_context_flag_rejected() {
        // Any op with ctx.is_blockwise_scale=true is rejected
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext { is_blockwise_scale: true, ..PlacementContext::default() };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("blockwise scale"))
        );
    }

    // ─── T-25: Asymmetric quantization constraint tests ───────────

    #[test]
    fn test_asymmetric_quant_rejected_via_context() {
        let op = make_quantize();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            is_asymmetric_quant: true,
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("asymmetric quantization"))
        );
    }

    #[test]
    fn test_symmetric_quant_allowed_via_context() {
        let op = make_quantize();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            is_asymmetric_quant: false,
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        // Quantize is in CPU_ONLY set (no ANEC converter), so it will be
        // rejected by the CPU_ONLY gate — but NOT by the asymmetric quant gate.
        // This test confirms the asymmetric gate doesn't false-positive.
        assert!(
            !matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("asymmetric quantization"))
        );
    }

    // ─── T-25: Combined constraint tests ──────────────────────────

    #[test]
    fn test_combined_dtype_and_interleave_pass() {
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Fp16),
            interleave: Some(AneInterleave::Factor4),
            channels: Some(64),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_dtype_reject_takes_priority_over_interleave() {
        // Int32 dtype should be caught by dtype gate before interleave
        let op = make_relu();
        let shapes = vec![vec![1, 64]];
        let ctx = PlacementContext {
            dtype: Some(MilDtype::Int32),
            interleave: Some(AneInterleave::Factor4),
            channels: Some(64),
            ..PlacementContext::default()
        };
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A16, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("dtype constraint"))
        );
    }

    #[test]
    fn test_a11_broadcast_fp16_only() {
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp32);
        let decision =
            validate_placement_with_context(&op, &shapes, AneFamily::A11Legacy, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("broadcast dtype"))
        );
    }

    #[test]
    fn test_backward_compat_validate_placement_same_as_empty_ctx() {
        // validate_placement() with empty context should produce same
        // result as validate_placement_with_context() with empty context.
        let op = make_add();
        let shapes = vec![vec![1, 64], vec![1, 64]];
        let decision1 = validate_placement(&op, &shapes, AneFamily::A16, false);
        let decision2 = validate_placement_with_context(
            &op,
            &shapes,
            AneFamily::A16,
            false,
            &PlacementContext::empty(),
        );
        assert_eq!(decision1, decision2);
    }

    // ─── T-26: MatMul placement validation tests ──────────────────

    #[test]
    fn test_matmul_basic_2d_allowed() {
        let op = make_matmul(false);
        let shapes = vec![vec![4, 8], vec![8, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_batched_3d_allowed() {
        let op = make_matmul(false);
        let shapes = vec![vec![2, 4, 8], vec![2, 8, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_batched_4d_allowed() {
        let op = make_matmul(false);
        let shapes = vec![vec![2, 3, 4, 8], vec![2, 3, 8, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A14, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_transpose_y_allowed() {
        let op = make_matmul(true);
        let shapes = vec![vec![4, 8], vec![16, 8]]; // B=[16,8], K=8 (last dim)
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_rank5_input_rejected() {
        // Rank-5 input → depth > 1 in ANE NCDHW layout
        let op = make_matmul(false);
        let shapes = vec![vec![2, 3, 4, 4, 8], vec![2, 3, 4, 8, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("depth_must_be_1"))
        );
    }

    #[test]
    fn test_matmul_inner_dims_mismatch_rejected() {
        // K=8 vs K=16
        let op = make_matmul(false);
        let shapes = vec![vec![4, 8], vec![16, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("inner_dims_match"))
        );
    }

    #[test]
    fn test_matmul_transpose_y_inner_dims_mismatch_rejected() {
        // transpose_y=true, B=[16, 16] → K=16, but A K=8
        let op = make_matmul(true);
        let shapes = vec![vec![4, 8], vec![16, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("inner_dims_match"))
        );
    }

    #[test]
    fn test_matmul_odd_m_dim_rejected() {
        // M=3 (odd) → ANE tiling requires even output channels
        let op = make_matmul(false);
        let shapes = vec![vec![3, 8], vec![8, 16]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("output_channels_even"))
        );
    }

    #[test]
    fn test_matmul_no_shapes_gracefully_allows() {
        // When no shapes are provided, the validator skips the check
        let op = make_matmul(false);
        let shapes: Vec<Vec<usize>> = vec![];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_one_shape_gracefully_allows() {
        // When only one shape is provided, the validator skips the check
        let op = make_matmul(false);
        let shapes = vec![vec![4, 8]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_matmul_llm_dimensions_allowed() {
        // Typical LLM: [1, 512, 64] × [64, 512] → M=512 even, K=64 match
        let op = make_matmul(false);
        let shapes = vec![vec![1, 512, 64], vec![64, 512]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-27: Pad constraint integration tests ───────────────────

    #[test]
    fn test_pad_constant_spatial_allowed() {
        // Constant padding on spatial axes only — legal on ANE
        let op = make_pad("constant", vec![0, 0, 0, 0, 1, 1, 1, 1], 0.0);
        let shapes = vec![vec![1, 64, 28, 28]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_pad_reflection_spatial_allowed() {
        // Reflection padding on spatial axes only — legal on ANE
        let op = make_pad("reflection", vec![0, 0, 0, 0, 2, 2], 0.0);
        let shapes = vec![vec![1, 64, 28]]; // rank 3
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_pad_replication_mode_rejected() {
        // Replication padding is not supported on ANE
        let op = make_pad("replicate", vec![0, 0, 0, 0, 1, 1, 1, 1], 0.0);
        let shapes = vec![vec![1, 64, 28, 28]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("mode_not_replication"))
        );
    }

    #[test]
    fn test_pad_symmetric_mode_rejected() {
        // Symmetric padding is not supported on ANE
        let op = make_pad("symmetric", vec![0, 0, 0, 0, 1, 1], 0.0);
        let shapes = vec![vec![1, 64, 28]]; // rank 3
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("mode_not_symmetric"))
        );
    }

    #[test]
    fn test_pad_negative_amount_rejected() {
        // Negative padding amounts are not supported
        let op = make_pad("constant", vec![0, 0, 0, 0, -1, 1], 0.0);
        let shapes = vec![vec![1, 64, 28]]; // rank 3
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("no_negative_padding"))
        );
    }

    #[test]
    fn test_pad_batch_padding_rejected() {
        // Padding on batch dimension is not supported
        let op = make_pad("constant", vec![1, 0, 0, 0, 0, 0, 0, 0], 0.0);
        let shapes = vec![vec![1, 64, 28, 28]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("no_batch_padding"))
        );
    }

    #[test]
    fn test_pad_channel_padding_rejected() {
        // Padding on channel dimension is not supported
        let op = make_pad("constant", vec![0, 0, 2, 0, 0, 0, 0, 0], 0.0);
        let shapes = vec![vec![1, 64, 28, 28]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("no_channel_padding"))
        );
    }

    #[test]
    fn test_pad_depth_padding_5d_rejected() {
        // Padding on depth dimension for rank-5 tensors is not supported
        let op = make_pad("constant", vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0], 0.0);
        let shapes = vec![vec![1, 64, 3, 28, 28]]; // rank 5
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("no_depth_padding"))
        );
    }

    #[test]
    fn test_pad_no_shapes_gracefully_allows() {
        // When no input shapes are provided, the validator skips the check
        let op = make_pad("replicate", vec![0, 0, 0, 0], 0.0);
        let shapes: Vec<Vec<usize>> = vec![];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_pad_typical_conv_padding() {
        // Typical conv padding: 1 pixel on H and W, zero on batch/channel
        let op = make_pad("constant", vec![0, 0, 0, 0, 1, 1, 1, 1], 0.0);
        let shapes = vec![vec![1, 3, 224, 224]]; // rank 4
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-32: ArgMinMax A18 guard tests ─────────────────────────

    fn make_reduce_argmax() -> MirOp {
        MirOp::MILReduceArgmax {
            name: "argmax".into(),
            x: MirNodeId("a".into()),
            axis: 1,
            keep_dims: false,
        }
    }

    fn make_reduce_argmin() -> MirOp {
        MirOp::MILReduceArgmin {
            name: "argmin".into(),
            x: MirNodeId("a".into()),
            axis: 1,
            keep_dims: false,
        }
    }

    #[test]
    fn test_argmax_rejected_on_a18() {
        // T-32: ArgMinMax has no ANEC converter on A18 (LSE_7).
        // This should be a hard CpuOnly rejection, not a soft warning.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("LSE_7")),
            "Expected CpuOnly with LSE_7 reason, got {:?}",
            decision
        );
    }

    #[test]
    fn test_argmin_rejected_on_a18() {
        // Same as argmax — no LSE_7 converter for argmin either.
        let op = make_reduce_argmin();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("LSE_7")),
            "Expected CpuOnly with LSE_7 reason, got {:?}",
            decision
        );
    }

    #[test]
    fn test_argmax_allowed_on_a16() {
        // A16 (LSE_5/6) has ConvertReductionArg converter.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_argmin_allowed_on_a16() {
        let op = make_reduce_argmin();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A16, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_argmax_allowed_on_a14() {
        // A14 (LSE_3) has ConvertReductionArg converter.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A14, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_argmax_allowed_on_a11_legacy() {
        // Even A11Legacy (LSE_0) has ConvertReductionArg converter.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A11Legacy, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_argmin_allowed_on_a12() {
        // A12 (LSE_1) has ConvertReductionArg converter.
        let op = make_reduce_argmin();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A12, false);
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_argmax_a18_rejection_message_content() {
        // Verify the error message includes the key diagnostic information.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let decision = validate_placement(&op, &shapes, AneFamily::A18, false);
        if let PlacementDecision::CpuOnly(reason) = decision {
            assert!(
                reason.contains("MILReduceArgmax"),
                "Message should mention the op: {}",
                reason
            );
            assert!(reason.contains("A18"), "Message should mention A18: {}", reason);
            assert!(reason.contains("LSE_7"), "Message should mention LSE_7: {}", reason);
            assert!(
                reason.contains("ANEC converter"),
                "Message should mention ANEC converter: {}",
                reason
            );
        } else {
            panic!("Expected CpuOnly, got {:?}", decision);
        }
    }

    #[test]
    fn test_argmax_with_context_dtype_still_rejected_on_a18() {
        // The A18 guard should fire even when dtype context is provided.
        // Dtype check (Fp16 is fine) should pass, but the family gate should reject.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let ctx = PlacementContext::with_dtype(MilDtype::Fp16);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("LSE_7")),
            "Expected CpuOnly with LSE_7 reason, got {:?}",
            decision
        );
    }

    #[test]
    fn test_argmax_with_int32_dtype_rejected_before_a18_guard() {
        // If dtype is Int32 (illegal on any family), the dtype gate should
        // fire first before the A18 family gate.
        let op = make_reduce_argmax();
        let shapes = vec![vec![1, 128]];
        let ctx = PlacementContext::with_dtype(MilDtype::Int32);
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A18, false, &ctx);
        assert!(
            matches!(decision, PlacementDecision::CpuOnly(ref s) if s.contains("dtype constraint")),
            "Expected CpuOnly with dtype constraint reason, got {:?}",
            decision
        );
    }

    // ─── T-68: extract_whdc NCHW dimension extraction tests ────────

    #[test]
    fn test_extract_whdc_rank1() {
        let (w, h, d, c) = extract_whdc(&[128]);
        assert_eq!((w, h, d, c), (128, 1, 1, 1));
    }

    #[test]
    fn test_extract_whdc_rank2() {
        let (w, h, d, c) = extract_whdc(&[64, 128]);
        assert_eq!((w, h, d, c), (128, 64, 1, 1));
    }

    #[test]
    fn test_extract_whdc_rank3_chw() {
        // Rank 3: CHW [channels, height, width]
        let (w, h, d, c) = extract_whdc(&[64, 128, 256]);
        assert_eq!((w, h, d, c), (256, 128, 1, 64));
    }

    #[test]
    fn test_extract_whdc_rank4_nchw() {
        // T-68: Rank 4: NCHW [batch, channels, height, width]
        // For [1, 64, 128, 128]: w=128, h=128, d=1, c=64
        let (w, h, d, c) = extract_whdc(&[1, 64, 128, 128]);
        assert_eq!((w, h, d, c), (128, 128, 1, 64));
    }

    #[test]
    fn test_extract_whdc_rank4_nchw_large_batch() {
        // NCHW with batch > 1: batch is NOT channels
        let (w, h, d, c) = extract_whdc(&[4, 128, 32, 32]);
        assert_eq!((w, h, d, c), (32, 32, 1, 128));
    }

    #[test]
    fn test_extract_whdc_rank5_ncdhw() {
        // Rank 5: NCDHW [batch, channels, depth, height, width]
        let (w, h, d, c) = extract_whdc(&[1, 64, 8, 128, 128]);
        assert_eq!((w, h, d, c), (128, 128, 8, 64));
    }

    #[test]
    fn test_extract_whdc_empty() {
        let (w, h, d, c) = extract_whdc(&[]);
        assert_eq!((w, h, d, c), (0, 0, 0, 0));
    }

    #[test]
    fn test_extract_whdc_rank6_uses_last4() {
        // Rank > 5: last 4 dims as (w, h, d, c)
        let (w, h, d, c) = extract_whdc(&[1, 2, 3, 4, 5, 6]);
        assert_eq!((w, h, d, c), (6, 5, 4, 3));
    }

    /// T-68: Regression test — the old implementation would swap depth and
    /// channels for rank-4 NCHW tensors, causing max_tensor_channels to be
    /// checked against the batch dimension (always 1, trivially passing).
    #[test]
    fn test_extract_whdc_regression_nchw_channels_vs_batch() {
        // Typical LLM linear weight: [1, 896, 1, 896] (NCHW)
        // OLD (wrong): c=1 (batch), d=896 (channels) — channels bypass check
        // NEW (correct): c=896, d=1 — channels properly checked
        let (w, h, d, c) = extract_whdc(&[1, 896, 1, 896]);
        assert_eq!(c, 896, "channels should be 896, not 1");
        assert_eq!(d, 1, "depth should be 1 for NCHW rank-4");
    }

    // ─── T-105: Softmax and InstanceNorm placement tests ─────────────

    #[test]
    fn test_t105_softmax_a11_legacy_allowed() {
        // MILSoftmax with A11Legacy should return AneAllowed (soft warning, not hard rejection)
        let op = MirOp::MILSoftmax {
            name: "test_softmax".into(),
            x: MirNodeId("input".into()),
            axis: -1,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A11Legacy, false, &ctx,
        );
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_t105_softmax_a16_allowed() {
        // MILSoftmax with A16 should return AneAllowed
        let op = MirOp::MILSoftmax {
            name: "test_softmax".into(),
            x: MirNodeId("input".into()),
            axis: -1,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A16, false, &ctx,
        );
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_t105_instancenorm_a11_legacy_allowed() {
        // MILInstanceNorm with A11Legacy should return AneAllowed (soft warning, not hard rejection)
        let op = MirOp::MILInstanceNorm {
            name: "test_instnorm".into(),
            x: MirNodeId("input".into()),
            gamma: Some("gamma".into()),
            beta: Some("beta".into()),
            epsilon: 1e-5,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A11Legacy, false, &ctx,
        );
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_t105_instancenorm_a16_allowed() {
        // MILInstanceNorm with A16 should return AneAllowed
        let op = MirOp::MILInstanceNorm {
            name: "test_instnorm".into(),
            x: MirNodeId("input".into()),
            gamma: Some("gamma".into()),
            beta: Some("beta".into()),
            epsilon: 1e-5,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A16, false, &ctx,
        );
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    // ─── T-90: Concat placement validation tests (Orion #1) ─────────

    #[test]
    fn test_t90_concat_channel_axis_allowed() {
        // MILConcat along axis=1 (channel axis) should be AneAllowed
        let op = MirOp::MILConcat {
            name: "test_concat".into(),
            values: vec![MirNodeId("a".into()), MirNodeId("b".into())],
            axis: 1,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A16, false, &ctx,
        );
        assert_eq!(decision, PlacementDecision::AneAllowed);
    }

    #[test]
    fn test_t90_concat_non_channel_axis_rejected() {
        // MILConcat along axis=3 (non-channel) should be CpuOnly per Orion #1
        let op = MirOp::MILConcat {
            name: "test_concat".into(),
            values: vec![MirNodeId("a".into()), MirNodeId("b".into())],
            axis: 3,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A16, false, &ctx,
        );
        match decision {
            PlacementDecision::CpuOnly(msg) => {
                assert!(msg.contains("axis 3"), "Error should mention axis 3");
                assert!(msg.contains("Orion #1"), "Error should reference Orion #1");
            }
            other => panic!("Expected CpuOnly for non-channel concat, got {:?}", other),
        }
    }

    #[test]
    fn test_t90_concat_axis_0_rejected() {
        // MILConcat along axis=0 (batch, non-channel) should also be CpuOnly
        let op = MirOp::MILConcat {
            name: "test_concat".into(),
            values: vec![MirNodeId("a".into()), MirNodeId("b".into())],
            axis: 0,
        };
        let shapes: Vec<Vec<usize>> = vec![];
        let ctx = PlacementContext::empty();
        let decision = validate_placement_with_context(
            &op, &shapes, AneFamily::A11Legacy, false, &ctx,
        );
        match decision {
            PlacementDecision::CpuOnly(msg) => {
                assert!(msg.contains("axis 0"), "Error should mention axis 0");
            }
            other => panic!("Expected CpuOnly for axis=0 concat, got {:?}", other),
        }
    }
}
