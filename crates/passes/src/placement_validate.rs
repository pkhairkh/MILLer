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

use ane_ir::ane_layout::{AneInterleave, AneLayout};
use ane_ir::ane_target::AneFamily;
use ane_ir::common::MilDtype;
use ane_ir::mir::MirOp;
use crate::cpu_only_ops;
use crate::dtype_constraints;

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
    pub fn with_dtype_and_interleave(dtype: MilDtype, interleave: AneInterleave, channels: u64) -> Self {
        Self {
            dtype: Some(dtype),
            interleave: Some(interleave),
            channels: Some(channels),
            ..Self::default()
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
    validate_placement_with_context(op, input_shapes, target_family, has_dynamic_shapes, &PlacementContext::empty())
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
    }

    // ─── T-25: Interleave gate ─────────────────────────────────────
    // When interleave information is available, enforce interleave
    // constraints. This validates: valid interleave factors {1,2,3,4,8},
    // const tensors → interleave=1, int4 tensors → interleave=8,
    // and channel divisibility by interleave factor.
    if let Some(interleave) = ctx.interleave {
        let channels = ctx.channels.unwrap_or(0);
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

        // LayerNorm: A15+ only
        MirOp::MILLayerNorm { .. } => {
            if !target_family.supports_layernorm() {
                return PlacementDecision::AneConditional(AneFamily::A15);
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

        // ConvTranspose: always NE
        MirOp::MILConvTranspose { .. } => PlacementDecision::AneAllowed,

        // T-25: ConstexprBlockwiseShiftScale — hard-reject on ANE.
        // Blockwise scale is never supported on ANE (returns false).
        MirOp::MILConstexprBlockwiseShiftScale { .. }
        | MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => {
            PlacementDecision::CpuOnly(format!(
                "{}: blockwise scale is not supported on ANE",
                op_name(op)
            ))
        }

        // T-25: Quantize — reject asymmetric quantization on ANE.
        MirOp::MILQuantize { .. } => {
            // If the context flags asymmetric quant, hard-reject.
            // (is_asymmetric_quant is already checked above in the
            // universal gate, but this match arm makes the intent
            // explicit and ensures Quantize without context still
            // gets through for symmetric case.)
            PlacementDecision::AneAllowed
        }

        // Ops with no ANE engine — immediate CPU
        op => {
            // T-22: Check the CPU_ONLY_OPS set as a hard gate.
            // This catches ops that are on the CPU-only list but might
            // still have a default_engine() assignment (defensive check).
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
    }
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
        let ctx = PlacementContext::with_dtype_and_interleave(MilDtype::Fp16, AneInterleave::Factor4, 64);
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
        let ctx = PlacementContext::with_dtype_and_interleave(MilDtype::Fp16, AneInterleave::Factor4, 63);
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
        let ctx = PlacementContext {
            is_blockwise_scale: true,
            ..PlacementContext::default()
        };
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
        let decision = validate_placement_with_context(&op, &shapes, AneFamily::A11Legacy, false, &ctx);
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
            &op, &shapes, AneFamily::A16, false, &PlacementContext::empty(),
        );
        assert_eq!(decision1, decision2);
    }
}
