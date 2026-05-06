//! MIL-Emission IR (MIR)
//!
//! One-to-one mapping to Core ML MIL Builder calls.
//! Each MIR node corresponds to exactly one `mb.<op>()` call.
//!
//! ## MIR Coverage — 167/167 ops (100%)
//!
//! Every documented Core ML MIL operation is represented as a MirOp variant.
//! All ops are wired through SIR → AIR → MIR lowering paths.

use serde::{Deserialize, Serialize};

use super::common::IrNodeId;
pub use super::common::{ComputeUnitHint, MilDtype};
use crate::toproto::ToProto;

/// Target-specific annotation for a MIR op node.
///
/// Captures ANE placement attributes that are determined after
/// engine assignment but before emission. This separates target
/// concerns from the pure IR representation.
///
/// T-P5-08: ANE-specific quantization attributes (kernel_scale,
/// kernel_zero_point, kernel_palettized_lut) have been moved here from
/// the MirOp::MILConv variant, and palette_bits from SirOp variants
/// has been moved to `SirTargetAnnotation`. The IR should be
/// target-agnostic; ANE-specific metadata lives in the annotation layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MirOpTargetAnnotation {
    /// The assigned ANE engine for this op, if any.
    pub assigned_engine: Option<super::ane_engine::AneEngine>,
    /// Whether this op was demoted from ANE to CPU by a family override.
    pub demoted_to_cpu: bool,
    /// The target ANE revision, if known.
    pub target_revision: Option<super::ane_target::AneRevision>,
    /// T-P5-08: ANE quantization metadata for conv weights.
    ///
    /// Previously embedded directly in `MirOp::MILConv` as `kernel_scale`,
    /// `kernel_zero_point`, and `kernel_palettized_lut`. Moved here because
    /// these are ANEC-specific emission attributes, not part of the core IR.
    /// Populated by `palettize_weights::populate_conv_quantization_fields()`.
    #[serde(default)]
    pub ane_quant: Option<AneQuantMetadata>,
}

/// T-P5-08: ANE-specific quantization metadata for convolution weights.
///
/// Captures the ANEC `kernel_scale`, `kernel_zero_point`, and
/// `kernel_palettized_LUT` attributes needed for quantized/palettized
/// convolution emission on the ANE. These attributes are not part of
/// the core IR — they are ANE target-layer concerns populated during
/// the palettization pass and consumed during ANE lowering/emission.
///
/// This struct was previously represented as inline fields on
/// `MirOp::MILConv` (T-98). T-P5-08 moves them to the target
/// annotation layer so that the IR remains target-agnostic and
/// can represent non-ANE targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AneQuantMetadata {
    /// Per-op scale factor for quantized/palettized conv weights.
    /// Maps to ANEC `kernel_scale` attribute. Used to dequantize
    /// int4/int8 weights back to floating point.
    pub kernel_scale: f32,
    /// Zero-point offset for quantized conv weights.
    /// Maps to ANEC `kernel_zero_point` attribute. Zero indicates
    /// symmetric quantization.
    pub kernel_zero_point: i32,
    /// Name of the palettized LUT weight entry for this conv.
    /// Maps to ANEC `kernel_palettized_LUT` attribute. References a
    /// weight entry in the weight blob that contains the lookup table
    /// for palettized (kmeans) weights.
    pub kernel_palettized_lut: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirNodeId(pub String);

impl IrNodeId for MirNodeId {
    fn as_str(&self) -> &str {
        &self.0
    }

    fn from_string(s: String) -> Self {
        MirNodeId(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MirOp {
    // ─── Constants ───────────────────────────────────────────────
    MILConst {
        name: String,
        value_path: String,
        dtype: MilDtype,
    },

    // ─── Linear / FC ─────────────────────────────────────────────
    MILLinear {
        name: String,
        x: MirNodeId,
        weight: String,
        bias: Option<String>,
    },
    MILMatMul {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
        transpose_y: bool,
    },
    MILEinsum {
        name: String,
        inputs: Vec<MirNodeId>,
        equation: String,
    },

    // ─── Convolution ─────────────────────────────────────────────
    MILConv {
        name: String,
        x: MirNodeId,
        weight: MirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
        // T-P5-08: kernel_scale, kernel_zero_point, kernel_palettized_lut
        // moved to MirOpTargetAnnotation::ane_quant (AneQuantMetadata).
        // The IR is now target-agnostic; ANE quant metadata is in the
        // target annotation layer, populated by palettize_weights pass.
    },
    MILConvTranspose {
        name: String,
        x: MirNodeId,
        weight: MirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
        output_shape: Vec<usize>,
    },

    // ─── Elementwise Binary ──────────────────────────────────────
    MILAdd {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILMul {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILSub {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILMaximum {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILMinimum {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILRealDiv {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILFloorDiv {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILMod {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILPow {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILEqual {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILNotEqual {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILGreater {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILGreaterEqual {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILLess {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILLessEqual {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILLogicalAnd {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILLogicalOr {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILLogicalXor {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
    },

    // ─── Elementwise Unary ───────────────────────────────────────
    MILAbs {
        name: String,
        x: MirNodeId,
    },
    MILNeg {
        name: String,
        x: MirNodeId,
    },
    MILSigmoid {
        name: String,
        x: MirNodeId,
    },
    MILTanh {
        name: String,
        x: MirNodeId,
    },
    MILRelu {
        name: String,
        x: MirNodeId,
    },
    MILRelu6 {
        name: String,
        x: MirNodeId,
    },
    MILLeakyRelu {
        name: String,
        x: MirNodeId,
        alpha: f32,
    },
    MILSigmoidHard {
        name: String,
        x: MirNodeId,
        alpha: f32,
        beta: f32,
    },
    MILThresholdedRelu {
        name: String,
        x: MirNodeId,
        alpha: f32,
    },
    MILClampedRelu {
        name: String,
        x: MirNodeId,
        alpha: f32,
        beta: f32,
    },
    MILLinearActivation {
        name: String,
        x: MirNodeId,
        alpha: f32,
        beta: f32,
    },
    MILPrelu {
        name: String,
        x: MirNodeId,
        alpha: String,
    },
    MILSoftsign {
        name: String,
        x: MirNodeId,
    },
    MILSilu {
        name: String,
        x: MirNodeId,
    },
    MILScaledTanh {
        name: String,
        x: MirNodeId,
        alpha: f32,
        beta: f32,
    },
    MILElu {
        name: String,
        x: MirNodeId,
        alpha: f32,
    },
    MILSoftplus {
        name: String,
        x: MirNodeId,
    },
    MILSoftplusParametric {
        name: String,
        x: MirNodeId,
        alpha: String,
        beta: String,
    },
    MILGelu {
        name: String,
        x: MirNodeId,
        mode: String,
    },
    MILClip {
        name: String,
        x: MirNodeId,
        min_val: f32,
        max_val: f32,
    },
    MILSquare {
        name: String,
        x: MirNodeId,
    },
    MILThreshold {
        name: String,
        x: MirNodeId,
        alpha: f32,
    },
    MILSqrt {
        name: String,
        x: MirNodeId,
    },
    MILRsqrt {
        name: String,
        x: MirNodeId,
    },
    MILInverse {
        name: String,
        x: MirNodeId,
        epsilon: f32,
    },
    MILCeil {
        name: String,
        x: MirNodeId,
    },
    MILFloor {
        name: String,
        x: MirNodeId,
    },
    MILRound {
        name: String,
        x: MirNodeId,
    },
    MILExp {
        name: String,
        x: MirNodeId,
    },
    MILExp2 {
        name: String,
        x: MirNodeId,
    },
    MILLog {
        name: String,
        x: MirNodeId,
        epsilon: f32,
    },
    MILSign {
        name: String,
        x: MirNodeId,
    },
    MILCos {
        name: String,
        x: MirNodeId,
    },
    MILSin {
        name: String,
        x: MirNodeId,
    },
    MILTan {
        name: String,
        x: MirNodeId,
    },
    MILAcos {
        name: String,
        x: MirNodeId,
    },
    MILAsin {
        name: String,
        x: MirNodeId,
    },
    MILAtan {
        name: String,
        x: MirNodeId,
    },
    MILCosh {
        name: String,
        x: MirNodeId,
    },
    MILSinh {
        name: String,
        x: MirNodeId,
    }, // Sprint 58 (S58.6): renamed from MLSinh (was missing the I in MIL prefix)
    MILAtanh {
        name: String,
        x: MirNodeId,
    }, // Sprint 58 (S58.6): renamed from MILTanhInverse
    MILErf {
        name: String,
        x: MirNodeId,
    },
    MILLogicalNot {
        name: String,
        x: MirNodeId,
    },
    MILCast {
        name: String,
        x: MirNodeId,
        dtype: MilDtype,
    },
    MILSelect {
        name: String,
        condition: MirNodeId,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILWhere {
        name: String,
        condition: MirNodeId,
        x: MirNodeId,
        y: MirNodeId,
    },
    MILSoftmax {
        name: String,
        x: MirNodeId,
        axis: isize,
    },

    // ─── Reduction ───────────────────────────────────────────────
    MILReduceSum {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceMean {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceMax {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceMin {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceProd {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceSumSquare {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceL2Norm {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceL1Norm {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceLogSumExp {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceLogSum {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    MILReduceArgmax {
        name: String,
        x: MirNodeId,
        axis: usize,
        keep_dims: bool,
    },
    MILReduceArgmin {
        name: String,
        x: MirNodeId,
        axis: usize,
        keep_dims: bool,
    },

    // ─── Normalization ───────────────────────────────────────────
    MILBatchNorm {
        name: String,
        x: MirNodeId,
        mean: String,
        variance: String,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    MILInstanceNorm {
        name: String,
        x: MirNodeId,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    MILLayerNorm {
        name: String,
        x: MirNodeId,
        weight: String,
        bias: Option<String>,
        epsilon: f32,
        axes: Vec<usize>,
    },
    MILL2Norm {
        name: String,
        x: MirNodeId,
        epsilon: f32,
        axes: Vec<usize>,
    },
    MILLocalResponseNorm {
        name: String,
        x: MirNodeId,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    },

    // ─── Pooling ─────────────────────────────────────────────────
    MILMaxPool {
        name: String,
        x: MirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },
    MILAvgPool {
        name: String,
        x: MirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
        count_include_padding: bool,
    },
    MILL2Pool {
        name: String,
        x: MirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },

    // ─── Image Resizing ──────────────────────────────────────────
    MILResize {
        name: String,
        x: MirNodeId,
        target_size: Vec<usize>,
        mode: String,
        sampling_mode: String,
        nearest_rounding_mode: String,
    },
    MILResizeNearestNeighbor {
        name: String,
        x: MirNodeId,
        target_height: usize,
        target_width: usize,
    },
    MILResizeBilinear {
        name: String,
        x: MirNodeId,
        target_height: usize,
        target_width: usize,
        align_corners: bool,
    },
    MILUpsampleNearestNeighbor {
        name: String,
        x: MirNodeId,
        scale: Vec<usize>,
    },
    MILUpsampleBilinear {
        name: String,
        x: MirNodeId,
        scale: Vec<usize>,
        align_corners: bool,
        half_pixel_centers: bool,
    },
    MILCropResize {
        name: String,
        x: MirNodeId,
        boxes: MirNodeId,
        box_indices: MirNodeId,
        crop_height: usize,
        crop_width: usize,
    },
    MILAffine {
        name: String,
        x: MirNodeId,
        transform: MirNodeId,
        output_height: usize,
        output_width: usize,
        sampling_mode: String,
        pad_value: f32,
    },
    MILResample {
        name: String,
        x: MirNodeId,
        coordinates: MirNodeId,
        sampling_mode: String,
        pad_value: f32,
    },

    // ─── Tensor Transform ────────────────────────────────────────
    MILReshape {
        name: String,
        x: MirNodeId,
        shape: Vec<usize>,
    },
    MILReshapeLike {
        name: String,
        x: MirNodeId,
        ref_tensor: MirNodeId,
    },
    MILTranspose {
        name: String,
        x: MirNodeId,
        perm: Vec<usize>,
    },
    MILSplit {
        name: String,
        x: MirNodeId,
        axis: usize,
        num_splits: usize,
    },
    MILConcat {
        name: String,
        values: Vec<MirNodeId>,
        axis: usize,
    },
    MILExpandDims {
        name: String,
        x: MirNodeId,
        axis: Vec<usize>,
    },
    MILSqueeze {
        name: String,
        x: MirNodeId,
        axis: Vec<usize>,
    },
    MILFlatten2d {
        name: String,
        x: MirNodeId,
        axis: usize,
    },
    MILReverse {
        name: String,
        x: MirNodeId,
        axes: Vec<usize>,
    },
    MILReverseSequence {
        name: String,
        x: MirNodeId,
        lengths: MirNodeId,
        batch_axis: usize,
        seq_axis: usize,
    },
    MILSliceByIndex {
        name: String,
        x: MirNodeId,
        begin: Vec<i64>,
        end: Vec<i64>,
        stride: Vec<i64>,
        begin_mask: Vec<bool>,
        end_mask: Vec<bool>,
        squeeze_mask: Vec<bool>,
    },
    MILSliceBySize {
        name: String,
        x: MirNodeId,
        begin: Vec<i64>,
        size: Vec<i64>,
    },
    MILSliceUpdate {
        name: String,
        x: MirNodeId,
        update: MirNodeId,
        begin: Vec<i64>,
        end: Vec<i64>,
    },
    MILSlidingWindows {
        name: String,
        x: MirNodeId,
        axis: usize,
        window_size: usize,
        stride: usize,
    },
    MILDepthToSpace {
        name: String,
        x: MirNodeId,
        block_size: usize,
    },
    MILSpaceToDepth {
        name: String,
        x: MirNodeId,
        block_size: usize,
    },
    MILPixelShuffle {
        name: String,
        x: MirNodeId,
        upscale_factor: usize,
    },
    MILPixelUnshuffle {
        name: String,
        x: MirNodeId,
        downscale_factor: usize,
    },
    MILBatchToSpace {
        name: String,
        x: MirNodeId,
        block_shape: Vec<usize>,
        crops: Vec<(usize, usize)>,
    },
    MILSpaceToBatch {
        name: String,
        x: MirNodeId,
        block_shape: Vec<usize>,
        paddings: Vec<(usize, usize)>,
    },
    MILPad {
        name: String,
        x: MirNodeId,
        pad_amounts: Vec<i64>,
        mode: String,
        constant_value: f32,
    },
    MILStack {
        name: String,
        values: Vec<MirNodeId>,
        axis: usize,
    },
    MILTile {
        name: String,
        x: MirNodeId,
        reps: Vec<usize>,
    },
    MILCumsum {
        name: String,
        x: MirNodeId,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    },
    MILFill {
        name: String,
        shape: Vec<usize>,
        value: f32,
        dtype: MilDtype,
    },
    MILFillLike {
        name: String,
        ref_tensor: MirNodeId,
        value: f32,
        dtype: MilDtype,
    },
    MILIdentity {
        name: String,
        x: MirNodeId,
    },
    MILOneHot {
        name: String,
        indices: MirNodeId,
        one_hot_vector_size: usize,
        on_value: f32,
        off_value: f32,
        axis: usize,
        dtype: MilDtype,
    },
    MILNonZero {
        name: String,
        x: MirNodeId,
    },
    MILArgsort {
        name: String,
        x: MirNodeId,
        axis: usize,
        ascending: bool,
    },
    MILBandPart {
        name: String,
        x: MirNodeId,
        num_lower: i64,
        num_upper: i64,
    },
    MILRange1d {
        name: String,
        start: f32,
        end: f32,
        step: f32,
    },
    MILShape {
        name: String,
        x: MirNodeId,
    },
    MILCrop {
        name: String,
        x: MirNodeId,
        crop_height: usize,
        crop_width: usize,
        offset_height: usize,
        offset_width: usize,
    },

    // ─── Scatter / Gather ────────────────────────────────────────
    MILGather {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
        axis: isize,
    },
    MILGatherAlongAxis {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
        axis: isize,
    },
    MILGatherNd {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
    },
    MILScatter {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
        updates: MirNodeId,
        axis: isize,
        mode: String,
    },
    MILScatterAlongAxis {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
        updates: MirNodeId,
        axis: isize,
    },
    MILScatterNd {
        name: String,
        x: MirNodeId,
        indices: MirNodeId,
        updates: MirNodeId,
    },
    MILNonMaximumSuppression {
        name: String,
        boxes: MirNodeId,
        scores: MirNodeId,
        iou_threshold: f32,
        score_threshold: f32,
        max_detections: usize,
    },

    // ─── Attention ───────────────────────────────────────────────
    MILScaledDotProductAttention {
        name: String,
        query: MirNodeId,
        key: MirNodeId,
        value: MirNodeId,
        attention_mask: Option<MirNodeId>,
        scale: Option<f32>,
    },

    // ─── Quantization ────────────────────────────────────────────
    MILQuantize {
        name: String,
        x: MirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: MilDtype,
    },
    MILDequantize {
        name: String,
        x: MirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: MilDtype,
    },

    // ─── Constexpr / Compression ─────────────────────────────────
    MILConstexprAffineDequantize {
        name: String,
        quantized_data: String,
        scale: f32,
        zero_point: i32,
        axis: isize,
    },
    MILConstexprBlockwiseShiftScale {
        name: String,
        data: String,
        scale: String,
        offset: String,
        block_size: Vec<usize>,
    },
    MILConstexprLutToDense {
        name: String,
        indices: String,
        lut: String,
        num_bits: usize,
    },
    MILConstexprSparseToDense {
        name: String,
        nonzero_data: String,
        shape: Vec<usize>,
        default_value: f32,
    },
    MILConstexprCast {
        name: String,
        data: String,
        dtype: MilDtype,
    },
    MILConstexprLutToSparse {
        name: String,
        data: String,
        num_bits: usize,
    },
    MILConstexprSparseBlockwiseShiftScale {
        name: String,
        data: String,
        scale: String,
        offset: String,
        block_size: Vec<usize>,
        block_axis: usize,
    },

    // ─── Recurrent ───────────────────────────────────────────────
    MILRnn {
        name: String,
        x: MirNodeId,
        initial_h: MirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        mode: String,
        output_sequence: bool,
    },
    MILGru {
        name: String,
        x: MirNodeId,
        initial_h: MirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        reset_after: bool,
        output_sequence: bool,
    },
    MILLstm {
        name: String,
        x: MirNodeId,
        initial_h: MirNodeId,
        initial_c: MirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        output_sequence: bool,
    },

    // ─── Control Flow ────────────────────────────────────────────
    MILCond {
        name: String,
        pred: MirNodeId,
        true_graph: String,
        false_graph: String,
    },
    MILWhileLoop {
        name: String,
        condition: String,
        body: String,
        loop_vars: Vec<MirNodeId>,
    },
    MILMakeList {
        name: String,
        elems: Vec<MirNodeId>,
        dtype: MilDtype,
    },
    MILListLength {
        name: String,
        ls: MirNodeId,
    },
    MILListWrite {
        name: String,
        ls: MirNodeId,
        index: MirNodeId,
        value: MirNodeId,
    },
    MILListRead {
        name: String,
        ls: MirNodeId,
        index: MirNodeId,
    },
    MILListGather {
        name: String,
        ls: MirNodeId,
        indices: MirNodeId,
    },
    MILListScatter {
        name: String,
        ls: MirNodeId,
        indices: MirNodeId,
        values: MirNodeId,
    },

    // ─── Random ──────────────────────────────────────────────────
    MILRandomBernoulli {
        name: String,
        shape: Vec<usize>,
        prob: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    MILRandomNormal {
        name: String,
        shape: Vec<usize>,
        mean: f32,
        stddev: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    MILRandomUniform {
        name: String,
        shape: Vec<usize>,
        low: f32,
        high: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    MILRandomCategorical {
        name: String,
        logits: MirNodeId,
        num_samples: usize,
        seed: Option<u64>,
        dtype: MilDtype,
    },

    // ─── State ───────────────────────────────────────────────────
    MILReadState {
        name: String,
        state_id: String,
        shape: Vec<usize>,
        dtype: MilDtype,
    },
    MILCoremlUpdateState {
        name: String,
        state_id: String,
        value: MirNodeId,
    },
    MILStateWrite {
        name: String,
        state_ref: String,
        value: MirNodeId,
    },

    // ─── Metadata / Misc ─────────────────────────────────────────
    MILTopk {
        name: String,
        x: MirNodeId,
        k: usize,
        axis: isize,
    },
    MILClassify {
        name: String,
        x: MirNodeId,
    },

    // ─── ANEC Internal Ops (T-P4-08) ────────────────────────────────
    // These represent operations performed by the ANEC binary internally.
    // They are not exposed as MIL builder calls and are used for
    // internal tracking and debugging only.
    /// ANEC-internal: fused conv+activation operation.
    /// F-OPS-01: Promoted from CPU-only stub — now carries conv parameters
    /// for MIR-to-proto emission (decomposed to conv + activation in proto).
    AnecFusedConvActivate {
        name: String,
        x: MirNodeId,
        weight: MirNodeId,
        activation: String,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
        groups: usize,
    },
    /// ANEC-internal: fused linear+activation operation.
    AnecFusedLinearActivate {
        name: String,
        x: MirNodeId,
        weight: String,
        activation: String,
    },
    // ─── T-P4-08: Unmapped ANEC operation stubs ────────────────────
    // These 14 ops are genuinely unmapped ANEC operations identified
    // during forensic analysis. They are CPU-only stubs — no ANEC
    // converter exists for any family.
    /// ANEC-internal: broadcast operation.
    AnecBroadcast {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: scaled elementwise operation.
    /// F-OPS-01: Promoted from CPU-only stub — now carries per-operand
    /// scaling factors for MIR-to-proto emission (decomposed to
    /// mul(x, scale_a) + mul(y, scale_b) in proto).
    AnecScaledElementwise {
        name: String,
        x: MirNodeId,
        y: MirNodeId,
        scale_a: f32,
        scale_b: f32,
    },
    /// ANEC-internal: global arg min/max operation.
    AnecGlobalArgMinMax {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: degamma operation.
    AnecDegamma {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: Dirac delta operation.
    AnecDirac {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: gain/offset control operation.
    AnecGainOffsetControl {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: N-ReLU operation.
    AnecNRelu {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: high-precision sigmoid operation.
    AnecHighPrecisionSigmoid {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: log2 operation.
    AnecLog2 {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: trunc(ate) operation.
    AnecTrunc {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: invert operation.
    AnecInvert {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: unflatten operation.
    AnecUnflatten {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: channel-to-space operation.
    AnecChannelToSpace {
        name: String,
        x: MirNodeId,
    },
    /// ANEC-internal: space-to-channel operation.
    AnecSpaceToChannel {
        name: String,
        x: MirNodeId,
    },
}

impl MirOp {
    /// Returns the base ANE engine assignment for this op, ignoring revision-specific
    /// constraints. This is the static per-op engine mapping used before applying
    /// family capability overrides.
    ///
    /// **Deprecated** (T-P5-07): Use `ane_placement::engine_for_op()` instead,
    /// which provides revision-aware placement with family-specific overrides.
    /// This method will be removed in a future version once all callers are
    /// migrated.
    #[deprecated(
        since = "0.7.0",
        note = "T-P5-07: Use ane_placement::engine_for_op() for revision-aware engine assignment. \
                This method only provides the static per-op mapping without family overrides."
    )]
    pub(crate) fn base_engine(&self) -> Option<super::ane_engine::AneEngine> {
        use super::ane_engine::AneEngine;
        match self {
            // ─── NE pipeline: conv/pool/matmul/attention ────────────
            MirOp::MILLinear { .. }
            | MirOp::MILMatMul { .. }
            // MILEinsum moved to CPU-only (T-22): no ANEC converter.
            | MirOp::MILConv { .. }
            | MirOp::MILConvTranspose { .. }
            | MirOp::MILScaledDotProductAttention { .. }
            // ─── F-OPS-01: ANEC ops promoted from CPU-only ───────────
            // AnecFusedConvActivate: decomposed to conv + activation in proto
            | MirOp::AnecFusedConvActivate { .. }
            // AnecScaledElementwise: decomposed to scaled mul + add in proto
            | MirOp::AnecScaledElementwise { .. }
            | MirOp::MILMaxPool { .. }
            | MirOp::MILAvgPool { .. }
            | MirOp::MILL2Pool { .. }
            | MirOp::MILResample { .. }
            | MirOp::MILResize { .. }
            | MirOp::MILResizeNearestNeighbor { .. }
            | MirOp::MILResizeBilinear { .. }
            | MirOp::MILUpsampleNearestNeighbor { .. }
            | MirOp::MILUpsampleBilinear { .. }
            | MirOp::MILCropResize { .. }
            | MirOp::MILAffine { .. }
            | MirOp::MILDepthToSpace { .. }
            | MirOp::MILSpaceToDepth { .. }
            | MirOp::MILPixelShuffle { .. }
            | MirOp::MILPixelUnshuffle { .. }
            | MirOp::MILBatchToSpace { .. }
            | MirOp::MILSpaceToBatch { .. } => Some(AneEngine::NE),

            // ─── PE pipeline: elementwise/reduction/cast/shape ──────
            //
            // T-22 NOTE: Many ops were moved from PE to the None (CPU-only)
            // branch because they have NO ANEC converter in any ANE family.
            // See the None branch for the complete CPU-only list.
            //
            // Elementwise binary (ANE-legal per per-op support matrix)
            MirOp::MILAdd { .. }
            | MirOp::MILMul { .. }
            | MirOp::MILSub { .. }
            | MirOp::MILMaximum { .. }
            | MirOp::MILMinimum { .. }
            | MirOp::MILRealDiv { .. }
            | MirOp::MILFloorDiv { .. }
            // MILMod moved to CPU-only (T-22): no ANEC converter
            | MirOp::MILPow { .. }
            | MirOp::MILEqual { .. }
            | MirOp::MILNotEqual { .. }
            | MirOp::MILGreater { .. }
            | MirOp::MILGreaterEqual { .. }
            | MirOp::MILLess { .. }
            | MirOp::MILLessEqual { .. }
            // MILLogicalAnd/Or/Xor moved to CPU-only (T-22): no ANEC converter.
            // Comparison ops (Equal, NotEqual, Greater, Less) ARE ANE-legal.
            // Elementwise unary / activations (ANE-legal per per-op matrix)
            | MirOp::MILAbs { .. }
            // MILNeg moved to CPU-only (T-67): no ANEC converter for "neg".
            // Per-op matrix row 197: mps.negative has no ANEC converter.
            // Combined with I-41 fix: was incorrectly in PE branch.
            | MirOp::MILSigmoid { .. }
            | MirOp::MILTanh { .. }
            | MirOp::MILRelu { .. }
            // MILRelu6, SigmoidHard, ThresholdedRelu, ClampedRelu,
            // LinearActivation, Prelu, Softsign, ScaledTanh, Softplus,
            // SoftplusParametric moved to CPU-only (T-22).
            | MirOp::MILLeakyRelu { .. }
            | MirOp::MILSilu { .. }
            | MirOp::MILElu { .. }   // ANE-legal: per-op matrix row 34
            | MirOp::MILGelu { .. }
            // MILClip moved to CPU-only (T-22): no ANEC converter for clamp.
            | MirOp::MILSquare { .. }  // ANE-legal: per-op matrix row 19 (A13Minus/A14Plus split)
            // MILThreshold moved to CPU-only (T-22).
            | MirOp::MILSqrt { .. }
            | MirOp::MILRsqrt { .. }
            // MILInverse moved to CPU-only (T-22).
            | MirOp::MILCeil { .. }
            | MirOp::MILFloor { .. }
            | MirOp::MILRound { .. }
            | MirOp::MILExp { .. }
            | MirOp::MILExp2 { .. }    // ANE-legal: per-op matrix row 22
            | MirOp::MILLog { .. }
            | MirOp::MILSign { .. }
            | MirOp::MILCos { .. }
            | MirOp::MILSin { .. }
            // MILTan, Acos, Asin, Atan, Cosh, Sinh, Atanh moved to CPU-only (T-22).
            | MirOp::MILErf { .. }     // ANE-legal: per-op matrix row 25
            // MILLogicalNot moved to CPU-only (T-22).
            // Cast / softmax
            | MirOp::MILCast { .. }
            | MirOp::MILSoftmax { .. }
            // Reductions (ANE-legal per per-op matrix)
            | MirOp::MILReduceSum { .. }
            | MirOp::MILReduceMean { .. }
            | MirOp::MILReduceMax { .. }
            | MirOp::MILReduceMin { .. }
            | MirOp::MILReduceProd { .. }
            | MirOp::MILReduceSumSquare { .. }
            | MirOp::MILReduceL2Norm { .. }
            | MirOp::MILReduceL1Norm { .. }
            | MirOp::MILReduceLogSumExp { .. }
            | MirOp::MILReduceLogSum { .. }
            | MirOp::MILReduceArgmax { .. }
            | MirOp::MILReduceArgmin { .. }
            // Normalization (ANE-legal per per-op matrix)
            | MirOp::MILBatchNorm { .. }
            | MirOp::MILInstanceNorm { .. }
            | MirOp::MILLayerNorm { .. }
            | MirOp::MILL2Norm { .. }
            | MirOp::MILLocalResponseNorm { .. }
            // Shape / rearrange (ANE-legal per per-op matrix)
            | MirOp::MILReshape { .. }
            | MirOp::MILReshapeLike { .. }
            | MirOp::MILExpandDims { .. }
            | MirOp::MILSqueeze { .. }
            | MirOp::MILFlatten2d { .. }
            | MirOp::MILConcat { .. }
            | MirOp::MILSplit { .. }
            | MirOp::MILStack { .. }
            | MirOp::MILTile { .. }
            | MirOp::MILPad { .. }
            // Slice (ANE-legal)
            | MirOp::MILSliceByIndex { .. }
            | MirOp::MILSliceBySize { .. }
            // MILSliceUpdate moved to CPU-only (T-47): no ANEC converter.
            // MILSlidingWindows moved to CPU-only (T-47): no ANEC converter.
            // MILReverse moved to CPU-only (T-47): no ANEC converter.
            // MILReverseSequence moved to CPU-only (T-22).
            // Quantize / dequantize (ANE-legal per per-op matrix rows 84-85,
            // but currently lack MirOpCompat variants — see T-39)
            | MirOp::MILQuantize { .. }
            | MirOp::MILDequantize { .. }
            // Sort / topk
            // MILArgsort moved to CPU-only (T-47): no ANEC converter.
            | MirOp::MILTopk { .. }
            // MILBandPart moved to CPU-only (T-22).
            // Identity / misc (ANE-legal)
            | MirOp::MILIdentity { .. }
            | MirOp::MILCrop { .. } => Some(AneEngine::PE),

            // ─── TransposeEngine ───────────────────────────────────
            MirOp::MILTranspose { .. } => Some(AneEngine::TransposeEngine),

            // ─── CPU-only: no ANE engine assignment ────────────────
            // Constants, constexpr ops, control flow, random, scatter,
            // state, RNN, classify — these never execute on the ANE.
            MirOp::MILConst { .. }
            | MirOp::MILConstexprAffineDequantize { .. }
            | MirOp::MILConstexprBlockwiseShiftScale { .. }
            | MirOp::MILConstexprLutToDense { .. }
            | MirOp::MILConstexprSparseToDense { .. }
            | MirOp::MILConstexprCast { .. }
            | MirOp::MILConstexprLutToSparse { .. }
            | MirOp::MILConstexprSparseBlockwiseShiftScale { .. }
            | MirOp::MILScatter { .. }
            | MirOp::MILScatterAlongAxis { .. }
            | MirOp::MILScatterNd { .. }
            | MirOp::MILNonMaximumSuppression { .. }
            | MirOp::MILRnn { .. }
            | MirOp::MILGru { .. }
            | MirOp::MILLstm { .. }
            | MirOp::MILCond { .. }
            | MirOp::MILWhileLoop { .. }
            | MirOp::MILMakeList { .. }
            | MirOp::MILListLength { .. }
            | MirOp::MILListWrite { .. }
            | MirOp::MILListRead { .. }
            | MirOp::MILListGather { .. }
            | MirOp::MILListScatter { .. }
            | MirOp::MILRandomBernoulli { .. }
            | MirOp::MILRandomNormal { .. }
            | MirOp::MILRandomUniform { .. }
            | MirOp::MILRandomCategorical { .. }
            | MirOp::MILReadState { .. }
            | MirOp::MILCoremlUpdateState { .. }
            | MirOp::MILStateWrite { .. }
            | MirOp::MILClassify { .. }
            | MirOp::MILCumsum { .. }
            // ANE-illegal conditional/tensor creation ops (no ANE converter)
            // select / where: Despite per-op matrix row 69, empirical testing shows
            //   mb.select causes CPU fallback. Decompose to arithmetic instead:
            //   select(cond, a, b) → cond*a + (1-cond)*b
            // fill: ANE has no fill converter; use precomputed Const instead
            // fill_like: ANE has no fill_like converter; decomposed to mul+add
            // one_hot: ANE has no one_hot converter
            // non_zero: ANE has no non_zero converter
            // range1d: ANE has no range converter
            // shape: ANE has no shape query converter
            | MirOp::MILSelect { .. }
            | MirOp::MILWhere { .. }
            | MirOp::MILFill { .. }
            | MirOp::MILFillLike { .. }
            | MirOp::MILOneHot { .. }
            | MirOp::MILNonZero { .. }
            | MirOp::MILRange1d { .. }
            | MirOp::MILShape { .. }
            // Gather ops: CPU-only due to ANE plannability ~0.26.
            // Only embedding uses Gather (runs on CPU anyway).
            | MirOp::MILGather { .. }
            | MirOp::MILGatherAlongAxis { .. }
            | MirOp::MILGatherNd { .. }
            // ─── T-22: CPU-only ops moved from PE/NE pipeline ──────
            // These ops have NO ANEC converter in any ANE family.
            // Source: ane-constraints-docs/04-operation-support/ per-op matrix
            // and the CPU_ONLY_OPS set in ane-passes/src/cpu_only_ops.rs.
            //
            // Trig inverse / hyperbolic: no ANEC converter
            | MirOp::MILAcos { .. }
            | MirOp::MILAsin { .. }
            | MirOp::MILAtan { .. }
            | MirOp::MILAtanh { .. }
            | MirOp::MILTan { .. }
            | MirOp::MILCosh { .. }
            | MirOp::MILSinh { .. }
            // Logical: no ANEC converter for logical_and/or/xor/not
            // (comparison ops equal/not_equal/greater/less ARE ANE-legal)
            | MirOp::MILLogicalAnd { .. }
            | MirOp::MILLogicalOr { .. }
            | MirOp::MILLogicalXor { .. }
            | MirOp::MILLogicalNot { .. }
            // Activation variants with no ANEC converter:
            // relu6, sigmoid_hard, thresholded_relu, clamped_relu,
            // linear_activation, prelu, softsign, scaled_tanh, softplus,
            // softplus_parametric — none appear in the per-op support matrix.
            | MirOp::MILRelu6 { .. }
            | MirOp::MILSigmoidHard { .. }
            | MirOp::MILThresholdedRelu { .. }
            | MirOp::MILClampedRelu { .. }
            | MirOp::MILLinearActivation { .. }
            | MirOp::MILPrelu { .. }
            | MirOp::MILSoftsign { .. }
            | MirOp::MILScaledTanh { .. }
            | MirOp::MILSoftplus { .. }
            | MirOp::MILSoftplusParametric { .. }
            // Other elementwise with no ANEC converter:
            // threshold, inverse, modulo, clamp — not in per-op matrix.
            | MirOp::MILThreshold { .. }
            | MirOp::MILInverse { .. }
            | MirOp::MILMod { .. }
            | MirOp::MILClip { .. }
            // Miscellaneous CPU-only:
            // band_part, reverse_sequence, einsum — no ANEC converter.
            | MirOp::MILBandPart { .. }
            | MirOp::MILReverseSequence { .. }
            // Einsum: no ANEC converter in any family.
            | MirOp::MILEinsum { .. }
            // ─── T-47: Ops with PE engine but no ANEC converter ──────
            // These were incorrectly assigned Some(AneEngine::PE) but map
            // to MirOpCompat::Unsupported at emission time. Moving to
            // CPU-only prevents silent emission failures.
            | MirOp::MILSliceUpdate { .. }
            | MirOp::MILSlidingWindows { .. }
            | MirOp::MILReverse { .. }
            | MirOp::MILArgsort { .. }
            // ─── T-67: MILNeg has no ANEC converter ──────────────────
            // Per-op matrix row 197: mps.negative has no ANEC converter.
            // Was incorrectly assigned Some(AneEngine::PE), causing MILNeg
            // to pass the CPU-only gate (I-41) because CPU_ONLY_OPS had
            // "negative" instead of "neg" (I-42).
            | MirOp::MILNeg { .. }
            // ─── T-P4-08: ANEC internal ops (CPU-only stubs) ──────
            | MirOp::AnecFusedLinearActivate { .. }
            // ─── T-P4-08: Unmapped ANEC operation stubs (CPU-only) ──
            | MirOp::AnecBroadcast { .. }
            | MirOp::AnecGlobalArgMinMax { .. }
            | MirOp::AnecDegamma { .. }
            | MirOp::AnecDirac { .. }
            | MirOp::AnecGainOffsetControl { .. }
            | MirOp::AnecNRelu { .. }
            | MirOp::AnecHighPrecisionSigmoid { .. }
            | MirOp::AnecLog2 { .. }
            | MirOp::AnecTrunc { .. }
            | MirOp::AnecInvert { .. }
            | MirOp::AnecUnflatten { .. }
            | MirOp::AnecChannelToSpace { .. }
            | MirOp::AnecSpaceToChannel { .. } => None,
        }
    }

    /// Returns the revision-aware default ANE engine for this op.
    ///
    /// This method extends the static engine assignment with family-specific
    /// capability checks. When a revision is provided, it resolves the
    /// corresponding [`AneFamily`] and overrides the base engine assignment
    /// for ops that lack an ANEC converter on that family.
    ///
    /// # Family-specific overrides
    ///
    /// | Op | Family | Base engine | Override | Reason |
    /// |----|--------|-------------|----------|--------|
    /// | `ReduceArgmax` | A18 | `PE` | `None` | No LSE_7 converter |
    /// | `ReduceArgmin` | A18 | `PE` | `None` | No LSE_7 converter |
    /// | `ReduceL2Norm` | A11Legacy, A12 | `PE` | `None` | No converter on these families |
    /// | `MILSquare` | A11Legacy, A12, A13 | `PE` | `None` | A14Minus converters; split at A13Minus/A14Plus boundary |
    /// | `ScaledDotProductAttention` | A16+ | `NE` | `NE` (unchanged) | Reliable SDPA converter on A16+ |
    ///
    /// When `revision` is `None`, the base engine assignment is returned
    /// without any family-specific overrides (backward-compatible behavior).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ane_ir::mir::MirOp;
    /// use ane_ir::ane_target::AneRevision;
    /// // On A18, ReduceArgmax has no ANEC converter → None
    /// let op = MirOp::MILReduceArgmax { name: "argmax".into(), x: "x".into(), axis: 1, keep_dims: false };
    /// assert_eq!(op.default_engine_for_revision(Some(AneRevision::V19)), None);
    /// ```
    pub fn default_engine_for_revision(
        &self,
        revision: Option<super::ane_target::AneRevision>,
    ) -> Option<super::ane_engine::AneEngine> {
        super::ane_placement::engine_for_op(self, revision)
    }

    /// Returns the default ANE execution engine for this op (revision-agnostic).
    ///
    /// Based on observed ANE fusion boundaries:
    /// - **NE**: conv/pool/matmul/attention pipeline
    /// - **PE**: elementwise/reduction/scaled-EW pipeline
    /// - **TransposeEngine**: data rearrangement
    /// - **None**: CPU-only ops (control flow, random, scatter, state, constexpr, etc.)
    ///
    /// This method delegates to [`Self::default_engine_for_revision`] with `None`,
    /// preserving backward compatibility. For revision-aware engine assignment,
    /// use [`Self::default_engine_for_revision`] instead.
    ///
    /// Source: ane-constraints-docs/03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md
    #[deprecated(
        since = "0.6.0",
        note = "use default_engine_for_revision() for revision-aware engine assignment"
    )]
    pub fn default_engine(&self) -> Option<super::ane_engine::AneEngine> {
        self.default_engine_for_revision(None)
    }

    /// Returns the canonical lowercase MIL op name for this variant.
    ///
    /// This is used for CPU-only op set lookups and cross-referencing
    /// with the constraint documentation. The name matches the MIL
    /// builder function name (e.g., "add", "conv", "scaled_dot_product_attention").
    pub fn mil_op_name(&self) -> &'static str {
        match self {
            MirOp::MILConst { .. } => "const",
            MirOp::MILLinear { .. } => "linear",
            MirOp::MILMatMul { .. } => "matmul",
            MirOp::MILEinsum { .. } => "einsum",
            MirOp::MILConv { .. } => "conv",
            MirOp::MILConvTranspose { .. } => "conv_transpose",
            MirOp::MILAdd { .. } => "add",
            MirOp::MILMul { .. } => "mul",
            MirOp::MILSub { .. } => "sub",
            MirOp::MILMaximum { .. } => "maximum",
            MirOp::MILMinimum { .. } => "minimum",
            MirOp::MILRealDiv { .. } => "real_div",
            MirOp::MILFloorDiv { .. } => "floor_div",
            MirOp::MILMod { .. } => "modulo",
            MirOp::MILPow { .. } => "pow",
            MirOp::MILEqual { .. } => "equal",
            MirOp::MILNotEqual { .. } => "not_equal",
            MirOp::MILGreater { .. } => "greater",
            MirOp::MILGreaterEqual { .. } => "greater_equal",
            MirOp::MILLess { .. } => "less",
            MirOp::MILLessEqual { .. } => "less_equal",
            MirOp::MILLogicalAnd { .. } => "logical_and",
            MirOp::MILLogicalOr { .. } => "logical_or",
            MirOp::MILLogicalXor { .. } => "logical_xor",
            MirOp::MILAbs { .. } => "abs",
            MirOp::MILNeg { .. } => "neg",
            MirOp::MILSigmoid { .. } => "sigmoid",
            MirOp::MILTanh { .. } => "tanh",
            MirOp::MILRelu { .. } => "relu",
            MirOp::MILRelu6 { .. } => "relu6",
            MirOp::MILLeakyRelu { .. } => "leaky_relu",
            MirOp::MILSigmoidHard { .. } => "sigmoid_hard",
            MirOp::MILThresholdedRelu { .. } => "thresholded_relu",
            MirOp::MILClampedRelu { .. } => "clamped_relu",
            MirOp::MILLinearActivation { .. } => "linear_activation",
            MirOp::MILPrelu { .. } => "prelu",
            MirOp::MILSoftsign { .. } => "softsign",
            MirOp::MILSilu { .. } => "silu",
            MirOp::MILScaledTanh { .. } => "scaled_tanh",
            MirOp::MILElu { .. } => "elu",
            MirOp::MILSoftplus { .. } => "softplus",
            MirOp::MILSoftplusParametric { .. } => "softplus_parametric",
            MirOp::MILGelu { .. } => "gelu",
            MirOp::MILClip { .. } => "clamp",
            MirOp::MILSquare { .. } => "square",
            MirOp::MILThreshold { .. } => "threshold",
            MirOp::MILSqrt { .. } => "sqrt",
            MirOp::MILRsqrt { .. } => "rsqrt",
            MirOp::MILInverse { .. } => "inverse",
            MirOp::MILCeil { .. } => "ceil",
            MirOp::MILFloor { .. } => "floor",
            MirOp::MILRound { .. } => "round",
            MirOp::MILExp { .. } => "exp",
            MirOp::MILExp2 { .. } => "exp2",
            MirOp::MILLog { .. } => "log",
            MirOp::MILSign { .. } => "sign",
            MirOp::MILCos { .. } => "cos",
            MirOp::MILSin { .. } => "sin",
            MirOp::MILTan { .. } => "tan",
            MirOp::MILAcos { .. } => "acos",
            MirOp::MILAsin { .. } => "asin",
            MirOp::MILAtan { .. } => "atan",
            MirOp::MILCosh { .. } => "cosh",
            MirOp::MILSinh { .. } => "sinh",
            MirOp::MILAtanh { .. } => "atanh",
            MirOp::MILErf { .. } => "erf",
            MirOp::MILLogicalNot { .. } => "logical_not",
            MirOp::MILCast { .. } => "cast",
            MirOp::MILSelect { .. } => "select",
            MirOp::MILWhere { .. } => "where",
            MirOp::MILSoftmax { .. } => "softmax",
            MirOp::MILReduceSum { .. } => "reduce_sum",
            MirOp::MILReduceMean { .. } => "reduce_mean",
            MirOp::MILReduceMax { .. } => "reduce_max",
            MirOp::MILReduceMin { .. } => "reduce_min",
            MirOp::MILReduceProd { .. } => "reduce_prod",
            MirOp::MILReduceSumSquare { .. } => "reduce_sum_square",
            MirOp::MILReduceL2Norm { .. } => "reduce_l2_norm",
            MirOp::MILReduceL1Norm { .. } => "reduce_l1_norm",
            MirOp::MILReduceLogSumExp { .. } => "reduce_log_sum_exp",
            MirOp::MILReduceLogSum { .. } => "reduce_log_sum",
            MirOp::MILReduceArgmax { .. } => "reduce_argmax",
            MirOp::MILReduceArgmin { .. } => "reduce_argmin",
            MirOp::MILBatchNorm { .. } => "batch_norm",
            MirOp::MILInstanceNorm { .. } => "instance_norm",
            MirOp::MILLayerNorm { .. } => "layer_norm",
            MirOp::MILL2Norm { .. } => "l2_norm",
            MirOp::MILLocalResponseNorm { .. } => "local_response_norm",
            MirOp::MILMaxPool { .. } => "max_pool",
            MirOp::MILAvgPool { .. } => "avg_pool",
            MirOp::MILL2Pool { .. } => "l2_pool",
            MirOp::MILResize { .. } => "resize",
            MirOp::MILResizeNearestNeighbor { .. } => "resize_nearest_neighbor",
            MirOp::MILResizeBilinear { .. } => "resize_bilinear",
            MirOp::MILUpsampleNearestNeighbor { .. } => "upsample_nearest_neighbor",
            MirOp::MILUpsampleBilinear { .. } => "upsample_bilinear",
            MirOp::MILCropResize { .. } => "crop_resize",
            MirOp::MILAffine { .. } => "affine",
            MirOp::MILResample { .. } => "resample",
            MirOp::MILReshape { .. } => "reshape",
            MirOp::MILReshapeLike { .. } => "reshape_like",
            MirOp::MILTranspose { .. } => "transpose",
            MirOp::MILSplit { .. } => "split",
            MirOp::MILConcat { .. } => "concat",
            MirOp::MILExpandDims { .. } => "expand_dims",
            MirOp::MILSqueeze { .. } => "squeeze",
            MirOp::MILFlatten2d { .. } => "flatten2d",
            MirOp::MILReverse { .. } => "reverse",
            MirOp::MILReverseSequence { .. } => "reverse_sequence",
            MirOp::MILSliceByIndex { .. } => "slice_by_index",
            MirOp::MILSliceBySize { .. } => "slice_by_size",
            MirOp::MILSliceUpdate { .. } => "slice_update",
            MirOp::MILSlidingWindows { .. } => "sliding_windows",
            MirOp::MILDepthToSpace { .. } => "depth_to_space",
            MirOp::MILSpaceToDepth { .. } => "space_to_depth",
            MirOp::MILPixelShuffle { .. } => "pixel_shuffle",
            MirOp::MILPixelUnshuffle { .. } => "pixel_unshuffle",
            MirOp::MILBatchToSpace { .. } => "batch_to_space",
            MirOp::MILSpaceToBatch { .. } => "space_to_batch",
            MirOp::MILPad { .. } => "pad",
            MirOp::MILStack { .. } => "stack",
            MirOp::MILTile { .. } => "tile",
            MirOp::MILCumsum { .. } => "cumsum",
            MirOp::MILFill { .. } => "fill",
            MirOp::MILFillLike { .. } => "fill_like",
            MirOp::MILIdentity { .. } => "identity",
            MirOp::MILOneHot { .. } => "one_hot",
            MirOp::MILNonZero { .. } => "non_zero",
            MirOp::MILArgsort { .. } => "argsort",
            MirOp::MILBandPart { .. } => "band_part",
            MirOp::MILRange1d { .. } => "range1d",
            MirOp::MILShape { .. } => "shape",
            MirOp::MILCrop { .. } => "crop",
            MirOp::MILGather { .. } => "gather",
            MirOp::MILGatherAlongAxis { .. } => "gather_along_axis",
            MirOp::MILGatherNd { .. } => "gather_nd",
            MirOp::MILScatter { .. } => "scatter",
            MirOp::MILScatterAlongAxis { .. } => "scatter_along_axis",
            MirOp::MILScatterNd { .. } => "scatter_nd",
            MirOp::MILNonMaximumSuppression { .. } => "non_maximum_suppression",
            MirOp::MILScaledDotProductAttention { .. } => "scaled_dot_product_attention",
            MirOp::MILQuantize { .. } => "quantize",
            MirOp::MILDequantize { .. } => "dequantize",
            MirOp::MILConstexprAffineDequantize { .. } => "constexpr_affine_dequantize",
            MirOp::MILConstexprBlockwiseShiftScale { .. } => "constexpr_blockwise_shift_scale",
            MirOp::MILConstexprLutToDense { .. } => "constexpr_lut_to_dense",
            MirOp::MILConstexprSparseToDense { .. } => "constexpr_sparse_to_dense",
            MirOp::MILConstexprCast { .. } => "constexpr_cast",
            MirOp::MILConstexprLutToSparse { .. } => "constexpr_lut_to_sparse",
            MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => {
                "constexpr_sparse_blockwise_shift_scale"
            }
            MirOp::MILRnn { .. } => "rnn",
            MirOp::MILGru { .. } => "gru",
            MirOp::MILLstm { .. } => "lstm",
            MirOp::MILCond { .. } => "cond",
            MirOp::MILWhileLoop { .. } => "while_loop",
            MirOp::MILMakeList { .. } => "make_list",
            MirOp::MILListLength { .. } => "list_length",
            MirOp::MILListWrite { .. } => "list_write",
            MirOp::MILListRead { .. } => "list_read",
            MirOp::MILListGather { .. } => "list_gather",
            MirOp::MILListScatter { .. } => "list_scatter",
            MirOp::MILRandomBernoulli { .. } => "random_bernoulli",
            MirOp::MILRandomNormal { .. } => "random_normal",
            MirOp::MILRandomUniform { .. } => "random_uniform",
            MirOp::MILRandomCategorical { .. } => "random_categorical",
            MirOp::MILReadState { .. } => "read_state",
            MirOp::MILCoremlUpdateState { .. } => "coreml_update_state",
            MirOp::MILStateWrite { .. } => "state_write",
            MirOp::MILTopk { .. } => "topk",
            MirOp::MILClassify { .. } => "classify",
            MirOp::AnecFusedConvActivate { .. } => "anec_fused_conv_activate",
            MirOp::AnecFusedLinearActivate { .. } => "anec_fused_linear_activate",
            // ─── T-P4-08: Unmapped ANEC operation stubs ──────────────
            MirOp::AnecBroadcast { .. } => "anec_broadcast",
            MirOp::AnecScaledElementwise { .. } => "anec_scaled_elementwise",
            MirOp::AnecGlobalArgMinMax { .. } => "anec_global_arg_min_max",
            MirOp::AnecDegamma { .. } => "anec_degamma",
            MirOp::AnecDirac { .. } => "anec_dirac",
            MirOp::AnecGainOffsetControl { .. } => "anec_gain_offset_control",
            MirOp::AnecNRelu { .. } => "anec_n_relu",
            MirOp::AnecHighPrecisionSigmoid { .. } => "anec_high_precision_sigmoid",
            MirOp::AnecLog2 { .. } => "anec_log2",
            MirOp::AnecTrunc { .. } => "anec_trunc",
            MirOp::AnecInvert { .. } => "anec_invert",
            MirOp::AnecUnflatten { .. } => "anec_unflatten",
            MirOp::AnecChannelToSpace { .. } => "anec_channel_to_space",
            MirOp::AnecSpaceToChannel { .. } => "anec_space_to_channel",
        }
    }
}

// ─── ToProto trait implementation ──────────────────────────────
// T-38 (I-17): Unified proto-emission interface for MirOp.
// Replaces per-variant match-arm boilerplate across 5+ separate match expressions.

impl ToProto for MirOp {
    fn proto_op_type(&self) -> &'static str {
        self.mil_op_name()
    }

    fn proto_output_name(&self) -> &str {
        match self {
            MirOp::MILConst { name, .. } => name,
            MirOp::MILLinear { name, .. } => name,
            MirOp::MILMatMul { name, .. } => name,
            MirOp::MILEinsum { name, .. } => name,
            MirOp::MILConv { name, .. } => name,
            MirOp::MILConvTranspose { name, .. } => name,
            MirOp::MILAdd { name, .. } => name,
            MirOp::MILMul { name, .. } => name,
            MirOp::MILSub { name, .. } => name,
            MirOp::MILMaximum { name, .. } => name,
            MirOp::MILMinimum { name, .. } => name,
            MirOp::MILRealDiv { name, .. } => name,
            MirOp::MILFloorDiv { name, .. } => name,
            MirOp::MILMod { name, .. } => name,
            MirOp::MILPow { name, .. } => name,
            MirOp::MILEqual { name, .. } => name,
            MirOp::MILNotEqual { name, .. } => name,
            MirOp::MILGreater { name, .. } => name,
            MirOp::MILGreaterEqual { name, .. } => name,
            MirOp::MILLess { name, .. } => name,
            MirOp::MILLessEqual { name, .. } => name,
            MirOp::MILLogicalAnd { name, .. } => name,
            MirOp::MILLogicalOr { name, .. } => name,
            MirOp::MILLogicalXor { name, .. } => name,
            MirOp::MILAbs { name, .. } => name,
            MirOp::MILNeg { name, .. } => name,
            MirOp::MILSigmoid { name, .. } => name,
            MirOp::MILTanh { name, .. } => name,
            MirOp::MILRelu { name, .. } => name,
            MirOp::MILRelu6 { name, .. } => name,
            MirOp::MILLeakyRelu { name, .. } => name,
            MirOp::MILSigmoidHard { name, .. } => name,
            MirOp::MILThresholdedRelu { name, .. } => name,
            MirOp::MILClampedRelu { name, .. } => name,
            MirOp::MILLinearActivation { name, .. } => name,
            MirOp::MILPrelu { name, .. } => name,
            MirOp::MILSoftsign { name, .. } => name,
            MirOp::MILSilu { name, .. } => name,
            MirOp::MILScaledTanh { name, .. } => name,
            MirOp::MILElu { name, .. } => name,
            MirOp::MILSoftplus { name, .. } => name,
            MirOp::MILSoftplusParametric { name, .. } => name,
            MirOp::MILGelu { name, .. } => name,
            MirOp::MILClip { name, .. } => name,
            MirOp::MILSquare { name, .. } => name,
            MirOp::MILThreshold { name, .. } => name,
            MirOp::MILSqrt { name, .. } => name,
            MirOp::MILRsqrt { name, .. } => name,
            MirOp::MILInverse { name, .. } => name,
            MirOp::MILCeil { name, .. } => name,
            MirOp::MILFloor { name, .. } => name,
            MirOp::MILRound { name, .. } => name,
            MirOp::MILExp { name, .. } => name,
            MirOp::MILExp2 { name, .. } => name,
            MirOp::MILLog { name, .. } => name,
            MirOp::MILSign { name, .. } => name,
            MirOp::MILCos { name, .. } => name,
            MirOp::MILSin { name, .. } => name,
            MirOp::MILTan { name, .. } => name,
            MirOp::MILAcos { name, .. } => name,
            MirOp::MILAsin { name, .. } => name,
            MirOp::MILAtan { name, .. } => name,
            MirOp::MILCosh { name, .. } => name,
            MirOp::MILSinh { name, .. } => name,
            MirOp::MILAtanh { name, .. } => name,
            MirOp::MILErf { name, .. } => name,
            MirOp::MILLogicalNot { name, .. } => name,
            MirOp::MILCast { name, .. } => name,
            MirOp::MILSelect { name, .. } => name,
            MirOp::MILWhere { name, .. } => name,
            MirOp::MILSoftmax { name, .. } => name,
            MirOp::MILReduceSum { name, .. } => name,
            MirOp::MILReduceMean { name, .. } => name,
            MirOp::MILReduceMax { name, .. } => name,
            MirOp::MILReduceMin { name, .. } => name,
            MirOp::MILReduceProd { name, .. } => name,
            MirOp::MILReduceSumSquare { name, .. } => name,
            MirOp::MILReduceL2Norm { name, .. } => name,
            MirOp::MILReduceL1Norm { name, .. } => name,
            MirOp::MILReduceLogSumExp { name, .. } => name,
            MirOp::MILReduceLogSum { name, .. } => name,
            MirOp::MILReduceArgmax { name, .. } => name,
            MirOp::MILReduceArgmin { name, .. } => name,
            MirOp::MILBatchNorm { name, .. } => name,
            MirOp::MILInstanceNorm { name, .. } => name,
            MirOp::MILLayerNorm { name, .. } => name,
            MirOp::MILL2Norm { name, .. } => name,
            MirOp::MILLocalResponseNorm { name, .. } => name,
            MirOp::MILMaxPool { name, .. } => name,
            MirOp::MILAvgPool { name, .. } => name,
            MirOp::MILL2Pool { name, .. } => name,
            MirOp::MILResize { name, .. } => name,
            MirOp::MILResizeNearestNeighbor { name, .. } => name,
            MirOp::MILResizeBilinear { name, .. } => name,
            MirOp::MILUpsampleNearestNeighbor { name, .. } => name,
            MirOp::MILUpsampleBilinear { name, .. } => name,
            MirOp::MILCropResize { name, .. } => name,
            MirOp::MILAffine { name, .. } => name,
            MirOp::MILResample { name, .. } => name,
            MirOp::MILReshape { name, .. } => name,
            MirOp::MILReshapeLike { name, .. } => name,
            MirOp::MILTranspose { name, .. } => name,
            MirOp::MILSplit { name, .. } => name,
            MirOp::MILConcat { name, .. } => name,
            MirOp::MILExpandDims { name, .. } => name,
            MirOp::MILSqueeze { name, .. } => name,
            MirOp::MILFlatten2d { name, .. } => name,
            MirOp::MILReverse { name, .. } => name,
            MirOp::MILReverseSequence { name, .. } => name,
            MirOp::MILSliceByIndex { name, .. } => name,
            MirOp::MILSliceBySize { name, .. } => name,
            MirOp::MILSliceUpdate { name, .. } => name,
            MirOp::MILSlidingWindows { name, .. } => name,
            MirOp::MILDepthToSpace { name, .. } => name,
            MirOp::MILSpaceToDepth { name, .. } => name,
            MirOp::MILPixelShuffle { name, .. } => name,
            MirOp::MILPixelUnshuffle { name, .. } => name,
            MirOp::MILBatchToSpace { name, .. } => name,
            MirOp::MILSpaceToBatch { name, .. } => name,
            MirOp::MILPad { name, .. } => name,
            MirOp::MILStack { name, .. } => name,
            MirOp::MILTile { name, .. } => name,
            MirOp::MILCumsum { name, .. } => name,
            MirOp::MILFill { name, .. } => name,
            MirOp::MILFillLike { name, .. } => name,
            MirOp::MILIdentity { name, .. } => name,
            MirOp::MILOneHot { name, .. } => name,
            MirOp::MILNonZero { name, .. } => name,
            MirOp::MILArgsort { name, .. } => name,
            MirOp::MILBandPart { name, .. } => name,
            MirOp::MILRange1d { name, .. } => name,
            MirOp::MILShape { name, .. } => name,
            MirOp::MILCrop { name, .. } => name,
            MirOp::MILGather { name, .. } => name,
            MirOp::MILGatherAlongAxis { name, .. } => name,
            MirOp::MILGatherNd { name, .. } => name,
            MirOp::MILScatter { name, .. } => name,
            MirOp::MILScatterAlongAxis { name, .. } => name,
            MirOp::MILScatterNd { name, .. } => name,
            MirOp::MILNonMaximumSuppression { name, .. } => name,
            MirOp::MILScaledDotProductAttention { name, .. } => name,
            MirOp::MILQuantize { name, .. } => name,
            MirOp::MILDequantize { name, .. } => name,
            MirOp::MILConstexprAffineDequantize { name, .. } => name,
            MirOp::MILConstexprBlockwiseShiftScale { name, .. } => name,
            MirOp::MILConstexprLutToDense { name, .. } => name,
            MirOp::MILConstexprSparseToDense { name, .. } => name,
            MirOp::MILConstexprCast { name, .. } => name,
            MirOp::MILConstexprLutToSparse { name, .. } => name,
            MirOp::MILConstexprSparseBlockwiseShiftScale { name, .. } => name,
            MirOp::MILRnn { name, .. } => name,
            MirOp::MILGru { name, .. } => name,
            MirOp::MILLstm { name, .. } => name,
            MirOp::MILCond { name, .. } => name,
            MirOp::MILWhileLoop { name, .. } => name,
            MirOp::MILMakeList { name, .. } => name,
            MirOp::MILListLength { name, .. } => name,
            MirOp::MILListWrite { name, .. } => name,
            MirOp::MILListRead { name, .. } => name,
            MirOp::MILListGather { name, .. } => name,
            MirOp::MILListScatter { name, .. } => name,
            MirOp::MILRandomBernoulli { name, .. } => name,
            MirOp::MILRandomNormal { name, .. } => name,
            MirOp::MILRandomUniform { name, .. } => name,
            MirOp::MILRandomCategorical { name, .. } => name,
            MirOp::MILReadState { name, .. } => name,
            MirOp::MILCoremlUpdateState { name, .. } => name,
            MirOp::MILStateWrite { name, .. } => name,
            MirOp::MILTopk { name, .. } => name,
            MirOp::MILClassify { name, .. } => name,
            MirOp::AnecFusedConvActivate { name, .. } => name,
            MirOp::AnecFusedLinearActivate { name, .. } => name,
            // ─── T-P4-08: Unmapped ANEC operation stubs ──────────────
            MirOp::AnecBroadcast { name, .. } => name,
            MirOp::AnecScaledElementwise { name, .. } => name,
            MirOp::AnecGlobalArgMinMax { name, .. } => name,
            MirOp::AnecDegamma { name, .. } => name,
            MirOp::AnecDirac { name, .. } => name,
            MirOp::AnecGainOffsetControl { name, .. } => name,
            MirOp::AnecNRelu { name, .. } => name,
            MirOp::AnecHighPrecisionSigmoid { name, .. } => name,
            MirOp::AnecLog2 { name, .. } => name,
            MirOp::AnecTrunc { name, .. } => name,
            MirOp::AnecInvert { name, .. } => name,
            MirOp::AnecUnflatten { name, .. } => name,
            MirOp::AnecChannelToSpace { name, .. } => name,
            MirOp::AnecSpaceToChannel { name, .. } => name,
        }
    }

    fn proto_input_refs(&self) -> Vec<String> {
        match self {
            // ─── Constants ───────────────────────────────────────────
            MirOp::MILConst { .. } => vec![],

            // ─── Linear / FC ─────────────────────────────────────────
            MirOp::MILLinear { x, weight, bias, .. } => {
                let mut refs = vec![x.0.clone(), weight.clone()];
                if let Some(b) = bias {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILMatMul { x, y, .. } => vec![x.0.clone(), y.0.clone()],
            MirOp::MILEinsum { inputs, .. } => inputs.iter().map(|id| id.0.clone()).collect(),

            // ─── Convolution ─────────────────────────────────────────
            MirOp::MILConv { x, weight, .. } => vec![x.0.clone(), weight.0.clone()],
            MirOp::MILConvTranspose { x, weight, .. } => vec![x.0.clone(), weight.0.clone()],

            // ─── Elementwise Binary ──────────────────────────────────
            MirOp::MILAdd { x, y, .. }
            | MirOp::MILMul { x, y, .. }
            | MirOp::MILSub { x, y, .. }
            | MirOp::MILMaximum { x, y, .. }
            | MirOp::MILMinimum { x, y, .. }
            | MirOp::MILRealDiv { x, y, .. }
            | MirOp::MILFloorDiv { x, y, .. }
            | MirOp::MILMod { x, y, .. }
            | MirOp::MILPow { x, y, .. }
            | MirOp::MILEqual { x, y, .. }
            | MirOp::MILNotEqual { x, y, .. }
            | MirOp::MILGreater { x, y, .. }
            | MirOp::MILGreaterEqual { x, y, .. }
            | MirOp::MILLess { x, y, .. }
            | MirOp::MILLessEqual { x, y, .. }
            | MirOp::MILLogicalAnd { x, y, .. }
            | MirOp::MILLogicalOr { x, y, .. }
            | MirOp::MILLogicalXor { x, y, .. } => vec![x.0.clone(), y.0.clone()],

            // ─── Elementwise Unary (simple) ──────────────────────────
            MirOp::MILAbs { x, .. }
            | MirOp::MILNeg { x, .. }
            | MirOp::MILSigmoid { x, .. }
            | MirOp::MILTanh { x, .. }
            | MirOp::MILRelu { x, .. }
            | MirOp::MILRelu6 { x, .. }
            | MirOp::MILSoftsign { x, .. }
            | MirOp::MILSilu { x, .. }
            | MirOp::MILSoftplus { x, .. }
            | MirOp::MILSquare { x, .. }
            | MirOp::MILSqrt { x, .. }
            | MirOp::MILRsqrt { x, .. }
            | MirOp::MILCeil { x, .. }
            | MirOp::MILFloor { x, .. }
            | MirOp::MILRound { x, .. }
            | MirOp::MILExp { x, .. }
            | MirOp::MILExp2 { x, .. }
            | MirOp::MILSign { x, .. }
            | MirOp::MILCos { x, .. }
            | MirOp::MILSin { x, .. }
            | MirOp::MILTan { x, .. }
            | MirOp::MILAcos { x, .. }
            | MirOp::MILAsin { x, .. }
            | MirOp::MILAtan { x, .. }
            | MirOp::MILCosh { x, .. }
            | MirOp::MILSinh { x, .. }
            | MirOp::MILAtanh { x, .. }
            | MirOp::MILErf { x, .. }
            | MirOp::MILLogicalNot { x, .. }
            | MirOp::MILIdentity { x, .. }
            | MirOp::MILNonZero { x, .. }
            | MirOp::MILShape { x, .. } => vec![x.0.clone()],

            // ─── Elementwise Unary (with scalar params) ─────────────
            MirOp::MILLeakyRelu { x, .. }
            | MirOp::MILSigmoidHard { x, .. }
            | MirOp::MILThresholdedRelu { x, .. }
            | MirOp::MILClampedRelu { x, .. }
            | MirOp::MILLinearActivation { x, .. }
            | MirOp::MILScaledTanh { x, .. }
            | MirOp::MILElu { x, .. }
            | MirOp::MILClip { x, .. }
            | MirOp::MILThreshold { x, .. }
            | MirOp::MILInverse { x, .. }
            | MirOp::MILLog { x, .. } => vec![x.0.clone()],

            // ─── Elementwise Unary (with String weight refs) ─────────
            MirOp::MILPrelu { x, alpha, .. } => {
                vec![x.0.clone(), alpha.clone()]
            }
            MirOp::MILSoftplusParametric { x, alpha, beta, .. } => {
                vec![x.0.clone(), alpha.clone(), beta.clone()]
            }

            // ─── Gelu (mode is an enum string, not a ref) ───────────
            MirOp::MILGelu { x, .. } => vec![x.0.clone()],

            // ─── Cast / Softmax / Select / Where ─────────────────────
            MirOp::MILCast { x, .. } => vec![x.0.clone()],
            MirOp::MILSoftmax { x, .. } => vec![x.0.clone()],
            MirOp::MILSelect { condition, x, y, .. } | MirOp::MILWhere { condition, x, y, .. } => {
                vec![condition.0.clone(), x.0.clone(), y.0.clone()]
            }

            // ─── Reduction ───────────────────────────────────────────
            MirOp::MILReduceSum { x, .. }
            | MirOp::MILReduceMean { x, .. }
            | MirOp::MILReduceMax { x, .. }
            | MirOp::MILReduceMin { x, .. }
            | MirOp::MILReduceProd { x, .. }
            | MirOp::MILReduceSumSquare { x, .. }
            | MirOp::MILReduceL2Norm { x, .. }
            | MirOp::MILReduceL1Norm { x, .. }
            | MirOp::MILReduceLogSumExp { x, .. }
            | MirOp::MILReduceLogSum { x, .. } => vec![x.0.clone()],

            MirOp::MILReduceArgmax { x, .. } | MirOp::MILReduceArgmin { x, .. } => {
                vec![x.0.clone()]
            }

            // ─── Normalization ───────────────────────────────────────
            MirOp::MILBatchNorm { x, mean, variance, gamma, beta, .. } => {
                let mut refs = vec![x.0.clone(), mean.clone(), variance.clone()];
                if let Some(g) = gamma {
                    refs.push(g.clone());
                }
                if let Some(b) = beta {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILInstanceNorm { x, gamma, beta, .. } => {
                let mut refs = vec![x.0.clone()];
                if let Some(g) = gamma {
                    refs.push(g.clone());
                }
                if let Some(b) = beta {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILLayerNorm { x, weight, bias, .. } => {
                let mut refs = vec![x.0.clone(), weight.clone()];
                if let Some(b) = bias {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILL2Norm { x, .. } => vec![x.0.clone()],
            MirOp::MILLocalResponseNorm { x, .. } => vec![x.0.clone()],

            // ─── Pooling ─────────────────────────────────────────────
            MirOp::MILMaxPool { x, .. }
            | MirOp::MILAvgPool { x, .. }
            | MirOp::MILL2Pool { x, .. } => vec![x.0.clone()],

            // ─── Image Resizing ──────────────────────────────────────
            MirOp::MILResize { x, .. }
            | MirOp::MILResizeNearestNeighbor { x, .. }
            | MirOp::MILResizeBilinear { x, .. }
            | MirOp::MILUpsampleNearestNeighbor { x, .. }
            | MirOp::MILUpsampleBilinear { x, .. } => vec![x.0.clone()],

            MirOp::MILCropResize { x, boxes, box_indices, .. } => {
                vec![x.0.clone(), boxes.0.clone(), box_indices.0.clone()]
            }
            MirOp::MILAffine { x, transform, .. } => {
                vec![x.0.clone(), transform.0.clone()]
            }
            MirOp::MILResample { x, coordinates, .. } => {
                vec![x.0.clone(), coordinates.0.clone()]
            }

            // ─── Tensor Transform ────────────────────────────────────
            MirOp::MILReshape { x, .. } => vec![x.0.clone()],
            MirOp::MILReshapeLike { x, ref_tensor, .. } => {
                vec![x.0.clone(), ref_tensor.0.clone()]
            }
            MirOp::MILTranspose { x, .. } => vec![x.0.clone()],
            MirOp::MILSplit { x, .. } => vec![x.0.clone()],
            MirOp::MILConcat { values, .. } => values.iter().map(|id| id.0.clone()).collect(),
            MirOp::MILExpandDims { x, .. }
            | MirOp::MILSqueeze { x, .. }
            | MirOp::MILFlatten2d { x, .. } => vec![x.0.clone()],

            MirOp::MILReverse { x, .. } => vec![x.0.clone()],
            MirOp::MILReverseSequence { x, lengths, .. } => {
                vec![x.0.clone(), lengths.0.clone()]
            }
            MirOp::MILSliceByIndex { x, .. } => vec![x.0.clone()],
            MirOp::MILSliceBySize { x, .. } => vec![x.0.clone()],
            MirOp::MILSliceUpdate { x, update, .. } => {
                vec![x.0.clone(), update.0.clone()]
            }
            MirOp::MILSlidingWindows { x, .. } => vec![x.0.clone()],

            MirOp::MILDepthToSpace { x, .. }
            | MirOp::MILSpaceToDepth { x, .. }
            | MirOp::MILPixelShuffle { x, .. }
            | MirOp::MILPixelUnshuffle { x, .. }
            | MirOp::MILBatchToSpace { x, .. }
            | MirOp::MILSpaceToBatch { x, .. } => vec![x.0.clone()],

            MirOp::MILPad { x, .. } => vec![x.0.clone()],
            MirOp::MILStack { values, .. } => values.iter().map(|id| id.0.clone()).collect(),
            MirOp::MILTile { x, .. } => vec![x.0.clone()],
            MirOp::MILCumsum { x, .. } => vec![x.0.clone()],
            MirOp::MILFill { .. } => vec![],
            MirOp::MILFillLike { ref_tensor, .. } => vec![ref_tensor.0.clone()],
            MirOp::MILOneHot { indices, .. } => vec![indices.0.clone()],
            MirOp::MILArgsort { x, .. } => vec![x.0.clone()],
            MirOp::MILBandPart { x, .. } => vec![x.0.clone()],
            MirOp::MILRange1d { .. } => vec![],
            MirOp::MILCrop { x, .. } => vec![x.0.clone()],

            // ─── Scatter / Gather ────────────────────────────────────
            MirOp::MILGather { x, indices, .. }
            | MirOp::MILGatherAlongAxis { x, indices, .. }
            | MirOp::MILGatherNd { x, indices, .. } => {
                vec![x.0.clone(), indices.0.clone()]
            }
            MirOp::MILScatter { x, indices, updates, .. }
            | MirOp::MILScatterAlongAxis { x, indices, updates, .. }
            | MirOp::MILScatterNd { x, indices, updates, .. } => {
                vec![x.0.clone(), indices.0.clone(), updates.0.clone()]
            }
            MirOp::MILNonMaximumSuppression { boxes, scores, .. } => {
                vec![boxes.0.clone(), scores.0.clone()]
            }

            // ─── Attention ───────────────────────────────────────────
            MirOp::MILScaledDotProductAttention { query, key, value, attention_mask, .. } => {
                let mut refs = vec![query.0.clone(), key.0.clone(), value.0.clone()];
                if let Some(m) = attention_mask {
                    refs.push(m.0.clone());
                }
                refs
            }

            // ─── Quantization ────────────────────────────────────────
            MirOp::MILQuantize { x, .. } | MirOp::MILDequantize { x, .. } => vec![x.0.clone()],

            // ─── Constexpr / Compression ─────────────────────────────
            MirOp::MILConstexprAffineDequantize { quantized_data, .. } => {
                vec![quantized_data.clone()]
            }
            MirOp::MILConstexprBlockwiseShiftScale { data, scale, offset, .. } => {
                vec![data.clone(), scale.clone(), offset.clone()]
            }
            MirOp::MILConstexprLutToDense { indices, lut, .. } => {
                vec![indices.clone(), lut.clone()]
            }
            MirOp::MILConstexprSparseToDense { nonzero_data, .. } => {
                vec![nonzero_data.clone()]
            }
            MirOp::MILConstexprCast { data, .. } => {
                vec![data.clone()]
            }
            MirOp::MILConstexprLutToSparse { data, .. } => {
                vec![data.clone()]
            }
            MirOp::MILConstexprSparseBlockwiseShiftScale { data, scale, offset, .. } => {
                vec![data.clone(), scale.clone(), offset.clone()]
            }

            // ─── Recurrent ───────────────────────────────────────────
            MirOp::MILRnn { x, initial_h, weight_ih, weight_hh, bias, .. } => {
                let mut refs =
                    vec![x.0.clone(), initial_h.0.clone(), weight_ih.clone(), weight_hh.clone()];
                if let Some(b) = bias {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILGru { x, initial_h, weight_ih, weight_hh, bias, .. } => {
                let mut refs =
                    vec![x.0.clone(), initial_h.0.clone(), weight_ih.clone(), weight_hh.clone()];
                if let Some(b) = bias {
                    refs.push(b.clone());
                }
                refs
            }
            MirOp::MILLstm { x, initial_h, initial_c, weight_ih, weight_hh, bias, .. } => {
                let mut refs = vec![
                    x.0.clone(),
                    initial_h.0.clone(),
                    initial_c.0.clone(),
                    weight_ih.clone(),
                    weight_hh.clone(),
                ];
                if let Some(b) = bias {
                    refs.push(b.clone());
                }
                refs
            }

            // ─── Control Flow ────────────────────────────────────────
            MirOp::MILCond { pred, .. } => vec![pred.0.clone()],
            MirOp::MILWhileLoop { loop_vars, .. } => {
                loop_vars.iter().map(|id| id.0.clone()).collect()
            }
            MirOp::MILMakeList { elems, .. } => elems.iter().map(|id| id.0.clone()).collect(),
            MirOp::MILListLength { ls, .. } => vec![ls.0.clone()],
            MirOp::MILListWrite { ls, index, value, .. } => {
                vec![ls.0.clone(), index.0.clone(), value.0.clone()]
            }
            MirOp::MILListRead { ls, index, .. } => {
                vec![ls.0.clone(), index.0.clone()]
            }
            MirOp::MILListGather { ls, indices, .. } => {
                vec![ls.0.clone(), indices.0.clone()]
            }
            MirOp::MILListScatter { ls, indices, values, .. } => {
                vec![ls.0.clone(), indices.0.clone(), values.0.clone()]
            }

            // ─── Random ──────────────────────────────────────────────
            MirOp::MILRandomBernoulli { .. }
            | MirOp::MILRandomNormal { .. }
            | MirOp::MILRandomUniform { .. } => vec![],
            MirOp::MILRandomCategorical { logits, .. } => vec![logits.0.clone()],

            // ─── State ───────────────────────────────────────────────
            MirOp::MILReadState { .. } => vec![],
            MirOp::MILCoremlUpdateState { value, .. } => vec![value.0.clone()],
            MirOp::MILStateWrite { value, .. } => vec![value.0.clone()],

            // ─── Metadata / Misc ─────────────────────────────────────
            MirOp::MILTopk { x, .. } => vec![x.0.clone()],
            MirOp::MILClassify { x, .. } => vec![x.0.clone()],
            // ─── ANEC Internal Ops (T-P4-08) ─────────────────────────
            MirOp::AnecFusedConvActivate { x, weight, .. } => vec![x.0.clone(), weight.0.clone()],
            MirOp::AnecFusedLinearActivate { x, weight, .. } => vec![x.0.clone(), weight.clone()],
            // ─── T-P4-08: Unmapped ANEC operation stubs ──────────────
            MirOp::AnecBroadcast { x, .. } => vec![x.0.clone()],
            MirOp::AnecScaledElementwise { x, y, .. } => vec![x.0.clone(), y.0.clone()],
            MirOp::AnecGlobalArgMinMax { x, .. } => vec![x.0.clone()],
            MirOp::AnecDegamma { x, .. } => vec![x.0.clone()],
            MirOp::AnecDirac { x, .. } => vec![x.0.clone()],
            MirOp::AnecGainOffsetControl { x, .. } => vec![x.0.clone()],
            MirOp::AnecNRelu { x, .. } => vec![x.0.clone()],
            MirOp::AnecHighPrecisionSigmoid { x, .. } => vec![x.0.clone()],
            MirOp::AnecLog2 { x, .. } => vec![x.0.clone()],
            MirOp::AnecTrunc { x, .. } => vec![x.0.clone()],
            MirOp::AnecInvert { x, .. } => vec![x.0.clone()],
            MirOp::AnecUnflatten { x, .. } => vec![x.0.clone()],
            MirOp::AnecChannelToSpace { x, .. } => vec![x.0.clone()],
            MirOp::AnecSpaceToChannel { x, .. } => vec![x.0.clone()],
        }
    }

    fn is_proto_supported(&self) -> bool {
        matches!(
            self,
            // Constants
            MirOp::MILConst { .. }
            // Linear / FC
            | MirOp::MILLinear { .. }
            | MirOp::MILMatMul { .. }
            // Convolution
            | MirOp::MILConv { .. }
            // Elementwise Binary
            | MirOp::MILAdd { .. }
            | MirOp::MILMul { .. }
            | MirOp::MILSub { .. }
            | MirOp::MILMaximum { .. }
            | MirOp::MILMinimum { .. }
            | MirOp::MILRealDiv { .. }
            | MirOp::MILFloorDiv { .. }
            | MirOp::MILMod { .. }
            | MirOp::MILPow { .. }
            | MirOp::MILEqual { .. }
            | MirOp::MILNotEqual { .. }
            | MirOp::MILGreater { .. }
            | MirOp::MILGreaterEqual { .. }
            | MirOp::MILLess { .. }
            | MirOp::MILLessEqual { .. }
            | MirOp::MILLogicalAnd { .. }
            | MirOp::MILLogicalOr { .. }
            | MirOp::MILLogicalNot { .. }
            // Elementwise Unary
            | MirOp::MILAbs { .. }
            | MirOp::MILNeg { .. }
            | MirOp::MILSigmoid { .. }
            | MirOp::MILTanh { .. }
            | MirOp::MILRelu { .. }
            | MirOp::MILLeakyRelu { .. }
            | MirOp::MILSilu { .. }
            | MirOp::MILGelu { .. }
            | MirOp::MILClip { .. }
            | MirOp::MILSqrt { .. }
            | MirOp::MILRsqrt { .. }
            | MirOp::MILCeil { .. }
            | MirOp::MILFloor { .. }
            | MirOp::MILRound { .. }
            | MirOp::MILExp { .. }
            | MirOp::MILLog { .. }
            | MirOp::MILSign { .. }
            | MirOp::MILCos { .. }
            | MirOp::MILSin { .. }
            // Cast / Select / Where / Softmax
            | MirOp::MILCast { .. }
            | MirOp::MILSelect { .. }
            | MirOp::MILWhere { .. }
            | MirOp::MILSoftmax { .. }
            // Reduction
            | MirOp::MILReduceSum { .. }
            | MirOp::MILReduceMean { .. }
            | MirOp::MILReduceMax { .. }
            | MirOp::MILReduceMin { .. }
            | MirOp::MILReduceProd { .. }
            // Normalization
            | MirOp::MILLayerNorm { .. }
            // Tensor Transform
            | MirOp::MILReshape { .. }
            | MirOp::MILTranspose { .. }
            | MirOp::MILSplit { .. }
            | MirOp::MILConcat { .. }
            | MirOp::MILExpandDims { .. }
            | MirOp::MILSqueeze { .. }
            | MirOp::MILSliceByIndex { .. }
            | MirOp::MILSliceUpdate { .. }
            | MirOp::MILPad { .. }
            | MirOp::MILTile { .. }
            | MirOp::MILFill { .. }
            | MirOp::MILFillLike { .. }
            | MirOp::MILIdentity { .. }
            // Scatter / Gather
            | MirOp::MILGather { .. }
            | MirOp::MILTopk { .. }
            // Attention
            | MirOp::MILScaledDotProductAttention { .. }
            // State
            | MirOp::MILReadState { .. }
            | MirOp::MILCoremlUpdateState { .. }
            | MirOp::MILStateWrite { .. }
            // Constexpr / Compression
            | MirOp::MILConstexprAffineDequantize { .. }
            | MirOp::MILConstexprBlockwiseShiftScale { .. }
            | MirOp::MILConstexprLutToDense { .. }
            | MirOp::MILConstexprSparseToDense { .. }
            | MirOp::MILConstexprCast { .. }
            | MirOp::MILConstexprLutToSparse { .. }
            | MirOp::MILConstexprSparseBlockwiseShiftScale { .. }
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirNode {
    pub id: MirNodeId,
    pub op: MirOp,
    pub dtype: MilDtype,
    pub shape: Vec<usize>,
    pub compute_unit_hint: Option<ComputeUnitHint>,
    pub air_source: Option<super::air::AirNodeId>,
    /// Target-specific annotation capturing ANE placement attributes.
    /// Populated after engine assignment but before emission.
    #[serde(default)]
    pub target_annotation: MirOpTargetAnnotation,
}

// ComputeUnitHint moved to common.rs; re-exported via `pub use super::common::ComputeUnitHint;` above.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirGraph {
    pub nodes: Vec<MirNode>,
    pub inputs: Vec<MirNodeId>,
    pub outputs: Vec<MirNodeId>,
    pub opset_version: String,
    pub shard_name: String,
    /// Explicit input shapes for graph inputs that don't have corresponding
    /// MirNode entries. This is critical for multi-function models like
    /// decode_step where the inputs (e.g., "sir_hidden_input") are
    /// referenced by ops but aren't MirNode themselves. Without this,
    /// mir_to_compat falls back to shape [1] which breaks all downstream
    /// shape inference.
    #[serde(default)]
    pub input_shapes: std::collections::HashMap<MirNodeId, Vec<usize>>,
}

impl MirGraph {
    /// Verify graph invariants:
    /// - No duplicate node IDs
    /// - All required fields are populated (name, op kind)
    /// - All dtype values are valid
    /// - All edge references resolve to existing nodes
    /// - All inputs/outputs reference existing nodes
    /// - All node shapes are non-empty or the op is a known
    ///   shape-exception (MILConst, MILStateWrite, MILCoremlUpdateState,
    ///   MILRange1d, MILFill, or ops that produce dynamic/0-rank tensors)
    pub fn verify(&self) -> Result<(), super::common::VerifyError> {
        self.verify_with_shape_check(true)
    }

    /// Verify graph invariants without checking shapes.
    /// Useful for early-stage graphs where shapes haven't been inferred yet.
    pub fn verify_no_shape_check(&self) -> Result<(), super::common::VerifyError> {
        self.verify_with_shape_check(false)
    }

    /// Verify graph invariants with strict shape checking.
    /// Same as `verify()` — requires non-empty shapes on all non-exception nodes.
    pub fn verify_strict(&self) -> Result<(), super::common::VerifyError> {
        self.verify_with_shape_check(true)
    }

    fn verify_with_shape_check(
        &self,
        check_shapes: bool,
    ) -> Result<(), super::common::VerifyError> {
        use super::common::VerifyError;
        use std::collections::HashSet;

        // Collect all node IDs, checking for duplicates
        let mut seen_ids: HashSet<&str> = HashSet::new();
        for node in &self.nodes {
            let id = node.id.as_str();
            if !seen_ids.insert(id) {
                return Err(VerifyError::DuplicateNodeId { node_id: id.to_string() });
            }
        }

        // Check inputs reference existing nodes
        for input_id in &self.inputs {
            if !seen_ids.contains(input_id.as_str()) {
                return Err(VerifyError::GraphInvariant {
                    message: format!("MirGraph input '{}' not found in nodes", input_id.0),
                });
            }
        }

        // Check outputs reference existing nodes
        for output_id in &self.outputs {
            if !seen_ids.contains(output_id.as_str()) {
                return Err(VerifyError::GraphInvariant {
                    message: format!("MirGraph output '{}' not found in nodes", output_id.0),
                });
            }
        }

        // Per-node checks
        for node in &self.nodes {
            let node_id = node.id.as_str();

            // Check shapes: require non-empty shapes for ops that produce tensors.
            // Exceptions: ops that carry no output shape (constants, state writes,
            // range1d, fill — these have their shape encoded differently or
            // produce 0-rank/scalar tensors).
            if check_shapes && node.shape.is_empty() && !Self::is_shape_exception(&node.op) {
                return Err(VerifyError::EmptyShape { node_id: node_id.to_string() });
            }

            // Validate node references inside ops
            let ref_err = Self::validate_op_refs(&node.op, node_id, &seen_ids);
            ref_err?
        }

        Ok(())
    }

    /// Ops that are allowed to have an empty shape vector.
    ///
    /// These are ops where the output shape is not meaningful (state
    /// mutation ops), is encoded inline (Fill, Range1d, MILConst carry
    /// shape info in their fields), or produce 0-rank/scalar tensors.
    fn is_shape_exception(op: &MirOp) -> bool {
        matches!(
            op,
            MirOp::MILConst { .. }
                | MirOp::MILStateWrite { .. }
                | MirOp::MILCoremlUpdateState { .. }
                | MirOp::MILRange1d { .. }
                | MirOp::MILFill { .. }
        )
    }

    /// Validate that all MirNodeId references inside an op resolve to existing nodes.
    fn validate_op_refs(
        op: &MirOp,
        node_id: &str,
        seen_ids: &std::collections::HashSet<&str>,
    ) -> Result<(), super::common::VerifyError> {
        use super::common::VerifyError;
        for referenced in op.node_refs() {
            if !seen_ids.contains(referenced.as_str()) {
                return Err(VerifyError::UnresolvedReference {
                    node_id: node_id.to_string(),
                    reference: referenced.0.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Helper: extract all MirNodeId references from a MirOp.
impl MirOp {
    /// Returns all `MirNodeId` references within this op variant.
    pub fn node_refs(&self) -> Vec<&MirNodeId> {
        match self {
            MirOp::MILConst { .. } => vec![],
            MirOp::MILLinear { x, .. } => vec![x],
            MirOp::MILMatMul { x, y, .. } => vec![x, y],
            MirOp::MILEinsum { inputs, .. } => inputs.iter().collect(),
            MirOp::MILConv { x, weight, .. } => vec![x, weight],
            MirOp::MILConvTranspose { x, weight, .. } => vec![x, weight],
            MirOp::MILAdd { x, y, .. } => vec![x, y],
            MirOp::MILMul { x, y, .. } => vec![x, y],
            MirOp::MILSub { x, y, .. } => vec![x, y],
            MirOp::MILMaximum { x, y, .. } => vec![x, y],
            MirOp::MILMinimum { x, y, .. } => vec![x, y],
            MirOp::MILRealDiv { x, y, .. } => vec![x, y],
            MirOp::MILFloorDiv { x, y, .. } => vec![x, y],
            MirOp::MILMod { x, y, .. } => vec![x, y],
            MirOp::MILPow { x, y, .. } => vec![x, y],
            MirOp::MILEqual { x, y, .. } => vec![x, y],
            MirOp::MILNotEqual { x, y, .. } => vec![x, y],
            MirOp::MILGreater { x, y, .. } => vec![x, y],
            MirOp::MILGreaterEqual { x, y, .. } => vec![x, y],
            MirOp::MILLess { x, y, .. } => vec![x, y],
            MirOp::MILLessEqual { x, y, .. } => vec![x, y],
            MirOp::MILLogicalAnd { x, y, .. } => vec![x, y],
            MirOp::MILLogicalOr { x, y, .. } => vec![x, y],
            MirOp::MILLogicalXor { x, y, .. } => vec![x, y],
            MirOp::MILAbs { x, .. } => vec![x],
            MirOp::MILNeg { x, .. } => vec![x],
            MirOp::MILSigmoid { x, .. } => vec![x],
            MirOp::MILTanh { x, .. } => vec![x],
            MirOp::MILRelu { x, .. } => vec![x],
            MirOp::MILRelu6 { x, .. } => vec![x],
            MirOp::MILLeakyRelu { x, .. } => vec![x],
            MirOp::MILSigmoidHard { x, .. } => vec![x],
            MirOp::MILThresholdedRelu { x, .. } => vec![x],
            MirOp::MILClampedRelu { x, .. } => vec![x],
            MirOp::MILLinearActivation { x, .. } => vec![x],
            MirOp::MILPrelu { x, .. } => vec![x],
            MirOp::MILSoftsign { x, .. } => vec![x],
            MirOp::MILSilu { x, .. } => vec![x],
            MirOp::MILScaledTanh { x, .. } => vec![x],
            MirOp::MILElu { x, .. } => vec![x],
            MirOp::MILSoftplus { x, .. } => vec![x],
            MirOp::MILSoftplusParametric { x, .. } => vec![x],
            MirOp::MILGelu { x, .. } => vec![x],
            MirOp::MILClip { x, .. } => vec![x],
            MirOp::MILSquare { x, .. } => vec![x],
            MirOp::MILThreshold { x, .. } => vec![x],
            MirOp::MILSqrt { x, .. } => vec![x],
            MirOp::MILRsqrt { x, .. } => vec![x],
            MirOp::MILInverse { x, .. } => vec![x],
            MirOp::MILCeil { x, .. } => vec![x],
            MirOp::MILFloor { x, .. } => vec![x],
            MirOp::MILRound { x, .. } => vec![x],
            MirOp::MILExp { x, .. } => vec![x],
            MirOp::MILExp2 { x, .. } => vec![x],
            MirOp::MILLog { x, .. } => vec![x],
            MirOp::MILSign { x, .. } => vec![x],
            MirOp::MILCos { x, .. } => vec![x],
            MirOp::MILSin { x, .. } => vec![x],
            MirOp::MILTan { x, .. } => vec![x],
            MirOp::MILAcos { x, .. } => vec![x],
            MirOp::MILAsin { x, .. } => vec![x],
            MirOp::MILAtan { x, .. } => vec![x],
            MirOp::MILCosh { x, .. } => vec![x],
            MirOp::MILSinh { x, .. } => vec![x],
            MirOp::MILAtanh { x, .. } => vec![x],
            MirOp::MILErf { x, .. } => vec![x],
            MirOp::MILLogicalNot { x, .. } => vec![x],
            MirOp::MILCast { x, .. } => vec![x],
            MirOp::MILSelect { condition, x, y, .. } => vec![condition, x, y],
            MirOp::MILWhere { condition, x, y, .. } => vec![condition, x, y],
            MirOp::MILSoftmax { x, .. } => vec![x],
            MirOp::MILReduceSum { x, .. } => vec![x],
            MirOp::MILReduceMean { x, .. } => vec![x],
            MirOp::MILReduceMax { x, .. } => vec![x],
            MirOp::MILReduceMin { x, .. } => vec![x],
            MirOp::MILReduceProd { x, .. } => vec![x],
            MirOp::MILReduceSumSquare { x, .. } => vec![x],
            MirOp::MILReduceL2Norm { x, .. } => vec![x],
            MirOp::MILReduceL1Norm { x, .. } => vec![x],
            MirOp::MILReduceLogSumExp { x, .. } => vec![x],
            MirOp::MILReduceLogSum { x, .. } => vec![x],
            MirOp::MILReduceArgmax { x, .. } => vec![x],
            MirOp::MILReduceArgmin { x, .. } => vec![x],
            MirOp::MILBatchNorm { x, .. } => vec![x],
            MirOp::MILInstanceNorm { x, .. } => vec![x],
            MirOp::MILLayerNorm { x, .. } => vec![x],
            MirOp::MILL2Norm { x, .. } => vec![x],
            MirOp::MILLocalResponseNorm { x, .. } => vec![x],
            MirOp::MILMaxPool { x, .. } => vec![x],
            MirOp::MILAvgPool { x, .. } => vec![x],
            MirOp::MILL2Pool { x, .. } => vec![x],
            MirOp::MILResize { x, .. } => vec![x],
            MirOp::MILResizeNearestNeighbor { x, .. } => vec![x],
            MirOp::MILResizeBilinear { x, .. } => vec![x],
            MirOp::MILUpsampleNearestNeighbor { x, .. } => vec![x],
            MirOp::MILUpsampleBilinear { x, .. } => vec![x],
            MirOp::MILCropResize { x, boxes, box_indices, .. } => vec![x, boxes, box_indices],
            MirOp::MILAffine { x, transform, .. } => vec![x, transform],
            MirOp::MILResample { x, coordinates, .. } => vec![x, coordinates],
            MirOp::MILReshape { x, .. } => vec![x],
            MirOp::MILReshapeLike { x, ref_tensor, .. } => vec![x, ref_tensor],
            MirOp::MILTranspose { x, .. } => vec![x],
            MirOp::MILSplit { x, .. } => vec![x],
            MirOp::MILConcat { values, .. } => values.iter().collect(),
            MirOp::MILExpandDims { x, .. } => vec![x],
            MirOp::MILSqueeze { x, .. } => vec![x],
            MirOp::MILFlatten2d { x, .. } => vec![x],
            MirOp::MILReverse { x, .. } => vec![x],
            MirOp::MILReverseSequence { x, lengths, .. } => vec![x, lengths],
            MirOp::MILSliceByIndex { x, .. } => vec![x],
            MirOp::MILSliceBySize { x, .. } => vec![x],
            MirOp::MILSliceUpdate { x, update, .. } => vec![x, update],
            MirOp::MILSlidingWindows { x, .. } => vec![x],
            MirOp::MILDepthToSpace { x, .. } => vec![x],
            MirOp::MILSpaceToDepth { x, .. } => vec![x],
            MirOp::MILPixelShuffle { x, .. } => vec![x],
            MirOp::MILPixelUnshuffle { x, .. } => vec![x],
            MirOp::MILBatchToSpace { x, .. } => vec![x],
            MirOp::MILSpaceToBatch { x, .. } => vec![x],
            MirOp::MILPad { x, .. } => vec![x],
            MirOp::MILStack { values, .. } => values.iter().collect(),
            MirOp::MILTile { x, .. } => vec![x],
            MirOp::MILCumsum { x, .. } => vec![x],
            MirOp::MILFill { .. } => vec![],
            MirOp::MILFillLike { ref_tensor, .. } => vec![ref_tensor],
            MirOp::MILIdentity { x, .. } => vec![x],
            MirOp::MILOneHot { indices, .. } => vec![indices],
            MirOp::MILNonZero { x, .. } => vec![x],
            MirOp::MILArgsort { x, .. } => vec![x],
            MirOp::MILBandPart { x, .. } => vec![x],
            MirOp::MILRange1d { .. } => vec![],
            MirOp::MILShape { x, .. } => vec![x],
            MirOp::MILCrop { x, .. } => vec![x],
            MirOp::MILGather { x, indices, .. } => vec![x, indices],
            MirOp::MILGatherAlongAxis { x, indices, .. } => vec![x, indices],
            MirOp::MILGatherNd { x, indices, .. } => vec![x, indices],
            MirOp::MILScatter { x, indices, updates, .. } => vec![x, indices, updates],
            MirOp::MILScatterAlongAxis { x, indices, updates, .. } => vec![x, indices, updates],
            MirOp::MILScatterNd { x, indices, updates, .. } => vec![x, indices, updates],
            MirOp::MILNonMaximumSuppression { boxes, scores, .. } => vec![boxes, scores],
            MirOp::MILScaledDotProductAttention { query, key, value, attention_mask, .. } => {
                let mut refs = vec![query, key, value];
                if let Some(m) = attention_mask {
                    refs.push(m);
                }
                refs
            }
            MirOp::MILQuantize { x, .. } => vec![x],
            MirOp::MILDequantize { x, .. } => vec![x],
            MirOp::MILConstexprAffineDequantize { .. } => vec![],
            MirOp::MILConstexprBlockwiseShiftScale { .. } => vec![],
            MirOp::MILConstexprLutToDense { .. } => vec![],
            MirOp::MILConstexprSparseToDense { .. } => vec![],
            MirOp::MILConstexprCast { .. } => vec![],
            MirOp::MILConstexprLutToSparse { .. } => vec![],
            MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => vec![],
            MirOp::MILRnn { x, initial_h, .. } => vec![x, initial_h],
            MirOp::MILGru { x, initial_h, .. } => vec![x, initial_h],
            MirOp::MILLstm { x, initial_h, initial_c, .. } => vec![x, initial_h, initial_c],
            MirOp::MILCond { pred, .. } => vec![pred],
            MirOp::MILWhileLoop { loop_vars, .. } => loop_vars.iter().collect(),
            MirOp::MILMakeList { elems, .. } => elems.iter().collect(),
            MirOp::MILListLength { ls, .. } => vec![ls],
            MirOp::MILListWrite { ls, index, value, .. } => vec![ls, index, value],
            MirOp::MILListRead { ls, index, .. } => vec![ls, index],
            MirOp::MILListGather { ls, indices, .. } => vec![ls, indices],
            MirOp::MILListScatter { ls, indices, values, .. } => vec![ls, indices, values],
            MirOp::MILRandomBernoulli { .. } => vec![],
            MirOp::MILRandomNormal { .. } => vec![],
            MirOp::MILRandomUniform { .. } => vec![],
            MirOp::MILRandomCategorical { logits, .. } => vec![logits],
            MirOp::MILReadState { .. } => vec![],
            MirOp::MILCoremlUpdateState { value, .. } => vec![value],
            MirOp::MILStateWrite { value, .. } => vec![value],
            MirOp::MILTopk { x, .. } => vec![x],
            MirOp::MILClassify { x, .. } => vec![x],
            MirOp::AnecFusedConvActivate { x, weight, .. } => vec![x, weight],
            MirOp::AnecFusedLinearActivate { x, .. } => vec![x],
            // ─── T-P4-08: Unmapped ANEC operation stubs ──────────────
            MirOp::AnecBroadcast { x, .. } => vec![x],
            MirOp::AnecScaledElementwise { x, y, .. } => vec![x, y],
            MirOp::AnecGlobalArgMinMax { x, .. } => vec![x],
            MirOp::AnecDegamma { x, .. } => vec![x],
            MirOp::AnecDirac { x, .. } => vec![x],
            MirOp::AnecGainOffsetControl { x, .. } => vec![x],
            MirOp::AnecNRelu { x, .. } => vec![x],
            MirOp::AnecHighPrecisionSigmoid { x, .. } => vec![x],
            MirOp::AnecLog2 { x, .. } => vec![x],
            MirOp::AnecTrunc { x, .. } => vec![x],
            MirOp::AnecInvert { x, .. } => vec![x],
            MirOp::AnecUnflatten { x, .. } => vec![x],
            MirOp::AnecChannelToSpace { x, .. } => vec![x],
            MirOp::AnecSpaceToChannel { x, .. } => vec![x],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ane_engine::AneEngine;
    use super::super::ane_target::AneRevision;
    use super::{AneQuantMetadata, MirNodeId, MirOp, MirOpTargetAnnotation};

    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    // ─── T-113: default_engine_for_revision tests ──────────────────

    /// Test that default_engine() still returns the same results as before
    /// (backward compatibility). With None revision, no overrides are applied.
    #[test]
    fn test_default_engine_backward_compat() {
        // NE pipeline ops
        let conv = MirOp::MILConv {
            name: "c".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        assert_eq!(conv.default_engine(), Some(AneEngine::NE));

        // PE pipeline ops
        let add = MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") };
        assert_eq!(add.default_engine(), Some(AneEngine::PE));

        let relu = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        assert_eq!(relu.default_engine(), Some(AneEngine::PE));

        // TransposeEngine
        let transpose = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
        assert_eq!(transpose.default_engine(), Some(AneEngine::TransposeEngine));

        // CPU-only ops
        let const_op = MirOp::MILConst {
            name: "c".into(),
            value_path: "v".into(),
            dtype: super::MilDtype::Fp16,
        };
        assert_eq!(const_op.default_engine(), None);

        // ReduceArgmax returns PE when revision is None (backward compat)
        let argmax =
            MirOp::MILReduceArgmax { name: "am".into(), x: nid("x"), axis: 1, keep_dims: false };
        assert_eq!(argmax.default_engine(), Some(AneEngine::PE));

        // ReduceL2Norm returns PE when revision is None
        let l2 = MirOp::MILReduceL2Norm {
            name: "l2".into(),
            x: nid("x"),
            axes: vec![1],
            keep_dims: false,
        };
        assert_eq!(l2.default_engine(), Some(AneEngine::PE));

        // MILSquare returns PE when revision is None
        let square = MirOp::MILSquare { name: "sq".into(), x: nid("x") };
        assert_eq!(square.default_engine(), Some(AneEngine::PE));

        // ScaledDotProductAttention returns NE when revision is None
        let sdpa = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: None,
        };
        assert_eq!(sdpa.default_engine(), Some(AneEngine::NE));
    }

    /// Test that ReduceArgmax returns None on A18 (V19 — no LSE_7 converter).
    #[test]
    fn test_reduce_argmax_none_on_a18() {
        let argmax =
            MirOp::MILReduceArgmax { name: "am".into(), x: nid("x"), axis: 1, keep_dims: false };

        // A18 revisions: V19, V20, V26
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V19)), None);
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V20)), None);
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V26)), None);

        // Non-A18 revisions should still return PE (they have converters)
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V4)), Some(AneEngine::PE));
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V7)), Some(AneEngine::PE));
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V10)), Some(AneEngine::PE));
        assert_eq!(argmax.default_engine_for_revision(Some(AneRevision::V11)), Some(AneEngine::PE));
    }

    /// Test that ReduceArgmin returns None on A18 (V19 — no LSE_7 converter).
    #[test]
    fn test_reduce_argmin_none_on_a18() {
        let argmin =
            MirOp::MILReduceArgmin { name: "amin".into(), x: nid("x"), axis: 1, keep_dims: false };

        // A18 revisions
        assert_eq!(argmin.default_engine_for_revision(Some(AneRevision::V19)), None);
        assert_eq!(argmin.default_engine_for_revision(Some(AneRevision::V20)), None);
        assert_eq!(argmin.default_engine_for_revision(Some(AneRevision::V26)), None);

        // Non-A18 revisions should still return PE
        assert_eq!(argmin.default_engine_for_revision(Some(AneRevision::V5)), Some(AneEngine::PE));
        assert_eq!(argmin.default_engine_for_revision(Some(AneRevision::V8)), Some(AneEngine::PE));
    }

    /// Test that ReduceL2Norm returns None on A11Legacy/A12 families
    /// (uses_a14minus_converters — no reduce_l2_norm converter).
    #[test]
    fn test_reduce_l2norm_none_on_a11legacy_a12() {
        let l2 = MirOp::MILReduceL2Norm {
            name: "l2".into(),
            x: nid("x"),
            axes: vec![1],
            keep_dims: false,
        };

        // A11Legacy (V4) and A12 (V5) — uses_a14minus_converters
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V4)), None);
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V5)), None);

        // A13 (V6) also uses A14Minus converters
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V6)), None);

        // A14+ should return PE (A14Plus converters have reduce_l2_norm)
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V7)), Some(AneEngine::PE));
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V10)), Some(AneEngine::PE));
        assert_eq!(l2.default_engine_for_revision(Some(AneRevision::V19)), Some(AneEngine::PE));
    }

    /// Test that MILSquare returns None on A11Legacy/A12/A13 families
    /// (uses_a14minus_converters — per-op matrix row 19 A13Minus/A14Plus split).
    #[test]
    fn test_square_none_on_a14minus_families() {
        let square = MirOp::MILSquare { name: "sq".into(), x: nid("x") };

        // A14Minus converter families: A11Legacy, A12, A13
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V4)), None); // A11Legacy
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V5)), None); // A12
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V6)), None); // A13

        // A14+ should return PE (A14Plus converters have square)
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V7)), Some(AneEngine::PE)); // A14
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V8)), Some(AneEngine::PE)); // A15
        assert_eq!(square.default_engine_for_revision(Some(AneRevision::V19)), Some(AneEngine::PE));
        // A18
    }

    /// Test that MILConv returns Some(NE) for all revisions.
    /// Conv is a core NE pipeline op with converters on every family.
    #[test]
    fn test_conv_ne_for_all_revisions() {
        let conv = MirOp::MILConv {
            name: "c".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };

        // Conv should always return NE regardless of revision
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
            assert_eq!(
                conv.default_engine_for_revision(Some(rev)),
                Some(AneEngine::NE),
                "Conv should be NE on revision {:?}",
                rev
            );
        }

        // Also test with None (backward compat)
        assert_eq!(conv.default_engine_for_revision(None), Some(AneEngine::NE));
    }

    /// Test that ScaledDotProductAttention returns NE on A16+ but None on older families.
    #[test]
    fn test_sdpa_revision_aware() {
        let sdpa = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: None,
        };

        // Pre-A16 families: no reliable SDPA converter → None
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V4)), None); // A11Legacy
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V5)), None); // A12
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V6)), None); // A13
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V7)), None); // A14
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V8)), None); // A15

        // A16+ families: reliable SDPA converter → NE
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V10)), Some(AneEngine::NE)); // A16
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V11)), Some(AneEngine::NE)); // A17
        assert_eq!(sdpa.default_engine_for_revision(Some(AneRevision::V19)), Some(AneEngine::NE)); // A18

        // With None revision: backward compat returns base engine (NE)
        assert_eq!(sdpa.default_engine_for_revision(None), Some(AneEngine::NE));
    }

    // ─── T-P5-08: MILConv ANE attrs moved to target annotation ───────

    /// T-P5-08: Verify that MILConv no longer has ANE-specific quantization
    /// fields (kernel_scale, kernel_zero_point, kernel_palettized_lut).
    /// These have been moved to MirOpTargetAnnotation::ane_quant.
    #[test]
    fn test_tp5_08_conv_no_ane_quant_fields() {
        let conv = MirOp::MILConv {
            name: "plain_conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };

        // Conv still maps to NE engine (target-agnostic IR)
        assert_eq!(conv.default_engine(), Some(AneEngine::NE));

        // Conv has no ANE quant fields — they live on MirOpTargetAnnotation
        // Verify MirOpTargetAnnotation defaults to no ane_quant
        let annotation = MirOpTargetAnnotation::default();
        assert!(annotation.ane_quant.is_none());
    }

    /// T-P5-08: Verify that AneQuantMetadata captures conv quantization info
    /// and is stored in MirOpTargetAnnotation.
    #[test]
    fn test_tp5_08_ane_quant_metadata() {
        let quant = AneQuantMetadata {
            kernel_scale: 0.0078,
            kernel_zero_point: 0,
            kernel_palettized_lut: "conv_weight_lut_4bit".to_string(),
        };

        assert!((quant.kernel_scale - 0.0078).abs() < f32::EPSILON);
        assert_eq!(quant.kernel_zero_point, 0);
        assert_eq!(quant.kernel_palettized_lut, "conv_weight_lut_4bit");

        // Store in target annotation
        let annotation = MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::NE),
            demoted_to_cpu: false,
            target_revision: None,
            ane_quant: Some(quant.clone()),
        };

        let retrieved = annotation.ane_quant.as_ref().unwrap();
        assert!((retrieved.kernel_scale - 0.0078).abs() < f32::EPSILON);
        assert_eq!(retrieved.kernel_zero_point, 0);
        assert_eq!(retrieved.kernel_palettized_lut, "conv_weight_lut_4bit");
    }

    /// T-P5-08: Verify AneQuantMetadata serialization round-trip.
    #[test]
    fn test_tp5_08_ane_quant_metadata_serde() {
        let quant = AneQuantMetadata {
            kernel_scale: 0.0156,
            kernel_zero_point: -128,
            kernel_palettized_lut: "conv_weight_lut_8bit".to_string(),
        };
        let annotation = MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::NE),
            demoted_to_cpu: false,
            target_revision: None,
            ane_quant: Some(quant),
        };

        let json = serde_json::to_string(&annotation).unwrap();
        let deserialized: MirOpTargetAnnotation = serde_json::from_str(&json).unwrap();
        assert_eq!(annotation, deserialized);
    }

    /// T-P5-08: Verify MirOpTargetAnnotation with ane_quant=None deserializes
    /// from JSON that lacks the ane_quant field (backward compat).
    #[test]
    fn test_tp5_08_annotation_backward_compat_deser() {
        // JSON without ane_quant field (old format)
        let old_json = r#"{"assigned_engine":null,"demoted_to_cpu":false,"target_revision":null}"#;
        let deserialized: MirOpTargetAnnotation = serde_json::from_str(old_json).unwrap();
        assert!(deserialized.ane_quant.is_none());
        assert!(!deserialized.demoted_to_cpu);
    }

    // ─── T-131: MatMul transpose flags ──────────────────────────────────

    /// T-131: Verify MatMul retains transpose_y flag for named-const-node
    /// emission in the Apple proto path. The actual emission change is in
    /// coreml-proto, but the MIR struct must carry the flag correctly.
    #[test]
    fn test_t131_matmul_transpose_y_flag() {
        let mm_true =
            MirOp::MILMatMul { name: "mm_t".into(), x: nid("a"), y: nid("b"), transpose_y: true };
        let mm_false =
            MirOp::MILMatMul { name: "mm_f".into(), x: nid("a"), y: nid("b"), transpose_y: false };

        if let MirOp::MILMatMul { transpose_y, .. } = &mm_true {
            assert!(*transpose_y, "transpose_y should be true");
        }
        if let MirOp::MILMatMul { transpose_y, .. } = &mm_false {
            assert!(!*transpose_y, "transpose_y should be false");
        }
    }
}
