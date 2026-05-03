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
}

impl MirOp {
    /// Returns the default ANE execution engine for this op.
    ///
    /// Based on observed ANE fusion boundaries:
    /// - **NE**: conv/pool/matmul/attention pipeline
    /// - **PE**: elementwise/reduction/scaled-EW pipeline
    /// - **TransposeEngine**: data rearrangement
    /// - **None**: CPU-only ops (control flow, random, scatter, state, constexpr, etc.)
    ///
    /// Source: ane-constraints-docs/03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md
    pub fn default_engine(&self) -> Option<super::ane_engine::AneEngine> {
        use super::ane_engine::AneEngine;
        match self {
            // ─── NE pipeline: conv/pool/matmul/attention ────────────
            MirOp::MILLinear { .. }
            | MirOp::MILMatMul { .. }
            // MILEinsum moved to CPU-only (T-22): no ANEC converter.
            | MirOp::MILConv { .. }
            | MirOp::MILConvTranspose { .. }
            | MirOp::MILScaledDotProductAttention { .. }
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
            | MirOp::MILNeg { .. }
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
            // Slice / reverse
            | MirOp::MILSliceByIndex { .. }
            | MirOp::MILSliceBySize { .. }
            | MirOp::MILSliceUpdate { .. }
            | MirOp::MILSlidingWindows { .. }
            | MirOp::MILReverse { .. }
            // MILReverseSequence moved to CPU-only (T-22).
            // Quantize / dequantize (ANE-legal per per-op matrix rows 84-85,
            // but currently lack MirOpCompat variants — see T-39)
            | MirOp::MILQuantize { .. }
            | MirOp::MILDequantize { .. }
            // Sort / topk (ANE-legal)
            | MirOp::MILArgsort { .. }
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
            | MirOp::MILEinsum { .. } => None,
        }
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
            MirOp::MILConstexprSparseBlockwiseShiftScale { .. } => "constexpr_sparse_blockwise_shift_scale",
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
        }
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
