//! ANE-Legal IR (AIR)
//!
//! The graph after legality verification. All 167 MIL ops have
//! corresponding AIR representations for full coverage.

use super::common::IrNodeId;
use serde::{Deserialize, Serialize};

// ─── Legality Status ────────────────────────────────────────────

/// Legality status for an AIR node, replacing f32 risk/confidence fields.
///
/// Provides structured legality information instead of floating-point risk scores.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[non_exhaustive]
pub enum LegalityStatus {
    /// The op has been verified to run correctly on ANE.
    Verified,
    /// The op has not been verified on ANE (may work, may fallback).
    #[default]
    Unverified,
    /// The op is likely to fall back to CPU based on known constraints.
    LikelyFallback,
    /// The legality is unknown (insufficient information).
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AirNodeId(pub String);

impl IrNodeId for AirNodeId {
    fn as_str(&self) -> &str {
        &self.0
    }

    fn from_string(s: String) -> Self {
        AirNodeId(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AirOp {
    // ─── Constants ───────────────────────────────────────────────
    Const {
        value_path: String,
        dtype: super::common::MilDtype,
    },

    // ─── Linear / FC ─────────────────────────────────────────────
    Linear {
        input: AirNodeId,
        weight: String,
        bias: Option<String>,
    },
    MatMul {
        a: AirNodeId,
        b: AirNodeId,
    },
    Einsum {
        inputs: Vec<AirNodeId>,
        equation: String,
    },
    Conv1x1AsLinear {
        input: AirNodeId,
        weight: String,
        pad_type: String,
        /// Output feature dimension for this linear projection.
        /// When None, the output dim is unknown and shape inference must fall back
        /// to propagating the input shape.
        #[serde(default)]
        output_dim: Option<usize>,
    },

    // ─── Convolution ─────────────────────────────────────────────
    Conv {
        input: AirNodeId,
        weight: AirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
    },
    ConvTranspose {
        input: AirNodeId,
        weight: AirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
        output_shape: Vec<usize>,
    },

    // ─── Elementwise Binary ──────────────────────────────────────
    Add {
        x: AirNodeId,
        y: AirNodeId,
    },
    Mul {
        x: AirNodeId,
        y: AirNodeId,
    },
    Sub {
        x: AirNodeId,
        y: AirNodeId,
    },
    Maximum {
        x: AirNodeId,
        y: AirNodeId,
    },
    Minimum {
        x: AirNodeId,
        y: AirNodeId,
    },
    RealDiv {
        x: AirNodeId,
        y: AirNodeId,
    },
    FloorDiv {
        x: AirNodeId,
        y: AirNodeId,
    },
    Mod {
        x: AirNodeId,
        y: AirNodeId,
    },
    Pow {
        x: AirNodeId,
        y: AirNodeId,
    },
    Equal {
        x: AirNodeId,
        y: AirNodeId,
    },
    NotEqual {
        x: AirNodeId,
        y: AirNodeId,
    },
    Greater {
        x: AirNodeId,
        y: AirNodeId,
    },
    GreaterEqual {
        x: AirNodeId,
        y: AirNodeId,
    },
    Less {
        x: AirNodeId,
        y: AirNodeId,
    },
    LessEqual {
        x: AirNodeId,
        y: AirNodeId,
    },
    LogicalAnd {
        x: AirNodeId,
        y: AirNodeId,
    },
    LogicalOr {
        x: AirNodeId,
        y: AirNodeId,
    },
    LogicalXor {
        x: AirNodeId,
        y: AirNodeId,
    },

    // ─── Elementwise Unary ───────────────────────────────────────
    Abs {
        input: AirNodeId,
    },
    Neg {
        input: AirNodeId,
    },
    Sigmoid {
        input: AirNodeId,
    },
    Tanh {
        input: AirNodeId,
    },
    Relu {
        input: AirNodeId,
    },
    Relu6 {
        input: AirNodeId,
    },
    LeakyRelu {
        input: AirNodeId,
        alpha: f32,
    },
    SigmoidHard {
        input: AirNodeId,
        alpha: f32,
        beta: f32,
    },
    ThresholdedRelu {
        input: AirNodeId,
        alpha: f32,
    },
    ClampedRelu {
        input: AirNodeId,
        alpha: f32,
        beta: f32,
    },
    LinearActivation {
        input: AirNodeId,
        alpha: f32,
        beta: f32,
    },
    Prelu {
        input: AirNodeId,
        alpha: String,
    },
    Softsign {
        input: AirNodeId,
    },
    Silu {
        input: AirNodeId,
    },
    ScaledTanh {
        input: AirNodeId,
        alpha: f32,
        beta: f32,
    },
    Elu {
        input: AirNodeId,
        alpha: f32,
    },
    Softplus {
        input: AirNodeId,
    },
    SoftplusParametric {
        input: AirNodeId,
        alpha: String,
        beta: String,
    },
    Gelu {
        input: AirNodeId,
        mode: String,
    },
    Clip {
        input: AirNodeId,
        min_val: f32,
        max_val: f32,
    },
    Square {
        input: AirNodeId,
    },
    Threshold {
        input: AirNodeId,
        alpha: f32,
    },
    Sqrt {
        input: AirNodeId,
    },
    Rsqrt {
        input: AirNodeId,
    },
    Inverse {
        input: AirNodeId,
        epsilon: f32,
    },
    Ceil {
        input: AirNodeId,
    },
    Floor {
        input: AirNodeId,
    },
    Round {
        input: AirNodeId,
    },
    Exp {
        input: AirNodeId,
    },
    Exp2 {
        input: AirNodeId,
    },
    Log {
        input: AirNodeId,
        epsilon: f32,
    },
    Sign {
        input: AirNodeId,
    },
    Cos {
        input: AirNodeId,
    },
    Sin {
        input: AirNodeId,
    },
    Tan {
        input: AirNodeId,
    },
    Acos {
        input: AirNodeId,
    },
    Asin {
        input: AirNodeId,
    },
    Atan {
        input: AirNodeId,
    },
    Cosh {
        input: AirNodeId,
    },
    Sinh {
        input: AirNodeId,
    },
    Atanh {
        input: AirNodeId,
    },
    Erf {
        input: AirNodeId,
    },
    LogicalNot {
        input: AirNodeId,
    },
    Cast {
        input: AirNodeId,
        dtype: super::common::MilDtype,
    },
    Select {
        condition: AirNodeId,
        x: AirNodeId,
        y: AirNodeId,
    },
    Where {
        condition: AirNodeId,
        x: AirNodeId,
        y: AirNodeId,
    },
    Softmax {
        input: AirNodeId,
        axis: isize,
    },

    // ─── Reduction ───────────────────────────────────────────────
    ReduceSum {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMean {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMax {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMin {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceProd {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceSumSquare {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceL2Norm {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceL1Norm {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceLogSumExp {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceLogSum {
        input: AirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceArgmax {
        input: AirNodeId,
        axis: usize,
        keep_dims: bool,
    },
    ReduceArgmin {
        input: AirNodeId,
        axis: usize,
        keep_dims: bool,
    },

    // ─── Normalization ───────────────────────────────────────────
    BatchNorm {
        input: AirNodeId,
        mean: String,
        variance: String,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    InstanceNorm {
        input: AirNodeId,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    LayerNorm {
        input: AirNodeId,
        weight: String,
        bias: Option<String>,
        epsilon: f32,
        axes: Vec<usize>,
    },
    L2Norm {
        input: AirNodeId,
        epsilon: f32,
        axes: Vec<usize>,
    },
    LocalResponseNorm {
        input: AirNodeId,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    },

    // ─── Pooling ─────────────────────────────────────────────────
    MaxPool {
        input: AirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },
    AvgPool {
        input: AirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
        count_include_padding: bool,
    },
    L2Pool {
        input: AirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },

    // ─── Image Resizing ──────────────────────────────────────────
    Resize {
        input: AirNodeId,
        target_size: Vec<usize>,
        mode: String,
        sampling_mode: String,
        nearest_rounding_mode: String,
    },
    ResizeNearestNeighbor {
        input: AirNodeId,
        target_height: usize,
        target_width: usize,
    },
    ResizeBilinear {
        input: AirNodeId,
        target_height: usize,
        target_width: usize,
        align_corners: bool,
    },
    UpsampleNearestNeighbor {
        input: AirNodeId,
        scale: Vec<usize>,
    },
    UpsampleBilinear {
        input: AirNodeId,
        scale: Vec<usize>,
        align_corners: bool,
        half_pixel_centers: bool,
    },
    CropResize {
        input: AirNodeId,
        boxes: AirNodeId,
        box_indices: AirNodeId,
        crop_height: usize,
        crop_width: usize,
    },
    Affine {
        input: AirNodeId,
        transform: AirNodeId,
        output_height: usize,
        output_width: usize,
        sampling_mode: String,
        pad_value: f32,
    },
    Resample {
        input: AirNodeId,
        coordinates: AirNodeId,
        sampling_mode: String,
        pad_value: f32,
    },

    // ─── Tensor Transform ────────────────────────────────────────
    Reshape {
        input: AirNodeId,
        target_shape: Vec<usize>,
    },
    ReshapeLike {
        input: AirNodeId,
        ref_tensor: AirNodeId,
    },
    Transpose {
        input: AirNodeId,
        perm: Vec<usize>,
    },
    Split {
        input: AirNodeId,
        axis: usize,
        num_splits: usize,
    },
    Concat {
        inputs: Vec<AirNodeId>,
        axis: usize,
    },
    ExpandDims {
        input: AirNodeId,
        axis: Vec<usize>,
    },
    Squeeze {
        input: AirNodeId,
        axis: Vec<usize>,
    },
    Flatten2d {
        input: AirNodeId,
        axis: usize,
    },
    Reverse {
        input: AirNodeId,
        axes: Vec<usize>,
    },
    ReverseSequence {
        input: AirNodeId,
        lengths: AirNodeId,
        batch_axis: usize,
        seq_axis: usize,
    },
    SliceByIndex {
        input: AirNodeId,
        begin: Vec<i64>,
        end: Vec<i64>,
        stride: Vec<i64>,
        begin_mask: Vec<bool>,
        end_mask: Vec<bool>,
        squeeze_mask: Vec<bool>,
    },
    SliceBySize {
        input: AirNodeId,
        begin: Vec<i64>,
        size: Vec<i64>,
    },
    SliceUpdate {
        input: AirNodeId,
        update: AirNodeId,
        begin: Vec<i64>,
        end: Vec<i64>,
    },
    SlidingWindows {
        input: AirNodeId,
        axis: usize,
        window_size: usize,
        stride: usize,
    },
    DepthToSpace {
        input: AirNodeId,
        block_size: usize,
    },
    SpaceToDepth {
        input: AirNodeId,
        block_size: usize,
    },
    PixelShuffle {
        input: AirNodeId,
        upscale_factor: usize,
    },
    PixelUnshuffle {
        input: AirNodeId,
        downscale_factor: usize,
    },
    BatchToSpace {
        input: AirNodeId,
        block_shape: Vec<usize>,
        crops: Vec<(usize, usize)>,
    },
    SpaceToBatch {
        input: AirNodeId,
        block_shape: Vec<usize>,
        paddings: Vec<(usize, usize)>,
    },
    Pad {
        input: AirNodeId,
        pad_amounts: Vec<i64>,
        mode: String,
        constant_value: f32,
    },
    Stack {
        values: Vec<AirNodeId>,
        axis: usize,
    },
    Tile {
        input: AirNodeId,
        reps: Vec<usize>,
    },
    Cumsum {
        input: AirNodeId,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    },
    Fill {
        shape: Vec<usize>,
        value: f32,
        dtype: super::common::MilDtype,
    },
    FillLike {
        ref_tensor: AirNodeId,
        value: f32,
        dtype: super::common::MilDtype,
    },
    Identity {
        input: AirNodeId,
    },
    OneHot {
        indices: AirNodeId,
        one_hot_vector_size: usize,
        on_value: f32,
        off_value: f32,
        axis: usize,
        dtype: super::common::MilDtype,
    },
    NonZero {
        input: AirNodeId,
    },
    Argsort {
        input: AirNodeId,
        axis: usize,
        ascending: bool,
    },
    BandPart {
        input: AirNodeId,
        num_lower: i64,
        num_upper: i64,
    },
    Range1d {
        start: f32,
        end: f32,
        step: f32,
    },
    Shape {
        input: AirNodeId,
    },
    Crop {
        input: AirNodeId,
        crop_height: usize,
        crop_width: usize,
        offset_height: usize,
        offset_width: usize,
    },

    // ─── Scatter / Gather ────────────────────────────────────────
    Gather {
        input: AirNodeId,
        indices: AirNodeId,
        axis: isize,
    },
    GatherAlongAxis {
        input: AirNodeId,
        indices: AirNodeId,
        axis: isize,
    },
    GatherNd {
        input: AirNodeId,
        indices: AirNodeId,
    },
    Scatter {
        input: AirNodeId,
        indices: AirNodeId,
        updates: AirNodeId,
        axis: isize,
        mode: String,
    },
    ScatterAlongAxis {
        input: AirNodeId,
        indices: AirNodeId,
        updates: AirNodeId,
        axis: isize,
    },
    ScatterNd {
        input: AirNodeId,
        indices: AirNodeId,
        updates: AirNodeId,
    },
    NonMaximumSuppression {
        boxes: AirNodeId,
        scores: AirNodeId,
        iou_threshold: f32,
        score_threshold: f32,
        max_detections: usize,
    },

    // ─── Attention ───────────────────────────────────────────────
    ScaledDotProductAttention {
        query: AirNodeId,
        key: AirNodeId,
        value: AirNodeId,
        attention_mask: Option<AirNodeId>,
        scale: Option<f32>,
    },

    // ─── Quantization ────────────────────────────────────────────
    Quantize {
        input: AirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: super::common::MilDtype,
    },
    Dequantize {
        input: AirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: super::common::MilDtype,
    },

    // ─── Constexpr / Compression ─────────────────────────────────
    ConstexprAffineDequantize {
        quantized_data: String,
        scale: f32,
        zero_point: i32,
        axis: isize,
    },
    ConstexprBlockwiseShiftScale {
        data: String,
        scale: String,
        offset: String,
        block_size: Vec<usize>,
    },
    ConstexprLutToDense {
        indices: String,
        lut: String,
        num_bits: usize,
    },
    ConstexprSparseToDense {
        nonzero_data: String,
        shape: Vec<usize>,
        default_value: f32,
    },
    ConstexprCast {
        data: String,
        dtype: super::common::MilDtype,
    },
    ConstexprLutToSparse {
        data: String,
        num_bits: usize,
    },
    ConstexprSparseBlockwiseShiftScale {
        data: String,
        scale: String,
        offset: String,
        block_size: Vec<usize>,
        block_axis: usize,
    },

    // ─── Recurrent ───────────────────────────────────────────────
    Rnn {
        input: AirNodeId,
        initial_h: AirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        mode: String,
        output_sequence: bool,
    },
    Gru {
        input: AirNodeId,
        initial_h: AirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        reset_after: bool,
        output_sequence: bool,
    },
    Lstm {
        input: AirNodeId,
        initial_h: AirNodeId,
        initial_c: AirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        output_sequence: bool,
    },

    // ─── Control Flow ────────────────────────────────────────────
    Cond {
        pred: AirNodeId,
        true_graph: String,
        false_graph: String,
    },
    WhileLoop {
        condition: String,
        body: String,
        loop_vars: Vec<AirNodeId>,
    },
    MakeList {
        elems: Vec<AirNodeId>,
        dtype: super::common::MilDtype,
    },
    ListLength {
        ls: AirNodeId,
    },
    ListWrite {
        ls: AirNodeId,
        index: AirNodeId,
        value: AirNodeId,
    },
    ListRead {
        ls: AirNodeId,
        index: AirNodeId,
    },
    ListGather {
        ls: AirNodeId,
        indices: AirNodeId,
    },
    ListScatter {
        ls: AirNodeId,
        indices: AirNodeId,
        values: AirNodeId,
    },

    // ─── Random ──────────────────────────────────────────────────
    RandomBernoulli {
        shape: Vec<usize>,
        prob: f32,
        seed: Option<u64>,
        dtype: super::common::MilDtype,
    },
    RandomNormal {
        shape: Vec<usize>,
        mean: f32,
        stddev: f32,
        seed: Option<u64>,
        dtype: super::common::MilDtype,
    },
    RandomUniform {
        shape: Vec<usize>,
        low: f32,
        high: f32,
        seed: Option<u64>,
        dtype: super::common::MilDtype,
    },
    RandomCategorical {
        logits: AirNodeId,
        num_samples: usize,
        seed: Option<u64>,
        dtype: super::common::MilDtype,
    },

    // ─── State ───────────────────────────────────────────────────
    StateReadFixed {
        state_id: String,
        shape: Vec<usize>,
        dtype: super::common::MilDtype,
    },
    StateWriteFixed {
        state_id: String,
        value: AirNodeId,
    },

    // ─── Metadata / Misc ─────────────────────────────────────────
    Topk {
        input: AirNodeId,
        k: usize,
        axis: isize,
    },
    Classify {
        input: AirNodeId,
    },

    // ─── Legacy: kept for backward compat with existing code ─────
    StaticLUTProjection {
        input: AirNodeId,
        indices: String,
        lut: String,
        group_size: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirNode {
    pub id: AirNodeId,
    pub op: AirOp,
    pub name: String,
    pub legality_confidence: f32,
    pub sir_source: Option<super::sir::SirNodeId>,
    pub fallback_risk: f32,
    pub drift_risk: f32,
    pub precision_override: Option<String>,
    /// Structured legality status replacing f32 risk fields.
    #[serde(default)]
    pub legality_status: LegalityStatus,
}

/// Legacy fields for backward-compatible deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyAirNodeFields {
    id: AirNodeId,
    op: AirOp,
    name: String,
    legality_confidence: f32,
    sir_source: Option<super::sir::SirNodeId>,
    fallback_risk: f32,
    drift_risk: f32,
    precision_override: Option<String>,
}

impl TryFrom<LegacyAirNodeFields> for AirNode {
    type Error = String;
    fn try_from(legacy: LegacyAirNodeFields) -> Result<Self, Self::Error> {
        let legality_status = if legacy.legality_confidence >= 0.95 {
            LegalityStatus::Verified
        } else if legacy.fallback_risk > 0.5 {
            LegalityStatus::LikelyFallback
        } else if legacy.legality_confidence < 0.1 {
            LegalityStatus::Unknown
        } else {
            LegalityStatus::Unverified
        };
        Ok(AirNode {
            id: legacy.id,
            op: legacy.op,
            name: legacy.name,
            legality_confidence: legacy.legality_confidence,
            sir_source: legacy.sir_source,
            fallback_risk: legacy.fallback_risk,
            drift_risk: legacy.drift_risk,
            precision_override: legacy.precision_override,
            legality_status,
        })
    }
}

impl From<AirNode> for LegacyAirNodeFields {
    fn from(node: AirNode) -> Self {
        LegacyAirNodeFields {
            id: node.id,
            op: node.op,
            name: node.name,
            legality_confidence: node.legality_confidence,
            sir_source: node.sir_source,
            fallback_risk: node.fallback_risk,
            drift_risk: node.drift_risk,
            precision_override: node.precision_override,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirGraph {
    pub nodes: Vec<AirNode>,
    pub inputs: Vec<AirNodeId>,
    pub outputs: Vec<AirNodeId>,
    pub staticization_decisions: Vec<StaticizationDecision>,
}

impl AirGraph {
    /// Verify graph invariants: no duplicate node IDs, all inputs/outputs reference existing nodes.
    pub fn verify(&self) -> Result<(), super::common::VerifyError> {
        use std::collections::HashSet;
        let seen_ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        if seen_ids.len() != self.nodes.len() {
            return Err(super::common::VerifyError { message: "Duplicate node IDs in AirGraph".into() });
        }
        for input_id in &self.inputs {
            if !seen_ids.contains(input_id.as_str()) {
                return Err(super::common::VerifyError { message: format!("AirGraph input '{}' not found in nodes", input_id.0) });
            }
        }
        for output_id in &self.outputs {
            if !seen_ids.contains(output_id.as_str()) {
                return Err(super::common::VerifyError { message: format!("AirGraph output '{}' not found in nodes", output_id.0) });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticizationDecision {
    pub original_dynamic: String,
    pub resolved_static: String,
    pub method: String,
}
