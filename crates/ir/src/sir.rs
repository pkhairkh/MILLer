//! Semantic/Task IR (SIR)
//!
//! The highest level of abstraction. All 167 MIL ops have corresponding
//! SIR representations. Complex ops (AttentionBlock, DecodeStep, RMSNorm,
//! etc.) decompose into multiple AIR ops; simple ops map 1:1.

use super::common::IrNodeId;
use super::common::MilDtype;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SirNodeId(pub String);

impl IrNodeId for SirNodeId {
    fn as_str(&self) -> &str {
        &self.0
    }

    fn from_string(s: String) -> Self {
        SirNodeId(s)
    }
}

/// Default axes for RMSNorm: [2] for 3D [batch, seq, embed] tensors.
/// Used for serde backward compatibility when deserializing older SIR
/// graphs that lack the `axes` field.
fn default_rms_norm_axes() -> Vec<usize> {
    vec![2]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SirOp {
    // ─── Composite / High-Level Semantic Ops ─────────────────────
    LinearProjection {
        input: SirNodeId,
        weight: String,
        bias: Option<String>,
    },
    AttentionBlock {
        q: SirNodeId,
        k: SirNodeId,
        v: SirNodeId,
        mask: Option<SirNodeId>,
        rope: Option<SirNodeId>,
    },
    RMSNorm {
        input: SirNodeId,
        weight: String,
        epsilon: f32,
        /// Axes to reduce over for norm computation.
        /// For 3D [batch, seq, embed]: axes=[2] (standard layer norm / rms norm).
        /// For 4D [batch, seq, heads, head_dim]: axes=[3] (per-head q/k norm).
        #[serde(default = "default_rms_norm_axes")]
        axes: Vec<usize>,
    },
    RoPETransform {
        input: SirNodeId,
        tables: String,
    },
    DecodeStep {
        token: SirNodeId,
        state_map: Vec<String>,
    },
    Sampler {
        logits: SirNodeId,
        temperature: f32,
        top_p: f32,
        rep_penalty: f32,
        /// Min-p pruning threshold. Tokens with probability below
        /// min_p * max_probability are pruned before sampling.
        min_p: f32,
        /// K value for top-k pre-candidate selection.
        top_k: usize,
        /// Whether to use Gumbel noise injection for stochastic sampling.
        gumbel_noise: bool,
    },
    StateRead {
        state_id: String,
        offset: usize,
        shape: Vec<usize>,
    },
    StateWrite {
        state_id: String,
        offset: usize,
        value: SirNodeId,
    },

    // ─── Constants ───────────────────────────────────────────────
    Const {
        value_path: String,
        dtype: MilDtype,
    },

    // ─── Linear / FC ─────────────────────────────────────────────
    MatMul {
        a: SirNodeId,
        b: SirNodeId,
    },
    Einsum {
        inputs: Vec<SirNodeId>,
        equation: String,
    },

    // ─── Convolution ─────────────────────────────────────────────
    Conv {
        input: SirNodeId,
        weight: SirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
    },
    ConvTranspose {
        input: SirNodeId,
        weight: SirNodeId,
        pad_type: String,
        groups: usize,
        strides: Vec<usize>,
        pad_amounts: Vec<usize>,
        dilations: Vec<usize>,
        output_shape: Vec<usize>,
    },

    // ─── Elementwise Binary ──────────────────────────────────────
    Add {
        x: SirNodeId,
        y: SirNodeId,
    },
    Mul {
        x: SirNodeId,
        y: SirNodeId,
    },
    Sub {
        x: SirNodeId,
        y: SirNodeId,
    },
    Maximum {
        x: SirNodeId,
        y: SirNodeId,
    },
    Minimum {
        x: SirNodeId,
        y: SirNodeId,
    },
    RealDiv {
        x: SirNodeId,
        y: SirNodeId,
    },
    FloorDiv {
        x: SirNodeId,
        y: SirNodeId,
    },
    Mod {
        x: SirNodeId,
        y: SirNodeId,
    },
    Pow {
        x: SirNodeId,
        y: SirNodeId,
    },
    Equal {
        x: SirNodeId,
        y: SirNodeId,
    },
    NotEqual {
        x: SirNodeId,
        y: SirNodeId,
    },
    Greater {
        x: SirNodeId,
        y: SirNodeId,
    },
    GreaterEqual {
        x: SirNodeId,
        y: SirNodeId,
    },
    Less {
        x: SirNodeId,
        y: SirNodeId,
    },
    LessEqual {
        x: SirNodeId,
        y: SirNodeId,
    },
    LogicalAnd {
        x: SirNodeId,
        y: SirNodeId,
    },
    LogicalOr {
        x: SirNodeId,
        y: SirNodeId,
    },
    LogicalXor {
        x: SirNodeId,
        y: SirNodeId,
    },

    // ─── Elementwise Unary ───────────────────────────────────────
    Abs {
        input: SirNodeId,
    },
    Neg {
        input: SirNodeId,
    },
    Sigmoid {
        input: SirNodeId,
    },
    Tanh {
        input: SirNodeId,
    },
    Relu {
        input: SirNodeId,
    },
    Relu6 {
        input: SirNodeId,
    },
    LeakyRelu {
        input: SirNodeId,
        alpha: f32,
    },
    SigmoidHard {
        input: SirNodeId,
        alpha: f32,
        beta: f32,
    },
    ThresholdedRelu {
        input: SirNodeId,
        alpha: f32,
    },
    ClampedRelu {
        input: SirNodeId,
        alpha: f32,
        beta: f32,
    },
    LinearActivation {
        input: SirNodeId,
        alpha: f32,
        beta: f32,
    },
    Prelu {
        input: SirNodeId,
        alpha: String,
    },
    Softsign {
        input: SirNodeId,
    },
    Silu {
        input: SirNodeId,
    },
    ScaledTanh {
        input: SirNodeId,
        alpha: f32,
        beta: f32,
    },
    Elu {
        input: SirNodeId,
        alpha: f32,
    },
    Softplus {
        input: SirNodeId,
    },
    SoftplusParametric {
        input: SirNodeId,
        alpha: String,
        beta: String,
    },
    Gelu {
        input: SirNodeId,
        mode: String,
    },
    Clip {
        input: SirNodeId,
        min_val: f32,
        max_val: f32,
    },
    Square {
        input: SirNodeId,
    },
    Threshold {
        input: SirNodeId,
        alpha: f32,
    },
    Sqrt {
        input: SirNodeId,
    },
    Rsqrt {
        input: SirNodeId,
    },
    Inverse {
        input: SirNodeId,
        epsilon: f32,
    },
    Ceil {
        input: SirNodeId,
    },
    Floor {
        input: SirNodeId,
    },
    Round {
        input: SirNodeId,
    },
    Exp {
        input: SirNodeId,
    },
    Exp2 {
        input: SirNodeId,
    },
    Log {
        input: SirNodeId,
        epsilon: f32,
    },
    Sign {
        input: SirNodeId,
    },
    Cos {
        input: SirNodeId,
    },
    Sin {
        input: SirNodeId,
    },
    Tan {
        input: SirNodeId,
    },
    Acos {
        input: SirNodeId,
    },
    Asin {
        input: SirNodeId,
    },
    Atan {
        input: SirNodeId,
    },
    Cosh {
        input: SirNodeId,
    },
    Sinh {
        input: SirNodeId,
    },
    Atanh {
        input: SirNodeId,
    },
    Erf {
        input: SirNodeId,
    },
    LogicalNot {
        input: SirNodeId,
    },
    Cast {
        input: SirNodeId,
        dtype: MilDtype,
    },
    Select {
        condition: SirNodeId,
        x: SirNodeId,
        y: SirNodeId,
    },
    Where {
        condition: SirNodeId,
        x: SirNodeId,
        y: SirNodeId,
    },
    Softmax {
        input: SirNodeId,
        axis: isize,
    },

    // ─── Reduction ───────────────────────────────────────────────
    ReduceSum {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMean {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMax {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceMin {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceProd {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceSumSquare {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceL2Norm {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceL1Norm {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceLogSumExp {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceLogSum {
        input: SirNodeId,
        axes: Vec<usize>,
        keep_dims: bool,
    },
    ReduceArgmax {
        input: SirNodeId,
        axis: usize,
        keep_dims: bool,
    },
    ReduceArgmin {
        input: SirNodeId,
        axis: usize,
        keep_dims: bool,
    },

    // ─── Normalization ───────────────────────────────────────────
    BatchNorm {
        input: SirNodeId,
        mean: String,
        variance: String,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    InstanceNorm {
        input: SirNodeId,
        gamma: Option<String>,
        beta: Option<String>,
        epsilon: f32,
    },
    LayerNorm {
        input: SirNodeId,
        weight: String,
        bias: Option<String>,
        epsilon: f32,
        axes: Vec<usize>,
    },
    L2Norm {
        input: SirNodeId,
        epsilon: f32,
        axes: Vec<usize>,
    },
    LocalResponseNorm {
        input: SirNodeId,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    },

    // ─── Pooling ─────────────────────────────────────────────────
    MaxPool {
        input: SirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },
    AvgPool {
        input: SirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
        count_include_padding: bool,
    },
    L2Pool {
        input: SirNodeId,
        kernel_sizes: Vec<usize>,
        strides: Vec<usize>,
        pad_types: Vec<String>,
        pad_amounts: Vec<usize>,
    },

    // ─── Image Resizing ──────────────────────────────────────────
    Resize {
        input: SirNodeId,
        target_size: Vec<usize>,
        mode: String,
        sampling_mode: String,
        nearest_rounding_mode: String,
    },
    ResizeNearestNeighbor {
        input: SirNodeId,
        target_height: usize,
        target_width: usize,
    },
    ResizeBilinear {
        input: SirNodeId,
        target_height: usize,
        target_width: usize,
        align_corners: bool,
    },
    UpsampleNearestNeighbor {
        input: SirNodeId,
        scale: Vec<usize>,
    },
    UpsampleBilinear {
        input: SirNodeId,
        scale: Vec<usize>,
        align_corners: bool,
        half_pixel_centers: bool,
    },
    CropResize {
        input: SirNodeId,
        boxes: SirNodeId,
        box_indices: SirNodeId,
        crop_height: usize,
        crop_width: usize,
    },
    Affine {
        input: SirNodeId,
        transform: SirNodeId,
        output_height: usize,
        output_width: usize,
        sampling_mode: String,
        pad_value: f32,
    },
    Resample {
        input: SirNodeId,
        coordinates: SirNodeId,
        sampling_mode: String,
        pad_value: f32,
    },

    // ─── Tensor Transform ────────────────────────────────────────
    Reshape {
        input: SirNodeId,
        target_shape: Vec<usize>,
    },
    ReshapeLike {
        input: SirNodeId,
        ref_tensor: SirNodeId,
    },
    Transpose {
        input: SirNodeId,
        perm: Vec<usize>,
    },
    Split {
        input: SirNodeId,
        axis: usize,
        num_splits: usize,
    },
    Concat {
        inputs: Vec<SirNodeId>,
        axis: usize,
    },
    ExpandDims {
        input: SirNodeId,
        axis: Vec<usize>,
    },
    Squeeze {
        input: SirNodeId,
        axis: Vec<usize>,
    },
    Flatten2d {
        input: SirNodeId,
        axis: usize,
    },
    Reverse {
        input: SirNodeId,
        axes: Vec<usize>,
    },
    ReverseSequence {
        input: SirNodeId,
        lengths: SirNodeId,
        batch_axis: usize,
        seq_axis: usize,
    },
    SliceByIndex {
        input: SirNodeId,
        begin: Vec<i64>,
        end: Vec<i64>,
        stride: Vec<i64>,
        begin_mask: Vec<bool>,
        end_mask: Vec<bool>,
        squeeze_mask: Vec<bool>,
    },
    SliceBySize {
        input: SirNodeId,
        begin: Vec<i64>,
        size: Vec<i64>,
    },
    SlidingWindows {
        input: SirNodeId,
        axis: usize,
        window_size: usize,
        stride: usize,
    },
    DepthToSpace {
        input: SirNodeId,
        block_size: usize,
    },
    SpaceToDepth {
        input: SirNodeId,
        block_size: usize,
    },
    PixelShuffle {
        input: SirNodeId,
        upscale_factor: usize,
    },
    PixelUnshuffle {
        input: SirNodeId,
        downscale_factor: usize,
    },
    BatchToSpace {
        input: SirNodeId,
        block_shape: Vec<usize>,
        crops: Vec<(usize, usize)>,
    },
    SpaceToBatch {
        input: SirNodeId,
        block_shape: Vec<usize>,
        paddings: Vec<(usize, usize)>,
    },
    Pad {
        input: SirNodeId,
        pad_amounts: Vec<i64>,
        mode: String,
        constant_value: f32,
    },
    Stack {
        values: Vec<SirNodeId>,
        axis: usize,
    },
    Tile {
        input: SirNodeId,
        reps: Vec<usize>,
    },
    Cumsum {
        input: SirNodeId,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    },
    Fill {
        shape: Vec<usize>,
        value: f32,
        dtype: MilDtype,
    },
    FillLike {
        ref_tensor: SirNodeId,
        value: f32,
        dtype: MilDtype,
    },
    Identity {
        input: SirNodeId,
    },
    OneHot {
        indices: SirNodeId,
        one_hot_vector_size: usize,
        on_value: f32,
        off_value: f32,
        axis: usize,
        dtype: MilDtype,
    },
    NonZero {
        input: SirNodeId,
    },
    Argsort {
        input: SirNodeId,
        axis: usize,
        ascending: bool,
    },
    BandPart {
        input: SirNodeId,
        num_lower: i64,
        num_upper: i64,
    },
    Range1d {
        start: f32,
        end: f32,
        step: f32,
    },
    Shape {
        input: SirNodeId,
    },
    Crop {
        input: SirNodeId,
        crop_height: usize,
        crop_width: usize,
        offset_height: usize,
        offset_width: usize,
    },

    // ─── Scatter / Gather ────────────────────────────────────────
    Gather {
        input: SirNodeId,
        indices: SirNodeId,
        axis: isize,
    },
    GatherAlongAxis {
        input: SirNodeId,
        indices: SirNodeId,
        axis: isize,
    },
    GatherNd {
        input: SirNodeId,
        indices: SirNodeId,
    },
    Scatter {
        input: SirNodeId,
        indices: SirNodeId,
        updates: SirNodeId,
        axis: isize,
        mode: String,
    },
    ScatterAlongAxis {
        input: SirNodeId,
        indices: SirNodeId,
        updates: SirNodeId,
        axis: isize,
    },
    ScatterNd {
        input: SirNodeId,
        indices: SirNodeId,
        updates: SirNodeId,
    },
    NonMaximumSuppression {
        boxes: SirNodeId,
        scores: SirNodeId,
        iou_threshold: f32,
        score_threshold: f32,
        max_detections: usize,
    },

    // ─── Attention ───────────────────────────────────────────────
    ScaledDotProductAttention {
        query: SirNodeId,
        key: SirNodeId,
        value: SirNodeId,
        attention_mask: Option<SirNodeId>,
        scale: Option<f32>,
    },

    // ─── Quantization / Palettization ───────────────────────────
    Quantize {
        input: SirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: MilDtype,
    },
    Dequantize {
        input: SirNodeId,
        scale: f32,
        zero_point: i32,
        axis: isize,
        output_dtype: MilDtype,
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
        dtype: MilDtype,
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
        input: SirNodeId,
        initial_h: SirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        mode: String,
        output_sequence: bool,
    },
    Gru {
        input: SirNodeId,
        initial_h: SirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        reset_after: bool,
        output_sequence: bool,
    },
    Lstm {
        input: SirNodeId,
        initial_h: SirNodeId,
        initial_c: SirNodeId,
        weight_ih: String,
        weight_hh: String,
        bias: Option<String>,
        output_sequence: bool,
    },

    // ─── Control Flow ────────────────────────────────────────────
    Cond {
        pred: SirNodeId,
        true_graph: String,
        false_graph: String,
    },
    WhileLoop {
        condition: String,
        body: String,
        loop_vars: Vec<SirNodeId>,
    },
    MakeList {
        elems: Vec<SirNodeId>,
        dtype: MilDtype,
    },
    ListLength {
        ls: SirNodeId,
    },
    ListWrite {
        ls: SirNodeId,
        index: SirNodeId,
        value: SirNodeId,
    },
    ListRead {
        ls: SirNodeId,
        index: SirNodeId,
    },
    ListGather {
        ls: SirNodeId,
        indices: SirNodeId,
    },
    ListScatter {
        ls: SirNodeId,
        indices: SirNodeId,
        values: SirNodeId,
    },

    // ─── Random ──────────────────────────────────────────────────
    RandomBernoulli {
        shape: Vec<usize>,
        prob: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    RandomNormal {
        shape: Vec<usize>,
        mean: f32,
        stddev: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    RandomUniform {
        shape: Vec<usize>,
        low: f32,
        high: f32,
        seed: Option<u64>,
        dtype: MilDtype,
    },
    RandomCategorical {
        logits: SirNodeId,
        num_samples: usize,
        seed: Option<u64>,
        dtype: MilDtype,
    },

    // ─── Topk / Classify ─────────────────────────────────────────
    Topk {
        input: SirNodeId,
        k: usize,
        axis: isize,
    },
    Classify {
        input: SirNodeId,
    },

    // ─── KV Cache ──────────────────────────────────────────────
    // KV cache operations are represented by StateRead/StateWrite.
    // Optimization strategies (masked blend, ring buffer, paged) are
    // applied by passes that transform StateRead/StateWrite sequences
    // into appropriate primitive op patterns. No specialized KV cache
    // op variants are needed — the strategy framework discovers and
    // applies the right pattern dynamically.

    // ─── Legacy compat: ElementWise used by linear_slice.rs ──────
    ElementWise {
        op: ElementWiseOp,
        inputs: Vec<SirNodeId>,
    },
}

// Sprint 58 (S58.2): MilDtypeRepr was removed. SIR now uses
// `super::mir::MilDtype` directly, eliminating the duplicate type.

// Legacy compatibility: ElementWiseOp still used by some existing code paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementWiseOp {
    Add,
    Mul,
    Abs,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirNode {
    pub id: SirNodeId,
    pub op: SirOp,
    pub name: String,
    pub metadata: SirMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirMetadata {
    pub task_origin: TaskOrigin,
    pub model_id: Option<String>,
    pub quality_contract: Option<QualityContract>,
    pub precision_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOrigin {
    Synthetic,
    RealModel {
        name: String,
    },
    MilImport {
        source: String,
    },
    Manual,
    /// Traced from a HuggingFace transformers model via torch.fx.
    TransformersTrace {
        name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityContract {
    pub max_perplexity_delta: Option<f32>,
    pub max_latency_ms: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirGraph {
    pub nodes: Vec<SirNode>,
    pub inputs: Vec<SirNodeId>,
    pub outputs: Vec<SirNodeId>,
}

/// KV cache layout strategy — determines how KV cache updates are structured.
///
/// The layout choice directly affects ANE provisioning behavior.
/// Strategies are discovered dynamically by the optimization framework
/// based on graph structure and target hardware, not hardcoded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum KvCacheLayout {
    /// Naive append/shift scheme — new tokens appended at the end,
    /// old tokens shifted out when context is full. Requires scatter
    /// operations that often force CPU fallback on ANE.
    #[default]
    Naive,
    /// Masked-blend write pattern: active context in contiguous suffix,
    /// new K/V values written by masked blending instead of scatter.
    /// The strategy framework discovers this pattern when the target
    /// hardware supports the required primitive ops (Where, Mul, Add).
    MaskedBlend,
    /// Paged KV cache with fixed-size blocks. Not yet implemented;
    /// reserved for future paged-attention support.
    Paged,
}

/// Quantization strategy for a weight tensor or layer.
///
/// Parameterized by method and bit-width rather than named after
/// specific projects. The strategy framework discovers which
/// combination of parameters is appropriate for each weight tensor
/// based on sensitivity analysis and target hardware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum QuantizationStrategy {
    /// No quantization — full fp16/fp32 weights.
    #[default]
    Unquantized,
    /// Blockwise weight-only quantization with per-group scales and offsets.
    /// Good for embedding/LM head matrices.
    Blockwise {
        /// Block group size (e.g., 128).
        group_size: usize,
        /// Bits per weight element (4, 6, or 8).
        bits: usize,
    },
    /// Grouped look-up table (LUT) quantization.
    /// Each group of `group_size` weights shares a palette of 2^bits entries.
    /// Good for attention and MLP projection matrices.
    GroupedLut {
        /// Block group size (typically 128).
        group_size: usize,
        /// Bits per index (4, 6, or 8).
        bits: usize,
        /// Number of LUT groups.
        num_groups: usize,
    },
    /// Post-hoc palettization applied to constants after Core ML emission.
    /// Good for KV/mask tables and other less sensitive tensors.
    Palettized {
        /// Palettization mode (e.g., "kmeans").
        mode: String,
        /// Bits per palette index (1, 2, 4, or 8).
        nbits: usize,
        /// Group size for grouped channel palettization.
        group_size: usize,
    },
}

/// Specification for an on-device sampler model.
///
/// Sampling as a first-class model in the deployment package keeps
/// the decode loop fully on-device. Parameters are discovered by
/// the strategy framework based on model characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerSpec {
    /// K value for top-k pre-candidate selection.
    pub pre_candidate_k: usize,
    /// K value for final top-k selection after min-p pruning.
    pub final_top_k: usize,
    /// Number of noise samples for Gumbel noise table.
    pub num_noise_samples: usize,
    /// Default temperature.
    pub default_temperature: f32,
    /// Default min-p threshold.
    pub default_min_p: f32,
    /// Default repetition penalty.
    pub default_rep_penalty: f32,
    /// Token history size for repetition tracking.
    pub history_size: usize,
}

impl Default for SamplerSpec {
    fn default() -> Self {
        SamplerSpec {
            pre_candidate_k: 64,
            final_top_k: 16,
            num_noise_samples: 8192,
            default_temperature: 1.0,
            default_min_p: 0.05,
            default_rep_penalty: 2.8,
            history_size: 1024,
        }
    }
}

/// Specification for a conditional IO model (shared embedding + LM head).
///
/// For models with tied embedding/LM-head weights, a conditional IO
/// model can share the weight matrix with a mode switch, halving the
/// memory footprint. The strategy framework discovers when tied weights
/// are present and recommends this pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoModelSpec {
    /// Whether embedding and LM head weights are shared (tied).
    pub tied_weights: bool,
    /// Quantization strategy for the embedding/LM head weights.
    pub quantization: QuantizationStrategy,
    /// Embedding mode value (typically 0).
    pub embedding_mode_value: i32,
    /// Logit mode value (typically 1).
    pub logit_mode_value: i32,
}

impl Default for IoModelSpec {
    fn default() -> Self {
        IoModelSpec {
            tied_weights: true,
            quantization: QuantizationStrategy::Blockwise { group_size: 128, bits: 4 },
            embedding_mode_value: 0,
            logit_mode_value: 1,
        }
    }
}
