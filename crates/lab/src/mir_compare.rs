//! MIR-vs-Emitted-Structure Comparison (Sprint 34)
//!
//! Compares the MIR ops that the compiler intended to emit against the
//! actual ops found in the emitted model structure (via MLModelStructure
//! or fallback inspection). Produces op fidelity metrics and diff reports.
//!
//! This module is the Rust-side complement to the Python-side
//! `model_structure.compare_mir_vs_structure()`. The Python version runs
//! inside the bridge and compares against MLModelStructure output directly.
//! This Rust version can compare a MIR graph against a structure result
//! returned from the bridge.
//!
//! ## MIR-to-MIL Name Mapping
//!
//! MIR uses Rust-style enum variant names (e.g., `MILLinear`, `MILGelu`),
//! while Core ML's MIL uses lowercase snake_case names (e.g., `linear`,
//! `gelu`). This module maintains the canonical mapping so that comparisons
//! are by semantic op identity, not by string coincidence.

use ane_ir::mir::{MirGraph, MirOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of comparing a MIR graph against an emitted model structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirComparisonResult {
    /// Fraction of MIR ops that matched structure ops (0.0 - 1.0).
    pub op_fidelity_score: f64,
    /// Number of MIR ops that were matched in the structure.
    pub matched_count: usize,
    /// Number of MIR ops that were NOT found in the structure.
    pub missing_count: usize,
    /// Number of structure ops that were NOT expected by the MIR.
    pub extra_count: usize,
    /// Total MIR ops compared.
    pub mir_op_count: usize,
    /// Total structure ops found.
    pub structure_op_count: usize,
    /// Detailed match information for each MIR op.
    pub matches: Vec<MirOpMatch>,
    /// MIR ops missing from the emitted structure.
    pub missing_ops: Vec<String>,
    /// Structure ops not expected by the MIR (by MIL op name).
    pub extra_ops: Vec<String>,
}

/// Match status for a single MIR op against the structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirOpMatch {
    /// MIR op type name (e.g., "MILLinear").
    pub mir_op_type: String,
    /// Expected MIL op name (e.g., "linear").
    pub expected_mil_name: String,
    /// Whether this MIR op was found in the structure.
    pub matched: bool,
    /// MIR node name (if available).
    pub mir_node_name: Option<String>,
}

/// Canonical mapping from MIR op variant names to Core ML MIL op names.
///
/// This is the single source of truth for the MIR→MIL name correspondence.
/// When a new MIR op is added, it MUST be registered here.
pub fn mir_to_mil_name(mir_op_type: &str) -> Option<&'static str> {
    match mir_op_type {
        "MILConst" => Some("const"),
        "MILMatMul" => Some("matmul"),
        "MILLinear" => Some("linear"),
        "MILConv" => Some("conv"),
        "MILAdd" => Some("add"),
        "MILMul" => Some("mul"),
        "MILSub" => Some("sub"),
        "MILAbs" => Some("abs"),
        "MILMaximum" => Some("maximum"),
        "MILMinimum" => Some("minimum"),
        "MILReshape" => Some("reshape"),
        "MILTranspose" => Some("transpose"),
        "MILSplit" => Some("split"),
        "MILConcat" => Some("concat"),
        "MILSoftmax" => Some("softmax"),
        "MILScaledDotProductAttention" => Some("scaled_dot_product_attention"),
        "MILSliceByIndex" => Some("slice_by_index"),
        "MILGelu" => Some("gelu"),
        "MILReadState" => Some("read_state"),
        "MILCoremlUpdateState" => Some("coreml_update_state"),
        "MILStateWrite" => Some("state_write"),
        "MILReduceSum" => Some("reduce_sum"),
        "MILReduceMean" => Some("reduce_mean"),
        "MILRsqrt" => Some("rsqrt"),
        "MILRealDiv" => Some("real_div"),
        "MILLayerNorm" => Some("layer_norm"),
        "MILTopk" => Some("topk"),
        "MILGather" => Some("gather"),
        "MILCos" => Some("cos"),
        "MILSin" => Some("sin"),
        "MILCast" => Some("cast"),
        // Sprint 50: P2 MIR ops
        "MILSliceUpdate" => Some("slice_update"),
        "MILExp" => Some("exp"),
        "MILSigmoid" => Some("sigmoid"),
        "MILTanh" => Some("tanh"),
        "MILRelu" => Some("relu"),
        "MILWhere" => Some("where"),
        // P3+ MIR ops
        "MILEinsum" => Some("einsum"),
        "MILConvTranspose" => Some("conv_transpose"),
        "MILFloorDiv" => Some("floor_div"),
        "MILMod" => Some("mod"),
        "MILPow" => Some("pow"),
        "MILEqual" => Some("equal"),
        "MILNotEqual" => Some("not_equal"),
        "MILGreater" => Some("greater"),
        "MILGreaterEqual" => Some("greater_equal"),
        "MILLess" => Some("less"),
        "MILLessEqual" => Some("less_equal"),
        "MILLogicalAnd" => Some("logical_and"),
        "MILLogicalOr" => Some("logical_or"),
        "MILLogicalXor" => Some("logical_xor"),
        "MILNeg" => Some("neg"),
        "MILRelu6" => Some("relu6"),
        "MILLeakyRelu" => Some("leaky_relu"),
        "MILSigmoidHard" => Some("sigmoid_hard"),
        "MILThresholdedRelu" => Some("thresholded_relu"),
        "MILClampedRelu" => Some("clamped_relu"),
        "MILLinearActivation" => Some("linear_activation"),
        "MILPrelu" => Some("prelu"),
        "MILSoftsign" => Some("softsign"),
        "MILSilu" => Some("silu"),
        "MILScaledTanh" => Some("scaled_tanh"),
        "MILElu" => Some("elu"),
        "MILSoftplus" => Some("softplus"),
        "MILSoftplusParametric" => Some("softplus_parametric"),
        "MILClip" => Some("clip"),
        "MILSquare" => Some("square"),
        "MILThreshold" => Some("threshold"),
        "MILSqrt" => Some("sqrt"),
        "MILInverse" => Some("inverse"),
        "MILCeil" => Some("ceil"),
        "MILFloor" => Some("floor"),
        "MILRound" => Some("round"),
        "MILExp2" => Some("exp2"),
        "MILLog" => Some("log"),
        "MILSign" => Some("sign"),
        "MILTan" => Some("tan"),
        "MILAcos" => Some("acos"),
        "MILAsin" => Some("asin"),
        "MILAtan" => Some("atan"),
        "MILCosh" => Some("cosh"),
        "MILSinh" => Some("sinh"),
        "MILAtanh" => Some("atanh"),
        "MILErf" => Some("erf"),
        "MILLogicalNot" => Some("logical_not"),
        "MILSelect" => Some("select"),
        "MILReduceMax" => Some("reduce_max"),
        "MILReduceMin" => Some("reduce_min"),
        "MILReduceProd" => Some("reduce_prod"),
        "MILReduceSumSquare" => Some("reduce_sum_square"),
        "MILReduceL2Norm" => Some("reduce_l2_norm"),
        "MILReduceL1Norm" => Some("reduce_l1_norm"),
        "MILReduceLogSumExp" => Some("reduce_log_sum_exp"),
        "MILReduceLogSum" => Some("reduce_log_sum"),
        "MILReduceArgmax" => Some("reduce_argmax"),
        "MILReduceArgmin" => Some("reduce_argmin"),
        "MILBatchNorm" => Some("batch_norm"),
        "MILInstanceNorm" => Some("instance_norm"),
        "MILL2Norm" => Some("l2_norm"),
        "MILLocalResponseNorm" => Some("local_response_norm"),
        "MILMaxPool" => Some("max_pool"),
        "MILAvgPool" => Some("avg_pool"),
        "MILL2Pool" => Some("l2_pool"),
        "MILResize" => Some("resize"),
        "MILResizeNearestNeighbor" => Some("resize_nearest_neighbor"),
        "MILResizeBilinear" => Some("resize_bilinear"),
        "MILUpsampleNearestNeighbor" => Some("upsample_nearest_neighbor"),
        "MILUpsampleBilinear" => Some("upsample_bilinear"),
        "MILCropResize" => Some("crop_resize"),
        "MILAffine" => Some("affine"),
        "MILResample" => Some("resample"),
        "MILReshapeLike" => Some("reshape_like"),
        "MILExpandDims" => Some("expand_dims"),
        "MILSqueeze" => Some("squeeze"),
        "MILFlatten2d" => Some("flatten2d"),
        "MILReverse" => Some("reverse"),
        "MILReverseSequence" => Some("reverse_sequence"),
        "MILSliceBySize" => Some("slice_by_size"),
        "MILSlidingWindows" => Some("sliding_windows"),
        "MILDepthToSpace" => Some("depth_to_space"),
        "MILSpaceToDepth" => Some("space_to_depth"),
        "MILPixelShuffle" => Some("pixel_shuffle"),
        "MILPixelUnshuffle" => Some("pixel_unshuffle"),
        "MILBatchToSpace" => Some("batch_to_space"),
        "MILSpaceToBatch" => Some("space_to_batch"),
        "MILPad" => Some("pad"),
        "MILStack" => Some("stack"),
        "MILTile" => Some("tile"),
        "MILCumsum" => Some("cumsum"),
        "MILFill" => Some("fill"),
        "MILFillLike" => Some("fill_like"),
        "MILIdentity" => Some("identity"),
        "MILOneHot" => Some("one_hot"),
        "MILNonZero" => Some("non_zero"),
        "MILArgsort" => Some("argsort"),
        "MILBandPart" => Some("band_part"),
        "MILRange1d" => Some("range_1d"),
        "MILShape" => Some("shape"),
        "MILCrop" => Some("crop"),
        "MILGatherAlongAxis" => Some("gather_along_axis"),
        "MILGatherNd" => Some("gather_nd"),
        "MILScatter" => Some("scatter"),
        "MILScatterAlongAxis" => Some("scatter_along_axis"),
        "MILScatterNd" => Some("scatter_nd"),
        "MILNonMaximumSuppression" => Some("non_maximum_suppression"),
        "MILQuantize" => Some("quantize"),
        "MILDequantize" => Some("dequantize"),
        "MILConstexprAffineDequantize" => Some("constexpr_affine_dequantize"),
        "MILConstexprBlockwiseShiftScale" => Some("constexpr_blockwise_shift_scale"),
        "MILConstexprLutToDense" => Some("constexpr_lut_to_dense"),
        "MILConstexprSparseToDense" => Some("constexpr_sparse_to_dense"),
        "MILConstexprCast" => Some("constexpr_cast"),
        "MILConstexprLutToSparse" => Some("constexpr_lut_to_sparse"),
        "MILConstexprSparseBlockwiseShiftScale" => Some("constexpr_sparse_blockwise_shift_scale"),
        "MILRnn" => Some("rnn"),
        "MILGru" => Some("gru"),
        "MILLstm" => Some("lstm"),
        "MILCond" => Some("cond"),
        "MILWhileLoop" => Some("while_loop"),
        "MILMakeList" => Some("make_list"),
        "MILListLength" => Some("list_length"),
        "MILListWrite" => Some("list_write"),
        "MILListRead" => Some("list_read"),
        "MILListGather" => Some("list_gather"),
        "MILListScatter" => Some("list_scatter"),
        "MILRandomBernoulli" => Some("random_bernoulli"),
        "MILRandomNormal" => Some("random_normal"),
        "MILRandomUniform" => Some("random_uniform"),
        "MILRandomCategorical" => Some("random_categorical"),
        "MILClassify" => Some("classify"),
        _ => None,
    }
}

/// Get the MIR op type name for a MirOp variant.
pub fn mir_op_type_name(op: &MirOp) -> &'static str {
    match op {
        MirOp::MILConst { .. } => "MILConst",
        MirOp::MILMatMul { .. } => "MILMatMul",
        MirOp::MILLinear { .. } => "MILLinear",
        MirOp::MILConv { .. } => "MILConv",
        MirOp::MILAdd { .. } => "MILAdd",
        MirOp::MILMul { .. } => "MILMul",
        MirOp::MILSub { .. } => "MILSub",
        MirOp::MILAbs { .. } => "MILAbs",
        MirOp::MILMaximum { .. } => "MILMaximum",
        MirOp::MILMinimum { .. } => "MILMinimum",
        MirOp::MILReshape { .. } => "MILReshape",
        MirOp::MILTranspose { .. } => "MILTranspose",
        MirOp::MILSplit { .. } => "MILSplit",
        MirOp::MILConcat { .. } => "MILConcat",
        MirOp::MILSoftmax { .. } => "MILSoftmax",
        MirOp::MILScaledDotProductAttention { .. } => "MILScaledDotProductAttention",
        MirOp::MILSliceByIndex { .. } => "MILSliceByIndex",
        MirOp::MILGelu { .. } => "MILGelu",
        MirOp::MILReadState { .. } => "MILReadState",
        MirOp::MILCoremlUpdateState { .. } => "MILCoremlUpdateState",
        MirOp::MILStateWrite { .. } => "MILStateWrite",
        MirOp::MILReduceSum { .. } => "MILReduceSum",
        MirOp::MILReduceMean { .. } => "MILReduceMean",
        MirOp::MILRsqrt { .. } => "MILRsqrt",
        MirOp::MILRealDiv { .. } => "MILRealDiv",
        MirOp::MILLayerNorm { .. } => "MILLayerNorm",
        MirOp::MILTopk { .. } => "MILTopk",
        MirOp::MILGather { .. } => "MILGather",
        MirOp::MILCos { .. } => "MILCos",
        MirOp::MILSin { .. } => "MILSin",
        MirOp::MILCast { .. } => "MILCast",
        // Sprint 50: P2 MIR ops
        MirOp::MILSliceUpdate { .. } => "MILSliceUpdate",
        MirOp::MILExp { .. } => "MILExp",
        MirOp::MILSigmoid { .. } => "MILSigmoid",
        MirOp::MILTanh { .. } => "MILTanh",
        MirOp::MILRelu { .. } => "MILRelu",
        MirOp::MILWhere { .. } => "MILWhere",
        // P3+ MIR ops
        MirOp::MILEinsum { .. } => "MILEinsum",
        MirOp::MILConvTranspose { .. } => "MILConvTranspose",
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
        MirOp::MILNeg { .. } => "MILNeg",
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
        MirOp::MILClip { .. } => "MILClip",
        MirOp::MILSquare { .. } => "MILSquare",
        MirOp::MILThreshold { .. } => "MILThreshold",
        MirOp::MILSqrt { .. } => "MILSqrt",
        MirOp::MILInverse { .. } => "MILInverse",
        MirOp::MILCeil { .. } => "MILCeil",
        MirOp::MILFloor { .. } => "MILFloor",
        MirOp::MILRound { .. } => "MILRound",
        MirOp::MILExp2 { .. } => "MILExp2",
        MirOp::MILLog { .. } => "MILLog",
        MirOp::MILSign { .. } => "MILSign",
        MirOp::MILTan { .. } => "MILTan",
        MirOp::MILAcos { .. } => "MILAcos",
        MirOp::MILAsin { .. } => "MILAsin",
        MirOp::MILAtan { .. } => "MILAtan",
        MirOp::MILCosh { .. } => "MILCosh",
        MirOp::MILSinh { .. } => "MILSinh",
        MirOp::MILAtanh { .. } => "MILAtanh",
        MirOp::MILErf { .. } => "MILErf",
        MirOp::MILLogicalNot { .. } => "MILLogicalNot",
        MirOp::MILSelect { .. } => "MILSelect",
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
        MirOp::MILReshapeLike { .. } => "MILReshapeLike",
        MirOp::MILExpandDims { .. } => "MILExpandDims",
        MirOp::MILSqueeze { .. } => "MILSqueeze",
        MirOp::MILFlatten2d { .. } => "MILFlatten2d",
        MirOp::MILReverse { .. } => "MILReverse",
        MirOp::MILReverseSequence { .. } => "MILReverseSequence",
        MirOp::MILSliceBySize { .. } => "MILSliceBySize",
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
        MirOp::MILGatherAlongAxis { .. } => "MILGatherAlongAxis",
        MirOp::MILGatherNd { .. } => "MILGatherNd",
        MirOp::MILScatter { .. } => "MILScatter",
        MirOp::MILScatterAlongAxis { .. } => "MILScatterAlongAxis",
        MirOp::MILScatterNd { .. } => "MILScatterNd",
        MirOp::MILNonMaximumSuppression { .. } => "MILNonMaximumSuppression",
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
        MirOp::MILClassify { .. } => "MILClassify",
    }
}

/// Extract the op names from a MIR graph as MIL op names.
///
/// Returns a vector of (mir_op_type, mil_name, node_name) triples.
/// Ops without a known MIL mapping are included with mil_name = None.
pub fn extract_mir_mil_names(mir: &MirGraph) -> Vec<(String, Option<String>, Option<String>)> {
    mir.nodes
        .iter()
        .map(|node| {
            let mir_type = mir_op_type_name(&node.op).to_string();
            let mil_name = mir_to_mil_name(&mir_type).map(|s| s.to_string());
            let node_name = match &node.op {
                MirOp::MILLinear { name, .. } => Some(name.clone()),
                MirOp::MILGelu { name, .. } => Some(name.clone()),
                MirOp::MILSoftmax { name, .. } => Some(name.clone()),
                MirOp::MILReshape { name, .. } => Some(name.clone()),
                MirOp::MILTranspose { name, .. } => Some(name.clone()),
                MirOp::MILMatMul { name, .. } => Some(name.clone()),
                MirOp::MILConst { name, .. } => Some(name.clone()),
                MirOp::MILAdd { name, .. } => Some(name.clone()),
                MirOp::MILMul { name, .. } => Some(name.clone()),
                MirOp::MILScaledDotProductAttention { name, .. } => Some(name.clone()),
                MirOp::MILSliceByIndex { name, .. } => Some(name.clone()),
                MirOp::MILConcat { name, .. } => Some(name.clone()),
                MirOp::MILSplit { name, .. } => Some(name.clone()),
                MirOp::MILReduceSum { name, .. } => Some(name.clone()),
                MirOp::MILReduceMean { name, .. } => Some(name.clone()),
                MirOp::MILRsqrt { name, .. } => Some(name.clone()),
                MirOp::MILRealDiv { name, .. } => Some(name.clone()),
                MirOp::MILLayerNorm { name, .. } => Some(name.clone()),
                MirOp::MILTopk { name, .. } => Some(name.clone()),
                MirOp::MILGather { name, .. } => Some(name.clone()),
                MirOp::MILCos { name, .. } => Some(name.clone()),
                MirOp::MILSin { name, .. } => Some(name.clone()),
                MirOp::MILCast { name, .. } => Some(name.clone()),
                MirOp::MILReadState { name, .. } => Some(name.clone()),
                MirOp::MILCoremlUpdateState { name, .. } => Some(name.clone()),
                MirOp::MILStateWrite { name, .. } => Some(name.clone()),
                // Sprint 50: P2 MIR ops
                MirOp::MILSliceUpdate { name, .. } => Some(name.clone()),
                MirOp::MILExp { name, .. } => Some(name.clone()),
                MirOp::MILSigmoid { name, .. } => Some(name.clone()),
                MirOp::MILTanh { name, .. } => Some(name.clone()),
                MirOp::MILRelu { name, .. } => Some(name.clone()),
                MirOp::MILWhere { name, .. } => Some(name.clone()),
                _ => None,
            };
            (mir_type, mil_name, node_name)
        })
        .collect()
}

/// Compare a MIR graph against a list of structure op names.
///
/// This performs a multiset comparison: for each MIR op, it checks whether
/// the corresponding MIL op name appears in the structure. The comparison
/// accounts for op multiplicity (e.g., two MILLinear ops in the MIR should
/// match two "linear" ops in the structure).
///
/// Args:
/// - `mir`: The MIR graph representing the compiler's intent.
/// - `structure_op_names`: List of MIL op names found in the emitted model.
///
/// Returns a `MirComparisonResult` with fidelity metrics and diffs.
pub fn compare_mir_vs_structure(
    mir: &MirGraph,
    structure_op_names: &[String],
) -> MirComparisonResult {
    let mir_entries = extract_mir_mil_names(mir);

    // Build multisets of expected vs actual MIL op names
    let mut expected_counts: HashMap<String, usize> = HashMap::new();
    let mut actual_counts: HashMap<String, usize> = HashMap::new();

    for (_mir_type, mil_name, _node_name) in &mir_entries {
        if let Some(name) = mil_name {
            *expected_counts.entry(name.clone()).or_insert(0) += 1;
        }
        // MIR ops without a known MIL mapping are excluded from the comparison
        // but reported in the matches list
    }

    for name in structure_op_names {
        *actual_counts.entry(name.clone()).or_insert(0) += 1;
    }

    // Compute matches
    let mut matches = Vec::new();
    let mut matched_count = 0usize;
    let mut missing_ops = Vec::new();

    for (mir_type, mil_name, node_name) in &mir_entries {
        let expected_mil = mil_name.as_deref().unwrap_or("unknown");
        let was_matched = if let Some(name) = mil_name {
            let _expected_n = expected_counts.get(name).copied().unwrap_or(0);
            let actual_n = actual_counts.get(name).copied().unwrap_or(0);
            // This op is matched if the expected count <= actual count
            // (simplified: we count the total matched fraction)
            actual_n > 0
        } else {
            false
        };

        matches.push(MirOpMatch {
            mir_op_type: mir_type.clone(),
            expected_mil_name: expected_mil.to_string(),
            matched: was_matched,
            mir_node_name: node_name.clone(),
        });

        if was_matched {
            matched_count += 1;
        }
    }

    // Compute missing from structure (expected but not fully present)
    for (name, &expected_n) in &expected_counts {
        let actual_n = actual_counts.get(name).copied().unwrap_or(0);
        if actual_n < expected_n {
            missing_ops.push(format!("{} (expected {}, found {})", name, expected_n, actual_n));
        }
    }

    // Compute extra in structure (present but not expected)
    let mut extra_ops = Vec::new();
    for (name, &actual_n) in &actual_counts {
        let expected_n = expected_counts.get(name).copied().unwrap_or(0);
        if expected_n == 0 {
            extra_ops.push(format!("{} (unexpected, found {})", name, actual_n));
        } else if actual_n > expected_n {
            extra_ops.push(format!(
                "{} (expected {}, found {} — surplus {})",
                name,
                expected_n,
                actual_n,
                actual_n - expected_n
            ));
        }
    }

    let mir_op_count = mir_entries.len();
    let structure_op_count = structure_op_names.len();
    let missing_count = mir_op_count - matched_count;

    let op_fidelity_score = if mir_op_count > 0 {
        matched_count as f64 / mir_op_count as f64
    } else {
        1.0 // No MIR ops means perfect fidelity vacuously
    };

    MirComparisonResult {
        op_fidelity_score,
        matched_count,
        missing_count,
        extra_count: extra_ops.len(),
        mir_op_count,
        structure_op_count,
        matches,
        missing_ops,
        extra_ops,
    }
}

/// Build the MIR ops list suitable for passing to the Python bridge's
/// `model_structure` command for comparison.
///
/// Returns a Vec of dicts, each with "op_type" and optionally "name" keys,
/// formatted as JSON-serializable values.
pub fn mir_ops_for_bridge(mir: &MirGraph) -> Vec<serde_json::Value> {
    mir.nodes
        .iter()
        .map(|node| {
            let mir_type = mir_op_type_name(&node.op).to_string();
            let mut obj = serde_json::Map::new();
            obj.insert("op_type".to_string(), serde_json::Value::String(mir_type));
            if let Some(name) = mir_op_name(&node.op) {
                obj.insert("name".to_string(), serde_json::Value::String(name));
            }
            serde_json::Value::Object(obj)
        })
        .collect()
}

/// Extract the name from a MirOp variant.
fn mir_op_name(op: &MirOp) -> Option<String> {
    match op {
        MirOp::MILConst { name, .. } => Some(name.clone()),
        MirOp::MILMatMul { name, .. } => Some(name.clone()),
        MirOp::MILLinear { name, .. } => Some(name.clone()),
        MirOp::MILConv { name, .. } => Some(name.clone()),
        MirOp::MILAdd { name, .. } => Some(name.clone()),
        MirOp::MILMul { name, .. } => Some(name.clone()),
        MirOp::MILSub { name, .. } => Some(name.clone()),
        MirOp::MILAbs { name, .. } => Some(name.clone()),
        MirOp::MILMaximum { name, .. } => Some(name.clone()),
        MirOp::MILMinimum { name, .. } => Some(name.clone()),
        MirOp::MILReshape { name, .. } => Some(name.clone()),
        MirOp::MILTranspose { name, .. } => Some(name.clone()),
        MirOp::MILSplit { name, .. } => Some(name.clone()),
        MirOp::MILConcat { name, .. } => Some(name.clone()),
        MirOp::MILSoftmax { name, .. } => Some(name.clone()),
        MirOp::MILScaledDotProductAttention { name, .. } => Some(name.clone()),
        MirOp::MILSliceByIndex { name, .. } => Some(name.clone()),
        MirOp::MILGelu { name, .. } => Some(name.clone()),
        MirOp::MILReadState { name, .. } => Some(name.clone()),
        MirOp::MILCoremlUpdateState { name, .. } => Some(name.clone()),
        MirOp::MILStateWrite { name, .. } => Some(name.clone()),
        MirOp::MILReduceSum { name, .. } => Some(name.clone()),
        MirOp::MILReduceMean { name, .. } => Some(name.clone()),
        MirOp::MILRsqrt { name, .. } => Some(name.clone()),
        MirOp::MILRealDiv { name, .. } => Some(name.clone()),
        MirOp::MILLayerNorm { name, .. } => Some(name.clone()),
        MirOp::MILTopk { name, .. } => Some(name.clone()),
        MirOp::MILGather { name, .. } => Some(name.clone()),
        MirOp::MILCos { name, .. } => Some(name.clone()),
        MirOp::MILSin { name, .. } => Some(name.clone()),
        MirOp::MILCast { name, .. } => Some(name.clone()),
        MirOp::MILSliceUpdate { name, .. } => Some(name.clone()),
        MirOp::MILExp { name, .. } => Some(name.clone()),
        MirOp::MILSigmoid { name, .. } => Some(name.clone()),
        MirOp::MILTanh { name, .. } => Some(name.clone()),
        MirOp::MILRelu { name, .. } => Some(name.clone()),
        MirOp::MILWhere { name, .. } => Some(name.clone()),
        // P3+ MIR ops
        MirOp::MILEinsum { name, .. } => Some(name.clone()),
        MirOp::MILConvTranspose { name, .. } => Some(name.clone()),
        MirOp::MILFloorDiv { name, .. } => Some(name.clone()),
        MirOp::MILMod { name, .. } => Some(name.clone()),
        MirOp::MILPow { name, .. } => Some(name.clone()),
        MirOp::MILEqual { name, .. } => Some(name.clone()),
        MirOp::MILNotEqual { name, .. } => Some(name.clone()),
        MirOp::MILGreater { name, .. } => Some(name.clone()),
        MirOp::MILGreaterEqual { name, .. } => Some(name.clone()),
        MirOp::MILLess { name, .. } => Some(name.clone()),
        MirOp::MILLessEqual { name, .. } => Some(name.clone()),
        MirOp::MILLogicalAnd { name, .. } => Some(name.clone()),
        MirOp::MILLogicalOr { name, .. } => Some(name.clone()),
        MirOp::MILLogicalXor { name, .. } => Some(name.clone()),
        MirOp::MILNeg { name, .. } => Some(name.clone()),
        MirOp::MILRelu6 { name, .. } => Some(name.clone()),
        MirOp::MILLeakyRelu { name, .. } => Some(name.clone()),
        MirOp::MILSigmoidHard { name, .. } => Some(name.clone()),
        MirOp::MILThresholdedRelu { name, .. } => Some(name.clone()),
        MirOp::MILClampedRelu { name, .. } => Some(name.clone()),
        MirOp::MILLinearActivation { name, .. } => Some(name.clone()),
        MirOp::MILPrelu { name, .. } => Some(name.clone()),
        MirOp::MILSoftsign { name, .. } => Some(name.clone()),
        MirOp::MILSilu { name, .. } => Some(name.clone()),
        MirOp::MILScaledTanh { name, .. } => Some(name.clone()),
        MirOp::MILElu { name, .. } => Some(name.clone()),
        MirOp::MILSoftplus { name, .. } => Some(name.clone()),
        MirOp::MILSoftplusParametric { name, .. } => Some(name.clone()),
        MirOp::MILClip { name, .. } => Some(name.clone()),
        MirOp::MILSquare { name, .. } => Some(name.clone()),
        MirOp::MILThreshold { name, .. } => Some(name.clone()),
        MirOp::MILSqrt { name, .. } => Some(name.clone()),
        MirOp::MILInverse { name, .. } => Some(name.clone()),
        MirOp::MILCeil { name, .. } => Some(name.clone()),
        MirOp::MILFloor { name, .. } => Some(name.clone()),
        MirOp::MILRound { name, .. } => Some(name.clone()),
        MirOp::MILExp2 { name, .. } => Some(name.clone()),
        MirOp::MILLog { name, .. } => Some(name.clone()),
        MirOp::MILSign { name, .. } => Some(name.clone()),
        MirOp::MILTan { name, .. } => Some(name.clone()),
        MirOp::MILAcos { name, .. } => Some(name.clone()),
        MirOp::MILAsin { name, .. } => Some(name.clone()),
        MirOp::MILAtan { name, .. } => Some(name.clone()),
        MirOp::MILCosh { name, .. } => Some(name.clone()),
        MirOp::MILSinh { name, .. } => Some(name.clone()),
        MirOp::MILAtanh { name, .. } => Some(name.clone()),
        MirOp::MILErf { name, .. } => Some(name.clone()),
        MirOp::MILLogicalNot { name, .. } => Some(name.clone()),
        MirOp::MILSelect { name, .. } => Some(name.clone()),
        MirOp::MILReduceMax { name, .. } => Some(name.clone()),
        MirOp::MILReduceMin { name, .. } => Some(name.clone()),
        MirOp::MILReduceProd { name, .. } => Some(name.clone()),
        MirOp::MILReduceSumSquare { name, .. } => Some(name.clone()),
        MirOp::MILReduceL2Norm { name, .. } => Some(name.clone()),
        MirOp::MILReduceL1Norm { name, .. } => Some(name.clone()),
        MirOp::MILReduceLogSumExp { name, .. } => Some(name.clone()),
        MirOp::MILReduceLogSum { name, .. } => Some(name.clone()),
        MirOp::MILReduceArgmax { name, .. } => Some(name.clone()),
        MirOp::MILReduceArgmin { name, .. } => Some(name.clone()),
        MirOp::MILBatchNorm { name, .. } => Some(name.clone()),
        MirOp::MILInstanceNorm { name, .. } => Some(name.clone()),
        MirOp::MILL2Norm { name, .. } => Some(name.clone()),
        MirOp::MILLocalResponseNorm { name, .. } => Some(name.clone()),
        MirOp::MILMaxPool { name, .. } => Some(name.clone()),
        MirOp::MILAvgPool { name, .. } => Some(name.clone()),
        MirOp::MILL2Pool { name, .. } => Some(name.clone()),
        MirOp::MILResize { name, .. } => Some(name.clone()),
        MirOp::MILResizeNearestNeighbor { name, .. } => Some(name.clone()),
        MirOp::MILResizeBilinear { name, .. } => Some(name.clone()),
        MirOp::MILUpsampleNearestNeighbor { name, .. } => Some(name.clone()),
        MirOp::MILUpsampleBilinear { name, .. } => Some(name.clone()),
        MirOp::MILCropResize { name, .. } => Some(name.clone()),
        MirOp::MILAffine { name, .. } => Some(name.clone()),
        MirOp::MILResample { name, .. } => Some(name.clone()),
        MirOp::MILReshapeLike { name, .. } => Some(name.clone()),
        MirOp::MILExpandDims { name, .. } => Some(name.clone()),
        MirOp::MILSqueeze { name, .. } => Some(name.clone()),
        MirOp::MILFlatten2d { name, .. } => Some(name.clone()),
        MirOp::MILReverse { name, .. } => Some(name.clone()),
        MirOp::MILReverseSequence { name, .. } => Some(name.clone()),
        MirOp::MILSliceBySize { name, .. } => Some(name.clone()),
        MirOp::MILSlidingWindows { name, .. } => Some(name.clone()),
        MirOp::MILDepthToSpace { name, .. } => Some(name.clone()),
        MirOp::MILSpaceToDepth { name, .. } => Some(name.clone()),
        MirOp::MILPixelShuffle { name, .. } => Some(name.clone()),
        MirOp::MILPixelUnshuffle { name, .. } => Some(name.clone()),
        MirOp::MILBatchToSpace { name, .. } => Some(name.clone()),
        MirOp::MILSpaceToBatch { name, .. } => Some(name.clone()),
        MirOp::MILPad { name, .. } => Some(name.clone()),
        MirOp::MILStack { name, .. } => Some(name.clone()),
        MirOp::MILTile { name, .. } => Some(name.clone()),
        MirOp::MILCumsum { name, .. } => Some(name.clone()),
        MirOp::MILFill { name, .. } => Some(name.clone()),
        MirOp::MILFillLike { name, .. } => Some(name.clone()),
        MirOp::MILIdentity { name, .. } => Some(name.clone()),
        MirOp::MILOneHot { name, .. } => Some(name.clone()),
        MirOp::MILNonZero { name, .. } => Some(name.clone()),
        MirOp::MILArgsort { name, .. } => Some(name.clone()),
        MirOp::MILBandPart { name, .. } => Some(name.clone()),
        MirOp::MILRange1d { name, .. } => Some(name.clone()),
        MirOp::MILShape { name, .. } => Some(name.clone()),
        MirOp::MILCrop { name, .. } => Some(name.clone()),
        MirOp::MILGatherAlongAxis { name, .. } => Some(name.clone()),
        MirOp::MILGatherNd { name, .. } => Some(name.clone()),
        MirOp::MILScatter { name, .. } => Some(name.clone()),
        MirOp::MILScatterAlongAxis { name, .. } => Some(name.clone()),
        MirOp::MILScatterNd { name, .. } => Some(name.clone()),
        MirOp::MILNonMaximumSuppression { name, .. } => Some(name.clone()),
        MirOp::MILQuantize { name, .. } => Some(name.clone()),
        MirOp::MILDequantize { name, .. } => Some(name.clone()),
        MirOp::MILConstexprAffineDequantize { name, .. } => Some(name.clone()),
        MirOp::MILConstexprBlockwiseShiftScale { name, .. } => Some(name.clone()),
        MirOp::MILConstexprLutToDense { name, .. } => Some(name.clone()),
        MirOp::MILConstexprSparseToDense { name, .. } => Some(name.clone()),
        MirOp::MILConstexprCast { name, .. } => Some(name.clone()),
        MirOp::MILConstexprLutToSparse { name, .. } => Some(name.clone()),
        MirOp::MILConstexprSparseBlockwiseShiftScale { name, .. } => Some(name.clone()),
        MirOp::MILRnn { name, .. } => Some(name.clone()),
        MirOp::MILGru { name, .. } => Some(name.clone()),
        MirOp::MILLstm { name, .. } => Some(name.clone()),
        MirOp::MILCond { name, .. } => Some(name.clone()),
        MirOp::MILWhileLoop { name, .. } => Some(name.clone()),
        MirOp::MILMakeList { name, .. } => Some(name.clone()),
        MirOp::MILListLength { name, .. } => Some(name.clone()),
        MirOp::MILListWrite { name, .. } => Some(name.clone()),
        MirOp::MILListRead { name, .. } => Some(name.clone()),
        MirOp::MILListGather { name, .. } => Some(name.clone()),
        MirOp::MILListScatter { name, .. } => Some(name.clone()),
        MirOp::MILRandomBernoulli { name, .. } => Some(name.clone()),
        MirOp::MILRandomNormal { name, .. } => Some(name.clone()),
        MirOp::MILRandomUniform { name, .. } => Some(name.clone()),
        MirOp::MILRandomCategorical { name, .. } => Some(name.clone()),
        MirOp::MILClassify { name, .. } => Some(name.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::mir::{MilDtype, MirNode, MirNodeId};

    fn make_simple_mir_graph() -> MirGraph {
        MirGraph {
            nodes: vec![
                MirNode {
                    id: MirNodeId("const_w".into()),
                    op: MirOp::MILConst {
                        name: "weight".into(),
                        value_path: "weights/w.bin".into(),
                        dtype: MilDtype::Fp16,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![64, 32],
                    compute_unit_hint: None,
                    air_source: None,
                target_annotation: Default::default(),
                },
                MirNode {
                    id: MirNodeId("linear".into()),
                    op: MirOp::MILLinear {
                        name: "output".into(),
                        x: MirNodeId("input".into()),
                        weight: "weight".into(),
                        bias: None,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![1, 32],
                    compute_unit_hint: None,
                    air_source: None,
                target_annotation: Default::default(),
                },
            ],
            inputs: vec![MirNodeId("input".into())],
            outputs: vec![MirNodeId("linear".into())],
            opset_version: "iOS18".into(),
            shard_name: "shard_0".into(),
            input_shapes: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_mir_to_mil_name_mapping() {
        assert_eq!(mir_to_mil_name("MILLinear"), Some("linear"));
        assert_eq!(mir_to_mil_name("MILGelu"), Some("gelu"));
        assert_eq!(
            mir_to_mil_name("MILScaledDotProductAttention"),
            Some("scaled_dot_product_attention")
        );
        assert_eq!(mir_to_mil_name("UNKNOWN"), None);
    }

    #[test]
    fn test_extract_mir_mil_names() {
        let mir = make_simple_mir_graph();
        let names = extract_mir_mil_names(&mir);
        assert_eq!(names.len(), 2);
        assert_eq!(names[0].0, "MILConst");
        assert_eq!(names[0].1, Some("const".to_string()));
        assert_eq!(names[1].0, "MILLinear");
        assert_eq!(names[1].1, Some("linear".to_string()));
    }

    #[test]
    fn test_compare_mir_vs_structure_perfect_match() {
        let mir = make_simple_mir_graph();
        let structure_ops = vec!["const".to_string(), "linear".to_string()];
        let result = compare_mir_vs_structure(&mir, &structure_ops);
        assert_eq!(result.mir_op_count, 2);
        assert!(result.op_fidelity_score > 0.99);
        assert!(result.missing_ops.is_empty());
    }

    #[test]
    fn test_compare_mir_vs_structure_missing_op() {
        let mir = make_simple_mir_graph();
        // Structure missing "linear" — only has "const"
        let structure_ops = vec!["const".to_string()];
        let result = compare_mir_vs_structure(&mir, &structure_ops);
        assert!(result.missing_ops.iter().any(|m| m.contains("linear")));
        assert!(result.op_fidelity_score < 1.0);
    }

    #[test]
    fn test_compare_mir_vs_structure_extra_op() {
        let mir = make_simple_mir_graph();
        let structure_ops = vec![
            "const".to_string(),
            "linear".to_string(),
            "reshape".to_string(), // Extra op not in MIR
        ];
        let result = compare_mir_vs_structure(&mir, &structure_ops);
        assert!(result.extra_ops.iter().any(|e| e.contains("reshape")));
    }

    #[test]
    fn test_mir_ops_for_bridge() {
        let mir = make_simple_mir_graph();
        let bridge_ops = mir_ops_for_bridge(&mir);
        assert_eq!(bridge_ops.len(), 2);
        assert_eq!(bridge_ops[0].get("op_type").unwrap().as_str(), Some("MILConst"));
        assert_eq!(bridge_ops[1].get("op_type").unwrap().as_str(), Some("MILLinear"));
    }

    #[test]
    fn test_op_fidelity_score_empty_mir() {
        let mir = MirGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            opset_version: "iOS18".into(),
            shard_name: "shard_0".into(),
            input_shapes: std::collections::HashMap::new(),
        };
        let result = compare_mir_vs_structure(&mir, &[]);
        assert!((result.op_fidelity_score - 1.0).abs() < 1e-10);
    }
}
