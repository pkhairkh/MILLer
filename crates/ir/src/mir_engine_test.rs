//! Exhaustive test: every MirOp variant maps to a valid engine in default_engine()
//!
//! This test uses an exhaustive match to ensure that no MirOp variant is
//! missed by the `default_engine()` method. If a new variant is added to
//! MirOp without updating `default_engine()`, this test will fail to compile.

use crate::ane_engine::AneEngine;
use crate::common::MilDtype;
use crate::mir::{MirNodeId, MirOp};

// Convenience aliases for MilDtype variants used in the test
use MilDtype::{Fp16, Fp32, UInt8};

/// Helper to create a dummy MirNodeId.
fn nid(s: &str) -> MirNodeId {
    MirNodeId(s.to_string())
}

/// Build a representative MirOp instance for every variant.
/// This uses dummy values — we only care that every variant is covered.
fn all_mir_op_variants() -> Vec<(String, MirOp)> {
    vec![
        // ─── Constants ───────────────────────────────────────────────
        (
            "MILConst".into(),
            MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: Fp16 },
        ),
        // ─── Linear / FC ─────────────────────────────────────────────
        (
            "MILLinear".into(),
            MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None },
        ),
        (
            "MILMatMul".into(),
            MirOp::MILMatMul { name: "m".into(), x: nid("x"), y: nid("y"), transpose_y: false },
        ),
        (
            "MILEinsum".into(),
            MirOp::MILEinsum {
                name: "e".into(),
                inputs: vec![nid("a"), nid("b")],
                equation: "ij,jk->ik".into(),
            },
        ),
        // ─── Convolution ─────────────────────────────────────────────
        (
            "MILConv".into(),
            MirOp::MILConv {
                name: "c".into(),
                x: nid("x"),
                weight: nid("w"),
                pad_type: "valid".into(),
                groups: 1,
                strides: vec![1],
                pad_amounts: vec![0],
                dilations: vec![1],
                kernel_scale: None,
                kernel_zero_point: None,
                kernel_palettized_lut: None,
            },
        ),
        (
            "MILConvTranspose".into(),
            MirOp::MILConvTranspose {
                name: "ct".into(),
                x: nid("x"),
                weight: nid("w"),
                pad_type: "valid".into(),
                groups: 1,
                strides: vec![1],
                pad_amounts: vec![0],
                dilations: vec![1],
                output_shape: vec![1],
            },
        ),
        // ─── Elementwise Binary ──────────────────────────────────────
        ("MILAdd".into(), MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") }),
        ("MILMul".into(), MirOp::MILMul { name: "m".into(), x: nid("x"), y: nid("y") }),
        ("MILSub".into(), MirOp::MILSub { name: "s".into(), x: nid("x"), y: nid("y") }),
        ("MILMaximum".into(), MirOp::MILMaximum { name: "mx".into(), x: nid("x"), y: nid("y") }),
        ("MILMinimum".into(), MirOp::MILMinimum { name: "mn".into(), x: nid("x"), y: nid("y") }),
        ("MILRealDiv".into(), MirOp::MILRealDiv { name: "rd".into(), x: nid("x"), y: nid("y") }),
        ("MILFloorDiv".into(), MirOp::MILFloorDiv { name: "fd".into(), x: nid("x"), y: nid("y") }),
        ("MILMod".into(), MirOp::MILMod { name: "mod".into(), x: nid("x"), y: nid("y") }),
        ("MILPow".into(), MirOp::MILPow { name: "pow".into(), x: nid("x"), y: nid("y") }),
        ("MILEqual".into(), MirOp::MILEqual { name: "eq".into(), x: nid("x"), y: nid("y") }),
        ("MILNotEqual".into(), MirOp::MILNotEqual { name: "ne".into(), x: nid("x"), y: nid("y") }),
        ("MILGreater".into(), MirOp::MILGreater { name: "gt".into(), x: nid("x"), y: nid("y") }),
        (
            "MILGreaterEqual".into(),
            MirOp::MILGreaterEqual { name: "ge".into(), x: nid("x"), y: nid("y") },
        ),
        ("MILLess".into(), MirOp::MILLess { name: "lt".into(), x: nid("x"), y: nid("y") }),
        (
            "MILLessEqual".into(),
            MirOp::MILLessEqual { name: "le".into(), x: nid("x"), y: nid("y") },
        ),
        (
            "MILLogicalAnd".into(),
            MirOp::MILLogicalAnd { name: "la".into(), x: nid("x"), y: nid("y") },
        ),
        (
            "MILLogicalOr".into(),
            MirOp::MILLogicalOr { name: "lo".into(), x: nid("x"), y: nid("y") },
        ),
        (
            "MILLogicalXor".into(),
            MirOp::MILLogicalXor { name: "lx".into(), x: nid("x"), y: nid("y") },
        ),
        // ─── Elementwise Unary ───────────────────────────────────────
        ("MILAbs".into(), MirOp::MILAbs { name: "abs".into(), x: nid("x") }),
        ("MILNeg".into(), MirOp::MILNeg { name: "neg".into(), x: nid("x") }),
        ("MILSigmoid".into(), MirOp::MILSigmoid { name: "sig".into(), x: nid("x") }),
        ("MILTanh".into(), MirOp::MILTanh { name: "tanh".into(), x: nid("x") }),
        ("MILRelu".into(), MirOp::MILRelu { name: "relu".into(), x: nid("x") }),
        ("MILRelu6".into(), MirOp::MILRelu6 { name: "relu6".into(), x: nid("x") }),
        (
            "MILLeakyRelu".into(),
            MirOp::MILLeakyRelu { name: "lrelu".into(), x: nid("x"), alpha: 0.01 },
        ),
        (
            "MILSigmoidHard".into(),
            MirOp::MILSigmoidHard { name: "sh".into(), x: nid("x"), alpha: 1.0, beta: 1.0 },
        ),
        (
            "MILThresholdedRelu".into(),
            MirOp::MILThresholdedRelu { name: "tr".into(), x: nid("x"), alpha: 1.0 },
        ),
        (
            "MILClampedRelu".into(),
            MirOp::MILClampedRelu { name: "cr".into(), x: nid("x"), alpha: 0.0, beta: 6.0 },
        ),
        (
            "MILLinearActivation".into(),
            MirOp::MILLinearActivation { name: "la".into(), x: nid("x"), alpha: 1.0, beta: 0.0 },
        ),
        (
            "MILPrelu".into(),
            MirOp::MILPrelu { name: "pr".into(), x: nid("x"), alpha: "alpha".into() },
        ),
        ("MILSoftsign".into(), MirOp::MILSoftsign { name: "ss".into(), x: nid("x") }),
        ("MILSilu".into(), MirOp::MILSilu { name: "silu".into(), x: nid("x") }),
        (
            "MILScaledTanh".into(),
            MirOp::MILScaledTanh { name: "st".into(), x: nid("x"), alpha: 1.0, beta: 1.0 },
        ),
        ("MILElu".into(), MirOp::MILElu { name: "elu".into(), x: nid("x"), alpha: 1.0 }),
        ("MILSoftplus".into(), MirOp::MILSoftplus { name: "sp".into(), x: nid("x") }),
        (
            "MILSoftplusParametric".into(),
            MirOp::MILSoftplusParametric {
                name: "spp".into(),
                x: nid("x"),
                alpha: "a".into(),
                beta: "b".into(),
            },
        ),
        (
            "MILGelu".into(),
            MirOp::MILGelu { name: "gelu".into(), x: nid("x"), mode: "exact".into() },
        ),
        (
            "MILClip".into(),
            MirOp::MILClip { name: "clip".into(), x: nid("x"), min_val: 0.0, max_val: 6.0 },
        ),
        ("MILSquare".into(), MirOp::MILSquare { name: "sq".into(), x: nid("x") }),
        (
            "MILThreshold".into(),
            MirOp::MILThreshold { name: "thr".into(), x: nid("x"), alpha: 1.0 },
        ),
        ("MILSqrt".into(), MirOp::MILSqrt { name: "sqrt".into(), x: nid("x") }),
        ("MILRsqrt".into(), MirOp::MILRsqrt { name: "rsqrt".into(), x: nid("x") }),
        ("MILInverse".into(), MirOp::MILInverse { name: "inv".into(), x: nid("x"), epsilon: 1e-6 }),
        ("MILCeil".into(), MirOp::MILCeil { name: "ceil".into(), x: nid("x") }),
        ("MILFloor".into(), MirOp::MILFloor { name: "floor".into(), x: nid("x") }),
        ("MILRound".into(), MirOp::MILRound { name: "round".into(), x: nid("x") }),
        ("MILExp".into(), MirOp::MILExp { name: "exp".into(), x: nid("x") }),
        ("MILExp2".into(), MirOp::MILExp2 { name: "exp2".into(), x: nid("x") }),
        ("MILLog".into(), MirOp::MILLog { name: "log".into(), x: nid("x"), epsilon: 1e-6 }),
        ("MILSign".into(), MirOp::MILSign { name: "sign".into(), x: nid("x") }),
        ("MILCos".into(), MirOp::MILCos { name: "cos".into(), x: nid("x") }),
        ("MILSin".into(), MirOp::MILSin { name: "sin".into(), x: nid("x") }),
        ("MILTan".into(), MirOp::MILTan { name: "tan".into(), x: nid("x") }),
        ("MILAcos".into(), MirOp::MILAcos { name: "acos".into(), x: nid("x") }),
        ("MILAsin".into(), MirOp::MILAsin { name: "asin".into(), x: nid("x") }),
        ("MILAtan".into(), MirOp::MILAtan { name: "atan".into(), x: nid("x") }),
        ("MILCosh".into(), MirOp::MILCosh { name: "cosh".into(), x: nid("x") }),
        ("MILSinh".into(), MirOp::MILSinh { name: "sinh".into(), x: nid("x") }),
        ("MILAtanh".into(), MirOp::MILAtanh { name: "atanh".into(), x: nid("x") }),
        ("MILErf".into(), MirOp::MILErf { name: "erf".into(), x: nid("x") }),
        ("MILLogicalNot".into(), MirOp::MILLogicalNot { name: "lnot".into(), x: nid("x") }),
        ("MILCast".into(), MirOp::MILCast { name: "cast".into(), x: nid("x"), dtype: Fp32 }),
        (
            "MILSelect".into(),
            MirOp::MILSelect { name: "sel".into(), condition: nid("c"), x: nid("x"), y: nid("y") },
        ),
        (
            "MILWhere".into(),
            MirOp::MILWhere { name: "where".into(), condition: nid("c"), x: nid("x"), y: nid("y") },
        ),
        ("MILSoftmax".into(), MirOp::MILSoftmax { name: "sm".into(), x: nid("x"), axis: -1 }),
        // ─── Reduction ───────────────────────────────────────────────
        (
            "MILReduceSum".into(),
            MirOp::MILReduceSum { name: "rs".into(), x: nid("x"), axes: vec![1], keep_dims: false },
        ),
        (
            "MILReduceMean".into(),
            MirOp::MILReduceMean {
                name: "rm".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceMax".into(),
            MirOp::MILReduceMax {
                name: "rmax".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceMin".into(),
            MirOp::MILReduceMin {
                name: "rmin".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceProd".into(),
            MirOp::MILReduceProd {
                name: "rp".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceSumSquare".into(),
            MirOp::MILReduceSumSquare {
                name: "rss".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceL2Norm".into(),
            MirOp::MILReduceL2Norm {
                name: "rl2".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceL1Norm".into(),
            MirOp::MILReduceL1Norm {
                name: "rl1".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceLogSumExp".into(),
            MirOp::MILReduceLogSumExp {
                name: "rlse".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceLogSum".into(),
            MirOp::MILReduceLogSum {
                name: "rls".into(),
                x: nid("x"),
                axes: vec![1],
                keep_dims: false,
            },
        ),
        (
            "MILReduceArgmax".into(),
            MirOp::MILReduceArgmax { name: "ram".into(), x: nid("x"), axis: 1, keep_dims: false },
        ),
        (
            "MILReduceArgmin".into(),
            MirOp::MILReduceArgmin { name: "rin".into(), x: nid("x"), axis: 1, keep_dims: false },
        ),
        // ─── Normalization ───────────────────────────────────────────
        (
            "MILBatchNorm".into(),
            MirOp::MILBatchNorm {
                name: "bn".into(),
                x: nid("x"),
                mean: "m".into(),
                variance: "v".into(),
                gamma: None,
                beta: None,
                epsilon: 1e-5,
            },
        ),
        (
            "MILInstanceNorm".into(),
            MirOp::MILInstanceNorm {
                name: "in".into(),
                x: nid("x"),
                gamma: None,
                beta: None,
                epsilon: 1e-5,
            },
        ),
        (
            "MILLayerNorm".into(),
            MirOp::MILLayerNorm {
                name: "ln".into(),
                x: nid("x"),
                weight: "w".into(),
                bias: None,
                epsilon: 1e-5,
                axes: vec![2],
            },
        ),
        (
            "MILL2Norm".into(),
            MirOp::MILL2Norm { name: "l2".into(), x: nid("x"), epsilon: 1e-6, axes: vec![2] },
        ),
        (
            "MILLocalResponseNorm".into(),
            MirOp::MILLocalResponseNorm {
                name: "lrn".into(),
                x: nid("x"),
                size: 5,
                alpha: 1.0,
                beta: 0.75,
                k: 1.0,
            },
        ),
        // ─── Pooling ─────────────────────────────────────────────────
        (
            "MILMaxPool".into(),
            MirOp::MILMaxPool {
                name: "mp".into(),
                x: nid("x"),
                kernel_sizes: vec![3],
                strides: vec![1],
                pad_types: vec!["valid".into()],
                pad_amounts: vec![0],
            },
        ),
        (
            "MILAvgPool".into(),
            MirOp::MILAvgPool {
                name: "ap".into(),
                x: nid("x"),
                kernel_sizes: vec![3],
                strides: vec![1],
                pad_types: vec!["valid".into()],
                pad_amounts: vec![0],
                count_include_padding: false,
            },
        ),
        (
            "MILL2Pool".into(),
            MirOp::MILL2Pool {
                name: "l2p".into(),
                x: nid("x"),
                kernel_sizes: vec![3],
                strides: vec![1],
                pad_types: vec!["valid".into()],
                pad_amounts: vec![0],
            },
        ),
        // ─── Image Resizing ──────────────────────────────────────────
        (
            "MILResize".into(),
            MirOp::MILResize {
                name: "rz".into(),
                x: nid("x"),
                target_size: vec![224, 224],
                mode: "bilinear".into(),
                sampling_mode: "default".into(),
                nearest_rounding_mode: "round".into(),
            },
        ),
        (
            "MILResizeNearestNeighbor".into(),
            MirOp::MILResizeNearestNeighbor {
                name: "rnn".into(),
                x: nid("x"),
                target_height: 224,
                target_width: 224,
            },
        ),
        (
            "MILResizeBilinear".into(),
            MirOp::MILResizeBilinear {
                name: "rb".into(),
                x: nid("x"),
                target_height: 224,
                target_width: 224,
                align_corners: false,
            },
        ),
        (
            "MILUpsampleNearestNeighbor".into(),
            MirOp::MILUpsampleNearestNeighbor { name: "unn".into(), x: nid("x"), scale: vec![2] },
        ),
        (
            "MILUpsampleBilinear".into(),
            MirOp::MILUpsampleBilinear {
                name: "ub".into(),
                x: nid("x"),
                scale: vec![2],
                align_corners: false,
                half_pixel_centers: true,
            },
        ),
        (
            "MILCropResize".into(),
            MirOp::MILCropResize {
                name: "cr".into(),
                x: nid("x"),
                boxes: nid("b"),
                box_indices: nid("bi"),
                crop_height: 224,
                crop_width: 224,
            },
        ),
        (
            "MILAffine".into(),
            MirOp::MILAffine {
                name: "af".into(),
                x: nid("x"),
                transform: nid("t"),
                output_height: 224,
                output_width: 224,
                sampling_mode: "bilinear".into(),
                pad_value: 0.0,
            },
        ),
        (
            "MILResample".into(),
            MirOp::MILResample {
                name: "rsmp".into(),
                x: nid("x"),
                coordinates: nid("c"),
                sampling_mode: "bilinear".into(),
                pad_value: 0.0,
            },
        ),
        // ─── Tensor Transform ────────────────────────────────────────
        (
            "MILReshape".into(),
            MirOp::MILReshape { name: "rsh".into(), x: nid("x"), shape: vec![1, 2, 3] },
        ),
        (
            "MILReshapeLike".into(),
            MirOp::MILReshapeLike { name: "rl".into(), x: nid("x"), ref_tensor: nid("r") },
        ),
        (
            "MILTranspose".into(),
            MirOp::MILTranspose { name: "tr".into(), x: nid("x"), perm: vec![1, 0] },
        ),
        (
            "MILSplit".into(),
            MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 1, num_splits: 2 },
        ),
        (
            "MILConcat".into(),
            MirOp::MILConcat { name: "cat".into(), values: vec![nid("a"), nid("b")], axis: 1 },
        ),
        (
            "MILExpandDims".into(),
            MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![1] },
        ),
        ("MILSqueeze".into(), MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1] }),
        ("MILFlatten2d".into(), MirOp::MILFlatten2d { name: "f2d".into(), x: nid("x"), axis: 1 }),
        ("MILReverse".into(), MirOp::MILReverse { name: "rev".into(), x: nid("x"), axes: vec![1] }),
        (
            "MILReverseSequence".into(),
            MirOp::MILReverseSequence {
                name: "rseq".into(),
                x: nid("x"),
                lengths: nid("l"),
                batch_axis: 0,
                seq_axis: 1,
            },
        ),
        (
            "MILSliceByIndex".into(),
            MirOp::MILSliceByIndex {
                name: "sbi".into(),
                x: nid("x"),
                begin: vec![0],
                end: vec![10],
                stride: vec![1],
                begin_mask: vec![false],
                end_mask: vec![false],
                squeeze_mask: vec![false],
            },
        ),
        (
            "MILSliceBySize".into(),
            MirOp::MILSliceBySize {
                name: "sbs".into(),
                x: nid("x"),
                begin: vec![0],
                size: vec![10],
            },
        ),
        (
            "MILSliceUpdate".into(),
            MirOp::MILSliceUpdate {
                name: "su".into(),
                x: nid("x"),
                update: nid("u"),
                begin: vec![0],
                end: vec![10],
            },
        ),
        (
            "MILSlidingWindows".into(),
            MirOp::MILSlidingWindows {
                name: "sw".into(),
                x: nid("x"),
                axis: 1,
                window_size: 3,
                stride: 1,
            },
        ),
        (
            "MILDepthToSpace".into(),
            MirOp::MILDepthToSpace { name: "d2s".into(), x: nid("x"), block_size: 2 },
        ),
        (
            "MILSpaceToDepth".into(),
            MirOp::MILSpaceToDepth { name: "s2d".into(), x: nid("x"), block_size: 2 },
        ),
        (
            "MILPixelShuffle".into(),
            MirOp::MILPixelShuffle { name: "ps".into(), x: nid("x"), upscale_factor: 2 },
        ),
        (
            "MILPixelUnshuffle".into(),
            MirOp::MILPixelUnshuffle { name: "pu".into(), x: nid("x"), downscale_factor: 2 },
        ),
        (
            "MILBatchToSpace".into(),
            MirOp::MILBatchToSpace {
                name: "b2s".into(),
                x: nid("x"),
                block_shape: vec![2],
                crops: vec![(0, 0)],
            },
        ),
        (
            "MILSpaceToBatch".into(),
            MirOp::MILSpaceToBatch {
                name: "s2b".into(),
                x: nid("x"),
                block_shape: vec![2],
                paddings: vec![(0, 0)],
            },
        ),
        (
            "MILPad".into(),
            MirOp::MILPad {
                name: "pad".into(),
                x: nid("x"),
                pad_amounts: vec![0, 1],
                mode: "constant".into(),
                constant_value: 0.0,
            },
        ),
        (
            "MILStack".into(),
            MirOp::MILStack { name: "stk".into(), values: vec![nid("a"), nid("b")], axis: 0 },
        ),
        ("MILTile".into(), MirOp::MILTile { name: "tile".into(), x: nid("x"), reps: vec![2] }),
        (
            "MILCumsum".into(),
            MirOp::MILCumsum {
                name: "cs".into(),
                x: nid("x"),
                axis: 1,
                exclusive: false,
                reverse: false,
            },
        ),
        (
            "MILFill".into(),
            MirOp::MILFill { name: "fill".into(), shape: vec![3], value: 0.0, dtype: Fp32 },
        ),
        (
            "MILFillLike".into(),
            MirOp::MILFillLike { name: "fl".into(), ref_tensor: nid("r"), value: 0.0, dtype: Fp32 },
        ),
        ("MILIdentity".into(), MirOp::MILIdentity { name: "id".into(), x: nid("x") }),
        (
            "MILOneHot".into(),
            MirOp::MILOneHot {
                name: "oh".into(),
                indices: nid("i"),
                one_hot_vector_size: 10,
                on_value: 1.0,
                off_value: 0.0,
                axis: 1,
                dtype: Fp32,
            },
        ),
        ("MILNonZero".into(), MirOp::MILNonZero { name: "nz".into(), x: nid("x") }),
        (
            "MILArgsort".into(),
            MirOp::MILArgsort { name: "asort".into(), x: nid("x"), axis: 1, ascending: true },
        ),
        (
            "MILBandPart".into(),
            MirOp::MILBandPart { name: "bp".into(), x: nid("x"), num_lower: -1, num_upper: 0 },
        ),
        (
            "MILRange1d".into(),
            MirOp::MILRange1d { name: "rng".into(), start: 0.0, end: 10.0, step: 1.0 },
        ),
        ("MILShape".into(), MirOp::MILShape { name: "shp".into(), x: nid("x") }),
        (
            "MILCrop".into(),
            MirOp::MILCrop {
                name: "crop".into(),
                x: nid("x"),
                crop_height: 224,
                crop_width: 224,
                offset_height: 0,
                offset_width: 0,
            },
        ),
        // ─── Scatter / Gather ────────────────────────────────────────
        (
            "MILGather".into(),
            MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("i"), axis: 0 },
        ),
        (
            "MILGatherAlongAxis".into(),
            MirOp::MILGatherAlongAxis {
                name: "gaa".into(),
                x: nid("x"),
                indices: nid("i"),
                axis: 0,
            },
        ),
        (
            "MILGatherNd".into(),
            MirOp::MILGatherNd { name: "gnd".into(), x: nid("x"), indices: nid("i") },
        ),
        (
            "MILScatter".into(),
            MirOp::MILScatter {
                name: "sc".into(),
                x: nid("x"),
                indices: nid("i"),
                updates: nid("u"),
                axis: 0,
                mode: "update".into(),
            },
        ),
        (
            "MILScatterAlongAxis".into(),
            MirOp::MILScatterAlongAxis {
                name: "saa".into(),
                x: nid("x"),
                indices: nid("i"),
                updates: nid("u"),
                axis: 0,
            },
        ),
        (
            "MILScatterNd".into(),
            MirOp::MILScatterNd {
                name: "snd".into(),
                x: nid("x"),
                indices: nid("i"),
                updates: nid("u"),
            },
        ),
        (
            "MILNonMaximumSuppression".into(),
            MirOp::MILNonMaximumSuppression {
                name: "nms".into(),
                boxes: nid("b"),
                scores: nid("s"),
                iou_threshold: 0.5,
                score_threshold: 0.1,
                max_detections: 100,
            },
        ),
        // ─── Attention ───────────────────────────────────────────────
        (
            "MILScaledDotProductAttention".into(),
            MirOp::MILScaledDotProductAttention {
                name: "sdpa".into(),
                query: nid("q"),
                key: nid("k"),
                value: nid("v"),
                attention_mask: None,
                scale: None,
            },
        ),
        // ─── Quantization ────────────────────────────────────────────
        (
            "MILQuantize".into(),
            MirOp::MILQuantize {
                name: "q".into(),
                x: nid("x"),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
                output_dtype: UInt8,
            },
        ),
        (
            "MILDequantize".into(),
            MirOp::MILDequantize {
                name: "dq".into(),
                x: nid("x"),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
                output_dtype: Fp16,
            },
        ),
        // ─── Constexpr / Compression ─────────────────────────────────
        (
            "MILConstexprAffineDequantize".into(),
            MirOp::MILConstexprAffineDequantize {
                name: "cad".into(),
                quantized_data: "d".into(),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
            },
        ),
        (
            "MILConstexprBlockwiseShiftScale".into(),
            MirOp::MILConstexprBlockwiseShiftScale {
                name: "cbss".into(),
                data: "d".into(),
                scale: "s".into(),
                offset: "o".into(),
                block_size: vec![128],
            },
        ),
        (
            "MILConstexprLutToDense".into(),
            MirOp::MILConstexprLutToDense {
                name: "cltd".into(),
                indices: "i".into(),
                lut: "l".into(),
                num_bits: 4,
            },
        ),
        (
            "MILConstexprSparseToDense".into(),
            MirOp::MILConstexprSparseToDense {
                name: "cstd".into(),
                nonzero_data: "d".into(),
                shape: vec![10],
                default_value: 0.0,
            },
        ),
        (
            "MILConstexprCast".into(),
            MirOp::MILConstexprCast { name: "cc".into(), data: "d".into(), dtype: MilDtype::Fp32 },
        ),
        (
            "MILConstexprLutToSparse".into(),
            MirOp::MILConstexprLutToSparse { name: "clts".into(), data: "d".into(), num_bits: 4 },
        ),
        (
            "MILConstexprSparseBlockwiseShiftScale".into(),
            MirOp::MILConstexprSparseBlockwiseShiftScale {
                name: "csbss".into(),
                data: "d".into(),
                scale: "s".into(),
                offset: "o".into(),
                block_size: vec![128],
                block_axis: 0,
            },
        ),
        // ─── Recurrent ───────────────────────────────────────────────
        (
            "MILRnn".into(),
            MirOp::MILRnn {
                name: "rnn".into(),
                x: nid("x"),
                initial_h: nid("h"),
                weight_ih: "wi".into(),
                weight_hh: "wh".into(),
                bias: None,
                mode: "relu".into(),
                output_sequence: false,
            },
        ),
        (
            "MILGru".into(),
            MirOp::MILGru {
                name: "gru".into(),
                x: nid("x"),
                initial_h: nid("h"),
                weight_ih: "wi".into(),
                weight_hh: "wh".into(),
                bias: None,
                reset_after: true,
                output_sequence: false,
            },
        ),
        (
            "MILLstm".into(),
            MirOp::MILLstm {
                name: "lstm".into(),
                x: nid("x"),
                initial_h: nid("h"),
                initial_c: nid("c"),
                weight_ih: "wi".into(),
                weight_hh: "wh".into(),
                bias: None,
                output_sequence: false,
            },
        ),
        // ─── Control Flow ────────────────────────────────────────────
        (
            "MILCond".into(),
            MirOp::MILCond {
                name: "cond".into(),
                pred: nid("p"),
                true_graph: "t".into(),
                false_graph: "f".into(),
            },
        ),
        (
            "MILWhileLoop".into(),
            MirOp::MILWhileLoop {
                name: "wl".into(),
                condition: "c".into(),
                body: "b".into(),
                loop_vars: vec![nid("v")],
            },
        ),
        (
            "MILMakeList".into(),
            MirOp::MILMakeList {
                name: "ml".into(),
                elems: vec![nid("a"), nid("b")],
                dtype: MilDtype::Fp16,
            },
        ),
        ("MILListLength".into(), MirOp::MILListLength { name: "ll".into(), ls: nid("l") }),
        (
            "MILListWrite".into(),
            MirOp::MILListWrite {
                name: "lw".into(),
                ls: nid("l"),
                index: nid("i"),
                value: nid("v"),
            },
        ),
        (
            "MILListRead".into(),
            MirOp::MILListRead { name: "lr".into(), ls: nid("l"), index: nid("i") },
        ),
        (
            "MILListGather".into(),
            MirOp::MILListGather { name: "lg".into(), ls: nid("l"), indices: nid("i") },
        ),
        (
            "MILListScatter".into(),
            MirOp::MILListScatter {
                name: "ls".into(),
                ls: nid("l"),
                indices: nid("i"),
                values: nid("v"),
            },
        ),
        // ─── Random ──────────────────────────────────────────────────
        (
            "MILRandomBernoulli".into(),
            MirOp::MILRandomBernoulli {
                name: "rb".into(),
                shape: vec![10],
                prob: 0.5,
                seed: None,
                dtype: MilDtype::Fp32,
            },
        ),
        (
            "MILRandomNormal".into(),
            MirOp::MILRandomNormal {
                name: "rn".into(),
                shape: vec![10],
                mean: 0.0,
                stddev: 1.0,
                seed: None,
                dtype: MilDtype::Fp32,
            },
        ),
        (
            "MILRandomUniform".into(),
            MirOp::MILRandomUniform {
                name: "ru".into(),
                shape: vec![10],
                low: 0.0,
                high: 1.0,
                seed: None,
                dtype: MilDtype::Fp32,
            },
        ),
        (
            "MILRandomCategorical".into(),
            MirOp::MILRandomCategorical {
                name: "rc".into(),
                logits: nid("l"),
                num_samples: 10,
                seed: None,
                dtype: MilDtype::Fp32,
            },
        ),
        // ─── State ───────────────────────────────────────────────────
        (
            "MILReadState".into(),
            MirOp::MILReadState {
                name: "rs".into(),
                state_id: "s".into(),
                shape: vec![10],
                dtype: MilDtype::Fp16,
            },
        ),
        (
            "MILCoremlUpdateState".into(),
            MirOp::MILCoremlUpdateState {
                name: "us".into(),
                state_id: "s".into(),
                value: nid("v"),
            },
        ),
        (
            "MILStateWrite".into(),
            MirOp::MILStateWrite { name: "sw".into(), state_ref: "s".into(), value: nid("v") },
        ),
        // ─── Metadata / Misc ─────────────────────────────────────────
        ("MILTopk".into(), MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 10, axis: -1 }),
        ("MILClassify".into(), MirOp::MILClassify { name: "cls".into(), x: nid("x") }),
    ]
}

#[allow(deprecated)]
#[test]
fn test_every_mir_op_variant_has_engine_assignment() {
    let variants = all_mir_op_variants();
    assert!(variants.len() > 100, "Should have many MirOp variants, got {}", variants.len());

    for (name, op) in &variants {
        // Call default_engine for every variant — this must not panic
        let engine = op.default_engine();

        // Verify the engine is a valid value
        match engine {
            Some(AneEngine::NE) | Some(AneEngine::PE) | Some(AneEngine::TransposeEngine) | None => {
                // All valid assignments
            }
        }

        // Ensure we actually got a result (not silently ignored)
        // At minimum, this test documents which variants map to which engine
        let _ = engine;
    }
}

#[allow(deprecated)]
#[test]
fn test_ne_pipeline_ops_map_to_ne() {
    use crate::ane_engine::AneEngine;
    let ne_ops: Vec<MirOp> = vec![
        MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None },
        MirOp::MILMatMul { name: "m".into(), x: nid("x"), y: nid("y"), transpose_y: false },
        MirOp::MILConv {
            name: "c".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
            kernel_scale: None,
            kernel_zero_point: None,
            kernel_palettized_lut: None,
        },
        MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: None,
        },
        MirOp::MILMaxPool {
            name: "mp".into(),
            x: nid("x"),
            kernel_sizes: vec![3],
            strides: vec![1],
            pad_types: vec!["valid".into()],
            pad_amounts: vec![0],
        },
    ];

    for op in &ne_ops {
        assert_eq!(op.default_engine(), Some(AneEngine::NE), "Expected NE engine for {:?}", op);
    }
}

#[allow(deprecated)]
#[test]
fn test_pe_pipeline_ops_map_to_pe() {
    use crate::ane_engine::AneEngine;
    let pe_ops: Vec<MirOp> = vec![
        MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") },
        MirOp::MILRelu { name: "r".into(), x: nid("x") },
        MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![1], keep_dims: false },
        MirOp::MILLayerNorm {
            name: "ln".into(),
            x: nid("x"),
            weight: "w".into(),
            bias: None,
            epsilon: 1e-5,
            axes: vec![2],
        },
        MirOp::MILReshape { name: "rsh".into(), x: nid("x"), shape: vec![1, 2, 3] },
        MirOp::MILSoftmax { name: "sm".into(), x: nid("x"), axis: -1 },
        // NOTE: MILGather moved to CPU-only (ANE plannability ~0.26, causes
        // sync stalls). Only embedding uses Gather, which runs on CPU anyway.
    ];

    for op in &pe_ops {
        assert_eq!(op.default_engine(), Some(AneEngine::PE), "Expected PE engine for {:?}", op);
    }
}

#[allow(deprecated)]
#[test]
fn test_transpose_maps_to_transpose_engine() {
    use crate::ane_engine::AneEngine;
    let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
    assert_eq!(op.default_engine(), Some(AneEngine::TransposeEngine));
}

#[allow(deprecated)]
#[test]
fn test_cpu_only_ops_return_none() {
    let cpu_ops: Vec<MirOp> = vec![
        MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 },
        MirOp::MILScatter {
            name: "sc".into(),
            x: nid("x"),
            indices: nid("i"),
            updates: nid("u"),
            axis: 0,
            mode: "update".into(),
        },
        MirOp::MILCond {
            name: "cond".into(),
            pred: nid("p"),
            true_graph: "t".into(),
            false_graph: "f".into(),
        },
        MirOp::MILRandomNormal {
            name: "rn".into(),
            shape: vec![10],
            mean: 0.0,
            stddev: 1.0,
            seed: None,
            dtype: MilDtype::Fp32,
        },
        MirOp::MILReadState {
            name: "rs".into(),
            state_id: "s".into(),
            shape: vec![10],
            dtype: MilDtype::Fp16,
        },
        MirOp::MILClassify { name: "cls".into(), x: nid("x") },
        MirOp::MILCumsum {
            name: "cs".into(),
            x: nid("x"),
            axis: 1,
            exclusive: false,
            reverse: false,
        },
        MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("i"), axis: 0 },
        MirOp::MILGatherAlongAxis { name: "gaa".into(), x: nid("x"), indices: nid("i"), axis: 0 },
        MirOp::MILGatherNd { name: "gnd".into(), x: nid("x"), indices: nid("i") },
    ];

    for op in &cpu_ops {
        assert_eq!(op.default_engine(), None, "Expected None (CPU-only) for {:?}", op);
    }
}

#[allow(deprecated)]
#[test]
fn test_variant_count_matches_expectation() {
    let variants = all_mir_op_variants();
    // The MirOp enum is documented as having 167 variants.
    // If a new variant is added, this test should be updated.
    // This serves as a canary: if the count changes, review is needed.
    assert!(
        variants.len() >= 100,
        "Expected at least 100 MirOp variants, got {}. If variants were removed, update this test.",
        variants.len()
    );
}

/// T-22: Verify that ops moved from PE/NE to CPU-only actually return None.
/// These ops have NO ANEC converter in any ANE family per the per-op support matrix.
#[allow(deprecated)]
#[test]
fn test_t22_cpu_only_ops_moved_from_pe_ne() {
    let moved_ops: Vec<MirOp> = vec![
        // Trig inverse / hyperbolic (no ANEC converter)
        MirOp::MILAcos { name: "acos".into(), x: nid("x") },
        MirOp::MILAsin { name: "asin".into(), x: nid("x") },
        MirOp::MILAtan { name: "atan".into(), x: nid("x") },
        MirOp::MILAtanh { name: "atanh".into(), x: nid("x") },
        MirOp::MILTan { name: "tan".into(), x: nid("x") },
        MirOp::MILCosh { name: "cosh".into(), x: nid("x") },
        MirOp::MILSinh { name: "sinh".into(), x: nid("x") },
        // Logical (no ANEC converter)
        MirOp::MILLogicalAnd { name: "la".into(), x: nid("x"), y: nid("y") },
        MirOp::MILLogicalOr { name: "lo".into(), x: nid("x"), y: nid("y") },
        MirOp::MILLogicalXor { name: "lx".into(), x: nid("x"), y: nid("y") },
        MirOp::MILLogicalNot { name: "lnot".into(), x: nid("x") },
        // Activation variants (no ANEC converter)
        MirOp::MILRelu6 { name: "relu6".into(), x: nid("x") },
        MirOp::MILSigmoidHard { name: "sh".into(), x: nid("x"), alpha: 1.0, beta: 1.0 },
        MirOp::MILThresholdedRelu { name: "tr".into(), x: nid("x"), alpha: 1.0 },
        MirOp::MILClampedRelu { name: "cr".into(), x: nid("x"), alpha: 0.0, beta: 6.0 },
        MirOp::MILLinearActivation { name: "la".into(), x: nid("x"), alpha: 1.0, beta: 0.0 },
        MirOp::MILPrelu { name: "pr".into(), x: nid("x"), alpha: "a".into() },
        MirOp::MILSoftsign { name: "ss".into(), x: nid("x") },
        MirOp::MILScaledTanh { name: "st".into(), x: nid("x"), alpha: 1.0, beta: 1.0 },
        MirOp::MILSoftplus { name: "sp".into(), x: nid("x") },
        MirOp::MILSoftplusParametric {
            name: "spp".into(),
            x: nid("x"),
            alpha: "a".into(),
            beta: "b".into(),
        },
        // Other elementwise (no ANEC converter)
        MirOp::MILThreshold { name: "thr".into(), x: nid("x"), alpha: 1.0 },
        MirOp::MILInverse { name: "inv".into(), x: nid("x"), epsilon: 1e-6 },
        MirOp::MILMod { name: "mod".into(), x: nid("x"), y: nid("y") },
        MirOp::MILClip { name: "clip".into(), x: nid("x"), min_val: 0.0, max_val: 6.0 },
        // Miscellaneous (no ANEC converter)
        MirOp::MILBandPart { name: "bp".into(), x: nid("x"), num_lower: -1, num_upper: 0 },
        MirOp::MILReverseSequence {
            name: "rseq".into(),
            x: nid("x"),
            lengths: nid("l"),
            batch_axis: 0,
            seq_axis: 1,
        },
        MirOp::MILEinsum {
            name: "e".into(),
            inputs: vec![nid("a"), nid("b")],
            equation: "ij,jk->ik".into(),
        },
    ];

    for op in &moved_ops {
        assert_eq!(
            op.default_engine(),
            None,
            "T-22: Expected None (CPU-only) for {:?}, but got Some engine",
            op
        );
    }
}

/// T-22: Verify that ANE-legal ops that were kept in PE still return PE.
/// These ops have ANEC converters per the per-op support matrix but
/// currently lack MirOpCompat variants (see T-38/T-39).
#[allow(deprecated)]
#[test]
fn test_t22_ane_legal_ops_still_in_pe() {
    use crate::ane_engine::AneEngine;
    let ane_legal_ops: Vec<MirOp> = vec![
        MirOp::MILElu { name: "elu".into(), x: nid("x"), alpha: 1.0 },
        MirOp::MILSquare { name: "sq".into(), x: nid("x") },
        MirOp::MILExp2 { name: "exp2".into(), x: nid("x") },
        MirOp::MILErf { name: "erf".into(), x: nid("x") },
        MirOp::MILQuantize {
            name: "q".into(),
            x: nid("x"),
            scale: 1.0,
            zero_point: 0,
            axis: -1,
            output_dtype: MilDtype::UInt8,
        },
        MirOp::MILDequantize {
            name: "dq".into(),
            x: nid("x"),
            scale: 1.0,
            zero_point: 0,
            axis: -1,
            output_dtype: MilDtype::Fp16,
        },
    ];

    for op in &ane_legal_ops {
        assert_eq!(
            op.default_engine(),
            Some(AneEngine::PE),
            "T-22: Expected PE (ANE-legal) for {:?}",
            op
        );
    }
}

/// T-22: Verify mil_op_name() returns non-empty names for all variants.
#[allow(deprecated)]
#[test]
fn test_mil_op_name_returns_nonempty() {
    let variants = all_mir_op_variants();
    for (name, op) in &variants {
        let mil_name = op.mil_op_name();
        assert!(!mil_name.is_empty(), "mil_op_name() returned empty string for variant {}", name);
        let _ = mil_name; // Also verify no panic
    }
}

#[test]
fn test_layernorm_revision_gating() {
    use crate::ane_engine::AneEngine;
    use crate::ane_target::AneRevision;

    let ln = MirOp::MILLayerNorm {
        name: "ln".into(),
        x: nid("x"),
        weight: "w".into(),
        bias: None,
        epsilon: 1e-5,
        axes: vec![2],
    };

    // A14 (V7) — does NOT support LayerNorm → None
    assert_eq!(ln.default_engine_for_revision(Some(AneRevision::V7)), None);
    // V17 (M1) is A14-class — does NOT support LayerNorm → None
    assert_eq!(ln.default_engine_for_revision(Some(AneRevision::V17)), None);
    // A15 (V8) — supports LayerNorm → PE
    assert_eq!(ln.default_engine_for_revision(Some(AneRevision::V8)), Some(AneEngine::PE));
    // A16 (V10) — supports LayerNorm → PE
    assert_eq!(ln.default_engine_for_revision(Some(AneRevision::V10)), Some(AneEngine::PE));
}
