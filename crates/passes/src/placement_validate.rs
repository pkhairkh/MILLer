//! ANE Placement Validator — deterministic hard-constraint checker.
//!
//! Runs before the soft scoring pass to reject ops that can never run
//! on the ANE, regardless of heuristics. This is a compile-time guard
//! that mirrors the hard constraints documented in:
//! - ane-constraints-docs/02-hardware-and-limits/
//! - ane-constraints-docs/03-placement-and-compiler/mil-to-ane-placement-constraint-system.md

use ane_ir::ane_target::AneFamily;
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

/// Validate whether a MIR op can be placed on the ANE given its tensor
/// shapes, the target family, and whether dynamic shapes are present.
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

        // Broadcast on A11/A12 is FP16-only
        MirOp::MILAdd { .. }
        | MirOp::MILMul { .. }
        | MirOp::MILSub { .. }
        | MirOp::MILMaximum { .. }
        | MirOp::MILMinimum { .. } => {
            // This is a soft constraint for now — the validator doesn't
            // know the dtype of the broadcast operand. We just flag it.
            PlacementDecision::AneAllowed
        }

        // ConvTranspose: always NE
        MirOp::MILConvTranspose { .. } => PlacementDecision::AneAllowed,

        // Ops with no ANE engine — immediate CPU
        op => {
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
}
