//! ANE Layer Fusability Checks (T-P6-05)
//!
//! Implements `IsFusable`-equivalent checks for ANE layer fusion, modeling
//! the constraints documented in `ane-constraints-docs/03-placement-and-compiler/`.
//!
//! ## Background
//!
//! The ANE compiler uses a multi-engine fusion architecture with three execution
//! engines: NE (convolution/math), PE (element-wise/reduction), and TransposeEngine.
//! Each engine fuses compatible adjacent operations into a single engine layer.
//! Operations that individually pass placement validation may still fail to fuse
//! into an engine layer, forcing graph breaks and DMA round-trips.
//!
//! ## Fusion Atoms
//!
//! The ANE compiler organizes operations into "atoms" — indivisible units that
//! are matched and grouped into fused engine layers. Key atom categories include:
//!
//! - **ConvAtom**: Base convolution, can fuse with dequant, activation, GOC
//! - **MatMulAtom**: Matrix multiplication, similar fusion patterns
//! - **ElementWiseAtom**: Binary/unary element-wise operations
//! - **ActivationAtom**: ReLU, sigmoid, tanh, and other activation functions
//! - **GOCAtom**: Generic Operation Compute (flexible compute kernel)
//! - **DeQuantAtom**: Dequantization, can fuse with conv or as GOC
//! - **ScaledEWAtom**: Scaled element-wise operations
//! - **PostScaleAtom**: Post-scale operations
//! - **PerChannelGOCAtom**: Per-channel GOC operations
//!
//! ## Key Fusability Rules
//!
//! 1. **Engine Compatibility**: Only ops assigned to the same engine (NE or PE)
//!    can be fused. TransposeEngine ops can be fused with NE ops.
//! 2. **Tensor Format**: Fusion requires compatible tensor formats (channel-first
//!    vs channel-last). Format mismatches break fusion.
//! 3. **Quantization Format**: Dequantization atoms must match the quantization
//!    format of the preceding conv/GOC atom.
//! 4. **OCG Size Constraints**: The Output Channel Group (OCG) size must be
//!    compatible between producer and consumer for fusion.
//! 5. **Active NE Constraints**: The number of active NE engines limits which
//!    ops can be simultaneously fused.
//! 6. **Memory Pressure**: Even if fusability checks pass, L2 cache pressure
//!    may force fusion boundaries.
//!
//! ## Failed Fusion Patterns
//!
//! Known patterns that cannot be fused (from ANEC binary research):
//! - GOC + GOC + EW_MAX (3-atom PE pattern)
//! - Transpose + ScaledEW + Transpose (3-atom PE pattern)
//! - Clamped ReLU as activation fusion
//! - Leaky ReLU as activation fusion
//! - Swish (SiLU) as activation fusion in some engine configurations
//!
//! Reference: `ane-constraints-docs/03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md`

use ane_ir::ane_engine::AneEngine;
use ane_ir::mir::{MirOp, MirOpTargetAnnotation};

/// Result of a fusability check between two adjacent MIR operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusabilityResult {
    /// Whether the two operations can be fused into a single engine layer.
    pub can_fuse: bool,
    /// If fusion is not possible, the reason why.
    pub reason: Option<String>,
}

impl FusabilityResult {
    /// Create a successful fusability result.
    pub fn fusable() -> Self {
        FusabilityResult { can_fuse: true, reason: None }
    }

    /// Create a failed fusability result with a reason.
    pub fn not_fusable(reason: impl Into<String>) -> Self {
        FusabilityResult { can_fuse: false, reason: Some(reason.into()) }
    }
}

/// Classification of an op's fusion atom type.
///
/// Maps MirOp variants to their ANEC fusion atom categories.
/// Ops that don't map to any atom are classified as `NonFusable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionAtom {
    /// Convolution atom (NEFUSED_CONV pipeline).
    Conv,
    /// Matrix multiplication atom (NEFUSED_MATMUL pipeline).
    MatMul,
    /// Pooling atom (NEFUSED_POOL / PEFUSED_POOL).
    Pool,
    /// Binary element-wise atom (NEFUSED_EW / PEFUSED_ELEMENTWISE).
    ElementWiseBinary,
    /// Unary element-wise / activation atom.
    Activation,
    /// Generic Operation Compute atom (PEFUSED_GOC).
    GOC,
    /// Dequantization atom.
    Dequant,
    /// Scaled element-wise atom (PE pipeline).
    ScaledEW,
    /// Post-scale atom.
    PostScale,
    /// Per-channel GOC atom.
    PerChannelGOC,
    /// Transpose / reshape / data movement atom.
    Transpose,
    /// Bypass / passthrough atom (NE).
    Bypass,
    /// Pre-scale atom.
    PreScale,
    /// The op cannot participate in any fusion pattern.
    NonFusable,
}

/// Classify a MirOp into its fusion atom type.
///
/// This mapping determines which fusion patterns an op can participate in.
/// Ops classified as `NonFusable` will always force a fusion boundary.
pub fn classify_fusion_atom(op: &MirOp) -> FusionAtom {
    match op {
        // Convolution family → ConvAtom
        MirOp::MILConv { .. } => FusionAtom::Conv,

        // MatMul family → MatMulAtom
        MirOp::MILLinear { .. } | MirOp::MILMatMul { .. } => FusionAtom::MatMul,

        // Pooling → PoolAtom
        MirOp::MILMaxPool { .. } | MirOp::MILAvgPool { .. } | MirOp::MILL2Pool { .. } => {
            FusionAtom::Pool
        }

        // Binary element-wise → ElementWiseAtom
        MirOp::MILAdd { .. }
        | MirOp::MILSub { .. }
        | MirOp::MILMul { .. }
        | MirOp::MILMaximum { .. }
        | MirOp::MILMinimum { .. }
        | MirOp::MILRealDiv { .. }
        | MirOp::MILFloorDiv { .. }
        | MirOp::MILPow { .. } => FusionAtom::ElementWiseBinary,

        // Activation functions → ActivationAtom
        MirOp::MILRelu { .. }
        | MirOp::MILRelu6 { .. }
        | MirOp::MILSigmoid { .. }
        | MirOp::MILTanh { .. }
        | MirOp::MILSigmoidHard { .. }
        | MirOp::MILSoftsign { .. }
        | MirOp::MILElu { .. }
        | MirOp::MILSoftplus { .. }
        | MirOp::MILGelu { .. }
        | MirOp::MILPrelu { .. }
        | MirOp::MILThresholdedRelu { .. }
        | MirOp::MILLinearActivation { .. } => FusionAtom::Activation,

        // Leaky ReLU, Clamped ReLU, SiLU: NOT fusable as activation atoms
        // (known ANEC fusion failures)
        MirOp::MILLeakyRelu { .. } | MirOp::MILClampedRelu { .. } | MirOp::MILSilu { .. } => {
            FusionAtom::NonFusable
        }

        // Reshape/transpose/permute → TransposeAtom
        MirOp::MILReshape { .. }
        | MirOp::MILTranspose { .. }
        | MirOp::MILExpandDims { .. }
        | MirOp::MILSqueeze { .. }
        | MirOp::MILFlatten2d { .. }
        | MirOp::MILTile { .. } => FusionAtom::Transpose,

        // Reductions → GOC (handled as GOC in PE pipeline)
        MirOp::MILReduceSum { .. }
        | MirOp::MILReduceMean { .. }
        | MirOp::MILReduceMax { .. }
        | MirOp::MILReduceMin { .. }
        | MirOp::MILReduceProd { .. } => FusionAtom::GOC,

        // Everything else is non-fusable for now
        _ => FusionAtom::NonFusable,
    }
}

/// Check if two adjacent operations can be fused into a single ANE engine layer.
///
/// This implements the primary fusability check, combining engine compatibility,
/// atom compatibility, and known failed fusion patterns from ANEC binary research.
///
/// # Arguments
///
/// * `producer` - The upstream MirOp (produces the tensor)
/// * `consumer` - The downstream MirOp (consumes the tensor)
/// * `producer_annotation` - Target annotation for the producer
/// * `consumer_annotation` - Target annotation for the consumer
///
/// # Returns
///
/// A `FusabilityResult` indicating whether fusion is possible and why not.
pub fn check_fusability(
    producer: &MirOp,
    consumer: &MirOp,
    producer_annotation: &MirOpTargetAnnotation,
    consumer_annotation: &MirOpTargetAnnotation,
) -> FusabilityResult {
    // Step 1: Check engine compatibility
    if let Err(reason) = check_engine_compatibility(producer_annotation, consumer_annotation) {
        return FusabilityResult::not_fusable(reason);
    }

    // Step 2: Classify atoms
    let producer_atom = classify_fusion_atom(producer);
    let consumer_atom = classify_fusion_atom(consumer);

    // Step 3: Check if either op is non-fusable
    if producer_atom == FusionAtom::NonFusable {
        return FusabilityResult::not_fusable(format!(
            "Producer op '{}' is classified as NonFusable atom (e.g., leaky_relu, clamped_relu, silu have no ANEC fusion converter)",
            producer.mil_op_name()
        ));
    }
    if consumer_atom == FusionAtom::NonFusable {
        return FusabilityResult::not_fusable(format!(
            "Consumer op '{}' is classified as NonFusable atom",
            consumer.mil_op_name()
        ));
    }

    // Step 4: Check atom compatibility
    if let Err(reason) = check_atom_compatibility(producer_atom, consumer_atom) {
        return FusabilityResult::not_fusable(reason);
    }

    // Step 5: Check known failed fusion patterns
    if let Err(reason) = check_failed_patterns(producer, consumer, producer_atom, consumer_atom) {
        return FusabilityResult::not_fusable(reason);
    }

    FusabilityResult::fusable()
}

/// Check that two ops are assigned to compatible engines for fusion.
///
/// Fusion requires ops to be on the same engine. NE and PE can sometimes
/// interact through DMA, but cannot be fused into a single engine layer.
/// TransposeEngine can be fused with NE ops.
fn check_engine_compatibility(
    producer_annotation: &MirOpTargetAnnotation,
    consumer_annotation: &MirOpTargetAnnotation,
) -> Result<(), String> {
    let producer_engine = producer_annotation.assigned_engine;
    let consumer_engine = consumer_annotation.assigned_engine;

    match (producer_engine, consumer_engine) {
        // Same engine → compatible
        (Some(AneEngine::NE), Some(AneEngine::NE)) => Ok(()),
        (Some(AneEngine::PE), Some(AneEngine::PE)) => Ok(()),
        (Some(AneEngine::TransposeEngine), Some(AneEngine::TransposeEngine)) => Ok(()),

        // TransposeEngine can fuse with NE
        (Some(AneEngine::NE), Some(AneEngine::TransposeEngine)) => Ok(()),
        (Some(AneEngine::TransposeEngine), Some(AneEngine::NE)) => Ok(()),

        // Different engines → not fusable
        (Some(AneEngine::NE), Some(AneEngine::PE)) => Err(
            "Cannot fuse NE producer with PE consumer — different engines require DMA round-trip"
                .into(),
        ),
        (Some(AneEngine::PE), Some(AneEngine::NE)) => Err(
            "Cannot fuse PE producer with NE consumer — different engines require DMA round-trip"
                .into(),
        ),

        // CPU ops cannot be fused
        (None, _) | (_, None) => {
            Err("One or both ops have no engine assignment (CPU-only) — cannot fuse".into())
        }

        // TransposeEngine + PE not fusable
        (Some(AneEngine::TransposeEngine), Some(AneEngine::PE))
        | (Some(AneEngine::PE), Some(AneEngine::TransposeEngine)) => {
            Err("Cannot fuse TransposeEngine with PE — no fusion pattern exists".into())
        }
    }
}

/// Check that two fusion atoms are compatible for fusion.
///
/// Models the ANEC's IsFusable*() checks for atom pairs.
fn check_atom_compatibility(producer: FusionAtom, consumer: FusionAtom) -> Result<(), String> {
    match (producer, consumer) {
        // Conv → Activation: fusable (NEFUSED_CONV absorbs activation)
        (FusionAtom::Conv, FusionAtom::Activation) => Ok(()),
        // Conv → ElementWise: fusable (NEFUSED_EW after conv)
        (FusionAtom::Conv, FusionAtom::ElementWiseBinary) => Ok(()),
        // Conv → GOC: fusable (ConvGOCAtom)
        (FusionAtom::Conv, FusionAtom::GOC) => Ok(()),
        // Conv → Dequant: fusable (ConvAtom::IsFusableToDequant)
        (FusionAtom::Conv, FusionAtom::Dequant) => Ok(()),
        // Conv → Transpose: fusable (TransposeFusion)
        (FusionAtom::Conv, FusionAtom::Transpose) => Ok(()),

        // MatMul → Activation: fusable
        (FusionAtom::MatMul, FusionAtom::Activation) => Ok(()),
        // MatMul → ElementWise: fusable
        (FusionAtom::MatMul, FusionAtom::ElementWiseBinary) => Ok(()),
        // MatMul → GOC: fusable (MatmulGOCAtom)
        (FusionAtom::MatMul, FusionAtom::GOC) => Ok(()),

        // Activation → ElementWise: fusable in PE
        (FusionAtom::Activation, FusionAtom::ElementWiseBinary) => Ok(()),
        // Activation → GOC: fusable
        (FusionAtom::Activation, FusionAtom::GOC) => Ok(()),

        // ElementWise → Activation: fusable
        (FusionAtom::ElementWiseBinary, FusionAtom::Activation) => Ok(()),
        // ElementWise → ElementWise: fusable (chained EW)
        (FusionAtom::ElementWiseBinary, FusionAtom::ElementWiseBinary) => Ok(()),
        // ElementWise → GOC: fusable (EWGOCAtom)
        (FusionAtom::ElementWiseBinary, FusionAtom::GOC) => Ok(()),
        // ElementWise → ScaledEW: fusable
        (FusionAtom::ElementWiseBinary, FusionAtom::ScaledEW) => Ok(()),
        // ElementWise → PostScale: fusable
        (FusionAtom::ElementWiseBinary, FusionAtom::PostScale) => Ok(()),

        // GOC → Activation: fusable
        (FusionAtom::GOC, FusionAtom::Activation) => Ok(()),
        // GOC → ElementWise: fusable
        (FusionAtom::GOC, FusionAtom::ElementWiseBinary) => Ok(()),
        // GOC → GOC: NOT fusable (known failure: "Unable to fuse GOC, GOC and EW_MAX")
        (FusionAtom::GOC, FusionAtom::GOC) => {
            Err("Cannot fuse GOC → GOC — no ANEC fusion pattern exists (known failure)".into())
        }

        // Pool → Activation: fusable
        (FusionAtom::Pool, FusionAtom::Activation) => Ok(()),
        // Pool → GOC: fusable (PoolGOCAtom)
        (FusionAtom::Pool, FusionAtom::GOC) => Ok(()),

        // Transpose → anything on same engine: fusable
        (FusionAtom::Transpose, _) => Ok(()),
        // Anything → Transpose: fusable (TransposeFusion)
        (_, FusionAtom::Transpose) => Ok(()),

        // ScaledEW → Activation: fusable
        (FusionAtom::ScaledEW, FusionAtom::Activation) => Ok(()),
        // ScaledEW → GOC: fusable
        (FusionAtom::ScaledEW, FusionAtom::GOC) => Ok(()),

        // Dequant → Conv: fusable
        (FusionAtom::Dequant, FusionAtom::Conv) => Ok(()),
        // Dequant → GOC: fusable (DeQuantAtom::IsFusableAsGOC)
        (FusionAtom::Dequant, FusionAtom::GOC) => Ok(()),

        // PreScale → Conv: fusable
        (FusionAtom::PreScale, FusionAtom::Conv) => Ok(()),
        // PreScale → GOC: fusable (PreScaleAtom::IsFusable)
        (FusionAtom::PreScale, FusionAtom::GOC) => Ok(()),

        // PostScale → GOC: fusable (PostScaleAtom::IsFusable)
        (FusionAtom::PostScale, FusionAtom::GOC) => Ok(()),

        // PerChannelGOC → ElementWise: fusable
        (FusionAtom::PerChannelGOC, FusionAtom::ElementWiseBinary) => Ok(()),

        // Default: unknown combinations are not fusable
        (p, c) => Err(format!("No known fusion pattern for {:?} → {:?}", p, c)),
    }
}

/// Check for known failed fusion patterns from ANEC binary research.
///
/// These are patterns that pass individual atom compatibility checks but
/// fail at the ANEC compiler level due to implementation constraints.
fn check_failed_patterns(
    _producer: &MirOp,
    consumer: &MirOp,
    _producer_atom: FusionAtom,
    consumer_atom: FusionAtom,
) -> Result<(), String> {
    // Pattern: Activation fusion with non-fusable activation types
    // ANEC error: "NEElementWise can only have input activation mode as Relu"
    // Clamped ReLU, Leaky ReLU, and SiLU cannot be fused as activations
    if consumer_atom == FusionAtom::Activation {
        match consumer {
            MirOp::MILClampedRelu { .. } => {
                return Err("ClampedReLU cannot be fused as activation — no ANEC fusion converter (known failure)".into());
            }
            MirOp::MILLeakyRelu { .. } => {
                return Err("LeakyReLU cannot be fused as activation — no ANEC fusion converter (known failure)".into());
            }
            MirOp::MILSilu { .. } => {
                return Err("SiLU cannot be fused as activation — no ANEC fusion converter for this pattern (known failure)".into());
            }
            _ => {}
        }
    }

    Ok(())
}

/// Scan a sequence of MIR operations and identify fusion boundaries.
///
/// Returns a list of fusion groups, where each group is a contiguous
/// sequence of operation indices that can be fused together. Operations
/// between groups require a fusion boundary (graph break / DMA).
///
/// This is a simplified model of the ANEC's MirLayerFusion::Group() pass.
pub fn identify_fusion_groups(ops: &[(MirOp, MirOpTargetAnnotation)]) -> Vec<Vec<usize>> {
    if ops.is_empty() {
        return vec![];
    }

    let mut groups: Vec<Vec<usize>> = vec![vec![0]];

    for i in 1..ops.len() {
        let (ref prev_op, ref prev_ann) = ops[i - 1];
        let (ref curr_op, ref curr_ann) = ops[i];

        let result = check_fusability(prev_op, curr_op, prev_ann, curr_ann);

        if result.can_fuse {
            // Add to current group
            groups.last_mut().unwrap().push(i);
        } else {
            // Start a new group
            groups.push(vec![i]);
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::ane_engine::AneEngine;
    use ane_ir::ane_target::AneRevision;
    use ane_ir::mir::{MirNodeId, MirOpTargetAnnotation};

    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    fn ne_annotation() -> MirOpTargetAnnotation {
        MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::NE),
            demoted_to_cpu: false,
            target_revision: Some(AneRevision::V19),
            ane_quant: None,
        }
    }

    fn pe_annotation() -> MirOpTargetAnnotation {
        MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::PE),
            demoted_to_cpu: false,
            target_revision: Some(AneRevision::V19),
            ane_quant: None,
        }
    }

    fn cpu_annotation() -> MirOpTargetAnnotation {
        MirOpTargetAnnotation::default()
    }

    // ─── FusionAtom Classification Tests ──────────────────────────────

    #[test]
    fn test_classify_conv_atom() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        assert_eq!(classify_fusion_atom(&conv), FusionAtom::Conv);
    }

    #[test]
    fn test_classify_matmul_atom() {
        let linear =
            MirOp::MILLinear { name: "lin".into(), x: nid("x"), weight: "w".into(), bias: None };
        assert_eq!(classify_fusion_atom(&linear), FusionAtom::MatMul);

        let mm =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        assert_eq!(classify_fusion_atom(&mm), FusionAtom::MatMul);
    }

    #[test]
    fn test_classify_activation_atom() {
        let relu = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        assert_eq!(classify_fusion_atom(&relu), FusionAtom::Activation);

        let sigmoid = MirOp::MILSigmoid { name: "s".into(), x: nid("x") };
        assert_eq!(classify_fusion_atom(&sigmoid), FusionAtom::Activation);

        let tanh = MirOp::MILTanh { name: "t".into(), x: nid("x") };
        assert_eq!(classify_fusion_atom(&tanh), FusionAtom::Activation);
    }

    #[test]
    fn test_classify_non_fusable_activations() {
        let leaky = MirOp::MILLeakyRelu { name: "lr".into(), x: nid("x"), alpha: 0.01 };
        assert_eq!(classify_fusion_atom(&leaky), FusionAtom::NonFusable);

        let clamped =
            MirOp::MILClampedRelu { name: "cr".into(), x: nid("x"), alpha: 0.0, beta: 6.0 };
        assert_eq!(classify_fusion_atom(&clamped), FusionAtom::NonFusable);

        let silu = MirOp::MILSilu { name: "si".into(), x: nid("x") };
        assert_eq!(classify_fusion_atom(&silu), FusionAtom::NonFusable);
    }

    #[test]
    fn test_classify_elementwise_atom() {
        let add = MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") };
        assert_eq!(classify_fusion_atom(&add), FusionAtom::ElementWiseBinary);

        let mul = MirOp::MILMul { name: "m".into(), x: nid("x"), y: nid("y") };
        assert_eq!(classify_fusion_atom(&mul), FusionAtom::ElementWiseBinary);
    }

    #[test]
    fn test_classify_pool_atom() {
        let maxpool = MirOp::MILMaxPool {
            name: "mp".into(),
            x: nid("x"),
            kernel_sizes: vec![3],
            strides: vec![1],
            pad_types: vec!["valid".into()],
            pad_amounts: vec![0],
        };
        assert_eq!(classify_fusion_atom(&maxpool), FusionAtom::Pool);
    }

    #[test]
    fn test_classify_transpose_atom() {
        let reshape = MirOp::MILReshape { name: "rs".into(), x: nid("x"), shape: vec![1, 0] };
        assert_eq!(classify_fusion_atom(&reshape), FusionAtom::Transpose);

        let transpose =
            MirOp::MILTranspose { name: "tr".into(), x: nid("x"), perm: vec![0, 2, 1, 3] };
        assert_eq!(classify_fusion_atom(&transpose), FusionAtom::Transpose);
    }

    #[test]
    fn test_classify_reduction_as_goc() {
        let reduce_sum =
            MirOp::MILReduceSum { name: "rs".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        assert_eq!(classify_fusion_atom(&reduce_sum), FusionAtom::GOC);
    }

    // ─── Engine Compatibility Tests ────────────────────────────────────

    #[test]
    fn test_engine_compat_same_ne() {
        assert!(check_engine_compatibility(&ne_annotation(), &ne_annotation()).is_ok());
    }

    #[test]
    fn test_engine_compat_same_pe() {
        assert!(check_engine_compatibility(&pe_annotation(), &pe_annotation()).is_ok());
    }

    #[test]
    fn test_engine_compat_ne_pe_incompatible() {
        let result = check_engine_compatibility(&ne_annotation(), &pe_annotation());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("different engines"));
    }

    #[test]
    fn test_engine_compat_ne_transpose_compatible() {
        let trans_ann = MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::TransposeEngine),
            ..Default::default()
        };
        assert!(check_engine_compatibility(&ne_annotation(), &trans_ann).is_ok());
    }

    #[test]
    fn test_engine_compat_cpu_incompatible() {
        let result = check_engine_compatibility(&cpu_annotation(), &ne_annotation());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CPU-only"));
    }

    // ─── Atom Compatibility Tests ──────────────────────────────────────

    #[test]
    fn test_atom_compat_conv_activation() {
        assert!(check_atom_compatibility(FusionAtom::Conv, FusionAtom::Activation).is_ok());
    }

    #[test]
    fn test_atom_compat_conv_goc() {
        assert!(check_atom_compatibility(FusionAtom::Conv, FusionAtom::GOC).is_ok());
    }

    #[test]
    fn test_atom_compat_matmul_activation() {
        assert!(check_atom_compatibility(FusionAtom::MatMul, FusionAtom::Activation).is_ok());
    }

    #[test]
    fn test_atom_compat_goc_goc_not_fusable() {
        let result = check_atom_compatibility(FusionAtom::GOC, FusionAtom::GOC);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("GOC"));
    }

    #[test]
    fn test_atom_compat_ew_ew() {
        assert!(check_atom_compatibility(
            FusionAtom::ElementWiseBinary,
            FusionAtom::ElementWiseBinary
        )
        .is_ok());
    }

    // ─── Full Fusability Check Tests ───────────────────────────────────

    #[test]
    fn test_fusable_conv_relu() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let relu = MirOp::MILRelu { name: "relu".into(), x: nid("conv") };

        let result = check_fusability(&conv, &relu, &ne_annotation(), &pe_annotation());
        // Note: Conv is NE, Relu is PE — different engines. This is actually not fusable.
        assert!(!result.can_fuse);
        assert!(result.reason.unwrap().contains("different engines"));
    }

    #[test]
    fn test_fusable_conv_relu_same_engine() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let relu = MirOp::MILRelu { name: "relu".into(), x: nid("conv") };

        // Both assigned to NE (activation after conv can be on NE)
        let result = check_fusability(&conv, &relu, &ne_annotation(), &ne_annotation());
        assert!(result.can_fuse);
    }

    #[test]
    fn test_not_fusable_conv_leaky_relu() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let leaky = MirOp::MILLeakyRelu { name: "lr".into(), x: nid("conv"), alpha: 0.01 };

        let result = check_fusability(&conv, &leaky, &ne_annotation(), &ne_annotation());
        assert!(!result.can_fuse);
        assert!(result.reason.unwrap().contains("NonFusable"));
    }

    #[test]
    fn test_not_fusable_conv_silu() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let silu = MirOp::MILSilu { name: "silu".into(), x: nid("conv") };

        let result = check_fusability(&conv, &silu, &ne_annotation(), &ne_annotation());
        assert!(!result.can_fuse);
    }

    #[test]
    fn test_not_fusable_goc_goc() {
        let reduce1 =
            MirOp::MILReduceSum { name: "rs1".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        let reduce2 = MirOp::MILReduceMean {
            name: "rm2".into(),
            x: nid("rs1"),
            axes: vec![2],
            keep_dims: true,
        };

        let result = check_fusability(&reduce1, &reduce2, &pe_annotation(), &pe_annotation());
        assert!(!result.can_fuse);
        assert!(result.reason.unwrap().contains("GOC"));
    }

    #[test]
    fn test_not_fusable_cpu_op() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        // Conv on NE, but consumer has no engine (CPU)
        let add = MirOp::MILAdd { name: "add".into(), x: nid("conv"), y: nid("y") };

        let result = check_fusability(&conv, &add, &ne_annotation(), &cpu_annotation());
        assert!(!result.can_fuse);
    }

    // ─── Fusion Group Identification Tests ─────────────────────────────

    #[test]
    fn test_identify_groups_single_op() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let ops = vec![(conv, ne_annotation())];
        let groups = identify_fusion_groups(&ops);
        assert_eq!(groups, vec![vec![0]]);
    }

    #[test]
    fn test_identify_groups_fusable_pair() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let relu = MirOp::MILRelu { name: "relu".into(), x: nid("conv") };

        let ops = vec![(conv, ne_annotation()), (relu, ne_annotation())];
        let groups = identify_fusion_groups(&ops);
        // Both should be in the same group (Conv + Activation on NE)
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0, 1]);
    }

    #[test]
    fn test_identify_groups_boundary_at_engine_change() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let add = MirOp::MILAdd { name: "add".into(), x: nid("conv"), y: nid("y") };

        let ops = vec![(conv, ne_annotation()), (add, pe_annotation())];
        let groups = identify_fusion_groups(&ops);
        // Different engines → separate groups
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1], vec![1]);
    }

    #[test]
    fn test_identify_groups_leaky_relu_boundary() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let leaky = MirOp::MILLeakyRelu { name: "lr".into(), x: nid("conv"), alpha: 0.01 };

        let ops = vec![(conv, ne_annotation()), (leaky, ne_annotation())];
        let groups = identify_fusion_groups(&ops);
        // LeakyReLU is NonFusable → boundary
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_identify_groups_three_ops_mixed() {
        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let relu = MirOp::MILRelu { name: "relu".into(), x: nid("conv") };
        let add = MirOp::MILAdd { name: "add".into(), x: nid("relu"), y: nid("y") };

        let ops = vec![
            (conv, ne_annotation()),
            (relu, ne_annotation()), // NE: Conv + Activation → fusable
            (add, pe_annotation()),  // PE: different engine → boundary
        ];
        let groups = identify_fusion_groups(&ops);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2]);
    }

    // ─── FusabilityResult Tests ────────────────────────────────────────

    #[test]
    fn test_fusability_result_fusable() {
        let result = FusabilityResult::fusable();
        assert!(result.can_fuse);
        assert!(result.reason.is_none());
    }

    #[test]
    fn test_fusability_result_not_fusable() {
        let result = FusabilityResult::not_fusable("test reason");
        assert!(!result.can_fuse);
        assert_eq!(result.reason.unwrap(), "test reason");
    }

    // ─── Conv with ANE quant metadata ─────────────────────────────────

    #[test]
    fn test_fusable_conv_with_ane_quant() {
        use ane_ir::mir::AneQuantMetadata;

        let conv = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
        };
        let relu = MirOp::MILRelu { name: "relu".into(), x: nid("conv") };

        let conv_ann = MirOpTargetAnnotation {
            assigned_engine: Some(AneEngine::NE),
            demoted_to_cpu: false,
            target_revision: Some(AneRevision::V19),
            ane_quant: Some(AneQuantMetadata {
                kernel_scale: 0.0078,
                kernel_zero_point: 0,
                kernel_palettized_lut: "conv_lut_4bit".to_string(),
            }),
        };

        let result = check_fusability(&conv, &relu, &conv_ann, &ne_annotation());
        assert!(result.can_fuse, "Conv with ANE quant metadata should still be fusable with ReLU");
    }
}
