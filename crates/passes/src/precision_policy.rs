//! Precision Policy pass.
//!
//! Applies precision annotations to SIR nodes based on
//! knowledge about safe precision boundaries and hazard rules.
//!
//! This is the first pass in the pipeline that materially changes
//! a compilation decision based on stored empirical knowledge.
//! When a precision hazard is known for an operation (e.g., fp16
//! is known to cause quality degradation for certain linear
//! projections), this pass overrides the default dtype to fp32.
//!
//! Without knowledge, all operations use the default fp16 precision.
//! This is a concrete, testable adaptation: the compiler changes
//! its decision because stored knowledge says the default is unsafe.

use crate::knowledge_query::PassKnowledgeQuery;
use ane_ir::sir::SirGraph;
use anyhow::Result;

/// Default precision for operations without specific knowledge.
const DEFAULT_DTYPE: &str = "fp16";

/// Minimum confidence threshold for a precision hazard to trigger
/// a dtype override. Hazards below this confidence are ignored,
/// keeping the default precision.
const HAZARD_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Record of a precision adaptation decision.
///
/// Captures the full provenance of why a dtype was changed,
/// enabling downstream artifacts to report which knowledge
/// entry influenced the decision and why.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrecisionAdaptation {
    /// The node that was adapted.
    pub node_name: String,
    /// The original dtype before adaptation.
    pub original_dtype: String,
    /// The dtype after adaptation.
    pub adapted_dtype: String,
    /// The knowledge source that triggered the adaptation.
    pub source_id: Option<String>,
    /// Confidence of the hazard knowledge.
    pub confidence: f32,
    /// Human-readable reason for the adaptation.
    pub reason: String,
}

/// Precision Policy pass implementation.
///
/// This pass queries the knowledge store for precision hazards
/// and overrides the default fp16 precision when a known hazard
/// with sufficient confidence exists. Without matching knowledge,
/// the pass uses fp16 (the ANE's native precision) throughout.
///
/// This is the first pass that changes a compilation decision
/// because of stored empirical knowledge — the hallmark of
/// "knowledge-affecting" vs "knowledge-aware" behavior.
pub struct PrecisionPolicyPass {
    /// Default dtype to assign when no knowledge is available.
    pub default_dtype: String,
    /// Minimum confidence threshold for a hazard to trigger override.
    pub hazard_confidence_threshold: f32,
    /// Records of all adaptations made during this pass run.
    pub adaptations: Vec<PrecisionAdaptation>,
}

impl Default for PrecisionPolicyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecisionPolicyPass {
    pub fn new() -> Self {
        Self {
            default_dtype: DEFAULT_DTYPE.to_string(),
            hazard_confidence_threshold: HAZARD_CONFIDENCE_THRESHOLD,
            adaptations: Vec::new(),
        }
    }

    /// Create a pass with a custom confidence threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.hazard_confidence_threshold = threshold;
        self
    }

    /// Derive an op pattern string from a SIR node's operation.
    ///
    /// This maps SIR op types to the op pattern strings used in
    /// knowledge store queries. The pattern must match what the
    /// seed/observation entries use.
    ///
    /// Every SirOp variant has a specific pattern. Related ops are
    /// grouped into meaningful categories (Comparison_*, Activation_*,
    /// Reduce*, etc.) so that knowledge entries can target entire
    /// families or individual ops. The catch-all arm produces a
    /// "Misc_{VariantName}" pattern for any future variants that
    /// have not yet been categorized.
    fn op_pattern_for_node(node: &ane_ir::sir::SirNode) -> String {
        match &node.op {
            // ─── Composite / High-Level Semantic Ops ─────────────
            ane_ir::sir::SirOp::LinearProjection { .. } => "LinearProjection".to_string(),
            ane_ir::sir::SirOp::AttentionBlock { .. } => "AttentionBlock".to_string(),
            ane_ir::sir::SirOp::DecodeStep { .. } => "DecodeStep".to_string(),
            ane_ir::sir::SirOp::Sampler { .. } => "Sampler".to_string(),

            // ─── Normalization ───────────────────────────────────
            ane_ir::sir::SirOp::RMSNorm { .. } => "RMSNorm".to_string(),
            ane_ir::sir::SirOp::LayerNorm { .. } => "LayerNorm".to_string(),
            ane_ir::sir::SirOp::BatchNorm { .. } => "BatchNorm".to_string(),
            ane_ir::sir::SirOp::InstanceNorm { .. } => "InstanceNorm".to_string(),
            ane_ir::sir::SirOp::L2Norm { .. } => "L2Norm".to_string(),
            ane_ir::sir::SirOp::LocalResponseNorm { .. } => "LocalResponseNorm".to_string(),

            // ─── Positional Encoding ─────────────────────────────
            ane_ir::sir::SirOp::RoPETransform { .. } => "RoPETransform".to_string(),

            // ─── State / KV-Cache ────────────────────────────────
            ane_ir::sir::SirOp::StateRead { .. } => "StateRead".to_string(),
            ane_ir::sir::SirOp::StateWrite { .. } => "StateWrite".to_string(),

            // ─── Linear / FC ─────────────────────────────────────
            ane_ir::sir::SirOp::MatMul { .. } => "MatMul".to_string(),
            ane_ir::sir::SirOp::Einsum { .. } => "Einsum".to_string(),

            // ─── Convolution ─────────────────────────────────────
            ane_ir::sir::SirOp::Conv { .. } => "Conv".to_string(),
            ane_ir::sir::SirOp::ConvTranspose { .. } => "ConvTranspose".to_string(),

            // ─── Elementwise Binary ──────────────────────────────
            ane_ir::sir::SirOp::Add { .. } => "Add".to_string(),
            ane_ir::sir::SirOp::Mul { .. } => "Mul".to_string(),
            ane_ir::sir::SirOp::Sub { .. } => "Sub".to_string(),
            ane_ir::sir::SirOp::Maximum { .. } => "Maximum".to_string(),
            ane_ir::sir::SirOp::Minimum { .. } => "Minimum".to_string(),
            ane_ir::sir::SirOp::RealDiv { .. } => "RealDiv".to_string(),
            ane_ir::sir::SirOp::FloorDiv { .. } => "FloorDiv".to_string(),
            ane_ir::sir::SirOp::Mod { .. } => "Mod".to_string(),
            ane_ir::sir::SirOp::Pow { .. } => "Pow".to_string(),

            // ─── Elementwise Comparison ──────────────────────────
            ane_ir::sir::SirOp::Equal { .. } => "Comparison_Equal".to_string(),
            ane_ir::sir::SirOp::NotEqual { .. } => "Comparison_NotEqual".to_string(),
            ane_ir::sir::SirOp::Greater { .. } => "Comparison_Greater".to_string(),
            ane_ir::sir::SirOp::GreaterEqual { .. } => "Comparison_GreaterEqual".to_string(),
            ane_ir::sir::SirOp::Less { .. } => "Comparison_Less".to_string(),
            ane_ir::sir::SirOp::LessEqual { .. } => "Comparison_LessEqual".to_string(),

            // ─── Elementwise Logical ─────────────────────────────
            ane_ir::sir::SirOp::LogicalAnd { .. } => "LogicalAnd".to_string(),
            ane_ir::sir::SirOp::LogicalOr { .. } => "LogicalOr".to_string(),
            ane_ir::sir::SirOp::LogicalXor { .. } => "LogicalXor".to_string(),
            ane_ir::sir::SirOp::LogicalNot { .. } => "LogicalNot".to_string(),

            // ─── Elementwise Unary ───────────────────────────────
            ane_ir::sir::SirOp::Abs { .. } => "Abs".to_string(),
            ane_ir::sir::SirOp::Neg { .. } => "Neg".to_string(),
            ane_ir::sir::SirOp::Sigmoid { .. } => "Sigmoid".to_string(),
            ane_ir::sir::SirOp::Tanh { .. } => "Tanh".to_string(),
            ane_ir::sir::SirOp::Relu { .. } => "Relu".to_string(),
            ane_ir::sir::SirOp::Silu { .. } => "Silu".to_string(),
            ane_ir::sir::SirOp::Gelu { .. } => "Gelu".to_string(),
            ane_ir::sir::SirOp::Sqrt { .. } => "Sqrt".to_string(),
            ane_ir::sir::SirOp::Rsqrt { .. } => "Rsqrt".to_string(),
            ane_ir::sir::SirOp::Exp { .. } => "Exp".to_string(),
            ane_ir::sir::SirOp::Log { .. } => "Log".to_string(),
            ane_ir::sir::SirOp::Cast { .. } => "Cast".to_string(),
            ane_ir::sir::SirOp::Clip { .. } => "Clip".to_string(),
            ane_ir::sir::SirOp::Softmax { .. } => "Softmax".to_string(),

            // ─── Activation ──────────────────────────────────────
            ane_ir::sir::SirOp::Relu6 { .. } => "Activation_Relu6".to_string(),
            ane_ir::sir::SirOp::LeakyRelu { .. } => "Activation_LeakyRelu".to_string(),
            ane_ir::sir::SirOp::SigmoidHard { .. } => "Activation_SigmoidHard".to_string(),
            ane_ir::sir::SirOp::ThresholdedRelu { .. } => "Activation_ThresholdedRelu".to_string(),
            ane_ir::sir::SirOp::ClampedRelu { .. } => "Activation_ClampedRelu".to_string(),
            ane_ir::sir::SirOp::LinearActivation { .. } => "Activation_Linear".to_string(),
            ane_ir::sir::SirOp::Prelu { .. } => "Activation_Prelu".to_string(),
            ane_ir::sir::SirOp::Softsign { .. } => "Activation_Softsign".to_string(),
            ane_ir::sir::SirOp::ScaledTanh { .. } => "Activation_ScaledTanh".to_string(),
            ane_ir::sir::SirOp::Elu { .. } => "Activation_Elu".to_string(),
            ane_ir::sir::SirOp::Softplus { .. } => "Activation_Softplus".to_string(),
            ane_ir::sir::SirOp::SoftplusParametric { .. } => "Activation_SoftplusParametric".to_string(),
            ane_ir::sir::SirOp::Square { .. } => "Square".to_string(),
            ane_ir::sir::SirOp::Threshold { .. } => "Threshold".to_string(),

            // ─── Rounding ────────────────────────────────────────
            ane_ir::sir::SirOp::Ceil { .. } => "Ceil".to_string(),
            ane_ir::sir::SirOp::Floor { .. } => "Floor".to_string(),
            ane_ir::sir::SirOp::Round { .. } => "Round".to_string(),

            // ─── Mathematical / Inverse ──────────────────────────
            ane_ir::sir::SirOp::Inverse { .. } => "Inverse".to_string(),
            ane_ir::sir::SirOp::Exp2 { .. } => "Exp2".to_string(),
            ane_ir::sir::SirOp::Sign { .. } => "Sign".to_string(),
            ane_ir::sir::SirOp::Erf { .. } => "Erf".to_string(),

            // ─── Trigonometric ───────────────────────────────────
            ane_ir::sir::SirOp::Cos { .. } => "Trig_Cos".to_string(),
            ane_ir::sir::SirOp::Sin { .. } => "Trig_Sin".to_string(),
            ane_ir::sir::SirOp::Tan { .. } => "Trig_Tan".to_string(),
            ane_ir::sir::SirOp::Acos { .. } => "Trig_Acos".to_string(),
            ane_ir::sir::SirOp::Asin { .. } => "Trig_Asin".to_string(),
            ane_ir::sir::SirOp::Atan { .. } => "Trig_Atan".to_string(),
            ane_ir::sir::SirOp::Cosh { .. } => "Trig_Cosh".to_string(),
            ane_ir::sir::SirOp::Sinh { .. } => "Trig_Sinh".to_string(),
            ane_ir::sir::SirOp::Atanh { .. } => "Trig_Atanh".to_string(),

            // ─── Conditional / Select ────────────────────────────
            ane_ir::sir::SirOp::Select { .. } => "Select".to_string(),
            ane_ir::sir::SirOp::Where { .. } => "Where".to_string(),

            // ─── Reduction ───────────────────────────────────────
            ane_ir::sir::SirOp::ReduceSum { .. } => "ReduceSum".to_string(),
            ane_ir::sir::SirOp::ReduceMean { .. } => "ReduceMean".to_string(),
            ane_ir::sir::SirOp::ReduceMax { .. } => "ReduceMax".to_string(),
            ane_ir::sir::SirOp::ReduceMin { .. } => "ReduceMin".to_string(),
            ane_ir::sir::SirOp::ReduceProd { .. } => "ReduceProd".to_string(),
            ane_ir::sir::SirOp::ReduceSumSquare { .. } => "ReduceSumSquare".to_string(),
            ane_ir::sir::SirOp::ReduceL2Norm { .. } => "ReduceL2Norm".to_string(),
            ane_ir::sir::SirOp::ReduceL1Norm { .. } => "ReduceL1Norm".to_string(),
            ane_ir::sir::SirOp::ReduceLogSumExp { .. } => "ReduceLogSumExp".to_string(),
            ane_ir::sir::SirOp::ReduceLogSum { .. } => "ReduceLogSum".to_string(),
            ane_ir::sir::SirOp::ReduceArgmax { .. } => "ReduceArgmax".to_string(),
            ane_ir::sir::SirOp::ReduceArgmin { .. } => "ReduceArgmin".to_string(),

            // ─── Pooling ─────────────────────────────────────────
            ane_ir::sir::SirOp::MaxPool { .. } => "MaxPool".to_string(),
            ane_ir::sir::SirOp::AvgPool { .. } => "AvgPool".to_string(),
            ane_ir::sir::SirOp::L2Pool { .. } => "L2Pool".to_string(),

            // ─── Image Resizing ──────────────────────────────────
            ane_ir::sir::SirOp::Resize { .. } => "Resize".to_string(),
            ane_ir::sir::SirOp::ResizeNearestNeighbor { .. } => "ResizeNearestNeighbor".to_string(),
            ane_ir::sir::SirOp::ResizeBilinear { .. } => "ResizeBilinear".to_string(),
            ane_ir::sir::SirOp::UpsampleNearestNeighbor { .. } => "UpsampleNearestNeighbor".to_string(),
            ane_ir::sir::SirOp::UpsampleBilinear { .. } => "UpsampleBilinear".to_string(),
            ane_ir::sir::SirOp::CropResize { .. } => "CropResize".to_string(),
            ane_ir::sir::SirOp::Affine { .. } => "Affine".to_string(),
            ane_ir::sir::SirOp::Resample { .. } => "Resample".to_string(),

            // ─── Tensor Transform ────────────────────────────────
            ane_ir::sir::SirOp::Reshape { .. } => "Reshape".to_string(),
            ane_ir::sir::SirOp::ReshapeLike { .. } => "ReshapeLike".to_string(),
            ane_ir::sir::SirOp::Transpose { .. } => "Transpose".to_string(),
            ane_ir::sir::SirOp::Split { .. } => "Split".to_string(),
            ane_ir::sir::SirOp::Concat { .. } => "Concat".to_string(),
            ane_ir::sir::SirOp::ExpandDims { .. } => "ExpandDims".to_string(),
            ane_ir::sir::SirOp::Squeeze { .. } => "Squeeze".to_string(),
            ane_ir::sir::SirOp::Flatten2d { .. } => "Flatten2d".to_string(),
            ane_ir::sir::SirOp::Reverse { .. } => "Reverse".to_string(),
            ane_ir::sir::SirOp::ReverseSequence { .. } => "ReverseSequence".to_string(),
            ane_ir::sir::SirOp::SliceByIndex { .. } => "SliceByIndex".to_string(),
            ane_ir::sir::SirOp::SliceBySize { .. } => "SliceBySize".to_string(),
            ane_ir::sir::SirOp::SlidingWindows { .. } => "SlidingWindows".to_string(),
            ane_ir::sir::SirOp::DepthToSpace { .. } => "DepthToSpace".to_string(),
            ane_ir::sir::SirOp::SpaceToDepth { .. } => "SpaceToDepth".to_string(),
            ane_ir::sir::SirOp::PixelShuffle { .. } => "PixelShuffle".to_string(),
            ane_ir::sir::SirOp::PixelUnshuffle { .. } => "PixelUnshuffle".to_string(),
            ane_ir::sir::SirOp::BatchToSpace { .. } => "BatchToSpace".to_string(),
            ane_ir::sir::SirOp::SpaceToBatch { .. } => "SpaceToBatch".to_string(),
            ane_ir::sir::SirOp::Pad { .. } => "Pad".to_string(),
            ane_ir::sir::SirOp::Stack { .. } => "Stack".to_string(),
            ane_ir::sir::SirOp::Tile { .. } => "Tile".to_string(),
            ane_ir::sir::SirOp::Cumsum { .. } => "Cumsum".to_string(),
            ane_ir::sir::SirOp::Fill { .. } => "Fill".to_string(),
            ane_ir::sir::SirOp::FillLike { .. } => "FillLike".to_string(),
            ane_ir::sir::SirOp::Identity { .. } => "Identity".to_string(),
            ane_ir::sir::SirOp::OneHot { .. } => "OneHot".to_string(),
            ane_ir::sir::SirOp::NonZero { .. } => "NonZero".to_string(),
            ane_ir::sir::SirOp::Argsort { .. } => "Argsort".to_string(),
            ane_ir::sir::SirOp::BandPart { .. } => "BandPart".to_string(),
            ane_ir::sir::SirOp::Range1d { .. } => "Range1d".to_string(),
            ane_ir::sir::SirOp::Shape { .. } => "Shape".to_string(),
            ane_ir::sir::SirOp::Crop { .. } => "Crop".to_string(),

            // ─── Scatter / Gather ────────────────────────────────
            ane_ir::sir::SirOp::Gather { .. } => "Gather".to_string(),
            ane_ir::sir::SirOp::GatherAlongAxis { .. } => "GatherAlongAxis".to_string(),
            ane_ir::sir::SirOp::GatherNd { .. } => "GatherNd".to_string(),
            ane_ir::sir::SirOp::Scatter { .. } => "Scatter".to_string(),
            ane_ir::sir::SirOp::ScatterAlongAxis { .. } => "ScatterAlongAxis".to_string(),
            ane_ir::sir::SirOp::ScatterNd { .. } => "ScatterNd".to_string(),
            ane_ir::sir::SirOp::NonMaximumSuppression { .. } => "NonMaximumSuppression".to_string(),

            // ─── Attention ───────────────────────────────────────
            ane_ir::sir::SirOp::ScaledDotProductAttention { .. } => "ScaledDotProductAttention".to_string(),

            // ─── Quantization ────────────────────────────────────
            ane_ir::sir::SirOp::Quantize { .. } => "Quantize".to_string(),
            ane_ir::sir::SirOp::Dequantize { .. } => "Dequantize".to_string(),

            // ─── Constexpr / Compression ─────────────────────────
            ane_ir::sir::SirOp::ConstexprAffineDequantize { .. } => "Constexpr_AffineDequantize".to_string(),
            ane_ir::sir::SirOp::ConstexprBlockwiseShiftScale { .. } => "Constexpr_BlockwiseShiftScale".to_string(),
            ane_ir::sir::SirOp::ConstexprLutToDense { .. } => "Constexpr_LutToDense".to_string(),
            ane_ir::sir::SirOp::ConstexprSparseToDense { .. } => "Constexpr_SparseToDense".to_string(),
            ane_ir::sir::SirOp::ConstexprCast { .. } => "Constexpr_Cast".to_string(),
            ane_ir::sir::SirOp::ConstexprLutToSparse { .. } => "Constexpr_LutToSparse".to_string(),
            ane_ir::sir::SirOp::ConstexprSparseBlockwiseShiftScale { .. } => {
                "Constexpr_SparseBlockwiseShiftScale".to_string()
            }

            // ─── Constants ───────────────────────────────────────
            ane_ir::sir::SirOp::Const { .. } => "Const".to_string(),

            // ─── Recurrent ───────────────────────────────────────
            ane_ir::sir::SirOp::Rnn { .. } => "Recurrent_Rnn".to_string(),
            ane_ir::sir::SirOp::Gru { .. } => "Recurrent_Gru".to_string(),
            ane_ir::sir::SirOp::Lstm { .. } => "Recurrent_Lstm".to_string(),

            // ─── Control Flow ────────────────────────────────────
            ane_ir::sir::SirOp::Cond { .. } => "ControlFlow_Cond".to_string(),
            ane_ir::sir::SirOp::WhileLoop { .. } => "ControlFlow_WhileLoop".to_string(),
            ane_ir::sir::SirOp::MakeList { .. } => "ControlFlow_MakeList".to_string(),
            ane_ir::sir::SirOp::ListLength { .. } => "ControlFlow_ListLength".to_string(),
            ane_ir::sir::SirOp::ListWrite { .. } => "ControlFlow_ListWrite".to_string(),
            ane_ir::sir::SirOp::ListRead { .. } => "ControlFlow_ListRead".to_string(),
            ane_ir::sir::SirOp::ListGather { .. } => "ControlFlow_ListGather".to_string(),
            ane_ir::sir::SirOp::ListScatter { .. } => "ControlFlow_ListScatter".to_string(),

            // ─── Random ──────────────────────────────────────────
            ane_ir::sir::SirOp::RandomBernoulli { .. } => "Random_Bernoulli".to_string(),
            ane_ir::sir::SirOp::RandomNormal { .. } => "Random_Normal".to_string(),
            ane_ir::sir::SirOp::RandomUniform { .. } => "Random_Uniform".to_string(),
            ane_ir::sir::SirOp::RandomCategorical { .. } => "Random_Categorical".to_string(),

            // ─── Topk / Classify ─────────────────────────────────
            ane_ir::sir::SirOp::Topk { .. } => "Topk".to_string(),
            ane_ir::sir::SirOp::Classify { .. } => "Classify".to_string(),

            // ─── Future / Uncategorized ──────────────────────────
            // This catch-all ensures any newly added SirOp variant
            // gets a "Misc_{VariantName}" pattern instead of bare
            // "Other", so it can be identified and categorized later.
            #[allow(unreachable_patterns)]
            _ => {
                let debug_str = format!("{:?}", node.op);
                let variant_name = debug_str.split('{').next().unwrap_or("Unknown").trim();
                format!("Misc_{}", variant_name)
            }
        }
    }

    /// Run the precision policy pass.
    ///
    /// For each SIR node, queries the knowledge store for precision
    /// hazards. When a hazard is found with confidence above the
    /// threshold, the node's quality_contract is updated to record
    /// the dtype override, and an adaptation record is created.
    ///
    /// The SIR graph itself carries the precision override in the
    /// metadata (quality_contract field), and the adaptation records
    /// are available for downstream artifact generation.
    ///
    /// Without knowledge, all nodes use fp16 and no adaptations
    /// are recorded. This ensures behavior is identical to the
    /// pre-adaptation pass when no knowledge store is available.
    pub fn run(
        &mut self,
        input: SirGraph,
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> Result<SirGraph> {
        // Reset adaptations for this run
        self.adaptations.clear();

        let nodes = input.nodes.into_iter().map(|mut node| {
            let op_pattern = Self::op_pattern_for_node(&node);

            // Query knowledge for precision hazards for this op
            if let Some(hazard) = knowledge_query.query_precision_hazard(
                &op_pattern,
                &self.default_dtype,
                None,
            ) {
                // Only override if confidence exceeds threshold
                if hazard.confidence >= self.hazard_confidence_threshold {
                    // Record the adaptation
                    let adaptation = PrecisionAdaptation {
                        node_name: node.name.clone(),
                        original_dtype: self.default_dtype.clone(),
                        adapted_dtype: hazard.recommended_dtype.clone(),
                        source_id: hazard.source_id.clone(),
                        confidence: hazard.confidence,
                        reason: format!(
                            "Precision hazard: {} at {} is unsafe (confidence={:.2}, evidence={}), overriding to {}",
                            op_pattern,
                            hazard.hazardous_dtype,
                            hazard.confidence,
                            hazard.evidence_count,
                            hazard.recommended_dtype,
                        ),
                    };
                    self.adaptations.push(adaptation);

                    // Record the override in the SIR metadata.
                    // The precision_override field carries the adapted dtype
                    // through the pipeline, allowing downstream passes and
                    // bridge payload generation to use the knowledge-informed
                    // precision instead of the spec default.
                    node.metadata.precision_override = Some(hazard.recommended_dtype.clone());
                }
            }

            node
        }).collect();

        Ok(SirGraph { nodes, inputs: input.inputs, outputs: input.outputs })
    }

    /// Get the adapted dtype for a given op, considering all adaptations.
    ///
    /// Returns the recommended dtype if an adaptation was recorded for
    /// this op, otherwise returns the default dtype.
    pub fn adapted_dtype_for(&self, node_name: &str) -> &str {
        self.adaptations
            .iter()
            .find(|a| a.node_name == node_name)
            .map(|a| a.adapted_dtype.as_str())
            .unwrap_or(&self.default_dtype)
    }

    /// Check whether any adaptations were made.
    pub fn has_adaptations(&self) -> bool {
        !self.adaptations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_query::{
        ComputePlanPlacementInfo, LegalityInfo, NoKnowledge, PrecisionHazardInfo, RiskInfo,
    };
    use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    /// A mock knowledge query that reports a precision hazard for LinearProjection.
    struct MockPrecisionHazardKnowledge;

    impl PassKnowledgeQuery for MockPrecisionHazardKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(
            &self,
            op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            if op_pattern == "LinearProjection" {
                Some(PrecisionHazardInfo {
                    op_pattern: "LinearProjection".to_string(),
                    hazardous_dtype: "fp16".to_string(),
                    recommended_dtype: "fp32".to_string(),
                    confidence: 0.7,
                    evidence_count: 3,
                    source_id: Some("hazard_wq_4bit_deep_layers".to_string()),
                    description: Some("Qwen3 uses 8-bit for W_Q in layers 24-27".to_string()),
                })
            } else {
                None
            }
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// A mock knowledge query that reports a hazard below the confidence threshold.
    struct MockLowConfidenceHazardKnowledge;

    impl PassKnowledgeQuery for MockLowConfidenceHazardKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(
            &self,
            op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            if op_pattern == "LinearProjection" {
                Some(PrecisionHazardInfo {
                    op_pattern: "LinearProjection".to_string(),
                    hazardous_dtype: "fp16".to_string(),
                    recommended_dtype: "fp32".to_string(),
                    confidence: 0.3, // Below threshold
                    evidence_count: 1,
                    source_id: Some("low_confidence_hazard".to_string()),
                    description: Some("Weak evidence of precision issue".to_string()),
                })
            } else {
                None
            }
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// A mock knowledge query that reports no hazards at all.
    struct MockSafeKnowledge;

    impl PassKnowledgeQuery for MockSafeKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    fn make_linear_sir() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("weight".into()),
                    op: SirOp::Mul { x: SirNodeId(String::new()), y: SirNodeId(String::new()) },
                    name: "weight".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("output".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "weight".into(),
                        bias: Some("bias".into()),
                        palette_bits: None,
                    },
                    name: "linear_out".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("output".into())],
        }
    }

    /// Test that precision hazard knowledge changes the pass output.
    ///
    /// This is the core Sprint 16 integration test: it proves that
    /// stored empirical knowledge materially changes a compilation
    /// decision. When a hazard is known for LinearProjection at fp16,
    /// the pass records an adaptation, proving the compiler is not
    /// just "aware" of knowledge but is "affected" by it.
    #[test]
    fn test_precision_hazard_changes_dtype_decision() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        // Run with hazard knowledge
        let hazard_knowledge = MockPrecisionHazardKnowledge;
        let _result = pass.run(sir.clone(), &hazard_knowledge).unwrap();

        // Verify an adaptation was recorded for the linear projection node
        assert!(
            pass.has_adaptations(),
            "Pass must record adaptations when hazard knowledge is present"
        );
        assert_eq!(
            pass.adaptations.len(),
            1,
            "Exactly one adaptation for the LinearProjection node"
        );

        let adaptation = &pass.adaptations[0];
        assert_eq!(adaptation.node_name, "linear_out");
        assert_eq!(adaptation.original_dtype, "fp16");
        assert_eq!(adaptation.adapted_dtype, "fp32");
        assert_eq!(adaptation.source_id, Some("hazard_wq_4bit_deep_layers".to_string()));
        assert!((adaptation.confidence - 0.7).abs() < 0.001);
        assert!(adaptation.reason.contains("LinearProjection"));
        assert!(adaptation.reason.contains("fp32"));

        // Verify adapted_dtype_for returns fp32 for the adapted node
        assert_eq!(pass.adapted_dtype_for("linear_out"), "fp32");
        // And fp16 for the weight node (no hazard)
        assert_eq!(pass.adapted_dtype_for("weight"), "fp16");
    }

    /// Test that NoKnowledge produces no adaptations.
    ///
    /// Without knowledge, the pass must behave identically to the
    /// pre-adaptation version: no dtype overrides, no adaptation records.
    #[test]
    fn test_no_knowledge_no_adaptation() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        let no_knowledge = NoKnowledge;
        let _result = pass.run(sir, &no_knowledge).unwrap();

        assert!(!pass.has_adaptations(), "NoKnowledge must produce zero adaptations");
        assert_eq!(pass.adapted_dtype_for("linear_out"), "fp16");
        assert_eq!(pass.adapted_dtype_for("weight"), "fp16");
    }

    /// Test that low-confidence hazards do not trigger adaptation.
    ///
    /// The confidence threshold prevents weak or speculative knowledge
    /// from overriding the default precision.
    #[test]
    fn test_low_confidence_hazard_no_adaptation() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        let low_conf = MockLowConfidenceHazardKnowledge;
        let _result = pass.run(sir, &low_conf).unwrap();

        assert!(!pass.has_adaptations(), "Low confidence hazard must not trigger adaptation");
        assert_eq!(pass.adapted_dtype_for("linear_out"), "fp16");
    }

    /// Test that safe knowledge (no hazards) produces no adaptations.
    #[test]
    fn test_safe_knowledge_no_adaptation() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        let safe = MockSafeKnowledge;
        let _result = pass.run(sir, &safe).unwrap();

        assert!(!pass.has_adaptations(), "Safe knowledge must produce zero adaptations");
    }

    /// Test that the adaptation record contains the correct source_id.
    ///
    /// This ensures that artifact provenance can trace each adaptation
    /// back to the specific knowledge entry that caused it.
    #[test]
    fn test_adaptation_provenance() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        let hazard = MockPrecisionHazardKnowledge;
        let _result = pass.run(sir, &hazard).unwrap();

        assert!(pass.has_adaptations());
        let adaptation = &pass.adaptations[0];
        assert!(adaptation.source_id.is_some());
        assert_eq!(adaptation.source_id.as_ref().unwrap(), "hazard_wq_4bit_deep_layers");
    }

    /// Test that adaptations are reset between runs.
    #[test]
    fn test_adaptations_reset_between_runs() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new();

        let hazard = MockPrecisionHazardKnowledge;
        let _ = pass.run(sir.clone(), &hazard).unwrap();
        assert!(pass.has_adaptations());

        // Run again with NoKnowledge — adaptations should be reset
        let no_knowledge = NoKnowledge;
        let _ = pass.run(sir, &no_knowledge).unwrap();
        assert!(!pass.has_adaptations(), "Adaptations must be reset between runs");
    }

    /// Test custom confidence threshold.
    #[test]
    fn test_custom_confidence_threshold() {
        let sir = make_linear_sir();
        let mut pass = PrecisionPolicyPass::new().with_threshold(0.9);

        let hazard = MockPrecisionHazardKnowledge; // confidence 0.7
        let _ = pass.run(sir, &hazard).unwrap();

        assert!(
            !pass.has_adaptations(),
            "Hazard below custom threshold must not trigger adaptation"
        );
    }

    /// Test that expanded op patterns cover key SIR op variants (T-89 / CQ-11).
    ///
    /// Previously only 14/167+ SIR ops had specific pattern strings.
    /// This test verifies that the expanded pattern coverage maps
    /// important op categories to meaningful pattern strings rather
    /// than falling through to "Other".
    #[test]
    fn test_expanded_op_patterns_cover_key_categories() {
        fn node_for_op(op: SirOp) -> SirNode {
            SirNode {
                id: SirNodeId("test".into()),
                op,
                name: "test_node".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }
        }

        // Composite ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::DecodeStep {
                token: SirNodeId("t".into()),
                state_map: vec![],
                q_weight: None,
                k_weight: None,
                v_weight: None,
                out_weight: None,
                rope_tables: None,
                position: None,
                q_norm_weight: None,
                k_norm_weight: None,
                norm_epsilon: 1e-6,
                qk_norm_type: "rms".into(),
                mask_ref: None,
            })),
            "DecodeStep"
        );
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Sampler {
                logits: SirNodeId("l".into()),
                temperature: 1.0,
                top_p: 0.9,
                rep_penalty: 1.0,
                min_p: 0.0,
                top_k: 50,
                gumbel_noise: false,
            })),
            "Sampler"
        );

        // Normalization ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LayerNorm {
                input: SirNodeId("x".into()),
                weight: "w".into(),
                bias: None,
                epsilon: 1e-5,
                axes: vec![2],
            })),
            "LayerNorm"
        );
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::BatchNorm {
                input: SirNodeId("x".into()),
                mean: "m".into(),
                variance: "v".into(),
                gamma: None,
                beta: None,
                epsilon: 1e-5,
            })),
            "BatchNorm"
        );

        // Linear/FC ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::MatMul {
                a: SirNodeId("a".into()),
                b: SirNodeId("b".into()),
            })),
            "MatMul"
        );

        // Convolution ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Conv {
                input: SirNodeId("x".into()),
                weight: SirNodeId("w".into()),
                pad_type: "valid".into(),
                groups: 1,
                strides: vec![1],
                pad_amounts: vec![0],
                dilations: vec![1],
            })),
            "Conv"
        );

        // Elementwise unary ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Silu {
                input: SirNodeId("x".into()),
            })),
            "Silu"
        );
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Gelu {
                input: SirNodeId("x".into()),
                mode: "exact".into(),
            })),
            "Gelu"
        );

        // Pooling ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::MaxPool {
                input: SirNodeId("x".into()),
                kernel_sizes: vec![3],
                strides: vec![1],
                pad_types: vec!["valid".into()],
                pad_amounts: vec![0],
            })),
            "MaxPool"
        );

        // Reduction ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ReduceMean {
                input: SirNodeId("x".into()),
                axes: vec![1],
                keep_dims: false,
            })),
            "ReduceMean"
        );

        // Attention
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(
                SirOp::ScaledDotProductAttention {
                    query: SirNodeId("q".into()),
                    key: SirNodeId("k".into()),
                    value: SirNodeId("v".into()),
                    attention_mask: None,
                    scale: None,
                }
            )),
            "ScaledDotProductAttention"
        );

        // Quantization ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Quantize {
                input: SirNodeId("x".into()),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
                output_dtype: ane_ir::mir::MilDtype::Int8,
            })),
            "Quantize"
        );

        // Scatter/Gather ops
        assert_eq!(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Gather {
                input: SirNodeId("x".into()),
                indices: SirNodeId("i".into()),
                axis: 0,
            })),
            "Gather"
        );
    }

    /// Test that at least 30 specific op patterns exist (T-108).
    ///
    /// Verifies that the expanded pattern coverage includes at least
    /// 30 distinct non-"Other" patterns. This ensures meaningful
    /// coverage of the SirOp variant space.
    #[test]
    fn test_at_least_30_specific_op_patterns() {
        fn node_for_op(op: SirOp) -> SirNode {
            SirNode {
                id: SirNodeId("test".into()),
                op,
                name: "test_node".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }
        }

        use std::collections::HashSet;
        let mut specific_patterns: HashSet<String> = HashSet::new();

        // Composite
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LinearProjection {
                input: SirNodeId("i".into()),
                weight: "w".into(),
                bias: None,
                palette_bits: None,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::AttentionBlock {
                q: SirNodeId("q".into()),
                k: SirNodeId("k".into()),
                v: SirNodeId("v".into()),
                mask: None,
                rope: None,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::DecodeStep {
                token: SirNodeId("t".into()),
                state_map: vec![],
                q_weight: None,
                k_weight: None,
                v_weight: None,
                out_weight: None,
                rope_tables: None,
                position: None,
                q_norm_weight: None,
                k_norm_weight: None,
                norm_epsilon: 1e-6,
                qk_norm_type: "rms".into(),
                mask_ref: None,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Sampler {
                logits: SirNodeId("l".into()),
                temperature: 1.0,
                top_p: 0.9,
                rep_penalty: 1.0,
                min_p: 0.0,
                top_k: 50,
                gumbel_noise: false,
            })),
        );

        // Normalization
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::RMSNorm {
                input: SirNodeId("x".into()),
                weight: "w".into(),
                epsilon: 1e-5,
                axes: vec![2],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LayerNorm {
                input: SirNodeId("x".into()),
                weight: "w".into(),
                bias: None,
                epsilon: 1e-5,
                axes: vec![2],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::BatchNorm {
                input: SirNodeId("x".into()),
                mean: "m".into(),
                variance: "v".into(),
                gamma: None,
                beta: None,
                epsilon: 1e-5,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::InstanceNorm {
                input: SirNodeId("x".into()),
                gamma: None,
                beta: None,
                epsilon: 1e-5,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::L2Norm {
                input: SirNodeId("x".into()),
                epsilon: 1e-5,
                axes: vec![1],
            })),
        );

        // Positional Encoding
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::RoPETransform {
                input: SirNodeId("x".into()),
                tables: "t".into(),
            })),
        );

        // State
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::StateRead {
                state_id: "s".into(),
                offset: 0,
                shape: vec![1],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::StateWrite {
                state_id: "s".into(),
                offset: 0,
                value: SirNodeId("v".into()),
            })),
        );

        // Linear/FC
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::MatMul {
                a: SirNodeId("a".into()),
                b: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Einsum {
                inputs: vec![SirNodeId("a".into())],
                equation: "ij->j".into(),
            })),
        );

        // Convolution
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Conv {
                input: SirNodeId("x".into()),
                weight: SirNodeId("w".into()),
                pad_type: "valid".into(),
                groups: 1,
                strides: vec![1],
                pad_amounts: vec![0],
                dilations: vec![1],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ConvTranspose {
                input: SirNodeId("x".into()),
                weight: SirNodeId("w".into()),
                pad_type: "valid".into(),
                groups: 1,
                strides: vec![1],
                pad_amounts: vec![0],
                dilations: vec![1],
                output_shape: vec![1],
            })),
        );

        // Elementwise Binary
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Add {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Mul {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::RealDiv {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Pow {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );

        // Comparison
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Equal {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Greater {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LessEqual {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );

        // Logical
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LogicalAnd {
                x: SirNodeId("a".into()),
                y: SirNodeId("b".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LogicalNot {
                input: SirNodeId("x".into()),
            })),
        );

        // Activation
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::LeakyRelu {
                input: SirNodeId("x".into()),
                alpha: 0.01,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Elu {
                input: SirNodeId("x".into()),
                alpha: 1.0,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Softplus {
                input: SirNodeId("x".into()),
            })),
        );

        // Reduction
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ReduceProd {
                input: SirNodeId("x".into()),
                axes: vec![1],
                keep_dims: false,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ReduceL2Norm {
                input: SirNodeId("x".into()),
                axes: vec![1],
                keep_dims: false,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ReduceArgmax {
                input: SirNodeId("x".into()),
                axis: 1,
                keep_dims: false,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ReduceLogSumExp {
                input: SirNodeId("x".into()),
                axes: vec![1],
                keep_dims: false,
            })),
        );

        // Tensor Transform
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ExpandDims {
                input: SirNodeId("x".into()),
                axis: vec![1],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Squeeze {
                input: SirNodeId("x".into()),
                axis: vec![1],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Stack {
                values: vec![SirNodeId("a".into())],
                axis: 0,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Tile {
                input: SirNodeId("x".into()),
                reps: vec![1],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::SliceByIndex {
                input: SirNodeId("x".into()),
                begin: vec![0],
                end: vec![1],
                stride: vec![1],
                begin_mask: vec![false],
                end_mask: vec![false],
                squeeze_mask: vec![false],
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Flatten2d {
                input: SirNodeId("x".into()),
                axis: 1,
            })),
        );

        // Trigonometric
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Cos {
                input: SirNodeId("x".into()),
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Sin {
                input: SirNodeId("x".into()),
            })),
        );

        // Scatter/Gather
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ScatterNd {
                input: SirNodeId("x".into()),
                indices: SirNodeId("i".into()),
                updates: SirNodeId("u".into()),
            })),
        );

        // Recurrent
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Lstm {
                input: SirNodeId("x".into()),
                initial_h: SirNodeId("h".into()),
                initial_c: SirNodeId("c".into()),
                weight_ih: "wih".into(),
                weight_hh: "whh".into(),
                bias: None,
                output_sequence: false,
            })),
        );

        // Control Flow
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Cond {
                pred: SirNodeId("p".into()),
                true_graph: "t".into(),
                false_graph: "f".into(),
            })),
        );

        // Random
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::RandomNormal {
                shape: vec![1],
                mean: 0.0,
                stddev: 1.0,
                seed: None,
                dtype: ane_ir::mir::MilDtype::Fp16,
            })),
        );

        // Quantization
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Quantize {
                input: SirNodeId("x".into()),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
                output_dtype: ane_ir::mir::MilDtype::Int8,
            })),
        );
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::Dequantize {
                input: SirNodeId("x".into()),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
                output_dtype: ane_ir::mir::MilDtype::Fp16,
            })),
        );

        // Constexpr
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(SirOp::ConstexprAffineDequantize {
                quantized_data: "q".into(),
                scale: 1.0,
                zero_point: 0,
                axis: -1,
            })),
        );

        // Attention
        specific_patterns.insert(
            PrecisionPolicyPass::op_pattern_for_node(&node_for_op(
                SirOp::ScaledDotProductAttention {
                    query: SirNodeId("q".into()),
                    key: SirNodeId("k".into()),
                    value: SirNodeId("v".into()),
                    attention_mask: None,
                    scale: None,
                },
            )),
        );

        // None of the patterns should be bare "Other"
        for pattern in &specific_patterns {
            assert_ne!(
                pattern, "Other",
                "No op pattern should be bare 'Other'"
            );
        }

        // Must have at least 30 distinct specific patterns
        assert!(
            specific_patterns.len() >= 30,
            "Expected at least 30 specific op patterns, got {}",
            specific_patterns.len()
        );
    }

    /// Test that NO SirOp variant maps to bare "Other" (T-108).
    ///
    /// Every variant should produce a specific pattern or at least a
    /// "Misc_*" pattern. The bare "Other" catch-all has been replaced
    /// with a dynamic "Misc_{VariantName}" pattern so that uncategorized
    /// ops can be identified.
    #[test]
    fn test_no_sir_op_maps_to_bare_other() {
        fn node_for_op(op: SirOp) -> SirNode {
            SirNode {
                id: SirNodeId("test".into()),
                op,
                name: "test_node".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }
        }

        // Test all the previously-uncategorized ops to ensure they
        // no longer produce bare "Other".
        let previously_uncategorized: Vec<SirOp> = vec![
            // Comparison
            SirOp::Equal { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::NotEqual { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::Greater { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::GreaterEqual { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::Less { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::LessEqual { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            // Logical
            SirOp::LogicalAnd { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::LogicalOr { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::LogicalXor { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::LogicalNot { input: SirNodeId("x".into()) },
            // Activation
            SirOp::Relu6 { input: SirNodeId("x".into()) },
            SirOp::LeakyRelu { input: SirNodeId("x".into()), alpha: 0.01 },
            SirOp::SigmoidHard { input: SirNodeId("x".into()), alpha: 1.0, beta: 0.2 },
            SirOp::ThresholdedRelu { input: SirNodeId("x".into()), alpha: 1.0 },
            SirOp::ClampedRelu { input: SirNodeId("x".into()), alpha: 0.0, beta: 6.0 },
            SirOp::LinearActivation { input: SirNodeId("x".into()), alpha: 1.0, beta: 0.0 },
            SirOp::Prelu { input: SirNodeId("x".into()), alpha: "a".into() },
            SirOp::Softsign { input: SirNodeId("x".into()) },
            SirOp::ScaledTanh { input: SirNodeId("x".into()), alpha: 1.0, beta: 1.0 },
            SirOp::Elu { input: SirNodeId("x".into()), alpha: 1.0 },
            SirOp::Softplus { input: SirNodeId("x".into()) },
            SirOp::SoftplusParametric { input: SirNodeId("x".into()), alpha: "a".into(), beta: "b".into() },
            // Math
            SirOp::Square { input: SirNodeId("x".into()) },
            SirOp::Threshold { input: SirNodeId("x".into()), alpha: 0.0 },
            SirOp::Inverse { input: SirNodeId("x".into()), epsilon: 1e-6 },
            SirOp::Ceil { input: SirNodeId("x".into()) },
            SirOp::Floor { input: SirNodeId("x".into()) },
            SirOp::Round { input: SirNodeId("x".into()) },
            SirOp::Exp2 { input: SirNodeId("x".into()) },
            SirOp::Sign { input: SirNodeId("x".into()) },
            SirOp::Erf { input: SirNodeId("x".into()) },
            // Trig
            SirOp::Cos { input: SirNodeId("x".into()) },
            SirOp::Sin { input: SirNodeId("x".into()) },
            SirOp::Tan { input: SirNodeId("x".into()) },
            SirOp::Acos { input: SirNodeId("x".into()) },
            SirOp::Asin { input: SirNodeId("x".into()) },
            SirOp::Atan { input: SirNodeId("x".into()) },
            SirOp::Cosh { input: SirNodeId("x".into()) },
            SirOp::Sinh { input: SirNodeId("x".into()) },
            SirOp::Atanh { input: SirNodeId("x".into()) },
            // Conditional
            SirOp::Select { condition: SirNodeId("c".into()), x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            SirOp::Where { condition: SirNodeId("c".into()), x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            // Reduction
            SirOp::ReduceProd { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceSumSquare { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceL2Norm { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceL1Norm { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceLogSumExp { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceLogSum { input: SirNodeId("x".into()), axes: vec![1], keep_dims: false },
            SirOp::ReduceArgmax { input: SirNodeId("x".into()), axis: 1, keep_dims: false },
            SirOp::ReduceArgmin { input: SirNodeId("x".into()), axis: 1, keep_dims: false },
            // Resize
            SirOp::Resize { input: SirNodeId("x".into()), target_size: vec![1], mode: "linear".into(), sampling_mode: "default".into(), nearest_rounding_mode: "round".into() },
            SirOp::ResizeNearestNeighbor { input: SirNodeId("x".into()), target_height: 1, target_width: 1 },
            SirOp::ResizeBilinear { input: SirNodeId("x".into()), target_height: 1, target_width: 1, align_corners: false },
            SirOp::UpsampleNearestNeighbor { input: SirNodeId("x".into()), scale: vec![2] },
            SirOp::UpsampleBilinear { input: SirNodeId("x".into()), scale: vec![2], align_corners: false, half_pixel_centers: false },
            SirOp::CropResize { input: SirNodeId("x".into()), boxes: SirNodeId("b".into()), box_indices: SirNodeId("i".into()), crop_height: 1, crop_width: 1 },
            SirOp::Affine { input: SirNodeId("x".into()), transform: SirNodeId("t".into()), output_height: 1, output_width: 1, sampling_mode: "bilinear".into(), pad_value: 0.0 },
            SirOp::Resample { input: SirNodeId("x".into()), coordinates: SirNodeId("c".into()), sampling_mode: "bilinear".into(), pad_value: 0.0 },
            // Tensor Transform
            SirOp::ReshapeLike { input: SirNodeId("x".into()), ref_tensor: SirNodeId("r".into()) },
            SirOp::ExpandDims { input: SirNodeId("x".into()), axis: vec![1] },
            SirOp::Squeeze { input: SirNodeId("x".into()), axis: vec![1] },
            SirOp::Flatten2d { input: SirNodeId("x".into()), axis: 1 },
            SirOp::Reverse { input: SirNodeId("x".into()), axes: vec![1] },
            SirOp::ReverseSequence { input: SirNodeId("x".into()), lengths: SirNodeId("l".into()), batch_axis: 0, seq_axis: 1 },
            SirOp::SliceByIndex { input: SirNodeId("x".into()), begin: vec![0], end: vec![1], stride: vec![1], begin_mask: vec![false], end_mask: vec![false], squeeze_mask: vec![false] },
            SirOp::SliceBySize { input: SirNodeId("x".into()), begin: vec![0], size: vec![1] },
            SirOp::SlidingWindows { input: SirNodeId("x".into()), axis: 1, window_size: 3, stride: 1 },
            SirOp::DepthToSpace { input: SirNodeId("x".into()), block_size: 2 },
            SirOp::SpaceToDepth { input: SirNodeId("x".into()), block_size: 2 },
            SirOp::PixelShuffle { input: SirNodeId("x".into()), upscale_factor: 2 },
            SirOp::PixelUnshuffle { input: SirNodeId("x".into()), downscale_factor: 2 },
            SirOp::BatchToSpace { input: SirNodeId("x".into()), block_shape: vec![2], crops: vec![(0, 0)] },
            SirOp::SpaceToBatch { input: SirNodeId("x".into()), block_shape: vec![2], paddings: vec![(0, 0)] },
            SirOp::Stack { values: vec![SirNodeId("a".into())], axis: 0 },
            SirOp::Tile { input: SirNodeId("x".into()), reps: vec![1] },
            SirOp::Cumsum { input: SirNodeId("x".into()), axis: 0, exclusive: false, reverse: false },
            SirOp::FillLike { ref_tensor: SirNodeId("r".into()), value: 0.0, dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::OneHot { indices: SirNodeId("i".into()), one_hot_vector_size: 10, on_value: 1.0, off_value: 0.0, axis: 0, dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::NonZero { input: SirNodeId("x".into()) },
            SirOp::Argsort { input: SirNodeId("x".into()), axis: 0, ascending: true },
            SirOp::BandPart { input: SirNodeId("x".into()), num_lower: -1, num_upper: 0 },
            SirOp::Range1d { start: 0.0, end: 10.0, step: 1.0 },
            SirOp::Shape { input: SirNodeId("x".into()) },
            SirOp::Crop { input: SirNodeId("x".into()), crop_height: 1, crop_width: 1, offset_height: 0, offset_width: 0 },
            SirOp::Mod { x: SirNodeId("a".into()), y: SirNodeId("b".into()) },
            // Scatter/Gather
            SirOp::GatherAlongAxis { input: SirNodeId("x".into()), indices: SirNodeId("i".into()), axis: 0 },
            SirOp::ScatterAlongAxis { input: SirNodeId("x".into()), indices: SirNodeId("i".into()), updates: SirNodeId("u".into()), axis: 0 },
            SirOp::ScatterNd { input: SirNodeId("x".into()), indices: SirNodeId("i".into()), updates: SirNodeId("u".into()) },
            SirOp::NonMaximumSuppression { boxes: SirNodeId("b".into()), scores: SirNodeId("s".into()), iou_threshold: 0.5, score_threshold: 0.1, max_detections: 100 },
            // Constexpr
            SirOp::ConstexprAffineDequantize { quantized_data: "q".into(), scale: 1.0, zero_point: 0, axis: -1 },
            SirOp::ConstexprBlockwiseShiftScale { data: "d".into(), scale: "s".into(), offset: "o".into(), block_size: vec![128] },
            SirOp::ConstexprLutToDense { indices: "i".into(), lut: "l".into(), num_bits: 4 },
            SirOp::ConstexprSparseToDense { nonzero_data: "n".into(), shape: vec![1], default_value: 0.0 },
            SirOp::ConstexprCast { data: "d".into(), dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::ConstexprLutToSparse { data: "d".into(), num_bits: 4 },
            SirOp::ConstexprSparseBlockwiseShiftScale { data: "d".into(), scale: "s".into(), offset: "o".into(), block_size: vec![128], block_axis: 0 },
            // Recurrent
            SirOp::Rnn { input: SirNodeId("x".into()), initial_h: SirNodeId("h".into()), weight_ih: "wih".into(), weight_hh: "whh".into(), bias: None, mode: "relu".into(), output_sequence: false },
            SirOp::Gru { input: SirNodeId("x".into()), initial_h: SirNodeId("h".into()), weight_ih: "wih".into(), weight_hh: "whh".into(), bias: None, reset_after: true, output_sequence: false },
            SirOp::Lstm { input: SirNodeId("x".into()), initial_h: SirNodeId("h".into()), initial_c: SirNodeId("c".into()), weight_ih: "wih".into(), weight_hh: "whh".into(), bias: None, output_sequence: false },
            // Control Flow
            SirOp::Cond { pred: SirNodeId("p".into()), true_graph: "t".into(), false_graph: "f".into() },
            SirOp::WhileLoop { condition: "c".into(), body: "b".into(), loop_vars: vec![SirNodeId("v".into())] },
            SirOp::MakeList { elems: vec![SirNodeId("e".into())], dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::ListLength { ls: SirNodeId("l".into()) },
            SirOp::ListWrite { ls: SirNodeId("l".into()), index: SirNodeId("i".into()), value: SirNodeId("v".into()) },
            SirOp::ListRead { ls: SirNodeId("l".into()), index: SirNodeId("i".into()) },
            SirOp::ListGather { ls: SirNodeId("l".into()), indices: SirNodeId("i".into()) },
            SirOp::ListScatter { ls: SirNodeId("l".into()), indices: SirNodeId("i".into()), values: SirNodeId("v".into()) },
            // Random
            SirOp::RandomBernoulli { shape: vec![1], prob: 0.5, seed: None, dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::RandomNormal { shape: vec![1], mean: 0.0, stddev: 1.0, seed: None, dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::RandomUniform { shape: vec![1], low: 0.0, high: 1.0, seed: None, dtype: ane_ir::mir::MilDtype::Fp16 },
            SirOp::RandomCategorical { logits: SirNodeId("l".into()), num_samples: 1, seed: None, dtype: ane_ir::mir::MilDtype::Int32 },
            // Topk/Classify
            SirOp::Topk { input: SirNodeId("x".into()), k: 5, axis: -1 },
            SirOp::Classify { input: SirNodeId("x".into()) },
        ];

        for op in previously_uncategorized {
            let pattern = PrecisionPolicyPass::op_pattern_for_node(&node_for_op(op));
            assert_ne!(
                pattern, "Other",
                "No SirOp variant should map to bare 'Other' — all should have specific patterns or Misc_* prefix"
            );
        }
    }
}
