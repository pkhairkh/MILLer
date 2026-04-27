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

use ane_ir::sir::SirGraph;
use anyhow::Result;
use crate::knowledge_query::PassKnowledgeQuery;

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
    fn op_pattern_for_node(node: &ane_ir::sir::SirNode) -> &str {
        match &node.op {
            ane_ir::sir::SirOp::LinearProjection { .. } => "LinearProjection",
            ane_ir::sir::SirOp::AttentionBlock { .. } => "AttentionBlock",
            ane_ir::sir::SirOp::RMSNorm { .. } => "RMSNorm",
            ane_ir::sir::SirOp::RoPETransform { .. } => "RoPETransform",
            ane_ir::sir::SirOp::ElementWise { .. } => "ElementWise",
            ane_ir::sir::SirOp::Reshape { .. } => "Reshape",
            ane_ir::sir::SirOp::Transpose { .. } => "Transpose",
            ane_ir::sir::SirOp::Split { .. } => "Split",
            ane_ir::sir::SirOp::Concat { .. } => "Concat",
            ane_ir::sir::SirOp::Softmax { .. } => "Softmax",
            ane_ir::sir::SirOp::StateRead { .. } => "StateRead",
            ane_ir::sir::SirOp::StateWrite { .. } => "StateWrite",
            ane_ir::sir::SirOp::DecodeStep { .. } => "DecodeStep",
            ane_ir::sir::SirOp::Sampler { .. } => "Sampler",
            _ => "Other",
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
    pub fn run(&mut self, input: SirGraph, knowledge_query: &dyn PassKnowledgeQuery) -> Result<SirGraph> {
        // Reset adaptations for this run
        self.adaptations.clear();

        let nodes = input.nodes.into_iter().map(|mut node| {
            let op_pattern = Self::op_pattern_for_node(&node);

            // Query knowledge for precision hazards for this op
            if let Some(hazard) = knowledge_query.query_precision_hazard(
                op_pattern,
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

        Ok(SirGraph {
            nodes,
            inputs: input.inputs,
            outputs: input.outputs,
        })
    }

    /// Get the adapted dtype for a given op, considering all adaptations.
    ///
    /// Returns the recommended dtype if an adaptation was recorded for
    /// this op, otherwise returns the default dtype.
    pub fn adapted_dtype_for(&self, node_name: &str) -> &str {
        self.adaptations.iter()
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
    use crate::knowledge_query::{LegalityInfo, RiskInfo, PrecisionHazardInfo, ComputePlanPlacementInfo, NoKnowledge};
    use ane_ir::sir::{SirGraph, SirNode, SirNodeId, SirOp, SirMetadata, TaskOrigin, ElementWiseOp};

    /// A mock knowledge query that reports a precision hazard for LinearProjection.
    struct MockPrecisionHazardKnowledge;

    impl PassKnowledgeQuery for MockPrecisionHazardKnowledge {
        fn query_legality(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(&self, op_pattern: &str, _current_dtype: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<PrecisionHazardInfo> {
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

        fn query_compute_plan_placement(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// A mock knowledge query that reports a hazard below the confidence threshold.
    struct MockLowConfidenceHazardKnowledge;

    impl PassKnowledgeQuery for MockLowConfidenceHazardKnowledge {
        fn query_legality(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(&self, op_pattern: &str, _current_dtype: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<PrecisionHazardInfo> {
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

        fn query_compute_plan_placement(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// A mock knowledge query that reports no hazards at all.
    struct MockSafeKnowledge;

    impl PassKnowledgeQuery for MockSafeKnowledge {
        fn query_legality(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(&self, _op_pattern: &str, _current_dtype: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(&self, _op_pattern: &str, _scope: Option<&ane_ir::kir::KnowledgeScope>) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    fn make_linear_sir() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("weight".into()),
                    op: SirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![] },
                    name: "weight".into(),
                    metadata: SirMetadata { task_origin: TaskOrigin::Synthetic, model_id: None, quality_contract: None, precision_override: None },
                },
                SirNode {
                    id: SirNodeId("output".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "weight".into(),
                        bias: Some("bias".into()),
                    },
                    name: "linear_out".into(),
                    metadata: SirMetadata { task_origin: TaskOrigin::Synthetic, model_id: None, quality_contract: None, precision_override: None },
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
        assert!(pass.has_adaptations(), "Pass must record adaptations when hazard knowledge is present");
        assert_eq!(pass.adaptations.len(), 1, "Exactly one adaptation for the LinearProjection node");

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

        assert!(!pass.has_adaptations(), "Hazard below custom threshold must not trigger adaptation");
    }
}
