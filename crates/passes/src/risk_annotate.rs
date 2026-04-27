//! Risk Annotate pass.
//!
//! Annotates AIR nodes with fallback risk and drift risk scores
//! derived from knowledge about known hazards and survival data.
//!
//! This pass queries the knowledge store for per-op risk data
//! and updates each AIR node's risk scores accordingly. When
//! no knowledge is available, default risk scores are used.
//!
//! Sprint 35 addition: when compute plan evidence shows that an op
//! was NOT placed on the NeuralEngine (ane_placed=False), the
//! fallback_risk score is increased. Compute plan evidence is
//! deterministic for a given hardware+OS, so it carries high weight.

use ane_ir::air::{AirGraph, AirOp};
use ane_ir::sir::ElementWiseOp;
use anyhow::Result;
use crate::knowledge_query::PassKnowledgeQuery;

/// Default fallback risk score for operations without specific knowledge.
const DEFAULT_FALLBACK_RISK: f32 = 0.1;

/// Default drift risk score for operations without specific knowledge.
const DEFAULT_DRIFT_RISK: f32 = 0.05;

/// Fallback risk penalty when compute plan evidence shows ane_placed=False.
///
/// This is a significant increase because compute plan evidence is
/// deterministic for a given hardware+OS combination (confidence 0.9).
/// If the compute planner chose not to place an op on the NeuralEngine,
/// it means the op genuinely cannot run on ANE for that configuration.
const COMPUTE_PLAN_FALLBACK_PENALTY: f32 = 0.7;

/// Risk Annotate pass implementation.
pub struct RiskAnnotatePass {
    /// Fallback risk score to assign when no knowledge is available.
    pub default_fallback_risk: f32,
    /// Drift risk score to assign when no knowledge is available.
    pub default_drift_risk: f32,
}

impl RiskAnnotatePass {
    pub fn new() -> Self {
        Self {
            default_fallback_risk: DEFAULT_FALLBACK_RISK,
            default_drift_risk: DEFAULT_DRIFT_RISK,
        }
    }

    /// Run the risk annotation pass.
    ///
    /// Queries the knowledge store for each operation's risk data
    /// and annotates each AIR node with appropriate fallback and
    /// drift risk scores. When no knowledge is available, the
    /// pass's default risk scores are used.
    ///
    /// Sprint 35: after applying knowledge-based risk scores, also
    /// queries compute plan placement. If compute plan evidence
    /// shows ane_placed=False for an op, fallback_risk is increased
    /// by COMPUTE_PLAN_FALLBACK_PENALTY (clamped to 1.0).
    pub fn run(&self, input: AirGraph, knowledge_query: &dyn PassKnowledgeQuery) -> Result<AirGraph> {
        let annotated_nodes: Vec<ane_ir::air::AirNode> = input.nodes.into_iter().map(|mut node| {
            // Derive op pattern from the AIR node's operation type
            let op_pattern = match &node.op {
                AirOp::MatMul { .. } => "mb.matmul",
                AirOp::ElementWise { op, .. } => match op {
                    ElementWiseOp::Add => "mb.add",
                    ElementWiseOp::Mul => "mb.mul",
                    ElementWiseOp::Abs => "mb.abs",
                    ElementWiseOp::Maximum => "mb.maximum",
                    ElementWiseOp::Minimum => "mb.minimum",
                },
                AirOp::Reshape { .. } => "mb.reshape",
                AirOp::Transpose { .. } => "mb.transpose",
                AirOp::Split { .. } => "mb.split",
                AirOp::Concat { .. } => "mb.concat",
                AirOp::Softmax { .. } => "mb.softmax",
                // Normalization ops (Sprint 33)
                AirOp::ReduceMean { .. } => "mb.reduce_mean",
                AirOp::Rsqrt { .. } => "mb.rsqrt",
                AirOp::RealDiv { .. } => "mb.real_div",
                AirOp::LayerNorm { .. } => "mb.layer_norm",
                // Sampling ops (Sprint 33)
                AirOp::Topk { .. } => "mb.topk",
                AirOp::Gather { .. } => "mb.gather",
                // RoPE ops (Sprint 33)
                AirOp::Cos { .. } => "mb.cos",
                AirOp::Sin { .. } => "mb.sin",
                // Attention ops (Sprint 36)
                AirOp::Conv1x1AsLinear { .. } => "mb.linear",
                AirOp::ScaledDotProductAttention { .. } => "mb.scaled_dot_product_attention",
                AirOp::SliceByIndex { .. } => "mb.slice_by_index",
                // Activation ops (Sprint 36)
                AirOp::Gelu { .. } => "mb.gelu",
                AirOp::Relu { .. } => "mb.relu",
                // State ops (Sprint 36)
                AirOp::StateReadFixed { .. } => "mb.read_state",
                AirOp::StateWriteFixed { .. } => "mb.coreml_update_state",
                // Sprint 50: P2 ops
                AirOp::SliceUpdate { .. } => "mb.slice_update",
                AirOp::Exp { .. } => "mb.exp",
                AirOp::Sigmoid { .. } => "mb.sigmoid",
                AirOp::Tanh { .. } => "mb.tanh",
                AirOp::Where { .. } => "mb.where",
                _ => "unknown",
            };

            // Step 1: Apply knowledge-based risk scores
            match knowledge_query.query_risk(op_pattern, None) {
                Some(risk_info) => {
                    node.fallback_risk = risk_info.fallback_risk;
                    node.drift_risk = risk_info.drift_risk;
                }
                None => {
                    node.fallback_risk = self.default_fallback_risk;
                    node.drift_risk = self.default_drift_risk;
                }
            }

            // Step 2: Apply compute plan placement evidence (Sprint 35)
            // If compute plan shows ane_placed=False, increase fallback_risk
            if let Some(placement) = knowledge_query.query_compute_plan_placement(op_pattern, None) {
                if !placement.ane_placed {
                    // Compute plan evidence: op was NOT placed on NeuralEngine.
                    // This is deterministic and high-confidence, so apply a
                    // significant fallback risk penalty.
                    node.fallback_risk = (node.fallback_risk + COMPUTE_PLAN_FALLBACK_PENALTY).min(1.0);
                }
            }

            node
        }).collect();

        Ok(AirGraph {
            nodes: annotated_nodes,
            inputs: input.inputs,
            outputs: input.outputs,
            staticization_decisions: input.staticization_decisions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::air::{AirNode, AirNodeId};
    use ane_ir::kir::KnowledgeScope;
    use crate::knowledge_query::{
        NoKnowledge, PassKnowledgeQuery, RiskInfo, ComputePlanPlacementInfo, LegalityInfo, PrecisionHazardInfo,
    };

    /// A mock knowledge query that can return configurable compute plan placement.
    struct MockKnowledge {
        risk_info: Option<RiskInfo>,
        compute_plan_placement: Option<ComputePlanPlacementInfo>,
    }

    impl MockKnowledge {
        fn new() -> Self {
            Self {
                risk_info: None,
                compute_plan_placement: None,
            }
        }

        fn with_compute_plan_not_ane(mut self, op_pattern: &str) -> Self {
            self.compute_plan_placement = Some(ComputePlanPlacementInfo {
                op_pattern: op_pattern.to_string(),
                ane_placed: false,
                preferred_device: "CPU".to_string(),
                confidence: 0.9,
                evidence_count: 1,
                source_id: Some("cp_test".to_string()),
            });
            self
        }

        fn with_compute_plan_ane(mut self, op_pattern: &str) -> Self {
            self.compute_plan_placement = Some(ComputePlanPlacementInfo {
                op_pattern: op_pattern.to_string(),
                ane_placed: true,
                preferred_device: "NeuralEngine".to_string(),
                confidence: 0.9,
                evidence_count: 1,
                source_id: Some("cp_test".to_string()),
            });
            self
        }

        fn with_risk(mut self, fallback_risk: f32, drift_risk: f32) -> Self {
            self.risk_info = Some(RiskInfo {
                fallback_risk,
                drift_risk,
                confidence: 0.5,
                evidence_count: 1,
                source_id: Some("test_risk".to_string()),
            });
            self
        }
    }

    impl PassKnowledgeQuery for MockKnowledge {
        fn query_legality(&self, _op_pattern: &str, _scope: Option<&KnowledgeScope>) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(&self, _op_pattern: &str, _scope: Option<&KnowledgeScope>) -> Option<RiskInfo> {
            self.risk_info.clone()
        }

        fn query_precision_hazard(&self, _op_pattern: &str, _current_dtype: &str, _scope: Option<&KnowledgeScope>) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(&self, _op_pattern: &str, _scope: Option<&KnowledgeScope>) -> Option<ComputePlanPlacementInfo> {
            self.compute_plan_placement.clone()
        }
    }

    fn make_simple_graph(op: AirOp) -> AirGraph {
        AirGraph {
            nodes: vec![AirNode {
                id: AirNodeId("test_node".to_string()),
                op,
                name: "test".to_string(),
                legality_confidence: 0.5,
                sir_source: None,
                fallback_risk: 0.0,
                drift_risk: 0.0,
                precision_override: None,
            }],
            inputs: vec![],
            outputs: vec![],
            staticization_decisions: vec![],
        }
    }

    #[test]
    fn test_no_knowledge_uses_defaults() {
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = NoKnowledge;
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(result.nodes[0].fallback_risk, DEFAULT_FALLBACK_RISK);
        assert_eq!(result.nodes[0].drift_risk, DEFAULT_DRIFT_RISK);
    }

    #[test]
    fn test_compute_plan_ane_placed_no_penalty() {
        // When compute plan shows ane_placed=True, no penalty is applied
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_compute_plan_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        // Default fallback risk should be unchanged (no penalty)
        assert_eq!(result.nodes[0].fallback_risk, DEFAULT_FALLBACK_RISK);
    }

    #[test]
    fn test_compute_plan_not_ane_increases_fallback_risk() {
        // When compute plan shows ane_placed=False, fallback risk should increase
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::Reshape {
            input: AirNodeId("a".to_string()),
            target_shape: vec![1, 64],
        });
        let knowledge = MockKnowledge::new().with_compute_plan_not_ane("mb.reshape");
        let result = pass.run(graph, &knowledge).unwrap();
        // Fallback risk should be default + COMPUTE_PLAN_FALLBACK_PENALTY
        let expected = (DEFAULT_FALLBACK_RISK + COMPUTE_PLAN_FALLBACK_PENALTY).min(1.0);
        assert!(
            (result.nodes[0].fallback_risk - expected).abs() < 0.001,
            "expected fallback_risk ~{}, got {}",
            expected,
            result.nodes[0].fallback_risk
        );
    }

    #[test]
    fn test_compute_plan_penalty_stacks_with_risk_knowledge() {
        // When there is already risk knowledge AND compute plan shows not-ANE,
        // the penalty should be added to the risk knowledge's fallback_risk
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_risk(0.3, 0.1)
            .with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        // Fallback risk should be 0.3 (from risk knowledge) + 0.7 (penalty) = 1.0
        let expected = (0.3 + COMPUTE_PLAN_FALLBACK_PENALTY).min(1.0);
        assert!(
            (result.nodes[0].fallback_risk - expected).abs() < 0.001,
            "expected fallback_risk ~{}, got {}",
            expected,
            result.nodes[0].fallback_risk
        );
    }

    #[test]
    fn test_compute_plan_penalty_clamped_to_one() {
        // Even with high existing risk + compute plan penalty, result clamps to 1.0
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_risk(0.5, 0.2)
            .with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert!(
            result.nodes[0].fallback_risk <= 1.0,
            "fallback_risk should be clamped to 1.0, got {}",
            result.nodes[0].fallback_risk
        );
    }
}
