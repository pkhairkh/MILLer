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
//!
//! Sprint 61 addition: the pass now directly derives and sets
//! `legality_status` on each AIR node from the combined evidence
//! of legality knowledge, risk knowledge, and compute plan placement.
//! Previously, `legality_status` was only populated indirectly through
//! the `TryFrom<LegacyAirNodeFields>` migration path during
//! deserialization, which meant it stayed at the default `Unverified`
//! even after this pass updated the f32 risk scores — creating an
//! inconsistency between the f32 fields and the structured enum.
//!
//! M-011/V-026: default risk scores changed from 0.1/0.05 to 0.5/0.5
//! to signal "unknown" rather than the false safety of near-zero.
//! The legality_confidence field is now also updated from legality
//! knowledge, with a conservative 0.5 default when no knowledge
//! is available.

use crate::knowledge_query::{LegalityInfo, PassKnowledgeQuery};
use ane_ir::air::{AirGraph, AirOp, LegalityStatus};
use anyhow::Result;

/// Conservative default legality confidence when no knowledge is available.
///
/// 0.5 signals "unknown" rather than the false certainty of 1.0.
const DEFAULT_LEGALITY_CONFIDENCE: f32 = 0.5;

/// Conservative default fallback risk score when no knowledge is available.
///
/// 0.5 signals "unknown" rather than the false safety of 0.0.
/// M-011/V-026: changed from 0.1 to avoid false confidence.
const DEFAULT_FALLBACK_RISK: f32 = 0.5;

/// Conservative default drift risk score when no knowledge is available.
///
/// 0.5 signals "unknown" rather than the false safety of 0.0.
/// M-011/V-026: changed from 0.05 to avoid false confidence.
const DEFAULT_DRIFT_RISK: f32 = 0.5;

/// Fallback risk penalty when compute plan evidence shows ane_placed=False.
///
/// This is a significant increase because compute plan evidence is
/// deterministic for a given hardware+OS combination (confidence 0.9).
/// If the compute planner chose not to place an op on the NeuralEngine,
/// it means the op genuinely cannot run on ANE for that configuration.
const COMPUTE_PLAN_FALLBACK_PENALTY: f32 = 0.7;

/// Derive `LegalityStatus` from the combined evidence of all knowledge
/// sources available to the risk annotation pass.
///
/// The decision logic follows a strict priority cascade where stronger
/// signals override weaker ones. This ensures the structured enum field
/// is always consistent with the f32 risk scores that this pass writes.
///
/// # Priority order
///
/// 1. **Compute plan ane_placed=false** → `LikelyFallback`
///    Deterministic hardware evidence; if the planner says no, it's no.
///
/// 2. **Knowledge-explicit illegality** (ane_legal=false with confidence
///    ≥ 0.5) → `LikelyFallback`
///    The legality knowledge store explicitly flags this op as ANE-illegal.
///
/// 3. **No knowledge from any source** → `Unknown`
///    If legality, risk, and compute plan queries all returned None,
///    we truly have insufficient information to make any claim.
///    This MUST come before the fallback_risk threshold check because
///    conservative defaults (0.5) are not evidence of actual risk.
///
/// 4. **High fallback_risk from actual knowledge** (≥ 0.5 AND
///    had_risk_knowledge=true) → `LikelyFallback`
///    Real risk evidence (not default values) pushes the op into the
///    fallback regime.
///
/// 5. **Strong positive legality evidence** (ane_legal=true, confidence
///    ≥ 0.95, ≥ 2 observations) → `Verified`
///    Multiple independent observations confirm the op runs correctly.
///
/// 6. **Strong positive metrics** (legality_confidence ≥ 0.95 AND
///    fallback_risk < 0.1 AND drift_risk < 0.1) → `Verified`
///    The combination of high confidence and low risk qualifies the op.
///
/// 7. **Default** → `Unverified`
///    Some knowledge exists but it is not conclusive enough for a
///    Verified or LikelyFallback classification.
fn determine_legality_status(
    node: &ane_ir::air::AirNode,
    compute_plan_not_ane: bool,
    legality_info: Option<&LegalityInfo>,
    had_risk_knowledge: bool,
) -> LegalityStatus {
    // Priority 1: Compute plan says NOT ANE-placed.
    // This is deterministic, high-confidence evidence that the op
    // cannot run on ANE for the target hardware+OS configuration.
    if compute_plan_not_ane {
        return LegalityStatus::LikelyFallback;
    }

    // Priority 2: Legality knowledge explicitly flags this op as
    // ANE-illegal with reasonable confidence. This is a direct
    // negative signal from the legality store.
    if let Some(info) = legality_info {
        if !info.ane_legal && info.confidence >= 0.5 {
            return LegalityStatus::LikelyFallback;
        }
    }

    // Priority 3: No knowledge from any source.
    // If the risk store returned nothing AND the legality store
    // returned nothing, we have genuinely insufficient information
    // to classify this op. This MUST come before the fallback_risk
    // threshold check because conservative defaults (0.5) are not
    // evidence of actual risk — they signal "unknown".
    if !had_risk_knowledge && legality_info.is_none() {
        return LegalityStatus::Unknown;
    }

    // Priority 4: High fallback risk from actual knowledge.
    // Only fires when had_risk_knowledge is true, meaning the
    // fallback_risk value came from the knowledge store (not from
    // the conservative default). The compute plan penalty also
    // sets compute_plan_not_ane=true, handled by priority 1 above.
    if had_risk_knowledge && node.fallback_risk >= 0.5 {
        return LegalityStatus::LikelyFallback;
    }

    // Priority 5: Strong positive evidence from the legality store.
    // Multiple independent observations at high confidence confirm
    // the op runs correctly on ANE. This is more reliable than
    // just having high legality_confidence alone.
    if let Some(info) = legality_info {
        if info.ane_legal && info.confidence >= 0.95 && info.evidence_count >= 2 {
            return LegalityStatus::Verified;
        }
    }

    // Priority 6: Strong positive signal from combined metrics.
    // High legality_confidence from the legality pass combined with
    // very low risk scores from this pass. This catches cases where
    // legality_rewrite set high confidence and risk_annotate found
    // no contradicting risk evidence.
    if node.legality_confidence >= 0.95 && node.fallback_risk < 0.1 && node.drift_risk < 0.1 {
        return LegalityStatus::Verified;
    }

    // Default: some knowledge exists but it is not conclusive enough
    // for Verified or LikelyFallback. The op may or may not work on
    // ANE — we simply don't have strong enough evidence either way.
    LegalityStatus::Unverified
}

/// Risk Annotate pass implementation.
pub struct RiskAnnotatePass {
    /// Legality confidence to assign when no knowledge is available.
    pub default_legality_confidence: f32,
    /// Fallback risk score to assign when no knowledge is available.
    pub default_fallback_risk: f32,
    /// Drift risk score to assign when no knowledge is available.
    pub default_drift_risk: f32,
}

impl Default for RiskAnnotatePass {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskAnnotatePass {
    pub fn new() -> Self {
        Self {
            default_legality_confidence: DEFAULT_LEGALITY_CONFIDENCE,
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
    ///
    /// Sprint 61: derives and sets `legality_status` from all
    /// evidence sources (legality, risk, compute plan), ensuring
    /// the structured enum is always consistent with the f32 risk
    /// scores written by this pass.
    ///
    /// M-011/V-026: also updates `legality_confidence` from legality
    /// knowledge, with a conservative default when no knowledge is
    /// available. Uses conservative 0.5 defaults for all fields.
    pub fn run(
        &self,
        input: AirGraph,
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> Result<AirGraph> {
        let annotated_nodes: Vec<ane_ir::air::AirNode> = input
            .nodes
            .into_iter()
            .map(|mut node| {
                // Derive op pattern from the AIR node's operation type
                let op_pattern = match &node.op {
                    AirOp::MatMul { .. } => "mb.matmul",
                    AirOp::Add { .. } => "mb.add",
                    AirOp::Mul { .. } => "mb.mul",
                    AirOp::Abs { .. } => "mb.abs",
                    AirOp::Maximum { .. } => "mb.maximum",
                    AirOp::Minimum { .. } => "mb.minimum",
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

                // Step 1: Apply knowledge-based legality confidence.
                // Store the legality_info for later status derivation.
                let legality_info = match knowledge_query.query_legality(op_pattern, None) {
                    Some(info) => {
                        node.legality_confidence = info.confidence;
                        Some(info)
                    }
                    None => {
                        log::warn!(
                            "risk_annotate: no legality knowledge for '{}', \
                             using conservative legality_confidence={}",
                            op_pattern,
                            self.default_legality_confidence
                        );
                        node.legality_confidence = self.default_legality_confidence;
                        None
                    }
                };

                // Step 2: Apply knowledge-based risk scores
                let had_risk_knowledge = match knowledge_query.query_risk(op_pattern, None) {
                    Some(risk_info) => {
                        node.fallback_risk = risk_info.fallback_risk;
                        node.drift_risk = risk_info.drift_risk;
                        true
                    }
                    None => {
                        log::warn!(
                            "risk_annotate: no risk knowledge for '{}', \
                             using conservative fallback_risk={}, drift_risk={}",
                            op_pattern,
                            self.default_fallback_risk,
                            self.default_drift_risk
                        );
                        node.fallback_risk = self.default_fallback_risk;
                        node.drift_risk = self.default_drift_risk;
                        false
                    }
                };

                // Step 3: Apply compute plan placement evidence (Sprint 35)
                // If compute plan shows ane_placed=False, increase fallback_risk
                let mut compute_plan_not_ane = false;
                if let Some(placement) =
                    knowledge_query.query_compute_plan_placement(op_pattern, None)
                {
                    if !placement.ane_placed {
                        // Compute plan evidence: op was NOT placed on NeuralEngine.
                        // This is deterministic and high-confidence, so apply a
                        // significant fallback risk penalty.
                        compute_plan_not_ane = true;
                        node.fallback_risk =
                            (node.fallback_risk + COMPUTE_PLAN_FALLBACK_PENALTY).min(1.0);
                    }
                }

                // Step 4: Derive legality_status from all combined evidence (Sprint 61)
                node.legality_status = determine_legality_status(
                    &node,
                    compute_plan_not_ane,
                    legality_info.as_ref(),
                    had_risk_knowledge,
                );

                node
            })
            .collect();

        Ok(AirGraph {
            nodes: annotated_nodes,
            inputs: input.inputs,
            outputs: input.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_query::NoKnowledge;
    use crate::test_utils::MockKnowledge;
    use ane_ir::air::{AirNode, AirNodeId};

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
                legality_status: LegalityStatus::Unverified,
            }],
            inputs: vec![],
            outputs: vec![],
        }
    }

    // ─── f32 risk score tests ────────────────────────────────────

    #[test]
    fn test_no_knowledge_uses_conservative_defaults() {
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = NoKnowledge;
        let result = pass.run(graph, &knowledge).unwrap();
        // All three fields should use conservative 0.5 defaults (not ideal 1.0/0.0/0.0)
        assert_eq!(result.nodes[0].legality_confidence, DEFAULT_LEGALITY_CONFIDENCE);
        assert_eq!(result.nodes[0].fallback_risk, DEFAULT_FALLBACK_RISK);
        assert_eq!(result.nodes[0].drift_risk, DEFAULT_DRIFT_RISK);
    }

    #[test]
    fn test_knowledge_overrides_defaults() {
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_legality(true, 0.95)
            .with_risk(0.1, 0.05);
        let result = pass.run(graph, &knowledge).unwrap();
        // Knowledge-provided values should override defaults
        assert!((result.nodes[0].legality_confidence - 0.95).abs() < 0.001);
        assert!((result.nodes[0].fallback_risk - 0.1).abs() < 0.001);
        assert!((result.nodes[0].drift_risk - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_legality_knowledge_only_uses_defaults_for_risk() {
        // When only legality knowledge is available, risk fields should still
        // use conservative defaults
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_legality(true, 0.9);
        let result = pass.run(graph, &knowledge).unwrap();
        assert!((result.nodes[0].legality_confidence - 0.9).abs() < 0.001);
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
        // Conservative default fallback risk should be unchanged (no penalty)
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
        let knowledge =
            MockKnowledge::new().with_risk(0.3, 0.1).with_compute_plan_not_ane("mb.matmul");
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
        let knowledge =
            MockKnowledge::new().with_risk(0.5, 0.2).with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert!(
            result.nodes[0].fallback_risk <= 1.0,
            "fallback_risk should be clamped to 1.0, got {}",
            result.nodes[0].fallback_risk
        );
    }

    // ─── legality_status derivation tests ─────────────────────────

    #[test]
    fn test_no_knowledge_sets_unknown() {
        // When ALL knowledge sources return None, legality_status should be
        // Unknown — we have genuinely insufficient information.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = NoKnowledge;
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unknown,
            "No knowledge from any source should produce Unknown"
        );
    }

    #[test]
    fn test_compute_plan_not_ane_sets_likely_fallback() {
        // Compute plan ane_placed=false is the strongest negative signal.
        // It should override everything else.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::LikelyFallback,
            "Compute plan ane_placed=false should produce LikelyFallback"
        );
    }

    #[test]
    fn test_compute_plan_not_ane_overrides_high_legality_confidence() {
        // Even if legality_confidence is high, compute plan not-ANE wins.
        let pass = RiskAnnotatePass::new();
        let mut graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        graph.nodes[0].legality_confidence = 0.99;
        let knowledge = MockKnowledge::new().with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::LikelyFallback,
            "Compute plan not-ANE should override high legality_confidence"
        );
    }

    #[test]
    fn test_high_fallback_risk_sets_likely_fallback() {
        // When fallback_risk >= 0.5 (from risk knowledge), the op is
        // likely to fall back to CPU.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_risk(0.6, 0.2);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::LikelyFallback,
            "High fallback_risk from risk knowledge should produce LikelyFallback"
        );
    }

    #[test]
    fn test_conservative_default_risk_with_legality_sets_likely_fallback() {
        // When legality knowledge exists but risk knowledge does not,
        // the conservative default fallback_risk=0.5 should NOT trigger
        // LikelyFallback (it's just a default, not evidence). But with
        // legality knowledge present, the status is Unverified, not Unknown.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_legality(true, 0.9);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unverified,
            "Conservative default risk with legality knowledge should produce Unverified, not LikelyFallback"
        );
    }

    #[test]
    fn test_moderate_risk_knowledge_sets_unverified() {
        // When risk knowledge exists but fallback_risk < 0.5, the op
        // is Unverified — we have some information but it's not strong
        // enough for a definitive classification.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_risk(0.3, 0.1);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unverified,
            "Moderate risk knowledge should produce Unverified"
        );
    }

    #[test]
    fn test_high_confidence_low_risk_sets_verified() {
        // When legality_confidence >= 0.95 and both risk scores are
        // very low (< 0.1), the op is Verified.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_legality(true, 0.97)
            .with_risk(0.05, 0.02);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Verified,
            "High confidence + low risk should produce Verified"
        );
    }

    #[test]
    fn test_high_confidence_moderate_risk_stays_unverified() {
        // Even with high legality_confidence, if fallback_risk is not
        // very low (< 0.1), the op stays Unverified rather than
        // Verified — the risk evidence prevents full verification.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_legality(true, 0.97)
            .with_risk(0.2, 0.1);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unverified,
            "High confidence + moderate risk should produce Unverified, not Verified"
        );
    }

    #[test]
    fn test_strong_legality_evidence_sets_verified() {
        // Priority 4: ane_legal=true, confidence >= 0.95, >= 2 observations
        // should produce Verified even without risk knowledge being low.
        // (The risk knowledge provides 0.1 fallback_risk, which is < 0.5
        // but not < 0.1, so this tests the priority 4 path specifically.)
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_legality_and_evidence(true, 0.97, 3)
            .with_risk(0.1, 0.05);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Verified,
            "Strong legality evidence (high conf, multiple obs) should produce Verified"
        );
    }

    #[test]
    fn test_risk_knowledge_prevents_unknown() {
        // If risk knowledge exists (even with moderate scores), the
        // status should NOT be Unknown — we have some information.
        // With moderate fallback_risk < 0.5 and no legality info,
        // the result is Unverified (not Unknown).
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new().with_risk(0.3, 0.1);
        let result = pass.run(graph, &knowledge).unwrap();
        assert_ne!(
            result.nodes[0].legality_status,
            LegalityStatus::Unknown,
            "Having risk knowledge should prevent Unknown"
        );
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unverified,
            "Moderate risk without legality info should produce Unverified"
        );
    }

    #[test]
    fn test_compute_plan_penalty_upgrades_to_likely_fallback() {
        // Even moderate risk (0.3) becomes LikelyFallback when the
        // compute plan penalty pushes fallback_risk >= 0.5.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge =
            MockKnowledge::new().with_risk(0.3, 0.1).with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::LikelyFallback,
            "Compute plan not-ANE should produce LikelyFallback even with moderate risk"
        );
    }

    #[test]
    fn test_default_risk_no_legality_no_compute_plan_is_unknown() {
        // When only default risk scores are applied (no actual risk
        // knowledge), no legality knowledge, and no compute plan,
        // the result should be Unknown.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::Add {
            x: AirNodeId("a".to_string()),
            y: AirNodeId("b".to_string()),
        });
        let knowledge = NoKnowledge;
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::Unknown,
            "Default risk scores with no actual knowledge should produce Unknown"
        );
    }
}
