//! Risk Annotate pass.
//!
//! Annotates AIR nodes with `LegalityStatus` derived from knowledge
//! about known hazards, legality evidence, and compute plan placement.
//!
//! T-P3-03: This pass now sets `legality_status: LegalityStatus` directly
//! on each AIR node. The legacy f32 fields (legality_confidence, fallback_risk,
//! drift_risk) have been removed from `AirNode`. The pass derives the structured
//! enum from three knowledge sources:
//!
//! 1. **Legality knowledge** — whether the op is known-legal for ANE.
//! 2. **Risk knowledge** — fallback and drift risk scores from the knowledge store.
//! 3. **Compute plan placement** — whether the op was placed on the NeuralEngine.
//!
//! Sprint 35 addition: when compute plan evidence shows that an op
//! was NOT placed on the NeuralEngine (ane_placed=False), the
//! `LegalityStatus` is set to `LikelyFallback`. Compute plan evidence
//! is deterministic for a given hardware+OS, so it carries high weight.

use crate::knowledge_query::{LegalityInfo, PassKnowledgeQuery};
use ane_ir::air::{AirGraph, AirOp, LegalityStatus};
use anyhow::Result;

/// Derive `LegalityStatus` from the combined evidence of all knowledge
/// sources available to the risk annotation pass.
///
/// The decision logic follows a strict priority cascade where stronger
/// signals override weaker ones.
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
/// 6. **Strong positive metrics** (legality confidence ≥ 0.95 AND
///    fallback_risk < 0.1 AND drift_risk < 0.1) → `Verified`
///    The combination of high confidence and low risk qualifies the op.
///
/// 7. **Default** → `Unverified`
///    Some knowledge exists but it is not conclusive enough for a
///    Verified or LikelyFallback classification.
fn determine_legality_status(
    compute_plan_not_ane: bool,
    legality_info: Option<&LegalityInfo>,
    risk_info: Option<&crate::knowledge_query::RiskInfo>,
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
    // to classify this op.
    if !had_risk_knowledge && legality_info.is_none() {
        return LegalityStatus::Unknown;
    }

    // Priority 4: High fallback risk from actual knowledge.
    // Only fires when had_risk_knowledge is true, meaning the
    // fallback_risk value came from the knowledge store (not from
    // the conservative default).
    if had_risk_knowledge {
        if let Some(risk) = risk_info {
            if risk.fallback_risk >= 0.5 {
                return LegalityStatus::LikelyFallback;
            }
        }
    }

    // Priority 5: Strong positive evidence from the legality store.
    // Multiple independent observations at high confidence confirm
    // the op runs correctly on ANE.
    if let Some(info) = legality_info {
        if info.ane_legal && info.confidence >= 0.95 && info.evidence_count >= 2 {
            return LegalityStatus::Verified;
        }
    }

    // Priority 6: Strong positive signal from combined metrics.
    // High legality confidence combined with very low risk scores.
    if let Some(info) = legality_info {
        if info.confidence >= 0.95 {
            if let Some(risk) = risk_info {
                if risk.fallback_risk < 0.1 && risk.drift_risk < 0.1 {
                    return LegalityStatus::Verified;
                }
            }
        }
    }

    // Default: some knowledge exists but it is not conclusive enough
    // for Verified or LikelyFallback.
    LegalityStatus::Unverified
}

/// Risk Annotate pass implementation.
pub struct RiskAnnotatePass;

impl Default for RiskAnnotatePass {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskAnnotatePass {
    pub fn new() -> Self {
        Self
    }

    /// Run the risk annotation pass.
    ///
    /// Queries the knowledge store for each operation's risk data
    /// and annotates each AIR node with the appropriate `LegalityStatus`.
    ///
    /// T-P3-03: This pass now sets `legality_status` directly from the
    /// knowledge store evidence, without using intermediate f32 fields.
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

                // Step 1: Query legality knowledge.
                let legality_info = match knowledge_query.query_legality(op_pattern, None) {
                    Some(info) => {
                        log::debug!(
                            "risk_annotate: legality knowledge for '{}' = ane_legal={}, confidence={:.2}",
                            op_pattern, info.ane_legal, info.confidence
                        );
                        Some(info)
                    }
                    None => {
                        log::warn!(
                            "risk_annotate: no legality knowledge for '{}'",
                            op_pattern
                        );
                        None
                    }
                };

                // Step 2: Query risk knowledge
                let risk_info;
                let had_risk_knowledge = match knowledge_query.query_risk(op_pattern, None) {
                    Some(info) => {
                        log::debug!(
                            "risk_annotate: risk knowledge for '{}' = fallback_risk={:.2}, drift_risk={:.2}",
                            op_pattern, info.fallback_risk, info.drift_risk
                        );
                        risk_info = Some(info);
                        true
                    }
                    None => {
                        log::warn!(
                            "risk_annotate: no risk knowledge for '{}'",
                            op_pattern
                        );
                        risk_info = None;
                        false
                    }
                };

                // Step 3: Query compute plan placement evidence (Sprint 35)
                let mut compute_plan_not_ane = false;
                if let Some(placement) =
                    knowledge_query.query_compute_plan_placement(op_pattern, None)
                {
                    if !placement.ane_placed {
                        compute_plan_not_ane = true;
                    }
                }

                // Step 4: Derive legality_status from all combined evidence (T-P3-03)
                node.legality_status = determine_legality_status(
                    compute_plan_not_ane,
                    legality_info.as_ref(),
                    risk_info.as_ref(),
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
                sir_source: None,
                precision_override: None,
                legality_status: LegalityStatus::Unverified,
            }],
            inputs: vec![],
            outputs: vec![],
        }
    }

    // ─── LegalityStatus derivation tests ─────────────────────────

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
        // Even if legality confidence is high, compute plan not-ANE wins.
        let pass = RiskAnnotatePass::new();
        let graph = make_simple_graph(AirOp::MatMul {
            a: AirNodeId("a".to_string()),
            b: AirNodeId("b".to_string()),
        });
        let knowledge = MockKnowledge::new()
            .with_legality(true, 0.99)
            .with_compute_plan_not_ane("mb.matmul");
        let result = pass.run(graph, &knowledge).unwrap();
        assert_eq!(
            result.nodes[0].legality_status,
            LegalityStatus::LikelyFallback,
            "Compute plan not-ANE should override high legality confidence"
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
    fn test_conservative_default_risk_with_legality_sets_unverified() {
        // When legality knowledge exists but risk knowledge does not,
        // the status is Unverified, not Unknown or LikelyFallback.
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
            "Legality knowledge without risk knowledge should produce Unverified"
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
        // When legality confidence >= 0.95 and both risk scores are
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
        // Even with high legality confidence, if fallback_risk is not
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
        // Priority 5: ane_legal=true, confidence >= 0.95, >= 2 observations
        // should produce Verified even without risk knowledge being low.
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
        // compute plan says not-ANE.
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

    // ─── LegalityStatus enum tests ────────────────────────────────

    #[test]
    fn test_legality_status_default_is_unverified() {
        assert_eq!(LegalityStatus::default(), LegalityStatus::Unverified);
    }

    #[test]
    fn test_legality_status_equality() {
        assert_eq!(LegalityStatus::Verified, LegalityStatus::Verified);
        assert_ne!(LegalityStatus::Verified, LegalityStatus::Unverified);
        assert_ne!(LegalityStatus::LikelyFallback, LegalityStatus::Unknown);
    }

    // ─── Legacy mapping tests ─────────────────────────────────────

    #[test]
    fn test_legacy_high_confidence_maps_to_verified() {
        // legality_confidence > 0.8 → Verified
        use ane_ir::air::LegacyAirNodeFields;
        let legacy = LegacyAirNodeFields {
            id: AirNodeId("test".into()),
            op: AirOp::Add { x: AirNodeId("a".into()), y: AirNodeId("b".into()) },
            name: "test".into(),
            legality_confidence: 0.9,
            sir_source: None,
            fallback_risk: 0.1,
            drift_risk: 0.05,
            precision_override: None,
        };
        let node = AirNode::try_from(legacy).unwrap();
        assert_eq!(node.legality_status, LegalityStatus::Verified);
    }

    #[test]
    fn test_legacy_high_fallback_risk_maps_to_likely_fallback() {
        // fallback_risk > 0.5 → LikelyFallback
        use ane_ir::air::LegacyAirNodeFields;
        let legacy = LegacyAirNodeFields {
            id: AirNodeId("test".into()),
            op: AirOp::Add { x: AirNodeId("a".into()), y: AirNodeId("b".into()) },
            name: "test".into(),
            legality_confidence: 0.5,
            sir_source: None,
            fallback_risk: 0.7,
            drift_risk: 0.3,
            precision_override: None,
        };
        let node = AirNode::try_from(legacy).unwrap();
        assert_eq!(node.legality_status, LegalityStatus::LikelyFallback);
    }

    #[test]
    fn test_legacy_low_confidence_maps_to_unknown() {
        // legality_confidence < 0.1 → Unknown
        use ane_ir::air::LegacyAirNodeFields;
        let legacy = LegacyAirNodeFields {
            id: AirNodeId("test".into()),
            op: AirOp::Add { x: AirNodeId("a".into()), y: AirNodeId("b".into()) },
            name: "test".into(),
            legality_confidence: 0.05,
            sir_source: None,
            fallback_risk: 0.1,
            drift_risk: 0.05,
            precision_override: None,
        };
        let node = AirNode::try_from(legacy).unwrap();
        assert_eq!(node.legality_status, LegalityStatus::Unknown);
    }

    #[test]
    fn test_legacy_default_fields_map_to_unverified() {
        // Default/missing fields → Unverified
        use ane_ir::air::LegacyAirNodeFields;
        let legacy = LegacyAirNodeFields {
            id: AirNodeId("test".into()),
            op: AirOp::Add { x: AirNodeId("a".into()), y: AirNodeId("b".into()) },
            name: "test".into(),
            legality_confidence: 0.5,
            sir_source: None,
            fallback_risk: 0.1,
            drift_risk: 0.05,
            precision_override: None,
        };
        let node = AirNode::try_from(legacy).unwrap();
        assert_eq!(node.legality_status, LegalityStatus::Unverified);
    }
}
