//! Strategy Discovery for Traced Models
//!
//! Bridges the strategy framework (`ane_ir::strategy`) with the tracing
//! pipeline. When a model is traced, this module analyzes the resulting
//! `TracedGraph` and `SirGraph` to discover applicable optimization
//! strategies, producing a `CompilationPlan` that drives the pass pipeline.
//!
//! This is the glue between "trace any model ad-hoc" and "apply the right
//! optimizations dynamically" — no model registry needed.

use ane_ir::ane_target::AneFamily;
use ane_ir::sir::SirGraph;
use ane_ir::strategy::{discover_strategies, CompilationPlan, DiscoveryReport};

/// Discover optimization strategies for a traced model.
///
/// This is the primary entry point for strategy-driven compilation.
/// It analyzes the SIR graph (built from the trace) and the target
/// hardware to determine which optimizations apply and with what
/// parameters.
///
/// The returned `DiscoveryReport` can be converted into a
/// `CompilationPlan` that specifies which passes to run and in what
/// order — all driven by the graph structure, not a hardcoded registry.
pub fn discover_for_trace(sir: &SirGraph, target_family: AneFamily) -> DiscoveryReport {
    discover_strategies(sir, target_family)
}

/// Discover and plan for a traced model in one step.
///
/// Convenience function that discovers strategies and creates a
/// compilation plan. The plan specifies which passes to run and
/// with what parameters.
pub fn plan_for_trace(sir: &SirGraph, target_family: AneFamily) -> CompilationPlan {
    let report = discover_for_trace(sir, target_family);
    CompilationPlan::from_discovery(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    fn make_rms_norm_graph() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("input".to_string()),
                    op: SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                    name: "input".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("rms_0".to_string()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input".to_string()),
                        weight: "norm_weight".to_string(),
                        epsilon: 1e-6,
                    },
                    name: "rms_norm".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        }
    }

    #[test]
    fn test_discover_for_trace_finds_normalization() {
        let graph = make_rms_norm_graph();
        let report = discover_for_trace(&graph, AneFamily::A16);

        assert!(
            report.evaluated.iter().any(|s| {
                s.id.category == ane_ir::strategy::StrategyCategory::Normalization && s.applicable
            }),
            "Should discover applicable normalization strategies"
        );
    }

    #[test]
    fn test_plan_for_trace_produces_ordered_plan() {
        let graph = make_rms_norm_graph();
        let plan = plan_for_trace(&graph, AneFamily::A16);

        assert!(!plan.strategy_order.is_empty(), "Plan should include at least one strategy");
    }
}
