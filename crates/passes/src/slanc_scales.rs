//! Normalization Stabilization Pass — **STUB-MIMIC (M-005)**
//!
//! **WARNING**: This pass is a structural scaffold only. It inserts
//! `Const + Mul` ops before RMSNorm nodes, but the scale values are
//! **NOT computed from weight metadata** — they are placeholder strings
//! that will resolve to uncomputed/zero values. This can silently
//! corrupt the graph if downstream passes treat these as real constants.
//!
//! The `computed_scales` field in the result is always `false` to
//! honestly reflect this limitation.
//!
//! This pass is NOT wired into the compilation pipeline. It exists
//! as a structural placeholder for future implementation.
//!
//! # Intended Scale Computation (not yet implemented)
//!
//! For each RMSNorm node:
//! 1. `h_in` scale = 1/norm(input_norm_weight * [I || W_D_prev])
//! 2. `h_mid` scale = 1/norm(post_attn_norm_weight * [I || W_O])
//! 3. For Q/K paths: per-group scale = 1/norm(norm_weight[group] * W[group]^T)
//! 4. For output: `h_out` = 1/norm(final_norm_weight * [I || W_D_last])
//!
//! When implemented, these scales should be emitted as `Const + Mul`
//! ops inserted before the RMSNorm in the SIR graph.

use ane_ir::sir::{SirGraph, SirNode, SirNodeId, SirOp, SirTargetAnnotation};

/// Result of the normalization stabilization pass.
#[derive(Debug, Clone)]
pub struct NormStabilizationResult {
    /// Number of RMSNorm nodes that received pre-scales.
    pub scales_applied: usize,
    /// Number of new Mul + Const op pairs inserted.
    pub ops_inserted: usize,
    /// Whether the scale factors were actually computed from weight metadata.
    ///
    /// When `false`, the inserted Const+Mul ops use placeholder scale values
    /// that are NOT derived from real weight tensors. Downstream passes must
    /// treat these as structural scaffolding, not validated numeric constants.
    pub computed_scales: bool,
}

/// Run the normalization stabilization pass on a SIR graph.
///
/// **STUB-MIMIC WARNING (M-005)**: This pass inserts `Const + Mul` ops
/// with **uncomputed scale values**. The Const ops contain placeholder
/// value_path strings, not actual tensor data derived from weight metadata.
/// The `computed_scales` field in the result will always be `false`.
///
/// Do NOT use this pass in production pipelines until real scale
/// computation is implemented.
///
/// # Arguments
/// * `graph` - The SIR graph to transform (modified in place)
///
/// # Returns
/// Statistics about how many scales were applied.
#[deprecated(
    since = "0.6.0",
    note = "M-005: This pass inserts Const+Mul ops with uncomputed scale values. \
           Do not use in production until real weight-derived computation is implemented. \
           The inserted ops silently corrupt the graph if treated as real constants."
)]
pub fn run_slanc_scales_pass(graph: &mut SirGraph) -> NormStabilizationResult {
    let mut result = NormStabilizationResult {
        scales_applied: 0,
        ops_inserted: 0,
        computed_scales: false, // M-005: current impl inserts structural ops only
    };

    // Collect indices of RMSNorm nodes that need pre-scales
    let rms_norm_indices: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| match &node.op {
            SirOp::RMSNorm { .. } => Some(idx),
            _ => None,
        })
        .collect();

    // For each RMSNorm node, insert a Const + Mul op pair before it.
    //
    // Note: In a full implementation, the scale values would be computed
    // from the actual weight tensors. Here we insert the structural
    // ops and mark them for later weight-dependent computation.
    for idx in rms_norm_indices {
        let node = &graph.nodes[idx];

        log::warn!(
            "slanc_scales: inserting UNCOMPUTED scale placeholder for RMSNorm node `{}` \
             — Const+Mul ops are structural scaffolding, not derived from weight metadata",
            node.id.0
        );

        let (input_id, weight_name, epsilon, axes) = match &node.op {
            SirOp::RMSNorm { input, weight, epsilon, axes } => {
                (input.clone(), weight.clone(), *epsilon, axes.clone())
            }
            _ => unreachable!(),
        };

        // Create the Const op for the scale factor
        let scale_name = format!("norm_stabilization_scale_{}", node.id.0);
        let const_id = SirNodeId(format!("sir_norm_stabilization_const_{}", node.id.0));

        let const_node = SirNode {
            id: const_id.clone(),
            op: SirOp::Const {
                value_path: scale_name.clone(),
                dtype: ane_ir::mir::MilDtype::Fp16,
            },
            name: format!("norm_stabilization_const_{}", node.id.0),
            metadata: node.metadata.clone(),
        target_annotation: SirTargetAnnotation::default(),
        };

        // Create the Mul op that applies the pre-scale
        let mul_id = SirNodeId(format!("sir_norm_stabilization_prescale_{}", node.id.0));

        let mul_node = SirNode {
            id: mul_id.clone(),
            op: SirOp::Mul { x: input_id, y: const_id },
            name: format!("norm_stabilization_prescale_{}", node.id.0),
            metadata: node.metadata.clone(),
        target_annotation: SirTargetAnnotation::default(),
        };

        // Update the RMSNorm to use the pre-scaled input
        let updated_rms = SirNode {
            id: node.id.clone(),
            op: SirOp::RMSNorm { input: mul_id, weight: weight_name, epsilon, axes: axes.clone() },
            name: node.name.clone(),
            metadata: node.metadata.clone(),
        target_annotation: SirTargetAnnotation::default(),
        };

        // Replace the RMSNorm node with the updated version
        graph.nodes[idx] = updated_rms;

        // Insert the Const and Mul nodes (appended; canonicalize pass
        // will fix ordering)
        graph.nodes.push(const_node);
        graph.nodes.push(mul_node);

        result.scales_applied += 1;
        result.ops_inserted += 2; // Const + Mul pair
    }

    result
}

#[cfg(test)]
#[allow(deprecated)] // M-005: Tests intentionally call deprecated function
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, TaskOrigin};

    #[test]
    fn test_norm_stabilization_inserts_prescale_ops() {
        let mut graph = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("rms_0".to_string()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("input_0".to_string()),
                    weight: "norm_weight_0".to_string(),
                    epsilon: 1e-6,
                    axes: vec![2],
                },
                name: "rms_norm_0".to_string(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            target_annotation: SirTargetAnnotation::default(),
            }],
            inputs: vec![SirNodeId("input_0".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        };

        let result = run_slanc_scales_pass(&mut graph);

        assert_eq!(result.scales_applied, 1);
        assert_eq!(result.ops_inserted, 2); // Const + Mul pair
        assert!(!result.computed_scales, "computed_scales must be false — scales are not derived from weights");

        // Verify the RMSNorm now takes input from the Mul op
        let rms_node = graph.nodes.iter().find(|n| n.id.0 == "rms_0").unwrap();
        match &rms_node.op {
            SirOp::RMSNorm { input, .. } => {
                assert!(
                    input.0.contains("norm_stabilization_prescale"),
                    "RMSNorm input should come from the pre-scale Mul op"
                );
            }
            _ => panic!("Expected RMSNorm op"),
        }

        // Verify a Const op was inserted
        let has_const = graph.nodes.iter().any(|n| {
            matches!(n.op, SirOp::Const { .. }) && n.name.contains("norm_stabilization_const")
        });
        assert!(has_const, "Graph should contain a Const op for the scale factor");

        // Verify a Mul op was inserted
        let has_mul = graph.nodes.iter().any(|n| {
            matches!(n.op, SirOp::Mul { .. }) && n.name.contains("norm_stabilization_prescale")
        });
        assert!(has_mul, "Graph should contain a Mul op for the pre-scale");
    }

    #[test]
    fn test_norm_stabilization_processes_all_rmsnorms() {
        let mut graph = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("rms_0".to_string()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input_0".to_string()),
                        weight: "norm_weight_0".to_string(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                    name: "rms_norm_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                target_annotation: SirTargetAnnotation::default(),
                },
                SirNode {
                    id: SirNodeId("rms_1".to_string()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input_1".to_string()),
                        weight: "norm_weight_1".to_string(),
                        epsilon: 1e-5,
                        axes: vec![2],
                    },
                    name: "rms_norm_1".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                target_annotation: SirTargetAnnotation::default(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
        };

        let result = run_slanc_scales_pass(&mut graph);
        assert_eq!(result.scales_applied, 2);
        assert_eq!(result.ops_inserted, 4); // 2 × (Const + Mul) pairs
        assert!(!result.computed_scales, "computed_scales must be false — scales are not derived from weights");
    }

    #[test]
    fn test_norm_stabilization_computed_scales_field() {
        // Verify that computed_scales is false when no real weight computation occurs.
        let mut graph = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("rms_0".to_string()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("input_0".to_string()),
                    weight: "norm_weight_0".to_string(),
                    epsilon: 1e-6,
                    axes: vec![2],
                },
                name: "rms_norm_0".to_string(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            target_annotation: SirTargetAnnotation::default(),
            }],
            inputs: vec![SirNodeId("input_0".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        };

        let result = run_slanc_scales_pass(&mut graph);

        // The current implementation inserts Const+Mul ops as structural
        // placeholders without computing actual scale values from weights,
        // so computed_scales MUST be false (STUB-MIMIC compliance).
        assert!(!result.computed_scales,
            "computed_scales should be false: current pass does not compute scales from weight metadata");
        assert_eq!(result.scales_applied, 1, "one RMSNorm should have been pre-scaled");
        assert_eq!(result.ops_inserted, 2, "one Const+Mul pair should be inserted");
    }
}
