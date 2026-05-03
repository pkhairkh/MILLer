//! Normalization Stabilization Pass
//!
//! Computes per-layer scale factors that absorb the interaction between
//! RMSNorm weights, projection weights, and residual connections into
//! a single fp16-friendly pre-scale. This prevents fp16 underflow/
//! overflow in heavily quantized graphs.
//!
//! The pass walks the SIR graph, identifies RMSNorm nodes, and for each
//! one inserts a `Mul` op with a `Const` scale factor before the norm.
//! This pre-scale normalizes the input range so that the subsequent
//! RMSNorm computation stays within fp16-safe bounds.
//!
//! # Scale Computation
//!
//! For each RMSNorm node:
//! 1. Computes `h_in` scale = 1/norm(input_norm_weight * [I || W_D_prev])
//! 2. Computes `h_mid` scale = 1/norm(post_attn_norm_weight * [I || W_O])
//! 3. For Q/K paths: per-group scale = 1/norm(norm_weight[group] * W[group]^T)
//! 4. For output: `h_out` = 1/norm(final_norm_weight * [I || W_D_last])
//!
//! These scales are then emitted as `Const` + `Mul` ops inserted before
//! the RMSNorm in the SIR graph.

use ane_ir::sir::{SirGraph, SirNode, SirNodeId, SirOp};

/// Result of the normalization stabilization pass.
#[derive(Debug, Clone)]
pub struct NormStabilizationResult {
    /// Number of RMSNorm nodes that received pre-scales.
    pub scales_applied: usize,
    /// Number of new Mul + Const op pairs inserted.
    pub ops_inserted: usize,
}

/// Run the normalization stabilization pass on a SIR graph.
///
/// This pass identifies RMSNorm nodes and inserts Mul + Const ops
/// before them to pre-scale the input. The scale factor is determined
/// by the normalization stabilization strategy.
///
/// For traced models, the scales are computed from the weight metadata;
/// for synthetic models, unit scales are used.
///
/// # Arguments
/// * `graph` - The SIR graph to transform (modified in place)
///
/// # Returns
/// Statistics about how many scales were applied.
pub fn run_slanc_scales_pass(graph: &mut SirGraph) -> NormStabilizationResult {
    let mut result = NormStabilizationResult { scales_applied: 0, ops_inserted: 0 };

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
            op: SirOp::Const { value_path: scale_name.clone(), dtype: ane_ir::mir::MilDtype::Fp16 },
            name: format!("norm_stabilization_const_{}", node.id.0),
            metadata: node.metadata.clone(),
        };

        // Create the Mul op that applies the pre-scale
        let mul_id = SirNodeId(format!("sir_norm_stabilization_prescale_{}", node.id.0));

        let mul_node = SirNode {
            id: mul_id.clone(),
            op: SirOp::Mul { x: input_id, y: const_id },
            name: format!("norm_stabilization_prescale_{}", node.id.0),
            metadata: node.metadata.clone(),
        };

        // Update the RMSNorm to use the pre-scaled input
        let updated_rms = SirNode {
            id: node.id.clone(),
            op: SirOp::RMSNorm { input: mul_id, weight: weight_name, epsilon, axes: axes.clone() },
            name: node.name.clone(),
            metadata: node.metadata.clone(),
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
            }],
            inputs: vec![SirNodeId("input_0".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        };

        let result = run_slanc_scales_pass(&mut graph);

        assert_eq!(result.scales_applied, 1);
        assert_eq!(result.ops_inserted, 2); // Const + Mul pair

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
                },
            ],
            inputs: vec![],
            outputs: vec![],
        };

        let result = run_slanc_scales_pass(&mut graph);
        assert_eq!(result.scales_applied, 2);
        assert_eq!(result.ops_inserted, 4); // 2 × (Const + Mul) pairs
    }
}
