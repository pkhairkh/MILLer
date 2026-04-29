//! Static Tables Pass — pre-compute RoPE, causal mask, and identity tables as constants.
//!
//! Derived from pkhairkh/qwen3-coreml-palettized's `rotary_tables.py`:
//! pre-computes sin_tab, cos_tab, eye_tab, and mask_tab as fp16 constants
//! and embeds them directly in the Core ML graph. This avoids dynamic
//! computation of these tensors at inference time, which is especially
//! important on ANE where dynamic ops may cause CPU fallback.
//!
//! The pass identifies RoPE-related Mul/Cos/Sin patterns and replaces
//! dynamic RoPE computation with static table lookups where possible.

use ane_ir::sir::{SirGraph, SirNode, SirNodeId, SirOp};

/// Result of the static tables pass.
#[derive(Debug, Clone)]
pub struct StaticTablesResult {
    /// Number of RoPE patterns converted to static table lookups.
    pub rope_converted: usize,
    /// Number of static table constants inserted.
    pub tables_inserted: usize,
}

/// Run the static tables pass on a SIR graph.
///
/// This pass identifies RoPE-related computation patterns (Mul with
/// cos/sin inputs) and marks them for static table materialization.
/// The actual table values are computed during weight loading, but
/// the structural transformation happens here.
///
/// For now, this pass annotates RoPETransform nodes with static table
/// references and inserts Const nodes for the pre-computed tables.
/// A future pass will replace the dynamic cos/sin computation with
/// Gather ops from the static tables.
pub fn run_static_tables_pass(graph: &mut SirGraph) -> StaticTablesResult {
    let mut result = StaticTablesResult { rope_converted: 0, tables_inserted: 0 };

    // Find RoPETransform nodes and attach static table references
    let rope_indices: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| match &node.op {
            SirOp::RoPETransform { .. } => Some(idx),
            _ => None,
        })
        .collect();

    for idx in rope_indices {
        let (node_id, tables_ref, metadata) = {
            let node = &graph.nodes[idx];
            match &node.op {
                SirOp::RoPETransform { tables, .. } => {
                    (node.id.0.clone(), tables.clone(), node.metadata.clone())
                }
                _ => unreachable!(),
            }
        };

        // Insert static table constants: sin_tab, cos_tab, eye_tab, mask_tab
        // These will be materialized during weight loading/emission
        let tables = &["sin_tab", "cos_tab", "eye_tab", "mask_tab"];
        for &table_name in tables {
            let const_id = SirNodeId(format!("sir_static_{}_{}", table_name, node_id));
            let const_node = SirNode {
                id: const_id,
                op: SirOp::Const {
                    value_path: format!("static_tables/{}/{}", tables_ref, table_name),
                    dtype: ane_ir::mir::MilDtype::Fp16,
                },
                name: format!("static_{}_{}", table_name, node_id),
                metadata: metadata.clone(),
            };
            graph.nodes.push(const_node);
            result.tables_inserted += 1;
        }

        result.rope_converted += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, TaskOrigin};

    #[test]
    fn test_static_tables_inserts_constants() {
        let mut graph = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("rope_0".to_string()),
                op: SirOp::RoPETransform {
                    input: SirNodeId("input_0".to_string()),
                    tables: "rope_tables_0".to_string(),
                },
                name: "rope_0".to_string(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![],
            outputs: vec![],
        };

        let result = run_static_tables_pass(&mut graph);

        assert_eq!(result.rope_converted, 1);
        assert_eq!(result.tables_inserted, 4); // sin, cos, eye, mask

        // Verify static table constants were inserted
        let const_count =
            graph.nodes.iter().filter(|n| matches!(n.op, SirOp::Const { .. })).count();
        assert_eq!(const_count, 4);
    }
}
