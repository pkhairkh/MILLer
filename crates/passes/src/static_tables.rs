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

use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp};

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
/// This pass identifies `SirOp::RoPETransform` nodes and inserts
/// `SirOp::Const` nodes for the pre-computed RoPE cos/sin/eye/mask
/// tables. These Const nodes are prepended to the node list so that
/// `LegalityRewritePass` processes them first — when `decompose_rope`
/// later checks the `sir_to_air` map for the Const node IDs, they
/// will already be present.
///
/// The actual tensor values are computed at emission time by the
/// `StaticTableResolver` in the bridge crate, which resolves
/// `value_path` strings like `static_tables/rope_tables_0/sin_tab`.
pub fn run_static_tables_pass(graph: &mut SirGraph) -> StaticTablesResult {
    let mut result = StaticTablesResult { rope_converted: 0, tables_inserted: 0 };

    // Find RoPETransform nodes and collect their table references
    let rope_info: Vec<(String, String, SirMetadata)> = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            SirOp::RoPETransform { tables, .. } => {
                Some((node.id.0.clone(), tables.clone(), node.metadata.clone()))
            }
            _ => None,
        })
        .collect();

    // Collect all new Const nodes, then prepend them to the graph.
    // Prepending is critical: LegalityRewritePass processes nodes in order,
    // and decompose_rope checks sir_to_air for the Const node IDs.
    // If the Const nodes come AFTER the RoPETransform nodes, they won't
    // be in sir_to_air yet when decompose_rope looks for them, causing
    // a fallback to AirOp::Cos/AirOp::Sin with unresolved references.
    let mut new_const_nodes = Vec::new();
    for (node_id, tables_ref, metadata) in &rope_info {
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
            new_const_nodes.push(const_node);
            result.tables_inserted += 1;
        }
        result.rope_converted += 1;
    }

    // Prepend all Const nodes before the existing nodes
    let mut new_nodes = new_const_nodes;
    new_nodes.append(&mut graph.nodes);
    graph.nodes = new_nodes;

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

        // Verify Const nodes are PREPENDED before the RoPETransform node.
        // This is critical: LegalityRewritePass processes nodes in order,
        // and decompose_rope checks sir_to_air for the Const IDs. If they
        // come after the RoPETransform, the fallback path emits unresolved
        // AirOp::Cos/AirOp::Sin references.
        let rope_idx = graph
            .nodes
            .iter()
            .position(|n| matches!(n.op, SirOp::RoPETransform { .. }))
            .expect("RoPETransform node should exist");
        let first_const_idx = graph
            .nodes
            .iter()
            .position(|n| matches!(n.op, SirOp::Const { .. }))
            .expect("at least one Const node should exist");
        assert!(
            first_const_idx < rope_idx,
            "Const nodes must come before RoPETransform: first_const={}, rope={}",
            first_const_idx,
            rope_idx
        );
    }
}
