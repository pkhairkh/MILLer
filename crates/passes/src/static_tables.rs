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

use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, SirTargetAnnotation};

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
/// `AneLegalityRewritePass` processes them first — when `decompose_rope`
/// later checks the `sir_to_air` map for the Const node IDs, they
/// will already be present.
///
/// The actual tensor values are computed at emission time by the
/// `StaticTableResolver` in the bridge crate, which resolves
/// `value_path` strings like `static_tables/rope_tables_0/sin_tab`.
pub fn run_static_tables_pass(graph: &mut SirGraph) -> StaticTablesResult {
    let mut result = StaticTablesResult { rope_converted: 0, tables_inserted: 0 };

    // Find RoPETransform and DecodeStep nodes and collect their table references
    let rope_info: Vec<(String, String, SirMetadata)> = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            SirOp::RoPETransform { tables, .. } => {
                Some((node.id.0.clone(), tables.clone(), node.metadata.clone()))
            }
            SirOp::DecodeStep { rope_tables: Some(tables), .. } => {
                // DecodeStep also needs static tables for RoPE
                Some((node.id.0.clone(), tables.clone(), node.metadata.clone()))
            }
            _ => None,
        })
        .collect();

    // Deduplicate by tables_ref: only insert one set of Const nodes per
    // unique tables reference. All 56 RoPE nodes (28 layers × 2 for Q/K)
    // share the same "rope_tables_shared" ref, so only one set of 4 Const
    // nodes is inserted instead of 56×4 = 224.
    let mut seen_tables_refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut new_const_nodes = Vec::new();
    for (_node_id, tables_ref, metadata) in &rope_info {
        if seen_tables_refs.contains(tables_ref) {
            // Already inserted Const nodes for this tables_ref — skip
            result.rope_converted += 1;
            continue;
        }
        seen_tables_refs.insert(tables_ref.clone());

        let tables: &[(&str, ane_ir::mir::MilDtype)] = &[
            ("sin_tab", ane_ir::mir::MilDtype::Fp16),
            ("cos_tab", ane_ir::mir::MilDtype::Fp16),
            // NOTE: eye_tab and mask_tab REMOVED — these are no longer used.
            // The decode_step now uses pure-arithmetic mask computation
            // (Sub+Abs+Minimum+Maximum) instead of Gather(eye_tab/mask_tab).
            // This eliminates the quadratic-memory [seq,seq] tables and
            // the scalar-serialization bug when seq_len > 8192.
            //
            // arange_tab and arange_fp16_tab remain — arange_fp16 is used
            // by the arithmetic mask computation path.
            ("arange_tab", ane_ir::mir::MilDtype::Int32),
            ("arange_fp16_tab", ane_ir::mir::MilDtype::Fp16),
        ];
        for &(table_name, ref dtype) in tables {
            let const_id = SirNodeId(format!("sir_static_{}_{}", table_name, tables_ref));
            let const_node = SirNode {
                id: const_id,
                op: SirOp::Const {
                    value_path: format!("static_tables/{}/{}", tables_ref, table_name),
                    dtype: dtype.clone(),
                },
                name: format!("static_{}_{}", table_name, tables_ref),
                metadata: metadata.clone(),
            target_annotation: SirTargetAnnotation::default(),
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
                    tables: "rope_tables_shared".to_string(),
                },
                name: "rope_0".to_string(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            target_annotation: SirTargetAnnotation::default(),
            }],
            inputs: vec![],
            outputs: vec![],
        };

        let result = run_static_tables_pass(&mut graph);

        assert_eq!(result.rope_converted, 1);
        assert_eq!(result.tables_inserted, 4); // sin, cos, arange, arange_fp16

        // Verify static table constants were inserted
        let const_count =
            graph.nodes.iter().filter(|n| matches!(n.op, SirOp::Const { .. })).count();
        assert_eq!(const_count, 4);
        // This is critical: AneLegalityRewritePass processes nodes in order,
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

    #[test]
    fn test_static_tables_deduplicates_shared_refs() {
        // Multiple RoPE nodes sharing the same tables_ref should only
        // produce one set of Const nodes (4, not 12).
        let mut graph = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("rope_q_0".to_string()),
                    op: SirOp::RoPETransform {
                        input: SirNodeId("q_0".to_string()),
                        tables: "rope_tables_shared".to_string(),
                    },
                    name: "rope_q_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                target_annotation: SirTargetAnnotation::default(),
                },
                SirNode {
                    id: SirNodeId("rope_k_0".to_string()),
                    op: SirOp::RoPETransform {
                        input: SirNodeId("k_0".to_string()),
                        tables: "rope_tables_shared".to_string(),
                    },
                    name: "rope_k_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                target_annotation: SirTargetAnnotation::default(),
                },
                SirNode {
                    id: SirNodeId("rope_q_1".to_string()),
                    op: SirOp::RoPETransform {
                        input: SirNodeId("q_1".to_string()),
                        tables: "rope_tables_shared".to_string(),
                    },
                    name: "rope_q_1".to_string(),
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

        let result = run_static_tables_pass(&mut graph);

        assert_eq!(result.rope_converted, 3); // 3 RoPE patterns found
        assert_eq!(result.tables_inserted, 4); // only 1 set of 4 tables (deduped)

        let const_count =
            graph.nodes.iter().filter(|n| matches!(n.op, SirOp::Const { .. })).count();
        assert_eq!(const_count, 4);
    }
}
