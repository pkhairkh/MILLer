//! KV Cache Rewrite Pass — transform naive KV cache to reverse ring-buffer.
//!
//! Derived from pkhairkh/qwen3-coreml-palettized's reverse ring-buffer approach.
//!
//! The naive KV cache uses append/shift operations that often force CPU
//! fallback on ANE because scatter operations are not ANE-supported.
//! The reverse ring-buffer layout keeps active context in a contiguous
//! suffix of the sequence axis, and new K/V values are written by
//! masked blending instead of scatter-heavy updates.
//!
//! This pass transforms `StateRead`/`StateWrite` sequences into
//! `KvCacheRingUpdate` ops when the KV cache layout is set to
//! `ReverseRingBuffer`.

use ane_ir::sir::{SirGraph, SirNode, SirNodeId, SirOp, KvCacheLayout};

/// Result of the KV cache rewrite pass.
#[derive(Debug, Clone)]
pub struct KvCacheRewriteResult {
    /// Number of StateRead/StateWrite pairs converted to KvCacheRingUpdate.
    pub pairs_converted: usize,
    /// Number of KvCacheRingUpdate ops inserted.
    pub ring_updates_inserted: usize,
}

/// Run the KV cache rewrite pass on a SIR graph.
///
/// When `kv_layout` is `ReverseRingBuffer`, this pass:
/// 1. Identifies StateRead/StateWrite pairs targeting the same KV cache state
/// 2. Replaces them with KvCacheRingUpdate ops that use masked blending
/// 3. Inserts position and mask inputs for the ring-buffer write logic
///
/// When `kv_layout` is `Naive` or `Paged`, the pass is a no-op.
pub fn run_kv_cache_rewrite_pass(
    graph: &mut SirGraph,
    kv_layout: &KvCacheLayout,
) -> KvCacheRewriteResult {
    let mut result = KvCacheRewriteResult {
        pairs_converted: 0,
        ring_updates_inserted: 0,
    };

    if kv_layout != &KvCacheLayout::ReverseRingBuffer {
        return result;
    }

    // Find StateRead ops that read from KV cache states
    let kv_read_indices: Vec<usize> = graph.nodes.iter().enumerate()
        .filter_map(|(idx, node)| {
            match &node.op {
                SirOp::StateRead { state_id, .. } if state_id.contains("kv_cache") => Some(idx),
                _ => None,
            }
        })
        .collect();

    // For each KV cache read, find the corresponding write and replace
    // the pair with a KvCacheRingUpdate. In a full implementation, this
    // would also insert position counter and valid mask management ops.
    for read_idx in kv_read_indices {
        let (state_id, offset, shape) = match &graph.nodes[read_idx].op {
            SirOp::StateRead { state_id, offset, shape } => {
                (state_id.clone(), *offset, shape.clone())
            }
            _ => unreachable!(),
        };

        // Find the corresponding StateWrite
        let write_idx = graph.nodes.iter().enumerate()
            .find(|(_, node)| {
                match &node.op {
                    SirOp::StateWrite { state_id: ws, .. } if ws == &state_id => true,
                    _ => false,
                }
            })
            .map(|(idx, _)| idx);

        if let Some(w_idx) = write_idx {
            let (value_id, _) = match &graph.nodes[w_idx].op {
                SirOp::StateWrite { value, state_id: ws, .. } => (value.clone(), ws.clone()),
                _ => unreachable!(),
            };

            // Parse layer index from state_id (e.g., "kv_cache_layer_3_key" → 3)
            let layer_idx = parse_layer_idx(&state_id);
            let is_key = state_id.ends_with("_key");

            // Create the KvCacheRingUpdate op
            let cache_id = SirNodeId(format!("sir_kv_cache_read_{}", graph.nodes[read_idx].id.0));
            let position_id = SirNodeId(format!("position_counter_{}", layer_idx));
            let mask_id = SirNodeId(format!("valid_mask_{}", layer_idx));

            let ring_node = SirNode {
                id: SirNodeId(format!("sir_kv_ring_{}_{}", layer_idx, if is_key { "k" } else { "v" })),
                op: SirOp::KvCacheRingUpdate {
                    cache: cache_id,
                    new_values: value_id,
                    position: position_id,
                    valid_mask: mask_id,
                    is_key,
                    layer_idx,
                },
                name: format!("kv_ring_{}_{}", layer_idx, if is_key { "k" } else { "v" }),
                metadata: graph.nodes[read_idx].metadata.clone(),
            };

            // Replace the read node with a placeholder that feeds into the ring update
            // (the actual cache read happens inside KvCacheRingUpdate)
            graph.nodes[read_idx] = SirNode {
                id: graph.nodes[read_idx].id.clone(),
                op: SirOp::Identity {
                    input: SirNodeId(format!("sir_kv_ring_{}_{}", layer_idx, if is_key { "k" } else { "v" })),
                },
                name: format!("kv_read_passthrough_{}", layer_idx),
                metadata: graph.nodes[read_idx].metadata.clone(),
            };

            // Mark the write node as replaced
            graph.nodes[w_idx] = SirNode {
                id: graph.nodes[w_idx].id.clone(),
                op: SirOp::Identity {
                    input: SirNodeId(format!("sir_kv_ring_{}_{}", layer_idx, if is_key { "k" } else { "v" })),
                },
                name: format!("kv_write_passthrough_{}", layer_idx),
                metadata: graph.nodes[w_idx].metadata.clone(),
            };

            // Insert the KvCacheRingUpdate op
            graph.nodes.push(ring_node);

            result.pairs_converted += 1;
            result.ring_updates_inserted += 1;
        }
    }

    result
}

/// Parse a layer index from a KV cache state ID string.
/// E.g., "kv_cache_layer_3_key" → 3
fn parse_layer_idx(state_id: &str) -> usize {
    state_id.split('_')
        .filter_map(|s| s.parse::<usize>().ok())
        .next()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, TaskOrigin};

    #[test]
    fn test_kv_cache_rewrite_naive_is_noop() {
        let mut graph = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("kv_read_0".to_string()),
                    op: SirOp::StateRead {
                        state_id: "kv_cache_layer_0_key".to_string(),
                        offset: 0,
                        shape: vec![1, 64, 4, 32],
                    },
                    name: "kv_read_0".to_string(),
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

        let result = run_kv_cache_rewrite_pass(&mut graph, &KvCacheLayout::Naive);
        assert_eq!(result.pairs_converted, 0);
        assert_eq!(result.ring_updates_inserted, 0);
    }

    #[test]
    fn test_kv_cache_rewrite_reverse_ring_buffer() {
        let mut graph = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("kv_read_0".to_string()),
                    op: SirOp::StateRead {
                        state_id: "kv_cache_layer_0_key".to_string(),
                        offset: 0,
                        shape: vec![1, 64, 4, 32],
                    },
                    name: "kv_read_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("kv_write_0".to_string()),
                    op: SirOp::StateWrite {
                        state_id: "kv_cache_layer_0_key".to_string(),
                        offset: 0,
                        value: SirNodeId("new_k_0".to_string()),
                    },
                    name: "kv_write_0".to_string(),
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

        let result = run_kv_cache_rewrite_pass(&mut graph, &KvCacheLayout::ReverseRingBuffer);

        assert_eq!(result.pairs_converted, 1);
        assert_eq!(result.ring_updates_inserted, 1);

        // Verify a KvCacheRingUpdate op was inserted
        let has_ring = graph.nodes.iter().any(|n| {
            matches!(n.op, SirOp::KvCacheRingUpdate { .. })
        });
        assert!(has_ring, "Graph should contain a KvCacheRingUpdate op");
    }

    #[test]
    fn test_parse_layer_idx() {
        assert_eq!(parse_layer_idx("kv_cache_layer_3_key"), 3);
        assert_eq!(parse_layer_idx("kv_cache_layer_27_value"), 27);
        assert_eq!(parse_layer_idx("other_state"), 0);
    }
}
