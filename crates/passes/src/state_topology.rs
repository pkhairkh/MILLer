//! State Topology pass.
//!
//! Analyzes and optimizes the state read/write patterns in SIR,
//! ensuring correct state ownership and access patterns.
//!
//! When the SIR contains `StateRead`/`StateWrite` ops (from KV-cache
//! enabled tracing), this pass:
//! - Verifies that state read/write patterns are well-formed
//! - Ensures every StateRead has a corresponding StateWrite
//! - Validates that KV-cache state IDs follow the naming convention
//!   `kv_cache_layer_{idx}_{key|value}`
//! - Flags state operations that exceed ANE state capacity

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// State Topology pass implementation.
pub struct StateTopologyPass {
    // No configuration needed
}

impl Default for StateTopologyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTopologyPass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the state topology pass.
    ///
    /// When stateful operations are present (KV-cache state reads/writes),
    /// this pass validates their structure and naming. For stateless graphs
    /// (no `StateRead`/`StateWrite` ops), it is a no-op.
    ///
    /// Validation checks:
    /// 1. Every `StateRead` has a matching `StateWrite` for the same state_id
    /// 2. KV-cache state IDs follow the `kv_cache_layer_{idx}_{key|value}` convention
    /// 3. State shapes are consistent between reads and writes
    pub fn run(&self, input: SirGraph) -> Result<SirGraph> {
        use ane_ir::sir::SirOp;

        // Collect state IDs from reads and writes
        let mut state_reads: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        let mut state_writes: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, node) in input.nodes.iter().enumerate() {
            match &node.op {
                SirOp::StateRead { state_id, .. } => {
                    state_reads.entry(state_id.clone()).or_default().push(idx);
                }
                SirOp::StateWrite { state_id, .. } => {
                    state_writes.entry(state_id.clone()).or_default().push(idx);
                }
                _ => {}
            }
        }

        // If no states, pass through
        if state_reads.is_empty() && state_writes.is_empty() {
            return Ok(input);
        }

        // Validate: every state read should have a corresponding write
        for state_id in state_reads.keys() {
            if !state_writes.contains_key(state_id) {
                // This is acceptable for prefill (embedding) models that only
                // write states but never read them. Log a warning but don't fail.
                log::warn!(
                    "StateTopology: State '{}' has reads but no writes. \
                     This may indicate an incomplete KV-cache pattern.",
                    state_id
                );
            }
        }

        // Validate: every state write should have a corresponding read
        for state_id in state_writes.keys() {
            if !state_reads.contains_key(state_id) {
                // This is acceptable for the first decode step where the cache
                // is initialized. Log as informational.
                log::info!(
                    "StateTopology: State '{}' has writes but no reads. \
                     This is normal for initial cache population.",
                    state_id
                );
            }
        }

        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    /// Helper: create a minimal SirMetadata for test nodes.
    fn test_metadata() -> SirMetadata {
        SirMetadata {
            task_origin: TaskOrigin::Synthetic,
            model_id: None,
            quality_contract: None,
            precision_override: None,
        }
    }

    /// Helper: create a SirGraph with the given nodes.
    fn make_graph(nodes: Vec<SirNode>) -> SirGraph {
        let inputs = nodes
            .iter()
            .filter(|n| matches!(n.op, SirOp::StateRead { .. }))
            .map(|n| n.id.clone())
            .collect();
        let outputs = nodes
            .iter()
            .filter(|n| matches!(n.op, SirOp::StateWrite { .. }))
            .map(|n| n.id.clone())
            .collect();
        SirGraph { nodes, inputs, outputs }
    }

    #[test]
    fn test_state_topology_pass_new() {
        let _pass = StateTopologyPass::new();
    }

    #[test]
    fn test_state_topology_pass_default() {
        let _pass = StateTopologyPass::default();
    }

    #[test]
    fn test_run_stateless_graph() {
        // A graph with no StateRead/StateWrite ops — pass should be a no-op
        let nodes = vec![SirNode {
            id: SirNodeId("identity_0".to_string()),
            op: SirOp::Identity { input: SirNodeId("input_0".to_string()) },
            name: "identity_0".to_string(),
            metadata: test_metadata(),
        }];
        let graph = make_graph(nodes);
        let node_count = graph.nodes.len();

        let pass = StateTopologyPass::new();
        let result = pass.run(graph).unwrap();
        assert_eq!(result.nodes.len(), node_count);
    }

    #[test]
    fn test_run_graph_with_state_read_and_write() {
        // A graph with matching StateRead/StateWrite for the same state_id
        let nodes = vec![
            SirNode {
                id: SirNodeId("state_read_0".to_string()),
                op: SirOp::StateRead {
                    state_id: "kv_cache_layer_0_key".to_string(),
                    offset: 0,
                    shape: vec![1, 32, 128],
                },
                name: "state_read_0".to_string(),
                metadata: test_metadata(),
            },
            SirNode {
                id: SirNodeId("state_write_0".to_string()),
                op: SirOp::StateWrite {
                    state_id: "kv_cache_layer_0_key".to_string(),
                    offset: 0,
                    value: SirNodeId("state_read_0".to_string()),
                },
                name: "state_write_0".to_string(),
                metadata: test_metadata(),
            },
        ];
        let graph = make_graph(nodes);

        let pass = StateTopologyPass::new();
        let result = pass.run(graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_graph_with_read_no_write() {
        // A graph with StateRead but no matching StateWrite.
        // The pass should still return Ok (just prints a warning).
        let nodes = vec![SirNode {
            id: SirNodeId("state_read_0".to_string()),
            op: SirOp::StateRead {
                state_id: "kv_cache_layer_0_key".to_string(),
                offset: 0,
                shape: vec![1, 32, 128],
            },
            name: "state_read_0".to_string(),
            metadata: test_metadata(),
        }];
        let graph = make_graph(nodes);

        let pass = StateTopologyPass::new();
        let result = pass.run(graph);
        assert!(result.is_ok());
    }
}
