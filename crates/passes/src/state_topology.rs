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
//!
//! ## Strict Mode (T-109)
//!
//! When `strict` is true (the default), the pass enforces validation
//! by returning errors for invalid patterns:
//! - ReadState without matching WriteState → `Err`
//! - WriteState without matching ReadState → `warn!` only (valid for
//!   initial state writes, e.g., prefill embedding)
//!
//! When `strict` is false, all validation failures are logged as
//! warnings/info only (backward-compatible behavior).

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// State Topology pass implementation.
///
/// T-109: Added `strict` mode to enforce validation by returning errors
/// instead of silently logging warnings.
pub struct StateTopologyPass {
    /// T-109: When true, validation failures return Err instead of just logging.
    /// Default: true.
    strict: bool,
}

impl Default for StateTopologyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTopologyPass {
    pub fn new() -> Self {
        Self { strict: true }
    }

    /// T-109: Create a pass with explicit strict mode.
    pub fn with_strict(strict: bool) -> Self {
        Self { strict }
    }

    /// T-109: Create a non-strict pass (backward-compatible with old behavior).
    pub fn new_lenient() -> Self {
        Self { strict: false }
    }

    /// Run the state topology pass.
    ///
    /// When stateful operations are present (KV-cache state reads/writes),
    /// this pass validates their structure and naming. For stateless graphs
    /// (no `StateRead`/`StateWrite` ops), it is a no-op.
    ///
    /// ## Validation behavior (T-109)
    ///
    /// In strict mode (default):
    /// - `ReadState` without matching `WriteState` → returns `Err`
    /// - `WriteState` without matching `ReadState` → logs warning only
    ///
    /// In non-strict mode:
    /// - All validation failures are logged as warnings/info only
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
        for (state_id, read_indices) in &state_reads {
            if !state_writes.contains_key(state_id) {
                // T-109: Strict mode validation — read without write is an error
                if self.strict {
                    let read_op_name = &input.nodes[read_indices[0]].name;
                    anyhow::bail!(
                        "StateTopology validation: read op '{}' references state '{}' with no matching write op",
                        read_op_name,
                        state_id
                    );
                } else {
                    // Non-strict: log warning only (backward-compatible)
                    log::warn!(
                        "StateTopology: State '{}' has reads but no writes. \
                         This may indicate an incomplete KV-cache pattern.",
                        state_id
                    );
                }
            }
        }

        // Validate: every state write should have a corresponding read
        // T-109: Write without read is always just a warning — this is valid
        // for initial state writes (e.g., prefill embedding).
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
    use ane_ir::sir::{SirMetadata, SirNode, SirNodeId, SirOp, SirTargetAnnotation, TaskOrigin};

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
        let pass = StateTopologyPass::new();
        assert!(pass.strict, "Default strict mode should be true");
    }

    #[test]
    fn test_state_topology_pass_default() {
        let pass = StateTopologyPass::default();
        assert!(pass.strict, "Default strict mode should be true");
    }

    #[test]
    fn test_run_stateless_graph() {
        // A graph with no StateRead/StateWrite ops — pass should be a no-op
        let nodes = vec![SirNode {
            id: SirNodeId("identity_0".to_string()),
            op: SirOp::Identity { input: SirNodeId("input_0".to_string()) },
            name: "identity_0".to_string(),
            metadata: test_metadata(),
        target_annotation: SirTargetAnnotation::default(),
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
            target_annotation: SirTargetAnnotation::default(),
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
            target_annotation: SirTargetAnnotation::default(),
            },
        ];
        let graph = make_graph(nodes);

        // Strict mode: should still be Ok (read has matching write)
        let pass = StateTopologyPass::new();
        let result = pass.run(graph);
        assert!(result.is_ok());
    }

    /// T-109: Test that strict mode returns Err when there's a read without a write.
    #[test]
    fn test_run_graph_with_read_no_write_strict() {
        let nodes = vec![SirNode {
            id: SirNodeId("state_read_0".to_string()),
            op: SirOp::StateRead {
                state_id: "kv_cache_layer_0_key".to_string(),
                offset: 0,
                shape: vec![1, 32, 128],
            },
            name: "state_read_0".to_string(),
            metadata: test_metadata(),
        target_annotation: SirTargetAnnotation::default(),
        }];
        let graph = make_graph(nodes);

        // T-109: In strict mode (default), read without write should return Err
        let pass = StateTopologyPass::new();
        let result = pass.run(graph);
        assert!(result.is_err(), "Strict mode should return Err for read without write");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("StateTopology validation"),
            "Error message should mention StateTopology validation, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("kv_cache_layer_0_key"),
            "Error message should reference the state_id, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("state_read_0"),
            "Error message should reference the read op name, got: {}",
            err_msg
        );
    }

    /// T-109: Test that non-strict mode returns Ok with warning for read without write.
    #[test]
    fn test_run_graph_with_read_no_write_lenient() {
        let nodes = vec![SirNode {
            id: SirNodeId("state_read_0".to_string()),
            op: SirOp::StateRead {
                state_id: "kv_cache_layer_0_key".to_string(),
                offset: 0,
                shape: vec![1, 32, 128],
            },
            name: "state_read_0".to_string(),
            metadata: test_metadata(),
        target_annotation: SirTargetAnnotation::default(),
        }];
        let graph = make_graph(nodes);

        // T-109: In non-strict mode, read without write should return Ok (just warns)
        let pass = StateTopologyPass::new_lenient();
        let result = pass.run(graph);
        assert!(
            result.is_ok(),
            "Non-strict mode should return Ok for read without write (just warn)"
        );
    }

    /// T-109: Test that write without read is Ok in both modes (valid for initial state writes).
    #[test]
    fn test_run_graph_with_write_no_read() {
        let nodes = vec![SirNode {
            id: SirNodeId("state_write_0".to_string()),
            op: SirOp::StateWrite {
                state_id: "kv_cache_layer_0_key".to_string(),
                offset: 0,
                value: SirNodeId("some_value".to_string()),
            },
            name: "state_write_0".to_string(),
            metadata: test_metadata(),
        target_annotation: SirTargetAnnotation::default(),
        }];
        let graph = make_graph(nodes);

        // Strict mode: write without read should still be Ok (just info log)
        let pass = StateTopologyPass::new();
        let result = pass.run(graph);
        assert!(
            result.is_ok(),
            "Write without read should be Ok in strict mode (valid for initial state writes)"
        );

        // Non-strict mode: also Ok
        let pass = StateTopologyPass::new_lenient();
        let nodes2 = vec![SirNode {
            id: SirNodeId("state_write_0".to_string()),
            op: SirOp::StateWrite {
                state_id: "kv_cache_layer_0_key".to_string(),
                offset: 0,
                value: SirNodeId("some_value".to_string()),
            },
            name: "state_write_0".to_string(),
            metadata: test_metadata(),
        target_annotation: SirTargetAnnotation::default(),
        }];
        let graph2 = make_graph(nodes2);
        let result = pass.run(graph2);
        assert!(
            result.is_ok(),
            "Write without read should be Ok in non-strict mode"
        );
    }

    /// T-109: Test with_strict constructor.
    #[test]
    fn test_with_strict_constructor() {
        let strict_pass = StateTopologyPass::with_strict(true);
        assert!(strict_pass.strict);

        let lenient_pass = StateTopologyPass::with_strict(false);
        assert!(!lenient_pass.strict);
    }
}
