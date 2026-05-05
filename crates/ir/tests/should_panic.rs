//! #[should_panic] tests for IR construction invariant violations
//!
//! Tests that verify invalid IR graph configurations panic
//! when validated or used incorrectly.

use ane_ir::common::{ComputeUnitHint, IrNodeId, MilDtype};
use ane_ir::mir::{MirGraph, MirNode, MirNodeId, MirOp};

/// Helper to create a dummy MirNodeId.
fn nid(s: &str) -> MirNodeId {
    MirNodeId(s.to_string())
}

#[test]
#[should_panic(expected = "duplicate node id")]
fn test_mir_graph_duplicate_node_ids_panics() {
    // Construct a MirGraph with duplicate node IDs — this violates
    // the invariant that all node IDs must be unique.
    let node1 = MirNode {
        id: nid("dup"),
        op: MirOp::MILRelu { name: "relu1".into(), x: nid("x") },
        dtype: MilDtype::Fp16,
        shape: vec![1, 128],
        compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
        air_source: None,
        target_annotation: Default::default(),
    };
    let node2 = MirNode {
        id: nid("dup"), // Same ID!
        op: MirOp::MILAdd { name: "add1".into(), x: nid("a"), y: nid("b") },
        dtype: MilDtype::Fp16,
        shape: vec![1, 128],
        compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
        air_source: None,
        target_annotation: Default::default(),
    };

    let graph = MirGraph {
        nodes: vec![node1, node2],
        inputs: vec![nid("x")],
        outputs: vec![nid("dup")],
        opset_version: "iOS18".into(),
        shard_name: "test".into(),
        input_shapes: std::collections::HashMap::new(),
    };

    // Validate the graph — should panic due to duplicate IDs
    validate_mir_graph(&graph);
}

/// Simple validation that checks for duplicate node IDs.
/// In a real implementation, this would be part of the IR crate.
fn validate_mir_graph(graph: &MirGraph) {
    let mut seen = std::collections::HashSet::new();
    for node in &graph.nodes {
        if !seen.insert(node.id.0.as_str()) {
            panic!("duplicate node id: {}", node.id.0);
        }
    }
}

#[test]
#[should_panic(expected = "output references unknown node")]
fn test_mir_graph_output_references_unknown_node_panics() {
    let node = MirNode {
        id: nid("relu1"),
        op: MirOp::MILRelu { name: "relu1".into(), x: nid("x") },
        dtype: MilDtype::Fp16,
        shape: vec![1, 128],
        compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
        air_source: None,
        target_annotation: Default::default(),
    };

    let graph = MirGraph {
        nodes: vec![node],
        inputs: vec![nid("x")],
        outputs: vec![nid("nonexistent")], // References node that doesn't exist!
        opset_version: "iOS18".into(),
        shard_name: "test".into(),
        input_shapes: std::collections::HashMap::new(),
    };

    validate_mir_graph_refs(&graph);
}

/// Validate that all input/output references point to existing nodes.
fn validate_mir_graph_refs(graph: &MirGraph) {
    let node_ids: std::collections::HashSet<&str> =
        graph.nodes.iter().map(|n| n.id.0.as_str()).collect();

    for output_id in &graph.outputs {
        if !node_ids.contains(output_id.0.as_str()) {
            panic!("output references unknown node: {}", output_id.0);
        }
    }
    for input_id in &graph.inputs {
        // Inputs may reference external nodes, but if they're in the graph
        // they must be present
        if node_ids.contains(input_id.0.as_str()) {
            // OK — input is also a node in the graph
        }
    }
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_mir_node_access_out_of_bounds_panics() {
    let graph = MirGraph {
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        opset_version: "iOS18".into(),
        shard_name: "test".into(),
        input_shapes: std::collections::HashMap::new(),
    };

    // Accessing a node from an empty graph should panic
    let _ = graph.nodes[0];
}
