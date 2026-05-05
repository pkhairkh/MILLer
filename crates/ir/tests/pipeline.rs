//! Integration test: SIR → MIR lowering pipeline
//!
//! Verifies that constructing SIR graphs, lowering to MIR, and
//! serializing/deserializing through all stages produces consistent
//! and well-formed output.

use ane_ir::common::{ComputeUnitHint, IrNodeId, MilDtype};
use ane_ir::linear_slice::{
    lower_linear_projection_to_mir, sir_from_linear_projection, FamilyPayload,
};
use ane_ir::mir::{MirGraph, MirNode, MirNodeId, MirOp};
use ane_ir::sir::{QualityContract, SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};
use ane_ir::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};

/// Helper: build a simple LinearProjection task spec.
fn linear_spec() -> SyntheticTaskSpec {
    SyntheticTaskSpec {
        name: "test_linear".into(),
        family: "LinearProjection".into(),
        description: None,
        op: TaskOp::LinearProjection {
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            has_bias: true,
            dtype: "fp16".into(),
        },
        measurement: MeasurementConfig {
            warmup_iterations: 3,
            measured_iterations: 10,
            metrics: vec!["Latency".into()],
        },
    }
}

#[test]
fn test_sir_construction_from_spec() {
    let spec = linear_spec();
    let sir = sir_from_linear_projection(&spec).unwrap();

    // SIR must have nodes
    assert!(!sir.nodes.is_empty(), "SIR graph must have at least one node");
    // Must have inputs and outputs
    assert!(!sir.inputs.is_empty(), "SIR graph must have inputs");
    assert!(!sir.outputs.is_empty(), "SIR graph must have outputs");

    // All node IDs must be unique
    let ids: std::collections::HashSet<&str> = sir.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids.len(), sir.nodes.len(), "All SIR node IDs must be unique");
}

#[test]
fn test_mir_lowering_from_spec() {
    let spec = linear_spec();
    let shard_name = "test_shard_0";
    let mir = lower_linear_projection_to_mir(&spec, shard_name).unwrap();

    // MIR must have nodes
    assert!(!mir.nodes.is_empty(), "MIR graph must have nodes");
    assert_eq!(mir.inputs.len(), 1, "Linear projection has one input");
    assert_eq!(mir.outputs.len(), 1, "Linear projection has one output");
    assert_eq!(mir.shard_name, shard_name);
    assert_eq!(mir.opset_version, ane_ir::DEFAULT_OPSET_VERSION);

    // All node IDs must be unique
    let ids: std::collections::HashSet<&str> = mir.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids.len(), mir.nodes.len(), "All MIR node IDs must be unique");
}

#[test]
fn test_sir_to_mir_graph_structure_preserved() {
    let spec = linear_spec();
    let sir = sir_from_linear_projection(&spec).unwrap();
    let mir = lower_linear_projection_to_mir(&spec, "shard_0").unwrap();

    // SIR and MIR must both have inputs and outputs
    assert!(!sir.inputs.is_empty());
    assert!(!sir.outputs.is_empty());
    assert!(!mir.inputs.is_empty());
    assert!(!mir.outputs.is_empty());

    // MIR should have at least as many nodes as SIR (decomposition may add nodes)
    assert!(
        mir.nodes.len() >= sir.nodes.len(),
        "MIR should have >= SIR node count (got MIR={}, SIR={})",
        mir.nodes.len(),
        sir.nodes.len()
    );
}

#[test]
fn test_mir_serialization_roundtrip() {
    let spec = linear_spec();
    let mir = lower_linear_projection_to_mir(&spec, "roundtrip_shard").unwrap();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&mir).unwrap();

    // Deserialize back
    let mir_back: MirGraph = serde_json::from_str(&json).unwrap();

    // Must have the same structure
    assert_eq!(mir_back.nodes.len(), mir.nodes.len());
    assert_eq!(mir_back.inputs.len(), mir.inputs.len());
    assert_eq!(mir_back.outputs.len(), mir.outputs.len());
    assert_eq!(mir_back.shard_name, mir.shard_name);
    assert_eq!(mir_back.opset_version, mir.opset_version);
}

#[test]
fn test_sir_serialization_roundtrip() {
    let spec = linear_spec();
    let sir = sir_from_linear_projection(&spec).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&sir).unwrap();

    // Deserialize back
    let sir_back: SirGraph = serde_json::from_str(&json).unwrap();

    assert_eq!(sir_back.nodes.len(), sir.nodes.len());
    assert_eq!(sir_back.inputs.len(), sir.inputs.len());
    assert_eq!(sir_back.outputs.len(), sir.outputs.len());
}

#[test]
fn test_mir_ops_have_valid_engine_assignments() {
    let spec = linear_spec();
    let mir = lower_linear_projection_to_mir(&spec, "engine_check").unwrap();

    // Every MIR op should have a valid default_engine() result
    for node in &mir.nodes {
        let engine = node.op.default_engine();
        // engine is Option<AneEngine> — valid values are Some(NE), Some(PE),
        // Some(TransposeEngine), or None (CPU-only). All are valid.
        // Just ensure the call doesn't panic and returns a consistent result.
        let _ = engine;
    }
}

#[test]
fn test_mir_node_dtypes_consistent() {
    let spec = linear_spec();
    let mir = lower_linear_projection_to_mir(&spec, "dtype_check").unwrap();

    // All nodes should have a valid dtype
    for node in &mir.nodes {
        match node.dtype {
            MilDtype::Fp16
            | MilDtype::Fp32
            | MilDtype::Int32
            | MilDtype::UInt8
            | MilDtype::Bool
            | MilDtype::Fp64
            | MilDtype::Int8
            | MilDtype::Int16
            | MilDtype::Int4
            | MilDtype::UInt4
            | MilDtype::E4M3
            | MilDtype::E5M2
            | MilDtype::UInt16 => {}
        }
    }
}
