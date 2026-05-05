//! Serialization utilities for all IR types.

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Serialize any IR graph to MessagePack bytes.
pub fn serialize_graph<T: Serialize>(graph: &T) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("IR serialization failed: {}", e))
}

/// Deserialize any IR graph from MessagePack bytes.
pub fn deserialize_graph<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("IR deserialization failed: {}", e))
}

// Keep type-specific convenience wrappers for backward compatibility
use crate::{air::AirGraph, mir::MirGraph, pir::PirGraph, sir::SirGraph};

pub fn serialize_sir(graph: &SirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_sir(bytes: &[u8]) -> Result<SirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_air(graph: &AirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_air(bytes: &[u8]) -> Result<AirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_mir(graph: &MirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_mir(bytes: &[u8]) -> Result<MirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_pir(graph: &PirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_pir(bytes: &[u8]) -> Result<PirGraph, String> {
    deserialize_graph(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::{AirGraph, AirNode, AirNodeId, AirOp};
    use crate::common::MilDtype;
    use crate::mir::{ComputeUnitHint, MirGraph, MirNode, MirNodeId, MirOp};
    use crate::pir::{
        FunctionEntry, Handoff, HandoffKind, Package, PackageRole, PirGraph, ShardPartitionEntry,
        ShardRole, ShardTemplate, TensorSpec,
    };
    use crate::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};
    use std::collections::HashMap;

    // ─── Helpers ──────────────────────────────────────────────────────

    fn minimal_sir_graph() -> SirGraph {
        let input_id = SirNodeId("input".into());
        let weight_id = SirNodeId("weight".into());
        let output_id = SirNodeId("output".into());
        SirGraph {
            nodes: vec![
                SirNode {
                    id: weight_id,
                    op: SirOp::Const {
                        value_path: "weight.npy".into(),
                        dtype: MilDtype::Fp16,
                        palette_bits: None,
                    },
                    name: "weight".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: output_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "weight.npy".into(),
                        bias: Some("bias.npy".into()),
                        palette_bits: None,
                    },
                    name: "linear".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![input_id],
            outputs: vec![output_id],
        }
    }

    fn minimal_air_graph() -> AirGraph {
        let input_id = AirNodeId("input".into());
        let output_id = AirNodeId("output".into());
        AirGraph {
            nodes: vec![AirNode {
                id: output_id,
                op: AirOp::Linear {
                    input: input_id,
                    weight: "weight.npy".into(),
                    bias: Some("bias.npy".into()),
                },
                name: "linear".into(),
                sir_source: None,
                precision_override: None,
                legality_status: crate::air::LegalityStatus::Verified,
            }],
            inputs: vec![AirNodeId("input".into())],
            outputs: vec![AirNodeId("output".into())],
        }
    }

    fn minimal_mir_graph() -> MirGraph {
        let input_id = MirNodeId("input".into());
        let weight_id = MirNodeId("weight".into());
        let matmul_id = MirNodeId("matmul".into());
        MirGraph {
            nodes: vec![
                MirNode {
                    id: weight_id.clone(),
                    op: MirOp::MILConst {
                        name: "weight".into(),
                        value_path: "weight.npy".into(),
                        dtype: MilDtype::Fp16,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![64, 128],
                    compute_unit_hint: None,
                    air_source: None,
                    target_annotation: Default::default(),
                },
                MirNode {
                    id: matmul_id.clone(),
                    op: MirOp::MILMatMul {
                        name: "matmul".into(),
                        x: input_id.clone(),
                        y: weight_id,
                        transpose_y: false,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![1, 128],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                    target_annotation: Default::default(),
                },
            ],
            inputs: vec![input_id],
            outputs: vec![matmul_id],
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            shard_name: "test_shard".into(),
            input_shapes: HashMap::new(),
        }
    }

    fn minimal_pir_graph() -> PirGraph {
        PirGraph {
            packages: vec![Package {
                name: "test_package".into(),
                role: PackageRole::DecoderShard(ShardRole::Entry),
                compute_units: ComputeUnitHint::CPUAndNE,
                mil_program_ref: "test_package".into(),
                functions: vec![FunctionEntry {
                    name: "main".into(),
                    inputs: vec![TensorSpec {
                        name: "x".into(),
                        shape: vec![1, 64],
                        dtype: "fp16".into(),
                    }],
                    outputs: vec![TensorSpec {
                        name: "output".into(),
                        shape: vec![1, 128],
                        dtype: "fp16".into(),
                    }],
                    stateful: false,
                }],
            }],
            state_declarations: vec![],
            handoffs: vec![Handoff {
                from_package: "pkg_a".into(),
                to_package: "pkg_b".into(),
                tensor_name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
                handoff_kind: HandoffKind::TensorPassThrough,
                execution_order: 0,
                source_output_name: "output".into(),
                target_input_name: "x".into(),
            }],
            shard_template: Some(ShardTemplate {
                template_id: "test_template".into(),
                partition_spec: vec![ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 0,
                    compute_units: ComputeUnitHint::CPUAndNE,
                }],
                io_compute_units: None,
                sampler_compute_units: None,
                state_config: None,
                context_length: 0,
            }),
            context_length: 0,
            opset_version: crate::DEFAULT_OPSET_VERSION.into(),
            // T-115: Use DEFAULT_MINIMUM_DEPLOYMENT_TARGET instead of DEFAULT_OPSET_VERSION
            minimum_deployment_target: crate::DEFAULT_MINIMUM_DEPLOYMENT_TARGET.into(),
            kv_cache_layout: crate::sir::KvCacheLayout::default(),
            sampler_spec: None,
            io_model_spec: None,
        }
    }

    // ─── Roundtrip tests ──────────────────────────────────────────────

    #[test]
    fn test_serialize_deserialize_sir_roundtrip() {
        let graph = minimal_sir_graph();
        let bytes = serialize_sir(&graph).unwrap();
        let de: SirGraph = deserialize_sir(&bytes).unwrap();
        assert_eq!(de.nodes.len(), graph.nodes.len());
        assert_eq!(de.inputs.len(), graph.inputs.len());
        assert_eq!(de.outputs.len(), graph.outputs.len());
    }

    #[test]
    fn test_serialize_deserialize_air_roundtrip() {
        let graph = minimal_air_graph();
        let bytes = serialize_air(&graph).unwrap();
        let de: AirGraph = deserialize_air(&bytes).unwrap();
        assert_eq!(de.nodes.len(), graph.nodes.len());
        assert_eq!(de.inputs.len(), graph.inputs.len());
        assert_eq!(de.outputs.len(), graph.outputs.len());
    }

    #[test]
    fn test_serialize_deserialize_mir_roundtrip() {
        let graph = minimal_mir_graph();
        let bytes = serialize_mir(&graph).unwrap();
        let de: MirGraph = deserialize_mir(&bytes).unwrap();
        assert_eq!(de.nodes.len(), graph.nodes.len());
        assert_eq!(de.inputs.len(), graph.inputs.len());
        assert_eq!(de.outputs.len(), graph.outputs.len());
    }

    #[test]
    fn test_serialize_deserialize_pir_roundtrip() {
        let graph = minimal_pir_graph();
        let bytes = serialize_pir(&graph).unwrap();
        let de: PirGraph = deserialize_pir(&bytes).unwrap();
        assert_eq!(de.packages.len(), graph.packages.len());
        assert_eq!(de.handoffs.len(), graph.handoffs.len());
    }

    // ─── Error handling ───────────────────────────────────────────────

    #[test]
    fn test_deserialize_corrupt_bytes_returns_error() {
        let result: Result<SirGraph, String> = deserialize_graph(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("deserialization failed"));
    }

    #[test]
    fn test_deserialize_empty_bytes_returns_error() {
        let result: Result<MirGraph, String> = deserialize_graph(&[]);
        assert!(result.is_err());
    }

    // ─── Generic functions ────────────────────────────────────────────

    #[test]
    fn test_serialize_graph_generic() {
        let graph = minimal_mir_graph();
        let bytes = serialize_graph(&graph).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_deserialize_graph_generic() {
        let graph = minimal_mir_graph();
        let bytes = serialize_graph(&graph).unwrap();
        let de: MirGraph = deserialize_graph(&bytes).unwrap();
        assert_eq!(de.nodes.len(), graph.nodes.len());
    }

    // ─── Preservation tests ───────────────────────────────────────────

    #[test]
    fn test_sir_roundtrip_preserves_nodes() {
        let graph = minimal_sir_graph();
        let bytes = serialize_sir(&graph).unwrap();
        let de: SirGraph = deserialize_sir(&bytes).unwrap();

        // Verify node count
        assert_eq!(de.nodes.len(), 2);

        // Verify op types preserved
        assert!(matches!(de.nodes[0].op, SirOp::Const { .. }));
        assert!(matches!(de.nodes[1].op, SirOp::LinearProjection { .. }));

        // Verify metadata preserved
        assert!(matches!(de.nodes[0].metadata.task_origin, TaskOrigin::Synthetic));
    }

    #[test]
    fn test_mir_roundtrip_preserves_nodes() {
        let graph = minimal_mir_graph();
        let bytes = serialize_mir(&graph).unwrap();
        let de: MirGraph = deserialize_mir(&bytes).unwrap();

        // Verify node count preserved
        assert_eq!(de.nodes.len(), 2);

        // Verify op types preserved
        assert!(matches!(de.nodes[0].op, MirOp::MILConst { .. }));
        assert!(matches!(de.nodes[1].op, MirOp::MILMatMul { .. }));

        // Verify dtype preserved
        assert_eq!(de.nodes[0].dtype, MilDtype::Fp16);

        // Verify shard_name preserved
        assert_eq!(de.shard_name, "test_shard");
    }
}
