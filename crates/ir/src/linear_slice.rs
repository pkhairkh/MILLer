//! Linear Projection Slice
//!
//! End-to-end path for a synthetic linear projection task:
//! task spec → SIR → MIR → bridge payload.
//! This is the narrowest vertical slice proving the pipeline.
//!
//! Also includes the sharded linear pipeline path (S9.2):
//! task spec → per-shard SIR → per-shard MIR → per-shard bridge payload.
//! Each shard has explicit role semantics (Entry/Interior/Exit).
//!
//! Bridge payload types live in [`super::payload`] and shard-related types
//! live in [`super::shard_desc`]; both are re-exported here for backward
//! compatibility.

// Re-exports for backward compatibility: consumers that import from
// `ane_ir::linear_slice::{…}` continue to compile without changes.
pub use super::payload::*;
pub use super::shard_desc::*;

use crate::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirNodeId, MirOp};
use crate::sir::{
    SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, SirTargetAnnotation, TaskOrigin,
};
use crate::task_spec::{SyntheticTaskSpec, TaskOp};

/// Build a SIR graph from a synthetic linear projection task spec.
pub fn sir_from_linear_projection(spec: &SyntheticTaskSpec) -> Result<SirGraph, String> {
    // Extract dimensions from the task spec. The wildcard pattern is kept for
    // forward compatibility: when more TaskOp variants are added, this match
    // will need to handle them. For now, only LinearProjection is supported.
    let (_input_dim, _output_dim, _batch_size, _dtype) = match &spec.op {
        TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        TaskOp::LutProjection { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::DecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::ShardedDecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::Attention { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::MlpBlock { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        // ShardedLinearPipeline is handled by the sharded path, not single-shard
        #[allow(unreachable_patterns)]
        _ => return Err("Expected single-shard task type for sir_from_linear_projection".into()),
    };

    let input_id = SirNodeId("input".into());
    let weight_id = SirNodeId("weight".into());
    let bias_id = SirNodeId("bias".into());
    let output_id = SirNodeId("output".into());

    let nodes = vec![
        SirNode {
            id: weight_id.clone(),
            op: SirOp::Mul { x: SirNodeId(String::new()), y: SirNodeId(String::new()) },
            name: "weight".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
            target_annotation: SirTargetAnnotation::default(),
        },
        SirNode {
            id: bias_id.clone(),
            op: SirOp::Add { x: SirNodeId(String::new()), y: SirNodeId(String::new()) },
            name: "bias".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
            target_annotation: SirTargetAnnotation::default(),
        },
        SirNode {
            id: output_id.clone(),
            op: SirOp::LinearProjection {
                input: input_id.clone(),
                weight: "weight".into(),
                bias: Some("bias".into()),
            },
            name: "linear_out".into(),
            metadata: SirMetadata {
                task_origin: TaskOrigin::Synthetic,
                model_id: None,
                quality_contract: None,
                precision_override: None,
            },
            target_annotation: SirTargetAnnotation::default(),
        },
    ];

    Ok(SirGraph { nodes, inputs: vec![input_id], outputs: vec![output_id] })
}

/// Lower a linear projection SIR graph directly to MIR.
pub fn lower_linear_projection_to_mir(
    spec: &SyntheticTaskSpec,
    shard_name: &str,
) -> Result<MirGraph, String> {
    let (input_dim, output_dim, batch_size, dtype) = match &spec.op {
        TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        TaskOp::LutProjection { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::DecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::ShardedDecodeStep { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::Attention { embed_dim, batch_size, dtype, .. } => {
            (*embed_dim, *embed_dim, *batch_size, dtype.clone())
        }
        TaskOp::MlpBlock { input_dim, output_dim, batch_size, dtype, .. } => {
            (*input_dim, *output_dim, *batch_size, dtype.clone())
        }
        #[allow(unreachable_patterns)]
        _ => return Err("Expected single-shard task type for MIR lowering".into()),
    };

    let mil_dtype = match dtype.as_str() {
        "fp16" => MilDtype::Fp16,
        "fp32" => MilDtype::Fp32,
        "int4" => MilDtype::Int4,
        "uint4" => MilDtype::UInt4,
        "e4m3" => MilDtype::E4M3,
        "e5m2" => MilDtype::E5M2,
        "uint16" => MilDtype::UInt16,
        _ => {
            log::warn!("linear_slice: unrecognized dtype string '{}', defaulting to Fp16", dtype);
            MilDtype::Fp16
        }
    };

    let weight_id = MirNodeId("weight".into());
    let bias_id = MirNodeId("bias".into());
    let input_id = MirNodeId("input".into());
    let matmul_id = MirNodeId("matmul".into());
    let add_id = MirNodeId("add".into());

    let nodes = vec![
        MirNode {
            id: weight_id.clone(),
            op: MirOp::MILConst {
                name: "weight".into(),
                value_path: "weight.npy".into(),
                dtype: mil_dtype.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![input_dim, output_dim],
            compute_unit_hint: None,
            air_source: None,
            target_annotation: Default::default(),
        },
        MirNode {
            id: bias_id.clone(),
            op: MirOp::MILConst {
                name: "bias".into(),
                value_path: "bias.npy".into(),
                dtype: mil_dtype.clone(),
            },
            dtype: mil_dtype.clone(),
            shape: vec![output_dim],
            compute_unit_hint: None,
            air_source: None,
            target_annotation: Default::default(),
        },
        MirNode {
            id: matmul_id.clone(),
            op: MirOp::MILMatMul {
                name: "matmul".into(),
                x: input_id.clone(),
                y: weight_id.clone(),
                transpose_y: false,
            },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, output_dim],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
            target_annotation: Default::default(),
        },
        MirNode {
            id: add_id.clone(),
            op: MirOp::MILAdd { name: "add".into(), x: matmul_id.clone(), y: bias_id.clone() },
            dtype: mil_dtype.clone(),
            shape: vec![batch_size, output_dim],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
            target_annotation: Default::default(),
        },
    ];

    Ok(MirGraph {
        nodes,
        inputs: vec![input_id],
        outputs: vec![add_id],
        opset_version: crate::DEFAULT_OPSET_VERSION.into(),
        shard_name: shard_name.into(),
        input_shapes: std::collections::HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::{PackageRole, ShardRole};
    use crate::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};

    fn test_sharded_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_shard".into(),
            family: "ShardedLinearPipeline".into(),
            description: None,
            op: TaskOp::ShardedLinearPipeline {
                input_dim: 64,
                hidden_dim: 48,
                output_dim: 32,
                batch_size: 1,
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
    fn test_sharded_pipeline_three_shards() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        assert_eq!(shards.len(), 3, "ShardedLinearPipeline must produce 3 shards");

        assert_eq!(shards[0].role, ShardRole::Entry);
        assert_eq!(shards[0].input_dim, 64);
        assert_eq!(shards[0].output_dim, 48);
        assert_eq!(shards[0].compute_units, ComputeUnitHint::CPUAndNE);

        assert_eq!(shards[1].role, ShardRole::Interior);
        assert_eq!(shards[1].input_dim, 48);
        assert_eq!(shards[1].output_dim, 48);
        assert_eq!(shards[1].compute_units, ComputeUnitHint::CPUAndNE);

        assert_eq!(shards[2].role, ShardRole::Exit);
        assert_eq!(shards[2].input_dim, 48);
        assert_eq!(shards[2].output_dim, 32);
        assert_eq!(shards[2].compute_units, ComputeUnitHint::CPUAndNE);
    }

    #[test]
    fn test_sharded_pipeline_mir_per_shard() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();

        for shard in &shards {
            let mir = lower_shard_to_mir(shard, 1, "fp16").unwrap();
            assert_eq!(
                mir.nodes.len(),
                4,
                "Each shard MIR must have 4 nodes (weight, bias, matmul, add)"
            );
            assert_eq!(mir.inputs.len(), 1);
            assert_eq!(mir.outputs.len(), 1);
            assert_eq!(mir.shard_name, shard.shard_name);
        }
    }

    #[test]
    fn test_sharded_pipeline_pir() {
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        assert_eq!(pir.packages.len(), 3, "PIR must have 3 packages");

        // Verify package roles
        assert!(matches!(pir.packages[0].role, PackageRole::DecoderShard(ShardRole::Entry)));
        assert!(matches!(pir.packages[1].role, PackageRole::DecoderShard(ShardRole::Interior)));
        assert!(matches!(pir.packages[2].role, PackageRole::DecoderShard(ShardRole::Exit)));

        // Verify handoffs with concrete runtime semantics
        assert_eq!(pir.handoffs.len(), 2, "3 shards must have 2 handoffs");
        assert_eq!(pir.handoffs[0].from_package, "test_shard_entry");
        assert_eq!(pir.handoffs[0].to_package, "test_shard_interior");
        assert_eq!(pir.handoffs[1].from_package, "test_shard_interior");
        assert_eq!(pir.handoffs[1].to_package, "test_shard_exit");

        // Verify concrete handoff semantics (Sprint 17, S17.1)
        assert_eq!(pir.handoffs[0].handoff_kind, crate::pir::HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[0].execution_order, 0);
        assert_eq!(pir.handoffs[0].source_output_name, "output");
        assert_eq!(pir.handoffs[0].target_input_name, "x");
        assert_eq!(pir.handoffs[1].handoff_kind, crate::pir::HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[1].execution_order, 1);
        assert_eq!(pir.handoffs[1].source_output_name, "output");
        assert_eq!(pir.handoffs[1].target_input_name, "x");

        // Verify shard template
        assert!(pir.shard_template.is_some());
        let template = pir.shard_template.as_ref().unwrap();
        assert_eq!(template.partition_spec.len(), 3);
    }

    #[test]
    fn test_concrete_handoff_execution_order() {
        // Verify that handoff execution orders are sequential and start from 0
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        let orders: Vec<usize> = pir.handoffs.iter().map(|h| h.execution_order).collect();
        assert_eq!(
            orders,
            vec![0, 1],
            "Handoff execution orders must be sequential starting from 0"
        );
    }

    #[test]
    fn test_concrete_handoff_source_target_names() {
        // Verify that handoff source_output_name and target_input_name
        // reference actual function I/O names in the packages
        let spec = test_sharded_spec();
        let pir = build_sharded_pipeline_pir(&spec).unwrap();

        for handoff in &pir.handoffs {
            // Find the source package
            let source_pkg = pir
                .packages
                .iter()
                .find(|p| p.name == handoff.from_package)
                .expect("Source package must exist");
            let target_pkg = pir
                .packages
                .iter()
                .find(|p| p.name == handoff.to_package)
                .expect("Target package must exist");

            // Verify source output name matches a function output
            let source_outputs: Vec<&String> = source_pkg
                .functions
                .iter()
                .flat_map(|f| f.outputs.iter().map(|o| &o.name))
                .collect();
            assert!(
                source_outputs.contains(&&handoff.source_output_name),
                "Source output '{}' must exist in package '{}' outputs",
                handoff.source_output_name,
                handoff.from_package
            );

            // Verify target input name matches a function input
            let target_inputs: Vec<&String> = target_pkg
                .functions
                .iter()
                .flat_map(|f| f.inputs.iter().map(|i| &i.name))
                .collect();
            assert!(
                target_inputs.contains(&&handoff.target_input_name),
                "Target input '{}' must exist in package '{}' inputs",
                handoff.target_input_name,
                handoff.to_package
            );
        }
    }

    #[test]
    fn test_shard_payload_roundtrip() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        let shard = &shards[0];

        let payload = ShardedShardPayload::from_shard(
            shard,
            &spec.name,
            &spec.family,
            1,
            "fp16",
            "/tmp/test",
            42,
        );

        assert_eq!(payload.shard_role, "Entry");
        assert_eq!(payload.input_dim, 64);
        assert_eq!(payload.output_dim, 48);
        assert_eq!(payload.compute_units, "CPU_AND_NE");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_sharded_pipeline_rejects_linear_projection() {
        let spec = SyntheticTaskSpec {
            name: "test".into(),
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
                metrics: vec![],
            },
        };
        let result = sharded_pipeline_shards(&spec);
        assert!(result.is_err(), "sharded_pipeline_shards must reject LinearProjection tasks");
    }

    // ─── Precision Override Propagation Tests (Sprint 18) ─────────────────

    fn test_linear_spec_fp16() -> SyntheticTaskSpec {
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
    fn test_payload_dtype_override_changes_bridge_dtype() {
        let spec = test_linear_spec_fp16();

        // Without override: uses spec's fp16
        let payload_no = LinearProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16", "Without override, dtype should be fp16");

        // With fp32 override: uses overridden dtype
        let payload_fp32 =
            LinearProjectionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32"))
                .unwrap();
        assert_eq!(
            payload_fp32.dtype, "fp32",
            "With fp32 override, bridge payload dtype must be fp32"
        );

        // Function descriptors must also reflect the overridden dtype
        assert_eq!(
            payload_fp32.functions[0].inputs[0].dtype, "fp32",
            "Function input dtype must reflect override"
        );
        assert_eq!(
            payload_fp32.functions[0].outputs[0].dtype, "fp32",
            "Function output dtype must reflect override"
        );
    }

    #[test]
    fn test_payload_dtype_no_override_preserves_spec() {
        let spec = test_linear_spec_fp16();
        let payload =
            LinearProjectionPayload::from_spec_with_override(&spec, "/tmp/test", None).unwrap();
        assert_eq!(payload.dtype, "fp16", "Without override, dtype must match spec default");
    }

    #[test]
    fn test_shard_payload_dtype_override() {
        let spec = test_sharded_spec();
        let shards = sharded_pipeline_shards(&spec).unwrap();
        let shard = &shards[0];

        // Without override
        let payload_no = ShardedShardPayload::from_shard(
            shard,
            &spec.name,
            &spec.family,
            1,
            "fp16",
            "/tmp/test",
            42,
        );
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override
        let payload_fp32 = ShardedShardPayload::from_shard_with_override(
            shard,
            &spec.name,
            &spec.family,
            1,
            "fp16",
            "/tmp/test",
            42,
            Some("fp32"),
        );
        assert_eq!(
            payload_fp32.dtype, "fp32",
            "Shard payload with fp32 override must use fp32 dtype"
        );
        assert_eq!(payload_fp32.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_fp32.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_precision_override_propagates_full_pipeline() {
        // End-to-end test: SIR with precision_override → AIR → MIR
        // This proves that precision adaptation propagates through the IR pipeline.
        // The bridge payload propagation is tested separately in linear_slice tests.
        let spec = test_linear_spec_fp16();

        // Step 1: Build SIR (initially no precision override)
        let sir = sir_from_linear_projection(&spec).unwrap();

        // Step 2: Simulate precision policy setting override on the linear_out node
        let mut sir_adapted = sir.clone();
        for node in &mut sir_adapted.nodes {
            if node.name == "linear_out" {
                node.metadata.precision_override = Some("fp32".to_string());
            }
        }

        // Step 3: Verify SIR override is set
        let linear_sir_node = sir_adapted
            .nodes
            .iter()
            .find(|n| n.name == "linear_out")
            .expect("Expected linear_out SIR node");
        assert_eq!(
            linear_sir_node.metadata.precision_override,
            Some("fp32".to_string()),
            "Precision override must be set on SIR node"
        );

        // Step 4: Bridge payload with fp32 override must use fp32 dtype
        let payload =
            LinearProjectionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32"))
                .unwrap();
        assert_eq!(
            payload.dtype, "fp32",
            "Bridge payload dtype must reflect the precision adaptation"
        );
    }

    // ─── Sprint 20 — Dedicated LUT Path Tests ──────────────────────────────

    fn test_lut_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_lut".into(),
            family: "LutProjection".into(),
            description: None,
            op: TaskOp::LutProjection {
                vocab_size: 32000,
                embed_dim: 512,
                num_groups: 64,
                lut_bitwidth: 4,
                batch_size: 1,
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
    fn test_lut_payload_from_spec_succeeds() {
        let spec = test_lut_spec();
        let payload = LutProjectionPayload::from_spec(&spec, "/tmp/lut_test").unwrap();
        assert_eq!(
            payload.command, "emit_lut_projection",
            "LUT payload must use dedicated emit_lut_projection command"
        );
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
        assert_eq!(payload.vocab_size, 32000);
        assert_eq!(payload.embed_dim, 512);
        assert_eq!(payload.num_groups, 64);
        assert_eq!(payload.lut_bitwidth, 4);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
        assert_eq!(payload.family, "LutProjection");
    }

    #[test]
    fn test_lut_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = LutProjectionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "LutProjectionPayload must reject LinearProjection specs");
    }

    #[test]
    fn test_linear_payload_rejects_lut_spec() {
        let spec = test_lut_spec();
        let result = LinearProjectionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(),
            "LinearProjectionPayload must reject LutProjection specs — use LutProjectionPayload instead");
    }

    #[test]
    fn test_linear_vs_lut_payload_command_divergence() {
        // S20.4: Prove that linear and LUT compile paths generate
        // different bridge commands/payloads.
        let linear_spec = test_linear_spec_fp16();
        let lut_spec = test_lut_spec();

        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();
        let lut_payload = LutProjectionPayload::from_spec(&lut_spec, "/tmp/lut_test").unwrap();

        // Commands must differ
        assert_eq!(linear_payload.command, "emit_linear_projection");
        assert_eq!(lut_payload.command, "emit_lut_projection");
        assert_ne!(
            linear_payload.command, lut_payload.command,
            "Linear and LUT payloads must use different bridge commands"
        );

        // Payloads must have different fields
        // Linear has input_dim/output_dim; LUT has vocab_size/embed_dim/num_groups/lut_bitwidth
        let linear_json = serde_json::to_value(&linear_payload).unwrap();
        let lut_json = serde_json::to_value(&lut_payload).unwrap();

        // Verify LUT-specific fields are present
        assert!(lut_json.get("vocab_size").is_some(), "LUT payload must have vocab_size field");
        assert!(lut_json.get("lut_bitwidth").is_some(), "LUT payload must have lut_bitwidth field");
        assert!(lut_json.get("num_groups").is_some(), "LUT payload must have num_groups field");

        // Verify linear-specific fields are absent from LUT payload
        assert!(lut_json.get("input_dim").is_none(), "LUT payload must NOT have input_dim field");
        assert!(lut_json.get("output_dim").is_none(), "LUT payload must NOT have output_dim field");

        // Verify LUT-specific fields are absent from linear payload
        assert!(
            linear_json.get("vocab_size").is_none(),
            "Linear payload must NOT have vocab_size field"
        );
        assert!(
            linear_json.get("lut_bitwidth").is_none(),
            "Linear payload must NOT have lut_bitwidth field"
        );
    }

    #[test]
    fn test_lut_payload_deterministic_serialization() {
        let spec = test_lut_spec();
        let payload1 = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        let payload2 = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();

        let json1 = serde_json::to_string(&payload1).unwrap();
        let json2 = serde_json::to_string(&payload2).unwrap();
        assert_eq!(json1, json2, "LUT payload serialization must be deterministic");
    }

    #[test]
    fn test_lut_payload_function_descriptors() {
        let spec = test_lut_spec();
        let payload = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();

        // LUT payload function descriptor has int32 indices input (not float)
        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "indices");
        assert_eq!(
            payload.functions[0].inputs[0].dtype, "int32",
            "LUT function input must be int32 indices"
        );
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].dtype, "fp16");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 512]);
    }

    // ─── Decode-Step Payload Divergence Tests ─────────────────────────────

    fn test_decode_step_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_decode".into(),
            family: "DecodeStep".into(),
            description: None,
            op: TaskOp::DecodeStep {
                embed_dim: 128,
                num_heads: 4,
                head_dim: 32,
                kv_len: 64,
                batch_size: 1,
                kv_heads: 4,            // num_heads (no GQA)
                intermediate_size: 512, // embed_dim * 4
                vocab_size: 0,
                dtype: "fp16".into(),
                uses_rope: true,
                has_qk_norm: false,
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".into()],
            },
        }
    }

    #[test]
    fn test_decode_step_payload_from_spec_succeeds() {
        let spec = test_decode_step_spec();
        let payload = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_stateful_decode_step");
        assert_eq!(payload.family, "DecodeStep");
        assert_eq!(payload.embed_dim, 128);
        assert_eq!(payload.num_heads, 4);
        assert_eq!(payload.head_dim, 32);
        assert_eq!(payload.kv_len, 64);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
        assert_eq!(payload.compute_units, "CPU_AND_NE");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_decode_step_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = DecodeStepPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "DecodeStepPayload must reject LinearProjection specs");
    }

    #[test]
    fn test_decode_step_payload_command_differs_from_linear() {
        let linear_spec = test_linear_spec_fp16();
        let decode_spec = test_decode_step_spec();
        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();
        let decode_payload = DecodeStepPayload::from_spec(&decode_spec, "/tmp/test").unwrap();
        assert_ne!(
            linear_payload.command, decode_payload.command,
            "Decode-step and linear projection must use different bridge commands"
        );
    }

    #[test]
    fn test_decode_step_payload_command_differs_from_lut() {
        let lut_spec = SyntheticTaskSpec {
            name: "test_lut".into(),
            family: "LutProjection".into(),
            description: None,
            op: TaskOp::LutProjection {
                vocab_size: 16,
                embed_dim: 128,
                num_groups: 16,
                lut_bitwidth: 4,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 3,
                measured_iterations: 10,
                metrics: vec![],
            },
        };
        let decode_spec = test_decode_step_spec();
        let lut_payload = LutProjectionPayload::from_spec(&lut_spec, "/tmp/test").unwrap();
        let decode_payload = DecodeStepPayload::from_spec(&decode_spec, "/tmp/test").unwrap();
        assert_ne!(
            lut_payload.command, decode_payload.command,
            "Decode-step and LUT projection must use different bridge commands"
        );
    }

    #[test]
    fn test_decode_step_payload_deterministic_serialization() {
        let spec = test_decode_step_spec();
        let payload1 = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        let payload2 = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        let json1 = serde_json::to_string(&payload1).unwrap();
        let json2 = serde_json::to_string(&payload2).unwrap();
        assert_eq!(json1, json2, "Same spec must produce deterministic serialization");
    }

    #[test]
    fn test_decode_step_payload_function_descriptors() {
        let spec = test_decode_step_spec();
        let payload = DecodeStepPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.functions.len(), 1);
        let func = &payload.functions[0];
        assert_eq!(func.name, "main");
        assert_eq!(func.inputs.len(), 1);
        assert_eq!(func.inputs[0].name, "x");
        assert_eq!(func.inputs[0].shape, vec![1, 128]);
        assert_eq!(func.outputs.len(), 1);
        assert_eq!(func.outputs[0].name, "output");
        assert_eq!(func.outputs[0].shape, vec![1, 128]);
        assert!(func.stateful); // DecodeStep is stateful (manages KV cache)
    }

    // ─── MLP Block Payload Divergence Tests (Sprint 28) ──────────────────

    fn test_mlp_block_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_mlp".into(),
            family: "MlpBlock".into(),
            description: None,
            op: TaskOp::MlpBlock {
                input_dim: 128,
                hidden_dim: 512,
                output_dim: 128,
                activation: "gelu".into(),
                batch_size: 1,
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
    fn test_mlp_block_payload_creation() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_mlp_block");
        assert_eq!(payload.family, "MlpBlock");
        assert_eq!(payload.input_dim, 128);
        assert_eq!(payload.hidden_dim, 512);
        assert_eq!(payload.output_dim, 128);
        assert_eq!(payload.activation, "gelu");
        assert_eq!(payload.bridge_version, BRIDGE_VERSION);
    }

    #[test]
    fn test_mlp_block_payload_rejects_linear() {
        let spec = test_linear_spec_fp16();
        let result = MlpBlockPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "MlpBlockPayload must reject LinearProjection tasks");
    }

    #[test]
    fn test_mlp_block_payload_command_diverges_from_linear() {
        let linear_spec = test_linear_spec_fp16();
        let linear_payload = LinearProjectionPayload::from_spec(&linear_spec, "/tmp/test").unwrap();

        let mlp_spec = test_mlp_block_spec();
        let mlp_payload = MlpBlockPayload::from_spec(&mlp_spec, "/tmp/test").unwrap();

        assert_ne!(
            linear_payload.command, mlp_payload.command,
            "MLP block and linear projection must use different bridge commands"
        );
        assert_eq!(linear_payload.command, "emit_linear_projection");
        assert_eq!(mlp_payload.command, "emit_mlp_block");
    }

    #[test]
    fn test_mlp_block_payload_deterministic_serialization() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();

        let json1 = serde_json::to_string(&payload).unwrap();
        let json2 = serde_json::to_string(&payload).unwrap();
        assert_eq!(json1, json2, "Serialization must be deterministic");
    }

    #[test]
    fn test_mlp_block_payload_function_descriptors() {
        let spec = test_mlp_block_spec();
        let payload = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();

        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "x");
        assert_eq!(payload.functions[0].inputs[0].shape, vec![1, 128]);
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 128]);
        assert!(!payload.functions[0].stateful);
    }

    // ─── Attention Payload Tests (Sprint 29) ─────────────────────────────────

    fn test_attention_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "attn_128h4_s32_b1_fp16".into(),
            family: "Attention".into(),
            description: None,
            op: TaskOp::Attention {
                embed_dim: 128,
                num_heads: 4,
                head_dim: 32,
                seq_len: 32,
                batch_size: 1,
                dtype: "fp16".into(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".into(), "Drift".into()],
            },
        }
    }

    #[test]
    fn test_attention_payload_from_spec_succeeds() {
        let spec = test_attention_spec();
        let payload = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload.command, "emit_attention");
        assert_eq!(payload.embed_dim, 128);
        assert_eq!(payload.num_heads, 4);
        assert_eq!(payload.head_dim, 32);
        assert_eq!(payload.seq_len, 32);
        assert_eq!(payload.batch_size, 1);
        assert_eq!(payload.dtype, "fp16");
    }

    #[test]
    fn test_attention_payload_rejects_linear_spec() {
        let spec = test_linear_spec_fp16();
        let result = AttentionPayload::from_spec(&spec, "/tmp/test");
        assert!(result.is_err(), "AttentionPayload must reject LinearProjection tasks");
    }

    #[test]
    fn test_attention_payload_dtype_override() {
        let spec = test_attention_spec();

        // Without override: uses spec's fp16
        let payload_no = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override: uses overridden dtype
        let payload_yes =
            AttentionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32")).unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
        assert_eq!(payload_yes.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_yes.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_attention_payload_function_descriptors() {
        let spec = test_attention_spec();
        let payload = AttentionPayload::from_spec(&spec, "/tmp/test").unwrap();

        assert_eq!(payload.functions.len(), 1);
        assert_eq!(payload.functions[0].name, "main");
        assert_eq!(payload.functions[0].inputs.len(), 1);
        assert_eq!(payload.functions[0].inputs[0].name, "x");
        // Attention input shape: [batch_size, seq_len, embed_dim]
        assert_eq!(payload.functions[0].inputs[0].shape, vec![1, 32, 128]);
        assert_eq!(payload.functions[0].outputs.len(), 1);
        assert_eq!(payload.functions[0].outputs[0].name, "output");
        assert_eq!(payload.functions[0].outputs[0].shape, vec![1, 32, 128]);
        assert!(!payload.functions[0].stateful);
    }

    #[test]
    fn test_mlp_block_payload_dtype_override() {
        let spec = test_mlp_block_spec();

        // Without override: uses spec's fp16
        let payload_no = MlpBlockPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        // With fp32 override: uses overridden dtype
        let payload_yes =
            MlpBlockPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32")).unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
        assert_eq!(payload_yes.functions[0].inputs[0].dtype, "fp32");
        assert_eq!(payload_yes.functions[0].outputs[0].dtype, "fp32");
    }

    #[test]
    fn test_lut_projection_payload_dtype_override() {
        let spec = test_lut_spec();
        let payload_no = LutProjectionPayload::from_spec(&spec, "/tmp/test").unwrap();
        assert_eq!(payload_no.dtype, "fp16");

        let payload_yes =
            LutProjectionPayload::from_spec_with_override(&spec, "/tmp/test", Some("fp32"))
                .unwrap();
        assert_eq!(payload_yes.dtype, "fp32");
    }

    #[test]
    fn test_sir_accepts_attention() {
        let spec = test_attention_spec();
        let sir = sir_from_linear_projection(&spec);
        assert!(sir.is_ok(), "sir_from_linear_projection must accept Attention tasks");
    }

    #[test]
    fn test_mir_accepts_attention() {
        let spec = test_attention_spec();
        let mir = lower_linear_projection_to_mir(&spec, "attn_shard_0");
        assert!(mir.is_ok(), "lower_linear_projection_to_mir must accept Attention tasks");
    }
}
