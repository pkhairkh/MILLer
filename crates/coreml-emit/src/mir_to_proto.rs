//! MIR-to-Proto Conversion
//!
//! Converts MIR graph representations into Core ML protobuf model structures.
//! This is the bridge between the compiler's IR and the Core ML serialization
//! format.
//!
//! ## Conversion Strategy
//!
//! Each `MirOpCompat` variant maps to exactly one MIL operation in the
//! protobuf format. The conversion preserves:
//! - Operation names (SSA value names)
//! - Operand references (input/output edges)
//! - Weight data (stored in weight.bin, referenced by offset)
//! - Type information (dtype, shape)
//!
//! ## Weight Handling
//!
//! Constants (`MirOpCompat::Const`) are converted to weight entries
//! that will be stored in `weight.bin`. The protobuf references them
//! by offset and size, not by inline data.
//!
//! For shared weights across functions, the same weight name produces
//! the same offset — the `WeightBinBuilder` deduplicates automatically.

use anyhow::Result;
use ane_coreml_proto::{
    CoreMlModel, CoreMlFunction, CoreMlDataType, CoreMlComputeUnit, SpecVersion,
    TensorDesc, WeightEntry, SharedWeightRef, ModelDescriptionCompat,
    mir_compat::{MirGraphCompat, MirOpCompat, MilDtypeCompat},
};
use prost::Message;

/// Convert a single-function MIR graph to a CoreMlModel.
pub fn convert_mir_to_proto(
    graph: &MirGraphCompat,
    spec_version: SpecVersion,
    compute_unit: CoreMlComputeUnit,
) -> Result<CoreMlModel> {
    convert_mir_to_proto_multifunction(
        std::slice::from_ref(graph),
        &[],
        spec_version,
        compute_unit,
    )
}

/// Convert one or more MIR graphs to a multi-function CoreMlModel.
///
/// When `shared_weight_names` is non-empty, weights with those names
/// will be deduplicated across functions — each shared weight appears
/// once in `weight.bin` and is referenced by all functions that use it.
pub fn convert_mir_to_proto_multifunction(
    graphs: &[MirGraphCompat],
    shared_weight_names: &[String],
    spec_version: SpecVersion,
    compute_unit: CoreMlComputeUnit,
) -> Result<CoreMlModel> {
    let mut functions = Vec::new();
    let mut all_weights = Vec::new();
    let mut shared_weights = Vec::new();
    let shared_name_set: std::collections::HashSet<String> =
        shared_weight_names.iter().cloned().collect();

    // Track which functions reference which shared weights
    let mut shared_weight_refs: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for graph in graphs {
        // Extract weights from constants
        let mut graph_weights = Vec::new();
        let mut graph_inputs = Vec::new();
        let mut graph_outputs = Vec::new();
        let mut graph_states = Vec::new();

        for op in &graph.ops {
            match op {
                MirOpCompat::Const { name, data, dtype, shape } => {
                    let coreml_dtype = CoreMlDataType::from_mir_dtype(dtype);
                    let is_shared = shared_name_set.contains(name);

                    graph_weights.push(WeightEntry {
                        name: name.clone(),
                        offset: 0, // Will be set by WeightBinBuilder
                        size: data.len() as u64,
                        shape: shape.iter().map(|&d| d as u64).collect(),
                        dtype: coreml_dtype,
                        data: data.clone(),
                    });

                    if is_shared {
                        shared_weight_refs
                            .entry(name.clone())
                            .or_default()
                            .push(graph.function_name.clone());
                    }
                }
                MirOpCompat::ReadState { name: _, state_id, shape, dtype } => {
                    graph_states.push(TensorDesc {
                        name: state_id.clone(),
                        shape: shape.iter().map(|&d| d as u64).collect(),
                        dtype: CoreMlDataType::from_mir_dtype(dtype),
                        is_state: true,
                    });
                }
                _ => {}
            }
        }

        // Build input/output descriptions from the graph
        for input_name in &graph.inputs {
            // Find the input's dtype and shape from the graph ops
            graph_inputs.push(TensorDesc {
                name: input_name.clone(),
                shape: vec![], // Shape would be derived from actual op analysis
                dtype: CoreMlDataType::Float16, // Default
                is_state: false,
            });
        }
        for output_name in &graph.outputs {
            graph_outputs.push(TensorDesc {
                name: output_name.clone(),
                shape: vec![], // Shape would be derived from actual op analysis
                dtype: CoreMlDataType::Float16, // Default
                is_state: false,
            });
        }

        functions.push(CoreMlFunction {
            name: graph.function_name.clone(),
            inputs: graph_inputs,
            outputs: graph_outputs,
            states: graph_states,
            operations: graph.ops.clone(),
        });

        all_weights.extend(graph_weights);
    }

    // Build shared weight references
    for (weight_name, referencing_functions) in shared_weight_refs {
        if let Some(weight_entry) = all_weights.iter().find(|w| w.name == weight_name) {
            shared_weights.push(SharedWeightRef {
                weight: weight_entry.clone(),
                referencing_functions,
            });
        }
    }

    // Default function name
    let default_function_name = graphs
        .first()
        .map(|g| g.function_name.clone())
        .unwrap_or_else(|| "main".to_string());

    // Model description uses the default function's I/O
    let default_fn = functions.first();
    let description = ModelDescriptionCompat {
        inputs: default_fn.map(|f| f.inputs.clone()).unwrap_or_default(),
        outputs: default_fn.map(|f| f.outputs.clone()).unwrap_or_default(),
        states: default_fn.map(|f| f.states.clone()).unwrap_or_default(),
    };

    Ok(CoreMlModel {
        spec_version,
        description,
        functions,
        default_function_name,
        weights: all_weights,
        shared_weights,
        compute_unit,
        user_defined_metadata: std::collections::HashMap::new(),
    })
}

/// Serialize a CoreMlModel to protobuf bytes.
///
/// This produces the raw bytes that will be written as `model.mlmodel`
/// inside the mlpackage directory. The serialization follows the
/// Core ML protobuf format as defined in Model.proto.
///
/// Uses prost's `Message::encode_to_vec()` for real protobuf serialization,
/// converting the hand-written `CoreMlModel` into a `proto::Model` first.
pub fn model_to_protobuf_bytes(
    model: &CoreMlModel,
    weight_entries: &[WeightEntry],
) -> Result<Vec<u8>> {
    let proto_model = ane_coreml_proto::convert_to_proto_model(model, weight_entries);
    Ok(proto_model.encode_to_vec())
}

/// Build a linear projection MIR graph (for testing and as a
/// convenient entry point for proto-direct emission).
pub fn build_linear_projection_mir(
    _task_name: &str,
    input_dim: usize,
    output_dim: usize,
    _batch_size: usize,
    dtype: MilDtypeCompat,
    seed: u64,
) -> MirGraphCompat {
    // Generate deterministic weight data using a simple PRNG
    let weight_size = input_dim * output_dim;
    let bias_size = output_dim;

    let coreml_dtype = CoreMlDataType::from_mir_dtype(&dtype);
    let element_size = coreml_dtype.element_size();

    // Simple deterministic weight generation (matching the Python emitter's np.random.seed)
    let weight_data = generate_deterministic_data(weight_size, element_size, seed);
    let bias_data = generate_deterministic_data(bias_size, element_size, seed + 1);

    MirGraphCompat {
        ops: vec![
            MirOpCompat::Const {
                name: "weight".to_string(),
                data: weight_data,
                dtype: dtype,
                shape: vec![output_dim, input_dim], // mb.linear convention: [out, in]
            },
            MirOpCompat::Const {
                name: "bias".to_string(),
                data: bias_data,
                dtype: dtype,
                shape: vec![output_dim],
            },
            MirOpCompat::Linear {
                name: "output".to_string(),
                x: "x".to_string(),
                weight_name: "weight".to_string(),
                bias_name: Some("bias".to_string()),
            },
        ],
        inputs: vec!["x".to_string()],
        outputs: vec!["output".to_string()],
        opset_version: "iOS18".to_string(),
        function_name: "main".to_string(),
    }
}

/// Build a multi-function model with shared weights (for testing
/// proto-direct weight sharing).
pub fn build_multifunction_shared_weights_mir(
    _task_name: &str,
    embed_dim: usize,
    _batch_size: usize,
    dtype: MilDtypeCompat,
    seed: u64,
) -> (Vec<MirGraphCompat>, Vec<String>) {
    let coreml_dtype = CoreMlDataType::from_mir_dtype(&dtype);
    let element_size = coreml_dtype.element_size();

    // Shared weight: used by both embedding and decode_step functions
    let shared_weight_size = embed_dim * embed_dim;
    let shared_weight_data = generate_deterministic_data(shared_weight_size, element_size, seed);

    // Embedding function: uses shared weight for a linear projection
    let embedding_graph = MirGraphCompat {
        ops: vec![
            MirOpCompat::Const {
                name: "shared_projection_weight".to_string(),
                data: shared_weight_data.clone(),
                dtype: dtype,
                shape: vec![embed_dim, embed_dim],
            },
            MirOpCompat::Linear {
                name: "embedding_output".to_string(),
                x: "x".to_string(),
                weight_name: "shared_projection_weight".to_string(),
                bias_name: None,
            },
        ],
        inputs: vec!["x".to_string()],
        outputs: vec!["embedding_output".to_string()],
        opset_version: "iOS18".to_string(),
        function_name: "embedding".to_string(),
    };

    // Decode step function: uses the SAME shared weight
    let decode_step_graph = MirGraphCompat {
        ops: vec![
            MirOpCompat::Const {
                name: "shared_projection_weight".to_string(),
                data: shared_weight_data,
                dtype: dtype,
                shape: vec![embed_dim, embed_dim],
            },
            MirOpCompat::Linear {
                name: "decode_output".to_string(),
                x: "hidden".to_string(),
                weight_name: "shared_projection_weight".to_string(),
                bias_name: None,
            },
        ],
        inputs: vec!["hidden".to_string()],
        outputs: vec!["decode_output".to_string()],
        opset_version: "iOS18".to_string(),
        function_name: "decode_step".to_string(),
    };

    let shared_weight_names = vec!["shared_projection_weight".to_string()];

    (vec![embedding_graph, decode_step_graph], shared_weight_names)
}

/// Generate deterministic data for weight tensors.
///
/// This produces pseudo-random data using a simple LCG PRNG,
/// matching the pattern used by np.random.seed() in the Python emitter.
fn generate_deterministic_data(num_elements: usize, element_size: usize, seed: u64) -> Vec<u8> {
    let total_bytes = num_elements * element_size;
    let mut data = Vec::with_capacity(total_bytes);
    let mut state = seed;

    // Simple LCG PRNG
    for _ in 0..total_bytes {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push((state >> 56) as u8);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_single_function() {
        let graph = build_linear_projection_mir(
            "test_linear", 64, 32, 1, MilDtypeCompat::Fp16, 42,
        );

        let model = convert_mir_to_proto(&graph, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe)
            .unwrap();

        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "main");
        assert_eq!(model.default_function_name, "main");
        assert_eq!(model.weights.len(), 2); // weight + bias
        assert_eq!(model.shared_weights.len(), 0);
    }

    #[test]
    fn test_convert_multifunction_with_shared_weights() {
        let (graphs, shared_names) = build_multifunction_shared_weights_mir(
            "test_shared", 128, 1, MilDtypeCompat::Fp16, 42,
        );

        let model = convert_mir_to_proto_multifunction(
            &graphs, &shared_names, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe,
        ).unwrap();

        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "embedding");
        assert_eq!(model.functions[1].name, "decode_step");

        // The shared weight should appear once in the shared_weights list
        assert_eq!(model.shared_weights.len(), 1);
        assert_eq!(model.shared_weights[0].weight.name, "shared_projection_weight");
        assert_eq!(model.shared_weights[0].referencing_functions.len(), 2);
    }

    #[test]
    fn test_build_linear_projection_mir() {
        let graph = build_linear_projection_mir(
            "test", 64, 32, 1, MilDtypeCompat::Fp16, 42,
        );

        assert_eq!(graph.ops.len(), 3); // const + const + linear
        assert_eq!(graph.inputs, vec!["x".to_string()]);
        assert_eq!(graph.outputs, vec!["output".to_string()]);
    }

    #[test]
    fn test_deterministic_data_generation() {
        let data1 = generate_deterministic_data(100, 2, 42);
        let data2 = generate_deterministic_data(100, 2, 42);
        assert_eq!(data1, data2); // Deterministic

        let data3 = generate_deterministic_data(100, 2, 43);
        assert_ne!(data1, data3); // Different seed = different data
    }

    // ─── Real protobuf serialization tests ─────────────────────────────────

    #[test]
    fn test_model_to_protobuf_bytes_linear() {
        let graph = build_linear_projection_mir(
            "test_linear", 64, 32, 1, MilDtypeCompat::Fp16, 42,
        );
        let model = convert_mir_to_proto(&graph, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe)
            .unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        assert!(!bytes.is_empty());

        // Verify the bytes are valid protobuf by parsing back
        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            parsed.specification_version,
            ane_coreml_proto::proto::SpecificationVersion::SpecificationVersion8 as i32,
        );
        assert!(parsed.ml_program.is_some());
    }

    #[test]
    fn test_model_to_protobuf_bytes_multifunction() {
        let (graphs, shared_names) = build_multifunction_shared_weights_mir(
            "test_shared", 64, 1, MilDtypeCompat::Fp16, 42,
        );
        let model = convert_mir_to_proto_multifunction(
            &graphs, &shared_names, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe,
        ).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        assert!(!bytes.is_empty());

        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();
        assert!(parsed.ml_program.is_some());
        let ml_prog = parsed.ml_program.as_ref().unwrap();
        assert!(ml_prog.functions.contains_key("embedding"));
        assert!(ml_prog.functions.contains_key("decode_step"));
    }

    #[test]
    fn test_protobuf_roundtrip_preserves_fields() {
        let graph = build_linear_projection_mir(
            "test_rt", 32, 16, 1, MilDtypeCompat::Fp16, 99,
        );
        let model = convert_mir_to_proto(&graph, SpecVersion::V7, CoreMlComputeUnit::All)
            .unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();

        // Check spec version
        assert_eq!(
            parsed.specification_version,
            ane_coreml_proto::proto::SpecificationVersion::SpecificationVersion7 as i32,
        );

        // Check model description
        assert!(parsed.description.is_some());
        let desc = parsed.description.as_ref().unwrap();
        assert_eq!(desc.default_function_name, "main");

        // Check optimization hints
        assert!(parsed.optimization_hints.is_some());
        let hints = parsed.optimization_hints.as_ref().unwrap();
        assert_eq!(
            hints.preferred_compute_unit,
            ane_coreml_proto::proto::ComputeUnit::All as i32,
        );
    }

    #[test]
    fn test_protobuf_roundtrip_ops_preserved() {
        let graph = build_linear_projection_mir(
            "test_ops", 16, 8, 1, MilDtypeCompat::Fp16, 7,
        );
        let model = convert_mir_to_proto(&graph, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe)
            .unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();

        let ml_prog = parsed.ml_program.as_ref().unwrap();
        let main_func = ml_prog.functions.get("main").unwrap();
        let block = main_func.block.as_ref().unwrap();

        // Should have 3 operations: const (weight) + const (bias) + linear
        assert_eq!(block.operations.len(), 3);

        // Check SSA names
        assert_eq!(block.operations[0].name, "weight");
        assert_eq!(block.operations[1].name, "bias");
        assert_eq!(block.operations[2].name, "output");

        // Check operation types
        assert!(matches!(
            &block.operations[0].operation,
            Some(ane_coreml_proto::proto::mil_operation::Operation::ConstOp(_))
        ));
        assert!(matches!(
            &block.operations[2].operation,
            Some(ane_coreml_proto::proto::mil_operation::Operation::LinearOp(_))
        ));
    }

    #[test]
    fn test_protobuf_roundtrip_with_state_ops() {
        // Build a model with ReadState and CoremlUpdateState ops
        let ops = vec![
            MirOpCompat::ReadState {
                name: "kv_cache".to_string(),
                state_id: "state_0".to_string(),
                shape: vec![128],
                dtype: MilDtypeCompat::Fp16,
            },
            MirOpCompat::CoremlUpdateState {
                name: "updated_state".to_string(),
                state_id: "state_0".to_string(),
                value: "new_val".to_string(),
            },
        ];

        let graph = MirGraphCompat {
            ops,
            inputs: vec!["new_val".to_string()],
            outputs: vec!["updated_state".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "decode_step".to_string(),
        };

        let model = convert_mir_to_proto(&graph, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe)
            .unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();

        let ml_prog = parsed.ml_program.as_ref().unwrap();
        let func = ml_prog.functions.get("decode_step").unwrap();
        let block = func.block.as_ref().unwrap();

        assert_eq!(block.operations.len(), 2);

        // ReadState op
        assert!(matches!(
            &block.operations[0].operation,
            Some(ane_coreml_proto::proto::mil_operation::Operation::ReadStateOp(_))
        ));

        // CoremlUpdateState op
        assert!(matches!(
            &block.operations[1].operation,
            Some(ane_coreml_proto::proto::mil_operation::Operation::CoremlUpdateStateOp(_))
        ));
    }

    #[test]
    fn test_weight_file_references_in_proto() {
        let graph = build_linear_projection_mir(
            "test_wref", 16, 8, 1, MilDtypeCompat::Fp16, 7,
        );
        let model = convert_mir_to_proto(&graph, SpecVersion::V8, CoreMlComputeUnit::CpuAndNe)
            .unwrap();

        // Simulate weight entries with real offsets (as WeightBinBuilder would produce)
        let weight_entries = vec![
            WeightEntry {
                name: "weight".to_string(),
                offset: 0,
                size: 256,
                shape: vec![8, 16],
                dtype: CoreMlDataType::Float16,
                data: vec![0u8; 256],
            },
            WeightEntry {
                name: "bias".to_string(),
                offset: 256,
                size: 16,
                shape: vec![8],
                dtype: CoreMlDataType::Float16,
                data: vec![0u8; 16],
            },
        ];

        let bytes = model_to_protobuf_bytes(&model, &weight_entries).unwrap();
        let parsed = ane_coreml_proto::proto::Model::decode(bytes.as_slice()).unwrap();

        let ml_prog = parsed.ml_program.as_ref().unwrap();
        let func = ml_prog.functions.get("main").unwrap();
        let block = func.block.as_ref().unwrap();

        // First const op should have FileReference with offset 0
        let const_op0 = match &block.operations[0].operation {
            Some(ane_coreml_proto::proto::mil_operation::Operation::ConstOp(op)) => op,
            _ => panic!("Expected ConstOp"),
        };
        let wd0 = const_op0.value.as_ref().unwrap();
        match &wd0.weight_data {
            Some(ane_coreml_proto::proto::weight_data::WeightData::FileRef(fr)) => {
                assert_eq!(fr.offset, 0);
                assert_eq!(fr.size, 256);
            }
            _ => panic!("Expected FileReference for weight"),
        }

        // Second const op should have FileReference with offset 256
        let const_op1 = match &block.operations[1].operation {
            Some(ane_coreml_proto::proto::mil_operation::Operation::ConstOp(op)) => op,
            _ => panic!("Expected ConstOp"),
        };
        let wd1 = const_op1.value.as_ref().unwrap();
        match &wd1.weight_data {
            Some(ane_coreml_proto::proto::weight_data::WeightData::FileRef(fr)) => {
                assert_eq!(fr.offset, 256);
                assert_eq!(fr.size, 16);
            }
            _ => panic!("Expected FileReference for bias"),
        }
    }
}
