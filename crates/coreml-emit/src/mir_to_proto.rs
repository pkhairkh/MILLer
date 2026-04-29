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

use ane_coreml_proto::{
    mir_compat::{MilDtypeCompat, MirGraphCompat, MirOpCompat},
    CoreMlComputeUnit, CoreMlDataType, CoreMlFunction, CoreMlModel, ModelDescriptionCompat,
    SharedWeightRef, SpecVersion, TensorDesc, WeightEntry,
};
use anyhow::Result;
use prost::Message;

/// Convert a single-function MIR graph to a CoreMlModel.
pub fn convert_mir_to_proto(
    graph: &MirGraphCompat,
    spec_version: SpecVersion,
    compute_unit: CoreMlComputeUnit,
) -> Result<CoreMlModel> {
    convert_mir_to_proto_multifunction(std::slice::from_ref(graph), &[], spec_version, compute_unit)
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
        // Validation gate: reject any Unsupported ops before they can produce
        // illegal MIL operator names like "identity__unsupported_tile" that
        // coremlcompiler will reject. Fail early with a clear error rather
        // than writing an invalid .mlpackage.
        let unsupported: Vec<_> = graph.ops.iter().filter_map(|op| {
            if let MirOpCompat::Unsupported { op_kind, name, .. } = op {
                Some(format!("  {name} (kind={op_kind})"))
            } else {
                None
            }
        }).collect();
        if !unsupported.is_empty() {
            anyhow::bail!(
                "Cannot emit Core ML package: {} unsupported MIR operation(s) in function '{}'. \
                 These would produce illegal 'identity__unsupported_*' operator names that \
                 coremlcompiler rejects.\nUnsupported ops:\n{}\n\
                 Each unsupported op needs a specialized MirOpCompat variant and emission path.",
                unsupported.len(),
                graph.function_name,
                unsupported.join("\n")
            );
        }

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

        // Build input/output descriptions from the graph.
        // If input_descs/output_descs are provided (from MIR node shapes),
        // use those for accurate shape information. Otherwise, fall back
        // to name-only descriptors with empty shapes.
        for input_name in &graph.inputs {
            if let Some(desc) = graph.input_descs.iter().find(|d| d.name == *input_name) {
                graph_inputs.push(TensorDesc {
                    name: desc.name.clone(),
                    shape: desc.shape.iter().map(|&d| d as u64).collect(),
                    dtype: CoreMlDataType::from_mir_dtype(&desc.dtype),
                    is_state: false,
                });
            } else {
                graph_inputs.push(TensorDesc {
                    name: input_name.clone(),
                    shape: vec![], // Shape unknown — Core ML may infer from graph
                    dtype: CoreMlDataType::Float16, // Default
                    is_state: false,
                });
            }
        }
        for output_name in &graph.outputs {
            if let Some(desc) = graph.output_descs.iter().find(|d| d.name == *output_name) {
                graph_outputs.push(TensorDesc {
                    name: desc.name.clone(),
                    shape: desc.shape.iter().map(|&d| d as u64).collect(),
                    dtype: CoreMlDataType::from_mir_dtype(&desc.dtype),
                    is_state: false,
                });
            } else {
                graph_outputs.push(TensorDesc {
                    name: output_name.clone(),
                    shape: vec![], // Shape unknown — Core ML may infer from graph
                    dtype: CoreMlDataType::Float16, // Default
                    is_state: false,
                });
            }
        }

        functions.push(CoreMlFunction {
            name: graph.function_name.clone(),
            inputs: graph_inputs,
            outputs: graph_outputs,
            states: graph_states,
            operations: graph.ops.clone(),
            node_shapes: graph.node_shapes.clone(),
        });

        all_weights.extend(graph_weights);
    }

    // Build shared weight references
    for (weight_name, referencing_functions) in shared_weight_refs {
        if let Some(weight_entry) = all_weights.iter().find(|w| w.name == weight_name) {
            shared_weights
                .push(SharedWeightRef { weight: weight_entry.clone(), referencing_functions });
        }
    }

    // Default function name
    let default_function_name =
        graphs.first().map(|g| g.function_name.clone()).unwrap_or_else(|| "main".to_string());

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

/// Serialize a CoreMlModel to protobuf bytes using Apple's actual wire format.
///
/// This produces the raw bytes that will be written as `model.mlmodel`
/// inside the mlpackage directory. The serialization uses Apple's actual
/// protobuf format (packages `CoreML.Specification` / `CoreML.Specification.MILSpec`),
/// which Core ML's runtime can decode correctly.
///
/// Key differences from the legacy format:
/// - Uses `MILSpec.Program` (field 502) instead of `MLProgram` (field 20)
/// - Operations use generic `type` + `inputs` + `outputs` format
/// - Data types use Apple enum values (FLOAT16=10, FLOAT32=11, etc.)
/// - Weight references use `BlobFileValue` with `fileName="weight.bin"`
pub fn model_to_protobuf_bytes(
    model: &CoreMlModel,
    weight_entries: &[WeightEntry],
) -> Result<Vec<u8>> {
    let apple_model = ane_coreml_proto::convert_to_apple_proto_model(model, weight_entries);
    Ok(apple_model.encode_to_vec())
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
                dtype,
                shape: vec![output_dim, input_dim], // mb.linear convention: [out, in]
            },
            MirOpCompat::Const {
                name: "bias".to_string(),
                data: bias_data,
                dtype,
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
        input_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "x".to_string(),
            shape: vec![1, input_dim],
            dtype,
        }],
        output_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "output".to_string(),
            shape: vec![1, output_dim],
            dtype,
        }],
        node_shapes: std::collections::HashMap::new(),
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
                dtype,
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
        input_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "x".to_string(),
            shape: vec![1, embed_dim],
            dtype,
        }],
        output_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "embedding_output".to_string(),
            shape: vec![1, embed_dim],
            dtype,
        }],
        node_shapes: std::collections::HashMap::new(),
    };

    // Decode step function: uses the SAME shared weight
    let decode_step_graph = MirGraphCompat {
        ops: vec![
            MirOpCompat::Const {
                name: "shared_projection_weight".to_string(),
                data: shared_weight_data,
                dtype,
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
        input_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "hidden".to_string(),
            shape: vec![1, embed_dim],
            dtype,
        }],
        output_descs: vec![ane_coreml_proto::mir_compat::TensorDescCompat {
            name: "decode_output".to_string(),
            shape: vec![1, embed_dim],
            dtype,
        }],
        node_shapes: std::collections::HashMap::new(),
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
        let graph = build_linear_projection_mir("test_linear", 64, 32, 1, MilDtypeCompat::Fp16, 42);

        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "main");
        assert_eq!(model.default_function_name, "main");
        assert_eq!(model.weights.len(), 2); // weight + bias
        assert_eq!(model.shared_weights.len(), 0);
    }

    #[test]
    fn test_convert_multifunction_with_shared_weights() {
        let (graphs, shared_names) =
            build_multifunction_shared_weights_mir("test_shared", 128, 1, MilDtypeCompat::Fp16, 42);

        let model = convert_mir_to_proto_multifunction(
            &graphs,
            &shared_names,
            SpecVersion::V10,
            CoreMlComputeUnit::CpuAndNe,
        )
        .unwrap();

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
        let graph = build_linear_projection_mir("test", 64, 32, 1, MilDtypeCompat::Fp16, 42);

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

    // ─── Apple-format protobuf serialization tests ────────────────────────

    #[test]
    fn test_model_to_protobuf_bytes_linear() {
        let graph = build_linear_projection_mir("test_linear", 64, 32, 1, MilDtypeCompat::Fp16, 42);
        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        assert!(!bytes.is_empty());

        // Verify the bytes are valid Apple protobuf by parsing back
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();
        assert_eq!(parsed.specification_version, 10);
        assert!(parsed.description.is_some());

        // Check that mlProgram is present (field 502 in Apple's format)
        let model_type = parsed.r#type.as_ref().unwrap();
        match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(program) => {
                assert!(program.functions.contains_key("main"));
            }
        }
    }

    #[test]
    fn test_model_to_protobuf_bytes_multifunction() {
        let (graphs, shared_names) =
            build_multifunction_shared_weights_mir("test_shared", 64, 1, MilDtypeCompat::Fp16, 42);
        let model = convert_mir_to_proto_multifunction(
            &graphs,
            &shared_names,
            SpecVersion::V10,
            CoreMlComputeUnit::CpuAndNe,
        )
        .unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        assert!(!bytes.is_empty());

        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();
        let model_type = parsed.r#type.as_ref().unwrap();
        match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(program) => {
                assert!(program.functions.contains_key("embedding"));
                assert!(program.functions.contains_key("decode_step"));
            }
        }
    }

    #[test]
    fn test_apple_proto_spec_version() {
        let graph = build_linear_projection_mir("test_rt", 32, 16, 1, MilDtypeCompat::Fp16, 99);
        let model = convert_mir_to_proto(&graph, SpecVersion::V7, CoreMlComputeUnit::All).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();

        assert_eq!(parsed.specification_version, 7);
    }

    #[test]
    fn test_apple_proto_ops_preserved() {
        let graph = build_linear_projection_mir("test_ops", 16, 8, 1, MilDtypeCompat::Fp16, 7);
        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();

        let model_type = parsed.r#type.as_ref().unwrap();
        let program = match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(p) => p,
        };
        let main_func = program.functions.get("main").unwrap();
        assert_eq!(main_func.opset, "CoreML9");

        let block = main_func.block_specializations.get("CoreML9").unwrap();

        // Should have 3 operations: const (weight) + const (bias) + linear
        assert_eq!(block.operations.len(), 3);

        // Check operation types (Apple format uses string type field)
        assert_eq!(block.operations[0].r#type, "const");
        assert_eq!(block.operations[1].r#type, "const");
        assert_eq!(block.operations[2].r#type, "linear");

        // Check output names
        assert_eq!(block.operations[0].outputs[0].name, "weight");
        assert_eq!(block.operations[1].outputs[0].name, "bias");
        assert_eq!(block.operations[2].outputs[0].name, "output");

        // Check that const ops use attributes instead of inputs for values
        let const_op0 = &block.operations[0];
        assert!(const_op0.inputs.is_empty()); // const ops have no inputs in Apple format
        assert!(const_op0.attributes.contains_key("val")); // value is in attributes["val"]
        assert!(const_op0.attributes.contains_key("name")); // name attribute present

        // Check linear op inputs
        let linear_op = &block.operations[2];
        assert!(linear_op.inputs.contains_key("x"));
        assert!(linear_op.inputs.contains_key("weight"));
        assert!(linear_op.inputs.contains_key("bias"));
        // All ops should have attributes["name"]
        assert!(linear_op.attributes.contains_key("name"));
    }

    #[test]
    fn test_apple_proto_weight_blob_file_references() {
        let graph = build_linear_projection_mir("test_wref", 16, 8, 1, MilDtypeCompat::Fp16, 7);
        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

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
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();

        let model_type = parsed.r#type.as_ref().unwrap();
        let program = match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(p) => p,
        };
        let main_func = program.functions.get("main").unwrap();
        let block = main_func.block_specializations.get("CoreML9").unwrap();

        // First const op: should have BlobFileValue in attributes["val"] with offset 0
        let const_op0 = &block.operations[0];
        assert_eq!(const_op0.r#type, "const");
        let val_attr0 = const_op0.attributes.get("val").unwrap();
        match val_attr0.value.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::value::Value::BlobFileValue(bfv) => {
                assert_eq!(bfv.file_name, "@model_path/weights/weight.bin");
                assert_eq!(bfv.offset, 0);
            }
            other => panic!("Expected BlobFileValue for weight, got {:?}", other),
        }

        // Second const op: should have BlobFileValue in attributes["val"] with offset 256
        let const_op1 = &block.operations[1];
        assert_eq!(const_op1.r#type, "const");
        let val_attr1 = const_op1.attributes.get("val").unwrap();
        match val_attr1.value.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::value::Value::BlobFileValue(bfv) => {
                assert_eq!(bfv.file_name, "@model_path/weights/weight.bin");
                assert_eq!(bfv.offset, 256);
            }
            other => panic!("Expected BlobFileValue for bias, got {:?}", other),
        }
    }

    #[test]
    fn test_apple_proto_model_description_functions() {
        // Single-function model: should use single-function schema pattern
        // (top-level I/O populated, functions empty, MIL program key="main")
        let graph = build_linear_projection_mir("test_desc", 32, 16, 1, MilDtypeCompat::Fp16, 99);
        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();

        let desc = parsed.description.as_ref().unwrap();
        // Single-function: no defaultFunctionName, empty functions list
        assert_eq!(desc.default_function_name, "");
        assert!(desc.functions.is_empty());

        // Single-function: top-level I/O should be populated
        assert!(!desc.input.is_empty());
        assert!(!desc.output.is_empty());

        // Single-function: MIL Program function key should be "main"
        let model_type = parsed.r#type.as_ref().unwrap();
        match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(program) => {
                assert!(program.functions.contains_key("main"));
            }
        }

        // Check metadata — reference models have metadata = None
        assert!(desc.metadata.is_none());
    }

    #[test]
    fn test_apple_proto_array_feature_type_values() {
        // Verify that ArrayFeatureType uses Apple's enum values
        assert_eq!(
            ane_coreml_proto::apple_proto::array_feature_type::ArrayDataType::Float16 as i32,
            65552
        );
        assert_eq!(
            ane_coreml_proto::apple_proto::array_feature_type::ArrayDataType::Float32 as i32,
            65568
        );
        assert_eq!(
            ane_coreml_proto::apple_proto::array_feature_type::ArrayDataType::Int32 as i32,
            131104
        );
    }

    #[test]
    fn test_apple_proto_mil_data_type_values() {
        // Verify that MILSpec.DataType uses Apple's enum values
        assert_eq!(ane_coreml_proto::apple_proto::mil_spec::DataType::Float16 as i32, 10);
        assert_eq!(ane_coreml_proto::apple_proto::mil_spec::DataType::Float32 as i32, 11);
        assert_eq!(ane_coreml_proto::apple_proto::mil_spec::DataType::Int32 as i32, 23);
    }

    #[test]
    fn test_apple_proto_state_ops() {
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
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let model =
            convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe).unwrap();

        let bytes = model_to_protobuf_bytes(&model, &model.weights).unwrap();
        let parsed = ane_coreml_proto::apple_proto::Model::decode(bytes.as_slice()).unwrap();

        let model_type = parsed.r#type.as_ref().unwrap();
        let program = match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(p) => p,
        };
        let func = program.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();

        assert_eq!(block.operations.len(), 2);

        // Check op types (now using Apple's "write_state" instead of "coreml_update_state")
        assert_eq!(block.operations[0].r#type, "read_state");
        assert_eq!(block.operations[1].r#type, "write_state");

        // Check that state ops use name references (not inline strings)
        let read_state_op = &block.operations[0];
        let state_arg = read_state_op.inputs.get("state").unwrap();
        // Should be a name reference, not an immediate value
        match state_arg.arguments[0].binding.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::argument::binding::Binding::Name(name) => {
                assert_eq!(name, "state_0");
            }
            other => panic!("Expected Name binding for state, got {:?}", other),
        }

        let write_state_op = &block.operations[1];
        let state_arg2 = write_state_op.inputs.get("state").unwrap();
        match state_arg2.arguments[0].binding.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::argument::binding::Binding::Name(name) => {
                assert_eq!(name, "state_0");
            }
            other => panic!("Expected Name binding for state in write_state, got {:?}", other),
        }

        // All ops should have attributes["name"]
        assert!(read_state_op.attributes.contains_key("name"));
        assert!(write_state_op.attributes.contains_key("name"));
    }
}
