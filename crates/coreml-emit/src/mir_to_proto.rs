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
        let unsupported: Vec<_> = graph
            .ops
            .iter()
            .filter_map(|op| {
                if let MirOpCompat::Unsupported { op_kind, name, .. } = op {
                    Some(format!("  {name} (kind={op_kind})"))
                } else {
                    None
                }
            })
            .collect();
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

        // Validation gate: reject ANE-illegal ops that should have been
        // decomposed during lowering. These ops force CPU fallback and
        // indicate a bug in the compilation pipeline.
        {
            let mut illegal_ops: Vec<String> = Vec::new();
            for op in &graph.ops {
                let illegal = match op {
                    MirOpCompat::Fill { name, .. } => Some(format!("  {name}: mb.fill is ANE-illegal, should have been replaced with MILConst during lowering")),
                    MirOpCompat::Select { name, .. } => Some(format!("  {name}: mb.select is ANE-illegal in practice (causes CPU fallback despite ConvertSelect in per-op matrix). Should have been decomposed to arithmetic (cond*x + (1-cond)*y) at SIR→AIR level")),
                    MirOpCompat::Where { name, .. } => Some(format!("  {name}: mb.where is ANE-illegal (no ANE converter). Should have been decomposed to arithmetic (cond*x + (1-cond)*y) at SIR→AIR level")),
                    // FillLike is NOT rejected — the Apple proto emitter decomposes it
                    // to ANE-legal mul(ref,0)+add(zero,val) ops.
                    _ => None,
                };
                if let Some(msg) = illegal {
                    illegal_ops.push(msg);
                }
            }
            if !illegal_ops.is_empty() {
                anyhow::bail!(
                    "Cannot emit Core ML package: {} ANE-illegal operation(s) in function '{}'. \
                     These ops force CPU fallback and should have been replaced earlier in the pipeline.\n\
                     ANE-illegal ops:\n{}\n\
                     Fix: Fill → MILConst (mil_lower). Select/Where → arithmetic decomposition \
                     (cond*x + (1-cond)*y) in legality_rewrite. FillLike → decomposed by proto emitter (mul+add).",
                    illegal_ops.len(),
                    graph.function_name,
                    illegal_ops.join("\n")
                );
            }
        }

        // Validation gate: reject duplicate output names.
        // Core ML MIL is SSA-like: each output value name may be defined only once
        // in a block. If two operations produce the same output name, coremlcompiler
        // rejects the model with "Block redefines I/O name:<name>".
        // This commonly happens with GQA when the same KV head is sliced multiple
        // times in a per-Q-head loop instead of being sliced once and reused.
        {
            let mut seen_names: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut duplicates: Vec<String> = Vec::new();
            for op in &graph.ops {
                let output_names = op_output_names(op);
                for name in output_names {
                    if !seen_names.insert(name.clone()) {
                        duplicates.push(name.clone());
                    }
                }
            }
            if !duplicates.is_empty() {
                // Deduplicate the error list for readability
                duplicates.sort();
                duplicates.dedup();
                anyhow::bail!(
                    "Cannot emit Core ML package: {} duplicate output name(s) in function '{}'. \
                     Core ML MIL requires each output name to be defined exactly once in a block. \
                     Duplicate names:\n  {}\n\
                     This is typically caused by GQA KV-head slicing inside a per-Q-head loop \
                     instead of pre-slicing each KV head once and reusing the result.",
                    duplicates.len(),
                    graph.function_name,
                    duplicates.join("\n  ")
                );
            }
        }

        // Validation gate: reject impossible reshapes.
        // Core ML rejects reshapes where input element count ≠ target element count.
        // A scalar (empty shape) being reshaped into a large tensor is the canonical
        // failure mode — it indicates that shape inference failed upstream (e.g., a
        // concat whose output shape was never computed).
        {
            let mut bad_reshapes: Vec<String> = Vec::new();
            for op in &graph.ops {
                if let MirOpCompat::Reshape { name, x, shape } = op {
                    let input_elements: usize =
                        graph.node_shapes.get(x).map(|s| s.iter().product::<usize>()).unwrap_or(0);
                    let target_elements: usize = shape
                        .iter()
                        .map(|&d| {
                            if d > 0 {
                                d as usize
                            } else {
                                1
                            } // treat 0-dims as 1 for element count
                        })
                        .product();
                    // Only validate when both are known and non-zero
                    if input_elements > 0
                        && target_elements > 0
                        && input_elements != target_elements
                    {
                        bad_reshapes.push(format!(
                            "  {name}: input '{}' has {} elements, target shape {:?} has {} elements",
                            x, input_elements, shape, target_elements
                        ));
                    }
                }
            }
            if !bad_reshapes.is_empty() {
                anyhow::bail!(
                    "Cannot emit Core ML package: {} impossible reshape(s) in function '{}'. \
                     Core ML rejects reshapes where input element count ≠ target element count. \
                     This typically indicates a shape inference bug (e.g., a concat whose output \
                     was never computed, producing a scalar that gets reshaped).\nImpossible reshapes:\n{}\n\
                     Fix: ensure all concats and other shape-propagating ops have correct output shapes \
                     in the node_shapes map before emission.",
                    bad_reshapes.len(),
                    graph.function_name,
                    bad_reshapes.join("\n")
                );
            }
        }

        // Validation gate: reject concats with scalar outputs.
        // A concat of ranked tensors must produce a ranked tensor, not a scalar.
        // This is the direct cause of the "cannot reshape tensor of size 1" error.
        {
            let mut bad_concats: Vec<String> = Vec::new();
            for op in &graph.ops {
                if let MirOpCompat::Concat { name, values, axis: _ } = op {
                    let output_shape = graph.node_shapes.get(name);
                    let has_ranked_inputs = values
                        .iter()
                        .any(|v| graph.node_shapes.get(v).map(|s| !s.is_empty()).unwrap_or(false));
                    let output_is_scalar = output_shape.map(|s| s.is_empty()).unwrap_or(true);
                    if has_ranked_inputs && output_is_scalar {
                        bad_concats.push(format!(
                            "  {name}: concat of {} values (some ranked) but output is scalar (empty shape)",
                            values.len()
                        ));
                    }
                }
            }
            if !bad_concats.is_empty() {
                anyhow::bail!(
                    "Cannot emit Core ML package: {} concat(s) with scalar output in function '{}'. \
                     A concat of ranked tensors must produce a ranked tensor. This causes \
                     downstream reshapes to fail with 'cannot reshape tensor of size 1'.\n\
                     Bad concats:\n{}\n\
                     Fix: add MILConcat shape inference to the forward shape propagation pass.",
                    bad_concats.len(),
                    graph.function_name,
                    bad_concats.join("\n")
                );
            }
        }

        // Validation gate: reject ops with zero dimensions in shape vectors.
        // (T-29 / I-08) Zero dimensions in reshape target shapes or fill shape
        // vectors produce invalid Core ML models. Core ML treats 0 as a literal
        // zero dimension, not "infer from input". While the mir_to_compat layer
        // should catch these first (hard bail), this provides defense-in-depth
        // in case zeros slip through via a different code path.
        {
            let mut zero_dim_ops: Vec<String> = Vec::new();
            for op in &graph.ops {
                match op {
                    MirOpCompat::Reshape { name, shape, .. } if shape.contains(&0) => {
                        let zero_pos: Vec<usize> = shape
                            .iter()
                            .enumerate()
                            .filter(|(_, &d)| d == 0)
                            .map(|(i, _)| i)
                            .collect();
                        zero_dim_ops.push(format!(
                                "  {name}: reshape target shape {:?} has zero dimension(s) at position(s) {:?}",
                                shape, zero_pos
                            ));
                    }
                    MirOpCompat::Fill { name, shape, .. } if shape.contains(&0) => {
                        let zero_pos: Vec<usize> = shape
                            .iter()
                            .enumerate()
                            .filter(|(_, &d)| d == 0)
                            .map(|(i, _)| i)
                            .collect();
                        zero_dim_ops.push(format!(
                            "  {name}: fill shape {:?} has zero dimension(s) at position(s) {:?}",
                            shape, zero_pos
                        ));
                    }
                    _ => {}
                }
            }
            if !zero_dim_ops.is_empty() {
                anyhow::bail!(
                    "Cannot emit Core ML package: {} operation(s) with zero dimensions in \
                     shape vectors in function '{}'. Core ML treats 0 as a literal zero \
                     dimension, producing invalid models. This indicates that shape inference \
                     failed to resolve placeholder zeros before emission.\n\
                     Zero-dimension ops:\n{}\n\
                     Fix: ensure shape inference (infer_shape / resolve_reshape_zeros / \
                     resolve_reshape_shape) resolves all zero placeholders before the compat \
                     conversion step.",
                    zero_dim_ops.len(),
                    graph.function_name,
                    zero_dim_ops.join("\n")
                );
            }
        }

        // T-104: Before the state-building loop, collect all ReadState ops into
        // a map so that CoremlUpdateState and StateWrite can derive their shape
        // and dtype from ReadState when present. Core ML rejects protos with
        // empty-dimension state tensors, so we must not default to shape=[].
        let mut read_state_map: std::collections::HashMap<String, (Vec<usize>, CoreMlDataType)> =
            std::collections::HashMap::new();
        for op in &graph.ops {
            if let MirOpCompat::ReadState { state_id, shape, dtype, .. } = op {
                read_state_map
                    .entry(state_id.clone())
                    .or_insert_with(|| (shape.clone(), CoreMlDataType::from_mir_dtype(dtype)));
            }
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
                MirOpCompat::ReadState { name: _, state_id, shape, dtype }
                    // Collect state declarations from ReadState ops.
                    // Deduplicate by state_id since the same state may be
                    // read multiple times in one function.
                    if !graph_states.iter().any(|s: &TensorDesc| s.name == *state_id) => {
                        graph_states.push(TensorDesc {
                            name: state_id.clone(),
                            shape: shape.iter().map(|&d| d as u64).collect(),
                            dtype: CoreMlDataType::from_mir_dtype(dtype),
                            is_state: true,
                        });
                    }
                MirOpCompat::CoremlUpdateState { state_id, .. }
                    // T-104: Also collect state declarations from CoremlUpdateState ops.
                    // A function might only write to a state without reading
                    // it first (e.g., initial fill), so we need to capture
                    // these declarations too. Look up ReadState map for shape/dtype.
                    if !graph_states.iter().any(|s: &TensorDesc| s.name == *state_id) => {
                        // T-104: Derive shape/dtype from ReadState if available
                        if let Some((shape, dtype)) = read_state_map.get(state_id) {
                            graph_states.push(TensorDesc {
                                name: state_id.clone(),
                                shape: shape.iter().map(|&d| d as u64).collect(),
                                dtype: dtype.clone(),
                                is_state: true,
                            });
                        } else {
                            // T-104: No ReadState exists and shape is empty — Core ML
                            // rejects empty-dimension state tensors.
                            anyhow::bail!(
                                "State '{}' has no ReadState op and no explicit shape — \
                                 Core ML rejects empty-dimension state tensors",
                                state_id
                            );
                        }
                    }
                MirOpCompat::StateWrite { state_ref, .. }
                    // T-104: StateWrite uses state_ref instead of state_id.
                    // Look up ReadState map for shape/dtype.
                    if !graph_states.iter().any(|s: &TensorDesc| s.name == *state_ref) => {
                        // T-104: Derive shape/dtype from ReadState if available
                        if let Some((shape, dtype)) = read_state_map.get(state_ref) {
                            graph_states.push(TensorDesc {
                                name: state_ref.clone(),
                                shape: shape.iter().map(|&d| d as u64).collect(),
                                dtype: dtype.clone(),
                                is_state: true,
                            });
                        } else {
                            // T-104: No ReadState exists and shape is empty — Core ML
                            // rejects empty-dimension state tensors.
                            anyhow::bail!(
                                "State '{}' has no ReadState op and no explicit shape — \
                                 Core ML rejects empty-dimension state tensors",
                                state_ref
                            );
                        }
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

    // Default function name and model-level I/O description.
    // For multi-function models (e.g., embedding + decode_step with KV cache),
    // the default function should be the one used repeatedly at inference time,
    // which is typically the last function (decode_step). The embedding function
    // is called only once during prefill.
    // For single-function models, just use the first (only) function.
    let default_function_name = if graphs.len() > 1 {
        // Multi-function: prefer the last function as default (typically decode_step)
        graphs.last().map(|g| g.function_name.clone()).unwrap_or_else(|| "main".to_string())
    } else {
        graphs.first().map(|g| g.function_name.clone()).unwrap_or_else(|| "main".to_string())
    };

    // Model description uses the default function's I/O
    let default_fn =
        functions.iter().find(|f| f.name == default_function_name).or_else(|| functions.first());
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
    let apple_model = ane_coreml_proto::convert_to_apple_proto_model(model, weight_entries)
        .map_err(|e| {
            anyhow::anyhow!("Core ML proto validation failed: [{}] {}", e.kind, e.message)
        })?;
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

/// Extract the output name(s) from a `MirOpCompat` operation.
///
/// Every `MirOpCompat` variant defines one output value in the MIL block.
/// The `name` field serves as the SSA value name that other ops reference.
/// For duplicate detection, we return a single-element Vec with the name.
///
// TODO(T-38): Remove this wrapper once all callers use MirOpCompat::output_name() directly
fn op_output_names(op: &MirOpCompat) -> Vec<String> {
    vec![op.output_name()]
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

        // Check op types: read_state uses "input" param, write_state uses "state"/"value"
        assert_eq!(block.operations[0].r#type, "read_state");
        assert_eq!(block.operations[1].r#type, "write_state");

        // Check that state ops use name references (not inline strings)
        let read_state_op = &block.operations[0];
        // read_state uses "input" parameter name (Core ML schema: mb.read_state(input=k_state))
        let input_arg = read_state_op.inputs.get("input").unwrap();
        // Should be a name reference, not an immediate value
        match input_arg.arguments[0].binding.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::argument::binding::Binding::Name(name) => {
                assert_eq!(name, "state_0");
            }
            other => panic!("Expected Name binding for input, got {:?}", other),
        }

        let update_state_op = &block.operations[1];
        // write_state uses "state" parameter name
        let state_arg2 = update_state_op.inputs.get("state").unwrap();
        match state_arg2.arguments[0].binding.as_ref().unwrap() {
            ane_coreml_proto::apple_proto::mil_spec::argument::binding::Binding::Name(name) => {
                assert_eq!(name, "state_0");
            }
            other => panic!("Expected Name binding for state in write_state, got {:?}", other),
        }

        // All ops should have attributes["name"]
        assert!(read_state_op.attributes.contains_key("name"));
        assert!(update_state_op.attributes.contains_key("name"));
    }

    #[test]
    fn test_duplicate_output_names_rejected() {
        // Build a graph with two ops producing the same output name.
        // This simulates the GQA KV-head duplicate bug where the same
        // k_head_0 is sliced twice in a per-Q-head loop.
        let ops = vec![
            MirOpCompat::SliceByIndex {
                name: "k_head_0".to_string(), // First definition
                x: "k_split".to_string(),
                begin: vec![0, 0, 0, 0],
                end: vec![0, 1, 0, 0],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, false, true, true],
                end_mask: vec![true, false, true, true],
                squeeze_mask: vec![false, true, false, false],
            },
            MirOpCompat::SliceByIndex {
                name: "k_head_0".to_string(), // DUPLICATE — same name!
                x: "k_split".to_string(),
                begin: vec![0, 0, 0, 0],
                end: vec![0, 1, 0, 0],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, false, true, true],
                end_mask: vec![true, false, true, true],
                squeeze_mask: vec![false, true, false, false],
            },
        ];

        let graph = MirGraphCompat {
            ops,
            inputs: vec!["k_split".to_string()],
            outputs: vec!["k_head_0".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "decode_step".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);

        assert!(result.is_err(), "Duplicate output names should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("duplicate output name"),
            "Error should mention duplicate output names, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("k_head_0"),
            "Error should mention the specific duplicate name, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("decode_step"),
            "Error should mention the function name, got: {}",
            err_msg
        );
    }

    // ─── Zero-dimension validation tests (T-29 / I-08) ──────────────────

    #[test]
    fn test_zero_dim_reshape_rejected() {
        // A reshape with zero dimensions in its target shape must be
        // rejected by the emission validation gate. Core ML treats 0
        // as a literal zero dimension, not "infer from input".
        let graph = MirGraphCompat {
            ops: vec![MirOpCompat::Reshape {
                name: "bad_reshape".to_string(),
                x: "input".to_string(),
                shape: vec![0, 0, 16, 128], // zeros at positions 0 and 1
            }],
            inputs: vec!["input".to_string()],
            outputs: vec!["bad_reshape".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        assert!(result.is_err(), "Reshape with zero dimensions should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("zero dimension"),
            "Error should mention zero dimensions, got: {err_msg}"
        );
        assert!(
            err_msg.contains("bad_reshape"),
            "Error should mention the op name, got: {err_msg}"
        );
    }

    #[test]
    fn test_zero_dim_reshape_single_zero_rejected() {
        // Even a single zero dimension should be caught.
        let graph = MirGraphCompat {
            ops: vec![MirOpCompat::Reshape {
                name: "single_zero".to_string(),
                x: "input".to_string(),
                shape: vec![1, 0, 128], // zero at position 1
            }],
            inputs: vec!["input".to_string()],
            outputs: vec!["single_zero".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        assert!(result.is_err(), "Reshape with even one zero dimension should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("zero dimension"),
            "Error should mention zero dimensions, got: {err_msg}"
        );
        assert!(
            err_msg.contains("position(s) [1]"),
            "Error should mention position 1, got: {err_msg}"
        );
    }

    #[test]
    fn test_zero_dim_fill_rejected_by_ane_illegal_gate() {
        // A Fill op with zero dimensions in its shape vector is caught by
        // the ANE-illegal gate (which rejects ALL Fill ops) BEFORE the
        // zero-dim gate runs. This test verifies that Fill ops with zero
        // dimensions are still rejected, albeit via the earlier gate.
        let graph = MirGraphCompat {
            ops: vec![MirOpCompat::Fill {
                name: "bad_fill".to_string(),
                shape: vec![0, 512], // zero at position 0
                value: 1.0,
                dtype: MilDtypeCompat::Fp16,
            }],
            inputs: vec![],
            outputs: vec!["bad_fill".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        assert!(result.is_err(), "Fill with zero dimensions should be rejected");
        let err_msg = result.unwrap_err().to_string();
        // Fill is caught by the ANE-illegal gate first (which rejects ALL Fill ops)
        assert!(
            err_msg.contains("ANE-illegal") || err_msg.contains("zero dimension"),
            "Fill should be rejected by ANE-illegal or zero-dim gate, got: {err_msg}"
        );
        assert!(err_msg.contains("bad_fill"), "Error should mention the op name, got: {err_msg}");
    }

    #[test]
    fn test_zero_dim_reshape_all_zeros_rejected() {
        // An all-zero reshape shape is definitely invalid.
        let graph = MirGraphCompat {
            ops: vec![MirOpCompat::Reshape {
                name: "all_zeros".to_string(),
                x: "input".to_string(),
                shape: vec![0, 0, 0, 0],
            }],
            inputs: vec!["input".to_string()],
            outputs: vec!["all_zeros".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        assert!(result.is_err(), "All-zero reshape should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("zero dimension"),
            "Error should mention zero dimensions, got: {err_msg}"
        );
        assert!(
            err_msg.contains("position(s) [0, 1, 2, 3]"),
            "Error should list all zero positions, got: {err_msg}"
        );
    }

    #[test]
    fn test_concrete_reshape_passes_validation() {
        // A reshape with no zero dimensions should pass the zero-dim gate
        // (though it may fail other gates like impossible reshape if the
        // element counts don't match).
        let mut node_shapes = std::collections::HashMap::new();
        node_shapes.insert("input".to_string(), vec![1, 512, 16, 128]);

        let graph = MirGraphCompat {
            ops: vec![MirOpCompat::Reshape {
                name: "good_reshape".to_string(),
                x: "input".to_string(),
                shape: vec![1, 512, 2048], // no zeros, valid reshape
            }],
            inputs: vec!["input".to_string()],
            outputs: vec!["good_reshape".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes,
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        // Should NOT fail on zero-dim gate (may or may not succeed on
        // other gates, but zero-dim should pass).
        if let Err(e) = &result {
            let err_msg = e.to_string();
            assert!(
                !err_msg.contains("zero dimension"),
                "Concrete reshape should not trigger zero-dim rejection, got: {err_msg}"
            );
        }
    }

    #[test]
    fn test_multiple_zero_dim_ops_all_reported() {
        // When multiple ops have zero dimensions, all of them should be
        // reported in the error message (not just the first one).
        let graph = MirGraphCompat {
            ops: vec![
                MirOpCompat::Reshape {
                    name: "bad_reshape_1".to_string(),
                    x: "input".to_string(),
                    shape: vec![0, 128],
                },
                MirOpCompat::Reshape {
                    name: "bad_reshape_2".to_string(),
                    x: "input".to_string(),
                    shape: vec![1, 0, 64],
                },
            ],
            inputs: vec!["input".to_string()],
            outputs: vec!["bad_reshape_2".to_string()],
            opset_version: "iOS18".to_string(),
            function_name: "main".to_string(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };

        let result = convert_mir_to_proto(&graph, SpecVersion::V10, CoreMlComputeUnit::CpuAndNe);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bad_reshape_1"),
            "Error should mention first bad reshape, got: {err_msg}"
        );
        assert!(
            err_msg.contains("bad_reshape_2"),
            "Error should mention second bad reshape, got: {err_msg}"
        );
        assert!(
            err_msg.contains("2 operation(s)"),
            "Error should report count of 2, got: {err_msg}"
        );
    }
}
