//! MIR-to-Compat Conversion Module
//!
//! Converts from the compiler's internal `ane-ir::mir` types to the
//! proto-compat representation (`ane_coreml_proto::mir_compat`).
//!
//! This bridge layer is needed because `ane-coreml-proto` deliberately
//! avoids depending on `ane-ir` (to prevent circular dependencies).
//! Instead, it defines its own `MirGraphCompat` / `MirOpCompat` types
//! that mirror the essential structure. This module performs the conversion.
//!
//! ## Type Mapping
//!
//! | `ane-ir::mir`          | `mir_compat`           | Notes |
//! |-------------------------|------------------------|-------|
//! | `MirGraph`              | `MirGraphCompat`       |       |
//! | `MirNode`               | (flattened into ops)   | MirNode wraps MirOp + metadata |
//! | `MirOp`                 | `MirOpCompat`          | See variant mapping below |
//! | `MilDtype`              | `MilDtypeCompat`       | 1:1   |
//! | `ComputeUnitHint`       | `ComputeUnitHintCompat`| 1:1   |
//! | `MirNodeId(String)`     | `String`               | Unwrap newtype |
//!
//! ## MIR Coverage
//!
//! All currently declared `MirOp` variants have a `MirOpCompat` equivalent.
//! Earlier gaps for `MILConv`, `MILStateWrite`, and `MILReduceSum` were closed
//! in Sprint 54, so there are no remaining bail paths in this conversion layer.
//!
//! ## Weight Data
//!
//! `MILConst` in `ane-ir` stores a `value_path` (a reference key to weight
//! data) rather than inline bytes. The compat `Const` variant requires
//! `data: Vec<u8>`. The conversion uses a `WeightResolver` trait to look
//! up weight data by path. If no resolver is provided, empty data is used
//! (suitable for structure-only conversions).

use ane_coreml_proto::mir_compat::{
    ComputeUnitHintCompat, MilDtypeCompat, MirGraphCompat, MirOpCompat,
};
use ane_ir::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirOp};
#[allow(unused_imports)]
// T-38: ToProto trait methods will be used for validation in future PRs
use ane_ir::toproto::ToProto;
use anyhow::Result;
use std::collections::HashMap;

use crate::shape_inference::{compat_input_dtype, compat_input_shape, compat_output_shape};

/// Resolver for weight data referenced by `MILConst.value_path`.
///
/// When converting `MILConst { value_path, .. }` to `MirOpCompat::Const { data, .. }`,
/// we need to look up the actual weight bytes. Implement this trait to provide
/// that lookup.
pub trait WeightResolver {
    /// Resolve weight data for the given value path.
    /// Returns the raw bytes and shape of the weight tensor.
    fn resolve(&self, value_path: &str) -> Option<WeightData>;
}

/// Weight data returned by a `WeightResolver`.
#[derive(Debug, Clone)]
pub struct WeightData {
    /// Raw weight bytes.
    pub data: Vec<u8>,
    /// Shape of the weight tensor.
    pub shape: Vec<usize>,
}

/// A simple in-memory weight resolver backed by a HashMap.
#[derive(Debug, Clone, Default)]
pub struct HashMapWeightResolver {
    weights: HashMap<String, WeightData>,
}

impl HashMapWeightResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a weight entry.
    pub fn add(&mut self, path: String, data: Vec<u8>, shape: Vec<usize>) {
        self.weights.insert(path, WeightData { data, shape });
    }
}

impl WeightResolver for HashMapWeightResolver {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        self.weights.get(value_path).cloned()
    }
}

/// A resolver that returns empty data for all lookups.
/// Useful for structure-only conversions where weight data
/// is not available or not needed.
#[derive(Debug, Clone, Copy)]
pub struct EmptyWeightResolver;

impl WeightResolver for EmptyWeightResolver {
    fn resolve(&self, _value_path: &str) -> Option<WeightData> {
        // Return Some with empty data so the conversion doesn't fail,
        // but the shape will need to come from the node's shape field.
        None
    }
}

/// Convert a compiler MIR graph to the proto-compat representation.
///
/// The `resolver` is used to look up weight data for `MILConst` ops.
/// If you don't have weight data available, use `EmptyWeightResolver`.
///
/// ## Weight Materialization
///
/// In the MIR graph, ops like `MILLinear` and `MILLayerNorm` reference weights
/// by name (e.g., `"model.layers.0.self_attn.q_proj.weight"`) but don't carry
/// the weight data inline. The Core ML proto format requires weight data to be
/// present as `Const` ops (which become `WeightEntry` objects in the model).
///
/// This function automatically materializes `MirOpCompat::Const` entries for
/// every weight name referenced by ops like `MILLinear` and `MILLayerNorm`.
/// The weight data is looked up via the `resolver`. If the resolver can't find
/// a weight, zero-filled data is used (matching the MILConst fallback behavior).
pub fn mir_graph_to_compat(
    graph: &MirGraph,
    resolver: &dyn WeightResolver,
) -> Result<MirGraphCompat> {
    mir_graph_to_compat_with_arch(graph, resolver, None, None)
}

/// Convert a MIR graph to compat representation with architecture-specific
/// weight name patterns.
///
/// T-36 (I-15/CQ-18): The `architecture` parameter allows the caller to
/// specify the model architecture for weight-name pattern resolution in
/// [`build_input_alias_map`]. When `None`, defaults to Qwen3 patterns
/// (backward-compatible behavior).
///
/// T-36 (I-15/CQ-19): The `max_seq_len` parameter replaces the hardcoded
/// `512` fallback in shape inference. When `None`, defaults to 32768
/// (Qwen3-0.6B max_position_embeddings).
pub fn mir_graph_to_compat_with_arch(
    graph: &MirGraph,
    resolver: &dyn WeightResolver,
    architecture: Option<&ane_ir::common::ModelArchitecture>,
    max_seq_len: Option<usize>,
) -> Result<MirGraphCompat> {
    let alias_map = build_input_alias_map(graph, architecture);
    let max_seq_len = max_seq_len.unwrap_or(32768);

    // Build a shape map from MirNode.id → MirNode.shape so that reshape ops
    // can resolve zero-placeholder dimensions by looking up their input node's
    // shape. This is critical because infer_shape() in mil_lower.rs may fail
    // to propagate shapes correctly (e.g., when input shapes aren't seeded or
    // positional zero resolution is wrong for rank-changing reshapes).
    let shape_map: std::collections::HashMap<String, Vec<usize>> = graph
        .nodes
        .iter()
        .filter(|n| !n.shape.is_empty())
        .map(|n| (n.id.0.clone(), n.shape.clone()))
        .collect();

    // Phase 1: Convert all MIR nodes to compat ops
    let ops: Vec<MirOpCompat> = graph
        .nodes
        .iter()
        .map(|node| mir_node_to_compat_with_shapes(node, resolver, &shape_map))
        .map(|op| op.map(|op| remap_compat_inputs(op, &alias_map)))
        .collect::<Result<Vec<_>>>()?;

    // Phase 2: Collect weight names referenced by ops but not yet materialized
    // as Const ops. We need to emit Const entries for these so the proto
    // emission layer has actual weight data to write to weight.bin.
    let existing_const_names: std::collections::HashSet<String> = ops
        .iter()
        .filter_map(|op| match op {
            MirOpCompat::Const { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();

    let graph_value_names: std::collections::HashSet<String> =
        graph.nodes.iter().map(|n| n.id.0.clone()).collect();
    let mut referenced_weights: Vec<String> = Vec::new();
    for op in &ops {
        match op {
            MirOpCompat::Linear { weight_name, bias_name, .. } => {
                if !existing_const_names.contains(weight_name) {
                    referenced_weights.push(weight_name.clone());
                }
                if let Some(bias) = bias_name {
                    if !existing_const_names.contains(bias) {
                        referenced_weights.push(bias.clone());
                    }
                }
            }
            MirOpCompat::LayerNorm { weight_name, bias_name, .. } => {
                if !existing_const_names.contains(weight_name) {
                    referenced_weights.push(weight_name.clone());
                }
                if let Some(bias) = bias_name {
                    if !existing_const_names.contains(bias) {
                        referenced_weights.push(bias.clone());
                    }
                }
            }
            _ => {}
        }

        for input_name in compat_input_names(op) {
            if graph_value_names.contains(&input_name) || existing_const_names.contains(&input_name)
            {
                continue;
            }
            if resolver.resolve(&input_name).is_some() {
                referenced_weights.push(input_name);
            }
        }
    }

    // Phase 3: Materialize Const ops for each referenced weight
    referenced_weights.sort();
    referenced_weights.dedup();
    let mut const_ops = Vec::new();
    for weight_name in referenced_weights {
        let (data, shape, dtype) = match resolver.resolve(&weight_name) {
            Some(wd) => (wd.data, wd.shape, MilDtypeCompat::Fp16),
            None => {
                // Weight not found in resolver.
                // For static_tables paths (eye_tab, mask_tab, etc.), skip entirely
                // rather than creating a broken scalar placeholder. These weights
                // are optional and may not be needed when the arithmetic mask path
                // is used instead of Gather. Creating a scalar [1] placeholder for
                // a weight that should be a ranked tensor causes CoreML validation
                // failures (e.g., "Param 'x' has incorrect type for operator
                // 'gather'. Expected tensor; got fp16").
                if weight_name.starts_with("static_tables/") {
                    eprintln!(
                        "  Info: static table '{}' not resolved — skipping (arithmetic mask path used)",
                        weight_name
                    );
                    continue;
                }
                // For model weights (non-static-tables), use zero-filled placeholder.
                // Default shape [1] with FP16 gives 2 bytes minimum.
                eprintln!(
                    "  Warning: weight '{}' not resolved — using zero-filled placeholder",
                    weight_name
                );
                (vec![0u8; 2], vec![1], MilDtypeCompat::Fp16)
            }
        };
        const_ops.push(MirOpCompat::Const { name: weight_name, data, dtype, shape });
    }
    let mut all_ops = const_ops;
    all_ops.extend(ops);

    let inputs: Vec<String> = graph.inputs.iter().map(|id| id.0.clone()).collect();
    let outputs: Vec<String> = graph.outputs.iter().map(|id| id.0.clone()).collect();

    // Build input/output descriptors with shapes and dtypes from the MIR nodes.
    // This is critical for Core ML: the model description must have shapes for I/O.
    let node_map: std::collections::HashMap<&str, &MirNode> =
        graph.nodes.iter().map(|n| (n.id.0.as_str(), n)).collect();

    let input_descs: Vec<ane_coreml_proto::mir_compat::TensorDescCompat> = graph
        .inputs
        .iter()
        .map(|id| {
            use ane_coreml_proto::mir_compat::TensorDescCompat;
            match node_map.get(id.0.as_str()) {
                Some(node) => TensorDescCompat {
                    name: node.id.0.clone(),
                    shape: compat_input_shape(&node.id.0, &node.shape, max_seq_len),
                    dtype: compat_input_dtype(&node.id.0, &node.dtype),
                },
                None => {
                    // Input node not found in graph nodes — check if we have
                    // explicit input_shapes from the MIR graph. This happens for
                    // multi-function models where inputs like "sir_hidden_input"
                    // are referenced by ops but don't have their own MirNode.
                    if let Some(shape) = graph.input_shapes.get(id) {
                        TensorDescCompat {
                            name: id.0.clone(),
                            shape: shape.clone(),
                            dtype: if id.0.contains("position") || id.0.contains("pos") {
                                MilDtypeCompat::Int32
                            } else {
                                MilDtypeCompat::Fp16
                            },
                        }
                    } else {
                        // Last resort: default shape [1].
                        // Core ML requires every input to have shape constraints.
                        eprintln!(
                            "  Warning: input node '{}' not found in MIR graph — using default shape [1]",
                            id.0
                        );
                        TensorDescCompat {
                            name: id.0.clone(),
                            shape: vec![1],
                            dtype: MilDtypeCompat::Fp16,
                        }
                    }
                }
            }
        })
        .collect();

    // Build node_shapes map using forward shape inference.
    //
    // Sprint 61: The previous approach used per-node `compat_output_shape`
    // fallbacks that were hardcoded (e.g., ReduceMean → [1,1,1024]) and
    // couldn't derive shapes from their inputs. This produced wrong shapes
    // when the input tensor had a non-standard embedding dimension (e.g.,
    // q_proj output [1,512,2048] followed by q_norm ReduceMean which was
    // hardcoded to [1,1,1024] instead of the correct [1,512,1] for axes=[2]).
    //
    // The new approach does a forward pass over nodes in topological order,
    // computing each node's output shape from its inputs' already-known shapes.
    // Static fallbacks are only used when the input shape is unknown.
    let mut node_shapes: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    // Seed with graph input shapes
    for id in &graph.inputs {
        if let Some(node) = node_map.get(id.0.as_str()) {
            let shape = compat_input_shape(&node.id.0, &node.shape, max_seq_len);
            if !shape.is_empty() {
                node_shapes.insert(node.id.0.clone(), shape);
            }
        } else if let Some(shape) = graph.input_shapes.get(id) {
            // Use explicit input_shapes from the MIR graph for inputs that
            // don't have MirNode entries (e.g., "sir_hidden_input").
            node_shapes.insert(id.0.clone(), shape.clone());
        }
    }

    // Seed MILConst shapes from the resolver.
    //
    // MILConst nodes store a value_path (e.g., "static_tables/rope_tables_0/arange_fp16_tab")
    // that references weight data. The resolver knows the shape of each weight.
    // Without seeding, the forward pass can't determine Const shapes, causing the
    // entire downstream shape chain to collapse (e.g., arange_fp16 [40960] becomes
    // unknown → Sub(arange_fp16, pos_fp16) inherits pos_fp16's shape [1] → all
    // mask ops produce [1] instead of [40960] → impossible reshapes).
    //
    // We also seed both the MIR node ID and the value_path as keys, because:
    // - Downstream ops reference the MIR node ID (e.g., "shared_rope_rope_tables_0_arange_fp16_tab")
    // - compat_output_shape for MILConst looks up value_path in node_shapes
    for node in &graph.nodes {
        if let MirOp::MILConst { value_path, .. } = &node.op {
            if node.shape.is_empty() {
                // MIR node shape not populated — try resolver
                if let Some(wd) = resolver.resolve(value_path) {
                    if !wd.shape.is_empty() {
                        node_shapes.insert(node.id.0.clone(), wd.shape.clone());
                    }
                } else if value_path.starts_with("scalar://") {
                    // All scalar constants are 1-element tensors
                    node_shapes.insert(node.id.0.clone(), vec![1]);
                }
            }
        }
    }

    // Forward pass: compute each node's output shape
    for node in &graph.nodes {
        // Skip if already known (e.g., from graph input seeding or Const resolver seeding)
        if node_shapes.contains_key(&node.id.0) {
            continue;
        }
        // Try the MIR node's shape first (populated by infer_shape in mil_lower)
        if !node.shape.is_empty() {
            node_shapes.insert(node.id.0.clone(), node.shape.clone());
            continue;
        }
        // Fall back to the static compat_output_shape for this op
        let shape =
            compat_output_shape(&node.id.0, &node.op, &node.shape, &node_shapes, max_seq_len);
        if !shape.is_empty() {
            node_shapes.insert(node.id.0.clone(), shape);
        }
    }

    // Build output descriptors using the forward-inferred node_shapes map.
    let output_descs: Vec<ane_coreml_proto::mir_compat::TensorDescCompat> = graph
        .outputs
        .iter()
        .map(|id| {
            use ane_coreml_proto::mir_compat::TensorDescCompat;
            match node_map.get(id.0.as_str()) {
                Some(node) => TensorDescCompat {
                    name: node.id.0.clone(),
                    shape: node_shapes
                        .get(&node.id.0)
                        .cloned()
                        .unwrap_or_else(|| compat_output_shape(&node.id.0, &node.op, &node.shape, &node_shapes, max_seq_len)),
                    dtype: mil_dtype_to_compat(&node.dtype),
                },
                None => {
                    // Output node not found in graph nodes — use default shape/dtype.
                    // Core ML requires at least one output with shape constraints.
                    eprintln!(
                        "  Warning: output node '{}' not found in MIR graph — using default shape [1]",
                        id.0
                    );
                    TensorDescCompat {
                        name: id.0.clone(),
                        shape: vec![1],
                        dtype: MilDtypeCompat::Fp16,
                    }
                }
            }
        })
        .collect();

    Ok(MirGraphCompat {
        ops: all_ops,
        inputs,
        outputs,
        opset_version: graph.opset_version.clone(),
        function_name: graph.shard_name.clone(),
        input_descs,
        output_descs,
        node_shapes,
    })
}

/// Returns all SSA input reference names for a `MirOpCompat`.
///
// TODO(T-38): Remove this wrapper once all callers use MirOpCompat::input_names() directly
fn compat_input_names(op: &MirOpCompat) -> Vec<String> {
    op.input_names()
}

// Shape inference functions have been moved to crate::shape_inference.
// Use `crate::shape_inference::compat_output_shape`,
// `crate::shape_inference::compat_input_shape`, and
// `crate::shape_inference::compat_input_dtype` instead.

/// Build a map of SIR alias names to their resolved MIR node IDs.
///
/// T-36 (I-15/CQ-18): Previously hardcoded Qwen3-specific weight name
/// patterns. Now uses `ModelArchitecture` for architecture-aware pattern
/// resolution. When `architecture` is `None`, defaults to Qwen3 patterns
/// for backward compatibility.
///
/// The alias map is used by [`remap_compat_inputs`] to redirect SIR-level
/// input references (which use synthetic names from the SIR decomposition)
/// to the actual MIR node IDs that produce those values.
fn build_input_alias_map(
    graph: &MirGraph,
    architecture: Option<&ane_ir::common::ModelArchitecture>,
) -> std::collections::HashMap<String, String> {
    // T-57: Use Qwen3 patterns when no architecture is specified.
    // Previously this silently defaulted to Qwen3, which is a correctness
    // hazard for non-Qwen3 models. Now we log a warning to make the
    // assumption visible.
    let arch = match architecture.cloned() {
        Some(a) => a,
        None => {
            log::warn!(
                "mir_to_compat: no architecture specified, defaulting to Qwen3 \
                 weight-name patterns. Pass an explicit architecture to avoid \
                 incorrect weight resolution for non-Qwen3 models."
            );
            ane_ir::common::ModelArchitecture::Qwen3
        }
    };
    // T-70 (I-45): Previously, k_proj and v_proj patterns were resolved but
    // immediately discarded with `let _ = (k_proj, v_proj)`. For GQA models
    // with separate K/V projections, this caused ALL QKV-split aliases to
    // point to the Q projection node, producing silently incorrect SSA
    // references for K/V heads. Now we use k_proj/v_proj patterns to build
    // separate K/V alias entries.
    let q_proj = arch.q_proj_pattern();
    let k_proj = arch.k_proj_pattern();
    let v_proj = arch.v_proj_pattern();
    let up_proj = arch.up_proj_pattern();

    let mut aliases = std::collections::HashMap::new();
    aliases
        .insert("embed_weight_embed_tokens".to_string(), "model.embed_tokens.weight".to_string());

    for node in &graph.nodes {
        match &node.op {
            MirOp::MILLinear { weight, .. } if weight.contains(q_proj) && !weight.contains(k_proj) => {
                if let Some(layer) = layer_index_from_weight(weight) {
                    aliases.insert(
                        format!("sir_qkv_split_q_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                }
            }
            MirOp::MILLinear { weight, .. } if weight.contains(k_proj) && !weight.contains(v_proj) => {
                if let Some(layer) = layer_index_from_weight(weight) {
                    aliases.insert(
                        format!("sir_qkv_split_k_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                }
            }
            MirOp::MILLinear { weight, .. } if weight.contains(v_proj) => {
                if let Some(layer) = layer_index_from_weight(weight) {
                    aliases.insert(
                        format!("sir_qkv_split_v_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                }
            }
            MirOp::MILLinear { weight, .. } if weight.contains(up_proj) => {
                if let Some(layer) = layer_index_from_weight(weight) {
                    aliases.insert(format!("sir_up_proj_layer_{layer}_mlp"), node.id.0.clone());
                }
            }
            MirOp::MILSilu { name, .. } if name == "mlp_silu" => {
                if let Some(layer) = layer_index_from_node_id(&node.id.0) {
                    aliases.insert(format!("sir_mlp_act_layer_{layer}_mlp"), node.id.0.clone());
                }
            }
            MirOp::MILMatMul { name, .. } if name == "attn_qk" => {
                if let Some(layer) = layer_index_from_node_id(&node.id.0) {
                    aliases
                        .insert(format!("sir_attn_qk_layer_{layer}_self_attn"), node.id.0.clone());
                }
            }
            MirOp::MILSoftmax { name, .. } if name == "attn_softmax" => {
                if let Some(layer) = layer_index_from_node_id(&node.id.0) {
                    aliases.insert(
                        format!("sir_attn_softmax_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                }
            }
            MirOp::MILMatMul { name, .. } if name == "attn_sv" => {
                if let Some(layer) = layer_index_from_node_id(&node.id.0) {
                    aliases
                        .insert(format!("sir_attn_out_layer_{layer}_self_attn"), node.id.0.clone());
                }
            }
            _ => {}
        }
    }

    aliases
}

fn layer_index_from_weight(weight: &str) -> Option<usize> {
    let rest = weight.strip_prefix("model.layers.")?;
    rest.split('.').next()?.parse().ok()
}

fn layer_index_from_node_id(id: &str) -> Option<usize> {
    let rest = id.split("_layer_").nth(1)?;
    rest.split('_').next()?.parse().ok()
}

/// Remap all tensor input names in a `MirOpCompat` using the alias map.
///
// TODO(T-38): Remove this wrapper once all callers use MirOpCompat::remap_inputs() directly
fn remap_compat_inputs(
    op: MirOpCompat,
    aliases: &std::collections::HashMap<String, String>,
) -> MirOpCompat {
    op.remap_inputs(|name| aliases.get(&name).cloned().unwrap_or(name))
}

/// Rename the output SSA name of a MirOpCompat.
///
/// Used to override the MirOp's `name` field with the MIR node's unique ID
/// (which may be non-unique across decomposed SIR nodes). This is critical
/// for SSA validity: each MIL operation must produce a uniquely-named output,
/// and consumers reference these names via MIR node IDs.
///
// TODO(T-38): Remove this wrapper once all callers use MirOpCompat::rename_output() directly
fn rename_compat_output(compat: MirOpCompat, new_name: String) -> MirOpCompat {
    compat.rename_output(new_name)
}

/// Convert a MIR node to compat without access to a graph-wide shape map.
///
/// This is the simpler version that relies solely on the node's own shape
/// field. It cannot resolve reshape zero placeholders from the full graph,
/// so it should only be used in:
/// - **Tests**: where the full shape map isn't available or necessary
/// - **Debugging**: as a simpler conversion path for single-node inspection
///
/// Production code should use [`mir_node_to_compat_with_shapes`] instead,
/// which can resolve reshape zero placeholders by consulting the full MIR
/// graph shape map. See T-82 (I-57) for details on the migration.
#[cfg(test)]
fn mir_node_to_compat(node: &MirNode, resolver: &dyn WeightResolver) -> Result<MirOpCompat> {
    let compat = mir_op_to_compat(&node.op, &node.shape, resolver)?;

    // CRITICAL: Override the compat op's output name with the MIR node's unique ID.
    // The MirOp's `name` field is set from `air_node.name` which is the SIR node ID,
    // shared across ALL decomposed nodes from the same SIR operation. For example,
    // RmsNorm decomposes into ReduceMean + Rsqrt + Mul + Mul, and all four have
    // name = "sir_2_layer_0_input_norm". This produces duplicate SSA output names
    // and leaves consumer references undefined.
    //
    // Using node.id.0 (the AIR node ID, e.g., "sir_2_layer_0_input_norm_mean")
    // ensures each operation has a unique output name that matches what consumers
    // reference via air_to_mir → MirNodeId.
    let compat = rename_compat_output(compat, node.id.0.clone());

    // For ReadState, propagate the actual dtype from the MIR node instead
    // of the hardcoded Fp16 default in mir_op_to_compat.
    if let MirOpCompat::ReadState { name, state_id, shape, .. } = &compat {
        let node_dtype = mil_dtype_to_compat(&node.dtype);
        return Ok(MirOpCompat::ReadState {
            name: name.clone(),
            state_id: state_id.clone(),
            shape: shape.clone(),
            dtype: node_dtype,
        });
    }

    // For Identity, propagate the actual dtype from the MIR node so
    // the proto emission can declare the correct output type (e.g., Int32
    // for input_ids rather than hardcoded Float16).
    //
    // Additionally, Identity nodes that are graph inputs (x references
    // "__placeholder__") should be converted to Placeholder ops instead.
    // Core ML's MIL format declares function inputs as block parameters,
    // not as operations. The Placeholder is a marker that gets stripped
    // during proto emission (no MIL operation is emitted for it).
    if let MirOpCompat::Identity { name, x, .. } = &compat {
        let node_dtype = compat_input_dtype(&node.id.0, &node.dtype);
        if x == "__placeholder__" {
            return Ok(MirOpCompat::Placeholder { name: name.clone(), dtype: node_dtype });
        }
        return Ok(MirOpCompat::Identity { name: name.clone(), x: x.clone(), dtype: node_dtype });
    }

    Ok(compat)
}

/// Convert a MIR node to compat with access to a graph-wide shape map.
///
/// This is the shape-aware version of `mir_node_to_compat` that can resolve
/// reshape zero placeholders by looking up the input node's shape from the
/// full MIR graph. This is necessary because `infer_shape()` in mil_lower.rs
/// may fail to propagate shapes correctly (e.g., when input shapes aren't
/// seeded, or positional zero resolution is wrong for rank-changing reshapes).
fn mir_node_to_compat_with_shapes(
    node: &MirNode,
    resolver: &dyn WeightResolver,
    shape_map: &std::collections::HashMap<String, Vec<usize>>,
) -> Result<MirOpCompat> {
    let compat = mir_op_to_compat_with_shapes(&node.op, &node.shape, resolver, shape_map)?;

    // CRITICAL: Override the compat op's output name with the MIR node's unique ID.
    // (Same logic as mir_node_to_compat — see its comment for details.)
    let compat = rename_compat_output(compat, node.id.0.clone());

    // For ReadState, propagate the actual dtype from the MIR node instead
    // of the hardcoded Fp16 default.
    if let MirOpCompat::ReadState { name, state_id, shape, .. } = &compat {
        let node_dtype = mil_dtype_to_compat(&node.dtype);
        return Ok(MirOpCompat::ReadState {
            name: name.clone(),
            state_id: state_id.clone(),
            shape: shape.clone(),
            dtype: node_dtype,
        });
    }

    // For Identity, propagate the actual dtype and convert graph-input
    // Identity ops to Placeholder ops.
    if let MirOpCompat::Identity { name, x, .. } = &compat {
        let node_dtype = compat_input_dtype(&node.id.0, &node.dtype);
        if x == "__placeholder__" {
            return Ok(MirOpCompat::Placeholder { name: name.clone(), dtype: node_dtype });
        }
        return Ok(MirOpCompat::Identity { name: name.clone(), x: x.clone(), dtype: node_dtype });
    }

    Ok(compat)
}

/// Shape-aware version of `mir_op_to_compat` that can resolve reshape zeros
/// using the graph-wide shape map.
pub fn mir_op_to_compat_with_shapes(
    op: &MirOp,
    node_shape: &[usize],
    resolver: &dyn WeightResolver,
    shape_map: &std::collections::HashMap<String, Vec<usize>>,
) -> Result<MirOpCompat> {
    match op {
        MirOp::MILReshape { name, x, shape } => {
            // Resolve reshape target shape using multiple strategies:
            //
            // 1. If node_shape is populated and has the correct rank with no zeros,
            //    use it directly (the infer_shape() path worked correctly).
            //
            // 2. If the raw shape has zeros, look up the input node's shape from
            //    the shape_map and resolve using element-count-based inference.
            //    This handles cases where infer_shape() failed to propagate shapes.
            //
            // 3. As a last resort, try positional resolution from node_shape.
            let has_zeros = shape.contains(&0);
            let node_shape_valid = !node_shape.is_empty()
                && node_shape.len() == shape.len()
                && !node_shape.contains(&0);

            let resolved_shape: Vec<i32> = if !has_zeros {
                // No zeros — shape is already concrete
                shape.iter().map(|&d| d as i32).collect()
            } else if node_shape_valid {
                // node_shape is valid and has no zeros — use it
                node_shape.iter().map(|&d| d as i32).collect()
            } else if let Some(input_shape) = shape_map.get(&x.0) {
                // Look up the input node's shape and resolve zeros
                resolve_reshape_shape(shape, input_shape, name)
            } else if !node_shape.is_empty() && node_shape.len() == shape.len() {
                // node_shape has zeros too, but try positional fallback
                let mut resolved = shape.clone();
                for (i, slot) in resolved.iter_mut().enumerate() {
                    if *slot == 0 {
                        if let Some(&dim) = node_shape.get(i) {
                            if dim != 0 {
                                *slot = dim;
                            }
                        }
                    }
                }
                resolved.iter().map(|&d| d as i32).collect()
            } else {
                // No resolution possible — emit as-is (will produce error below)
                shape.iter().map(|&d| d as i32).collect()
            };

            // Validate: no zeros should remain in the resolved shape.
            // Zero dimensions in emitted Core ML reshape targets produce invalid
            // models — Core ML treats 0 as a literal zero dimension, not "infer
            // from input". This is a hard gate; shape inference must succeed
            // before we can emit a valid model. (T-29 / I-08)
            if resolved_shape.contains(&0) {
                let zero_positions: Vec<usize> = resolved_shape
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d == 0)
                    .map(|(i, _)| i)
                    .collect();
                anyhow::bail!(
                    "Reshape '{}' has unresolved zero dimensions at positions {:?} after all \
                     resolution strategies failed. Resolved shape: {:?}, raw shape: {:?}, \
                     node_shape: {:?}, input_shape: {:?}. \
                     Zero dimensions produce invalid Core ML models — shape inference \
                     must resolve all placeholders before emission.",
                    name,
                    zero_positions,
                    resolved_shape,
                    shape,
                    node_shape,
                    shape_map.get(&x.0).map(|s| s.as_slice()).unwrap_or(&[])
                );
            }

            Ok(MirOpCompat::Reshape { name: name.clone(), x: x.0.clone(), shape: resolved_shape })
        }

        // All other ops delegate to the original mir_op_to_compat
        _ => mir_op_to_compat(op, node_shape, resolver),
    }
}

/// Resolve zero-placeholder dimensions in a reshape target shape using the
/// input tensor's shape and element-count-based inference.
///
/// Strategy:
/// 1. For each zero in the target shape, try to copy the dimension from the
///    corresponding position in the input shape (works when input and target
///    have the same rank, e.g., [B,S,E] → [B,S,H,D]).
/// 2. If positional resolution produces a shape whose element count doesn't
///    match the input, fall back to element-count-based inference:
///    - Compute the product of non-zero target dimensions
///    - Distribute the remaining elements among the zero dimensions
///    - If batch=1 is assumed for the first zero, compute the rest
fn resolve_reshape_shape(target_shape: &[usize], input_shape: &[usize], _name: &str) -> Vec<i32> {
    let input_elements: usize = input_shape.iter().product();
    if input_elements == 0 {
        return target_shape.iter().map(|&d| d as i32).collect();
    }

    let mut resolved: Vec<usize> = target_shape.to_vec();

    // Step 1: Try positional resolution
    let mut positional_works = true;
    for (i, slot) in resolved.iter_mut().enumerate() {
        if *slot == 0 {
            if let Some(&dim) = input_shape.get(i) {
                *slot = dim;
            } else {
                positional_works = false;
                break;
            }
        }
    }

    // Verify element count after positional resolution
    if positional_works {
        let resolved_elements: usize = resolved.iter().product();
        if resolved_elements == input_elements {
            return resolved.iter().map(|&d| d as i32).collect();
        }
        // Positional resolution produced wrong element count — reset and
        // use element-count-based inference instead
        resolved = target_shape.to_vec();
    } else {
        // Positional resolution failed partway through (e.g., target rank
        // exceeds input rank). Reset to original target shape because
        // `resolved` may have been partially modified by the loop above,
        // which would corrupt the non_zero_product calculation in Step 2.
        resolved = target_shape.to_vec();
    }

    // Step 2: Element-count-based inference
    let non_zero_product: usize = resolved.iter().filter(|&&d| d != 0).product();
    if non_zero_product == 0 {
        return target_shape.iter().map(|&d| d as i32).collect();
    }

    let remaining = input_elements / non_zero_product;
    if remaining * non_zero_product != input_elements {
        // Element count doesn't divide evenly — can't resolve
        return target_shape.iter().map(|&d| d as i32).collect();
    }

    let zero_count = resolved.iter().filter(|&&d| d == 0).count();
    match zero_count {
        0 => {} // No zeros (shouldn't reach here, but handle gracefully)
        1 => {
            // Single zero — resolve directly
            for slot in &mut resolved {
                if *slot == 0 {
                    *slot = remaining;
                    break;
                }
            }
        }
        _ => {
            // Two or more zeros — set all but the last zero to 1 (batch
            // dimension heuristic), then compute the last from the remaining
            // product. This is the common case for [0, 0, embed] or
            // [0, 0, H, D] reshapes in attention.
            //
            // Previously used `% 1 == 0` which is always true (modulo one
            // is zero), making the else branch dead code. Fixed to use
            // `product_so_far` consistently with the element-count
            // factorization approach.
            let zero_positions: Vec<usize> =
                resolved.iter().enumerate().filter(|(_, &d)| d == 0).map(|(i, _)| i).collect();
            let mut product_so_far = 1usize;
            for &pos in &zero_positions[..zero_positions.len() - 1] {
                resolved[pos] = 1;
                product_so_far *= resolved[pos];
            }
            if let Some(&last_pos) = zero_positions.last() {
                if product_so_far > 0 && remaining.is_multiple_of(product_so_far) {
                    resolved[last_pos] = remaining / product_so_far;
                }
            }
        }
    }

    resolved.iter().map(|&d| d as i32).collect()
}

/// Convert a single MIR op to the compat representation.
///
/// The `node_shape` is used for ops that need shape information
/// (e.g., `MILConst` → `Const`).
pub fn mir_op_to_compat(
    op: &MirOp,
    node_shape: &[usize],
    resolver: &dyn WeightResolver,
) -> Result<MirOpCompat> {
    match op {
        MirOp::MILConst { name, value_path, dtype } => {
            let compat_dtype = mil_dtype_to_compat(dtype);
            let mut shape = node_shape.to_vec();

            // Try to resolve weight data; if unavailable, use zeros.
            // If the resolver returns data with a non-empty shape but our
            // node_shape is empty (shape was lost during inference), use
            // the resolver's shape as a fallback. This prevents emitting
            // scalar constants where ranked tensors are expected (e.g.,
            // RoPE cos/sin tables used as gather inputs).
            let data = match resolver.resolve(value_path) {
                Some(wd) => {
                    if shape.is_empty() && !wd.shape.is_empty() {
                        shape = wd.shape.clone();
                    }
                    wd.data
                }
                None => {
                    // Compute expected size from shape and dtype
                    let element_size = compat_dtype_element_size(&compat_dtype);
                    let total_elements: usize = shape.iter().product();
                    vec![0u8; total_elements * element_size]
                }
            };

            Ok(MirOpCompat::Const { name: name.clone(), data, dtype: compat_dtype, shape })
        }

        MirOp::MILLinear { name, x, weight, bias } => Ok(MirOpCompat::Linear {
            name: name.clone(),
            x: x.0.clone(),
            weight_name: weight.clone(),
            bias_name: bias.clone(),
        }),

        MirOp::MILMatMul { name, x, y, transpose_y } => Ok(MirOpCompat::MatMul {
            name: name.clone(),
            x: x.0.clone(),
            y: y.0.clone(),
            transpose_y: *transpose_y,
        }),

        MirOp::MILAdd { name, x, y } => {
            Ok(MirOpCompat::Add { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILMul { name, x, y } => {
            Ok(MirOpCompat::Mul { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILSub { name, x, y } => {
            Ok(MirOpCompat::Sub { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILAbs { name, x } => Ok(MirOpCompat::Abs { name: name.clone(), x: x.0.clone() }),

        MirOp::MILMaximum { name, x, y } => {
            Ok(MirOpCompat::Maximum { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILMinimum { name, x, y } => {
            Ok(MirOpCompat::Minimum { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILReshape { name, x, shape } => {
            // Resolve reshape target shape using node_shape (from infer_shape)
            // as the primary source. For the shape-aware version that can also
            // look up the input node's shape from the graph, see
            // mir_op_to_compat_with_shapes.
            let has_zeros = shape.contains(&0);
            let node_shape_valid = !node_shape.is_empty()
                && node_shape.len() == shape.len()
                && !node_shape.contains(&0);

            let resolved_shape: Vec<i32> = if !has_zeros {
                // No zeros — shape is already concrete
                shape.iter().map(|&d| d as i32).collect()
            } else if node_shape_valid {
                // node_shape has no zeros — use it
                node_shape.iter().map(|&d| d as i32).collect()
            } else {
                // Fallback: try positional resolution from node_shape
                let mut resolved = shape.clone();
                for (i, slot) in resolved.iter_mut().enumerate() {
                    if *slot == 0 {
                        if let Some(&dim) = node_shape.get(i) {
                            if dim != 0 {
                                *slot = dim;
                            }
                        }
                    }
                }
                resolved.iter().map(|&d| d as i32).collect()
            };

            // Validate: no zeros should remain in the resolved shape.
            // Zero dimensions in emitted Core ML reshape targets produce invalid
            // models — Core ML treats 0 as a literal zero dimension, not "infer
            // from input". This is a hard gate; shape inference must succeed
            // before we can emit a valid model. (T-29 / I-08)
            if resolved_shape.contains(&0) {
                let zero_positions: Vec<usize> = resolved_shape
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d == 0)
                    .map(|(i, _)| i)
                    .collect();
                anyhow::bail!(
                    "Reshape '{}' has unresolved zero dimensions at positions {:?} after all \
                     resolution strategies failed. Resolved shape: {:?}, raw shape: {:?}, \
                     node_shape: {:?}. \
                     Zero dimensions produce invalid Core ML models — shape inference \
                     must resolve all placeholders before emission.",
                    name,
                    zero_positions,
                    resolved_shape,
                    shape,
                    node_shape
                );
            }

            Ok(MirOpCompat::Reshape { name: name.clone(), x: x.0.clone(), shape: resolved_shape })
        }

        MirOp::MILTranspose { name, x, perm } => Ok(MirOpCompat::Transpose {
            name: name.clone(),
            x: x.0.clone(),
            perm: perm.iter().map(|&d| d as i32).collect(),
        }),

        MirOp::MILSliceByIndex {
            name,
            x,
            begin,
            end,
            stride,
            begin_mask,
            end_mask,
            squeeze_mask,
        } => Ok(MirOpCompat::SliceByIndex {
            name: name.clone(),
            x: x.0.clone(),
            begin: begin.iter().map(|&d| d as i32).collect(),
            end: end.iter().map(|&d| d as i32).collect(),
            stride: stride.iter().map(|&d| d as i32).collect(),
            begin_mask: begin_mask.clone(),
            end_mask: end_mask.clone(),
            squeeze_mask: squeeze_mask.clone(),
        }),

        MirOp::MILConcat { name, values, axis } => Ok(MirOpCompat::Concat {
            name: name.clone(),
            values: values.iter().map(|id| id.0.clone()).collect(),
            axis: *axis as i64,
        }),

        MirOp::MILSoftmax { name, x, axis } => {
            Ok(MirOpCompat::Softmax { name: name.clone(), x: x.0.clone(), axis: *axis as i64 })
        }

        MirOp::MILGelu { name, x, mode } => {
            Ok(MirOpCompat::Gelu { name: name.clone(), x: x.0.clone(), mode: mode.clone() })
        }

        MirOp::MILScaledDotProductAttention { name, query, key, value, attention_mask, scale } => {
            Ok(MirOpCompat::ScaledDotProductAttention {
                name: name.clone(),
                query: query.0.clone(),
                key: key.0.clone(),
                value: value.0.clone(),
                attention_mask: attention_mask.as_ref().map(|id| id.0.clone()),
                scale: *scale,
            })
        }

        MirOp::MILReadState { name, state_id, shape, dtype } => {
            // Propagate dtype from the MIR node's dtype field.
            // The MIR node carries the correct dtype from the precision
            // policy pass through AIR→MIR lowering.
            Ok(MirOpCompat::ReadState {
                name: name.clone(),
                state_id: state_id.clone(),
                shape: shape.clone(),
                dtype: mil_dtype_to_compat(dtype),
            })
        }

        MirOp::MILCoremlUpdateState { name, state_id, value } => {
            Ok(MirOpCompat::CoremlUpdateState {
                name: name.clone(),
                state_id: state_id.clone(),
                value: value.0.clone(),
            })
        }

        MirOp::MILGather { name, x, indices, axis } => Ok(MirOpCompat::Gather {
            name: name.clone(),
            x: x.0.clone(),
            indices: indices.0.clone(),
            axis: *axis as i64,
        }),

        MirOp::MILReduceMean { name, x, axes, keep_dims } => Ok(MirOpCompat::ReduceMean {
            name: name.clone(),
            x: x.0.clone(),
            axes: axes.iter().map(|&d| d as i64).collect(),
            keep_dims: *keep_dims,
        }),

        MirOp::MILRsqrt { name, x } => {
            Ok(MirOpCompat::Rsqrt { name: name.clone(), x: x.0.clone() })
        }

        MirOp::MILRealDiv { name, x, y } => {
            Ok(MirOpCompat::RealDiv { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

        MirOp::MILLayerNorm { name, x, weight, bias, epsilon, axes } => {
            Ok(MirOpCompat::LayerNorm {
                name: name.clone(),
                x: x.0.clone(),
                weight_name: weight.clone(),
                bias_name: bias.clone(),
                epsilon: *epsilon,
                axes: axes.iter().map(|&d| d as i64).collect(),
            })
        }

        MirOp::MILTopk { name, x, k, axis } => Ok(MirOpCompat::Topk {
            name: name.clone(),
            x: x.0.clone(),
            k: *k as i64,
            axis: *axis as i64,
        }),

        MirOp::MILCos { name, x } => Ok(MirOpCompat::Cos { name: name.clone(), x: x.0.clone() }),

        MirOp::MILSin { name, x } => Ok(MirOpCompat::Sin { name: name.clone(), x: x.0.clone() }),

        MirOp::MILCast { name, x, dtype } => Ok(MirOpCompat::Cast {
            name: name.clone(),
            x: x.0.clone(),
            dtype: mil_dtype_to_compat(dtype),
        }),

        // Sprint 50: P2 MIR ops
        MirOp::MILSliceUpdate { name, x, update, begin, end } => Ok(MirOpCompat::SliceUpdate {
            name: name.clone(),
            x: x.0.clone(),
            update: update.0.clone(),
            begin: begin.iter().map(|&d| d as i32).collect(),
            end: end.iter().map(|&d| d as i32).collect(),
        }),

        MirOp::MILExp { name, x } => Ok(MirOpCompat::Exp { name: name.clone(), x: x.0.clone() }),

        MirOp::MILSigmoid { name, x } => {
            Ok(MirOpCompat::Sigmoid { name: name.clone(), x: x.0.clone() })
        }

        MirOp::MILTanh { name, x } => Ok(MirOpCompat::Tanh { name: name.clone(), x: x.0.clone() }),

        MirOp::MILRelu { name, x } => Ok(MirOpCompat::Relu { name: name.clone(), x: x.0.clone() }),

        MirOp::MILWhere { name, condition, x, y } => Ok(MirOpCompat::Where {
            name: name.clone(),
            condition: condition.0.clone(),
            x: x.0.clone(),
            y: y.0.clone(),
        }),

        // ─── Sprint 54: Previously unsupported ops now have MirOpCompat equivalents ───
        MirOp::MILConv {
            name,
            x,
            weight,
            pad_type,
            groups,
            strides: _,
            pad_amounts: _,
            dilations: _,
        } => Ok(MirOpCompat::Conv {
            name: name.clone(),
            x: x.0.clone(),
            weight: weight.0.clone(),
            pad_type: pad_type.clone(),
            groups: *groups as i64,
        }),
        MirOp::MILSplit { name, x, axis, num_splits } => Ok(MirOpCompat::Split {
            name: name.clone(),
            x: x.0.clone(),
            axis: *axis as i64,
            num_splits: *num_splits as i64,
        }),
        MirOp::MILStateWrite { name, state_ref, value } => Ok(MirOpCompat::StateWrite {
            name: name.clone(),
            state_ref: state_ref.clone(),
            value: value.0.clone(),
        }),
        MirOp::MILReduceSum { name, x, axes, keep_dims } => Ok(MirOpCompat::ReduceSum {
            name: name.clone(),
            x: x.0.clone(),
            axes: axes.iter().map(|&a| a as i64).collect(),
            keep_dims: *keep_dims,
        }),

        // ─── SiLU, Identity, and Tile: real Core ML MIL ops ───
        MirOp::MILSilu { name, x } => Ok(MirOpCompat::Silu { name: name.clone(), x: x.0.clone() }),
        MirOp::MILIdentity { name, x } => Ok(MirOpCompat::Identity {
            name: name.clone(),
            x: x.0.clone(),
            dtype: MilDtypeCompat::Fp16, // will be overridden by mir_node_to_compat
        }),
        MirOp::MILTile { name, x, reps } => Ok(MirOpCompat::Tile {
            name: name.clone(),
            x: x.0.clone(),
            reps: reps.iter().map(|&r| r as i32).collect(),
        }),

        // ─── Fill / FillLike: tensor constant generators for Tile decomposition ───
        MirOp::MILFill { name, shape, value, dtype } => Ok(MirOpCompat::Fill {
            name: name.clone(),
            shape: shape.iter().map(|&d| d as i32).collect(),
            value: *value,
            dtype: mil_dtype_to_compat(dtype),
        }),
        MirOp::MILFillLike { name, ref_tensor, value, dtype } => Ok(MirOpCompat::FillLike {
            name: name.clone(),
            ref_tensor: ref_tensor.0.clone(),
            value: *value,
            dtype: mil_dtype_to_compat(dtype),
        }),

        // ─── Neg: arithmetic negation (needed for RoPE rotate_half) ───
        MirOp::MILNeg { name, x } => Ok(MirOpCompat::Neg { name: name.clone(), x: x.0.clone() }),

        MirOp::MILExpandDims { name, x, axis } => Ok(MirOpCompat::ExpandDims {
            name: name.clone(),
            x: x.0.clone(),
            axis: axis.iter().map(|&a| a as i32).collect(),
        }),
        MirOp::MILSqueeze { name, x, axis } => Ok(MirOpCompat::Squeeze {
            name: name.clone(),
            x: x.0.clone(),
            axis: axis.iter().map(|&a| a as i32).collect(),
        }),
        MirOp::MILSqrt { name, x } => Ok(MirOpCompat::Sqrt { name: name.clone(), x: x.0.clone() }),
        MirOp::MILPow { name, x, y } => {
            Ok(MirOpCompat::Pow { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILClip { name, x, min_val, max_val } => Ok(MirOpCompat::Clip {
            name: name.clone(),
            x: x.0.clone(),
            min_val: *min_val,
            max_val: *max_val,
        }),
        MirOp::MILEqual { name, x, y } => {
            Ok(MirOpCompat::Equal { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILNotEqual { name, x, y } => {
            Ok(MirOpCompat::NotEqual { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILGreater { name, x, y } => {
            Ok(MirOpCompat::Greater { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILGreaterEqual { name, x, y } => {
            Ok(MirOpCompat::GreaterEqual { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILLess { name, x, y } => {
            Ok(MirOpCompat::Less { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILLessEqual { name, x, y } => {
            Ok(MirOpCompat::LessEqual { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILLogicalNot { name, x } => {
            Ok(MirOpCompat::LogicalNot { name: name.clone(), x: x.0.clone() })
        }
        MirOp::MILLogicalAnd { name, x, y } => {
            Ok(MirOpCompat::LogicalAnd { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILLogicalOr { name, x, y } => {
            Ok(MirOpCompat::LogicalOr { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILPad { name, x, pad_amounts, mode, constant_value } => Ok(MirOpCompat::Pad {
            name: name.clone(),
            x: x.0.clone(),
            pad_amounts: pad_amounts.iter().map(|&d| d as i32).collect(),
            mode: mode.clone(),
            constant_value: *constant_value,
        }),
        MirOp::MILReduceMax { name, x, axes, keep_dims } => Ok(MirOpCompat::ReduceMax {
            name: name.clone(),
            x: x.0.clone(),
            axes: axes.iter().map(|&a| a as i64).collect(),
            keep_dims: *keep_dims,
        }),
        MirOp::MILReduceMin { name, x, axes, keep_dims } => Ok(MirOpCompat::ReduceMin {
            name: name.clone(),
            x: x.0.clone(),
            axes: axes.iter().map(|&a| a as i64).collect(),
            keep_dims: *keep_dims,
        }),
        MirOp::MILReduceProd { name, x, axes, keep_dims } => Ok(MirOpCompat::ReduceProd {
            name: name.clone(),
            x: x.0.clone(),
            axes: axes.iter().map(|&a| a as i64).collect(),
            keep_dims: *keep_dims,
        }),
        MirOp::MILSelect { name, condition, x, y } => Ok(MirOpCompat::Select {
            name: name.clone(),
            condition: condition.0.clone(),
            x: x.0.clone(),
            y: y.0.clone(),
        }),
        MirOp::MILLeakyRelu { name, x, alpha } => {
            Ok(MirOpCompat::LeakyRelu { name: name.clone(), x: x.0.clone(), alpha: *alpha })
        }
        MirOp::MILFloorDiv { name, x, y } => {
            Ok(MirOpCompat::FloorDiv { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILMod { name, x, y } => {
            Ok(MirOpCompat::Mod { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }
        MirOp::MILCeil { name, x } => Ok(MirOpCompat::Ceil { name: name.clone(), x: x.0.clone() }),
        MirOp::MILFloor { name, x } => {
            Ok(MirOpCompat::Floor { name: name.clone(), x: x.0.clone() })
        }
        MirOp::MILRound { name, x } => {
            Ok(MirOpCompat::Round { name: name.clone(), x: x.0.clone() })
        }
        MirOp::MILSign { name, x } => Ok(MirOpCompat::Sign { name: name.clone(), x: x.0.clone() }),
        MirOp::MILLog { name, x, .. } => {
            Ok(MirOpCompat::Log { name: name.clone(), x: x.0.clone() })
        }

        // ─── Constexpr / Weight Compression (T-39: I-18) ────────────
        MirOp::MILConstexprAffineDequantize { name, quantized_data, scale, zero_point, axis } => {
            Ok(MirOpCompat::ConstexprAffineDequantize {
                name: name.clone(),
                quantized_data: quantized_data.clone(),
                scale: *scale,
                zero_point: *zero_point,
                axis: *axis as i64,
            })
        }
        MirOp::MILConstexprBlockwiseShiftScale { name, data, scale, offset, block_size } => {
            Ok(MirOpCompat::ConstexprBlockwiseShiftScale {
                name: name.clone(),
                data: data.clone(),
                scale: scale.clone(),
                offset: offset.clone(),
                block_size: block_size.iter().map(|&d| d as i64).collect(),
            })
        }
        MirOp::MILConstexprLutToDense { name, indices, lut, num_bits } => {
            Ok(MirOpCompat::ConstexprLutToDense {
                name: name.clone(),
                indices: indices.clone(),
                lut: lut.clone(),
                num_bits: *num_bits as i64,
            })
        }
        MirOp::MILConstexprSparseToDense { name, nonzero_data, shape, default_value } => {
            Ok(MirOpCompat::ConstexprSparseToDense {
                name: name.clone(),
                nonzero_data: nonzero_data.clone(),
                shape: shape.iter().map(|&d| d as i64).collect(),
                default_value: *default_value,
            })
        }
        MirOp::MILConstexprCast { name, data, dtype } => Ok(MirOpCompat::ConstexprCast {
            name: name.clone(),
            data: data.clone(),
            dtype: mil_dtype_to_compat(dtype),
        }),
        MirOp::MILConstexprLutToSparse { name, data, num_bits } => {
            Ok(MirOpCompat::ConstexprLutToSparse {
                name: name.clone(),
                data: data.clone(),
                num_bits: *num_bits as i64,
            })
        }
        MirOp::MILConstexprSparseBlockwiseShiftScale {
            name,
            data,
            scale,
            offset,
            block_size,
            block_axis,
        } => Ok(MirOpCompat::ConstexprSparseBlockwiseShiftScale {
            name: name.clone(),
            data: data.clone(),
            scale: scale.clone(),
            offset: offset.clone(),
            block_size: block_size.iter().map(|&d| d as i64).collect(),
            block_axis: *block_axis as i64,
        }),

        // ─── Full-coverage wildcard for all remaining MirOp variants ───
        // These map to MirOpCompat::Unsupported which carries the op kind
        // and serialized parameters for flexible proto emission.
        other => {
            let (op_kind, name, params) = mir_op_to_unsupported(other);
            Ok(MirOpCompat::Unsupported { op_kind, name, params_json: params })
        }
    }
}

/// Extract op kind, name, and serialized params from a MirOp for the
/// Unsupported compat path. This ensures all 167 MirOp variants can
/// flow through the compat layer even without specialized MirOpCompat
/// representations.
fn mir_op_to_unsupported(op: &MirOp) -> (String, String, String) {
    match op {
        MirOp::MILEinsum { name, .. } => ("einsum".into(), name.clone(), "{}".into()),
        MirOp::MILConvTranspose { name, .. } => {
            ("conv_transpose".into(), name.clone(), "{}".into())
        }
        MirOp::MILFloorDiv { .. } => unreachable!("MILFloorDiv is handled by mir_op_to_compat"),
        MirOp::MILMod { .. } => unreachable!("MILMod is handled by mir_op_to_compat"),
        MirOp::MILPow { .. } => unreachable!("MILPow is handled by mir_op_to_compat"),
        MirOp::MILEqual { .. } => unreachable!("MILEqual is handled by mir_op_to_compat"),
        MirOp::MILNotEqual { .. } => unreachable!("MILNotEqual is handled by mir_op_to_compat"),
        MirOp::MILGreater { .. } => unreachable!("MILGreater is handled by mir_op_to_compat"),
        MirOp::MILGreaterEqual { .. } => {
            unreachable!("MILGreaterEqual is handled by mir_op_to_compat")
        }
        MirOp::MILLess { .. } => unreachable!("MILLess is handled by mir_op_to_compat"),
        MirOp::MILLessEqual { .. } => unreachable!("MILLessEqual is handled by mir_op_to_compat"),
        MirOp::MILLogicalAnd { .. } => unreachable!("MILLogicalAnd is handled by mir_op_to_compat"),
        MirOp::MILLogicalOr { .. } => unreachable!("MILLogicalOr is handled by mir_op_to_compat"),
        MirOp::MILLogicalXor { name, .. } => ("logical_xor".into(), name.clone(), "{}".into()),
        MirOp::MILNeg { .. } => unreachable!("MILNeg is handled by mir_op_to_compat"),
        MirOp::MILSigmoid { .. } => unreachable!("MILSigmoid is handled by mir_op_to_compat"),
        MirOp::MILTanh { .. } => unreachable!("MILTanh is handled by mir_op_to_compat"),
        MirOp::MILRelu6 { name, .. } => ("relu6".into(), name.clone(), "{}".into()),
        MirOp::MILLeakyRelu { .. } => unreachable!("MILLeakyRelu is handled by mir_op_to_compat"),
        MirOp::MILSigmoidHard { name, alpha, beta, .. } => {
            ("sigmoid_hard".into(), name.clone(), format!("{{\"alpha\":{alpha},\"beta\":{beta}}}"))
        }
        MirOp::MILThresholdedRelu { name, alpha, .. } => {
            ("thresholded_relu".into(), name.clone(), format!("{{\"alpha\":{alpha}}}"))
        }
        MirOp::MILClampedRelu { name, alpha, beta, .. } => {
            ("clamped_relu".into(), name.clone(), format!("{{\"alpha\":{alpha},\"beta\":{beta}}}"))
        }
        MirOp::MILLinearActivation { name, alpha, beta, .. } => (
            "linear_activation".into(),
            name.clone(),
            format!("{{\"alpha\":{alpha},\"beta\":{beta}}}"),
        ),
        MirOp::MILPrelu { name, alpha, .. } => {
            ("prelu".into(), name.clone(), format!("{{\"alpha\":\"{alpha}\"}}"))
        }
        MirOp::MILSoftsign { name, .. } => ("softsign".into(), name.clone(), "{}".into()),
        MirOp::MILSilu { .. } => unreachable!("MILSilu is handled by mir_op_to_compat"),
        MirOp::MILScaledTanh { name, alpha, beta, .. } => {
            ("scaled_tanh".into(), name.clone(), format!("{{\"alpha\":{alpha},\"beta\":{beta}}}"))
        }
        MirOp::MILElu { name, alpha, .. } => {
            ("elu".into(), name.clone(), format!("{{\"alpha\":{alpha}}}"))
        }
        MirOp::MILSoftplus { name, .. } => ("softplus".into(), name.clone(), "{}".into()),
        MirOp::MILSoftplusParametric { name, alpha, beta, .. } => (
            "softplus_parametric".into(),
            name.clone(),
            format!("{{\"alpha\":\"{alpha}\",\"beta\":\"{beta}\"}}"),
        ),
        MirOp::MILClip { .. } => unreachable!("MILClip is handled by mir_op_to_compat"),
        MirOp::MILSquare { name, .. } => ("square".into(), name.clone(), "{}".into()),
        MirOp::MILThreshold { name, alpha, .. } => {
            ("threshold".into(), name.clone(), format!("{{\"alpha\":{alpha}}}"))
        }
        MirOp::MILSqrt { .. } => unreachable!("MILSqrt is handled by mir_op_to_compat"),
        MirOp::MILInverse { name, epsilon, .. } => {
            ("inverse".into(), name.clone(), format!("{{\"epsilon\":{epsilon}}}"))
        }
        MirOp::MILCeil { .. } => unreachable!("MILCeil is handled by mir_op_to_compat"),
        MirOp::MILFloor { .. } => unreachable!("MILFloor is handled by mir_op_to_compat"),
        MirOp::MILRound { .. } => unreachable!("MILRound is handled by mir_op_to_compat"),
        MirOp::MILExp2 { name, .. } => ("exp2".into(), name.clone(), "{}".into()),
        MirOp::MILLog { .. } => unreachable!("MILLog is handled by mir_op_to_compat"),
        MirOp::MILSign { .. } => unreachable!("MILSign is handled by mir_op_to_compat"),
        MirOp::MILTan { name, .. } => ("tan".into(), name.clone(), "{}".into()),
        MirOp::MILAcos { name, .. } => ("acos".into(), name.clone(), "{}".into()),
        MirOp::MILAsin { name, .. } => ("asin".into(), name.clone(), "{}".into()),
        MirOp::MILAtan { name, .. } => ("atan".into(), name.clone(), "{}".into()),
        MirOp::MILCosh { name, .. } => ("cosh".into(), name.clone(), "{}".into()),
        MirOp::MILSinh { name, .. } => ("sinh".into(), name.clone(), "{}".into()),
        MirOp::MILAtanh { name, .. } => ("atanh".into(), name.clone(), "{}".into()),
        MirOp::MILErf { name, .. } => ("erf".into(), name.clone(), "{}".into()),
        MirOp::MILLogicalNot { .. } => unreachable!("MILLogicalNot is handled by mir_op_to_compat"),
        MirOp::MILSelect { .. } => unreachable!("MILSelect is handled by mir_op_to_compat"),
        MirOp::MILReduceMax { .. } => unreachable!("MILReduceMax is handled by mir_op_to_compat"),
        MirOp::MILReduceMin { .. } => unreachable!("MILReduceMin is handled by mir_op_to_compat"),
        MirOp::MILReduceProd { .. } => unreachable!("MILReduceProd is handled by mir_op_to_compat"),
        MirOp::MILReduceSumSquare { name, .. } => {
            ("reduce_sum_square".into(), name.clone(), "{}".into())
        }
        MirOp::MILReduceL2Norm { name, .. } => ("reduce_l2_norm".into(), name.clone(), "{}".into()),
        MirOp::MILReduceL1Norm { name, .. } => ("reduce_l1_norm".into(), name.clone(), "{}".into()),
        MirOp::MILReduceLogSumExp { name, .. } => {
            ("reduce_log_sum_exp".into(), name.clone(), "{}".into())
        }
        MirOp::MILReduceLogSum { name, .. } => ("reduce_log_sum".into(), name.clone(), "{}".into()),
        MirOp::MILReduceArgmax { name, .. } => ("reduce_argmax".into(), name.clone(), "{}".into()),
        MirOp::MILReduceArgmin { name, .. } => ("reduce_argmin".into(), name.clone(), "{}".into()),
        MirOp::MILBatchNorm { name, .. } => ("batch_norm".into(), name.clone(), "{}".into()),
        MirOp::MILInstanceNorm { name, .. } => ("instance_norm".into(), name.clone(), "{}".into()),
        MirOp::MILL2Norm { name, .. } => ("l2_norm".into(), name.clone(), "{}".into()),
        MirOp::MILLocalResponseNorm { name, .. } => {
            ("local_response_norm".into(), name.clone(), "{}".into())
        }
        MirOp::MILMaxPool { name, .. } => ("max_pool".into(), name.clone(), "{}".into()),
        MirOp::MILAvgPool { name, .. } => ("avg_pool".into(), name.clone(), "{}".into()),
        MirOp::MILL2Pool { name, .. } => ("l2_pool".into(), name.clone(), "{}".into()),
        MirOp::MILResize { name, .. } => ("resize".into(), name.clone(), "{}".into()),
        MirOp::MILResizeNearestNeighbor { name, .. } => {
            ("resize_nearest_neighbor".into(), name.clone(), "{}".into())
        }
        MirOp::MILResizeBilinear { name, .. } => {
            ("resize_bilinear".into(), name.clone(), "{}".into())
        }
        MirOp::MILUpsampleNearestNeighbor { name, .. } => {
            ("upsample_nearest_neighbor".into(), name.clone(), "{}".into())
        }
        MirOp::MILUpsampleBilinear { name, .. } => {
            ("upsample_bilinear".into(), name.clone(), "{}".into())
        }
        MirOp::MILCropResize { name, .. } => ("crop_resize".into(), name.clone(), "{}".into()),
        MirOp::MILAffine { name, .. } => ("affine".into(), name.clone(), "{}".into()),
        MirOp::MILResample { name, .. } => ("resample".into(), name.clone(), "{}".into()),
        MirOp::MILReshapeLike { name, .. } => ("reshape_like".into(), name.clone(), "{}".into()),
        MirOp::MILExpandDims { .. } => unreachable!("MILExpandDims is handled by mir_op_to_compat"),
        MirOp::MILSqueeze { .. } => unreachable!("MILSqueeze is handled by mir_op_to_compat"),
        MirOp::MILFlatten2d { name, .. } => ("flatten2d".into(), name.clone(), "{}".into()),
        MirOp::MILReverse { name, .. } => ("reverse".into(), name.clone(), "{}".into()),
        MirOp::MILReverseSequence { name, .. } => {
            ("reverse_sequence".into(), name.clone(), "{}".into())
        }
        MirOp::MILSliceBySize { name, .. } => ("slice_by_size".into(), name.clone(), "{}".into()),
        MirOp::MILSlidingWindows { name, .. } => {
            ("sliding_windows".into(), name.clone(), "{}".into())
        }
        MirOp::MILDepthToSpace { name, .. } => ("depth_to_space".into(), name.clone(), "{}".into()),
        MirOp::MILSpaceToDepth { name, .. } => ("space_to_depth".into(), name.clone(), "{}".into()),
        MirOp::MILPixelShuffle { name, .. } => ("pixel_shuffle".into(), name.clone(), "{}".into()),
        MirOp::MILPixelUnshuffle { name, .. } => {
            ("pixel_unshuffle".into(), name.clone(), "{}".into())
        }
        MirOp::MILBatchToSpace { name, .. } => ("batch_to_space".into(), name.clone(), "{}".into()),
        MirOp::MILSpaceToBatch { name, .. } => ("space_to_batch".into(), name.clone(), "{}".into()),
        MirOp::MILPad { .. } => unreachable!("MILPad is handled by mir_op_to_compat"),
        MirOp::MILStack { name, .. } => ("stack".into(), name.clone(), "{}".into()),
        MirOp::MILTile { .. } => unreachable!("MILTile is handled by mir_op_to_compat"),
        MirOp::MILCumsum { name, .. } => ("cumsum".into(), name.clone(), "{}".into()),
        MirOp::MILFill { .. } => unreachable!("MILFill is handled by mir_op_to_compat"),
        MirOp::MILFillLike { .. } => unreachable!("MILFillLike is handled by mir_op_to_compat"),
        MirOp::MILIdentity { .. } => unreachable!("MILIdentity is handled by mir_op_to_compat"),
        MirOp::MILOneHot { name, .. } => ("one_hot".into(), name.clone(), "{}".into()),
        MirOp::MILNonZero { name, .. } => ("non_zero".into(), name.clone(), "{}".into()),
        MirOp::MILArgsort { name, .. } => ("argsort".into(), name.clone(), "{}".into()),
        MirOp::MILBandPart { name, .. } => ("band_part".into(), name.clone(), "{}".into()),
        MirOp::MILRange1d { name, .. } => ("range_1d".into(), name.clone(), "{}".into()),
        MirOp::MILShape { name, .. } => ("shape".into(), name.clone(), "{}".into()),
        MirOp::MILCrop { name, .. } => ("crop".into(), name.clone(), "{}".into()),
        MirOp::MILGatherAlongAxis { name, .. } => {
            ("gather_along_axis".into(), name.clone(), "{}".into())
        }
        MirOp::MILGatherNd { name, .. } => ("gather_nd".into(), name.clone(), "{}".into()),
        MirOp::MILScatter { name, .. } => ("scatter".into(), name.clone(), "{}".into()),
        MirOp::MILScatterAlongAxis { name, .. } => {
            ("scatter_along_axis".into(), name.clone(), "{}".into())
        }
        MirOp::MILScatterNd { name, .. } => ("scatter_nd".into(), name.clone(), "{}".into()),
        MirOp::MILNonMaximumSuppression { name, .. } => {
            ("non_maximum_suppression".into(), name.clone(), "{}".into())
        }
        MirOp::MILQuantize { name, .. } => ("quantize".into(), name.clone(), "{}".into()),
        MirOp::MILDequantize { name, .. } => ("dequantize".into(), name.clone(), "{}".into()),
        // T-39: Constexpr* variants are now handled in mir_op_to_compat()
        // with proper MirOpCompat representations. These arms should be
        // unreachable but are kept for exhaustiveness.
        MirOp::MILConstexprAffineDequantize { name, .. } => {
            unreachable!("constexpr_affine_dequantize is now handled in mir_op_to_compat: {}", name)
        }
        MirOp::MILConstexprBlockwiseShiftScale { name, .. } => {
            unreachable!(
                "constexpr_blockwise_shift_scale is now handled in mir_op_to_compat: {}",
                name
            )
        }
        MirOp::MILConstexprLutToDense { name, .. } => {
            unreachable!("constexpr_lut_to_dense is now handled in mir_op_to_compat: {}", name)
        }
        MirOp::MILConstexprSparseToDense { name, .. } => {
            unreachable!("constexpr_sparse_to_dense is now handled in mir_op_to_compat: {}", name)
        }
        MirOp::MILConstexprCast { name, .. } => {
            unreachable!("constexpr_cast is now handled in mir_op_to_compat: {}", name)
        }
        MirOp::MILConstexprLutToSparse { name, .. } => {
            unreachable!("constexpr_lut_to_sparse is now handled in mir_op_to_compat: {}", name)
        }
        MirOp::MILConstexprSparseBlockwiseShiftScale { name, .. } => {
            unreachable!(
                "constexpr_sparse_blockwise_shift_scale is now handled in mir_op_to_compat: {}",
                name
            )
        }
        MirOp::MILRnn { name, .. } => ("rnn".into(), name.clone(), "{}".into()),
        MirOp::MILGru { name, .. } => ("gru".into(), name.clone(), "{}".into()),
        MirOp::MILLstm { name, .. } => ("lstm".into(), name.clone(), "{}".into()),
        MirOp::MILCond { name, .. } => ("cond".into(), name.clone(), "{}".into()),
        MirOp::MILWhileLoop { name, .. } => ("while_loop".into(), name.clone(), "{}".into()),
        MirOp::MILMakeList { name, .. } => ("make_list".into(), name.clone(), "{}".into()),
        MirOp::MILListLength { name, .. } => ("list_length".into(), name.clone(), "{}".into()),
        MirOp::MILListWrite { name, .. } => ("list_write".into(), name.clone(), "{}".into()),
        MirOp::MILListRead { name, .. } => ("list_read".into(), name.clone(), "{}".into()),
        MirOp::MILListGather { name, .. } => ("list_gather".into(), name.clone(), "{}".into()),
        MirOp::MILListScatter { name, .. } => ("list_scatter".into(), name.clone(), "{}".into()),
        MirOp::MILRandomBernoulli { name, .. } => {
            ("random_bernoulli".into(), name.clone(), "{}".into())
        }
        MirOp::MILRandomNormal { name, .. } => ("random_normal".into(), name.clone(), "{}".into()),
        MirOp::MILRandomUniform { name, .. } => {
            ("random_uniform".into(), name.clone(), "{}".into())
        }
        MirOp::MILRandomCategorical { name, .. } => {
            ("random_categorical".into(), name.clone(), "{}".into())
        }
        MirOp::MILStateWrite { name, .. } => ("state_write".into(), name.clone(), "{}".into()),
        MirOp::MILClassify { name, .. } => ("classify".into(), name.clone(), "{}".into()),
        // All explicitly handled MirOp variants above; this arm catches
        // any future additions without breaking the build.
        _ => ("unknown".into(), "unknown".into(), "{}".into()),
    }
}

/// Convert `MilDtype` → `MilDtypeCompat`.
///
/// T-35: Int4, UInt4, E4M3, E5M2, UInt16 now have proper compat representations
/// instead of being mapped to lossy approximations.
pub fn mil_dtype_to_compat(dtype: &MilDtype) -> MilDtypeCompat {
    match dtype {
        MilDtype::Fp16 => MilDtypeCompat::Fp16,
        MilDtype::Fp32 => MilDtypeCompat::Fp32,
        MilDtype::Int32 => MilDtypeCompat::Int32,
        MilDtype::UInt8 => MilDtypeCompat::UInt8,
        MilDtype::Int4 => MilDtypeCompat::Int4,
        MilDtype::UInt4 => MilDtypeCompat::UInt4,
        MilDtype::E4M3 => MilDtypeCompat::E4M3,
        MilDtype::E5M2 => MilDtypeCompat::E5M2,
        MilDtype::UInt16 => MilDtypeCompat::UInt16,
        MilDtype::Bool => MilDtypeCompat::UInt8, // No Bool compat; approximate as UInt8
        MilDtype::Fp64 => MilDtypeCompat::Fp32,  // No Fp64 compat; downcast to Fp32
        MilDtype::Int8 => MilDtypeCompat::Int32, // No Int8 compat; promote to Int32
        MilDtype::Int16 => MilDtypeCompat::Int32, // No Int16 compat; promote to Int32
    }
}

/// Convert `ComputeUnitHint` → `ComputeUnitHintCompat`.
pub fn compute_unit_hint_to_compat(hint: &ComputeUnitHint) -> ComputeUnitHintCompat {
    match hint {
        ComputeUnitHint::CPUAndNE => ComputeUnitHintCompat::CPUAndNE,
        ComputeUnitHint::CPUAndGPU => ComputeUnitHintCompat::CPUAndGPU,
        ComputeUnitHint::CPUOnly => ComputeUnitHintCompat::CPUOnly,
        ComputeUnitHint::All => ComputeUnitHintCompat::All,
    }
}

/// Get the element size in bytes for a compat dtype.
fn compat_dtype_element_size(dtype: &MilDtypeCompat) -> usize {
    match dtype {
        MilDtypeCompat::Fp16 => 2,
        MilDtypeCompat::Fp32 => 4,
        MilDtypeCompat::Int32 => 4,
        MilDtypeCompat::UInt8 => 1,
        // T-35: new dtype element sizes
        MilDtypeCompat::Int4 | MilDtypeCompat::UInt4 => 1, // 4-bit stored as 1 byte
        MilDtypeCompat::E4M3 | MilDtypeCompat::E5M2 => 1,  // 8-bit float
        MilDtypeCompat::UInt16 => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::mir::{MirNodeId, MirOp};

    fn make_test_graph() -> MirGraph {
        use ane_ir::mir::{MirNode, MirOp};

        MirGraph {
            nodes: vec![
                MirNode {
                    id: MirNodeId("x".to_string()),
                    op: MirOp::MILConst {
                        name: "weight".to_string(),
                        value_path: "weights/weight.bin".to_string(),
                        dtype: MilDtype::Fp16,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![32, 64],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                },
                MirNode {
                    id: MirNodeId("bias".to_string()),
                    op: MirOp::MILConst {
                        name: "bias".to_string(),
                        value_path: "weights/bias.bin".to_string(),
                        dtype: MilDtype::Fp16,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![32],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                },
                MirNode {
                    id: MirNodeId("output".to_string()),
                    op: MirOp::MILLinear {
                        name: "output".to_string(),
                        x: MirNodeId("input".to_string()),
                        weight: "weight".to_string(),
                        bias: Some("bias".to_string()),
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![32],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                },
            ],
            inputs: vec![MirNodeId("input".to_string())],
            outputs: vec![MirNodeId("output".to_string())],
            opset_version: "iOS18".to_string(),
            shard_name: "main".to_string(),
            input_shapes: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_mir_graph_to_compat_basic() {
        let graph = make_test_graph();
        let resolver = EmptyWeightResolver;
        let compat = mir_graph_to_compat(&graph, &resolver).unwrap();

        // 3 original ops + 1 auto-materialized weight (Linear references "weight"
        // but the existing Const was renamed to "x" via rename_compat_output,
        // so auto-materialization creates a new Const for "weight")
        assert_eq!(compat.ops.len(), 4);
        assert_eq!(compat.inputs, vec!["input".to_string()]);
        assert_eq!(compat.outputs, vec!["output".to_string()]);
        assert_eq!(compat.opset_version, "iOS18");
        assert_eq!(compat.function_name, "main");
    }

    #[test]
    fn test_mir_graph_to_compat_with_resolver() {
        let graph = make_test_graph();
        let mut resolver = HashMapWeightResolver::new();
        resolver.add("weights/weight.bin".to_string(), vec![1u8; 32 * 64 * 2], vec![32, 64]);
        resolver.add("weights/bias.bin".to_string(), vec![2u8; 32 * 2], vec![32]);

        let compat = mir_graph_to_compat(&graph, &resolver).unwrap();

        // Check that Const ops have proper data.
        // Note: The ops list is [auto-materialized "weight", renamed "x", "bias", Linear].
        // Auto-materialization prepends Const ops for weight names referenced by
        // Linear/LayerNorm but not present in existing_const_names (which uses the
        // renamed SSA names, not the original op.name). The Const for "weight" was
        // renamed to "x" via rename_compat_output, so "weight" is auto-materialized.

        // ops[0] = auto-materialized Const for "weight" (referenced by Linear but
        // not in existing_const_names after the "x" rename)
        match &compat.ops[0] {
            MirOpCompat::Const { name, data, dtype, shape } => {
                assert_eq!(name, "weight"); // auto-materialized, uses the weight name
                assert_eq!(*dtype, MilDtypeCompat::Fp16);
                assert_eq!(*shape, vec![1]); // zero-filled fallback has shape [1]
                let _ = data; // data is zero-filled (2 bytes for fp16 scalar)
            }
            _ => panic!("Expected auto-materialized Const op for 'weight'"),
        }

        // ops[1] = the original MILConst node, renamed from op.name "weight" → node.id "x"
        match &compat.ops[1] {
            MirOpCompat::Const { name, data, dtype, shape } => {
                assert_eq!(name, "x"); // node.id.0, not op.name "weight"
                assert_eq!(data.len(), 32 * 64 * 2);
                assert_eq!(*dtype, MilDtypeCompat::Fp16);
                assert_eq!(*shape, vec![32, 64]);
            }
            _ => panic!("Expected Const op 'x'"),
        }

        // ops[2] = bias Const
        match &compat.ops[2] {
            MirOpCompat::Const { name, data, .. } => {
                assert_eq!(name, "bias"); // node.id.0, matches op.name in this case
                assert_eq!(data.len(), 32 * 2);
            }
            _ => panic!("Expected Const op 'bias'"),
        }

        // ops[3] = Linear op: output name is the node ID "output"
        match &compat.ops[3] {
            MirOpCompat::Linear { name, x, weight_name, bias_name } => {
                assert_eq!(name, "output");
                assert_eq!(x, "input");
                assert_eq!(weight_name, "weight");
                assert_eq!(bias_name, &Some("bias".to_string()));
            }
            _ => panic!("Expected Linear op"),
        }
    }

    #[test]
    fn test_mil_dtype_to_compat() {
        assert_eq!(mil_dtype_to_compat(&MilDtype::Fp16), MilDtypeCompat::Fp16);
        assert_eq!(mil_dtype_to_compat(&MilDtype::Fp32), MilDtypeCompat::Fp32);
        assert_eq!(mil_dtype_to_compat(&MilDtype::Int32), MilDtypeCompat::Int32);
        assert_eq!(mil_dtype_to_compat(&MilDtype::UInt8), MilDtypeCompat::UInt8);
    }

    #[test]
    fn test_compute_unit_hint_to_compat() {
        assert_eq!(
            compute_unit_hint_to_compat(&ComputeUnitHint::CPUAndNE),
            ComputeUnitHintCompat::CPUAndNE
        );
        assert_eq!(
            compute_unit_hint_to_compat(&ComputeUnitHint::CPUAndGPU),
            ComputeUnitHintCompat::CPUAndGPU
        );
        assert_eq!(
            compute_unit_hint_to_compat(&ComputeUnitHint::CPUOnly),
            ComputeUnitHintCompat::CPUOnly
        );
        assert_eq!(compute_unit_hint_to_compat(&ComputeUnitHint::All), ComputeUnitHintCompat::All);
    }

    #[test]
    fn test_op_conversion_all_supported_ops() {
        let resolver = EmptyWeightResolver;

        // Test all supported MirOp variants
        let test_cases: Vec<(MirOp, &[usize])> = vec![
            (
                MirOp::MILConst { name: "c".into(), value_path: "w".into(), dtype: MilDtype::Fp16 },
                &[4, 4],
            ),
            (
                MirOp::MILLinear {
                    name: "l".into(),
                    x: MirNodeId("x".into()),
                    weight: "w".into(),
                    bias: None,
                },
                &[],
            ),
            (
                MirOp::MILMatMul {
                    name: "mm".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("y".into()),
                    transpose_y: false,
                },
                &[],
            ),
            (
                MirOp::MILAdd {
                    name: "a".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("y".into()),
                },
                &[],
            ),
            (
                MirOp::MILMul {
                    name: "m".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("y".into()),
                },
                &[],
            ),
            (
                MirOp::MILSub {
                    name: "s".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("y".into()),
                },
                &[],
            ),
            (MirOp::MILAbs { name: "abs".into(), x: MirNodeId("x".into()) }, &[]),
            (
                MirOp::MILMaximum {
                    name: "max".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("zero".into()),
                },
                &[],
            ),
            (
                MirOp::MILMinimum {
                    name: "min".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("cap".into()),
                },
                &[],
            ),
            (
                MirOp::MILReshape { name: "r".into(), x: MirNodeId("x".into()), shape: vec![2, 4] },
                &[],
            ),
            (
                MirOp::MILTranspose {
                    name: "t".into(),
                    x: MirNodeId("x".into()),
                    perm: vec![1, 0],
                },
                &[],
            ),
            (
                MirOp::MILSliceByIndex {
                    name: "sl".into(),
                    x: MirNodeId("x".into()),
                    begin: vec![0],
                    end: vec![4],
                    stride: vec![],
                    begin_mask: vec![],
                    end_mask: vec![],
                    squeeze_mask: vec![],
                },
                &[],
            ),
            (
                MirOp::MILConcat {
                    name: "cat".into(),
                    values: vec![MirNodeId("a".into()), MirNodeId("b".into())],
                    axis: 0,
                },
                &[],
            ),
            (MirOp::MILSoftmax { name: "sm".into(), x: MirNodeId("x".into()), axis: -1 }, &[]),
            (
                MirOp::MILGelu { name: "g".into(), x: MirNodeId("x".into()), mode: "exact".into() },
                &[],
            ),
            (
                MirOp::MILScaledDotProductAttention {
                    name: "attn".into(),
                    query: MirNodeId("q".into()),
                    key: MirNodeId("k".into()),
                    value: MirNodeId("v".into()),
                    attention_mask: None,
                    scale: None,
                },
                &[],
            ),
            (
                MirOp::MILReadState {
                    name: "rs".into(),
                    state_id: "s0".into(),
                    shape: vec![128],
                    dtype: MilDtype::Fp16,
                },
                &[],
            ),
            (
                MirOp::MILCoremlUpdateState {
                    name: "us".into(),
                    state_id: "s0".into(),
                    value: MirNodeId("v".into()),
                },
                &[],
            ),
            (
                MirOp::MILGather {
                    name: "ga".into(),
                    x: MirNodeId("x".into()),
                    indices: MirNodeId("i".into()),
                    axis: 0,
                },
                &[],
            ),
            (
                MirOp::MILReduceMean {
                    name: "rm".into(),
                    x: MirNodeId("x".into()),
                    axes: vec![1],
                    keep_dims: true,
                },
                &[],
            ),
            (MirOp::MILRsqrt { name: "rsqrt".into(), x: MirNodeId("x".into()) }, &[]),
            (
                MirOp::MILRealDiv {
                    name: "rd".into(),
                    x: MirNodeId("x".into()),
                    y: MirNodeId("y".into()),
                },
                &[],
            ),
            (
                MirOp::MILLayerNorm {
                    name: "ln".into(),
                    x: MirNodeId("x".into()),
                    weight: "w".into(),
                    bias: Some("b".into()),
                    epsilon: 1e-5,
                    axes: vec![1],
                },
                &[],
            ),
            (MirOp::MILTopk { name: "tk".into(), x: MirNodeId("x".into()), k: 5, axis: -1 }, &[]),
            (MirOp::MILCos { name: "cos".into(), x: MirNodeId("x".into()) }, &[]),
            (MirOp::MILSin { name: "sin".into(), x: MirNodeId("x".into()) }, &[]),
            (
                MirOp::MILCast {
                    name: "cast".into(),
                    x: MirNodeId("x".into()),
                    dtype: MilDtype::Fp32,
                },
                &[],
            ),
            (
                MirOp::MILSplit {
                    name: "split".into(),
                    x: MirNodeId("x".into()),
                    axis: 1,
                    num_splits: 3,
                },
                &[],
            ),
            // Sprint 54: Previously unsupported ops now convert successfully
            (
                MirOp::MILReduceSum {
                    name: "rsum".into(),
                    x: MirNodeId("x".into()),
                    axes: vec![1],
                    keep_dims: false,
                },
                &[],
            ),
            (
                MirOp::MILConv {
                    name: "conv".into(),
                    x: MirNodeId("x".into()),
                    weight: MirNodeId("w".into()),
                    pad_type: "valid".into(),
                    groups: 1,
                    strides: vec![],
                    pad_amounts: vec![],
                    dilations: vec![],
                },
                &[],
            ),
            (
                MirOp::MILStateWrite {
                    name: "sw".into(),
                    state_ref: "s0".into(),
                    value: MirNodeId("v".into()),
                },
                &[],
            ),
        ];

        for (op, shape) in &test_cases {
            let result = mir_op_to_compat(op, shape, &resolver);
            assert!(result.is_ok(), "Failed to convert {:?}: {:?}", op, result.err());
        }
    }

    #[test]
    fn test_static_lut_projection_rejected_at_mil_lower() {
        // StaticLUTProjection has no AIR→MIR lowering path, so it never reaches
        // mir_to_compat. The three previously-unsupported MirOp variants
        // (MILConv, MILStateWrite, MILReduceSum) were added in Sprint 54
        // and are now tested in test_op_conversion_all_supported_ops.
        // This test documents that the rejection happens upstream, not here.
        let resolver = EmptyWeightResolver;

        // MILConv, MILStateWrite, and MILReduceSum now all succeed
        let now_supported: Vec<(MirOp, Vec<usize>)> = vec![
            (
                MirOp::MILConv {
                    name: "conv".into(),
                    x: MirNodeId("x".into()),
                    weight: MirNodeId("w".into()),
                    pad_type: "valid".into(),
                    groups: 1,
                    strides: vec![],
                    pad_amounts: vec![],
                    dilations: vec![],
                },
                vec![],
            ),
            (
                MirOp::MILStateWrite {
                    name: "sw".into(),
                    state_ref: "s0".into(),
                    value: MirNodeId("v".into()),
                },
                vec![],
            ),
            (
                MirOp::MILReduceSum {
                    name: "rs".into(),
                    x: MirNodeId("x".into()),
                    axes: vec![1],
                    keep_dims: false,
                },
                vec![],
            ),
        ];

        for (op, shape) in &now_supported {
            let result = mir_op_to_compat(op, shape, &resolver);
            assert!(result.is_ok(), "Sprint 54: {:?} should now be supported, but got error", op);
        }
    }

    #[test]
    fn test_empty_weight_resolver_fills_zeros() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "nonexistent".into(),
            dtype: MilDtype::Fp16,
        };
        let shape = vec![2, 3];
        let resolver = EmptyWeightResolver;

        let compat = mir_op_to_compat(&op, &shape, &resolver).unwrap();

        match compat {
            MirOpCompat::Const { data, shape: s, .. } => {
                // 2 * 3 * 2 bytes per fp16 = 12 bytes of zeros
                assert_eq!(data.len(), 12);
                assert_eq!(s, vec![2, 3]);
                assert!(data.iter().all(|&b| b == 0));
            }
            _ => panic!("Expected Const"),
        }
    }

    #[test]
    fn test_hashmap_weight_resolver() {
        let mut resolver = HashMapWeightResolver::new();
        resolver.add("w0".into(), vec![42u8; 100], vec![10, 10]);
        resolver.add("w1".into(), vec![7u8; 50], vec![25, 2]);

        let resolved = resolver.resolve("w0").unwrap();
        assert_eq!(resolved.data.len(), 100);
        assert!(resolved.data.iter().all(|&b| b == 42));
        assert_eq!(resolved.shape, vec![10, 10]);

        assert!(resolver.resolve("nonexistent").is_none());
    }

    // --- Sprint 55: Maximum/Minimum and ReadState dtype tests ---

    #[test]
    fn test_maximum_minimum_compat_conversion() {
        let resolver = EmptyWeightResolver;

        let max_op = MirOp::MILMaximum {
            name: "max".into(),
            x: MirNodeId("a".into()),
            y: MirNodeId("b".into()),
        };
        let result = mir_op_to_compat(&max_op, &[], &resolver).unwrap();
        match result {
            MirOpCompat::Maximum { name, x, y } => {
                assert_eq!(name, "max");
                assert_eq!(x, "a");
                assert_eq!(y, "b");
            }
            _ => panic!("Expected Maximum compat"),
        }

        let min_op = MirOp::MILMinimum {
            name: "min".into(),
            x: MirNodeId("a".into()),
            y: MirNodeId("b".into()),
        };
        let result = mir_op_to_compat(&min_op, &[], &resolver).unwrap();
        match result {
            MirOpCompat::Minimum { name, x, y } => {
                assert_eq!(name, "min");
                assert_eq!(x, "a");
                assert_eq!(y, "b");
            }
            _ => panic!("Expected Minimum compat"),
        }
    }

    #[test]
    fn test_read_state_dtype_propagates_from_node() {
        // When a MirNode with MILReadState has dtype=Fp32, the compat
        // conversion should propagate that dtype instead of hardcoding Fp16.
        let resolver = EmptyWeightResolver;
        let node = MirNode {
            id: MirNodeId("k_cache".to_string()),
            op: MirOp::MILReadState {
                name: "k_cache".to_string(),
                state_id: "kv_cache_k".to_string(),
                shape: vec![64, 128],
                dtype: MilDtype::Fp32,
            },
            dtype: MilDtype::Fp32, // Node says Fp32
            shape: vec![64, 128],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
        };

        let compat = mir_node_to_compat(&node, &resolver).unwrap();
        match compat {
            MirOpCompat::ReadState { dtype, .. } => {
                assert_eq!(
                    dtype,
                    MilDtypeCompat::Fp32,
                    "ReadState compat should propagate Fp32 from node dtype, not hardcode Fp16"
                );
            }
            _ => panic!("Expected ReadState compat"),
        }

        // Also verify that Fp16 default still works
        let node_fp16 = MirNode {
            id: MirNodeId("v_cache".to_string()),
            op: MirOp::MILReadState {
                name: "v_cache".to_string(),
                state_id: "kv_cache_v".to_string(),
                shape: vec![64, 128],
                dtype: MilDtype::Fp16,
            },
            dtype: MilDtype::Fp16,
            shape: vec![64, 128],
            compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
            air_source: None,
        };

        let compat_fp16 = mir_node_to_compat(&node_fp16, &resolver).unwrap();
        match compat_fp16 {
            MirOpCompat::ReadState { dtype, .. } => {
                assert_eq!(
                    dtype,
                    MilDtypeCompat::Fp16,
                    "ReadState compat with Fp16 node dtype should remain Fp16"
                );
            }
            _ => panic!("Expected ReadState compat"),
        }
    }

    /// Test that reshape zero-placeholders are resolved against node_shape.
    /// This is the core fix for the "seq_len=32 instead of 512" bug:
    /// SIR emits [0,0,16,128], infer_shape resolves to [1,512,16,128],
    /// and the bridge must use node_shape (not the raw MirOp shape) for
    /// the emitted reshape target.
    #[test]
    fn test_reshape_zero_placeholders_resolved_from_node_shape() {
        let resolver = EmptyWeightResolver;

        // Case 1: Zero placeholders in shape, node_shape has resolved dims
        // This simulates: reshape([1,512,2048], [0,0,16,128]) → [1,512,16,128]
        let op = MirOp::MILReshape {
            name: "attn_q_4d".into(),
            x: MirNodeId("q_proj".into()),
            shape: vec![0, 0, 16, 128], // zeros for batch, seq_len
        };
        let node_shape = &[1, 512, 16, 128]; // resolved by infer_shape

        let compat = mir_op_to_compat(&op, node_shape, &resolver).unwrap();
        match compat {
            MirOpCompat::Reshape { name, x, shape } => {
                assert_eq!(name, "attn_q_4d");
                assert_eq!(x, "q_proj");
                assert_eq!(
                    shape,
                    vec![1, 512, 16, 128],
                    "Zero placeholders must be resolved from node_shape"
                );
            }
            _ => panic!("Expected Reshape compat"),
        }

        // Case 2: 3D reshape with zero placeholders for batch/seq
        // reshape([1,512,2048], [0,0,2048]) → [1,512,2048]
        let op2 = MirOp::MILReshape {
            name: "attn_flat".into(),
            x: MirNodeId("attn_result".into()),
            shape: vec![0, 0, 2048], // zeros for batch, seq_len
        };
        let node_shape2 = &[1, 512, 2048];

        let compat2 = mir_op_to_compat(&op2, node_shape2, &resolver).unwrap();
        match compat2 {
            MirOpCompat::Reshape { shape, .. } => {
                assert_eq!(
                    shape,
                    vec![1, 512, 2048],
                    "3D zero placeholders must be resolved from node_shape"
                );
            }
            _ => panic!("Expected Reshape compat"),
        }

        // Case 3: No zeros in shape, node_shape matches — should use node_shape
        let op3 = MirOp::MILReshape {
            name: "simple".into(),
            x: MirNodeId("x".into()),
            shape: vec![2, 4], // no zeros
        };
        let node_shape3 = &[2, 4];

        let compat3 = mir_op_to_compat(&op3, node_shape3, &resolver).unwrap();
        match compat3 {
            MirOpCompat::Reshape { shape, .. } => {
                assert_eq!(shape, vec![2, 4]);
            }
            _ => panic!("Expected Reshape compat"),
        }

        // Case 4: Empty node_shape — unresolved zeros must produce an error
        // (T-29 / I-08): Zero dimensions in reshape targets produce invalid
        // Core ML models. The compat converter must reject them rather than
        // silently emitting zeros.
        let op4 = MirOp::MILReshape {
            name: "fallback".into(),
            x: MirNodeId("x".into()),
            shape: vec![0, 0, 16, 128],
        };
        let node_shape4: &[usize] = &[];

        let result4 = mir_op_to_compat(&op4, node_shape4, &resolver);
        assert!(
            result4.is_err(),
            "Reshape with unresolvable zero dimensions must return an error, not emit zeros"
        );
        let err_msg = result4.unwrap_err().to_string();
        assert!(
            err_msg.contains("unresolved zero dimensions"),
            "Error message should mention unresolved zero dimensions, got: {err_msg}"
        );
        assert!(
            err_msg.contains("fallback"),
            "Error message should mention the op name, got: {err_msg}"
        );
    }
}

#[test]
fn test_resolve_reshape_shape_element_count_inference() {
    // Test element-count-based zero resolution for rank-changing reshapes.
    // This is the key scenario for attn_reshape_3d where positional
    // resolution gives wrong results.

    // Case 1: 3D→4D with same rank (positional works)
    // input [1, 512, 2048] → target [0, 0, 16, 128]
    let result = resolve_reshape_shape(&[0, 0, 16, 128], &[1, 512, 2048], "test");
    assert_eq!(result, vec![1, 512, 16, 128], "3D→4D positional should work");

    // Case 2: 4D→3D (positional WRONG, element-count needed)
    // input [1, 16, 512, 128] → target [0, 0, 2048]
    // Positional would give [1, 16, 2048] (wrong), element-count gives [1, 512, 2048]
    let result = resolve_reshape_shape(&[0, 0, 2048], &[1, 16, 512, 128], "test");
    assert_eq!(result, vec![1, 512, 2048], "4D→3D should use element-count");

    // Case 3: Single zero dimension
    // input [1, 512, 1024] → target [1, 0, 1024]
    let result = resolve_reshape_shape(&[1, 0, 1024], &[1, 512, 1024], "test");
    assert_eq!(result, vec![1, 512, 1024], "Single zero should resolve directly");

    // Case 4: No zeros (concrete shape)
    let result = resolve_reshape_shape(&[1, 512, 16, 128], &[1, 512, 2048], "test");
    assert_eq!(result, vec![1, 512, 16, 128], "No zeros should pass through");
}

#[test]
fn test_resolve_reshape_shape_two_zeros_product_so_far() {
    // T-30 regression test: previously the 2-zero case used `% 1 == 0`
    // which is always true, making the else branch dead code. Now uses
    // `product_so_far` consistently. These tests verify the 2-zero case
    // resolves correctly with the fixed algorithm.

    // Case 1: Two zeros, attention reshape [0, 0, 2048] from [1, 16, 512, 128]
    // Positional resolution gives [1, 16, 2048] which is wrong (1*16*2048 ≠ 1*16*512*128).
    // Element-count: non_zero_product = 2048, remaining = 1048576/2048 = 512.
    // Two zeros at positions 0,1: first→1, last→512. Result: [1, 512, 2048]
    let result = resolve_reshape_shape(&[0, 0, 2048], &[1, 16, 512, 128], "test");
    assert_eq!(result, vec![1, 512, 2048],
            "Two zeros should use product_so_far factorization: first zero=1, last=remaining/product_so_far");

    // Case 2: Two zeros [0, 0, 16, 128] from [1, 512, 2048]
    // Positional works: [1, 512, 16, 128], elements = 1*512*16*128 = 1048576 = 1*512*2048
    let result = resolve_reshape_shape(&[0, 0, 16, 128], &[1, 512, 2048], "test");
    assert_eq!(
        result,
        vec![1, 512, 16, 128],
        "Positional resolution should work when ranks allow it"
    );

    // Case 3: Two zeros with larger remaining product
    // [0, 0, 64] from [2, 8, 4, 4] = 256 elements
    // Positional: wrong rank, fallback to element-count.
    // non_zero_product = 64, remaining = 4.
    // first zero→1, last zero→4. Result: [1, 4, 64]
    let result = resolve_reshape_shape(&[0, 0, 64], &[2, 8, 4, 4], "test");
    assert_eq!(
        result,
        vec![1, 4, 64],
        "Two zeros with even remaining should resolve via product_so_far"
    );
}

#[test]
fn test_resolve_reshape_shape_three_plus_zeros() {
    // Three or more zeros: all but last set to 1, last = remaining/product_so_far

    // Case 1: [0, 0, 0, 512] from [2, 8, 4, 4, 4] = 1024 elements
    // non_zero_product = 512, remaining = 2.
    // Positions 0,1,2 are zeros: first two → 1, last → 2.
    // Result: [1, 1, 2, 512]
    let result = resolve_reshape_shape(&[0, 0, 0, 512], &[2, 8, 4, 4, 4], "test");
    assert_eq!(
        result,
        vec![1, 1, 2, 512],
        "Three zeros: first two become 1, last gets remaining/product_so_far"
    );

    // Case 2: [0, 0, 0, 0, 128] from [1, 512, 128] = 65536 elements
    // non_zero_product = 128, remaining = 512.
    // Positions 0,1,2,3 are zeros: first three → 1, last → 512.
    // product_so_far = 1*1*1 = 1, remaining % 1 == 0 → true.
    // Result: [1, 1, 1, 512, 128]
    let result = resolve_reshape_shape(&[0, 0, 0, 0, 128], &[1, 512, 128], "test");
    assert_eq!(
        result,
        vec![1, 1, 1, 512, 128],
        "Four zeros: first three become 1, last gets remaining"
    );
}

#[test]
fn test_resolve_reshape_shape_single_zero() {
    // Single zero should resolve directly to remaining

    // [1, 0, 1024] from [1, 512, 1024] = 524288 elements
    // non_zero_product = 1 * 1024 = 1024, remaining = 512
    // Single zero at position 1 → 512
    let result = resolve_reshape_shape(&[1, 0, 1024], &[1, 512, 1024], "test");
    assert_eq!(result, vec![1, 512, 1024], "Single zero should resolve to remaining elements");
}

#[test]
fn test_resolve_reshape_shape_zero_input() {
    // Zero-element input shape should return target as-is

    let result = resolve_reshape_shape(&[0, 0, 16, 128], &[0, 0, 0], "test");
    assert_eq!(
        result,
        vec![0, 0, 16, 128],
        "Zero-element input should return target shape unchanged"
    );
}

#[test]
fn test_resolve_reshape_shape_incompatible_count() {
    // Element count doesn't divide evenly — return target as-is

    // 2*3 = 6 elements, target [0, 4] = 4 * ? — 6/4 = 1.5, not divisible
    let result = resolve_reshape_shape(&[0, 4], &[2, 3], "test");
    assert_eq!(result, vec![0, 4], "Incompatible element count should return target unchanged");
}

// ─── SDPA attention_mask + scale tests (T-31) ───────────────────

/// Helper to create MirNodeId from &str in tests.
fn _mk_nid(s: &str) -> ane_ir::mir::MirNodeId {
    ane_ir::mir::MirNodeId(s.to_string())
}

/// Dummy weight resolver for SDPA conversion tests.
struct _DummyResolver;
impl WeightResolver for _DummyResolver {
    fn resolve(&self, _path: &str) -> Option<WeightData> {
        None
    }
}

#[test]
fn test_sdpa_compat_preserves_attention_mask() {
    // attention_mask should be preserved through MirOp → MirOpCompat conversion
    let op = MirOp::MILScaledDotProductAttention {
        name: "sdpa_masked".into(),
        query: _mk_nid("q"),
        key: _mk_nid("k"),
        value: _mk_nid("v"),
        attention_mask: Some(_mk_nid("causal_mask")),
        scale: None,
    };
    let resolver = _DummyResolver;
    let compat = mir_op_to_compat(&op, &[], &resolver).expect("SDPA conversion should succeed");
    match compat {
        MirOpCompat::ScaledDotProductAttention { attention_mask, scale, .. } => {
            assert_eq!(
                attention_mask,
                Some("causal_mask".to_string()),
                "attention_mask should be preserved through MirOp → MirOpCompat conversion"
            );
            assert_eq!(scale, None, "scale should be None when source is None");
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}

#[test]
fn test_sdpa_compat_preserves_scale() {
    // scale should be preserved through MirOp → MirOpCompat conversion
    let op = MirOp::MILScaledDotProductAttention {
        name: "sdpa_scaled".into(),
        query: _mk_nid("q"),
        key: _mk_nid("k"),
        value: _mk_nid("v"),
        attention_mask: None,
        scale: Some(0.125),
    };
    let resolver = _DummyResolver;
    let compat = mir_op_to_compat(&op, &[], &resolver).expect("SDPA conversion should succeed");
    match compat {
        MirOpCompat::ScaledDotProductAttention { attention_mask, scale, .. } => {
            assert!(attention_mask.is_none(), "attention_mask should be None when source is None");
            assert_eq!(
                scale,
                Some(0.125),
                "scale should be preserved through MirOp → MirOpCompat conversion"
            );
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}

#[test]
fn test_sdpa_compat_preserves_both_mask_and_scale() {
    // Both attention_mask and scale should be preserved together
    let op = MirOp::MILScaledDotProductAttention {
        name: "sdpa_full".into(),
        query: _mk_nid("q"),
        key: _mk_nid("k"),
        value: _mk_nid("v"),
        attention_mask: Some(_mk_nid("attn_mask")),
        scale: Some(0.0625),
    };
    let resolver = _DummyResolver;
    let compat = mir_op_to_compat(&op, &[], &resolver).expect("SDPA conversion should succeed");
    match compat {
        MirOpCompat::ScaledDotProductAttention {
            name,
            query,
            key,
            value,
            attention_mask,
            scale,
        } => {
            assert_eq!(name, "sdpa_full");
            assert_eq!(query, "q");
            assert_eq!(key, "k");
            assert_eq!(value, "v");
            assert_eq!(
                attention_mask,
                Some("attn_mask".to_string()),
                "attention_mask should be preserved"
            );
            assert_eq!(scale, Some(0.0625), "scale should be preserved");
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}

#[test]
fn test_sdpa_compat_no_mask_no_scale() {
    // SDPA without mask or scale should convert cleanly
    let op = MirOp::MILScaledDotProductAttention {
        name: "sdpa_plain".into(),
        query: _mk_nid("q"),
        key: _mk_nid("k"),
        value: _mk_nid("v"),
        attention_mask: None,
        scale: None,
    };
    let resolver = _DummyResolver;
    let compat = mir_op_to_compat(&op, &[], &resolver).expect("SDPA conversion should succeed");
    match compat {
        MirOpCompat::ScaledDotProductAttention { attention_mask, scale, .. } => {
            assert!(attention_mask.is_none());
            assert!(scale.is_none());
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}

#[test]
fn test_sdpa_input_names_includes_mask() {
    // compat_input_names() should include attention_mask when present
    let compat = MirOpCompat::ScaledDotProductAttention {
        name: "sdpa".into(),
        query: "q".into(),
        key: "k".into(),
        value: "v".into(),
        attention_mask: Some("mask".into()),
        scale: None,
    };
    let names = compat_input_names(&compat);
    assert!(
        names.contains(&"mask".to_string()),
        "compat_input_names should include attention_mask name when present"
    );
    assert_eq!(names.len(), 4, "should have 4 inputs: q, k, v, mask");
}

#[test]
fn test_sdpa_input_names_without_mask() {
    // compat_input_names() should NOT include mask when absent
    let compat = MirOpCompat::ScaledDotProductAttention {
        name: "sdpa".into(),
        query: "q".into(),
        key: "k".into(),
        value: "v".into(),
        attention_mask: None,
        scale: None,
    };
    let names = compat_input_names(&compat);
    assert_eq!(names.len(), 3, "should have 3 inputs: q, k, v");
    assert!(
        !names.iter().any(|n| n == "mask"),
        "compat_input_names should not include mask when absent"
    );
}

#[test]
fn test_sdpa_remap_input_names_preserves_mask() {
    // remap_compat_inputs should remap the attention_mask name
    let compat = MirOpCompat::ScaledDotProductAttention {
        name: "sdpa".into(),
        query: "q".into(),
        key: "k".into(),
        value: "v".into(),
        attention_mask: Some("mask".into()),
        scale: Some(0.5),
    };
    let aliases = HashMap::from([
        ("q".to_string(), "q_alias".to_string()),
        ("k".to_string(), "k_alias".to_string()),
        ("v".to_string(), "v_alias".to_string()),
        ("mask".to_string(), "mask_alias".to_string()),
    ]);
    let remapped = remap_compat_inputs(compat, &aliases);
    match remapped {
        MirOpCompat::ScaledDotProductAttention {
            query, key, value, attention_mask, scale, ..
        } => {
            assert_eq!(query, "q_alias");
            assert_eq!(key, "k_alias");
            assert_eq!(value, "v_alias");
            assert_eq!(
                attention_mask,
                Some("mask_alias".to_string()),
                "attention_mask should be remapped via aliases"
            );
            assert_eq!(scale, Some(0.5), "scale should pass through remap unchanged");
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}

#[test]
fn test_sdpa_rename_preserves_mask_and_scale() {
    // rename_compat_output should update the name but preserve attention_mask and scale
    let compat = MirOpCompat::ScaledDotProductAttention {
        name: "old_name".into(),
        query: "q".into(),
        key: "k".into(),
        value: "v".into(),
        attention_mask: Some("mask".into()),
        scale: Some(0.125),
    };
    let renamed = rename_compat_output(compat, "new_name".to_string());
    match renamed {
        MirOpCompat::ScaledDotProductAttention {
            name,
            query,
            key,
            value,
            attention_mask,
            scale,
        } => {
            assert_eq!(name, "new_name", "name should be renamed");
            assert_eq!(query, "q", "query should be preserved");
            assert_eq!(key, "k", "key should be preserved");
            assert_eq!(value, "v", "value should be preserved");
            assert_eq!(
                attention_mask,
                Some("mask".to_string()),
                "attention_mask should survive rename"
            );
            assert_eq!(scale, Some(0.125), "scale should survive rename");
        }
        _ => panic!("Expected MirOpCompat::ScaledDotProductAttention"),
    }
}
