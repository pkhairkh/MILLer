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
use anyhow::Result;
use std::collections::HashMap;

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
    let alias_map = build_input_alias_map(graph);

    // Phase 1: Convert all MIR nodes to compat ops
    let ops: Vec<MirOpCompat> = graph
        .nodes
        .iter()
        .map(|node| mir_node_to_compat(node, resolver))
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
                // Weight not found in resolver — use zero-filled placeholder.
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
                    shape: compat_input_shape(&node.id.0, &node.shape),
                    dtype: compat_input_dtype(&node.id.0, &node.dtype),
                },
                None => {
                    // Input node not found in graph nodes — use default shape/dtype.
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
        })
        .collect();

    let output_descs: Vec<ane_coreml_proto::mir_compat::TensorDescCompat> = graph
        .outputs
        .iter()
        .map(|id| {
            use ane_coreml_proto::mir_compat::TensorDescCompat;
            match node_map.get(id.0.as_str()) {
                Some(node) => TensorDescCompat {
                    name: node.id.0.clone(),
                    shape: compat_output_shape(&node.id.0, &node.op, &node.shape),
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

    // Build node_shapes map from MIR node metadata. This is used by the
    // Apple proto emitter to set correct output types on MIL operations.
    // Without this, all op outputs default to scalar fp16, causing
    // "Tensor storage and type have different number of elements" errors.
    let mut node_shapes: std::collections::HashMap<String, Vec<usize>> = graph
        .nodes
        .iter()
        .map(|n| (n.id.0.clone(), compat_output_shape(&n.id.0, &n.op, &n.shape)))
        .filter(|(_, shape)| !shape.is_empty())
        .collect();
    for id in &graph.inputs {
        if let Some(node) = node_map.get(id.0.as_str()) {
            let shape = compat_input_shape(&node.id.0, &node.shape);
            if !shape.is_empty() {
                node_shapes.insert(node.id.0.clone(), shape);
            }
        }
    }

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

fn compat_input_names(op: &MirOpCompat) -> Vec<String> {
    match op {
        MirOpCompat::Const { .. } | MirOpCompat::Placeholder { .. } => vec![],
        MirOpCompat::Linear { x, weight_name, bias_name, .. } => {
            let mut names = vec![x.clone(), weight_name.clone()];
            if let Some(bias) = bias_name {
                names.push(bias.clone());
            }
            names
        }
        MirOpCompat::MatMul { x, y, .. }
        | MirOpCompat::Add { x, y, .. }
        | MirOpCompat::Mul { x, y, .. }
        | MirOpCompat::Sub { x, y, .. }
        | MirOpCompat::Maximum { x, y, .. }
        | MirOpCompat::Minimum { x, y, .. }
        | MirOpCompat::RealDiv { x, y, .. } => vec![x.clone(), y.clone()],
        MirOpCompat::Abs { x, .. }
        | MirOpCompat::Reshape { x, .. }
        | MirOpCompat::Transpose { x, .. }
        | MirOpCompat::Softmax { x, .. }
        | MirOpCompat::Gelu { x, .. }
        | MirOpCompat::ReduceMean { x, .. }
        | MirOpCompat::ReduceSum { x, .. }
        | MirOpCompat::Rsqrt { x, .. }
        | MirOpCompat::Topk { x, .. }
        | MirOpCompat::Cos { x, .. }
        | MirOpCompat::Sin { x, .. }
        | MirOpCompat::Cast { x, .. }
        | MirOpCompat::Split { x, .. }
        | MirOpCompat::Exp { x, .. }
        | MirOpCompat::Sigmoid { x, .. }
        | MirOpCompat::Tanh { x, .. }
        | MirOpCompat::Relu { x, .. }
        | MirOpCompat::Silu { x, .. }
        | MirOpCompat::Identity { x, .. }
        | MirOpCompat::Tile { x, .. } => vec![x.clone()],
        MirOpCompat::Gather { x, indices, .. } => vec![x.clone(), indices.clone()],
        MirOpCompat::SliceByIndex { x, .. } => vec![x.clone()],
        MirOpCompat::SliceUpdate { x, update, .. } => vec![x.clone(), update.clone()],
        MirOpCompat::Concat { values, .. } => values.clone(),
        MirOpCompat::ScaledDotProductAttention { query, key, value, .. } => {
            vec![query.clone(), key.clone(), value.clone()]
        }
        MirOpCompat::ReadState { state_id, .. } => vec![state_id.clone()],
        MirOpCompat::CoremlUpdateState { state_id, value, .. } => {
            vec![state_id.clone(), value.clone()]
        }
        MirOpCompat::Conv { x, weight, .. } => vec![x.clone(), weight.clone()],
        MirOpCompat::StateWrite { state_ref, value, .. } => vec![state_ref.clone(), value.clone()],
        MirOpCompat::LayerNorm { x, weight_name, bias_name, .. } => {
            let mut names = vec![x.clone(), weight_name.clone()];
            if let Some(bias) = bias_name {
                names.push(bias.clone());
            }
            names
        }
        MirOpCompat::Where { condition, x, y, .. } => vec![condition.clone(), x.clone(), y.clone()],
        MirOpCompat::Unsupported { .. } => vec![],
    }
}

fn compat_input_dtype(name: &str, dtype: &MilDtype) -> MilDtypeCompat {
    if name.contains("input_ids") {
        MilDtypeCompat::Int32
    } else {
        mil_dtype_to_compat(dtype)
    }
}

fn compat_input_shape(name: &str, shape: &[usize]) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        vec![1, 512]
    } else {
        vec![1]
    }
}

fn compat_output_shape(name: &str, op: &MirOp, shape: &[usize]) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        return vec![1, 512];
    }
    match op {
        MirOp::MILGather { .. } => vec![1, 512, 1024],
        MirOp::MILReduceMean { keep_dims: true, .. } => vec![1, 1, 1024],
        MirOp::MILRsqrt { x, .. } if x.0.contains("_mean") => vec![1, 1, 1024],
        MirOp::MILLinear { weight, .. } if weight == "lm_head.weight" => vec![1, 512, 151936],
        MirOp::MILLinear { weight, .. } if weight.contains(".mlp.up_proj.weight") => {
            vec![1, 512, 3072]
        }
        MirOp::MILLinear { weight, .. } if weight.contains(".self_attn.q_proj.weight") => {
            vec![1, 512, 2048]
        }
        MirOp::MILLinear { .. } => vec![1, 512, 1024],
        MirOp::MILMatMul { name, .. } if name == "attn_qk" => vec![1, 512, 512],
        MirOp::MILMatMul { name, .. } if name == "attn_sv" => vec![1, 512, 2048],
        MirOp::MILSoftmax { name, .. } if name == "attn_softmax" => vec![1, 512, 512],
        MirOp::MILSilu { .. } => vec![1, 512, 3072],
        MirOp::MILIdentity { x, .. } if x.0 == "__placeholder__" => vec![1, 512],
        _ if name.contains("_mean") || name.contains("_rsqrt") => vec![1, 1, 1024],
        _ if name.contains("lm_head") || name.contains("_output") => vec![1, 512, 151936],
        _ => vec![1, 512, 1024],
    }
}

fn build_input_alias_map(graph: &MirGraph) -> std::collections::HashMap<String, String> {
    let mut aliases = std::collections::HashMap::new();
    aliases
        .insert("embed_weight_embed_tokens".to_string(), "model.embed_tokens.weight".to_string());

    for node in &graph.nodes {
        match &node.op {
            MirOp::MILLinear { weight, .. } if weight.contains(".self_attn.q_proj.weight") => {
                if let Some(layer) = layer_index_from_weight(weight) {
                    aliases.insert(
                        format!("sir_qkv_split_q_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                    aliases.insert(
                        format!("sir_qkv_split_k_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                    aliases.insert(
                        format!("sir_qkv_split_v_layer_{layer}_self_attn"),
                        node.id.0.clone(),
                    );
                }
            }
            MirOp::MILLinear { weight, .. } if weight.contains(".mlp.up_proj.weight") => {
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

fn remap_name(name: String, aliases: &std::collections::HashMap<String, String>) -> String {
    aliases.get(&name).cloned().unwrap_or(name)
}

fn remap_compat_inputs(
    op: MirOpCompat,
    aliases: &std::collections::HashMap<String, String>,
) -> MirOpCompat {
    match op {
        MirOpCompat::Linear { name, x, weight_name, bias_name } => MirOpCompat::Linear {
            name,
            x: remap_name(x, aliases),
            weight_name: remap_name(weight_name, aliases),
            bias_name: bias_name.map(|v| remap_name(v, aliases)),
        },
        MirOpCompat::MatMul { name, x, y } => {
            MirOpCompat::MatMul { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Add { name, x, y } => {
            MirOpCompat::Add { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Mul { name, x, y } => {
            MirOpCompat::Mul { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Sub { name, x, y } => {
            MirOpCompat::Sub { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Maximum { name, x, y } => {
            MirOpCompat::Maximum { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Minimum { name, x, y } => {
            MirOpCompat::Minimum { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::RealDiv { name, x, y } => {
            MirOpCompat::RealDiv { name, x: remap_name(x, aliases), y: remap_name(y, aliases) }
        }
        MirOpCompat::Abs { name, x } => MirOpCompat::Abs { name, x: remap_name(x, aliases) },
        MirOpCompat::Reshape { name, x, shape } => {
            MirOpCompat::Reshape { name, x: remap_name(x, aliases), shape }
        }
        MirOpCompat::Transpose { name, x, perm } => {
            MirOpCompat::Transpose { name, x: remap_name(x, aliases), perm }
        }
        MirOpCompat::SliceByIndex { name, x, begin, end } => {
            MirOpCompat::SliceByIndex { name, x: remap_name(x, aliases), begin, end }
        }
        MirOpCompat::SliceUpdate { name, x, update, begin, end } => MirOpCompat::SliceUpdate {
            name,
            x: remap_name(x, aliases),
            update: remap_name(update, aliases),
            begin,
            end,
        },
        MirOpCompat::Concat { name, values, axis } => MirOpCompat::Concat {
            name,
            values: values.into_iter().map(|v| remap_name(v, aliases)).collect(),
            axis,
        },
        MirOpCompat::Softmax { name, x, axis } => {
            MirOpCompat::Softmax { name, x: remap_name(x, aliases), axis }
        }
        MirOpCompat::Gelu { name, x, mode } => {
            MirOpCompat::Gelu { name, x: remap_name(x, aliases), mode }
        }
        MirOpCompat::ScaledDotProductAttention { name, query, key, value } => {
            MirOpCompat::ScaledDotProductAttention {
                name,
                query: remap_name(query, aliases),
                key: remap_name(key, aliases),
                value: remap_name(value, aliases),
            }
        }
        MirOpCompat::CoremlUpdateState { name, state_id, value } => {
            MirOpCompat::CoremlUpdateState {
                name,
                state_id: remap_name(state_id, aliases),
                value: remap_name(value, aliases),
            }
        }
        MirOpCompat::Gather { name, x, indices, axis } => MirOpCompat::Gather {
            name,
            x: remap_name(x, aliases),
            indices: remap_name(indices, aliases),
            axis,
        },
        MirOpCompat::ReduceMean { name, x, axes, keep_dims } => {
            MirOpCompat::ReduceMean { name, x: remap_name(x, aliases), axes, keep_dims }
        }
        MirOpCompat::ReduceSum { name, x, axes, keep_dims } => {
            MirOpCompat::ReduceSum { name, x: remap_name(x, aliases), axes, keep_dims }
        }
        MirOpCompat::Conv { name, x, weight, pad_type, groups } => MirOpCompat::Conv {
            name,
            x: remap_name(x, aliases),
            weight: remap_name(weight, aliases),
            pad_type,
            groups,
        },
        MirOpCompat::StateWrite { name, state_ref, value } => MirOpCompat::StateWrite {
            name,
            state_ref: remap_name(state_ref, aliases),
            value: remap_name(value, aliases),
        },
        MirOpCompat::Rsqrt { name, x } => MirOpCompat::Rsqrt { name, x: remap_name(x, aliases) },
        MirOpCompat::LayerNorm { name, x, weight_name, bias_name, epsilon, axes } => {
            MirOpCompat::LayerNorm {
                name,
                x: remap_name(x, aliases),
                weight_name: remap_name(weight_name, aliases),
                bias_name: bias_name.map(|v| remap_name(v, aliases)),
                epsilon,
                axes,
            }
        }
        MirOpCompat::Topk { name, x, k, axis } => {
            MirOpCompat::Topk { name, x: remap_name(x, aliases), k, axis }
        }
        MirOpCompat::Cos { name, x } => MirOpCompat::Cos { name, x: remap_name(x, aliases) },
        MirOpCompat::Sin { name, x } => MirOpCompat::Sin { name, x: remap_name(x, aliases) },
        MirOpCompat::Cast { name, x, dtype } => {
            MirOpCompat::Cast { name, x: remap_name(x, aliases), dtype }
        }
        MirOpCompat::Split { name, x, axis, num_splits } => {
            MirOpCompat::Split { name, x: remap_name(x, aliases), axis, num_splits }
        }
        MirOpCompat::Exp { name, x } => MirOpCompat::Exp { name, x: remap_name(x, aliases) },
        MirOpCompat::Sigmoid { name, x } => {
            MirOpCompat::Sigmoid { name, x: remap_name(x, aliases) }
        }
        MirOpCompat::Tanh { name, x } => MirOpCompat::Tanh { name, x: remap_name(x, aliases) },
        MirOpCompat::Relu { name, x } => MirOpCompat::Relu { name, x: remap_name(x, aliases) },
        MirOpCompat::Where { name, condition, x, y } => MirOpCompat::Where {
            name,
            condition: remap_name(condition, aliases),
            x: remap_name(x, aliases),
            y: remap_name(y, aliases),
        },
        MirOpCompat::Silu { name, x } => MirOpCompat::Silu { name, x: remap_name(x, aliases) },
        MirOpCompat::Identity { name, x, dtype } => {
            MirOpCompat::Identity { name, x: remap_name(x, aliases), dtype }
        }
        MirOpCompat::Tile { name, x, reps } => {
            MirOpCompat::Tile { name, x: remap_name(x, aliases), reps }
        }
        other => other,
    }
}

/// Convert a single MIR node to a compat op.
///
/// Each `MirNode` wraps a `MirOp` plus metadata (dtype, shape, compute_unit_hint).
/// The compat representation only stores the op, so metadata like dtype/shape
/// is folded into the op where relevant (e.g., `Const`, `ReadState`).
/// Rename the output SSA name of a MirOpCompat to use the MIR node's unique ID
/// instead of the MirOp's `name` field (which may be non-unique across decomposed
/// SIR nodes). This is critical for SSA validity: each MIL operation must produce
/// a uniquely-named output, and consumers reference these names via MIR node IDs.
fn rename_compat_output(compat: MirOpCompat, new_name: String) -> MirOpCompat {
    match compat {
        MirOpCompat::Const { name: _, data, dtype, shape } => {
            MirOpCompat::Const { name: new_name, data, dtype, shape }
        }
        MirOpCompat::Linear { name: _, x, weight_name, bias_name } => {
            MirOpCompat::Linear { name: new_name, x, weight_name, bias_name }
        }
        MirOpCompat::MatMul { name: _, x, y } => MirOpCompat::MatMul { name: new_name, x, y },
        MirOpCompat::Add { name: _, x, y } => MirOpCompat::Add { name: new_name, x, y },
        MirOpCompat::Mul { name: _, x, y } => MirOpCompat::Mul { name: new_name, x, y },
        MirOpCompat::Sub { name: _, x, y } => MirOpCompat::Sub { name: new_name, x, y },
        MirOpCompat::Abs { name: _, x } => MirOpCompat::Abs { name: new_name, x },
        MirOpCompat::Maximum { name: _, x, y } => MirOpCompat::Maximum { name: new_name, x, y },
        MirOpCompat::Minimum { name: _, x, y } => MirOpCompat::Minimum { name: new_name, x, y },
        MirOpCompat::Reshape { name: _, x, shape } => {
            MirOpCompat::Reshape { name: new_name, x, shape }
        }
        MirOpCompat::Transpose { name: _, x, perm } => {
            MirOpCompat::Transpose { name: new_name, x, perm }
        }
        MirOpCompat::SliceByIndex { name: _, x, begin, end } => {
            MirOpCompat::SliceByIndex { name: new_name, x, begin, end }
        }
        MirOpCompat::SliceUpdate { name: _, x, update, begin, end } => {
            MirOpCompat::SliceUpdate { name: new_name, x, update, begin, end }
        }
        MirOpCompat::Concat { name: _, values, axis } => {
            MirOpCompat::Concat { name: new_name, values, axis }
        }
        MirOpCompat::Softmax { name: _, x, axis } => {
            MirOpCompat::Softmax { name: new_name, x, axis }
        }
        MirOpCompat::Gelu { name: _, x, mode } => MirOpCompat::Gelu { name: new_name, x, mode },
        MirOpCompat::ScaledDotProductAttention { name: _, query, key, value } => {
            MirOpCompat::ScaledDotProductAttention { name: new_name, query, key, value }
        }
        MirOpCompat::ReadState { name: _, state_id, shape, dtype } => {
            MirOpCompat::ReadState { name: new_name, state_id, shape, dtype }
        }
        MirOpCompat::CoremlUpdateState { name: _, state_id, value } => {
            MirOpCompat::CoremlUpdateState { name: new_name, state_id, value }
        }
        MirOpCompat::Gather { name: _, x, indices, axis } => {
            MirOpCompat::Gather { name: new_name, x, indices, axis }
        }
        MirOpCompat::ReduceMean { name: _, x, axes, keep_dims } => {
            MirOpCompat::ReduceMean { name: new_name, x, axes, keep_dims }
        }
        MirOpCompat::ReduceSum { name: _, x, axes, keep_dims } => {
            MirOpCompat::ReduceSum { name: new_name, x, axes, keep_dims }
        }
        MirOpCompat::Conv { name: _, x, weight, pad_type, groups } => {
            MirOpCompat::Conv { name: new_name, x, weight, pad_type, groups }
        }
        MirOpCompat::StateWrite { name: _, state_ref, value } => {
            MirOpCompat::StateWrite { name: new_name, state_ref, value }
        }
        MirOpCompat::Rsqrt { name: _, x } => MirOpCompat::Rsqrt { name: new_name, x },
        MirOpCompat::RealDiv { name: _, x, y } => MirOpCompat::RealDiv { name: new_name, x, y },
        MirOpCompat::LayerNorm { name: _, x, weight_name, bias_name, epsilon, axes } => {
            MirOpCompat::LayerNorm { name: new_name, x, weight_name, bias_name, epsilon, axes }
        }
        MirOpCompat::Topk { name: _, x, k, axis } => {
            MirOpCompat::Topk { name: new_name, x, k, axis }
        }
        MirOpCompat::Cos { name: _, x } => MirOpCompat::Cos { name: new_name, x },
        MirOpCompat::Sin { name: _, x } => MirOpCompat::Sin { name: new_name, x },
        MirOpCompat::Cast { name: _, x, dtype } => MirOpCompat::Cast { name: new_name, x, dtype },
        MirOpCompat::Split { name: _, x, axis, num_splits } => {
            MirOpCompat::Split { name: new_name, x, axis, num_splits }
        }
        MirOpCompat::Exp { name: _, x } => MirOpCompat::Exp { name: new_name, x },
        MirOpCompat::Sigmoid { name: _, x } => MirOpCompat::Sigmoid { name: new_name, x },
        MirOpCompat::Tanh { name: _, x } => MirOpCompat::Tanh { name: new_name, x },
        MirOpCompat::Relu { name: _, x } => MirOpCompat::Relu { name: new_name, x },
        MirOpCompat::Where { name: _, condition, x, y } => {
            MirOpCompat::Where { name: new_name, condition, x, y }
        }
        MirOpCompat::Silu { name: _, x } => MirOpCompat::Silu { name: new_name, x },
        MirOpCompat::Identity { name: _, x, dtype } => {
            MirOpCompat::Identity { name: new_name, x, dtype }
        }
        MirOpCompat::Tile { name: _, x, reps } => {
            MirOpCompat::Tile { name: new_name, x, reps }
        }
        MirOpCompat::Placeholder { name: _, dtype } => {
            MirOpCompat::Placeholder { name: new_name, dtype }
        }
        MirOpCompat::Unsupported { op_kind, name: _, params_json } => {
            MirOpCompat::Unsupported { op_kind, name: new_name, params_json }
        }
    }
}

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
            let shape = node_shape.to_vec();

            // Try to resolve weight data; if unavailable, use zeros
            let data = match resolver.resolve(value_path) {
                Some(wd) => wd.data,
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

        MirOp::MILMatMul { name, x, y } => {
            Ok(MirOpCompat::MatMul { name: name.clone(), x: x.0.clone(), y: y.0.clone() })
        }

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

        MirOp::MILReshape { name, x, shape } => Ok(MirOpCompat::Reshape {
            name: name.clone(),
            x: x.0.clone(),
            shape: shape.iter().map(|&d| d as i64).collect(),
        }),

        MirOp::MILTranspose { name, x, perm } => Ok(MirOpCompat::Transpose {
            name: name.clone(),
            x: x.0.clone(),
            perm: perm.iter().map(|&d| d as i64).collect(),
        }),

        MirOp::MILSliceByIndex {
            name,
            x,
            begin,
            end,
            stride: _,
            begin_mask: _,
            end_mask: _,
            squeeze_mask: _,
        } => Ok(MirOpCompat::SliceByIndex {
            name: name.clone(),
            x: x.0.clone(),
            begin: begin.clone(),
            end: end.clone(),
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

        MirOp::MILScaledDotProductAttention {
            name,
            query,
            key,
            value,
            attention_mask: _,
            scale: _,
        } => Ok(MirOpCompat::ScaledDotProductAttention {
            name: name.clone(),
            query: query.0.clone(),
            key: key.0.clone(),
            value: value.0.clone(),
        }),

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
            begin: begin.to_vec(),
            end: end.to_vec(),
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
            reps: reps.iter().map(|&r| r as i64).collect(),
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
        MirOp::MILFloorDiv { name, .. } => ("floor_div".into(), name.clone(), "{}".into()),
        MirOp::MILMod { name, .. } => ("mod".into(), name.clone(), "{}".into()),
        MirOp::MILPow { name, .. } => ("pow".into(), name.clone(), "{}".into()),
        MirOp::MILEqual { name, .. } => ("equal".into(), name.clone(), "{}".into()),
        MirOp::MILNotEqual { name, .. } => ("not_equal".into(), name.clone(), "{}".into()),
        MirOp::MILGreater { name, .. } => ("greater".into(), name.clone(), "{}".into()),
        MirOp::MILGreaterEqual { name, .. } => ("greater_equal".into(), name.clone(), "{}".into()),
        MirOp::MILLess { name, .. } => ("less".into(), name.clone(), "{}".into()),
        MirOp::MILLessEqual { name, .. } => ("less_equal".into(), name.clone(), "{}".into()),
        MirOp::MILLogicalAnd { name, .. } => ("logical_and".into(), name.clone(), "{}".into()),
        MirOp::MILLogicalOr { name, .. } => ("logical_or".into(), name.clone(), "{}".into()),
        MirOp::MILLogicalXor { name, .. } => ("logical_xor".into(), name.clone(), "{}".into()),
        MirOp::MILNeg { name, .. } => ("neg".into(), name.clone(), "{}".into()),
        MirOp::MILSigmoid { name, .. } => ("sigmoid".into(), name.clone(), "{}".into()),
        MirOp::MILTanh { name, .. } => ("tanh".into(), name.clone(), "{}".into()),
        MirOp::MILRelu6 { name, .. } => ("relu6".into(), name.clone(), "{}".into()),
        MirOp::MILLeakyRelu { name, alpha, .. } => {
            ("leaky_relu".into(), name.clone(), format!("{{\"alpha\":{alpha}}}"))
        }
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
        MirOp::MILSilu { name, .. } => ("silu".into(), name.clone(), "{}".into()),
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
        MirOp::MILClip { name, min_val, max_val, .. } => (
            "clip".into(),
            name.clone(),
            format!("{{\"min_val\":{min_val},\"max_val\":{max_val}}}"),
        ),
        MirOp::MILSquare { name, .. } => ("square".into(), name.clone(), "{}".into()),
        MirOp::MILThreshold { name, alpha, .. } => {
            ("threshold".into(), name.clone(), format!("{{\"alpha\":{alpha}}}"))
        }
        MirOp::MILSqrt { name, .. } => ("sqrt".into(), name.clone(), "{}".into()),
        MirOp::MILInverse { name, epsilon, .. } => {
            ("inverse".into(), name.clone(), format!("{{\"epsilon\":{epsilon}}}"))
        }
        MirOp::MILCeil { name, .. } => ("ceil".into(), name.clone(), "{}".into()),
        MirOp::MILFloor { name, .. } => ("floor".into(), name.clone(), "{}".into()),
        MirOp::MILRound { name, .. } => ("round".into(), name.clone(), "{}".into()),
        MirOp::MILExp2 { name, .. } => ("exp2".into(), name.clone(), "{}".into()),
        MirOp::MILLog { name, epsilon, .. } => {
            ("log".into(), name.clone(), format!("{{\"epsilon\":{epsilon}}}"))
        }
        MirOp::MILSign { name, .. } => ("sign".into(), name.clone(), "{}".into()),
        MirOp::MILTan { name, .. } => ("tan".into(), name.clone(), "{}".into()),
        MirOp::MILAcos { name, .. } => ("acos".into(), name.clone(), "{}".into()),
        MirOp::MILAsin { name, .. } => ("asin".into(), name.clone(), "{}".into()),
        MirOp::MILAtan { name, .. } => ("atan".into(), name.clone(), "{}".into()),
        MirOp::MILCosh { name, .. } => ("cosh".into(), name.clone(), "{}".into()),
        MirOp::MILSinh { name, .. } => ("sinh".into(), name.clone(), "{}".into()),
        MirOp::MILAtanh { name, .. } => ("atanh".into(), name.clone(), "{}".into()),
        MirOp::MILErf { name, .. } => ("erf".into(), name.clone(), "{}".into()),
        MirOp::MILLogicalNot { name, .. } => ("logical_not".into(), name.clone(), "{}".into()),
        MirOp::MILSelect { name, .. } => ("select".into(), name.clone(), "{}".into()),
        MirOp::MILReduceMax { name, .. } => ("reduce_max".into(), name.clone(), "{}".into()),
        MirOp::MILReduceMin { name, .. } => ("reduce_min".into(), name.clone(), "{}".into()),
        MirOp::MILReduceProd { name, .. } => ("reduce_prod".into(), name.clone(), "{}".into()),
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
        MirOp::MILExpandDims { name, .. } => ("expand_dims".into(), name.clone(), "{}".into()),
        MirOp::MILSqueeze { name, .. } => ("squeeze".into(), name.clone(), "{}".into()),
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
        MirOp::MILPad { name, .. } => ("pad".into(), name.clone(), "{}".into()),
        MirOp::MILStack { name, .. } => ("stack".into(), name.clone(), "{}".into()),
        MirOp::MILTile { .. } => unreachable!("MILTile is handled by mir_op_to_compat"),
        MirOp::MILCumsum { name, .. } => ("cumsum".into(), name.clone(), "{}".into()),
        MirOp::MILFill { name, .. } => ("fill".into(), name.clone(), "{}".into()),
        MirOp::MILFillLike { name, .. } => ("fill_like".into(), name.clone(), "{}".into()),
        MirOp::MILIdentity { name, .. } => ("identity".into(), name.clone(), "{}".into()),
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
        MirOp::MILConstexprAffineDequantize { name, .. } => {
            ("constexpr_affine_dequantize".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprBlockwiseShiftScale { name, .. } => {
            ("constexpr_blockwise_shift_scale".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprLutToDense { name, .. } => {
            ("constexpr_lut_to_dense".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprSparseToDense { name, .. } => {
            ("constexpr_sparse_to_dense".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprCast { name, .. } => {
            ("constexpr_cast".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprLutToSparse { name, .. } => {
            ("constexpr_lut_to_sparse".into(), name.clone(), "{}".into())
        }
        MirOp::MILConstexprSparseBlockwiseShiftScale { name, .. } => {
            ("constexpr_sparse_blockwise_shift_scale".into(), name.clone(), "{}".into())
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
pub fn mil_dtype_to_compat(dtype: &MilDtype) -> MilDtypeCompat {
    match dtype {
        MilDtype::Fp16 => MilDtypeCompat::Fp16,
        MilDtype::Fp32 => MilDtypeCompat::Fp32,
        MilDtype::Int32 => MilDtypeCompat::Int32,
        MilDtype::UInt8 => MilDtypeCompat::UInt8,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::mir::MirNodeId;

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
        // Note: SSA output names use the MIR node ID (from node.id.0),
        // not the MirOp's name field. This ensures unique SSA names.
        match &compat.ops[0] {
            MirOpCompat::Const { name, data, dtype, shape } => {
                assert_eq!(name, "x"); // node.id.0, not op.name "weight"
                assert_eq!(data.len(), 32 * 64 * 2);
                assert_eq!(*dtype, MilDtypeCompat::Fp16);
                assert_eq!(*shape, vec![32, 64]);
            }
            _ => panic!("Expected Const op"),
        }

        match &compat.ops[1] {
            MirOpCompat::Const { name, data, .. } => {
                assert_eq!(name, "bias"); // node.id.0, matches op.name in this case
                assert_eq!(data.len(), 32 * 2);
            }
            _ => panic!("Expected Const op"),
        }

        // Linear op: output name is the node ID "output"
        match &compat.ops[2] {
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
}
