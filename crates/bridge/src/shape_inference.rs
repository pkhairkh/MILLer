//! Shape Inference for MIR-to-Compat Conversion
//!
//! This module provides forward shape inference for the bridge layer,
//! computing output tensor shapes from input tensor shapes when the
//! MIR node's shape field is empty (i.e., when `infer_shape` in
//! `mil_lower` failed to compute a shape).
//!
//! ## Design
//!
//! The main entry point is [`compat_output_shape`], which takes a `MirOp`
//! and a map of already-known node shapes, and returns the inferred output
//! shape. The inference propagates shapes forward through the graph:
//!
//! - **Unary ops** (e.g., `Silu`, `Relu`, `Abs`): propagate the input shape.
//! - **Binary ops** (e.g., `Add`, `Mul`): propagate the first operand shape.
//! - **Reduction ops** (e.g., `ReduceMean`, `ReduceSum`): collapse or set-to-1
//!   the specified axes.
//! - **Structural ops** (e.g., `Reshape`, `Transpose`, `ExpandDims`, `Squeeze`):
//!   compute the output shape from their parameters.
//!
//! When the input shape is unknown, an empty `Vec` is returned, meaning
//! "unknown" — Core ML will attempt to infer from the graph at runtime.

use ane_coreml_proto::mir_compat::MilDtypeCompat;
use ane_ir::mir::{MilDtype, MirOp};
use std::collections::HashMap;

/// Infer the output dtype for a compat graph input node.
///
/// Special-cases `input_ids` (which is `Int32`) and falls back to
/// the MIR node's declared dtype for everything else.
pub fn compat_input_dtype(name: &str, dtype: &MilDtype) -> MilDtypeCompat {
    if name.contains("input_ids") {
        MilDtypeCompat::Int32
    } else {
        crate::mir_to_compat::mil_dtype_to_compat(dtype)
    }
}

/// Infer the shape for a compat graph input node.
///
/// Returns the node's own shape if available, otherwise falls back
/// to a default shape based on the node name (e.g., `[1, 512]` for
/// `input_ids` inputs, `[1]` for others).
///
/// Note: this fallback shape is only for the model's I/O description
/// in the proto, NOT for shape inference. Shape inference uses the
/// `input_shapes` seed from `mil_lower.rs` which is populated from
/// the traced graph's actual input dimensions.
pub fn compat_input_shape(name: &str, shape: &[usize]) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        vec![1, 512]
    } else {
        vec![1]
    }
}

/// Infer the output shape of a MIR operation.
///
/// This function is called when `MirNode.shape` is empty (i.e., when
/// `infer_shape` in `mil_lower` failed to compute a shape). It uses
/// forward shape propagation: looking up input shapes from the
/// `node_shapes` map (populated from earlier nodes in topological order)
/// and computing the output shape from the op's parameters.
///
/// Returns an empty `Vec` when the shape cannot be determined, meaning
/// "unknown" — Core ML will attempt to infer it from the graph.
pub fn compat_output_shape(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        return vec![1, 512];
    }
    match op {
        // ─── Shape-propagating ops: derive from input shapes in node_shapes ───
        // These fallbacks are only hit when MIR node.shape is empty (i.e., when
        // infer_shape in mil_lower failed to compute a shape). When shape inference
        // works correctly, these branches are never reached.
        MirOp::MILReduceMean { x, axes, keep_dims, .. } => {
            reduce_shape(x, axes, *keep_dims, node_shapes)
        }
        MirOp::MILRsqrt { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        MirOp::MILGather { x, indices, axis, .. } => {
            // Embedding: output replaces axis dim of x with indices shape
            match (node_shapes.get(&x.0), node_shapes.get(&indices.0)) {
                (Some(input_shape), Some(indices_shape)) => {
                    let ax = *axis as usize;
                    let mut out = Vec::new();
                    for (i, &dim) in input_shape.iter().enumerate() {
                        if i == ax {
                            out.extend_from_slice(indices_shape);
                        } else {
                            out.push(dim);
                        }
                    }
                    out
                }
                (Some(input_shape), None) => input_shape.clone(),
                _ => vec![],
            }
        }
        // Linear: propagate input shape
        MirOp::MILLinear { x, .. } => {
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
        // Unary ops: propagate input shape
        MirOp::MILSilu { x, .. }
        | MirOp::MILAbs { x, .. }
        | MirOp::MILRelu { x, .. }
        | MirOp::MILSigmoid { x, .. }
        | MirOp::MILTanh { x, .. }
        | MirOp::MILGelu { x, .. }
        | MirOp::MILExp { x, .. }
        | MirOp::MILCos { x, .. }
        | MirOp::MILSin { x, .. }
        | MirOp::MILCast { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // Binary ops: compute broadcast output shape (not just x's shape).
        // Core ML's type inference applies standard numpy-style broadcasting,
        // so the declared output shape must match the broadcast result.
        MirOp::MILAdd { x, y, .. }
        | MirOp::MILMul { x, y, .. }
        | MirOp::MILSub { x, y, .. }
        | MirOp::MILMaximum { x, y, .. }
        | MirOp::MILMinimum { x, y, .. }
        | MirOp::MILRealDiv { x, y, .. } => {
            let shape_a = node_shapes.get(&x.0).cloned().unwrap_or_default();
            let shape_b = node_shapes.get(&y.0).cloned().unwrap_or_default();
            if !shape_a.is_empty() && !shape_b.is_empty() {
                broadcast_shape_compat(&shape_a, &shape_b)
                    .unwrap_or_else(|| shape_a.clone())
            } else if !shape_a.is_empty() {
                shape_a
            } else {
                shape_b
            }
        }
        MirOp::MILSoftmax { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        MirOp::MILMatMul { x, y, .. } => {
            // Sprint 63: proper batched matmul shape inference.
            // [*, M, K] × [*, K, N] → [*, M, N] with right-aligned batch broadcast.
            match (node_shapes.get(&x.0), node_shapes.get(&y.0)) {
                (Some(x_shape), Some(y_shape)) => {
                    let x_rank = x_shape.len();
                    let y_rank = y_shape.len();
                    if x_rank >= 2 && y_rank >= 2 {
                        let lhs_rows = x_shape[x_rank - 2];
                        let _lhs_cols = x_shape[x_rank - 1];
                        let _rhs_rows = y_shape[y_rank - 2];
                        let rhs_cols = y_shape[y_rank - 1];
                        let batch_x = &x_shape[..x_rank - 2];
                        let batch_y = &y_shape[..y_rank - 2];
                        let batch = if batch_x.is_empty() && batch_y.is_empty() {
                            vec![]
                        } else {
                            broadcast_shape_compat(batch_x, batch_y)
                                .unwrap_or_else(|| batch_x.to_vec())
                        };
                        let mut out = batch;
                        out.push(lhs_rows);
                        out.push(rhs_cols);
                        out
                    } else {
                        // Fallback: propagate x shape for degenerate cases
                        x_shape.clone()
                    }
                }
                (Some(x_shape), None) => x_shape.clone(),
                _ => vec![],
            }
        }
        MirOp::MILReshape { shape, .. } => shape.iter().map(|&d| d as usize).collect(),
        MirOp::MILTranspose { x, perm, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                perm.iter().map(|&p| input_shape.get(p as usize).copied().unwrap_or(0)).collect()
            } else {
                vec![]
            }
        }
        MirOp::MILTile { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        MirOp::MILFill { shape, .. } => shape.iter().map(|&d| d as usize).collect(),
        MirOp::MILFillLike { ref_tensor, .. } => {
            node_shapes.get(&ref_tensor.0).cloned().unwrap_or_default()
        }
        MirOp::MILNeg { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // New unary ops: propagate input shape
        MirOp::MILSqrt { x, .. }
        | MirOp::MILLogicalNot { x, .. }
        | MirOp::MILCeil { x, .. }
        | MirOp::MILFloor { x, .. }
        | MirOp::MILRound { x, .. }
        | MirOp::MILSign { x, .. }
        | MirOp::MILLog { x, .. }
        | MirOp::MILLeakyRelu { x, .. }
        | MirOp::MILClip { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // New binary ops: compute broadcast output shape
        MirOp::MILPow { x, y, .. }
        | MirOp::MILFloorDiv { x, y, .. }
        | MirOp::MILMod { x, y, .. }
        | MirOp::MILEqual { x, y, .. }
        | MirOp::MILNotEqual { x, y, .. }
        | MirOp::MILGreater { x, y, .. }
        | MirOp::MILGreaterEqual { x, y, .. }
        | MirOp::MILLess { x, y, .. }
        | MirOp::MILLessEqual { x, y, .. }
        | MirOp::MILLogicalAnd { x, y, .. }
        | MirOp::MILLogicalOr { x, y, .. } => {
            let shape_a = node_shapes.get(&x.0).cloned().unwrap_or_default();
            let shape_b = node_shapes.get(&y.0).cloned().unwrap_or_default();
            if !shape_a.is_empty() && !shape_b.is_empty() {
                broadcast_shape_compat(&shape_a, &shape_b)
                    .unwrap_or_else(|| shape_a.clone())
            } else if !shape_a.is_empty() {
                shape_a
            } else {
                shape_b
            }
        }
        // ExpandDims: insert 1-sized dims at specified axes
        MirOp::MILExpandDims { x, axis, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let mut out = input_shape.clone();
                let mut sorted_axes: Vec<usize> = axis.iter().map(|&a| a as usize).collect();
                sorted_axes.sort_unstable();
                for (i, &ax) in sorted_axes.iter().enumerate() {
                    let insert_pos = if ax >= out.len() { out.len() } else { ax + i };
                    out.insert(insert_pos, 1);
                }
                out
            } else {
                vec![]
            }
        }
        // Squeeze: remove dims at specified axes
        MirOp::MILSqueeze { x, axis, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let mut out = input_shape.clone();
                let mut sorted_axes: Vec<usize> = axis.iter().map(|&a| a as usize).collect();
                sorted_axes.sort_unstable_by(|a, b| b.cmp(a)); // Remove from back to front
                for &ax in &sorted_axes {
                    if ax < out.len() {
                        out.remove(ax);
                    }
                }
                out
            } else {
                vec![]
            }
        }
        // Pad: output shape = input shape + pad amounts
        MirOp::MILPad { x, pad_amounts, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let rank = input_shape.len();
                let mut out = input_shape.clone();
                for i in 0..rank {
                    let before = pad_amounts.get(i).copied().unwrap_or(0) as usize;
                    let after = pad_amounts.get(i + rank).copied().unwrap_or(0) as usize;
                    out[i] += before + after;
                }
                out
            } else {
                vec![]
            }
        }
        // ReduceMax/Min/Prod: same as ReduceMean shape propagation
        MirOp::MILReduceMax { x, axes, keep_dims, .. }
        | MirOp::MILReduceMin { x, axes, keep_dims, .. }
        | MirOp::MILReduceProd { x, axes, keep_dims, .. } => {
            reduce_shape(x, axes, *keep_dims, node_shapes)
        }
        // ─── Concat: sum input dims along the concat axis ───────────────
        // CRITICAL: This was the root cause of the "cannot reshape tensor
        // of size 1 into shape [1, 512, 2048]" error. Without this case,
        // MILConcat fell through to the catch-all `_ => vec![]`, producing
        // an empty shape (scalar) for multi-input attention concats.
        MirOp::MILConcat { values, axis, .. } => {
            if let Some(first_shape) = values.first().and_then(|id| node_shapes.get(&id.0)) {
                let mut out = first_shape.clone();
                let ax = *axis as usize;
                if ax < out.len() {
                    let mut total_dim = 0usize;
                    for id in values {
                        if let Some(shape) = node_shapes.get(&id.0) {
                            if let Some(&dim) = shape.get(ax) {
                                total_dim += dim;
                            }
                        }
                    }
                    out[ax] = total_dim;
                }
                out
            } else {
                vec![]
            }
        }
        // Where: broadcast of condition, x, y (like a ternary elementwise)
        MirOp::MILWhere { condition, x, y, .. } => {
            let shape_c = node_shapes.get(&condition.0).cloned().unwrap_or_default();
            let shape_a = node_shapes.get(&x.0).cloned().unwrap_or_default();
            let shape_b = node_shapes.get(&y.0).cloned().unwrap_or_default();
            // Try broadcasting all three; fall back to x's shape
            if !shape_a.is_empty() && !shape_b.is_empty() && !shape_c.is_empty() {
                broadcast_shape_compat(&shape_a, &shape_b)
                    .and_then(|ab| broadcast_shape_compat(&ab, &shape_c))
                    .unwrap_or_else(|| shape_a.clone())
            } else if !shape_a.is_empty() {
                shape_a
            } else if !shape_c.is_empty() {
                shape_c
            } else {
                shape_b
            }
        }
        // LayerNorm: output shape = input shape
        MirOp::MILLayerNorm { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // Topk: output shape = input shape with axis dim replaced by k
        MirOp::MILTopk { x, k, axis, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let mut out = input_shape.clone();
                let ax = if *axis >= 0 { *axis as usize } else { out.len().saturating_add(*axis as usize) };
                if ax < out.len() {
                    out[ax] = *k;
                }
                out
            } else {
                vec![]
            }
        }
        // ScaledDotProductAttention: output shape = query shape
        MirOp::MILScaledDotProductAttention { query, .. } => {
            node_shapes.get(&query.0).cloned().unwrap_or_default()
        }
        // ReadState: shape is explicitly provided in the op
        MirOp::MILReadState { shape, .. } => shape.clone(),
        // CoremlUpdateState / StateWrite: propagate value shape
        MirOp::MILCoremlUpdateState { value, .. } => {
            node_shapes.get(&value.0).cloned().unwrap_or_default()
        }
        MirOp::MILStateWrite { value, .. } => {
            node_shapes.get(&value.0).cloned().unwrap_or_default()
        }
        // Conv: propagate input shape (output_dim handled at AIR level)
        MirOp::MILConv { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // Select: propagate first operand shape (like Where)
        MirOp::MILSelect { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        MirOp::MILSplit { x, axis, num_splits, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let mut out = input_shape.clone();
                if let Some(dim) = out.get_mut(*axis) {
                    *dim /= num_splits;
                }
                out
            } else {
                vec![]
            }
        }
        // SliceByIndex: compute output shape respecting begin_mask/end_mask/squeeze_mask
        MirOp::MILSliceByIndex { x, begin, end, begin_mask, end_mask, squeeze_mask, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let sliced: Vec<usize> = (0..begin.len())
                    .map(|i| {
                        let b = if begin_mask.get(i).copied().unwrap_or(false) {
                            0
                        } else {
                            begin[i].max(0) as usize
                        };
                        let e = if end_mask.get(i).copied().unwrap_or(false) {
                            input_shape.get(i).copied().unwrap_or(0)
                        } else if end[i] < 0 {
                            let dim = input_shape.get(i).copied().unwrap_or(0) as i64;
                            (dim + end[i]).max(0) as usize
                        } else {
                            end[i] as usize
                        };
                        e.saturating_sub(b)
                    })
                    .collect();
                // Apply squeeze_mask: remove dimensions where squeeze_mask[i] is true
                if !squeeze_mask.is_empty() {
                    sliced.into_iter()
                        .enumerate()
                        .filter(|(i, _)| !squeeze_mask.get(*i).copied().unwrap_or(false))
                        .map(|(_, d)| d)
                        .collect()
                } else {
                    sliced
                }
            } else {
                vec![]
            }
        }
        // Identity for graph inputs (placeholder): use known input shape
        MirOp::MILIdentity { x, .. } if x.0 == "__placeholder__" => vec![1, 512],
        // Identity: propagate input shape
        MirOp::MILIdentity { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
        // ─── Stack: like Concat but inserts a new dimension at `axis` ───
        // Stack([t1, t2, ..., tN], axis) → shape is same as t1 but with a new
        // dim of size N inserted at `axis`. E.g. Stack([a,b], axis=0) where
        // a=[3,4] → [2,3,4].
        MirOp::MILStack { values, axis, .. } => {
            if let Some(first_shape) = values.first().and_then(|id| node_shapes.get(&id.0)) {
                let mut out = first_shape.clone();
                let ax = if *axis <= out.len() { *axis } else { out.len() };
                out.insert(ax, values.len());
                out
            } else {
                vec![]
            }
        }
        // ─── MILConst: look up shape from node_shapes ───
        // Const nodes are seeded into node_shapes by the resolver during the
        // forward pass in mir_graph_to_compat (keyed by MIR node ID, not value_path).
        // The value_path is used to check for scalar:// patterns which are always [1].
        MirOp::MILConst { value_path, .. } => {
            // Check node ID first (seeded by resolver in mir_to_compat forward pass)
            if let Some(shape) = node_shapes.get(name) {
                shape.clone()
            } else if value_path.starts_with("scalar://") {
                // All scalar constants (scalar://fp16/*, scalar://fp32/*) are
                // 1-element tensors. This is critical for mask computation:
                // without it, Sub(arange_fp16[40960], scalar[UNKNOWN]) produces
                // a wrong broadcast result because the scalar shape is unknown.
                vec![1]
            } else {
                vec![]
            }
        }
        // Catch-all: return empty shape rather than a wrong hardcoded value.
        // An empty shape means "unknown" which Core ML will try to infer from
        // the graph — better than a wrong shape that causes type inference failure.
        _ => vec![],
    }
}

/// Compute the broadcast output shape from two input shapes (numpy-style).
///
/// For each dimension pair (right-aligned), the output dimension is the larger
/// of the two inputs. Missing dimensions in the shorter shape are treated as 1.
/// Returns `None` if the shapes are not broadcast-compatible.
fn broadcast_shape_compat(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() { 1 } else { a[i - (max_rank - a.len())] };
        let db = if i < max_rank - b.len() { 1 } else { b[i - (max_rank - b.len())] };
        if da != db && da != 1 && db != 1 {
            return None; // incompatible
        }
        result.push(da.max(db));
    }
    Some(result)
}

/// Compute the output shape of a reduction operation.
///
/// Shared by `ReduceMean`, `ReduceMax`, `ReduceMin`, and `ReduceProd`.
fn reduce_shape(
    x: &ane_ir::mir::MirNodeId,
    axes: &[usize],
    keep_dims: bool,
    node_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    if let Some(input_shape) = node_shapes.get(&x.0) {
        let mut out = input_shape.clone();
        if keep_dims {
            for &ax in axes {
                if ax < out.len() {
                    out[ax] = 1;
                }
            }
        } else {
            let mut sorted_axes: Vec<usize> = axes.to_vec();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
            for &ax in &sorted_axes {
                if ax < out.len() {
                    out.remove(ax);
                }
            }
        }
        out
    } else {
        vec![]
    }
}
