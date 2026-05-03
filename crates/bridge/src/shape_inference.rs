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
use ane_ir::shape_ops;
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
/// to a default shape based on the node name (e.g., `[1, max_seq_len]` for
/// `input_ids` inputs, `[1]` for others).
///
/// T-36 (I-15/CQ-19): The `max_seq_len` parameter replaces the hardcoded
/// `512` fallback. Callers should pass the model's actual max sequence
/// length from `ModelArchConfig::max_seq_len` or equivalent.
///
/// Note: this fallback shape is only for the model's I/O description
/// in the proto, NOT for shape inference. Shape inference uses the
/// `input_shapes` seed from `mil_lower.rs` which is populated from
/// the traced graph's actual input dimensions.
pub fn compat_input_shape(name: &str, shape: &[usize], max_seq_len: usize) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        vec![1, max_seq_len]
    } else {
        vec![1]
    }
}

/// Infer the shape for a compat graph input node using default Qwen3-0.6B
/// max sequence length (32768).
///
/// Convenience wrapper for callers that don't have a `ModelArchConfig` available.
/// Prefer [`compat_input_shape`] with an explicit `max_seq_len` for correctness.
pub fn compat_input_shape_default(name: &str, shape: &[usize]) -> Vec<usize> {
    compat_input_shape(name, shape, 32768)
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
///
/// T-36 (I-15/CQ-19): The `max_seq_len` parameter replaces the hardcoded
/// `512` fallback for input_ids and placeholder nodes. Callers should pass
/// the model's actual max sequence length from `ModelArchConfig::max_seq_len`.
pub fn compat_output_shape(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
    max_seq_len: usize,
) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    if name.contains("input_ids") {
        return vec![1, max_seq_len];
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
        MirOp::MILLinear { x, .. } => node_shapes.get(&x.0).cloned().unwrap_or_default(),
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
                broadcast_shape_compat(&shape_a, &shape_b).unwrap_or_else(|| shape_a.clone())
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
        MirOp::MILReshape { shape, .. } => shape.to_vec(),
        MirOp::MILTranspose { x, perm, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                shape_ops::transpose_shape(input_shape, perm)
            } else {
                vec![]
            }
        }
        MirOp::MILTile { x, reps, .. } => {
            // CQ-22 fix: previously just propagated input shape, ignoring reps.
            // Now correctly computes out[i] = input_shape[i] * reps[i].
            if let Some(input_shape) = node_shapes.get(&x.0) {
                shape_ops::tile_shape(input_shape, reps)
            } else {
                vec![]
            }
        }
        MirOp::MILFill { shape, .. } => shape.to_vec(),
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
                broadcast_shape_compat(&shape_a, &shape_b).unwrap_or_else(|| shape_a.clone())
            } else if !shape_a.is_empty() {
                shape_a
            } else {
                shape_b
            }
        }
        // ExpandDims: insert 1-sized dims at specified axes
        MirOp::MILExpandDims { x, axis, .. } => {
            // CQ-22: Delegates to shape_ops::expand_dims_shape which uses
            // Core ML semantics (axes specify output positions).
            if let Some(input_shape) = node_shapes.get(&x.0) {
                shape_ops::expand_dims_shape(input_shape, axis)
            } else {
                vec![]
            }
        }
        // Squeeze: remove dims at specified axes
        MirOp::MILSqueeze { x, axis, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                shape_ops::squeeze_shape(input_shape, axis)
            } else {
                vec![]
            }
        }
        // Pad: output shape = input shape + pad amounts
        MirOp::MILPad { x, pad_amounts, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let pad: Vec<usize> = pad_amounts.iter().map(|&p| p as usize).collect();
                shape_ops::pad_shape(input_shape, &pad)
            } else {
                vec![]
            }
        }
        // ReduceMax/Min/Prod: same as ReduceMean shape propagation
        MirOp::MILReduceMax { x, axes, keep_dims, .. }
        | MirOp::MILReduceMin { x, axes, keep_dims, .. }
        | MirOp::MILReduceProd { x, axes, keep_dims, .. }
        // Additional reduce ops (CQ-22: previously fell through to empty vec)
        | MirOp::MILReduceSum { x, axes, keep_dims, .. }
        | MirOp::MILReduceSumSquare { x, axes, keep_dims, .. }
        | MirOp::MILReduceL2Norm { x, axes, keep_dims, .. }
        | MirOp::MILReduceL1Norm { x, axes, keep_dims, .. }
        | MirOp::MILReduceLogSumExp { x, axes, keep_dims, .. }
        | MirOp::MILReduceLogSum { x, axes, keep_dims, .. } => {
            reduce_shape(x, axes, *keep_dims, node_shapes)
        }
        // ReduceArgmax/Argmin: reduce along a single axis
        MirOp::MILReduceArgmax { x, axis, keep_dims, .. }
        | MirOp::MILReduceArgmin { x, axis, keep_dims, .. } => {
            reduce_shape(x, &[*axis], *keep_dims, node_shapes)
        }
        // ─── Concat: sum input dims along the concat axis ───────────────
        // CRITICAL: This was the root cause of the "cannot reshape tensor
        // of size 1 into shape [1, 512, 2048]" error. Without this case,
        // MILConcat fell through to the catch-all `_ => vec![]`, producing
        // an empty shape (scalar) for multi-input attention concats.
        MirOp::MILConcat { values, axis, .. } => {
            if let Some(first_shape) = values.first().and_then(|id| node_shapes.get(&id.0)) {
                let mut out = first_shape.clone();
                let ax = *axis;
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
                let rank = out.len() as isize;
                let ax = if *axis >= 0 { *axis as usize } else { (rank + axis) as usize };
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
                    sliced
                        .into_iter()
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
        MirOp::MILIdentity { x, .. } if x.0 == "__placeholder__" => vec![1, max_seq_len],
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
        // ─── Additional shape-propagating ops (CQ-22: previously fell through to empty vec) ───
        // These fallbacks are only hit when MIR node.shape is empty (i.e., when
        // infer_shape in mil_lower failed to compute a shape). When shape inference
        // works correctly, these branches are never reached.

        // Normalization ops: propagate input shape
        MirOp::MILBatchNorm { x, .. }
        | MirOp::MILInstanceNorm { x, .. }
        | MirOp::MILL2Norm { x, .. }
        | MirOp::MILLocalResponseNorm { x, .. } => {
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
        // Quantize/Dequantize: propagate input shape
        MirOp::MILQuantize { x, .. }
        | MirOp::MILDequantize { x, .. } => {
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
        // Unary elementwise ops: propagate input shape
        MirOp::MILSquare { x, .. }
        | MirOp::MILPrelu { x, .. }
        | MirOp::MILSoftsign { x, .. }
        | MirOp::MILElu { x, .. }
        | MirOp::MILReverse { x, .. }
        | MirOp::MILDepthToSpace { x, .. }
        | MirOp::MILSpaceToDepth { x, .. }
        | MirOp::MILPixelShuffle { x, .. }
        | MirOp::MILPixelUnshuffle { x, .. }
        | MirOp::MILCumsum { x, .. } => {
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
        // Parametric activations: propagate input shape
        MirOp::MILRelu6 { x, .. }
        | MirOp::MILSigmoidHard { x, .. }
        | MirOp::MILThresholdedRelu { x, .. }
        | MirOp::MILClampedRelu { x, .. }
        | MirOp::MILLinearActivation { x, .. }
        | MirOp::MILScaledTanh { x, .. }
        | MirOp::MILSoftplusParametric { x, .. }
        | MirOp::MILThreshold { x, .. }
        | MirOp::MILInverse { x, .. }
        | MirOp::MILExp2 { x, .. } => {
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
        // ConvTranspose: use output_shape field if available, else propagate
        MirOp::MILConvTranspose { x, output_shape, .. } => {
            if !output_shape.is_empty() {
                output_shape.clone()
            } else {
                node_shapes.get(&x.0).cloned().unwrap_or_default()
            }
        }
        // ReshapeLike: derive shape from ref_tensor
        MirOp::MILReshapeLike { ref_tensor, .. } => {
            node_shapes.get(&ref_tensor.0).cloned().unwrap_or_default()
        }
        // Flatten2d: reshape-like, compute from input
        MirOp::MILFlatten2d { x, axis, .. } => {
            if let Some(input_shape) = node_shapes.get(&x.0) {
                let product: usize = input_shape[*axis..].iter().product();
                let mut out = input_shape[..*axis].to_vec();
                out.push(product);
                out
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
    shape_ops::broadcast_shape(a, b)
}

/// Compute the output shape of a reduction operation.
///
/// Shared by `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`, and the
/// additional reduce variants added below.
///
/// Delegates to [`shape_ops::reduce_shape`].
fn reduce_shape(
    x: &ane_ir::mir::MirNodeId,
    axes: &[usize],
    keep_dims: bool,
    node_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    if let Some(input_shape) = node_shapes.get(&x.0) {
        shape_ops::reduce_shape(input_shape, axes, keep_dims)
    } else {
        vec![]
    }
}

/// Convenience wrapper for [`compat_output_shape`] using default Qwen3-0.6B
/// max sequence length (32768).
///
/// Prefer [`compat_output_shape`] with an explicit `max_seq_len` for correctness
/// when compiling models with different sequence lengths.
pub fn compat_output_shape_default(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    compat_output_shape(name, op, shape, node_shapes, 32768)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::mir::{MilDtype, MirNodeId, MirOp};
    use std::collections::HashMap;

    // ─── Helper constructors ───────────────────────────────────────────────

    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    fn shapes() -> HashMap<String, Vec<usize>> {
        HashMap::new()
    }

    fn shapes_with(entries: Vec<(&str, Vec<usize>)>) -> HashMap<String, Vec<usize>> {
        entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    // ─── compat_input_dtype ───────────────────────────────────────────────

    #[test]
    fn test_compat_input_dtype_input_ids_returns_int32() {
        let result = compat_input_dtype("input_ids", &MilDtype::Fp16);
        assert_eq!(result, MilDtypeCompat::Int32);
    }

    #[test]
    fn test_compat_input_dtype_input_ids_prefix_returns_int32() {
        let result = compat_input_dtype("model_input_ids_seq", &MilDtype::Fp16);
        assert_eq!(result, MilDtypeCompat::Int32);
    }

    #[test]
    fn test_compat_input_dtype_non_input_ids_returns_compat_dtype() {
        let result = compat_input_dtype("attention_mask", &MilDtype::Fp16);
        assert_eq!(result, MilDtypeCompat::Fp16);
    }

    #[test]
    fn test_compat_input_dtype_fp32_passthrough() {
        let result = compat_input_dtype("weights", &MilDtype::Fp32);
        assert_eq!(result, MilDtypeCompat::Fp32);
    }

    // ─── compat_input_shape ───────────────────────────────────────────────

    #[test]
    fn test_compat_input_shape_non_empty_returns_as_is() {
        assert_eq!(compat_input_shape("x", &[2, 3, 4], 512), vec![2, 3, 4]);
    }

    #[test]
    fn test_compat_input_shape_empty_input_ids_fallback() {
        assert_eq!(compat_input_shape("input_ids", &[], 512), vec![1, 512]);
    }

    #[test]
    fn test_compat_input_shape_empty_generic_fallback() {
        assert_eq!(compat_input_shape("attention_mask", &[], 512), vec![1]);
    }

    #[test]
    fn test_compat_input_shape_non_empty_overrides_input_ids_name() {
        // When shape is known, the name heuristic is irrelevant
        assert_eq!(compat_input_shape("input_ids", &[1, 128], 512), vec![1, 128]);
    }

    // ─── compat_output_shape: early-return paths ──────────────────────────

    #[test]
    fn test_compat_output_shape_non_empty_shape_returns_immediately() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        // Even though x has no shape in node_shapes, non-empty shape is returned
        assert_eq!(compat_output_shape("node", &op, &[3, 4], &ns, 512), vec![3, 4]);
    }

    #[test]
    fn test_compat_output_shape_input_ids_name_returns_default() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("input_ids_node", &op, &[], &ns), vec![1, 32768]);
    }

    // ─── compat_output_shape: unary shape-propagating ops ─────────────────

    #[test]
    fn test_unary_silu_propagates_shape() {
        let op = MirOp::MILSilu { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1024]);
    }

    #[test]
    fn test_unary_abs_propagates_shape() {
        let op = MirOp::MILAbs { name: "a".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![4, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 5]);
    }

    #[test]
    fn test_unary_relu_propagates_shape() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_unary_sigmoid_propagates_shape() {
        let op = MirOp::MILSigmoid { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![8]);
    }

    #[test]
    fn test_unary_tanh_propagates_shape() {
        let op = MirOp::MILTanh { name: "t".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_unary_gelu_propagates_shape() {
        let op = MirOp::MILGelu { name: "g".into(), x: nid("x"), mode: "exact".into() };
        let ns = shapes_with(vec![("x", vec![2, 4, 6])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 4, 6]);
    }

    #[test]
    fn test_unary_exp_propagates_shape() {
        let op = MirOp::MILExp { name: "e".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 3]);
    }

    #[test]
    fn test_unary_cos_propagates_shape() {
        let op = MirOp::MILCos { name: "c".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![7])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![7]);
    }

    #[test]
    fn test_unary_sin_propagates_shape() {
        let op = MirOp::MILSin { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![7])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![7]);
    }

    #[test]
    fn test_unary_cast_propagates_shape() {
        let op = MirOp::MILCast { name: "c".into(), x: nid("x"), dtype: MilDtype::Fp32 };
        let ns = shapes_with(vec![("x", vec![1, 128])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 128]);
    }

    #[test]
    fn test_unary_rsqrt_propagates_shape() {
        let op = MirOp::MILRsqrt { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![4, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 8]);
    }

    #[test]
    fn test_unary_neg_propagates_shape() {
        let op = MirOp::MILNeg { name: "n".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_unary_sqrt_propagates_shape() {
        let op = MirOp::MILSqrt { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![5, 6])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![5, 6]);
    }

    #[test]
    fn test_unary_logical_not_propagates_shape() {
        let op = MirOp::MILLogicalNot { name: "ln".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 10])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 10]);
    }

    #[test]
    fn test_unary_ceil_propagates_shape() {
        let op = MirOp::MILCeil { name: "c".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3]);
    }

    #[test]
    fn test_unary_floor_propagates_shape() {
        let op = MirOp::MILFloor { name: "f".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3]);
    }

    #[test]
    fn test_unary_round_propagates_shape() {
        let op = MirOp::MILRound { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3]);
    }

    #[test]
    fn test_unary_sign_propagates_shape() {
        let op = MirOp::MILSign { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3]);
    }

    #[test]
    fn test_unary_log_propagates_shape() {
        let op = MirOp::MILLog { name: "l".into(), x: nid("x"), epsilon: 1e-7 };
        let ns = shapes_with(vec![("x", vec![2, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 4]);
    }

    #[test]
    fn test_unary_leaky_relu_propagates_shape() {
        let op = MirOp::MILLeakyRelu { name: "lr".into(), x: nid("x"), alpha: 0.01 };
        let ns = shapes_with(vec![("x", vec![1, 32])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 32]);
    }

    #[test]
    fn test_unary_clip_propagates_shape() {
        let op = MirOp::MILClip { name: "c".into(), x: nid("x"), min_val: 0.0, max_val: 6.0 };
        let ns = shapes_with(vec![("x", vec![1, 32])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 32]);
    }

    #[test]
    fn test_unary_unknown_input_returns_empty() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("unknown") };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: binary ops with broadcast ───────────────────

    #[test]
    fn test_binary_add_same_shape() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3]), ("b", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_binary_mul_broadcast_different_rank() {
        let op = MirOp::MILMul { name: "m".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 512, 64]), ("b", vec![64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 64]);
    }

    #[test]
    fn test_binary_sub_broadcast_scalar() {
        let op = MirOp::MILSub { name: "s".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 512]), ("b", vec![1])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512]);
    }

    #[test]
    fn test_binary_maximum_broadcast() {
        let op = MirOp::MILMaximum { name: "mx".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![4, 1, 8]), ("b", vec![1, 6, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 6, 8]);
    }

    #[test]
    fn test_binary_minimum_same_shape() {
        let op = MirOp::MILMinimum { name: "mn".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![5, 5]), ("b", vec![5, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![5, 5]);
    }

    #[test]
    fn test_binary_realdiv_broadcast() {
        let op = MirOp::MILRealDiv { name: "d".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3, 4]), ("b", vec![4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3, 4]);
    }

    #[test]
    fn test_binary_pow_broadcast() {
        let op = MirOp::MILPow { name: "p".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4]), ("b", vec![1])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4]);
    }

    #[test]
    fn test_binary_floordiv_same_shape() {
        let op = MirOp::MILFloorDiv { name: "fd".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![8, 8]), ("b", vec![8, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![8, 8]);
    }

    #[test]
    fn test_binary_mod_broadcast() {
        let op = MirOp::MILMod { name: "md".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 10]), ("b", vec![10])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 10]);
    }

    #[test]
    fn test_binary_equal_broadcast() {
        let op = MirOp::MILEqual { name: "eq".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 64]), ("b", vec![64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_binary_not_equal_same_shape() {
        let op = MirOp::MILNotEqual { name: "ne".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 3]), ("b", vec![3, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 3]);
    }

    #[test]
    fn test_binary_greater_broadcast() {
        let op = MirOp::MILGreater { name: "gt".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![4, 1]), ("b", vec![1, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 5]);
    }

    #[test]
    fn test_binary_greater_equal_broadcast() {
        let op = MirOp::MILGreaterEqual { name: "ge".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3]), ("b", vec![3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_binary_less_same_shape() {
        let op = MirOp::MILLess { name: "lt".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 2]), ("b", vec![2, 2])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 2]);
    }

    #[test]
    fn test_binary_less_equal_broadcast() {
        let op = MirOp::MILLessEqual { name: "le".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 8]), ("b", vec![8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 8]);
    }

    #[test]
    fn test_binary_logical_and_broadcast() {
        let op = MirOp::MILLogicalAnd { name: "la".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 16]), ("b", vec![16])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 16]);
    }

    #[test]
    fn test_binary_logical_or_broadcast() {
        let op = MirOp::MILLogicalOr { name: "lo".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 16]), ("b", vec![16])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 16]);
    }

    #[test]
    fn test_binary_only_x_known_returns_x() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_binary_only_y_known_returns_y() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("b", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_binary_both_unknown_returns_empty() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    #[test]
    fn test_binary_incompatible_broadcast_returns_x_fallback() {
        // [3, 4] and [5, 6] are not broadcast-compatible → falls back to shape_a
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4]), ("b", vec![5, 6])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4]);
    }

    // ─── compat_output_shape: softmax ─────────────────────────────────────

    #[test]
    fn test_softmax_propagates_shape() {
        let op = MirOp::MILSoftmax { name: "sm".into(), x: nid("x"), axis: -1 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 64, 64]);
    }

    // ─── compat_output_shape: linear ──────────────────────────────────────

    #[test]
    fn test_linear_propagates_input_shape() {
        let op = MirOp::MILLinear {
            name: "lin".into(),
            x: nid("x"),
            weight: "w.bin".into(),
            bias: None,
        };
        let ns = shapes_with(vec![("x", vec![1, 512])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512]);
    }

    #[test]
    fn test_linear_unknown_input_returns_empty() {
        let op = MirOp::MILLinear {
            name: "lin".into(),
            x: nid("x"),
            weight: "w.bin".into(),
            bias: None,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: matmul ──────────────────────────────────────

    #[test]
    fn test_matmul_2d() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![4, 8]), ("b", vec![8, 16])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 16]);
    }

    #[test]
    fn test_matmul_batched() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        // [2, 3, 4] × [2, 4, 5] → [2, 3, 5]
        let ns = shapes_with(vec![("a", vec![2, 3, 4]), ("b", vec![2, 4, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3, 5]);
    }

    #[test]
    fn test_matmul_batched_broadcast() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        // [2, 1, 3, 4] × [1, 6, 4, 5] → [2, 6, 3, 5]
        let ns = shapes_with(vec![("a", vec![2, 1, 3, 4]), ("b", vec![1, 6, 4, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 6, 3, 5]);
    }

    #[test]
    fn test_matmul_only_x_known() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![4, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![4, 8]);
    }

    #[test]
    fn test_matmul_both_unknown() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    #[test]
    fn test_matmul_degenerate_1d_fallback() {
        // 1D × 2D falls back to x shape
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![8]), ("b", vec![8, 16])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![8]);
    }

    // ─── compat_output_shape: reshape ─────────────────────────────────────

    #[test]
    fn test_reshape_returns_target_shape() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![2, 3, 4] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3, 4]);
    }

    #[test]
    fn test_reshape_flatten() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![1, 24] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 24]);
    }

    // ─── compat_output_shape: transpose ───────────────────────────────────

    #[test]
    fn test_transpose_2d() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
        let ns = shapes_with(vec![("x", vec![3, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![5, 3]);
    }

    #[test]
    fn test_transpose_4d_nchw_to_nhwc() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![0, 2, 3, 1] };
        let ns = shapes_with(vec![("x", vec![1, 64, 8, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 8, 8, 64]);
    }

    #[test]
    fn test_transpose_unknown_input_returns_empty() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: tile ────────────────────────────────────────

    #[test]
    fn test_tile_multiplies_by_reps() {
        // CQ-22 fix: tile now correctly multiplies dims by reps instead of
        // just propagating the input shape.
        let op = MirOp::MILTile { name: "t".into(), x: nid("x"), reps: vec![2, 3] };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 192]);
    }

    #[test]
    fn test_tile_identity_reps() {
        // Tile with all-1 reps preserves input shape
        let op = MirOp::MILTile { name: "t".into(), x: nid("x"), reps: vec![1, 1] };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_tile_gqa_style() {
        // GQA tile: repeat along head dimension
        let op = MirOp::MILTile {
            name: "t".into(),
            x: nid("x"),
            reps: vec![1, 1, 2, 1, 1],
        };
        let ns = shapes_with(vec![("x", vec![1, 8, 1, 512, 128])]);
        assert_eq!(
            compat_output_shape_default("node", &op, &[], &ns),
            vec![1, 8, 2, 512, 128]
        );
    }

    #[test]
    fn test_tile_unknown_input_returns_empty() {
        let op = MirOp::MILTile { name: "t".into(), x: nid("unknown"), reps: vec![2, 3] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: fill / fill_like ────────────────────────────

    #[test]
    fn test_fill_returns_shape_param() {
        let op = MirOp::MILFill {
            name: "f".into(),
            shape: vec![2, 3, 4],
            value: 0.0,
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3, 4]);
    }

    #[test]
    fn test_fill_like_propagates_ref_shape() {
        let op = MirOp::MILFillLike {
            name: "fl".into(),
            ref_tensor: nid("x"),
            value: 1.0,
            dtype: MilDtype::Fp16,
        };
        let ns = shapes_with(vec![("x", vec![1, 128])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 128]);
    }

    #[test]
    fn test_fill_like_unknown_ref_returns_empty() {
        let op = MirOp::MILFillLike {
            name: "fl".into(),
            ref_tensor: nid("x"),
            value: 1.0,
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: gather (embedding) ──────────────────────────

    #[test]
    fn test_gather_embedding_2d() {
        // x=[151936, 1024], indices=[1, 512], axis=0 → [1, 512, 1024]
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![151936, 1024]), ("idx", vec![1, 512])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1024]);
    }

    #[test]
    fn test_gather_axis_last() {
        // x=[1, 512, 1024], indices=[1, 512, 10], axis=2 → [1, 512, 1, 512, 10]
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 2 };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024]), ("idx", vec![1, 512, 10])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1, 512, 10]);
    }

    #[test]
    fn test_gather_no_indices_shape_falls_back_to_input() {
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![100, 50])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![100, 50]);
    }

    #[test]
    fn test_gather_both_unknown_returns_empty() {
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: reduce_mean / reduce_max / reduce_min / reduce_prod ───

    #[test]
    fn test_reduce_mean_keep_dims() {
        let op =
            MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 1, 64]);
    }

    #[test]
    fn test_reduce_mean_no_keep_dims() {
        let op = MirOp::MILReduceMean {
            name: "rm".into(),
            x: nid("x"),
            axes: vec![2],
            keep_dims: false,
        };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 64]);
    }

    #[test]
    fn test_reduce_mean_multiple_axes_no_keep() {
        let op = MirOp::MILReduceMean {
            name: "rm".into(),
            x: nid("x"),
            axes: vec![1, 2],
            keep_dims: false,
        };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_reduce_max_keep_dims() {
        let op =
            MirOp::MILReduceMax { name: "rx".into(), x: nid("x"), axes: vec![1], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 1, 4]);
    }

    #[test]
    fn test_reduce_min_no_keep_dims() {
        let op =
            MirOp::MILReduceMin { name: "rn".into(), x: nid("x"), axes: vec![0], keep_dims: false };
        let ns = shapes_with(vec![("x", vec![5, 6, 7])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![6, 7]);
    }

    #[test]
    fn test_reduce_prod_keep_dims_all_axes() {
        let op = MirOp::MILReduceProd {
            name: "rp".into(),
            x: nid("x"),
            axes: vec![0, 1, 2],
            keep_dims: true,
        };
        let ns = shapes_with(vec![("x", vec![3, 4, 5])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 1, 1]);
    }

    #[test]
    fn test_reduce_unknown_input_returns_empty() {
        let op = MirOp::MILReduceMean {
            name: "rm".into(),
            x: nid("x"),
            axes: vec![1],
            keep_dims: false,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: expand_dims ─────────────────────────────────

    #[test]
    fn test_expand_dims_single_axis() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_multiple_axes() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![0, 2] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        // Insert 1 at pos 0 → [1, 3, 4], then at pos 2 → [1, 3, 1, 4]
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_axis_at_end() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![2] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4, 1]);
    }

    #[test]
    fn test_expand_dims_unknown_input_returns_empty() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![0] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: squeeze ─────────────────────────────────────

    #[test]
    fn test_squeeze_single_axis() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes_with(vec![("x", vec![3, 1, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4]);
    }

    #[test]
    fn test_squeeze_multiple_axes() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1, 3] };
        let ns = shapes_with(vec![("x", vec![1, 1, 4, 1])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 4]);
    }

    #[test]
    fn test_squeeze_axis_out_of_range_is_noop() {
        // Axis 5 is out of range for a 3D tensor; the removal is skipped
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![5] };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3, 4]);
    }

    #[test]
    fn test_squeeze_unknown_input_returns_empty() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: pad ─────────────────────────────────────────

    #[test]
    fn test_pad_symmetric_padding() {
        // Pad [1, 64, 8, 8] with [0,0,1,1,0,0,1,1] → [1, 64, 10, 10]
        let op = MirOp::MILPad {
            name: "p".into(),
            x: nid("x"),
            pad_amounts: vec![0, 0, 1, 1, 0, 0, 1, 1],
            mode: "constant".into(),
            constant_value: 0.0,
        };
        let ns = shapes_with(vec![("x", vec![1, 64, 8, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 10, 10]);
    }

    #[test]
    fn test_pad_uneven_padding() {
        // Pad [3, 4] with [2, 0, 0, 3] → [5, 7]
        let op = MirOp::MILPad {
            name: "p".into(),
            x: nid("x"),
            pad_amounts: vec![2, 0, 0, 3],
            mode: "constant".into(),
            constant_value: 0.0,
        };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![5, 7]);
    }

    #[test]
    fn test_pad_no_padding_returns_same_shape() {
        let op = MirOp::MILPad {
            name: "p".into(),
            x: nid("x"),
            pad_amounts: vec![0, 0, 0, 0],
            mode: "constant".into(),
            constant_value: 0.0,
        };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_pad_unknown_input_returns_empty() {
        let op = MirOp::MILPad {
            name: "p".into(),
            x: nid("x"),
            pad_amounts: vec![0, 0],
            mode: "constant".into(),
            constant_value: 0.0,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: concat ──────────────────────────────────────

    #[test]
    fn test_concat_along_axis() {
        let op = MirOp::MILConcat {
            name: "c".into(),
            values: vec![nid("a"), nid("b"), nid("c")],
            axis: 2,
        };
        let ns = shapes_with(vec![
            ("a", vec![1, 12, 64, 64]),
            ("b", vec![1, 12, 32, 64]),
            ("c", vec![1, 12, 16, 64]),
        ]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 112, 64]);
    }

    #[test]
    fn test_concat_single_input() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a")], axis: 1 };
        let ns = shapes_with(vec![("a", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_concat_unknown_first_input_returns_empty() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a"), nid("b")], axis: 1 };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    #[test]
    fn test_concat_axis_out_of_range_only_first_shape_used() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a"), nid("b")], axis: 5 };
        let ns = shapes_with(vec![("a", vec![1, 2, 3]), ("b", vec![1, 2, 3])]);
        // axis 5 is out of range for 3D tensor, so out[5] is never written
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 2, 3]);
    }

    // ─── compat_output_shape: where ───────────────────────────────────────

    #[test]
    fn test_where_all_same_shape() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![2, 3]), ("a", vec![2, 3]), ("b", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_where_broadcast_condition_scalar() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![1]), ("a", vec![1, 64]), ("b", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    #[test]
    fn test_where_only_x_known_returns_x() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4]);
    }

    #[test]
    fn test_where_only_condition_known_returns_condition() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 4]);
    }

    #[test]
    fn test_where_all_unknown_returns_empty() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: layer_norm ──────────────────────────────────

    #[test]
    fn test_layer_norm_propagates_shape() {
        let op = MirOp::MILLayerNorm {
            name: "ln".into(),
            x: nid("x"),
            weight: "w.bin".into(),
            bias: Some("b.bin".into()),
            epsilon: 1e-5,
            axes: vec![2],
        };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 64]);
    }

    // ─── compat_output_shape: topk ────────────────────────────────────────

    #[test]
    fn test_topk_positive_axis() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 10, axis: 2 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 10]);
    }

    #[test]
    fn test_topk_negative_axis() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 5, axis: -1 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        // axis=-1 → axis = 3 - 1 = 2
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 5]);
    }

    #[test]
    fn test_topk_unknown_input_returns_empty() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 10, axis: 1 };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: scaled_dot_product_attention ────────────────

    #[test]
    fn test_sdpa_propagates_query_shape() {
        let op = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: Some(0.125),
        };
        let ns = shapes_with(vec![("q", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 12, 64, 64]);
    }

    #[test]
    fn test_sdpa_unknown_query_returns_empty() {
        let op = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"),
            key: nid("k"),
            value: nid("v"),
            attention_mask: None,
            scale: None,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: read_state ──────────────────────────────────

    #[test]
    fn test_read_state_returns_explicit_shape() {
        let op = MirOp::MILReadState {
            name: "rs".into(),
            state_id: "kv_cache".into(),
            shape: vec![1, 2, 128, 64],
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 2, 128, 64]);
    }

    // ─── compat_output_shape: coreml_update_state / state_write ───────────

    #[test]
    fn test_coreml_update_state_propagates_value_shape() {
        let op = MirOp::MILCoremlUpdateState {
            name: "us".into(),
            state_id: "kv".into(),
            value: nid("v"),
        };
        let ns = shapes_with(vec![("v", vec![1, 2, 128, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 2, 128, 64]);
    }

    #[test]
    fn test_state_write_propagates_value_shape() {
        let op =
            MirOp::MILStateWrite { name: "sw".into(), state_ref: "kv".into(), value: nid("v") };
        let ns = shapes_with(vec![("v", vec![2, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 4]);
    }

    // ─── compat_output_shape: conv ────────────────────────────────────────

    #[test]
    fn test_conv_propagates_input_shape() {
        let op = MirOp::MILConv {
            name: "conv".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1, 1],
            pad_amounts: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
        };
        let ns = shapes_with(vec![("x", vec![1, 3, 32, 32])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 3, 32, 32]);
    }

    // ─── compat_output_shape: select ──────────────────────────────────────

    #[test]
    fn test_select_propagates_x_shape() {
        let op =
            MirOp::MILSelect { name: "sel".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 128])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 128]);
    }

    // ─── compat_output_shape: split ───────────────────────────────────────

    #[test]
    fn test_split_evenly() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 1, num_splits: 4 };
        let ns = shapes_with(vec![("x", vec![1, 64, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 16, 8]);
    }

    #[test]
    fn test_split_axis_0() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 0, num_splits: 2 };
        let ns = shapes_with(vec![("x", vec![4, 8])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 8]);
    }

    #[test]
    fn test_split_unknown_input_returns_empty() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 1, num_splits: 2 };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: slice_by_index ──────────────────────────────

    #[test]
    fn test_slice_by_index_simple() {
        let op = MirOp::MILSliceByIndex {
            name: "sli".into(),
            x: nid("x"),
            begin: vec![0, 0, 0],
            end: vec![1, 6, 32],
            stride: vec![1, 1, 1],
            begin_mask: vec![true, false, false],
            end_mask: vec![true, false, false],
            squeeze_mask: vec![],
        };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        // axis 0: begin_mask=true → b=0, end_mask=true → e=1 → 1-0=1
        // axis 1: begin=0, end=6 → 6-0=6
        // axis 2: begin=0, end=32 → 32-0=32
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 6, 32]);
    }

    #[test]
    fn test_slice_by_index_with_begin_end_masks() {
        let op = MirOp::MILSliceByIndex {
            name: "sli".into(),
            x: nid("x"),
            begin: vec![0, 2, 0],
            end: vec![0, 5, 0],
            stride: vec![1, 1, 1],
            begin_mask: vec![true, false, true],
            end_mask: vec![true, false, true],
            squeeze_mask: vec![],
        };
        let ns = shapes_with(vec![("x", vec![1, 10, 64])]);
        // axis 0: begin_mask → b=0, end_mask → e=1 → 1
        // axis 1: begin=2, end=5 → 3
        // axis 2: begin_mask → b=0, end_mask → e=64 → 64
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 3, 64]);
    }

    #[test]
    fn test_slice_by_index_with_squeeze_mask() {
        let op = MirOp::MILSliceByIndex {
            name: "sli".into(),
            x: nid("x"),
            begin: vec![0, 0, 0],
            end: vec![1, 10, 64],
            stride: vec![1, 1, 1],
            begin_mask: vec![false, false, false],
            end_mask: vec![false, false, false],
            squeeze_mask: vec![true, false, false],
        };
        let ns = shapes_with(vec![("x", vec![1, 10, 64])]);
        // Before squeeze: [1, 10, 64]; squeeze axis 0 → [10, 64]
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![10, 64]);
    }

    #[test]
    fn test_slice_by_index_negative_end() {
        let op = MirOp::MILSliceByIndex {
            name: "sli".into(),
            x: nid("x"),
            begin: vec![0, 0],
            end: vec![0, -1],
            stride: vec![1, 1],
            begin_mask: vec![true, false],
            end_mask: vec![true, false],
            squeeze_mask: vec![],
        };
        let ns = shapes_with(vec![("x", vec![1, 10])]);
        // axis 0: begin_mask → 1; axis 1: begin=0, end=-1 → 10+(-1)=9, 9-0=9
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 9]);
    }

    #[test]
    fn test_slice_by_index_unknown_input_returns_empty() {
        let op = MirOp::MILSliceByIndex {
            name: "sli".into(),
            x: nid("x"),
            begin: vec![0, 0],
            end: vec![1, 5],
            stride: vec![1, 1],
            begin_mask: vec![],
            end_mask: vec![],
            squeeze_mask: vec![],
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: identity ────────────────────────────────────

    #[test]
    fn test_identity_propagates_input_shape() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 4]);
    }

    #[test]
    fn test_identity_placeholder_returns_default() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("__placeholder__") };
        let ns = shapes();
        // T-36: placeholder now uses max_seq_len (32768) instead of hardcoded 512
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 32768]);
    }

    #[test]
    fn test_identity_unknown_input_returns_empty() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: stack ───────────────────────────────────────

    #[test]
    fn test_stack_axis_0() {
        let op = MirOp::MILStack {
            name: "st".into(),
            values: vec![nid("a"), nid("b"), nid("c")],
            axis: 0,
        };
        let ns = shapes_with(vec![("a", vec![3, 4]), ("b", vec![3, 4]), ("c", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![3, 3, 4]);
    }

    #[test]
    fn test_stack_axis_2() {
        let op = MirOp::MILStack { name: "st".into(), values: vec![nid("a"), nid("b")], axis: 2 };
        let ns = shapes_with(vec![("a", vec![1, 64]), ("b", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 2]);
    }

    #[test]
    fn test_stack_unknown_first_input_returns_empty() {
        let op = MirOp::MILStack { name: "st".into(), values: vec![nid("a"), nid("b")], axis: 0 };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: const ───────────────────────────────────────

    #[test]
    fn test_const_from_node_shapes_lookup() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "weights/w.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes_with(vec![("const_node", vec![64, 64])]);
        assert_eq!(compat_output_shape_default("const_node", &op, &[], &ns), vec![64, 64]);
    }

    #[test]
    fn test_const_scalar_pattern() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "scalar://fp16/0.5".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("const_node", &op, &[], &ns), vec![1]);
    }

    #[test]
    fn test_const_scalar_fp32_pattern() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "scalar://fp32/1.0".into(),
            dtype: MilDtype::Fp32,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("const_node", &op, &[], &ns), vec![1]);
    }

    #[test]
    fn test_const_node_shapes_takes_priority_over_scalar() {
        // When the node name exists in node_shapes, it takes priority even for scalar:// paths
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "scalar://fp16/0.5".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes_with(vec![("const_node", vec![4])]);
        assert_eq!(compat_output_shape_default("const_node", &op, &[], &ns), vec![4]);
    }

    #[test]
    fn test_const_unknown_returns_empty() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "weights/w.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape_default("const_node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── compat_output_shape: catch-all unknown op ────────────────────────

    #[test]
    fn test_unknown_op_returns_empty_shape() {
        // Use an op variant that doesn't have a specific case in compat_output_shape.
        // MILResizeBilinear is not handled (falls through to `_ => vec![]`)
        let op = MirOp::MILResizeBilinear {
            name: "rb".into(),
            x: nid("x"),
            target_height: 8,
            target_width: 8,
            align_corners: false,
        };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), Vec::<usize>::new());
    }

    // ─── broadcast_shape_compat ───────────────────────────────────────────

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(broadcast_shape_compat(&[3, 4], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_different_rank() {
        assert_eq!(broadcast_shape_compat(&[1, 512, 64], &[64]), Some(vec![1, 512, 64]));
    }

    #[test]
    fn test_broadcast_scalar_like() {
        assert_eq!(broadcast_shape_compat(&[1], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_mixed_ones() {
        assert_eq!(broadcast_shape_compat(&[4, 1, 8], &[1, 6, 8]), Some(vec![4, 6, 8]));
    }

    #[test]
    fn test_broadcast_incompatible_returns_none() {
        assert_eq!(broadcast_shape_compat(&[3, 4], &[5, 6]), None);
    }

    #[test]
    fn test_broadcast_both_scalars() {
        assert_eq!(broadcast_shape_compat(&[1], &[1]), Some(vec![1]));
    }

    #[test]
    fn test_broadcast_empty_left() {
        // Empty shape treated as scalar
        assert_eq!(broadcast_shape_compat(&[], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_empty_right() {
        assert_eq!(broadcast_shape_compat(&[3, 4], &[]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_both_empty() {
        assert_eq!(broadcast_shape_compat(&[], &[]), Some(vec![]));
    }

    #[test]
    fn test_broadcast_3d_with_1d() {
        assert_eq!(broadcast_shape_compat(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
    }

    // ─── reduce_shape ─────────────────────────────────────────────────────

    #[test]
    fn test_reduce_shape_keep_dims_single_axis() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3, 4, 5])]);
        assert_eq!(reduce_shape(&x, &[2], true, &ns), vec![2, 3, 1, 5]);
    }

    #[test]
    fn test_reduce_shape_no_keep_dims_single_axis() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3, 4, 5])]);
        assert_eq!(reduce_shape(&x, &[2], false, &ns), vec![2, 3, 5]);
    }

    #[test]
    fn test_reduce_shape_no_keep_dims_multiple_axes() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3, 4, 5])]);
        // Axes [1, 3] — removed in reverse order (3 then 1) to preserve indices
        assert_eq!(reduce_shape(&x, &[1, 3], false, &ns), vec![2, 4]);
    }

    #[test]
    fn test_reduce_shape_keep_dims_multiple_axes() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3, 4, 5])]);
        assert_eq!(reduce_shape(&x, &[0, 2], true, &ns), vec![1, 3, 1, 5]);
    }

    #[test]
    fn test_reduce_shape_all_axes_no_keep() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(reduce_shape(&x, &[0, 1], false, &ns), Vec::<usize>::new());
    }

    #[test]
    fn test_reduce_shape_all_axes_keep() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(reduce_shape(&x, &[0, 1], true, &ns), vec![1, 1]);
    }

    #[test]
    fn test_reduce_shape_unknown_input_returns_empty() {
        let x = nid("x");
        let ns = shapes();
        assert_eq!(reduce_shape(&x, &[0], false, &ns), Vec::<usize>::new());
    }

    #[test]
    fn test_reduce_shape_axis_out_of_range_no_keep_is_noop() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        // axis 5 is out of range; removal is skipped
        assert_eq!(reduce_shape(&x, &[5], false, &ns), vec![2, 3]);
    }

    #[test]
    fn test_reduce_shape_axis_out_of_range_keep_is_noop() {
        let x = nid("x");
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(reduce_shape(&x, &[5], true, &ns), vec![2, 3]);
    }

    // ─── CQ-22: Additional reduce ops ─────────────────────────────────

    #[test]
    fn test_reduce_sum_no_keep_dims() {
        let op = MirOp::MILReduceSum { name: "rs".into(), x: nid("x"), axes: vec![1], keep_dims: false };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 1024]);
    }

    #[test]
    fn test_reduce_sum_square_keep_dims() {
        let op = MirOp::MILReduceSumSquare { name: "rss".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1]);
    }

    #[test]
    fn test_reduce_l2_norm_no_keep() {
        let op = MirOp::MILReduceL2Norm { name: "rl2".into(), x: nid("x"), axes: vec![1, 2], keep_dims: false };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1]);
    }

    #[test]
    fn test_reduce_argmax_keep_dims() {
        let op = MirOp::MILReduceArgmax { name: "ram".into(), x: nid("x"), axis: 2, keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1]);
    }

    #[test]
    fn test_reduce_argmin_no_keep_dims() {
        let op = MirOp::MILReduceArgmin { name: "rin".into(), x: nid("x"), axis: 1, keep_dims: false };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 1024]);
    }

    // ─── CQ-22: Normalization ops ─────────────────────────────────────

    #[test]
    fn test_batch_norm_propagates_shape() {
        let op = MirOp::MILBatchNorm {
            name: "bn".into(),
            x: nid("x"),
            mean: "m.bin".into(),
            variance: "v.bin".into(),
            gamma: Some("g.bin".into()),
            beta: Some("b.bin".into()),
            epsilon: 1e-5,
        };
        let ns = shapes_with(vec![("x", vec![1, 512, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 64]);
    }

    #[test]
    fn test_instance_norm_propagates_shape() {
        let op = MirOp::MILInstanceNorm {
            name: "in".into(),
            x: nid("x"),
            gamma: Some("g.bin".into()),
            beta: Some("b.bin".into()),
            epsilon: 1e-5,
        };
        let ns = shapes_with(vec![("x", vec![1, 64, 32])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 32]);
    }

    #[test]
    fn test_l2_norm_propagates_shape() {
        let op = MirOp::MILL2Norm { name: "l2".into(), x: nid("x"), epsilon: 1e-12, axes: vec![1] };
        let ns = shapes_with(vec![("x", vec![1, 512])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512]);
    }

    // ─── CQ-22: Quantize/Dequantize ───────────────────────────────────

    #[test]
    fn test_quantize_propagates_shape() {
        let op = MirOp::MILQuantize {
            name: "q".into(),
            x: nid("x"),
            scale: 0.1,
            zero_point: 128,
            axis: 0,
            output_dtype: MilDtype::UInt8,
        };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1024]);
    }

    #[test]
    fn test_dequantize_propagates_shape() {
        let op = MirOp::MILDequantize {
            name: "dq".into(),
            x: nid("x"),
            scale: 0.1,
            zero_point: 128,
            axis: 0,
            output_dtype: MilDtype::UInt8,
        };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 512, 1024]);
    }

    // ─── CQ-22: Additional unary ops ──────────────────────────────────

    #[test]
    fn test_square_propagates_shape() {
        let op = MirOp::MILSquare { name: "sq".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![2, 3]);
    }

    #[test]
    fn test_prelu_propagates_shape() {
        let op = MirOp::MILPrelu { name: "pr".into(), x: nid("x"), alpha: "a.bin".into() };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64]);
    }

    // ─── CQ-22: ConvTranspose ─────────────────────────────────────────

    #[test]
    fn test_conv_transpose_with_output_shape() {
        let op = MirOp::MILConvTranspose {
            name: "ct".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
            output_shape: vec![1, 64, 256],
        };
        let ns = shapes_with(vec![("x", vec![1, 64, 128])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 256]);
    }

    #[test]
    fn test_conv_transpose_without_output_shape() {
        let op = MirOp::MILConvTranspose {
            name: "ct".into(),
            x: nid("x"),
            weight: nid("w"),
            pad_type: "valid".into(),
            groups: 1,
            strides: vec![1],
            pad_amounts: vec![0],
            dilations: vec![1],
            output_shape: vec![],
        };
        let ns = shapes_with(vec![("x", vec![1, 64, 128])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 128]);
    }

    // ─── CQ-22: ReshapeLike ───────────────────────────────────────────

    #[test]
    fn test_reshape_like_uses_ref_tensor_shape() {
        let op = MirOp::MILReshapeLike { name: "rl".into(), x: nid("x"), ref_tensor: nid("r") };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024]), ("r", vec![1, 64, 8192])]);
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 64, 8192]);
    }

    // ─── CQ-22: Flatten2d ─────────────────────────────────────────────

    #[test]
    fn test_flatten2d_axis_1() {
        let op = MirOp::MILFlatten2d { name: "f2".into(), x: nid("x"), axis: 1 };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        // Flatten dims from axis 1 onwards: product = 512 * 1024 = 524288
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![1, 524288]);
    }

    #[test]
    fn test_flatten2d_axis_0() {
        let op = MirOp::MILFlatten2d { name: "f2".into(), x: nid("x"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        // Flatten all dims: product = 2 * 3 * 4 = 24
        assert_eq!(compat_output_shape_default("node", &op, &[], &ns), vec![24]);
    }
}
