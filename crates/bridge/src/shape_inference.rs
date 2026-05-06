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

/// T-P5-04: Shape inference error type for explicit error reporting.
///
/// When shape inference fails (e.g., unknown op, missing input shape),
/// this error provides context about why inference failed rather than
/// silently returning an empty `Vec` (which means "unknown shape" but
/// can hide bugs in the inference logic).
#[derive(Debug, Clone)]
pub enum ShapeInferenceError {
    /// The input shape for a referenced node is not available.
    MissingInputShape {
        node_name: String,
        input_id: String,
    },
    /// The op variant has no shape inference rule.
    UnknownOp {
        node_name: String,
        op_name: String,
    },
    /// Name-based heuristic was used (T-P5-09: should be replaced with explicit fields).
    NameHeuristicUsed {
        node_name: String,
        heuristic: String,
        inferred_shape: Vec<usize>,
    },
    /// Shape could not be determined for any reason.
    Indeterminate {
        node_name: String,
        reason: String,
    },
}

impl std::fmt::Display for ShapeInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeInferenceError::MissingInputShape { node_name, input_id } => {
                write!(f, "Shape inference failed for '{}': input shape for '{}' is not available", node_name, input_id)
            }
            ShapeInferenceError::UnknownOp { node_name, op_name } => {
                write!(f, "Shape inference failed for '{}': no inference rule for op '{}'", node_name, op_name)
            }
            ShapeInferenceError::NameHeuristicUsed { node_name, heuristic, inferred_shape } => {
                write!(f, "Shape inference for '{}' used name heuristic '{}' (shape: {:?})", node_name, heuristic, inferred_shape)
            }
            ShapeInferenceError::Indeterminate { node_name, reason } => {
                write!(f, "Shape inference failed for '{}': {}", node_name, reason)
            }
        }
    }
}

impl std::error::Error for ShapeInferenceError {}

/// Infer the output dtype for a compat graph input node.
///
/// T-81 (I-56): Previously, this function used `name.contains("input_ids")`
/// to force `Int32` dtype, which is a string-heuristic that can misfire on
/// tensors with "input_ids" in their name that are not actually Int32.
/// Now, the function trusts the MIR node's declared `dtype` field directly,
/// since the MIR builder correctly assigns `MilDtype::Int32` to input_ids
/// tensors during graph construction. The `name` parameter is retained for
/// API compatibility but is no longer used for dtype override decisions.
pub fn compat_input_dtype(_name: &str, dtype: &MilDtype) -> MilDtypeCompat {
    crate::mir_to_compat::mil_dtype_to_compat(dtype)
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
///
/// **Deprecated (M-018):** Prefer [`compat_input_shape_explicit`] which accepts
/// an explicit shape hint instead of relying on name-based heuristics.
pub fn compat_input_shape(name: &str, shape: &[usize], max_seq_len: usize) -> Vec<usize> {
    compat_input_shape_explicit(name, shape, max_seq_len, None)
}

/// Infer the shape for a compat graph input node, with an explicit shape hint.
///
/// This is the preferred API over [`compat_input_shape`]. When `explicit_shape`
/// is `Some` and non-empty, it is used directly, bypassing any name-based
/// heuristics. When `explicit_shape` is `None` or empty, the function falls
/// back to the legacy name-matching behavior but logs a deprecation warning.
///
/// M-018 fix: Callers that have access to shape annotations (e.g., from
/// `AirOp::Identity::shape_hint`) should pass them here to avoid relying
/// on fragile name-based heuristics like `name.contains("input_ids")`.
pub fn compat_input_shape_explicit(
    name: &str,
    shape: &[usize],
    max_seq_len: usize,
    explicit_shape: Option<&[usize]>,
) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    // M-018: Use explicit shape hint when available.
    if let Some(hint) = explicit_shape {
        if !hint.is_empty() {
            return hint.to_vec();
        }
    }
    // Legacy name-based heuristic fallback — deprecated.
    // M-018: `name.contains("input_ids")` assumes Qwen3-style token ID
    // inputs with shape [1, max_seq_len]. This is fragile and will produce
    // incorrect shapes for non-Qwen3 models or tensors that happen to
    // contain "input_ids" in their name but have different shapes.
    // Use compat_input_shape_explicit with an explicit_shape to avoid this.
    if name.contains("input_ids") {
        log::warn!(
            "M-018 DEPRECATED: compat_input_shape: using name-based heuristic \
             name.contains(\"input_ids\") to infer shape [1, {}]. This is fragile \
             and deprecated. Pass explicit_shape via compat_input_shape_explicit \
             instead. Node name: {:?}",
             max_seq_len, name
        );
        vec![1, max_seq_len]
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
///
/// T-36 (I-15/CQ-19): The `max_seq_len` parameter replaces the hardcoded
/// `512` fallback for input_ids and placeholder nodes. Callers should pass
/// the model's actual max sequence length from `ModelArchConfig::max_seq_len`.
///
/// **Deprecated (M-018):** Prefer [`compat_output_shape_explicit`] which accepts
/// an explicit shape hint instead of relying on name-based heuristics.
pub fn compat_output_shape(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
    max_seq_len: usize,
) -> Vec<usize> {
    compat_output_shape_explicit(name, op, shape, node_shapes, max_seq_len, None)
}

/// Infer the output shape of a MIR operation, with an explicit shape hint.
///
/// This is the preferred API over [`compat_output_shape`]. When `explicit_shape`
/// is `Some` and non-empty, it is used directly, bypassing any name-based
/// heuristics. When `explicit_shape` is `None` or empty, the function falls
/// back to the legacy name-matching behavior but logs a deprecation warning.
///
/// M-018 fix: Callers that have access to shape annotations (e.g., from
/// `AirOp::Identity::shape_hint`) should pass them here to avoid relying
/// on fragile name-based heuristics like `name.contains("input_ids")`.
pub fn compat_output_shape_explicit(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
    max_seq_len: usize,
    explicit_shape: Option<&[usize]>,
) -> Vec<usize> {
    if !shape.is_empty() {
        return shape.to_vec();
    }
    // M-018: Use explicit shape hint when available.
    if let Some(hint) = explicit_shape {
        if !hint.is_empty() {
            return hint.to_vec();
        }
    }
    // Legacy name-based heuristic fallback — deprecated.
    // M-018: `name.contains("input_ids")` assumes Qwen3-style token ID
    // inputs with shape [1, max_seq_len]. This is fragile and will produce
    // incorrect shapes for non-Qwen3 models or tensors that happen to
    // contain "input_ids" in their name but have different shapes.
    // Use compat_output_shape_explicit with an explicit_shape to avoid this.
    if name.contains("input_ids") {
        log::warn!(
            "M-018 DEPRECATED: compat_output_shape: using name-based heuristic \
             name.contains(\"input_ids\") to infer shape [1, {}]. This is fragile \
             and deprecated. Pass explicit_shape via compat_output_shape_explicit \
             instead. Node name: {:?}",
             max_seq_len, name
        );
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
        // Linear: propagate input shape (M-014: output dim unknown without weight metadata)
        // The MILLinear weight field is a String (not a MirNodeId), so we cannot
        // look up the weight shape at this layer to compute [B, output_features].
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
        // M-014: Conv — compute output shape from input + weight + conv params.
        // Falls back to input shape propagation if weight shape is unavailable.
        MirOp::MILConv { x, weight, strides, pad_amounts, dilations, .. } => {
            conv_output_shape(x, weight, strides, pad_amounts, dilations, node_shapes)
        }
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
        // M-041/T-P5-09: Magic-string shape heuristic — `"__placeholder__"` is a
        // Qwen3-specific sentinel for graph input placeholders, hardcoded to
        // [1, max_seq_len]. This is implicit semantics via node naming and will
        // produce incorrect shapes for non-Qwen3 models or any placeholder with
        // a different shape. This should be replaced with explicit shape annotations
        // carried through SIR→AIR→MIR.
        MirOp::MILIdentity { x, .. } if x.0 == "__placeholder__" => {
            log::warn!(
                "T-P5-09/M-041: compat_output_shape: using magic string \"__placeholder__\" \
                 heuristic to infer shape [1, {}]. This is fragile and should be replaced \
                 with explicit shape annotations. Node name: {:?}",
                 max_seq_len, name
            );
            vec![1, max_seq_len]
        }
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
        // T-P5-04: Log a warning when the catch-all is hit so that missing
        // inference rules are discoverable.
        _ => {
            log::warn!(
                "T-P5-04: compat_output_shape: no inference rule for op '{}' on node '{}'. \
                 Returning empty shape (unknown). Consider adding an explicit shape inference \
                 rule or using compat_output_shape_fallible for error handling.",
                op.mil_op_name(), name
            );
            vec![]
        }
    }
}

/// T-P5-04: Fallible version of [`compat_output_shape`] that returns
/// explicit errors instead of silently returning empty shapes.
///
/// This function has the same logic as `compat_output_shape` but returns
/// `Result<Vec<usize>, ShapeInferenceError>` instead of `Vec<usize>`.
/// When shape inference fails, it returns an error that describes *why*
/// inference failed, rather than silently returning `vec![]` (which
/// ambiguously means "unknown shape").
///
/// Use this function when you need to distinguish between:
/// - A known empty shape (e.g., scalar)
/// - A genuinely unknown shape (inference failed)
/// - A shape derived from a name heuristic (fragile)
///
/// For backward compatibility, [`compat_output_shape`] continues to
/// return `vec![]` for unknown shapes.
///
/// **Deprecated (M-018):** Prefer the `_explicit` variants which accept
/// explicit shape hints instead of relying on name-based heuristics.
pub fn compat_output_shape_fallible(
    name: &str,
    op: &MirOp,
    shape: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
    max_seq_len: usize,
) -> Result<Vec<usize>, ShapeInferenceError> {
    if !shape.is_empty() {
        return Ok(shape.to_vec());
    }
    // M-018: Name heuristic — reports as a NameHeuristicUsed error
    // rather than silently returning the heuristic result. This is
    // deprecated; use compat_output_shape_explicit instead.
    if name.contains("input_ids") {
        let inferred = vec![1, max_seq_len];
        return Err(ShapeInferenceError::NameHeuristicUsed {
            node_name: name.to_string(),
            heuristic: "name.contains(\"input_ids\") → [1, max_seq_len] (deprecated, use explicit_shape)".to_string(),
            inferred_shape: inferred,
        });
    }
    match op {
        MirOp::MILReduceMean { x, axes, keep_dims, .. } => {
            reduce_shape_fallible(x, axes, *keep_dims, node_shapes, name)
        }
        MirOp::MILRsqrt { x, .. } => {
            node_shapes.get(&x.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        // M-014: Linear propagates input shape because the output dimension
        // depends on weight metadata (weight is a String name, not a MirNodeId),
        // so we cannot look up the weight shape at this layer. The fallible
        // version returns Indeterminate when input shape is missing.
        MirOp::MILLinear { x, .. } => {
            node_shapes.get(&x.0).cloned().ok_or_else(|| {
                ShapeInferenceError::Indeterminate {
                    node_name: name.to_string(),
                    reason: "MILLinear: output dim unknown without weight metadata \
                             (M-014: weight is a String name, not a MirNodeId)"
                        .to_string(),
                }
            })
        }
        // M-014: Conv — compute output shape from input + weight + conv params.
        // Weight is a MirNodeId, so we can look up its shape from node_shapes.
        MirOp::MILConv { x, weight, strides, pad_amounts, dilations, .. } => {
            conv_output_shape_fallible(x, weight, strides, pad_amounts, dilations, node_shapes, name)
        }
        MirOp::MILGather { x, indices, axis, .. } => {
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
                    Ok(out)
                }
                (Some(_), None) | (None, Some(_)) => {
                    Err(ShapeInferenceError::MissingInputShape {
                        node_name: name.to_string(),
                        input_id: if node_shapes.get(&x.0).is_none() { x.0.clone() } else { indices.0.clone() },
                    })
                }
                _ => Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }),
            }
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
        | MirOp::MILCast { x, .. }
        | MirOp::MILSoftmax { x, .. }
        | MirOp::MILNeg { x, .. }
        | MirOp::MILSqrt { x, .. }
        | MirOp::MILLogicalNot { x, .. }
        | MirOp::MILCeil { x, .. }
        | MirOp::MILFloor { x, .. }
        | MirOp::MILRound { x, .. }
        | MirOp::MILSign { x, .. }
        | MirOp::MILLog { x, .. }
        | MirOp::MILLeakyRelu { x, .. }
        | MirOp::MILClip { x, .. }
        | MirOp::MILLayerNorm { x, .. }
        | MirOp::MILFillLike { ref_tensor: x, .. }
        | MirOp::MILSelect { x, .. }
        | MirOp::MILBatchNorm { x, .. }
        | MirOp::MILInstanceNorm { x, .. }
        | MirOp::MILL2Norm { x, .. }
        | MirOp::MILLocalResponseNorm { x, .. }
        | MirOp::MILQuantize { x, .. }
        | MirOp::MILDequantize { x, .. }
        | MirOp::MILSquare { x, .. }
        | MirOp::MILPrelu { x, .. }
        | MirOp::MILSoftsign { x, .. }
        | MirOp::MILElu { x, .. }
        | MirOp::MILReverse { x, .. }
        | MirOp::MILDepthToSpace { x, .. }
        | MirOp::MILSpaceToDepth { x, .. }
        | MirOp::MILPixelShuffle { x, .. }
        | MirOp::MILPixelUnshuffle { x, .. }
        | MirOp::MILCumsum { x, .. }
        | MirOp::MILRelu6 { x, .. }
        | MirOp::MILSigmoidHard { x, .. }
        | MirOp::MILThresholdedRelu { x, .. }
        | MirOp::MILClampedRelu { x, .. }
        | MirOp::MILLinearActivation { x, .. }
        | MirOp::MILScaledTanh { x, .. }
        | MirOp::MILSoftplusParametric { x, .. }
        | MirOp::MILThreshold { x, .. }
        | MirOp::MILInverse { x, .. }
        | MirOp::MILExp2 { x, .. }
        | MirOp::MILReshapeLike { ref_tensor: x, .. } => {
            node_shapes.get(&x.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        // Binary ops: compute broadcast output shape
        MirOp::MILAdd { x, y, .. }
        | MirOp::MILMul { x, y, .. }
        | MirOp::MILSub { x, y, .. }
        | MirOp::MILMaximum { x, y, .. }
        | MirOp::MILMinimum { x, y, .. }
        | MirOp::MILRealDiv { x, y, .. }
        | MirOp::MILPow { x, y, .. }
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
            let shape_a = node_shapes.get(&x.0).cloned();
            let shape_b = node_shapes.get(&y.0).cloned();
            match (shape_a, shape_b) {
                (Some(a), Some(b)) => {
                    Ok(broadcast_shape_compat(&a, &b).unwrap_or_else(|| a.clone()))
                }
                (Some(a), None) => Ok(a),
                (None, Some(b)) => Ok(b),
                _ => Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }),
            }
        }
        MirOp::MILMatMul { x, y, .. } => {
            match (node_shapes.get(&x.0), node_shapes.get(&y.0)) {
                (Some(x_shape), Some(y_shape)) => {
                    let x_rank = x_shape.len();
                    let y_rank = y_shape.len();
                    if x_rank >= 2 && y_rank >= 2 {
                        let lhs_rows = x_shape[x_rank - 2];
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
                        Ok(out)
                    } else {
                        Ok(x_shape.clone())
                    }
                }
                (Some(x_shape), None) => Ok(x_shape.clone()),
                _ => Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }),
            }
        }
        MirOp::MILReshape { shape, .. } => Ok(shape.to_vec()),
        MirOp::MILFill { shape, .. } => Ok(shape.to_vec()),
        MirOp::MILReadState { shape, .. } => Ok(shape.clone()),
        MirOp::MILTranspose { x, perm, .. } => {
            node_shapes.get(&x.0).map(|s| shape_ops::transpose_shape(s, perm)).ok_or_else(|| {
                ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }
            })
        }
        MirOp::MILTile { x, reps, .. } => {
            node_shapes.get(&x.0).map(|s| shape_ops::tile_shape(s, reps)).ok_or_else(|| {
                ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }
            })
        }
        MirOp::MILExpandDims { x, axis, .. } => {
            node_shapes.get(&x.0).map(|s| shape_ops::expand_dims_shape(s, axis)).ok_or_else(|| {
                ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }
            })
        }
        MirOp::MILSqueeze { x, axis, .. } => {
            node_shapes.get(&x.0).map(|s| shape_ops::squeeze_shape(s, axis)).ok_or_else(|| {
                ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }
            })
        }
        MirOp::MILPad { x, pad_amounts, .. } => {
            node_shapes.get(&x.0).map(|s| {
                let pad: Vec<usize> = pad_amounts.iter().map(|&p| p as usize).collect();
                shape_ops::pad_shape(s, &pad)
            }).ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        // Reduce ops
        MirOp::MILReduceMax { x, axes, keep_dims, .. }
        | MirOp::MILReduceMin { x, axes, keep_dims, .. }
        | MirOp::MILReduceProd { x, axes, keep_dims, .. }
        | MirOp::MILReduceSum { x, axes, keep_dims, .. }
        | MirOp::MILReduceSumSquare { x, axes, keep_dims, .. }
        | MirOp::MILReduceL2Norm { x, axes, keep_dims, .. }
        | MirOp::MILReduceL1Norm { x, axes, keep_dims, .. }
        | MirOp::MILReduceLogSumExp { x, axes, keep_dims, .. }
        | MirOp::MILReduceLogSum { x, axes, keep_dims, .. } => {
            reduce_shape_fallible(x, axes, *keep_dims, node_shapes, name)
        }
        MirOp::MILReduceArgmax { x, axis, keep_dims, .. }
        | MirOp::MILReduceArgmin { x, axis, keep_dims, .. } => {
            reduce_shape_fallible(x, &[*axis], *keep_dims, node_shapes, name)
        }
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
                Ok(out)
            } else {
                Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: values.first().map(|id| id.0.clone()).unwrap_or_default(),
                })
            }
        }
        MirOp::MILWhere { condition, x, y, .. } => {
            let shape_c = node_shapes.get(&condition.0).cloned();
            let shape_a = node_shapes.get(&x.0).cloned();
            let shape_b = node_shapes.get(&y.0).cloned();
            match (shape_a, shape_b, shape_c) {
                (Some(a), Some(b), Some(c)) => {
                    Ok(broadcast_shape_compat(&a, &b)
                        .and_then(|ab| broadcast_shape_compat(&ab, &c))
                        .unwrap_or_else(|| a.clone()))
                }
                (Some(a), _, _) => Ok(a),
                (_, _, Some(c)) => Ok(c),
                (_, Some(b), _) => Ok(b),
                _ => Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                }),
            }
        }
        MirOp::MILTopk { x, k, axis, .. } => {
            node_shapes.get(&x.0).map(|input_shape| {
                let mut out = input_shape.clone();
                let rank = out.len() as isize;
                let ax = if *axis >= 0 { *axis as usize } else { (rank + axis) as usize };
                if ax < out.len() {
                    out[ax] = *k;
                }
                out
            }).ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        MirOp::MILScaledDotProductAttention { query, .. } => {
            node_shapes.get(&query.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: query.0.clone(),
            })
        }
        MirOp::MILCoremlUpdateState { value, .. }
        | MirOp::MILStateWrite { value, .. } => {
            node_shapes.get(&value.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: value.0.clone(),
            })
        }
        MirOp::MILSplit { x, axis, num_splits, .. } => {
            node_shapes.get(&x.0).map(|input_shape| {
                let mut out = input_shape.clone();
                if let Some(dim) = out.get_mut(*axis) {
                    *dim /= num_splits;
                }
                out
            }).ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        MirOp::MILSliceByIndex { x, begin, end, begin_mask, end_mask, squeeze_mask, .. } => {
            node_shapes.get(&x.0).map(|input_shape| {
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
            }).ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        MirOp::MILIdentity { x, .. } if x.0 == "__placeholder__" => {
            let inferred = vec![1, max_seq_len];
            Err(ShapeInferenceError::NameHeuristicUsed {
                node_name: name.to_string(),
                heuristic: "\"__placeholder__\" → [1, max_seq_len]".to_string(),
                inferred_shape: inferred,
            })
        }
        MirOp::MILIdentity { x, .. } => {
            node_shapes.get(&x.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        MirOp::MILStack { values, axis, .. } => {
            if let Some(first_shape) = values.first().and_then(|id| node_shapes.get(&id.0)) {
                let mut out = first_shape.clone();
                let ax = if *axis <= out.len() { *axis } else { out.len() };
                out.insert(ax, values.len());
                Ok(out)
            } else {
                Err(ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: values.first().map(|id| id.0.clone()).unwrap_or_default(),
                })
            }
        }
        MirOp::MILConst { value_path, .. } => {
            if let Some(shape) = node_shapes.get(name) {
                Ok(shape.clone())
            } else if value_path.starts_with("scalar://") {
                Ok(vec![1])
            } else {
                Err(ShapeInferenceError::Indeterminate {
                    node_name: name.to_string(),
                    reason: format!("MILConst: shape for value_path '{}' not available in node_shapes", value_path),
                })
            }
        }
        MirOp::MILConvTranspose { x, output_shape, .. } => {
            if !output_shape.is_empty() {
                Ok(output_shape.clone())
            } else {
                node_shapes.get(&x.0).cloned().ok_or_else(|| ShapeInferenceError::MissingInputShape {
                    node_name: name.to_string(),
                    input_id: x.0.clone(),
                })
            }
        }
        MirOp::MILFlatten2d { x, axis, .. } => {
            node_shapes.get(&x.0).map(|input_shape| {
                let product: usize = input_shape[*axis..].iter().product();
                let mut out = input_shape[..*axis].to_vec();
                out.push(product);
                out
            }).ok_or_else(|| ShapeInferenceError::MissingInputShape {
                node_name: name.to_string(),
                input_id: x.0.clone(),
            })
        }
        // Catch-all: explicit error for unsupported ops
        _ => Err(ShapeInferenceError::UnknownOp {
            node_name: name.to_string(),
            op_name: op.mil_op_name().to_string(),
        }),
    }
}

/// M-014: Non-fallible version of conv shape inference for `compat_output_shape`.
/// Falls back to input shape propagation when weight shape is unavailable.
fn conv_output_shape(
    x: &ane_ir::mir::MirNodeId,
    weight: &ane_ir::mir::MirNodeId,
    strides: &[usize],
    pad_amounts: &[usize],
    dilations: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    match conv_output_shape_fallible(x, weight, strides, pad_amounts, dilations, node_shapes, "") {
        Ok(shape) => shape,
        Err(_) => {
            // Fallback: propagate input shape (old behavior)
            node_shapes.get(&x.0).cloned().unwrap_or_default()
        }
    }
}

/// M-014: Compute MILConv output shape from input shape, weight shape, and conv parameters.
///
/// For a 2D convolution with:
/// - input shape: [B, C_in, H, W]
/// - weight shape: [C_out, C_in/groups, kH, kW]
///
/// Output shape: [B, C_out, out_H, out_W] where:
///   out_H = (H + pad_h_total - dilation_h * (kH - 1) - 1) / stride_h + 1
///   out_W = (W + pad_w_total - dilation_w * (kW - 1) - 1) / stride_w + 1
///
/// `pad_amounts` layout for 2D conv: [pad_b, pad_a, pad_r, pad_l] (Core ML convention).
/// For 1D conv: [pad_e, pad_s].
fn conv_output_shape_fallible(
    x: &ane_ir::mir::MirNodeId,
    weight: &ane_ir::mir::MirNodeId,
    strides: &[usize],
    pad_amounts: &[usize],
    dilations: &[usize],
    node_shapes: &HashMap<String, Vec<usize>>,
    node_name: &str,
) -> Result<Vec<usize>, ShapeInferenceError> {
    let input_shape = node_shapes.get(&x.0).cloned().ok_or_else(|| {
        ShapeInferenceError::MissingInputShape {
            node_name: node_name.to_string(),
            input_id: x.0.clone(),
        }
    })?;
    let weight_shape = node_shapes.get(&weight.0).cloned().ok_or_else(|| {
        ShapeInferenceError::MissingInputShape {
            node_name: node_name.to_string(),
            input_id: weight.0.clone(),
        }
    })?;

    // C_out is the first dimension of the weight tensor
    let c_out = weight_shape.first().copied().unwrap_or(0);

    // Determine spatial dimensions from input (skip batch and channel dims)
    let spatial_dims: Vec<usize> = if input_shape.len() > 2 {
        input_shape[2..].to_vec()
    } else {
        vec![]
    };

    // Compute output spatial dimensions
    let n_spatial = strides.len().max(1);
    let mut out_spatial = Vec::with_capacity(n_spatial);

    for i in 0..n_spatial {
        let input_dim = spatial_dims.get(i).copied().unwrap_or(1);
        let stride = strides.get(i).copied().unwrap_or(1).max(1);
        let dilation = dilations.get(i).copied().unwrap_or(1).max(1);
        // kernel_size from weight: for 2D conv, weight is [C_out, C_in/groups, kH, kW]
        // so kernel dims start at index 2
        let kernel_size = if weight_shape.len() > 2 + i {
            weight_shape[2 + i]
        } else {
            // Fallback: assume kernel_size = 1 if not available
            1
        };

        // pad_amounts layout for 2D: [pad_b, pad_a, pad_r, pad_l]
        // For 1D: [pad_e, pad_s]
        // Total padding for spatial dim i:
        //   1D: dim 0 → pad_e + pad_s
        //   2D: dim 0 (height) → pad_b + pad_a, dim 1 (width) → pad_r + pad_l
        let pad_total = if pad_amounts.len() >= 2 * n_spatial {
            pad_amounts[2 * i] + pad_amounts[2 * i + 1]
        } else if pad_amounts.len() > 2 * i {
            pad_amounts[2 * i]
        } else {
            0
        };

        // Standard conv output formula:
        // out = floor((in + pad_total - dilation * (kernel - 1) - 1) / stride) + 1
        let effective_kernel = dilation * (kernel_size.saturating_sub(1)) + 1;
        let out_dim = if input_dim == 0 {
            0
        } else {
            (input_dim + pad_total).saturating_sub(effective_kernel) / stride + 1
        };
        out_spatial.push(out_dim);
    }

    // Assemble output: [batch, c_out, out_spatial...]
    let mut out = Vec::with_capacity(2 + out_spatial.len());
    if let Some(&batch) = input_shape.first() {
        out.push(batch);
    }
    out.push(c_out);
    out.extend(out_spatial);
    Ok(out)
}

/// Fallible version of reduce_shape for T-P5-04.
fn reduce_shape_fallible(
    x: &ane_ir::mir::MirNodeId,
    axes: &[usize],
    keep_dims: bool,
    node_shapes: &HashMap<String, Vec<usize>>,
    node_name: &str,
) -> Result<Vec<usize>, ShapeInferenceError> {
    match node_shapes.get(&x.0) {
        Some(input_shape) => Ok(shape_ops::reduce_shape(input_shape, axes, keep_dims)),
        None => Err(ShapeInferenceError::MissingInputShape {
            node_name: node_name.to_string(),
            input_id: x.0.clone(),
        }),
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
    // T-81 (I-56): Tests updated to reflect that compat_input_dtype now
    // trusts the MIR node's declared dtype directly instead of using
    // name-based string heuristics.

    #[test]
    fn test_compat_input_dtype_uses_declared_dtype() {
        // T-81: The function now uses the declared MilDtype directly,
        // regardless of the tensor name. input_ids with Int32 dtype → Int32.
        let result = compat_input_dtype("input_ids", &MilDtype::Int32);
        assert_eq!(result, MilDtypeCompat::Int32);
    }

    #[test]
    fn test_compat_input_dtype_input_ids_with_fp16_returns_fp16() {
        // T-81: Previously, name.contains("input_ids") would override to Int32
        // even when the declared dtype was Fp16. Now the declared dtype is used.
        let result = compat_input_dtype("input_ids", &MilDtype::Fp16);
        assert_eq!(result, MilDtypeCompat::Fp16);
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

    #[test]
    fn test_compat_input_dtype_int32_passthrough() {
        // T-81: Verify Int32 tensors are correctly mapped regardless of name.
        let result = compat_input_dtype("my_input_ids_special", &MilDtype::Int32);
        assert_eq!(result, MilDtypeCompat::Int32);
    }

    #[test]
    fn test_compat_input_dtype_no_name_based_override() {
        // T-81: Verify that name heuristics no longer override dtype.
        // A tensor named "something_input_ids_something" with Fp32 dtype
        // should return Fp32, not Int32.
        let result = compat_input_dtype("my_input_ids_special", &MilDtype::Fp32);
        assert_eq!(result, MilDtypeCompat::Fp32);
    }

    // ─── compat_input_shape ───────────────────────────────────────────────

    #[test]
    fn test_compat_input_shape_non_empty_returns_as_is() {
        assert_eq!(compat_input_shape("x", &[2, 3, 4], 512), vec![2, 3, 4]);
    }

    #[test]
    fn test_compat_input_shape_empty_input_ids_fallback() {
        // Legacy heuristic still works when no explicit_shape is provided
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

    // ─── compat_input_shape_explicit (M-018) ────────────────────────────

    #[test]
    fn test_compat_input_shape_explicit_overrides_name_heuristic() {
        // When explicit_shape is provided, it should be used even for "input_ids"
        assert_eq!(
            compat_input_shape_explicit("input_ids", &[], 512, Some(&[1, 2048])),
            vec![1, 2048]
        );
    }

    #[test]
    fn test_compat_input_shape_explicit_none_falls_back_to_heuristic() {
        // When explicit_shape is None, legacy heuristic still applies
        assert_eq!(
            compat_input_shape_explicit("input_ids", &[], 512, None),
            vec![1, 512]
        );
    }

    #[test]
    fn test_compat_input_shape_explicit_empty_treated_as_none() {
        // When explicit_shape is Some(&[]), treat as no hint
        assert_eq!(
            compat_input_shape_explicit("input_ids", &[], 512, Some(&[])),
            vec![1, 512]
        );
    }

    #[test]
    fn test_compat_input_shape_explicit_non_empty_shape_takes_priority() {
        // When node shape is non-empty, it takes priority over explicit_shape
        assert_eq!(
            compat_input_shape_explicit("input_ids", &[2, 3], 512, Some(&[1, 2048])),
            vec![2, 3]
        );
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
        // Legacy heuristic still works when no explicit_shape is provided
        assert_eq!(compat_output_shape("input_ids_node", &op, &[], &ns, 32768), vec![1, 32768]);
    }

    // ─── compat_output_shape_explicit (M-018) ───────────────────────────

    #[test]
    fn test_compat_output_shape_explicit_overrides_name_heuristic() {
        // When explicit_shape is provided, it should be used even for "input_ids" names
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(
            compat_output_shape_explicit("input_ids_node", &op, &[], &ns, 32768, Some(&[1, 2048])),
            vec![1, 2048]
        );
    }

    #[test]
    fn test_compat_output_shape_explicit_none_falls_back_to_heuristic() {
        // When explicit_shape is None, legacy heuristic still applies
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(
            compat_output_shape_explicit("input_ids_node", &op, &[], &ns, 32768, None),
            vec![1, 32768]
        );
    }

    #[test]
    fn test_compat_output_shape_explicit_empty_treated_as_none() {
        // When explicit_shape is Some(&[]), treat as no hint
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(
            compat_output_shape_explicit("input_ids_node", &op, &[], &ns, 32768, Some(&[])),
            vec![1, 32768]
        );
    }

    #[test]
    fn test_compat_output_shape_explicit_non_empty_shape_takes_priority() {
        // When node shape is non-empty, it takes priority over explicit_shape
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(
            compat_output_shape_explicit("input_ids_node", &op, &[2, 3], &ns, 32768, Some(&[1, 2048])),
            vec![2, 3]
        );
    }

    // ─── compat_output_shape: unary shape-propagating ops ─────────────────

    #[test]
    fn test_unary_silu_propagates_shape() {
        let op = MirOp::MILSilu { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1024]);
    }

    #[test]
    fn test_unary_abs_propagates_shape() {
        let op = MirOp::MILAbs { name: "a".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![4, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 5]);
    }

    #[test]
    fn test_unary_relu_propagates_shape() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_unary_sigmoid_propagates_shape() {
        let op = MirOp::MILSigmoid { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![8]);
    }

    #[test]
    fn test_unary_tanh_propagates_shape() {
        let op = MirOp::MILTanh { name: "t".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_unary_gelu_propagates_shape() {
        let op = MirOp::MILGelu { name: "g".into(), x: nid("x"), mode: "exact".into() };
        let ns = shapes_with(vec![("x", vec![2, 4, 6])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 4, 6]);
    }

    #[test]
    fn test_unary_exp_propagates_shape() {
        let op = MirOp::MILExp { name: "e".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 3]);
    }

    #[test]
    fn test_unary_cos_propagates_shape() {
        let op = MirOp::MILCos { name: "c".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![7])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![7]);
    }

    #[test]
    fn test_unary_sin_propagates_shape() {
        let op = MirOp::MILSin { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![7])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![7]);
    }

    #[test]
    fn test_unary_cast_propagates_shape() {
        let op = MirOp::MILCast { name: "c".into(), x: nid("x"), dtype: MilDtype::Fp32 };
        let ns = shapes_with(vec![("x", vec![1, 128])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 128]);
    }

    #[test]
    fn test_unary_rsqrt_propagates_shape() {
        let op = MirOp::MILRsqrt { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![4, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 8]);
    }

    #[test]
    fn test_unary_neg_propagates_shape() {
        let op = MirOp::MILNeg { name: "n".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_unary_sqrt_propagates_shape() {
        let op = MirOp::MILSqrt { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![5, 6])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![5, 6]);
    }

    #[test]
    fn test_unary_logical_not_propagates_shape() {
        let op = MirOp::MILLogicalNot { name: "ln".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![1, 10])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 10]);
    }

    #[test]
    fn test_unary_ceil_propagates_shape() {
        let op = MirOp::MILCeil { name: "c".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3]);
    }

    #[test]
    fn test_unary_floor_propagates_shape() {
        let op = MirOp::MILFloor { name: "f".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3]);
    }

    #[test]
    fn test_unary_round_propagates_shape() {
        let op = MirOp::MILRound { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3]);
    }

    #[test]
    fn test_unary_sign_propagates_shape() {
        let op = MirOp::MILSign { name: "s".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3]);
    }

    #[test]
    fn test_unary_log_propagates_shape() {
        let op = MirOp::MILLog { name: "l".into(), x: nid("x"), epsilon: 1e-7 };
        let ns = shapes_with(vec![("x", vec![2, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 4]);
    }

    #[test]
    fn test_unary_leaky_relu_propagates_shape() {
        let op = MirOp::MILLeakyRelu { name: "lr".into(), x: nid("x"), alpha: 0.01 };
        let ns = shapes_with(vec![("x", vec![1, 32])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 32]);
    }

    #[test]
    fn test_unary_clip_propagates_shape() {
        let op = MirOp::MILClip { name: "c".into(), x: nid("x"), min_val: 0.0, max_val: 6.0 };
        let ns = shapes_with(vec![("x", vec![1, 32])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 32]);
    }

    #[test]
    fn test_unary_unknown_input_returns_empty() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("unknown") };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: binary ops with broadcast ───────────────────

    #[test]
    fn test_binary_add_same_shape() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3]), ("b", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_binary_mul_broadcast_different_rank() {
        let op = MirOp::MILMul { name: "m".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 512, 64]), ("b", vec![64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 64]);
    }

    #[test]
    fn test_binary_sub_broadcast_scalar() {
        let op = MirOp::MILSub { name: "s".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 512]), ("b", vec![1])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512]);
    }

    #[test]
    fn test_binary_maximum_broadcast() {
        let op = MirOp::MILMaximum { name: "mx".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![4, 1, 8]), ("b", vec![1, 6, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 6, 8]);
    }

    #[test]
    fn test_binary_minimum_same_shape() {
        let op = MirOp::MILMinimum { name: "mn".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![5, 5]), ("b", vec![5, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![5, 5]);
    }

    #[test]
    fn test_binary_realdiv_broadcast() {
        let op = MirOp::MILRealDiv { name: "d".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3, 4]), ("b", vec![4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3, 4]);
    }

    #[test]
    fn test_binary_pow_broadcast() {
        let op = MirOp::MILPow { name: "p".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4]), ("b", vec![1])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4]);
    }

    #[test]
    fn test_binary_floordiv_same_shape() {
        let op = MirOp::MILFloorDiv { name: "fd".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![8, 8]), ("b", vec![8, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![8, 8]);
    }

    #[test]
    fn test_binary_mod_broadcast() {
        let op = MirOp::MILMod { name: "md".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 10]), ("b", vec![10])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 10]);
    }

    #[test]
    fn test_binary_equal_broadcast() {
        let op = MirOp::MILEqual { name: "eq".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 64]), ("b", vec![64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_binary_not_equal_same_shape() {
        let op = MirOp::MILNotEqual { name: "ne".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 3]), ("b", vec![3, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 3]);
    }

    #[test]
    fn test_binary_greater_broadcast() {
        let op = MirOp::MILGreater { name: "gt".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![4, 1]), ("b", vec![1, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 5]);
    }

    #[test]
    fn test_binary_greater_equal_broadcast() {
        let op = MirOp::MILGreaterEqual { name: "ge".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3]), ("b", vec![3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_binary_less_same_shape() {
        let op = MirOp::MILLess { name: "lt".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 2]), ("b", vec![2, 2])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 2]);
    }

    #[test]
    fn test_binary_less_equal_broadcast() {
        let op = MirOp::MILLessEqual { name: "le".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 8]), ("b", vec![8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 8]);
    }

    #[test]
    fn test_binary_logical_and_broadcast() {
        let op = MirOp::MILLogicalAnd { name: "la".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 16]), ("b", vec![16])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 16]);
    }

    #[test]
    fn test_binary_logical_or_broadcast() {
        let op = MirOp::MILLogicalOr { name: "lo".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 16]), ("b", vec![16])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 16]);
    }

    #[test]
    fn test_binary_only_x_known_returns_x() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_binary_only_y_known_returns_y() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("b", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_binary_both_unknown_returns_empty() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    #[test]
    fn test_binary_incompatible_broadcast_returns_x_fallback() {
        // [3, 4] and [5, 6] are not broadcast-compatible → falls back to shape_a
        let op = MirOp::MILAdd { name: "a".into(), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4]), ("b", vec![5, 6])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4]);
    }

    // ─── compat_output_shape: softmax ─────────────────────────────────────

    #[test]
    fn test_softmax_propagates_shape() {
        let op = MirOp::MILSoftmax { name: "sm".into(), x: nid("x"), axis: -1 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 64, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: matmul ──────────────────────────────────────

    #[test]
    fn test_matmul_2d() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![4, 8]), ("b", vec![8, 16])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 16]);
    }

    #[test]
    fn test_matmul_batched() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        // [2, 3, 4] × [2, 4, 5] → [2, 3, 5]
        let ns = shapes_with(vec![("a", vec![2, 3, 4]), ("b", vec![2, 4, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3, 5]);
    }

    #[test]
    fn test_matmul_batched_broadcast() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        // [2, 1, 3, 4] × [1, 6, 4, 5] → [2, 6, 3, 5]
        let ns = shapes_with(vec![("a", vec![2, 1, 3, 4]), ("b", vec![1, 6, 4, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 6, 3, 5]);
    }

    #[test]
    fn test_matmul_only_x_known() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![4, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![4, 8]);
    }

    #[test]
    fn test_matmul_both_unknown() {
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    #[test]
    fn test_matmul_degenerate_1d_fallback() {
        // 1D × 2D falls back to x shape
        let op =
            MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
        let ns = shapes_with(vec![("a", vec![8]), ("b", vec![8, 16])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![8]);
    }

    // ─── compat_output_shape: reshape ─────────────────────────────────────

    #[test]
    fn test_reshape_returns_target_shape() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![2, 3, 4] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3, 4]);
    }

    #[test]
    fn test_reshape_flatten() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![1, 24] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 24]);
    }

    // ─── compat_output_shape: transpose ───────────────────────────────────

    #[test]
    fn test_transpose_2d() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
        let ns = shapes_with(vec![("x", vec![3, 5])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![5, 3]);
    }

    #[test]
    fn test_transpose_4d_nchw_to_nhwc() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![0, 2, 3, 1] };
        let ns = shapes_with(vec![("x", vec![1, 64, 8, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 8, 8, 64]);
    }

    #[test]
    fn test_transpose_unknown_input_returns_empty() {
        let op = MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![1, 0] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: tile ────────────────────────────────────────

    #[test]
    fn test_tile_multiplies_by_reps() {
        // CQ-22 fix: tile now correctly multiplies dims by reps instead of
        // just propagating the input shape.
        let op = MirOp::MILTile { name: "t".into(), x: nid("x"), reps: vec![2, 3] };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 192]);
    }

    #[test]
    fn test_tile_identity_reps() {
        // Tile with all-1 reps preserves input shape
        let op = MirOp::MILTile { name: "t".into(), x: nid("x"), reps: vec![1, 1] };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_tile_gqa_style() {
        // GQA tile: repeat along head dimension
        let op = MirOp::MILTile { name: "t".into(), x: nid("x"), reps: vec![1, 1, 2, 1, 1] };
        let ns = shapes_with(vec![("x", vec![1, 8, 1, 512, 128])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 8, 2, 512, 128]);
    }

    #[test]
    fn test_tile_unknown_input_returns_empty() {
        let op = MirOp::MILTile { name: "t".into(), x: nid("unknown"), reps: vec![2, 3] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3, 4]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 128]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: gather (embedding) ──────────────────────────

    #[test]
    fn test_gather_embedding_2d() {
        // x=[151936, 1024], indices=[1, 512], axis=0 → [1, 512, 1024]
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![151936, 1024]), ("idx", vec![1, 512])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1024]);
    }

    #[test]
    fn test_gather_axis_last() {
        // x=[1, 512, 1024], indices=[1, 512, 10], axis=2 → [1, 512, 1, 512, 10]
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 2 };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024]), ("idx", vec![1, 512, 10])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1, 512, 10]);
    }

    #[test]
    fn test_gather_no_indices_shape_falls_back_to_input() {
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![100, 50])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![100, 50]);
    }

    #[test]
    fn test_gather_both_unknown_returns_empty() {
        let op = MirOp::MILGather { name: "g".into(), x: nid("x"), indices: nid("idx"), axis: 0 };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: reduce_mean / reduce_max / reduce_min / reduce_prod ───

    #[test]
    fn test_reduce_mean_keep_dims() {
        let op =
            MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 12, 64, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 1, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_reduce_max_keep_dims() {
        let op =
            MirOp::MILReduceMax { name: "rx".into(), x: nid("x"), axes: vec![1], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 1, 4]);
    }

    #[test]
    fn test_reduce_min_no_keep_dims() {
        let op =
            MirOp::MILReduceMin { name: "rn".into(), x: nid("x"), axes: vec![0], keep_dims: false };
        let ns = shapes_with(vec![("x", vec![5, 6, 7])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![6, 7]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 1, 1]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: expand_dims ─────────────────────────────────

    #[test]
    fn test_expand_dims_single_axis() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_multiple_axes() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![0, 2] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        // Insert 1 at pos 0 → [1, 3, 4], then at pos 2 → [1, 3, 1, 4]
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_axis_at_end() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![2] };
        let ns = shapes_with(vec![("x", vec![3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4, 1]);
    }

    #[test]
    fn test_expand_dims_unknown_input_returns_empty() {
        let op = MirOp::MILExpandDims { name: "ed".into(), x: nid("x"), axis: vec![0] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: squeeze ─────────────────────────────────────

    #[test]
    fn test_squeeze_single_axis() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes_with(vec![("x", vec![3, 1, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4]);
    }

    #[test]
    fn test_squeeze_multiple_axes() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1, 3] };
        let ns = shapes_with(vec![("x", vec![1, 1, 4, 1])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 4]);
    }

    #[test]
    fn test_squeeze_axis_out_of_range_is_noop() {
        // Axis 5 is out of range for a 3D tensor; the removal is skipped
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![5] };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3, 4]);
    }

    #[test]
    fn test_squeeze_unknown_input_returns_empty() {
        let op = MirOp::MILSqueeze { name: "sq".into(), x: nid("x"), axis: vec![1] };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 10, 10]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![5, 7]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 112, 64]);
    }

    #[test]
    fn test_concat_single_input() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a")], axis: 1 };
        let ns = shapes_with(vec![("a", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_concat_unknown_first_input_returns_empty() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a"), nid("b")], axis: 1 };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    #[test]
    fn test_concat_axis_out_of_range_only_first_shape_used() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a"), nid("b")], axis: 5 };
        let ns = shapes_with(vec![("a", vec![1, 2, 3]), ("b", vec![1, 2, 3])]);
        // axis 5 is out of range for 3D tensor, so out[5] is never written
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 2, 3]);
    }

    // ─── compat_output_shape: where ───────────────────────────────────────

    #[test]
    fn test_where_all_same_shape() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![2, 3]), ("a", vec![2, 3]), ("b", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_where_broadcast_condition_scalar() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![1]), ("a", vec![1, 64]), ("b", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
    }

    #[test]
    fn test_where_only_x_known_returns_x() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4]);
    }

    #[test]
    fn test_where_only_condition_known_returns_condition() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("c", vec![3, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 4]);
    }

    #[test]
    fn test_where_all_unknown_returns_empty() {
        let op =
            MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 64]);
    }

    // ─── compat_output_shape: topk ────────────────────────────────────────

    #[test]
    fn test_topk_positive_axis() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 10, axis: 2 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 10]);
    }

    #[test]
    fn test_topk_negative_axis() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 5, axis: -1 };
        let ns = shapes_with(vec![("x", vec![1, 12, 64])]);
        // axis=-1 → axis = 3 - 1 = 2
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 5]);
    }

    #[test]
    fn test_topk_unknown_input_returns_empty() {
        let op = MirOp::MILTopk { name: "tk".into(), x: nid("x"), k: 10, axis: 1 };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 12, 64, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 2, 128, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 2, 128, 64]);
    }

    #[test]
    fn test_state_write_propagates_value_shape() {
        let op =
            MirOp::MILStateWrite { name: "sw".into(), state_ref: "kv".into(), value: nid("v") };
        let ns = shapes_with(vec![("v", vec![2, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 4]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 3, 32, 32]);
    }

    // ─── compat_output_shape: select ──────────────────────────────────────

    #[test]
    fn test_select_propagates_x_shape() {
        let op =
            MirOp::MILSelect { name: "sel".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        let ns = shapes_with(vec![("a", vec![1, 128])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 128]);
    }

    // ─── compat_output_shape: split ───────────────────────────────────────

    #[test]
    fn test_split_evenly() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 1, num_splits: 4 };
        let ns = shapes_with(vec![("x", vec![1, 64, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 16, 8]);
    }

    #[test]
    fn test_split_axis_0() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 0, num_splits: 2 };
        let ns = shapes_with(vec![("x", vec![4, 8])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 8]);
    }

    #[test]
    fn test_split_unknown_input_returns_empty() {
        let op = MirOp::MILSplit { name: "sp".into(), x: nid("x"), axis: 1, num_splits: 2 };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 6, 32]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 3, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![10, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 9]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
    }

    // ─── compat_output_shape: identity ────────────────────────────────────

    #[test]
    fn test_identity_propagates_input_shape() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 4])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 4]);
    }

    #[test]
    fn test_identity_placeholder_returns_default() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("__placeholder__") };
        let ns = shapes();
        // T-36: placeholder now uses max_seq_len (32768) instead of hardcoded 512
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 32768]);
    }

    #[test]
    fn test_identity_unknown_input_returns_empty() {
        let op = MirOp::MILIdentity { name: "id".into(), x: nid("x") };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![3, 3, 4]);
    }

    #[test]
    fn test_stack_axis_2() {
        let op = MirOp::MILStack { name: "st".into(), values: vec![nid("a"), nid("b")], axis: 2 };
        let ns = shapes_with(vec![("a", vec![1, 64]), ("b", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 2]);
    }

    #[test]
    fn test_stack_unknown_first_input_returns_empty() {
        let op = MirOp::MILStack { name: "st".into(), values: vec![nid("a"), nid("b")], axis: 0 };
        let ns = shapes();
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("const_node", &op, &[], &ns, 32768), vec![64, 64]);
    }

    #[test]
    fn test_const_scalar_pattern() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "scalar://fp16/0.5".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape("const_node", &op, &[], &ns, 32768), vec![1]);
    }

    #[test]
    fn test_const_scalar_fp32_pattern() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "scalar://fp32/1.0".into(),
            dtype: MilDtype::Fp32,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape("const_node", &op, &[], &ns, 32768), vec![1]);
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
        assert_eq!(compat_output_shape("const_node", &op, &[], &ns, 32768), vec![4]);
    }

    #[test]
    fn test_const_unknown_returns_empty() {
        let op = MirOp::MILConst {
            name: "c".into(),
            value_path: "weights/w.bin".into(),
            dtype: MilDtype::Fp16,
        };
        let ns = shapes();
        assert_eq!(compat_output_shape("const_node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), Vec::<usize>::new());
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
        let op =
            MirOp::MILReduceSum { name: "rs".into(), x: nid("x"), axes: vec![1], keep_dims: false };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 1024]);
    }

    #[test]
    fn test_reduce_sum_square_keep_dims() {
        let op = MirOp::MILReduceSumSquare {
            name: "rss".into(),
            x: nid("x"),
            axes: vec![2],
            keep_dims: true,
        };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1]);
    }

    #[test]
    fn test_reduce_l2_norm_no_keep() {
        let op = MirOp::MILReduceL2Norm {
            name: "rl2".into(),
            x: nid("x"),
            axes: vec![1, 2],
            keep_dims: false,
        };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1]);
    }

    #[test]
    fn test_reduce_argmax_keep_dims() {
        let op =
            MirOp::MILReduceArgmax { name: "ram".into(), x: nid("x"), axis: 2, keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1]);
    }

    #[test]
    fn test_reduce_argmin_no_keep_dims() {
        let op =
            MirOp::MILReduceArgmin { name: "rin".into(), x: nid("x"), axis: 1, keep_dims: false };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 1024]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 32]);
    }

    #[test]
    fn test_l2_norm_propagates_shape() {
        let op = MirOp::MILL2Norm { name: "l2".into(), x: nid("x"), epsilon: 1e-12, axes: vec![1] };
        let ns = shapes_with(vec![("x", vec![1, 512])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1024]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 512, 1024]);
    }

    // ─── CQ-22: Additional unary ops ──────────────────────────────────

    #[test]
    fn test_square_propagates_shape() {
        let op = MirOp::MILSquare { name: "sq".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![2, 3]);
    }

    #[test]
    fn test_prelu_propagates_shape() {
        let op = MirOp::MILPrelu { name: "pr".into(), x: nid("x"), alpha: "a.bin".into() };
        let ns = shapes_with(vec![("x", vec![1, 64])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 256]);
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
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 128]);
    }

    // ─── CQ-22: ReshapeLike ───────────────────────────────────────────

    #[test]
    fn test_reshape_like_uses_ref_tensor_shape() {
        let op = MirOp::MILReshapeLike { name: "rl".into(), x: nid("x"), ref_tensor: nid("r") };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024]), ("r", vec![1, 64, 8192])]);
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 64, 8192]);
    }

    // ─── CQ-22: Flatten2d ─────────────────────────────────────────────

    #[test]
    fn test_flatten2d_axis_1() {
        let op = MirOp::MILFlatten2d { name: "f2".into(), x: nid("x"), axis: 1 };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        // Flatten dims from axis 1 onwards: product = 512 * 1024 = 524288
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![1, 524288]);
    }

    #[test]
    fn test_flatten2d_axis_0() {
        let op = MirOp::MILFlatten2d { name: "f2".into(), x: nid("x"), axis: 0 };
        let ns = shapes_with(vec![("x", vec![2, 3, 4])]);
        // Flatten all dims: product = 2 * 3 * 4 = 24
        assert_eq!(compat_output_shape("node", &op, &[], &ns, 32768), vec![24]);
    }

    // ─── T-P5-04: compat_output_shape_fallible tests ─────────────────────

    #[test]
    fn test_fallible_known_shape_returns_ok() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        let result = compat_output_shape_fallible("node", &op, &[3, 4], &ns, 512);
        assert_eq!(result.unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_fallible_input_ids_returns_name_heuristic_error() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes();
        let result = compat_output_shape_fallible("input_ids_node", &op, &[], &ns, 512);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            ShapeInferenceError::NameHeuristicUsed { node_name, heuristic, inferred_shape } => {
                assert_eq!(node_name, "input_ids_node");
                assert!(heuristic.contains("input_ids"));
                assert_eq!(inferred_shape, vec![1, 512]);
            }
            _ => panic!("Expected NameHeuristicUsed error, got: {:?}", err),
        }
    }

    #[test]
    fn test_fallible_unary_with_shape_returns_ok() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("x") };
        let ns = shapes_with(vec![("x", vec![2, 3])]);
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert_eq!(result.unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_fallible_unary_missing_input_returns_error() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("unknown") };
        let ns = shapes();
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeInferenceError::MissingInputShape { node_name, input_id } => {
                assert_eq!(node_name, "node");
                assert_eq!(input_id, "unknown");
            }
            other => panic!("Expected MissingInputShape, got: {:?}", other),
        }
    }

    #[test]
    fn test_fallible_unknown_op_returns_error() {
        // MILConst with no matching shape in node_shapes returns Indeterminate
        // (not UnknownOp, since MILConst IS handled in the fallible function)
        let op = MirOp::MILConst { name: "c".into(), value_path: "weights.bin".into(), dtype: MilDtype::Fp16 };
        let ns = shapes();
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeInferenceError::Indeterminate { node_name, reason } => {
                assert_eq!(node_name, "node");
                assert!(reason.contains("MILConst"), "Reason should mention MILConst: {}", reason);
            }
            other => panic!("Expected Indeterminate, got: {:?}", other),
        }
    }

    #[test]
    fn test_fallible_reduce_with_shape_returns_ok() {
        let op = MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![2], keep_dims: true };
        let ns = shapes_with(vec![("x", vec![1, 512, 1024])]);
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 512, 1]);
    }

    #[test]
    fn test_fallible_reduce_missing_input_returns_error() {
        let op = MirOp::MILReduceMean { name: "rm".into(), x: nid("unknown"), axes: vec![2], keep_dims: true };
        let ns = shapes();
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeInferenceError::MissingInputShape { .. } => {}
            other => panic!("Expected MissingInputShape, got: {:?}", other),
        }
    }

    #[test]
    fn test_fallible_reshape_returns_shape() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![2, 3, 4] };
        let ns = shapes();
        let result = compat_output_shape_fallible("node", &op, &[], &ns, 512);
        assert_eq!(result.unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_inference_error_display() {
        let err = ShapeInferenceError::MissingInputShape {
            node_name: "test_node".into(),
            input_id: "input_0".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test_node"));
        assert!(msg.contains("input_0"));

        let err2 = ShapeInferenceError::UnknownOp {
            node_name: "test_node".into(),
            op_name: "conv".into(),
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("conv"));
    }
}
