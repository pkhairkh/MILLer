//! Shared Shape Operations — Pure Functions on `Vec<usize>`
//!
//! This module centralizes shape-computation logic that was previously duplicated
//! across `passes/mil_lower.rs` (AIR-level) and `bridge/shape_inference.rs`
//! (MIR-level). Every function is a pure function on `&[usize]` slices with
//! no dependency on IR types — the callers extract the relevant slices from
//! their type-specific contexts.
//!
//! ## Design Principles
//!
//! - **No IR types**: Functions operate on `&[usize]` only, keeping `ane-ir`
//!   free of cross-crate type dependencies.
//! - **No panics**: Functions return `Option<Vec<usize>>` or `Result<Vec<usize>>`
//!   for fallible operations. Callers decide how to handle errors.
//! - **Bug-free**: The MILTile and MILExpandDims implementations here fix
//!   bugs that existed in the bridge's local copies (CQ-22).
//!
//! ## Bug Fixes
//!
//! 1. **MILTile** (bridge was wrong): Previously propagated the input shape
//!    unchanged. Correctly computes `out[i] = input_shape[i] * reps[i]`.
//! 2. **MILExpandDims**: The bridge and mil_lower implementations diverged
//!    on multi-axis handling. This module uses the correct Core ML semantics
//!    where axes specify **output positions** (not input positions), so no
//!    insertion offset is needed.

/// Compute the broadcast output shape from two input shapes (numpy-style).
///
/// For each dimension pair (right-aligned), the output dimension is the larger
/// of the two inputs. Missing dimensions in the shorter shape are treated as 1.
/// Returns `None` if the shapes are not broadcast-compatible.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::broadcast_shape;
/// assert_eq!(broadcast_shape(&[1, 512, 64], &[64]), Some(vec![1, 512, 64]));
/// assert_eq!(broadcast_shape(&[3, 4], &[5, 6]), None); // incompatible
/// ```
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
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
/// If `keep_dims` is true, reduced dimensions are set to 1 (preserving rank).
/// If `keep_dims` is false, reduced dimensions are removed entirely.
///
/// Uses the filter approach (non-mutating) rather than remove-from-back which
/// requires sorting axes in reverse order.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::reduce_shape;
/// assert_eq!(reduce_shape(&[1, 512, 1024], &[1, 2], true), vec![1, 1, 1]);
/// assert_eq!(reduce_shape(&[1, 512, 1024], &[1, 2], false), vec![1]);
/// ```
pub fn reduce_shape(shape: &[usize], axes: &[usize], keep_dims: bool) -> Vec<usize> {
    if keep_dims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| if axes.contains(&i) { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &dim)| dim)
            .collect()
    }
}

/// Compute the output shape of a Tile (repeat) operation.
///
/// Each output dimension is `input_shape[i] * reps[i]`. If `reps` is shorter
/// than the input shape, missing reps are treated as 1 (no tiling). If `reps`
/// is longer, missing input dims are treated as 1.
///
/// # Bug Note (CQ-22)
///
/// The previous bridge implementation just propagated the input shape unchanged,
/// which is wrong for any Tile with `reps != [1, 1, ...]`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::tile_shape;
/// assert_eq!(tile_shape(&[1, 8, 512, 128], &[1, 1, 2, 1]), vec![1, 8, 1024, 128]);
/// ```
pub fn tile_shape(input_shape: &[usize], reps: &[usize]) -> Vec<usize> {
    let max_len = input_shape.len().max(reps.len());
    let mut out = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let dim = input_shape.get(i).copied().unwrap_or(1);
        let rep = reps.get(i).copied().unwrap_or(1);
        out.push(dim * rep);
    }
    out
}

/// Compute the output shape of an ExpandDims operation (insert 1-sized dims).
///
/// In Core ML's `expand_dims`, the `axes` parameter specifies positions in
/// the **output** shape where new 1-sized dimensions should be inserted.
/// Axes are sorted and then inserted in order. Since each axis refers to an
/// output position (not an input position), we insert at the raw sorted axis
/// values — the shifting of subsequent dimensions happens naturally.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::expand_dims_shape;
/// assert_eq!(expand_dims_shape(&[3, 4], &[0]), vec![1, 3, 4]);
/// assert_eq!(expand_dims_shape(&[3, 4], &[0, 2]), vec![1, 3, 1, 4]);
/// assert_eq!(expand_dims_shape(&[3, 4], &[1, 2]), vec![3, 1, 1, 4]);
/// ```
pub fn expand_dims_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    for &ax in &sorted_axes {
        let insert_pos = if ax >= out.len() { out.len() } else { ax };
        out.insert(insert_pos, 1);
    }
    out
}

/// Compute the output shape of a Squeeze operation (remove dims at specified axes).
///
/// Axes are sorted in descending order so that removals from back to front
/// don't shift positions of earlier axes.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::squeeze_shape;
/// assert_eq!(squeeze_shape(&[1, 3, 1, 4], &[0, 2]), vec![3, 4]);
/// ```
pub fn squeeze_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a)); // Remove from back to front
    for &ax in &sorted_axes {
        if ax < out.len() {
            out.remove(ax);
        }
    }
    out
}

/// Compute the output shape of a Transpose operation.
///
/// Each output dimension is `input_shape[perm[i]]`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::transpose_shape;
/// assert_eq!(transpose_shape(&[1, 512, 8, 128], &[0, 2, 1, 3]), vec![1, 8, 512, 128]);
/// ```
pub fn transpose_shape(input_shape: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| input_shape.get(p).copied().unwrap_or(0)).collect()
}

/// Resolve zero-placeholder dimensions in a reshape target shape.
///
/// Core ML's reshape treats 0 as a literal zero dimension, not as an
/// "infer from input" sentinel (unlike PyTorch's -1). This function
/// attempts to resolve zero placeholders using two strategies:
///
/// 1. **Positional**: copy dim from `input_shape[i]` → `target[i]`.
///    Works when input and target have the same rank (e.g., 3D→3D
///    where the first dims align: `[B,S,E]`→`[B,S,H]`).
/// 2. **Element-count-based**: compute zeros from the total element count.
///    Works for rank-changing reshapes where positional is wrong
///    (e.g., `[B,H,S,D]`→`[B,S,E]`: pos 1 gives H instead of S).
///    For 2+ zeros, assumes batch=1 for all but the last.
///
/// Returns `Ok(resolved_shape)` or an error message if zero-resolution fails.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::resolve_reshape_zeros;
/// assert_eq!(resolve_reshape_zeros(&[1, 512, 1024], &[1, 0, 1024]).unwrap(), vec![1, 512, 1024]);
/// ```
pub fn resolve_reshape_zeros(
    input_shape: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>, String> {
    let input_elements: usize = input_shape.iter().product();
    let mut resolved = target_shape.to_vec();
    let has_zeros = resolved.contains(&0);

    if !has_zeros || input_elements == 0 {
        return Ok(resolved);
    }

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
        if resolved_elements != input_elements {
            // Positional resolution produced wrong count —
            // reset and use element-count-based inference
            resolved = target_shape.to_vec();
            positional_works = false;
        }
    }

    if !positional_works {
        // Step 2: Element-count-based inference
        let non_zero_product: usize = resolved.iter().filter(|&&d| d != 0).product();
        if let Some(remaining) = input_elements
            .checked_div(non_zero_product)
            .filter(|&r| r * non_zero_product == input_elements)
        {
            let zero_positions: Vec<usize> =
                resolved.iter().enumerate().filter(|(_, &d)| d == 0).map(|(i, _)| i).collect();

            if zero_positions.is_empty() {
                return Err(format!(
                    "Reshape zero-resolution internal error: zero_count indicated \
                     zeros exist in target_shape {:?} but no zero positions found",
                    target_shape
                ));
            }

            match zero_positions.len() {
                1 => {
                    resolved[zero_positions[0]] = remaining;
                }
                _ => {
                    // Two or more zeros: assume batch=1 for all but the last,
                    // compute last from remaining elements
                    for &pos in &zero_positions[..zero_positions.len() - 1] {
                        resolved[pos] = 1;
                    }
                    if let Some(&last_pos) = zero_positions.last() {
                        resolved[last_pos] = remaining;
                    }
                }
            }
        }
    }

    // Final validation: if zeros remain after resolution, this reshape is
    // malformed and would produce a Core ML model with literal zero
    // dimensions, which is invalid.
    if resolved.contains(&0) {
        return Err(format!(
            "Reshape zero-resolution failed: could not resolve all zero placeholders \
             in target_shape {:?} with input_shape {:?}. Resolved shape still contains \
             zeros: {:?}. Input has {} elements, non-zero target dims product is {}.",
            target_shape,
            input_shape,
            resolved,
            input_elements,
            resolved.iter().filter(|&&d| d != 0).product::<usize>()
        ));
    }

    Ok(resolved)
}

/// Compute the output shape of a Pad operation.
///
/// The `pad_amounts` slice has length `2 * rank`, where `pad_amounts[i]` is
/// the before-padding for axis `i` and `pad_amounts[i + rank]` is the
/// after-padding for axis `i`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::pad_shape;
/// assert_eq!(pad_shape(&[1, 512, 64], &[0, 0, 0, 0, 0, 0]), vec![1, 512, 64]);
/// assert_eq!(pad_shape(&[1, 512, 64], &[0, 1, 0, 0, 0, 0]), vec![1, 513, 64]);
/// ```
pub fn pad_shape(input_shape: &[usize], pad_amounts: &[usize]) -> Vec<usize> {
    let rank = input_shape.len();
    let mut out = input_shape.to_vec();
    for (i, slot) in out.iter_mut().enumerate().take(rank) {
        let before = pad_amounts.get(i).copied().unwrap_or(0);
        let after = pad_amounts.get(i + rank).copied().unwrap_or(0);
        *slot += before + after;
    }
    out
}

/// Compute the output shape of a Concat operation along an axis.
///
/// The output shape matches the first input's shape except at the concat
/// axis, where the dimension is the sum of all inputs' dimensions at that axis.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::concat_shape;
/// let shapes: Vec<&[usize]> = vec![&[1, 512, 64], &[1, 512, 64]];
/// assert_eq!(concat_shape(&shapes, 2), Some(vec![1, 512, 128]));
/// ```
pub fn concat_shape(input_shapes: &[&[usize]], axis: usize) -> Option<Vec<usize>> {
    let first = input_shapes.first()?;
    let mut out = first.to_vec();
    if axis < out.len() {
        let mut total_dim = 0usize;
        for shape in input_shapes {
            if let Some(&dim) = shape.get(axis) {
                total_dim += dim;
            }
        }
        out[axis] = total_dim;
    }
    Some(out)
}

/// Compute the output shape of a Split operation.
///
/// The output shape is the same as the input shape except that the dimension
/// at `axis` is divided by `num_splits`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::split_shape;
/// assert_eq!(split_shape(&[1, 512, 128], 1, 2), vec![1, 256, 128]);
/// ```
pub fn split_shape(input_shape: &[usize], axis: usize, num_splits: usize) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    if let Some(dim) = out.get_mut(axis) {
        *dim /= num_splits;
    }
    out
}

/// Compute the output shape of a Stack operation.
///
/// Like Concat but inserts a new dimension of size `num_values` at `axis`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::stack_shape;
/// assert_eq!(stack_shape(&[3, 4], 0, 2), vec![2, 3, 4]);
/// ```
pub fn stack_shape(first_input_shape: &[usize], axis: usize, num_values: usize) -> Vec<usize> {
    let mut out = first_input_shape.to_vec();
    let ax = if axis <= out.len() { axis } else { out.len() };
    out.insert(ax, num_values);
    out
}

/// Compute the output shape of a Gather (embedding lookup) operation.
///
/// The output shape replaces the `axis` dimension of `input_shape` with
/// `indices_shape`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::gather_shape;
/// assert_eq!(gather_shape(&[32000, 1024], &[1, 512], 0), vec![1, 512, 1024]);
/// ```
pub fn gather_shape(input_shape: &[usize], indices_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if i == axis {
            out.extend_from_slice(indices_shape);
        } else {
            out.push(dim);
        }
    }
    out
}

/// Compute the output shape of a Topk operation.
///
/// The output shape is the same as the input shape except that the dimension
/// at `axis` is replaced by `k`.
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::topk_shape;
/// assert_eq!(topk_shape(&[1, 512, 1024], 5, 2), vec![1, 512, 5]);
/// ```
pub fn topk_shape(input_shape: &[usize], k: usize, axis: usize) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    if axis < out.len() {
        out[axis] = k;
    }
    out
}

/// Format a shape as a human-readable string like "[1, 512, 2048]".
///
/// # Examples
/// ```
/// use ane_ir::shape_ops::format_shape;
/// assert_eq!(format_shape(&[1, 512, 2048]), "[1, 512, 2048]");
/// ```
pub fn format_shape(shape: &[usize]) -> String {
    format!("[{}]", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── broadcast_shape ──────────────────────────────────────────────

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn test_broadcast_different_rank() {
        assert_eq!(broadcast_shape(&[1, 512, 64], &[64]), Some(vec![1, 512, 64]));
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shape(&[1, 512], &[1]), Some(vec![1, 512]));
    }

    #[test]
    fn test_broadcast_3d() {
        assert_eq!(broadcast_shape(&[4, 1, 8], &[1, 6, 8]), Some(vec![4, 6, 8]));
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert_eq!(broadcast_shape(&[3, 4], &[5, 6]), None);
    }

    #[test]
    fn test_broadcast_empty() {
        assert_eq!(broadcast_shape(&[], &[]), Some(vec![]));
    }

    #[test]
    fn test_broadcast_one_empty() {
        assert_eq!(broadcast_shape(&[3], &[]), Some(vec![3]));
    }

    // ─── reduce_shape ────────────────────────────────────────────────

    #[test]
    fn test_reduce_keep_dims() {
        assert_eq!(reduce_shape(&[1, 512, 1024], &[1, 2], true), vec![1, 1, 1]);
    }

    #[test]
    fn test_reduce_no_keep_dims() {
        assert_eq!(reduce_shape(&[1, 512, 1024], &[1, 2], false), vec![1]);
    }

    #[test]
    fn test_reduce_single_axis_keep() {
        assert_eq!(reduce_shape(&[1, 512, 1024], &[1], true), vec![1, 1, 1024]);
    }

    #[test]
    fn test_reduce_single_axis_no_keep() {
        assert_eq!(reduce_shape(&[1, 512, 1024], &[1], false), vec![1, 1024]);
    }

    #[test]
    fn test_reduce_all_axes() {
        assert_eq!(reduce_shape(&[2, 3], &[0, 1], false), Vec::<usize>::new());
    }

    #[test]
    fn test_reduce_no_axes() {
        assert_eq!(reduce_shape(&[2, 3], &[], false), vec![2, 3]);
    }

    // ─── tile_shape ──────────────────────────────────────────────────

    #[test]
    fn test_tile_identity() {
        assert_eq!(tile_shape(&[1, 8, 512, 128], &[1, 1, 1, 1]), vec![1, 8, 512, 128]);
    }

    #[test]
    fn test_tile_gqa() {
        // GQA tile: repeat along head dimension
        assert_eq!(
            tile_shape(&[1, 8, 1, 512, 128], &[1, 1, 2, 1, 1]),
            vec![1, 8, 2, 512, 128]
        );
    }

    #[test]
    fn test_tile_short_reps() {
        // Reps shorter than input: missing reps treated as 1 (no tiling)
        assert_eq!(tile_shape(&[2, 3], &[3]), vec![6, 3]);
    }

    #[test]
    fn test_tile_long_reps() {
        // Reps longer than input: missing input dims treated as 1
        assert_eq!(tile_shape(&[3], &[2, 4]), vec![6, 4]);
    }

    #[test]
    fn test_tile_empty() {
        assert_eq!(tile_shape(&[], &[]), Vec::<usize>::new());
    }

    // ─── expand_dims_shape ──────────────────────────────────────────

    #[test]
    fn test_expand_dims_single_axis() {
        assert_eq!(expand_dims_shape(&[3, 4], &[0]), vec![1, 3, 4]);
        assert_eq!(expand_dims_shape(&[3, 4], &[1]), vec![3, 1, 4]);
        assert_eq!(expand_dims_shape(&[3, 4], &[2]), vec![3, 4, 1]);
    }

    #[test]
    fn test_expand_dims_multi_axis_output_positions() {
        // Core ML semantics: axes=[1, 2] means insert 1s at output positions 1 and 2
        assert_eq!(expand_dims_shape(&[3, 4], &[1, 2]), vec![3, 1, 1, 4]);
    }

    #[test]
    fn test_expand_dims_multi_axis_front_and_mid() {
        // axes=[0, 2] on [3, 4] → insert at output pos 0 and 2
        // [3,4] → insert at 0 → [1,3,4] → insert at 2 → [1,3,1,4]
        assert_eq!(expand_dims_shape(&[3, 4], &[0, 2]), vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_multi_axis_front() {
        // axes=[0, 0] on [3, 4] → [1, 1, 3, 4]
        assert_eq!(expand_dims_shape(&[3, 4], &[0, 0]), vec![1, 1, 3, 4]);
    }

    #[test]
    fn test_expand_dims_unsorted_axes() {
        // Axes should be sorted internally: [2, 0] sorted → [0, 2]
        // Insert at 0 → [1, 3, 4], insert at 2 → [1, 3, 1, 4]
        assert_eq!(expand_dims_shape(&[3, 4], &[2, 0]), vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_empty_input() {
        assert_eq!(expand_dims_shape(&[], &[0]), vec![1]);
    }

    // ─── squeeze_shape ──────────────────────────────────────────────

    #[test]
    fn test_squeeze_basic() {
        assert_eq!(squeeze_shape(&[1, 3, 1, 4], &[0, 2]), vec![3, 4]);
    }

    #[test]
    fn test_squeeze_single() {
        assert_eq!(squeeze_shape(&[1, 3, 4], &[0]), vec![3, 4]);
    }

    #[test]
    fn test_squeeze_out_of_range() {
        // Axis beyond shape length is ignored
        assert_eq!(squeeze_shape(&[3, 4], &[5]), vec![3, 4]);
    }

    // ─── transpose_shape ────────────────────────────────────────────

    #[test]
    fn test_transpose_basic() {
        assert_eq!(
            transpose_shape(&[1, 512, 8, 128], &[0, 2, 1, 3]),
            vec![1, 8, 512, 128]
        );
    }

    // ─── resolve_reshape_zeros ──────────────────────────────────────

    #[test]
    fn test_reshape_no_zeros() {
        assert_eq!(
            resolve_reshape_zeros(&[1, 512, 1024], &[1, 512, 1024]).unwrap(),
            vec![1, 512, 1024]
        );
    }

    #[test]
    fn test_reshape_single_zero() {
        assert_eq!(
            resolve_reshape_zeros(&[1, 512, 1024], &[1, 0, 1024]).unwrap(),
            vec![1, 512, 1024]
        );
    }

    #[test]
    fn test_reshape_positional() {
        // Positional resolution: 0 at pos 1 gets input_shape[1]=512
        assert_eq!(
            resolve_reshape_zeros(&[1, 512, 1024], &[1, 0, 1024]).unwrap(),
            vec![1, 512, 1024]
        );
    }

    #[test]
    fn test_reshape_element_count_fallback() {
        // [2,3,4] = 24 elements → [0, 12] → element-count: 24/12 = 2
        assert_eq!(
            resolve_reshape_zeros(&[2, 3, 4], &[0, 12]).unwrap(),
            vec![2, 12]
        );
    }

    #[test]
    fn test_reshape_two_zeros() {
        // [2,3,4] = 24 elements → [0, 0, 12] → batch=1, 24/12=2 → [1, 2, 12]
        assert_eq!(
            resolve_reshape_zeros(&[2, 3, 4], &[0, 0, 12]).unwrap(),
            vec![1, 2, 12]
        );
    }

    #[test]
    fn test_reshape_zero_input() {
        // Input with zero elements: return as-is
        assert_eq!(
            resolve_reshape_zeros(&[0, 3, 4], &[0, 12]).unwrap(),
            vec![0, 12]
        );
    }

    // ─── pad_shape ──────────────────────────────────────────────────

    #[test]
    fn test_pad_no_padding() {
        assert_eq!(pad_shape(&[1, 512, 64], &[0, 0, 0, 0, 0, 0]), vec![1, 512, 64]);
    }

    #[test]
    fn test_pad_with_padding() {
        // pad_amounts: [before_0, before_1, before_2, after_0, after_1, after_2]
        // Add 1 before-padding on axis 1 → 512+1=513
        assert_eq!(pad_shape(&[1, 512, 64], &[0, 1, 0, 0, 0, 0]), vec![1, 513, 64]);
    }

    // ─── concat_shape ──────────────────────────────────────────────

    #[test]
    fn test_concat_basic() {
        assert_eq!(
            concat_shape(&[&[1, 512, 64], &[1, 512, 64]], 2),
            Some(vec![1, 512, 128])
        );
    }

    #[test]
    fn test_concat_empty() {
        assert_eq!(concat_shape(&[], 0), None);
    }

    // ─── split_shape ──────────────────────────────────────────────

    #[test]
    fn test_split_basic() {
        assert_eq!(split_shape(&[1, 512, 128], 1, 2), vec![1, 256, 128]);
    }

    // ─── stack_shape ──────────────────────────────────────────────

    #[test]
    fn test_stack_basic() {
        assert_eq!(stack_shape(&[3, 4], 0, 2), vec![2, 3, 4]);
    }

    // ─── gather_shape ──────────────────────────────────────────────

    #[test]
    fn test_gather_embedding() {
        assert_eq!(gather_shape(&[32000, 1024], &[1, 512], 0), vec![1, 512, 1024]);
    }

    // ─── topk_shape ──────────────────────────────────────────────

    #[test]
    fn test_topk_basic() {
        assert_eq!(topk_shape(&[1, 512, 1024], 5, 2), vec![1, 512, 5]);
    }

    // ─── format_shape ──────────────────────────────────────────────

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[1, 512, 2048]), "[1, 512, 2048]");
    }

    #[test]
    fn test_format_shape_empty() {
        assert_eq!(format_shape(&[]), "[]");
    }
}
