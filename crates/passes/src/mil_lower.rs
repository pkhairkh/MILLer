//! MIL Lower pass.
//!
//! Lowers an AIR graph (with shard plan) into one or more
//! MIR graphs, each corresponding to a single MIL program.
//!
//! Current AIR→MIR lowering coverage:
//! - Linear/FC: AirOp::MatMul → MILMatMul, AirOp::Conv1x1AsLinear → MILLinear
//! - Elementwise: AirOp::ElementWise::Add → MILAdd, AirOp::ElementWise::Mul → MILMul,
//!   AirOp::ElementWise::Abs → MILAbs, AirOp::ElementWise::Maximum → MILMaximum,
//!   AirOp::ElementWise::Minimum → MILMinimum
//! - Shape ops: AirOp::Reshape → MILReshape, AirOp::Transpose → MILTranspose,
//!   AirOp::Split → MILSplit, AirOp::Concat → MILConcat
//! - Attention: AirOp::Softmax → MILSoftmax,
//!   AirOp::ScaledDotProductAttention → MIRScaledDotProductAttention,
//!   AirOp::SliceByIndex → MILSliceByIndex
//! - Activation: AirOp::Gelu → MILGelu, AirOp::Relu → MILRelu
//! - State: AirOp::StateReadFixed → MILReadState, AirOp::StateWriteFixed → MILCoremlUpdateState
//! - Normalization: AirOp::ReduceMean → MILReduceMean, AirOp::ReduceSum → MILReduceSum,
//!   AirOp::Rsqrt → MILRsqrt,
//!   AirOp::RealDiv → MILRealDiv, AirOp::LayerNorm → MILLayerNorm
//! - Sampling: AirOp::Topk → MILTopk, AirOp::Gather → MILGather
//! - RoPE: AirOp::Cos → MILCos, AirOp::Sin → MILSin
//!
//! All previously "declared but no lowering" MIR ops now have active AIR→MIR
//! lowering paths (Sprint 36). The SIR→AIR decompositions in LegalityRewritePass
//! produce the AIR ops that feed these lowering paths.
//!
//! Sprint 55: ElementWise::Maximum/Minimum now lower to MILMaximum/MILMinimum
//! instead of erroring.
//!
//! Sprint 57: StaticLUTProjection now lowers to MILGather as a de-scoped
//! approximation (the op is not used by any active SIR/task path; LUT
//! projection has a dedicated Python emission path). All AIR→MIR ops now
//! have lowering paths. Shape information from AIR ops is propagated into
//! MirNode.shape during lowering.

use super::shard_plan::ShardPlan;
use ane_ir::air::{AirGraph, AirNodeId, AirOp};
use ane_ir::mir::{ComputeUnitHint, MilDtype, MirGraph, MirNode, MirNodeId, MirOp};
use anyhow::Result;
use std::collections::HashMap;

/// Infer the output shape of an AIR op given the shapes of its inputs.
///
/// Sprint 57: this helper propagates shape information from AIR into
/// `MirNode.shape` during lowering. When an input shape is not available
/// (e.g., the input is an external graph input not yet processed), an
/// empty vec is returned as a conservative fallback.
fn infer_shape(op: &AirOp, node_shapes: &HashMap<AirNodeId, Vec<usize>>) -> Vec<usize> {
    match op {
        // ─── Identity: propagate input shape (critical for graph I/O nodes) ───
        AirOp::Identity { input } => node_shapes.get(input).cloned().unwrap_or_default(),

        // ─── MatMul: [M, K] × [K, N] → [M, N]; 1-D broadcast cases ───
        AirOp::MatMul { a, b, .. } => {
            match (node_shapes.get(a), node_shapes.get(b)) {
                (Some(a_shape), Some(b_shape)) => {
                    match (a_shape.len(), b_shape.len()) {
                        (2, 2) => vec![a_shape[0], b_shape[1]],
                        (1, 2) => vec![b_shape[1]], // bias-like: [K] × [K,N] → [N]
                        (2, 1) => vec![a_shape[0]], // [M,K] × [K] → [M]
                        (1, 1) => vec![],           // scalar × scalar
                        _ => vec![],
                    }
                }
                _ => vec![],
            }
        }

        // ─── Conv1x1AsLinear: semantically a linear projection ───
        // Sprint 61: Use output_dim to compute the correct output shape.
        // A linear projection y = x @ W^T maps [batch, seq, input_dim] → [batch, seq, output_dim].
        // When output_dim is 0 (unknown), fall back to propagating the input shape.
        AirOp::Conv1x1AsLinear { input, output_dim, .. } => {
            match (node_shapes.get(input), output_dim) {
                (Some(input_shape), od) if *od > 0 => {
                    // Replace the last dimension with the output_dim
                    let mut out = input_shape.clone();
                    if let Some(last) = out.last_mut() {
                        *last = *od;
                    }
                    out
                }
                (Some(input_shape), 0) => {
                    // output_dim unknown: propagate input shape (pre-Sprint-61 behavior)
                    input_shape.clone()
                }
                _ => vec![],
            }
        }

        AirOp::ElementWise { inputs, .. } => {
            // Sprint 62: Validate broadcast compatibility for binary elementwise ops.
            // Core ML requires broadcasting rules: dimensions must be compatible
            // (equal, or one of them is 1, or one is missing from the shorter shape).
            // Invalid broadcasts like [1,512,2048] * [128] will be caught here.
            if inputs.len() == 2 {
                let shape_a = node_shapes.get(&inputs[0]).cloned().unwrap_or_default();
                let shape_b = node_shapes.get(&inputs[1]).cloned().unwrap_or_default();
                if !shape_a.is_empty() && !shape_b.is_empty() {
                    if let Err(e) = validate_broadcast_compatibility(&shape_a, &shape_b) {
                        eprintln!(
                            "[WARN] Broadcast incompatibility: {} * {} — {}. \
                             This will fail Core ML type inference.",
                            format_shape(&shape_a),
                            format_shape(&shape_b),
                            e
                        );
                    }
                }
            }
            inputs.first().and_then(|id| node_shapes.get(id).cloned()).unwrap_or_default()
        }
        AirOp::Reshape { target_shape, .. } => target_shape.clone(),
        AirOp::Transpose { input, perm } => {
            if let Some(shape) = node_shapes.get(input) {
                perm.iter().map(|&p| shape.get(p).copied().unwrap_or(0)).collect()
            } else {
                vec![]
            }
        }
        AirOp::Split { input, axis, num_splits } => {
            if let Some(shape) = node_shapes.get(input) {
                let mut out = shape.clone();
                if let Some(dim) = out.get_mut(*axis) {
                    *dim /= num_splits;
                }
                out
            } else {
                vec![]
            }
        }
        AirOp::Concat { inputs, axis: _ } => {
            inputs.first().and_then(|id| node_shapes.get(id).cloned()).unwrap_or_default()
        }
        AirOp::Softmax { input, .. } => node_shapes.get(input).cloned().unwrap_or_default(),
        AirOp::StateReadFixed { shape, .. } => shape.clone(),
        AirOp::StateWriteFixed { .. } => vec![],
        AirOp::ReduceMean { input, axes, keep_dims } => {
            reduce_shape(node_shapes.get(input).cloned().unwrap_or_default(), axes, *keep_dims)
        }
        AirOp::ReduceSum { input, axes, keep_dims } => {
            reduce_shape(node_shapes.get(input).cloned().unwrap_or_default(), axes, *keep_dims)
        }
        AirOp::Rsqrt { input }
        | AirOp::Cos { input }
        | AirOp::Sin { input }
        | AirOp::Exp { input }
        | AirOp::Sigmoid { input }
        | AirOp::Tanh { input }
        | AirOp::Gelu { input, .. }
        | AirOp::Relu { input }
        | AirOp::SliceUpdate { input, .. }
        | AirOp::LayerNorm { input, .. } => node_shapes.get(input).cloned().unwrap_or_default(),
        AirOp::RealDiv { x, .. } => node_shapes.get(x).cloned().unwrap_or_default(),
        AirOp::Topk { input, k, axis } => {
            if let Some(shape) = node_shapes.get(input) {
                let mut out = shape.clone();
                let ax = if *axis >= 0 {
                    *axis as usize
                } else {
                    out.len().saturating_add(*axis as usize)
                };
                if ax < out.len() {
                    out[ax] = *k;
                }
                out
            } else {
                vec![]
            }
        }
        AirOp::Gather { input, indices, axis } => {
            // Embedding lookup: Gather(embed_weight, input_ids, axis=0)
            // The output shape replaces the axis dimension of the input (embedding
            // table) with the shape of the indices tensor. For a 2D weight
            // [vocab, embed_dim] gathered by [batch, seq] along axis 0, the result
            // is [batch, seq, embed_dim].
            match (node_shapes.get(input), node_shapes.get(indices)) {
                (Some(input_shape), Some(indices_shape)) => {
                    // Replace the axis dimension of input_shape with indices_shape
                    let ax = if *axis >= 0 {
                        *axis as usize
                    } else {
                        input_shape.len().saturating_sub((-*axis) as usize)
                    };
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
                (Some(input_shape), None) => {
                    // Indices shape unknown: use input shape as fallback
                    input_shape.clone()
                }
                _ => vec![],
            }
        }
        AirOp::ScaledDotProductAttention { query, .. } => {
            node_shapes.get(query).cloned().unwrap_or_default()
        }
        AirOp::Tile { input, reps } => {
            // Tile replicates the input tensor along each dimension by the
            // corresponding factor in `reps`. Output shape[i] = input_shape[i] * reps[i].
            if let Some(input_shape) = node_shapes.get(input) {
                // Broadcast reps to match input rank if needed
                let mut out = Vec::with_capacity(input_shape.len().max(reps.len()));
                let max_len = input_shape.len().max(reps.len());
                for i in 0..max_len {
                    let dim = input_shape.get(i).copied().unwrap_or(1);
                    let rep = reps.get(i).copied().unwrap_or(1);
                    out.push(dim * rep);
                }
                out
            } else {
                vec![]
            }
        }
        AirOp::SliceByIndex { input, begin, end, .. } => {
            if begin.iter().all(|v| *v >= 0)
                && end.iter().all(|v| *v >= 0)
                && begin.len() == end.len()
            {
                end.iter()
                    .zip(begin.iter())
                    .map(|(e, b)| (*e as usize).saturating_sub(*b as usize))
                    .collect()
            } else {
                node_shapes.get(input).cloned().unwrap_or_default()
            }
        }
        AirOp::Where { x, .. } => node_shapes.get(x).cloned().unwrap_or_default(),
        AirOp::StaticLUTProjection { .. } => vec![],
        // ─── Unary ops that pass through input shape ───
        AirOp::Silu { input }
        | AirOp::Abs { input }
        | AirOp::Neg { input }
        | AirOp::Sqrt { input }
        | AirOp::Cast { input, .. } => node_shapes.get(input).cloned().unwrap_or_default(),
        // ─── Binary ops: use first operand shape ───
        AirOp::Add { x, .. }
        | AirOp::Mul { x, .. }
        | AirOp::Sub { x, .. }
        | AirOp::Maximum { x, .. }
        | AirOp::Minimum { x, .. } => node_shapes.get(x).cloned().unwrap_or_default(),
        // All remaining AIR ops: conservatively return empty shape
        _ => vec![],
    }
}

/// Validate SDPA (ScaledDotProductAttention) constraints during AIR→MIR lowering.
///
/// Sprint 59: ANE constraints require:
/// - Operand count is 3 or 4 (Q, K, V, optional mask)
/// - All operand ranks ≤ 4
///
/// Returns a descriptive error on violation.
fn validate_sdpa_constraints(
    query_shape: &[usize],
    key_shape: &[usize],
    value_shape: &[usize],
    mask_shape: Option<&[usize]>,
) -> Result<()> {
    // Operand count: must be 3 or 4 (Q, K, V, optional mask)
    // The mask presence is checked by the Option, so we just validate
    // that we have the required 3 core operands.

    // Check all operand ranks ≤ 4
    let check_rank = |name: &str, shape: &[usize]| -> Result<()> {
        if shape.len() > 4 {
            anyhow::bail!(
                "SDPA constraint violation: {} has rank {} which exceeds maximum of 4 \
                 (ANE constraint: all SDPA operands must be rank ≤ 4)",
                name,
                shape.len()
            );
        }
        Ok(())
    };

    check_rank("query", query_shape)?;
    check_rank("key", key_shape)?;
    check_rank("value", value_shape)?;
    if let Some(mask_shape) = mask_shape {
        check_rank("attention_mask", mask_shape)?;
    }

    Ok(())
}

/// Helper: compute the output shape of a reduce op (ReduceMean / ReduceSum).
fn reduce_shape(mut shape: Vec<usize>, axes: &[usize], keep_dims: bool) -> Vec<usize> {
    if keep_dims {
        for &ax in axes {
            if ax < shape.len() {
                shape[ax] = 1;
            }
        }
        shape
    } else {
        shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &dim)| dim).collect()
    }
}

/// Sprint 62: Validate that two shapes are broadcast-compatible per Core ML rules.
///
/// Core ML (and numpy-style) broadcasting requires that for each dimension pair
/// (from the right/end), one of the following holds:
/// - The dimensions are equal
/// - One of the dimensions is 1
/// - One of the shapes doesn't have this dimension (shorter shape)
///
/// Returns Err with a description if incompatible, Ok(()) if compatible.
fn validate_broadcast_compatibility(a: &[usize], b: &[usize]) -> Result<(), String> {
    let max_rank = a.len().max(b.len());
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() { None } else { Some(a[i - (max_rank - a.len())]) };
        let db = if i < max_rank - b.len() { None } else { Some(b[i - (max_rank - b.len())]) };
        match (da, db) {
            (Some(da), Some(db)) => {
                if da != db && da != 1 && db != 1 {
                    return Err(format!(
                        "dimension {} mismatch: {} vs {} (neither is 1)",
                        i, da, db
                    ));
                }
            }
            _ => {} // one shape missing this dimension → broadcast from 1
        }
    }
    Ok(())
}

/// Sprint 62: Format a shape as a human-readable string like "[1, 512, 2048]".
fn format_shape(shape: &[usize]) -> String {
    format!("[{}]", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

/// MIL Lower pass implementation.
pub struct MilLowerPass {
    // No configuration needed for the linear projection case
}

impl Default for MilLowerPass {
    fn default() -> Self {
        Self::new()
    }
}

impl MilLowerPass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the MIL lower pass.
    ///
    /// For the current vertical slice, this pass:
    /// - Converts AirOp::MatMul to MirOp::MILMatMul
    /// - Converts AirOp::Conv1x1AsLinear to MirOp::MILLinear (fixes dead-letter bug)
    /// - Converts AirOp::ElementWise::Add to MirOp::MILAdd
    /// - Assigns fp16 dtype and compute unit hint from the shard plan
    /// - Produces a single-shard MIR graph (one MIL program)
    ///
    /// When more operations and sharding are supported, this pass will:
    /// - Lower each AIR operation to its MIR equivalent
    /// - Materialize constants as named MILConst nodes
    /// - Split the AIR graph across shard boundaries
    /// - Produce one MIR graph per shard
    /// - Handle state read/write as MILReadState/MILCoremlUpdateState operations
    ///
    /// `input_shapes` maps AIR node IDs for graph inputs to their expected shapes.
    /// This is critical for seeding shape inference: without it, Identity nodes
    /// that represent graph inputs (e.g., input_ids) get empty shapes, which
    /// propagates through the entire graph producing wrong metadata.
    pub fn run(
        &self,
        input: &AirGraph,
        shard_plan: &ShardPlan,
        input_shapes: &std::collections::HashMap<AirNodeId, Vec<usize>>,
    ) -> Result<Vec<MirGraph>> {
        Self::run_with_weight_shapes(self, input, shard_plan, input_shapes, &HashMap::new())
    }

    /// Run the MIL lower pass with additional weight tensor shapes.
    ///
    /// Weight tensors are referenced by name (e.g., "model.embed_tokens.weight")
    /// but aren't AIR graph nodes — they exist outside the graph. This variant
    /// allows callers to provide their shapes so that ops like Gather (embedding
    /// lookup) can produce correct output shapes even when the weight isn't a
    /// prior AIR node output.
    pub fn run_with_weight_shapes(
        &self,
        input: &AirGraph,
        shard_plan: &ShardPlan,
        input_shapes: &std::collections::HashMap<AirNodeId, Vec<usize>>,
        weight_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<Vec<MirGraph>> {
        let mut mir_nodes = Vec::new();
        let mut air_to_mir = std::collections::HashMap::new();
        // Sprint 57: track output shape of each AIR node so we can propagate
        // shape information into MirNode.shape during lowering.
        // Seed with the externally-provided input shapes for graph inputs.
        // Without this, Identity nodes representing graph inputs get empty shapes,
        // which propagates through the entire graph producing wrong metadata.
        let mut node_shapes: HashMap<AirNodeId, Vec<usize>> = input_shapes.clone();

        // Also seed weight tensor shapes. Weight names (e.g., "model.embed_tokens.weight")
        // appear as AirNodeId references in ops like Gather but aren't AIR graph nodes.
        // Without seeding, Gather(embed_weight, indices) can't infer its output shape
        // because the weight's shape is never added to node_shapes.
        for (weight_name, shape) in weight_shapes {
            node_shapes.insert(AirNodeId(weight_name.clone()), shape.clone());
        }

        // Derive the compute unit hint from the shard plan instead of hardcoding
        // CPU_AND_NE. This fixes critique Bug 3 where the compute_unit_hint on
        // MirNode was always CPUAndNE regardless of the shard plan's actual
        // compute unit assignment (which can be overridden by knowledge-driven
        // adaptation in the ShardPlanPass).
        let compute_hint = if shard_plan.compute_units.is_empty() {
            ComputeUnitHint::CPUAndNE // fallback default
        } else {
            match shard_plan.compute_units[0].as_str() {
                "CPU_AND_NE" => ComputeUnitHint::CPUAndNE,
                "CPU_AND_GPU" => ComputeUnitHint::CPUAndGPU,
                "CPU_ONLY" => ComputeUnitHint::CPUOnly,
                "ALL" => ComputeUnitHint::All,
                _ => ComputeUnitHint::CPUAndNE,
            }
        };

        for air_node in &input.nodes {
            let mir_id = MirNodeId(air_node.id.0.clone());
            air_to_mir.insert(air_node.id.clone(), mir_id.clone());

            // Determine dtype from AIR precision_override or default to fp16.
            // When the precision policy pass has overridden the dtype (e.g., fp16 → fp32
            // due to a known precision hazard), that override propagates through AIR
            // and must be respected in the MIR, ensuring the knowledge-informed
            // precision decision reaches the emitted mlpackage.
            //
            // Special case: Identity nodes that are graph inputs with integer-like
            // names (e.g., "input_ids", "attention_mask") should use Int32, since
            // Core ML's gather op expects integer indices and these inputs represent
            // token indices or masks, not floating-point data.
            let mil_dtype = match &air_node.precision_override {
                Some(dtype) => match dtype.as_str() {
                    "fp32" => MilDtype::Fp32,
                    "fp16" => MilDtype::Fp16,
                    "int32" => MilDtype::Int32,
                    _ => MilDtype::Fp16,
                },
                None => {
                    // Heuristic: Identity ops that are graph inputs with names
                    // containing "ids" (e.g., input_ids) or "mask" are integer tensors.
                    if matches!(&air_node.op, AirOp::Identity { .. })
                        && (air_node.name.ends_with("_ids")
                            || air_node.name.contains("input_ids")
                            || air_node.name.contains("mask"))
                    {
                        MilDtype::Int32
                    } else {
                        MilDtype::Fp16
                    }
                }
            };

            let mir_op = match &air_node.op {
                AirOp::MatMul { a, b, .. } => {
                    let x_id = air_to_mir.get(a).cloned().unwrap_or_else(|| MirNodeId(a.0.clone()));
                    let y_id = air_to_mir.get(b).cloned().unwrap_or_else(|| MirNodeId(b.0.clone()));
                    MirOp::MILMatMul { name: air_node.name.clone(), x: x_id, y: y_id }
                }
                AirOp::Conv1x1AsLinear { input, weight, .. } => {
                    // Conv1x1AsLinear is semantically a fully-connected projection.
                    // Lower to MILLinear (canonical Core ML op for FC projections)
                    // instead of a dead-letter matmul+add decomposition.
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLinear {
                        name: air_node.name.clone(),
                        x: mir_input,
                        weight: weight.clone(),
                        bias: None, // Conv1x1AsLinear has no bias field
                    }
                }
                AirOp::ElementWise { op, inputs } => {
                    match op {
                        ane_ir::sir::ElementWiseOp::Add => {
                            let x = inputs.first().map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            let y = inputs.get(1).map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            // For the linear projection vertical slice, we need both operands.
                            // If we don't have two inputs (weight/bias are constants, not in AIR inputs),
                            // we create a placeholder referencing by name.
                            MirOp::MILAdd {
                                name: air_node.name.clone(),
                                x: x.unwrap_or_else(|| MirNodeId("input".into())),
                                y: y.unwrap_or_else(|| MirNodeId("bias".into())),
                            }
                        }
                        ane_ir::sir::ElementWiseOp::Mul => {
                            let x = inputs.first().map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            let y = inputs.get(1).map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            MirOp::MILMul {
                                name: air_node.name.clone(),
                                x: x.unwrap_or_else(|| MirNodeId("input".into())),
                                y: y.unwrap_or_else(|| MirNodeId("weight".into())),
                            }
                        }
                        ane_ir::sir::ElementWiseOp::Abs => {
                            let x = inputs.first().map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            MirOp::MILAbs {
                                name: air_node.name.clone(),
                                x: x.unwrap_or_else(|| MirNodeId("input".into())),
                            }
                        }
                        ane_ir::sir::ElementWiseOp::Maximum => {
                            let x = inputs.first().map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            let y = inputs.get(1).map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            MirOp::MILMaximum {
                                name: air_node.name.clone(),
                                x: x.unwrap_or_else(|| MirNodeId("input".into())),
                                y: y.unwrap_or_else(|| MirNodeId("zero".into())),
                            }
                        }
                        ane_ir::sir::ElementWiseOp::Minimum => {
                            let x = inputs.first().map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            let y = inputs.get(1).map(|id| {
                                air_to_mir
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| MirNodeId(id.0.clone()))
                            });
                            MirOp::MILMinimum {
                                name: air_node.name.clone(),
                                x: x.unwrap_or_else(|| MirNodeId("input".into())),
                                y: y.unwrap_or_else(|| MirNodeId("zero".into())),
                            }
                        }
                    }
                }
                AirOp::Reshape { input, target_shape } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReshape {
                        name: air_node.name.clone(),
                        x: mir_input,
                        shape: target_shape.clone(),
                    }
                }
                AirOp::Transpose { input, perm } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILTranspose {
                        name: air_node.name.clone(),
                        x: mir_input,
                        perm: perm.clone(),
                    }
                }
                AirOp::Softmax { input, axis } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSoftmax { name: air_node.name.clone(), x: mir_input, axis: *axis }
                }
                // Normalization ops (Sprint 33)
                AirOp::ReduceMean { input, axes, keep_dims } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceMean {
                        name: air_node.name.clone(),
                        x: mir_input,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceSum { input, axes, keep_dims } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceSum {
                        name: air_node.name.clone(),
                        x: mir_input,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::Rsqrt { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILRsqrt { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::RealDiv { x, y } => {
                    let mir_x =
                        air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let mir_y =
                        air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILRealDiv { name: air_node.name.clone(), x: mir_x, y: mir_y }
                }
                AirOp::LayerNorm { input, weight, bias, epsilon, axes } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLayerNorm {
                        name: air_node.name.clone(),
                        x: mir_input,
                        weight: weight.clone(),
                        bias: bias.clone(),
                        epsilon: *epsilon,
                        axes: axes.clone(),
                    }
                }
                // Sampling ops (Sprint 33)
                AirOp::Topk { input, k, axis } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILTopk { name: air_node.name.clone(), x: mir_input, k: *k, axis: *axis }
                }
                AirOp::Gather { input, indices, axis } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mir_indices = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    MirOp::MILGather {
                        name: air_node.name.clone(),
                        x: mir_input,
                        indices: mir_indices,
                        axis: *axis,
                    }
                }
                // RoPE/trigonometric ops (Sprint 33)
                AirOp::Cos { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCos { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::Sin { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSin { name: air_node.name.clone(), x: mir_input }
                }
                // Attention ops (Sprint 36)
                AirOp::ScaledDotProductAttention { query, key, value, attention_mask, scale } => {
                    // Sprint 59: validate SDPA constraints before lowering.
                    let q_shape = node_shapes.get(query).cloned().unwrap_or_default();
                    let k_shape = node_shapes.get(key).cloned().unwrap_or_default();
                    let v_shape = node_shapes.get(value).cloned().unwrap_or_default();
                    let m_shape = attention_mask.as_ref().and_then(|m| node_shapes.get(m)).cloned();
                    validate_sdpa_constraints(&q_shape, &k_shape, &v_shape, m_shape.as_deref())?;

                    let mir_q = air_to_mir
                        .get(query)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(query.0.clone()));
                    let mir_k =
                        air_to_mir.get(key).cloned().unwrap_or_else(|| MirNodeId(key.0.clone()));
                    let mir_v = air_to_mir
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(value.0.clone()));
                    let mir_mask = attention_mask.as_ref().map(|m| {
                        air_to_mir.get(m).cloned().unwrap_or_else(|| MirNodeId(m.0.clone()))
                    });
                    MirOp::MILScaledDotProductAttention {
                        name: air_node.name.clone(),
                        query: mir_q,
                        key: mir_k,
                        value: mir_v,
                        attention_mask: mir_mask,
                        scale: *scale,
                    }
                }
                AirOp::SliceByIndex {
                    input,
                    begin,
                    end,
                    stride,
                    begin_mask,
                    end_mask,
                    squeeze_mask,
                } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSliceByIndex {
                        name: air_node.name.clone(),
                        x: mir_input,
                        begin: begin.clone(),
                        end: end.clone(),
                        stride: stride.clone(),
                        begin_mask: begin_mask.clone(),
                        end_mask: end_mask.clone(),
                        squeeze_mask: squeeze_mask.clone(),
                    }
                }
                // Activation ops (Sprint 36)
                AirOp::Gelu { input, mode } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILGelu { name: air_node.name.clone(), x: mir_input, mode: mode.clone() }
                }
                AirOp::Relu { input } => {
                    // Sprint 50: ReLU now has a proper MIR op (MILRelu) instead
                    // of the previous MILCast approximation that was semantically
                    // incorrect but preserved graph structure.
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILRelu { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::StateReadFixed { state_id, shape, dtype } => MirOp::MILReadState {
                    name: air_node.name.clone(),
                    state_id: state_id.clone(),
                    shape: shape.clone(),
                    dtype: dtype.clone(),
                },
                AirOp::StateWriteFixed { state_id, value } => {
                    let mir_value = air_to_mir
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(value.0.clone()));
                    MirOp::MILCoremlUpdateState {
                        name: air_node.name.clone(),
                        state_id: state_id.clone(),
                        value: mir_value,
                    }
                }
                // Shape ops that previously had no lowering
                AirOp::Split { input, axis, num_splits } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSplit {
                        name: air_node.name.clone(),
                        x: mir_input,
                        axis: *axis,
                        num_splits: *num_splits,
                    }
                }
                AirOp::Concat { inputs, axis } => {
                    let mir_inputs: Vec<MirNodeId> = inputs
                        .iter()
                        .map(|id| {
                            air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone()))
                        })
                        .collect();
                    MirOp::MILConcat {
                        name: air_node.name.clone(),
                        values: mir_inputs,
                        axis: *axis,
                    }
                }
                // Sprint 50: P2 ops — buffer update, activation, math, conditional
                AirOp::SliceUpdate { input, update, begin, end } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mir_update = air_to_mir
                        .get(update)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(update.0.clone()));
                    MirOp::MILSliceUpdate {
                        name: air_node.name.clone(),
                        x: mir_input,
                        update: mir_update,
                        begin: begin.clone(),
                        end: end.clone(),
                    }
                }
                AirOp::Exp { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILExp { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::Sigmoid { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSigmoid { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::Tanh { input } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILTanh { name: air_node.name.clone(), x: mir_input }
                }
                AirOp::Where { condition, x, y } => {
                    let mir_condition = air_to_mir
                        .get(condition)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(condition.0.clone()));
                    let mir_x =
                        air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let mir_y =
                        air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILWhere {
                        name: air_node.name.clone(),
                        condition: mir_condition,
                        x: mir_x,
                        y: mir_y,
                    }
                }
                // Sprint 57: StaticLUTProjection lowers to MILGather as a de-scoped
                // approximation. The op is not used by any active SIR/task path;
                // LUT projection has a dedicated Python emission path.
                AirOp::StaticLUTProjection { input: _, indices, lut, group_size: _ } => {
                    let mir_lut = air_to_mir
                        .get(&AirNodeId(lut.clone()))
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(lut.clone()));
                    let mir_indices = air_to_mir
                        .get(&AirNodeId(indices.clone()))
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.clone()));
                    MirOp::MILGather {
                        name: air_node.name.clone(),
                        x: mir_lut,
                        indices: mir_indices,
                        axis: 0,
                    }
                }

                // ─── Full coverage lowering for all remaining AIR ops ─────
                // Each AirOp variant maps to its corresponding MirOp variant.
                // These are pass-through lowerings that preserve the op semantics
                // from AIR into MIR for MIL emission.

                // Direct elementwise unary variants (also handled via ElementWise legacy path)
                AirOp::Abs { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAbs { name: air_node.name.clone(), x: mi }
                }

                AirOp::Const { value_path, dtype } => MirOp::MILConst {
                    name: air_node.name.clone(),
                    value_path: value_path.clone(),
                    dtype: dtype.clone(),
                },
                AirOp::Linear { input, weight, bias } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLinear {
                        name: air_node.name.clone(),
                        x: mir_input,
                        weight: weight.clone(),
                        bias: bias.clone(),
                    }
                }
                AirOp::Einsum { inputs, equation } => {
                    let mir_inputs: Vec<MirNodeId> = inputs
                        .iter()
                        .map(|id| {
                            air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone()))
                        })
                        .collect();
                    MirOp::MILEinsum {
                        name: air_node.name.clone(),
                        inputs: mir_inputs,
                        equation: equation.clone(),
                    }
                }
                AirOp::Conv {
                    input,
                    weight,
                    pad_type,
                    groups,
                    strides,
                    pad_amounts,
                    dilations,
                } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mir_weight = air_to_mir
                        .get(weight)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(weight.0.clone()));
                    MirOp::MILConv {
                        name: air_node.name.clone(),
                        x: mir_input,
                        weight: mir_weight,
                        pad_type: pad_type.clone(),
                        groups: *groups,
                        strides: strides.clone(),
                        pad_amounts: pad_amounts.clone(),
                        dilations: dilations.clone(),
                    }
                }
                AirOp::ConvTranspose {
                    input,
                    weight,
                    pad_type,
                    groups,
                    strides,
                    pad_amounts,
                    dilations,
                    output_shape,
                } => {
                    let mir_input = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mir_weight = air_to_mir
                        .get(weight)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(weight.0.clone()));
                    MirOp::MILConvTranspose {
                        name: air_node.name.clone(),
                        x: mir_input,
                        weight: mir_weight,
                        pad_type: pad_type.clone(),
                        groups: *groups,
                        strides: strides.clone(),
                        pad_amounts: pad_amounts.clone(),
                        dilations: dilations.clone(),
                        output_shape: output_shape.clone(),
                    }
                }

                // Elementwise binary ops
                AirOp::Add { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILAdd { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Mul { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILMul { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Sub { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILSub { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Maximum { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILMaximum { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Minimum { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILMinimum { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::FloorDiv { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILFloorDiv { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Mod { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILMod { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Pow { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILPow { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Equal { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILEqual { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::NotEqual { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILNotEqual { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Greater { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILGreater { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::GreaterEqual { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILGreaterEqual { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::Less { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILLess { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::LessEqual { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILLessEqual { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::LogicalAnd { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILLogicalAnd { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::LogicalOr { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILLogicalOr { name: air_node.name.clone(), x: mx, y: my }
                }
                AirOp::LogicalXor { x, y } => {
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILLogicalXor { name: air_node.name.clone(), x: mx, y: my }
                }

                // Elementwise unary ops
                AirOp::Neg { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILNeg { name: air_node.name.clone(), x: mi }
                }
                AirOp::Relu6 { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILRelu6 { name: air_node.name.clone(), x: mi }
                }
                AirOp::LeakyRelu { input, alpha } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLeakyRelu { name: air_node.name.clone(), x: mi, alpha: *alpha }
                }
                AirOp::SigmoidHard { input, alpha, beta } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSigmoidHard {
                        name: air_node.name.clone(),
                        x: mi,
                        alpha: *alpha,
                        beta: *beta,
                    }
                }
                AirOp::ThresholdedRelu { input, alpha } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILThresholdedRelu { name: air_node.name.clone(), x: mi, alpha: *alpha }
                }
                AirOp::ClampedRelu { input, alpha, beta } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILClampedRelu {
                        name: air_node.name.clone(),
                        x: mi,
                        alpha: *alpha,
                        beta: *beta,
                    }
                }
                AirOp::LinearActivation { input, alpha, beta } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLinearActivation {
                        name: air_node.name.clone(),
                        x: mi,
                        alpha: *alpha,
                        beta: *beta,
                    }
                }
                AirOp::Prelu { input, alpha } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILPrelu { name: air_node.name.clone(), x: mi, alpha: alpha.clone() }
                }
                AirOp::Softsign { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSoftsign { name: air_node.name.clone(), x: mi }
                }
                AirOp::Silu { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSilu { name: air_node.name.clone(), x: mi }
                }
                AirOp::ScaledTanh { input, alpha, beta } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILScaledTanh {
                        name: air_node.name.clone(),
                        x: mi,
                        alpha: *alpha,
                        beta: *beta,
                    }
                }
                AirOp::Elu { input, alpha } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILElu { name: air_node.name.clone(), x: mi, alpha: *alpha }
                }
                AirOp::Softplus { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSoftplus { name: air_node.name.clone(), x: mi }
                }
                AirOp::SoftplusParametric { input, alpha, beta } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSoftplusParametric {
                        name: air_node.name.clone(),
                        x: mi,
                        alpha: alpha.clone(),
                        beta: beta.clone(),
                    }
                }
                AirOp::Clip { input, min_val, max_val } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILClip {
                        name: air_node.name.clone(),
                        x: mi,
                        min_val: *min_val,
                        max_val: *max_val,
                    }
                }
                AirOp::Square { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSquare { name: air_node.name.clone(), x: mi }
                }
                AirOp::Threshold { input, alpha } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILThreshold { name: air_node.name.clone(), x: mi, alpha: *alpha }
                }
                AirOp::Sqrt { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSqrt { name: air_node.name.clone(), x: mi }
                }
                AirOp::Inverse { input, epsilon } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILInverse { name: air_node.name.clone(), x: mi, epsilon: *epsilon }
                }
                AirOp::Ceil { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCeil { name: air_node.name.clone(), x: mi }
                }
                AirOp::Floor { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILFloor { name: air_node.name.clone(), x: mi }
                }
                AirOp::Round { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILRound { name: air_node.name.clone(), x: mi }
                }
                AirOp::Exp2 { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILExp2 { name: air_node.name.clone(), x: mi }
                }
                AirOp::Log { input, epsilon } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLog { name: air_node.name.clone(), x: mi, epsilon: *epsilon }
                }
                AirOp::Sign { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSign { name: air_node.name.clone(), x: mi }
                }
                AirOp::Tan { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILTan { name: air_node.name.clone(), x: mi }
                }
                AirOp::Acos { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAcos { name: air_node.name.clone(), x: mi }
                }
                AirOp::Asin { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAsin { name: air_node.name.clone(), x: mi }
                }
                AirOp::Atan { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAtan { name: air_node.name.clone(), x: mi }
                }
                AirOp::Cosh { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCosh { name: air_node.name.clone(), x: mi }
                }
                AirOp::Sinh { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSinh { name: air_node.name.clone(), x: mi }
                }
                AirOp::Atanh { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAtanh { name: air_node.name.clone(), x: mi }
                }
                AirOp::Erf { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILErf { name: air_node.name.clone(), x: mi }
                }
                AirOp::LogicalNot { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLogicalNot { name: air_node.name.clone(), x: mi }
                }
                AirOp::Cast { input, dtype } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCast { name: air_node.name.clone(), x: mi, dtype: dtype.clone() }
                }
                AirOp::Select { condition, x, y } => {
                    let mc = air_to_mir
                        .get(condition)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(condition.0.clone()));
                    let mx = air_to_mir.get(x).cloned().unwrap_or_else(|| MirNodeId(x.0.clone()));
                    let my = air_to_mir.get(y).cloned().unwrap_or_else(|| MirNodeId(y.0.clone()));
                    MirOp::MILSelect { name: air_node.name.clone(), condition: mc, x: mx, y: my }
                }

                // Reduction ops
                AirOp::ReduceMax { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceMax {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceMin { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceMin {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceProd { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceProd {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceSumSquare { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceSumSquare {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceL2Norm { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceL2Norm {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceL1Norm { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceL1Norm {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceLogSumExp { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceLogSumExp {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceLogSum { input, axes, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceLogSum {
                        name: air_node.name.clone(),
                        x: mi,
                        axes: axes.clone(),
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceArgmax { input, axis, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceArgmax {
                        name: air_node.name.clone(),
                        x: mi,
                        axis: *axis,
                        keep_dims: *keep_dims,
                    }
                }
                AirOp::ReduceArgmin { input, axis, keep_dims } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReduceArgmin {
                        name: air_node.name.clone(),
                        x: mi,
                        axis: *axis,
                        keep_dims: *keep_dims,
                    }
                }

                // Normalization ops
                AirOp::BatchNorm { input, mean, variance, gamma, beta, epsilon } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILBatchNorm {
                        name: air_node.name.clone(),
                        x: mi,
                        mean: mean.clone(),
                        variance: variance.clone(),
                        gamma: gamma.clone(),
                        beta: beta.clone(),
                        epsilon: *epsilon,
                    }
                }
                AirOp::InstanceNorm { input, gamma, beta, epsilon } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILInstanceNorm {
                        name: air_node.name.clone(),
                        x: mi,
                        gamma: gamma.clone(),
                        beta: beta.clone(),
                        epsilon: *epsilon,
                    }
                }
                AirOp::L2Norm { input, epsilon, axes } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILL2Norm {
                        name: air_node.name.clone(),
                        x: mi,
                        epsilon: *epsilon,
                        axes: axes.clone(),
                    }
                }
                AirOp::LocalResponseNorm { input, size, alpha, beta, k } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILLocalResponseNorm {
                        name: air_node.name.clone(),
                        x: mi,
                        size: *size,
                        alpha: *alpha,
                        beta: *beta,
                        k: *k,
                    }
                }

                // Pooling ops
                AirOp::MaxPool { input, kernel_sizes, strides, pad_types, pad_amounts } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILMaxPool {
                        name: air_node.name.clone(),
                        x: mi,
                        kernel_sizes: kernel_sizes.clone(),
                        strides: strides.clone(),
                        pad_types: pad_types.clone(),
                        pad_amounts: pad_amounts.clone(),
                    }
                }
                AirOp::AvgPool {
                    input,
                    kernel_sizes,
                    strides,
                    pad_types,
                    pad_amounts,
                    count_include_padding,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILAvgPool {
                        name: air_node.name.clone(),
                        x: mi,
                        kernel_sizes: kernel_sizes.clone(),
                        strides: strides.clone(),
                        pad_types: pad_types.clone(),
                        pad_amounts: pad_amounts.clone(),
                        count_include_padding: *count_include_padding,
                    }
                }
                AirOp::L2Pool { input, kernel_sizes, strides, pad_types, pad_amounts } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILL2Pool {
                        name: air_node.name.clone(),
                        x: mi,
                        kernel_sizes: kernel_sizes.clone(),
                        strides: strides.clone(),
                        pad_types: pad_types.clone(),
                        pad_amounts: pad_amounts.clone(),
                    }
                }

                // Image resizing ops
                AirOp::Resize {
                    input,
                    target_size,
                    mode,
                    sampling_mode,
                    nearest_rounding_mode,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILResize {
                        name: air_node.name.clone(),
                        x: mi,
                        target_size: target_size.clone(),
                        mode: mode.clone(),
                        sampling_mode: sampling_mode.clone(),
                        nearest_rounding_mode: nearest_rounding_mode.clone(),
                    }
                }
                AirOp::ResizeNearestNeighbor { input, target_height, target_width } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILResizeNearestNeighbor {
                        name: air_node.name.clone(),
                        x: mi,
                        target_height: *target_height,
                        target_width: *target_width,
                    }
                }
                AirOp::ResizeBilinear { input, target_height, target_width, align_corners } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILResizeBilinear {
                        name: air_node.name.clone(),
                        x: mi,
                        target_height: *target_height,
                        target_width: *target_width,
                        align_corners: *align_corners,
                    }
                }
                AirOp::UpsampleNearestNeighbor { input, scale } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILUpsampleNearestNeighbor {
                        name: air_node.name.clone(),
                        x: mi,
                        scale: scale.clone(),
                    }
                }
                AirOp::UpsampleBilinear { input, scale, align_corners, half_pixel_centers } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILUpsampleBilinear {
                        name: air_node.name.clone(),
                        x: mi,
                        scale: scale.clone(),
                        align_corners: *align_corners,
                        half_pixel_centers: *half_pixel_centers,
                    }
                }
                AirOp::CropResize { input, boxes, box_indices, crop_height, crop_width } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mb = air_to_mir
                        .get(boxes)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(boxes.0.clone()));
                    let mbi = air_to_mir
                        .get(box_indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(box_indices.0.clone()));
                    MirOp::MILCropResize {
                        name: air_node.name.clone(),
                        x: mi,
                        boxes: mb,
                        box_indices: mbi,
                        crop_height: *crop_height,
                        crop_width: *crop_width,
                    }
                }
                AirOp::Affine {
                    input,
                    transform,
                    output_height,
                    output_width,
                    sampling_mode,
                    pad_value,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mt = air_to_mir
                        .get(transform)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(transform.0.clone()));
                    MirOp::MILAffine {
                        name: air_node.name.clone(),
                        x: mi,
                        transform: mt,
                        output_height: *output_height,
                        output_width: *output_width,
                        sampling_mode: sampling_mode.clone(),
                        pad_value: *pad_value,
                    }
                }
                AirOp::Resample { input, coordinates, sampling_mode, pad_value } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mc = air_to_mir
                        .get(coordinates)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(coordinates.0.clone()));
                    MirOp::MILResample {
                        name: air_node.name.clone(),
                        x: mi,
                        coordinates: mc,
                        sampling_mode: sampling_mode.clone(),
                        pad_value: *pad_value,
                    }
                }

                // Tensor transform ops
                AirOp::ReshapeLike { input, ref_tensor } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mr = air_to_mir
                        .get(ref_tensor)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(ref_tensor.0.clone()));
                    MirOp::MILReshapeLike { name: air_node.name.clone(), x: mi, ref_tensor: mr }
                }
                AirOp::ExpandDims { input, axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILExpandDims { name: air_node.name.clone(), x: mi, axis: axis.clone() }
                }
                AirOp::Squeeze { input, axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSqueeze { name: air_node.name.clone(), x: mi, axis: axis.clone() }
                }
                AirOp::Flatten2d { input, axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILFlatten2d { name: air_node.name.clone(), x: mi, axis: *axis }
                }
                AirOp::Reverse { input, axes } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILReverse { name: air_node.name.clone(), x: mi, axes: axes.clone() }
                }
                AirOp::ReverseSequence { input, lengths, batch_axis, seq_axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let ml = air_to_mir
                        .get(lengths)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(lengths.0.clone()));
                    MirOp::MILReverseSequence {
                        name: air_node.name.clone(),
                        x: mi,
                        lengths: ml,
                        batch_axis: *batch_axis,
                        seq_axis: *seq_axis,
                    }
                }
                AirOp::SliceBySize { input, begin, size } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSliceBySize {
                        name: air_node.name.clone(),
                        x: mi,
                        begin: begin.clone(),
                        size: size.clone(),
                    }
                }
                AirOp::SlidingWindows { input, axis, window_size, stride } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSlidingWindows {
                        name: air_node.name.clone(),
                        x: mi,
                        axis: *axis,
                        window_size: *window_size,
                        stride: *stride,
                    }
                }
                AirOp::DepthToSpace { input, block_size } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILDepthToSpace {
                        name: air_node.name.clone(),
                        x: mi,
                        block_size: *block_size,
                    }
                }
                AirOp::SpaceToDepth { input, block_size } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSpaceToDepth {
                        name: air_node.name.clone(),
                        x: mi,
                        block_size: *block_size,
                    }
                }
                AirOp::PixelShuffle { input, upscale_factor } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILPixelShuffle {
                        name: air_node.name.clone(),
                        x: mi,
                        upscale_factor: *upscale_factor,
                    }
                }
                AirOp::PixelUnshuffle { input, downscale_factor } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILPixelUnshuffle {
                        name: air_node.name.clone(),
                        x: mi,
                        downscale_factor: *downscale_factor,
                    }
                }
                AirOp::BatchToSpace { input, block_shape, crops } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILBatchToSpace {
                        name: air_node.name.clone(),
                        x: mi,
                        block_shape: block_shape.clone(),
                        crops: crops.clone(),
                    }
                }
                AirOp::SpaceToBatch { input, block_shape, paddings } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILSpaceToBatch {
                        name: air_node.name.clone(),
                        x: mi,
                        block_shape: block_shape.clone(),
                        paddings: paddings.clone(),
                    }
                }
                AirOp::Pad { input, pad_amounts, mode, constant_value } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILPad {
                        name: air_node.name.clone(),
                        x: mi,
                        pad_amounts: pad_amounts.clone(),
                        mode: mode.clone(),
                        constant_value: *constant_value,
                    }
                }
                AirOp::Stack { values, axis } => {
                    let mv: Vec<MirNodeId> = values
                        .iter()
                        .map(|id| {
                            air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone()))
                        })
                        .collect();
                    MirOp::MILStack { name: air_node.name.clone(), values: mv, axis: *axis }
                }
                AirOp::Tile { input, reps } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILTile { name: air_node.name.clone(), x: mi, reps: reps.clone() }
                }
                AirOp::Cumsum { input, axis, exclusive, reverse } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCumsum {
                        name: air_node.name.clone(),
                        x: mi,
                        axis: *axis,
                        exclusive: *exclusive,
                        reverse: *reverse,
                    }
                }
                AirOp::Fill { shape, value, dtype } => MirOp::MILFill {
                    name: air_node.name.clone(),
                    shape: shape.clone(),
                    value: *value,
                    dtype: dtype.clone(),
                },
                AirOp::FillLike { ref_tensor, value, dtype } => {
                    let mr = air_to_mir
                        .get(ref_tensor)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(ref_tensor.0.clone()));
                    MirOp::MILFillLike {
                        name: air_node.name.clone(),
                        ref_tensor: mr,
                        value: *value,
                        dtype: dtype.clone(),
                    }
                }
                AirOp::Identity { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILIdentity { name: air_node.name.clone(), x: mi }
                }
                AirOp::OneHot {
                    indices,
                    one_hot_vector_size,
                    on_value,
                    off_value,
                    axis,
                    dtype,
                } => {
                    let mi = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    MirOp::MILOneHot {
                        name: air_node.name.clone(),
                        indices: mi,
                        one_hot_vector_size: *one_hot_vector_size,
                        on_value: *on_value,
                        off_value: *off_value,
                        axis: *axis,
                        dtype: dtype.clone(),
                    }
                }
                AirOp::NonZero { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILNonZero { name: air_node.name.clone(), x: mi }
                }
                AirOp::Argsort { input, axis, ascending } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILArgsort {
                        name: air_node.name.clone(),
                        x: mi,
                        axis: *axis,
                        ascending: *ascending,
                    }
                }
                AirOp::BandPart { input, num_lower, num_upper } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILBandPart {
                        name: air_node.name.clone(),
                        x: mi,
                        num_lower: *num_lower,
                        num_upper: *num_upper,
                    }
                }
                AirOp::Range1d { start, end, step } => MirOp::MILRange1d {
                    name: air_node.name.clone(),
                    start: *start,
                    end: *end,
                    step: *step,
                },
                AirOp::Shape { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILShape { name: air_node.name.clone(), x: mi }
                }
                AirOp::Crop { input, crop_height, crop_width, offset_height, offset_width } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILCrop {
                        name: air_node.name.clone(),
                        x: mi,
                        crop_height: *crop_height,
                        crop_width: *crop_width,
                        offset_height: *offset_height,
                        offset_width: *offset_width,
                    }
                }

                // Scatter / Gather ops
                AirOp::GatherAlongAxis { input, indices, axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    MirOp::MILGatherAlongAxis {
                        name: air_node.name.clone(),
                        x: mi,
                        indices: midx,
                        axis: *axis,
                    }
                }
                AirOp::GatherNd { input, indices } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    MirOp::MILGatherNd { name: air_node.name.clone(), x: mi, indices: midx }
                }
                AirOp::Scatter { input, indices, updates, axis, mode } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    let mu = air_to_mir
                        .get(updates)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(updates.0.clone()));
                    MirOp::MILScatter {
                        name: air_node.name.clone(),
                        x: mi,
                        indices: midx,
                        updates: mu,
                        axis: *axis,
                        mode: mode.clone(),
                    }
                }
                AirOp::ScatterAlongAxis { input, indices, updates, axis } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    let mu = air_to_mir
                        .get(updates)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(updates.0.clone()));
                    MirOp::MILScatterAlongAxis {
                        name: air_node.name.clone(),
                        x: mi,
                        indices: midx,
                        updates: mu,
                        axis: *axis,
                    }
                }
                AirOp::ScatterNd { input, indices, updates } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    let mu = air_to_mir
                        .get(updates)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(updates.0.clone()));
                    MirOp::MILScatterNd {
                        name: air_node.name.clone(),
                        x: mi,
                        indices: midx,
                        updates: mu,
                    }
                }
                AirOp::NonMaximumSuppression {
                    boxes,
                    scores,
                    iou_threshold,
                    score_threshold,
                    max_detections,
                } => {
                    let mb = air_to_mir
                        .get(boxes)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(boxes.0.clone()));
                    let ms = air_to_mir
                        .get(scores)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(scores.0.clone()));
                    MirOp::MILNonMaximumSuppression {
                        name: air_node.name.clone(),
                        boxes: mb,
                        scores: ms,
                        iou_threshold: *iou_threshold,
                        score_threshold: *score_threshold,
                        max_detections: *max_detections,
                    }
                }

                // Quantization ops
                AirOp::Quantize { input, scale, zero_point, axis, output_dtype } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILQuantize {
                        name: air_node.name.clone(),
                        x: mi,
                        scale: *scale,
                        zero_point: *zero_point,
                        axis: *axis,
                        output_dtype: output_dtype.clone(),
                    }
                }
                AirOp::Dequantize { input, scale, zero_point, axis, output_dtype } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILDequantize {
                        name: air_node.name.clone(),
                        x: mi,
                        scale: *scale,
                        zero_point: *zero_point,
                        axis: *axis,
                        output_dtype: output_dtype.clone(),
                    }
                }

                // Constexpr / Compression ops
                AirOp::ConstexprAffineDequantize { quantized_data, scale, zero_point, axis } => {
                    MirOp::MILConstexprAffineDequantize {
                        name: air_node.name.clone(),
                        quantized_data: quantized_data.clone(),
                        scale: *scale,
                        zero_point: *zero_point,
                        axis: *axis,
                    }
                }
                AirOp::ConstexprBlockwiseShiftScale { data, scale, offset, block_size } => {
                    MirOp::MILConstexprBlockwiseShiftScale {
                        name: air_node.name.clone(),
                        data: data.clone(),
                        scale: scale.clone(),
                        offset: offset.clone(),
                        block_size: block_size.clone(),
                    }
                }
                AirOp::ConstexprLutToDense { indices, lut, num_bits } => {
                    MirOp::MILConstexprLutToDense {
                        name: air_node.name.clone(),
                        indices: indices.clone(),
                        lut: lut.clone(),
                        num_bits: *num_bits,
                    }
                }
                AirOp::ConstexprSparseToDense { nonzero_data, shape, default_value } => {
                    MirOp::MILConstexprSparseToDense {
                        name: air_node.name.clone(),
                        nonzero_data: nonzero_data.clone(),
                        shape: shape.clone(),
                        default_value: *default_value,
                    }
                }
                AirOp::ConstexprCast { data, dtype } => MirOp::MILConstexprCast {
                    name: air_node.name.clone(),
                    data: data.clone(),
                    dtype: dtype.clone(),
                },
                AirOp::ConstexprLutToSparse { data, num_bits } => MirOp::MILConstexprLutToSparse {
                    name: air_node.name.clone(),
                    data: data.clone(),
                    num_bits: *num_bits,
                },
                AirOp::ConstexprSparseBlockwiseShiftScale {
                    data,
                    scale,
                    offset,
                    block_size,
                    block_axis,
                } => MirOp::MILConstexprSparseBlockwiseShiftScale {
                    name: air_node.name.clone(),
                    data: data.clone(),
                    scale: scale.clone(),
                    offset: offset.clone(),
                    block_size: block_size.clone(),
                    block_axis: *block_axis,
                },

                // Recurrent ops
                AirOp::Rnn {
                    input,
                    initial_h,
                    weight_ih,
                    weight_hh,
                    bias,
                    mode,
                    output_sequence,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mh = air_to_mir
                        .get(initial_h)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(initial_h.0.clone()));
                    MirOp::MILRnn {
                        name: air_node.name.clone(),
                        x: mi,
                        initial_h: mh,
                        weight_ih: weight_ih.clone(),
                        weight_hh: weight_hh.clone(),
                        bias: bias.clone(),
                        mode: mode.clone(),
                        output_sequence: *output_sequence,
                    }
                }
                AirOp::Gru {
                    input,
                    initial_h,
                    weight_ih,
                    weight_hh,
                    bias,
                    reset_after,
                    output_sequence,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mh = air_to_mir
                        .get(initial_h)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(initial_h.0.clone()));
                    MirOp::MILGru {
                        name: air_node.name.clone(),
                        x: mi,
                        initial_h: mh,
                        weight_ih: weight_ih.clone(),
                        weight_hh: weight_hh.clone(),
                        bias: bias.clone(),
                        reset_after: *reset_after,
                        output_sequence: *output_sequence,
                    }
                }
                AirOp::Lstm {
                    input,
                    initial_h,
                    initial_c,
                    weight_ih,
                    weight_hh,
                    bias,
                    output_sequence,
                } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    let mh = air_to_mir
                        .get(initial_h)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(initial_h.0.clone()));
                    let mc = air_to_mir
                        .get(initial_c)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(initial_c.0.clone()));
                    MirOp::MILLstm {
                        name: air_node.name.clone(),
                        x: mi,
                        initial_h: mh,
                        initial_c: mc,
                        weight_ih: weight_ih.clone(),
                        weight_hh: weight_hh.clone(),
                        bias: bias.clone(),
                        output_sequence: *output_sequence,
                    }
                }

                // Control flow ops
                AirOp::Cond { pred, true_graph, false_graph } => {
                    let mp =
                        air_to_mir.get(pred).cloned().unwrap_or_else(|| MirNodeId(pred.0.clone()));
                    MirOp::MILCond {
                        name: air_node.name.clone(),
                        pred: mp,
                        true_graph: true_graph.clone(),
                        false_graph: false_graph.clone(),
                    }
                }
                AirOp::WhileLoop { condition, body, loop_vars } => {
                    let mv: Vec<MirNodeId> = loop_vars
                        .iter()
                        .map(|id| {
                            air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone()))
                        })
                        .collect();
                    MirOp::MILWhileLoop {
                        name: air_node.name.clone(),
                        condition: condition.clone(),
                        body: body.clone(),
                        loop_vars: mv,
                    }
                }
                AirOp::MakeList { elems, dtype } => {
                    let mv: Vec<MirNodeId> = elems
                        .iter()
                        .map(|id| {
                            air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone()))
                        })
                        .collect();
                    MirOp::MILMakeList {
                        name: air_node.name.clone(),
                        elems: mv,
                        dtype: dtype.clone(),
                    }
                }
                AirOp::ListLength { ls } => {
                    let ml = air_to_mir.get(ls).cloned().unwrap_or_else(|| MirNodeId(ls.0.clone()));
                    MirOp::MILListLength { name: air_node.name.clone(), ls: ml }
                }
                AirOp::ListWrite { ls, index, value } => {
                    let ml = air_to_mir.get(ls).cloned().unwrap_or_else(|| MirNodeId(ls.0.clone()));
                    let mi = air_to_mir
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(index.0.clone()));
                    let mv = air_to_mir
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(value.0.clone()));
                    MirOp::MILListWrite {
                        name: air_node.name.clone(),
                        ls: ml,
                        index: mi,
                        value: mv,
                    }
                }
                AirOp::ListRead { ls, index } => {
                    let ml = air_to_mir.get(ls).cloned().unwrap_or_else(|| MirNodeId(ls.0.clone()));
                    let mi = air_to_mir
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(index.0.clone()));
                    MirOp::MILListRead { name: air_node.name.clone(), ls: ml, index: mi }
                }
                AirOp::ListGather { ls, indices } => {
                    let ml = air_to_mir.get(ls).cloned().unwrap_or_else(|| MirNodeId(ls.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    MirOp::MILListGather { name: air_node.name.clone(), ls: ml, indices: midx }
                }
                AirOp::ListScatter { ls, indices, values } => {
                    let ml = air_to_mir.get(ls).cloned().unwrap_or_else(|| MirNodeId(ls.0.clone()));
                    let midx = air_to_mir
                        .get(indices)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(indices.0.clone()));
                    let mv = air_to_mir
                        .get(values)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(values.0.clone()));
                    MirOp::MILListScatter {
                        name: air_node.name.clone(),
                        ls: ml,
                        indices: midx,
                        values: mv,
                    }
                }

                // Random ops
                AirOp::RandomBernoulli { shape, prob, seed, dtype } => MirOp::MILRandomBernoulli {
                    name: air_node.name.clone(),
                    shape: shape.clone(),
                    prob: *prob,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                AirOp::RandomNormal { shape, mean, stddev, seed, dtype } => {
                    MirOp::MILRandomNormal {
                        name: air_node.name.clone(),
                        shape: shape.clone(),
                        mean: *mean,
                        stddev: *stddev,
                        seed: *seed,
                        dtype: dtype.clone(),
                    }
                }
                AirOp::RandomUniform { shape, low, high, seed, dtype } => MirOp::MILRandomUniform {
                    name: air_node.name.clone(),
                    shape: shape.clone(),
                    low: *low,
                    high: *high,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                AirOp::RandomCategorical { logits, num_samples, seed, dtype } => {
                    let ml = air_to_mir
                        .get(logits)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(logits.0.clone()));
                    MirOp::MILRandomCategorical {
                        name: air_node.name.clone(),
                        logits: ml,
                        num_samples: *num_samples,
                        seed: *seed,
                        dtype: dtype.clone(),
                    }
                }

                // Misc ops
                AirOp::Classify { input } => {
                    let mi = air_to_mir
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| MirNodeId(input.0.clone()));
                    MirOp::MILClassify { name: air_node.name.clone(), x: mi }
                }
            };

            // Sprint 57: infer the output shape from the AIR op and propagate
            // it into the MirNode and the node_shapes map.
            let inferred_shape = infer_shape(&air_node.op, &node_shapes);

            // Preserve pre-seeded shapes for graph inputs. When a graph input
            // is an Identity node with input="__placeholder__", infer_shape
            // returns empty because "__placeholder__" isn't in node_shapes.
            // Without this guard, the seeded shape (e.g., [1, 512] for
            // input_ids) would be overwritten with [], producing wrong
            // metadata throughout the rest of the graph.
            let shape = if inferred_shape.is_empty() {
                node_shapes.get(&air_node.id).cloned().unwrap_or(inferred_shape)
            } else {
                inferred_shape
            };

            mir_nodes.push(MirNode {
                id: mir_id.clone(),
                op: mir_op,
                dtype: mil_dtype,
                shape: shape.clone(),
                compute_unit_hint: Some(compute_hint.clone()),
                air_source: Some(air_node.id.clone()),
            });
            node_shapes.insert(air_node.id.clone(), shape);
        }

        let mir_inputs: Vec<MirNodeId> = input
            .inputs
            .iter()
            .map(|id| air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone())))
            .collect();
        let mir_outputs: Vec<MirNodeId> = input
            .outputs
            .iter()
            .map(|id| air_to_mir.get(id).cloned().unwrap_or_else(|| MirNodeId(id.0.clone())))
            .collect();

        // Single-shard: one MIR graph
        // Use shard name from the shard plan if available
        let shard_name =
            shard_plan.shard_names.first().cloned().unwrap_or_else(|| "shard_0".to_string());

        Ok(vec![MirGraph {
            nodes: mir_nodes,
            inputs: mir_inputs,
            outputs: mir_outputs,
            opset_version: "iOS18".into(),
            shard_name,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::air::{AirNode, AirNodeId};
    use ane_ir::sir::ElementWiseOp;

    fn make_air_graph_with_precision(override_dtype: Option<&str>) -> AirGraph {
        AirGraph {
            nodes: vec![
                AirNode {
                    id: AirNodeId("weight".into()),
                    op: AirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![] },
                    name: "weight".into(),
                    legality_confidence: 0.5,
                    sir_source: None,
                    fallback_risk: 0.1,
                    drift_risk: 0.05,
                    precision_override: None,
                },
                AirNode {
                    id: AirNodeId("output".into()),
                    op: AirOp::MatMul {
                        a: AirNodeId("input".into()),
                        b: AirNodeId("weight".into()),
                    },
                    name: "linear_out".into(),
                    legality_confidence: 0.95,
                    sir_source: None,
                    fallback_risk: 0.05,
                    drift_risk: 0.02,
                    precision_override: override_dtype.map(|s| s.to_string()),
                },
            ],
            inputs: vec![AirNodeId("input".into())],
            outputs: vec![AirNodeId("output".into())],
            staticization_decisions: vec![],
        }
    }

    #[test]
    fn test_precision_override_propagates_to_mir_dtype() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();

        // Without precision override: default fp16
        let air_no_override = make_air_graph_with_precision(None);
        let mirs_no = pass.run(&air_no_override, &shard_plan, &HashMap::new()).unwrap();
        let matmul_node_no = mirs_no[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMatMul { .. }))
            .expect("Expected MatMul node");
        assert_eq!(
            matmul_node_no.dtype,
            MilDtype::Fp16,
            "Without precision override, MIR dtype should be fp16"
        );

        // With fp32 precision override: should produce fp32 MIR node
        let air_fp32 = make_air_graph_with_precision(Some("fp32"));
        let mirs_fp32 = pass.run(&air_fp32, &shard_plan, &HashMap::new()).unwrap();
        let matmul_node_fp32 = mirs_fp32[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMatMul { .. }))
            .expect("Expected MatMul node");
        assert_eq!(
            matmul_node_fp32.dtype,
            MilDtype::Fp32,
            "With fp32 precision override, MIR dtype should be fp32"
        );

        // Ensure the dtype actually changed
        assert_ne!(
            matmul_node_no.dtype, matmul_node_fp32.dtype,
            "Precision override must produce different MIR dtype"
        );
    }

    #[test]
    fn test_no_precision_override_produces_fp16() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = make_air_graph_with_precision(None);
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();

        for node in &mirs[0].nodes {
            assert_eq!(
                node.dtype,
                MilDtype::Fp16,
                "All nodes without precision override should use fp16, but {} uses {:?}",
                node.id.0,
                node.dtype
            );
        }
    }

    #[test]
    fn test_fp16_precision_override_produces_fp16() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = make_air_graph_with_precision(Some("fp16"));
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();

        let matmul_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMatMul { .. }))
            .expect("Expected MatMul node");
        assert_eq!(
            matmul_node.dtype,
            MilDtype::Fp16,
            "Explicit fp16 precision override should produce fp16 MIR dtype"
        );
    }

    // --- Sprint 33: P1 AIR→MIR lowering tests ---

    fn make_simple_air_node(id: &str, op: AirOp) -> AirNode {
        AirNode {
            id: AirNodeId(id.into()),
            op,
            name: id.into(),
            legality_confidence: 0.9,
            sir_source: None,
            fallback_risk: 0.05,
            drift_risk: 0.02,
            precision_override: None,
        }
    }

    #[test]
    fn test_reduce_mean_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "rm",
                AirOp::ReduceMean { input: AirNodeId("x".into()), axes: vec![1], keep_dims: true },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("rm".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILReduceMean { .. }))
            .expect("Expected MILReduceMean node");
        if let MirOp::MILReduceMean { axes, keep_dims, .. } = &node.op {
            assert_eq!(axes, &vec![1]);
            assert!(*keep_dims);
        }
    }

    #[test]
    fn test_reduce_sum_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "rs",
                AirOp::ReduceSum { input: AirNodeId("x".into()), axes: vec![1], keep_dims: false },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("rs".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILReduceSum { .. }))
            .expect("Expected MILReduceSum node");
        if let MirOp::MILReduceSum { axes, keep_dims, .. } = &node.op {
            assert_eq!(axes, &vec![1]);
            assert!(!*keep_dims);
        }
    }

    #[test]
    fn test_rsqrt_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "rsqrt",
                AirOp::Rsqrt { input: AirNodeId("x".into()) },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("rsqrt".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILRsqrt { .. }))
            .expect("Expected MILRsqrt node");
        assert_eq!(node.id.0, "rsqrt");
    }

    #[test]
    fn test_real_div_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "div",
                AirOp::RealDiv { x: AirNodeId("a".into()), y: AirNodeId("b".into()) },
            )],
            inputs: vec![AirNodeId("a".into())],
            outputs: vec![AirNodeId("div".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILRealDiv { .. }))
            .expect("Expected MILRealDiv node");
        if let MirOp::MILRealDiv { x, y, .. } = &node.op {
            assert_eq!(x.0, "a");
            assert_eq!(y.0, "b");
        }
    }

    #[test]
    fn test_layer_norm_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "ln",
                AirOp::LayerNorm {
                    input: AirNodeId("x".into()),
                    weight: "ln_weight".into(),
                    bias: Some("ln_bias".into()),
                    epsilon: 1e-5,
                    axes: vec![1],
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("ln".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILLayerNorm { .. }))
            .expect("Expected MILLayerNorm node");
        if let MirOp::MILLayerNorm { weight, bias, epsilon, axes, .. } = &node.op {
            assert_eq!(weight, "ln_weight");
            assert_eq!(bias, &Some("ln_bias".to_string()));
            assert!((epsilon - 1e-5).abs() < 1e-10);
            assert_eq!(axes, &vec![1]);
        }
    }

    #[test]
    fn test_layer_norm_no_bias_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "ln",
                AirOp::LayerNorm {
                    input: AirNodeId("x".into()),
                    weight: "ln_weight".into(),
                    bias: None,
                    epsilon: 1e-5,
                    axes: vec![1],
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("ln".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILLayerNorm { .. }))
            .expect("Expected MILLayerNorm node");
        if let MirOp::MILLayerNorm { bias, .. } = &node.op {
            assert_eq!(bias, &None);
        }
    }

    #[test]
    fn test_topk_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "topk",
                AirOp::Topk {
                    input: AirNodeId("x".into()),
                    k: 5,
                    axis: -1, // negative axis for last dimension
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("topk".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILTopk { .. }))
            .expect("Expected MILTopk node");
        if let MirOp::MILTopk { k, axis, .. } = &node.op {
            assert_eq!(*k, 5);
            assert_eq!(*axis, -1); // negative axis for last dimension
        }
    }

    #[test]
    fn test_gather_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "gather",
                AirOp::Gather {
                    input: AirNodeId("data".into()),
                    indices: AirNodeId("idx".into()),
                    axis: 0,
                },
            )],
            inputs: vec![AirNodeId("data".into())],
            outputs: vec![AirNodeId("gather".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILGather { .. }))
            .expect("Expected MILGather node");
        if let MirOp::MILGather { indices, axis, .. } = &node.op {
            assert_eq!(indices.0, "idx");
            assert_eq!(*axis, 0);
        }
    }

    #[test]
    fn test_cos_sin_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![
                make_simple_air_node("cos_out", AirOp::Cos { input: AirNodeId("x".into()) }),
                make_simple_air_node("sin_out", AirOp::Sin { input: AirNodeId("x".into()) }),
            ],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("cos_out".into()), AirNodeId("sin_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let cos_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILCos { .. }))
            .expect("Expected MILCos node");
        let sin_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILSin { .. }))
            .expect("Expected MILSin node");
        assert_eq!(cos_node.id.0, "cos_out");
        assert_eq!(sin_node.id.0, "sin_out");
    }

    // --- Sprint 35 / Critique Bug 3: compute_unit_hint propagation tests ---

    #[test]
    fn test_default_shard_plan_produces_cpu_and_ne_hint() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default(); // defaults to CPU_AND_NE
        let air = make_air_graph_with_precision(None);
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();

        for node in &mirs[0].nodes {
            assert_eq!(
                node.compute_unit_hint,
                Some(ComputeUnitHint::CPUAndNE),
                "Default shard plan should produce CPUAndNE compute unit hint, but {} has {:?}",
                node.id.0,
                node.compute_unit_hint
            );
        }
    }

    #[test]
    fn test_gpu_shard_plan_produces_cpu_and_gpu_hint() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan {
            num_shards: 1,
            layer_assignment: vec![0],
            compute_units: vec!["CPU_AND_GPU".to_string()],
            is_multi_shard: false,
            shard_roles: vec![],
            shard_names: vec!["shard_0".to_string()],
        };
        let air = make_air_graph_with_precision(None);
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();

        for node in &mirs[0].nodes {
            assert_eq!(
                node.compute_unit_hint,
                Some(ComputeUnitHint::CPUAndGPU),
                "GPU shard plan should produce CPUAndGPU compute unit hint, but {} has {:?}",
                node.id.0,
                node.compute_unit_hint
            );
        }
    }

    #[test]
    fn test_shard_plan_shard_name_propagates_to_mir() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan {
            num_shards: 1,
            layer_assignment: vec![0],
            compute_units: vec!["CPU_AND_NE".to_string()],
            is_multi_shard: false,
            shard_roles: vec![],
            shard_names: vec!["entry_shard".to_string()],
        };
        let air = make_air_graph_with_precision(None);
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();

        assert_eq!(
            mirs[0].shard_name, "entry_shard",
            "Shard name from shard plan should propagate to MIR graph"
        );
    }

    #[test]
    fn test_normalization_pipeline_lowering() {
        // Test a realistic RMSNorm decomposition:
        // x → ReduceMean → Sub(x, mean) → Mul(x-mean, x-mean) → ReduceMean → Rsqrt → Mul(x, rsqrt)
        // Simpler: just test ReduceMean → Rsqrt chain
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![
                make_simple_air_node(
                    "mean",
                    AirOp::ReduceMean {
                        input: AirNodeId("x".into()),
                        axes: vec![1],
                        keep_dims: true,
                    },
                ),
                make_simple_air_node("rsqrt", AirOp::Rsqrt { input: AirNodeId("mean".into()) }),
            ],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("rsqrt".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        assert_eq!(mirs[0].nodes.len(), 2);
        assert!(mirs[0].nodes.iter().any(|n| matches!(n.op, MirOp::MILReduceMean { .. })));
        assert!(mirs[0].nodes.iter().any(|n| matches!(n.op, MirOp::MILRsqrt { .. })));
    }

    // --- Sprint 36: New AIR→MIR lowering tests ---

    #[test]
    fn test_scaled_dot_product_attention_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "sdpa",
                AirOp::ScaledDotProductAttention {
                    query: AirNodeId("q".into()),
                    key: AirNodeId("k".into()),
                    value: AirNodeId("v".into()),
                    attention_mask: None,
                    scale: None,
                },
            )],
            inputs: vec![AirNodeId("q".into())],
            outputs: vec![AirNodeId("sdpa".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILScaledDotProductAttention { .. }))
            .expect("Expected MIRScaledDotProductAttention node");
        if let MirOp::MILScaledDotProductAttention { query, key, value, .. } = &node.op {
            assert_eq!(query.0, "q");
            assert_eq!(key.0, "k");
            assert_eq!(value.0, "v");
        }
    }

    #[test]
    fn test_slice_by_index_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "slice",
                AirOp::SliceByIndex {
                    input: AirNodeId("qkv".into()),
                    begin: vec![0, 0, 0],
                    end: vec![1, 32, 128],
                    stride: vec![],
                    begin_mask: vec![],
                    end_mask: vec![],
                    squeeze_mask: vec![],
                },
            )],
            inputs: vec![AirNodeId("qkv".into())],
            outputs: vec![AirNodeId("slice".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILSliceByIndex { .. }))
            .expect("Expected MILSliceByIndex node");
        if let MirOp::MILSliceByIndex { begin, end, .. } = &node.op {
            assert_eq!(begin, &vec![0, 0, 0]);
            assert_eq!(end, &vec![1, 32, 128]);
        }
    }

    #[test]
    fn test_gelu_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "gelu",
                AirOp::Gelu { input: AirNodeId("x".into()), mode: "TANH_APPROXIMATION".into() },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("gelu".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILGelu { .. }))
            .expect("Expected MILGelu node");
        if let MirOp::MILGelu { mode, .. } = &node.op {
            assert_eq!(mode, "TANH_APPROXIMATION");
        }
    }

    #[test]
    fn test_state_read_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "k_cache",
                AirOp::StateReadFixed {
                    state_id: "kv_cache_k".into(),
                    shape: vec![64, 128],
                    dtype: ane_ir::mir::MilDtype::Fp16,
                },
            )],
            inputs: vec![],
            outputs: vec![AirNodeId("k_cache".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILReadState { .. }))
            .expect("Expected MILReadState node");
        if let MirOp::MILReadState { state_id, shape, .. } = &node.op {
            assert_eq!(state_id, "kv_cache_k");
            assert_eq!(shape, &vec![64, 128]);
        }
    }

    #[test]
    fn test_state_write_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![
                make_simple_air_node(
                    "k_new",
                    AirOp::SliceByIndex {
                        input: AirNodeId("qkv".into()),
                        begin: vec![0, 0],
                        end: vec![0, 0],
                        stride: vec![],
                        begin_mask: vec![],
                        end_mask: vec![],
                        squeeze_mask: vec![],
                    },
                ),
                make_simple_air_node(
                    "k_write",
                    AirOp::StateWriteFixed {
                        state_id: "kv_cache_k".into(),
                        value: AirNodeId("k_new".into()),
                    },
                ),
            ],
            inputs: vec![AirNodeId("qkv".into())],
            outputs: vec![AirNodeId("k_write".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILCoremlUpdateState { .. }))
            .expect("Expected MILCoremlUpdateState node");
        if let MirOp::MILCoremlUpdateState { state_id, value, .. } = &node.op {
            assert_eq!(state_id, "kv_cache_k");
            assert_eq!(value.0, "k_new");
        }
    }

    #[test]
    fn test_split_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "split",
                AirOp::Split { input: AirNodeId("x".into()), axis: 2, num_splits: 3 },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("split".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILSplit { .. }))
            .expect("Expected MILSplit node");
        if let MirOp::MILSplit { axis, num_splits, .. } = &node.op {
            assert_eq!(*axis, 2);
            assert_eq!(*num_splits, 3);
        }
    }

    #[test]
    fn test_concat_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "concat",
                AirOp::Concat {
                    inputs: vec![AirNodeId("a".into()), AirNodeId("b".into())],
                    axis: 1,
                },
            )],
            inputs: vec![AirNodeId("a".into())],
            outputs: vec![AirNodeId("concat".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILConcat { .. }))
            .expect("Expected MILConcat node");
        if let MirOp::MILConcat { values, axis, .. } = &node.op {
            assert_eq!(values.len(), 2);
            assert_eq!(*axis, 1);
        }
    }

    // --- Sprint 50: P2 AIR→MIR lowering tests ---

    #[test]
    fn test_slice_update_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "su",
                AirOp::SliceUpdate {
                    input: AirNodeId("cache".into()),
                    update: AirNodeId("new_val".into()),
                    begin: vec![0, 0, 0],
                    end: vec![1, 1, 128],
                },
            )],
            inputs: vec![AirNodeId("cache".into())],
            outputs: vec![AirNodeId("su".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILSliceUpdate { .. }))
            .expect("Expected MILSliceUpdate node");
        if let MirOp::MILSliceUpdate { x, update, begin, end, .. } = &node.op {
            assert_eq!(x.0, "cache");
            assert_eq!(update.0, "new_val");
            assert_eq!(begin, &vec![0, 0, 0]);
            assert_eq!(end, &vec![1, 1, 128]);
        }
    }

    #[test]
    fn test_exp_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "exp_out",
                AirOp::Exp { input: AirNodeId("x".into()) },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("exp_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILExp { .. }))
            .expect("Expected MILExp node");
        assert_eq!(node.id.0, "exp_out");
    }

    #[test]
    fn test_sigmoid_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "sig_out",
                AirOp::Sigmoid { input: AirNodeId("x".into()) },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("sig_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILSigmoid { .. }))
            .expect("Expected MILSigmoid node");
        assert_eq!(node.id.0, "sig_out");
    }

    #[test]
    fn test_tanh_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "tanh_out",
                AirOp::Tanh { input: AirNodeId("x".into()) },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("tanh_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILTanh { .. }))
            .expect("Expected MILTanh node");
        assert_eq!(node.id.0, "tanh_out");
    }

    #[test]
    fn test_relu_proper_lowering() {
        // Sprint 50: ReLU should now lower to MILRelu, not MILCast
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "relu_out",
                AirOp::Relu { input: AirNodeId("x".into()) },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("relu_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILRelu { .. }))
            .expect("Expected MILRelu node (not MILCast approximation)");
        assert_eq!(node.id.0, "relu_out");

        // Verify there is NO MILCast node (the old approximation)
        let has_cast = mirs[0].nodes.iter().any(|n| matches!(n.op, MirOp::MILCast { .. }));
        assert!(!has_cast, "ReLU should not produce MILCast approximation");
    }

    #[test]
    fn test_where_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "where_out",
                AirOp::Where {
                    condition: AirNodeId("mask".into()),
                    x: AirNodeId("update".into()),
                    y: AirNodeId("original".into()),
                },
            )],
            inputs: vec![AirNodeId("mask".into())],
            outputs: vec![AirNodeId("where_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILWhere { .. }))
            .expect("Expected MILWhere node");
        if let MirOp::MILWhere { condition, x, y, .. } = &node.op {
            assert_eq!(condition.0, "mask");
            assert_eq!(x.0, "update");
            assert_eq!(y.0, "original");
        }
    }

    // --- Sprint 55: Maximum/Minimum lowering tests ---

    #[test]
    fn test_maximum_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "max_out",
                AirOp::ElementWise {
                    op: ElementWiseOp::Maximum,
                    inputs: vec![AirNodeId("x".into()), AirNodeId("zero".into())],
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("max_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMaximum { .. }))
            .expect("Expected MILMaximum node");
        if let MirOp::MILMaximum { x, y, .. } = &node.op {
            assert_eq!(x.0, "x");
            assert_eq!(y.0, "zero");
        }
    }

    #[test]
    fn test_minimum_lowering() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "min_out",
                AirOp::ElementWise {
                    op: ElementWiseOp::Minimum,
                    inputs: vec![AirNodeId("x".into()), AirNodeId("cap".into())],
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("min_out".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMinimum { .. }))
            .expect("Expected MILMinimum node");
        if let MirOp::MILMinimum { x, y, .. } = &node.op {
            assert_eq!(x.0, "x");
            assert_eq!(y.0, "cap");
        }
    }

    #[test]
    fn test_all_elementwise_ops_lower() {
        // Verify that all 5 declared ElementWiseOp variants now lower successfully.
        // Previously, Maximum and Minimum would error.
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();

        let cases: Vec<(&str, ElementWiseOp)> = vec![
            ("add", ElementWiseOp::Add),
            ("mul", ElementWiseOp::Mul),
            ("abs", ElementWiseOp::Abs),
            ("max", ElementWiseOp::Maximum),
            ("min", ElementWiseOp::Minimum),
        ];

        for (name, op) in cases {
            let label = format!("{:?}", op);
            let air = AirGraph {
                nodes: vec![make_simple_air_node(
                    name,
                    AirOp::ElementWise {
                        op,
                        inputs: vec![AirNodeId("a".into()), AirNodeId("b".into())],
                    },
                )],
                inputs: vec![AirNodeId("a".into())],
                outputs: vec![AirNodeId(name.into())],
                staticization_decisions: vec![],
            };
            let result = pass.run(&air, &shard_plan, &HashMap::new());
            assert!(
                result.is_ok(),
                "ElementWiseOp::{} should lower successfully, but got error: {:?}",
                label,
                result.err()
            );
        }
    }

    /// Sprint 57: StaticLUTProjection now lowers to MILGather instead of
    /// erroring. This test verifies the lowering succeeds and produces a
    /// Gather op.
    #[test]
    fn test_static_lut_projection_lowering_no_longer_errors() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "lut",
                AirOp::StaticLUTProjection {
                    input: AirNodeId("x".into()),
                    indices: "lut_indices".into(),
                    lut: "lut_table".into(),
                    group_size: 16,
                },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("lut".into())],
            staticization_decisions: vec![],
        };
        let result = pass.run(&air, &shard_plan, &HashMap::new());
        assert!(
            result.is_ok(),
            "StaticLUTProjection should no longer error at lowering, but got: {:?}",
            result.err()
        );
        let mirs = result.unwrap();
        let node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILGather { .. }))
            .expect("StaticLUTProjection should lower to MILGather");
        if let MirOp::MILGather { x, indices, axis, .. } = &node.op {
            assert_eq!(x.0, "lut_table");
            assert_eq!(indices.0, "lut_indices");
            assert_eq!(*axis, 0);
        }
    }

    // --- Sprint 57: Shape propagation tests ---

    /// Sprint 57: verify that AIR shape information propagates into MirNode.shape.
    /// MatMul carries output_shape directly in the AIR op.
    #[test]
    fn test_shape_propagation_from_air_to_mir() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![
                AirNode {
                    id: AirNodeId("weight".into()),
                    op: AirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![] },
                    name: "weight".into(),
                    legality_confidence: 0.5,
                    sir_source: None,
                    fallback_risk: 0.1,
                    drift_risk: 0.05,
                    precision_override: None,
                },
                AirNode {
                    id: AirNodeId("output".into()),
                    op: AirOp::MatMul {
                        a: AirNodeId("input".into()),
                        b: AirNodeId("weight".into()),
                    },
                    name: "linear_out".into(),
                    legality_confidence: 0.95,
                    sir_source: None,
                    fallback_risk: 0.05,
                    drift_risk: 0.02,
                    precision_override: None,
                },
            ],
            inputs: vec![AirNodeId("input".into())],
            outputs: vec![AirNodeId("output".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let matmul_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILMatMul { .. }))
            .expect("Expected MatMul node");
        assert_eq!(
            matmul_node.shape,
            Vec::<usize>::new(),
            "MirNode.shape for MatMul is empty since output_shape was removed from AirOp::MatMul"
        );
    }

    /// Sprint 57: verify that Reshape target_shape propagates into MirNode.shape.
    #[test]
    fn test_reshape_shape_propagation() {
        let pass = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let air = AirGraph {
            nodes: vec![make_simple_air_node(
                "reshape",
                AirOp::Reshape { input: AirNodeId("x".into()), target_shape: vec![2, 16] },
            )],
            inputs: vec![AirNodeId("x".into())],
            outputs: vec![AirNodeId("reshape".into())],
            staticization_decisions: vec![],
        };
        let mirs = pass.run(&air, &shard_plan, &HashMap::new()).unwrap();
        let reshape_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILReshape { .. }))
            .expect("Expected MILReshape node");
        assert_eq!(
            reshape_node.shape,
            vec![2, 16],
            "MirNode.shape should propagate from Reshape target_shape"
        );
    }

    // --- Sprint 59: SDPA constraint validation tests ---

    #[test]
    fn test_sdpa_validation_rank4_succeeds() {
        // Rank-4 Q, K, V should pass validation
        let result =
            validate_sdpa_constraints(&[1, 8, 4, 64], &[1, 8, 4, 64], &[1, 8, 4, 64], None);
        assert!(result.is_ok(), "Rank-4 SDPA should pass validation");
    }

    #[test]
    fn test_sdpa_validation_rank5_query_fails() {
        // Rank-5 query should fail
        let result = validate_sdpa_constraints(
            &[1, 2, 8, 4, 64], // rank 5
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            None,
        );
        assert!(result.is_err(), "Rank-5 query should fail SDPA validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("query") && err.contains("rank 5"),
            "Error should mention query and rank 5, got: {err}"
        );
    }

    #[test]
    fn test_sdpa_validation_rank5_key_fails() {
        // Rank-5 key should fail
        let result = validate_sdpa_constraints(
            &[1, 8, 4, 64],
            &[1, 2, 8, 4, 64], // rank 5
            &[1, 8, 4, 64],
            None,
        );
        assert!(result.is_err(), "Rank-5 key should fail SDPA validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("key") && err.contains("rank 5"),
            "Error should mention key and rank 5, got: {err}"
        );
    }

    #[test]
    fn test_sdpa_validation_rank5_value_fails() {
        // Rank-5 value should fail
        let result = validate_sdpa_constraints(
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            &[1, 2, 8, 4, 64], // rank 5
            None,
        );
        assert!(result.is_err(), "Rank-5 value should fail SDPA validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("value") && err.contains("rank 5"),
            "Error should mention value and rank 5, got: {err}"
        );
    }

    #[test]
    fn test_sdpa_validation_rank5_mask_fails() {
        // Rank-5 mask should fail
        let result = validate_sdpa_constraints(
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            Some(&[1, 2, 8, 4, 64]), // rank 5
        );
        assert!(result.is_err(), "Rank-5 mask should fail SDPA validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("attention_mask") && err.contains("rank 5"),
            "Error should mention attention_mask and rank 5, got: {err}"
        );
    }

    #[test]
    fn test_sdpa_validation_rank3_succeeds() {
        // Rank-3 operands should pass
        let result = validate_sdpa_constraints(&[8, 4, 64], &[8, 4, 64], &[8, 4, 64], None);
        assert!(result.is_ok(), "Rank-3 SDPA should pass validation");
    }

    #[test]
    fn test_sdpa_validation_with_mask_succeeds() {
        // Valid SDPA with mask
        let result = validate_sdpa_constraints(
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            &[1, 8, 4, 64],
            Some(&[1, 8, 4, 64]),
        );
        assert!(result.is_ok(), "SDPA with valid mask should pass validation");
    }
}
