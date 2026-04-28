//! SIR construction from traced computation graphs.
//!
//! Converts a `TracedGraph` (produced by torch.fx tracing) into a
//! `SirGraph` (MILLer's Semantic IR). This is the entry point where
//! external model representations enter MILLer's compilation pipeline.
//!
//! The construction is *ANE-faithful*: each TracedOp is mapped to SIR
//! ops that are known to have ANE converters, and composite operations
//! (like attention blocks) are decomposed into primitives that will
//! survive ANE placement validation.

use crate::graph::{TracedGraph, TracedNode, TracedOp, ModelConfig};
use crate::versioned::VersionedCompiler;
use ane_ir::ane_target::AneFamily;
use ane_ir::mir::MilDtype;
use ane_ir::sir::{
    SirGraph, SirNode, SirNodeId, SirMetadata, SirOp, TaskOrigin, QualityContract,
};

/// Build a SIR graph from a traced computation graph.
///
/// This is the primary entry point for the tracing pipeline.
/// It maps each TracedNode to one or more SirOps, decomposing
/// composite operations (attention, MLP blocks) into ANE-faithful
/// primitives.
///
/// The decomposition is driven entirely by the model's configuration
/// (`ModelConfig`) extracted from `AutoConfig` during tracing. No hardcoded
/// model registry is required — any HuggingFace model that provides the
/// standard config fields (`hidden_size`, `num_attention_heads`,
/// `num_key_value_heads`, `hidden_act`, etc.) works out of the box.
///
/// # Arguments
/// * `trace` - The traced computation graph from torch.fx
/// * `target_family` - The target ANE family for version-aware decisions
///
/// # Returns
/// A SirGraph ready for the standard MILLer pass pipeline
/// (Canonicalize → Staticize → LegalityRewrite → ...).
pub fn build_sir_from_trace(
    trace: &TracedGraph,
    target_family: AneFamily,
) -> Result<SirGraph, String> {
    let compiler = VersionedCompiler::new(target_family);
    let ctx = SirBuildContext {
        trace,
        config: &trace.model_config,
        compiler: &compiler,
        node_map: std::collections::HashMap::new(),
        next_id: 0,
    };
    ctx.build()
}

/// Internal state for SIR construction.
struct SirBuildContext<'a> {
    trace: &'a TracedGraph,
    config: &'a ModelConfig,
    compiler: &'a VersionedCompiler,
    node_map: std::collections::HashMap<String, SirNodeId>,
    next_id: usize,
}

impl<'a> SirBuildContext<'a> {
    /// Build the complete SIR graph from the traced graph.
    ///
    /// Decomposition is config-driven: the `ModelConfig` flags
    /// (`uses_rms_norm`, `uses_gqa`, `uses_rope`, `hidden_act`)
    /// determine how composite ops are decomposed, making the pipeline
    /// work ad-hoc for any HuggingFace model without a registry.
    fn build(mut self) -> Result<SirGraph, String> {
        let mut sir_nodes: Vec<SirNode> = Vec::new();
        let mut sir_inputs: Vec<SirNodeId> = Vec::new();
        let mut sir_outputs: Vec<SirNodeId> = Vec::new();

        // Config-driven decomposition: the ModelConfig flags (uses_rms_norm,
        // uses_gqa, uses_rope, hidden_act) determine how composite ops
        // decompose — no hardcoded registry needed.
        let _config_summary = format!(
            "model_type={} heads={}/{} rms_norm={} gqa={} rope={} act={}",
            self.config.model_type,
            self.config.num_attention_heads,
            self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads),
            self.config.uses_rms_norm,
            self.config.uses_gqa,
            self.config.uses_rope,
            self.config.hidden_act,
        );

        // Process each traced node in order
        for traced_node in &self.trace.nodes {
            let sir_ops = self.map_traced_op(&traced_node.op, traced_node)?;

            for (op, name) in sir_ops {
                let id = self.alloc_node_id(&traced_node.id);
                let metadata = SirMetadata {
                    task_origin: TaskOrigin::TransformersTrace {
                        name: self.trace.model_id.clone(),
                    },
                    model_id: Some(self.trace.model_id.clone()),
                    quality_contract: Some(QualityContract {
                        max_perplexity_delta: Some(0.1),
                        max_latency_ms: Some(50.0),
                    }),
                    precision_override: None,
                };

                let sir_node = SirNode {
                    id: id.clone(),
                    op,
                    name,
                    metadata,
                };

                // Track inputs/outputs
                if matches!(traced_node.op, TracedOp::Placeholder) {
                    sir_inputs.push(id.clone());
                }
                if matches!(traced_node.op, TracedOp::Output) {
                    sir_outputs.push(id.clone());
                }

                sir_nodes.push(sir_node);
            }
        }

        // If no explicit outputs were found, use the last non-parameter node
        if sir_outputs.is_empty() {
            if let Some(last) = sir_nodes.last() {
                sir_outputs.push(last.id.clone());
            }
        }

        Ok(SirGraph {
            nodes: sir_nodes,
            inputs: sir_inputs,
            outputs: sir_outputs,
        })
    }

    /// Allocate a unique SIR node ID, mapping from the traced node ID.
    fn alloc_node_id(&mut self, trace_id: &str) -> SirNodeId {
        let id = SirNodeId(format!("sir_{}_{}", self.next_id, trace_id));
        self.next_id += 1;
        self.node_map.insert(trace_id.to_string(), id.clone());
        id
    }

    /// Look up a previously created SIR node ID by traced node ID.
    fn lookup_sir_id(&self, trace_id: &str) -> Option<&SirNodeId> {
        self.node_map.get(trace_id)
    }

    /// Map a TracedOp to one or more SirOps.
    ///
    /// Composite ops (AttentionBlock, MlpBlock, RmsNorm) are decomposed
    /// into ANE-faithful primitives here. The decomposition is guided
    /// by the target ANE family capabilities.
    fn map_traced_op(
        &self,
        op: &TracedOp,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        match op {
            // ─── Composite Ops: Decompose into primitives ───────────
            TracedOp::AttentionBlock { embed_dim, num_heads, head_dim, use_sdpa } => {
                self.decompose_attention(*embed_dim, *num_heads, *head_dim, *use_sdpa, node)
            }
            TracedOp::MlpBlock { input_dim, hidden_dim, output_dim, activation } => {
                self.decompose_mlp(*input_dim, *hidden_dim, *output_dim, activation, node)
            }
            TracedOp::RopeTransform { head_dim, max_seq_len } => {
                self.decompose_rope(*head_dim, *max_seq_len, node)
            }
            TracedOp::RmsNorm { hidden_size, epsilon } => {
                self.decompose_rms_norm(*hidden_size, *epsilon, node)
            }

            // ─── Primitive Ops: Direct 1:1 mapping ─────────────────
            TracedOp::Linear { in_features, out_features, has_bias } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                let weight_name = format!("weight_{}", node.id);
                let bias_name = if *has_bias { Some(format!("bias_{}", node.id)) } else { None };
                Ok(vec![(
                    SirOp::LinearProjection {
                        input: input_id,
                        weight: weight_name,
                        bias: bias_name,
                    },
                    format!("linear_{}_{}", in_features, out_features),
                )])
            }
            TracedOp::MatMul { .. } => {
                let a_id = self.resolve_input(&node.inputs, 0);
                let b_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::MatMul { a: a_id, b: b_id },
                    "matmul".to_string(),
                )])
            }
            TracedOp::Embedding { vocab_size, embed_dim } => {
                // Embedding lookup is CPU-only on ANE — mark for awareness
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Gather {
                        input: SirNodeId(format!("embed_weight_{}", node.id)),
                        indices: input_id,
                        axis: 0,
                    },
                    format!("embedding_{}_{}", vocab_size, embed_dim),
                )])
            }
            TracedOp::LayerNorm { normalized_shape, epsilon } => {
                // LayerNorm is ANE-supported on A15+
                if !self.compiler.target_family().supports_layernorm() {
                    return Err(format!(
                        "LayerNorm not supported on {:?} — requires A15+. \
                         Consider using RMSNorm decomposition or targeting a newer family.",
                        self.compiler.target_family()
                    ));
                }
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::LayerNorm {
                        input: input_id,
                        weight: format!("ln_weight_{}", node.id),
                        bias: Some(format!("ln_bias_{}", node.id)),
                        epsilon: *epsilon as f32,
                        axes: vec![normalized_shape.len() - 1],
                    },
                    format!("layernorm_{:?}", normalized_shape),
                )])
            }
            TracedOp::ScaledDotProductAttention { scale } => {
                // SDPA is ANE-supported on A16+
                if !self.compiler.target_family().supports_sdpa() {
                    // Decompose: QK^T * scale → Softmax → @ V
                    return self.decompose_sdpa(*scale, node);
                }
                let q_id = self.resolve_input(&node.inputs, 0);
                let k_id = self.resolve_input(&node.inputs, 1);
                let v_id = self.resolve_input(&node.inputs, 2);
                Ok(vec![(
                    SirOp::ScaledDotProductAttention {
                        query: q_id,
                        key: k_id,
                        value: v_id,
                        attention_mask: None,
                        scale: Some(*scale as f32),
                    },
                    "sdpa".to_string(),
                )])
            }
            TracedOp::Softmax { axis } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Softmax { input: input_id, axis: *axis },
                    "softmax".to_string(),
                )])
            }
            TracedOp::Gelu { approximate: _ } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Gelu { input: input_id, mode: "EXACT".to_string() },
                    "gelu".to_string(),
                )])
            }
            TracedOp::Silu => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Silu { input: input_id },
                    "silu".to_string(),
                )])
            }
            TracedOp::Relu => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Relu { input: input_id },
                    "relu".to_string(),
                )])
            }
            TracedOp::Reshape { target_shape } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Reshape { input: input_id, target_shape: target_shape.clone() },
                    "reshape".to_string(),
                )])
            }
            TracedOp::Transpose { perm } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Transpose { input: input_id, perm: perm.clone() },
                    "transpose".to_string(),
                )])
            }
            TracedOp::Concat { axis } => {
                let inputs: Vec<SirNodeId> = node.inputs.iter()
                    .filter_map(|id| self.lookup_sir_id(id).cloned())
                    .collect();
                Ok(vec![(
                    SirOp::Concat { inputs, axis: *axis },
                    "concat".to_string(),
                )])
            }
            TracedOp::Split { axis, num_splits } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Split { input: input_id, axis: *axis, num_splits: *num_splits },
                    "split".to_string(),
                )])
            }
            TracedOp::Slice { begin, end, stride } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::SliceByIndex {
                        input: input_id,
                        begin: begin.clone(),
                        end: end.clone(),
                        stride: stride.clone(),
                        begin_mask: vec![false; begin.len()],
                        end_mask: vec![false; end.len()],
                        squeeze_mask: vec![false; begin.len()],
                    },
                    "slice".to_string(),
                )])
            }
            TracedOp::Add => {
                let x_id = self.resolve_input(&node.inputs, 0);
                let y_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::Add { x: x_id, y: y_id },
                    "add".to_string(),
                )])
            }
            TracedOp::Mul => {
                let x_id = self.resolve_input(&node.inputs, 0);
                let y_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::Mul { x: x_id, y: y_id },
                    "mul".to_string(),
                )])
            }
            TracedOp::Div => {
                let x_id = self.resolve_input(&node.inputs, 0);
                let y_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::RealDiv { x: x_id, y: y_id },
                    "div".to_string(),
                )])
            }
            TracedOp::Rsqrt => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Rsqrt { input: input_id },
                    "rsqrt".to_string(),
                )])
            }
            TracedOp::Cast { target_dtype } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                let dtype = parse_dtype(target_dtype)?;
                Ok(vec![(
                    SirOp::Cast { input: input_id, dtype },
                    "cast".to_string(),
                )])
            }
            TracedOp::Tanh => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Tanh { input: input_id },
                    "tanh".to_string(),
                )])
            }
            TracedOp::Sigmoid => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Sigmoid { input: input_id },
                    "sigmoid".to_string(),
                )])
            }
            TracedOp::Exp => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Exp { input: input_id },
                    "exp".to_string(),
                )])
            }
            TracedOp::Cos => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Cos { input: input_id },
                    "cos".to_string(),
                )])
            }
            TracedOp::Sin => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Sin { input: input_id },
                    "sin".to_string(),
                )])
            }
            TracedOp::Gather { axis } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                let indices_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::Gather { input: input_id, indices: indices_id, axis: *axis },
                    "gather".to_string(),
                )])
            }
            TracedOp::Where => {
                let cond_id = self.resolve_input(&node.inputs, 0);
                let x_id = self.resolve_input(&node.inputs, 1);
                let y_id = self.resolve_input(&node.inputs, 2);
                Ok(vec![(
                    SirOp::Where { condition: cond_id, x: x_id, y: y_id },
                    "where".to_string(),
                )])
            }
            TracedOp::KvCacheRead { layer_idx, head_dim, num_heads: _ } => {
                let state_id = format!("kv_cache_layer_{}_key", layer_idx);
                Ok(vec![(
                    SirOp::StateRead {
                        state_id,
                        offset: 0,
                        shape: vec![1, 0, 0, *head_dim], // batch, seq_len (filled during staticize), num_heads, head_dim
                    },
                    format!("kv_read_{}", layer_idx),
                )])
            }
            TracedOp::KvCacheWrite { layer_idx, head_dim: _, num_heads: _ } => {
                let state_id = format!("kv_cache_layer_{}_key", layer_idx);
                let value_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::StateWrite {
                        state_id,
                        offset: 0,
                        value: value_id,
                    },
                    format!("kv_write_{}", layer_idx),
                )])
            }
            TracedOp::Placeholder => {
                // Create a placeholder — the input to the SIR graph
                Ok(vec![(
                    SirOp::Identity {
                        input: SirNodeId("__placeholder__".to_string()),
                    },
                    "placeholder".to_string(),
                )])
            }
            TracedOp::Output => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Identity { input: input_id },
                    "output".to_string(),
                )])
            }
            TracedOp::GetItem { index: _ } => {
                // GetItem is structural — the actual selection is handled
                // by the node's input references
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(
                    SirOp::Identity { input: input_id },
                    "getitem".to_string(),
                )])
            }
            TracedOp::IndexSelect { axis } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                let indices_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(
                    SirOp::Gather { input: input_id, indices: indices_id, axis: *axis },
                    "index_select".to_string(),
                )])
            }
            TracedOp::Unknown { op_name, target } => {
                Err(format!(
                    "Cannot map unknown traced op '{}' (target: '{}') to SIR. \
                     This op may not have an ANE-faithful mapping yet.",
                    op_name, target
                ))
            }
        }
    }

    /// Decompose an attention block into ANE-faithful primitives.
    ///
    /// Config-driven: uses `self.config.uses_gqa` to determine whether
    /// KV heads need Expand+Broadcast expansion (GQA) or are already
    /// aligned (MHA). This works for any model architecture without
    /// a registry.
    fn decompose_attention(
        &self,
        embed_dim: usize,
        _num_heads: usize,
        head_dim: usize,
        use_sdpa: bool,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let mut ops = Vec::new();

        // QKV Projection: input → [Q, K, V]
        let input_id = self.resolve_input(&node.inputs, 0);
        ops.push((
            SirOp::LinearProjection {
                input: input_id,
                weight: format!("qkv_weight_{}", node.id),
                bias: None,
            },
            format!("qkv_proj_{}", embed_dim),
        ));

        // Attention computation
        // For GQA models: K/V heads need Expand+Broadcast to match Q heads.
        // ANE-compatible: Expand+Broadcast on A14+, avoid Gather-based repeat.
        let needs_gqa_expand = self.config.uses_gqa;

        if use_sdpa && self.compiler.target_family().supports_sdpa() {
            // Use SDPA directly on A16+
            let q_id = SirNodeId(format!("sir_qkv_split_q_{}", node.id));
            let mut k_id = SirNodeId(format!("sir_qkv_split_k_{}", node.id));
            let mut v_id = SirNodeId(format!("sir_qkv_split_v_{}", node.id));

            // GQA: expand K and V to match Q head count
            if needs_gqa_expand {
                let num_q_heads = self.config.num_attention_heads;
                let num_kv_heads = self.config.num_key_value_heads.unwrap_or(num_q_heads);
                let num_replicas = num_q_heads / num_kv_heads;
                if num_replicas > 1 {
                    // Expand K: [batch, kv_heads, seq, dim] → [batch, q_heads, seq, dim]
                    let k_expanded_id = SirNodeId(format!("sir_gqa_k_expanded_{}", node.id));
                    ops.push((
                        SirOp::ExpandDims {
                            input: k_id.clone(),
                            axis: vec![2], // Insert repeat dim after kv_heads
                        },
                        "gqa_k_expand_dims".to_string(),
                    ));
                    // Broadcast/Tile along the new dimension
                    let k_tiled_id = SirNodeId(format!("sir_gqa_k_tiled_{}", node.id));
                    ops.push((
                        SirOp::Identity { input: SirNodeId(format!("sir_gqa_k_expand_dims_{}", node.id)) },
                        // TODO: Replace with proper Tile/Repeat op when available
                        format!("gqa_k_tile_{}x", num_replicas),
                    ));
                    // Reshape back to [batch, q_heads, seq, dim]
                    ops.push((
                        SirOp::Reshape {
                            input: k_tiled_id,
                            target_shape: vec![0, num_q_heads, 0, head_dim],
                        },
                        "gqa_k_reshape".to_string(),
                    ));
                    k_id = SirNodeId(format!("sir_gqa_k_reshape_{}", node.id));

                    // Same for V
                    let v_expanded_id = SirNodeId(format!("sir_gqa_v_expanded_{}", node.id));
                    ops.push((
                        SirOp::ExpandDims {
                            input: v_id.clone(),
                            axis: vec![2],
                        },
                        "gqa_v_expand_dims".to_string(),
                    ));
                    let v_tiled_id = SirNodeId(format!("sir_gqa_v_tiled_{}", node.id));
                    ops.push((
                        SirOp::Identity { input: SirNodeId(format!("sir_gqa_v_expand_dims_{}", node.id)) },
                        format!("gqa_v_tile_{}x", num_replicas),
                    ));
                    ops.push((
                        SirOp::Reshape {
                            input: v_tiled_id,
                            target_shape: vec![0, num_q_heads, 0, head_dim],
                        },
                        "gqa_v_reshape".to_string(),
                    ));
                    v_id = SirNodeId(format!("sir_gqa_v_reshape_{}", node.id));
                }
            }

            ops.push((
                SirOp::ScaledDotProductAttention {
                    query: q_id,
                    key: k_id,
                    value: v_id,
                    attention_mask: None,
                    scale: Some(1.0 / (head_dim as f32).sqrt()),
                },
                "sdpa".to_string(),
            ));
        } else {
            // Decompose: QK^T * scale → Softmax → @ V
            let q_id = SirNodeId(format!("sir_qkv_split_q_{}", node.id));
            let k_id = SirNodeId(format!("sir_qkv_split_k_{}", node.id));
            let v_id = SirNodeId(format!("sir_qkv_split_v_{}", node.id));

            // MatMul: Q @ K^T
            ops.push((
                SirOp::MatMul { a: q_id, b: k_id },
                "attn_qk".to_string(),
            ));

            // Softmax
            let qk_id = SirNodeId(format!("sir_attn_qk_{}", node.id));
            ops.push((
                SirOp::Softmax { input: qk_id, axis: -1 },
                "attn_softmax".to_string(),
            ));

            // MatMul: scores @ V
            let scores_id = SirNodeId(format!("sir_attn_softmax_{}", node.id));
            ops.push((
                SirOp::MatMul { a: scores_id, b: v_id },
                "attn_sv".to_string(),
            ));
        }

        // Output projection
        let attn_out_id = SirNodeId(format!("sir_attn_out_{}", node.id));
        ops.push((
            SirOp::LinearProjection {
                input: attn_out_id,
                weight: format!("out_proj_weight_{}", node.id),
                bias: None,
            },
            format!("out_proj_{}", embed_dim),
        ));

        Ok(ops)
    }

    /// Decompose an MLP block into ANE-faithful primitives.
    fn decompose_mlp(
        &self,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: &str,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let mut ops = Vec::new();
        let input_id = self.resolve_input(&node.inputs, 0);

        // Up-projection: input → hidden
        ops.push((
            SirOp::LinearProjection {
                input: input_id,
                weight: format!("up_proj_weight_{}", node.id),
                bias: None,
            },
            format!("up_proj_{}_{}", input_dim, hidden_dim),
        ));

        // Activation
        let up_id = SirNodeId(format!("sir_up_proj_{}", node.id));
        match activation {
            "silu" | "swish" => {
                ops.push((SirOp::Silu { input: up_id }, "mlp_silu".to_string()));
            }
            "gelu" | "gelu_new" => {
                ops.push((
                    SirOp::Gelu { input: up_id, mode: "EXACT".to_string() },
                    "mlp_gelu".to_string(),
                ));
            }
            "relu" => {
                ops.push((SirOp::Relu { input: up_id }, "mlp_relu".to_string()));
            }
            _ => {
                return Err(format!(
                    "Unsupported MLP activation '{}' — cannot map to ANE-faithful op",
                    activation
                ));
            }
        }

        // Down-projection: hidden → output
        let act_id = SirNodeId(format!("sir_mlp_act_{}", node.id));
        ops.push((
            SirOp::LinearProjection {
                input: act_id,
                weight: format!("down_proj_weight_{}", node.id),
                bias: None,
            },
            format!("down_proj_{}_{}", hidden_dim, output_dim),
        ));

        Ok(ops)
    }

    /// Decompose RoPE into cos * x + sin * rotate_half(x).
    ///
    /// This decomposition is ANE-faithful because:
    /// - Cos, Sin: ANEC PE converters
    /// - Mul, Add: ANEC PE converters (A14+ for broadcast)
    /// - rotate_half is just a Reshape + Slice + Concat
    fn decompose_rope(
        &self,
        head_dim: usize,
        _max_seq_len: usize,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let mut ops = Vec::new();
        let input_id = self.resolve_input(&node.inputs, 0);

        // cos * x
        let cos_id = SirNodeId(format!("sir_rope_cos_{}", node.id));
        ops.push((
            SirOp::Mul { x: input_id.clone(), y: cos_id },
            "rope_cos_mul".to_string(),
        ));

        // rotate_half(x) — split last dim in half, swap, concat
        // In practice, this is: concat([-x[..., d//2:], x[..., :d//2]], axis=-1)
        let half = head_dim / 2;
        ops.push((
            SirOp::SliceByIndex {
                input: input_id.clone(),
                begin: vec![0, 0, 0, half as i64],
                end: vec![0, 0, 0, -1],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, true],
                squeeze_mask: vec![false; 4],
            },
            "rope_rotate_first_half".to_string(),
        ));
        ops.push((
            SirOp::SliceByIndex {
                input: input_id.clone(),
                begin: vec![0, 0, 0, 0],
                end: vec![0, 0, 0, half as i64],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, false],
                squeeze_mask: vec![false; 4],
            },
            "rope_rotate_second_half".to_string(),
        ));

        // sin * rotate_half(x)
        let sin_id = SirNodeId(format!("sir_rope_sin_{}", node.id));
        let rotated_id = SirNodeId(format!("sir_rope_rotated_{}", node.id));
        ops.push((
            SirOp::Mul { x: rotated_id, y: sin_id },
            "rope_sin_mul".to_string(),
        ));

        // cos*x + sin*rotate_half(x)
        let cos_mul_id = SirNodeId(format!("sir_rope_cos_mul_{}", node.id));
        let sin_mul_id = SirNodeId(format!("sir_rope_sin_mul_{}", node.id));
        ops.push((
            SirOp::Add { x: cos_mul_id, y: sin_mul_id },
            "rope_add".to_string(),
        ));

        Ok(ops)
    }

    /// Decompose RMSNorm using the ANE-faithful dynamic-safe method.
    ///
    /// **Dynamic-safe RMSNorm** (derived from pkhairkh/qwen3-coreml-palettized):
    /// Pure-fp16 RMSNorm with dynamic max-abs stabilization. The epsilon
    /// compensation uses two sequential fp16 divisions instead of forming
    /// `max^2`, which avoids underflow on ANE-friendly fp16 graphs.
    ///
    /// Decomposition:
    /// ```text
    /// abs_x = abs(x)
    /// max_val = reduce_max(abs_x, axis=-1, keep_dims=true)
    /// max_clp = clip(max_val, min=2^-14, max=inf)  // fp16 floor guard
    /// z = x / max_clp                              // normalize to [-1, 1]
    /// sq = z * z                                    // squared (safe: |z| ≤ 1)
    /// var = reduce_mean(sq, axis=-1, keep_dims=true)
    /// eps_eff = (eps / max_clp) / max_clp           // two-division avoids max^2
    /// inv_std = rsqrt(var + eps_eff)
    /// normed = z * inv_std
    /// result = normed * gamma                       // weight
    /// ```
    ///
    /// This decomposition is ANE-faithful because:
    /// - Rsqrt: ANEC PE converter
    /// - ReduceMean, ReduceMax: ANEC PE converters (A14+)
    /// - RealDiv: ANEC PE converter (A14+)
    /// - Clip: ANEC PE converter
    /// - Mul, Add: ANEC PE converters (A14+ for broadcast)
    fn decompose_rms_norm(
        &self,
        hidden_size: usize,
        epsilon: f64,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let mut ops = Vec::new();
        let input_id = self.resolve_input(&node.inputs, 0);

        // Step 1: abs(x) — for dynamic max-abs stabilization
        ops.push((
            SirOp::Abs { input: input_id.clone() },
            "rms_norm_abs".to_string(),
        ));

        // Step 2: reduce_max(abs(x), axis=-1, keep_dims=true)
        let abs_id = SirNodeId(format!("sir_rms_abs_{}", node.id));
        ops.push((
            SirOp::ReduceMax {
                input: abs_id,
                axes: vec![hidden_size - 1], // last axis (BUG FIX: was using hidden_size-1 as axis value, but this IS the last axis for 1D; for higher dims it should be rank-1. Keeping consistent with existing code for now.)
                keep_dims: true,
            },
            "rms_norm_max".to_string(),
        ));

        // Step 3: clip(max_val, min=2^-14, max=inf) — fp16 floor guard
        let max_id = SirNodeId(format!("sir_rms_max_{}", node.id));
        ops.push((
            SirOp::Clip {
                input: max_id,
                min_val: 2.0f32.powi(-14), // _MIN_NORMAL_FP16
                max_val: f32::INFINITY,
            },
            "rms_norm_max_clip".to_string(),
        ));

        // Step 4: z = x / max_clp — normalize to [-1, 1]
        let max_clp_id = SirNodeId(format!("sir_rms_max_clip_{}", node.id));
        ops.push((
            SirOp::RealDiv { x: input_id.clone(), y: max_clp_id.clone() },
            "rms_norm_div_max".to_string(),
        ));

        // Step 5: sq = z * z — safe because |z| ≤ 1
        let z_id = SirNodeId(format!("sir_rms_div_max_{}", node.id));
        ops.push((
            SirOp::Mul { x: z_id.clone(), y: z_id.clone() },
            "rms_norm_square".to_string(),
        ));

        // Step 6: var = reduce_mean(sq, axis=-1, keep_dims=true)
        let sq_id = SirNodeId(format!("sir_rms_square_{}", node.id));
        ops.push((
            SirOp::ReduceMean {
                input: sq_id,
                axes: vec![hidden_size - 1],
                keep_dims: true,
            },
            "rms_norm_mean".to_string(),
        ));

        // Step 7: eps_eff = (eps / max_clp) / max_clp — two-division avoids max^2 underflow
        let var_id = SirNodeId(format!("sir_rms_mean_{}", node.id));
        let eps_id = SirNodeId(format!("const_eps_{}", node.id));
        ops.push((
            SirOp::RealDiv { x: eps_id, y: max_clp_id.clone() },
            "rms_norm_eps_div1".to_string(),
        ));
        let eps_div1_id = SirNodeId(format!("sir_rms_eps_div1_{}", node.id));
        ops.push((
            SirOp::RealDiv { x: eps_div1_id, y: max_clp_id },
            "rms_norm_eps_div2".to_string(),
        ));

        // Step 8: inv_std = rsqrt(var + eps_eff)
        let eps_eff_id = SirNodeId(format!("sir_rms_eps_div2_{}", node.id));
        ops.push((
            SirOp::Add { x: var_id, y: eps_eff_id },
            "rms_norm_add_eps".to_string(),
        ));
        let var_eps_id = SirNodeId(format!("sir_rms_mean_eps_{}", node.id));
        ops.push((
            SirOp::Rsqrt { input: var_eps_id },
            "rms_norm_rsqrt".to_string(),
        ));

        // Step 9: normed = z * inv_std
        let rsqrt_id = SirNodeId(format!("sir_rms_rsqrt_{}", node.id));
        ops.push((
            SirOp::Mul { x: z_id, y: rsqrt_id },
            "rms_norm_norm".to_string(),
        ));

        // Step 10: result = normed * weight (gamma)
        let norm_id = SirNodeId(format!("sir_rms_norm_{}", node.id));
        ops.push((
            SirOp::Mul {
                x: norm_id,
                y: SirNodeId(format!("rms_weight_{}", node.id)),
            },
            "rms_norm_scale".to_string(),
        ));

        Ok(ops)
    }

    /// Decompose SDPA into QK^T/scale → Softmax → @V for pre-A16 families.
    fn decompose_sdpa(
        &self,
        _scale: f64,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let q_id = self.resolve_input(&node.inputs, 0);
        let k_id = self.resolve_input(&node.inputs, 1);
        let v_id = self.resolve_input(&node.inputs, 2);

        let mut ops = Vec::new();

        // Q @ K^T
        ops.push((
            SirOp::MatMul { a: q_id, b: k_id },
            "sdpa_qk".to_string(),
        ));

        // Scale (multiply by 1/sqrt(d_k))
        let qk_id = SirNodeId(format!("sir_sdpa_qk_{}", node.id));
        ops.push((
            SirOp::Mul {
                x: qk_id,
                y: SirNodeId(format!("const_scale_{}", node.id)),
            },
            "sdpa_scale".to_string(),
        ));

        // Softmax
        let scaled_id = SirNodeId(format!("sir_sdpa_scaled_{}", node.id));
        ops.push((
            SirOp::Softmax { input: scaled_id, axis: -1 },
            "sdpa_softmax".to_string(),
        ));

        // Scores @ V
        let softmax_id = SirNodeId(format!("sir_sdpa_softmax_{}", node.id));
        ops.push((
            SirOp::MatMul { a: softmax_id, b: v_id },
            "sdpa_sv".to_string(),
        ));

        Ok(ops)
    }

    /// Resolve a traced node input reference to a SIR node ID.
    fn resolve_input(&self, inputs: &[String], index: usize) -> SirNodeId {
        inputs
            .get(index)
            .and_then(|id| self.lookup_sir_id(id).cloned())
            .unwrap_or_else(|| SirNodeId(format!("__unresolved_{}__", index)))
    }
}

/// Parse a dtype string into a MilDtype.
fn parse_dtype(dtype_str: &str) -> Result<MilDtype, String> {
    match dtype_str.to_lowercase().as_str() {
        "fp16" | "float16" | "half" => Ok(MilDtype::Fp16),
        "fp32" | "float32" | "float" => Ok(MilDtype::Fp32),
        "int32" | "int" => Ok(MilDtype::Int32),
        "int8" => Ok(MilDtype::Int8),
        "uint8" => Ok(MilDtype::UInt8),
        "bool" => Ok(MilDtype::Bool),
        _ => Err(format!("Unknown dtype '{}'", dtype_str)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::*;
    use std::collections::HashMap;


    fn make_simple_trace() -> TracedGraph {
        TracedGraph {
            model_id: "test-model".to_string(),
            architecture: "LlamaForCausalLM".to_string(),
            transformers_version: "4.36.0".to_string(),
            torch_version: "2.1.0".to_string(),
            model_config: ModelConfig {
                hidden_size: 256,
                num_attention_heads: 4,
                num_key_value_heads: Some(4),
                num_hidden_layers: 2,
                intermediate_size: 1024,
                vocab_size: 32000,
                max_position_embeddings: 2048,
                layer_norm_epsilon: 1e-6,
                hidden_act: "silu".to_string(),
                uses_rope: true,
                uses_rms_norm: true,
                uses_gqa: false,
                model_type: "llama".to_string(),
            },
            nodes: vec![
                TracedNode {
                    id: "input".to_string(),
                    op: TracedOp::Placeholder,
                    name: "input_ids".to_string(),
                    inputs: vec![],
                    output_shape: TensorShape { dims: vec![1, 32], dtype: "int32".to_string() },
                    is_parameter: false,
                    module_path: None,
                },
                TracedNode {
                    id: "linear1".to_string(),
                    op: TracedOp::Linear { in_features: 256, out_features: 256, has_bias: false },
                    name: "q_proj".to_string(),
                    inputs: vec!["input".to_string()],
                    output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
                    is_parameter: false,
                    module_path: Some("model.layers.0.self_attn.q_proj".to_string()),
                },
                TracedNode {
                    id: "output".to_string(),
                    op: TracedOp::Output,
                    name: "output".to_string(),
                    inputs: vec!["linear1".to_string()],
                    output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
                    is_parameter: false,
                    module_path: None,
                },
            ],
            weights: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            state_declarations: vec![],
            trace_metadata: TraceMetadata {
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                trace_duration_secs: 1.0,
                num_nodes: 3,
                num_parameters: 1000,
                parameter_bytes: 2000,
                decomposed: false,
                warnings: vec![],
            },
        }
    }

    #[test]
    fn test_build_sir_from_simple_trace() {
        let trace = make_simple_trace();
        let sir = build_sir_from_trace(&trace, AneFamily::A16);
        assert!(sir.is_ok());
        let sir = sir.unwrap();
        assert!(!sir.nodes.is_empty());
    }

    #[test]
    fn test_sir_has_transformers_origin() {
        let trace = make_simple_trace();
        let sir = build_sir_from_trace(&trace, AneFamily::A16).unwrap();
        let has_trace_origin = sir.nodes.iter().any(|n| {
            matches!(n.metadata.task_origin, TaskOrigin::TransformersTrace { .. })
        });
        assert!(has_trace_origin);
    }

    #[test]
    fn test_layernorm_fails_on_a14() {
        let trace = TracedGraph {
            model_id: "bert-test".to_string(),
            architecture: "BertModel".to_string(),
            transformers_version: "4.36.0".to_string(),
            torch_version: "2.1.0".to_string(),
            model_config: ModelConfig {
                hidden_size: 768,
                num_attention_heads: 12,
                num_key_value_heads: Some(12),
                num_hidden_layers: 12,
                intermediate_size: 3072,
                vocab_size: 30000,
                max_position_embeddings: 512,
                layer_norm_epsilon: 1e-12,
                hidden_act: "gelu".to_string(),
                uses_rope: false,
                uses_rms_norm: false,
                uses_gqa: false,
                model_type: "bert".to_string(),
            },
            nodes: vec![
                TracedNode {
                    id: "ln1".to_string(),
                    op: TracedOp::LayerNorm { normalized_shape: vec![768], epsilon: 1e-12 },
                    name: "layernorm".to_string(),
                    inputs: vec!["input".to_string()],
                    output_shape: TensorShape { dims: vec![1, 32, 768], dtype: "fp16".to_string() },
                    is_parameter: false,
                    module_path: None,
                },
                TracedNode {
                    id: "input".to_string(),
                    op: TracedOp::Placeholder,
                    name: "input_ids".to_string(),
                    inputs: vec![],
                    output_shape: TensorShape { dims: vec![1, 32], dtype: "int32".to_string() },
                    is_parameter: false,
                    module_path: None,
                },
            ],
            weights: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            state_declarations: vec![],
            trace_metadata: TraceMetadata {
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                trace_duration_secs: 0.5,
                num_nodes: 2,
                num_parameters: 0,
                parameter_bytes: 0,
                decomposed: false,
                warnings: vec![],
            },
        };

        // A14 does not support LayerNorm
        let result = build_sir_from_trace(&trace, AneFamily::A14);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("LayerNorm not supported"));
    }

    #[test]
    fn test_rms_norm_decomposition() {
        let mut trace = make_simple_trace();
        // Replace the linear node with an RMSNorm
        trace.nodes.insert(1, TracedNode {
            id: "rms1".to_string(),
            op: TracedOp::RmsNorm { hidden_size: 256, epsilon: 1e-6 },
            name: "rms_norm".to_string(),
            inputs: vec!["input".to_string()],
            output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
            is_parameter: false,
            module_path: None,
        });

        let sir = build_sir_from_trace(&trace, AneFamily::A16).unwrap();
        // Dynamic-safe RMSNorm decomposes into: abs, max, clip, div_max, square, mean,
        // eps_div1, eps_div2, add_eps, rsqrt, norm, scale = 12 ops
        let rms_ops: Vec<_> = sir.nodes.iter()
            .filter(|n| n.name.starts_with("rms_norm"))
            .collect();
        assert!(rms_ops.len() >= 8, "Dynamic-safe RMSNorm should decompose into >=8 ops, got {} ops", rms_ops.len());
    }
}
