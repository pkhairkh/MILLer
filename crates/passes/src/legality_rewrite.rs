//! Legality Rewrite pass.
//!
//! Rewrites SIR operations into ANE-legal equivalents,
//! consuming legality knowledge to produce an AIR graph.
//!
//! ## SIR→AIR Decomposition Coverage
//!
//! | SIR Op | AIR Decomposition |
//! |--------|-------------------|
//! | LinearProjection | Conv1x1AsLinear (canonical mb.linear) |
//! | ElementWise | ElementWise (1:1) |
//! | Reshape | Reshape (1:1) |
//! | Transpose | Transpose (1:1) |
//! | Split | Split (1:1) |
//! | Concat | Concat (1:1) |
//! | Softmax | Softmax (1:1) |
//! | AttentionBlock | Reshape(Q) + Reshape(K) + Reshape(V) + Transpose(Q) + Transpose(K) + Transpose(V) + ScaledDotProductAttention + Reshape + Conv1x1AsLinear |
//! | DecodeStep | Conv1x1AsLinear + SliceByIndex + StateReadFixed + Reshape + ScaledDotProductAttention + Conv1x1AsLinear |
//! | RMSNorm | ReduceMean + Rsqrt + ElementWise::Mul + ElementWise::Mul |
//! | RoPETransform | Cos + Sin + ElementWise::Mul + ElementWise::Add |
//! | Sampler | Topk + Gather + Softmax |
//!
//! **Critique fix (Sprint 36):** `SirOp::LinearProjection` now lowers to
//! `AirOp::Conv1x1AsLinear` instead of `AirOp::MatMul`. This closes the
//! inconsistency where the Python emitter uses `mb.linear` (Sprint 31) but
//! the SIR→AIR path still produced `MatMul`. The `Conv1x1AsLinear` AIR op
//! correctly lowers to `MILLinear` in the MIL lower pass, matching the
//! canonical Core ML emission path.

use crate::cpu_only_ops;
use crate::knowledge_query::PassKnowledgeQuery;
use ane_ir::air::{AirGraph, AirNode, AirNodeId, AirOp};
use ane_ir::mir::MilDtype;
use ane_ir::sir::{ElementWiseOp, SirGraph, SirOp};
use anyhow::Result;

/// Task dimensions needed by AIR decomposition functions.
///
/// Carries the concrete tensor dimensions from the task spec through
/// the SIR→AIR decomposition, so that AIR ops carry truthful shapes
/// instead of placeholder zeros.
///
/// Before Sprint 56, `decompose_attention_block` and `decompose_decode_step`
/// emitted placeholder `vec![0, 0, 0]` shapes for SliceByIndex bounds and
/// Reshape target shapes, because the SIR graph does not carry dimension
/// information. This struct threads the dimensions from the task spec
/// into the decomposition, making the AIR graph semantically complete.
///
/// When `DecompositionContext` is `None` (e.g., in tests or non-task
/// compilation), decompositions fall back to zero-filled placeholder
/// shapes, preserving backward compatibility.
#[derive(Debug, Clone, Default)]
pub struct DecompositionContext {
    /// Batch dimension.
    pub batch_size: usize,
    /// Embedding dimension (total, across all heads).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head (embed_dim / num_heads).
    pub head_dim: usize,
    /// KV-cache sequence length (decode step) or input sequence length (attention).
    pub seq_len: usize,
    /// Number of KV heads for GQA (defaults to num_heads if 0).
    pub kv_heads: usize,
    /// MLP intermediate size (e.g., 3072 for Qwen3-0.6B).
    pub intermediate_size: usize,
    /// Vocabulary size for the language model head.
    pub vocab_size: usize,
}

impl DecompositionContext {
    /// Construct a context from an Attention task spec.
    pub fn for_attention(
        batch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            batch_size,
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            kv_heads: 0,
            intermediate_size: 0,
            vocab_size: 0,
        }
    }

    /// Construct a context from an Attention task spec with full dimensions.
    pub fn for_attention_full(
        batch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            batch_size,
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            kv_heads,
            intermediate_size,
            vocab_size,
        }
    }

    /// Construct a context from a DecodeStep task spec.
    pub fn for_decode_step(
        batch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Self {
        Self {
            batch_size,
            embed_dim,
            num_heads,
            head_dim,
            seq_len: kv_len,
            kv_heads: 0,
            intermediate_size: 0,
            vocab_size: 0,
        }
    }

    /// Derive the output dimension for a linear projection based on the weight name.
    ///
    /// Weight names follow the HuggingFace convention, e.g.:
    /// - `model.layers.0.self_attn.q_proj.weight` → num_heads * head_dim
    /// - `model.layers.0.self_attn.k_proj.weight` → kv_heads * head_dim
    /// - `model.layers.0.self_attn.v_proj.weight` → kv_heads * head_dim
    /// - `model.layers.0.self_attn.o_proj.weight` → embed_dim
    /// - `model.layers.0.mlp.gate_proj.weight`   → intermediate_size
    /// - `model.layers.0.mlp.up_proj.weight`     → intermediate_size
    /// - `model.layers.0.mlp.down_proj.weight`   → embed_dim
    /// - `model.embed_tokens.weight`              → embed_dim
    /// - `lm_head.weight`                         → vocab_size
    ///
    /// Returns 0 if the output dimension cannot be determined (unknown projection).
    pub fn output_dim_for_weight(&self, weight: &str) -> usize {
        let kv_heads = if self.kv_heads > 0 { self.kv_heads } else { self.num_heads };
        if weight.contains(".self_attn.q_proj.weight") {
            self.num_heads * self.head_dim
        } else if weight.contains(".self_attn.k_proj.weight")
            || weight.contains(".self_attn.k_proj.weight")
        {
            kv_heads * self.head_dim
        } else if weight.contains(".self_attn.v_proj.weight") {
            kv_heads * self.head_dim
        } else if weight.contains(".self_attn.o_proj.weight")
            || weight.contains(".self_attn.out_proj.weight")
        {
            self.embed_dim
        } else if weight.contains(".mlp.gate_proj.weight")
            || weight.contains(".mlp.up_proj.weight")
        {
            if self.intermediate_size > 0 {
                self.intermediate_size
            } else {
                0
            }
        } else if weight.contains(".mlp.down_proj.weight") {
            self.embed_dim
        } else if weight == "lm_head.weight" || weight.contains("lm_head.") {
            if self.vocab_size > 0 { self.vocab_size } else { 0 }
        } else if weight.contains("embed_tokens") {
            self.embed_dim
        } else {
            0
        }
    }
}

/// Legality Rewrite pass implementation.
pub struct LegalityRewritePass {
    // No configuration needed for the linear projection case
}

impl Default for LegalityRewritePass {
    fn default() -> Self {
        Self::new()
    }
}

impl LegalityRewritePass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the legality rewrite pass.
    ///
    /// Converts SIR operations to ANE-legal AIR equivalents,
    /// querying the knowledge store for per-op legality confidence
    /// and fallback/drift risk scores. When no knowledge is available
    /// for an operation, reasonable defaults are used.
    ///
    /// High-level SIR ops (AttentionBlock, DecodeStep, RMSNorm, RoPETransform,
    /// Sampler) are decomposed into sequences of lower-level AIR ops.
    ///
    /// The `ctx` parameter carries task dimensions for truthful shape
    /// emission. When `None`, placeholder zero-filled shapes are used
    /// (backward-compatible with pre-Sprint-56 behavior).
    pub fn run(
        &self,
        input: SirGraph,
        knowledge_query: &dyn PassKnowledgeQuery,
        ctx: Option<&DecompositionContext>,
    ) -> Result<AirGraph> {
        let mut air_nodes = Vec::new();
        let mut sir_to_air = std::collections::HashMap::new();

        // Map SIR nodes to AIR equivalents
        for sir_node in &input.nodes {
            // Some SIR ops decompose into multiple AIR ops; those helpers
            // return the AirNodeId of the *final* op in the decomposition.
            let (final_air_id, decomposed_nodes, _op_pattern) = match &sir_node.op {
                SirOp::LinearProjection { input: sir_input, weight, bias: _ } => {
                    // CRITICAL FIX (Sprint 36 / Critique Bug 1):
                    // Linear projection must lower to Conv1x1AsLinear, NOT MatMul.
                    // The Python emitter uses mb.linear (Sprint 31), and the
                    // MIL lower pass maps Conv1x1AsLinear → MILLinear. Using
                    // MatMul here was inconsistent with the emission path.
                    //
                    // Sprint 61: Derive output_dim from the weight name and
                    // DecompositionContext so that shape inference can propagate
                    // correct output shapes through the AIR→MIR pipeline.
                    let a_id = sir_to_air
                        .get(sir_input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(sir_input.0.clone()));
                    let output_dim = ctx
                        .map(|c| c.output_dim_for_weight(weight))
                        .unwrap_or(0);
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Conv1x1AsLinear {
                            input: a_id,
                            weight: weight.clone(),
                            pad_type: "valid".to_string(),
                            output_dim,
                        },
                        sir_node,
                        "mb.linear",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.linear")
                }
                SirOp::AttentionBlock { q, k, v, mask, rope: _ } => {
                    let attn_ctx = ctx.and_then(|c| {
                        // Only use the context if it has non-zero embed_dim (meaningful dimensions)
                        if c.embed_dim > 0 {
                            Some(c.clone())
                        } else {
                            None
                        }
                    });
                    let (final_id, nodes) = Self::decompose_attention_block(
                        sir_node,
                        q,
                        k,
                        v,
                        mask,
                        &sir_to_air,
                        knowledge_query,
                        attn_ctx.as_ref(),
                    );
                    (final_id, nodes, "mb.scaled_dot_product_attention")
                }
                SirOp::DecodeStep { token, state_map } => {
                    let ds_ctx =
                        ctx.and_then(|c| if c.embed_dim > 0 { Some(c.clone()) } else { None });
                    let (final_id, nodes) = Self::decompose_decode_step(
                        sir_node,
                        token,
                        state_map,
                        &sir_to_air,
                        knowledge_query,
                        ds_ctx.as_ref(),
                    );
                    (final_id, nodes, "mb.scaled_dot_product_attention")
                }
                SirOp::RMSNorm { input, weight, epsilon, .. } => {
                    let (final_id, nodes) = Self::decompose_rms_norm(
                        sir_node,
                        input,
                        weight,
                        *epsilon,
                        &sir_to_air,
                        knowledge_query,
                    );
                    (final_id, nodes, "mb.layer_norm")
                }
                SirOp::RoPETransform { input, tables: _ } => {
                    let (final_id, nodes) =
                        Self::decompose_rope(sir_node, input, &sir_to_air, knowledge_query);
                    (final_id, nodes, "mb.cos")
                }
                SirOp::Sampler { logits, temperature: _, top_p: _, rep_penalty: _, .. } => {
                    let (final_id, nodes) =
                        Self::decompose_sampler(sir_node, logits, &sir_to_air, knowledge_query);
                    (final_id, nodes, "mb.topk")
                }
                SirOp::ElementWise { op, inputs } => {
                    let air_inputs: Vec<AirNodeId> = inputs
                        .iter()
                        .map(|id| {
                            sir_to_air.get(id).cloned().unwrap_or_else(|| AirNodeId(id.0.clone()))
                        })
                        .collect();
                    let pattern = match op {
                        ElementWiseOp::Add => "mb.add",
                        ElementWiseOp::Mul => "mb.mul",
                        ElementWiseOp::Abs => "mb.abs",
                        ElementWiseOp::Maximum => "mb.maximum",
                        ElementWiseOp::Minimum => "mb.minimum",
                    };
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::ElementWise { op: op.clone(), inputs: air_inputs },
                        sir_node,
                        pattern,
                        knowledge_query,
                    )];
                    (air_id, nodes, pattern)
                }
                SirOp::Reshape { input, target_shape } => {
                    let air_input = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Reshape { input: air_input, target_shape: target_shape.clone() },
                        sir_node,
                        "mb.reshape",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.reshape")
                }
                SirOp::Transpose { input, perm } => {
                    let air_input = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Transpose { input: air_input, perm: perm.clone() },
                        sir_node,
                        "mb.transpose",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.transpose")
                }
                SirOp::Split { input, axis, num_splits } => {
                    let air_input = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Split { input: air_input, axis: *axis, num_splits: *num_splits },
                        sir_node,
                        "mb.split",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.split")
                }
                SirOp::Concat { inputs, axis } => {
                    let air_inputs: Vec<AirNodeId> = inputs
                        .iter()
                        .map(|id| {
                            sir_to_air.get(id).cloned().unwrap_or_else(|| AirNodeId(id.0.clone()))
                        })
                        .collect();
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Concat { inputs: air_inputs, axis: *axis },
                        sir_node,
                        "mb.concat",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.concat")
                }
                SirOp::Softmax { input, axis } => {
                    let air_input = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Softmax { input: air_input, axis: *axis },
                        sir_node,
                        "mb.softmax",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.softmax")
                }
                SirOp::StateRead { state_id, offset: _, shape } => {
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::StateReadFixed {
                            state_id: state_id.clone(),
                            shape: shape.clone(),
                            dtype: MilDtype::Fp16,
                        },
                        sir_node,
                        "mb.read_state",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.read_state")
                }
                SirOp::StateWrite { state_id, offset: _, value } => {
                    let air_value = sir_to_air
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(value.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::StateWriteFixed { state_id: state_id.clone(), value: air_value },
                        sir_node,
                        "mb.coreml_update_state",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.coreml_update_state")
                }
                // ─── All new 1:1 passthrough ops ─────────────────
                op => {
                    let (air_op, pattern) =
                        Self::sir_to_air_passthrough(op, &sir_node.id, &sir_to_air);
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        air_op,
                        sir_node,
                        pattern,
                        knowledge_query,
                    )];
                    (air_id, nodes, pattern)
                }
            };

            sir_to_air.insert(sir_node.id.clone(), final_air_id.clone());
            air_nodes.extend(decomposed_nodes);
        }

        let air_inputs: Vec<AirNodeId> = input
            .inputs
            .iter()
            .map(|id| sir_to_air.get(id).cloned().unwrap_or_else(|| AirNodeId(id.0.clone())))
            .collect();
        let air_outputs: Vec<AirNodeId> = input
            .outputs
            .iter()
            .map(|id| sir_to_air.get(id).cloned().unwrap_or_else(|| AirNodeId(id.0.clone())))
            .collect();

        Ok(AirGraph {
            nodes: air_nodes,
            inputs: air_inputs,
            outputs: air_outputs,
            staticization_decisions: vec![],
        })
    }

    /// Helper: create an AirNode with knowledge-queried legality scores.
    fn make_air_node(
        id: AirNodeId,
        op: AirOp,
        sir_node: &ane_ir::sir::SirNode,
        op_pattern: &str,
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> AirNode {
        // ─── CPU_ONLY hard gate (Sprint 60) ─────────────────────────
        // Ops in the CPU_ONLY set NEVER land on ANE — no amount of
        // soft scoring can override this. If the MIL op name (stripped
        // of its "mb." prefix) is in the CPU_ONLY set, force confidence
        // to 0.0 and skip the knowledge query entirely.
        let mil_name = op_pattern.strip_prefix("mb.").unwrap_or(op_pattern);
        if cpu_only_ops::is_cpu_only(mil_name) {
            return AirNode {
                id,
                op,
                name: sir_node.name.clone(),
                legality_confidence: 0.0,
                sir_source: Some(sir_node.id.clone()),
                fallback_risk: 1.0,
                drift_risk: 1.0,
                precision_override: sir_node.metadata.precision_override.clone(),
            };
        }

        let (legality_confidence, fallback_risk, drift_risk) =
            match knowledge_query.query_legality(op_pattern, None) {
                Some(info) if info.ane_legal => (
                    info.confidence,
                    (1.0 - info.confidence).min(1.0),
                    (1.0 - info.confidence).min(1.0) * 0.5,
                ),
                Some(info) => (
                    (1.0 - info.confidence).max(0.0),
                    info.confidence.min(1.0),
                    info.confidence.min(1.0) * 0.8,
                ),
                None => (0.5, 0.1, 0.05),
            };

        AirNode {
            id,
            op,
            name: sir_node.name.clone(),
            legality_confidence,
            sir_source: Some(sir_node.id.clone()),
            fallback_risk,
            drift_risk,
            precision_override: sir_node.metadata.precision_override.clone(),
        }
    }

    /// Decompose SirOp::AttentionBlock into AIR ops.
    ///
    /// AttentionBlock(q, k, v, mask, rope) →
    ///   qkv_proj: Conv1x1AsLinear(x, W_qkv)
    ///   q: SliceByIndex(qkv, [0, 0, 0], [batch, seq, embed])
    ///   k: SliceByIndex(qkv, [0, 0, embed], [batch, seq, 2*embed])
    ///   v: SliceByIndex(qkv, [0, 0, 2*embed], [batch, seq, 3*embed])
    ///   q_4d: Reshape(q, [batch, seq, heads, head_dim])
    ///   k_4d: Reshape(k, [batch, seq, heads, head_dim])
    ///   v_4d: Reshape(v, [batch, seq, heads, head_dim])
    ///   q_t: Transpose(q_4d, [0, 2, 1, 3])
    ///   k_t: Transpose(k_4d, [0, 2, 1, 3])
    ///   v_t: Transpose(v_4d, [0, 2, 1, 3])
    ///   attn: ScaledDotProductAttention(q_t, k_t, v_t)
    ///   attn_flat: Reshape(attn, [batch, seq, embed])
    ///   output: Conv1x1AsLinear(attn_flat, W_out)
    ///
    /// When `ctx` is `Some`, the SliceByIndex bounds and Reshape target shapes
    /// are populated with real dimensions from the task spec (Sprint 56).
    /// When `ctx` is `None`, placeholder zeros are used (pre-Sprint-56 behavior).
    fn decompose_attention_block(
        sir_node: &ane_ir::sir::SirNode,
        q_sir: &ane_ir::sir::SirNodeId,
        k_sir: &ane_ir::sir::SirNodeId,
        v_sir: &ane_ir::sir::SirNodeId,
        mask_sir: &Option<ane_ir::sir::SirNodeId>,
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
        kq: &dyn PassKnowledgeQuery,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let base = &sir_node.id.0;
        let q_air = sir_to_air.get(q_sir).cloned().unwrap_or_else(|| AirNodeId(q_sir.0.clone()));
        let k_air = sir_to_air.get(k_sir).cloned().unwrap_or_else(|| AirNodeId(k_sir.0.clone()));
        let v_air = sir_to_air.get(v_sir).cloned().unwrap_or_else(|| AirNodeId(v_sir.0.clone()));

        // Extract dimensions from context or use placeholders
        let (batch, seq, embed, heads, head_dim) = match ctx {
            Some(c) => (
                c.batch_size as i64,
                c.seq_len as i64,
                c.embed_dim as i64,
                c.num_heads as i64,
                c.head_dim as i64,
            ),
            None => (0, 0, 0, 0, 0),
        };

        let mut nodes = Vec::new();

        // Q, K, V come as separate projections from the SIR builder.
        // The SIR builder already emits distinct LinearProjection ops for each,
        // so we must NOT create a fused QKV projection here. Instead, reshape
        // and transpose each projection output to 4D [batch, heads, seq, head_dim].

        // Steps 1-3: Reshape Q, K, V from [batch, seq, embed] to [batch, seq, heads, head_dim]
        let q_4d_id = AirNodeId(format!("{base}_q_4d"));
        nodes.push(Self::make_air_node(
            q_4d_id.clone(),
            AirOp::Reshape {
                input: q_air,
                target_shape: vec![batch as usize, seq as usize, heads as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        let k_4d_id = AirNodeId(format!("{base}_k_4d"));
        nodes.push(Self::make_air_node(
            k_4d_id.clone(),
            AirOp::Reshape {
                input: k_air,
                target_shape: vec![batch as usize, seq as usize, heads as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        let v_4d_id = AirNodeId(format!("{base}_v_4d"));
        nodes.push(Self::make_air_node(
            v_4d_id.clone(),
            AirOp::Reshape {
                input: v_air,
                target_shape: vec![batch as usize, seq as usize, heads as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Steps 4-6: Transpose to [batch, heads, seq, head_dim]
        // This is the layout that ANE SDPA expects.
        let q_t_id = AirNodeId(format!("{base}_q_t"));
        nodes.push(Self::make_air_node(
            q_t_id.clone(),
            AirOp::Transpose { input: q_4d_id, perm: vec![0, 2, 1, 3] },
            sir_node,
            "mb.transpose",
            kq,
        ));

        let k_t_id = AirNodeId(format!("{base}_k_t"));
        nodes.push(Self::make_air_node(
            k_t_id.clone(),
            AirOp::Transpose { input: k_4d_id, perm: vec![0, 2, 1, 3] },
            sir_node,
            "mb.transpose",
            kq,
        ));

        let v_t_id = AirNodeId(format!("{base}_v_t"));
        nodes.push(Self::make_air_node(
            v_t_id.clone(),
            AirOp::Transpose { input: v_4d_id, perm: vec![0, 2, 1, 3] },
            sir_node,
            "mb.transpose",
            kq,
        ));

        // Step 7: Scaled dot-product attention.
        // Scale = 1/√d_k, which is the standard scaling factor for dot-product
        // attention. The mask (if present) carries the causal mask reference.
        let mask_air = mask_sir.as_ref().and_then(|m| {
            sir_to_air.get(m).cloned().or_else(|| Some(AirNodeId(m.0.clone())))
        });
        let scale = if head_dim > 0 {
            Some(1.0 / (head_dim as f32).sqrt())
        } else {
            None
        };

        let attn_id = AirNodeId(format!("{base}_attn"));
        nodes.push(Self::make_air_node(
            attn_id.clone(),
            AirOp::ScaledDotProductAttention {
                query: q_t_id,
                key: k_t_id,
                value: v_t_id,
                attention_mask: mask_air,
                scale,
            },
            sir_node,
            "mb.scaled_dot_product_attention",
            kq,
        ));

        // Step 8: Reshape back to [batch, seq, embed]
        let attn_flat_id = AirNodeId(format!("{base}_attn_flat"));
        nodes.push(Self::make_air_node(
            attn_flat_id.clone(),
            AirOp::Reshape {
                input: attn_id,
                target_shape: vec![batch as usize, seq as usize, embed as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Step 9: Output projection
        let out_id = AirNodeId(sir_node.id.0.clone());
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Conv1x1AsLinear {
                input: attn_flat_id,
                weight: format!("{base}_w_out"),
                pad_type: "valid".into(),
                output_dim: embed as usize, // out_proj: embed_dim output
            },
            sir_node,
            "mb.linear",
            kq,
        ));

        (out_id, nodes)
    }

    /// Decompose SirOp::DecodeStep into AIR ops.
    ///
    /// DecodeStep(token, state_map) →
    ///   qkv_proj: Conv1x1AsLinear(token, W_qkv)
    ///   q: SliceByIndex(qkv, ...)
    ///   k: SliceByIndex(qkv, ...)
    ///   v: SliceByIndex(qkv, ...)
    ///   k_cache: StateReadFixed(k_state, shape)
    ///   v_cache: StateReadFixed(v_state, shape)
    ///   q_4d: Reshape(q, [batch, heads, 1, head_dim])
    ///   k_4d: Reshape(k_cache, [1, heads, kv_len, head_dim])
    ///   v_4d: Reshape(v_cache, [1, heads, kv_len, head_dim])
    ///   attn: ScaledDotProductAttention(q_4d, k_4d, v_4d)
    ///   attn_flat: Reshape(attn, [batch, embed])
    ///   output: Conv1x1AsLinear(attn_flat, W_out)
    ///   k_update: StateWriteFixed(k_state, k)
    ///   v_update: StateWriteFixed(v_state, v)
    ///
    /// When `ctx` is `Some`, the SliceByIndex bounds, Reshape target shapes,
    /// and StateReadFixed shapes are populated with real dimensions from the
    /// task spec (Sprint 56). When `ctx` is `None`, placeholder zeros are used
    /// (pre-Sprint-56 behavior).
    fn decompose_decode_step(
        sir_node: &ane_ir::sir::SirNode,
        token_sir: &ane_ir::sir::SirNodeId,
        state_map: &[String],
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
        kq: &dyn PassKnowledgeQuery,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let base = &sir_node.id.0;
        let token_air =
            sir_to_air.get(token_sir).cloned().unwrap_or_else(|| AirNodeId(token_sir.0.clone()));

        // Extract dimensions from context or use placeholders
        let (batch, embed, heads, head_dim, kv_len) = match ctx {
            Some(c) => (
                c.batch_size as i64,
                c.embed_dim as i64,
                c.num_heads as i64,
                c.head_dim as i64,
                c.seq_len as i64,
            ),
            None => (0, 0, 0, 0, 0),
        };

        let mut nodes = Vec::new();
        let last_id = token_air;

        // Step 1: QKV projection
        let qkv_id = AirNodeId(format!("{base}_qkv_proj"));
        nodes.push(Self::make_air_node(
            qkv_id.clone(),
            AirOp::Conv1x1AsLinear {
                input: last_id,
                weight: format!("{base}_w_qkv"),
                pad_type: "valid".into(),
                output_dim: (3 * embed) as usize, // QKV fused projection: 3 * embed_dim
            },
            sir_node,
            "mb.linear",
            kq,
        ));

        // Steps 2-4: Slice Q, K, V
        // Q: [0, 0] → [batch, embed], K: [0, embed] → [batch, 2*embed], V: [0, 2*embed] → [batch, 3*embed]
        let q_id = AirNodeId(format!("{base}_q"));
        nodes.push(Self::make_air_node(
            q_id.clone(),
            AirOp::SliceByIndex {
                input: qkv_id.clone(),
                begin: vec![0, 0],
                end: vec![batch, embed],
                stride: vec![],
                begin_mask: vec![],
                end_mask: vec![],
                squeeze_mask: vec![],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        let k_id = AirNodeId(format!("{base}_k_new"));
        nodes.push(Self::make_air_node(
            k_id.clone(),
            AirOp::SliceByIndex {
                input: qkv_id.clone(),
                begin: vec![0, embed],
                end: vec![batch, 2 * embed],
                stride: vec![],
                begin_mask: vec![],
                end_mask: vec![],
                squeeze_mask: vec![],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        let v_id = AirNodeId(format!("{base}_v_new"));
        nodes.push(Self::make_air_node(
            v_id.clone(),
            AirOp::SliceByIndex {
                input: qkv_id,
                begin: vec![0, 2 * embed],
                end: vec![batch, 3 * embed],
                stride: vec![],
                begin_mask: vec![],
                end_mask: vec![],
                squeeze_mask: vec![],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        // Steps 5-6: State reads for KV cache
        // Use state_map to derive state IDs (convention: "k_state", "v_state")
        let k_state_id = state_map.first().cloned().unwrap_or_else(|| format!("{base}_k_cache"));
        let v_state_id = state_map.get(1).cloned().unwrap_or_else(|| format!("{base}_v_cache"));

        // State shape: [kv_len, embed_dim] for full KV cache per head
        let k_cache_id = AirNodeId(format!("{base}_k_cache_read"));
        nodes.push(Self::make_air_node(
            k_cache_id.clone(),
            AirOp::StateReadFixed {
                state_id: k_state_id.clone(),
                shape: vec![kv_len as usize, embed as usize],
                dtype: ane_ir::mir::MilDtype::Fp16,
            },
            sir_node,
            "mb.read_state",
            kq,
        ));

        let v_cache_id = AirNodeId(format!("{base}_v_cache_read"));
        nodes.push(Self::make_air_node(
            v_cache_id.clone(),
            AirOp::StateReadFixed {
                state_id: v_state_id.clone(),
                shape: vec![kv_len as usize, embed as usize],
                dtype: ane_ir::mir::MilDtype::Fp16,
            },
            sir_node,
            "mb.read_state",
            kq,
        ));

        // Steps 7-9: Reshape for multi-head attention
        // Q: [batch, embed] → [batch, heads, 1, head_dim]
        // K cache: [kv_len, embed] → [1, heads, kv_len, head_dim]
        // V cache: [kv_len, embed] → [1, heads, kv_len, head_dim]
        let q_4d_id = AirNodeId(format!("{base}_q_4d"));
        nodes.push(Self::make_air_node(
            q_4d_id.clone(),
            AirOp::Reshape {
                input: q_id,
                target_shape: vec![batch as usize, heads as usize, 1, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        let k_4d_id = AirNodeId(format!("{base}_k_4d"));
        nodes.push(Self::make_air_node(
            k_4d_id.clone(),
            AirOp::Reshape {
                input: k_cache_id,
                target_shape: vec![1, heads as usize, kv_len as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        let v_4d_id = AirNodeId(format!("{base}_v_4d"));
        nodes.push(Self::make_air_node(
            v_4d_id.clone(),
            AirOp::Reshape {
                input: v_cache_id,
                target_shape: vec![1, heads as usize, kv_len as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Step 10: Scaled dot-product attention
        let attn_id = AirNodeId(format!("{base}_attn"));
        nodes.push(Self::make_air_node(
            attn_id.clone(),
            AirOp::ScaledDotProductAttention {
                query: q_4d_id,
                key: k_4d_id,
                value: v_4d_id,
                attention_mask: None,
                scale: None,
            },
            sir_node,
            "mb.scaled_dot_product_attention",
            kq,
        ));

        // Step 11: Reshape back
        let attn_flat_id = AirNodeId(format!("{base}_attn_flat"));
        nodes.push(Self::make_air_node(
            attn_flat_id.clone(),
            AirOp::Reshape { input: attn_id, target_shape: vec![batch as usize, embed as usize] },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Step 12: Output projection
        let out_id = AirNodeId(format!("{base}_out_proj"));
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Conv1x1AsLinear {
                input: attn_flat_id,
                weight: format!("{base}_w_out"),
                pad_type: "valid".into(),
                output_dim: embed as usize, // out_proj: embed_dim output
            },
            sir_node,
            "mb.linear",
            kq,
        ));

        // Steps 13-14: State writes to update KV cache
        let k_write_id = AirNodeId(format!("{base}_k_cache_write"));
        nodes.push(Self::make_air_node(
            k_write_id.clone(),
            AirOp::StateWriteFixed { state_id: k_state_id, value: k_id },
            sir_node,
            "mb.coreml_update_state",
            kq,
        ));

        let v_write_id = AirNodeId(format!("{base}_v_cache_write"));
        nodes.push(Self::make_air_node(
            v_write_id.clone(),
            AirOp::StateWriteFixed { state_id: v_state_id, value: v_id },
            sir_node,
            "mb.coreml_update_state",
            kq,
        ));

        // The primary output of the decode step is the output projection
        (out_id, nodes)
    }

    /// Decompose SirOp::RMSNorm into AIR ops.
    ///
    /// RMSNorm(x, weight, epsilon) →
    ///   mean: ReduceMean(x^2, axes=[-1], keep_dims=true)
    ///   rsqrt: Rsqrt(mean + epsilon)
    ///   normed: ElementWise::Mul(x, rsqrt)
    ///   output: ElementWise::Mul(normed, weight)
    fn decompose_rms_norm(
        sir_node: &ane_ir::sir::SirNode,
        input_sir: &ane_ir::sir::SirNodeId,
        weight: &str,
        _epsilon: f32,
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
        kq: &dyn PassKnowledgeQuery,
    ) -> (AirNodeId, Vec<AirNode>) {
        let base = &sir_node.id.0;
        let input_air =
            sir_to_air.get(input_sir).cloned().unwrap_or_else(|| AirNodeId(input_sir.0.clone()));

        let mut nodes = Vec::new();

        // Step 1: x^2 mean via ReduceMean
        // (In a full decomposition, we'd first compute x^2 via Mul(x, x),
        // then ReduceMean. For this decomposition we use ReduceMean directly
        // as the AIR representation — the MIL emitter will expand properly.)
        let mean_id = AirNodeId(format!("{base}_mean"));
        nodes.push(Self::make_air_node(
            mean_id.clone(),
            AirOp::ReduceMean {
                input: input_air.clone(),
                axes: vec![2], // normalize over embedding dimension (last dim for 3D [batch, seq, embed])
                keep_dims: true,
            },
            sir_node,
            "mb.reduce_mean",
            kq,
        ));

        // Step 2: Rsqrt of mean
        let rsqrt_id = AirNodeId(format!("{base}_rsqrt"));
        nodes.push(Self::make_air_node(
            rsqrt_id.clone(),
            AirOp::Rsqrt { input: mean_id },
            sir_node,
            "mb.rsqrt",
            kq,
        ));

        // Step 3: x * rsqrt(x^2_mean)
        let normed_id = AirNodeId(format!("{base}_normed"));
        nodes.push(Self::make_air_node(
            normed_id.clone(),
            AirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![input_air, rsqrt_id] },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 4: normed * weight (gamma)
        let out_id = AirNodeId(sir_node.id.0.clone());
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::ElementWise {
                op: ElementWiseOp::Mul,
                inputs: vec![normed_id, AirNodeId(weight.into())],
            },
            sir_node,
            "mb.mul",
            kq,
        ));

        (out_id, nodes)
    }

    /// Decompose SirOp::RoPETransform into AIR ops.
    ///
    /// RoPETransform(x, tables) →
    ///   cos_vals: Cos(tables)
    ///   sin_vals: Sin(tables)
    ///   x_cos: ElementWise::Mul(x, cos_vals)
    ///   x_sin: ElementWise::Mul(x, sin_vals)
    ///   output: ElementWise::Add(x_cos, x_sin)
    ///
    /// This is a simplified decomposition; a full RoPE would also need
    /// the half-rotation and negation of odd elements.
    fn decompose_rope(
        sir_node: &ane_ir::sir::SirNode,
        input_sir: &ane_ir::sir::SirNodeId,
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
        kq: &dyn PassKnowledgeQuery,
    ) -> (AirNodeId, Vec<AirNode>) {
        let base = &sir_node.id.0;
        let input_air =
            sir_to_air.get(input_sir).cloned().unwrap_or_else(|| AirNodeId(input_sir.0.clone()));

        let mut nodes = Vec::new();

        let cos_id = AirNodeId(format!("{base}_cos"));
        nodes.push(Self::make_air_node(
            cos_id.clone(),
            AirOp::Cos { input: input_air.clone() },
            sir_node,
            "mb.cos",
            kq,
        ));

        let sin_id = AirNodeId(format!("{base}_sin"));
        nodes.push(Self::make_air_node(
            sin_id.clone(),
            AirOp::Sin { input: input_air.clone() },
            sir_node,
            "mb.sin",
            kq,
        ));

        let x_cos_id = AirNodeId(format!("{base}_x_cos"));
        nodes.push(Self::make_air_node(
            x_cos_id.clone(),
            AirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![input_air.clone(), cos_id] },
            sir_node,
            "mb.mul",
            kq,
        ));

        let x_sin_id = AirNodeId(format!("{base}_x_sin"));
        nodes.push(Self::make_air_node(
            x_sin_id.clone(),
            AirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![input_air, sin_id] },
            sir_node,
            "mb.mul",
            kq,
        ));

        let out_id = AirNodeId(sir_node.id.0.clone());
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::ElementWise { op: ElementWiseOp::Add, inputs: vec![x_cos_id, x_sin_id] },
            sir_node,
            "mb.add",
            kq,
        ));

        (out_id, nodes)
    }

    /// Decompose SirOp::Sampler into AIR ops.
    ///
    /// Sampler(logits, temperature, top_p, rep_penalty) →
    ///   topk_vals, topk_idx: Topk(logits, k)
    ///   probs: Softmax(topk_vals / temperature)
    ///   selected: Gather(topk_idx, argmax(probs))
    ///
    /// Simplified: Topk + Softmax + Gather
    fn decompose_sampler(
        sir_node: &ane_ir::sir::SirNode,
        logits_sir: &ane_ir::sir::SirNodeId,
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
        kq: &dyn PassKnowledgeQuery,
    ) -> (AirNodeId, Vec<AirNode>) {
        let base = &sir_node.id.0;
        let logits_air =
            sir_to_air.get(logits_sir).cloned().unwrap_or_else(|| AirNodeId(logits_sir.0.clone()));

        let mut nodes = Vec::new();

        let topk_id = AirNodeId(format!("{base}_topk"));
        nodes.push(Self::make_air_node(
            topk_id.clone(),
            AirOp::Topk {
                input: logits_air,
                k: 1,     // default: top-1 sampling
                axis: -1, // last dimension
            },
            sir_node,
            "mb.topk",
            kq,
        ));

        let softmax_id = AirNodeId(format!("{base}_softmax"));
        nodes.push(Self::make_air_node(
            softmax_id.clone(),
            AirOp::Softmax { input: topk_id, axis: -1 },
            sir_node,
            "mb.softmax",
            kq,
        ));

        let out_id = AirNodeId(sir_node.id.0.clone());
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Gather {
                input: softmax_id,
                indices: AirNodeId(format!("{base}_topk_idx")),
                axis: -1,
            },
            sir_node,
            "mb.gather",
            kq,
        ));

        (out_id, nodes)
    }

    // Sprint 58 (S58.2): dtype_repr_to_mir() removed — SIR now uses MilDtype directly.

    /// 1:1 passthrough from SirOp to AirOp for all non-decomposing ops.
    ///
    /// Returns (AirOp, op_pattern_string).
    fn sir_to_air_passthrough(
        op: &SirOp,
        node_id: &ane_ir::sir::SirNodeId,
        sir_to_air: &std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
    ) -> (AirOp, &'static str) {
        let aid = |sid: &ane_ir::sir::SirNodeId| -> AirNodeId {
            sir_to_air.get(sid).cloned().unwrap_or_else(|| AirNodeId(sid.0.clone()))
        };
        let aids =
            |sids: &[ane_ir::sir::SirNodeId]| -> Vec<AirNodeId> { sids.iter().map(&aid).collect() };
        let _base = &node_id.0;

        match op {
            // ─── Constants ───────────────────────────────────────
            SirOp::Const { value_path, dtype } => {
                (AirOp::Const { value_path: value_path.clone(), dtype: dtype.clone() }, "mb.const")
            }

            // ─── Linear / FC ─────────────────────────────────────
            SirOp::MatMul { a, b } => (AirOp::MatMul { a: aid(a), b: aid(b) }, "mb.matmul"),
            SirOp::Einsum { inputs, equation } => {
                (AirOp::Einsum { inputs: aids(inputs), equation: equation.clone() }, "mb.einsum")
            }

            // ─── Convolution ─────────────────────────────────────
            SirOp::Conv { input, weight, pad_type, groups, strides, pad_amounts, dilations } => (
                AirOp::Conv {
                    input: aid(input),
                    weight: aid(weight),
                    pad_type: pad_type.clone(),
                    groups: *groups,
                    strides: strides.clone(),
                    pad_amounts: pad_amounts.clone(),
                    dilations: dilations.clone(),
                },
                "mb.conv",
            ),
            SirOp::ConvTranspose {
                input,
                weight,
                pad_type,
                groups,
                strides,
                pad_amounts,
                dilations,
                output_shape,
            } => (
                AirOp::ConvTranspose {
                    input: aid(input),
                    weight: aid(weight),
                    pad_type: pad_type.clone(),
                    groups: *groups,
                    strides: strides.clone(),
                    pad_amounts: pad_amounts.clone(),
                    dilations: dilations.clone(),
                    output_shape: output_shape.clone(),
                },
                "mb.conv_transpose",
            ),

            // ─── Elementwise Binary ──────────────────────────────
            SirOp::Add { x, y } => (AirOp::Add { x: aid(x), y: aid(y) }, "mb.add"),
            SirOp::Mul { x, y } => (AirOp::Mul { x: aid(x), y: aid(y) }, "mb.mul"),
            SirOp::Sub { x, y } => (AirOp::Sub { x: aid(x), y: aid(y) }, "mb.sub"),
            SirOp::Maximum { x, y } => (AirOp::Maximum { x: aid(x), y: aid(y) }, "mb.maximum"),
            SirOp::Minimum { x, y } => (AirOp::Minimum { x: aid(x), y: aid(y) }, "mb.minimum"),
            SirOp::RealDiv { x, y } => (AirOp::RealDiv { x: aid(x), y: aid(y) }, "mb.real_div"),
            SirOp::FloorDiv { x, y } => (AirOp::FloorDiv { x: aid(x), y: aid(y) }, "mb.floor_div"),
            SirOp::Mod { x, y } => (AirOp::Mod { x: aid(x), y: aid(y) }, "mb.mod"),
            SirOp::Pow { x, y } => (AirOp::Pow { x: aid(x), y: aid(y) }, "mb.pow"),
            SirOp::Equal { x, y } => (AirOp::Equal { x: aid(x), y: aid(y) }, "mb.equal"),
            SirOp::NotEqual { x, y } => (AirOp::NotEqual { x: aid(x), y: aid(y) }, "mb.not_equal"),
            SirOp::Greater { x, y } => (AirOp::Greater { x: aid(x), y: aid(y) }, "mb.greater"),
            SirOp::GreaterEqual { x, y } => {
                (AirOp::GreaterEqual { x: aid(x), y: aid(y) }, "mb.greater_equal")
            }
            SirOp::Less { x, y } => (AirOp::Less { x: aid(x), y: aid(y) }, "mb.less"),
            SirOp::LessEqual { x, y } => {
                (AirOp::LessEqual { x: aid(x), y: aid(y) }, "mb.less_equal")
            }
            SirOp::LogicalAnd { x, y } => {
                (AirOp::LogicalAnd { x: aid(x), y: aid(y) }, "mb.logical_and")
            }
            SirOp::LogicalOr { x, y } => {
                (AirOp::LogicalOr { x: aid(x), y: aid(y) }, "mb.logical_or")
            }
            SirOp::LogicalXor { x, y } => {
                (AirOp::LogicalXor { x: aid(x), y: aid(y) }, "mb.logical_xor")
            }

            // ─── Elementwise Unary ───────────────────────────────
            SirOp::Abs { input } => (AirOp::Abs { input: aid(input) }, "mb.abs"),
            SirOp::Neg { input } => (AirOp::Neg { input: aid(input) }, "mb.neg"),
            SirOp::Sigmoid { input } => (AirOp::Sigmoid { input: aid(input) }, "mb.sigmoid"),
            SirOp::Tanh { input } => (AirOp::Tanh { input: aid(input) }, "mb.tanh"),
            SirOp::Relu { input } => (AirOp::Relu { input: aid(input) }, "mb.relu"),
            SirOp::Relu6 { input } => (AirOp::Relu6 { input: aid(input) }, "mb.relu6"),
            SirOp::LeakyRelu { input, alpha } => {
                (AirOp::LeakyRelu { input: aid(input), alpha: *alpha }, "mb.leaky_relu")
            }
            SirOp::SigmoidHard { input, alpha, beta } => (
                AirOp::SigmoidHard { input: aid(input), alpha: *alpha, beta: *beta },
                "mb.sigmoid_hard",
            ),
            SirOp::ThresholdedRelu { input, alpha } => {
                (AirOp::ThresholdedRelu { input: aid(input), alpha: *alpha }, "mb.thresholded_relu")
            }
            SirOp::ClampedRelu { input, alpha, beta } => (
                AirOp::ClampedRelu { input: aid(input), alpha: *alpha, beta: *beta },
                "mb.clamped_relu",
            ),
            SirOp::LinearActivation { input, alpha, beta } => (
                AirOp::LinearActivation { input: aid(input), alpha: *alpha, beta: *beta },
                "mb.linear_activation",
            ),
            SirOp::Prelu { input, alpha } => {
                (AirOp::Prelu { input: aid(input), alpha: alpha.clone() }, "mb.prelu")
            }
            SirOp::Softsign { input } => (AirOp::Softsign { input: aid(input) }, "mb.softsign"),
            SirOp::Silu { input } => (AirOp::Silu { input: aid(input) }, "mb.silu"),
            SirOp::ScaledTanh { input, alpha, beta } => (
                AirOp::ScaledTanh { input: aid(input), alpha: *alpha, beta: *beta },
                "mb.scaled_tanh",
            ),
            SirOp::Elu { input, alpha } => {
                (AirOp::Elu { input: aid(input), alpha: *alpha }, "mb.elu")
            }
            SirOp::Softplus { input } => (AirOp::Softplus { input: aid(input) }, "mb.softplus"),
            SirOp::SoftplusParametric { input, alpha, beta } => (
                AirOp::SoftplusParametric {
                    input: aid(input),
                    alpha: alpha.clone(),
                    beta: beta.clone(),
                },
                "mb.softplus_parametric",
            ),
            SirOp::Gelu { input, mode } => {
                (AirOp::Gelu { input: aid(input), mode: mode.clone() }, "mb.gelu")
            }
            SirOp::Clip { input, min_val, max_val } => {
                (AirOp::Clip { input: aid(input), min_val: *min_val, max_val: *max_val }, "mb.clip")
            }
            SirOp::Square { input } => (AirOp::Square { input: aid(input) }, "mb.square"),
            SirOp::Threshold { input, alpha } => {
                (AirOp::Threshold { input: aid(input), alpha: *alpha }, "mb.threshold")
            }
            SirOp::Sqrt { input } => (AirOp::Sqrt { input: aid(input) }, "mb.sqrt"),
            SirOp::Rsqrt { input } => (AirOp::Rsqrt { input: aid(input) }, "mb.rsqrt"),
            SirOp::Inverse { input, epsilon } => {
                (AirOp::Inverse { input: aid(input), epsilon: *epsilon }, "mb.inverse")
            }
            SirOp::Ceil { input } => (AirOp::Ceil { input: aid(input) }, "mb.ceil"),
            SirOp::Floor { input } => (AirOp::Floor { input: aid(input) }, "mb.floor"),
            SirOp::Round { input } => (AirOp::Round { input: aid(input) }, "mb.round"),
            SirOp::Exp { input } => (AirOp::Exp { input: aid(input) }, "mb.exp"),
            SirOp::Exp2 { input } => (AirOp::Exp2 { input: aid(input) }, "mb.exp2"),
            SirOp::Log { input, epsilon } => {
                (AirOp::Log { input: aid(input), epsilon: *epsilon }, "mb.log")
            }
            SirOp::Sign { input } => (AirOp::Sign { input: aid(input) }, "mb.sign"),
            SirOp::Cos { input } => (AirOp::Cos { input: aid(input) }, "mb.cos"),
            SirOp::Sin { input } => (AirOp::Sin { input: aid(input) }, "mb.sin"),
            SirOp::Tan { input } => (AirOp::Tan { input: aid(input) }, "mb.tan"),
            SirOp::Acos { input } => (AirOp::Acos { input: aid(input) }, "mb.acos"),
            SirOp::Asin { input } => (AirOp::Asin { input: aid(input) }, "mb.asin"),
            SirOp::Atan { input } => (AirOp::Atan { input: aid(input) }, "mb.atan"),
            SirOp::Cosh { input } => (AirOp::Cosh { input: aid(input) }, "mb.cosh"),
            SirOp::Sinh { input } => (AirOp::Sinh { input: aid(input) }, "mb.sinh"),
            SirOp::Atanh { input } => (AirOp::Atanh { input: aid(input) }, "mb.atanh"),
            SirOp::Erf { input } => (AirOp::Erf { input: aid(input) }, "mb.erf"),
            SirOp::LogicalNot { input } => {
                (AirOp::LogicalNot { input: aid(input) }, "mb.logical_not")
            }
            SirOp::Cast { input, dtype } => {
                (AirOp::Cast { input: aid(input), dtype: dtype.clone() }, "mb.cast")
            }
            SirOp::Select { condition, x, y } => {
                (AirOp::Select { condition: aid(condition), x: aid(x), y: aid(y) }, "mb.select")
            }
            SirOp::Where { condition, x, y } => {
                (AirOp::Where { condition: aid(condition), x: aid(x), y: aid(y) }, "mb.where")
            }
            SirOp::Softmax { input, axis } => {
                (AirOp::Softmax { input: aid(input), axis: *axis }, "mb.softmax")
            }

            // ─── Reduction ───────────────────────────────────────
            SirOp::ReduceSum { input, axes, keep_dims } => (
                AirOp::ReduceSum { input: aid(input), axes: axes.clone(), keep_dims: *keep_dims },
                "mb.reduce_sum",
            ),
            SirOp::ReduceMean { input, axes, keep_dims } => (
                AirOp::ReduceMean { input: aid(input), axes: axes.clone(), keep_dims: *keep_dims },
                "mb.reduce_mean",
            ),
            SirOp::ReduceMax { input, axes, keep_dims } => (
                AirOp::ReduceMax { input: aid(input), axes: axes.clone(), keep_dims: *keep_dims },
                "mb.reduce_max",
            ),
            SirOp::ReduceMin { input, axes, keep_dims } => (
                AirOp::ReduceMin { input: aid(input), axes: axes.clone(), keep_dims: *keep_dims },
                "mb.reduce_min",
            ),
            SirOp::ReduceProd { input, axes, keep_dims } => (
                AirOp::ReduceProd { input: aid(input), axes: axes.clone(), keep_dims: *keep_dims },
                "mb.reduce_prod",
            ),
            SirOp::ReduceSumSquare { input, axes, keep_dims } => (
                AirOp::ReduceSumSquare {
                    input: aid(input),
                    axes: axes.clone(),
                    keep_dims: *keep_dims,
                },
                "mb.reduce_sum_square",
            ),
            SirOp::ReduceL2Norm { input, axes, keep_dims } => (
                AirOp::ReduceL2Norm {
                    input: aid(input),
                    axes: axes.clone(),
                    keep_dims: *keep_dims,
                },
                "mb.reduce_l2_norm",
            ),
            SirOp::ReduceL1Norm { input, axes, keep_dims } => (
                AirOp::ReduceL1Norm {
                    input: aid(input),
                    axes: axes.clone(),
                    keep_dims: *keep_dims,
                },
                "mb.reduce_l1_norm",
            ),
            SirOp::ReduceLogSumExp { input, axes, keep_dims } => (
                AirOp::ReduceLogSumExp {
                    input: aid(input),
                    axes: axes.clone(),
                    keep_dims: *keep_dims,
                },
                "mb.reduce_log_sum_exp",
            ),
            SirOp::ReduceLogSum { input, axes, keep_dims } => (
                AirOp::ReduceLogSum {
                    input: aid(input),
                    axes: axes.clone(),
                    keep_dims: *keep_dims,
                },
                "mb.reduce_log_sum",
            ),
            SirOp::ReduceArgmax { input, axis, keep_dims } => (
                AirOp::ReduceArgmax { input: aid(input), axis: *axis, keep_dims: *keep_dims },
                "mb.reduce_argmax",
            ),
            SirOp::ReduceArgmin { input, axis, keep_dims } => (
                AirOp::ReduceArgmin { input: aid(input), axis: *axis, keep_dims: *keep_dims },
                "mb.reduce_argmin",
            ),

            // ─── Normalization ───────────────────────────────────
            SirOp::BatchNorm { input, mean, variance, gamma, beta, epsilon } => (
                AirOp::BatchNorm {
                    input: aid(input),
                    mean: mean.clone(),
                    variance: variance.clone(),
                    gamma: gamma.clone(),
                    beta: beta.clone(),
                    epsilon: *epsilon,
                },
                "mb.batch_norm",
            ),
            SirOp::InstanceNorm { input, gamma, beta, epsilon } => (
                AirOp::InstanceNorm {
                    input: aid(input),
                    gamma: gamma.clone(),
                    beta: beta.clone(),
                    epsilon: *epsilon,
                },
                "mb.instance_norm",
            ),
            SirOp::LayerNorm { input, weight, bias, epsilon, axes } => (
                AirOp::LayerNorm {
                    input: aid(input),
                    weight: weight.clone(),
                    bias: bias.clone(),
                    epsilon: *epsilon,
                    axes: axes.clone(),
                },
                "mb.layer_norm",
            ),
            SirOp::L2Norm { input, epsilon, axes } => (
                AirOp::L2Norm { input: aid(input), epsilon: *epsilon, axes: axes.clone() },
                "mb.l2_norm",
            ),
            SirOp::LocalResponseNorm { input, size, alpha, beta, k } => (
                AirOp::LocalResponseNorm {
                    input: aid(input),
                    size: *size,
                    alpha: *alpha,
                    beta: *beta,
                    k: *k,
                },
                "mb.local_response_norm",
            ),

            // ─── Pooling ─────────────────────────────────────────
            SirOp::MaxPool { input, kernel_sizes, strides, pad_types, pad_amounts } => (
                AirOp::MaxPool {
                    input: aid(input),
                    kernel_sizes: kernel_sizes.clone(),
                    strides: strides.clone(),
                    pad_types: pad_types.clone(),
                    pad_amounts: pad_amounts.clone(),
                },
                "mb.max_pool",
            ),
            SirOp::AvgPool {
                input,
                kernel_sizes,
                strides,
                pad_types,
                pad_amounts,
                count_include_padding,
            } => (
                AirOp::AvgPool {
                    input: aid(input),
                    kernel_sizes: kernel_sizes.clone(),
                    strides: strides.clone(),
                    pad_types: pad_types.clone(),
                    pad_amounts: pad_amounts.clone(),
                    count_include_padding: *count_include_padding,
                },
                "mb.avg_pool",
            ),
            SirOp::L2Pool { input, kernel_sizes, strides, pad_types, pad_amounts } => (
                AirOp::L2Pool {
                    input: aid(input),
                    kernel_sizes: kernel_sizes.clone(),
                    strides: strides.clone(),
                    pad_types: pad_types.clone(),
                    pad_amounts: pad_amounts.clone(),
                },
                "mb.l2_pool",
            ),

            // ─── Image Resizing ──────────────────────────────────
            SirOp::Resize { input, target_size, mode, sampling_mode, nearest_rounding_mode } => (
                AirOp::Resize {
                    input: aid(input),
                    target_size: target_size.clone(),
                    mode: mode.clone(),
                    sampling_mode: sampling_mode.clone(),
                    nearest_rounding_mode: nearest_rounding_mode.clone(),
                },
                "mb.resize",
            ),
            SirOp::ResizeNearestNeighbor { input, target_height, target_width } => (
                AirOp::ResizeNearestNeighbor {
                    input: aid(input),
                    target_height: *target_height,
                    target_width: *target_width,
                },
                "mb.resize_nearest_neighbor",
            ),
            SirOp::ResizeBilinear { input, target_height, target_width, align_corners } => (
                AirOp::ResizeBilinear {
                    input: aid(input),
                    target_height: *target_height,
                    target_width: *target_width,
                    align_corners: *align_corners,
                },
                "mb.resize_bilinear",
            ),
            SirOp::UpsampleNearestNeighbor { input, scale } => (
                AirOp::UpsampleNearestNeighbor { input: aid(input), scale: scale.clone() },
                "mb.upsample_nearest_neighbor",
            ),
            SirOp::UpsampleBilinear { input, scale, align_corners, half_pixel_centers } => (
                AirOp::UpsampleBilinear {
                    input: aid(input),
                    scale: scale.clone(),
                    align_corners: *align_corners,
                    half_pixel_centers: *half_pixel_centers,
                },
                "mb.upsample_bilinear",
            ),
            SirOp::CropResize { input, boxes, box_indices, crop_height, crop_width } => (
                AirOp::CropResize {
                    input: aid(input),
                    boxes: aid(boxes),
                    box_indices: aid(box_indices),
                    crop_height: *crop_height,
                    crop_width: *crop_width,
                },
                "mb.crop_resize",
            ),
            SirOp::Affine {
                input,
                transform,
                output_height,
                output_width,
                sampling_mode,
                pad_value,
            } => (
                AirOp::Affine {
                    input: aid(input),
                    transform: aid(transform),
                    output_height: *output_height,
                    output_width: *output_width,
                    sampling_mode: sampling_mode.clone(),
                    pad_value: *pad_value,
                },
                "mb.affine",
            ),
            SirOp::Resample { input, coordinates, sampling_mode, pad_value } => (
                AirOp::Resample {
                    input: aid(input),
                    coordinates: aid(coordinates),
                    sampling_mode: sampling_mode.clone(),
                    pad_value: *pad_value,
                },
                "mb.resample",
            ),

            // ─── Tensor Transform ────────────────────────────────
            SirOp::Reshape { input, target_shape } => (
                AirOp::Reshape { input: aid(input), target_shape: target_shape.clone() },
                "mb.reshape",
            ),
            SirOp::ReshapeLike { input, ref_tensor } => (
                AirOp::ReshapeLike { input: aid(input), ref_tensor: aid(ref_tensor) },
                "mb.reshape_like",
            ),
            SirOp::Transpose { input, perm } => {
                (AirOp::Transpose { input: aid(input), perm: perm.clone() }, "mb.transpose")
            }
            SirOp::Split { input, axis, num_splits } => (
                AirOp::Split { input: aid(input), axis: *axis, num_splits: *num_splits },
                "mb.split",
            ),
            SirOp::Concat { inputs, axis } => {
                (AirOp::Concat { inputs: aids(inputs), axis: *axis }, "mb.concat")
            }
            SirOp::ExpandDims { input, axis } => {
                (AirOp::ExpandDims { input: aid(input), axis: axis.clone() }, "mb.expand_dims")
            }
            SirOp::Squeeze { input, axis } => {
                (AirOp::Squeeze { input: aid(input), axis: axis.clone() }, "mb.squeeze")
            }
            SirOp::Flatten2d { input, axis } => {
                (AirOp::Flatten2d { input: aid(input), axis: *axis }, "mb.flatten2d")
            }
            SirOp::Reverse { input, axes } => {
                (AirOp::Reverse { input: aid(input), axes: axes.clone() }, "mb.reverse")
            }
            SirOp::ReverseSequence { input, lengths, batch_axis, seq_axis } => (
                AirOp::ReverseSequence {
                    input: aid(input),
                    lengths: aid(lengths),
                    batch_axis: *batch_axis,
                    seq_axis: *seq_axis,
                },
                "mb.reverse_sequence",
            ),
            SirOp::SliceByIndex {
                input,
                begin,
                end,
                stride,
                begin_mask,
                end_mask,
                squeeze_mask,
            } => (
                AirOp::SliceByIndex {
                    input: aid(input),
                    begin: begin.clone(),
                    end: end.clone(),
                    stride: stride.clone(),
                    begin_mask: begin_mask.clone(),
                    end_mask: end_mask.clone(),
                    squeeze_mask: squeeze_mask.clone(),
                },
                "mb.slice_by_index",
            ),
            SirOp::SliceBySize { input, begin, size } => (
                AirOp::SliceBySize { input: aid(input), begin: begin.clone(), size: size.clone() },
                "mb.slice_by_size",
            ),
            SirOp::SlidingWindows { input, axis, window_size, stride } => (
                AirOp::SlidingWindows {
                    input: aid(input),
                    axis: *axis,
                    window_size: *window_size,
                    stride: *stride,
                },
                "mb.sliding_windows",
            ),
            SirOp::DepthToSpace { input, block_size } => (
                AirOp::DepthToSpace { input: aid(input), block_size: *block_size },
                "mb.depth_to_space",
            ),
            SirOp::SpaceToDepth { input, block_size } => (
                AirOp::SpaceToDepth { input: aid(input), block_size: *block_size },
                "mb.space_to_depth",
            ),
            SirOp::PixelShuffle { input, upscale_factor } => (
                AirOp::PixelShuffle { input: aid(input), upscale_factor: *upscale_factor },
                "mb.pixel_shuffle",
            ),
            SirOp::PixelUnshuffle { input, downscale_factor } => (
                AirOp::PixelUnshuffle { input: aid(input), downscale_factor: *downscale_factor },
                "mb.pixel_unshuffle",
            ),
            SirOp::BatchToSpace { input, block_shape, crops } => (
                AirOp::BatchToSpace {
                    input: aid(input),
                    block_shape: block_shape.clone(),
                    crops: crops.clone(),
                },
                "mb.batch_to_space",
            ),
            SirOp::SpaceToBatch { input, block_shape, paddings } => (
                AirOp::SpaceToBatch {
                    input: aid(input),
                    block_shape: block_shape.clone(),
                    paddings: paddings.clone(),
                },
                "mb.space_to_batch",
            ),
            SirOp::Pad { input, pad_amounts, mode, constant_value } => (
                AirOp::Pad {
                    input: aid(input),
                    pad_amounts: pad_amounts.clone(),
                    mode: mode.clone(),
                    constant_value: *constant_value,
                },
                "mb.pad",
            ),
            SirOp::Stack { values, axis } => {
                (AirOp::Stack { values: aids(values), axis: *axis }, "mb.stack")
            }
            SirOp::Tile { input, reps } => {
                (AirOp::Tile { input: aid(input), reps: reps.clone() }, "mb.tile")
            }
            SirOp::Cumsum { input, axis, exclusive, reverse } => (
                AirOp::Cumsum {
                    input: aid(input),
                    axis: *axis,
                    exclusive: *exclusive,
                    reverse: *reverse,
                },
                "mb.cumsum",
            ),
            SirOp::Fill { shape, value, dtype } => (
                AirOp::Fill { shape: shape.clone(), value: *value, dtype: dtype.clone() },
                "mb.fill",
            ),
            SirOp::FillLike { ref_tensor, value, dtype } => (
                AirOp::FillLike {
                    ref_tensor: aid(ref_tensor),
                    value: *value,
                    dtype: dtype.clone(),
                },
                "mb.fill_like",
            ),
            SirOp::Identity { input } => (AirOp::Identity { input: aid(input) }, "mb.identity"),
            SirOp::OneHot { indices, one_hot_vector_size, on_value, off_value, axis, dtype } => (
                AirOp::OneHot {
                    indices: aid(indices),
                    one_hot_vector_size: *one_hot_vector_size,
                    on_value: *on_value,
                    off_value: *off_value,
                    axis: *axis,
                    dtype: dtype.clone(),
                },
                "mb.one_hot",
            ),
            SirOp::NonZero { input } => (AirOp::NonZero { input: aid(input) }, "mb.non_zero"),
            SirOp::Argsort { input, axis, ascending } => (
                AirOp::Argsort { input: aid(input), axis: *axis, ascending: *ascending },
                "mb.argsort",
            ),
            SirOp::BandPart { input, num_lower, num_upper } => (
                AirOp::BandPart { input: aid(input), num_lower: *num_lower, num_upper: *num_upper },
                "mb.band_part",
            ),
            SirOp::Range1d { start, end, step } => {
                (AirOp::Range1d { start: *start, end: *end, step: *step }, "mb.range_1d")
            }
            SirOp::Shape { input } => (AirOp::Shape { input: aid(input) }, "mb.shape"),
            SirOp::Crop { input, crop_height, crop_width, offset_height, offset_width } => (
                AirOp::Crop {
                    input: aid(input),
                    crop_height: *crop_height,
                    crop_width: *crop_width,
                    offset_height: *offset_height,
                    offset_width: *offset_width,
                },
                "mb.crop",
            ),

            // ─── Scatter / Gather ────────────────────────────────
            SirOp::Gather { input, indices, axis } => (
                AirOp::Gather { input: aid(input), indices: aid(indices), axis: *axis },
                "mb.gather",
            ),
            SirOp::GatherAlongAxis { input, indices, axis } => (
                AirOp::GatherAlongAxis { input: aid(input), indices: aid(indices), axis: *axis },
                "mb.gather_along_axis",
            ),
            SirOp::GatherNd { input, indices } => {
                (AirOp::GatherNd { input: aid(input), indices: aid(indices) }, "mb.gather_nd")
            }
            SirOp::Scatter { input, indices, updates, axis, mode } => (
                AirOp::Scatter {
                    input: aid(input),
                    indices: aid(indices),
                    updates: aid(updates),
                    axis: *axis,
                    mode: mode.clone(),
                },
                "mb.scatter",
            ),
            SirOp::ScatterAlongAxis { input, indices, updates, axis } => (
                AirOp::ScatterAlongAxis {
                    input: aid(input),
                    indices: aid(indices),
                    updates: aid(updates),
                    axis: *axis,
                },
                "mb.scatter_along_axis",
            ),
            SirOp::ScatterNd { input, indices, updates } => (
                AirOp::ScatterNd {
                    input: aid(input),
                    indices: aid(indices),
                    updates: aid(updates),
                },
                "mb.scatter_nd",
            ),
            SirOp::NonMaximumSuppression {
                boxes,
                scores,
                iou_threshold,
                score_threshold,
                max_detections,
            } => (
                AirOp::NonMaximumSuppression {
                    boxes: aid(boxes),
                    scores: aid(scores),
                    iou_threshold: *iou_threshold,
                    score_threshold: *score_threshold,
                    max_detections: *max_detections,
                },
                "mb.non_maximum_suppression",
            ),

            // ─── Attention ───────────────────────────────────────
            SirOp::ScaledDotProductAttention { query, key, value, attention_mask, scale } => (
                AirOp::ScaledDotProductAttention {
                    query: aid(query),
                    key: aid(key),
                    value: aid(value),
                    attention_mask: attention_mask.as_ref().map(aid),
                    scale: *scale,
                },
                "mb.scaled_dot_product_attention",
            ),

            // ─── Quantization ────────────────────────────────────
            SirOp::Quantize { input, scale, zero_point, axis, output_dtype } => (
                AirOp::Quantize {
                    input: aid(input),
                    scale: *scale,
                    zero_point: *zero_point,
                    axis: *axis,
                    output_dtype: output_dtype.clone(),
                },
                "mb.quantize",
            ),
            SirOp::Dequantize { input, scale, zero_point, axis, output_dtype } => (
                AirOp::Dequantize {
                    input: aid(input),
                    scale: *scale,
                    zero_point: *zero_point,
                    axis: *axis,
                    output_dtype: output_dtype.clone(),
                },
                "mb.dequantize",
            ),

            // ─── Constexpr / Compression ─────────────────────────
            SirOp::ConstexprAffineDequantize { quantized_data, scale, zero_point, axis } => (
                AirOp::ConstexprAffineDequantize {
                    quantized_data: quantized_data.clone(),
                    scale: *scale,
                    zero_point: *zero_point,
                    axis: *axis,
                },
                "mb.constexpr_affine_dequantize",
            ),
            SirOp::ConstexprBlockwiseShiftScale { data, scale, offset, block_size } => (
                AirOp::ConstexprBlockwiseShiftScale {
                    data: data.clone(),
                    scale: scale.clone(),
                    offset: offset.clone(),
                    block_size: block_size.clone(),
                },
                "mb.constexpr_blockwise_shift_scale",
            ),
            SirOp::ConstexprLutToDense { indices, lut, num_bits } => (
                AirOp::ConstexprLutToDense {
                    indices: indices.clone(),
                    lut: lut.clone(),
                    num_bits: *num_bits,
                },
                "mb.constexpr_lut_to_dense",
            ),
            SirOp::ConstexprSparseToDense { nonzero_data, shape, default_value } => (
                AirOp::ConstexprSparseToDense {
                    nonzero_data: nonzero_data.clone(),
                    shape: shape.clone(),
                    default_value: *default_value,
                },
                "mb.constexpr_sparse_to_dense",
            ),
            SirOp::ConstexprCast { data, dtype } => (
                AirOp::ConstexprCast { data: data.clone(), dtype: dtype.clone() },
                "mb.constexpr_cast",
            ),
            SirOp::ConstexprLutToSparse { data, num_bits } => (
                AirOp::ConstexprLutToSparse { data: data.clone(), num_bits: *num_bits },
                "mb.constexpr_lut_to_sparse",
            ),
            SirOp::ConstexprSparseBlockwiseShiftScale {
                data,
                scale,
                offset,
                block_size,
                block_axis,
            } => (
                AirOp::ConstexprSparseBlockwiseShiftScale {
                    data: data.clone(),
                    scale: scale.clone(),
                    offset: offset.clone(),
                    block_size: block_size.clone(),
                    block_axis: *block_axis,
                },
                "mb.constexpr_sparse_blockwise_shift_scale",
            ),

            // ─── Recurrent ───────────────────────────────────────
            SirOp::Rnn { input, initial_h, weight_ih, weight_hh, bias, mode, output_sequence } => (
                AirOp::Rnn {
                    input: aid(input),
                    initial_h: aid(initial_h),
                    weight_ih: weight_ih.clone(),
                    weight_hh: weight_hh.clone(),
                    bias: bias.clone(),
                    mode: mode.clone(),
                    output_sequence: *output_sequence,
                },
                "mb.rnn",
            ),
            SirOp::Gru {
                input,
                initial_h,
                weight_ih,
                weight_hh,
                bias,
                reset_after,
                output_sequence,
            } => (
                AirOp::Gru {
                    input: aid(input),
                    initial_h: aid(initial_h),
                    weight_ih: weight_ih.clone(),
                    weight_hh: weight_hh.clone(),
                    bias: bias.clone(),
                    reset_after: *reset_after,
                    output_sequence: *output_sequence,
                },
                "mb.gru",
            ),
            SirOp::Lstm {
                input,
                initial_h,
                initial_c,
                weight_ih,
                weight_hh,
                bias,
                output_sequence,
            } => (
                AirOp::Lstm {
                    input: aid(input),
                    initial_h: aid(initial_h),
                    initial_c: aid(initial_c),
                    weight_ih: weight_ih.clone(),
                    weight_hh: weight_hh.clone(),
                    bias: bias.clone(),
                    output_sequence: *output_sequence,
                },
                "mb.lstm",
            ),

            // ─── Control Flow ────────────────────────────────────
            SirOp::Cond { pred, true_graph, false_graph } => (
                AirOp::Cond {
                    pred: aid(pred),
                    true_graph: true_graph.clone(),
                    false_graph: false_graph.clone(),
                },
                "mb.cond",
            ),
            SirOp::WhileLoop { condition, body, loop_vars } => (
                AirOp::WhileLoop {
                    condition: condition.clone(),
                    body: body.clone(),
                    loop_vars: aids(loop_vars),
                },
                "mb.while_loop",
            ),
            SirOp::MakeList { elems, dtype } => {
                (AirOp::MakeList { elems: aids(elems), dtype: dtype.clone() }, "mb.make_list")
            }
            SirOp::ListLength { ls } => (AirOp::ListLength { ls: aid(ls) }, "mb.list_length"),
            SirOp::ListWrite { ls, index, value } => (
                AirOp::ListWrite { ls: aid(ls), index: aid(index), value: aid(value) },
                "mb.list_write",
            ),
            SirOp::ListRead { ls, index } => {
                (AirOp::ListRead { ls: aid(ls), index: aid(index) }, "mb.list_read")
            }
            SirOp::ListGather { ls, indices } => {
                (AirOp::ListGather { ls: aid(ls), indices: aid(indices) }, "mb.list_gather")
            }
            SirOp::ListScatter { ls, indices, values } => (
                AirOp::ListScatter { ls: aid(ls), indices: aid(indices), values: aid(values) },
                "mb.list_scatter",
            ),

            // ─── Random ──────────────────────────────────────────
            SirOp::RandomBernoulli { shape, prob, seed, dtype } => (
                AirOp::RandomBernoulli {
                    shape: shape.clone(),
                    prob: *prob,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                "mb.random_bernoulli",
            ),
            SirOp::RandomNormal { shape, mean, stddev, seed, dtype } => (
                AirOp::RandomNormal {
                    shape: shape.clone(),
                    mean: *mean,
                    stddev: *stddev,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                "mb.random_normal",
            ),
            SirOp::RandomUniform { shape, low, high, seed, dtype } => (
                AirOp::RandomUniform {
                    shape: shape.clone(),
                    low: *low,
                    high: *high,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                "mb.random_uniform",
            ),
            SirOp::RandomCategorical { logits, num_samples, seed, dtype } => (
                AirOp::RandomCategorical {
                    logits: aid(logits),
                    num_samples: *num_samples,
                    seed: *seed,
                    dtype: dtype.clone(),
                },
                "mb.random_categorical",
            ),

            // ─── Topk / Classify ─────────────────────────────────
            SirOp::Topk { input, k, axis } => {
                (AirOp::Topk { input: aid(input), k: *k, axis: *axis }, "mb.topk")
            }
            SirOp::Classify { input } => (AirOp::Classify { input: aid(input) }, "mb.classify"),

            // ─── Legacy compat (handled explicitly above, fallback) ──
            SirOp::ElementWise { .. } => {
                (AirOp::ElementWise { op: ElementWiseOp::Add, inputs: vec![] }, "mb.add")
            }

            // ─── Composite ops (should be handled by decompositions, not here) ──
            SirOp::LinearProjection { .. }
            | SirOp::AttentionBlock { .. }
            | SirOp::RMSNorm { .. }
            | SirOp::RoPETransform { .. }
            | SirOp::DecodeStep { .. }
            | SirOp::Sampler { .. }
            | SirOp::StateRead { .. }
            | SirOp::StateWrite { .. } => {
                unreachable!("composite ops should be handled by explicit decompositions above")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_query::{
        ComputePlanPlacementInfo, LegalityInfo, NoKnowledge, PassKnowledgeQuery,
        PrecisionHazardInfo, RiskInfo,
    };
    use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    /// A mock knowledge query that reports mb.linear as ANE-legal with high confidence.
    struct MockLinearLegalKnowledge;

    impl PassKnowledgeQuery for MockLinearLegalKnowledge {
        fn query_legality(
            &self,
            op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            if op_pattern == "mb.linear" {
                Some(LegalityInfo {
                    ane_legal: true,
                    confidence: 0.95,
                    evidence_count: 10,
                    source_id: Some("test_seed_linear_legal".to_string()),
                })
            } else {
                None
            }
        }

        fn query_risk(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// A mock knowledge query that reports mb.linear as ANE-illegal.
    struct MockLinearIllegalKnowledge;

    impl PassKnowledgeQuery for MockLinearIllegalKnowledge {
        fn query_legality(
            &self,
            op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            if op_pattern == "mb.linear" {
                Some(LegalityInfo {
                    ane_legal: false,
                    confidence: 0.8,
                    evidence_count: 5,
                    source_id: Some("test_seed_linear_illegal".to_string()),
                })
            } else {
                None
            }
        }

        fn query_risk(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            None
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    fn make_linear_sir() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("weight".into()),
                    op: SirOp::ElementWise { op: ElementWiseOp::Mul, inputs: vec![] },
                    name: "weight".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("output".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "weight".into(),
                        bias: Some("bias".into()),
                    },
                    name: "linear_out".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("output".into())],
        }
    }

    /// Test that knowledge changes legality_rewrite pass outputs.
    #[test]
    fn test_knowledge_influences_legality_pass_output() {
        let sir = make_linear_sir();
        let pass = LegalityRewritePass::new();

        let no_knowledge = NoKnowledge;
        let air_no_knowledge = pass.run(sir.clone(), &no_knowledge, None).unwrap();

        let legal_knowledge = MockLinearLegalKnowledge;
        let air_legal = pass.run(sir.clone(), &legal_knowledge, None).unwrap();

        let illegal_knowledge = MockLinearIllegalKnowledge;
        let air_illegal = pass.run(sir.clone(), &illegal_knowledge, None).unwrap();

        // After the Sprint 36 fix, LinearProjection → Conv1x1AsLinear (not MatMul)
        let lp_node_no_knowledge: Vec<_> = air_no_knowledge
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }))
            .collect();
        let lp_node_legal: Vec<_> = air_legal
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }))
            .collect();
        let lp_node_illegal: Vec<_> = air_illegal
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }))
            .collect();

        assert_eq!(lp_node_no_knowledge.len(), 1, "Expected exactly one Conv1x1AsLinear node");
        assert_eq!(lp_node_legal.len(), 1, "Expected exactly one Conv1x1AsLinear node");
        assert_eq!(lp_node_illegal.len(), 1, "Expected exactly one Conv1x1AsLinear node");

        let no_k_conf = lp_node_no_knowledge[0].legality_confidence;
        let legal_conf = lp_node_legal[0].legality_confidence;
        let illegal_conf = lp_node_illegal[0].legality_confidence;

        assert!(
            legal_conf > no_k_conf,
            "Legal knowledge ({}) should produce higher confidence than NoKnowledge ({})",
            legal_conf,
            no_k_conf
        );

        assert!(
            illegal_conf < no_k_conf,
            "Illegal knowledge ({}) should produce lower confidence than NoKnowledge ({})",
            illegal_conf,
            no_k_conf
        );

        let no_k_risk = lp_node_no_knowledge[0].fallback_risk;
        let legal_risk = lp_node_legal[0].fallback_risk;
        let illegal_risk = lp_node_illegal[0].fallback_risk;

        assert!(legal_risk < no_k_risk);
        assert!(illegal_risk > no_k_risk);
    }

    /// Test that NoKnowledge produces the expected default values.
    #[test]
    fn test_no_knowledge_default_confidence() {
        let sir = make_linear_sir();
        let pass = LegalityRewritePass::new();
        let no_knowledge = NoKnowledge;
        let air = pass.run(sir, &no_knowledge, None).unwrap();

        let lp_node = air
            .nodes
            .iter()
            .find(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }))
            .expect("Expected Conv1x1AsLinear node");

        assert!((lp_node.legality_confidence - 0.5).abs() < 0.001);
        assert!((lp_node.fallback_risk - 0.1).abs() < 0.001);
        assert!((lp_node.drift_risk - 0.05).abs() < 0.001);
    }

    /// Test that LinearProjection now lowers to Conv1x1AsLinear (not MatMul).
    ///
    /// This is the Sprint 36 / Critique Bug 1 fix verification.
    #[test]
    fn test_linear_projection_lowers_to_conv1x1aslinear_not_matmul() {
        let sir = make_linear_sir();
        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Should produce Conv1x1AsLinear, NOT MatMul
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_conv1x1 = air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));

        assert!(
            !has_matmul,
            "LinearProjection should NOT lower to MatMul — use Conv1x1AsLinear instead"
        );
        assert!(has_conv1x1, "LinearProjection should lower to Conv1x1AsLinear");
    }

    /// Test that precision_override propagates from SIR through AIR to MIR.
    #[test]
    fn test_precision_override_propagates_sir_to_air_to_mir() {
        use crate::mil_lower::MilLowerPass;
        use crate::shard_plan::ShardPlan;
        use ane_ir::mir::{MilDtype, MirOp};

        let sir = make_linear_sir();

        let mut sir_adapted = sir;
        for node in &mut sir_adapted.nodes {
            if node.name == "linear_out" {
                node.metadata.precision_override = Some("fp32".to_string());
            }
        }

        let pass = LegalityRewritePass::new();
        let no_knowledge = NoKnowledge;
        let air = pass.run(sir_adapted, &no_knowledge, None).unwrap();

        let linear_air_node = air
            .nodes
            .iter()
            .find(|n| n.name == "linear_out")
            .expect("Expected linear_out AIR node");
        assert_eq!(
            linear_air_node.precision_override,
            Some("fp32".to_string()),
            "Precision override must propagate from SIR to AIR"
        );

        let mil_lower = MilLowerPass::new();
        let shard_plan = ShardPlan::default();
        let input_shapes = std::collections::HashMap::new();
        let mirs = mil_lower.run(&air, &shard_plan, &input_shapes).unwrap();

        // After the fix, Conv1x1AsLinear → MILLinear (not MatMul)
        let linear_node = mirs[0]
            .nodes
            .iter()
            .find(|n| matches!(n.op, MirOp::MILLinear { .. }))
            .expect("Expected MILLinear node");
        assert_eq!(
            linear_node.dtype,
            MilDtype::Fp32,
            "MILLinear node dtype must be fp32 when AIR precision_override is fp32"
        );
    }

    /// Test that AttentionBlock decomposes into the expected AIR ops.
    #[test]
    fn test_attention_block_decomposition() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("attn".into()),
                op: SirOp::AttentionBlock {
                    q: SirNodeId("input".into()),
                    k: SirNodeId("input".into()),
                    v: SirNodeId("input".into()),
                    mask: None,
                    rope: None,
                },
                name: "attn".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("attn".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Should have: 3 reshape (Q/K/V) + 3 transpose (Q/K/V) + SDPA + reshape + out proj = 10 nodes
        // The new decomposition uses separate Q/K/V directly (no fused QKV projection or slicing).
        let has_reshape = air.nodes.iter().any(|n| matches!(n.op, AirOp::Reshape { .. }));
        let has_transpose = air.nodes.iter().any(|n| matches!(n.op, AirOp::Transpose { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_out_proj = air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));

        assert!(
            has_reshape,
            "AttentionBlock decomposition must include Reshape for multi-head layout"
        );
        assert!(
            has_transpose,
            "AttentionBlock decomposition must include Transpose for multi-head layout"
        );
        assert!(has_sdpa, "AttentionBlock decomposition must include ScaledDotProductAttention");
        assert!(
            has_out_proj,
            "AttentionBlock decomposition must include Conv1x1AsLinear for output projection"
        );

        // Verify that SDPA has scale set (1/√d_k) when context is available
        let sdpa_node = air.nodes.iter().find(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        if let Some(node) = sdpa_node {
            if let AirOp::ScaledDotProductAttention { scale, attention_mask, .. } = &node.op {
                // Without context (head_dim=0), scale is None. With context, scale = Some(1/√d_k).
                // This test has no context, so scale should be None.
                assert_eq!(*scale, None, "SDPA scale should be None without DecompositionContext");
                assert_eq!(*attention_mask, None, "SDPA mask should be None when SIR mask is None");
            }
        }
    }

    /// Test that DecodeStep decomposes into AIR ops including state read/write.
    #[test]
    fn test_decode_step_decomposition() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("decode".into()),
                op: SirOp::DecodeStep {
                    token: SirNodeId("input".into()),
                    state_map: vec!["k_cache".into(), "v_cache".into()],
                },
                name: "decode".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("decode".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        let has_state_read = air.nodes.iter().any(|n| matches!(n.op, AirOp::StateReadFixed { .. }));
        let has_state_write =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::StateWriteFixed { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_linear = air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));

        assert!(
            has_state_read,
            "DecodeStep decomposition must include StateReadFixed for KV cache"
        );
        assert!(
            has_state_write,
            "DecodeStep decomposition must include StateWriteFixed for KV cache update"
        );
        assert!(has_sdpa, "DecodeStep decomposition must include ScaledDotProductAttention");
        assert!(
            has_linear,
            "DecodeStep decomposition must include Conv1x1AsLinear for QKV and output projections"
        );
    }

    /// Test that RMSNorm decomposes into ReduceMean + Rsqrt + Mul + Mul.
    #[test]
    fn test_rms_norm_decomposition() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("norm".into()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("input".into()),
                    weight: "gamma".into(),
                    epsilon: 1e-5,
                },
                name: "norm".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("norm".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        let has_reduce_mean = air.nodes.iter().any(|n| matches!(n.op, AirOp::ReduceMean { .. }));
        let has_rsqrt = air.nodes.iter().any(|n| matches!(n.op, AirOp::Rsqrt { .. }));
        let has_mul = air
            .nodes
            .iter()
            .any(|n| matches!(n.op, AirOp::ElementWise { op: ElementWiseOp::Mul, .. }));

        assert!(has_reduce_mean, "RMSNorm decomposition must include ReduceMean");
        assert!(has_rsqrt, "RMSNorm decomposition must include Rsqrt");
        assert!(has_mul, "RMSNorm decomposition must include ElementWise::Mul");
    }

    /// Test that RoPETransform decomposes into Cos + Sin + Mul + Mul + Add.
    #[test]
    fn test_rope_decomposition() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("rope".into()),
                op: SirOp::RoPETransform {
                    input: SirNodeId("input".into()),
                    tables: "rope_tables".into(),
                },
                name: "rope".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("rope".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        let has_cos = air.nodes.iter().any(|n| matches!(n.op, AirOp::Cos { .. }));
        let has_sin = air.nodes.iter().any(|n| matches!(n.op, AirOp::Sin { .. }));

        assert!(has_cos, "RoPETransform decomposition must include Cos");
        assert!(has_sin, "RoPETransform decomposition must include Sin");
    }

    /// Test that Sampler decomposes into Topk + Softmax + Gather.
    #[test]
    fn test_sampler_decomposition() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("sampler".into()),
                op: SirOp::Sampler {
                    logits: SirNodeId("input".into()),
                    temperature: 1.0,
                    top_p: 0.9,
                    rep_penalty: 1.0,
                    min_p: 0.0,
                    top_k: 0,
                    gumbel_noise: false,
                },
                name: "sampler".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("sampler".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        let has_topk = air.nodes.iter().any(|n| matches!(n.op, AirOp::Topk { .. }));
        let has_softmax = air.nodes.iter().any(|n| matches!(n.op, AirOp::Softmax { .. }));
        let has_gather = air.nodes.iter().any(|n| matches!(n.op, AirOp::Gather { .. }));

        assert!(has_topk, "Sampler decomposition must include Topk");
        assert!(has_softmax, "Sampler decomposition must include Softmax");
        assert!(has_gather, "Sampler decomposition must include Gather");
    }

    // ─── Sprint 56: DecompositionContext shape propagation tests ─────────

    /// Test that DecompositionContext populates real SliceByIndex bounds in attention decomposition.
    #[test]
    fn test_attention_decomposition_with_context_has_real_shapes() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("attn".into()),
                op: SirOp::AttentionBlock {
                    q: SirNodeId("input".into()),
                    k: SirNodeId("input".into()),
                    v: SirNodeId("input".into()),
                    mask: None,
                    rope: None,
                },
                name: "attn".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("attn".into())],
        };

        // batch=2, embed_dim=128, num_heads=4, head_dim=32, seq_len=16
        let ctx = DecompositionContext::for_attention(2, 128, 4, 32, 16);
        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Find the Q reshape: should have [batch, seq, heads, head_dim] = [2, 16, 4, 32]
        let q_4d =
            air.nodes.iter().find(|n| n.id.0 == "attn_q_4d").expect("Expected attn_q_4d node");
        match &q_4d.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![2, 16, 4, 32],
                    "Q reshape should be [batch, seq, heads, head_dim] = [2, 16, 4, 32]"
                );
            }
            other => panic!("Expected Reshape for attn_q_4d, got {:?}", other),
        }

        // Verify SDPA has correct scale = 1/√32 ≈ 0.17678
        let sdpa_node = air
            .nodes
            .iter()
            .find(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }))
            .expect("Expected SDPA node");
        if let AirOp::ScaledDotProductAttention { scale, .. } = &sdpa_node.op {
            let expected_scale = 1.0 / (32.0_f32).sqrt();
            let actual_scale = scale.expect("SDPA scale must be Some with DecompositionContext providing head_dim");
            assert!(
                (actual_scale - expected_scale).abs() < 1e-5,
                "SDPA scale should be 1/√32 ≈ {:.5}, got {:.5}",
                expected_scale,
                actual_scale
            );
        }

        // Verify attn_flat reshape has [batch, seq, embed]
        let attn_flat = air
            .nodes
            .iter()
            .find(|n| n.id.0 == "attn_attn_flat")
            .expect("Expected attn_attn_flat node");
        match &attn_flat.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![2, 16, 128],
                    "attn_flat reshape should be [batch, seq, embed] = [2, 16, 128]"
                );
            }
            other => panic!("Expected Reshape for attn_attn_flat, got {:?}", other),
        }
    }

    /// Test that attention decomposition without context still uses placeholder zeros.
    #[test]
    fn test_attention_decomposition_without_context_has_placeholder_shapes() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("attn".into()),
                op: SirOp::AttentionBlock {
                    q: SirNodeId("input".into()),
                    k: SirNodeId("input".into()),
                    v: SirNodeId("input".into()),
                    mask: None,
                    rope: None,
                },
                name: "attn".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("attn".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Without context, reshape should have zero-filled target shapes
        let q_4d =
            air.nodes.iter().find(|n| n.id.0 == "attn_q_4d").expect("Expected attn_q_4d node");
        match &q_4d.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![0, 0, 0, 0],
                    "Without context, Q reshape should be placeholder [0, 0, 0, 0]"
                );
            }
            other => panic!("Expected Reshape for attn_q_4d, got {:?}", other),
        }

        // SDPA scale should be None without context (head_dim=0)
        let sdpa_node = air
            .nodes
            .iter()
            .find(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }))
            .expect("Expected SDPA node");
        if let AirOp::ScaledDotProductAttention { scale, .. } = &sdpa_node.op {
            assert_eq!(*scale, None, "Without context, SDPA scale should be None");
        }
    }

    /// Test that DecompositionContext populates real shapes in decode-step decomposition.
    #[test]
    fn test_decode_step_decomposition_with_context_has_real_shapes() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("decode".into()),
                op: SirOp::DecodeStep {
                    token: SirNodeId("input".into()),
                    state_map: vec!["k_cache".into(), "v_cache".into()],
                },
                name: "decode".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("decode".into())],
        };

        // batch=1, embed_dim=128, num_heads=4, head_dim=32, kv_len=64
        let ctx = DecompositionContext::for_decode_step(1, 128, 4, 32, 64);
        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Q slice: begin=[0,0], end=[1,128]
        let q_slice =
            air.nodes.iter().find(|n| n.id.0 == "decode_q").expect("Expected decode_q node");
        match &q_slice.op {
            AirOp::SliceByIndex { begin, end, .. } => {
                assert_eq!(begin, &vec![0, 0]);
                assert_eq!(end, &vec![1, 128], "Q slice end should be [batch, embed] = [1, 128]");
            }
            other => panic!("Expected SliceByIndex for decode_q, got {:?}", other),
        }

        // K cache state read: shape=[64, 128]
        let k_cache_read = air
            .nodes
            .iter()
            .find(|n| n.id.0 == "decode_k_cache_read")
            .expect("Expected decode_k_cache_read node");
        match &k_cache_read.op {
            AirOp::StateReadFixed { shape, .. } => {
                assert_eq!(
                    shape,
                    &vec![64, 128],
                    "K cache state shape should be [kv_len, embed_dim] = [64, 128]"
                );
            }
            other => panic!("Expected StateReadFixed for k_cache_read, got {:?}", other),
        }

        // Q reshape: [batch, heads, 1, head_dim] = [1, 4, 1, 32]
        let q_4d =
            air.nodes.iter().find(|n| n.id.0 == "decode_q_4d").expect("Expected decode_q_4d node");
        match &q_4d.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![1, 4, 1, 32],
                    "Q 4D reshape should be [batch, heads, 1, head_dim] = [1, 4, 1, 32]"
                );
            }
            other => panic!("Expected Reshape for decode_q_4d, got {:?}", other),
        }

        // K reshape: [1, heads, kv_len, head_dim] = [1, 4, 64, 32]
        let k_4d =
            air.nodes.iter().find(|n| n.id.0 == "decode_k_4d").expect("Expected decode_k_4d node");
        match &k_4d.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![1, 4, 64, 32],
                    "K 4D reshape should be [1, heads, kv_len, head_dim] = [1, 4, 64, 32]"
                );
            }
            other => panic!("Expected Reshape for decode_k_4d, got {:?}", other),
        }

        // attn_flat reshape: [batch, embed] = [1, 128]
        let attn_flat = air
            .nodes
            .iter()
            .find(|n| n.id.0 == "decode_attn_flat")
            .expect("Expected decode_attn_flat node");
        match &attn_flat.op {
            AirOp::Reshape { target_shape, .. } => {
                assert_eq!(
                    target_shape,
                    &vec![1, 128],
                    "attn_flat reshape should be [batch, embed] = [1, 128]"
                );
            }
            other => panic!("Expected Reshape for decode_attn_flat, got {:?}", other),
        }
    }

    /// Test that decode-step decomposition without context still uses placeholder zeros.
    #[test]
    fn test_decode_step_decomposition_without_context_has_placeholder_shapes() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("decode".into()),
                op: SirOp::DecodeStep {
                    token: SirNodeId("input".into()),
                    state_map: vec!["k_cache".into(), "v_cache".into()],
                },
                name: "decode".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("decode".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Without context, state read shapes should be zero
        let k_cache_read = air
            .nodes
            .iter()
            .find(|n| n.id.0 == "decode_k_cache_read")
            .expect("Expected decode_k_cache_read node");
        match &k_cache_read.op {
            AirOp::StateReadFixed { shape, .. } => {
                assert_eq!(
                    shape,
                    &vec![0, 0],
                    "Without context, state shape should be placeholder [0, 0]"
                );
            }
            other => panic!("Expected StateReadFixed, got {:?}", other),
        }
    }

    /// Test that DecompositionContext::default has all-zero fields.
    #[test]
    fn test_decomposition_context_default() {
        let ctx = DecompositionContext::default();
        assert_eq!(ctx.batch_size, 0);
        assert_eq!(ctx.embed_dim, 0);
        assert_eq!(ctx.num_heads, 0);
        assert_eq!(ctx.head_dim, 0);
        assert_eq!(ctx.seq_len, 0);
    }

    /// Test that DecompositionContext::for_attention and for_decode_step construct correctly.
    #[test]
    fn test_decomposition_context_constructors() {
        let attn_ctx = DecompositionContext::for_attention(2, 256, 8, 32, 64);
        assert_eq!(attn_ctx.batch_size, 2);
        assert_eq!(attn_ctx.embed_dim, 256);
        assert_eq!(attn_ctx.num_heads, 8);
        assert_eq!(attn_ctx.head_dim, 32);
        assert_eq!(attn_ctx.seq_len, 64);

        let ds_ctx = DecompositionContext::for_decode_step(1, 128, 4, 32, 96);
        assert_eq!(ds_ctx.batch_size, 1);
        assert_eq!(ds_ctx.embed_dim, 128);
        assert_eq!(ds_ctx.num_heads, 4);
        assert_eq!(ds_ctx.head_dim, 32);
        assert_eq!(ds_ctx.seq_len, 96);
    }
}
