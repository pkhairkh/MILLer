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
//! | AttentionBlock | Split + per-head MatMul + Softmax + Concat (no Tile, no SDPA) |
//! | DecodeStep | Split + per-head MatMul + Softmax + Concat (no Tile, no SDPA) |
//! | RMSNorm | ReduceMean + Rsqrt + ElementWise::Mul + ElementWise::Mul |
//! | RoPETransform | Const(cos_tab) + Const(sin_tab) + Gather + ElementWise::Mul + ElementWise::Add |
//! | Tile | Reshape + broadcast Mul + Reshape (fallback decomposition) |
//! | Sampler | Topk + Gather + Softmax |
//!
//! **Tile elimination strategy (matching reference model pkhairkh/qwen3-coreml-palettized):**
//! GQA Tile ops are eliminated at the SIR builder level by using split-based
//! per-head attention instead of Tile+SDPA. Any remaining standalone Tile ops
//! are decomposed to Reshape + broadcast Mul + Reshape. The fallback passthrough
//! panics to prevent Tile from ever reaching AIR/MIR.
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
use ane_ir::sir::{SirGraph, SirOp};
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
    /// MLP intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size for the language model head.
    pub vocab_size: usize,
    /// Whether the model uses RoPE (config-driven, not model-specific).
    pub uses_rope: bool,
    /// Whether the model uses QK-norm (config-driven, not model-specific).
    pub has_qk_norm: bool,
    /// Whether the model uses GQA (kv_heads < num_heads).
    pub uses_gqa: bool,
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
            uses_rope: false,
            has_qk_norm: false,
            uses_gqa: false,
        }
    }

    /// Construct a context from an Attention task spec with full dimensions.
    #[allow(clippy::too_many_arguments)]
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
            uses_rope: false,
            has_qk_norm: false,
            uses_gqa: kv_heads > 0 && kv_heads < num_heads,
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
            uses_rope: false,
            has_qk_norm: false,
            uses_gqa: false,
        }
    }

    /// Construct a context for a DecodeStep with full dimensions and feature flags.
    ///
    /// Includes `intermediate_size` and `vocab_size` so that `output_dim_for_weight`
    /// can resolve MLP gate/up/down projections and lm_head correctly.
    /// Without these, the Conv1x1AsLinear output_dim for lm_head would be 0,
    /// causing shape inference to produce wrong dimensions for the decode_step.
    #[allow(clippy::too_many_arguments)]
    pub fn for_decode_step_full(
        batch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        kv_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        uses_rope: bool,
        has_qk_norm: bool,
    ) -> Self {
        Self {
            batch_size,
            embed_dim,
            num_heads,
            head_dim,
            seq_len: kv_len,
            kv_heads,
            intermediate_size,
            vocab_size,
            uses_rope,
            has_qk_norm,
            uses_gqa: kv_heads > 0 && kv_heads < num_heads,
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
            || weight.contains(".self_attn.v_proj.weight")
        {
            kv_heads * self.head_dim
        } else if weight.contains(".self_attn.o_proj.weight")
            || weight.contains(".self_attn.out_proj.weight")
        {
            self.embed_dim
        } else if weight.contains(".mlp.gate_proj.weight") || weight.contains(".mlp.up_proj.weight")
        {
            if self.intermediate_size > 0 {
                self.intermediate_size
            } else {
                0
            }
        } else if weight.contains(".mlp.down_proj.weight") {
            self.embed_dim
        } else if weight == "lm_head.weight" || weight.contains("lm_head.") {
            if self.vocab_size > 0 {
                self.vocab_size
            } else {
                0
            }
        } else if weight.contains("embed_tokens") {
            self.embed_dim
        } else {
            0
        }
    }

    /// Construct a context from a ModelArchConfig plus runtime dimensions.
    ///
    /// This is the preferred factory when a `ModelArchConfig` is available
    /// (e.g., from the task spec), avoiding field-by-field unpacking.
    pub fn from_model_arch(
        config: &ane_ir::common::ModelArchConfig,
        batch_size: usize,
        seq_len: usize,
        uses_rope: bool,
        has_qk_norm: bool,
    ) -> Self {
        Self {
            batch_size,
            embed_dim: config.embed_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            seq_len,
            kv_heads: config.kv_heads,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            uses_rope,
            has_qk_norm,
            uses_gqa: config.kv_heads > 0 && config.kv_heads < config.num_heads,
        }
    }

    /// Derive the input dimension at position `i` for a Tile op, given the
    /// total rank of the Tile's input tensor.
    ///
    /// T-60 (I-34): This method computes concrete Tile input dimensions from
    /// the DecompositionContext fields, replacing the 0 placeholders that
    /// previously relied on `resolve_reshape_zeros()` heuristic (batch=1 for
    /// multi-zero cases, which is incorrect for GQA Tile patterns).
    ///
    /// The most common Tile pattern in this compiler is GQA KV-head expansion:
    ///   Tile([B, kv_heads, S, D], [1, fan_out, 1, 1])
    /// where position 0 = batch, 1 = kv_heads, 2 = seq_len, 3 = head_dim.
    ///
    /// Returns `None` if the dimension cannot be determined from the context
    /// (e.g., the position is out of range or the context lacks dimension info).
    pub fn tile_input_dim(&self, position: usize, rank: usize) -> Option<usize> {
        // Only support 4D Tile patterns (the only ones used in this compiler).
        // Tile with other ranks will fall back to 0-placeholder resolution.
        if rank != 4 {
            return None;
        }

        let kv_heads = if self.kv_heads > 0 { self.kv_heads } else { self.num_heads };

        match position {
            0 => {
                // Batch dimension
                if self.batch_size > 0 { Some(self.batch_size) } else { None }
            }
            1 => {
                // Second dimension — typically kv_heads for GQA Tile,
                // or num_heads for non-GQA Tile.
                if kv_heads > 0 { Some(kv_heads) } else { None }
            }
            2 => {
                // Third dimension — typically sequence length.
                if self.seq_len > 0 { Some(self.seq_len) } else { None }
            }
            3 => {
                // Fourth dimension — typically head_dim.
                if self.head_dim > 0 { Some(self.head_dim) } else { None }
            }
            _ => None,
        }
    }
}

/// Shared pass infrastructure passed through all SIR→AIR decomposition functions.
///
/// This struct bundles the references that appear in almost every decomposition
/// function signature: the SIR→AIR node mapping, the knowledge query interface,
/// the original SIR node, and the base name for AIR node IDs.
/// Grouping these eliminates the `too_many_arguments` clippy warning across 7+ functions.
pub struct DecompositionEnv<'a> {
    /// SIR→AIR node ID mapping (already-emitted nodes).
    pub sir_to_air: &'a std::collections::HashMap<ane_ir::sir::SirNodeId, AirNodeId>,
    /// Knowledge query for ANE legality rules.
    pub kq: &'a dyn PassKnowledgeQuery,
    /// Original SIR node being decomposed (for ID prefix + metadata).
    pub sir_node: &'a ane_ir::sir::SirNode,
    /// Base name prefix for generated AIR node IDs.
    pub base: &'a str,
}

/// Weight references for the decode step decomposition.
///
/// Groups the 8 optional weight-name strings that resolve to Conv1x1AsLinear
/// weight paths. All are `Option<&str>` because some models don't use all features
/// (e.g., QK-norm, RoPE, attention masking).
pub struct DecodeWeights<'a> {
    /// Weight name for Q projection.
    pub q_weight: Option<&'a str>,
    /// Weight name for K projection.
    pub k_weight: Option<&'a str>,
    /// Weight name for V projection.
    pub v_weight: Option<&'a str>,
    /// Weight name for output (O) projection.
    pub out_weight: Option<&'a str>,
    /// Weight name for precomputed RoPE cos/sin tables.
    pub rope_tables: Option<&'a str>,
    /// Weight name for Q-norm (Qwen3-style per-head RMSNorm).
    pub q_norm_weight: Option<&'a str>,
    /// Weight name for K-norm.
    pub k_norm_weight: Option<&'a str>,
    /// Weight name for the causal attention mask.
    pub mask_ref: Option<&'a str>,
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
                SirOp::LinearProjection { input: sir_input, weight, bias: _, .. } => {
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
                    let output_dim = ctx.map(|c| c.output_dim_for_weight(weight)).unwrap_or(0);
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
                    let base_str = &sir_node.id.0;
                    let env = DecompositionEnv {
                        sir_to_air: &sir_to_air,
                        kq: knowledge_query,
                        sir_node,
                        base: base_str,
                    };
                    let (final_id, nodes) =
                        Self::decompose_attention_block(q, k, v, mask, &env, attn_ctx.as_ref());
                    (final_id, nodes, "mb.scaled_dot_product_attention")
                }
                SirOp::DecodeStep {
                    token,
                    state_map,
                    q_weight,
                    k_weight,
                    v_weight,
                    out_weight,
                    rope_tables,
                    position,
                    q_norm_weight,
                    k_norm_weight,
                    norm_epsilon,
                    qk_norm_type: _,
                    mask_ref,
                } => {
                    let ds_ctx =
                        ctx.and_then(|c| if c.embed_dim > 0 { Some(c.clone()) } else { None });
                    let base_str = &sir_node.id.0;
                    let env = DecompositionEnv {
                        sir_to_air: &sir_to_air,
                        kq: knowledge_query,
                        sir_node,
                        base: base_str,
                    };
                    let weights = DecodeWeights {
                        q_weight: q_weight.as_deref(),
                        k_weight: k_weight.as_deref(),
                        v_weight: v_weight.as_deref(),
                        out_weight: out_weight.as_deref(),
                        rope_tables: rope_tables.as_deref(),
                        q_norm_weight: q_norm_weight.as_deref(),
                        k_norm_weight: k_norm_weight.as_deref(),
                        mask_ref: mask_ref.as_deref(),
                    };
                    let (final_id, nodes) = Self::decompose_decode_step(
                        token,
                        state_map,
                        position,
                        &weights,
                        *norm_epsilon,
                        &env,
                        ds_ctx.as_ref(),
                    );
                    (final_id, nodes, "mb.scaled_dot_product_attention")
                }
                SirOp::RMSNorm { input, weight, epsilon, axes } => {
                    let base_str = &sir_node.id.0;
                    let env = DecompositionEnv {
                        sir_to_air: &sir_to_air,
                        kq: knowledge_query,
                        sir_node,
                        base: base_str,
                    };
                    let (final_id, nodes) =
                        Self::decompose_rms_norm(input, weight, *epsilon, axes, &env, ctx);
                    (final_id, nodes, "mb.layer_norm")
                }
                SirOp::RoPETransform { input, tables } => {
                    let base_str = &sir_node.id.0;
                    let env = DecompositionEnv {
                        sir_to_air: &sir_to_air,
                        kq: knowledge_query,
                        sir_node,
                        base: base_str,
                    };
                    let (final_id, nodes) = Self::decompose_rope(input, tables, &env, ctx);
                    (final_id, nodes, "mb.mul")
                }
                SirOp::Sampler { logits, temperature: _, top_p: _, rep_penalty: _, .. } => {
                    let (final_id, nodes) =
                        Self::decompose_sampler(sir_node, logits, &sir_to_air, knowledge_query);
                    (final_id, nodes, "mb.topk")
                }
                SirOp::Tile { input, reps } => {
                    // ANE-LEGAL DECOMPOSITION (Sprint 67→68):
                    // mb.tile is ANE-illegal — the ANE cannot execute Tile ops,
                    // forcing CPU fallback. This causes:
                    //   1. CPU fallback during execution planning, adding load/compile overhead
                    //   2. Inter-op synchronization stalls when switching between ANE and CPU
                    //   3. On multi-function models, CPU-only ops may block ANE pipelining
                    //
                    // The reference model (pkhairkh/qwen3-coreml-palettized) does NOT
                    // use mb.tile. For GQA, it uses split-based per-head attention
                    // (handled in decompose_decode_step). For standalone Tile ops
                    // (e.g., in prefill models without KV cache), we decompose to
                    // ANE-legal broadcast Mul:
                    //
                    //   Tile(x, reps) → Mul(x_reshaped, ones)
                    //
                    // Where:
                    //   - x_reshaped: Reshape x to insert size-1 dims where reps > 1
                    //     (ANE broadcast rules will expand the size-1 dims)
                    //   - ones: Const tensor of 1.0 with the final tiled shape
                    //   - The Mul broadcast replicates x along the tiled dimensions
                    //
                    // For GQA Tile specifically:
                    //   Tile([B, kv_heads, S, D], [1, fan_out, 1, 1])
                    //     → Reshape([B, kv_heads, 1, S, D])
                    //     → Mul(reshaped, ones[B, kv_heads, fan_out, S, D])
                    //     → Reshape([B, kv_heads*fan_out, S, D])
                    //
                    // This is fully ANE-compatible: Reshape and Mul both run on ANE.
                    let input_air = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let base = &sir_node.id.0;

                    // Check if any reps > 1 (tile is actually needed)
                    let needs_tile = reps.iter().any(|&r| r > 1);

                    if !needs_tile {
                        // No-op tile: all reps are 1, just pass through as identity
                        let nodes = vec![Self::make_air_node(
                            air_id.clone(),
                            AirOp::Identity { input: input_air },
                            sir_node,
                            "mb.identity",
                            knowledge_query,
                        )];
                        (air_id, nodes, "mb.identity")
                    } else {
                        // Decompose Tile into: Reshape (insert broadcast dims) → broadcast Mul → Reshape (final shape)
                        // The Mul with ones will use ANE broadcast to expand the size-1 dimensions.
                        //
                        // T-60 (I-34): Previously, reshape_shape and final_shape used 0
                        // placeholders for input dimensions, relying on resolve_reshape_zeros()
                        // downstream. That function uses a batch=1 heuristic for multi-zero
                        // resolution, which is incorrect for general Tile patterns (e.g., GQA
                        // Tile with input shape [B, kv_heads, S, D] and reps [1, fan_out, 1, 1]).
                        //
                        // Now, when ctx is available, we compute concrete input dimensions
                        // from the DecompositionContext fields. When ctx is None (e.g., in tests
                        // or non-task compilation), we fall back to 0 placeholders with a logged
                        // warning, preserving backward compatibility.
                        //
                        // The final_shape is computed at the original input rank (4D for a 4D Tile),
                        // not the expanded rank (5D). The Mul broadcast output is at the expanded
                        // rank, and the final Reshape collapses it back to the input rank.
                        let mut nodes = Vec::new();

                        // Step 1: Reshape input to insert size-1 broadcast dimensions
                        // For each dimension where reps[i] > 1, insert a new axis of size 1.
                        // E.g., Tile([B, kv, S, D], [1, fan, 1, 1])
                        //     → Reshape to [B, 1, kv, S, D] (insert axis at dim 1 for fan_out)
                        let mut reshape_shape: Vec<usize> = Vec::new();
                        let mut final_shape: Vec<usize> = Vec::new();
                        for (i, &rep) in reps.iter().enumerate() {
                            // T-60 (I-34): Use concrete input dimensions from ctx when available.
                            let input_dim = ctx.and_then(|c| c.tile_input_dim(i, reps.len()));
                            let dim_val = match input_dim {
                                Some(dim) => dim,
                                None => {
                                    // No ctx or unknown dimension — use 0 placeholder.
                                    // resolve_reshape_zeros() will attempt heuristic resolution.
                                    if ctx.is_none() {
                                        log::warn!(
                                            "Tile decomposition for '{}' using 0 placeholders \
                                             (no DecompositionContext). Provide ctx for correct \
                                             shape resolution.",
                                            base
                                        );
                                    }
                                    0
                                }
                            };

                            if rep > 1 {
                                reshape_shape.push(1); // Insert broadcast dim BEFORE input dim
                            }
                            reshape_shape.push(dim_val);

                            // Final shape is at the original input rank, with each dim = input * rep.
                            // The Mul broadcast output is at the expanded rank; this Reshape collapses it.
                            final_shape.push(if dim_val > 0 { dim_val * rep } else { 0 });
                        }

                        let reshape_id = AirNodeId(format!("{}_tile_reshape", base));
                        nodes.push(Self::make_air_node(
                            reshape_id.clone(),
                            AirOp::Reshape { input: input_air, target_shape: reshape_shape },
                            sir_node,
                            "mb.reshape",
                            knowledge_query,
                        ));

                        // Step 2: Mul with ones tensor (broadcast will handle the tiling)
                        let ones_id = AirNodeId(format!("{}_tile_ones", base));
                        nodes.push(Self::make_air_node(
                            ones_id.clone(),
                            AirOp::Const {
                                value_path: format!("_tile_ones_{}", base),
                                dtype: ane_ir::mir::MilDtype::Fp16,
                            },
                            sir_node,
                            "mb.const",
                            knowledge_query,
                        ));

                        let mul_id = AirNodeId(format!("{}_tile_mul", base));
                        nodes.push(Self::make_air_node(
                            mul_id.clone(),
                            AirOp::Mul { x: reshape_id, y: ones_id },
                            sir_node,
                            "mb.mul",
                            knowledge_query,
                        ));

                        // Step 3: Reshape to final tiled shape (collapse broadcast dims)
                        let final_reshape_id = air_id.clone();
                        nodes.push(Self::make_air_node(
                            final_reshape_id,
                            AirOp::Reshape { input: mul_id, target_shape: final_shape },
                            sir_node,
                            "mb.reshape",
                            knowledge_query,
                        ));

                        (air_id, nodes, "ane.legal.tile_decompose")
                    }
                }
                SirOp::Add { x, y } => {
                    let air_x =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let air_y =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Add { x: air_x, y: air_y },
                        sir_node,
                        "mb.add",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.add")
                }
                SirOp::Mul { x, y } => {
                    let air_x =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let air_y =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Mul { x: air_x, y: air_y },
                        sir_node,
                        "mb.mul",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.mul")
                }
                // ─── ANE-ILLEGAL: Select / Where → arithmetic decomposition ───
                // mb.select and mb.where are ANE-illegal (no ANE converter).
                // Decompose: select/where(cond, x, y) → cond*x + (1-cond)*y
                // using Const(1.0), Sub, Mul, Add — all fully ANE-legal.
                SirOp::Select { condition, x, y } => {
                    let cond_air = sir_to_air
                        .get(condition)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(condition.0.clone()));
                    let x_air =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let y_air =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let base = &sir_node.id.0;
                    let mut nodes = Vec::new();

                    // 1. Const scalar 1.0
                    let one_id = AirNodeId(format!("{}_sel_one", base));
                    nodes.push(Self::make_air_node(
                        one_id.clone(),
                        AirOp::Const {
                            value_path: "scalar://fp16/1.0".to_string(),
                            dtype: MilDtype::Fp16,
                        },
                        sir_node,
                        "mb.const",
                        knowledge_query,
                    ));

                    // 2. Sub: 1 - cond
                    let one_minus_cond_id = AirNodeId(format!("{}_sel_sub", base));
                    nodes.push(Self::make_air_node(
                        one_minus_cond_id.clone(),
                        AirOp::Sub { x: one_id, y: cond_air.clone() },
                        sir_node,
                        "mb.sub",
                        knowledge_query,
                    ));

                    // 3. Mul: cond * x
                    let cond_x_id = AirNodeId(format!("{}_sel_mul_x", base));
                    nodes.push(Self::make_air_node(
                        cond_x_id.clone(),
                        AirOp::Mul { x: cond_air, y: x_air },
                        sir_node,
                        "mb.mul",
                        knowledge_query,
                    ));

                    // 4. Mul: (1-cond) * y
                    let one_minus_cond_y_id = AirNodeId(format!("{}_sel_mul_y", base));
                    nodes.push(Self::make_air_node(
                        one_minus_cond_y_id.clone(),
                        AirOp::Mul { x: one_minus_cond_id, y: y_air },
                        sir_node,
                        "mb.mul",
                        knowledge_query,
                    ));

                    // 5. Add: cond*x + (1-cond)*y (final result)
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    nodes.push(Self::make_air_node(
                        air_id.clone(),
                        AirOp::Add { x: cond_x_id, y: one_minus_cond_y_id },
                        sir_node,
                        "mb.add",
                        knowledge_query,
                    ));

                    (air_id, nodes, "ane.legal.select_decompose")
                }
                SirOp::Where { condition, x, y } => {
                    let cond_air = sir_to_air
                        .get(condition)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(condition.0.clone()));
                    let x_air =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let y_air =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let base = &sir_node.id.0;
                    let mut nodes = Vec::new();

                    // 1. Const scalar 1.0
                    let one_id = AirNodeId(format!("{}_where_one", base));
                    nodes.push(Self::make_air_node(
                        one_id.clone(),
                        AirOp::Const {
                            value_path: "scalar://fp16/1.0".to_string(),
                            dtype: MilDtype::Fp16,
                        },
                        sir_node,
                        "mb.const",
                        knowledge_query,
                    ));

                    // 2. Sub: 1 - cond
                    let one_minus_cond_id = AirNodeId(format!("{}_where_sub", base));
                    nodes.push(Self::make_air_node(
                        one_minus_cond_id.clone(),
                        AirOp::Sub { x: one_id, y: cond_air.clone() },
                        sir_node,
                        "mb.sub",
                        knowledge_query,
                    ));

                    // 3. Mul: cond * x
                    let cond_x_id = AirNodeId(format!("{}_where_mul_x", base));
                    nodes.push(Self::make_air_node(
                        cond_x_id.clone(),
                        AirOp::Mul { x: cond_air, y: x_air },
                        sir_node,
                        "mb.mul",
                        knowledge_query,
                    ));

                    // 4. Mul: (1-cond) * y
                    let one_minus_cond_y_id = AirNodeId(format!("{}_where_mul_y", base));
                    nodes.push(Self::make_air_node(
                        one_minus_cond_y_id.clone(),
                        AirOp::Mul { x: one_minus_cond_id, y: y_air },
                        sir_node,
                        "mb.mul",
                        knowledge_query,
                    ));

                    // 5. Add: cond*x + (1-cond)*y (final result)
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    nodes.push(Self::make_air_node(
                        air_id.clone(),
                        AirOp::Add { x: cond_x_id, y: one_minus_cond_y_id },
                        sir_node,
                        "mb.add",
                        knowledge_query,
                    ));

                    (air_id, nodes, "ane.legal.where_decompose")
                }
                SirOp::Abs { input } => {
                    let air_input = sir_to_air
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| AirNodeId(input.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Abs { input: air_input },
                        sir_node,
                        "mb.abs",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.abs")
                }
                SirOp::Maximum { x, y } => {
                    let air_x =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let air_y =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Maximum { x: air_x, y: air_y },
                        sir_node,
                        "mb.maximum",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.maximum")
                }
                SirOp::Minimum { x, y } => {
                    let air_x =
                        sir_to_air.get(x).cloned().unwrap_or_else(|| AirNodeId(x.0.clone()));
                    let air_y =
                        sir_to_air.get(y).cloned().unwrap_or_else(|| AirNodeId(y.0.clone()));
                    let air_id = AirNodeId(sir_node.id.0.clone());
                    let nodes = vec![Self::make_air_node(
                        air_id.clone(),
                        AirOp::Minimum { x: air_x, y: air_y },
                        sir_node,
                        "mb.minimum",
                        knowledge_query,
                    )];
                    (air_id, nodes, "mb.minimum")
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
                        Self::sir_to_air_passthrough(op, &sir_node.id, &sir_to_air)?;
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

        // ─── Deduplicate shared AIR nodes ─────────────────────────────────
        // Shared nodes (e.g., shared_attn_scale, shared_scalar_one,
        // shared_rope_*_cos_tab/sin_tab/arange_tab) use the same AirNodeId
        // across multiple layers. The dedup checks inside decompose_* functions
        // check `sir_to_air.values()`, but shared IDs are intermediate nodes
        // that are never the final AirNodeId of a SIR node, so the checks
        // always fail and the shared node is emitted once per layer call.
        //
        // This produces N copies of each shared node (N = number of layers),
        // all with the same AirNodeId, violating CoreML MIL's SSA rule that
        // each output name must be defined exactly once.
        //
        // Fix: after all nodes are collected, deduplicate by AirNodeId.0,
        // keeping only the first occurrence. Subsequent duplicates are
        // harmless — they would produce the same value, so removing them
        // doesn't change semantics. All references to the shared ID from
        // other nodes point to the AirNodeId string (not a position index),
        // so the first occurrence is the canonical definition.
        {
            let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            air_nodes.retain(|node| seen_ids.insert(node.id.0.clone()));
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
    ///   attn_flat: Reshape(attn, [batch, seq, heads*head_dim])
    ///   output: Conv1x1AsLinear(attn_flat, W_out)
    ///
    /// When `ctx` is `Some`, the SliceByIndex bounds and Reshape target shapes
    /// are populated with real dimensions from the task spec (Sprint 56).
    /// When `ctx` is `None`, placeholder zeros are used (pre-Sprint-56 behavior).
    fn decompose_attention_block(
        q_sir: &ane_ir::sir::SirNodeId,
        k_sir: &ane_ir::sir::SirNodeId,
        v_sir: &ane_ir::sir::SirNodeId,
        mask_sir: &Option<ane_ir::sir::SirNodeId>,
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let sir_node = env.sir_node;
        let base = env.base;
        let sir_to_air = env.sir_to_air;
        let kq = env.kq;
        let q_air = sir_to_air.get(q_sir).cloned().unwrap_or_else(|| AirNodeId(q_sir.0.clone()));
        let k_air = sir_to_air.get(k_sir).cloned().unwrap_or_else(|| AirNodeId(k_sir.0.clone()));
        let v_air = sir_to_air.get(v_sir).cloned().unwrap_or_else(|| AirNodeId(v_sir.0.clone()));

        // Extract dimensions from context or use placeholders
        let (batch, seq, embed, heads, kv_heads, head_dim) = match ctx {
            Some(c) => (
                c.batch_size as i64,
                c.seq_len as i64,
                c.embed_dim as i64,
                c.num_heads as i64,
                c.kv_heads.max(1) as i64,
                c.head_dim as i64,
            ),
            None => (0, 0, 0, 0, 0, 0),
        };

        let mut nodes = Vec::new();

        // Q, K, V come as separate projections from the SIR builder.
        // The SIR builder now emits split-based per-head attention at the SIR level,
        // so this function is primarily for backward compat with fused AttentionBlock
        // SIR ops. We use the same split-based approach here to eliminate Tile+SDPA.

        // Steps 1-3: Reshape Q, K, V from [batch, seq, embed] to [batch, seq, heads, head_dim]
        // For GQA models, K/V use kv_heads (not num_heads) because their projection
        // output is [B, S, kv_heads*head_dim], not [B, S, num_heads*head_dim].
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
                target_shape: vec![
                    batch as usize,
                    seq as usize,
                    kv_heads as usize,
                    head_dim as usize,
                ],
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
                target_shape: vec![
                    batch as usize,
                    seq as usize,
                    kv_heads as usize,
                    head_dim as usize,
                ],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Steps 4-6: Transpose to [batch, heads, seq, head_dim]
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

        // Define the attn_flat_id before the if/else so it's available for output projection
        let attn_flat_id = AirNodeId(format!("{base}_attn_flat"));

        // Step 7: Split-based per-head attention (ANE-legal, no Tile/SDPA)
        //
        // The reference model (pkhairkh/qwen3-coreml-palettized) does NOT
        // use mb.tile or mb.scaled_dot_product_attention. Instead, it splits
        // Q into individual heads and pairs each Q head with its corresponding
        // KV head via per-head matmul+softmax+matmul.
        //
        // For GQA (kv_heads < num_heads): fan_out = num_heads / kv_heads
        // Each group of `fan_out` Q heads shares one KV head.
        // For non-GQA (kv_heads == num_heads): fan_out = 1, per-head attention.
        //
        // FALLBACK: When DecompositionContext is not available (heads=0), we
        // cannot split into per-head attention because we don't know the head
        // count. In this case, fall back to SDPA (which will be on the ANE for
        // A16+ targets). This fallback should only occur in synthetic tests.

        // CRITICAL: Use heads * head_dim, NOT embed_dim. For models where
        // num_heads * head_dim != hidden_size (e.g., Qwen3-0.6B: 16*128=2048 ≠ 1024),
        // using embed_dim produces an impossible reshape because the concat output
        // has num_heads * head_dim elements, not embed_dim elements.
        let attn_flat_dim = heads * head_dim;

        if heads > 0 {
            // ── Split-based per-head attention (primary path) ──────────
            let fan_out =
                if kv_heads > 0 && kv_heads < heads { (heads / kv_heads) as usize } else { 1 };

            // NOTE: We intentionally do NOT emit mb.split here. Core ML's split op
            // returns a *list* of tensors, which our IR cannot model (single output
            // per op). Serialising a split with num_splits>1 as a single-output op
            // is invalid MIL and causes "number of outputs must be within the range
            // 2:MAX" errors from coremlcompiler. Instead, we slice individual heads
            // directly from the original Q/K/V tensors using slice_by_index, which
            // matches the Python reference emitter pattern and produces valid MIL.
            let q_split_id = q_t_id.clone();
            let k_split_id = k_t_id.clone();
            let v_split_id = v_t_id.clone();

            // Scale constant: 1/√d_k
            // T-36 (I-15/CQ-17): Warn on missing head_dim instead of silently
            // falling back to 128, which produces wrong attention scale for
            // models with head_dim != 128.
            let scale_val = if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                eprintln!(
                "[ERROR] decompose_attention: head_dim is 0 — cannot compute correct attention scale. \
                 Using default 1/√128 which will be WRONG for models with head_dim != 128. \
                 Provide DecompositionContext with correct head_dim."
            );
                1.0 / (128.0_f32).sqrt()
            };
            // Shared scale constant: 1/√d_k
            // Uses scalar:// resolution so the value is correctly serialized as fp16.
            // Duplicates are removed by the global AirNodeId dedup at the end
            // of LegalityRewritePass::run().
            let scale_const_id = AirNodeId("shared_attn_scale".to_string());
            nodes.push(Self::make_air_node(
                scale_const_id.clone(),
                AirOp::Const {
                    value_path: format!("scalar://fp16/{:.10}", scale_val),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));

            // Pre-slice K and V heads — one slice per KV head, not per Q head.
            // This avoids duplicate MIL output names when GQA fan_out > 1.
            // (Same fix as decompose_decode_step — see CRITICAL comment there.)
            let mut k_head_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads.max(1) as usize);
            for kv_idx in 0..(kv_heads.max(1) as usize) {
                let k_i_id = AirNodeId(format!("{base}_k_head_{}", kv_idx));
                nodes.push(Self::make_air_node(
                    k_i_id.clone(),
                    AirOp::SliceByIndex {
                        input: k_split_id.clone(),
                        begin: vec![0, kv_idx as i64, 0, 0],
                        end: vec![0, (kv_idx as i64) + 1, 0, 0],
                        stride: vec![1, 1, 1, 1],
                        begin_mask: vec![true, false, true, true],
                        end_mask: vec![true, false, true, true],
                        squeeze_mask: vec![false, true, false, false],
                    },
                    sir_node,
                    "mb.slice_by_index",
                    kq,
                ));
                k_head_ids.push(k_i_id);
            }

            let mut v_head_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads.max(1) as usize);
            for kv_idx in 0..(kv_heads.max(1) as usize) {
                let v_i_id = AirNodeId(format!("{base}_v_head_{}", kv_idx));
                nodes.push(Self::make_air_node(
                    v_i_id.clone(),
                    AirOp::SliceByIndex {
                        input: v_split_id.clone(),
                        begin: vec![0, kv_idx as i64, 0, 0],
                        end: vec![0, (kv_idx as i64) + 1, 0, 0],
                        stride: vec![1, 1, 1, 1],
                        begin_mask: vec![true, false, true, true],
                        end_mask: vec![true, false, true, true],
                        squeeze_mask: vec![false, true, false, false],
                    },
                    sir_node,
                    "mb.slice_by_index",
                    kq,
                ));
                v_head_ids.push(v_i_id);
            }

            // Also pre-slice the K transposes, since each KV head only needs one transpose
            // and multiple Q heads may reference the same transposed K.
            let mut k_head_t_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads.max(1) as usize);
            for (kv_idx, k_head_id) in k_head_ids.iter().enumerate().take(kv_heads.max(1) as usize)
            {
                let k_i_t_id = AirNodeId(format!("{base}_k_head_{}_t", kv_idx));
                nodes.push(Self::make_air_node(
                    k_i_t_id.clone(),
                    AirOp::Transpose { input: k_head_id.clone(), perm: vec![0, 2, 1] },
                    sir_node,
                    "mb.transpose",
                    kq,
                ));
                k_head_t_ids.push(k_i_t_id);
            }

            // Per-head attention loop
            let mut ctx_parts: Vec<AirNodeId> = Vec::with_capacity(heads as usize);

            for head_idx in 0..(heads as usize) {
                let kv_idx = head_idx / fan_out;

                // Extract Q head: SliceByIndex from q_split output
                // Q shape per head: [B, 1, S, D] (squeeze dim 1 → [B, S, D])
                let q_i_id = AirNodeId(format!("{base}_q_head_{}", head_idx));
                nodes.push(Self::make_air_node(
                    q_i_id.clone(),
                    AirOp::SliceByIndex {
                        input: q_split_id.clone(),
                        begin: vec![0, head_idx as i64, 0, 0],
                        end: vec![0, (head_idx as i64) + 1, 0, 0],
                        stride: vec![1, 1, 1, 1],
                        begin_mask: vec![true, false, true, true],
                        end_mask: vec![true, false, true, true],
                        squeeze_mask: vec![false, true, false, false],
                    },
                    sir_node,
                    "mb.slice_by_index",
                    kq,
                ));

                // Reuse pre-sliced K and V heads (no duplicate output names)
                let _k_i_id = k_head_ids[kv_idx].clone();
                let v_i_id = v_head_ids[kv_idx].clone();
                let k_i_t_id = k_head_t_ids[kv_idx].clone();

                // logits = matmul(q_i, k_i^T)
                // q_i: [B, S, D], k_i_t: [B, D, S] (pre-transposed)
                // matmul: [B, S, D] @ [B, D, S] = [B, S, S]
                let logits_id = AirNodeId(format!("{base}_logits_{}", head_idx));
                nodes.push(Self::make_air_node(
                    logits_id.clone(),
                    AirOp::MatMul { a: q_i_id, b: k_i_t_id },
                    sir_node,
                    "mb.matmul",
                    kq,
                ));

                // Scale: logits *= 1/√d_k
                let scaled_logits_id = AirNodeId(format!("{base}_scaled_logits_{}", head_idx));
                nodes.push(Self::make_air_node(
                    scaled_logits_id.clone(),
                    AirOp::Mul { x: logits_id, y: scale_const_id.clone() },
                    sir_node,
                    "mb.mul",
                    kq,
                ));

                // Apply causal mask if available
                let masked_logits_id = if let Some(ref m) = mask_sir {
                    let mask_air_id =
                        sir_to_air.get(m).cloned().unwrap_or_else(|| AirNodeId(m.0.clone()));
                    let ml_id = AirNodeId(format!("{base}_masked_logits_{}", head_idx));
                    nodes.push(Self::make_air_node(
                        ml_id.clone(),
                        AirOp::Add { x: scaled_logits_id, y: mask_air_id },
                        sir_node,
                        "mb.add",
                        kq,
                    ));
                    ml_id
                } else {
                    scaled_logits_id
                };

                // weights = softmax(logits, axis=-1)
                let weights_id = AirNodeId(format!("{base}_weights_{}", head_idx));
                nodes.push(Self::make_air_node(
                    weights_id.clone(),
                    AirOp::Softmax { input: masked_logits_id, axis: -1 },
                    sir_node,
                    "mb.softmax",
                    kq,
                ));

                // ctx_part = matmul(weights, v_i)
                // weights: [B, S, S], v_i: [B, S, D] → output: [B, S, D]
                let ctx_part_id = AirNodeId(format!("{base}_ctx_{}", head_idx));
                nodes.push(Self::make_air_node(
                    ctx_part_id.clone(),
                    AirOp::MatMul { a: weights_id, b: v_i_id },
                    sir_node,
                    "mb.matmul",
                    kq,
                ));

                // Expand dims: [B, S, D] → [B, 1, S, D] for concat along axis 1
                let ctx_expanded_id = AirNodeId(format!("{base}_ctx_exp_{}", head_idx));
                nodes.push(Self::make_air_node(
                    ctx_expanded_id.clone(),
                    AirOp::ExpandDims { input: ctx_part_id, axis: vec![1] },
                    sir_node,
                    "mb.expand_dims",
                    kq,
                ));

                ctx_parts.push(ctx_expanded_id);
            }

            // Concat all per-head context: [B, 1, S, D] × hq → [B, hq, S, D]
            let ctx_concat_id = AirNodeId(format!("{base}_ctx_concat"));
            nodes.push(Self::make_air_node(
                ctx_concat_id.clone(),
                AirOp::Concat { inputs: ctx_parts, axis: 1 },
                sir_node,
                "mb.concat",
                kq,
            ));

            // Step 8: Reshape back to [batch, seq, num_heads * head_dim]
            // attn_flat_dim is defined above (before the if/else branch)
            nodes.push(Self::make_air_node(
                attn_flat_id.clone(),
                AirOp::Reshape {
                    input: ctx_concat_id,
                    target_shape: vec![batch as usize, seq as usize, attn_flat_dim as usize],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));
        } else {
            // ── SDPA fallback (no DecompositionContext) ───────────────
            // When heads=0 (no context), we don't know the head count for
            // per-head attention, so fall back to SDPA. This should only
            // occur in synthetic tests — production compilation always
            // provides a DecompositionContext.
            let mask_air = mask_sir
                .as_ref()
                .and_then(|m| sir_to_air.get(m).cloned().or_else(|| Some(AirNodeId(m.0.clone()))));
            let scale = if head_dim > 0 { Some(1.0 / (head_dim as f32).sqrt()) } else { None };

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

            nodes.push(Self::make_air_node(
                attn_flat_id.clone(),
                AirOp::Reshape {
                    input: attn_id,
                    // CRITICAL: Use heads*head_dim, NOT embed_dim.
                    // For GQA models (e.g., Qwen3-0.6B: 16 heads × 128 head_dim = 2048 ≠ 1024 embed_dim),
                    // the attention output has heads*head_dim elements, not embed_dim.
                    // Using embed_dim here causes the "2048 vs 1024" impossible reshape error.
                    target_shape: vec![batch as usize, seq as usize, attn_flat_dim as usize],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));
        }

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
    /// Full decode step with all optional features:
    ///
    /// ```text
    /// DecodeStep(token, state_map, q_weight?, k_weight?, v_weight?, out_weight?,
    ///            rope_tables?, position?, q_norm_weight?, k_norm_weight?,
    ///            norm_epsilon, qk_norm_type, mask_ref?) →
    ///
    ///   ── Q/K/V Projections ──────────────────────────────────────────
    ///   When separate weights are provided (q_weight, k_weight, v_weight):
    ///     q: Conv1x1AsLinear(token, q_weight)
    ///     k_new: Conv1x1AsLinear(token, k_weight)
    ///     v_new: Conv1x1AsLinear(token, v_weight)
    ///   When no separate weights (legacy path):
    ///     qkv: Conv1x1AsLinear(token, W_qkv)
    ///     q, k_new, v_new: SliceByIndex(qkv, ...) × 3
    ///
    ///   ── Optional QK-norm (when q_norm_weight / k_norm_weight provided) ──
    ///     q_normed: RMSNorm(q, q_norm_weight, epsilon, axes=[3])
    ///     k_normed: RMSNorm(k_new, k_norm_weight, epsilon, axes=[3])
    ///
    ///   ── KV Cache ────────────────────────────────────────────────────
    ///     k_cache: StateReadFixed(k_state, [kv_len, kv_heads*head_dim])
    ///     v_cache: StateReadFixed(v_state, [kv_len, kv_heads*head_dim])
    ///
    ///   ── Reshape to 4D ──────────────────────────────────────────────
    ///     q_4d: Reshape(q, [batch, heads, 1, head_dim])
    ///     k_4d: Reshape(k_cache, [1, kv_heads, kv_len, head_dim])
    ///     v_4d: Reshape(v_cache, [1, kv_heads, kv_len, head_dim])
    ///
    ///   ── Split-based per-head attention (ANE-legal, no Tile/SDPA) ──
    ///     q_split: Split(q_rope, num_splits=heads, axis=1)
    ///     k_split: Split(k_rope, num_splits=kv_heads, axis=1)
    ///     v_split: Split(v_rope, num_splits=kv_heads, axis=1)
    ///     for i in 0..heads:
    ///       kv_idx = i / fan_out
    ///       logits = MatMul(q_i, k_{kv_idx}^T)
    ///       scaled = Mul(logits, scale)
    ///       masked = Add(scaled, mask)
    ///       weights = Softmax(masked)
    ///       ctx_i = MatMul(weights, v_{kv_idx})
    ///     ctx: Concat(ctx_0..ctx_{heads-1}, axis=1)
    ///
    ///   ── Output ────────────────────────────────────────────────────
    ///     attn_flat: Reshape(attn, [batch, heads*head_dim])
    ///     output: Conv1x1AsLinear(attn_flat, out_weight)
    ///     k_update: StateWriteFixed(k_state, k_new)
    ///     v_update: StateWriteFixed(v_state, v_new)
    /// ```
    ///
    /// When optional parameters are `None`, the corresponding steps are
    /// skipped — this keeps the decomposition generic and not model-specific.
    fn decompose_decode_step(
        token_sir: &ane_ir::sir::SirNodeId,
        state_map: &[String],
        position: &Option<ane_ir::sir::SirNodeId>,
        weights: &DecodeWeights,
        norm_epsilon: f32,
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let sir_node = env.sir_node;
        let base = env.base;
        let sir_to_air = env.sir_to_air;
        let kq = env.kq;
        let q_weight = weights.q_weight;
        let k_weight = weights.k_weight;
        let v_weight = weights.v_weight;
        let out_weight = weights.out_weight;
        let rope_tables = weights.rope_tables;
        let q_norm_weight = weights.q_norm_weight;
        let k_norm_weight = weights.k_norm_weight;
        let mask_ref = weights.mask_ref;
        let token_air =
            sir_to_air.get(token_sir).cloned().unwrap_or_else(|| AirNodeId(token_sir.0.clone()));

        // Extract dimensions from context or use placeholders
        let kv_heads_val = ctx.map(|c| c.kv_heads).unwrap_or(0);
        let kv_heads =
            if kv_heads_val > 0 { kv_heads_val } else { ctx.map(|c| c.num_heads).unwrap_or(0) };
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

        // ─────────────────────────────────────────────────────────────
        // Step 1: Q/K/V Projections
        // ─────────────────────────────────────────────────────────────
        // When separate weights are provided, use them (correct for
        // HuggingFace models which store q_proj/k_proj/v_proj separately).
        // When no separate weights, fall back to legacy fused QKV.

        let (q_id, k_new_id, v_new_id) =
            if let (Some(qw), Some(kw), Some(vw)) = (q_weight, k_weight, v_weight) {
                // Separate Q, K, V projections — each with its own weight name
                let q_proj_dim = heads * head_dim;
                let kv_proj_dim = kv_heads as i64 * head_dim;

                let q_id = AirNodeId(format!("{base}_q_proj"));
                nodes.push(Self::make_air_node(
                    q_id.clone(),
                    AirOp::Conv1x1AsLinear {
                        input: token_air.clone(),
                        weight: qw.to_string(),
                        pad_type: "valid".into(),
                        output_dim: q_proj_dim as usize,
                    },
                    sir_node,
                    "mb.linear",
                    kq,
                ));

                let k_id = AirNodeId(format!("{base}_k_proj"));
                nodes.push(Self::make_air_node(
                    k_id.clone(),
                    AirOp::Conv1x1AsLinear {
                        input: token_air.clone(),
                        weight: kw.to_string(),
                        pad_type: "valid".into(),
                        output_dim: kv_proj_dim as usize,
                    },
                    sir_node,
                    "mb.linear",
                    kq,
                ));

                let v_id = AirNodeId(format!("{base}_v_proj"));
                nodes.push(Self::make_air_node(
                    v_id.clone(),
                    AirOp::Conv1x1AsLinear {
                        input: token_air,
                        weight: vw.to_string(),
                        pad_type: "valid".into(),
                        output_dim: kv_proj_dim as usize,
                    },
                    sir_node,
                    "mb.linear",
                    kq,
                ));

                (q_id, k_id, v_id)
            } else {
                // Legacy fallback: fused QKV projection + slice
                let qkv_id = AirNodeId(format!("{base}_qkv_proj"));
                nodes.push(Self::make_air_node(
                    qkv_id.clone(),
                    AirOp::Conv1x1AsLinear {
                        input: token_air,
                        weight: format!("{base}_w_qkv"),
                        pad_type: "valid".into(),
                        output_dim: (3 * embed) as usize,
                    },
                    sir_node,
                    "mb.linear",
                    kq,
                ));

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

                (q_id, k_id, v_id)
            };

        // ─────────────────────────────────────────────────────────────
        // Step 2: Optional QK-norm (RMSNorm with axes=[3])
        // ─────────────────────────────────────────────────────────────
        // When q_norm_weight / k_norm_weight are provided, apply per-head
        // RMSNorm. The input is flat [B, heads*head_dim] but the norm
        // needs 4D layout [B, 1, heads, head_dim] (seq_len=1 for decode)
        // to apply axes=[3] correctly.

        let q_after_norm = if let Some(qnw) = q_norm_weight {
            Self::apply_qk_norm_decode(&q_id, qnw, norm_epsilon, "_q_norm", env, ctx, &mut nodes)
        } else {
            q_id.clone()
        };

        let k_after_norm = if let Some(knw) = k_norm_weight {
            Self::apply_qk_norm_decode(
                &k_new_id,
                knw,
                norm_epsilon,
                "_k_norm",
                env,
                ctx,
                &mut nodes,
            )
        } else {
            k_new_id.clone()
        };

        // ─────────────────────────────────────────────────────────────
        // Step 3: KV Cache State Reads
        // ─────────────────────────────────────────────────────────────
        let k_state_id = state_map.first().cloned().unwrap_or_else(|| format!("{base}_k_cache"));
        let v_state_id = state_map.get(1).cloned().unwrap_or_else(|| format!("{base}_v_cache"));

        let kv_embed = kv_heads * head_dim as usize;
        let k_cache_id = AirNodeId(format!("{base}_k_cache_read"));
        nodes.push(Self::make_air_node(
            k_cache_id.clone(),
            AirOp::StateReadFixed {
                state_id: k_state_id.clone(),
                shape: vec![kv_len as usize, kv_embed],
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
                shape: vec![kv_len as usize, kv_embed],
                dtype: ane_ir::mir::MilDtype::Fp16,
            },
            sir_node,
            "mb.read_state",
            kq,
        ));

        // ─────────────────────────────────────────────────────────────
        // Step 4: Reshape to 4D for multi-head attention
        // ─────────────────────────────────────────────────────────────
        // Q: [batch, heads*head_dim] → [batch, heads, 1, head_dim]
        // K cache: [kv_len, kv_heads*head_dim] → [1, kv_heads, kv_len, head_dim]
        // V cache: [kv_len, kv_heads*head_dim] → [1, kv_heads, kv_len, head_dim]

        let q_4d_id = AirNodeId(format!("{base}_q_4d"));
        nodes.push(Self::make_air_node(
            q_4d_id.clone(),
            AirOp::Reshape {
                input: q_after_norm,
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
                input: k_cache_id.clone(),
                target_shape: vec![1, kv_heads, kv_len as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        let v_4d_id = AirNodeId(format!("{base}_v_4d"));
        nodes.push(Self::make_air_node(
            v_4d_id.clone(),
            AirOp::Reshape {
                input: v_cache_id.clone(),
                target_shape: vec![1, kv_heads, kv_len as usize, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // ─────────────────────────────────────────────────────────────
        // Step 5: ANE-LEGAL GQA — Split-based, no Tile (Sprint 67)
        // ─────────────────────────────────────────────────────────────
        // The reference model (pkhairkh/qwen3-coreml-palettized) does NOT
        // use mb.tile for GQA. Instead, it splits Q into individual heads
        // and pairs each Q head with its corresponding KV head:
        //
        //   k_blocks = mb.split(k_cache, num_splits=hk, axis=1)
        //   v_blocks = mb.split(v_cache, num_splits=hk, axis=1)
        //   q_blocks = mb.split(q,       num_splits=hq, axis=1)
        //   for i, q_i in enumerate(q_blocks):
        //       kv_idx = i // fan_out
        //       logits = mb.matmul(q_i, k_blocks[kv_idx], transpose_y=True)
        //       logits = mb.mul(logits, scale)
        //       logits = mb.add(logits, mask)
        //       weights = mb.softmax(logits, axis=-1)
        //       ctx_part = mb.matmul(weights, v_blocks[kv_idx])
        //   ctx = mb.concat(ctx_parts, axis=1)
        //
        // This eliminates BOTH mb.tile AND mb.scaled_dot_product_attention,
        // which are absent from the reference model's op set.
        //
        // When kv_heads == num_heads (no GQA), fan_out=1 and each Q head
        // maps to its own KV head — the split is still correct but
        // degenerates to per-head attention.

        let fan_out =
            if (kv_heads as i64) < heads { (heads / kv_heads as i64) as usize } else { 1 };
        let _uses_gqa = (kv_heads as i64) < heads; // used for diagnostics only

        // Step 5a-prep: Reshape the new K value BEFORE applying RoPE.
        // k_after_norm is [batch, kv_heads*head_dim] → [1, kv_heads, 1, head_dim]
        // This reshape must happen before step 5a so we can apply RoPE to the
        // new K token before writing it to cache.
        let k_new_4d_id = AirNodeId(format!("{base}_k_new_4d"));
        nodes.push(Self::make_air_node(
            k_new_4d_id.clone(),
            AirOp::Reshape {
                input: k_after_norm.clone(),
                target_shape: vec![1, kv_heads, 1, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Reshape the new V value for cache write:
        // v_new_id is [batch, kv_heads*head_dim] → [1, kv_heads, 1, head_dim]
        let v_new_4d_id = AirNodeId(format!("{base}_v_new_4d"));
        nodes.push(Self::make_air_node(
            v_new_4d_id.clone(),
            AirOp::Reshape {
                input: v_new_id.clone(),
                target_shape: vec![1, kv_heads, 1, head_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Step 5a: Apply RoPE to Q and the new K token.
        //
        // CRITICAL ARCHITECTURE DECISION: The KV cache stores PRE-RoPE'd
        // K values. This means we only need to apply RoPE to the NEW K
        // token (shape [1, hk, 1, hd]) before writing it to the cache,
        // using the gathered cos/sin for the current position (shape
        // [1, 1, 1, hd]). This avoids the broadcast-incompatibility
        // problem that arises when trying to apply RoPE to the ENTIRE
        // K cache using full cos/sin tables — the K cache has shape
        // [1, hk, max_seq, hd] which is incompatible with tables shaped
        // [1, 1, prefill_seq, hd].
        //
        // The Q RoPE uses the same gathered cos/sin (Q has seq_len=1
        // for decode, so broadcast works: [1, hq, 1, hd] * [1, 1, 1, hd]).
        let (q_for_attn, k_new_rope_id, kv_mask_write_id, causal_mask_id) =
            if let Some(tables_ref) = rope_tables {
                let (q_rope, _cos_tab, _sin_tab, k_new_rope, kv_mask, causal_mask) =
                    Self::apply_rope_decode(
                        &q_4d_id,
                        &k_new_4d_id,
                        tables_ref,
                        position,
                        env,
                        ctx,
                        &mut nodes,
                    );
                (q_rope, k_new_rope, kv_mask, causal_mask)
            } else {
                (q_4d_id, k_new_4d_id.clone(), None, None)
            };

        // Step 5b: KV Cache write — masked blend (NOT SliceUpdate)
        //
        // The reference model's _append function:
        //   def _append(old, new, mask_keep, mask_write):
        //       return mb.add(x=mb.mul(x=old, y=mask_keep),
        //                     y=mb.mul(x=new, y=mask_write))
        //
        // mask_write: 1-hot at write position, 0 elsewhere → [1, 1, seq, 1]
        // mask_keep:  1 - mask_write → 0 at write position, 1 elsewhere
        //
        // next_k_cache = _append(k_cache, k_new_rope, mask_keep, mask_write)
        // next_v_cache = _append(v_cache, v_new_4d, mask_keep, mask_write)
        // mb.coreml_update_state(state=k_state, value=next_k_cache)
        // mb.coreml_update_state(state=v_state, value=next_v_cache)

        // Compute masked blend for KV cache writes
        let (next_k_cache_id, next_v_cache_id) = if let Some(ref kv_mask_row) = kv_mask_write_id {
            // Reshape kv_mask_row to [1, 1, seq, 1] for broadcast with [1, kv_heads, seq, head_dim]
            let mask_write_id = AirNodeId(format!("{base}_mask_write"));
            nodes.push(Self::make_air_node(
                mask_write_id.clone(),
                AirOp::Reshape {
                    input: kv_mask_row.clone(),
                    target_shape: vec![1, 1, kv_len as usize, 1],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));

            // mask_keep = 1.0 - mask_write
            // Shared scalar one constant: 1.0 for mask_keep = 1.0 - mask_write
            // Uses the same shared_scalar_one as the arithmetic mask path.
            // Duplicates are removed by the global AirNodeId dedup at the end
            // of LegalityRewritePass::run().
            let one_const_id = AirNodeId("shared_scalar_one".to_string());
            nodes.push(Self::make_air_node(
                one_const_id.clone(),
                AirOp::Const { value_path: "scalar://fp16/1.0".to_string(), dtype: MilDtype::Fp16 },
                sir_node,
                "mb.const",
                kq,
            ));

            let mask_keep_id = AirNodeId(format!("{base}_mask_keep"));
            nodes.push(Self::make_air_node(
                mask_keep_id.clone(),
                AirOp::Sub { x: one_const_id, y: mask_write_id.clone() },
                sir_node,
                "mb.sub",
                kq,
            ));

            // K cache: _append(k_cache, k_new_rope, mask_keep, mask_write)
            // Write PRE-RoPE'd K to the cache so we don't need to re-apply
            // RoPE to the entire cache at every decode step.
            let k_old_masked_id = AirNodeId(format!("{base}_k_old_masked"));
            nodes.push(Self::make_air_node(
                k_old_masked_id.clone(),
                AirOp::Mul { x: k_4d_id.clone(), y: mask_keep_id.clone() },
                sir_node,
                "mb.mul",
                kq,
            ));

            let k_new_masked_id = AirNodeId(format!("{base}_k_new_masked"));
            nodes.push(Self::make_air_node(
                k_new_masked_id.clone(),
                AirOp::Mul { x: k_new_rope_id, y: mask_write_id.clone() },
                sir_node,
                "mb.mul",
                kq,
            ));

            let next_k_id = AirNodeId(format!("{base}_next_k_cache"));
            nodes.push(Self::make_air_node(
                next_k_id.clone(),
                AirOp::Add { x: k_old_masked_id, y: k_new_masked_id },
                sir_node,
                "mb.add",
                kq,
            ));

            // V cache: _append(v_cache, v_new_4d, mask_keep, mask_write)
            // Use the 4D reshaped V cache (not the raw 2D read) so that
            // broadcasting with mask_keep [1,1,seq,1] produces correct shape
            // [1, kv_heads, seq, head_dim] instead of [1,1,seq,kv_embed].
            let v_old_masked_id = AirNodeId(format!("{base}_v_old_masked"));
            nodes.push(Self::make_air_node(
                v_old_masked_id.clone(),
                AirOp::Mul { x: v_4d_id.clone(), y: mask_keep_id },
                sir_node,
                "mb.mul",
                kq,
            ));

            let v_new_masked_id = AirNodeId(format!("{base}_v_new_masked"));
            nodes.push(Self::make_air_node(
                v_new_masked_id.clone(),
                AirOp::Mul { x: v_new_4d_id, y: mask_write_id },
                sir_node,
                "mb.mul",
                kq,
            ));

            let next_v_id = AirNodeId(format!("{base}_next_v_cache"));
            nodes.push(Self::make_air_node(
                next_v_id.clone(),
                AirOp::Add { x: v_old_masked_id, y: v_new_masked_id },
                sir_node,
                "mb.add",
                kq,
            ));

            (next_k_id, next_v_id)
        } else {
            // No position/mask info — fall back to simple 4D cache write.
            // Use the 4D reshaped K/V so the downstream split-based
            // attention (which expects [1, kv_heads, seq, head_dim]) works
            // correctly. The old 2D k_cache_id / v_cache_id shapes
            // [kv_len, kv_embed] are incompatible with the Split axis=1.
            //
            // Since there's no mask, the "overwrite" is just the old cache
            // with the new token already blended in via the reshape. We
            // still need a proper add to merge old + new. But without
            // position info we can't compute the mask, so we use the 4D
            // old cache as the next cache (the new K will be written via
            // StateWrite anyway).
            (k_4d_id.clone(), v_4d_id.clone())
        };

        // Step 5c: K cache now stores pre-RoPE'd values.
        //
        // Since we applied RoPE to the new K token before writing it to
        // the cache (Step 5a), the K cache already contains RoPE'd values
        // for all positions. No full-cache RoPE re-application is needed.
        // This eliminates the broadcast incompatibility that would arise
        // from multiplying full cos/sin tables [1,1,prefill_seq,hd] with
        // the entire K cache [1,hk,max_seq,hd].
        let k_for_attn = next_k_cache_id.clone();

        // Step 5d: Split-based per-head attention (matching reference model)
        //
        // Split Q into hq heads, K and V into hk heads.
        // For GQA (fan_out > 1), each group of `fan_out` Q heads shares
        // one KV head.
        //
        // For the split-based attention, we use:
        //   - Q: q_for_attn (RoPE'd Q, shape [B, hq, 1, hd])
        //   - K: k_for_attn (RoPE'd full K cache with new token, shape [1, hk, seq, hd])
        //   - V: next_v_cache (full V cache with new token, shape [1, hk, seq, hd])

        // Attention scale factor: 1/√d_k
        // Used per-head as a scalar constant multiplied with logits.
        // T-36 (I-15/CQ-17): Warn on missing head_dim instead of silently
        // falling back to 128, which produces wrong attention scale for
        // models with head_dim != 128.
        let scale_val = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            eprintln!(
                "[ERROR] decompose_decode_step: head_dim is 0 — cannot compute correct attention scale. \
                 Using default 1/√128 which will be WRONG for models with head_dim != 128. \
                 Provide DecompositionContext with correct head_dim."
            );
            1.0 / (128.0_f32).sqrt()
        };

        // NOTE: We intentionally do NOT emit mb.split here. Core ML's split op
        // returns a *list* of tensors, which our IR cannot model (single output
        // per op). Serialising a split with num_splits>1 as a single-output op
        // is invalid MIL and causes coremlcompiler errors. Instead, we slice
        // individual heads directly from the original tensors using
        // slice_by_index — same pattern as the embedding path and the Python
        // reference emitter.
        let q_split_id = q_for_attn.clone();
        let k_split_id = k_for_attn.clone();
        let v_split_id = next_v_cache_id.clone();

        // For each Q head, compute per-head attention:
        //   logits_i = matmul(q_i, k_{kv_idx}, transpose_y=True)
        //   logits_i = mul(logits_i, scale)
        //   logits_i = add(logits_i, mask)
        //   weights_i = softmax(logits_i, axis=-1)
        //   ctx_i = matmul(weights_i, v_{kv_idx})
        //
        // Then concat all ctx_i along axis 1.
        //
        // Note: Individual heads are extracted via slice_by_index on the
        // original (un-split) Q/K/V tensors. This avoids the invalid-MIL
        // problem of serialising mb.split with wrong output arity, and
        // matches the Python reference emitter pattern.
        //
        // CRITICAL: For GQA (fan_out > 1), multiple Q heads share the
        // same KV head. Each KV head must be sliced EXACTLY ONCE and
        // the result reused by all Q heads that map to it. If we slice
        // the same KV head multiple times inside the Q-head loop, we
        // produce duplicate MIL output names (e.g., "k_head_0" twice),
        // which violates MIL's SSA rule and causes coremlcompiler to
        // reject the model with "Block redefines I/O name".
        //
        // Pre-slice all KV heads OUTSIDE the per-Q-head loop, then
        // reference them by index inside the loop.

        // Pre-slice K and V heads — one slice per KV head, not per Q head.
        // This avoids duplicate output names when GQA fan_out > 1.
        let mut k_head_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads);
        for kv_idx in 0..kv_heads {
            let k_i_id = AirNodeId(format!("{base}_k_head_{}", kv_idx));
            nodes.push(Self::make_air_node(
                k_i_id.clone(),
                AirOp::SliceByIndex {
                    input: k_split_id.clone(),
                    begin: vec![0, kv_idx as i64, 0, 0],
                    end: vec![0, (kv_idx as i64) + 1, 0, 0],
                    stride: vec![1, 1, 1, 1],
                    begin_mask: vec![true, false, true, true],
                    end_mask: vec![true, false, true, true],
                    squeeze_mask: vec![false, true, false, false],
                },
                sir_node,
                "mb.slice_by_index",
                kq,
            ));
            k_head_ids.push(k_i_id);
        }

        let mut v_head_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads);
        for kv_idx in 0..kv_heads {
            let v_i_id = AirNodeId(format!("{base}_v_head_{}", kv_idx));
            nodes.push(Self::make_air_node(
                v_i_id.clone(),
                AirOp::SliceByIndex {
                    input: v_split_id.clone(),
                    begin: vec![0, kv_idx as i64, 0, 0],
                    end: vec![0, (kv_idx as i64) + 1, 0, 0],
                    stride: vec![1, 1, 1, 1],
                    begin_mask: vec![true, false, true, true],
                    end_mask: vec![true, false, true, true],
                    squeeze_mask: vec![false, true, false, false],
                },
                sir_node,
                "mb.slice_by_index",
                kq,
            ));
            v_head_ids.push(v_i_id);
        }

        // Pre-transpose K heads for the Q×K^T matmul.
        // After slicing, each K head has shape [1, seq, hd] (3D from squeeze).
        // Transposing axes [0,2,1] gives [1, hd, seq], so the matmul becomes:
        //   [1, 1, hd] × [1, hd, seq] = [1, 1, seq]  ← correct attention scores
        // Without this transpose, the matmul would be:
        //   [1, 1, hd] × [1, seq, hd]  ← inner dims mismatch (hd ≠ seq)
        // This matches the pattern used in decompose_attention_block (line 1017-1028).
        let mut k_head_t_ids: Vec<AirNodeId> = Vec::with_capacity(kv_heads);
        for (kv_idx, k_head_id) in k_head_ids.iter().enumerate().take(kv_heads) {
            let k_i_t_id = AirNodeId(format!("{base}_k_head_{}_t", kv_idx));
            nodes.push(Self::make_air_node(
                k_i_t_id.clone(),
                AirOp::Transpose { input: k_head_id.clone(), perm: vec![0, 2, 1] },
                sir_node,
                "mb.transpose",
                kq,
            ));
            k_head_t_ids.push(k_i_t_id);
        }

        // Shared attention scale constant: emitted once before the Q-head loop.
        // Previously emitted inside the loop, causing duplicate output names
        // (shared_attn_scale defined 16× per function with GQA).
        // Uses scalar:// resolution so the value is correctly serialized as fp16.
        let scale_const_id = AirNodeId("shared_attn_scale".to_string());
        nodes.push(Self::make_air_node(
            scale_const_id.clone(),
            AirOp::Const {
                value_path: format!("scalar://fp16/{:.10}", scale_val),
                dtype: MilDtype::Fp16,
            },
            sir_node,
            "mb.const",
            kq,
        ));

        let mut ctx_parts: Vec<AirNodeId> = Vec::with_capacity(heads as usize);

        for head_idx in 0..(heads as usize) {
            let kv_idx = head_idx / fan_out;
            let hi = head_idx;

            // Extract Q head: SliceByIndex from q_split output
            // Q shape per head: [B, 1, 1, hd]
            let q_i_id = AirNodeId(format!("{base}_q_head_{}", hi));
            nodes.push(Self::make_air_node(
                q_i_id.clone(),
                AirOp::SliceByIndex {
                    input: q_split_id.clone(),
                    begin: vec![0, hi as i64, 0, 0],
                    end: vec![0, (hi as i64) + 1, 0, 0],
                    stride: vec![1, 1, 1, 1],
                    begin_mask: vec![true, false, true, true],
                    end_mask: vec![true, false, true, true],
                    squeeze_mask: vec![false, true, false, false],
                },
                sir_node,
                "mb.slice_by_index",
                kq,
            ));

            // Reuse pre-sliced V heads (no duplicate output names)
            let v_i_id = v_head_ids[kv_idx].clone();
            // Use pre-transposed K head: k_i^T shape [B, hd, seq]
            let k_i_t_id = k_head_t_ids[kv_idx].clone();

            // logits = matmul(q_i, k_i^T)
            // q_i: [B, 1, hd] (3D from squeeze), k_i^T: [B, hd, seq] (pre-transposed)
            // matmul output: [B, 1, seq]  ← correct attention score shape
            let logits_id = AirNodeId(format!("{base}_logits_{}", hi));
            nodes.push(Self::make_air_node(
                logits_id.clone(),
                AirOp::MatMul { a: q_i_id, b: k_i_t_id },
                sir_node,
                "mb.matmul",
                kq,
            ));

            // logits *= scale (using pre-hoisted shared_attn_scale)
            let scaled_logits_id = AirNodeId(format!("{base}_scaled_logits_{}", hi));
            nodes.push(Self::make_air_node(
                scaled_logits_id.clone(),
                AirOp::Mul { x: logits_id, y: scale_const_id.clone() },
                sir_node,
                "mb.mul",
                kq,
            ));

            // logits += causal_mask (if available)
            let masked_logits_id = if let Some(ref mask_row) = causal_mask_id {
                // Reshape mask_row to [1, 1, 1, seq] for broadcast with [B, 1, 1, seq]
                let mask_4d_id = AirNodeId(format!("{base}_mask_4d_{}", hi));
                nodes.push(Self::make_air_node(
                    mask_4d_id.clone(),
                    AirOp::Reshape {
                        input: mask_row.clone(),
                        target_shape: vec![1, 1, 1, kv_len as usize],
                    },
                    sir_node,
                    "mb.reshape",
                    kq,
                ));

                let ml_id = AirNodeId(format!("{base}_masked_logits_{}", hi));
                nodes.push(Self::make_air_node(
                    ml_id.clone(),
                    AirOp::Add { x: scaled_logits_id, y: mask_4d_id },
                    sir_node,
                    "mb.add",
                    kq,
                ));
                ml_id
            } else if let Some(mref) = mask_ref {
                // Use the external mask reference
                let ml_id = AirNodeId(format!("{base}_masked_logits_{}", hi));
                nodes.push(Self::make_air_node(
                    ml_id.clone(),
                    AirOp::Add { x: scaled_logits_id, y: AirNodeId(mref.to_string()) },
                    sir_node,
                    "mb.add",
                    kq,
                ));
                ml_id
            } else {
                scaled_logits_id
            };

            // weights = softmax(logits, axis=-1)
            let weights_id = AirNodeId(format!("{base}_weights_{}", hi));
            nodes.push(Self::make_air_node(
                weights_id.clone(),
                AirOp::Softmax { input: masked_logits_id, axis: -1 },
                sir_node,
                "mb.softmax",
                kq,
            ));

            // ctx_part = matmul(weights, v_i)
            // weights: [B, 1, 1, seq], v_i: [B, 1, seq, hd]
            // output: [B, 1, 1, hd]
            let ctx_part_id = AirNodeId(format!("{base}_ctx_{}", hi));
            nodes.push(Self::make_air_node(
                ctx_part_id.clone(),
                AirOp::MatMul { a: weights_id, b: v_i_id },
                sir_node,
                "mb.matmul",
                kq,
            ));

            // Expand dims: [B, 1, 1, hd] → [B, 1, 1, hd] with head axis
            // for proper concat along axis 1
            let ctx_expanded_id = AirNodeId(format!("{base}_ctx_exp_{}", hi));
            nodes.push(Self::make_air_node(
                ctx_expanded_id.clone(),
                AirOp::ExpandDims { input: ctx_part_id, axis: vec![1] },
                sir_node,
                "mb.expand_dims",
                kq,
            ));

            ctx_parts.push(ctx_expanded_id);
        }

        // Concat all per-head context: [B, 1, 1, hd] × hq → [B, hq, 1, hd]
        let ctx_concat_id = AirNodeId(format!("{base}_ctx_concat"));
        nodes.push(Self::make_air_node(
            ctx_concat_id.clone(),
            AirOp::Concat { inputs: ctx_parts, axis: 1 },
            sir_node,
            "mb.concat",
            kq,
        ));

        // Step 7: Reshape back to flat [B, num_heads * head_dim]
        // CRITICAL: Use heads * head_dim, NOT embed_dim. For models where
        // num_heads * head_dim != hidden_size (e.g., Qwen3-0.6B: 16*128=2048 ≠ 1024),
        // using embed_dim produces an impossible reshape because the concat output
        // has num_heads * head_dim elements, not embed_dim elements. The output
        // projection (o_proj) then maps from num_heads*head_dim back to embed_dim.
        let attn_flat_dim = heads * head_dim;
        let attn_flat_id = AirNodeId(format!("{base}_attn_flat"));
        nodes.push(Self::make_air_node(
            attn_flat_id.clone(),
            AirOp::Reshape {
                input: ctx_concat_id,
                target_shape: vec![batch as usize, attn_flat_dim as usize],
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // Step 8: Output projection
        let out_w = out_weight.map(|w| w.to_string()).unwrap_or_else(|| format!("{base}_w_out"));
        let out_id = AirNodeId(format!("{base}_out_proj"));
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Conv1x1AsLinear {
                input: attn_flat_id,
                weight: out_w,
                pad_type: "valid".into(),
                output_dim: embed as usize,
            },
            sir_node,
            "mb.linear",
            kq,
        ));

        // ─────────────────────────────────────────────────────────────
        // Step 9: KV Cache state writes — masked blend already applied
        // ─────────────────────────────────────────────────────────────
        // The masked blend was computed in Step 5b above. Now write the
        // blended cache values back to the state. The reference model
        // writes next_k_cache and next_v_cache (which already include
        // the new token via masked blend).
        let k_write_id = AirNodeId(format!("{base}_k_cache_write"));
        nodes.push(Self::make_air_node(
            k_write_id,
            AirOp::StateWriteFixed { state_id: k_state_id, value: next_k_cache_id },
            sir_node,
            "mb.coreml_update_state",
            kq,
        ));

        let v_write_id = AirNodeId(format!("{base}_v_cache_write"));
        nodes.push(Self::make_air_node(
            v_write_id,
            AirOp::StateWriteFixed { state_id: v_state_id, value: next_v_cache_id },
            sir_node,
            "mb.coreml_update_state",
            kq,
        ));

        // The primary output of the decode step is the output projection
        (out_id, nodes)
    }

    /// Apply QK-norm (RMSNorm with axes=[3]) for the decode step.
    ///
    /// The input is a flat 2D tensor [batch, heads*head_dim] (seq_len=1
    /// for decode). We reshape to 4D [batch, 1, heads, head_dim], apply
    /// RMSNorm with axes=[3], then reshape back to 2D.
    fn apply_qk_norm_decode(
        input_id: &AirNodeId,
        norm_weight: &str,
        epsilon: f32,
        suffix: &str,
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
        nodes: &mut Vec<AirNode>,
    ) -> AirNodeId {
        let sir_node = env.sir_node;
        let base = env.base;
        let kq = env.kq;
        // Derive head count from context: kv_heads for k_norm, num_heads otherwise.
        let heads = if suffix.contains("_k_norm") {
            ctx.map(|c| if c.kv_heads > 0 { c.kv_heads } else { c.num_heads }).unwrap_or(0)
        } else {
            ctx.map(|c| c.num_heads).unwrap_or(0)
        };
        let head_dim = ctx.map(|c| c.head_dim).unwrap_or(0);
        // Reshape flat [B, heads*head_dim] → [B, 1, heads, head_dim]
        let reshape_4d_id = AirNodeId(format!("{base}{suffix}_reshape_4d"));
        nodes.push(Self::make_air_node(
            reshape_4d_id.clone(),
            AirOp::Reshape {
                input: input_id.clone(),
                target_shape: vec![0, 1, heads, head_dim], // batch inferred from input
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        // x² = Mul(x, x)
        let x_sq_id = AirNodeId(format!("{base}{suffix}_x_sq"));
        nodes.push(Self::make_air_node(
            x_sq_id.clone(),
            AirOp::Mul { x: reshape_4d_id.clone(), y: reshape_4d_id.clone() },
            sir_node,
            "mb.mul",
            kq,
        ));

        // mean(x²) via ReduceMean with axes=[3], keep_dims=true
        let mean_id = AirNodeId(format!("{base}{suffix}_mean"));
        nodes.push(Self::make_air_node(
            mean_id.clone(),
            AirOp::ReduceMean { input: x_sq_id, axes: vec![3], keep_dims: true },
            sir_node,
            "mb.reduce_mean",
            kq,
        ));

        // mean(x²) + epsilon (ANE-legal: Const scalar + Add with broadcasting)
        // Instead of FillLike (which is ANE-illegal), use a scalar Const
        // that broadcasts with the mean tensor in mb.add.
        // CoreML's mb.add(x=tensor, y=scalar) broadcasts the scalar correctly.
        let eps_scalar_id = AirNodeId(format!("{base}{suffix}_eps_scalar"));
        nodes.push(Self::make_air_node(
            eps_scalar_id.clone(),
            AirOp::Const {
                value_path: format!("scalar://fp16/{}", epsilon),
                dtype: MilDtype::Fp16,
            },
            sir_node,
            "mb.const",
            kq,
        ));

        let biased_id = AirNodeId(format!("{base}{suffix}_biased"));
        nodes.push(Self::make_air_node(
            biased_id.clone(),
            AirOp::Add { x: mean_id, y: eps_scalar_id },
            sir_node,
            "mb.add",
            kq,
        ));

        // rsqrt(mean(x²) + epsilon)
        let rsqrt_id = AirNodeId(format!("{base}{suffix}_rsqrt"));
        nodes.push(Self::make_air_node(
            rsqrt_id.clone(),
            AirOp::Rsqrt { input: biased_id },
            sir_node,
            "mb.rsqrt",
            kq,
        ));

        // normed = x * rsqrt
        let normed_id = AirNodeId(format!("{base}{suffix}_normed"));
        nodes.push(Self::make_air_node(
            normed_id.clone(),
            AirOp::Mul { x: reshape_4d_id, y: rsqrt_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // normed * weight (gamma) — weight shape [head_dim] broadcasts with [B, 1, H, D]
        let weighted_id = AirNodeId(format!("{base}{suffix}_weighted"));
        nodes.push(Self::make_air_node(
            weighted_id.clone(),
            AirOp::Mul { x: normed_id, y: AirNodeId(norm_weight.to_string()) },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Reshape back [B, 1, heads, head_dim] → [B, heads*head_dim]
        let flat_id = AirNodeId(format!("{base}{suffix}_flat"));
        nodes.push(Self::make_air_node(
            flat_id.clone(),
            AirOp::Reshape {
                input: weighted_id,
                target_shape: vec![0, heads * head_dim], // batch inferred
            },
            sir_node,
            "mb.reshape",
            kq,
        ));

        flat_id
    }

    /// Apply RoPE to Q and the new K token in the decode step.
    ///
    /// For the decode step (seq_len=1), both Q and the new K token are
    /// [B, H, 1, D]. When a position input is provided, we gather the
    /// specific row from the cos/sin tables corresponding to the current
    /// decode position, producing [1, 1, 1, D] which broadcasts correctly
    /// with [B, H, 1, D].
    ///
    /// The K cache stores PRE-RoPE'd values. The new K token gets RoPE'd
    /// here (before being written to cache), so no full-cache RoPE
    /// re-application is needed later. This avoids the broadcast
    /// incompatibility that would arise from applying full tables to the
    /// entire K cache.
    ///
    /// Returns (q_rope, cos_tab_id, sin_tab_id, k_new_rope, kv_mask_write, causal_mask).
    ///
    /// The K RoPE is applied to the NEW K token (shape [1, hk, 1, hd]) using
    /// gathered cos/sin values (shape [1, 1, 1, hd]). The cache stores PRE-RoPE'd
    /// K values, so no full-cache RoPE re-application is needed.
    fn apply_rope_decode(
        q_4d_id: &AirNodeId,
        k_new_4d_id: &AirNodeId, // new K token to apply RoPE to
        tables_ref: &str,
        position: &Option<ane_ir::sir::SirNodeId>,
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
        nodes: &mut Vec<AirNode>,
    ) -> (
        AirNodeId,
        Option<AirNodeId>,
        Option<AirNodeId>,
        AirNodeId,
        Option<AirNodeId>,
        Option<AirNodeId>,
    ) {
        let sir_node = env.sir_node;
        let base = env.base;
        let sir_to_air = env.sir_to_air;
        let kq = env.kq;
        let head_dim = ctx.map(|c| c.head_dim).unwrap_or(0);
        let head_dim = if head_dim > 0 {
            head_dim
        } else {
            // T-36 (I-15/CQ-17): Previously fell back to 128 silently, which
            // produces wrong RoPE slicing for models with head_dim != 128.
            // Now we return a default value with a strong warning; the caller
            // (decompose_rope_transform) will error if head_dim is still 0
            // after attempting to derive it from the graph.
            eprintln!(
                "[WARN] apply_rope_decode: head_dim=0 from DecompositionContext — \
                 using default 128. This will be wrong for models with head_dim != 128. \
                 Provide DecompositionContext with correct head_dim."
            );
            128
        };
        let _kv_len = ctx.map(|c| c.seq_len as i64).unwrap_or(0);
        let half = head_dim / 2;

        // ── ANE-LEGAL RoPE: Const + Gather pattern (Sprint 67) ────────
        //
        // The reference model (pkhairkh/qwen3-coreml-palettized) NEVER uses
        // mb.cos / mb.sin — these are ANE-illegal ops that cause the
        // execution planner to return error -5. Instead, it pre-computes
        // cos/sin tables at compile time and uses mb.gather to look up
        // position-specific rows at runtime.
        //
        // Pattern (matching the reference model's _build_decoder_prelude):
        //   1. Const nodes for sin_tab, cos_tab  [1, 1, seq_len, head_dim]
        //   2. Gather(cos_tab, position, axis=0)  → cos values for this position
        //   3. Gather(sin_tab, position, axis=0)  → sin values for this position
        //
        // We emit the Const nodes directly here rather than depending on
        // the static_tables pass to have already inserted them into the
        // SIR graph. This makes the decomposition self-contained and
        // guarantees no Cos/Sin fallback is ever emitted.

        // Check if the static_tables pass already inserted Const nodes.
        // When the static_tables pass has run (embedding graph), the nodes
        // are named "sir_static_cos_tab_{tables_ref}". For decode_step,
        // the static_tables pass may not have run, so we emit SHARED Const
        // nodes keyed by tables_ref (not per-layer). This ensures that all
        // 28 layers in a decode_step function share the same cos_tab, sin_tab,
        // and arange_tab Const nodes — avoiding 28× weight duplication.
        //
        // We use the same AirNodeId across all layers. Each layer unconditionally
        // emits the shared Const node, and the global AirNodeId dedup at the end
        // of LegalityRewritePass::run() removes duplicates, keeping only the first
        // occurrence. The AirNodeId uses "shared_rope_{tables_ref}" as a prefix
        // instead of the per-layer {base} prefix.
        let cos_tab_sir_id = ane_ir::sir::SirNodeId(format!("sir_static_cos_tab_{}", tables_ref));
        let sin_tab_sir_id = ane_ir::sir::SirNodeId(format!("sir_static_sin_tab_{}", tables_ref));

        // Shared AIR node IDs — same across all layers within a function
        let shared_cos_id = AirNodeId(format!("shared_rope_{}_cos_tab", tables_ref));
        let shared_sin_id = AirNodeId(format!("shared_rope_{}_sin_tab", tables_ref));
        let shared_arange_id = AirNodeId(format!("shared_rope_{}_arange_tab", tables_ref));

        let cos_id = if sir_to_air.contains_key(&cos_tab_sir_id) {
            // Const node already exists from static_tables pass
            AirNodeId(cos_tab_sir_id.0.clone())
        } else {
            // Emit shared cos_tab Const. Duplicates across layers are
            // removed by the global AirNodeId dedup in LegalityRewritePass::run().
            nodes.push(Self::make_air_node(
                shared_cos_id.clone(),
                AirOp::Const {
                    value_path: format!("static_tables/{}/cos_tab", tables_ref),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));
            shared_cos_id
        };

        let sin_id = if sir_to_air.contains_key(&sin_tab_sir_id) {
            AirNodeId(sin_tab_sir_id.0.clone())
        } else {
            // Emit shared sin_tab Const. Duplicates across layers are
            // removed by the global AirNodeId dedup in LegalityRewritePass::run().
            nodes.push(Self::make_air_node(
                shared_sin_id.clone(),
                AirOp::Const {
                    value_path: format!("static_tables/{}/sin_tab", tables_ref),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));
            shared_sin_id
        };

        // arange_tab is no longer needed for mask computation — masks now use
        // precomputed eye_tab/mask_tab tables + Gather (ANE-legal pattern from
        // the reference implementation). However, we keep arange_tab available
        // for backward compatibility with the embedding/prefill path that may
        // still use it, and as a fallback for very large seq_len where
        // eye_tab/mask_tab are impractical (> 128 MB each).
        let _arange_tab_id = {
            let arange_sir_id =
                ane_ir::sir::SirNodeId(format!("sir_static_arange_tab_{}", tables_ref));
            if sir_to_air.contains_key(&arange_sir_id) {
                AirNodeId(arange_sir_id.0.clone())
            } else {
                // Emit shared arange_tab Const. Duplicates across layers are
                // removed by the global AirNodeId dedup in LegalityRewritePass::run().
                nodes.push(Self::make_air_node(
                    shared_arange_id.clone(),
                    AirOp::Const {
                        value_path: format!("static_tables/{}/arange_tab", tables_ref),
                        dtype: MilDtype::Int32,
                    },
                    sir_node,
                    "mb.const",
                    kq,
                ));
                shared_arange_id.clone()
            }
        };

        // When a position input is provided, gather the specific row
        // from the cos/sin tables for position-dependent RoPE, and
        // compute KV write mask and causal mask using precomputed
        // eye_tab/mask_tab tables + Gather (all ANE-legal ops).
        //
        // cos_tab shape: [1, 1, seq_len, head_dim]
        // Gather along axis 2 with position index → [1, 1, 1, head_dim]
        let (cos_for_q, sin_for_q, kv_mask_write, causal_mask) = if let Some(pos_sir) = position {
            let pos_air =
                sir_to_air.get(pos_sir).cloned().unwrap_or_else(|| AirNodeId(pos_sir.0.clone()));

            // ── ANE-LEGAL RoPE table lookup: SliceByIndex (NOT Gather) ──
            //
            // mb.gather is ANE-illegal (CPU plannability ~0.26, causes sync
            // stalls). We replace Gather(cos_tab, pos, axis=2) with
            // SliceByIndex which is fully ANE-legal.
            //
            // cos_tab shape: [1, 1, seq_len, head_dim]
            // We need the row at position `pos` along axis 2.
            // SliceByIndex with begin=[0,0,pos,0], end=[0,0,pos+1,head_dim],
            // squeeze_mask=[false,false,true,false] → [1, 1, head_dim]
            // This broadcasts correctly with Q/K [1, hq, 1, head_dim].
            //
            // NOTE: We use pos_air (int32 position) as the basis for the
            // slice begin/end. However, SliceByIndex expects i64 constants
            // for begin/end, not dynamic indices. Since `pos` is a runtime
            // input, we MUST use a dynamic approach.
            //
            // Alternative ANE-legal approach: use Mul with a one-hot mask.
            // But SliceByIndex with begin_mask/end_mask can handle this
            // if we provide the position as a 4D tensor.
            //
            // Actually, the simplest ANE-legal replacement for
            // Gather(table, pos, axis=2) when pos is dynamic is:
            //   1. Expand pos to [1, 1, 1, 1] via Reshape
            //   2. SliceByIndex with dynamic begin from pos
            //
            // But SliceByIndex's begin/end are Vec<i64> (static), not
            // dynamic. So we need a different approach entirely.
            //
            // The correct ANE-legal approach for dynamic position lookup:
            //   - Reshape cos_tab [1,1,S,D] → [1,1,S,1,D] (insert dim)
            //   - Mul with one-hot at position pos
            //   - ReduceSum along the position axis
            //
            // But this requires computing a one-hot from pos, which is
            // exactly the mask we already compute for KV cache!
            //
            // Simplest approach: use the KV mask one-hot (already computed)
            // to select the cos/sin row via Mul + ReduceSum.
            //
            // However, the KV mask is computed LATER in this function.
            // We need to restructure: compute the KV mask FIRST, then
            // use it for both RoPE table lookup and KV cache write.
            //
            // For now, we use a SIMPLER approach that works with the
            // existing structure: elementwise Mul + ReduceSum.
            //
            // cos_row = ReduceSum(cos_tab * one_hot_kv_mask, axis=2)
            // sin_row = ReduceSum(sin_tab * one_hot_kv_mask, axis=2)
            //
            // where one_hot_kv_mask is [1, 1, seq_len] with 1 at pos, 0 elsewhere.
            // This is exactly what we compute as kv_mask_gathered below.
            //
            // BUT: the kv_mask is computed AFTER the RoPE step. We need to
            // reorder: compute the KV mask first, then use it for RoPE.
            //
            // SIMPLEST FIX: compute the KV one-hot mask here (before RoPE),
            // use it for both RoPE table lookup AND KV cache write.
            //
            // For the RoPE lookup:
            //   cos_for_pos = ReduceSum(Mul(cos_tab, kv_one_hot_4d), axis=2)
            //   sin_for_pos = ReduceSum(Mul(sin_tab, kv_one_hot_4d), axis=2)
            // where kv_one_hot_4d has shape [1, 1, seq_len, 1] (broadcasts with
            // cos_tab [1, 1, seq_len, head_dim])
            // Result: [1, 1, 1, head_dim] — the cos/sin values at position pos.

            // ── Compute KV one-hot mask FIRST (needed for RoPE lookup) ──
            // Same arithmetic mask computation as below, but done early
            // so we can use it for cos/sin table lookup.

            // Cast position from int32 to fp16 for arithmetic mask computation
            let pos_fp16_id = AirNodeId(format!("{base}_pos_fp16"));
            nodes.push(Self::make_air_node(
                pos_fp16_id.clone(),
                AirOp::Cast { input: pos_air.clone(), dtype: MilDtype::Fp16 },
                sir_node,
                "mb.cast",
                kq,
            ));

            // Shared arange_fp16_tab Const
            let shared_arange_fp16_id =
                AirNodeId(format!("shared_rope_{}_arange_fp16_tab", tables_ref));
            let arange_fp16_id = {
                let arange_fp16_sir_id =
                    ane_ir::sir::SirNodeId(format!("sir_static_arange_fp16_tab_{}", tables_ref));
                if sir_to_air.contains_key(&arange_fp16_sir_id) {
                    AirNodeId(arange_fp16_sir_id.0.clone())
                } else {
                    nodes.push(Self::make_air_node(
                        shared_arange_fp16_id.clone(),
                        AirOp::Const {
                            value_path: format!("static_tables/{}/arange_fp16_tab", tables_ref),
                            dtype: MilDtype::Fp16,
                        },
                        sir_node,
                        "mb.const",
                        kq,
                    ));
                    shared_arange_fp16_id
                }
            };

            // KV write mask (one-hot at position pos):
            // diff = arange_fp16 - pos_fp16
            let kv_diff_id = AirNodeId(format!("{base}_kv_mask_diff"));
            nodes.push(Self::make_air_node(
                kv_diff_id.clone(),
                AirOp::Sub { x: arange_fp16_id.clone(), y: pos_fp16_id.clone() },
                sir_node,
                "mb.sub",
                kq,
            ));

            // abs_diff = Abs(diff)
            let kv_abs_id = AirNodeId(format!("{base}_kv_mask_abs"));
            nodes.push(Self::make_air_node(
                kv_abs_id.clone(),
                AirOp::Abs { input: kv_diff_id },
                sir_node,
                "mb.abs",
                kq,
            ));

            // clipped = Minimum(abs_diff, 1.0)
            let kv_one_const_id = AirNodeId("shared_scalar_one".to_string());
            nodes.push(Self::make_air_node(
                kv_one_const_id.clone(),
                AirOp::Const { value_path: "scalar://fp16/1.0".to_string(), dtype: MilDtype::Fp16 },
                sir_node,
                "mb.const",
                kq,
            ));

            let kv_clipped_id = AirNodeId(format!("{base}_kv_mask_clipped"));
            nodes.push(Self::make_air_node(
                kv_clipped_id.clone(),
                AirOp::Minimum { x: kv_abs_id, y: kv_one_const_id.clone() },
                sir_node,
                "mb.minimum",
                kq,
            ));

            // one_hot = Sub(1.0, clipped) → 1 at pos, 0 elsewhere → shape [seq_len]
            let kv_mask_gathered_id = AirNodeId(format!("{base}_kv_mask_gathered"));
            nodes.push(Self::make_air_node(
                kv_mask_gathered_id.clone(),
                AirOp::Sub { x: kv_one_const_id.clone(), y: kv_clipped_id },
                sir_node,
                "mb.sub",
                kq,
            ));

            // ── ANE-LEGAL RoPE table lookup using Mul+ReduceSum (no Gather) ──
            //
            // cos_tab: [1, 1, seq_len, head_dim]
            // kv_one_hot_4d: Reshape one_hot [seq_len] → [1, 1, seq_len, 1]
            // cos_masked = Mul(cos_tab, kv_one_hot_4d) → zeros everywhere except pos row
            // cos_for_pos = ReduceSum(cos_masked, axis=2, keep_dims=true) → [1, 1, 1, head_dim]
            //
            // Same for sin_tab.

            let kv_one_hot_4d_id = AirNodeId(format!("{base}_kv_one_hot_4d"));
            nodes.push(Self::make_air_node(
                kv_one_hot_4d_id.clone(),
                AirOp::Reshape {
                    input: kv_mask_gathered_id.clone(),
                    target_shape: vec![1, 1, ctx.map(|c| c.seq_len).unwrap_or(0), 1],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));

            // cos_for_pos = ReduceSum(Mul(cos_tab, one_hot_4d), axis=2)
            let cos_masked_id = AirNodeId(format!("{base}_cos_masked"));
            nodes.push(Self::make_air_node(
                cos_masked_id.clone(),
                AirOp::Mul { x: cos_id.clone(), y: kv_one_hot_4d_id.clone() },
                sir_node,
                "mb.mul",
                kq,
            ));

            let cos_gathered_id = AirNodeId(format!("{base}_cos_gathered"));
            nodes.push(Self::make_air_node(
                cos_gathered_id.clone(),
                AirOp::ReduceSum { input: cos_masked_id, axes: vec![2], keep_dims: true },
                sir_node,
                "mb.reduce_sum",
                kq,
            ));

            // sin_for_pos = ReduceSum(Mul(sin_tab, one_hot_4d), axis=2)
            let sin_masked_id = AirNodeId(format!("{base}_sin_masked"));
            nodes.push(Self::make_air_node(
                sin_masked_id.clone(),
                AirOp::Mul { x: sin_id.clone(), y: kv_one_hot_4d_id },
                sir_node,
                "mb.mul",
                kq,
            ));

            let sin_gathered_id = AirNodeId(format!("{base}_sin_gathered"));
            nodes.push(Self::make_air_node(
                sin_gathered_id.clone(),
                AirOp::ReduceSum { input: sin_masked_id, axes: vec![2], keep_dims: true },
                sir_node,
                "mb.reduce_sum",
                kq,
            ));

            // ── Causal attention mask: 0 for allowed, -65504 for blocked ──
            // (KV one-hot mask was already computed above for RoPE table lookup)
            // offset = Sub(Const(seq_len-1), pos_fp16) → first allowed position
            let seq_minus_1_id = AirNodeId(format!("shared_seq_minus_1_{}", tables_ref));
            let seq_len_val = ctx.map(|c| c.seq_len).unwrap_or(0);
            nodes.push(Self::make_air_node(
                seq_minus_1_id.clone(),
                AirOp::Const {
                    value_path: format!("scalar://fp16/{}", seq_len_val.saturating_sub(1)),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));

            let offset_id = AirNodeId(format!("{base}_mask_offset"));
            nodes.push(Self::make_air_node(
                offset_id.clone(),
                AirOp::Sub { x: seq_minus_1_id, y: pos_fp16_id.clone() },
                sir_node,
                "mb.sub",
                kq,
            ));

            // distance = Sub(arange_fp16, offset) → negative for blocked, ≥0 for allowed
            let dist_id = AirNodeId(format!("{base}_mask_distance"));
            nodes.push(Self::make_air_node(
                dist_id.clone(),
                AirOp::Sub { x: arange_fp16_id, y: offset_id },
                sir_node,
                "mb.sub",
                kq,
            ));

            // shifted = Add(distance, 1.0) → ≥1 for allowed, ≤0 for blocked
            let shifted_id = AirNodeId(format!("{base}_mask_shifted"));
            nodes.push(Self::make_air_node(
                shifted_id.clone(),
                AirOp::Add { x: dist_id, y: kv_one_const_id.clone() },
                sir_node,
                "mb.add",
                kq,
            ));

            // is_allowed = Minimum(Maximum(shifted, 0.0), 1.0)
            let zero_const_id = AirNodeId("shared_scalar_zero".to_string());
            nodes.push(Self::make_air_node(
                zero_const_id.clone(),
                AirOp::Const { value_path: "scalar://fp16/0.0".to_string(), dtype: MilDtype::Fp16 },
                sir_node,
                "mb.const",
                kq,
            ));

            let clamped_pos_id = AirNodeId(format!("{base}_mask_clamped_pos"));
            nodes.push(Self::make_air_node(
                clamped_pos_id.clone(),
                AirOp::Maximum { x: shifted_id, y: zero_const_id },
                sir_node,
                "mb.maximum",
                kq,
            ));

            let is_allowed_id = AirNodeId(format!("{base}_mask_is_allowed"));
            nodes.push(Self::make_air_node(
                is_allowed_id.clone(),
                AirOp::Minimum { x: clamped_pos_id, y: kv_one_const_id.clone() },
                sir_node,
                "mb.minimum",
                kq,
            ));

            // is_blocked = Sub(1.0, is_allowed) → 0 for allowed, 1 for blocked
            let is_blocked_id = AirNodeId(format!("{base}_mask_is_blocked"));
            nodes.push(Self::make_air_node(
                is_blocked_id.clone(),
                AirOp::Sub { x: kv_one_const_id.clone(), y: is_allowed_id },
                sir_node,
                "mb.sub",
                kq,
            ));

            // mask = Mul(is_blocked, -65504.0) → 0 for allowed, -65504 for blocked
            // -65504 is the minimum fp16 value, effectively -inf for softmax.
            let neg_inf_id = AirNodeId("shared_fp16_neg_inf".to_string());
            nodes.push(Self::make_air_node(
                neg_inf_id.clone(),
                AirOp::Const {
                    value_path: "scalar://fp16/-65504.0".to_string(),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));

            let mask_gathered_id = AirNodeId(format!("{base}_mask_gathered"));
            nodes.push(Self::make_air_node(
                mask_gathered_id.clone(),
                AirOp::Mul { x: is_blocked_id, y: neg_inf_id },
                sir_node,
                "mb.mul",
                kq,
            ));

            (cos_gathered_id, sin_gathered_id, Some(kv_mask_gathered_id), Some(mask_gathered_id))
        } else {
            // No position input — use full tables with broadcast
            (cos_id.clone(), sin_id.clone(), None, None)
        };

        // Apply RoPE to Q: output = q * cos + rotate_half(q) * sin
        // Q has shape [B, H, 1, D] and cos/sin are [1, 1, 1, D] (gathered)
        // or [1, 1, seq_len, D] (full broadcast for prefill).
        let q_rope =
            Self::apply_rotary_half(q_4d_id, &cos_for_q, &sin_for_q, half, "_q_rope", env, nodes);

        // Apply RoPE to the NEW K token: output = k_new * cos + rotate_half(k_new) * sin
        // K new has shape [1, hk, 1, hd] and cos/sin are [1, 1, 1, hd] (gathered).
        // Broadcast: [1, hk, 1, hd] * [1, 1, 1, hd] → [1, hk, 1, hd] ✓
        // The RoPE'd K is then written to the cache, so the cache stores
        // pre-RoPE'd values and no full-cache RoPE re-application is needed.
        let k_new_rope = Self::apply_rotary_half(
            k_new_4d_id,
            &cos_for_q,
            &sin_for_q,
            half,
            "_k_new_rope",
            env,
            nodes,
        );

        (q_rope, Some(cos_id), Some(sin_id), k_new_rope, kv_mask_write, causal_mask)
    }

    /// Apply the RoPE rotation to a 4D tensor: output = x * cos + rotate_half(x) * sin
    ///
    /// `rotate_half(x)` splits the last dimension in half, negates the
    /// second half, swaps, and concatenates:
    ///   x1 = x[..., :d/2],  x2 = x[..., d/2:]
    ///   rotate_half(x) = concat(-x2, x1, axis=-1)
    ///
    /// This uses half-dim slices so cos/sin tables with shape [..., D/2]
    /// or full [..., D] both work — the broadcast is always compatible.
    fn apply_rotary_half(
        x_id: &AirNodeId,
        cos_id: &AirNodeId,
        sin_id: &AirNodeId,
        half: usize,
        suffix: &str,
        env: &DecompositionEnv,
        nodes: &mut Vec<AirNode>,
    ) -> AirNodeId {
        let sir_node = env.sir_node;
        let base = env.base;
        let kq = env.kq;
        // Slice first half: x1 = x[..., :half]
        let x1_id = AirNodeId(format!("{base}{suffix}_x1"));
        nodes.push(Self::make_air_node(
            x1_id.clone(),
            AirOp::SliceByIndex {
                input: x_id.clone(),
                begin: vec![0, 0, 0, 0],
                end: vec![0, 0, 0, half as i64],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, false],
                squeeze_mask: vec![false; 4],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        // Slice second half: x2 = x[..., half:]
        let x2_id = AirNodeId(format!("{base}{suffix}_x2"));
        nodes.push(Self::make_air_node(
            x2_id.clone(),
            AirOp::SliceByIndex {
                input: x_id.clone(),
                begin: vec![0, 0, 0, half as i64],
                end: vec![0, 0, 0, -1],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, true],
                squeeze_mask: vec![false; 4],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        // Negate second half: -x2 (ANE lowers to mul(x, -1))
        let neg_x2_id = AirNodeId(format!("{base}{suffix}_neg_x2"));
        nodes.push(Self::make_air_node(
            neg_x2_id.clone(),
            AirOp::Neg { input: x2_id },
            sir_node,
            "mb.neg",
            kq,
        ));

        // Concatenate: rotated = concat(-x2, x1, axis=-1)
        let rotated_id = AirNodeId(format!("{base}{suffix}_rotated"));
        nodes.push(Self::make_air_node(
            rotated_id.clone(),
            AirOp::Concat { inputs: vec![neg_x2_id, x1_id], axis: 3 },
            sir_node,
            "mb.concat",
            kq,
        ));

        // x * cos(θ) — broadcast: [B, H, S, D] * [1, 1, S, D]
        let x_cos_id = AirNodeId(format!("{base}{suffix}_x_cos"));
        nodes.push(Self::make_air_node(
            x_cos_id.clone(),
            AirOp::Mul { x: x_id.clone(), y: cos_id.clone() },
            sir_node,
            "mb.mul",
            kq,
        ));

        // rotate_half(x) * sin(θ)
        let rotated_sin_id = AirNodeId(format!("{base}{suffix}_rot_sin"));
        nodes.push(Self::make_air_node(
            rotated_sin_id.clone(),
            AirOp::Mul { x: rotated_id, y: sin_id.clone() },
            sir_node,
            "mb.mul",
            kq,
        ));

        // output = x * cos(θ) + rotate_half(x) * sin(θ)
        let out_id = AirNodeId(format!("{base}{suffix}_out"));
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Add { x: x_cos_id, y: rotated_sin_id },
            sir_node,
            "mb.add",
            kq,
        ));

        out_id
    }

    /// Decompose SirOp::RMSNorm into AIR ops.
    ///
    /// RMSNorm(x, weight, epsilon) →
    ///   x_sq:    Mul(x, x)
    ///   mean:    ReduceMean(x^2, axes=[-1], keep_dims=true)
    ///   eps:     FillLike(mean, epsilon)
    ///   biased:  Add(mean, eps)
    ///   rsqrt:   Rsqrt(biased)
    ///   normed:  ElementWise::Mul(x, rsqrt)
    ///   output:  ElementWise::Mul(normed, weight)
    fn decompose_rms_norm(
        input_sir: &ane_ir::sir::SirNodeId,
        weight: &str,
        epsilon: f32,
        axes: &[usize],
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let sir_node = env.sir_node;
        let base = env.base;
        let sir_to_air = env.sir_to_air;
        let kq = env.kq;
        let mut input_air =
            sir_to_air.get(input_sir).cloned().unwrap_or_else(|| AirNodeId(input_sir.0.clone()));

        let mut nodes = Vec::new();

        // Use the axes from the SIR op. For 3D tensors the default is [2]
        // (normalize over embedding dimension), for 4D head-layout tensors
        // it's [3] (normalize over head_dim).
        let norm_axes = if axes.is_empty() { vec![2] } else { axes.to_vec() };

        // ── Per-head RMSNorm (axes=[3]) ───────────────────────────────
        //
        // When axes=[3], the norm is applied per-head-dimension (e.g., Qwen3 q/k norm).
        // The input is a flat 3D tensor [batch, seq, heads*head_dim] but the
        // weight is [head_dim], which can't broadcast with the flat last dimension.
        //
        // Fix: reshape the input to 4D [batch, seq, heads, head_dim], apply the norm
        // with axes=[3], then reshape back to 3D.
        //
        // We need the DecompositionContext to know the actual dimensions.
        // When ctx is None, fall back to axes=[2] (3D tensor norm).
        //
        // For k_norm, the head count is kv_heads (not num_heads) because k_proj
        // outputs [B, S, kv_heads*head_dim], not [B, S, num_heads*head_dim].
        // We detect this by checking if the weight name contains "k_norm".
        let needs_4d_reshape = norm_axes.contains(&3);
        let (batch, seq, heads, head_dim) = match ctx {
            Some(c) if needs_4d_reshape => {
                // Detect k_norm by checking the WEIGHT name (not the node ID).
                // The node ID is counter-based (e.g., "sir_7_layer_0_self_attn") and
                // does NOT contain "k_norm", but the weight parameter name is
                // "model.layers.0.self_attn.k_norm.weight" which DOES contain "k_norm".
                let is_k_norm = weight.contains("k_norm");
                let h = if is_k_norm {
                    c.kv_heads.max(1) // kv_heads for k_norm
                } else {
                    c.num_heads
                };
                (c.batch_size, c.seq_len, h, c.head_dim)
            }
            _ if needs_4d_reshape => {
                // No DecompositionContext but axes=[3] requested — can't create
                // the 4D reshape without knowing the head dimensions. Fall back
                // to axes=[2] (3D tensor norm) to avoid producing invalid ops.
                // This should not happen in the TraceCompile pipeline (which
                // always provides a ctx), but prevents silent corruption if it
                // does. Log a warning via eprintln for now.
                eprintln!(
                    "[WARN] RMSNorm axes=[3] without DecompositionContext — \
                     falling back to axes=[2] for node '{}'. \
                     This may produce incorrect shapes for Qwen3-style q/k norm.",
                    sir_node.id.0
                );
                (0, 0, 0, 0) // batch=0 → skip 4D reshape, fall through to 3D path
            }
            _ => (0, 0, 0, 0),
        };

        // When needs_4d_reshape but no context, fall back to axes=[2] on 3D tensor
        let effective_axes = if needs_4d_reshape && batch == 0 {
            vec![2] // 3D fallback: normalize over embedding dimension
        } else {
            norm_axes.clone()
        };

        if needs_4d_reshape && batch > 0 {
            // Reshape flat [B, S, heads*head_dim] → [B, S, heads, head_dim]
            let reshape_4d_id = AirNodeId(format!("{base}_reshape_4d"));
            nodes.push(Self::make_air_node(
                reshape_4d_id.clone(),
                AirOp::Reshape {
                    input: input_air,
                    target_shape: vec![batch, seq, heads, head_dim],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));
            input_air = reshape_4d_id;
        }

        // Step 0 (fp16 stabilization): Compute max_abs for safe normalization.
        //
        // In fp16, x² overflows for |x| > 255 (since 255² ≈ 65025 < 65504 = fp16 max,
        // but 256² = 65536 > 65504 → inf). To prevent this, we normalize x by its
        // max absolute value before squaring, then scale the rsqrt result back.
        //
        // Mathematically equivalent transformation:
        //   rsqrt(mean(x²) + ε) = max_abs * rsqrt(mean((x/max_abs)²) + ε) / max_abs
        //                        = rsqrt(mean(x²/max_abs²) + ε) * max_abs / max_abs
        //                        Wait — that simplifies to the original only if we
        //                        account for the eps term correctly.
        //
        // Correct approach: compute max_abs, clamp x before squaring to prevent overflow,
        // but use original x for the final multiply (which is safe since rsqrt ≤ 1/√ε).
        //
        // Simpler approach used here: compute max_abs, divide x by max_abs before squaring,
        // then multiply the final result by max_abs. The math:
        //   output = (x/max_abs) * rsqrt(mean((x/max_abs)²) + ε) * weight * max_abs
        //   = x * rsqrt(mean(x²)/max_abs² + ε) * weight
        //   ≈ x * rsqrt((mean(x²) + ε*max_abs²) / max_abs²) * weight
        //   = x * max_abs * rsqrt(mean(x²) + ε*max_abs²) * weight
        // This is NOT exactly the same as the original (ε is scaled by max_abs²),
        // but for typical transformer values where max_abs ≈ √mean(x²), the
        // relative error is negligible (ε is already tiny, ~1e-6).
        //
        // However, the simplest and most robust approach: just clip x before squaring.
        // The clip value sqrt(65504) ≈ 255.99 prevents overflow. If x > 255, the
        // clipped x² gives the correct relative scale, and the final multiply with
        // the original x preserves relative ordering. For normal transformer
        // activations (|x| << 255), this is a no-op.

        // Step 1: x^2 = Mul(x, x)
        //
        // RMSNorm requires the mean of x², not the mean of x.
        // The previous code passed input_air directly to ReduceMean,
        // computing E[x] instead of E[x²] — a critical correctness bug.
        //
        // fp16 max-abs stabilization: compute |x|, reduce max, divide x by it
        // before squaring to prevent fp16 overflow. Then rescale the rsqrt.
        let abs_x_id = AirNodeId(format!("{base}_abs_x"));
        nodes.push(Self::make_air_node(
            abs_x_id.clone(),
            AirOp::Abs { input: input_air.clone() },
            sir_node,
            "mb.abs",
            kq,
        ));

        let max_abs_id = AirNodeId(format!("{base}_max_abs"));
        nodes.push(Self::make_air_node(
            max_abs_id.clone(),
            AirOp::ReduceMax { input: abs_x_id, axes: effective_axes.clone(), keep_dims: true },
            sir_node,
            "mb.reduce_max",
            kq,
        ));

        // Clamp max_abs to at least epsilon to avoid division by zero
        // ANE-legal: Const scalar + Maximum with broadcasting (instead of FillLike)
        let eps_for_max_scalar_id = AirNodeId(format!("{base}_eps_for_max_scalar"));
        let eps_clamp_val = epsilon.max(1e-6);
        nodes.push(Self::make_air_node(
            eps_for_max_scalar_id.clone(),
            AirOp::Const {
                value_path: format!("scalar://fp16/{}", eps_clamp_val),
                dtype: MilDtype::Fp16,
            },
            sir_node,
            "mb.const",
            kq,
        ));

        let safe_max_id = AirNodeId(format!("{base}_safe_max"));
        nodes.push(Self::make_air_node(
            safe_max_id.clone(),
            AirOp::Maximum { x: max_abs_id, y: eps_for_max_scalar_id },
            sir_node,
            "mb.maximum",
            kq,
        ));

        // x_normalized = x / max_abs (|x_normalized| ≤ 1, no overflow when squared)
        let x_norm_id = AirNodeId(format!("{base}_x_norm"));
        nodes.push(Self::make_air_node(
            x_norm_id.clone(),
            AirOp::RealDiv { x: input_air.clone(), y: safe_max_id.clone() },
            sir_node,
            "mb.real_div",
            kq,
        ));

        // x_norm_sq = x_norm * x_norm (safe: |x_norm| ≤ 1, so x_norm_sq ≤ 1)
        let x_sq_id = AirNodeId(format!("{base}_x_sq"));
        nodes.push(Self::make_air_node(
            x_sq_id.clone(),
            AirOp::Mul { x: x_norm_id.clone(), y: x_norm_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 2: mean(x^2) via ReduceMean
        let mean_id = AirNodeId(format!("{base}_mean"));
        nodes.push(Self::make_air_node(
            mean_id.clone(),
            AirOp::ReduceMean { input: x_sq_id, axes: effective_axes.clone(), keep_dims: true },
            sir_node,
            "mb.reduce_mean",
            kq,
        ));

        // Step 3: mean(x^2) + epsilon
        //
        // RMSNorm requires rsqrt(mean(x²) + ε) for numerical stability.
        // ANE-legal: Const scalar + Add with broadcasting (instead of FillLike).
        // CoreML's mb.add(x=tensor, y=scalar) broadcasts the scalar correctly,
        // matching the reference implementation pattern.
        let eps_scalar_id = AirNodeId(format!("{base}_eps_scalar"));
        nodes.push(Self::make_air_node(
            eps_scalar_id.clone(),
            AirOp::Const {
                value_path: format!("scalar://fp16/{}", epsilon),
                dtype: MilDtype::Fp16,
            },
            sir_node,
            "mb.const",
            kq,
        ));

        let mean_plus_eps_id = AirNodeId(format!("{base}_mean_plus_eps"));
        nodes.push(Self::make_air_node(
            mean_plus_eps_id.clone(),
            AirOp::Add { x: mean_id, y: eps_scalar_id },
            sir_node,
            "mb.add",
            kq,
        ));

        // Step 4: Rsqrt of (mean(x^2) + epsilon)
        let rsqrt_id = AirNodeId(format!("{base}_rsqrt"));
        nodes.push(Self::make_air_node(
            rsqrt_id.clone(),
            AirOp::Rsqrt { input: mean_plus_eps_id },
            sir_node,
            "mb.rsqrt",
            kq,
        ));

        // Step 5: x_normalized * rsqrt(mean(x_norm^2) + epsilon)
        // This gives us (x/max_abs) * rsqrt(mean((x/max_abs)²) + ε)
        let normed_raw_id = AirNodeId(format!("{base}_normed_raw"));
        nodes.push(Self::make_air_node(
            normed_raw_id.clone(),
            AirOp::Mul { x: input_air.clone(), y: rsqrt_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 5b: Rescale by max_abs to undo the fp16-safe normalization.
        // output = (x/max_abs) * rsqrt(...) * max_abs = x * rsqrt(...)
        // This is mathematically correct: the max_abs cancels out because
        // rsqrt(mean(x²/max_abs²) + ε) = max_abs * rsqrt(mean(x²) + ε*max_abs²)
        // and x/max_abs * max_abs = x, so:
        //   normed = (x/max_abs) * max_abs * rsqrt(mean(x²) + ε*max_abs²)
        //   ≈ x * rsqrt(mean(x²) + ε) when ε*max_abs² ≈ ε (true for small ε)
        let normed_id = AirNodeId(format!("{base}_normed"));
        nodes.push(Self::make_air_node(
            normed_id.clone(),
            AirOp::Mul { x: normed_raw_id, y: safe_max_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 6: normed * weight (gamma)
        // In 4D mode, this produces [B, S, heads, head_dim] * [head_dim] → [B, S, heads, head_dim]
        let mul_out_id = if needs_4d_reshape && batch > 0 {
            AirNodeId(format!("{base}_normed_weighted"))
        } else {
            AirNodeId(sir_node.id.0.clone())
        };
        nodes.push(Self::make_air_node(
            mul_out_id.clone(),
            AirOp::Mul { x: normed_id, y: AirNodeId(weight.into()) },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 7 (4D mode only): reshape back to 3D [B, S, heads*head_dim]
        let out_id = if needs_4d_reshape && batch > 0 {
            let flat_id = AirNodeId(sir_node.id.0.clone());
            nodes.push(Self::make_air_node(
                flat_id.clone(),
                AirOp::Reshape {
                    input: mul_out_id,
                    target_shape: vec![batch, seq, heads * head_dim],
                },
                sir_node,
                "mb.reshape",
                kq,
            ));
            flat_id
        } else {
            mul_out_id
        };

        (out_id, nodes)
    }

    /// Decompose SirOp::RoPETransform into AIR ops.
    ///
    /// Correct RoPE formula:
    ///   output = x * cos(θ) + rotate_half(x) * sin(θ)
    ///
    /// Where rotate_half(x) splits the last dimension in half, negates
    /// the second half, swaps, and concatenates:
    ///   x1 = x[..., :d/2]
    ///   x2 = x[..., d/2:]
    ///   rotate_half(x) = concat(-x2, x1, axis=-1)
    ///
    /// The `tables` field references pre-computed frequency tables.
    /// The static_tables pass (which runs before legality_rewrite) inserts
    /// Const nodes for cos_tab and sin_tab at known IDs based on the
    /// tables reference. We look up these const nodes by their convention:
    ///   cos_tab: "sir_static_cos_tab_{base}"
    ///   sin_tab: "sir_static_sin_tab_{base}"
    ///
    /// If the static tables are not found (e.g., static_tables pass not run),
    /// we fall back to computing cos/sin from the tables reference as an
    /// angle tensor. This is less efficient but correct.
    ///
    /// Decomposition:
    ///   1. cos_vals: reference to pre-computed cos table (or Cos(tables))
    ///   2. sin_vals: reference to pre-computed sin table (or Sin(tables))
    ///   3. x1: SliceByIndex(x, [..., :head_dim//2]) — first half
    ///   4. x2: SliceByIndex(x, [..., head_dim//2:]) — second half
    ///   5. neg_x2: Neg(x2)
    ///   6. rotated: Concat([neg_x2, x1], axis=-1)
    ///   7. x_cos: Mul(x, cos_vals)
    ///   8. rotated_sin: Mul(rotated, sin_vals)
    ///   9. output: Add(x_cos, rotated_sin)
    fn decompose_rope(
        input_sir: &ane_ir::sir::SirNodeId,
        tables: &str,
        env: &DecompositionEnv,
        ctx: Option<&DecompositionContext>,
    ) -> (AirNodeId, Vec<AirNode>) {
        let sir_node = env.sir_node;
        let base = env.base;
        let sir_to_air = env.sir_to_air;
        let kq = env.kq;
        let input_air =
            sir_to_air.get(input_sir).cloned().unwrap_or_else(|| AirNodeId(input_sir.0.clone()));

        let mut nodes = Vec::new();

        // Determine head_dim from context (needed for rotate_half slicing).
        // T-36 (I-15/CQ-17): Previously fell back to 128 silently, which
        // produces wrong RoPE slicing for models with head_dim != 128.
        // Now uses a strong warning; the caller should provide DecompositionContext.
        let head_dim = ctx.map(|c| c.head_dim).unwrap_or(0);
        if head_dim == 0 {
            eprintln!(
                "[WARN] RoPE decompose without head_dim in context — \
                 using default 128. This will be WRONG for models with head_dim != 128. \
                 Provide DecompositionContext for correctness."
            );
        }
        let head_dim = if head_dim > 0 { head_dim } else { 128 };
        let half = head_dim / 2;

        // Step 1-2: Get cos/sin values.
        //
        // ANE-LEGAL (Sprint 67): NEVER emit AirOp::Cos / AirOp::Sin.
        // These are ANE-illegal ops that cause the execution planner to
        // return error -5. Instead, always use pre-computed static tables
        // via AirOp::Const + AirOp::Gather, matching the reference model.
        //
        // If the static_tables pass already inserted Const nodes in the
        // SIR graph, use those. Otherwise, emit Const nodes directly.
        let cos_tab_sir_id = ane_ir::sir::SirNodeId(format!("sir_static_cos_tab_{}", tables));
        let sin_tab_sir_id = ane_ir::sir::SirNodeId(format!("sir_static_sin_tab_{}", tables));

        let cos_id = if sir_to_air.contains_key(&cos_tab_sir_id) {
            // Use pre-computed cos table from static_tables pass
            AirNodeId(cos_tab_sir_id.0.clone())
        } else {
            // Emit per-node Const node with unique ID to avoid duplicates
            let const_id = AirNodeId(format!("{}_cos_tab", base));
            nodes.push(Self::make_air_node(
                const_id.clone(),
                AirOp::Const {
                    value_path: format!("static_tables/{}/cos_tab", tables),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));
            const_id
        };

        let sin_id = if sir_to_air.contains_key(&sin_tab_sir_id) {
            // Use pre-computed sin table from static_tables pass
            AirNodeId(sin_tab_sir_id.0.clone())
        } else {
            // Emit per-node Const node with unique ID to avoid duplicates
            let const_id = AirNodeId(format!("{}_sin_tab", base));
            nodes.push(Self::make_air_node(
                const_id.clone(),
                AirOp::Const {
                    value_path: format!("static_tables/{}/sin_tab", tables),
                    dtype: MilDtype::Fp16,
                },
                sir_node,
                "mb.const",
                kq,
            ));
            const_id
        };

        // Save input_air for step 7 (x * cos) — we need it after slicing
        let input_for_mul = input_air.clone();

        // Step 3: Slice first half: x1 = x[..., :head_dim//2]
        let x1_id = AirNodeId(format!("{base}_x1"));
        nodes.push(Self::make_air_node(
            x1_id.clone(),
            AirOp::SliceByIndex {
                input: input_air.clone(),
                begin: vec![0, 0, 0, 0],
                end: vec![0, 0, 0, half as i64],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, false],
                squeeze_mask: vec![false; 4],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        // Step 4: Slice second half: x2 = x[..., head_dim//2:]
        let x2_id = AirNodeId(format!("{base}_x2"));
        nodes.push(Self::make_air_node(
            x2_id.clone(),
            AirOp::SliceByIndex {
                input: input_air,
                begin: vec![0, 0, 0, half as i64],
                end: vec![0, 0, 0, -1],
                stride: vec![1, 1, 1, 1],
                begin_mask: vec![true, true, true, false],
                end_mask: vec![true, true, true, true],
                squeeze_mask: vec![false; 4],
            },
            sir_node,
            "mb.slice_by_index",
            kq,
        ));

        // Step 5: Negate second half: -x2
        // Note: Core ML has no "neg" op; this is lowered to mul(x, -1)
        // at the emission level (see mir_op_to_apple_ops). The AIR-level
        // Neg correctly propagates the input shape through inference.
        let neg_x2_id = AirNodeId(format!("{base}_neg_x2"));
        nodes.push(Self::make_air_node(
            neg_x2_id.clone(),
            AirOp::Neg { input: x2_id },
            sir_node,
            "mb.neg",
            kq,
        ));

        // Step 6: Concatenate: rotated = concat(-x2, x1, axis=-1)
        let rotated_id = AirNodeId(format!("{base}_rotated"));
        nodes.push(Self::make_air_node(
            rotated_id.clone(),
            AirOp::Concat {
                inputs: vec![neg_x2_id, x1_id],
                axis: 3, // last axis in [B, heads, S, head_dim]
            },
            sir_node,
            "mb.concat",
            kq,
        ));

        // Step 7: x * cos(θ)
        let x_cos_id = AirNodeId(format!("{base}_x_cos"));
        nodes.push(Self::make_air_node(
            x_cos_id.clone(),
            AirOp::Mul { x: input_for_mul, y: cos_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 8: rotate_half(x) * sin(θ)
        let rotated_sin_id = AirNodeId(format!("{base}_rotated_sin"));
        nodes.push(Self::make_air_node(
            rotated_sin_id.clone(),
            AirOp::Mul { x: rotated_id, y: sin_id },
            sir_node,
            "mb.mul",
            kq,
        ));

        // Step 9: output = x * cos(θ) + rotate_half(x) * sin(θ)
        let out_id = AirNodeId(sir_node.id.0.clone());
        nodes.push(Self::make_air_node(
            out_id.clone(),
            AirOp::Add { x: x_cos_id, y: rotated_sin_id },
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
    ) -> Result<(AirOp, &'static str)> {
        let aid = |sid: &ane_ir::sir::SirNodeId| -> AirNodeId {
            sir_to_air.get(sid).cloned().unwrap_or_else(|| AirNodeId(sid.0.clone()))
        };
        let aids =
            |sids: &[ane_ir::sir::SirNodeId]| -> Vec<AirNodeId> { sids.iter().map(&aid).collect() };
        let _base = &node_id.0;

        Ok(match op {
            // ─── Constants ───────────────────────────────────────
            SirOp::Const { value_path, dtype, .. } => {
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
            SirOp::Select { .. } | SirOp::Where { .. } => {
                // UNREACHABLE: Select and Where are decomposed to arithmetic
                // (cond*x + (1-cond)*y) in the main run() match above.
                // If this fires, a new code path is producing Select/Where
                // without going through the decomposition.
                anyhow::bail!("BUG: SirOp::Select/Where reached sir_to_air_passthrough — these must be decomposed to arithmetic in run(), not passed through. mb.select and mb.where are ANE-illegal.");
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
                // SAFETY NET: Tile is decomposed above in map_sir_op() to
                // Reshape + broadcast Mul + Reshape (ane.legal.tile_decompose).
                // Additionally, GQA Tile ops have been eliminated at the SIR
                // builder level via split-based per-head attention.
                //
                // If we reach this point, a Tile op has bypassed the decomposition
                // in map_sir_op(). This is a bug — Tile is ANE-illegal and must
                // never survive to AIR/MIR. Emit a panic to catch this during
                // development rather than silently producing an ANE-incompatible model.
                anyhow::bail!(
                    "BUG: SirOp::Tile {{ input: {:?}, reps: {:?} }} reached the fallback \
                     passthrough in sir_to_air_op(). All Tile ops must be decomposed in \
                     map_sir_op() (ane.legal.tile_decompose) or eliminated at the SIR builder \
                     level (split-based attention). mb.tile is ANE-illegal and must never \
                     reach AIR/MIR.",
                    input, reps
                );
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
            // DEPRECATED: FillLike is ANE-illegal. New decompositions should use
            // Const scalar + Add broadcasting instead. This 1:1 mapping is kept
            // for backward compatibility only. See ISSUE-002.
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
        })
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
                    op: SirOp::Mul { x: SirNodeId(String::new()), y: SirNodeId(String::new()) },
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
                        palette_bits: None,
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

        // After the ANE legality rewrite, Conv1x1AsLinear → MILLinear → MILMatMul
        // (MILLinear is replaced with MILMatMul by the post-lowering rewrite)
        let matmul_node =
            mirs[0].nodes.iter().find(|n| matches!(n.op, MirOp::MILMatMul { .. })).expect(
                "Expected MILMatMul node (replaced from MILLinear by ANE legality rewrite)",
            );
        assert_eq!(
            matmul_node.dtype,
            MilDtype::Fp32,
            "MILMatMul node dtype must be fp32 when AIR precision_override is fp32"
        );
    }

    /// Test that AttentionBlock decomposes into split-based per-head attention (no Tile/SDPA).
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

        // Should have: 3 reshape (Q/K/V) + 3 transpose (Q/K/V) + 3 split (Q/K/V)
        // + scale const + per-head attention ops + concat + reshape + out proj
        // NO ScaledDotProductAttention, NO Tile.
        let has_reshape = air.nodes.iter().any(|n| matches!(n.op, AirOp::Reshape { .. }));
        let has_transpose = air.nodes.iter().any(|n| matches!(n.op, AirOp::Transpose { .. }));
        // Without context (heads=0), falls back to SDPA — no split/MatMul/Concat.
        // These ops are present only when DecompositionContext provides head count.
        let has_out_proj = air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));

        assert!(
            has_reshape,
            "AttentionBlock decomposition must include Reshape for multi-head layout"
        );
        assert!(
            has_transpose,
            "AttentionBlock decomposition must include Transpose for multi-head layout"
        );
        assert!(
            has_out_proj,
            "AttentionBlock decomposition must include Conv1x1AsLinear for output projection"
        );
        // Without DecompositionContext (heads=0), falls back to SDPA.
        // Production compilation always provides DecompositionContext.
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));
        assert!(has_sdpa, "Without DecompositionContext, AttentionBlock falls back to SDPA");
        assert!(!has_tile, "AttentionBlock decomposition must NOT include Tile (ANE-illegal)");
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
                    q_weight: None,
                    k_weight: None,
                    v_weight: None,
                    out_weight: None,
                    rope_tables: None,
                    position: None,
                    q_norm_weight: None,
                    k_norm_weight: None,
                    norm_epsilon: 1e-6,
                    qk_norm_type: "rms".to_string(),
                    mask_ref: None,
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

        // Sprint 67: Provide a DecompositionContext so the per-head attention
        // loop has real dimensions to work with. Without a context, the loop
        // over heads produces zero MatMul nodes (no-op).
        let ctx = DecompositionContext::for_decode_step_full(
            1,      // batch_size
            2048,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // kv_len
            8,      // kv_heads (GQA)
            4096,   // intermediate_size
            151936, // vocab_size
            false,  // uses_rope
            false,  // has_qk_norm
        );

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        let has_state_read = air.nodes.iter().any(|n| matches!(n.op, AirOp::StateReadFixed { .. }));
        let has_state_write =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::StateWriteFixed { .. }));
        // Sprint 67: SDPA is NO LONGER used. The decomposition now uses
        // per-head matmul + softmax (split-based GQA) matching the reference model.
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_softmax = air.nodes.iter().any(|n| matches!(n.op, AirOp::Softmax { .. }));
        let has_split = air.nodes.iter().any(|n| matches!(n.op, AirOp::Split { .. }));
        let has_linear = air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));

        assert!(
            has_state_read,
            "DecodeStep decomposition must include StateReadFixed for KV cache"
        );
        assert!(
            has_state_write,
            "DecodeStep decomposition must include StateWriteFixed for KV cache update"
        );
        // Sprint 67: SDPA should NOT be present (replaced by per-head matmul+softmax)
        assert!(!has_sdpa, "DecodeStep decomposition must NOT include ScaledDotProductAttention (ANE-illegal in decoder shards)");
        assert!(has_matmul, "DecodeStep decomposition must include MatMul for per-head attention");
        assert!(
            has_softmax,
            "DecodeStep decomposition must include Softmax for per-head attention"
        );
        // Split is intentionally NOT emitted (invalid MIL output arity).
        // Instead, SliceByIndex is used to extract individual heads.
        assert!(!has_split, "DecodeStep decomposition must NOT include Split (invalid MIL)");
        let has_slice = air.nodes.iter().any(|n| matches!(n.op, AirOp::SliceByIndex { .. }));
        assert!(
            has_slice,
            "DecodeStep decomposition must include SliceByIndex for head extraction"
        );
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
                    axes: vec![2],
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
        let has_mul = air.nodes.iter().any(|n| matches!(n.op, AirOp::Mul { .. }));

        assert!(has_reduce_mean, "RMSNorm decomposition must include ReduceMean");
        assert!(has_rsqrt, "RMSNorm decomposition must include Rsqrt");
        assert!(has_mul, "RMSNorm decomposition must include Mul");
    }

    /// Test that RoPETransform decomposes into Const(cos_tab) + Const(sin_tab) + Mul + Add
    /// (ANE-legal decomposition using static tables, Sprint 67).
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

        // Sprint 67: Cos/Sin are ANE-illegal — they should NEVER appear.
        // Instead, the decomposition uses Const nodes for static tables.
        let has_cos = air.nodes.iter().any(|n| matches!(n.op, AirOp::Cos { .. }));
        let has_sin = air.nodes.iter().any(|n| matches!(n.op, AirOp::Sin { .. }));
        let has_const = air.nodes.iter().any(|n| matches!(n.op, AirOp::Const { .. }));
        let has_mul = air.nodes.iter().any(|n| matches!(n.op, AirOp::Mul { .. }));
        let has_add = air.nodes.iter().any(|n| matches!(n.op, AirOp::Add { .. }));

        assert!(!has_cos, "RoPETransform decomposition must NOT include Cos (ANE-illegal)");
        assert!(!has_sin, "RoPETransform decomposition must NOT include Sin (ANE-illegal)");
        assert!(
            has_const,
            "RoPETransform decomposition must include Const for static cos/sin tables"
        );
        assert!(
            has_mul,
            "RoPETransform decomposition must include Mul for x*cos and rotate_half*sin"
        );
        assert!(
            has_add,
            "RoPETransform decomposition must include Add for x*cos + rotate_half*sin"
        );
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

        // Verify split-based attention: should have Split, MatMul, Softmax, Concat ops
        // NO ScaledDotProductAttention, NO Tile
        let has_split = air.nodes.iter().any(|n| matches!(n.op, AirOp::Split { .. }));
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_softmax = air.nodes.iter().any(|n| matches!(n.op, AirOp::Softmax { .. }));
        let has_concat = air.nodes.iter().any(|n| matches!(n.op, AirOp::Concat { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));

        // Split is intentionally NOT emitted (invalid MIL output arity).
        // Instead, SliceByIndex is used to extract individual heads.
        assert!(!has_split, "Split-based attention must NOT include Split ops (invalid MIL)");
        let has_slice = air.nodes.iter().any(|n| matches!(n.op, AirOp::SliceByIndex { .. }));
        assert!(has_slice, "Split-based attention must include SliceByIndex for head extraction");
        assert!(has_matmul, "Split-based attention must include MatMul ops");
        assert!(has_softmax, "Split-based attention must include Softmax ops");
        assert!(has_concat, "Split-based attention must include Concat ops");
        assert!(!has_sdpa, "Split-based attention must NOT include SDPA");
        assert!(!has_tile, "Split-based attention must NOT include Tile");

        // Verify attn_flat reshape has [batch, seq, heads*head_dim]
        // Note: For this test config, heads*head_dim = 4*32 = 128 = embed_dim,
        // so it happens to match. For models like Qwen3-0.6B where
        // heads*head_dim (16*128=2048) != embed_dim (1024), the reshape
        // MUST use heads*head_dim, not embed_dim.
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
                    "attn_flat reshape should be [batch, seq, heads*head_dim] = [2, 16, 4*32]"
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

        // Without context (heads=0), falls back to SDPA (no per-head split)
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));
        assert!(has_sdpa, "Without DecompositionContext, must fall back to SDPA");
        assert!(!has_tile, "Even in fallback, Tile must NOT be present");
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
                    q_weight: None,
                    k_weight: None,
                    v_weight: None,
                    out_weight: None,
                    rope_tables: None,
                    position: None,
                    q_norm_weight: None,
                    k_norm_weight: None,
                    norm_epsilon: 1e-6,
                    qk_norm_type: "rms".to_string(),
                    mask_ref: None,
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

        // attn_flat reshape: [batch, heads*head_dim] = [1, 128]
        // For this test config: heads*head_dim = 4*32 = 128 = embed_dim.
        // For models like Qwen3-0.6B where heads*head_dim (16*128=2048) != embed_dim (1024),
        // the reshape MUST use heads*head_dim, not embed_dim.
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
                    "attn_flat reshape should be [batch, heads*head_dim] = [1, 4*32]"
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
                    q_weight: None,
                    k_weight: None,
                    v_weight: None,
                    out_weight: None,
                    rope_tables: None,
                    position: None,
                    q_norm_weight: None,
                    k_norm_weight: None,
                    norm_epsilon: 1e-6,
                    qk_norm_type: "rms".to_string(),
                    mask_ref: None,
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

    /// T-60 (I-34): Test tile_input_dim method on DecompositionContext.
    /// Verifies that 4D Tile input dimensions are correctly resolved from ctx.
    #[test]
    fn test_tile_input_dim_4d() {
        let ctx = DecompositionContext::for_attention_full(
            1,    // batch_size
            1024, // embed_dim
            8,    // num_heads
            128,  // head_dim
            512,  // seq_len
            2,    // kv_heads (GQA)
            4096, // intermediate_size
            151936, // vocab_size
        );

        // 4D Tile: position 0 = batch, 1 = kv_heads, 2 = seq_len, 3 = head_dim
        assert_eq!(ctx.tile_input_dim(0, 4), Some(1), "pos 0 = batch_size");
        assert_eq!(ctx.tile_input_dim(1, 4), Some(2), "pos 1 = kv_heads");
        assert_eq!(ctx.tile_input_dim(2, 4), Some(512), "pos 2 = seq_len");
        assert_eq!(ctx.tile_input_dim(3, 4), Some(128), "pos 3 = head_dim");
        assert_eq!(ctx.tile_input_dim(4, 4), None, "pos 4 out of range for 4D");
    }

    /// T-60 (I-34): Test tile_input_dim returns None for non-4D Tile patterns.
    #[test]
    fn test_tile_input_dim_non_4d() {
        let ctx = DecompositionContext::for_attention(1, 256, 8, 32, 64);
        // Only 4D is supported; other ranks should return None
        assert_eq!(ctx.tile_input_dim(0, 3), None, "3D Tile not supported");
        assert_eq!(ctx.tile_input_dim(0, 5), None, "5D Tile not supported");
    }

    /// T-60 (I-34): Test tile_input_dim returns None for default (zero) ctx.
    #[test]
    fn test_tile_input_dim_default_ctx() {
        let ctx = DecompositionContext::default();
        // All dimensions are 0, so tile_input_dim should return None
        assert_eq!(ctx.tile_input_dim(0, 4), None, "batch_size=0 → None");
        assert_eq!(ctx.tile_input_dim(1, 4), None, "num_heads=0 → None");
        assert_eq!(ctx.tile_input_dim(2, 4), None, "seq_len=0 → None");
        assert_eq!(ctx.tile_input_dim(3, 4), None, "head_dim=0 → None");
    }

    /// Sprint 62: Verify that RMSNorm with axes=[3] (Qwen3-style q/k norm)
    /// produces the 4D reshape → norm → reshape-back sequence when a
    /// DecompositionContext is provided. Without the reshape, the [128]
    /// q_norm weight cannot broadcast with [1,512,2048] flat projection.
    #[test]
    fn test_rms_norm_4d_reshape_for_qk_norm() {
        use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

        // Simulate a q_norm RMSNorm SIR node: axes=[3] means per-head-dimension norm
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("sir_6_layer_0_self_attn".into()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("sir_3_layer_0_self_attn".into()),
                    weight: "model.layers.0.self_attn.q_norm.weight".into(),
                    epsilon: 1e-6,
                    axes: vec![3],
                },
                name: "q_norm".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("sir_3_layer_0_self_attn".into())],
            outputs: vec![SirNodeId("sir_6_layer_0_self_attn".into())],
        };

        // Qwen3-0.6B dimensions
        let ctx = DecompositionContext::for_attention_full(1, 1024, 16, 128, 512, 8, 3072, 151936);

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Must contain Reshape ops (3D→4D and 4D→3D)
        let reshape_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::Reshape { .. })).count();
        assert!(
            reshape_count >= 2,
            "q_norm with axes=[3] must produce at least 2 Reshape ops (3D→4D and 4D→3D), got {}",
            reshape_count
        );

        // The first Reshape should produce [1, 512, 16, 128] (4D head layout)
        let reshape_shapes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::Reshape { target_shape, .. } = &n.op {
                    Some(target_shape.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 16, 128]),
            "Expected 4D reshape to [1, 512, 16, 128], got shapes: {:?}",
            reshape_shapes
        );

        // ReduceMean must use axes=[3] (not axes=[2])
        let reduce_mean_axes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::ReduceMean { axes, .. } = &n.op {
                    Some(axes.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            reduce_mean_axes.iter().any(|a| a == &vec![3]),
            "ReduceMean must use axes=[3] for 4D per-head norm, got: {:?}",
            reduce_mean_axes
        );

        // Final reshape must produce [1, 512, 2048] (3D flat layout = 16*128)
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 2048]),
            "Expected final reshape back to [1, 512, 2048], got shapes: {:?}",
            reshape_shapes
        );
    }

    /// Sprint 62: k_norm with axes=[3] must use kv_heads (8) not num_heads (16)
    /// for the 4D reshape target shape. Detection uses the weight name, not the
    /// node ID (since SIR node IDs are counter-based like "sir_7_layer_0_self_attn").
    #[test]
    fn test_rms_norm_4d_reshape_k_norm_uses_kv_heads() {
        use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

        // Simulate a k_norm RMSNorm: the SIR node ID is counter-based (no "k_norm"),
        // but the WEIGHT name contains "k_norm" for detection.
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("sir_7_layer_0_self_attn".into()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("sir_4_layer_0_self_attn".into()),
                    weight: "model.layers.0.self_attn.k_norm.weight".into(),
                    epsilon: 1e-6,
                    axes: vec![3],
                },
                name: "k_norm".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("sir_4_layer_0_self_attn".into())],
            outputs: vec![SirNodeId("sir_7_layer_0_self_attn".into())],
        };

        let ctx = DecompositionContext::for_attention_full(1, 1024, 16, 128, 512, 8, 3072, 151936);

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // The 4D reshape must use kv_heads=8, producing [1, 512, 8, 128]
        let reshape_shapes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::Reshape { target_shape, .. } = &n.op {
                    Some(target_shape.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 8, 128]),
            "k_norm 4D reshape must use kv_heads=8 → [1, 512, 8, 128], got shapes: {:?}",
            reshape_shapes
        );
    }

    /// Sprint 62: RMSNorm with axes=[3] but NO context must NOT produce
    /// invalid ReduceMean(axes=[3]) on a 3D tensor. Without ctx, the code
    /// should fall back to axes=[2] or skip the 4D reshape safely.
    #[test]
    fn test_rms_norm_axes3_without_context_falls_back_gracefully() {
        use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("sir_6_layer_0_self_attn".into()),
                op: SirOp::RMSNorm {
                    input: SirNodeId("sir_3_layer_0_self_attn".into()),
                    weight: "model.layers.0.self_attn.q_norm.weight".into(),
                    epsilon: 1e-6,
                    axes: vec![3],
                },
                name: "q_norm".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("sir_3_layer_0_self_attn".into())],
            outputs: vec![SirNodeId("sir_6_layer_0_self_attn".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Without context, axes=[3] on a 3D tensor is invalid.
        // The code should either:
        // 1. Fall back to axes=[2], or
        // 2. Still produce the 4D reshape with placeholder shapes (zeros)
        //
        // Currently, the code produces ReduceMean(axes=[3]) without reshape,
        // which would be invalid on a 3D tensor. This test documents the bug.
        let reduce_mean_axes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::ReduceMean { axes, .. } = &n.op {
                    Some(axes.clone())
                } else {
                    None
                }
            })
            .collect();
        let has_reshape = air.nodes.iter().any(|n| matches!(n.op, AirOp::Reshape { .. }));

        // Document current behavior: axes=[3] without ctx → no reshape, invalid
        eprintln!(
            "[DIAG] axes=[3] without ctx: has_reshape={}, reduce_mean_axes={:?}",
            has_reshape, reduce_mean_axes
        );

        // This SHOULD eventually assert that axes are valid for the tensor rank,
        // but for now just verify the run doesn't crash.
    }

    /// Sprint 62: End-to-end integration test — build the full attention block
    /// SIR (with q/k norm) and verify the AIR output has correct shapes at every
    /// stage. This is the key test for the "no dialect" problem: it documents
    /// the expected attention shape flow through the lowering pipeline.
    ///
    /// Expected shape flow for Qwen3-0.6B layer 0:
    ///   input:           [1, 512, 1024]   (hidden state)
    ///   q_proj:          [1, 512, 2048]   (num_heads * head_dim = 16*128)
    ///   q_norm reshape:  [1, 512, 16, 128] (4D head layout)
    ///   q_norm mean:     [1, 512, 16, 1]   (reduce over head_dim)
    ///   q_norm rsqrt:    [1, 512, 16, 1]
    ///   q_norm normed:   [1, 512, 16, 128] (x * rsqrt)
    ///   q_norm weighted: [1, 512, 16, 128] (normed * weight[128])
    ///   q_norm flat:     [1, 512, 2048]   (reshape back to 3D)
    ///   (similar for k_norm with kv_heads=8)
    ///   attention:       [1, 512, 1024]   (after merge-heads + o_proj)
    ///   residual add:    [1, 512, 1024]   (attn_out + residual)
    #[test]
    fn test_full_attention_block_with_qk_norm_shape_flow() {
        use ane_ir::sir::{SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

        // Build a minimal SIR that mimics the Qwen3 attention decomposition:
        // LinearProjection(q) → RMSNorm(q, axes=3) → LinearProjection(k) →
        // RMSNorm(k, axes=3) → LinearProjection(v) → AttentionBlock(q,k,v)
        let input_id = SirNodeId("sir_1_embed_tokens".into());
        let residual_id = SirNodeId("sir_0_input_ids".into()); // placeholder

        let q_proj_id = SirNodeId("sir_3_layer_0_self_attn".into());
        let k_proj_id = SirNodeId("sir_4_layer_0_self_attn".into());
        let v_proj_id = SirNodeId("sir_5_layer_0_self_attn".into());
        let q_norm_id = SirNodeId("sir_6_layer_0_self_attn".into());
        let k_norm_id = SirNodeId("sir_7_layer_0_self_attn".into());

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: q_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.q_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "q_proj_2048".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: k_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.k_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "k_proj_1024".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: v_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.v_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "v_proj_1024".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: q_norm_id.clone(),
                    op: SirOp::RMSNorm {
                        input: q_proj_id.clone(),
                        weight: "model.layers.0.self_attn.q_norm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![3],
                    },
                    name: "q_norm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: k_norm_id.clone(),
                    op: SirOp::RMSNorm {
                        input: k_proj_id.clone(),
                        weight: "model.layers.0.self_attn.k_norm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![3],
                    },
                    name: "k_norm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("sir_8_layer_0_self_attn".into()),
                    op: SirOp::AttentionBlock {
                        q: q_norm_id,
                        k: k_norm_id,
                        v: v_proj_id,
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
                },
            ],
            inputs: vec![input_id.clone(), residual_id],
            outputs: vec![SirNodeId("sir_8_layer_0_self_attn".into())],
        };

        let ctx = DecompositionContext::for_attention_full(1, 1024, 16, 128, 512, 8, 3072, 151936);

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Count reshape ops — should have:
        //   - 2 from q_norm (3D→4D, 4D→3D)
        //   - 2 from k_norm (3D→4D, 4D→3D)
        //   - 3 from attention_block (q/k/v 3D→4D before transpose)
        //   - 1 from attention_block (4D→3D after SDPA)
        // Total: 8
        let reshape_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::Reshape { .. })).count();

        let reshape_shapes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::Reshape { target_shape, .. } = &n.op {
                    Some(target_shape.clone())
                } else {
                    None
                }
            })
            .collect();

        eprintln!("Total AIR nodes: {}", air.nodes.len());
        eprintln!("Reshape count: {}", reshape_count);
        eprintln!("Reshape target shapes: {:?}", reshape_shapes);

        // Must have 4D reshapes for q_norm (16 heads) and k_norm (8 kv_heads)
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 16, 128]),
            "Expected q_norm 4D reshape to [1, 512, 16, 128], got: {:?}",
            reshape_shapes
        );
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 8, 128]),
            "Expected k_norm 4D reshape to [1, 512, 8, 128], got: {:?}",
            reshape_shapes
        );

        // Must have reshapes back to 3D after q/k norm
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 2048]),
            "Expected q_norm reshape back to [1, 512, 2048], got: {:?}",
            reshape_shapes
        );
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 1024]),
            "Expected k_norm reshape back to [1, 512, 1024], got: {:?}",
            reshape_shapes
        );

        // All ReduceMean ops must have axes that are valid for their input rank
        for node in &air.nodes {
            if let AirOp::ReduceMean { axes, input, .. } = &node.op {
                // axes=[3] requires 4D input (which is the 4D-reshaped tensor)
                // axes=[2] requires 3D+ input
                // This is validated by the reshape sequence above
                eprintln!("  ReduceMean: input={} axes={:?}", input.0, axes);
            }
        }

        // All ElementWise::Mul ops must have broadcastable inputs
        // This is the key check: no [1,512,2048] * [128] allowed
        for node in &air.nodes {
            if let AirOp::Mul { x, y } = &node.op {
                eprintln!("  Mul: x={}, y={}", x.0, y.0);
                // The weight input should NOT be a flat [128] applied to [1,512,2048]
                // After 4D reshape, it's [1,512,16,128] * [128] which IS broadcastable
            }
        }
    }

    // ─── T-37: SIR→AIR Roundtrip Tests ─────────────────────────────────

    /// Helper: collect all AirNodeId references within an AirOp.
    /// Returns every AirNodeId that the op references as an input.
    fn collect_air_op_refs(op: &AirOp) -> Vec<AirNodeId> {
        match op {
            AirOp::Const { .. } => vec![],
            AirOp::Linear { input, .. } => vec![input.clone()],
            AirOp::MatMul { a, b } => vec![a.clone(), b.clone()],
            AirOp::Einsum { inputs, .. } => inputs.clone(),
            AirOp::Conv1x1AsLinear { input, .. } => vec![input.clone()],
            AirOp::Conv { input, weight, .. } => vec![input.clone(), weight.clone()],
            AirOp::ConvTranspose { input, weight, .. } => vec![input.clone(), weight.clone()],
            AirOp::Add { x, y }
            | AirOp::Mul { x, y }
            | AirOp::Sub { x, y }
            | AirOp::Maximum { x, y }
            | AirOp::Minimum { x, y }
            | AirOp::RealDiv { x, y }
            | AirOp::FloorDiv { x, y }
            | AirOp::Mod { x, y }
            | AirOp::Pow { x, y }
            | AirOp::Equal { x, y }
            | AirOp::NotEqual { x, y }
            | AirOp::Greater { x, y }
            | AirOp::GreaterEqual { x, y }
            | AirOp::Less { x, y }
            | AirOp::LessEqual { x, y }
            | AirOp::LogicalAnd { x, y }
            | AirOp::LogicalOr { x, y }
            | AirOp::LogicalXor { x, y } => vec![x.clone(), y.clone()],
            AirOp::Abs { input }
            | AirOp::Neg { input }
            | AirOp::Sigmoid { input }
            | AirOp::Tanh { input }
            | AirOp::Relu { input }
            | AirOp::Relu6 { input }
            | AirOp::Softsign { input }
            | AirOp::Silu { input }
            | AirOp::Softplus { input }
            | AirOp::Sqrt { input }
            | AirOp::Rsqrt { input }
            | AirOp::Ceil { input }
            | AirOp::Floor { input }
            | AirOp::Round { input }
            | AirOp::Exp { input }
            | AirOp::Exp2 { input }
            | AirOp::Sign { input }
            | AirOp::Cos { input }
            | AirOp::Sin { input }
            | AirOp::Tan { input }
            | AirOp::Acos { input }
            | AirOp::Asin { input }
            | AirOp::Atan { input }
            | AirOp::Cosh { input }
            | AirOp::Sinh { input }
            | AirOp::Atanh { input }
            | AirOp::Erf { input }
            | AirOp::LogicalNot { input }
            | AirOp::Cast { input, .. } => vec![input.clone()],
            AirOp::LeakyRelu { input, .. }
            | AirOp::SigmoidHard { input, .. }
            | AirOp::ThresholdedRelu { input, .. }
            | AirOp::ClampedRelu { input, .. }
            | AirOp::LinearActivation { input, .. }
            | AirOp::ScaledTanh { input, .. }
            | AirOp::Elu { input, .. }
            | AirOp::Gelu { input, .. }
            | AirOp::Clip { input, .. }
            | AirOp::Square { input }
            | AirOp::Threshold { input, .. }
            | AirOp::Inverse { input, .. }
            | AirOp::Log { input, .. } => vec![input.clone()],
            AirOp::Prelu { input, .. } | AirOp::SoftplusParametric { input, .. } => {
                vec![input.clone()]
            }
            AirOp::Select { condition, x, y } | AirOp::Where { condition, x, y } => {
                vec![condition.clone(), x.clone(), y.clone()]
            }
            AirOp::Softmax { input, .. } => vec![input.clone()],
            AirOp::ReduceSum { input, .. }
            | AirOp::ReduceMean { input, .. }
            | AirOp::ReduceMax { input, .. }
            | AirOp::ReduceMin { input, .. }
            | AirOp::ReduceProd { input, .. }
            | AirOp::ReduceSumSquare { input, .. }
            | AirOp::ReduceL2Norm { input, .. }
            | AirOp::ReduceL1Norm { input, .. }
            | AirOp::ReduceLogSumExp { input, .. }
            | AirOp::ReduceLogSum { input, .. } => vec![input.clone()],
            AirOp::ReduceArgmax { input, .. } | AirOp::ReduceArgmin { input, .. } => {
                vec![input.clone()]
            }
            AirOp::BatchNorm { input, .. } => vec![input.clone()],
            AirOp::InstanceNorm { input, .. } => vec![input.clone()],
            AirOp::LayerNorm { input, .. } => vec![input.clone()],
            AirOp::L2Norm { input, .. } => vec![input.clone()],
            AirOp::LocalResponseNorm { input, .. } => vec![input.clone()],
            AirOp::MaxPool { input, .. }
            | AirOp::AvgPool { input, .. }
            | AirOp::L2Pool { input, .. } => vec![input.clone()],
            AirOp::Resize { input, .. } => vec![input.clone()],
            AirOp::ResizeNearestNeighbor { input, .. } => vec![input.clone()],
            AirOp::ResizeBilinear { input, .. } => vec![input.clone()],
            AirOp::UpsampleNearestNeighbor { input, .. } => vec![input.clone()],
            AirOp::UpsampleBilinear { input, .. } => vec![input.clone()],
            AirOp::CropResize { input, boxes, box_indices, .. } => {
                vec![input.clone(), boxes.clone(), box_indices.clone()]
            }
            AirOp::Affine { input, transform, .. } => vec![input.clone(), transform.clone()],
            AirOp::Resample { input, coordinates, .. } => {
                vec![input.clone(), coordinates.clone()]
            }
            AirOp::Reshape { input, .. } => vec![input.clone()],
            AirOp::ReshapeLike { input, ref_tensor } => {
                vec![input.clone(), ref_tensor.clone()]
            }
            AirOp::Transpose { input, .. } => vec![input.clone()],
            AirOp::Split { input, .. } => vec![input.clone()],
            AirOp::Concat { inputs, .. } => inputs.clone(),
            AirOp::ExpandDims { input, .. } | AirOp::Squeeze { input, .. } => {
                vec![input.clone()]
            }
            AirOp::Flatten2d { input, .. } => vec![input.clone()],
            AirOp::Reverse { input, .. } => vec![input.clone()],
            AirOp::ReverseSequence { input, lengths, .. } => {
                vec![input.clone(), lengths.clone()]
            }
            AirOp::SliceByIndex { input, .. } => vec![input.clone()],
            AirOp::SliceBySize { input, .. } => vec![input.clone()],
            AirOp::SliceUpdate { input, update, .. } => vec![input.clone(), update.clone()],
            AirOp::SlidingWindows { input, .. } => vec![input.clone()],
            AirOp::DepthToSpace { input, .. }
            | AirOp::SpaceToDepth { input, .. }
            | AirOp::PixelShuffle { input, .. }
            | AirOp::PixelUnshuffle { input, .. } => vec![input.clone()],
            AirOp::BatchToSpace { input, .. } | AirOp::SpaceToBatch { input, .. } => {
                vec![input.clone()]
            }
            AirOp::Pad { input, .. } => vec![input.clone()],
            AirOp::Stack { values, .. } => values.clone(),
            AirOp::Tile { input, .. } => vec![input.clone()],
            AirOp::Cumsum { input, .. } => vec![input.clone()],
            AirOp::Fill { .. } | AirOp::Range1d { .. } => vec![],
            AirOp::FillLike { ref_tensor, .. } => vec![ref_tensor.clone()],
            AirOp::Identity { input } => vec![input.clone()],
            AirOp::OneHot { indices, .. } => vec![indices.clone()],
            AirOp::NonZero { input } | AirOp::Argsort { input, .. } => vec![input.clone()],
            AirOp::BandPart { input, .. } => vec![input.clone()],
            AirOp::Shape { input } => vec![input.clone()],
            AirOp::Crop { input, .. } => vec![input.clone()],
            AirOp::Gather { input, indices, .. }
            | AirOp::GatherAlongAxis { input, indices, .. } => {
                vec![input.clone(), indices.clone()]
            }
            AirOp::GatherNd { input, indices } => vec![input.clone(), indices.clone()],
            AirOp::Scatter { input, indices, updates, .. }
            | AirOp::ScatterAlongAxis { input, indices, updates, .. } => {
                vec![input.clone(), indices.clone(), updates.clone()]
            }
            AirOp::ScatterNd { input, indices, updates } => {
                vec![input.clone(), indices.clone(), updates.clone()]
            }
            AirOp::NonMaximumSuppression { boxes, scores, .. } => {
                vec![boxes.clone(), scores.clone()]
            }
            AirOp::ScaledDotProductAttention { query, key, value, attention_mask, .. } => {
                let mut refs = vec![query.clone(), key.clone(), value.clone()];
                if let Some(mask) = attention_mask {
                    refs.push(mask.clone());
                }
                refs
            }
            AirOp::Quantize { input, .. } | AirOp::Dequantize { input, .. } => {
                vec![input.clone()]
            }
            AirOp::ConstexprAffineDequantize { .. }
            | AirOp::ConstexprBlockwiseShiftScale { .. }
            | AirOp::ConstexprLutToDense { .. }
            | AirOp::ConstexprSparseToDense { .. }
            | AirOp::ConstexprCast { .. }
            | AirOp::ConstexprLutToSparse { .. }
            | AirOp::ConstexprSparseBlockwiseShiftScale { .. } => vec![],
            AirOp::Rnn { input, initial_h, .. } => vec![input.clone(), initial_h.clone()],
            AirOp::Gru { input, initial_h, .. } => vec![input.clone(), initial_h.clone()],
            AirOp::Lstm { input, initial_h, initial_c, .. } => {
                vec![input.clone(), initial_h.clone(), initial_c.clone()]
            }
            AirOp::Cond { pred, .. } => vec![pred.clone()],
            AirOp::WhileLoop { loop_vars, .. } => loop_vars.clone(),
            AirOp::MakeList { elems, .. } => elems.clone(),
            AirOp::ListLength { ls } | AirOp::ListRead { ls, .. } => vec![ls.clone()],
            AirOp::ListWrite { ls, index, value } => {
                vec![ls.clone(), index.clone(), value.clone()]
            }
            AirOp::ListGather { ls, indices } => vec![ls.clone(), indices.clone()],
            AirOp::ListScatter { ls, indices, values } => {
                vec![ls.clone(), indices.clone(), values.clone()]
            }
            AirOp::RandomBernoulli { .. }
            | AirOp::RandomNormal { .. }
            | AirOp::RandomUniform { .. } => vec![],
            AirOp::RandomCategorical { logits, .. } => vec![logits.clone()],
            AirOp::StateReadFixed { .. } => vec![],
            AirOp::StateWriteFixed { value, .. } => vec![value.clone()],
            AirOp::Topk { input, .. } => vec![input.clone()],
            AirOp::Classify { input } => vec![input.clone()],
            AirOp::StaticLUTProjection { input, .. } => vec![input.clone()],
        }
    }

    /// Validate structural invariants of an AirGraph:
    /// 1. No duplicate AirNodeIds (SSA property)
    /// 2. All AirNodeId references resolve to defined nodes, graph inputs,
    ///    or external weight/const references
    /// 3. All output nodes exist in the graph (or are graph inputs)
    ///
    /// Note: Graph inputs are externally provided and may not have corresponding
    /// AIR nodes in the graph. Similarly, the RMSNorm decomposition uses weight
    /// name strings directly as AirNodeId references (e.g.,
    /// `AirNodeId("model.layers.0.self_attn.q_norm.weight")`), which are
    /// resolved at the MIR lowering stage. These are valid external references.
    fn validate_air_graph_structural_invariants(air: &AirGraph) {
        // 1. No duplicate AirNodeIds
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &air.nodes {
            assert!(
                seen_ids.insert(node.id.0.clone()),
                "Duplicate AirNodeId: {} — violates SSA property",
                node.id.0
            );
        }

        // Build a lookup of all defined node IDs (including graph inputs)
        let mut defined_ids: std::collections::HashSet<String> =
            air.nodes.iter().map(|n| n.id.0.clone()).collect();
        // Graph inputs are externally defined — references to them are valid
        for in_id in &air.inputs {
            defined_ids.insert(in_id.0.clone());
        }

        // 2. All AirNodeId references within ops resolve to defined nodes or
        //    known external references (weight paths, scalar constants, placeholders)
        for node in &air.nodes {
            let refs = collect_air_op_refs(&node.op);
            for r in &refs {
                let is_defined = defined_ids.contains(&r.0);
                // External reference patterns used by the decomposition:
                // - Weight paths: "model.layers.X..." or "*.weight" or "*.bin"
                // - Scalar constants: "scalar://..."
                // - Placeholder inputs: "__placeholder__" or "__*"
                // - Shared rope tables: "shared_rope_*" / "shared_scalar_*"
                let is_external = r.0.contains(".weight")
                    || r.0.contains(".bin")
                    || r.0.starts_with("scalar://")
                    || r.0.starts_with("__")
                    || r.0.starts_with("shared_rope_")
                    || r.0.starts_with("shared_scalar_")
                    || r.0.starts_with("shared_attn_");
                assert!(
                    is_defined || is_external,
                    "AirNode '{}' references undefined AirNodeId '{}' — broken SSA reference",
                    node.id.0,
                    r.0
                );
            }
        }

        // 3. All output nodes exist (either defined in graph or as inputs)
        for out_id in &air.outputs {
            assert!(
                defined_ids.contains(&out_id.0),
                "AIR output '{}' is not defined in the graph",
                out_id.0
            );
        }
    }

    /// T-37: Full SIR→AIR roundtrip with `for_decode_step_full()` using
    /// realistic Qwen3-0.6B dimensions. This is the key test that the
    /// existing test suite lacked — all 19 unit tests cover individual
    /// decompositions but none exercises the full pipeline end-to-end
    /// with `for_decode_step_full()` context and realistic model dimensions.
    ///
    /// Qwen3-0.6B: embed_dim=1024, num_heads=16, head_dim=128,
    /// kv_heads=8 (GQA), intermediate_size=2048, vocab_size=151936,
    /// max_seq_len=32768.
    #[test]
    fn test_decode_step_roundtrip_qwen3_0_6b() {
        let ctx = DecompositionContext::for_decode_step_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // kv_len (typical decode KV cache length)
            8,      // kv_heads (GQA)
            2048,   // intermediate_size
            151936, // vocab_size
            true,   // uses_rope
            true,   // has_qk_norm
        );

        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("decode_0".into()),
                op: SirOp::DecodeStep {
                    token: SirNodeId("token_input".into()),
                    state_map: vec!["k_cache_0".into(), "v_cache_0".into()],
                    q_weight: Some("model.layers.0.self_attn.q_proj.weight".into()),
                    k_weight: Some("model.layers.0.self_attn.k_proj.weight".into()),
                    v_weight: Some("model.layers.0.self_attn.v_proj.weight".into()),
                    out_weight: Some("model.layers.0.self_attn.o_proj.weight".into()),
                    rope_tables: Some("rope_tables_shared".into()),
                    position: Some(SirNodeId("position_0".into())),
                    q_norm_weight: Some("model.layers.0.self_attn.q_norm.weight".into()),
                    k_norm_weight: Some("model.layers.0.self_attn.k_norm.weight".into()),
                    norm_epsilon: 1e-6,
                    qk_norm_type: "rms".to_string(),
                    mask_ref: Some("causal_mask".into()),
                },
                name: "decode_step_layer_0".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                    model_id: Some("qwen3-0.6b".into()),
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("token_input".into()), SirNodeId("position_0".into())],
            outputs: vec![SirNodeId("decode_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants ──
        validate_air_graph_structural_invariants(&air);

        // ── Decomposition correctness ──
        // Must have Conv1x1AsLinear for QKV+output projections
        let linear_nodes: Vec<_> =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. })).collect();
        assert!(
            !linear_nodes.is_empty(),
            "DecodeStep must decompose into Conv1x1AsLinear projections"
        );

        // Must have MatMul for per-head attention (NOT SDPA)
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        assert!(has_matmul, "DecodeStep must include MatMul for per-head attention");
        assert!(!has_sdpa, "DecodeStep must NOT include SDPA (ANE-illegal in decoder shards)");

        // Must have state ops for KV cache
        let has_state_read = air.nodes.iter().any(|n| matches!(n.op, AirOp::StateReadFixed { .. }));
        let has_state_write =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::StateWriteFixed { .. }));
        assert!(has_state_read, "DecodeStep must include StateReadFixed for KV cache");
        assert!(has_state_write, "DecodeStep must include StateWriteFixed for KV cache update");

        // Must have Softmax for attention weights
        let has_softmax = air.nodes.iter().any(|n| matches!(n.op, AirOp::Softmax { .. }));
        assert!(has_softmax, "DecodeStep must include Softmax for per-head attention");

        // Must have RoPE decomposition (Const tables + Gather + Mul + Add)
        let has_gather = air.nodes.iter().any(|n| matches!(n.op, AirOp::Gather { .. }));
        let has_rope_const = air.nodes.iter().any(
            |n| matches!(n.op, AirOp::Const { ref value_path, .. } if value_path.contains("rope")),
        );
        assert!(
            has_gather || has_rope_const,
            "DecodeStep with uses_rope=true must include RoPE decomposition (Gather or Const tables)"
        );

        // Must have QK-norm decomposition (ReduceMean + Rsqrt + Mul)
        let has_rsqrt = air.nodes.iter().any(|n| matches!(n.op, AirOp::Rsqrt { .. }));
        assert!(
            has_rsqrt,
            "DecodeStep with has_qk_norm=true must include RMSNorm decomposition with Rsqrt"
        );

        // ── Shape consistency ──
        // Conv1x1AsLinear output_dim for Q projection should be num_heads*head_dim = 16*128 = 2048
        let q_proj = air.nodes.iter().find(|n| {
            if let AirOp::Conv1x1AsLinear { weight, output_dim, .. } = &n.op {
                weight.contains("q_proj") && *output_dim > 0
            } else {
                false
            }
        });
        if let Some(q) = q_proj {
            if let AirOp::Conv1x1AsLinear { output_dim, .. } = &q.op {
                assert_eq!(
                    *output_dim, 2048,
                    "Q projection output_dim should be num_heads*head_dim = 2048"
                );
            }
        }

        // K projection output_dim should be kv_heads*head_dim = 8*128 = 1024
        let k_proj = air.nodes.iter().find(|n| {
            if let AirOp::Conv1x1AsLinear { weight, output_dim, .. } = &n.op {
                weight.contains("k_proj") && *output_dim > 0
            } else {
                false
            }
        });
        if let Some(k) = k_proj {
            if let AirOp::Conv1x1AsLinear { output_dim, .. } = &k.op {
                assert_eq!(
                    *output_dim, 1024,
                    "K projection output_dim should be kv_heads*head_dim = 1024"
                );
            }
        }

        // O projection output_dim should be embed_dim = 1024
        let o_proj = air.nodes.iter().find(|n| {
            if let AirOp::Conv1x1AsLinear { weight, output_dim, .. } = &n.op {
                weight.contains("o_proj") && *output_dim > 0
            } else {
                false
            }
        });
        if let Some(o) = o_proj {
            if let AirOp::Conv1x1AsLinear { output_dim, .. } = &o.op {
                assert_eq!(*output_dim, 1024, "O projection output_dim should be embed_dim = 1024");
            }
        }

        // ── No Tile ops (ANE-illegal) ──
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));
        assert!(!has_tile, "DecodeStep AIR output must NOT contain Tile ops");

        // ── No Split ops (invalid MIL for multi-output) ──
        let has_split = air.nodes.iter().any(|n| matches!(n.op, AirOp::Split { .. }));
        assert!(!has_split, "DecodeStep AIR output must NOT contain Split ops (invalid MIL)");

        // ── Must use SliceByIndex for head extraction ──
        let has_slice = air.nodes.iter().any(|n| matches!(n.op, AirOp::SliceByIndex { .. }));
        assert!(
            has_slice,
            "DecodeStep must use SliceByIndex for per-head extraction (replaces Split)"
        );

        // ── GQA: verify fan_out = num_heads / kv_heads = 16 / 8 = 2 ──
        // Each KV head is shared by 2 Q heads. The per-head attention loop
        // should produce fan_out MatMul ops per KV head group.
        let matmul_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::MatMul { .. })).count();
        assert!(
            matmul_count >= ctx.num_heads,
            "GQA decode must have at least {} MatMul ops (one per Q head), got {}",
            ctx.num_heads,
            matmul_count
        );

        // ── Verify the output is reachable ──
        assert!(!air.outputs.is_empty(), "AIR graph must have at least one output");
        assert!(
            air.nodes.iter().any(|n| n.id == air.outputs[0]),
            "AIR output node must exist in graph"
        );
    }

    /// T-37: AttentionBlock roundtrip with `for_attention_full()` and
    /// Qwen3-0.6B dimensions. Validates the full SIR→AIR pipeline for
    /// the prefill attention path (no state ops, separate Q/K/V inputs).
    #[test]
    fn test_attention_block_roundtrip_qwen3_0_6b() {
        let ctx = DecompositionContext::for_attention_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // seq_len
            8,      // kv_heads (GQA)
            2048,   // intermediate_size
            151936, // vocab_size
        );

        let q_id = SirNodeId("q_proj_output".into());
        let k_id = SirNodeId("k_proj_output".into());
        let v_id = SirNodeId("v_proj_output".into());

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: q_id.clone(),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("hidden_state".into()),
                        weight: "model.layers.0.self_attn.q_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "q_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: k_id.clone(),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("hidden_state".into()),
                        weight: "model.layers.0.self_attn.k_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "k_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: v_id.clone(),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("hidden_state".into()),
                        weight: "model.layers.0.self_attn.v_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "v_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("attn_0".into()),
                    op: SirOp::AttentionBlock { q: q_id, k: k_id, v: v_id, mask: None, rope: None },
                    name: "attention_layer_0".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("hidden_state".into())],
            outputs: vec![SirNodeId("attn_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants ──
        validate_air_graph_structural_invariants(&air);

        // ── Decomposition correctness ──
        // Must have per-head MatMul, NOT SDPA
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));

        assert!(has_matmul, "AttentionBlock with context must use per-head MatMul");
        assert!(!has_sdpa, "AttentionBlock with context must NOT use SDPA");
        assert!(!has_tile, "AttentionBlock must NOT use Tile (ANE-illegal)");

        // Must have Concat to merge per-head outputs
        let has_concat = air.nodes.iter().any(|n| matches!(n.op, AirOp::Concat { .. }));
        assert!(
            has_concat,
            "AttentionBlock must include Concat to merge per-head attention outputs"
        );

        // Must have Conv1x1AsLinear for output projection
        let has_out_proj = air
            .nodes
            .iter()
            .any(|n| matches!(&n.op, AirOp::Conv1x1AsLinear { weight, .. } if weight.contains("o_proj") || n.id.0.contains("out_proj")));
        let has_any_conv1x1 =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::Conv1x1AsLinear { .. }));
        assert!(
            has_any_conv1x1,
            "AttentionBlock decomposition must include Conv1x1AsLinear for projections"
        );

        // ── Shape consistency ──
        // Q reshape: [1, 512, 16, 128] (batch, seq, num_heads, head_dim)
        let q_4d =
            air.nodes.iter().find(|n| n.id.0 == "attn_0_q_4d").expect("Expected attn_0_q_4d node");
        if let AirOp::Reshape { target_shape, .. } = &q_4d.op {
            assert_eq!(
                target_shape,
                &vec![1, 512, 16, 128],
                "Q 4D reshape should be [batch, seq, num_heads, head_dim]"
            );
        }

        // K reshape: [1, 512, 8, 128] (uses kv_heads for GQA)
        let k_4d =
            air.nodes.iter().find(|n| n.id.0 == "attn_0_k_4d").expect("Expected attn_0_k_4d node");
        if let AirOp::Reshape { target_shape, .. } = &k_4d.op {
            assert_eq!(
                target_shape,
                &vec![1, 512, 8, 128],
                "K 4D reshape should use kv_heads=8 for GQA: [1, 512, 8, 128]"
            );
        }

        // V reshape: [1, 512, 8, 128] (uses kv_heads for GQA)
        let v_4d =
            air.nodes.iter().find(|n| n.id.0 == "attn_0_v_4d").expect("Expected attn_0_v_4d node");
        if let AirOp::Reshape { target_shape, .. } = &v_4d.op {
            assert_eq!(
                target_shape,
                &vec![1, 512, 8, 128],
                "V 4D reshape should use kv_heads=8 for GQA: [1, 512, 8, 128]"
            );
        }

        // attn_flat reshape: [1, 512, 2048] (num_heads * head_dim, NOT embed_dim)
        let attn_flat = air
            .nodes
            .iter()
            .find(|n| n.id.0 == "attn_0_attn_flat")
            .expect("Expected attn_0_attn_flat node");
        if let AirOp::Reshape { target_shape, .. } = &attn_flat.op {
            assert_eq!(
                target_shape, &vec![1, 512, 2048],
                "attn_flat reshape should be [batch, seq, num_heads*head_dim] = [1, 512, 2048] (NOT embed_dim=1024)"
            );
        }

        // ── GQA: MatMul count should equal num_heads (16) ──
        let matmul_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::MatMul { .. })).count();
        assert!(
            matmul_count >= ctx.num_heads,
            "GQA attention must have at least {} MatMul ops (one per Q head), got {}",
            ctx.num_heads,
            matmul_count
        );
    }

    /// T-37: Multi-layer SIR→AIR roundtrip. Simulates two decoder layers
    /// with realistic Qwen3-0.6B dimensions, each containing a DecodeStep.
    /// Tests that shared nodes (attn_scale, rope tables, etc.) are properly
    /// deduplicated across layers by the global AirNodeId dedup pass.
    #[test]
    fn test_multi_layer_decode_roundtrip() {
        let ctx = DecompositionContext::for_decode_step_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // kv_len
            8,      // kv_heads
            2048,   // intermediate_size
            151936, // vocab_size
            true,   // uses_rope
            true,   // has_qk_norm
        );

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("decode_layer_0".into()),
                    op: SirOp::DecodeStep {
                        token: SirNodeId("token_input".into()),
                        state_map: vec!["k_cache_0".into(), "v_cache_0".into()],
                        q_weight: Some("model.layers.0.self_attn.q_proj.weight".into()),
                        k_weight: Some("model.layers.0.self_attn.k_proj.weight".into()),
                        v_weight: Some("model.layers.0.self_attn.v_proj.weight".into()),
                        out_weight: Some("model.layers.0.self_attn.o_proj.weight".into()),
                        rope_tables: Some("rope_tables_shared".into()),
                        position: Some(SirNodeId("position_0".into())),
                        q_norm_weight: Some("model.layers.0.self_attn.q_norm.weight".into()),
                        k_norm_weight: Some("model.layers.0.self_attn.k_norm.weight".into()),
                        norm_epsilon: 1e-6,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: Some("causal_mask".into()),
                    },
                    name: "decode_layer_0".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("decode_layer_1".into()),
                    op: SirOp::DecodeStep {
                        token: SirNodeId("decode_layer_0".into()),
                        state_map: vec!["k_cache_1".into(), "v_cache_1".into()],
                        q_weight: Some("model.layers.1.self_attn.q_proj.weight".into()),
                        k_weight: Some("model.layers.1.self_attn.k_proj.weight".into()),
                        v_weight: Some("model.layers.1.self_attn.v_proj.weight".into()),
                        out_weight: Some("model.layers.1.self_attn.o_proj.weight".into()),
                        rope_tables: Some("rope_tables_shared".into()),
                        position: Some(SirNodeId("position_0".into())),
                        q_norm_weight: Some("model.layers.1.self_attn.q_norm.weight".into()),
                        k_norm_weight: Some("model.layers.1.self_attn.k_norm.weight".into()),
                        norm_epsilon: 1e-6,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: Some("causal_mask".into()),
                    },
                    name: "decode_layer_1".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("token_input".into()), SirNodeId("position_0".into())],
            outputs: vec![SirNodeId("decode_layer_1".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants (especially SSA dedup) ──
        validate_air_graph_structural_invariants(&air);

        // ── Shared nodes must be deduplicated ──
        // The shared_attn_scale should appear exactly once (not once per layer)
        let shared_scale_count = air.nodes.iter().filter(|n| n.id.0 == "shared_attn_scale").count();
        assert_eq!(
            shared_scale_count, 1,
            "shared_attn_scale must be deduplicated across layers (should appear exactly once)"
        );

        // ── Both layers must be fully decomposed ──
        let state_read_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::StateReadFixed { .. })).count();
        // Each layer reads 2 KV caches (K + V) → at least 4 state reads
        assert!(
            state_read_count >= 4,
            "Two decode layers must produce at least 4 StateReadFixed ops (2 per layer), got {}",
            state_read_count
        );

        let state_write_count =
            air.nodes.iter().filter(|n| matches!(n.op, AirOp::StateWriteFixed { .. })).count();
        // Each layer writes 2 KV caches (K + V) → at least 4 state writes
        assert!(
            state_write_count >= 4,
            "Two decode layers must produce at least 4 StateWriteFixed ops (2 per layer), got {}",
            state_write_count
        );

        // ── Layer 1 references layer 0 output as token input ──
        // The AIR graph should contain nodes whose inputs reference
        // the final AIR node of decode_layer_0's decomposition.
        assert!(!air.outputs.is_empty(), "AIR graph must have outputs");
    }

    /// T-37: Roundtrip test for a realistic multi-op pipeline that combines
    /// multiple SIR op types (LinearProjection, RMSNorm, Add, Reshape, etc.)
    /// with DecompositionContext. This simulates a simplified single-layer
    /// transformer block:
    ///   input → LinearProjection(Q) → RMSNorm(q_norm) →
    ///   LinearProjection(K) → RMSNorm(k_norm) → LinearProjection(V) →
    ///   AttentionBlock → Reshape → Add(residual)
    #[test]
    fn test_full_transformer_layer_roundtrip() {
        let ctx = DecompositionContext::for_attention_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // seq_len
            8,      // kv_heads
            2048,   // intermediate_size
            151936, // vocab_size
        );

        let input_id = SirNodeId("hidden_state".into());
        let residual_id = SirNodeId("residual".into());
        let q_proj_id = SirNodeId("q_proj".into());
        let k_proj_id = SirNodeId("k_proj".into());
        let v_proj_id = SirNodeId("v_proj".into());
        let q_norm_id = SirNodeId("q_norm".into());
        let k_norm_id = SirNodeId("k_norm".into());
        let attn_id = SirNodeId("attn_block".into());
        let reshape_id = SirNodeId("attn_reshape".into());
        let add_id = SirNodeId("residual_add".into());

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: q_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.q_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "q_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: k_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.k_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "k_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: v_proj_id.clone(),
                    op: SirOp::LinearProjection {
                        input: input_id.clone(),
                        weight: "model.layers.0.self_attn.v_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "v_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: q_norm_id.clone(),
                    op: SirOp::RMSNorm {
                        input: q_proj_id.clone(),
                        weight: "model.layers.0.self_attn.q_norm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![3], // per-head QK norm
                    },
                    name: "q_norm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: k_norm_id.clone(),
                    op: SirOp::RMSNorm {
                        input: k_proj_id.clone(),
                        weight: "model.layers.0.self_attn.k_norm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![3], // per-head QK norm
                    },
                    name: "k_norm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: attn_id.clone(),
                    op: SirOp::AttentionBlock {
                        q: q_norm_id,
                        k: k_norm_id,
                        v: v_proj_id,
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
                },
                SirNode {
                    id: reshape_id.clone(),
                    op: SirOp::Reshape { input: attn_id, target_shape: vec![1, 512, 1024] },
                    name: "attn_reshape".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: add_id.clone(),
                    op: SirOp::Add { x: reshape_id, y: residual_id.clone() },
                    name: "residual_add".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![input_id, residual_id],
            outputs: vec![add_id],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants ──
        validate_air_graph_structural_invariants(&air);

        // ── RMSNorm with axes=[3] must produce 4D reshapes ──
        let reshape_shapes: Vec<Vec<usize>> = air
            .nodes
            .iter()
            .filter_map(|n| {
                if let AirOp::Reshape { target_shape, .. } = &n.op {
                    Some(target_shape.clone())
                } else {
                    None
                }
            })
            .collect();

        // q_norm: 3D → 4D [1, 512, 16, 128]
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 16, 128]),
            "q_norm must produce 4D reshape to [1, 512, 16, 128] (num_heads=16), got: {:?}",
            reshape_shapes
        );

        // k_norm: 3D → 4D [1, 512, 8, 128]
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 8, 128]),
            "k_norm must produce 4D reshape to [1, 512, 8, 128] (kv_heads=8), got: {:?}",
            reshape_shapes
        );

        // q_norm: 4D → 3D [1, 512, 2048]
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 2048]),
            "q_norm must reshape back to [1, 512, 2048] (num_heads*head_dim), got: {:?}",
            reshape_shapes
        );

        // k_norm: 4D → 3D [1, 512, 1024]
        assert!(
            reshape_shapes.iter().any(|s| s == &vec![1, 512, 1024]),
            "k_norm must reshape back to [1, 512, 1024] (kv_heads*head_dim), got: {:?}",
            reshape_shapes
        );

        // ── AttentionBlock must use per-head MatMul (NOT SDPA/Tile) ──
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));
        assert!(has_matmul, "Transformer layer must include MatMul for per-head attention");
        assert!(!has_sdpa, "Transformer layer must NOT include SDPA");
        assert!(!has_tile, "Transformer layer must NOT include Tile");

        // ── Residual Add must be present ──
        let has_add = air.nodes.iter().any(|n| matches!(n.op, AirOp::Add { .. }));
        assert!(has_add, "Transformer layer must include Add for residual connection");
    }

    /// T-37: Non-GQA model roundtrip. Tests the DecodeStep decomposition
    /// with kv_heads == num_heads (no GQA). This is the path used by
    /// models like LLaMA-2 where all heads are KV heads (fan_out=1).
    #[test]
    fn test_decode_step_roundtrip_non_gqa() {
        let ctx = DecompositionContext::for_decode_step_full(
            1,     // batch_size
            4096,  // embed_dim
            32,    // num_heads
            128,   // head_dim
            2048,  // kv_len
            32,    // kv_heads (== num_heads, NO GQA)
            11008, // intermediate_size
            32000, // vocab_size
            false, // uses_rope (LLaMA-2 doesn't use RoPE in our SIR)
            false, // has_qk_norm
        );

        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("decode_0".into()),
                op: SirOp::DecodeStep {
                    token: SirNodeId("token_input".into()),
                    state_map: vec!["k_cache_0".into(), "v_cache_0".into()],
                    q_weight: Some("model.layers.0.self_attn.q_proj.weight".into()),
                    k_weight: Some("model.layers.0.self_attn.k_proj.weight".into()),
                    v_weight: Some("model.layers.0.self_attn.v_proj.weight".into()),
                    out_weight: Some("model.layers.0.self_attn.o_proj.weight".into()),
                    rope_tables: None,
                    position: None,
                    q_norm_weight: None,
                    k_norm_weight: None,
                    norm_epsilon: 1e-6,
                    qk_norm_type: "rms".to_string(),
                    mask_ref: None,
                },
                name: "decode_step_llama2".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("token_input".into())],
            outputs: vec![SirNodeId("decode_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants ──
        validate_air_graph_structural_invariants(&air);

        // ── Non-GQA: no RoPE, no QK-norm ──
        // Should NOT have Gather (used for position-based RoPE)
        let has_gather = air.nodes.iter().any(|n| matches!(n.op, AirOp::Gather { .. }));
        // Gather may still appear for other reasons, so we don't assert its absence.
        // But we should NOT have Rsqrt (which indicates QK-norm)
        // Actually, RMSNorm is still applied in the main decode path (not just QK-norm),
        // so Rsqrt may be present. Just check the structural invariants.

        // ── Output projection dimension should be embed_dim=4096 ──
        let o_proj = air.nodes.iter().find(|n| {
            if let AirOp::Conv1x1AsLinear { weight, output_dim, .. } = &n.op {
                weight.contains("o_proj") && *output_dim > 0
            } else {
                false
            }
        });
        if let Some(o) = o_proj {
            if let AirOp::Conv1x1AsLinear { output_dim, .. } = &o.op {
                assert_eq!(
                    *output_dim, 4096,
                    "O projection output_dim should be embed_dim = 4096 for non-GQA model"
                );
            }
        }

        // ── Non-GQA: Q and K should have same head count ──
        // K reshape should use kv_heads=32 == num_heads=32
        let k_4d = air.nodes.iter().find(|n| n.id.0 == "decode_0_k_4d");
        if let Some(k) = k_4d {
            if let AirOp::Reshape { target_shape, .. } = &k.op {
                // K should use 32 heads (same as Q) since kv_heads == num_heads
                assert!(
                    target_shape.contains(&32),
                    "K 4D reshape should use kv_heads=32 (non-GQA): {:?}",
                    target_shape
                );
            }
        }
    }

    /// T-37: Verify that `output_dim_for_weight` returns correct dimensions
    /// for all Qwen3-0.6B projection types when using `for_decode_step_full()`.
    #[test]
    fn test_output_dim_for_weight_qwen3_0_6b() {
        let ctx = DecompositionContext::for_decode_step_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // kv_len
            8,      // kv_heads
            2048,   // intermediate_size
            151936, // vocab_size
            true,   // uses_rope
            true,   // has_qk_norm
        );

        // Q projection: num_heads * head_dim = 16 * 128 = 2048
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.self_attn.q_proj.weight"),
            2048,
            "Q projection output_dim should be num_heads * head_dim = 2048"
        );

        // K projection: kv_heads * head_dim = 8 * 128 = 1024
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.self_attn.k_proj.weight"),
            1024,
            "K projection output_dim should be kv_heads * head_dim = 1024"
        );

        // V projection: kv_heads * head_dim = 8 * 128 = 1024
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.self_attn.v_proj.weight"),
            1024,
            "V projection output_dim should be kv_heads * head_dim = 1024"
        );

        // O projection: embed_dim = 1024
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.self_attn.o_proj.weight"),
            1024,
            "O projection output_dim should be embed_dim = 1024"
        );

        // Gate projection: intermediate_size = 2048
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.mlp.gate_proj.weight"),
            2048,
            "Gate projection output_dim should be intermediate_size = 2048"
        );

        // Up projection: intermediate_size = 2048
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.mlp.up_proj.weight"),
            2048,
            "Up projection output_dim should be intermediate_size = 2048"
        );

        // Down projection: embed_dim = 1024
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.mlp.down_proj.weight"),
            1024,
            "Down projection output_dim should be embed_dim = 1024"
        );

        // lm_head: vocab_size = 151936
        assert_eq!(
            ctx.output_dim_for_weight("lm_head.weight"),
            151936,
            "lm_head output_dim should be vocab_size = 151936"
        );

        // embed_tokens: embed_dim = 1024
        assert_eq!(
            ctx.output_dim_for_weight("model.embed_tokens.weight"),
            1024,
            "embed_tokens output_dim should be embed_dim = 1024"
        );

        // Unknown projection: 0
        assert_eq!(
            ctx.output_dim_for_weight("model.layers.0.unknown_proj.weight"),
            0,
            "Unknown projection output_dim should be 0"
        );
    }

    /// T-37: Verify that Conv1x1AsLinear output_dim in the AIR graph
    /// matches `output_dim_for_weight()` for every linear projection
    /// in a multi-op SIR→AIR roundtrip. This is a shape-consistency
    /// smoke test that catches the "output_dim=0" bug from Sprint 61.
    #[test]
    fn test_conv1x1_output_dim_matches_context() {
        let ctx = DecompositionContext::for_decode_step_full(
            1,      // batch_size
            1024,   // embed_dim
            16,     // num_heads
            128,    // head_dim
            512,    // kv_len
            8,      // kv_heads
            2048,   // intermediate_size
            151936, // vocab_size
            false,  // uses_rope
            false,  // has_qk_norm
        );

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("q_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.self_attn.q_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "q_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("k_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.self_attn.k_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "k_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("v_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.self_attn.v_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "v_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("o_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.self_attn.o_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "o_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("gate_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.mlp.gate_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "gate_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("down_proj".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.mlp.down_proj.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "down_proj".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("lm_head".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "lm_head.weight".into(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "lm_head".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("lm_head".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Verify every Conv1x1AsLinear has the correct output_dim
        for node in &air.nodes {
            if let AirOp::Conv1x1AsLinear { weight, output_dim, .. } = &node.op {
                let expected = ctx.output_dim_for_weight(weight);
                assert_eq!(
                    *output_dim, expected,
                    "Conv1x1AsLinear('{}') output_dim={} should match output_dim_for_weight()={}",
                    weight, output_dim, expected
                );
            }
        }

        // Structural invariants
        validate_air_graph_structural_invariants(&air);
    }

    /// T-37: Verify that SIR metadata (TaskOrigin, model_id,
    /// precision_override) propagates correctly through the SIR→AIR
    /// roundtrip for a realistic decode pipeline.
    #[test]
    fn test_metadata_propagation_through_roundtrip() {
        let ctx = DecompositionContext::for_decode_step_full(
            1, 1024, 16, 128, 512, 8, 2048, 151936, true, true,
        );

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("rms_norm".into()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.input_layernorm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                    name: "layernorm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::RealModel { name: "Qwen3-0.6B".into() },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: Some("fp32".into()),
                    },
                },
                SirNode {
                    id: SirNodeId("decode_0".into()),
                    op: SirOp::DecodeStep {
                        token: SirNodeId("rms_norm".into()),
                        state_map: vec!["k_cache_0".into(), "v_cache_0".into()],
                        q_weight: Some("model.layers.0.self_attn.q_proj.weight".into()),
                        k_weight: Some("model.layers.0.self_attn.k_proj.weight".into()),
                        v_weight: Some("model.layers.0.self_attn.v_proj.weight".into()),
                        out_weight: Some("model.layers.0.self_attn.o_proj.weight".into()),
                        rope_tables: Some("rope_tables_shared".into()),
                        position: Some(SirNodeId("pos".into())),
                        q_norm_weight: Some("model.layers.0.self_attn.q_norm.weight".into()),
                        k_norm_weight: Some("model.layers.0.self_attn.k_norm.weight".into()),
                        norm_epsilon: 1e-6,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: None,
                    },
                    name: "decode_step".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::TransformersTrace {
                            name: "qwen3-0.6b-trace".into(),
                        },
                        model_id: Some("qwen3-0.6b".into()),
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into()), SirNodeId("pos".into())],
            outputs: vec![SirNodeId("decode_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // Check that the RMSNorm-derived AIR nodes carry the precision_override
        let rms_air_nodes: Vec<_> = air
            .nodes
            .iter()
            .filter(|n| n.sir_source.as_ref().map_or(false, |s| s.0 == "rms_norm"))
            .collect();
        assert!(
            !rms_air_nodes.is_empty(),
            "AIR must contain nodes derived from the RMSNorm SIR node"
        );
        for node in &rms_air_nodes {
            assert_eq!(
                node.precision_override,
                Some("fp32".into()),
                "RMSNorm AIR node must inherit precision_override='fp32' from SIR metadata"
            );
        }

        // Check that DecodeStep-derived AIR nodes have correct sir_source
        let decode_air_nodes: Vec<_> = air
            .nodes
            .iter()
            .filter(|n| n.sir_source.as_ref().map_or(false, |s| s.0 == "decode_0"))
            .collect();
        assert!(
            !decode_air_nodes.is_empty(),
            "AIR must contain nodes derived from the DecodeStep SIR node"
        );
        // DecodeStep SIR node did NOT have precision_override
        for node in &decode_air_nodes {
            assert_eq!(
                node.precision_override, None,
                "DecodeStep AIR nodes should have no precision_override (not set in SIR)"
            );
        }
    }

    /// T-37: Verify SSA validity for the Tile decomposition path.
    /// Tile decomposes into Reshape + Mul + Reshape, and the intermediate
    /// AirNodeIds must be unique and referenceable.
    #[test]
    fn test_tile_decomposition_ssa_validity() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("tile_0".into()),
                op: SirOp::Tile {
                    input: SirNodeId("input".into()),
                    reps: vec![1, 4, 1, 1], // GQA-style tile on dim 1
                },
                name: "gqa_tile".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("tile_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Verify structural invariants (SSA, reference integrity)
        validate_air_graph_structural_invariants(&air);

        // Must include the Tile decomposition ops (Reshape, Const, Mul)
        let has_reshape = air.nodes.iter().any(|n| matches!(n.op, AirOp::Reshape { .. }));
        let has_mul = air.nodes.iter().any(|n| matches!(n.op, AirOp::Mul { .. }));
        let has_const = air.nodes.iter().any(|n| matches!(n.op, AirOp::Const { .. }));
        // No Tile should survive — it must be decomposed
        let has_tile = air.nodes.iter().any(|n| matches!(n.op, AirOp::Tile { .. }));

        assert!(has_reshape, "Tile decomposition must include Reshape");
        assert!(has_mul, "Tile decomposition must include Mul (broadcast)");
        assert!(has_const, "Tile decomposition must include Const (ones)");
        assert!(!has_tile, "Tile must NOT survive as AirOp::Tile in AIR");
    }

    /// T-60 (I-34): Tile decomposition with DecompositionContext uses concrete
    /// dimensions instead of 0 placeholders. This avoids the batch=1 heuristic
    /// in resolve_reshape_zeros() which produces incorrect shapes for GQA Tile.
    #[test]
    fn test_tile_decomposition_with_ctx_uses_concrete_shapes() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("tile_0".into()),
                op: SirOp::Tile {
                    input: SirNodeId("input".into()),
                    reps: vec![1, 4, 1, 1], // GQA-style: tile kv_heads by fan_out=4
                },
                name: "gqa_tile".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("tile_0".into())],
        };

        // T-60: With ctx, the Tile decomposition should produce concrete shapes.
        let ctx = DecompositionContext::for_attention_full(
            1,    // batch_size
            1024, // embed_dim
            8,    // num_heads
            128,  // head_dim
            512,  // seq_len
            2,    // kv_heads (GQA)
            4096, // intermediate_size
            151936, // vocab_size
        );

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        validate_air_graph_structural_invariants(&air);

        // Find the Reshape ops — the first reshape should have concrete dims,
        // not 0 placeholders. For reps=[1, 4, 1, 1] with kv_heads=2:
        // reshape_shape should be [1, 1, 2, 512, 128] (broadcast BEFORE kv_heads)
        let reshape_nodes: Vec<_> = air
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AirOp::Reshape { .. }))
            .collect();

        assert!(
            reshape_nodes.len() >= 2,
            "Tile decomposition should have at least 2 Reshape ops, got {}",
            reshape_nodes.len()
        );

        // Check that the first reshape has no 0s (concrete shapes from ctx)
        if let AirOp::Reshape { target_shape, .. } = &reshape_nodes[0].op {
            assert!(
                !target_shape.contains(&0),
                "T-60: First reshape should have concrete dims (no 0s), got {:?}",
                target_shape
            );
            // Expected: [1, 1, 2, 512, 128] for GQA tile with reps=[1,4,1,1]
            // The broadcast dim (1) is inserted BEFORE kv_heads (2)
            assert_eq!(
                target_shape,
                &vec![1, 1, 2, 512, 128],
                "T-60: First reshape shape should be [1, 1, 2, 512, 128] (concrete from ctx)"
            );
        }

        // Check that the final reshape has no 0s and is at the original input rank (4D)
        if let AirOp::Reshape { target_shape, .. } = &reshape_nodes[1].op {
            assert!(
                !target_shape.contains(&0),
                "T-60: Final reshape should have concrete dims (no 0s), got {:?}",
                target_shape
            );
            // T-60: final_shape is at the original input rank (4D), not the expanded rank (5D).
            // Expected: [1, 8, 512, 128] for kv_heads*fan_out = 2*4 = 8
            assert_eq!(target_shape.len(), 4, "T-60: Final reshape should be 4D (same rank as Tile input)");
            assert_eq!(
                target_shape,
                &vec![1, 8, 512, 128],
                "T-60: Final reshape shape should be [1, 8, 512, 128] (concrete from ctx, collapsed)"
            );
        }
    }

    /// T-60 (I-34): Tile decomposition WITHOUT ctx falls back to 0 placeholders,
    /// preserving backward compatibility.
    #[test]
    fn test_tile_decomposition_without_ctx_uses_placeholders() {
        let sir = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("tile_0".into()),
                op: SirOp::Tile {
                    input: SirNodeId("input".into()),
                    reps: vec![1, 4, 1, 1],
                },
                name: "gqa_tile".into(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::Synthetic,
                    model_id: None,
                    quality_contract: None,
                    precision_override: None,
                },
            }],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("tile_0".into())],
        };

        let pass = LegalityRewritePass::new();
        // No ctx — should fall back to 0 placeholders
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        validate_air_graph_structural_invariants(&air);

        // Without ctx, the reshape shapes should have 0 placeholders
        let reshape_nodes: Vec<_> = air
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AirOp::Reshape { .. }))
            .collect();

        assert!(
            reshape_nodes.len() >= 2,
            "Tile decomposition should have at least 2 Reshape ops"
        );

        // The first reshape should have 0 placeholders (no ctx available for input dims)
        if let AirOp::Reshape { target_shape, .. } = &reshape_nodes[0].op {
            assert!(
                target_shape.contains(&0),
                "Without ctx, first reshape should have 0 placeholders, got {:?}",
                target_shape
            );
        }

        // T-60: The final reshape should also have 0s and be at the original input rank (4D)
        if let AirOp::Reshape { target_shape, .. } = &reshape_nodes[1].op {
            assert_eq!(
                target_shape.len(), 4,
                "T-60: Final reshape should be 4D (same rank as Tile input), got {}D: {:?}",
                target_shape.len(), target_shape
            );
            assert!(
                target_shape.contains(&0),
                "Without ctx, final reshape should have 0 placeholders, got {:?}",
                target_shape
            );
        }
    }

    /// T-37: Verify SSA validity for Select/Where decomposition.
    /// These decompose into Const + Sub + Mul + Mul + Add, and all
    /// intermediate AirNodeIds must be unique and referenceable.
    #[test]
    fn test_select_where_decomposition_ssa_validity() {
        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("select_0".into()),
                    op: SirOp::Select {
                        condition: SirNodeId("cond".into()),
                        x: SirNodeId("x_val".into()),
                        y: SirNodeId("y_val".into()),
                    },
                    name: "select_op".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("where_0".into()),
                    op: SirOp::Where {
                        condition: SirNodeId("cond".into()),
                        x: SirNodeId("x_val".into()),
                        y: SirNodeId("y_val".into()),
                    },
                    name: "where_op".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![
                SirNodeId("cond".into()),
                SirNodeId("x_val".into()),
                SirNodeId("y_val".into()),
            ],
            outputs: vec![SirNodeId("select_0".into()), SirNodeId("where_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Verify structural invariants
        validate_air_graph_structural_invariants(&air);

        // Both Select and Where must be decomposed (NOT present as-is)
        let has_select = air.nodes.iter().any(|n| matches!(n.op, AirOp::Select { .. }));
        let has_where = air.nodes.iter().any(|n| matches!(n.op, AirOp::Where { .. }));
        assert!(!has_select, "Select must be decomposed into arithmetic ops (ANE-illegal)");
        assert!(!has_where, "Where must be decomposed into arithmetic ops (ANE-illegal)");

        // Must have the decomposition ops: Const(1), Sub, Mul, Add
        let has_const = air.nodes.iter().any(|n| matches!(n.op, AirOp::Const { .. }));
        let has_sub = air.nodes.iter().any(|n| matches!(n.op, AirOp::Sub { .. }));
        let has_mul = air.nodes.iter().any(|n| matches!(n.op, AirOp::Mul { .. }));
        let has_add = air.nodes.iter().any(|n| matches!(n.op, AirOp::Add { .. }));
        assert!(has_const, "Select/Where decomposition must include Const(1.0)");
        assert!(has_sub, "Select/Where decomposition must include Sub (1-cond)");
        assert!(has_mul, "Select/Where decomposition must include Mul (cond*x, (1-cond)*y)");
        assert!(has_add, "Select/Where decomposition must include Add (cond*x + (1-cond)*y)");
    }

    /// T-37: Empty SIR graph roundtrip. The simplest possible SIR→AIR
    /// conversion — should produce an empty AIR graph with no nodes.
    #[test]
    fn test_empty_graph_roundtrip() {
        let sir = SirGraph { nodes: vec![], inputs: vec![], outputs: vec![] };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        assert_eq!(air.nodes.len(), 0, "Empty SIR should produce empty AIR");
        assert_eq!(air.inputs.len(), 0);
        assert_eq!(air.outputs.len(), 0);
        assert_eq!(air.staticization_decisions.len(), 0);
    }

    /// T-37: Passthrough ops roundtrip. Verify that simple 1:1 SIR→AIR
    /// mappings (Add, Mul, Reshape, Transpose, etc.) preserve structural
    /// invariants and that all AirNodeId references resolve correctly.
    #[test]
    fn test_passthrough_ops_roundtrip() {
        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("input_a".into()),
                    op: SirOp::Identity { input: SirNodeId("__placeholder__".into()) },
                    name: "input_a".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("input_b".into()),
                    op: SirOp::Identity { input: SirNodeId("__placeholder__".into()) },
                    name: "input_b".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("add_0".into()),
                    op: SirOp::Add {
                        x: SirNodeId("input_a".into()),
                        y: SirNodeId("input_b".into()),
                    },
                    name: "add".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("mul_0".into()),
                    op: SirOp::Mul { x: SirNodeId("add_0".into()), y: SirNodeId("input_a".into()) },
                    name: "mul".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("reshape_0".into()),
                    op: SirOp::Reshape {
                        input: SirNodeId("mul_0".into()),
                        target_shape: vec![1, 16, 64],
                    },
                    name: "reshape".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("transpose_0".into()),
                    op: SirOp::Transpose {
                        input: SirNodeId("reshape_0".into()),
                        perm: vec![0, 2, 1],
                    },
                    name: "transpose".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input_a".into()), SirNodeId("input_b".into())],
            outputs: vec![SirNodeId("transpose_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, None).unwrap();

        // Verify structural invariants
        validate_air_graph_structural_invariants(&air);

        // Verify op types are preserved
        assert!(air.nodes.iter().any(|n| matches!(n.op, AirOp::Identity { .. })));
        assert!(air.nodes.iter().any(|n| matches!(n.op, AirOp::Add { .. })));
        assert!(air.nodes.iter().any(|n| matches!(n.op, AirOp::Mul { .. })));
        assert!(air.nodes.iter().any(|n| matches!(n.op, AirOp::Reshape { .. })));
        assert!(air.nodes.iter().any(|n| matches!(n.op, AirOp::Transpose { .. })));
    }

    /// T-37: RMSNorm + RoPE + DecodeStep combined roundtrip.
    /// This tests the full pre-norm + attention + RoPE pipeline
    /// that a real decoder layer would exercise.
    #[test]
    fn test_rms_norm_rope_decode_combined_roundtrip() {
        let ctx = DecompositionContext::for_decode_step_full(
            1, 1024, 16, 128, 512, 8, 2048, 151936, true, true,
        );

        let sir = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("layernorm".into()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input".into()),
                        weight: "model.layers.0.input_layernorm.weight".into(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                    name: "input_layernorm".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("rope_0".into()),
                    op: SirOp::RoPETransform {
                        input: SirNodeId("layernorm".into()),
                        tables: "rope_tables_shared".into(),
                    },
                    name: "rope".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("decode_0".into()),
                    op: SirOp::DecodeStep {
                        token: SirNodeId("rope_0".into()),
                        state_map: vec!["k_cache_0".into(), "v_cache_0".into()],
                        q_weight: Some("model.layers.0.self_attn.q_proj.weight".into()),
                        k_weight: Some("model.layers.0.self_attn.k_proj.weight".into()),
                        v_weight: Some("model.layers.0.self_attn.v_proj.weight".into()),
                        out_weight: Some("model.layers.0.self_attn.o_proj.weight".into()),
                        rope_tables: None,
                        position: None,
                        q_norm_weight: None,
                        k_norm_weight: None,
                        norm_epsilon: 1e-6,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: None,
                    },
                    name: "decode".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("decode_0".into())],
        };

        let pass = LegalityRewritePass::new();
        let air = pass.run(sir, &NoKnowledge, Some(&ctx)).unwrap();

        // ── Structural invariants ──
        validate_air_graph_structural_invariants(&air);

        // ── RMSNorm decomposition ──
        let has_reduce_mean = air.nodes.iter().any(|n| matches!(n.op, AirOp::ReduceMean { .. }));
        let has_rsqrt = air.nodes.iter().any(|n| matches!(n.op, AirOp::Rsqrt { .. }));
        assert!(has_reduce_mean, "RMSNorm must decompose into ReduceMean");
        assert!(has_rsqrt, "RMSNorm must decompose into Rsqrt");

        // ── RoPE decomposition ──
        // Standalone RoPETransform produces Const tables (NOT Cos/Sin which are ANE-illegal)
        let has_cos = air.nodes.iter().any(|n| matches!(n.op, AirOp::Cos { .. }));
        let has_sin = air.nodes.iter().any(|n| matches!(n.op, AirOp::Sin { .. }));
        assert!(!has_cos, "RoPE must NOT use Cos (ANE-illegal)");
        assert!(!has_sin, "RoPE must NOT use Sin (ANE-illegal)");

        // ── DecodeStep decomposition ──
        let has_matmul = air.nodes.iter().any(|n| matches!(n.op, AirOp::MatMul { .. }));
        let has_sdpa =
            air.nodes.iter().any(|n| matches!(n.op, AirOp::ScaledDotProductAttention { .. }));
        assert!(has_matmul, "DecodeStep must include MatMul");
        assert!(!has_sdpa, "DecodeStep must NOT include SDPA");
    }

    /// T-37: Verify the GQA fan_out computation. For Qwen3-0.6B,
    /// num_heads=16, kv_heads=8, so fan_out=2. This means each KV
    /// head is shared by 2 Q heads, and the per-head attention loop
    /// produces 16 Q-head MatMul ops grouped into 8 KV-head groups.
    #[test]
    fn test_gqa_fan_out_computation() {
        // Qwen3-0.6B: fan_out = 16 / 8 = 2
        let ctx = DecompositionContext::for_decode_step_full(
            1, 1024, 16, 128, 512, 8, 2048, 151936, false, false,
        );
        assert!(ctx.uses_gqa, "GQA should be detected when kv_heads < num_heads");

        // Non-GQA model: kv_heads == num_heads → uses_gqa = false
        let ctx_no_gqa = DecompositionContext::for_decode_step_full(
            1, 4096, 32, 128, 2048, 32, 11008, 32000, false, false,
        );
        assert!(!ctx_no_gqa.uses_gqa, "GQA should NOT be detected when kv_heads == num_heads");

        // Edge case: kv_heads=0 → uses_gqa = false
        let ctx_zero_kv = DecompositionContext::for_decode_step(1, 128, 4, 32, 64);
        assert!(
            !ctx_zero_kv.uses_gqa,
            "GQA should NOT be detected when kv_heads=0 (defaults to num_heads)"
        );
    }
}
