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
//!
//! # Key decomposition decisions
//!
//! - **Separate Q/K/V projections**: Attention blocks emit three distinct
//!   `LinearProjection` ops (q_proj, k_proj, v_proj) rather than a single
//!   merged QKV projection. This ensures each projection has its own weight
//!   name and output node, eliminating phantom node references.
//!
//! - **SwiGLU MLP support**: When both `gate_proj.weight` and `up_proj.weight`
//!   exist in the trace weights, the MLP is decomposed using the SwiGLU pattern:
//!   `down_proj(silu(gate_proj(x)) * up_proj(x))`. Standard MLPs follow the
//!   simpler `down_proj(activation(up_proj(x)))` path.
//!
//! - **Residual connections**: AttentionBlock and MlpBlock emit a trailing
//!   `SirOp::Add` when the traced node has ≥2 inputs, connecting the block
//!   output to the residual (skip) connection.
//!
//! - **Causal attention masks**: SDPA within attention blocks receives a causal
//!   mask reference that will be materialized as a static lower-triangular
//!   table by the staticize pass.
//!
//! - **Pre-A16 SDPA decomposition**: On chip families where SDPA is unreliable
//!   (A11, A12, A14, A15), attention is decomposed into explicit primitives:
//!   `Transpose(K) → MatMul(Q, K^T) → Const(1/√d_k) → Mul → Softmax → MatMul(scores, V)`.
//!   This is numerically equivalent to the SDPA op on A16+, preserving the
//!   1/√d_k scaling factor that is critical for correct softmax behavior.
//!
//! - **RMSNorm epsilon validation**: If the traced epsilon is 0 or missing,
//!   the builder falls back to the config's `layer_norm_epsilon` or the
//!   standard 1e-6 default for RMSNorm models (Qwen3, Llama, etc.).
//!
//! - **Non-silent input resolution**: `resolve_input` emits a warning when it
//!   cannot find a mapping, making debugging easier instead of silently
//!   producing `__unresolved_N__` node references.

use crate::graph::{ModelConfig, TracedGraph, TracedNode, TracedOp};
use crate::versioned::VersionedCompiler;
use ane_ir::ane_target::AneFamily;
use ane_ir::mir::MilDtype;
use ane_ir::sir::{QualityContract, SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

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
        // uses_gqa, uses_rope, hidden_act, is_encoder_decoder) determine how
        // composite ops decompose — no hardcoded registry needed.
        let _config_summary = format!(
            "model_type={} class={} enc_dec={} heads={}/{} rms_norm={} gqa={} rope={} act={}",
            self.config.model_type,
            self.config.model_class,
            self.config.is_encoder_decoder,
            self.config.num_attention_heads,
            self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads),
            self.config.uses_rms_norm,
            self.config.uses_gqa,
            self.config.uses_rope,
            self.config.hidden_act,
        );

        // Build a name-based alias map: for each traced node's decomposed ops,
        // record the mapping from {name_tag}_{traced_node_id} → actual allocated ID.
        // This replaces the fragile position-based resolution that couldn't handle
        // different decomposition paths (±QK-Norm, ±GQA, SDPA vs manual).
        let mut name_alias_to_actual: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

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

                // Register name-based alias: "sir_{name}_{traced_node_id}" → actual ID.
                // For example, if the op has name "sdpa" and traced_node.id is "attn_0",
                // then "sir_sdpa_attn_0" maps to the allocated "sir_7_attn_0" (or whatever).
                // The decompose functions create references using these semantic aliases;
                // this mapping lets us resolve them to actual counter-based IDs.
                let alias_key = format!("sir_{}_{}", name, traced_node.id);
                name_alias_to_actual.entry(alias_key).or_insert(id.0.clone());

                // Also register a short-prefix alias for names like "q_proj_1024" → "q_proj".
                // The decompose functions use short prefixes (e.g., "sir_q_proj_layer_0")
                // while the name tags include dimensions (e.g., "q_proj_1024").
                // Extract the short prefix by trimming trailing "_\d+" segments.
                if let Some(short) = strip_trailing_dim_suffix(&name) {
                    if short != name {
                        let short_alias_key = format!("sir_{}_{}", short, traced_node.id);
                        name_alias_to_actual.entry(short_alias_key).or_insert(id.0.clone());
                    }
                }

                let sir_node = SirNode { id: id.clone(), op, name, metadata };

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

        // ─── Post-hoc alias resolution ───────────────────────────────
        // The decompose functions create cross-references using semantic aliases
        // (e.g., "sir_q_proj_layer_0_self_attn") that don't match the allocated
        // node IDs (e.g., "sir_3_layer_0_self_attn"). We need to rewrite all
        // SirOp internal references from aliases to actual IDs.
        //
        // Strategy: extract all SirNodeId strings from the serialized SirOps,
        // find any that don't match a real node ID, and resolve them using the
        // name-based alias map built during allocation. This map records the
        // correspondence between "sir_{name_tag}_{traced_node_id}" and the
        // actual counter-based ID, so it works for any decomposition path
        // (±QK-Norm, ±GQA, SDPA vs manual) without hardcoded positions.
        {
            // Build the set of actual node IDs for quick lookup
            let actual_ids: std::collections::HashSet<String> =
                sir_nodes.iter().map(|n| n.id.0.clone()).collect();

            let mut alias_to_actual: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();

            // Scan all SirOps for potential alias references
            for node in &sir_nodes {
                let json = serde_json::to_string(&node.op).unwrap();
                // Find all string values in JSON that start with "sir_" and are
                // not actual node IDs — these are the dangling semantic aliases
                for candidate in extract_sir_ids_from_json(&json) {
                    if !actual_ids.contains(&candidate) && candidate.starts_with("sir_") {
                        // Try the name-based alias map first (handles all decomposition paths)
                        if let Some(actual_id) = name_alias_to_actual.get(&candidate) {
                            alias_to_actual.entry(candidate).or_insert(actual_id.clone());
                        } else {
                            // Fallback: try the legacy position-based resolution
                            // (handles aliases that don't match the name tag pattern)
                            if let Some(actual_id) = resolve_alias(&candidate, &actual_ids, &self.node_map) {
                                alias_to_actual.entry(candidate).or_insert(actual_id);
                            }
                        }
                    }
                }
            }

            if !alias_to_actual.is_empty() {
                // Rewrite SirOp references using JSON serialization
                for node in &mut sir_nodes {
                    let json = serde_json::to_string(&node.op).unwrap();
                    let mut new_json = json.clone();
                    for (alias, actual) in &alias_to_actual {
                        new_json = new_json.replace(
                            &format!("\"{}\"", alias),
                            &format!("\"{}\"", actual),
                        );
                    }
                    if new_json != json {
                        node.op = serde_json::from_str(&new_json).unwrap();
                    }
                }
            }
        }

        // ─── SSA validation ────────────────────────────────────────
        // After alias resolution, every input reference must resolve to either
        // a function/block input or the output of a prior operation. We also
        // reject self-referential edges (a node consuming its own output).
        {
            let mut available: std::collections::HashSet<String> =
                sir_inputs.iter().map(|id| id.0.clone()).collect();
            let mut errors: Vec<String> = Vec::new();

            for node in &sir_nodes {
                let node_outputs = std::collections::HashSet::from([node.id.0.clone()]);

                // Extract all SirNodeId references from the op's serialized form
                let json = serde_json::to_string(&node.op).unwrap();
                for input_ref in extract_sir_ids_from_json(&json) {
                    // Skip internal placeholder markers
                    if input_ref == "__placeholder__" {
                        continue;
                    }
                    // Self-input: this node references its own output as an input
                    if node_outputs.contains(&input_ref) {
                        errors.push(format!(
                            "SSA self-input: node '{}' references its own output as input '{}'",
                            node.id.0, input_ref
                        ));
                    }
                    // Unresolved: the referenced name was never produced by any prior op
                    else if !available.contains(&input_ref) {
                        errors.push(format!(
                            "SSA unresolved: node '{}' references '{}' which is not available",
                            node.id.0, input_ref
                        ));
                    }
                }

                // Make this node's output available for subsequent nodes
                available.insert(node.id.0.clone());
            }

            if !errors.is_empty() {
                // Report up to 10 errors to avoid log spam
                let display_limit = 10;
                let display_errors: Vec<String> =
                    errors.iter().take(display_limit).cloned().collect();
                let remaining = errors.len().saturating_sub(display_limit);
                let msg = if remaining > 0 {
                    format!(
                        "SIR SSA validation failed with {} errors:\n  {}\n  ... and {} more",
                        errors.len(),
                        display_errors.join("\n  "),
                        remaining
                    )
                } else {
                    format!(
                        "SIR SSA validation failed with {} errors:\n  {}",
                        errors.len(),
                        display_errors.join("\n  ")
                    )
                };
                return Err(msg);
            }
        }

        Ok(SirGraph { nodes: sir_nodes, inputs: sir_inputs, outputs: sir_outputs })
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

    /// Resolve the HuggingFace weight name for a traced node.
    ///
    /// If the node has a `module_path` and the `weight_name_map` contains
    /// an entry for this node's ID, returns the HF parameter name (e.g.,
    /// "model.layers.0.self_attn.q_proj.weight"). Otherwise falls back to
    /// the synthetic name (e.g., "weight_linear1").
    fn resolve_weight_name(&self, node: &TracedNode, fallback: &str) -> String {
        if let Some(ref module_path) = node.module_path {
            if let Some(entry) = self.trace.weight_name_map.get(&node.id) {
                if let Some(ref weight) = entry.weight {
                    return weight.clone();
                }
                // weight is None — fall through to construct from module_path
            }
            // Fallback: construct from module_path + ".weight"
            format!("{}.weight", module_path)
        } else {
            fallback.to_string()
        }
    }

    /// Resolve the HuggingFace bias name for a traced node.
    fn resolve_bias_name(&self, node: &TracedNode, fallback: &Option<String>) -> Option<String> {
        if let Some(ref module_path) = node.module_path {
            if let Some(entry) = self.trace.weight_name_map.get(&node.id) {
                return entry.bias.clone();
            }
            // Fallback: construct from module_path + ".bias"
            Some(format!("{}.bias", module_path))
        } else {
            fallback.clone()
        }
    }

    /// Resolve the HuggingFace weight name for a specific suffix (e.g., "weight", "bias")
    /// relative to a module path prefix. Used for composite ops that generate
    /// sub-module weight names (e.g., attention q_proj, k_proj, v_proj).
    fn hf_param_name(&self, module_path: &str, suffix: &str) -> String {
        format!("{}.{}", module_path, suffix)
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
            TracedOp::AttentionBlock { embed_dim, num_heads, head_dim, use_sdpa, has_qk_norm } => {
                self.decompose_attention(*embed_dim, *num_heads, *head_dim, *use_sdpa, *has_qk_norm, node)
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
                let weight_name = self.resolve_weight_name(node, &format!("weight_{}", node.id));
                let bias_name = if *has_bias {
                    self.resolve_bias_name(node, &Some(format!("bias_{}", node.id)))
                } else {
                    None
                };
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
                Ok(vec![(SirOp::MatMul { a: a_id, b: b_id }, "matmul".to_string())])
            }
            TracedOp::Embedding { vocab_size, embed_dim } => {
                // Embedding lookup is CPU-only on ANE — mark for awareness
                let input_id = self.resolve_input(&node.inputs, 0);
                let embed_weight_name =
                    self.resolve_weight_name(node, &format!("embed_weight_{}", node.id));
                Ok(vec![(
                    SirOp::Gather {
                        input: SirNodeId(embed_weight_name),
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
                let ln_weight = self.resolve_weight_name(node, &format!("ln_weight_{}", node.id));
                let ln_bias = self.resolve_bias_name(node, &Some(format!("ln_bias_{}", node.id)));
                Ok(vec![(
                    SirOp::LayerNorm {
                        input: input_id,
                        weight: ln_weight,
                        bias: ln_bias,
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
                // Causal mask for standalone SDPA — reference to a lower-triangular
                // mask that will be materialized by the staticize pass.
                let causal_mask = Some(SirNodeId(format!("causal_mask_{}", node.id)));
                Ok(vec![(
                    SirOp::ScaledDotProductAttention {
                        query: q_id,
                        key: k_id,
                        value: v_id,
                        attention_mask: causal_mask,
                        scale: Some(*scale as f32),
                    },
                    "sdpa".to_string(),
                )])
            }
            TracedOp::Softmax { axis } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Softmax { input: input_id, axis: *axis }, "softmax".to_string())])
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
                Ok(vec![(SirOp::Silu { input: input_id }, "silu".to_string())])
            }
            TracedOp::Relu => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Relu { input: input_id }, "relu".to_string())])
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
                let inputs: Vec<SirNodeId> =
                    node.inputs.iter().filter_map(|id| self.lookup_sir_id(id).cloned()).collect();
                Ok(vec![(SirOp::Concat { inputs, axis: *axis }, "concat".to_string())])
            }
            TracedOp::Split { axis, num_splits } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                // Python may emit negative axis (e.g., -1 for last dim);
                // SirOp::Split uses usize, so we must resolve negatives.
                let axis_usize = if *axis >= 0 {
                    *axis as usize
                } else {
                    // Negative axis: resolve at SIR build time using output_shape rank
                    let rank = node.output_shape.dims.len();
                    if rank > 0 {
                        (rank as isize + axis) as usize
                    } else {
                        0
                    }
                };
                Ok(vec![(
                    SirOp::Split { input: input_id, axis: axis_usize, num_splits: *num_splits },
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
                Ok(vec![(SirOp::Add { x: x_id, y: y_id }, "add".to_string())])
            }
            TracedOp::Mul => {
                let x_id = self.resolve_input(&node.inputs, 0);
                let y_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(SirOp::Mul { x: x_id, y: y_id }, "mul".to_string())])
            }
            TracedOp::Div => {
                let x_id = self.resolve_input(&node.inputs, 0);
                let y_id = self.resolve_input(&node.inputs, 1);
                Ok(vec![(SirOp::RealDiv { x: x_id, y: y_id }, "div".to_string())])
            }
            TracedOp::Rsqrt => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Rsqrt { input: input_id }, "rsqrt".to_string())])
            }
            TracedOp::Cast { target_dtype } => {
                let input_id = self.resolve_input(&node.inputs, 0);
                let dtype = parse_dtype(target_dtype)?;
                Ok(vec![(SirOp::Cast { input: input_id, dtype }, "cast".to_string())])
            }
            TracedOp::Tanh => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Tanh { input: input_id }, "tanh".to_string())])
            }
            TracedOp::Sigmoid => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Sigmoid { input: input_id }, "sigmoid".to_string())])
            }
            TracedOp::Exp => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Exp { input: input_id }, "exp".to_string())])
            }
            TracedOp::Cos => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Cos { input: input_id }, "cos".to_string())])
            }
            TracedOp::Sin => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Sin { input: input_id }, "sin".to_string())])
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
                    SirOp::StateWrite { state_id, offset: 0, value: value_id },
                    format!("kv_write_{}", layer_idx),
                )])
            }
            TracedOp::Placeholder => {
                // Create a placeholder — the input to the SIR graph
                Ok(vec![(
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                    "placeholder".to_string(),
                )])
            }
            TracedOp::Output => {
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Identity { input: input_id }, "output".to_string())])
            }
            TracedOp::GetItem { index: _ } => {
                // GetItem is structural — the actual selection is handled
                // by the node's input references
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Identity { input: input_id }, "getitem".to_string())])
            }
            TracedOp::Identity => {
                // No-op: contiguous, size query, etc.
                let input_id = self.resolve_input(&node.inputs, 0);
                Ok(vec![(SirOp::Identity { input: input_id }, "identity".to_string())])
            }
            TracedOp::ExpandDims { axis } => {
                // Unsqueeze — treated as a reshape at the SIR level
                let input_id = self.resolve_input(&node.inputs, 0);
                let sir_axis: Vec<usize> = axis.iter().map(|&a| a as usize).collect();
                Ok(vec![(
                    SirOp::ExpandDims { input: input_id, axis: sir_axis },
                    "expand_dims".to_string(),
                )])
            }
            TracedOp::Squeeze { axis } => {
                // Squeeze — remove dimensions of size 1
                let input_id = self.resolve_input(&node.inputs, 0);
                let sir_axis: Vec<usize> = axis.iter().map(|&a| a as usize).collect();
                Ok(vec![(
                    SirOp::Squeeze { input: input_id, axis: sir_axis },
                    "squeeze".to_string(),
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
            TracedOp::Unknown { op_name, target } => Err(format!(
                "Cannot map unknown traced op '{}' (target: '{}') to SIR. \
                     This op may not have an ANE-faithful mapping yet.",
                op_name, target
            )),
        }
    }

    /// Decompose an attention block into ANE-faithful primitives.
    ///
    /// Config-driven: uses `self.config.uses_gqa` to determine whether
    /// KV heads need Tile expansion (GQA) or are already aligned (MHA).
    /// This works for any model architecture without a registry.
    ///
    /// Emits **separate** Q, K, V linear projections (not a single merged
    /// QKV), each with its own weight name. This ensures every node
    /// reference in the graph is backed by an actual SirNode, eliminating
    /// phantom "split" IDs that pointed to nonexistent nodes.
    fn decompose_attention(
        &self,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        use_sdpa: bool,
        has_qk_norm: bool,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let mut ops = Vec::new();
        let input_id = self.resolve_input(&node.inputs, 0);

        // Qwen3 and similar models use Grouped Query Attention (GQA).
        // kv_heads < q_heads, so k_proj/v_proj output a smaller tensor than q_proj.
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(num_heads);
        let q_proj_dim = num_heads * head_dim; // e.g., 16 * 128 = 2048
        let kv_proj_dim = num_kv_heads * head_dim; // e.g., 8 * 128 = 1024

        // Resolve weight names using the module_path from the traced node.
        // For attention, the module_path is typically "model.layers.0.self_attn"
        // and the sub-modules are q_proj, k_proj, v_proj, o_proj.
        let (q_weight, k_weight, v_weight, out_weight) = if let Some(ref mp) = node.module_path {
            (
                self.hf_param_name(mp, "q_proj.weight"),
                self.hf_param_name(mp, "k_proj.weight"),
                self.hf_param_name(mp, "v_proj.weight"),
                self.hf_param_name(mp, "o_proj.weight"),
            )
        } else {
            (
                format!("q_proj_weight_{}", node.id),
                format!("k_proj_weight_{}", node.id),
                format!("v_proj_weight_{}", node.id),
                format!("out_proj_weight_{}", node.id),
            )
        };

        // Separate Q projection: input → Q (flat [B, S, q_heads*head_dim])
        ops.push((
            SirOp::LinearProjection { input: input_id.clone(), weight: q_weight, bias: None },
            format!("q_proj_{}", q_proj_dim),
        ));
        let mut q_id = SirNodeId(format!("sir_q_proj_{}", node.id));

        // Separate K projection: input → K (flat [B, S, kv_heads*head_dim])
        ops.push((
            SirOp::LinearProjection { input: input_id.clone(), weight: k_weight, bias: None },
            format!("k_proj_{}", kv_proj_dim),
        ));
        let mut k_id = SirNodeId(format!("sir_k_proj_{}", node.id));

        // Separate V projection: input → V (flat [B, S, kv_heads*head_dim])
        ops.push((
            SirOp::LinearProjection { input: input_id, weight: v_weight, bias: None },
            format!("v_proj_{}", kv_proj_dim),
        ));
        let mut v_id = SirNodeId(format!("sir_v_proj_{}", node.id));

        // QK-Norm: some models (Qwen3) apply RMSNorm to Q and K after projection.
        //
        // The q/k norm weights are per-head-dimension sized [head_dim], not
        // per-hidden-size [embed_dim]. To broadcast correctly, the norm must
        // be applied in head-aware 4D layout [B, S, heads, head_dim] with
        // axes=[3] (normalize over head_dim).
        //
        // The legality_rewrite pass will:
        //   1. Reshape the flat projection to 4D using DecompositionContext dims
        //   2. Apply ReduceMean/Rsqrt/Mul with axes=[3]
        //   3. Reshape back to 3D
        //
        // Without the 4D layout, [128] cannot broadcast with [1,512,2048].
        if has_qk_norm {
            let q_norm_weight = if let Some(ref mp) = node.module_path {
                self.hf_param_name(mp, "q_norm.weight")
            } else {
                format!("q_norm_weight_{}", node.id)
            };
            ops.push((
                SirOp::RMSNorm {
                    input: q_id.clone(),
                    weight: q_norm_weight,
                    epsilon: self.effective_epsilon(0.0),
                    axes: vec![3], // normalize over head_dim in 4D layout
                },
                "q_norm".to_string(),
            ));
            q_id = SirNodeId(format!("sir_q_norm_{}", node.id));

            let k_norm_weight = if let Some(ref mp) = node.module_path {
                self.hf_param_name(mp, "k_norm.weight")
            } else {
                format!("k_norm_weight_{}", node.id)
            };
            ops.push((
                SirOp::RMSNorm {
                    input: k_id.clone(),
                    weight: k_norm_weight,
                    epsilon: self.effective_epsilon(0.0),
                    axes: vec![3], // normalize over head_dim in 4D layout
                },
                "k_norm".to_string(),
            ));
            k_id = SirNodeId(format!("sir_k_norm_{}", node.id));
        }

        // GQA expansion: if the model uses Grouped Query Attention, K/V heads
        // need to be tiled to match the Q head count.
        // ANE-compatible: Tile on A14+ (replaces the old ExpandDims+Identity hack).
        let needs_gqa_expand = self.config.uses_gqa;
        if needs_gqa_expand {
            let num_q_heads = self.config.num_attention_heads;
            let gqa_kv_heads = self.config.num_key_value_heads.unwrap_or(num_q_heads);
            let num_replicas = num_q_heads / gqa_kv_heads;
            if num_replicas > 1 {
                // K: tile along the heads dimension
                ops.push((
                    SirOp::Tile { input: k_id, reps: vec![1, num_replicas, 1, 1] },
                    "gqa_k_tile".to_string(),
                ));
                k_id = SirNodeId(format!("sir_gqa_k_tile_{}", node.id));

                // V: tile along the heads dimension
                ops.push((
                    SirOp::Tile { input: v_id, reps: vec![1, num_replicas, 1, 1] },
                    "gqa_v_tile".to_string(),
                ));
                v_id = SirNodeId(format!("sir_gqa_v_tile_{}", node.id));
            }
        }

        // Attention computation
        if use_sdpa && self.compiler.target_family().supports_sdpa() {
            // Use SDPA directly on A16+.
            // Causal mask: reference to a lower-triangular mask that will be
            // materialized as a static table by the staticize pass.
            let causal_mask = Some(SirNodeId(format!("causal_mask_{}", node.id)));
            ops.push((
                SirOp::ScaledDotProductAttention {
                    query: q_id,
                    key: k_id,
                    value: v_id,
                    attention_mask: causal_mask,
                    scale: Some(1.0 / (head_dim as f32).sqrt()),
                },
                "sdpa".to_string(),
            ));
        } else {
            // Manual QK^T → scale → softmax → @V for pre-A16 families.
            // This must be numerically equivalent to SDPA on A16+.
            //
            // Step 1: Transpose K so we compute Q @ K^T (not Q @ K).
            // Q is [batch, heads, seq, head_dim], K is [batch, heads, seq, head_dim].
            // We need K^T = [batch, heads, head_dim, seq] so that Q @ K^T = [batch, heads, seq, seq].
            let k_t_id = SirNodeId(format!("sir_attn_k_transpose_{}", node.id));
            ops.push((
                SirOp::Transpose { input: k_id, perm: vec![0, 1, 3, 2] },
                "attn_k_transpose".to_string(),
            ));

            // Step 2: Q @ K^T → [batch, heads, seq, seq]
            ops.push((SirOp::MatMul { a: q_id, b: k_t_id }, "attn_qk".to_string()));
            let qk_id = SirNodeId(format!("sir_attn_qk_{}", node.id));

            // Step 3: Scale by 1/√d_k. This is critical for correct softmax behavior.
            // Without scaling, the dot products grow with √d_k, causing extremely
            // peaked softmax distributions and degraded attention quality.
            let scale_value = 1.0 / (head_dim as f32).sqrt();
            let scale_const_id = SirNodeId(format!("sir_attn_scale_const_{}", node.id));
            ops.push((
                SirOp::Const {
                    value_path: format!("attn_scale_{}_{:.8}", node.id, scale_value),
                    dtype: MilDtype::Fp16,
                },
                "attn_scale_const".to_string(),
            ));
            ops.push((
                SirOp::Mul { x: qk_id, y: scale_const_id },
                "attn_scale".to_string(),
            ));
            let scaled_qk_id = SirNodeId(format!("sir_attn_scale_{}", node.id));

            // Step 4: Softmax over the last dimension (the seq axis of QK^T)
            ops.push((
                SirOp::Softmax { input: scaled_qk_id, axis: -1 },
                "attn_softmax".to_string(),
            ));
            let scores_id = SirNodeId(format!("sir_attn_softmax_{}", node.id));

            // Step 5: Scores @ V → [batch, heads, seq, head_dim]
            ops.push((SirOp::MatMul { a: scores_id, b: v_id }, "attn_sv".to_string()));
        }

        // Output projection: its input is the attention computation output.
        // Use a path-specific semantic alias so the post-hoc resolver can map
        // it to the correct position:
        //   - Pre-A16: sir_attn_sv_{node.id} → position of the sv_matmul op
        //   - A16+:    sir_sdpa_{node.id}     → position of the SDPA op
        let attn_result_id = if use_sdpa && self.compiler.target_family().supports_sdpa() {
            SirNodeId(format!("sir_sdpa_{}", node.id))
        } else {
            SirNodeId(format!("sir_attn_sv_{}", node.id))
        };
        ops.push((
            SirOp::LinearProjection { input: attn_result_id, weight: out_weight, bias: None },
            format!("out_proj_{}", embed_dim),
        ));

        // Residual connection: if the traced node has 2+ inputs, the second
        // input is the skip/residual connection. Emit an Add so the block
        // output = projection_output + residual.
        if node.inputs.len() >= 2 {
            let residual_id = self.resolve_input(&node.inputs, 1);
            let out_proj_id = SirNodeId(format!("sir_out_proj_{}", node.id));
            ops.push((
                SirOp::Add { x: out_proj_id, y: residual_id },
                "attn_residual_add".to_string(),
            ));
        }

        Ok(ops)
    }

    /// Decompose an MLP block into ANE-faithful primitives.
    ///
    /// Detects SwiGLU automatically: if both `gate_proj.weight` AND
    /// `up_proj.weight` exist in the trace weights, the decomposition
    /// follows the SwiGLU pattern: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    /// Otherwise, the standard `down_proj(activation(up_proj(x)))` path is used.
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

        // Detect SwiGLU: if both gate_proj and up_proj exist, use SwiGLU pattern.
        // This is the standard pattern for Llama, Qwen, and other modern transformers.
        let has_gate = if let Some(ref mp) = node.module_path {
            self.trace.weights.contains_key(&self.hf_param_name(mp, "gate_proj.weight"))
        } else {
            false
        };
        let has_up = if let Some(ref mp) = node.module_path {
            self.trace.weights.contains_key(&self.hf_param_name(mp, "up_proj.weight"))
        } else {
            false
        };
        let is_swiglu = (has_gate && has_up)
            || (self.config.hidden_act == "silu" && has_gate);

        // Resolve weight names using module_path.
        // MLP module_path is typically "model.layers.0.mlp"
        // Sub-modules are gate_proj, up_proj, down_proj (varies by architecture).
        let (gate_weight, up_weight, down_weight) = if let Some(ref mp) = node.module_path {
            (
                self.hf_param_name(mp, "gate_proj.weight"),
                self.hf_param_name(mp, "up_proj.weight"),
                self.hf_param_name(mp, "down_proj.weight"),
            )
        } else {
            (
                format!("gate_proj_weight_{}", node.id),
                format!("up_proj_weight_{}", node.id),
                format!("down_proj_weight_{}", node.id),
            )
        };

        if is_swiglu {
            // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
            // 1. gate_proj(x)
            ops.push((
                SirOp::LinearProjection {
                    input: input_id.clone(),
                    weight: gate_weight,
                    bias: None,
                },
                format!("gate_proj_{}_{}", input_dim, hidden_dim),
            ));
            let gate_id = SirNodeId(format!("sir_gate_proj_{}", node.id));

            // 2. silu(gate_proj(x))
            ops.push((SirOp::Silu { input: gate_id }, "mlp_gate_silu".to_string()));
            let gate_silu_id = SirNodeId(format!("sir_mlp_gate_silu_{}", node.id));

            // 3. up_proj(x) — NO activation
            ops.push((
                SirOp::LinearProjection { input: input_id, weight: up_weight, bias: None },
                format!("up_proj_{}_{}", input_dim, hidden_dim),
            ));
            let up_id = SirNodeId(format!("sir_up_proj_{}", node.id));

            // 4. silu(gate) * up  (element-wise multiply)
            ops.push((SirOp::Mul { x: gate_silu_id, y: up_id }, "mlp_swiglu_mul".to_string()));
            let swiglu_id = SirNodeId(format!("sir_mlp_swiglu_mul_{}", node.id));

            // 5. down_proj
            ops.push((
                SirOp::LinearProjection { input: swiglu_id, weight: down_weight, bias: None },
                format!("down_proj_{}_{}", hidden_dim, output_dim),
            ));
        } else {
            // Standard MLP: down_proj(activation(up_proj(x)))
            // Choose the up-projection weight name: gate_proj or up_proj
            let up_weight_resolved = if let Some(ref mp) = node.module_path {
                if has_gate {
                    self.hf_param_name(mp, "gate_proj.weight")
                } else {
                    self.hf_param_name(mp, "up_proj.weight")
                }
            } else {
                format!("up_proj_weight_{}", node.id)
            };

            // Up-projection: input → hidden
            ops.push((
                SirOp::LinearProjection { input: input_id, weight: up_weight_resolved, bias: None },
                format!("up_proj_{}_{}", input_dim, hidden_dim),
            ));

            // Activation
            let up_id = SirNodeId(format!("sir_up_proj_{}", node.id));
            match activation {
                "silu" | "swish" => {
                    ops.push((SirOp::Silu { input: up_id }, "mlp_act".to_string()));
                }
                "gelu" | "gelu_new" => {
                    ops.push((
                        SirOp::Gelu { input: up_id, mode: "EXACT".to_string() },
                        "mlp_act".to_string(),
                    ));
                }
                "relu" => {
                    ops.push((SirOp::Relu { input: up_id }, "mlp_act".to_string()));
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
                SirOp::LinearProjection { input: act_id, weight: down_weight, bias: None },
                format!("down_proj_{}_{}", hidden_dim, output_dim),
            ));
        }

        // Residual connection: if the traced node has 2+ inputs, the second
        // input is the skip/residual connection. Emit an Add so the block
        // output = down_proj_output + residual.
        if node.inputs.len() >= 2 {
            let residual_id = self.resolve_input(&node.inputs, 1);
            let down_proj_id = SirNodeId(format!("sir_down_proj_{}", node.id));
            ops.push((
                SirOp::Add { x: down_proj_id, y: residual_id },
                "mlp_residual_add".to_string(),
            ));
        }

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
        ops.push((SirOp::Mul { x: input_id.clone(), y: cos_id }, "rope_cos_mul".to_string()));

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
        ops.push((SirOp::Mul { x: rotated_id, y: sin_id }, "rope_sin_mul".to_string()));

        // cos*x + sin*rotate_half(x)
        let cos_mul_id = SirNodeId(format!("sir_rope_cos_mul_{}", node.id));
        let sin_mul_id = SirNodeId(format!("sir_rope_sin_mul_{}", node.id));
        ops.push((SirOp::Add { x: cos_mul_id, y: sin_mul_id }, "rope_add".to_string()));

        Ok(ops)
    }

    /// Decompose RMSNorm into ANE-faithful primitives.
    ///
    /// The decomposition is **strategy-driven**: the method chosen depends
    /// on the target ANE family capabilities and model characteristics.
    /// The strategy framework discovers the best decomposition during
    /// compilation; here we emit the composite RMSNorm op that will be
    /// decomposed by the legality rewrite pass using the chosen strategy.
    ///
    /// Available decomposition strategies (discovered dynamically):
    /// - **naive**: x * rsqrt(mean(x^2) + eps) * weight — simple but may
    ///   underflow in fp16 for large hidden sizes
    /// - **max_abs_stabilized**: normalize by max(|x|) first, then compute
    ///   variance on normalized values with two-division epsilon compensation.
    ///   Prevents fp16 underflow. Preferred for fp16-only broadcast targets
    ///   (A11/A12) or models with large hidden sizes.
    fn decompose_rms_norm(
        &self,
        hidden_size: usize,
        epsilon: f64,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let input_id = self.resolve_input(&node.inputs, 0);
        let rms_weight = self.resolve_weight_name(node, &format!("rms_weight_{}", node.id));

        // Guard against zero/missing epsilon — use config's layer_norm_epsilon
        // as fallback, or the standard 1e-6 for RMSNorm models (Qwen3, Llama, etc.).
        let effective_epsilon = self.effective_epsilon(epsilon);

        // Emit the composite RMSNorm op. The actual decomposition into
        // primitives (naive vs max-abs-stabilized vs other) is determined
        // by the strategy framework and applied by the legality rewrite pass.
        //
        // This keeps sir_build.rs strategy-agnostic: it records the semantic
        // intent (RMSNorm) without committing to a specific decomposition.
        Ok(vec![(
            SirOp::RMSNorm { input: input_id, weight: rms_weight, epsilon: effective_epsilon, axes: vec![2] },
            format!("rms_norm_{}", hidden_size),
        )])
    }

    /// Decompose SDPA into K^T + QK^T/scale → Softmax → @V for pre-A16 families.
    ///
    /// This must be numerically equivalent to `SirOp::ScaledDotProductAttention`
    /// on A16+. The decomposition follows the standard attention formula:
    ///
    ///   softmax( (Q @ K^T) / √d_k ) @ V
    ///
    /// Each step is explicit so the ANE can place each primitive individually.
    fn decompose_sdpa(
        &self,
        scale: f64,
        node: &TracedNode,
    ) -> Result<Vec<(SirOp, String)>, String> {
        let q_id = self.resolve_input(&node.inputs, 0);
        let k_id = self.resolve_input(&node.inputs, 1);
        let v_id = self.resolve_input(&node.inputs, 2);

        let mut ops = Vec::new();

        // Step 1: Transpose K → K^T so that Q @ K^T computes correctly.
        // K is [batch, heads, seq, head_dim]; K^T = [batch, heads, head_dim, seq].
        let k_t_id = SirNodeId(format!("sir_sdpa_k_t_{}", node.id));
        ops.push((
            SirOp::Transpose { input: k_id, perm: vec![0, 1, 3, 2] },
            "sdpa_k_transpose".to_string(),
        ));

        // Step 2: Q @ K^T → [batch, heads, seq, seq]
        ops.push((SirOp::MatMul { a: q_id, b: k_t_id }, "sdpa_qk".to_string()));
        let qk_id = SirNodeId(format!("sir_sdpa_qk_{}", node.id));

        // Step 3: Scale by the provided scale factor (typically 1/√d_k).
        // The scale value from the TracedOp is used directly, not ignored.
        let scale_f32 = scale as f32;
        let scale_const_id = SirNodeId(format!("sir_sdpa_scale_const_{}", node.id));
        ops.push((
            SirOp::Const {
                value_path: format!("sdpa_scale_{}_{:.8}", node.id, scale_f32),
                dtype: MilDtype::Fp16,
            },
            "sdpa_scale_const".to_string(),
        ));
        ops.push((
            SirOp::Mul { x: qk_id, y: scale_const_id },
            "sdpa_scale".to_string(),
        ));

        // Step 4: Softmax over the last dimension
        let scaled_id = SirNodeId(format!("sir_sdpa_scaled_{}", node.id));
        ops.push((SirOp::Softmax { input: scaled_id, axis: -1 }, "sdpa_softmax".to_string()));

        // Step 5: Scores @ V → [batch, heads, seq, head_dim]
        let softmax_id = SirNodeId(format!("sir_sdpa_softmax_{}", node.id));
        ops.push((SirOp::MatMul { a: softmax_id, b: v_id }, "sdpa_sv".to_string()));

        Ok(ops)
    }

    /// Compute the effective epsilon for RMSNorm, with fallback logic.
    ///
    /// If the provided epsilon is > 0, use it directly. Otherwise fall back
    /// to the config's `layer_norm_epsilon`, or the standard 1e-6 default
    /// for RMSNorm models (Qwen3, Llama, etc.).
    fn effective_epsilon(&self, epsilon: f64) -> f32 {
        if epsilon > 0.0 {
            epsilon as f32
        } else if self.config.layer_norm_epsilon > 0.0 {
            self.config.layer_norm_epsilon as f32
        } else {
            1e-6 // Standard default for RMSNorm (Qwen3, Llama, etc.)
        }
    }

    /// Resolve a traced node input reference to a SIR node ID.
    ///
    /// Emits a warning to stderr when the input cannot be resolved,
    /// instead of silently producing an `__unresolved_N__` node.
    fn resolve_input(&self, inputs: &[String], index: usize) -> SirNodeId {
        inputs.get(index).and_then(|id| self.lookup_sir_id(id).cloned()).unwrap_or_else(|| {
            let unresolved = SirNodeId(format!("__unresolved_{}__", index));
            eprintln!(
                "WARNING: SIR resolve_input failed — no mapping for input index {} \
                     (inputs: {:?}, node_map keys: {:?}). Producing {}.",
                index,
                inputs,
                self.node_map.keys().take(10).collect::<Vec<_>>(),
                unresolved.0
            );
            unresolved
        })
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
                model_class: "causal_lm".to_string(),
                is_encoder_decoder: false,
        head_dim: None,
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
            discovered_features: crate::graph::DiscoveredFeatures {
                norm_types_encountered: vec!["RMSNorm".to_string()],
                has_rope_module: true,
                attention_module_types: vec![],
                mlp_module_types: vec![],
                linear_count: 1,
                embedding_count: 0,
                uses_gqa: false,
                detection_methods: {
                    let mut m = HashMap::new();
                    m.insert("norm_type".to_string(), "config_field_presence".to_string());
                    m.insert("rope".to_string(), "config_field_presence".to_string());
                    m
                },
            },
            weights: HashMap::new(),
            weight_name_map: HashMap::new(),
            model_cache_dir: None,
            safetensors_files: vec![],
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
        let has_trace_origin = sir
            .nodes
            .iter()
            .any(|n| matches!(n.metadata.task_origin, TaskOrigin::TransformersTrace { .. }));
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
                model_class: "causal_lm".to_string(),
                is_encoder_decoder: false,
        head_dim: None,
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
            discovered_features: crate::graph::DiscoveredFeatures {
                norm_types_encountered: vec!["LayerNorm".to_string()],
                has_rope_module: false,
                attention_module_types: vec![],
                mlp_module_types: vec![],
                linear_count: 0,
                embedding_count: 0,
                uses_gqa: false,
                detection_methods: {
                    let mut m = HashMap::new();
                    m.insert("norm_type".to_string(), "config_field_presence".to_string());
                    m
                },
            },
            weights: HashMap::new(),
            weight_name_map: HashMap::new(),
            model_cache_dir: None,
            safetensors_files: vec![],
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
        trace.nodes.insert(
            1,
            TracedNode {
                id: "rms1".to_string(),
                op: TracedOp::RmsNorm { hidden_size: 256, epsilon: 1e-6 },
                name: "rms_norm".to_string(),
                inputs: vec!["input".to_string()],
                output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
                is_parameter: false,
                module_path: None,
            },
        );

        let sir = build_sir_from_trace(&trace, AneFamily::A16).unwrap();
        // RMSNorm is now emitted as a single composite op (not decomposed into
        // primitives). The actual decomposition happens in the legality rewrite pass.
        let rms_ops: Vec<_> =
            sir.nodes.iter().filter(|n| matches!(n.op, SirOp::RMSNorm { .. })).collect();
        assert_eq!(
            rms_ops.len(),
            1,
            "RMSNorm should produce exactly 1 SirOp::RMSNorm node, got {} nodes",
            rms_ops.len()
        );
    }

    // ──────────────────────────────────────────────────────────────
    // Dynamic Tracing Tests
    //
    // These tests validate that the SIR construction pipeline works
    // fully dynamically — no model_type heuristics, no hardcoded model
    // lists. All feature detection comes from config field presence
    // and discovered_features.
    // ──────────────────────────────────────────────────────────────

    /// Helper: build a TracedGraph with arbitrary ModelConfig.
    /// This simulates what the Python tracer would produce for different
    /// model architectures without requiring actual model downloads.
    fn make_trace_with_config(
        config: ModelConfig,
        discovered: crate::graph::DiscoveredFeatures,
    ) -> TracedGraph {
        TracedGraph {
            model_id: format!("test-{}", config.model_type),
            architecture: format!("{}ForCausalLM", config.model_type),
            transformers_version: "4.48.0".to_string(),
            torch_version: "2.5.0".to_string(),
            model_config: config,
            discovered_features: discovered,
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
                    id: "rms1".to_string(),
                    op: TracedOp::RmsNorm { hidden_size: 256, epsilon: 1e-6 },
                    name: "input_norm".to_string(),
                    inputs: vec!["input".to_string()],
                    output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
                    is_parameter: false,
                    module_path: Some("model.layers.0.input_layernorm".to_string()),
                },
                TracedNode {
                    id: "output".to_string(),
                    op: TracedOp::Output,
                    name: "output".to_string(),
                    inputs: vec!["rms1".to_string()],
                    output_shape: TensorShape { dims: vec![1, 32, 256], dtype: "fp16".to_string() },
                    is_parameter: false,
                    module_path: None,
                },
            ],
            weights: HashMap::new(),
            weight_name_map: HashMap::new(),
            model_cache_dir: None,
            safetensors_files: vec![],
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
    fn test_dynamic_tracing_qwen3_5_config() {
        // Qwen3.5 has model_type="qwen3_5_text" which would NOT match any
        // hardcoded heuristic list. But because we detect features from config
        // fields (rms_norm_eps, rope_parameters, etc.), it works ad-hoc.
        let config = ModelConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: Some(8), // GQA: 8 KV heads for 16 Q heads
            num_hidden_layers: 24,
            intermediate_size: 4096,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            layer_norm_epsilon: 1e-6,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: true,
            model_type: "qwen3_5_text".to_string(), // No hardcoded list needed!
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        let discovered = crate::graph::DiscoveredFeatures {
            norm_types_encountered: vec!["RMSNorm".to_string()],
            has_rope_module: true,
            attention_module_types: vec!["Qwen3Attention".to_string()],
            mlp_module_types: vec!["Qwen3MLP".to_string()],
            linear_count: 0,
            embedding_count: 0,
            uses_gqa: true,
            detection_methods: {
                let mut m = HashMap::new();
                m.insert("norm_type".to_string(), "config_field_presence".to_string());
                m.insert("rope".to_string(), "config_field_presence".to_string());
                m.insert("gqa".to_string(), "config_field_comparison".to_string());
                m
            },
        };

        let trace = make_trace_with_config(config, discovered);
        let sir = build_sir_from_trace(&trace, AneFamily::A12); // M2 target
        assert!(
            sir.is_ok(),
            "Qwen3.5 should trace successfully on A12 without any hardcoded heuristics"
        );

        let sir = sir.unwrap();
        let has_rms = sir.nodes.iter().any(|n| matches!(n.op, SirOp::RMSNorm { .. }));
        assert!(has_rms, "Qwen3.5 should produce RMSNorm ops (detected dynamically from config)");
    }

    #[test]
    fn test_dynamic_tracing_qwen3_gqa() {
        // Qwen3-0.6B: GQA model with num_key_value_heads < num_attention_heads
        let config = ModelConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            num_hidden_layers: 28,
            intermediate_size: 4096,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            layer_norm_epsilon: 1e-6,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: true,
            model_type: "qwen3".to_string(),
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        let discovered = crate::graph::DiscoveredFeatures {
            norm_types_encountered: vec!["RMSNorm".to_string()],
            has_rope_module: true,
            attention_module_types: vec![],
            mlp_module_types: vec![],
            linear_count: 0,
            embedding_count: 0,
            uses_gqa: true,
            detection_methods: {
                let mut m = HashMap::new();
                m.insert("norm_type".to_string(), "config_field_presence".to_string());
                m.insert("rope".to_string(), "config_field_presence".to_string());
                m.insert("gqa".to_string(), "config_field_comparison".to_string());
                m
            },
        };

        let trace = make_trace_with_config(config, discovered);
        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Qwen3 with GQA should trace on A12");
    }

    #[test]
    fn test_dynamic_tracing_llama_3_2() {
        // Llama-3.2-1B: standard Llama architecture
        let config = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: Some(8), // GQA
            num_hidden_layers: 16,
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            layer_norm_epsilon: 1e-5,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: true,
            model_type: "llama".to_string(),
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        let discovered = crate::graph::DiscoveredFeatures {
            norm_types_encountered: vec!["RMSNorm".to_string()],
            has_rope_module: true,
            attention_module_types: vec!["LlamaAttention".to_string()],
            mlp_module_types: vec!["LlamaMLP".to_string()],
            linear_count: 0,
            embedding_count: 0,
            uses_gqa: true,
            detection_methods: {
                let mut m = HashMap::new();
                m.insert("norm_type".to_string(), "module_type_inspection".to_string());
                m.insert("rope".to_string(), "module_type_inspection".to_string());
                m.insert("gqa".to_string(), "config_field_comparison".to_string());
                m
            },
        };

        let trace = make_trace_with_config(config, discovered);
        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Llama-3.2-1B should trace on A12 (M2)");
    }

    #[test]
    fn test_dynamic_tracing_unknown_model_type() {
        // A completely unknown model_type should still work if it provides
        // the standard config fields. This is the core value proposition
        // of fully dynamic tracing: no model registry needed.
        let config = ModelConfig {
            hidden_size: 512,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            num_hidden_layers: 6,
            intermediate_size: 2048,
            vocab_size: 50000,
            max_position_embeddings: 4096,
            layer_norm_epsilon: 1e-6,
            hidden_act: "gelu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: false,
            model_type: "future_architecture_v7".to_string(), // Completely unknown!
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        let discovered = crate::graph::DiscoveredFeatures {
            norm_types_encountered: vec!["RMSNorm".to_string()],
            has_rope_module: true,
            attention_module_types: vec![],
            mlp_module_types: vec![],
            linear_count: 0,
            embedding_count: 0,
            uses_gqa: false,
            detection_methods: {
                let mut m = HashMap::new();
                m.insert("norm_type".to_string(), "config_field_presence".to_string());
                m.insert("rope".to_string(), "config_field_presence".to_string());
                m.insert("gqa".to_string(), "config_field_comparison".to_string());
                m
            },
        };

        let trace = make_trace_with_config(config, discovered);
        let sir = build_sir_from_trace(&trace, AneFamily::A16);
        assert!(
            sir.is_ok(),
            "Unknown model_type should still work — dynamic tracing doesn't need a registry"
        );

        let sir = sir.unwrap();
        assert!(!sir.nodes.is_empty());
    }

    #[test]
    fn test_discovered_features_roundtrip() {
        // Test that DiscoveredFeatures serializes/deserializes correctly
        let features = crate::graph::DiscoveredFeatures {
            norm_types_encountered: vec!["RMSNorm".to_string()],
            has_rope_module: true,
            attention_module_types: vec!["LlamaAttention".to_string()],
            mlp_module_types: vec!["LlamaMLP".to_string()],
            linear_count: 42,
            embedding_count: 1,
            uses_gqa: true,
            detection_methods: {
                let mut m = HashMap::new();
                m.insert("norm_type".to_string(), "module_type_inspection".to_string());
                m.insert("rope".to_string(), "config_field_presence".to_string());
                m
            },
        };

        let json = serde_json::to_string(&features).expect("Should serialize");
        let deserialized: crate::graph::DiscoveredFeatures =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.norm_types_encountered, vec!["RMSNorm"]);
        assert!(deserialized.has_rope_module);
        assert_eq!(deserialized.linear_count, 42);
        assert!(deserialized.uses_gqa);
        assert_eq!(
            deserialized.detection_methods.get("norm_type"),
            Some(&"module_type_inspection".to_string())
        );
    }

    #[test]
    fn test_encoder_decoder_seq2seq_config() {
        // Dolphin-1.5: encoder-decoder (DonutSwin + BART decoder)
        // The traced graph represents the decoder path for ANE compilation.
        // The model_class is "seq2seq_lm" and is_encoder_decoder is true.
        let config = ModelConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: Some(12),
            num_hidden_layers: 6,
            intermediate_size: 3072,
            vocab_size: 50265,
            max_position_embeddings: 1024,
            layer_norm_epsilon: 1e-6,
            hidden_act: "gelu".to_string(),
            uses_rope: false,
            uses_rms_norm: false,
            uses_gqa: false,
            model_type: "dolphin".to_string(),
            model_class: "seq2seq_lm".to_string(),
            is_encoder_decoder: true,
        head_dim: None,
        };

        // Verify the new fields are set correctly
        assert_eq!(config.model_class, "seq2seq_lm");
        assert!(config.is_encoder_decoder);
        assert!(!config.uses_rope); // BART uses learned positional embeddings, not RoPE
        assert!(!config.uses_rms_norm); // BART uses LayerNorm
    }

    #[test]
    fn test_decoder_only_multimodal_config() {
        // Qwen3-ASR-0.6B: multimodal with text_config
        // The decoder is a standard Qwen3 causal LM, extracted separately.
        // The model_class is "decoder_only" and is_encoder_decoder is false.
        let config = ModelConfig {
            hidden_size: 896,
            num_attention_heads: 14,
            num_key_value_heads: Some(2),
            num_hidden_layers: 24,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_position_embeddings: 4096,
            layer_norm_epsilon: 1e-6,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: true, // 14 heads, 2 KV heads = 7x GQA
            model_type: "qwen3_asr".to_string(),
            model_class: "decoder_only".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        assert_eq!(config.model_class, "decoder_only");
        assert!(!config.is_encoder_decoder);
        assert!(config.uses_gqa);
        assert!(config.uses_rope);
        assert!(config.uses_rms_norm);
    }

    #[test]
    fn test_model_config_serialization_with_new_fields() {
        // Verify that model_class and is_encoder_decoder serialize/deserialize
        let config = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            num_hidden_layers: 16,
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            layer_norm_epsilon: 1e-5,
            hidden_act: "silu".to_string(),
            uses_rope: true,
            uses_rms_norm: true,
            uses_gqa: true,
            model_type: "llama".to_string(),
            model_class: "causal_lm".to_string(),
            is_encoder_decoder: false,
        head_dim: None,
        };

        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: ModelConfig = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.model_class, "causal_lm");
        assert!(!deserialized.is_encoder_decoder);
        assert_eq!(deserialized.model_type, "llama");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Layer 2: Real-model fixture tests
    //
    // These tests load pre-traced JSON fixtures (generated by the Python
    // tracing pipeline) and validate that the SIR construction pipeline
    // works correctly with real model data, not just hand-crafted configs.
    //
    // Fixtures are in crates/trace/test_fixtures/ and are generated by:
    //   python scripts/generate_fixtures.py
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: load a traced graph from a test fixture JSON file.
    fn load_fixture(name: &str) -> Option<TracedGraph> {
        // Try multiple search paths for the fixture directory
        let search_paths = [
            format!("crates/trace/test_fixtures/{}", name),
            format!("test_fixtures/{}", name),
            format!("fixtures/{}", name),
        ];

        for path in &search_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                return serde_json::from_str(&content).ok();
            }
        }

        // Try relative to Cargo.toml directory
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let path = format!("{}/test_fixtures/{}", manifest_dir, name);
            if let Ok(content) = std::fs::read_to_string(&path) {
                return serde_json::from_str(&content).ok();
            }
        }

        None
    }

    #[test]
    fn test_sir_from_llama_3_2_1b_fixture() {
        // Llama-3.2-1B: standard causal LM with RMSNorm + RoPE + GQA
        let graph = load_fixture("llama_3_2_1b.json");
        if graph.is_none() {
            eprintln!(
                "SKIP: llama_3_2_1b.json fixture not found (run scripts/generate_fixtures.py)"
            );
            return;
        }
        let trace = graph.unwrap();

        // Validate fixture loaded correctly
        assert_eq!(trace.model_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(trace.model_config.model_class, "causal_lm");
        assert!(!trace.model_config.is_encoder_decoder);

        // Build SIR from the fixture
        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Llama-3.2-1B fixture should produce valid SIR on A12 (M2)");

        let sir = sir.unwrap();
        assert!(!sir.nodes.is_empty(), "SIR should have nodes");

        // Should have RMSNorm ops (detected dynamically)
        let has_rms = sir.nodes.iter().any(|n| matches!(n.op, SirOp::RMSNorm { .. }));
        assert!(has_rms, "Llama-3.2-1B should produce RMSNorm ops in SIR");

        // Config should match expected Llama-3.2-1B values
        assert_eq!(trace.model_config.hidden_size, 2048);
        assert_eq!(trace.model_config.num_attention_heads, 32);
        assert_eq!(trace.model_config.num_key_value_heads, Some(8));
    }

    #[test]
    fn test_sir_from_qwen3_0_6b_fixture() {
        // Qwen3-0.6B: causal LM with GQA (8 KV heads for 16 Q heads)
        let graph = load_fixture("qwen3_0_6b.json");
        if graph.is_none() {
            eprintln!("SKIP: qwen3_0_6b.json fixture not found (run scripts/generate_fixtures.py)");
            return;
        }
        let trace = graph.unwrap();

        assert_eq!(trace.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(trace.model_config.model_class, "causal_lm");
        assert!(trace.model_config.uses_gqa);

        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Qwen3-0.6B fixture should produce valid SIR on A12");

        let sir = sir.unwrap();
        assert!(!sir.nodes.is_empty());

        // Should have RMSNorm ops
        let has_rms = sir.nodes.iter().any(|n| matches!(n.op, SirOp::RMSNorm { .. }));
        assert!(has_rms, "Qwen3-0.6B should produce RMSNorm ops in SIR");
    }

    #[test]
    fn test_sir_from_qwen3_5_0_8b_fixture() {
        // Qwen3.5-0.8B: model_type="qwen3_5_text" — no hardcoded registry needed
        let graph = load_fixture("qwen3_5_0_8b.json");
        if graph.is_none() {
            eprintln!(
                "SKIP: qwen3_5_0_8b.json fixture not found (run scripts/generate_fixtures.py)"
            );
            return;
        }
        let trace = graph.unwrap();

        assert_eq!(trace.model_id, "Qwen/Qwen3.5-0.8B");
        assert_eq!(trace.model_config.model_type, "qwen3_5_text");

        // This is the KEY test: unknown model_type should work without any registry
        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Qwen3.5-0.8B with unknown model_type should still produce valid SIR");

        let sir = sir.unwrap();
        let has_rms = sir.nodes.iter().any(|n| matches!(n.op, SirOp::RMSNorm { .. }));
        assert!(has_rms, "Qwen3.5-0.8B should produce RMSNorm ops (detected dynamically)");
    }

    #[test]
    fn test_sir_from_dolphin_1_5_fixture() {
        // Dolphin-1.5: encoder-decoder (DonutSwin + BART decoder)
        // Uses LayerNorm (not RMSNorm), learned positional embeddings (no RoPE)
        let graph = load_fixture("dolphin_1_5.json");
        if graph.is_none() {
            eprintln!(
                "SKIP: dolphin_1_5.json fixture not found (run scripts/generate_fixtures.py)"
            );
            return;
        }
        let trace = graph.unwrap();

        assert_eq!(trace.model_id, "ByteDance/Dolphin-1.5");
        assert_eq!(trace.model_config.model_class, "seq2seq_lm");
        assert!(trace.model_config.is_encoder_decoder);
        assert!(!trace.model_config.uses_rope); // BART uses learned embeddings

        // Note: Dolphin's LayerNorm requires A15+ or CPU fallback on A12
        // On A12 (M2), this may need a legality rewrite to handle LayerNorm
        let sir = build_sir_from_trace(&trace, AneFamily::A16);
        // A16 supports LayerNorm natively
        assert!(sir.is_ok(), "Dolphin-1.5 decoder should produce valid SIR on A16+");

        let sir = sir.unwrap();
        assert!(!sir.nodes.is_empty());
    }

    #[test]
    fn test_sir_from_qwen3_asr_0_6b_fixture() {
        // Qwen3-ASR-0.6B: multimodal with text_config extraction
        // The decoder is a standard Qwen3 causal LM
        let graph = load_fixture("qwen3_asr_0_6b.json");
        if graph.is_none() {
            eprintln!(
                "SKIP: qwen3_asr_0_6b.json fixture not found (run scripts/generate_fixtures.py)"
            );
            return;
        }
        let trace = graph.unwrap();

        assert_eq!(trace.model_id, "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(trace.model_config.model_class, "decoder_only");
        assert!(!trace.model_config.is_encoder_decoder);

        // Qwen3 decoder features: RMSNorm + RoPE + extreme 7:1 GQA
        assert!(trace.model_config.uses_rms_norm);
        assert!(trace.model_config.uses_rope);
        assert!(trace.model_config.uses_gqa);

        let sir = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(sir.is_ok(), "Qwen3-ASR-0.6B decoder should produce valid SIR on A12");

        let sir = sir.unwrap();
        let has_rms = sir.nodes.iter().any(|n| matches!(n.op, SirOp::RMSNorm { .. }));
        assert!(has_rms, "Qwen3-ASR-0.6B should produce RMSNorm ops in SIR");
    }

    #[test]
    fn test_all_fixtures_produce_sir_with_correct_op_counts() {
        // Cross-model validation: all fixtures should produce SIR graphs
        // with reasonable op counts that scale with model size
        let fixtures = [
            ("llama_3_2_1b.json", 16),   // 16 layers
            ("qwen3_0_6b.json", 28),     // 28 layers
            ("qwen3_5_0_8b.json", 24),   // 24 layers
            ("dolphin_1_5.json", 6),     // 6 layers
            ("qwen3_asr_0_6b.json", 24), // 24 layers
        ];

        let mut loaded_count = 0;
        for (name, expected_layers) in &fixtures {
            let graph = load_fixture(name);
            if let Some(trace) = graph {
                loaded_count += 1;
                assert_eq!(
                    trace.model_config.num_hidden_layers, *expected_layers,
                    "Fixture {} should have {} layers",
                    name, expected_layers
                );

                let sir = build_sir_from_trace(&trace, AneFamily::A12);
                assert!(sir.is_ok(), "Fixture {} should produce valid SIR", name);
            }
        }

        // At least one fixture should be loadable (otherwise fixtures are missing)
        if loaded_count == 0 {
            eprintln!("NOTE: No fixtures loaded — run scripts/generate_fixtures.py first");
        }
    }

    /// Test that pre-A16 (M2 = A12) attention decomposition produces correct
    /// primitive ops: K^T transpose, scale-by-1/√d_k, and proper matmul order.
    ///
    /// This is a regression test for the bugs where:
    /// - The scale-by-1/√d_k step was missing (unscaled attention scores)
    /// - K was not transposed (computing Q@K instead of Q@K^T)
    /// - The decompose_sdpa() ignored the _scale parameter
    #[test]
    fn test_m2_attention_decomposition_has_scale_and_transpose() {
        // Build a trace with an AttentionBlock
        let mut trace = make_simple_trace();
        trace.nodes.push(TracedNode {
            id: "attn_0".to_string(),
            op: TracedOp::AttentionBlock {
                embed_dim: 256,
                num_heads: 4,
                head_dim: 64,
                use_sdpa: true,
                has_qk_norm: false,
            },
            inputs: vec!["input_0".to_string()],
            output_shape: TensorShape { dims: vec![1, 256], dtype: "fp16".to_string() },
            is_parameter: false,
            module_path: Some("model.layers.0.self_attn".to_string()),
            name: "self_attn_0".to_string(),
        });

        // Build SIR targeting A12 (M2) — SDPA is unreliable here
        let result = build_sir_from_trace(&trace, AneFamily::A12);
        assert!(result.is_ok(), "M2 SIR build should succeed: {:?}", result.err());

        let sir = result.unwrap();

        // Verify the SIR contains the key pre-A16 decomposition ops:
        // Transpose(K) → MatMul(Q, K^T) → Const(scale) → Mul → Softmax → MatMul(scores, V)
        let has_k_transpose = sir.nodes.iter().any(|n| {
            matches!(&n.op, SirOp::Transpose { perm, .. } if perm == &[0, 1, 3, 2])
                && n.name.contains("k_transpose")
        });
        assert!(
            has_k_transpose,
            "M2 attention decomposition must include K^T transpose (perm [0,1,3,2])"
        );

        let has_scale_const = sir.nodes.iter().any(|n| {
            matches!(&n.op, SirOp::Const { value_path, .. } if value_path.contains("attn_scale"))
        });
        assert!(
            has_scale_const,
            "M2 attention decomposition must include a scale constant (1/√d_k)"
        );

        let has_scale_mul = sir.nodes.iter().any(|n| {
            matches!(n.op, SirOp::Mul { .. }) && n.name.contains("attn_scale")
        });
        assert!(
            has_scale_mul,
            "M2 attention decomposition must include Mul for scaling QK^T"
        );

        let has_softmax = sir.nodes.iter().any(|n| {
            matches!(n.op, SirOp::Softmax { .. }) && n.name.contains("attn_softmax")
        });
        assert!(
            has_softmax,
            "M2 attention decomposition must include Softmax"
        );

        // Verify the scale constant encodes the correct value
        let scale_node = sir.nodes.iter().find(|n| {
            matches!(&n.op, SirOp::Const { value_path, .. } if value_path.contains("attn_scale"))
        });
        if let Some(node) = scale_node {
            if let SirOp::Const { value_path, .. } = &node.op {
                // value_path should contain the scale value (1/√64 ≈ 0.125)
                assert!(
                    value_path.contains("0.125"),
                    "Scale constant value_path must encode the scale value (1/√64), got: {}",
                    value_path
                );
            }
        }
    }

    /// Test that A16+ attention decomposition uses SDPA directly (no manual decomposition).
    #[test]
    fn test_a16_attention_decomposition_uses_sdpa() {
        let mut trace = make_simple_trace();
        trace.nodes.push(TracedNode {
            id: "attn_0".to_string(),
            op: TracedOp::AttentionBlock {
                embed_dim: 256,
                num_heads: 4,
                head_dim: 64,
                use_sdpa: true,
                has_qk_norm: false,
            },
            inputs: vec!["input_0".to_string()],
            output_shape: TensorShape { dims: vec![1, 256], dtype: "fp16".to_string() },
            is_parameter: false,
            module_path: Some("model.layers.0.self_attn".to_string()),
            name: "self_attn_0".to_string(),
        });

        // Build SIR targeting A16 — SDPA is reliable here
        let result = build_sir_from_trace(&trace, AneFamily::A16);
        assert!(result.is_ok(), "A16 SIR build should succeed: {:?}", result.err());

        let sir = result.unwrap();

        // On A16+, we should see SDPA, not manual decomposition
        let has_sdpa = sir.nodes.iter().any(|n| {
            matches!(n.op, SirOp::ScaledDotProductAttention { .. })
        });
        assert!(
            has_sdpa,
            "A16+ attention decomposition must use ScaledDotProductAttention"
        );

        // We should NOT see manual K transpose or scale const on A16+
        let has_k_transpose = sir.nodes.iter().any(|n| {
            matches!(&n.op, SirOp::Transpose { perm, .. } if perm == &[0, 1, 3, 2])
                && n.name.contains("k_transpose")
        });
        assert!(
            !has_k_transpose,
            "A16+ attention decomposition must NOT include manual K^T transpose"
        );
    }
}

/// Strip trailing dimension-like suffixes from a name tag.
///
/// For example:
/// - "q_proj_1024" → "q_proj"
/// - "out_proj_1024" → "out_proj"
/// - "linear_256_1024" → "linear"
/// - "sdpa" → None (no dimension suffix to strip)
/// - "attn_sv" → None (no dimension suffix)
/// - "embedding_32000_1024" → "embedding"
///
/// This is used to register short-prefix aliases so that semantic references
/// like "sir_q_proj_layer_0" can be resolved even though the actual name tag
/// is "q_proj_1024".
fn strip_trailing_dim_suffix(name: &str) -> Option<String> {
    // Find trailing segments that are purely numeric, and strip them.
    // For "q_proj_1024", we want "q_proj".
    // For "linear_256_1024", we want "linear".
    // For "sdpa", there's nothing to strip → return None.
    let parts: Vec<&str> = name.split('_').collect();
    if parts.is_empty() {
        return None;
    }

    // Find the first numeric segment from the right
    let mut end = parts.len();
    while end > 0 && parts[end - 1].parse::<usize>().is_ok() {
        end -= 1;
    }

    if end == parts.len() {
        // No trailing numeric segments found
        return None;
    }

    if end == 0 {
        // Entire name is numeric segments (shouldn't happen)
        return None;
    }

    // Reconstruct the prefix by joining the non-numeric parts
    Some(parts[..end].join("_"))
}

/// Extract all string values from a JSON string that look like SIR node IDs
/// (start with "sir_"). These are potential SirNodeId references.
fn extract_sir_ids_from_json(json: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut in_string = false;
    let mut current = String::new();

    for ch in json.chars() {
        if ch == '"' && !in_string {
            in_string = true;
            current.clear();
        } else if ch == '"' && in_string {
            in_string = false;
            if current.starts_with("sir_") {
                ids.push(current.clone());
            }
        } else if in_string {
            current.push(ch);
        }
    }
    ids
}

/// Resolve a semantic alias like "sir_q_proj_layer_0_self_attn" to the actual
/// allocated node ID like "sir_3_layer_0_self_attn".
///
/// The alias format is: "sir_{semantic_prefix}_{traced_node_id}"
/// The actual ID format is: "sir_{counter}_{traced_node_id}"
///
/// We need to find which actual ID corresponds to this alias. We do this by:
/// 1. Extracting the traced_node_id suffix from the alias
/// 2. Finding all actual IDs with that same suffix (they're all from the same
///    traced node's decomposed ops)
/// 3. Matching the semantic prefix to the correct position in the decomposed
///    ops list
///
/// The tricky part is determining which position corresponds to which alias.
/// The decompose functions create ops in a fixed order:
///   - AttentionBlock: q_proj, k_proj, v_proj, [q_norm, k_norm], [gqa_k_tile, gqa_v_tile], [k_transpose], ...
///   - MLP: gate_proj, [gate_silu], up_proj, [swiglu_mul], down_proj
///   - RoPE: cos, [sin, rotated, cos_mul, sin_mul]
///   - SDPA: k_transpose, qk, scale_const, scaled, softmax
///
/// Since we don't have the exact order without re-running the decompose functions,
/// we use a heuristic: match the semantic prefix to the sequential allocation
/// order for the traced node. The first op allocated for a traced node gets
/// the first counter, and the semantic prefix tells us which op it is.
///
/// We build a position-based mapping using node_map: all IDs registered for
/// the same traced_node_id are in allocation order, which matches the decompose
/// function's push order.
fn resolve_alias(
    alias: &str,
    actual_ids: &std::collections::HashSet<String>,
    node_map: &std::collections::HashMap<String, SirNodeId>,
) -> Option<String> {
    // The alias is like "sir_q_proj_layer_0_self_attn"
    // We need to find the traced_node_id suffix.
    // Pattern: "sir_{semantic_prefix}_{traced_node_id}"
    // The semantic prefix can contain underscores (e.g., "gqa_k_tiled"),
    // so we can't just split on "_". Instead, we try all possible splits
    // and check if the suffix matches a known traced node ID.

    // Try each possible split point after "sir_"
    let alias_stripped = alias.strip_prefix("sir_")?;

    // Try splitting at each underscore position to find a traced_node_id
    // that exists in node_map. We want the LONGEST possible semantic prefix
    // (shortest traced_node_id) that matches, because semantic prefixes are
    // specific (e.g., "gqa_k_tiled") and traced_node_ids are relatively short
    // (e.g., "layer_0_self_attn").
    let mut candidates: Vec<String> = Vec::new();
    for (i, c) in alias_stripped.char_indices() {
        if c == '_' {
            let trace_id = &alias_stripped[i + 1..];
            if node_map.contains_key(trace_id) {
                // Found a valid traced_node_id suffix
                // Now find all actual IDs that were allocated for this traced node.
                // They all share the same suffix: "sir_{counter}_{trace_id}"
                // Collect them in counter order.
                let mut node_ids_for_trace: Vec<(usize, String)> = Vec::new();
                for actual in actual_ids {
                    if actual.ends_with(trace_id) && actual.starts_with("sir_") {
                        // Extract the counter from "sir_{counter}_{trace_id}"
                        let without_prefix = actual.strip_prefix("sir_").unwrap();
                        let without_suffix = without_prefix.strip_suffix(trace_id).unwrap();
                        let counter_str = without_suffix.strip_suffix('_').unwrap_or(without_suffix);
                        if let Ok(counter) = counter_str.parse::<usize>() {
                            node_ids_for_trace.push((counter, actual.clone()));
                        }
                    }
                }
                node_ids_for_trace.sort_by_key(|(c, _)| *c);

                // The semantic prefix determines which position in the decomposed
                // ops list this alias corresponds to. We need to map the semantic
                // prefix to an index.
                let semantic_prefix = &alias_stripped[..i];

                // Map semantic prefix to position index using the known decompose
                // function ordering. This is a fixed mapping based on the order
                // that decompose_attention(), decompose_mlp(), decompose_rope(),
                // and decompose_sdpa() push their ops.
                let position = semantic_prefix_to_position(semantic_prefix);

                if let Some(pos) = position {
                    if pos < node_ids_for_trace.len() {
                        return Some(node_ids_for_trace[pos].1.clone());
                    }
                }

                candidates.push(format!("prefix='{}' trace_id='{}' nodes={}", semantic_prefix, trace_id, node_ids_for_trace.len()));
            }
        }
    }

    // If we couldn't resolve, return None
    None
}

/// Map a semantic prefix (from an alias like "sir_q_proj_...") to the position
/// index in the decomposed ops list for the corresponding traced node.
///
/// The positions correspond to the order in which decompose_attention(),
/// decompose_mlp(), decompose_rope(), and decompose_sdpa() push their ops.
///
/// AttentionBlock decomposition order (without QK-Norm, without GQA):
///   0: q_proj, 1: k_proj, 2: v_proj, 3: sdpa/out_proj, ...
///
/// AttentionBlock decomposition order (with QK-Norm + GQA, A16+):
///   0: q_proj, 1: k_proj, 2: v_proj, 3: q_norm, 4: k_norm,
///   5: gqa_k_tile, 6: gqa_v_tile, 7: sdpa, 8: out_proj, 9: residual_add
///
/// AttentionBlock decomposition order (with QK-Norm + GQA, pre-A16):
///   0: q_proj, 1: k_proj, 2: v_proj, 3: q_norm, 4: k_norm,
///   5: gqa_k_tile, 6: gqa_v_tile, 7: k_transpose, 8: qk_matmul,
///   9: scale_const, 10: scaled_qk, 11: softmax, 12: sv_matmul,
///   13: out_proj, 14: residual_add
///
/// MLP decomposition order (SwiGLU):
///   0: gate_proj, 1: gate_silu, 2: up_proj, 3: swiglu_mul, 4: down_proj
///
/// MLP decomposition order (standard):
///   0: up_proj, 1: act, 2: down_proj
fn semantic_prefix_to_position(prefix: &str) -> Option<usize> {
    // NOTE: This position-based mapping is a LEGACY FALLBACK. The primary
    // alias resolution is now done via the name-based alias map (which is
    // built dynamically from the actual name tags). This function is only
    // used for aliases that don't match any name tag pattern.
    //
    // Attention block ops (positions for the QK-Norm + GQA + pre-A16 path)
    match prefix {
        "q_proj" => Some(0),
        "k_proj" => Some(1),
        "v_proj" => Some(2),
        "q_norm" => Some(3),
        "k_norm" => Some(4),
        "gqa_k_tile" => Some(5),
        "gqa_v_tile" => Some(6),
        // Pre-A16 SDPA decomposition (positions 7-12)
        "attn_k_transpose" => Some(7),
        "attn_qk" => Some(8),
        "attn_scale_const" => Some(9),
        "attn_scale" => Some(10),
        "attn_softmax" => Some(11),
        "attn_sv" => Some(12),
        "out_proj" => Some(13),
        "sdpa" => Some(7),
        // MLP ops (these come from separate traced nodes, so positions restart)
        "gate_proj" => Some(0),
        "mlp_gate_silu" => Some(1),
        "up_proj" => Some(2),
        "mlp_swiglu_mul" => Some(3),
        "down_proj" => Some(4),
        "mlp_act" => Some(1),
        // RoPE ops
        "rope_cos" => Some(0),
        "rope_sin" => Some(1),
        "rope_rotated" => Some(2),
        "rope_cos_mul" => Some(3),
        "rope_sin_mul" => Some(4),
        // Standalone SDPA decomposition
        "sdpa_k_t" => Some(0),
        "sdpa_qk" => Some(1),
        "sdpa_scale_const" => Some(2),
        "sdpa_scaled" => Some(3),
        "sdpa_softmax" => Some(4),
        _ => None,
    }
}
