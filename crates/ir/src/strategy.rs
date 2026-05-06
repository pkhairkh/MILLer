//! Optimization Strategy Framework
//!
//! Dynamic, composable optimization strategy discovery and application.
//! Instead of hardcoding specific optimizations as fixed op variants or
//! fixed pass paths, this framework allows the compiler to:
//!
//! 1. **Discover** which optimization strategies apply to a given graph
//!    based on its structure and target hardware
//! 2. **Configure** each strategy with parameters derived from the graph
//! 3. **Compose** multiple strategies in a principled order
//! 4. **Validate** that applied strategies preserve the quality contract
//!
//! # Design Philosophy
//!
//! Optimization strategies are **possible paths**, not fixed implementations.
//! The compiler discovers what makes sense for each model based on:
//! - Graph structure (what ops are present, what patterns exist)
//! - Target hardware (AneFamily capabilities and constraints)
//! - Quality contract (perplexity delta, latency requirements)
//! - Model characteristics (hidden size, quantization sensitivity)
//!
//! This means the same framework handles Qwen3, LLaMA, GPT-2, or any
//! future architecture — no model registry needed.

use super::ane_target::AneFamily;
use super::sir::SirGraph;
use serde::{Deserialize, Serialize};

// ─── Strategy Identity ──────────────────────────────────────────────

/// A unique optimization strategy identifier.
///
/// Strategies are identified by a category + variant pair rather than
/// a single name. This allows multiple variants within the same
/// optimization category (e.g., different RMSNorm decomposition methods).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrategyId {
    /// The optimization category (normalization, kv_cache, quantization, etc.)
    pub category: StrategyCategory,
    /// A variant name within the category (e.g., "dynamic_max_abs", "naive")
    pub variant: String,
}

impl StrategyId {
    pub fn new(category: StrategyCategory, variant: &str) -> Self {
        Self { category, variant: variant.to_string() }
    }
}

/// Broad optimization category — determines when and how a strategy
/// is applied in the compilation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyCategory {
    /// Normalization decomposition: how to lower RMSNorm/LayerNorm
    /// to ANE-faithful primitives.
    Normalization,
    /// KV cache layout: how KV cache reads/writes are structured
    /// for ANE compatibility.
    KvCache,
    /// Weight quantization: how weight tensors are compressed
    /// for on-device deployment.
    Quantization,
    /// Static constant hoisting: pre-computing tables (RoPE, masks, etc.)
    /// as constants instead of dynamic computation.
    ConstantHoisting,
    /// Sampling: how token selection is performed on-device.
    Sampling,
    /// IO model: how embedding and LM head are packaged.
    IoModel,
    /// Shard partitioning: how the model is split across packages.
    Sharding,
}

// ─── Strategy Specification ─────────────────────────────────────────

/// A fully-parameterized optimization strategy ready for application.
///
/// Unlike a `StrategyId` (which just names a strategy), a `StrategySpec`
/// carries the concrete parameters that the strategy needs. These parameters
/// are derived from graph analysis — the strategy discovers what it needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySpec {
    /// Which strategy this is.
    pub id: StrategyId,
    /// Whether this strategy applies to the target graph.
    pub applicable: bool,
    /// Estimated benefit (0.0 = none, 1.0 = maximum).
    pub benefit: f32,
    /// Estimated cost in compile time / complexity (0.0 = free, 1.0 = expensive).
    pub cost: f32,
    /// Strategy-specific parameters as key-value pairs.
    pub params: StrategyParams,
    /// Human-readable reason why this strategy applies (or doesn't).
    pub reason: String,
}

/// Strategy parameters — a typed key-value store.
///
/// This allows each strategy to carry its own parameters without
/// requiring a dedicated struct per strategy. Parameters are derived
/// from graph analysis during strategy discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    entries: Vec<(String, StrategyValue)>,
}

impl StrategyParams {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn set(&mut self, key: &str, value: StrategyValue) {
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value;
        } else {
            self.entries.push((key.to_string(), value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&StrategyValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(|v| match v {
            StrategyValue::F32(f) => Some(*f),
            _ => None,
        })
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.get(key).and_then(|v| match v {
            StrategyValue::Usize(n) => Some(*n),
            _ => None,
        })
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| match v {
            StrategyValue::Bool(b) => Some(*b),
            _ => None,
        })
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| match v {
            StrategyValue::Str(s) => Some(s.as_str()),
            _ => None,
        })
    }
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed value in strategy parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyValue {
    F32(f32),
    Usize(usize),
    Bool(bool),
    Str(String),
    /// A list of values (for per-layer configurations).
    List(Vec<StrategyValue>),
}

// ─── Strategy Discovery ─────────────────────────────────────────────

/// Result of strategy discovery — the set of applicable optimizations
/// for a given graph and target, ordered by benefit/cost ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    /// All strategies that were evaluated.
    pub evaluated: Vec<StrategySpec>,
    /// Strategies that apply and are recommended, sorted by benefit desc.
    pub recommended: Vec<StrategySpec>,
    /// The target ANE family that constrained discovery.
    pub target_family: AneFamily,
    /// Summary of graph characteristics that drove discovery.
    pub graph_summary: GraphSummary,
}

/// Summary of graph characteristics used during strategy discovery.
///
/// This is computed once and shared across all strategy probes,
/// avoiding redundant graph traversals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSummary {
    /// Number of RMSNorm ops in the graph.
    pub rms_norm_count: usize,
    /// Number of LayerNorm ops in the graph.
    pub layer_norm_count: usize,
    /// Number of attention ops (AttentionBlock + SDPA) in the graph.
    pub attention_count: usize,
    /// Number of linear projection ops in the graph.
    pub linear_count: usize,
    /// Number of StateRead/StateWrite ops (KV cache).
    pub state_op_count: usize,
    /// Number of RoPE-related ops (Cos, Sin, or RoPETransform).
    pub rope_op_count: usize,
    /// Whether the graph uses GQA.
    pub uses_gqa: bool,
    /// Whether the graph uses RMSNorm.
    pub uses_rms_norm: bool,
    /// Whether the graph has KV cache state.
    pub has_kv_cache: bool,
    /// Whether the graph has tied embedding/LM-head weights.
    pub has_tied_weights: bool,
    /// Hidden dimension (0 if not detected).
    pub hidden_size: usize,
    /// Number of transformer layers (0 if not detected).
    pub num_layers: usize,
}

impl GraphSummary {
    /// Compute a summary from a SIR graph.
    pub fn from_graph(graph: &SirGraph) -> Self {
        use super::sir::SirOp;

        let mut summary = GraphSummary {
            rms_norm_count: 0,
            layer_norm_count: 0,
            attention_count: 0,
            linear_count: 0,
            state_op_count: 0,
            rope_op_count: 0,
            uses_gqa: false,
            uses_rms_norm: false,
            has_kv_cache: false,
            has_tied_weights: false,
            hidden_size: 0,
            num_layers: 0,
        };

        for node in &graph.nodes {
            match &node.op {
                SirOp::RMSNorm { .. } => {
                    summary.rms_norm_count += 1;
                    summary.uses_rms_norm = true;
                }
                SirOp::LayerNorm { .. } => {
                    summary.layer_norm_count += 1;
                }
                SirOp::AttentionBlock { .. } | SirOp::ScaledDotProductAttention { .. } => {
                    summary.attention_count += 1;
                }
                SirOp::LinearProjection { .. } => {
                    summary.linear_count += 1;
                }
                SirOp::StateRead { state_id, .. } | SirOp::StateWrite { state_id, .. } => {
                    if state_id.contains("kv_cache") {
                        summary.has_kv_cache = true;
                    }
                    summary.state_op_count += 1;
                }
                SirOp::Cos { .. } | SirOp::Sin { .. } | SirOp::RoPETransform { .. } => {
                    summary.rope_op_count += 1;
                }
                _ => {}
            }
        }

        // Infer number of layers from KV cache state IDs
        // (e.g., "kv_cache_layer_3_key" → 4 layers)
        let max_layer = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                SirOp::StateRead { state_id, .. } | SirOp::StateWrite { state_id, .. } => {
                    state_id.split('_').filter_map(|s| s.parse::<usize>().ok()).max()
                }
                _ => None,
            })
            .max();

        if let Some(max_idx) = max_layer {
            summary.num_layers = max_idx + 1;
        } else if summary.rms_norm_count > 0 {
            // Rough estimate: 2 RMSNorms per layer (attention + MLP norm)
            summary.num_layers = summary.rms_norm_count.div_ceil(2);
        }

        summary
    }
}

/// Discover applicable optimization strategies for a graph and target.
///
/// This is the primary entry point for the strategy framework. It
/// evaluates all known strategies against the graph and target hardware,
/// returning a `DiscoveryReport` with recommendations.
///
/// The discovery is:
/// - **Dynamic**: based on what's actually in the graph
/// - **Target-aware**: considers AneFamily capabilities
/// - **Non-prescriptive**: returns possibilities, not mandates
pub fn discover_strategies(graph: &SirGraph, target_family: AneFamily) -> DiscoveryReport {
    let summary = GraphSummary::from_graph(graph);
    let mut evaluated = Vec::new();

    // Probe each strategy category
    evaluated.extend(probe_normalization_strategies(&summary, target_family));
    evaluated.extend(probe_kv_cache_strategies(&summary, target_family));
    evaluated.extend(probe_quantization_strategies(&summary, target_family));
    evaluated.extend(probe_constant_hoisting_strategies(&summary, target_family));
    evaluated.extend(probe_sampling_strategies(&summary, target_family));
    evaluated.extend(probe_io_model_strategies(&summary, target_family));

    // Sort applicable strategies by benefit/cost ratio (descending)
    let mut recommended: Vec<StrategySpec> =
        evaluated.iter().filter(|s| s.applicable).cloned().collect();

    recommended.sort_by(|a, b| {
        let ratio_a = if a.cost > 0.0 { a.benefit / a.cost } else { a.benefit * 100.0 };
        let ratio_b = if b.cost > 0.0 { b.benefit / b.cost } else { b.benefit * 100.0 };
        ratio_b.partial_cmp(&ratio_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    DiscoveryReport { evaluated, recommended, target_family, graph_summary: summary }
}

// ─── Strategy Probes ────────────────────────────────────────────────

/// Probe normalization strategies.
///
/// The key insight: RMSNorm decomposition should be chosen dynamically
/// based on target hardware and model characteristics, not hardcoded.
///
/// Possible paths:
/// - **naive**: x * rsqrt(mean(x^2) + eps) * weight — simple but may
///   underflow in fp16 for large hidden sizes
/// - **max_abs_stabilized**: normalize by max(|x|) first, then compute
///   variance on the normalized values. Uses two-division epsilon
///   compensation to avoid forming max^2 (which can underflow in fp16).
///   The "dynamic-safe" pattern from community Core ML deployments.
/// - **layernorm_fallback**: use LayerNorm directly (A15+ only)
fn probe_normalization_strategies(
    summary: &GraphSummary,
    target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    // Naive decomposition — always applicable for RMSNorm
    if summary.uses_rms_norm {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::Normalization, "naive_rms"),
            applicable: true,
            benefit: 0.1,
            cost: 0.0,
            params: {
                let mut p = StrategyParams::new();
                p.set("decomposition", StrategyValue::Str("x * rsqrt(mean(x^2) + eps) * weight".to_string()));
                p.set("stabilization", StrategyValue::Str("none".to_string()));
                p
            },
            reason: "Naive RMSNorm decomposition — simple but may underflow in fp16 for large hidden sizes".to_string(),
        });

        // Max-abs stabilized decomposition — better for fp16 on ANE
        // Applicable when: target has fp16 compute AND model uses RMSNorm
        // This is the pattern discovered in community Core ML deployments:
        // normalize by max(|x|) first, use two-division eps to avoid max^2
        let benefit = if target_family.broadcast_fp16_only() {
            0.9 // Critical for A11/A12 fp16-only broadcast
        } else if summary.hidden_size >= 2048 {
            0.8 // Large models benefit significantly
        } else {
            0.5 // Smaller models benefit moderately
        };

        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::Normalization, "max_abs_stabilized"),
            applicable: true,
            benefit,
            cost: 0.3, // More ops but straightforward
            params: {
                let mut p = StrategyParams::new();
                p.set("decomposition", StrategyValue::Str(
                    "abs → max → clip(min=2^-14) → div_max → square → mean → eps/two_div → rsqrt → mul".to_string()
                ));
                p.set("stabilization", StrategyValue::Str("max_abs".to_string()));
                p.set("eps_method", StrategyValue::Str("two_division".to_string()));
                p.set("fp16_floor_guard", StrategyValue::F32(2.0f32.powi(-14)));
                p
            },
            reason: format!(
                "Max-abs stabilized RMSNorm — prevents fp16 underflow. Benefit={:.1} because {}",
                benefit,
                if target_family.broadcast_fp16_only() {
                    "target uses fp16-only broadcast".to_string()
                } else if summary.hidden_size >= 2048 {
                    format!("large hidden_size={}", summary.hidden_size)
                } else {
                    "moderate fp16 risk".to_string()
                }
            ),
        });
    }

    // LayerNorm fallback — only on A15+
    if summary.layer_norm_count > 0 || summary.uses_rms_norm {
        let applicable = target_family.supports_layernorm();
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::Normalization, "layernorm_native"),
            applicable,
            benefit: if applicable { 0.7 } else { 0.0 },
            cost: 0.1,
            params: {
                let mut p = StrategyParams::new();
                p.set("requires_family", StrategyValue::Str("A15+".to_string()));
                p
            },
            reason: if applicable {
                "Target supports native LayerNorm — use directly".to_string()
            } else {
                format!("Target {:?} does not support LayerNorm on ANE", target_family)
            },
        });
    }

    strategies
}

/// Probe KV cache strategies.
///
/// The choice of KV cache layout directly affects ANE provisioning.
/// Possible paths:
/// - **naive**: append/shift — scatter-heavy, often CPU fallback
/// - **masked_blend**: contiguous suffix + masked blending for writes
/// - **paged**: fixed-size blocks (future, not yet implemented)
fn probe_kv_cache_strategies(
    summary: &GraphSummary,
    target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    if !summary.has_kv_cache {
        return strategies;
    }

    // Naive — always applicable but poor for ANE
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::KvCache, "naive"),
        applicable: true,
        benefit: 0.1,
        cost: 0.0,
        params: {
            let mut p = StrategyParams::new();
            p.set("write_mechanism", StrategyValue::Str("scatter".to_string()));
            p.set("read_mechanism", StrategyValue::Str("contiguous".to_string()));
            p
        },
        reason: "Naive KV cache — scatter writes often force CPU fallback on ANE".to_string(),
    });

    // Masked blend — ANE-friendly
    // This pattern uses Where/Mul/Add to blend new K/V values into
    // the cache instead of scatter. The "reverse ring-buffer" approach
    // from community deployments is one instance of this pattern.
    let benefit = if matches!(
        target_family,
        AneFamily::A14 | AneFamily::A15 | AneFamily::A16 | AneFamily::A17 | AneFamily::A18
    ) {
        0.9 // Major benefit on A14+ families (A14Plus converters, pool-to-conv available)
    } else {
        0.3 // Modest benefit on A13 and older (A14Minus converters, pool-to-conv blocked)
    };

    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::KvCache, "masked_blend"),
        applicable: true,
        benefit,
        cost: 0.4,
        params: {
            let mut p = StrategyParams::new();
            p.set("write_mechanism", StrategyValue::Str("masked_blend".to_string()));
            p.set("read_mechanism", StrategyValue::Str("contiguous_suffix".to_string()));
            p.set("layout", StrategyValue::Str("reverse_ring_buffer".to_string()));
            p.set("requires_position_tracking", StrategyValue::Bool(true));
            p.set("requires_valid_mask", StrategyValue::Bool(true));
            p
        },
        reason: format!(
            "Masked-blend KV cache — ANE-friendly write pattern. Benefit={:.1} on {:?}",
            benefit, target_family
        ),
    });

    // Paged — future option
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::KvCache, "paged"),
        applicable: false, // Not yet implemented
        benefit: 0.7,
        cost: 0.8,
        params: {
            let mut p = StrategyParams::new();
            p.set("write_mechanism", StrategyValue::Str("block_write".to_string()));
            p.set("read_mechanism", StrategyValue::Str("block_read".to_string()));
            p.set("block_size", StrategyValue::Usize(16));
            p
        },
        reason: "Paged KV cache — not yet implemented, reserved for future paged-attention support"
            .to_string(),
    });

    strategies
}

/// Probe quantization strategies.
///
/// Quantization is parameterized by bit-width, group size, and method
/// rather than named after specific projects. Different weight types
/// can use different parameters based on sensitivity analysis.
fn probe_quantization_strategies(
    summary: &GraphSummary,
    _target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    if summary.linear_count == 0 && summary.attention_count == 0 {
        return strategies;
    }

    // No quantization — always an option
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::Quantization, "none"),
        applicable: true,
        benefit: 0.0,
        cost: 0.0,
        params: StrategyParams::new(),
        reason: "No quantization — full fp16 weights, no quality loss".to_string(),
    });

    // LUT-based quantization — grouped look-up table
    // Each group of weights shares a palette of 2^bits entries.
    // Parameters: group_size, bits, per-layer sensitivity
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::Quantization, "grouped_lut"),
        applicable: true,
        benefit: 0.8, // Significant model size reduction
        cost: 0.5,    // Requires calibration and sensitivity analysis
        params: {
            let mut p = StrategyParams::new();
            p.set("method", StrategyValue::Str("lut".to_string()));
            p.set("default_bits", StrategyValue::Usize(4));
            p.set("default_group_size", StrategyValue::Usize(128));
            // Conservative defaults for attention projections
            p.set("attention_bits", StrategyValue::Usize(6));
            // Aggressive defaults for MLP projections
            p.set("mlp_bits", StrategyValue::Usize(4));
            // Very aggressive for mask/KV constants
            p.set("mask_kv_bits", StrategyValue::Usize(1));
            p
        },
        reason:
            "Grouped LUT quantization — significant size reduction with per-layer bit-width control"
                .to_string(),
    });

    // Blockwise quantization — per-group scales and offsets
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::Quantization, "blockwise"),
        applicable: true,
        benefit: 0.7,
        cost: 0.4,
        params: {
            let mut p = StrategyParams::new();
            p.set("method", StrategyValue::Str("blockwise".to_string()));
            p.set("default_bits", StrategyValue::Usize(4));
            p.set("default_group_size", StrategyValue::Usize(128));
            p
        },
        reason: "Blockwise quantization — good for embedding/LM head matrices".to_string(),
    });

    // Post-hoc palettization via kmeans
    strategies.push(StrategySpec {
        id: StrategyId::new(StrategyCategory::Quantization, "kmeans_palettize"),
        applicable: true,
        benefit: 0.6,
        cost: 0.3,
        params: {
            let mut p = StrategyParams::new();
            p.set("method", StrategyValue::Str("kmeans".to_string()));
            p.set("default_nbits", StrategyValue::Usize(4));
            p.set("default_group_size", StrategyValue::Usize(128));
            p
        },
        reason:
            "K-means palettization — post-hoc compression for constants and less sensitive weights"
                .to_string(),
    });

    strategies
}

/// Probe constant hoisting strategies.
///
/// Pre-computing tables as constants avoids dynamic computation at
/// inference time, which is especially important on ANE where dynamic
/// ops may cause CPU fallback.
fn probe_constant_hoisting_strategies(
    summary: &GraphSummary,
    _target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    // RoPE tables
    if summary.rope_op_count > 0 {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::ConstantHoisting, "rope_tables"),
            applicable: true,
            benefit: 0.6,
            cost: 0.2,
            params: {
                let mut p = StrategyParams::new();
                p.set(
                    "tables",
                    StrategyValue::List(vec![
                        StrategyValue::Str("sin_tab".to_string()),
                        StrategyValue::Str("cos_tab".to_string()),
                    ]),
                );
                p.set("dtype", StrategyValue::Str("fp16".to_string()));
                p
            },
            reason: format!(
                "Pre-compute RoPE sin/cos tables — {} RoPE ops in graph",
                summary.rope_op_count
            ),
        });
    }

    // Causal mask table
    if summary.attention_count > 0 {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::ConstantHoisting, "causal_mask"),
            applicable: true,
            benefit: 0.4,
            cost: 0.1,
            params: {
                let mut p = StrategyParams::new();
                p.set(
                    "tables",
                    StrategyValue::List(vec![
                        StrategyValue::Str("mask_tab".to_string()),
                        StrategyValue::Str("eye_tab".to_string()),
                    ]),
                );
                p.set("dtype", StrategyValue::Str("fp16".to_string()));
                p
            },
            reason: "Pre-compute causal mask and identity tables as fp16 constants".to_string(),
        });
    }

    strategies
}

/// Probe sampling strategies.
///
/// On-device sampling keeps the decode loop fully on-device rather
/// than requiring host-side post-processing.
fn probe_sampling_strategies(
    summary: &GraphSummary,
    _target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    // Only recommend sampling strategy if the graph looks like a
    // causal LM (has attention + enough layers for a real model)
    if summary.attention_count > 0 && summary.num_layers > 0 {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::Sampling, "on_device_topk"),
            applicable: true,
            benefit: 0.5,
            cost: 0.4,
            params: {
                let mut p = StrategyParams::new();
                p.set("method", StrategyValue::Str("topk_gumbel".to_string()));
                p.set("pre_candidate_k", StrategyValue::Usize(64));
                p.set("final_top_k", StrategyValue::Usize(16));
                p.set("num_noise_samples", StrategyValue::Usize(8192));
                p.set("default_temperature", StrategyValue::F32(1.0));
                p.set("default_min_p", StrategyValue::F32(0.05));
                p
            },
            reason: "On-device sampler — keeps decode loop on-device, avoids host round-trip"
                .to_string(),
        });

        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::Sampling, "host_side"),
            applicable: true,
            benefit: 0.1,
            cost: 0.0,
            params: StrategyParams::new(),
            reason: "Host-side sampling — simpler but requires CPU round-trip for each token"
                .to_string(),
        });
    }

    strategies
}

/// Probe IO model strategies.
///
/// For models with tied embedding/LM-head weights, a conditional IO
/// model can share the weight matrix and halve memory usage.
fn probe_io_model_strategies(
    summary: &GraphSummary,
    _target_family: AneFamily,
) -> Vec<StrategySpec> {
    let mut strategies = Vec::new();

    if summary.has_tied_weights {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::IoModel, "conditional_shared"),
            applicable: true,
            benefit: 0.7, // Halves memory for tied weights
            cost: 0.3,
            params: {
                let mut p = StrategyParams::new();
                p.set("mode", StrategyValue::Str("conditional".to_string()));
                p.set("shared_weights", StrategyValue::Bool(true));
                p.set("embedding_mode_value", StrategyValue::Usize(0));
                p.set("logit_mode_value", StrategyValue::Usize(1));
                p
            },
            reason: "Conditional IO model — shared embedding/LM-head weights halve memory"
                .to_string(),
        });
    }

    // Standard separate IO model — always applicable
    if summary.linear_count > 0 {
        strategies.push(StrategySpec {
            id: StrategyId::new(StrategyCategory::IoModel, "separate"),
            applicable: true,
            benefit: 0.1,
            cost: 0.0,
            params: {
                let mut p = StrategyParams::new();
                p.set("mode", StrategyValue::Str("separate".to_string()));
                p.set("shared_weights", StrategyValue::Bool(false));
                p
            },
            reason: "Separate IO model — embedding and LM head in separate packages".to_string(),
        });
    }

    strategies
}

// ─── Strategy Application ───────────────────────────────────────────

/// Result of applying a strategy to a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyApplicationResult {
    /// The strategy that was applied.
    pub strategy_id: StrategyId,
    /// Number of nodes added.
    pub nodes_added: usize,
    /// Number of nodes modified.
    pub nodes_modified: usize,
    /// Number of nodes removed.
    pub nodes_removed: usize,
    /// Whether the application was successful.
    pub success: bool,
    /// Optional message about what happened.
    pub message: Option<String>,
}

// ─── Compilation Plan ───────────────────────────────────────────────

/// A compilation plan derived from strategy discovery.
///
/// This is the bridge between strategy discovery and the pass pipeline.
/// It specifies which passes to run, in what order, and with what
/// parameters — all derived dynamically from graph analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationPlan {
    /// The discovery report that drove this plan.
    pub discovery: DiscoveryReport,
    /// Ordered list of strategies to apply.
    pub strategy_order: Vec<StrategyId>,
}

impl CompilationPlan {
    /// Create a compilation plan from a discovery report.
    ///
    /// Selects recommended strategies and orders them by:
    /// 1. Category order (Normalization → KvCache → ConstantHoisting → Quantization → Sampling → IoModel → Sharding)
    /// 2. Benefit/cost ratio within each category
    pub fn from_discovery(discovery: DiscoveryReport) -> Self {
        let category_order = [
            StrategyCategory::Normalization,
            StrategyCategory::KvCache,
            StrategyCategory::ConstantHoisting,
            StrategyCategory::Quantization,
            StrategyCategory::Sampling,
            StrategyCategory::IoModel,
            StrategyCategory::Sharding,
        ];

        let strategy_order: Vec<StrategyId> = category_order
            .iter()
            .flat_map(|cat| {
                let for_category: Vec<&StrategySpec> =
                    discovery.recommended.iter().filter(|s| s.id.category == *cat).collect();
                // Already sorted by benefit/cost ratio from discovery
                // Take the best strategy per category (user can override)
                for_category
                    .into_iter()
                    .filter(|s| s.benefit > 0.0)
                    .map(|s| s.id.clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        CompilationPlan { discovery, strategy_order }
    }

    /// Whether this plan includes a specific strategy variant.
    pub fn includes(&self, category: StrategyCategory, variant: &str) -> bool {
        self.strategy_order.iter().any(|id| id.category == category && id.variant == variant)
    }

    /// Get the parameters for a strategy in this plan.
    pub fn params_for(&self, id: &StrategyId) -> Option<&StrategyParams> {
        self.discovery.recommended.iter().find(|s| &s.id == id).map(|s| &s.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sir::{
        SirGraph, SirMetadata, SirNode, SirNodeId, SirOp, SirTargetAnnotation, TaskOrigin,
    };

    fn make_simple_graph() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("input".to_string()),
                    op: SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                    name: "input".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                    target_annotation: SirTargetAnnotation::default(),
                },
                SirNode {
                    id: SirNodeId("rms_0".to_string()),
                    op: SirOp::RMSNorm {
                        input: SirNodeId("input".to_string()),
                        weight: "norm_weight".to_string(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                    name: "rms_norm".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                    target_annotation: SirTargetAnnotation::default(),
                },
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        }
    }

    #[test]
    fn test_discover_normalization_strategies() {
        let graph = make_simple_graph();
        let report = discover_strategies(&graph, AneFamily::A16);

        let norm_strategies: Vec<_> = report
            .evaluated
            .iter()
            .filter(|s| s.id.category == StrategyCategory::Normalization)
            .collect();

        assert!(norm_strategies.len() >= 2, "Should discover at least 2 normalization strategies");

        let has_naive = norm_strategies.iter().any(|s| s.id.variant == "naive_rms");
        let has_stabilized = norm_strategies.iter().any(|s| s.id.variant == "max_abs_stabilized");
        assert!(has_naive, "Should discover naive_rms strategy");
        assert!(has_stabilized, "Should discover max_abs_stabilized strategy");
    }

    #[test]
    fn test_stabilized_higher_benefit_for_fp16_only() {
        let graph = make_simple_graph();
        let report_a12 = discover_strategies(&graph, AneFamily::A12);
        let report_a16 = discover_strategies(&graph, AneFamily::A16);

        let stabilized_a12 =
            report_a12.evaluated.iter().find(|s| s.id.variant == "max_abs_stabilized").unwrap();
        let stabilized_a16 =
            report_a16.evaluated.iter().find(|s| s.id.variant == "max_abs_stabilized").unwrap();

        // A12 (fp16-only broadcast) should have higher benefit for stabilization
        assert!(
            stabilized_a12.benefit > stabilized_a16.benefit,
            "Stabilized benefit should be higher on fp16-only families"
        );
    }

    #[test]
    fn test_kv_cache_strategies_only_when_cache_present() {
        let graph_no_cache = make_simple_graph();
        let report = discover_strategies(&graph_no_cache, AneFamily::A16);

        let kv_strategies: Vec<_> = report
            .evaluated
            .iter()
            .filter(|s| s.id.category == StrategyCategory::KvCache)
            .collect();

        assert!(kv_strategies.is_empty(), "No KV cache strategies without KV cache ops");
    }

    #[test]
    fn test_compilation_plan_ordering() {
        let graph = make_simple_graph();
        let report = discover_strategies(&graph, AneFamily::A16);
        let plan = CompilationPlan::from_discovery(report);

        // Normalization should come before Quantization
        let norm_idx = plan
            .strategy_order
            .iter()
            .position(|id| id.category == StrategyCategory::Normalization);
        let quant_idx =
            plan.strategy_order.iter().position(|id| id.category == StrategyCategory::Quantization);

        if let (Some(ni), Some(qi)) = (norm_idx, quant_idx) {
            assert!(ni < qi, "Normalization should run before Quantization");
        }
    }

    #[test]
    fn test_graph_summary_detects_rms_norm() {
        let graph = make_simple_graph();
        let summary = GraphSummary::from_graph(&graph);

        assert_eq!(summary.rms_norm_count, 1);
        assert!(summary.uses_rms_norm);
    }

    #[test]
    fn test_strategy_params_typed_access() {
        let mut params = StrategyParams::new();
        params.set("bits", StrategyValue::Usize(4));
        params.set("epsilon", StrategyValue::F32(1e-6));
        params.set("method", StrategyValue::Str("lut".to_string()));

        assert_eq!(params.get_usize("bits"), Some(4));
        assert_eq!(params.get_f32("epsilon"), Some(1e-6));
        assert_eq!(params.get_str("method"), Some("lut"));
    }
}
