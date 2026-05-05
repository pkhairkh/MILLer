//! Task Specification Types
//!
//! Concrete task definitions loadable from TOML files.
//! These are the input format for the compilation pipeline.

use serde::{Deserialize, Serialize};

/// A synthetic task specification loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticTaskSpec {
    /// Task name.
    pub name: String,
    /// Task family.
    pub family: String,
    /// Optional description.
    pub description: Option<String>,
    /// The operation to compile.
    pub op: TaskOp,
    /// Measurement configuration.
    pub measurement: MeasurementConfig,
}

/// Operation types for synthetic tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOp {
    /// Linear projection: y = x @ W + b
    LinearProjection {
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
        has_bias: bool,
        dtype: String,
    },
    /// Sharded linear pipeline: a sequence of linear projection shards
    /// with explicit role semantics (Entry/Interior/Exit), mimicking
    /// the Qwen3 three-shard decomposition at a micro scale.
    ///
    /// Each shard is a separate linear projection with its own dimensions,
    /// producing a separate mlpackage. The pipeline composes:
    ///   Entry shard:  [batch, input_dim] -> [batch, hidden_dim]
    ///   Interior shard: [batch, hidden_dim] -> [batch, hidden_dim]
    ///   Exit shard:  [batch, hidden_dim] -> [batch, output_dim]
    ShardedLinearPipeline {
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        batch_size: usize,
        dtype: String,
    },
    /// LUT (Look-Up Table) projection: grouped scalar-LUT palettized projection.
    ///
    /// Models the `constexpr_lut`-to-`gather` pattern used in ANE palettized
    /// inference. An input tensor of integer indices is used to look up values
    /// in a per-group LUT, approximating a dense linear projection at reduced
    /// bitwidth. This is the second nontrivial task family, more ANE-relevant
    /// than plain linear projection because it exercises the LUT/gather path
    /// that is central to ANE palettized inference.
    ///
    /// Parameters:
    /// - `vocab_size`: number of possible index values (LUT entries per group)
    /// - `embed_dim`: embedding dimension (number of groups / output features)
    /// - `num_groups`: number of independent LUT groups
    /// - `lut_bitwidth`: LUT precision in bits (1, 2, 3, 4, 6, or 8)
    /// - `batch_size`: batch dimension
    /// - `dtype`: data type of the LUT values and output ("fp16" or "fp32")
    LutProjection {
        vocab_size: usize,
        embed_dim: usize,
        num_groups: usize,
        lut_bitwidth: usize,
        batch_size: usize,
        dtype: String,
    },
    /// Decode step: autoregressive inference step with KV-cache.
    ///
    /// Models the dominant execution pattern in autoregressive LLM
    /// inference on Apple Silicon. A decode step consists of:
    /// - QKV projection on the new token embedding
    /// - Attention computation against a KV cache
    /// - Output projection
    ///
    /// This is the third real task family, more ANE-relevant than
    /// linear projection alone because it exercises the combined
    /// projection + attention pattern that is central to LLM
    /// token generation.
    ///
    /// Parameters:
    /// - `embed_dim`: total embedding dimension
    /// - `num_heads`: number of attention heads
    /// - `head_dim`: dimension per head (embed_dim / num_heads)
    /// - `kv_len`: KV-cache sequence length (number of cached tokens)
    /// - `batch_size`: batch dimension
    /// - `dtype`: data type ("fp16" or "fp32")
    DecodeStep {
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        batch_size: usize,
        kv_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        dtype: String,
        uses_rope: bool,
        has_qk_norm: bool,
    },
    /// Sharded decode step: a decode step decomposed into multiple shards.
    ///
    /// Sprint 23 (S23.3): this is the second multi-unit task op, proving
    /// the generalized shard model works beyond linear projection. The
    /// decode step is decomposed into three shards:
    ///   Entry: QKV projection (embed_dim → 3 * embed_dim)
    ///   Interior: attention computation (3 * embed_dim → embed_dim)
    ///   Exit: output projection (embed_dim → embed_dim)
    ///
    /// Unlike `ShardedLinearPipeline` which carries raw dimensions for
    /// a 3-shard linear decomposition, this variant carries the decode-step
    /// parameters and lets the shard planner construct the appropriate
    /// `ShardPipelineSpec` via `three_shard_decode_step`.
    ShardedDecodeStep {
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        batch_size: usize,
        kv_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        dtype: String,
        uses_rope: bool,
        has_qk_norm: bool,
    },
    /// MLP Block: fused feed-forward network block (linear_up + activation + linear_down).
    ///
    /// Models the feed-forward network block in transformer models, which is
    /// one of the two dominant ANE execution patterns (the other being attention).
    /// An MLP block consists of:
    /// - Up-projection: linear(input_dim -> hidden_dim)
    /// - Activation: GELU or ReLU
    /// - Down-projection: linear(hidden_dim -> output_dim)
    ///
    /// This is the fourth real task family. It exercises the fused
    /// linear-activation-linear pattern that is central to ANE placement
    /// for transformer feed-forward layers.
    ///
    /// Parameters:
    /// - `input_dim`: input dimension (typically equals embed_dim)
    /// - `hidden_dim`: intermediate (up-projected) dimension
    /// - `output_dim`: output dimension (typically equals embed_dim)
    /// - `activation`: activation function ("gelu" or "relu")
    /// - `batch_size`: batch dimension
    /// - `dtype`: data type ("fp16" or "fp32")
    MlpBlock {
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: String,
        batch_size: usize,
        dtype: String,
    },
    /// Attention: multi-head self-attention block.
    ///
    /// Models the multi-head self-attention pattern that is one of the two
    /// dominant ANE execution patterns in transformer inference (the other
    /// being the MLP block). An attention block consists of:
    /// - QKV projection: input @ W_qkv → Q, K, V
    /// - Multi-head scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
    /// - Output projection: attn_output @ W_out → output
    ///
    /// This is the fifth real task family. Unlike the `DecodeStep` family which
    /// models a single decode step with KV-cache, the `Attention` family models
    /// a standalone multi-head self-attention block without cache semantics.
    /// This exercises the attention-specific ANE path: QKV projection, softmax,
    /// and output projection as a fused unit.
    ///
    /// Parameters:
    /// - `embed_dim`: total embedding dimension
    /// - `num_heads`: number of attention heads
    /// - `head_dim`: dimension per head (embed_dim / num_heads)
    /// - `seq_len`: input sequence length
    /// - `batch_size`: batch dimension
    /// - `dtype`: data type ("fp16" or "fp32")
    Attention {
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        batch_size: usize,
        dtype: String,
    },
}

// ─── Generic TaskOp methods ─────────────────────────────────────────────
//
// These methods encapsulate per-variant behavior so that callers (main.rs,
// baseline.rs, etc.) do NOT need to match on every variant. Adding a new
// family only requires: (1) adding the enum variant, (2) adding the
// match arm in each method below, and (3) adding the TOML parser.
// No changes needed in main.rs, bridge.py, or emitter.py for basic
// family support.

impl TaskOp {
    /// Return the canonical family identifier string.
    ///
    /// This is the primary dispatch key used throughout the pipeline:
    /// bridge command selection, baseline dispatch, artifact naming, etc.
    pub fn family_id(&self) -> &'static str {
        match self {
            TaskOp::LinearProjection { .. } => "LinearProjection",
            TaskOp::ShardedLinearPipeline { .. } => "ShardedLinearPipeline",
            TaskOp::LutProjection { .. } => "LutProjection",
            TaskOp::DecodeStep { .. } => "DecodeStep",
            TaskOp::ShardedDecodeStep { .. } => "ShardedDecodeStep",
            TaskOp::MlpBlock { .. } => "MlpBlock",
            TaskOp::Attention { .. } => "Attention",
        }
    }

    /// Return the bridge command string for this family.
    ///
    /// The Python bridge dispatches on this string to select the correct
    /// emission handler. Each family gets a dedicated command.
    ///
    /// DecodeStep now dispatches to `emit_stateful_decode_step` by default
    /// (Sprint 40), which uses real `mb.read_state` / `mb.coreml_update_state`
    /// for KV-cache state semantics (iOS 18+). The previous stateless path
    /// (`emit_decode_step` using `mb.const` KV cache) is available as
    /// `emit_stateless_decode_step` for single-step testing.
    pub fn bridge_command(&self) -> &'static str {
        match self {
            TaskOp::LinearProjection { .. } => "emit_linear_projection",
            TaskOp::ShardedLinearPipeline { .. } => "emit_linear_projection",
            TaskOp::LutProjection { .. } => "emit_lut_projection",
            TaskOp::DecodeStep { .. } => "emit_stateful_decode_step",
            TaskOp::ShardedDecodeStep { .. } => "emit_shard_decode_step",
            TaskOp::MlpBlock { .. } => "emit_mlp_block",
            TaskOp::Attention { .. } => "emit_attention",
        }
    }

    /// Return a deterministic identity string for hash computation.
    ///
    /// This string uniquely identifies the task's operational parameters
    /// and is used for SHA-256 task hashing. The format is stable:
    /// "FamilyId:key1=val1,key2=val2,..."
    pub fn identity_string(&self) -> String {
        match self {
            TaskOp::LinearProjection { input_dim, output_dim, batch_size, has_bias, dtype } => {
                format!("LinearProjection:input_dim={},output_dim={},batch_size={},has_bias={},dtype={}",
                    input_dim, output_dim, batch_size, has_bias, dtype)
            }
            TaskOp::ShardedLinearPipeline {
                input_dim,
                hidden_dim,
                output_dim,
                batch_size,
                dtype,
            } => {
                format!("ShardedLinearPipeline:input_dim={},hidden_dim={},output_dim={},batch_size={},dtype={}",
                    input_dim, hidden_dim, output_dim, batch_size, dtype)
            }
            TaskOp::LutProjection {
                vocab_size,
                embed_dim,
                num_groups,
                lut_bitwidth,
                batch_size,
                dtype,
            } => {
                format!("LutProjection:vocab_size={},embed_dim={},num_groups={},lut_bitwidth={},batch_size={},dtype={}",
                    vocab_size, embed_dim, num_groups, lut_bitwidth, batch_size, dtype)
            }
            TaskOp::DecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                kv_heads,
                intermediate_size,
                vocab_size,
                dtype,
                uses_rope,
                has_qk_norm,
            } => {
                format!("DecodeStep:embed_dim={},num_heads={},head_dim={},kv_len={},batch_size={},kv_heads={},intermediate_size={},vocab_size={},dtype={},uses_rope={},has_qk_norm={}",
                    embed_dim, num_heads, head_dim, kv_len, batch_size, kv_heads, intermediate_size, vocab_size, dtype, uses_rope, has_qk_norm)
            }
            TaskOp::ShardedDecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                kv_heads,
                intermediate_size,
                vocab_size,
                dtype,
                uses_rope,
                has_qk_norm,
            } => {
                format!("ShardedDecodeStep:embed_dim={},num_heads={},head_dim={},kv_len={},batch_size={},kv_heads={},intermediate_size={},vocab_size={},dtype={},uses_rope={},has_qk_norm={}",
                    embed_dim, num_heads, head_dim, kv_len, batch_size, kv_heads, intermediate_size, vocab_size, dtype, uses_rope, has_qk_norm)
            }
            TaskOp::MlpBlock {
                input_dim,
                hidden_dim,
                output_dim,
                activation,
                batch_size,
                dtype,
            } => {
                format!("MlpBlock:input_dim={},hidden_dim={},output_dim={},activation={},batch_size={},dtype={}",
                    input_dim, hidden_dim, output_dim, activation, batch_size, dtype)
            }
            TaskOp::Attention { embed_dim, num_heads, head_dim, seq_len, batch_size, dtype } => {
                format!("Attention:embed_dim={},num_heads={},head_dim={},seq_len={},batch_size={},dtype={}",
                    embed_dim, num_heads, head_dim, seq_len, batch_size, dtype)
            }
        }
    }

    /// Return the primary dimensions for this task: (input_dim, output_dim, batch_size, dtype).
    ///
    /// For families where input_dim == output_dim == embed_dim (e.g., DecodeStep, Attention),
    /// embed_dim is returned for both input and output dimensions.
    pub fn primary_dims(&self) -> (usize, usize, usize, String) {
        match self {
            TaskOp::LinearProjection { input_dim, output_dim, batch_size, dtype, .. } => {
                (*input_dim, *output_dim, *batch_size, dtype.clone())
            }
            TaskOp::ShardedLinearPipeline { input_dim, output_dim, batch_size, dtype, .. } => {
                (*input_dim, *output_dim, *batch_size, dtype.clone())
            }
            TaskOp::LutProjection { embed_dim, batch_size, dtype, .. } => {
                (*embed_dim, *embed_dim, *batch_size, dtype.clone())
            }
            TaskOp::DecodeStep { embed_dim, batch_size, dtype, .. } => {
                (*embed_dim, *embed_dim, *batch_size, dtype.clone())
            }
            TaskOp::ShardedDecodeStep { embed_dim, batch_size, dtype, .. } => {
                (*embed_dim, *embed_dim, *batch_size, dtype.clone())
            }
            TaskOp::MlpBlock { input_dim, output_dim, batch_size, dtype, .. } => {
                (*input_dim, *output_dim, *batch_size, dtype.clone())
            }
            TaskOp::Attention { embed_dim, batch_size, dtype, .. } => {
                (*embed_dim, *embed_dim, *batch_size, dtype.clone())
            }
        }
    }

    /// Return a short operation type string for logging/display.
    pub fn op_type_str(&self) -> &'static str {
        match self {
            TaskOp::LinearProjection { .. } => "linear_projection",
            TaskOp::ShardedLinearPipeline { .. } => "sharded_linear_pipeline",
            TaskOp::LutProjection { .. } => "lut_projection",
            TaskOp::DecodeStep { .. } => "decode_step",
            TaskOp::ShardedDecodeStep { .. } => "sharded_decode_step",
            TaskOp::MlpBlock { .. } => "mlp_block",
            TaskOp::Attention { .. } => "attention",
        }
    }

    /// Return whether this is a sharded task type.
    pub fn is_sharded(&self) -> bool {
        matches!(self, TaskOp::ShardedLinearPipeline { .. } | TaskOp::ShardedDecodeStep { .. })
    }

    /// Return the family-specific parameters as a JSON value for bridge payload construction.
    ///
    /// This replaces the need for separate payload structs per family.
    /// Each variant serializes its unique fields into a serde_json::Value map.
    pub fn family_params(&self) -> serde_json::Value {
        match self {
            TaskOp::LinearProjection { input_dim, output_dim, batch_size, has_bias, dtype } => {
                serde_json::json!({
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "batch_size": batch_size,
                    "has_bias": has_bias,
                    "dtype": dtype,
                })
            }
            TaskOp::ShardedLinearPipeline {
                input_dim,
                hidden_dim,
                output_dim,
                batch_size,
                dtype,
            } => {
                serde_json::json!({
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "batch_size": batch_size,
                    "dtype": dtype,
                })
            }
            TaskOp::LutProjection {
                vocab_size,
                embed_dim,
                num_groups,
                lut_bitwidth,
                batch_size,
                dtype,
            } => {
                serde_json::json!({
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "num_groups": num_groups,
                    "lut_bitwidth": lut_bitwidth,
                    "batch_size": batch_size,
                    "dtype": dtype,
                })
            }
            TaskOp::DecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                kv_heads,
                intermediate_size,
                vocab_size,
                dtype,
                uses_rope,
                has_qk_norm,
            } => {
                serde_json::json!({
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "kv_len": kv_len,
                    "batch_size": batch_size,
                    "kv_heads": kv_heads,
                    "intermediate_size": intermediate_size,
                    "vocab_size": vocab_size,
                    "dtype": dtype,
                    "uses_rope": uses_rope,
                    "has_qk_norm": has_qk_norm,
                })
            }
            TaskOp::ShardedDecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                kv_heads,
                intermediate_size,
                vocab_size,
                dtype,
                uses_rope,
                has_qk_norm,
            } => {
                serde_json::json!({
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "kv_len": kv_len,
                    "batch_size": batch_size,
                    "kv_heads": kv_heads,
                    "intermediate_size": intermediate_size,
                    "vocab_size": vocab_size,
                    "dtype": dtype,
                    "uses_rope": uses_rope,
                    "has_qk_norm": has_qk_norm,
                })
            }
            TaskOp::MlpBlock {
                input_dim,
                hidden_dim,
                output_dim,
                activation,
                batch_size,
                dtype,
            } => {
                serde_json::json!({
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "activation": activation,
                    "batch_size": batch_size,
                    "dtype": dtype,
                })
            }
            TaskOp::Attention { embed_dim, num_heads, head_dim, seq_len, batch_size, dtype } => {
                serde_json::json!({
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "dtype": dtype,
                })
            }
        }
    }

    /// Return the input tensor shape for bridge payload function descriptors.
    pub fn input_tensor_shape(&self) -> Vec<usize> {
        match self {
            TaskOp::LutProjection { batch_size, .. } => vec![*batch_size],
            TaskOp::Attention { batch_size, seq_len, embed_dim, .. } => {
                vec![*batch_size, *seq_len, *embed_dim]
            }
            _ => {
                let (input_dim, _, batch_size, _) = self.primary_dims();
                vec![batch_size, input_dim]
            }
        }
    }

    /// Return the output tensor shape for bridge payload function descriptors.
    pub fn output_tensor_shape(&self) -> Vec<usize> {
        match self {
            TaskOp::Attention { batch_size, seq_len, embed_dim, .. } => {
                vec![*batch_size, *seq_len, *embed_dim]
            }
            _ => {
                let (_, output_dim, batch_size, _) = self.primary_dims();
                vec![batch_size, output_dim]
            }
        }
    }

    /// Return the input tensor name for bridge payload function descriptors.
    pub fn input_tensor_name(&self) -> &'static str {
        match self {
            TaskOp::LutProjection { .. } => "indices",
            _ => "x",
        }
    }

    /// Return the input tensor dtype for bridge payload function descriptors.
    pub fn input_tensor_dtype(&self) -> String {
        match self {
            TaskOp::LutProjection { .. } => "int32".to_string(),
            _ => self.primary_dims().3,
        }
    }
}

/// Measurement configuration for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConfig {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub metrics: Vec<String>,
}

/// Load a synthetic task spec from a TOML file.
pub fn load_synthetic_task(path: &str) -> Result<SyntheticTaskSpec, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read task file '{}': {}", path, e))?;
    parse_synthetic_task(&content)
}

/// Type alias for a TOML family parser function.
type FamilyParser = fn(&toml::Value, &toml::Value) -> Result<SyntheticTaskSpec, String>;

/// Registry of TOML section names to their parser functions.
///
/// This is the single source of truth for which task families can be
/// parsed from TOML. Adding a new family requires only:
/// 1. Adding the parser function
/// 2. Adding an entry to this registry
///
/// No changes needed in parse_synthetic_task itself.
const FAMILY_PARSERS: &[(&str, FamilyParser)] = &[
    ("sharded_linear_pipeline", parse_sharded_pipeline as FamilyParser),
    ("sharded_decode_step", parse_sharded_decode_step as FamilyParser),
    ("attention", parse_attention as FamilyParser),
    ("mlp_block", parse_mlp_block as FamilyParser),
    ("decode_step", parse_decode_step as FamilyParser),
    ("lut_projection", parse_lut_projection as FamilyParser),
];

/// Parse a synthetic task spec from TOML text.
///
/// Uses a registry of family parsers to dispatch to the correct
/// parser based on the TOML section name. Adding a new family
/// only requires adding an entry to FAMILY_PARSERS and implementing
/// the parser function — no changes needed here.
pub fn parse_synthetic_task(toml_text: &str) -> Result<SyntheticTaskSpec, String> {
    let value: toml::Value =
        toml::from_str(toml_text).map_err(|e| format!("TOML parse error: {}", e))?;

    // Navigate the synthetic task section
    let synth = value.get("synthetic").ok_or("Missing [synthetic] section")?;

    // Try each registered family parser
    for (section_name, parser) in FAMILY_PARSERS {
        if let Some(task_section) = synth.get(section_name) {
            return parser(task_section, &value);
        }
    }

    // Fall through to original linear_projection format (legacy cases-based TOML)
    if synth.get("linear_projection").is_some() {
        return parse_linear_projection_legacy(synth, &value);
    }

    // Build a helpful error message listing all valid section names
    let valid_sections: Vec<&str> = FAMILY_PARSERS
        .iter()
        .map(|(name, _)| *name)
        .chain(std::iter::once("linear_projection"))
        .collect();
    Err(format!(
        "Missing [synthetic.<family>] section. Valid sections: {}",
        valid_sections.join(", ")
    ))
}

/// Parse the legacy linear_projection format with [[cases]] arrays.
///
/// This format predates the registry-based approach and uses a
/// different structure (nested cases arrays) from the other families.
fn parse_linear_projection_legacy(
    synth: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let task_section =
        synth.get("linear_projection").ok_or("Missing [synthetic.linear_projection] section")?;

    let name = task_section.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed").to_string();

    let family = task_section
        .get("family")
        .and_then(|v| v.as_str())
        .unwrap_or("LinearProjection")
        .to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    // Extract the first case for the vertical slice
    let cases = task_section
        .get("cases")
        .ok_or("Missing [[synthetic.linear_projection.cases]]")?
        .as_array()
        .ok_or("cases must be an array")?;

    if cases.is_empty() {
        return Err("No cases defined".into());
    }

    let case = &cases[0];
    let input_shape =
        case.get("input_shape").and_then(|v| v.as_array()).ok_or("Missing input_shape in case")?;
    let weight_shape = case
        .get("weight_shape")
        .and_then(|v| v.as_array())
        .ok_or("Missing weight_shape in case")?;

    let input_dim =
        input_shape.get(1).and_then(|v| v.as_integer()).ok_or("input_shape must have 2 elements")?
            as usize;
    let output_dim = weight_shape
        .get(1)
        .and_then(|v| v.as_integer())
        .ok_or("weight_shape must have 2 elements")? as usize;
    let batch_size = input_shape.first().and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = case.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    let op = TaskOp::LinearProjection { input_dim, output_dim, batch_size, has_bias: true, dtype };

    // Parse measurement section
    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse a sharded linear pipeline task from the TOML `[synthetic.sharded_linear_pipeline]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.sharded_linear_pipeline]
/// name = "sharded_linear_3shard"
/// family = "ShardedLinearPipeline"
/// description = "..."
/// input_dim = 64
/// hidden_dim = 48
/// output_dim = 32
/// batch_size = 1
/// dtype = "fp16"
/// ```
/// Parse an attention task from the TOML `[synthetic.attention]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.attention]
/// name = "attention_128h4"
/// family = "Attention"
/// description = "128-dim 4-head self-attention"
/// embed_dim = 128
/// num_heads = 4
/// seq_len = 32
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_attention(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name = task_section.get("name").and_then(|v| v.as_str()).unwrap_or("attention").to_string();

    let family =
        task_section.get("family").and_then(|v| v.as_str()).unwrap_or("Attention").to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let embed_dim = task_section
        .get("embed_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing embed_dim in attention")? as usize;
    let num_heads = task_section
        .get("num_heads")
        .and_then(|v| v.as_integer())
        .ok_or("Missing num_heads in attention")? as usize;
    let seq_len = task_section
        .get("seq_len")
        .and_then(|v| v.as_integer())
        .ok_or("Missing seq_len in attention")? as usize;
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    // Validate embed_dim / num_heads divisibility
    if num_heads == 0 {
        return Err("num_heads must be > 0 in attention".to_string());
    }
    if !embed_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "Invalid attention config: embed_dim {} is not divisible by num_heads {}",
            embed_dim, num_heads
        ));
    }
    let head_dim = embed_dim / num_heads;

    let op = TaskOp::Attention { embed_dim, num_heads, head_dim, seq_len, batch_size, dtype };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse an MLP block task from the TOML `[synthetic.mlp_block]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.mlp_block]
/// name = "mlp_block_128_512_128"
/// family = "MlpBlock"
/// description = "128-dim MLP block with GELU activation"
/// input_dim = 128
/// hidden_dim = 512
/// output_dim = 128
/// activation = "gelu"
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_mlp_block(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name = task_section.get("name").and_then(|v| v.as_str()).unwrap_or("mlp_block").to_string();

    let family =
        task_section.get("family").and_then(|v| v.as_str()).unwrap_or("MlpBlock").to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let input_dim = task_section
        .get("input_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing input_dim in mlp_block")? as usize;
    let hidden_dim = task_section
        .get("hidden_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing hidden_dim in mlp_block")? as usize;
    let output_dim = task_section
        .get("output_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing output_dim in mlp_block")? as usize;
    let activation =
        task_section.get("activation").and_then(|v| v.as_str()).unwrap_or("gelu").to_string();
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    // Validate activation
    if activation != "gelu" && activation != "relu" {
        return Err(format!("Invalid activation '{}': must be 'gelu' or 'relu'", activation));
    }

    let op = TaskOp::MlpBlock { input_dim, hidden_dim, output_dim, activation, batch_size, dtype };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse a sharded linear pipeline task from the TOML `[synthetic.sharded_linear_pipeline]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.sharded_linear_pipeline]
/// name = "sharded_linear_3shard"
/// family = "ShardedLinearPipeline"
/// description = "..."
/// input_dim = 64
/// hidden_dim = 48
/// output_dim = 32
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_sharded_pipeline(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name = task_section
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("sharded_linear_pipeline")
        .to_string();

    let family = task_section
        .get("family")
        .and_then(|v| v.as_str())
        .unwrap_or("ShardedLinearPipeline")
        .to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let input_dim = task_section
        .get("input_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing input_dim in sharded_linear_pipeline")? as usize;
    let hidden_dim = task_section
        .get("hidden_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing hidden_dim in sharded_linear_pipeline")? as usize;
    let output_dim = task_section
        .get("output_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing output_dim in sharded_linear_pipeline")? as usize;
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    let op = TaskOp::ShardedLinearPipeline { input_dim, hidden_dim, output_dim, batch_size, dtype };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse the optional `[measurement]` section from the TOML root.
fn parse_measurement(root: &toml::Value) -> MeasurementConfig {
    root.get("measurement")
        .map(|m| MeasurementConfig {
            warmup_iterations: m.get("warmup_iterations").and_then(|v| v.as_integer()).unwrap_or(5)
                as usize,
            measured_iterations: m
                .get("measured_iterations")
                .and_then(|v| v.as_integer())
                .unwrap_or(20) as usize,
            metrics: m
                .get("metrics")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        })
        .unwrap_or(MeasurementConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            metrics: vec!["Latency".into()],
        })
}

/// Parse a LUT projection task from the TOML `[synthetic.lut_projection]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.lut_projection]
/// name = "lut_proj_4bit"
/// family = "LutProjection"
/// description = "..."
/// vocab_size = 32000
/// embed_dim = 512
/// num_groups = 64
/// lut_bitwidth = 4
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_lut_projection(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name =
        task_section.get("name").and_then(|v| v.as_str()).unwrap_or("lut_projection").to_string();

    let family =
        task_section.get("family").and_then(|v| v.as_str()).unwrap_or("LutProjection").to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let vocab_size = task_section
        .get("vocab_size")
        .and_then(|v| v.as_integer())
        .ok_or("Missing vocab_size in lut_projection")? as usize;
    let embed_dim = task_section
        .get("embed_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing embed_dim in lut_projection")? as usize;
    let num_groups = task_section
        .get("num_groups")
        .and_then(|v| v.as_integer())
        .ok_or("Missing num_groups in lut_projection")? as usize;
    let lut_bitwidth = task_section
        .get("lut_bitwidth")
        .and_then(|v| v.as_integer())
        .ok_or("Missing lut_bitwidth in lut_projection")? as usize;
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    // T-64 (I-38): Use centralized palette bit-width validation
    // from ane_ir::ane_layout instead of inline matches! pattern.
    crate::ane_layout::validate_palette_bits(lut_bitwidth)?;

    let op = TaskOp::LutProjection {
        vocab_size,
        embed_dim,
        num_groups,
        lut_bitwidth,
        batch_size,
        dtype,
    };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse a decode step task from the TOML `[synthetic.decode_step]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.decode_step]
/// name = "decode_step_128h4"
/// family = "DecodeStep"
/// description = "..."
/// embed_dim = 128
/// num_heads = 4
/// kv_len = 32
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_decode_step(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name =
        task_section.get("name").and_then(|v| v.as_str()).unwrap_or("decode_step").to_string();

    let family =
        task_section.get("family").and_then(|v| v.as_str()).unwrap_or("DecodeStep").to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let embed_dim = task_section
        .get("embed_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing embed_dim in decode_step")? as usize;
    let num_heads = task_section
        .get("num_heads")
        .and_then(|v| v.as_integer())
        .ok_or("Missing num_heads in decode_step")? as usize;
    let kv_len = task_section
        .get("kv_len")
        .and_then(|v| v.as_integer())
        .ok_or("Missing kv_len in decode_step")? as usize;
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    // Validate embed_dim / num_heads divisibility
    if num_heads == 0 {
        return Err("num_heads must be > 0 in decode_step".to_string());
    }
    if !embed_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "Invalid decode_step config: embed_dim {} is not divisible by num_heads {}",
            embed_dim, num_heads
        ));
    }
    let head_dim = embed_dim / num_heads;

    let op = TaskOp::DecodeStep {
        embed_dim,
        num_heads,
        head_dim,
        kv_len,
        batch_size,
        kv_heads: num_heads,
        intermediate_size: embed_dim * 4,
        vocab_size: 0,
        dtype,
        uses_rope: true,
        has_qk_norm: false,
    };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

/// Parse a sharded decode step task from the TOML `[synthetic.sharded_decode_step]` section.
///
/// Expected TOML format:
/// ```toml
/// [synthetic.sharded_decode_step]
/// name = "sharded_decode_128h4"
/// family = "ShardedDecodeStep"
/// description = "..."
/// embed_dim = 128
/// num_heads = 4
/// kv_len = 32
/// batch_size = 1
/// dtype = "fp16"
/// ```
fn parse_sharded_decode_step(
    task_section: &toml::Value,
    root: &toml::Value,
) -> Result<SyntheticTaskSpec, String> {
    let name = task_section
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("sharded_decode_step")
        .to_string();

    let family = task_section
        .get("family")
        .and_then(|v| v.as_str())
        .unwrap_or("ShardedDecodeStep")
        .to_string();

    let description =
        task_section.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());

    let embed_dim = task_section
        .get("embed_dim")
        .and_then(|v| v.as_integer())
        .ok_or("Missing embed_dim in sharded_decode_step")? as usize;
    let num_heads = task_section
        .get("num_heads")
        .and_then(|v| v.as_integer())
        .ok_or("Missing num_heads in sharded_decode_step")? as usize;
    let kv_len = task_section
        .get("kv_len")
        .and_then(|v| v.as_integer())
        .ok_or("Missing kv_len in sharded_decode_step")? as usize;
    let batch_size =
        task_section.get("batch_size").and_then(|v| v.as_integer()).unwrap_or(1) as usize;
    let dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string();

    // Validate embed_dim / num_heads divisibility
    if num_heads == 0 {
        return Err("num_heads must be > 0 in sharded_decode_step".to_string());
    }
    if !embed_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "Invalid sharded_decode_step config: embed_dim {} is not divisible by num_heads {}",
            embed_dim, num_heads
        ));
    }
    let head_dim = embed_dim / num_heads;

    let op = TaskOp::ShardedDecodeStep {
        embed_dim,
        num_heads,
        head_dim,
        kv_len,
        batch_size,
        kv_heads: num_heads,
        intermediate_size: embed_dim * 4,
        vocab_size: 0,
        dtype,
        uses_rope: true,
        has_qk_norm: false,
    };

    let measurement = parse_measurement(root);

    Ok(SyntheticTaskSpec { name, family, description, op, measurement })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sharded_linear_pipeline() {
        let toml = r#"
[synthetic.sharded_linear_pipeline]
name = "test_shard"
family = "ShardedLinearPipeline"
description = "test"
input_dim = 64
hidden_dim = 48
output_dim = 32
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "test_shard");
        assert_eq!(spec.family, "ShardedLinearPipeline");
        match spec.op {
            TaskOp::ShardedLinearPipeline {
                input_dim,
                hidden_dim,
                output_dim,
                batch_size,
                dtype,
            } => {
                assert_eq!(input_dim, 64);
                assert_eq!(hidden_dim, 48);
                assert_eq!(output_dim, 32);
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected ShardedLinearPipeline, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_sharded_pipeline_missing_dims() {
        let toml = r#"
[synthetic.sharded_linear_pipeline]
name = "bad_shard"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing dimensions");
    }

    #[test]
    fn test_parse_linear_projection_still_works() {
        let toml = r#"
[synthetic.linear_projection]
name = "test_lp"
family = "LinearProjection"

[[synthetic.linear_projection.cases]]
name = "64x32"
input_shape = [1, 64]
weight_shape = [64, 32]
dtype = "fp16"
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "test_lp");
        match spec.op {
            TaskOp::LinearProjection { input_dim, output_dim, .. } => {
                assert_eq!(input_dim, 64);
                assert_eq!(output_dim, 32);
            }
            _ => panic!("Expected LinearProjection"),
        }
    }

    #[test]
    fn test_parse_lut_projection() {
        let toml = r#"
[synthetic.lut_projection]
name = "test_lut"
family = "LutProjection"
description = "4-bit LUT projection test"
vocab_size = 32000
embed_dim = 512
num_groups = 64
lut_bitwidth = 4
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "test_lut");
        assert_eq!(spec.family, "LutProjection");
        match spec.op {
            TaskOp::LutProjection {
                vocab_size,
                embed_dim,
                num_groups,
                lut_bitwidth,
                batch_size,
                dtype,
            } => {
                assert_eq!(vocab_size, 32000);
                assert_eq!(embed_dim, 512);
                assert_eq!(num_groups, 64);
                assert_eq!(lut_bitwidth, 4);
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected LutProjection, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_lut_projection_invalid_bitwidth() {
        let toml = r#"
[synthetic.lut_projection]
name = "bad_lut"
vocab_size = 1000
embed_dim = 128
num_groups = 16
lut_bitwidth = 5
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with invalid bitwidth 5");
    }

    #[test]
    fn test_parse_lut_projection_missing_fields() {
        let toml = r#"
[synthetic.lut_projection]
name = "bad_lut"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing required fields");
    }

    #[test]
    fn test_parse_decode_step() {
        let toml = r#"
[synthetic.decode_step]
name = "test_decode"
family = "DecodeStep"
description = "128-dim decode step test"
embed_dim = 128
num_heads = 4
kv_len = 32
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "test_decode");
        assert_eq!(spec.family, "DecodeStep");
        match spec.op {
            TaskOp::DecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                dtype,
                ..
            } => {
                assert_eq!(embed_dim, 128);
                assert_eq!(num_heads, 4);
                assert_eq!(head_dim, 32); // 128 / 4
                assert_eq!(kv_len, 32);
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected DecodeStep, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_decode_step_invalid_divisibility() {
        let toml = r#"
[synthetic.decode_step]
name = "bad_decode"
embed_dim = 127
num_heads = 4
kv_len = 32
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with embed_dim not divisible by num_heads");
    }

    #[test]
    fn test_parse_decode_step_missing_fields() {
        let toml = r#"
[synthetic.decode_step]
name = "bad_decode"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing required fields");
    }

    #[test]
    fn test_parse_sharded_decode_step() {
        let toml = r#"
[synthetic.sharded_decode_step]
name = "test_sharded_decode"
family = "ShardedDecodeStep"
description = "128-dim sharded decode step"
embed_dim = 128
num_heads = 4
kv_len = 32
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "test_sharded_decode");
        assert_eq!(spec.family, "ShardedDecodeStep");
        match spec.op {
            TaskOp::ShardedDecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                dtype,
                ..
            } => {
                assert_eq!(embed_dim, 128);
                assert_eq!(num_heads, 4);
                assert_eq!(head_dim, 32); // 128 / 4
                assert_eq!(kv_len, 32);
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected ShardedDecodeStep, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_sharded_decode_step_invalid_divisibility() {
        let toml = r#"
[synthetic.sharded_decode_step]
name = "bad_sharded_decode"
embed_dim = 127
num_heads = 4
kv_len = 32
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with embed_dim not divisible by num_heads");
    }

    #[test]
    fn test_parse_sharded_decode_step_missing_fields() {
        let toml = r#"
[synthetic.sharded_decode_step]
name = "bad_sharded_decode"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing required fields");
    }

    #[test]
    fn test_parse_mlp_block() {
        let toml = r#"
[synthetic.mlp_block]
name = "mlp_block_128_512_128"
family = "MlpBlock"
description = "128-dim MLP block with GELU activation"
input_dim = 128
hidden_dim = 512
output_dim = 128
activation = "gelu"
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "mlp_block_128_512_128");
        assert_eq!(spec.family, "MlpBlock");
        match spec.op {
            TaskOp::MlpBlock {
                input_dim,
                hidden_dim,
                output_dim,
                activation,
                batch_size,
                dtype,
            } => {
                assert_eq!(input_dim, 128);
                assert_eq!(hidden_dim, 512);
                assert_eq!(output_dim, 128);
                assert_eq!(activation, "gelu");
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected MlpBlock, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_mlp_block_invalid_activation() {
        let toml = r#"
[synthetic.mlp_block]
name = "bad_mlp"
input_dim = 128
hidden_dim = 512
output_dim = 128
activation = "sigmoid"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with invalid activation");
    }

    #[test]
    fn test_parse_mlp_block_missing_fields() {
        let toml = r#"
[synthetic.mlp_block]
name = "bad_mlp"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing required fields");
    }

    #[test]
    fn test_parse_attention() {
        let toml = r#"
[synthetic.attention]
name = "attention_128h4"
family = "Attention"
description = "128-dim 4-head self-attention"
embed_dim = 128
num_heads = 4
seq_len = 32
batch_size = 1
dtype = "fp16"

[measurement]
warmup_iterations = 3
measured_iterations = 10
metrics = ["Latency"]
"#;
        let spec = parse_synthetic_task(toml).unwrap();
        assert_eq!(spec.name, "attention_128h4");
        assert_eq!(spec.family, "Attention");
        match spec.op {
            TaskOp::Attention { embed_dim, num_heads, head_dim, seq_len, batch_size, dtype } => {
                assert_eq!(embed_dim, 128);
                assert_eq!(num_heads, 4);
                assert_eq!(head_dim, 32); // 128 / 4
                assert_eq!(seq_len, 32);
                assert_eq!(batch_size, 1);
                assert_eq!(dtype, "fp16");
            }
            _ => panic!("Expected Attention, got {:?}", spec.op),
        }
    }

    #[test]
    fn test_parse_attention_invalid_divisibility() {
        let toml = r#"
[synthetic.attention]
name = "bad_attention"
embed_dim = 127
num_heads = 4
seq_len = 32
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with embed_dim not divisible by num_heads");
    }

    #[test]
    fn test_parse_attention_missing_fields() {
        let toml = r#"
[synthetic.attention]
name = "bad_attention"
"#;
        let result = parse_synthetic_task(toml);
        assert!(result.is_err(), "Should fail with missing required fields");
    }
}
