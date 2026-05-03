//! Shared type definitions across SIR/AIR/MIR.
//!
//! This module centralizes types that are used by multiple IR levels,
//! reducing duplication and ensuring consistency.

use serde::{Deserialize, Serialize};

// ─── Data Types ───────────────────────────────────────────────────

/// MIL data type enum shared across all IR levels.
///
/// Sprint 58 (S58.2): MilDtypeRepr was removed. All IR levels now
/// use this single unified type.
///
/// T-35 (I-14): Added Int4, UInt4, E4M3, E5M2, UInt16 for proper
/// quantization and float8 constraint enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MilDtype {
    Fp16,
    Fp32,
    Int32,
    UInt8,
    Bool,
    Fp64,
    Int8,
    Int16,
    /// 4-bit signed integer. Used for palettized/quantized weights.
    /// ANE constraint: must use interleave factor 8.
    /// ANE constraint: Int4 per-cout dequant is NOT supported.
    Int4,
    /// 4-bit unsigned integer. Used for palettized weights.
    UInt4,
    /// 8-bit floating point (4-bit exponent, 3-bit mantissa).
    /// ANE constraint: architecture-dependent support.
    /// NOT supported on most families; limited support on A17/A18.
    /// ANE constraint: zero point is NOT supported for E4M3 quant.
    E4M3,
    /// 8-bit floating point (5-bit exponent, 2-bit mantissa).
    /// ANE constraint: NOT supported on ANE ("E4M3 or E5M2 format not supported").
    E5M2,
    /// 16-bit unsigned integer.
    UInt16,
}

// ─── Compute Unit Hints ──────────────────────────────────────────

/// Compute unit hint for MIR nodes and PIR packages.
///
/// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs
/// to become the unified compute unit representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeUnitHint {
    CPUAndNE,
    CPUAndGPU,
    CPUOnly,
    All,
}

impl ComputeUnitHint {
    /// Parse compute unit hint from the string representation used in
    /// task specs, bridge payloads, and shard template seeds.
    ///
    /// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs.
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s {
            "CPU_AND_NE" | "CPUAndNE" => Some(ComputeUnitHint::CPUAndNE),
            "CPU_AND_GPU" | "CPUAndGPU" => Some(ComputeUnitHint::CPUAndGPU),
            "CPU_ONLY" | "CPUOnly" => Some(ComputeUnitHint::CPUOnly),
            "ALL" | "All" => Some(ComputeUnitHint::All),
            _ => None,
        }
    }

    /// Returns the Core ML compatible string for this compute unit setting.
    ///
    /// Sprint 58 (S58.3): moved from the removed `ComputeUnits` type in pir.rs.
    pub fn to_coreml_string(&self) -> &'static str {
        match self {
            ComputeUnitHint::CPUAndNE => "CPU_AND_NE",
            ComputeUnitHint::CPUAndGPU => "CPU_AND_GPU",
            ComputeUnitHint::CPUOnly => "CPU_ONLY",
            ComputeUnitHint::All => "ALL",
        }
    }
}

// ─── Model Architecture Configuration ────────────────────────────

/// Model architecture identifier for weight-name pattern resolution.
///
/// Different transformer architectures use different weight naming conventions
/// in their HuggingFace checkpoints. This enum allows the compiler to
/// correctly resolve weight names and build input alias maps without
/// hardcoding architecture-specific patterns (CQ-18).
///
/// Qwen3 and LLaMA share the same naming convention (`q_proj`, `k_proj`,
/// `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), so they are
/// grouped together. Other architectures (GPT-2, BART, T5, etc.) use
/// different patterns and require the `Generic` variant with explicit
/// pattern strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Qwen3 family (also covers LLaMA — same weight naming convention).
    ///
    /// Weight patterns:
    /// - `.self_attn.q_proj.weight`, `.self_attn.k_proj.weight`,
    ///   `.self_attn.v_proj.weight`, `.self_attn.o_proj.weight`
    /// - `.mlp.gate_proj.weight`, `.mlp.up_proj.weight`,
    ///   `.mlp.down_proj.weight`
    Qwen3,
    /// Generic architecture with explicitly-provided weight name patterns.
    Generic {
        /// Substring pattern for Q projection weights (e.g., ".attn.q_proj.weight")
        q_proj_pattern: String,
        /// Substring pattern for K projection weights
        k_proj_pattern: String,
        /// Substring pattern for V projection weights
        v_proj_pattern: String,
        /// Substring pattern for output projection weights
        o_proj_pattern: String,
        /// Substring pattern for MLP gate projection weights
        gate_proj_pattern: String,
        /// Substring pattern for MLP up projection weights
        up_proj_pattern: String,
        /// Substring pattern for MLP down projection weights
        down_proj_pattern: String,
    },
}

impl ModelArchitecture {
    /// Returns the Q projection weight substring pattern for this architecture.
    pub fn q_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".self_attn.q_proj.weight",
            ModelArchitecture::Generic { q_proj_pattern, .. } => q_proj_pattern,
        }
    }

    /// Returns the K projection weight substring pattern for this architecture.
    pub fn k_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".self_attn.k_proj.weight",
            ModelArchitecture::Generic { k_proj_pattern, .. } => k_proj_pattern,
        }
    }

    /// Returns the V projection weight substring pattern for this architecture.
    pub fn v_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".self_attn.v_proj.weight",
            ModelArchitecture::Generic { v_proj_pattern, .. } => v_proj_pattern,
        }
    }

    /// Returns the output projection weight substring pattern for this architecture.
    pub fn o_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".self_attn.o_proj.weight",
            ModelArchitecture::Generic { o_proj_pattern, .. } => o_proj_pattern,
        }
    }

    /// Returns the MLP gate projection weight substring pattern.
    pub fn gate_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".mlp.gate_proj.weight",
            ModelArchitecture::Generic { gate_proj_pattern, .. } => gate_proj_pattern,
        }
    }

    /// Returns the MLP up projection weight substring pattern.
    pub fn up_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".mlp.up_proj.weight",
            ModelArchitecture::Generic { up_proj_pattern, .. } => up_proj_pattern,
        }
    }

    /// Returns the MLP down projection weight substring pattern.
    pub fn down_proj_pattern(&self) -> &str {
        match self {
            ModelArchitecture::Qwen3 => ".mlp.down_proj.weight",
            ModelArchitecture::Generic { down_proj_pattern, .. } => down_proj_pattern,
        }
    }

    /// Returns true if this architecture uses the Qwen3/LLaMA naming convention.
    pub fn is_qwen3_like(&self) -> bool {
        matches!(self, ModelArchitecture::Qwen3)
    }
}

/// Model architecture configuration for the compilation pipeline.
///
/// Carries model-specific constants that were previously hardcoded throughout
/// the codebase (CQ-16: `vocab_size=32000`, CQ-17: `head_dim=128` fallback,
/// CQ-18: Qwen3 weight patterns, CQ-19: `vec![1, 512]` shape fallback).
///
/// Production compilation always provides this configuration. When absent,
/// the compiler errors rather than silently using wrong defaults — a
/// deliberate design choice to prevent miscompilation for non-Qwen3 models.
///
/// This struct is separate from `trace::ModelConfig` (which captures
/// HuggingFace config.json fields for tracing). `ModelArchConfig` focuses
/// on the values needed during the compilation pipeline.
///
/// # T-36 (I-15)
///
/// All hardcoded model-specific constants have been moved into this struct.
/// The previous fallback values were:
/// - `vocab_size = 32000` (LLaMA-2 default, wrong for Qwen3's 151936)
/// - `head_dim = 128` (fallback, silently produces wrong attention scale)
/// - `embed_dim = 128` (fallback in role_mir.rs, wrong for most models)
/// - `max_seq_len = 512` (shape inference fallback, Qwen3-0.6B assumption)
/// - Qwen3 weight patterns hardcoded in `build_input_alias_map`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchConfig {
    /// Vocabulary size for the language model head.
    /// Qwen3-0.6B: 151936, LLaMA-2: 32000, GPT-2: 50257.
    pub vocab_size: usize,
    /// Embedding dimension (hidden_size).
    /// Qwen3-0.6B: 1024, LLaMA-2-7B: 4096.
    pub embed_dim: usize,
    /// Dimension per attention head.
    /// Qwen3-0.6B: 128, LLaMA-2-7B: 128.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads for GQA (0 = use num_heads).
    pub kv_heads: usize,
    /// MLP intermediate size.
    /// Qwen3-0.6B: 2048, LLaMA-2-7B: 11008.
    pub intermediate_size: usize,
    /// Maximum sequence length for shape inference fallbacks.
    /// Qwen3-0.6B: 32768, LLaMA-2: 4096.
    pub max_seq_len: usize,
    /// Model architecture type for weight-name pattern resolution.
    pub architecture: ModelArchitecture,
}

impl Default for ModelArchConfig {
    /// Default configuration for Qwen3-0.6B (the primary compilation target).
    ///
    /// This default exists for backward compatibility with existing callers
    /// that do not yet pass an explicit config. New code should prefer
    /// constructing `ModelArchConfig` explicitly from model metadata.
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            embed_dim: 1024,
            head_dim: 128,
            num_heads: 16,
            kv_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 32768,
            architecture: ModelArchitecture::Qwen3,
        }
    }
}

impl ModelArchConfig {
    /// Construct from trace::ModelConfig fields.
    ///
    /// This is the primary factory method used by the CLI to build
    /// a compilation-ready config from the traced model's metadata.
    pub fn from_model_config(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        intermediate_size: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
        model_type: &str,
    ) -> Self {
        let kv_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = head_dim.unwrap_or_else(|| hidden_size / num_attention_heads);

        // Determine architecture from model_type string.
        // HuggingFace model_type values:
        //   "qwen2" / "qwen3" → Qwen3 (same naming convention)
        //   "llama" → Qwen3 (LLaMA shares the same weight naming)
        //   anything else → Generic with Qwen3 patterns as starting point
        let architecture = match model_type {
            "qwen2" | "qwen3" | "llama" => ModelArchitecture::Qwen3,
            _ => ModelArchitecture::Generic {
                q_proj_pattern: format!(".self_attn.q_proj.weight"),
                k_proj_pattern: format!(".self_attn.k_proj.weight"),
                v_proj_pattern: format!(".self_attn.v_proj.weight"),
                o_proj_pattern: format!(".self_attn.o_proj.weight"),
                gate_proj_pattern: format!(".mlp.gate_proj.weight"),
                up_proj_pattern: format!(".mlp.up_proj.weight"),
                down_proj_pattern: format!(".mlp.down_proj.weight"),
            },
        };

        Self {
            vocab_size,
            embed_dim: hidden_size,
            head_dim,
            num_heads: num_attention_heads,
            kv_heads,
            intermediate_size,
            max_seq_len: max_position_embeddings,
            architecture,
        }
    }

    /// Returns the effective KV head count (kv_heads if set, else num_heads).
    pub fn effective_kv_heads(&self) -> usize {
        if self.kv_heads > 0 { self.kv_heads } else { self.num_heads }
    }

    /// Returns the attention scale factor: 1/√head_dim.
    ///
    /// Replaces the hardcoded `1.0 / (128.0_f32).sqrt()` fallbacks
    /// in legality_rewrite.rs (CQ-17).
    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ModelArchitecture ──────────────────────────────────────────────

    #[test]
    fn test_qwen3_patterns() {
        let arch = ModelArchitecture::Qwen3;
        assert_eq!(arch.q_proj_pattern(), ".self_attn.q_proj.weight");
        assert_eq!(arch.k_proj_pattern(), ".self_attn.k_proj.weight");
        assert_eq!(arch.v_proj_pattern(), ".self_attn.v_proj.weight");
        assert_eq!(arch.o_proj_pattern(), ".self_attn.o_proj.weight");
        assert_eq!(arch.gate_proj_pattern(), ".mlp.gate_proj.weight");
        assert_eq!(arch.up_proj_pattern(), ".mlp.up_proj.weight");
        assert_eq!(arch.down_proj_pattern(), ".mlp.down_proj.weight");
        assert!(arch.is_qwen3_like());
    }

    #[test]
    fn test_generic_patterns() {
        let arch = ModelArchitecture::Generic {
            q_proj_pattern: ".attn.q.weight".to_string(),
            k_proj_pattern: ".attn.k.weight".to_string(),
            v_proj_pattern: ".attn.v.weight".to_string(),
            o_proj_pattern: ".attn.o.weight".to_string(),
            gate_proj_pattern: ".mlp.gate.weight".to_string(),
            up_proj_pattern: ".mlp.up.weight".to_string(),
            down_proj_pattern: ".mlp.down.weight".to_string(),
        };
        assert_eq!(arch.q_proj_pattern(), ".attn.q.weight");
        assert_eq!(arch.k_proj_pattern(), ".attn.k.weight");
        assert_eq!(arch.v_proj_pattern(), ".attn.v.weight");
        assert_eq!(arch.o_proj_pattern(), ".attn.o.weight");
        assert_eq!(arch.gate_proj_pattern(), ".mlp.gate.weight");
        assert_eq!(arch.up_proj_pattern(), ".mlp.up.weight");
        assert_eq!(arch.down_proj_pattern(), ".mlp.down.weight");
        assert!(!arch.is_qwen3_like());
    }

    // ─── ModelArchConfig ────────────────────────────────────────────────

    #[test]
    fn test_default_is_qwen3_0_6b() {
        let config = ModelArchConfig::default();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.embed_dim, 1024);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.kv_heads, 8);
        assert_eq!(config.intermediate_size, 2048);
        assert_eq!(config.max_seq_len, 32768);
        assert_eq!(config.architecture, ModelArchitecture::Qwen3);
    }

    #[test]
    fn test_from_model_config_qwen3() {
        let config = ModelArchConfig::from_model_config(
            1024,   // hidden_size
            16,     // num_attention_heads
            Some(8),// num_key_value_heads
            Some(128),// head_dim
            2048,   // intermediate_size
            151936, // vocab_size
            32768,  // max_position_embeddings
            "qwen3",// model_type
        );
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.embed_dim, 1024);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_heads, 8);
        assert_eq!(config.architecture, ModelArchitecture::Qwen3);
    }

    #[test]
    fn test_from_model_config_llama() {
        let config = ModelArchConfig::from_model_config(
            4096,   // hidden_size
            32,     // num_attention_heads
            None,   // num_key_value_heads → defaults to 32
            None,   // head_dim → derived as 4096/32 = 128
            11008,  // intermediate_size
            32000,  // vocab_size
            4096,   // max_position_embeddings
            "llama",// model_type
        );
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.embed_dim, 4096);
        assert_eq!(config.head_dim, 128); // 4096 / 32
        assert_eq!(config.kv_heads, 32);  // defaulted from num_heads
        assert_eq!(config.architecture, ModelArchitecture::Qwen3); // LLaMA uses same patterns
    }

    #[test]
    fn test_from_model_config_unknown_arch() {
        let config = ModelArchConfig::from_model_config(
            2560,   // hidden_size
            20,     // num_attention_heads
            None,   // num_key_value_heads
            None,   // head_dim → 2560/20 = 128
            6400,   // intermediate_size
            50257,  // vocab_size (GPT-2)
            1024,   // max_position_embeddings
            "gpt2", // model_type → Generic
        );
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.head_dim, 128);
        assert!(matches!(config.architecture, ModelArchitecture::Generic { .. }));
    }

    #[test]
    fn test_effective_kv_heads() {
        let config = ModelArchConfig {
            kv_heads: 8,
            num_heads: 16,
            ..ModelArchConfig::default()
        };
        assert_eq!(config.effective_kv_heads(), 8);

        let config_no_kv = ModelArchConfig {
            kv_heads: 0,
            num_heads: 16,
            ..ModelArchConfig::default()
        };
        assert_eq!(config_no_kv.effective_kv_heads(), 16); // Falls back to num_heads
    }

    #[test]
    fn test_attention_scale() {
        let config = ModelArchConfig {
            head_dim: 128,
            ..ModelArchConfig::default()
        };
        let expected = 1.0f32 / (128.0f32).sqrt();
        assert!((config.attention_scale() - expected).abs() < 1e-6);

        let config_64 = ModelArchConfig {
            head_dim: 64,
            ..ModelArchConfig::default()
        };
        let expected_64 = 1.0f32 / (64.0f32).sqrt();
        assert!((config_64.attention_scale() - expected_64).abs() < 1e-6);
    }
}

// ─── Shared Traits ───────────────────────────────────────────────

/// Trait for IR node identifiers.
///
/// All IR levels (SIR, AIR, MIR) use string-based node IDs.
/// This trait provides a uniform interface for common operations
/// on node IDs across the IR stack.
pub trait IrNodeId:
    std::fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Serialize + serde::de::DeserializeOwned
{
    /// Returns the string representation of this node ID.
    fn as_str(&self) -> &str;

    /// Construct a node ID from a string.
    fn from_string(s: String) -> Self;
}

/// Minimal trait for common graph operations across IR levels.
///
/// Each IR level (SIR, AIR, MIR) implements this trait to provide
/// a uniform interface for graph traversal and analysis.
pub trait IrGraph {
    /// The node ID type used by this IR level.
    type NodeId: IrNodeId;

    /// Returns the input node IDs of this graph.
    fn inputs(&self) -> &[Self::NodeId];

    /// Returns the output node IDs of this graph.
    fn outputs(&self) -> &[Self::NodeId];

    /// Returns the total number of nodes in this graph.
    fn node_count(&self) -> usize;
}
