//! Transformers model architecture registry.
//!
//! Maps HuggingFace model types to their ANE-faithful decomposition
//! patterns. Each registered pattern describes how to decompose the
//! model's transformer layers into ANE-compatible operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The global model registry. Maps HuggingFace model type strings
/// (e.g., "gpt2", "llama", "bert") to their decomposition patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    patterns: HashMap<String, ModelPattern>,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Create the default registry with built-in model patterns.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        registry.register_gpt2();
        registry.register_llama();
        registry.register_bert();
        registry.register_phi();
        registry.register_qwen();
        registry
    }

    /// Register a custom model pattern.
    pub fn register(&mut self, model_type: &str, pattern: ModelPattern) {
        self.patterns.insert(model_type.to_string(), pattern);
    }

    /// Look up a model pattern by HuggingFace model type.
    pub fn get(&self, model_type: &str) -> Option<&ModelPattern> {
        self.patterns.get(model_type)
    }

    /// Look up a model pattern, falling back to architecture-based matching.
    pub fn get_by_architecture(&self, architecture: &str) -> Option<&ModelPattern> {
        // First try exact match on model_type
        if let Some(pattern) = self.patterns.get(architecture) {
            return Some(pattern);
        }
        // Try to match by architecture family
        for (key, pattern) in &self.patterns {
            if architecture.to_lowercase().contains(key) {
                return Some(pattern);
            }
            if pattern.architectures.iter().any(|a| architecture.contains(a)) {
                return Some(pattern);
            }
        }
        None
    }

    /// List all registered model types.
    pub fn registered_models(&self) -> Vec<&str> {
        self.patterns.keys().map(|s| s.as_str()).collect()
    }

    fn register_gpt2(&mut self) {
        self.patterns.insert(
            "gpt2".to_string(),
            ModelPattern {
                model_type: "gpt2".to_string(),
                architectures: vec![
                    "GPT2LMHeadModel".to_string(),
                    "GPT2Model".to_string(),
                    "GPT2ForSequenceClassification".to_string(),
                ],
                layer_kind: TransformerLayerKind::Gpt2Block,
                default_config: DefaultModelConfig {
                    hidden_size: 768,
                    num_attention_heads: 12,
                    num_key_value_heads: Some(12), // MHA, not GQA
                    intermediate_size: 3072,
                    hidden_act: "gelu_new".to_string(),
                    uses_rope: false,
                    uses_rms_norm: false,
                    uses_gqa: false,
                    layer_norm_epsilon: 1e-5,
                    norm_type: NormType::LayerNorm,
                },
                ane_notes: vec![
                    "GPT-2 uses GELU (EXACT mode), which has an ANEC converter.".to_string(),
                    "LayerNorm is supported on A15+ only.".to_string(),
                    "Attention mask handling may require Cast ops that are CPU-only on older families.".to_string(),
                    "No RoPE — position embeddings are learned (Embedding lookup), which is CPU-only on ANE.".to_string(),
                ],
            },
        );
    }

    fn register_llama(&mut self) {
        self.patterns.insert(
            "llama".to_string(),
            ModelPattern {
                model_type: "llama".to_string(),
                architectures: vec![
                    "LlamaForCausalLM".to_string(),
                    "LlamaModel".to_string(),
                    "LlamaForSequenceClassification".to_string(),
                ],
                layer_kind: TransformerLayerKind::LlamaBlock,
                default_config: DefaultModelConfig {
                    hidden_size: 4096,
                    num_attention_heads: 32,
                    num_key_value_heads: Some(32), // May differ for GQA variants
                    intermediate_size: 11008,
                    hidden_act: "silu".to_string(),
                    uses_rope: true,
                    uses_rms_norm: true,
                    uses_gqa: false, // Set to true for GQA variants
                    layer_norm_epsilon: 1e-6,
                    norm_type: NormType::RmsNorm,
                },
                ane_notes: vec![
                    "LLaMA uses SiLU (Swish) activation — has ANEC PEFUSED converter.".to_string(),
                    "RMSNorm decomposes to: Rsqrt(ReduceMean(x^2, axis=-1)) * x * weight".to_string(),
                    "  - Rsqrt: ANEC PE converter available".to_string(),
                    "  - ReduceMean: ANEC PE converter available (A14+ for non-FP ReduceMin)".to_string(),
                    "  - This decomposition is ANE-faithful on A14+.".to_string(),
                    "RoPE decomposes to: cos * x + sin * rotate_half(x)".to_string(),
                    "  - Cos/Sin: ANEC PE converters available".to_string(),
                    "  - This decomposition is ANE-faithful.".to_string(),
                    "SDPA (scaled_dot_product_attention) is supported on A16+.".to_string(),
                    "For A14/A15: SDPA must decompose to MatMul + Softmax + MatMul.".to_string(),
                    "GQA (Grouped Query Attention): K/V repeat before attention, which is a Gather/Expand op.".to_string(),
                    "  - Gather on ANE requires constant axis and batch=1, depth=1.".to_string(),
                    "No bias in LLaMA linear projections — simplifies ANE placement.".to_string(),
                    "CRITICAL: Blockwise quantization (GPTQ/AWQ) is NOT supported on ANE.".to_string(),
                    "  Only per-tensor and per-output-channel quantization are ANE-compatible.".to_string(),
                ],
            },
        );
    }

    fn register_bert(&mut self) {
        self.patterns.insert(
            "bert".to_string(),
            ModelPattern {
                model_type: "bert".to_string(),
                architectures: vec![
                    "BertModel".to_string(),
                    "BertForMaskedLM".to_string(),
                    "BertForSequenceClassification".to_string(),
                    "BertForTokenClassification".to_string(),
                ],
                layer_kind: TransformerLayerKind::BertBlock,
                default_config: DefaultModelConfig {
                    hidden_size: 768,
                    num_attention_heads: 12,
                    num_key_value_heads: Some(12),
                    intermediate_size: 3072,
                    hidden_act: "gelu".to_string(),
                    uses_rope: false,
                    uses_rms_norm: false,
                    uses_gqa: false,
                    layer_norm_epsilon: 1e-12,
                    norm_type: NormType::LayerNorm,
                },
                ane_notes: vec![
                    "BERT uses GELU activation — ANEC converter available.".to_string(),
                    "BERT uses bidirectional attention with attention masks.".to_string(),
                    "  - Attention mask handling may require logical ops (CPU-only).".to_string(),
                    "  - Consider pre-computing the mask as a constant tensor.".to_string(),
                    "LayerNorm supported on A15+.".to_string(),
                    "Encoder-only: no KV-cache state needed for prefill-only compilation.".to_string(),
                    "Token type embeddings add a second embedding lookup (CPU-only on ANE).".to_string(),
                ],
            },
        );
    }

    fn register_phi(&mut self) {
        self.patterns.insert(
            "phi".to_string(),
            ModelPattern {
                model_type: "phi".to_string(),
                architectures: vec![
                    "PhiForCausalLM".to_string(),
                    "PhiModel".to_string(),
                ],
                layer_kind: TransformerLayerKind::PhiBlock,
                default_config: DefaultModelConfig {
                    hidden_size: 2048,
                    num_attention_heads: 32,
                    num_key_value_heads: Some(32),
                    intermediate_size: 8192,
                    hidden_act: "gelu_new".to_string(),
                    uses_rope: true,
                    uses_rms_norm: false,
                    uses_gqa: false,
                    layer_norm_epsilon: 1e-5,
                    norm_type: NormType::LayerNorm,
                },
                ane_notes: vec![
                    "Phi uses partial attention (Q, K, V are not the same size).".to_string(),
                    "Small form factor: fits entirely in ANE L2 cache on A16+.".to_string(),
                    "GELU activation — ANEC converter available.".to_string(),
                    "RoPE — same decomposition as LLaMA (ANE-faithful).".to_string(),
                ],
            },
        );
    }

    fn register_qwen(&mut self) {
        self.patterns.insert(
            "qwen2".to_string(),
            ModelPattern {
                model_type: "qwen2".to_string(),
                architectures: vec![
                    "Qwen2ForCausalLM".to_string(),
                    "Qwen2Model".to_string(),
                ],
                layer_kind: TransformerLayerKind::QwenBlock,
                default_config: DefaultModelConfig {
                    hidden_size: 4096,
                    num_attention_heads: 32,
                    num_key_value_heads: Some(4), // GQA with 4 KV heads
                    intermediate_size: 11008,
                    hidden_act: "silu".to_string(),
                    uses_rope: true,
                    uses_rms_norm: true,
                    uses_gqa: true,
                    layer_norm_epsilon: 1e-6,
                    norm_type: NormType::RmsNorm,
                },
                ane_notes: vec![
                    "Qwen2 uses GQA (Grouped Query Attention) with 4 KV heads.".to_string(),
                    "  - K/V repeat via Expand+Broadcast is ANE-compatible on A14+.".to_string(),
                    "  - Avoid Gather-based repeat (non-constant axis issues).".to_string(),
                    "SiLU activation — ANEC PEFUSED converter available.".to_string(),
                    "RMSNorm — same decomposition as LLaMA (ANE-faithful on A14+).".to_string(),
                    "RoPE — same decomposition as LLaMA (ANE-faithful).".to_string(),
                    "SDPA supported on A16+; decompose to MatMul+Softmax+MatMul on A14/A15.".to_string(),
                ],
            },
        );
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

/// Pattern describing how to decompose a specific transformer architecture
/// into ANE-faithful operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPattern {
    /// HuggingFace model type identifier (e.g., "llama", "gpt2").
    pub model_type: String,

    /// HuggingFace architecture class names that use this pattern.
    pub architectures: Vec<String>,

    /// The kind of transformer layer this model uses.
    pub layer_kind: TransformerLayerKind,

    /// Default model configuration values.
    pub default_config: DefaultModelConfig,

    /// ANE-specific notes and constraints for this model family.
    pub ane_notes: Vec<String>,
}

/// Kind of transformer layer — determines the decomposition strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformerLayerKind {
    /// GPT-2 style: LayerNorm pre-attention + GELU MLP
    Gpt2Block,
    /// LLaMA style: RMSNorm pre-attention + SiLU MLP + RoPE
    LlamaBlock,
    /// BERT style: LayerNorm post-attention + GELU MLP + bidirectional
    BertBlock,
    /// Phi style: LayerNorm + partial attention + GELU MLP + RoPE
    PhiBlock,
    /// Qwen2 style: RMSNorm + GQA + SiLU MLP + RoPE
    QwenBlock,
}

/// Default model configuration for a registered pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub uses_rope: bool,
    pub uses_rms_norm: bool,
    pub uses_gqa: bool,
    pub layer_norm_epsilon: f64,
    pub norm_type: NormType,
}

/// Type of normalization used in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_registry_has_models() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.get("gpt2").is_some());
        assert!(registry.get("llama").is_some());
        assert!(registry.get("bert").is_some());
        assert!(registry.get("phi").is_some());
        assert!(registry.get("qwen2").is_some());
    }

    #[test]
    fn test_architecture_lookup() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.get_by_architecture("LlamaForCausalLM").is_some());
        assert!(registry.get_by_architecture("GPT2LMHeadModel").is_some());
        assert!(registry.get_by_architecture("UnknownModel").is_none());
    }

    #[test]
    fn test_custom_registration() {
        let mut registry = ModelRegistry::new();
        registry.register("custom", ModelPattern {
            model_type: "custom".to_string(),
            architectures: vec!["CustomModel".to_string()],
            layer_kind: TransformerLayerKind::LlamaBlock,
            default_config: DefaultModelConfig {
                hidden_size: 256,
                num_attention_heads: 4,
                num_key_value_heads: Some(4),
                intermediate_size: 1024,
                hidden_act: "silu".to_string(),
                uses_rope: true,
                uses_rms_norm: true,
                uses_gqa: false,
                layer_norm_epsilon: 1e-6,
                norm_type: NormType::RmsNorm,
            },
            ane_notes: vec![],
        });
        assert!(registry.get("custom").is_some());
    }

    #[test]
    fn test_llama_ane_notes() {
        let registry = ModelRegistry::default_registry();
        let llama = registry.get("llama").unwrap();
        assert!(!llama.ane_notes.is_empty());
        assert!(llama.default_config.uses_rope);
        assert!(llama.default_config.uses_rms_norm);
    }

    #[test]
    fn test_qwen_uses_gqa() {
        let registry = ModelRegistry::default_registry();
        let qwen = registry.get("qwen2").unwrap();
        assert!(qwen.default_config.uses_gqa);
        assert_eq!(qwen.default_config.num_key_value_heads, Some(4));
    }
}
