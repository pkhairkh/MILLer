//! Attention task family.
//!
//! Generates profiling tasks for multi-head self-attention blocks,
//! one of the two dominant ANE execution patterns in transformer
//! inference (the other being the MLP block).
//!
//! Each generated task is a `SyntheticTaskSpec` that can be fed
//! directly into the compile/lab pipeline. The generator produces
//! deterministic variants controlled by a seed.
//!
//! Unlike the `DecodeStep` family which models a single autoregressive
//! decode step with KV-cache, the `Attention` family models a standalone
//! multi-head self-attention block without cache semantics. This exercises
//! the attention-specific ANE path: QKV projection, softmax, and output
//! projection as a fused unit.

use ane_ir::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};
use anyhow::Result;
use super::TaskFamilyTrait;

/// Configuration for the attention family generator.
#[derive(Debug, Clone)]
pub struct AttentionFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Embedding dimension variants to generate.
    pub embed_dim_variants: Vec<usize>,
    /// Number of attention heads per embedding dimension.
    pub num_heads_variants: Vec<usize>,
    /// Input sequence length variants.
    pub seq_len_variants: Vec<usize>,
    /// Batch sizes to generate.
    pub batch_sizes: Vec<usize>,
    /// Data types to generate.
    pub dtypes: Vec<String>,
}

impl Default for AttentionFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            embed_dim_variants: vec![128, 256],
            num_heads_variants: vec![4],
            seq_len_variants: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        }
    }
}

impl AttentionFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }
}

/// Attention task family generator.
///
/// Generates deterministic multi-head self-attention task specs that
/// can be compiled by the active compile path. Each task variant
/// specifies concrete dimensions, batch size, dtype, and a
/// deterministic seed for weight initialization.
///
/// This is the fifth real task family, implementing the attention
/// pattern that is central to ANE placement for transformer models.
/// Unlike the DecodeStep family which includes KV-cache semantics,
/// the Attention family models a standalone self-attention block.
pub struct AttentionFamily {
    config: AttentionFamilyConfig,
}

impl AttentionFamily {
    /// Create a new attention family generator with default config.
    pub fn new() -> Self {
        Self {
            config: AttentionFamilyConfig::default(),
        }
    }

    /// Create an attention family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: AttentionFamilyConfig::new(seed),
        }
    }

    /// Create an attention family generator with custom config.
    pub fn with_config(config: AttentionFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &AttentionFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for attention blocks.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (embed_dim, num_heads, seq_len, batch_size, dtype).
    /// Each task has a deterministic name derived from its parameters,
    /// and the same config always produces the same set of tasks.
    ///
    /// Only valid combinations are generated: embed_dim must be
    /// divisible by num_heads. Invalid combinations are skipped.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for embed_dim in &self.config.embed_dim_variants {
            for num_heads in &self.config.num_heads_variants {
                // Skip invalid configurations
                if *num_heads == 0 || embed_dim % num_heads != 0 {
                    continue;
                }
                let head_dim = embed_dim / num_heads;

                for seq_len in &self.config.seq_len_variants {
                    for batch_size in &self.config.batch_sizes {
                        for dtype in &self.config.dtypes {
                            let task_name = format!(
                                "attn_{}h{}_s{}_b{}_{}",
                                embed_dim, num_heads, seq_len, batch_size, dtype
                            );

                            let spec = SyntheticTaskSpec {
                                name: task_name.clone(),
                                family: "Attention".to_string(),
                                description: Some(format!(
                                    "Generated attention: embed={}, heads={}, head_dim={}, seq_len={}, batch={}, dtype={}",
                                    embed_dim, num_heads, head_dim, seq_len, batch_size, dtype
                                )),
                                op: TaskOp::Attention {
                                    embed_dim: *embed_dim,
                                    num_heads: *num_heads,
                                    head_dim,
                                    seq_len: *seq_len,
                                    batch_size: *batch_size,
                                    dtype: dtype.clone(),
                                },
                                measurement: MeasurementConfig {
                                    warmup_iterations: 5,
                                    measured_iterations: 20,
                                    metrics: vec!["Latency".to_string()],
                                },
                            };

                            tasks.push(spec);
                        }
                    }
                }
            }
        }

        Ok(tasks)
    }
}

impl Default for AttentionFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for AttentionFamily {
    fn family_name(&self) -> &'static str {
        "Attention"
    }

    fn generator_version(&self) -> &'static str {
        "1.0.0"
    }

    fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate_tasks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_tasks_default() {
        let family = AttentionFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 2 embed_dims × 1 num_heads × 1 seq_len × 1 batch × 1 dtype = 2 tasks
        assert!(tasks.len() >= 2, "Must generate at least 2 tasks, got {}", tasks.len());

        for task in &tasks {
            assert_eq!(task.family, "Attention");
            assert!(!task.name.is_empty());
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = AttentionFamily::with_seed(42);
        let family2 = AttentionFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (TaskOp::Attention { embed_dim: e1, num_heads: h1, head_dim: d1, seq_len: s1, batch_size: b1, dtype: dt1 },
                 TaskOp::Attention { embed_dim: e2, num_heads: h2, head_dim: d2, seq_len: s2, batch_size: b2, dtype: dt2 }) => {
                    assert_eq!((e1, h1, d1, s1, b1, dt1), (e2, h2, d2, s2, b2, dt2));
                }
                _ => panic!("Expected both to be Attention"),
            }
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = AttentionFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            match (&parsed.op, &task.op) {
                (TaskOp::Attention { embed_dim: e1, .. },
                 TaskOp::Attention { embed_dim: e2, .. }) => {
                    assert_eq!(e1, e2);
                }
                _ => panic!("Expected Attention"),
            }
        }
    }

    #[test]
    fn test_custom_config() {
        let config = AttentionFamilyConfig {
            seed: 99,
            embed_dim_variants: vec![512],
            num_heads_variants: vec![8],
            seq_len_variants: vec![64, 128],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = AttentionFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 1 embed × 1 heads × 2 seq_lens × 1 batch × 1 dtype = 2 tasks
        assert_eq!(tasks.len(), 2);

        for task in &tasks {
            match &task.op {
                TaskOp::Attention { embed_dim, num_heads, head_dim, .. } => {
                    assert_eq!(*embed_dim, 512);
                    assert_eq!(*num_heads, 8);
                    assert_eq!(*head_dim, 64); // 512 / 8
                }
                _ => panic!("Expected Attention"),
            }
        }
    }

    #[test]
    fn test_invalid_config_skipped() {
        let config = AttentionFamilyConfig {
            seed: 42,
            embed_dim_variants: vec![127], // Not divisible by 4
            num_heads_variants: vec![4],
            seq_len_variants: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = AttentionFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();
        assert!(tasks.is_empty(), "Invalid embed_dim/num_heads should produce no tasks");
    }

    #[test]
    fn test_trait_dispatch() {
        let family = AttentionFamily::new();
        assert_eq!(family.family_name(), "Attention");
        assert_eq!(family.generator_version(), "1.0.0");
        let tasks = family.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_variant_count() {
        let config = AttentionFamilyConfig {
            seed: 42,
            embed_dim_variants: vec![128, 256],
            num_heads_variants: vec![4, 8],
            seq_len_variants: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = AttentionFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 128 is divisible by 4 and 8 (2 valid)
        // 256 is divisible by 4 and 8 (2 valid)
        // Total: 4 tasks
        assert_eq!(tasks.len(), 4);
    }
}
