//! Decode Step task family.
//!
//! Generates profiling tasks for decode-step patterns, which model the
//! autoregressive inference step in transformer models. A decode step
//! consists of:
//! - A KV-cache state read (previous key/value pairs)
//! - A QKV projection on the new token embedding
//! - An attention computation against the cached KV
//! - An output projection
//!
//! This family is ANE-relevant because decode-step patterns are the
//! dominant execution path in autoregressive LLM inference on Apple
//! Silicon. The ANE's ability to efficiently execute these patterns
//! directly impacts token generation throughput.
//!
//! The generated tasks are narrow and deterministic: each task specifies
//! concrete dimensions, head counts, sequence lengths, and dtype. They
//! can be fed into the compile/lab pipeline like any other synthetic
//! task spec.

use ane_ir::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};
use anyhow::Result;
use super::TaskFamilyTrait;

/// Configuration for the decode step family generator.
#[derive(Debug, Clone)]
pub struct DecodeStepFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Embedding dimension variants to generate.
    pub embed_dim_variants: Vec<usize>,
    /// Number of attention heads variants.
    pub num_heads_variants: Vec<usize>,
    /// KV-cache sequence length variants.
    pub kv_len_variants: Vec<usize>,
    /// Batch sizes to generate.
    pub batch_sizes: Vec<usize>,
    /// Data types to generate.
    pub dtypes: Vec<String>,
}

impl Default for DecodeStepFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            embed_dim_variants: vec![128, 256],
            num_heads_variants: vec![4],
            kv_len_variants: vec![32, 64],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        }
    }
}

impl DecodeStepFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Create a config with custom embedding dimensions.
    pub fn with_embed_dims(mut self, dims: Vec<usize>) -> Self {
        self.embed_dim_variants = dims;
        self
    }

    /// Create a config with custom head counts.
    pub fn with_num_heads(mut self, heads: Vec<usize>) -> Self {
        self.num_heads_variants = heads;
        self
    }

    /// Create a config with custom KV-cache lengths.
    pub fn with_kv_lens(mut self, lens: Vec<usize>) -> Self {
        self.kv_len_variants = lens;
        self
    }

    /// Create a config with custom batch sizes.
    pub fn with_batch_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.batch_sizes = sizes;
        self
    }

    /// Create a config with custom dtypes.
    pub fn with_dtypes(mut self, dtypes: Vec<String>) -> Self {
        self.dtypes = dtypes;
        self
    }
}

/// Decode step task family generator.
///
/// Generates deterministic decode-step task specs that model the
/// autoregressive inference step in transformer models. Each task
/// variant specifies concrete embedding dimensions, head count,
/// KV-cache length, batch size, and dtype.
///
/// The decode step exercises a pattern that combines:
/// - QKV projection (linear layer on the new token)
/// - Attention against a KV cache of `kv_len` previous tokens
/// - Output projection
///
/// This is the third real task family, following LinearProjection
/// and LutProjection.
pub struct DecodeStepFamily {
    config: DecodeStepFamilyConfig,
}

impl DecodeStepFamily {
    /// Create a new decode step family generator with default config.
    pub fn new() -> Self {
        Self {
            config: DecodeStepFamilyConfig::default(),
        }
    }

    /// Create a decode step family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: DecodeStepFamilyConfig::new(seed),
        }
    }

    /// Create a decode step family generator with custom config.
    pub fn with_config(config: DecodeStepFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &DecodeStepFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for decode steps.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (embed_dim, num_heads, kv_len, batch_size, dtype). Each task
    /// has a deterministic name derived from its parameters, and the
    /// same config always produces the same set of tasks.
    ///
    /// At minimum, this produces 4 tasks (2 embed_dims × 2 kv_lens
    /// with default config) with batch_size=1 and dtype=fp16.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for embed_dim in &self.config.embed_dim_variants {
            for num_heads in &self.config.num_heads_variants {
                // Validate that embed_dim is divisible by num_heads
                if embed_dim % num_heads != 0 {
                    anyhow::bail!(
                        "Invalid decode_step config: embed_dim {} is not divisible by num_heads {}",
                        embed_dim, num_heads
                    );
                }

                let head_dim = embed_dim / num_heads;

                for kv_len in &self.config.kv_len_variants {
                    for batch_size in &self.config.batch_sizes {
                        for dtype in &self.config.dtypes {
                            let task_name = format!(
                                "decode_e{}_h{}_kv{}_b{}_{}",
                                embed_dim, num_heads, kv_len, batch_size, dtype
                            );

                            let spec = SyntheticTaskSpec {
                                name: task_name.clone(),
                                family: "DecodeStep".to_string(),
                                description: Some(format!(
                                    "Generated decode step: embed={}, heads={}, head_dim={}, kv_len={}, batch={}, dtype={}",
                                    embed_dim, num_heads, head_dim, kv_len, batch_size, dtype
                                )),
                                op: TaskOp::DecodeStep {
                                    embed_dim: *embed_dim,
                                    num_heads: *num_heads,
                                    head_dim,
                                    kv_len: *kv_len,
                                    batch_size: *batch_size,
                                    dtype: dtype.clone(),
                                },
                                measurement: MeasurementConfig {
                                    warmup_iterations: 5,
                                    measured_iterations: 20,
                                    metrics: vec!["Latency".to_string(), "Drift".to_string()],
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

impl Default for DecodeStepFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for DecodeStepFamily {
    fn family_name(&self) -> &'static str {
        "DecodeStep"
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
        let family = DecodeStepFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 2 embed_dims × 1 heads × 2 kv_lens × 1 batch × 1 dtype = 4 tasks
        assert!(tasks.len() >= 4, "Must generate at least 4 tasks, got {}", tasks.len());

        // Verify no unimplemented!() on the path
        for task in &tasks {
            assert_eq!(task.family, "DecodeStep");
            assert!(!task.name.is_empty());
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = DecodeStepFamily::with_seed(42);
        let family2 = DecodeStepFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (TaskOp::DecodeStep { embed_dim: e1, num_heads: h1, head_dim: hd1, kv_len: kv1, batch_size: b1, dtype: d1 },
                 TaskOp::DecodeStep { embed_dim: e2, num_heads: h2, head_dim: hd2, kv_len: kv2, batch_size: b2, dtype: d2 }) => {
                    assert_eq!((e1, h1, hd1, kv1, b1, d1), (e2, h2, hd2, kv2, b2, d2));
                }
                _ => panic!("Expected both to be DecodeStep"),
            }
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = DecodeStepFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Verify serialization roundtrip
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            match (&parsed.op, &task.op) {
                (TaskOp::DecodeStep { embed_dim: e1, num_heads: h1, .. },
                 TaskOp::DecodeStep { embed_dim: e2, num_heads: h2, .. }) => {
                    assert_eq!((e1, h1), (e2, h2));
                }
                _ => panic!("Expected DecodeStep"),
            }
        }
    }

    #[test]
    fn test_at_least_four_variants() {
        let family = DecodeStepFamily::new();
        let tasks = family.generate_tasks().unwrap();
        assert!(tasks.len() >= 4, "Must generate at least 4 deterministic variants");
    }

    #[test]
    fn test_custom_config() {
        let config = DecodeStepFamilyConfig {
            seed: 99,
            embed_dim_variants: vec![512],
            num_heads_variants: vec![8],
            kv_len_variants: vec![128],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string()],
        };
        let family = DecodeStepFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 1 embed × 1 heads × 1 kv_len × 2 batches × 1 dtype = 2 tasks
        assert_eq!(tasks.len(), 2);

        for task in &tasks {
            match &task.op {
                TaskOp::DecodeStep { embed_dim, num_heads, head_dim, kv_len, .. } => {
                    assert_eq!(*embed_dim, 512);
                    assert_eq!(*num_heads, 8);
                    assert_eq!(*head_dim, 64); // 512 / 8
                    assert_eq!(*kv_len, 128);
                }
                _ => panic!("Expected DecodeStep"),
            }
        }
    }

    #[test]
    fn test_invalid_embed_dim_divisibility() {
        let config = DecodeStepFamilyConfig {
            seed: 42,
            embed_dim_variants: vec![127], // Not divisible by any reasonable head count
            num_heads_variants: vec![4],
            kv_len_variants: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = DecodeStepFamily::with_config(config);
        let result = family.generate_tasks();
        assert!(result.is_err(), "Invalid embed_dim/num_heads must be rejected");
    }

    #[test]
    fn test_generated_task_hash_stability() {
        let family = DecodeStepFamily::new();
        let tasks = family.generate_tasks().unwrap();

        // Generate twice and verify names/families match
        let tasks2 = family.generate_tasks().unwrap();
        for (t1, t2) in tasks.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (TaskOp::DecodeStep { embed_dim: e1, num_heads: h1, head_dim: hd1, kv_len: kv1, batch_size: b1, dtype: d1 },
                 TaskOp::DecodeStep { embed_dim: e2, num_heads: h2, head_dim: hd2, kv_len: kv2, batch_size: b2, dtype: d2 }) => {
                    assert_eq!((e1, h1, hd1, kv1, b1, d1), (e2, h2, hd2, kv2, b2, d2));
                }
                _ => panic!("Expected DecodeStep"),
            }
        }
    }

    #[test]
    fn test_task_family_trait_dispatch() {
        let family = DecodeStepFamily::new();
        let trait_ref: &dyn TaskFamilyTrait = &family;
        assert_eq!(trait_ref.family_name(), "DecodeStep");
        assert_eq!(trait_ref.generator_version(), "1.0.0");
        let tasks = trait_ref.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }
}
