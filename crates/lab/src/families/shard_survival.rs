//! Shard Survival task family.
//!
//! Tests that sharded models survive compilation and produce correct
//! results across shard boundaries. This family generates tasks that
//! exercise the sharded compilation pipeline with various shard counts
//! and dimension configurations, targeting the known gap where shard
//! roles must produce genuinely different emitted programs.
//!
//! ## Critique gap addressed
//!
//! Critique gap #3 (sharding bugs) identified that shard emission was
//! too uniform and that `StateWriteRead` was defined but dead. While
//! Sprint 48 activated `StateWriteRead` on the decode-step shard
//! boundary, there was no automated task generation for systematically
//! testing shard survival across different configurations. This family
//! provides that systematic testing surface.
//!
//! ## Generated tasks
//!
//! Two categories of shard survival tasks:
//!
//! - **Linear pipeline shards**: 3-shard Entry/Interior/Exit decomposition
//!   of linear projections with different dimensions per role.
//! - **Decode-step shards**: 3-shard QKV/Attention/Output decomposition
//!   of decode-step workloads with KV cache state semantics.
//!
//! Each task uses `TaskOp::ShardedLinearPipeline` or
//! `TaskOp::ShardedDecodeStep` so the existing sharded compilation
//! pipeline can process them directly.

use super::TaskFamilyTrait;
use ane_ir::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};
use anyhow::Result;

/// Configuration for the shard-survival family generator.
#[derive(Debug, Clone)]
pub struct ShardSurvivalFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Shard configurations to test.
    pub configs: Vec<ShardTestConfig>,
    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,
    /// Data types to test.
    pub dtypes: Vec<String>,
}

/// A shard test configuration specifying the pipeline type and dimensions.
#[derive(Debug, Clone)]
pub enum ShardTestConfig {
    /// 3-shard linear pipeline: Entry/Interior/Exit.
    /// Fields: (input_dim, hidden_dim, output_dim)
    LinearPipeline { input_dim: usize, hidden_dim: usize, output_dim: usize },
    /// 3-shard decode-step pipeline: QKV/Attention/Output.
    /// Fields: (embed_dim, num_heads, head_dim, kv_len)
    DecodeStepPipeline { embed_dim: usize, num_heads: usize, head_dim: usize, kv_len: usize },
}

impl ShardTestConfig {
    /// Return a human-readable name for this configuration.
    pub fn config_name(&self) -> String {
        match self {
            ShardTestConfig::LinearPipeline { input_dim, hidden_dim, output_dim } => {
                format!("linear_{}x{}x{}", input_dim, hidden_dim, output_dim)
            }
            ShardTestConfig::DecodeStepPipeline { embed_dim, num_heads, head_dim, kv_len } => {
                format!("decode_{}d_{}h_{}k_{}kv", embed_dim, num_heads, head_dim, kv_len)
            }
        }
    }
}

impl Default for ShardSurvivalFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            configs: vec![
                // Linear pipeline variants
                ShardTestConfig::LinearPipeline { input_dim: 64, hidden_dim: 48, output_dim: 32 },
                ShardTestConfig::LinearPipeline { input_dim: 128, hidden_dim: 96, output_dim: 64 },
                // Decode-step pipeline variants
                ShardTestConfig::DecodeStepPipeline {
                    embed_dim: 64,
                    num_heads: 4,
                    head_dim: 16,
                    kv_len: 32,
                },
                ShardTestConfig::DecodeStepPipeline {
                    embed_dim: 128,
                    num_heads: 4,
                    head_dim: 32,
                    kv_len: 64,
                },
            ],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        }
    }
}

impl ShardSurvivalFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed, ..Default::default() }
    }

    /// Create a config with custom shard configurations.
    pub fn with_configs(mut self, configs: Vec<ShardTestConfig>) -> Self {
        self.configs = configs;
        self
    }
}

/// Shard survival task family generator.
///
/// Generates deterministic task specs that exercise the sharded
/// compilation pipeline across various configurations. The generated
/// tasks use the existing sharded task ops (`ShardedLinearPipeline`
/// and `ShardedDecodeStep`) so the `compile-full-sharded` and
/// `compile-sharded` CLI commands can process them directly.
///
/// This family provides systematic coverage of:
///
/// - Different shard dimension configurations
/// - Both linear and decode-step shard pipeline types
/// - Different compute unit assignments across shards
/// - KV cache state semantics in decode-step shards
pub struct ShardSurvivalFamily {
    config: ShardSurvivalFamilyConfig,
}

impl ShardSurvivalFamily {
    /// Create a new shard-survival family generator with default config.
    pub fn new() -> Self {
        Self { config: ShardSurvivalFamilyConfig::default() }
    }

    /// Create a shard-survival family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { config: ShardSurvivalFamilyConfig::new(seed) }
    }

    /// Create a shard-survival family generator with custom config.
    pub fn with_config(config: ShardSurvivalFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &ShardSurvivalFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for shard survival testing.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (shard_config, batch_size, dtype). Linear pipeline configs
    /// use `TaskOp::ShardedLinearPipeline`; decode-step configs
    /// use `TaskOp::ShardedDecodeStep`.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for shard_config in &self.config.configs {
            let config_name = shard_config.config_name();

            for batch_size in &self.config.batch_sizes {
                for dtype in &self.config.dtypes {
                    let task_name =
                        format!("shard_survival_{}_b{}_{}", config_name, batch_size, dtype);

                    let (op, description) = match shard_config {
                        ShardTestConfig::LinearPipeline { input_dim, hidden_dim, output_dim } => {
                            (
                                TaskOp::ShardedLinearPipeline {
                                    input_dim: *input_dim,
                                    hidden_dim: *hidden_dim,
                                    output_dim: *output_dim,
                                    batch_size: *batch_size,
                                    dtype: dtype.clone(),
                                },
                                format!(
                                    "Shard survival test (3-shard linear): [{}] -> [{}] -> [{}], batch={}, dtype={}",
                                    input_dim, hidden_dim, output_dim, batch_size, dtype
                                ),
                            )
                        }
                        ShardTestConfig::DecodeStepPipeline { embed_dim, num_heads, head_dim, kv_len } => {
                            (
                                TaskOp::ShardedDecodeStep {
                                    embed_dim: *embed_dim,
                                    num_heads: *num_heads,
                                    head_dim: *head_dim,
                                    kv_len: *kv_len,
                                    batch_size: *batch_size,
                                    kv_heads: *num_heads,
                                    intermediate_size: *embed_dim * 4,
                                    vocab_size: 0,
                                    dtype: dtype.clone(),
                                },
                                format!(
                                    "Shard survival test (3-shard decode-step): {}d/{}h/{}k/{}kv, batch={}, dtype={}",
                                    embed_dim, num_heads, head_dim, kv_len, batch_size, dtype
                                ),
                            )
                        }
                    };

                    let spec = SyntheticTaskSpec {
                        name: task_name.clone(),
                        family: "ShardSurvival".to_string(),
                        description: Some(description),
                        op,
                        measurement: MeasurementConfig {
                            warmup_iterations: 3,
                            measured_iterations: 10,
                            metrics: vec![
                                "Latency".to_string(),
                                "FallbackSuspicion".to_string(),
                                "ShardSurvival".to_string(),
                            ],
                        },
                    };

                    tasks.push(spec);
                }
            }
        }

        Ok(tasks)
    }
}

impl Default for ShardSurvivalFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for ShardSurvivalFamily {
    fn family_name(&self) -> &'static str {
        "ShardSurvival"
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
        let family = ShardSurvivalFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 4 shard configs × 1 batch × 1 dtype = 4 tasks
        assert!(tasks.len() >= 4, "Must generate at least 4 tasks, got {}", tasks.len());

        for task in &tasks {
            assert_eq!(task.family, "ShardSurvival");
            assert!(!task.name.is_empty());
            assert!(task.name.starts_with("shard_survival_"));
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = ShardSurvivalFamily::with_seed(42);
        let family2 = ShardSurvivalFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = ShardSurvivalFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
        }
    }

    #[test]
    fn test_linear_and_decode_step_tasks_present() {
        let family = ShardSurvivalFamily::new();
        let tasks = family.generate_tasks().unwrap();

        let names: Vec<&str> = tasks.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("_linear_")), "Must have linear pipeline tasks");
        assert!(
            names.iter().any(|n| n.contains("_decode_")),
            "Must have decode-step pipeline tasks"
        );
    }

    #[test]
    fn test_linear_configs_use_sharded_linear_pipeline_op() {
        let config = ShardSurvivalFamilyConfig {
            seed: 42,
            configs: vec![ShardTestConfig::LinearPipeline {
                input_dim: 64,
                hidden_dim: 48,
                output_dim: 32,
            }],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = ShardSurvivalFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            match &task.op {
                TaskOp::ShardedLinearPipeline { .. } => {} // correct
                other => panic!("Expected ShardedLinearPipeline, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_decode_step_configs_use_sharded_decode_step_op() {
        let config = ShardSurvivalFamilyConfig {
            seed: 42,
            configs: vec![ShardTestConfig::DecodeStepPipeline {
                embed_dim: 64,
                num_heads: 4,
                head_dim: 16,
                kv_len: 32,
            }],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = ShardSurvivalFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            match &task.op {
                TaskOp::ShardedDecodeStep { .. } => {} // correct
                other => panic!("Expected ShardedDecodeStep, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_custom_config() {
        let config = ShardSurvivalFamilyConfig {
            seed: 99,
            configs: vec![ShardTestConfig::LinearPipeline {
                input_dim: 256,
                hidden_dim: 192,
                output_dim: 128,
            }],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        };
        let family = ShardSurvivalFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 1 config × 2 batches × 2 dtypes = 4 tasks
        assert_eq!(tasks.len(), 4);
    }

    #[test]
    fn test_trait_dispatch_works() {
        let family = ShardSurvivalFamily::new();
        let trait_gen: &dyn TaskFamilyTrait = &family;
        assert_eq!(trait_gen.family_name(), "ShardSurvival");
        assert_eq!(trait_gen.generator_version(), "1.0.0");
        let tasks = trait_gen.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_measurement_includes_shard_survival() {
        let family = ShardSurvivalFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            assert!(
                task.measurement.metrics.contains(&"ShardSurvival".to_string()),
                "ShardSurvival tasks should measure ShardSurvival metric"
            );
        }
    }
}
