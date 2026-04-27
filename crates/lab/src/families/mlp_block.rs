//! MLP Block task family.
//!
//! Generates profiling tasks for fused MLP block patterns (linear +
//! activation + linear) for ANE compilation correctness and performance.
//!
//! An MLP block models the feed-forward network block in transformer
//! models, which is one of the two dominant ANE execution patterns
//! (the other being attention). The block consists of:
//! - Up-projection: linear(input_dim -> hidden_dim)
//! - Activation: GELU or ReLU
//! - Down-projection: linear(hidden_dim -> output_dim)
//!
//! This family is ANE-relevant because the fused linear-activation-linear
//! pattern is central to ANE placement for transformer feed-forward layers.
//! The generated tasks exercise this pattern at various scales and with
//! both GELU and ReLU activations.
//!
//! The generated tasks are narrow and deterministic: each task specifies
//! concrete dimensions, activation type, batch size, and dtype. They
//! can be fed into the compile/lab pipeline like any other synthetic
//! task spec.

use ane_ir::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};
use anyhow::Result;
use super::TaskFamilyTrait;

/// Configuration for the MLP block family generator.
#[derive(Debug, Clone)]
pub struct MlpBlockFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Input dimension variants to generate.
    pub input_dim_variants: Vec<usize>,
    /// Hidden (up-projected) dimension variants.
    pub hidden_dim_variants: Vec<usize>,
    /// Output dimension variants (typically same as input_dim).
    pub output_dim_variants: Vec<usize>,
    /// Activation functions to generate.
    pub activations: Vec<String>,
    /// Batch sizes to generate.
    pub batch_sizes: Vec<usize>,
    /// Data types to generate.
    pub dtypes: Vec<String>,
}

impl Default for MlpBlockFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            input_dim_variants: vec![128, 256],
            hidden_dim_variants: vec![512],
            output_dim_variants: vec![], // Empty means: use same as input_dim
            activations: vec!["gelu".to_string(), "relu".to_string()],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        }
    }
}

impl MlpBlockFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Create a config with custom input dimensions.
    pub fn with_input_dims(mut self, dims: Vec<usize>) -> Self {
        self.input_dim_variants = dims;
        self
    }

    /// Create a config with custom hidden dimensions.
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dim_variants = dims;
        self
    }

    /// Create a config with custom output dimensions.
    pub fn with_output_dims(mut self, dims: Vec<usize>) -> Self {
        self.output_dim_variants = dims;
        self
    }

    /// Create a config with custom activations.
    pub fn with_activations(mut self, activations: Vec<String>) -> Self {
        self.activations = activations;
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

/// MLP block task family generator.
///
/// Generates deterministic MLP block task specs that model the
/// feed-forward network block in transformer models. Each task
/// variant specifies concrete input/hidden/output dimensions,
/// activation function, batch size, and dtype.
///
/// The MLP block exercises the fused linear-activation-linear
/// pattern that is central to ANE placement for transformer
/// feed-forward layers.
///
/// This is the fourth real task family, following LinearProjection,
/// LutProjection, and DecodeStep.
pub struct MlpBlockFamily {
    config: MlpBlockFamilyConfig,
}

impl MlpBlockFamily {
    /// Create a new MLP block family generator with default config.
    pub fn new() -> Self {
        Self {
            config: MlpBlockFamilyConfig::default(),
        }
    }

    /// Create an MLP block family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: MlpBlockFamilyConfig::new(seed),
        }
    }

    /// Create an MLP block family generator with custom config.
    pub fn with_config(config: MlpBlockFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &MlpBlockFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for MLP blocks.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (input_dim, hidden_dim, output_dim, activation, batch_size, dtype).
    /// Each task has a deterministic name derived from its parameters, and the
    /// same config always produces the same set of tasks.
    ///
    /// At minimum, this produces 4 tasks (2 input_dims × 1 hidden_dim ×
    /// 2 activations with default config) with batch_size=1 and dtype=fp16.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for input_dim in &self.config.input_dim_variants {
            // Determine output_dim: use output_dim_variants if set,
            // otherwise default to same as input_dim (typical for FFN).
            let output_dims: Vec<usize> = if self.config.output_dim_variants.is_empty() {
                vec![*input_dim]
            } else {
                self.config.output_dim_variants.clone()
            };

            for hidden_dim in &self.config.hidden_dim_variants {
                for output_dim in &output_dims {
                    for activation in &self.config.activations {
                        for batch_size in &self.config.batch_sizes {
                            for dtype in &self.config.dtypes {
                                let task_name = format!(
                                    "mlp_i{}_h{}_o{}_{}_b{}_{}",
                                    input_dim, hidden_dim, output_dim,
                                    activation, batch_size, dtype
                                );

                                let spec = SyntheticTaskSpec {
                                    name: task_name.clone(),
                                    family: "MlpBlock".to_string(),
                                    description: Some(format!(
                                        "Generated MLP block: input={}, hidden={}, output={}, activation={}, batch={}, dtype={}",
                                        input_dim, hidden_dim, output_dim, activation, batch_size, dtype
                                    )),
                                    op: TaskOp::MlpBlock {
                                        input_dim: *input_dim,
                                        hidden_dim: *hidden_dim,
                                        output_dim: *output_dim,
                                        activation: activation.clone(),
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
        }

        Ok(tasks)
    }
}

impl Default for MlpBlockFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for MlpBlockFamily {
    fn family_name(&self) -> &'static str {
        "MlpBlock"
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
        let family = MlpBlockFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 2 input_dims × 1 hidden_dim × (output=input) × 2 activations × 1 batch × 1 dtype = 4 tasks
        assert!(tasks.len() >= 4, "Must generate at least 4 tasks, got {}", tasks.len());

        // Verify no unimplemented!() on the path
        for task in &tasks {
            assert_eq!(task.family, "MlpBlock");
            assert!(!task.name.is_empty());
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = MlpBlockFamily::with_seed(42);
        let family2 = MlpBlockFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (TaskOp::MlpBlock { input_dim: i1, hidden_dim: h1, output_dim: o1, activation: a1, batch_size: b1, dtype: d1 },
                 TaskOp::MlpBlock { input_dim: i2, hidden_dim: h2, output_dim: o2, activation: a2, batch_size: b2, dtype: d2 }) => {
                    assert_eq!((i1, h1, o1, a1, b1, d1), (i2, h2, o2, a2, b2, d2));
                }
                _ => panic!("Expected both to be MlpBlock"),
            }
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = MlpBlockFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Verify serialization roundtrip
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            match (&parsed.op, &task.op) {
                (TaskOp::MlpBlock { input_dim: i1, hidden_dim: h1, .. },
                 TaskOp::MlpBlock { input_dim: i2, hidden_dim: h2, .. }) => {
                    assert_eq!((i1, h1), (i2, h2));
                }
                _ => panic!("Expected MlpBlock"),
            }
        }
    }

    #[test]
    fn test_at_least_four_variants() {
        let family = MlpBlockFamily::new();
        let tasks = family.generate_tasks().unwrap();
        assert!(tasks.len() >= 4, "Must generate at least 4 deterministic variants");
    }

    #[test]
    fn test_custom_config() {
        let config = MlpBlockFamilyConfig {
            seed: 99,
            input_dim_variants: vec![512],
            hidden_dim_variants: vec![2048],
            output_dim_variants: vec![512],
            activations: vec!["gelu".to_string()],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string()],
        };
        let family = MlpBlockFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 1 input × 1 hidden × 1 output × 1 activation × 2 batches × 1 dtype = 2 tasks
        assert_eq!(tasks.len(), 2);

        for task in &tasks {
            match &task.op {
                TaskOp::MlpBlock { input_dim, hidden_dim, output_dim, activation, .. } => {
                    assert_eq!(*input_dim, 512);
                    assert_eq!(*hidden_dim, 2048);
                    assert_eq!(*output_dim, 512);
                    assert_eq!(activation, "gelu");
                }
                _ => panic!("Expected MlpBlock"),
            }
        }
    }

    #[test]
    fn test_output_dim_defaults_to_input_dim() {
        let config = MlpBlockFamilyConfig {
            seed: 42,
            input_dim_variants: vec![128],
            hidden_dim_variants: vec![512],
            output_dim_variants: vec![], // Empty: defaults to input_dim
            activations: vec!["gelu".to_string()],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = MlpBlockFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        assert_eq!(tasks.len(), 1);
        match &tasks[0].op {
            TaskOp::MlpBlock { input_dim, output_dim, .. } => {
                assert_eq!(*input_dim, *output_dim, "output_dim should default to input_dim");
            }
            _ => panic!("Expected MlpBlock"),
        }
    }

    #[test]
    fn test_task_family_trait_dispatch() {
        let family = MlpBlockFamily::new();
        let trait_ref: &dyn TaskFamilyTrait = &family;
        assert_eq!(trait_ref.family_name(), "MlpBlock");
        assert_eq!(trait_ref.generator_version(), "1.0.0");
        let tasks = trait_ref.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }
}
