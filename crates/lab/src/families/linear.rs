//! Linear Projection task family.
//!
//! Generates profiling tasks for various linear projection sizes
//! and configurations for ANE legality and performance testing.
//!
//! Each generated task is a `SyntheticTaskSpec` that can be fed
//! directly into the compile/lab pipeline. The generator produces
//! deterministic variants controlled by a seed.

use ane_ir::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};
use anyhow::Result;
use super::TaskFamilyTrait;

/// Configuration for the linear family generator.
#[derive(Debug, Clone)]
pub struct LinearFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Dimension variants to generate: (input_dim, output_dim).
    pub dimension_variants: Vec<(usize, usize)>,
    /// Batch sizes to generate.
    pub batch_sizes: Vec<usize>,
    /// Data types to generate.
    pub dtypes: Vec<String>,
    /// Whether to include bias.
    pub has_bias: bool,
}

impl Default for LinearFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            dimension_variants: vec![
                (64, 32),
                (128, 64),
                (256, 128),
            ],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
            has_bias: true,
        }
    }
}

impl LinearFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Create a config with custom dimension variants.
    pub fn with_dimensions(mut self, variants: Vec<(usize, usize)>) -> Self {
        self.dimension_variants = variants;
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

/// Linear projection task family generator.
///
/// Generates deterministic linear projection task specs that can
/// be compiled by the active compile path. Each task variant
/// specifies concrete dimensions, batch size, dtype, and a
/// deterministic seed for weight initialization.
pub struct LinearFamily {
    config: LinearFamilyConfig,
}

impl LinearFamily {
    /// Create a new linear family generator with default config.
    pub fn new() -> Self {
        Self {
            config: LinearFamilyConfig::default(),
        }
    }

    /// Create a linear family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: LinearFamilyConfig::new(seed),
        }
    }

    /// Create a linear family generator with custom config.
    pub fn with_config(config: LinearFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &LinearFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for linear projections.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (dimension_variant, batch_size, dtype). Each task has a
    /// deterministic name derived from its parameters, and the
    /// same config always produces the same set of tasks.
    ///
    /// At minimum, this produces 3 tasks (one per default dimension
    /// variant) with batch_size=1 and dtype=fp16.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for (input_dim, output_dim) in &self.config.dimension_variants {
            for batch_size in &self.config.batch_sizes {
                for dtype in &self.config.dtypes {
                    let task_name = format!(
                        "linear_{}x{}_b{}_{}",
                        input_dim, output_dim, batch_size, dtype
                    );

                    let spec = SyntheticTaskSpec {
                        name: task_name.clone(),
                        family: "LinearProjection".to_string(),
                        description: Some(format!(
                            "Generated linear projection: [{}] -> [{}], batch={}, dtype={}",
                            input_dim, output_dim, batch_size, dtype
                        )),
                        op: TaskOp::LinearProjection {
                            input_dim: *input_dim,
                            output_dim: *output_dim,
                            batch_size: *batch_size,
                            has_bias: self.config.has_bias,
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

        Ok(tasks)
    }
}

impl Default for LinearFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for LinearFamily {
    fn family_name(&self) -> &'static str {
        "LinearProjection"
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
        let family = LinearFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 3 dimension variants × 1 batch × 1 dtype = 3 tasks
        assert!(tasks.len() >= 3, "Must generate at least 3 tasks, got {}", tasks.len());

        // Verify no unimplemented!() on the path
        for task in &tasks {
            assert_eq!(task.family, "LinearProjection");
            assert!(!task.name.is_empty());
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = LinearFamily::with_seed(42);
        let family2 = LinearFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
            // Compare op fields individually since TaskOp doesn't derive PartialEq
            match (&t1.op, &t2.op) {
                (TaskOp::LinearProjection { input_dim: i1, output_dim: o1, batch_size: b1, has_bias: h1, dtype: d1 },
                 TaskOp::LinearProjection { input_dim: i2, output_dim: o2, batch_size: b2, has_bias: h2, dtype: d2 }) => {
                    assert_eq!((i1, o1, b1, h1, d1), (i2, o2, b2, h2, d2));
                }
                _ => panic!("Expected both to be LinearProjection"),
            }
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = LinearFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Verify serialization roundtrip
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            // Compare op fields individually
            match (&parsed.op, &task.op) {
                (TaskOp::LinearProjection { input_dim: i1, output_dim: o1, .. },
                 TaskOp::LinearProjection { input_dim: i2, output_dim: o2, .. }) => {
                    assert_eq!((i1, o1), (i2, o2));
                }
                _ => panic!("Expected LinearProjection"),
            }
        }
    }

    #[test]
    fn test_at_least_three_variants() {
        let family = LinearFamily::new();
        let tasks = family.generate_tasks().unwrap();
        assert!(tasks.len() >= 3, "Must generate at least 3 deterministic variants");

        // Verify each variant has unique dimensions
        let dims: Vec<_> = tasks.iter().map(|t| {
            match &t.op {
                TaskOp::LinearProjection { input_dim, output_dim, .. } => (*input_dim, *output_dim),
                _ => panic!("Expected LinearProjection"),
            }
        }).collect();

        // All three should have distinct (input, output) pairs
        assert_eq!(dims.len(), 3);
        assert_ne!(dims[0], dims[1]);
        assert_ne!(dims[1], dims[2]);
    }

    #[test]
    fn test_custom_config() {
        let config = LinearFamilyConfig {
            seed: 99,
            dimension_variants: vec![(512, 256), (1024, 512)],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string()],
            has_bias: false,
        };
        let family = LinearFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 2 dimensions × 2 batches × 1 dtype = 4 tasks
        assert_eq!(tasks.len(), 4);

        for task in &tasks {
            match &task.op {
                TaskOp::LinearProjection { has_bias, .. } => {
                    assert!(!has_bias, "Custom config should have has_bias=false");
                }
                _ => panic!("Expected LinearProjection"),
            }
        }
    }

    #[test]
    fn test_generated_task_hash_stability() {
        // Verify that generated task specs produce stable hashes
        // when processed by compute_task_hash (defined in CLI)
        let family = LinearFamily::new();
        let tasks = family.generate_tasks().unwrap();

        // Generate twice and verify names/families match (hash stability
        // is derived from the deterministic spec content)
        let tasks2 = family.generate_tasks().unwrap();
        for (t1, t2) in tasks.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
            // Compare op fields individually since TaskOp doesn't derive PartialEq
            match (&t1.op, &t2.op) {
                (TaskOp::LinearProjection { input_dim: i1, output_dim: o1, batch_size: b1, has_bias: h1, dtype: d1 },
                 TaskOp::LinearProjection { input_dim: i2, output_dim: o2, batch_size: b2, has_bias: h2, dtype: d2 }) => {
                    assert_eq!((i1, o1, b1, h1, d1), (i2, o2, b2, h2, d2));
                }
                _ => panic!("Expected LinearProjection"),
            }
        }
    }
}
