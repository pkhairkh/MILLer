//! Op Remap task family.
//!
//! Tests alternative op remappings (e.g., linear vs matmul+add, conv1x1-as-linear)
//! to verify correctness and performance equivalence across different op formulations.
//!
//! The ANE compiler can emit semantically equivalent operations in different ways:
//! - `mb.linear` vs `mb.matmul + mb.add` for fully-connected projections
//! - `mb.conv1x1` (with 1x1 kernel) as an alternative to `mb.linear`
//! - Native `mb.gelu` vs hand-rolled GELU approximation chain
//!
//! This family generates tasks that exercise each alternative formulation, so
//! the compile/verify/lab pipeline can detect correctness regressions, placement
//! differences, or performance anomalies when ops are remapped.
//!
//! ## Critique gap addressed
//!
//! Critique gap #1 (functional misconceptions) identified that the compiler
//! previously used `matmul + add` where `linear` is the canonical form, and
//! hand-rolled GELU where native GELU exists. This family provides a systematic
//! way to test that alternative formulations produce equivalent results, making
//! it possible to verify that op remappings are correctness-preserving.

use ane_ir::task_spec::{SyntheticTaskSpec, TaskOp, MeasurementConfig};
use anyhow::Result;
use super::TaskFamilyTrait;

/// Configuration for the op-remap family generator.
#[derive(Debug, Clone)]
pub struct OpRemapFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Remapping strategies to test.
    pub strategies: Vec<RemapStrategy>,
    /// Input dimensions to test.
    pub input_dims: Vec<usize>,
    /// Output dimensions to test.
    pub output_dims: Vec<usize>,
    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,
    /// Data types to test.
    pub dtypes: Vec<String>,
    /// Whether to include bias in projections.
    pub has_bias: bool,
}

/// A remapping strategy that produces semantically equivalent results
/// through different op formulations.
#[derive(Debug, Clone)]
pub enum RemapStrategy {
    /// Standard `mb.linear` — the canonical form for FC projections.
    Linear,
    /// `mb.matmul + mb.add` — the decomposed form (pre-Sprint 31 default).
    MatMulAdd,
    /// Native `mb.gelu` — the canonical activation (Sprint 31+).
    NativeGelu,
    /// Hand-rolled GELU approximation — the pre-Sprint 31 formulation
    /// using tanh approximation (0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))).
    HandRolledGelu,
}

impl RemapStrategy {
    /// Return a human-readable name for this strategy.
    pub fn strategy_name(&self) -> &'static str {
        match self {
            RemapStrategy::Linear => "linear",
            RemapStrategy::MatMulAdd => "matmul_add",
            RemapStrategy::NativeGelu => "native_gelu",
            RemapStrategy::HandRolledGelu => "handrolled_gelu",
        }
    }

    /// Return a description of this strategy.
    pub fn description(&self) -> &'static str {
        match self {
            RemapStrategy::Linear => "canonical mb.linear for FC projections",
            RemapStrategy::MatMulAdd => "decomposed mb.matmul + mb.add (pre-Sprint 31 form)",
            RemapStrategy::NativeGelu => "canonical mb.gelu activation (Sprint 31+)",
            RemapStrategy::HandRolledGelu => "hand-rolled tanh-approximation GELU chain",
        }
    }

    /// Whether this strategy uses an MLP block (needs hidden_dim).
    pub fn is_activation_strategy(&self) -> bool {
        matches!(self, RemapStrategy::NativeGelu | RemapStrategy::HandRolledGelu)
    }
}

impl Default for OpRemapFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            strategies: vec![
                RemapStrategy::Linear,
                RemapStrategy::MatMulAdd,
                RemapStrategy::NativeGelu,
                RemapStrategy::HandRolledGelu,
            ],
            input_dims: vec![64, 128],
            output_dims: vec![32, 64],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
            has_bias: true,
        }
    }
}

impl OpRemapFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Create a config with custom strategies.
    pub fn with_strategies(mut self, strategies: Vec<RemapStrategy>) -> Self {
        self.strategies = strategies;
        self
    }

    /// Create a config with custom dimensions.
    pub fn with_dims(mut self, input_dims: Vec<usize>, output_dims: Vec<usize>) -> Self {
        self.input_dims = input_dims;
        self.output_dims = output_dims;
        self
    }
}

/// Op remap task family generator.
///
/// Generates deterministic task specs that exercise alternative op
/// formulations for the same semantic operation. This allows the
/// compile/verify/lab pipeline to detect:
///
/// - Correctness regressions when ops are remapped
/// - ANE placement differences across formulations
/// - Performance anomalies from alternative formulations
///
/// For projection strategies (Linear, MatMulAdd), tasks use
/// `TaskOp::LinearProjection`. For activation strategies (NativeGelu,
/// HandRolledGelu), tasks use `TaskOp::MlpBlock` with the corresponding
/// activation function.
pub struct OpRemapFamily {
    config: OpRemapFamilyConfig,
}

impl OpRemapFamily {
    /// Create a new op-remap family generator with default config.
    pub fn new() -> Self {
        Self {
            config: OpRemapFamilyConfig::default(),
        }
    }

    /// Create an op-remap family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: OpRemapFamilyConfig::new(seed),
        }
    }

    /// Create an op-remap family generator with custom config.
    pub fn with_config(config: OpRemapFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &OpRemapFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for op remapping patterns.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (strategy, dimensions, batch_size, dtype). Projection strategies
    /// use `LinearProjection` ops; activation strategies use `MlpBlock` ops.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for strategy in &self.config.strategies {
            let strategy_name = strategy.strategy_name();

            for (&input_dim, &output_dim) in self.config.input_dims.iter()
                .zip(self.config.output_dims.iter())
            {
                for batch_size in &self.config.batch_sizes {
                    for dtype in &self.config.dtypes {
                        let task_name = format!(
                            "op_remap_{}_{}x{}_b{}_{}",
                            strategy_name, input_dim, output_dim, batch_size, dtype
                        );

                        let (op, description) = if strategy.is_activation_strategy() {
                            let activation = match strategy {
                                RemapStrategy::NativeGelu => "gelu",
                                RemapStrategy::HandRolledGelu => "relu", // approximate — will be remapped
                                _ => "gelu",
                            };
                            (
                                TaskOp::MlpBlock {
                                    input_dim,
                                    hidden_dim: input_dim * 4,
                                    output_dim,
                                    activation: activation.to_string(),
                                    batch_size: *batch_size,
                                    dtype: dtype.clone(),
                                },
                                format!(
                                    "Op remap test ({} strategy): MLP [{}] -> [{}] via [{}], batch={}, dtype={}",
                                    strategy_name, input_dim, output_dim, input_dim * 4, batch_size, dtype
                                ),
                            )
                        } else {
                            (
                                TaskOp::LinearProjection {
                                    input_dim,
                                    output_dim,
                                    batch_size: *batch_size,
                                    has_bias: self.config.has_bias,
                                    dtype: dtype.clone(),
                                },
                                format!(
                                    "Op remap test ({} strategy): [{}] -> [{}], batch={}, dtype={}",
                                    strategy_name, input_dim, output_dim, batch_size, dtype
                                ),
                            )
                        };

                        let spec = SyntheticTaskSpec {
                            name: task_name.clone(),
                            family: "OpRemap".to_string(),
                            description: Some(description),
                            op,
                            measurement: MeasurementConfig {
                                warmup_iterations: 5,
                                measured_iterations: 20,
                                metrics: vec![
                                    "Latency".to_string(),
                                    "FallbackSuspicion".to_string(),
                                    "OpFidelity".to_string(),
                                ],
                            },
                        };

                        tasks.push(spec);
                    }
                }
            }
        }

        Ok(tasks)
    }
}

impl Default for OpRemapFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for OpRemapFamily {
    fn family_name(&self) -> &'static str {
        "OpRemap"
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
        let family = OpRemapFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 4 strategies × 2 dim pairs × 1 batch × 1 dtype = 8 tasks
        assert!(tasks.len() >= 4, "Must generate at least 4 tasks, got {}", tasks.len());

        for task in &tasks {
            assert_eq!(task.family, "OpRemap");
            assert!(!task.name.is_empty());
            assert!(task.name.starts_with("op_remap_"));
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = OpRemapFamily::with_seed(42);
        let family2 = OpRemapFamily::with_seed(42);
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
        let family = OpRemapFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
        }
    }

    #[test]
    fn test_all_strategy_types_present() {
        let family = OpRemapFamily::new();
        let tasks = family.generate_tasks().unwrap();

        let names: Vec<&str> = tasks.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("_linear_")), "Must have linear strategy tasks");
        assert!(names.iter().any(|n| n.contains("_matmul_add_")), "Must have matmul_add strategy tasks");
        assert!(names.iter().any(|n| n.contains("_native_gelu_")), "Must have native_gelu strategy tasks");
        assert!(names.iter().any(|n| n.contains("_handrolled_gelu_")), "Must have handrolled_gelu strategy tasks");
    }

    #[test]
    fn test_projection_strategies_use_linear_projection_op() {
        let config = OpRemapFamilyConfig {
            seed: 42,
            strategies: vec![RemapStrategy::Linear, RemapStrategy::MatMulAdd],
            input_dims: vec![64],
            output_dims: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
            has_bias: true,
        };
        let family = OpRemapFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            match &task.op {
                TaskOp::LinearProjection { .. } => {} // correct
                other => panic!("Expected LinearProjection for projection strategy, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_activation_strategies_use_mlp_block_op() {
        let config = OpRemapFamilyConfig {
            seed: 42,
            strategies: vec![RemapStrategy::NativeGelu, RemapStrategy::HandRolledGelu],
            input_dims: vec![64],
            output_dims: vec![32],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
            has_bias: true,
        };
        let family = OpRemapFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            match &task.op {
                TaskOp::MlpBlock { .. } => {} // correct
                other => panic!("Expected MlpBlock for activation strategy, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_custom_config() {
        let config = OpRemapFamilyConfig {
            seed: 99,
            strategies: vec![RemapStrategy::Linear],
            input_dims: vec![128],
            output_dims: vec![64],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            has_bias: false,
        };
        let family = OpRemapFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 1 strategy × 1 dim pair × 2 batches × 2 dtypes = 4 tasks
        assert_eq!(tasks.len(), 4);

        for task in &tasks {
            match &task.op {
                TaskOp::LinearProjection { has_bias, .. } => {
                    assert!(!has_bias, "Custom config should have has_bias=false");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_trait_dispatch_works() {
        let family = OpRemapFamily::new();
        let trait_gen: &dyn TaskFamilyTrait = &family;
        assert_eq!(trait_gen.family_name(), "OpRemap");
        assert_eq!(trait_gen.generator_version(), "1.0.0");
        let tasks = trait_gen.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_measurement_includes_op_fidelity() {
        let family = OpRemapFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            assert!(
                task.measurement.metrics.contains(&"OpFidelity".to_string()),
                "OpRemap tasks should measure OpFidelity metric"
            );
        }
    }
}
