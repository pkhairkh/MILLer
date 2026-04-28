//! Shape Hostile task family.
//!
//! Generates profiling tasks with edge-case tensor shapes that are known to
//! cause ANE compilation issues, silent fallbacks, or placement failures.
//!
//! This family reuses `TaskOp::LinearProjection` as its op type because
//! shape-hostile testing is about the *dimensions* of the tensors, not about
//! exercising a different op. The generated tasks sweep shapes that are
//! empirically problematic for ANE compilation:
//!
//! - **Odd dimensions**: Non-power-of-2 input/output dims (e.g., 63, 127, 255)
//!   that may fail alignment requirements on the ANE.
//! - **Prime dimensions**: Dimensions that are prime numbers (e.g., 37, 73, 97)
//!   that resist factorization for tiling strategies.
//! - **Large dimensions**: Dimensions that push toward or past known ANE limits
//!   (e.g., 2048, 4096) to explore the feasibility frontier.
//! - **Mismatched ratios**: Input/output dimension ratios that are not powers of 2
//!   (e.g., 128x63, 256x97) that may cause suboptimal ANE scheduling.
//!
//! The family is a dimension-sweep tool, not a new op family. Generated tasks
//! can be compiled, verified, and profiled through the existing pipeline to
//! discover ANE placement boundaries and fallback patterns.

use super::TaskFamilyTrait;
use ane_ir::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};
use anyhow::Result;

/// Configuration for the shape-hostile family generator.
#[derive(Debug, Clone)]
pub struct ShapeHostileFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Hostile shape patterns to generate.
    pub patterns: Vec<HostilePattern>,
    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,
    /// Data types to test.
    pub dtypes: Vec<String>,
    /// Whether to include bias in projections.
    pub has_bias: bool,
}

/// A shape pattern known to be potentially hostile to ANE compilation.
#[derive(Debug, Clone)]
pub enum HostilePattern {
    /// Odd (non-power-of-2) dimensions: (input_dim, output_dim).
    /// Example: (63, 127) — neither dimension is a power of 2.
    OddDimensions { input_dim: usize, output_dim: usize },
    /// Prime dimensions: (input_dim, output_dim).
    /// Both dimensions are prime numbers, resisting factorization.
    PrimeDimensions { input_dim: usize, output_dim: usize },
    /// Large dimensions that push ANE limits: (input_dim, output_dim).
    /// These explore the feasibility frontier.
    LargeDimensions { input_dim: usize, output_dim: usize },
    /// Mismatched ratio dimensions: (input_dim, output_dim).
    /// The ratio input_dim/output_dim is not a power of 2.
    MismatchedRatio { input_dim: usize, output_dim: usize },
}

impl HostilePattern {
    /// Return a human-readable name for this pattern.
    pub fn pattern_name(&self) -> &'static str {
        match self {
            HostilePattern::OddDimensions { .. } => "odd",
            HostilePattern::PrimeDimensions { .. } => "prime",
            HostilePattern::LargeDimensions { .. } => "large",
            HostilePattern::MismatchedRatio { .. } => "mismatch",
        }
    }

    /// Return the (input_dim, output_dim) for this pattern.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            HostilePattern::OddDimensions { input_dim, output_dim } => (*input_dim, *output_dim),
            HostilePattern::PrimeDimensions { input_dim, output_dim } => (*input_dim, *output_dim),
            HostilePattern::LargeDimensions { input_dim, output_dim } => (*input_dim, *output_dim),
            HostilePattern::MismatchedRatio { input_dim, output_dim } => (*input_dim, *output_dim),
        }
    }
}

impl Default for ShapeHostileFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            patterns: vec![
                // Odd dimensions — non-power-of-2
                HostilePattern::OddDimensions { input_dim: 63, output_dim: 127 },
                HostilePattern::OddDimensions { input_dim: 127, output_dim: 63 },
                // Prime dimensions — resist factorization
                HostilePattern::PrimeDimensions { input_dim: 37, output_dim: 73 },
                HostilePattern::PrimeDimensions { input_dim: 97, output_dim: 37 },
                // Large dimensions — push toward ANE limits
                HostilePattern::LargeDimensions { input_dim: 2048, output_dim: 1024 },
                HostilePattern::LargeDimensions { input_dim: 4096, output_dim: 2048 },
                // Mismatched ratios — non-power-of-2 ratio
                HostilePattern::MismatchedRatio { input_dim: 128, output_dim: 63 },
                HostilePattern::MismatchedRatio { input_dim: 256, output_dim: 97 },
            ],
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
            has_bias: true,
        }
    }
}

impl ShapeHostileFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed, ..Default::default() }
    }

    /// Create a config with custom patterns.
    pub fn with_patterns(mut self, patterns: Vec<HostilePattern>) -> Self {
        self.patterns = patterns;
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

/// Shape-hostile task family generator.
///
/// Generates deterministic task specs with edge-case tensor dimensions
/// that are known to be problematic for ANE compilation. Each task
/// is a `LinearProjection` with a hostile shape pattern, allowing
/// the existing compile/lab pipeline to discover ANE placement
/// boundaries and fallback patterns.
///
/// This family addresses ISSUES.md item #3: the repo can sweep shapes
/// through family generators but `shape_hostile` was previously a
/// literal `unimplemented!()` stub. With this implementation, the
/// lab task generation subsystem can produce shape-hostile specs
/// that feed into the compile/verify/profile pipeline to explore
/// feasibility frontiers.
pub struct ShapeHostileFamily {
    config: ShapeHostileFamilyConfig,
}

impl ShapeHostileFamily {
    /// Create a new shape-hostile family generator with default config.
    pub fn new() -> Self {
        Self { config: ShapeHostileFamilyConfig::default() }
    }

    /// Create a shape-hostile family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { config: ShapeHostileFamilyConfig::new(seed) }
    }

    /// Create a shape-hostile family generator with custom config.
    pub fn with_config(config: ShapeHostileFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &ShapeHostileFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for shape-hostile patterns.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (pattern, batch_size, dtype). Each task has a deterministic
    /// name derived from its pattern type and dimensions, and the
    /// same config always produces the same set of tasks.
    ///
    /// At minimum, this produces 8 tasks (one per default pattern)
    /// with batch_size=1 and dtype=fp16.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for pattern in &self.config.patterns {
            let (input_dim, output_dim) = pattern.dims();
            let pattern_name = pattern.pattern_name();

            for batch_size in &self.config.batch_sizes {
                for dtype in &self.config.dtypes {
                    let task_name = format!(
                        "shape_hostile_{}_{}x{}_b{}_{}",
                        pattern_name, input_dim, output_dim, batch_size, dtype
                    );

                    let spec = SyntheticTaskSpec {
                        name: task_name.clone(),
                        family: "ShapeHostile".to_string(),
                        description: Some(format!(
                            "Shape-hostile linear projection ({} pattern): [{}] -> [{}], batch={}, dtype={}",
                            pattern_name, input_dim, output_dim, batch_size, dtype
                        )),
                        op: TaskOp::LinearProjection {
                            input_dim,
                            output_dim,
                            batch_size: *batch_size,
                            has_bias: self.config.has_bias,
                            dtype: dtype.clone(),
                        },
                        measurement: MeasurementConfig {
                            warmup_iterations: 5,
                            measured_iterations: 20,
                            metrics: vec!["Latency".to_string(), "FallbackSuspicion".to_string()],
                        },
                    };

                    tasks.push(spec);
                }
            }
        }

        Ok(tasks)
    }
}

impl Default for ShapeHostileFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for ShapeHostileFamily {
    fn family_name(&self) -> &'static str {
        "ShapeHostile"
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
        let family = ShapeHostileFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 8 patterns × 1 batch × 1 dtype = 8 tasks
        assert!(tasks.len() >= 8, "Must generate at least 8 tasks, got {}", tasks.len());

        // Verify no unimplemented!() on the path
        for task in &tasks {
            assert_eq!(task.family, "ShapeHostile");
            assert!(!task.name.is_empty());
            assert!(task.name.starts_with("shape_hostile_"));
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = ShapeHostileFamily::with_seed(42);
        let family2 = ShapeHostileFamily::with_seed(42);
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
        let family = ShapeHostileFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Verify serialization roundtrip
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            // All shape-hostile tasks are LinearProjection under the hood
            match (&parsed.op, &task.op) {
                (
                    TaskOp::LinearProjection { input_dim: i1, output_dim: o1, .. },
                    TaskOp::LinearProjection { input_dim: i2, output_dim: o2, .. },
                ) => {
                    assert_eq!((i1, o1), (i2, o2));
                }
                _ => panic!("Expected LinearProjection"),
            }
        }
    }

    #[test]
    fn test_all_pattern_types_present() {
        let family = ShapeHostileFamily::new();
        let tasks = family.generate_tasks().unwrap();

        // Check that all four pattern types are represented
        let names: Vec<&str> = tasks.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("_odd_")), "Must have odd dimension tasks");
        assert!(names.iter().any(|n| n.contains("_prime_")), "Must have prime dimension tasks");
        assert!(names.iter().any(|n| n.contains("_large_")), "Must have large dimension tasks");
        assert!(names.iter().any(|n| n.contains("_mismatch_")), "Must have mismatched ratio tasks");
    }

    #[test]
    fn test_custom_config() {
        let config = ShapeHostileFamilyConfig {
            seed: 99,
            patterns: vec![
                HostilePattern::OddDimensions { input_dim: 33, output_dim: 65 },
                HostilePattern::PrimeDimensions { input_dim: 13, output_dim: 29 },
            ],
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            has_bias: false,
        };
        let family = ShapeHostileFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 2 patterns × 2 batches × 2 dtypes = 8 tasks
        assert_eq!(tasks.len(), 8);

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
    fn test_task_names_include_pattern_info() {
        let family = ShapeHostileFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Names should include pattern type, dimensions, batch, and dtype
            assert!(task.name.contains("shape_hostile_"));
            // Description should mention the pattern type
            let desc = task.description.as_ref().unwrap();
            assert!(desc.contains("pattern") || desc.contains("Shape-hostile"));
        }
    }

    #[test]
    fn test_measurement_includes_fallback_suspicion() {
        let family = ShapeHostileFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            assert!(
                task.measurement.metrics.contains(&"FallbackSuspicion".to_string()),
                "Shape-hostile tasks should measure FallbackSuspicion metric"
            );
        }
    }

    #[test]
    fn test_trait_dispatch_works() {
        let family = ShapeHostileFamily::new();
        let trait_gen: &dyn TaskFamilyTrait = &family;
        assert_eq!(trait_gen.family_name(), "ShapeHostile");
        assert_eq!(trait_gen.generator_version(), "1.0.0");
        let tasks = trait_gen.generate_tasks().unwrap();
        assert!(!tasks.is_empty());
    }
}
