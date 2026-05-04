//! LUT Projection task family.
//!
//! Generates profiling tasks for grouped scalar-LUT palettized projection
//! patterns, which are central to ANE palettized inference. This family
//! exercises the `constexpr_lut`-to-`gather` path that Core ML Tools
//! uses for palettized model weights.
//!
//! The LUT projection models the pattern where:
//! - An input tensor of integer indices (derived from palettized weights)
//!   is used to look up values in a per-group LUT.
//! - Each group has its own LUT with `2^bitwidth` entries.
//! - The output is a tensor that approximates the result of a dense
//!   linear projection at reduced precision.
//!
//! This family is more ANE-relevant than plain linear projection because
//! it exercises the LUT/gather path that is central to ANE palettized
//! inference in production models like Qwen3.

use super::TaskFamilyTrait;
use ane_ir::ane_layout::validate_palette_bits;
use ane_ir::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};
use anyhow::Result;

/// Configuration for the LUT projection family generator.
#[derive(Debug, Clone)]
pub struct LutProjectionFamilyConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// LUT bitwidth variants to generate.
    /// Valid values: 1, 2, 3, 4, 6, 8 (per Core ML Tools palettization).
    pub bitwidth_variants: Vec<usize>,
    /// Embedding dimension variants to generate.
    pub embed_dim_variants: Vec<usize>,
    /// Number of LUT groups (per-group palettization granularity).
    pub num_groups: usize,
    /// Vocabulary size (number of possible index values / LUT entries per group).
    /// For bitwidth `b`, `vocab_size = 2^b` is typical.
    pub vocab_size: usize,
    /// Batch sizes to generate.
    pub batch_sizes: Vec<usize>,
    /// Data types to generate.
    pub dtypes: Vec<String>,
}

impl Default for LutProjectionFamilyConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            bitwidth_variants: vec![4, 6, 8],
            embed_dim_variants: vec![128, 256],
            num_groups: 16,
            vocab_size: 0, // 0 means derive from bitwidth: vocab_size = 2^bitwidth
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        }
    }
}

impl LutProjectionFamilyConfig {
    /// Create a new config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed, ..Default::default() }
    }

    /// Create a config with custom bitwidth variants.
    pub fn with_bitwidths(mut self, variants: Vec<usize>) -> Self {
        self.bitwidth_variants = variants;
        self
    }

    /// Create a config with custom embedding dimensions.
    pub fn with_embed_dims(mut self, dims: Vec<usize>) -> Self {
        self.embed_dim_variants = dims;
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

    /// Resolve the vocabulary size for a given bitwidth.
    ///
    /// If `vocab_size` is explicitly set (> 0), it is used directly.
    /// Otherwise, `vocab_size = 2^bitwidth`, which is the natural
    /// number of LUT entries for a given bitwidth.
    fn resolve_vocab_size(&self, bitwidth: usize) -> usize {
        if self.vocab_size > 0 {
            self.vocab_size
        } else {
            1usize << bitwidth
        }
    }
}

/// LUT projection task family generator.
///
/// Generates deterministic LUT projection task specs that can be
/// compiled by the active compile path. Each task variant specifies
/// concrete vocabulary size, embedding dimension, number of groups,
/// LUT bitwidth, and a deterministic seed for weight initialization.
///
/// This family produces tasks that exercise the palettized inference
/// path in Core ML, which is critical for ANE performance in
/// production transformer models.
pub struct LutProjectionFamily {
    config: LutProjectionFamilyConfig,
}

impl LutProjectionFamily {
    /// Create a new LUT projection family generator with default config.
    pub fn new() -> Self {
        Self { config: LutProjectionFamilyConfig::default() }
    }

    /// Create a LUT projection family generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { config: LutProjectionFamilyConfig::new(seed) }
    }

    /// Create a LUT projection family generator with custom config.
    pub fn with_config(config: LutProjectionFamilyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &LutProjectionFamilyConfig {
        &self.config
    }

    /// Generate profiling tasks for LUT projections.
    ///
    /// Produces one `SyntheticTaskSpec` per combination of
    /// (bitwidth_variant, embed_dim, batch_size, dtype). Each task
    /// has a deterministic name derived from its parameters, and the
    /// same config always produces the same set of tasks.
    ///
    /// At minimum, this produces 3 tasks (one per default bitwidth
    /// variant) with embed_dim=128, batch_size=1, and dtype=fp16.
    pub fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>> {
        let mut tasks = Vec::new();

        for bitwidth in &self.config.bitwidth_variants {
            // T-64 (I-38): Use centralized palette bit-width validation
            // from ane_ir::ane_layout instead of inline matches! pattern.
            if let Err(e) = validate_palette_bits(*bitwidth) {
                anyhow::bail!("{}", e);
            }

            let vocab_size = self.config.resolve_vocab_size(*bitwidth);

            for embed_dim in &self.config.embed_dim_variants {
                for batch_size in &self.config.batch_sizes {
                    for dtype in &self.config.dtypes {
                        let task_name = format!(
                            "lut_v{}_e{}_g{}_b{}_{}",
                            vocab_size, embed_dim, self.config.num_groups, bitwidth, dtype
                        );

                        let spec = SyntheticTaskSpec {
                            name: task_name.clone(),
                            family: "LutProjection".to_string(),
                            description: Some(format!(
                                "Generated LUT projection: vocab={}, embed={}, groups={}, bitwidth={}, batch={}, dtype={}",
                                vocab_size, embed_dim, self.config.num_groups, bitwidth, batch_size, dtype
                            )),
                            op: TaskOp::LutProjection {
                                vocab_size,
                                embed_dim: *embed_dim,
                                num_groups: self.config.num_groups,
                                lut_bitwidth: *bitwidth,
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

        Ok(tasks)
    }
}

impl Default for LutProjectionFamily {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskFamilyTrait for LutProjectionFamily {
    fn family_name(&self) -> &'static str {
        "LutProjection"
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
        let family = LutProjectionFamily::new();
        let tasks = family.generate_tasks().unwrap();
        // Default config: 3 bitwidths × 2 embed_dims × 1 batch × 1 dtype = 6 tasks
        assert!(tasks.len() >= 3, "Must generate at least 3 tasks, got {}", tasks.len());

        // Verify no unimplemented!() on the path
        for task in &tasks {
            assert_eq!(task.family, "LutProjection");
            assert!(!task.name.is_empty());
        }
    }

    #[test]
    fn test_generate_tasks_deterministic() {
        let family1 = LutProjectionFamily::with_seed(42);
        let family2 = LutProjectionFamily::with_seed(42);
        let tasks1 = family1.generate_tasks().unwrap();
        let tasks2 = family2.generate_tasks().unwrap();

        assert_eq!(tasks1.len(), tasks2.len(), "Same config must produce same task count");
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name, "Task names must be identical for same config");
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (
                    TaskOp::LutProjection {
                        vocab_size: v1,
                        embed_dim: e1,
                        num_groups: g1,
                        lut_bitwidth: b1,
                        batch_size: bs1,
                        dtype: d1,
                    },
                    TaskOp::LutProjection {
                        vocab_size: v2,
                        embed_dim: e2,
                        num_groups: g2,
                        lut_bitwidth: b2,
                        batch_size: bs2,
                        dtype: d2,
                    },
                ) => {
                    assert_eq!((v1, e1, g1, b1, bs1, d1), (v2, e2, g2, b2, bs2, d2));
                }
                _ => panic!("Expected both to be LutProjection"),
            }
        }
    }

    #[test]
    fn test_generated_tasks_serialize_and_parse() {
        let family = LutProjectionFamily::new();
        let tasks = family.generate_tasks().unwrap();

        for task in &tasks {
            // Verify serialization roundtrip
            let json = serde_json::to_string(task).unwrap();
            let parsed: SyntheticTaskSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.name, task.name);
            assert_eq!(parsed.family, task.family);
            match (&parsed.op, &task.op) {
                (
                    TaskOp::LutProjection { lut_bitwidth: b1, .. },
                    TaskOp::LutProjection { lut_bitwidth: b2, .. },
                ) => {
                    assert_eq!(b1, b2);
                }
                _ => panic!("Expected LutProjection"),
            }
        }
    }

    #[test]
    fn test_at_least_three_variants() {
        let family = LutProjectionFamily::new();
        let tasks = family.generate_tasks().unwrap();
        assert!(tasks.len() >= 3, "Must generate at least 3 deterministic variants");

        // Verify each variant has a distinct bitwidth
        let bitwidths: Vec<usize> = tasks
            .iter()
            .map(|t| match &t.op {
                TaskOp::LutProjection { lut_bitwidth, .. } => *lut_bitwidth,
                _ => panic!("Expected LutProjection"),
            })
            .collect();

        // Default config should have at least 3 distinct bitwidths
        let unique_bitwidths: std::collections::HashSet<usize> = bitwidths.into_iter().collect();
        assert!(unique_bitwidths.len() >= 3, "Should have at least 3 distinct bitwidths");
    }

    #[test]
    fn test_custom_config() {
        let config = LutProjectionFamilyConfig {
            seed: 99,
            bitwidth_variants: vec![4, 8],
            embed_dim_variants: vec![512],
            num_groups: 32,
            vocab_size: 256, // explicit vocab_size
            batch_sizes: vec![1, 2],
            dtypes: vec!["fp16".to_string()],
        };
        let family = LutProjectionFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();

        // 2 bitwidths × 1 embed_dim × 2 batches × 1 dtype = 4 tasks
        assert_eq!(tasks.len(), 4);

        for task in &tasks {
            match &task.op {
                TaskOp::LutProjection { vocab_size, num_groups, .. } => {
                    assert_eq!(*vocab_size, 256, "Custom config should have vocab_size=256");
                    assert_eq!(*num_groups, 32, "Custom config should have num_groups=32");
                }
                _ => panic!("Expected LutProjection"),
            }
        }
    }

    #[test]
    fn test_invalid_bitwidth_rejected() {
        let config = LutProjectionFamilyConfig {
            seed: 42,
            bitwidth_variants: vec![5], // Invalid
            embed_dim_variants: vec![128],
            num_groups: 16,
            vocab_size: 0,
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = LutProjectionFamily::with_config(config);
        let result = family.generate_tasks();
        assert!(result.is_err(), "Invalid bitwidth must be rejected");
    }

    #[test]
    fn test_vocab_size_derived_from_bitwidth() {
        let config = LutProjectionFamilyConfig {
            seed: 42,
            bitwidth_variants: vec![4],
            embed_dim_variants: vec![128],
            num_groups: 16,
            vocab_size: 0, // Derive from bitwidth
            batch_sizes: vec![1],
            dtypes: vec!["fp16".to_string()],
        };
        let family = LutProjectionFamily::with_config(config);
        let tasks = family.generate_tasks().unwrap();
        assert_eq!(tasks.len(), 1);
        match &tasks[0].op {
            TaskOp::LutProjection { vocab_size, lut_bitwidth, .. } => {
                assert_eq!(*vocab_size, 16); // 2^4 = 16
                assert_eq!(*lut_bitwidth, 4);
            }
            _ => panic!("Expected LutProjection"),
        }
    }

    #[test]
    fn test_generated_task_hash_stability() {
        let family = LutProjectionFamily::new();
        let tasks = family.generate_tasks().unwrap();

        // Generate twice and verify names/families match
        let tasks2 = family.generate_tasks().unwrap();
        for (t1, t2) in tasks.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
            match (&t1.op, &t2.op) {
                (
                    TaskOp::LutProjection {
                        vocab_size: v1,
                        embed_dim: e1,
                        num_groups: g1,
                        lut_bitwidth: b1,
                        batch_size: bs1,
                        dtype: d1,
                    },
                    TaskOp::LutProjection {
                        vocab_size: v2,
                        embed_dim: e2,
                        num_groups: g2,
                        lut_bitwidth: b2,
                        batch_size: bs2,
                        dtype: d2,
                    },
                ) => {
                    assert_eq!((v1, e1, g1, b1, bs1, d1), (v2, e2, g2, b2, bs2, d2));
                }
                _ => panic!("Expected LutProjection"),
            }
        }
    }
}
