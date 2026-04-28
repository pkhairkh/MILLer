//! Task Generation
//!
//! Generates profiling tasks from task families with
//! configurable parameters and input specifications.
//!
//! The TaskGenerator orchestrates family-specific generators through
//! the `TaskFamilyTrait` interface. Adding a new family requires only:
//! 1. implementing `TaskFamilyTrait` in the family module,
//! 2. adding a variant to `TaskFamilyId`,
//! 3. wiring the variant in `TaskFamilyId::create_generator`.
//!
//! The generator dispatch does not need family-specific special casing
//! beyond typed registration and dispatch.

use crate::families::attention::{AttentionFamily, AttentionFamilyConfig};
use crate::families::decode_step::{DecodeStepFamily, DecodeStepFamilyConfig};
use crate::families::linear::{LinearFamily, LinearFamilyConfig};
use crate::families::lut_projection::{LutProjectionFamily, LutProjectionFamilyConfig};
use crate::families::mlp_block::{MlpBlockFamily, MlpBlockFamilyConfig};
use crate::families::op_remap::OpRemapFamily;
use crate::families::shape_hostile::{ShapeHostileFamily, ShapeHostileFamilyConfig};
use crate::families::shard_survival::ShardSurvivalFamily;
use crate::families::TaskFamilyTrait;
use ane_ir::task_spec::SyntheticTaskSpec;
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Supported task families for generation.
///
/// Each variant corresponds to a family that has a real `TaskFamilyTrait`
/// implementation. All previously scaffolded families are now real:
/// ShapeHostile, OpRemap, and ShardSurvival have been promoted from
/// `unimplemented!()` stubs to full `TaskFamilyTrait` implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskFamilyId {
    LinearProjection,
    LutProjection,
    DecodeStep,
    MlpBlock,
    Attention,
    ShapeHostile,
    OpRemap,
    ShardSurvival,
}

impl TaskFamilyId {
    /// Parse a family identifier from a string.
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "linear" | "linearprojection" | "linear_projection" => {
                Some(TaskFamilyId::LinearProjection)
            }
            "lut" | "lutprojection" | "lut_projection" => Some(TaskFamilyId::LutProjection),
            "decode" | "decodestep" | "decode_step" => Some(TaskFamilyId::DecodeStep),
            "mlp" | "mlpblock" | "mlp_block" => Some(TaskFamilyId::MlpBlock),
            "attn" | "attention" => Some(TaskFamilyId::Attention),
            "shape" | "shapehostile" | "shape_hostile" => Some(TaskFamilyId::ShapeHostile),
            "opremap" | "op_remap" | "remap" => Some(TaskFamilyId::OpRemap),
            "shardsurvival" | "shard_survival" | "survival" => Some(TaskFamilyId::ShardSurvival),
            _ => None,
        }
    }

    /// Get the canonical string name.
    pub fn canonical_name(&self) -> &'static str {
        match self {
            TaskFamilyId::LinearProjection => "LinearProjection",
            TaskFamilyId::LutProjection => "LutProjection",
            TaskFamilyId::DecodeStep => "DecodeStep",
            TaskFamilyId::MlpBlock => "MlpBlock",
            TaskFamilyId::Attention => "Attention",
            TaskFamilyId::ShapeHostile => "ShapeHostile",
            TaskFamilyId::OpRemap => "OpRemap",
            TaskFamilyId::ShardSurvival => "ShardSurvival",
        }
    }

    /// Create the typed family generator for this family id.
    ///
    /// This is the single registration point that maps `TaskFamilyId` variants
    /// to their concrete `TaskFamilyTrait` implementations. The `TaskGenerator`
    /// calls this to obtain a trait object and then dispatches generically.
    fn create_generator(&self, seed: u64) -> Box<dyn TaskFamilyTrait> {
        match self {
            TaskFamilyId::LinearProjection => {
                Box::new(LinearFamily::with_config(LinearFamilyConfig::new(seed)))
            }
            TaskFamilyId::LutProjection => {
                Box::new(LutProjectionFamily::with_config(LutProjectionFamilyConfig::new(seed)))
            }
            TaskFamilyId::DecodeStep => {
                Box::new(DecodeStepFamily::with_config(DecodeStepFamilyConfig::new(seed)))
            }
            TaskFamilyId::MlpBlock => {
                Box::new(MlpBlockFamily::with_config(MlpBlockFamilyConfig::new(seed)))
            }
            TaskFamilyId::Attention => {
                Box::new(AttentionFamily::with_config(AttentionFamilyConfig::new(seed)))
            }
            TaskFamilyId::ShapeHostile => {
                Box::new(ShapeHostileFamily::with_config(ShapeHostileFamilyConfig::new(seed)))
            }
            TaskFamilyId::OpRemap => Box::new(OpRemapFamily::with_seed(seed)),
            TaskFamilyId::ShardSurvival => Box::new(ShardSurvivalFamily::with_seed(seed)),
        }
    }
}

/// Task generator configuration.
pub struct TaskGeneratorConfig {
    /// Random seed for deterministic generation.
    pub seed: u64,
}

impl Default for TaskGeneratorConfig {
    fn default() -> Self {
        Self { seed: 42 }
    }
}

impl TaskGeneratorConfig {
    /// Create a config with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

/// Task generator that orchestrates family-specific generators.
///
/// Dispatches through the `TaskFamilyTrait` interface, so adding a new
/// family requires only implementing the trait and registering it in
/// `TaskFamilyId::create_generator`. The generator orchestration does
/// not need family-specific special casing beyond typed dispatch.
///
/// Generated tasks are persisted as individual TOML files in a
/// predictable directory layout, and can be fed directly into
/// the compile/lab CLI commands.
pub struct TaskGenerator {
    config: TaskGeneratorConfig,
}

impl TaskGenerator {
    /// Create a new task generator with default config.
    pub fn new() -> Self {
        Self { config: TaskGeneratorConfig::default() }
    }

    /// Create a new task generator with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { config: TaskGeneratorConfig::new(seed) }
    }

    /// Create a new task generator with custom config.
    pub fn with_config(config: TaskGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate tasks for a specific family.
    ///
    /// Dispatches through the `TaskFamilyTrait` interface — no
    /// family-specific branching in the generator orchestration.
    /// Use `generate_and_persist` to also write tasks to disk.
    pub fn generate(&self, family: &TaskFamilyId) -> Result<Vec<SyntheticTaskSpec>> {
        let generator = family.create_generator(self.config.seed);
        generator.generate_tasks()
    }

    /// Generate tasks for the linear family specifically.
    ///
    /// Convenience method that avoids needing to construct a
    /// `TaskFamilyId` for the most common case.
    pub fn generate_linear(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::LinearProjection)
    }

    /// Generate tasks for the LUT projection family specifically.
    ///
    /// Convenience method for generating LUT projection tasks.
    pub fn generate_lut(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::LutProjection)
    }

    /// Generate tasks for the decode step family specifically.
    ///
    /// Convenience method for generating decode step tasks.
    pub fn generate_decode_step(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::DecodeStep)
    }

    /// Generate tasks for the MLP block family specifically.
    ///
    /// Convenience method for generating MLP block tasks.
    pub fn generate_mlp(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::MlpBlock)
    }

    /// Generate tasks for the attention family specifically.
    ///
    /// Convenience method for generating attention tasks.
    pub fn generate_attention(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::Attention)
    }

    /// Generate tasks for the shape-hostile family specifically.
    ///
    /// Convenience method for generating shape-hostile tasks with
    /// edge-case tensor dimensions.
    pub fn generate_shape_hostile(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::ShapeHostile)
    }

    /// Generate tasks for the op-remap family specifically.
    ///
    /// Convenience method for generating op remapping tasks that test
    /// alternative op formulations for correctness and performance.
    pub fn generate_op_remap(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::OpRemap)
    }

    /// Generate tasks for the shard-survival family specifically.
    ///
    /// Convenience method for generating shard survival tasks that verify
    /// sharded models compile correctly across shard boundaries.
    pub fn generate_shard_survival(&self) -> Result<Vec<SyntheticTaskSpec>> {
        self.generate(&TaskFamilyId::ShardSurvival)
    }

    /// Generate tasks and persist them to disk.
    ///
    /// Each task is written as a TOML file in the output directory.
    /// The directory structure is:
    ///
    /// ```text
    /// <output_dir>/
    ///   generated_tasks.json       — Manifest of all generated tasks
    ///   <family>/
    ///     <task_name>.toml         — Individual task spec as TOML
    /// ```
    ///
    /// Returns the list of generated specs and the paths where they
    /// were written.
    pub fn generate_and_persist(
        &self,
        family: &TaskFamilyId,
        output_dir: &Path,
    ) -> Result<Vec<(SyntheticTaskSpec, PathBuf)>> {
        let generator = family.create_generator(self.config.seed);
        let tasks = generator.generate_tasks()?;

        // Create family subdirectory
        let family_dir = output_dir.join(family.canonical_name());
        fs::create_dir_all(&family_dir).with_context(|| {
            format!("Failed to create task output dir: {}", family_dir.display())
        })?;

        let mut results = Vec::new();

        for task in &tasks {
            let task_path = family_dir.join(format!("{}.toml", task.name));

            // Serialize the task spec as TOML
            let toml_str = toml::to_string_pretty(task)
                .with_context(|| format!("Failed to serialize task '{}' to TOML", task.name))?;
            fs::write(&task_path, &toml_str)
                .with_context(|| format!("Failed to write task file: {}", task_path.display()))?;

            results.push((task.clone(), task_path));
        }

        // Write a manifest of all generated tasks
        let manifest = GeneratedTasksManifest {
            generator_version: generator.generator_version().to_string(),
            seed: self.config.seed,
            family: family.canonical_name().to_string(),
            task_count: tasks.len(),
            tasks: tasks
                .iter()
                .map(|t| GeneratedTaskEntry {
                    name: t.name.clone(),
                    family: t.family.clone(),
                    description: t.description.clone(),
                })
                .collect(),
        };

        let manifest_path = output_dir.join("generated_tasks.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .with_context(|| "Failed to serialize generated tasks manifest")?;
        fs::write(&manifest_path, &manifest_json)
            .with_context(|| format!("Failed to write manifest: {}", manifest_path.display()))?;

        Ok(results)
    }
}

impl Default for TaskGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Manifest of generated tasks, written to `generated_tasks.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneratedTasksManifest {
    /// Version of the generator that produced these tasks.
    pub generator_version: String,
    /// Seed used for generation.
    pub seed: u64,
    /// Family name.
    pub family: String,
    /// Number of tasks generated.
    pub task_count: usize,
    /// List of generated task entries.
    pub tasks: Vec<GeneratedTaskEntry>,
}

/// A single entry in the generated tasks manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneratedTaskEntry {
    /// Task name.
    pub name: String,
    /// Task family.
    pub family: String,
    /// Task description.
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_linear_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::LinearProjection).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one task");
        for task in &tasks {
            assert_eq!(task.family, "LinearProjection");
        }
    }

    #[test]
    fn test_generate_linear_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_linear().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_linear().unwrap();
        let tasks2 = gen2.generate_linear().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_different_seeds_same_structure() {
        // Different seeds should produce same task names (since names are
        // derived from dimensions, not from the seed value directly),
        // but the seed is available for downstream deterministic use.
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(99);
        let tasks1 = gen1.generate_linear().unwrap();
        let tasks2 = gen2.generate_linear().unwrap();

        // Same structure, same names (names depend on dimensions only)
        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_generate_and_persist() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path();

        let gen = TaskGenerator::new();
        let results =
            gen.generate_and_persist(&TaskFamilyId::LinearProjection, output_dir).unwrap();

        assert!(!results.is_empty());

        // Check that TOML files were written
        for (_, path) in &results {
            assert!(path.exists(), "Task file must exist: {}", path.display());
        }

        // Check that manifest was written
        let manifest_path = output_dir.join("generated_tasks.json");
        assert!(manifest_path.exists(), "Manifest must exist");

        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        let manifest: GeneratedTasksManifest = serde_json::from_str(&manifest_json).unwrap();
        assert_eq!(manifest.family, "LinearProjection");
        assert_eq!(manifest.task_count, results.len());
    }

    #[test]
    fn test_family_id_parsing() {
        assert_eq!(TaskFamilyId::from_str_flexible("linear"), Some(TaskFamilyId::LinearProjection));
        assert_eq!(
            TaskFamilyId::from_str_flexible("LinearProjection"),
            Some(TaskFamilyId::LinearProjection)
        );
        assert_eq!(
            TaskFamilyId::from_str_flexible("linear_projection"),
            Some(TaskFamilyId::LinearProjection)
        );
        assert_eq!(TaskFamilyId::from_str_flexible("lut"), Some(TaskFamilyId::LutProjection));
        assert_eq!(
            TaskFamilyId::from_str_flexible("LutProjection"),
            Some(TaskFamilyId::LutProjection)
        );
        assert_eq!(
            TaskFamilyId::from_str_flexible("lut_projection"),
            Some(TaskFamilyId::LutProjection)
        );
        assert_eq!(TaskFamilyId::from_str_flexible("decode"), Some(TaskFamilyId::DecodeStep));
        assert_eq!(TaskFamilyId::from_str_flexible("DecodeStep"), Some(TaskFamilyId::DecodeStep));
        assert_eq!(TaskFamilyId::from_str_flexible("decode_step"), Some(TaskFamilyId::DecodeStep));
        assert_eq!(TaskFamilyId::from_str_flexible("mlp"), Some(TaskFamilyId::MlpBlock));
        assert_eq!(TaskFamilyId::from_str_flexible("MlpBlock"), Some(TaskFamilyId::MlpBlock));
        assert_eq!(TaskFamilyId::from_str_flexible("mlp_block"), Some(TaskFamilyId::MlpBlock));
        assert_eq!(TaskFamilyId::from_str_flexible("attention"), Some(TaskFamilyId::Attention));
        assert_eq!(TaskFamilyId::from_str_flexible("attn"), Some(TaskFamilyId::Attention));
        assert_eq!(TaskFamilyId::from_str_flexible("Attention"), Some(TaskFamilyId::Attention)); // case-insensitive
        assert_eq!(TaskFamilyId::from_str_flexible("shape"), Some(TaskFamilyId::ShapeHostile));
        assert_eq!(
            TaskFamilyId::from_str_flexible("shapehostile"),
            Some(TaskFamilyId::ShapeHostile)
        );
        assert_eq!(
            TaskFamilyId::from_str_flexible("shape_hostile"),
            Some(TaskFamilyId::ShapeHostile)
        );
        assert_eq!(TaskFamilyId::from_str_flexible("remap"), Some(TaskFamilyId::OpRemap));
        assert_eq!(TaskFamilyId::from_str_flexible("opremap"), Some(TaskFamilyId::OpRemap));
        assert_eq!(TaskFamilyId::from_str_flexible("op_remap"), Some(TaskFamilyId::OpRemap));
        assert_eq!(TaskFamilyId::from_str_flexible("survival"), Some(TaskFamilyId::ShardSurvival));
        assert_eq!(
            TaskFamilyId::from_str_flexible("shardsurvival"),
            Some(TaskFamilyId::ShardSurvival)
        );
        assert_eq!(
            TaskFamilyId::from_str_flexible("shard_survival"),
            Some(TaskFamilyId::ShardSurvival)
        );
    }

    #[test]
    fn test_generate_lut_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::LutProjection).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one LUT task");
        for task in &tasks {
            assert_eq!(task.family, "LutProjection");
        }
    }

    #[test]
    fn test_generate_lut_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_lut().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_lut_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_lut().unwrap();
        let tasks2 = gen2.generate_lut().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_generate_and_persist_lut() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path();

        let gen = TaskGenerator::new();
        let results = gen.generate_and_persist(&TaskFamilyId::LutProjection, output_dir).unwrap();

        assert!(!results.is_empty());

        // Check that TOML files were written
        for (_, path) in &results {
            assert!(path.exists(), "Task file must exist: {}", path.display());
        }

        // Check that manifest was written
        let manifest_path = output_dir.join("generated_tasks.json");
        assert!(manifest_path.exists(), "Manifest must exist");

        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        let manifest: GeneratedTasksManifest = serde_json::from_str(&manifest_json).unwrap();
        assert_eq!(manifest.family, "LutProjection");
        assert_eq!(manifest.task_count, results.len());
    }

    #[test]
    fn test_generate_decode_step_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::DecodeStep).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one decode step task");
        for task in &tasks {
            assert_eq!(task.family, "DecodeStep");
        }
    }

    #[test]
    fn test_generate_decode_step_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_decode_step().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_decode_step_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_decode_step().unwrap();
        let tasks2 = gen2.generate_decode_step().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_generate_and_persist_decode_step() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path();

        let gen = TaskGenerator::new();
        let results = gen.generate_and_persist(&TaskFamilyId::DecodeStep, output_dir).unwrap();

        assert!(!results.is_empty());

        // Check that TOML files were written
        for (_, path) in &results {
            assert!(path.exists(), "Task file must exist: {}", path.display());
        }

        // Check that manifest was written
        let manifest_path = output_dir.join("generated_tasks.json");
        assert!(manifest_path.exists(), "Manifest must exist");

        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        let manifest: GeneratedTasksManifest = serde_json::from_str(&manifest_json).unwrap();
        assert_eq!(manifest.family, "DecodeStep");
        assert_eq!(manifest.task_count, results.len());
    }

    #[test]
    fn test_generate_mlp_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::MlpBlock).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one MLP block task");
        for task in &tasks {
            assert_eq!(task.family, "MlpBlock");
        }
    }

    #[test]
    fn test_generate_mlp_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_mlp().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_mlp_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_mlp().unwrap();
        let tasks2 = gen2.generate_mlp().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_trait_dispatch_matches_convenience() {
        // Verify that the trait-based dispatch produces the same
        // results as the convenience methods.
        let gen = TaskGenerator::with_seed(42);

        let linear_trait = gen.generate(&TaskFamilyId::LinearProjection).unwrap();
        let linear_conv = gen.generate_linear().unwrap();
        assert_eq!(linear_trait.len(), linear_conv.len());
        for (t1, t2) in linear_trait.iter().zip(linear_conv.iter()) {
            assert_eq!(t1.name, t2.name);
        }

        let lut_trait = gen.generate(&TaskFamilyId::LutProjection).unwrap();
        let lut_conv = gen.generate_lut().unwrap();
        assert_eq!(lut_trait.len(), lut_conv.len());
        for (t1, t2) in lut_trait.iter().zip(lut_conv.iter()) {
            assert_eq!(t1.name, t2.name);
        }

        let ds_trait = gen.generate(&TaskFamilyId::DecodeStep).unwrap();
        let ds_conv = gen.generate_decode_step().unwrap();
        assert_eq!(ds_trait.len(), ds_conv.len());
        for (t1, t2) in ds_trait.iter().zip(ds_conv.iter()) {
            assert_eq!(t1.name, t2.name);
        }

        let mlp_trait = gen.generate(&TaskFamilyId::MlpBlock).unwrap();
        let mlp_conv = gen.generate_mlp().unwrap();
        assert_eq!(mlp_trait.len(), mlp_conv.len());
        for (t1, t2) in mlp_trait.iter().zip(mlp_conv.iter()) {
            assert_eq!(t1.name, t2.name);
        }

        let attn_trait = gen.generate(&TaskFamilyId::Attention).unwrap();
        let attn_conv = gen.generate_attention().unwrap();
        assert_eq!(attn_trait.len(), attn_conv.len());
        for (t1, t2) in attn_trait.iter().zip(attn_conv.iter()) {
            assert_eq!(t1.name, t2.name);
        }
    }

    #[test]
    fn test_generate_attention_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::Attention).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one attention task");
        for task in &tasks {
            assert_eq!(task.family, "Attention");
        }
    }

    #[test]
    fn test_generate_attention_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_attention().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_attention_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_attention().unwrap();
        let tasks2 = gen2.generate_attention().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.family, t2.family);
        }
    }

    #[test]
    fn test_generate_and_persist_attention() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path();

        let gen = TaskGenerator::new();
        let results = gen.generate_and_persist(&TaskFamilyId::Attention, output_dir).unwrap();

        assert!(!results.is_empty());

        // Check that TOML files were written
        for (_, path) in &results {
            assert!(path.exists(), "Task file must exist: {}", path.display());
        }

        // Check that manifest was written
        let manifest_path = output_dir.join("generated_tasks.json");
        assert!(manifest_path.exists(), "Manifest must exist");

        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        let manifest: GeneratedTasksManifest = serde_json::from_str(&manifest_json).unwrap();
        assert_eq!(manifest.family, "Attention");
        assert_eq!(manifest.task_count, results.len());
    }

    // ─── ShapeHostile family tests ──────────────────────────────────────

    #[test]
    fn test_generate_shape_hostile_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::ShapeHostile).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one shape-hostile task");
        for task in &tasks {
            assert_eq!(task.family, "ShapeHostile");
        }
    }

    #[test]
    fn test_generate_shape_hostile_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_shape_hostile().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_shape_hostile_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_shape_hostile().unwrap();
        let tasks2 = gen2.generate_shape_hostile().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
        }
    }

    // ─── OpRemap family tests ────────────────────────────────────────────

    #[test]
    fn test_generate_op_remap_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::OpRemap).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one op-remap task");
        for task in &tasks {
            assert_eq!(task.family, "OpRemap");
        }
    }

    #[test]
    fn test_generate_op_remap_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_op_remap().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_op_remap_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_op_remap().unwrap();
        let tasks2 = gen2.generate_op_remap().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
        }
    }

    // ─── ShardSurvival family tests ──────────────────────────────────────

    #[test]
    fn test_generate_shard_survival_family() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate(&TaskFamilyId::ShardSurvival).unwrap();
        assert!(!tasks.is_empty(), "Must generate at least one shard-survival task");
        for task in &tasks {
            assert_eq!(task.family, "ShardSurvival");
        }
    }

    #[test]
    fn test_generate_shard_survival_convenience() {
        let gen = TaskGenerator::new();
        let tasks = gen.generate_shard_survival().unwrap();
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_generate_shard_survival_deterministic() {
        let gen1 = TaskGenerator::with_seed(42);
        let gen2 = TaskGenerator::with_seed(42);
        let tasks1 = gen1.generate_shard_survival().unwrap();
        let tasks2 = gen2.generate_shard_survival().unwrap();

        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.name, t2.name);
        }
    }
}
