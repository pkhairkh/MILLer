//! Task Families
//!
//! Collection of profiling task families, each targeting
//! a specific aspect of ANE compilation behavior.
//!
//! Each family implements the `TaskFamilyTrait` which provides a
//! uniform interface for the `TaskGenerator` to dispatch through,
//! eliminating ad hoc branching in orchestration code.

pub mod linear;
pub mod lut_projection;
pub mod decode_step;
pub mod mlp_block;
pub mod attention;
pub mod shape_hostile;
pub mod op_remap;
pub mod shard_survival;

use ane_ir::task_spec::SyntheticTaskSpec;
use anyhow::Result;

/// Trait for task family generators.
///
/// Each family that enters the active generation surface must implement
/// this trait. The `TaskGenerator` dispatches through this interface
/// rather than ad hoc branching, so adding a new family requires only:
/// 1. implementing this trait,
/// 2. registering the family in `TaskFamilyId::create_generator`.
///
/// Families on the active generation surface implement this trait.
/// As of the Sprint 54 audit pass, all currently registered families are
/// real `TaskFamilyTrait` implementations; adding a new scaffolded family
/// should not be considered part of the active surface until it implements
/// this trait and is wired into `TaskFamilyId`.
pub trait TaskFamilyTrait: Send + Sync {
    /// The canonical family name (e.g., "LinearProjection", "LutProjection").
    fn family_name(&self) -> &'static str;

    /// The generator version string, for provenance tracking.
    fn generator_version(&self) -> &'static str;

    /// Generate profiling tasks for this family.
    ///
    /// Returns a list of deterministic `SyntheticTaskSpec` instances.
    /// The same configuration always produces the same set of tasks.
    fn generate_tasks(&self) -> Result<Vec<SyntheticTaskSpec>>;
}
