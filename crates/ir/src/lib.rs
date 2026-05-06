//! ANE Compiler IR Definitions
//!
//! Multi-level intermediate representation stack:
//! - SIR: Semantic/Task IR
//! - AIR: ANE-Legal IR
//! - MIR: MIL-Emission IR
//! - PIR: Package/Deployment IR
//! - ProfIR: Profiling/Task IR
//! - KIR: Backend-Knowledge Representation IR
//! - TaskSpec: Concrete task specification types

/// Default Core ML opset version used for emission.
///
/// This constant centralizes the opset version string that was previously
/// hardcoded across multiple files. Update this when targeting newer
/// Core ML runtime versions.
pub const DEFAULT_OPSET_VERSION: &str = "iOS18";

/// T-115: Default minimum deployment target for Core ML models.
///
/// Decoupled from `DEFAULT_OPSET_VERSION` because the opset version
/// (which MIL opset the model uses) and the deployment target
/// (which OS version the model requires) can differ. For example,
/// a model may use iOS18 opset features but only require iOS17
/// as a deployment target if those features are backward-compatible.
pub const DEFAULT_MINIMUM_DEPLOYMENT_TARGET: &str = "iOS18";

pub mod air;
pub mod ane_engine;
pub mod ane_hw_limits;
pub mod ane_layout;
pub mod ane_placement;
pub mod ane_target;
pub mod common;
pub mod kir;
pub mod linear_slice;
pub mod mir;
pub mod multi_ane;
pub mod payload;
pub mod placement_dialect;
pub mod pir;
pub mod prof_ir;
pub mod serialize;
pub mod shape_ops;
pub mod shard_desc;
pub mod sir;
pub mod strategy;
pub mod task_spec;
pub mod toproto;

// Re-export key types for convenience.
pub use common::VerifyError;
pub use air::LegalityStatus;
pub use air::LegacyAirNodeFields;

#[cfg(test)]
mod mir_engine_test;
