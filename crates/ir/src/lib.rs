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

pub mod air;
pub mod ane_engine;
pub mod ane_hw_limits;
pub mod ane_layout;
pub mod ane_target;
pub mod kir;
pub mod linear_slice;
pub mod mir;
pub mod pir;
pub mod prof_ir;
pub mod serialize;
pub mod sir;
pub mod task_spec;
