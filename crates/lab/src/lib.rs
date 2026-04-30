//! MILLer
//!
//! Profiling lab for generating tasks, running benchmarks,
//! detecting drift and fallback, and managing task families.
//!
//! Key types:
//! - `LabRun` — structured record of a lab run (compilation + inspection + optional profiling)
//! - `HostInspector` — host-side inspection of mlpackage artifacts
//! - `LabRunWriter` — writes lab run artifacts in canonical directory layout
//! - `FallbackDetector` — honest, weak fallback suspicion assessment
//! - `mir_compare` — MIR-vs-emitted-structure comparison (Sprint 34)

pub mod baseline;
pub mod device_meta;
pub mod drift;
pub mod fallback;
pub mod families;
pub mod harness;
pub mod host_inspect;
pub mod mir_compare;
pub mod run_dir;
pub mod session;
pub mod task_gen;
