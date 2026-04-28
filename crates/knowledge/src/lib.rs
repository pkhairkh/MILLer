//! ANE Compiler Knowledge Store
//!
//! Structured knowledge management for compilation decisions,
//! including storage, querying, confidence tracking, conflict resolution,
//! and shard template seed loading.

pub mod compute_plan_verify;
pub mod confidence;
pub mod conflict;
pub mod query;
pub mod shard_template;
pub mod snapshot;
pub mod store;
pub mod transfer;
pub mod update;

use serde::{Deserialize, Serialize};

/// A single observation harvested from MLComputePlan per-op placement data.
///
/// Each observation records whether a specific op was placed on the
/// NeuralEngine by the Core ML compute planner. Because this data is
/// deterministic for a given hardware+OS combination, it carries a
/// high confidence score (0.9).
///
/// These observations are produced by the Python bridge's
/// `compute_plan_harvest` command and ingested into the knowledge
/// store via `ingest_compute_plan_observations()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePlanObservation {
    /// The op pattern (name) from the compute plan, e.g. "linear_1".
    pub op_pattern: String,
    /// The preferred compute device class, e.g. "NeuralEngine" or "CPU".
    pub device_class: String,
    /// Whether the compute planner placed this op on the NeuralEngine.
    pub ane_placed: bool,
    /// Confidence score (always 0.9 for compute plan observations).
    pub confidence: f32,
    /// Number of evidence points (always 1 for a single compute plan run).
    pub evidence_count: usize,
}
