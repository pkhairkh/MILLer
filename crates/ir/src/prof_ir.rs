//! Profiling/Task IR (ProfIR)
//!
//! Represents a profiling task: what to compile, run, measure,
//! and compare against.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileTask {
    pub task_id: String,
    pub family: TaskFamily,
    pub mil_package_ref: String,
    pub inputs: TaskInputSpec,
    pub baseline: BaselineReference,
    pub metrics: Vec<Metric>,
    pub device_requirements: DeviceRequirement,
    pub repetition_count: usize,
    pub warmup_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskFamily {
    LinearProjection,
    LutProjection,
    MlpBlock,
    AttentionMicroblock,
    DecodeStep,
    ShapeHostile,
    OpRemap,
    ShardSurvival,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInputSpec {
    pub shapes: Vec<Vec<usize>>,
    pub dtypes: Vec<String>,
    pub value_range: (f32, f32),
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineReference {
    Fp32CpuReference,
    NumpyComputation,
    ReferenceArtifact { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metric {
    Latency,
    Throughput,
    CosineDistance,
    MaxAbsoluteError,
    MeanAbsoluteError,
    RelativeErrorP99,
    FallbackSuspicion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRequirement {
    pub device_class: Option<String>,
    pub os_version_range: Option<(String, String)>,
    pub compute_units: String,
}
