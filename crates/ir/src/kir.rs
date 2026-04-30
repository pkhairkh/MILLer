//! Backend-Knowledge Representation IR (KIR)
//!
//! Schema for the knowledge store. Each knowledge unit
//! is structured, versioned, and confidence-scored.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum KnowledgeType {
    LegalityRule,
    MotifCatalog,
    SurvivalMatrixEntry,
    ShardTemplateKnowledge,
    PrecisionHazard,
    FallbackSignature,
    DeviceFingerprint,
    StateTopologyOutcome,
    SyntheticTransferAnnotation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeUnit {
    pub id: String,
    pub version: u64,
    pub timestamp: String,
    pub knowledge_type: KnowledgeType,
    pub confidence: f32,
    pub evidence_source: EvidenceSource,
    pub evidence_count: usize,
    pub scope: KnowledgeScope,
    pub conflict_priority: u32,
    pub payload: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvidenceSource {
    SyntheticRun,
    RealModelRun,
    CompileFailure,
    LoadFailure,
    RuntimeAnomaly,
    ManualEntry,
    CrossValidated,
    /// Evidence from MLComputePlan per-op placement data.
    /// This is deterministic for a given hardware+OS combination,
    /// so observations from this source carry confidence 0.9.
    ComputePlan,
}

impl std::fmt::Display for EvidenceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvidenceSource::SyntheticRun => write!(f, "SyntheticRun"),
            EvidenceSource::RealModelRun => write!(f, "RealModelRun"),
            EvidenceSource::CompileFailure => write!(f, "CompileFailure"),
            EvidenceSource::LoadFailure => write!(f, "LoadFailure"),
            EvidenceSource::RuntimeAnomaly => write!(f, "RuntimeAnomaly"),
            EvidenceSource::ManualEntry => write!(f, "ManualEntry"),
            EvidenceSource::CrossValidated => write!(f, "CrossValidated"),
            EvidenceSource::ComputePlan => write!(f, "ComputePlan"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeScope {
    pub device_classes: Vec<String>,
    pub os_versions: Vec<String>,
    pub opset_versions: Vec<String>,
}
