//! Shared test utilities for compilation passes.
//!
//! Provides a common `MockKnowledge` implementation that can be configured
//! for any test scenario, replacing the duplicated mock implementations
//! that were previously scattered across individual pass test modules.
//!
//! # Usage
//!
//! ```ignore
//! use crate::test_utils::MockKnowledge;
//!
//! let knowledge = MockKnowledge::new()
//!     .with_compute_plan_ane("mb.matmul")
//!     .with_risk(0.3, 0.1);
//! ```

use crate::knowledge_query::{
    ComputePlanPlacementInfo, LegalityInfo, PassKnowledgeQuery, PrecisionHazardInfo, RiskInfo,
};
use ane_ir::kir::KnowledgeScope;

/// A mock knowledge query that can return configurable results for any
/// query type. This replaces the duplicated `MockKnowledge`, `MockHighFallbackRiskKnowledge`,
/// `MockLowFallbackRiskKnowledge`, `MockBorderlineRiskKnowledge`, `MockLinearLegalKnowledge`,
/// `MockLinearIllegalKnowledge`, `MockPrecisionHazardKnowledge`, `MockLowConfidenceHazardKnowledge`,
/// and `MockSafeKnowledge` that were previously defined in each pass's test module.
#[derive(Debug, Clone, Default)]
pub struct MockKnowledge {
    legality: Option<LegalityInfo>,
    risk: Option<RiskInfo>,
    precision_hazard: Option<PrecisionHazardInfo>,
    compute_plan_placement: Option<ComputePlanPlacementInfo>,
}

impl MockKnowledge {
    /// Create a new mock with all queries returning `None`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the mock to return a legality result.
    pub fn with_legality(mut self, ane_legal: bool, confidence: f32) -> Self {
        self.legality = Some(LegalityInfo {
            ane_legal,
            confidence,
            evidence_count: 1,
            source_id: Some("mock_legality".to_string()),
        });
        self
    }

    /// Configure the mock to return a risk result.
    pub fn with_risk(mut self, fallback_risk: f32, drift_risk: f32) -> Self {
        self.risk = Some(RiskInfo {
            fallback_risk,
            drift_risk,
            confidence: 0.5,
            evidence_count: 1,
            source_id: Some("mock_risk".to_string()),
        });
        self
    }

    /// Configure the mock to return a risk result with custom confidence.
    pub fn with_risk_and_confidence(
        mut self,
        fallback_risk: f32,
        drift_risk: f32,
        confidence: f32,
    ) -> Self {
        self.risk = Some(RiskInfo {
            fallback_risk,
            drift_risk,
            confidence,
            evidence_count: 1,
            source_id: Some("mock_risk".to_string()),
        });
        self
    }

    /// Configure the mock to return a precision hazard result.
    pub fn with_precision_hazard(
        mut self,
        op_pattern: &str,
        hazardous_dtype: &str,
        recommended_dtype: &str,
    ) -> Self {
        self.precision_hazard = Some(PrecisionHazardInfo {
            op_pattern: op_pattern.to_string(),
            hazardous_dtype: hazardous_dtype.to_string(),
            recommended_dtype: recommended_dtype.to_string(),
            confidence: 0.8,
            evidence_count: 1,
            source_id: Some("mock_hazard".to_string()),
            description: Some("mock precision hazard".to_string()),
        });
        self
    }

    /// Configure the mock to return a compute plan placement with ANE placement.
    pub fn with_compute_plan_ane(mut self, op_pattern: &str) -> Self {
        self.compute_plan_placement = Some(ComputePlanPlacementInfo {
            op_pattern: op_pattern.to_string(),
            ane_placed: true,
            preferred_device: "NeuralEngine".to_string(),
            confidence: 0.9,
            evidence_count: 1,
            source_id: Some("mock_cp".to_string()),
        });
        self
    }

    /// Configure the mock to return a compute plan placement without ANE placement.
    pub fn with_compute_plan_not_ane(mut self, op_pattern: &str) -> Self {
        self.compute_plan_placement = Some(ComputePlanPlacementInfo {
            op_pattern: op_pattern.to_string(),
            ane_placed: false,
            preferred_device: "CPU".to_string(),
            confidence: 0.9,
            evidence_count: 1,
            source_id: Some("mock_cp".to_string()),
        });
        self
    }

    /// Configure the mock to return a compute plan placement on GPU.
    pub fn with_compute_plan_gpu(mut self, op_pattern: &str) -> Self {
        self.compute_plan_placement = Some(ComputePlanPlacementInfo {
            op_pattern: op_pattern.to_string(),
            ane_placed: false,
            preferred_device: "GPU".to_string(),
            confidence: 0.9,
            evidence_count: 1,
            source_id: Some("mock_cp".to_string()),
        });
        self
    }
}

impl PassKnowledgeQuery for MockKnowledge {
    fn query_legality(
        &self,
        _op_pattern: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<LegalityInfo> {
        self.legality.clone()
    }

    fn query_risk(&self, _op_pattern: &str, _scope: Option<&KnowledgeScope>) -> Option<RiskInfo> {
        self.risk.clone()
    }

    fn query_precision_hazard(
        &self,
        _op_pattern: &str,
        _current_dtype: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<PrecisionHazardInfo> {
        self.precision_hazard.clone()
    }

    fn query_compute_plan_placement(
        &self,
        _op_pattern: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<ComputePlanPlacementInfo> {
        self.compute_plan_placement.clone()
    }
}
