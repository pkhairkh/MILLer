//! Fallback Detection
//!
//! Detects when Core ML silently falls back from ANE to CPU/GPU,
//! which invalidates performance assumptions.
//!
//! Design principle: this module is deliberately weak and honest.
//! It does NOT make hard placement claims without evidence. The
//! suspicion levels represent what can honestly be concluded from
//! available data, which is often very little.

use crate::device_meta::DeviceMetadata;
use crate::harness::{FallbackSuspicionLevel, FallbackSuspicionResult, SuspicionEvidence};

/// Fallback detection engine.
///
/// The detector uses multiple weak signals to build a suspicion assessment.
/// No single signal is conclusive. The overall assessment is always conservative.
pub struct FallbackDetector {
    /// Latency threshold ratio above which we start getting suspicious.
    /// If observed latency exceeds expected ANE latency by this factor,
    /// we add a low-confidence suspicion signal.
    pub latency_threshold_ratio: f64,
}

impl Default for FallbackDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackDetector {
    /// Create a new fallback detector with default thresholds.
    pub fn new() -> Self {
        Self { latency_threshold_ratio: 3.0 }
    }

    /// Create a new fallback detector with a custom latency threshold ratio.
    pub fn with_threshold_ratio(ratio: f64) -> Self {
        Self { latency_threshold_ratio: ratio }
    }

    /// Detect fallback suspicion from timing data.
    ///
    /// This compares observed latency against an expected ANE latency estimate.
    /// The result is always a weak signal — latency alone cannot prove fallback.
    ///
    /// If `expected_ane_latency_ms` is None, no timing-based suspicion is possible.
    pub fn detect_from_timing(
        &self,
        observed_median_ms: f64,
        expected_ane_latency_ms: Option<f64>,
        device_meta: &DeviceMetadata,
    ) -> FallbackSuspicionResult {
        let mut evidence = Vec::new();

        // Evidence from device metadata
        if !device_meta.is_device_backed() {
            evidence.push(SuspicionEvidence {
                kind: "environment".to_string(),
                description:
                    "Run was not performed on Apple hardware; fallback assessment is not available"
                        .to_string(),
                strength: 0.0,
            });
            return FallbackSuspicionResult {
                suspicion_level: FallbackSuspicionLevel::Unavailable,
                explanation:
                    "Fallback assessment requires device-backed execution on Apple hardware"
                        .to_string(),
                evidence,
            };
        }

        // Evidence from compute plan availability
        if !device_meta.compute_plan_available {
            evidence.push(SuspicionEvidence {
                kind: "compute_plan_unavailable".to_string(),
                description:
                    "Compute plan inspection not available; cannot verify compute unit assignment"
                        .to_string(),
                strength: 0.0,
            });
        }

        // Evidence from timing comparison (weak signal only)
        if let Some(expected_ms) = expected_ane_latency_ms {
            let ratio = observed_median_ms / expected_ms;
            if ratio > self.latency_threshold_ratio {
                evidence.push(SuspicionEvidence {
                    kind: "latency_anomaly".to_string(),
                    description: format!(
                        "Observed latency ({:.2}ms) is {:.1}x expected ANE latency ({:.2}ms) — possible CPU fallback",
                        observed_median_ms, ratio, expected_ms
                    ),
                    strength: 0.4, // Weak: many other explanations for slow execution
                });
            } else {
                evidence.push(SuspicionEvidence {
                    kind: "latency_normal".to_string(),
                    description: format!(
                        "Observed latency ({:.2}ms) is within {:.1}x of expected ANE latency ({:.2}ms)",
                        observed_median_ms, ratio, expected_ms
                    ),
                    strength: 0.3, // Also weak: normal latency doesn't prove ANE execution
                });
            }
        } else {
            evidence.push(SuspicionEvidence {
                kind: "no_baseline".to_string(),
                description: "No expected ANE latency baseline available for comparison"
                    .to_string(),
                strength: 0.0,
            });
        }

        // Determine overall suspicion level
        let suspicion_level = self.assess_overall_level(&evidence);

        let explanation = match &suspicion_level {
            FallbackSuspicionLevel::Unavailable => {
                "Insufficient evidence to assess fallback status".to_string()
            }
            FallbackSuspicionLevel::LowConfidenceSuspicion => {
                "Some weak signals suggest possible compute unit fallback, but this is not conclusive".to_string()
            }
            FallbackSuspicionLevel::NoConclusion => {
                "No strong evidence of fallback was found, but absence of evidence is not evidence of absence".to_string()
            }
        };

        FallbackSuspicionResult { suspicion_level, explanation, evidence }
    }

    /// Assess the overall suspicion level from accumulated evidence.
    fn assess_overall_level(&self, evidence: &[SuspicionEvidence]) -> FallbackSuspicionLevel {
        // If any evidence indicates we can't make an assessment, return Unavailable
        let has_unavailable =
            evidence.iter().any(|e| e.kind == "environment" || e.kind == "no_baseline");

        if has_unavailable {
            // We have some evidence but not enough for any conclusion
            // Check if we have at least timing data
            let has_timing =
                evidence.iter().any(|e| e.kind == "latency_anomaly" || e.kind == "latency_normal");
            if !has_timing {
                return FallbackSuspicionLevel::Unavailable;
            }
        }

        // Check if there's a latency anomaly signal
        let max_suspicion_strength = evidence
            .iter()
            .filter(|e| e.kind == "latency_anomaly")
            .map(|e| e.strength)
            .fold(0.0f64, f64::max);

        if max_suspicion_strength > 0.3 {
            FallbackSuspicionLevel::LowConfidenceSuspicion
        } else {
            FallbackSuspicionLevel::NoConclusion
        }
    }
}

/// Evidence of fallback from Core ML diagnostic logs.
///
/// This type is reserved for future use when log parsing is implemented.
/// On Apple hardware, Core ML diagnostic logs can reveal per-op compute
/// unit assignment, which would provide strong (but still not certain)
/// evidence of fallback.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FallbackLogEvidence {
    /// Operation name in the Core ML model.
    pub op_name: String,
    /// Compute unit that was expected to execute this op.
    pub expected_compute_unit: String,
    /// Compute unit that actually executed this op (from log).
    pub actual_compute_unit: String,
    /// Source of this evidence (e.g., "coreml_diagnostics_log").
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_meta::{DeviceMetadata, MetadataSource};

    fn make_device_backed_meta() -> DeviceMetadata {
        DeviceMetadata {
            source: MetadataSource::DeviceBacked,
            host_os: "macOS arm64".to_string(),
            device_class: Some("Apple M2".to_string()),
            chip_name: Some("t6020".to_string()),
            os_version: "macOS 14.3.1".to_string(),
            core_ml_version: Some("7.2".to_string()),
            total_memory_gb: Some(16.0),
            ane_core_count: Some(16),
            coreml_runtime_available: true,
            compute_plan_available: true,
        }
    }

    #[test]
    fn test_fallback_detector_default_threshold() {
        let detector = FallbackDetector::new();
        assert!((detector.latency_threshold_ratio - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fallback_detector_custom_threshold() {
        let detector = FallbackDetector::with_threshold_ratio(5.0);
        assert!((detector.latency_threshold_ratio - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_from_timing_host_only_returns_unavailable() {
        let detector = FallbackDetector::new();
        let meta = DeviceMetadata::host_only();
        let result = detector.detect_from_timing(10.0, Some(1.0), &meta);
        assert_eq!(result.suspicion_level, FallbackSuspicionLevel::Unavailable);
    }

    #[test]
    fn test_detect_from_timing_no_baseline_returns_unavailable() {
        let detector = FallbackDetector::new();
        let meta = DeviceMetadata::host_only();
        let result = detector.detect_from_timing(10.0, None, &meta);
        assert_eq!(result.suspicion_level, FallbackSuspicionLevel::Unavailable);
    }

    #[test]
    fn test_detect_from_timing_latency_anomaly() {
        let detector = FallbackDetector::new();
        let meta = make_device_backed_meta();
        // observed=10ms, expected=1ms → ratio=10x > threshold(3.0)
        let result = detector.detect_from_timing(10.0, Some(1.0), &meta);
        assert_eq!(result.suspicion_level, FallbackSuspicionLevel::LowConfidenceSuspicion);
    }

    #[test]
    fn test_detect_from_timing_latency_normal() {
        let detector = FallbackDetector::new();
        let meta = make_device_backed_meta();
        // observed=1.5ms, expected=1.0ms → ratio=1.5x < threshold(3.0)
        let result = detector.detect_from_timing(1.5, Some(1.0), &meta);
        assert_eq!(result.suspicion_level, FallbackSuspicionLevel::NoConclusion);
    }

    #[test]
    fn test_detect_from_timing_no_compute_plan_evidence() {
        let detector = FallbackDetector::new();
        let mut meta = make_device_backed_meta();
        meta.compute_plan_available = false;
        let result = detector.detect_from_timing(1.0, Some(1.0), &meta);
        let kinds: Vec<&str> = result.evidence.iter().map(|e| e.kind.as_str()).collect();
        assert!(
            kinds.contains(&"compute_plan_unavailable"),
            "Expected compute_plan_unavailable evidence, got: {:?}",
            kinds
        );
    }

    #[test]
    fn test_detect_from_timing_evidence_kinds() {
        let detector = FallbackDetector::new();
        let meta = make_device_backed_meta();
        let result = detector.detect_from_timing(10.0, Some(1.0), &meta);
        let kinds: Vec<&str> = result.evidence.iter().map(|e| e.kind.as_str()).collect();
        // Should have latency_anomaly since ratio > threshold
        assert!(
            kinds.contains(&"latency_anomaly"),
            "Expected latency_anomaly evidence, got: {:?}",
            kinds
        );
    }

    #[test]
    fn test_detect_from_timing_suspicion_explanation_not_empty() {
        let detector = FallbackDetector::new();
        let meta = make_device_backed_meta();
        let result = detector.detect_from_timing(10.0, Some(1.0), &meta);
        assert!(!result.explanation.is_empty(), "Explanation should not be empty");
    }

    #[test]
    fn test_fallback_log_evidence_serialization() {
        let evidence = FallbackLogEvidence {
            op_name: "linear_1".to_string(),
            expected_compute_unit: "NeuralEngine".to_string(),
            actual_compute_unit: "CPU".to_string(),
            source: "coreml_diagnostics_log".to_string(),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let back: FallbackLogEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back.op_name, "linear_1");
        assert_eq!(back.actual_compute_unit, "CPU");
        assert_eq!(back.source, "coreml_diagnostics_log");
    }

    #[test]
    fn test_assess_overall_level_latency_anomaly() {
        let detector = FallbackDetector::new();
        let evidence = vec![SuspicionEvidence {
            kind: "latency_anomaly".to_string(),
            description: "slow".to_string(),
            strength: 0.4,
        }];
        let level = detector.assess_overall_level(&evidence);
        assert_eq!(level, FallbackSuspicionLevel::LowConfidenceSuspicion);
    }

    #[test]
    fn test_assess_overall_level_no_anomaly() {
        let detector = FallbackDetector::new();
        let evidence = vec![SuspicionEvidence {
            kind: "latency_normal".to_string(),
            description: "normal".to_string(),
            strength: 0.3,
        }];
        let level = detector.assess_overall_level(&evidence);
        assert_eq!(level, FallbackSuspicionLevel::NoConclusion);
    }
}
