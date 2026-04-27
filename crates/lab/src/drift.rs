//! Drift Detection
//!
//! Computes numerical drift metrics between baseline (FP32 reference)
//! and actual (compiled model) outputs. The metrics are honest: they
//! report what was measured, they distinguish available from unavailable
//! results, and they never infer ANE behavior from numerical drift alone.
//!
//! The three mandatory metrics per the task spec:
//! - **max abs**: maximum absolute element-wise error
//! - **mean abs**: mean absolute element-wise error
//! - **RMSE**: root mean squared error
//!
//! Additional metrics computed for diagnostic value:
//! - cosine distance
//! - relative error p99

use serde::{Deserialize, Serialize};

/// Schema version for the drift report format.
pub const DRIFT_REPORT_SCHEMA_VERSION: &str = "1.0.0";

/// Drift detection engine.
///
/// Compares two tensors element-by-element and produces a structured
/// drift report. The detector does NOT infer ANE behavior from drift;
/// it only reports the numerical differences.
pub struct DriftDetector {
    /// Threshold for cosine distance above which drift is flagged.
    pub cosine_threshold: f64,
    /// Threshold for max absolute error above which drift is flagged.
    pub max_error_threshold: f64,
}

impl DriftDetector {
    /// Create a new drift detector with default thresholds.
    pub fn new() -> Self {
        Self {
            cosine_threshold: 1e-3,
            max_error_threshold: 1e-2,
        }
    }

    /// Create a drift detector with custom thresholds.
    pub fn with_thresholds(cosine_threshold: f64, max_error_threshold: f64) -> Self {
        Self {
            cosine_threshold,
            max_error_threshold,
        }
    }

    /// Compute drift metrics between a baseline (FP32 reference) and
    /// an actual output tensor.
    ///
    /// Both slices must have the same length. If they differ, an error
    /// report is returned indicating the mismatch.
    pub fn detect(&self, baseline: &[f32], actual: &[f32]) -> DriftReport {
        if baseline.len() != actual.len() {
            return DriftReport {
                drift_report_schema_version: DRIFT_REPORT_SCHEMA_VERSION.to_string(),
                has_drift: true,
                computation_status: DriftComputationStatus::LengthMismatch {
                    baseline_len: baseline.len(),
                    actual_len: actual.len(),
                },
                max_absolute_error: f64::NAN,
                mean_absolute_error: f64::NAN,
                rmse: f64::NAN,
                cosine_distance: f64::NAN,
                relative_error_p99: f64::NAN,
                element_count: baseline.len().min(actual.len()),
                scope_note: "Baseline and actual tensor lengths do not match — \
                    drift metrics cannot be computed reliably."
                    .to_string(),
            };
        }

        if baseline.is_empty() {
            return DriftReport {
                drift_report_schema_version: DRIFT_REPORT_SCHEMA_VERSION.to_string(),
                has_drift: false,
                computation_status: DriftComputationStatus::EmptyInput,
                max_absolute_error: 0.0,
                mean_absolute_error: 0.0,
                rmse: 0.0,
                cosine_distance: 0.0,
                relative_error_p99: 0.0,
                element_count: 0,
                scope_note: "Empty input tensors — no drift to compute.".to_string(),
            };
        }

        let n = baseline.len();

        // Compute element-wise metrics
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut rel_errors: Vec<f64> = Vec::with_capacity(n);

        // For cosine distance
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for i in 0..n {
            let a = baseline[i] as f64;
            let b = actual[i] as f64;
            let diff = (a - b).abs();

            max_abs = max_abs.max(diff);
            sum_abs += diff;
            sum_sq += diff * diff;

            // Cosine distance components
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;

            // Relative error (avoid division by zero)
            let denom = a.abs().max(b.abs());
            if denom > 1e-10 {
                rel_errors.push(diff / denom);
            }
        }

        let mean_abs = sum_abs / n as f64;
        let rmse = (sum_sq / n as f64).sqrt();

        // Cosine distance = 1 - cosine_similarity
        let cosine_distance = if norm_a > 0.0 && norm_b > 0.0 {
            let cosine_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
            1.0 - cosine_sim
        } else {
            // If one or both vectors are zero, cosine distance is undefined.
            // Report 0.0 if both are zero (no drift), 1.0 if only one is zero
            // (maximal drift).
            if norm_a == 0.0 && norm_b == 0.0 { 0.0 } else { 1.0 }
        };

        // P99 of relative errors
        let relative_error_p99 = if rel_errors.is_empty() {
            0.0
        } else {
            rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((rel_errors.len() as f64) * 0.99).ceil() as usize;
            let idx = idx.saturating_sub(1).min(rel_errors.len() - 1);
            rel_errors[idx]
        };

        let has_drift = max_abs > self.max_error_threshold
            || cosine_distance > self.cosine_threshold;

        DriftReport {
            drift_report_schema_version: DRIFT_REPORT_SCHEMA_VERSION.to_string(),
            has_drift,
            computation_status: DriftComputationStatus::Computed,
            max_absolute_error: max_abs,
            mean_absolute_error: mean_abs,
            rmse,
            cosine_distance,
            relative_error_p99,
            element_count: n,
            scope_note: "Host-side numerical comparison of FP32 baseline vs actual output. \
                Drift indicates numerical difference only — it does not indicate ANE fallback \
                or compute unit assignment."
                .to_string(),
        }
    }

    /// Produce a drift report indicating that drift computation is unavailable.
    ///
    /// This is the honest result when actual model outputs cannot be obtained
    /// (e.g., no Apple hardware for predict(), or compilation failed).
    pub fn unavailable(reason: &str) -> DriftReport {
        DriftReport {
            drift_report_schema_version: DRIFT_REPORT_SCHEMA_VERSION.to_string(),
            has_drift: false,
            computation_status: DriftComputationStatus::Unavailable {
                reason: reason.to_string(),
            },
            max_absolute_error: f64::NAN,
            mean_absolute_error: f64::NAN,
            rmse: f64::NAN,
            cosine_distance: f64::NAN,
            relative_error_p99: f64::NAN,
            element_count: 0,
            scope_note: format!(
                "Drift computation unavailable: {}. \
                 No numerical comparison was performed.",
                reason
            ),
        }
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of drift computation — distinguishes computed results from
/// unavailable or error states. This prevents NaN metrics from being
/// misread as actual measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftComputationStatus {
    /// Drift metrics were computed successfully.
    Computed,
    /// Drift computation is unavailable (e.g., no device output).
    Unavailable { reason: String },
    /// Baseline and actual tensor lengths do not match.
    LengthMismatch { baseline_len: usize, actual_len: usize },
    /// Empty input tensors — nothing to compare.
    EmptyInput,
}

/// Report from drift detection.
///
/// This is a stable artifact format. The `computation_status` field
/// MUST be checked before interpreting any numeric fields: if it is
/// not `Computed`, the numeric fields are not meaningful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Schema version of this drift report format.
    pub drift_report_schema_version: String,
    /// Whether drift was detected above the configured thresholds.
    pub has_drift: bool,
    /// Status of the computation — check this before reading metrics.
    pub computation_status: DriftComputationStatus,
    /// Maximum absolute element-wise error: max(|baseline[i] - actual[i]|).
    pub max_absolute_error: f64,
    /// Mean absolute error: mean(|baseline[i] - actual[i]|).
    pub mean_absolute_error: f64,
    /// Root mean squared error: sqrt(mean((baseline[i] - actual[i])^2)).
    pub rmse: f64,
    /// Cosine distance: 1 - cos_sim(baseline, actual).
    pub cosine_distance: f64,
    /// 99th percentile of relative error (|a-b|/max(|a|,|b|)).
    pub relative_error_p99: f64,
    /// Number of elements compared.
    pub element_count: usize,
    /// What this drift report actually measures and what it does NOT imply.
    pub scope_note: String,
}

impl DriftReport {
    /// Serialize this report to pretty-printed JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write this report to a JSON file.
    pub fn write_to_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Whether this report contains meaningful numeric metrics.
    pub fn is_computed(&self) -> bool {
        matches!(self.computation_status, DriftComputationStatus::Computed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_tensors_no_drift() {
        let detector = DriftDetector::new();
        let baseline = vec![1.0f32, 2.0, 3.0, 4.0];
        let report = detector.detect(&baseline, &baseline);
        assert!(!report.has_drift);
        assert!(report.is_computed());
        assert_eq!(report.max_absolute_error, 0.0);
        assert_eq!(report.mean_absolute_error, 0.0);
        assert_eq!(report.rmse, 0.0);
        assert!((report.cosine_distance).abs() < 1e-10);
    }

    #[test]
    fn test_small_drift_below_threshold() {
        let detector = DriftDetector::new();
        let baseline = vec![1.0f32, 2.0, 3.0, 4.0];
        let actual = vec![1.001f32, 2.001, 3.001, 4.001];
        let report = detector.detect(&baseline, &actual);
        assert!(!report.has_drift, "Small drift should be below default threshold");
        assert!(report.is_computed());
        assert!(report.max_absolute_error > 0.0);
        assert!(report.max_absolute_error < detector.max_error_threshold);
    }

    #[test]
    fn test_large_drift_above_threshold() {
        let detector = DriftDetector::new();
        let baseline = vec![1.0f32, 2.0, 3.0, 4.0];
        let actual = vec![1.5f32, 2.5, 3.5, 4.5];
        let report = detector.detect(&baseline, &actual);
        assert!(report.has_drift, "Large drift should exceed threshold");
        assert!(report.max_absolute_error > detector.max_error_threshold);
    }

    #[test]
    fn test_length_mismatch() {
        let detector = DriftDetector::new();
        let baseline = vec![1.0f32, 2.0, 3.0];
        let actual = vec![1.0f32, 2.0];
        let report = detector.detect(&baseline, &actual);
        assert!(report.has_drift);
        assert!(matches!(report.computation_status, DriftComputationStatus::LengthMismatch { .. }));
    }

    #[test]
    fn test_empty_input() {
        let detector = DriftDetector::new();
        let report = detector.detect(&[], &[]);
        assert!(!report.has_drift);
        assert!(matches!(report.computation_status, DriftComputationStatus::EmptyInput));
    }

    #[test]
    fn test_unavailable_report() {
        let report = DriftDetector::unavailable("no Apple hardware");
        assert!(!report.has_drift);
        assert!(!report.is_computed());
        assert!(matches!(report.computation_status, DriftComputationStatus::Unavailable { .. }));
        assert!(report.max_absolute_error.is_nan());
    }

    #[test]
    fn test_rmse_calculation() {
        let detector = DriftDetector::new();
        let baseline = vec![0.0f32, 0.0, 0.0];
        let actual = vec![3.0f32, 4.0, 0.0];
        let report = detector.detect(&baseline, &actual);
        // RMSE = sqrt((9 + 16 + 0) / 3) = sqrt(25/3) ≈ 2.887
        let expected_rmse = (25.0f64 / 3.0).sqrt();
        assert!((report.rmse - expected_rmse).abs() < 1e-10);
    }

    #[test]
    fn test_drift_report_serialization() {
        let detector = DriftDetector::new();
        let baseline = vec![1.0f32, 2.0, 3.0];
        let actual = vec![1.1f32, 2.1, 3.1];
        let report = detector.detect(&baseline, &actual);
        let json = report.to_json().unwrap();
        let parsed: DriftReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.drift_report_schema_version, "1.0.0");
        assert_eq!(parsed.element_count, 3);
    }
}
