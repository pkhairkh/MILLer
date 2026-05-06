//! Confidence decay and pruning for knowledge entries.
//!
//! Implements SPEC §553-554 (linear confidence decay: 1% per 30 days)
//! and SPEC §531 (knowledge pruning: remove entries below threshold).

/// Apply time-based linear confidence decay.
///
/// Implements SPEC §553-554: "1% per 30 days" linear decay.
///
/// ```text
/// confidence_after_decay = confidence * (1.0 - 0.01 * (days_elapsed / 30.0))
/// ```
///
/// Minimum confidence is 0.0 (can't go negative).
pub fn apply_time_decay(current_confidence: f64, days_elapsed: f64) -> f64 {
    let decay_rate = 0.01; // 1% per 30 days
    let periods = days_elapsed / 30.0;
    let decayed = current_confidence * (1.0 - decay_rate * periods);
    decayed.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_time_decay_no_time() {
        // 0 days elapsed → no change
        let result = apply_time_decay(0.8, 0.0);
        assert!((result - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_time_decay_30_days() {
        // 30 days → 1% decay: 0.8 * 0.99 = 0.792
        let result = apply_time_decay(0.8, 30.0);
        assert!((result - 0.792).abs() < 1e-10);
    }

    #[test]
    fn test_apply_time_decay_300_days() {
        // 300 days → 10% decay: 0.8 * 0.90 = 0.72
        let result = apply_time_decay(0.8, 300.0);
        assert!((result - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_apply_time_decay_clamp_at_zero() {
        // Very large days → 0.0 (not negative)
        let result = apply_time_decay(0.5, 100000.0);
        assert_eq!(result, 0.0);
    }
}
