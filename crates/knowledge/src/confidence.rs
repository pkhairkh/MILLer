//! Confidence Utilities
//!
//! Helper functions for confidence score manipulation.
//!
//! **Note**: The canonical initial confidence computation is
//! [`update::initial_confidence`], which is used by the update
//! pipeline. That function's base values are deliberately
//! conservative (single observations never start above 0.5
//! for synthetic/real-model runs). The former `compute_confidence`
//! function had contradictory, more optimistic base values and
//! has been removed.

/// Decay confidence over time (simulated temporal decay).
///
/// Given a current confidence value and a half-life in days,
/// returns the confidence after `elapsed_days` have passed.
/// Uses exponential decay: `c * 0.5^(elapsed / halflife)`.
///
/// **Note**: This function is currently only used in tests. Production code
/// should use [`update::initial_confidence`] for computing initial confidence
/// values. If temporal decay is needed in production, integrate this into the
/// update pipeline rather than calling it directly.
pub fn decay_confidence(current: f32, halflife_days: f32, elapsed_days: f32) -> f32 {
    current * 0.5f32.powf(elapsed_days / halflife_days)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_confidence_half() {
        // After one half-life, confidence should be halved
        let decayed = decay_confidence(0.8, 30.0, 30.0);
        assert!((decayed - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_decay_confidence_zero_time() {
        // Zero elapsed time → no decay
        let decayed = decay_confidence(0.8, 30.0, 0.0);
        assert!((decayed - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_decay_confidence_long_time() {
        // After many half-lives, confidence should be near zero
        let decayed = decay_confidence(0.8, 30.0, 300.0);
        assert!(decayed < 0.01);
    }
}
