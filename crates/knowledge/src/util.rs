//! Shared Utilities
//!
//! Common functions extracted from multiple modules to avoid duplication.
//! Includes ID sanitization, scope overlap detection, and typed payload
//! accessor helpers for JSON payload fields.

use ane_ir::kir::KnowledgeScope;
use std::collections::HashMap;

/// Sanitize an entry ID for use as a filename.
///
/// Replaces any character that is not alphanumeric, hyphen, or underscore
/// with an underscore. This is safe for all common filesystems.
pub fn sanitize_id(id: &str) -> String {
    id.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_")
}

/// Check if two knowledge scopes overlap (share at least one device class,
/// OS version, and opset version).
///
/// An "unknown" value in device_classes or os_versions is treated as a
/// wildcard that overlaps with everything (conservative: assume overlap
/// unless we can prove otherwise).
pub fn scopes_overlap(a: &KnowledgeScope, b: &KnowledgeScope) -> bool {
    let devices_overlap = a.device_classes.iter().any(|d| b.device_classes.contains(d))
        || a.device_classes.contains(&"unknown".to_string())
        || b.device_classes.contains(&"unknown".to_string());
    let os_overlap = a.os_versions.iter().any(|v| b.os_versions.contains(v))
        || a.os_versions.contains(&"unknown".to_string())
        || b.os_versions.contains(&"unknown".to_string());
    let opset_overlap = a.opset_versions.iter().any(|v| b.opset_versions.contains(v));

    devices_overlap && os_overlap && opset_overlap
}

// ---------------------------------------------------------------------------
// Typed payload accessor helpers
// ---------------------------------------------------------------------------
//
// The knowledge store uses `HashMap<String, serde_json::Value>` for payload
// data.  Raw `.get("ane_legal").and_then(|v| v.as_bool())` is error-prone
// and provides no compile-time guarantee that keys exist.  These helpers
// centralise every payload key behind a typed function so that typos and
// type-mismatches are caught in one place.

/// Retrieve the `ane_legal` boolean from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a boolean.
pub fn payload_ane_legal(payload: &HashMap<String, serde_json::Value>) -> Option<bool> {
    payload.get("ane_legal").and_then(|v| v.as_bool())
}

/// Retrieve the `op_pattern` string from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a string.
pub fn payload_op_pattern(payload: &HashMap<String, serde_json::Value>) -> Option<&str> {
    payload.get("op_pattern").and_then(|v| v.as_str())
}

/// Retrieve the `quality_impact` string from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a string.
pub fn payload_quality_impact(payload: &HashMap<String, serde_json::Value>) -> Option<&str> {
    payload.get("quality_impact").and_then(|v| v.as_str())
}

/// Retrieve the `ane_placed` boolean from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a boolean.
pub fn payload_ane_placed(payload: &HashMap<String, serde_json::Value>) -> Option<bool> {
    payload.get("ane_placed").and_then(|v| v.as_bool())
}

/// Retrieve the `survival_rate` float from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a number.
/// T-112: Used for SurvivalMatrixEntry field-level comparison in claims_agree.
pub fn payload_survival_rate(payload: &HashMap<String, serde_json::Value>) -> Option<f64> {
    payload.get("survival_rate").and_then(|v| v.as_f64())
}

/// Retrieve the `fallback_engine` string from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not a string.
/// T-112: Used for FallbackSignature field-level comparison in claims_agree.
pub fn payload_fallback_engine(payload: &HashMap<String, serde_json::Value>) -> Option<&str> {
    payload.get("fallback_engine").and_then(|v| v.as_str())
}

/// Retrieve the `num_partitions` integer from a knowledge unit payload.
///
/// Returns `None` if the key is absent or the value is not an integer.
/// T-112: Used for ShardTemplateKnowledge field-level comparison in claims_agree.
pub fn payload_num_partitions(payload: &HashMap<String, serde_json::Value>) -> Option<u64> {
    payload.get("num_partitions").and_then(|v| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("simple_id-123"), "simple_id-123");
        assert_eq!(sanitize_id("has spaces"), "has_spaces");
        assert_eq!(sanitize_id("dot.separated"), "dot_separated");
        assert_eq!(sanitize_id("slash/separated"), "slash_separated");
    }

    #[test]
    fn test_scopes_overlap_basic() {
        let scope_a = KnowledgeScope {
            device_classes: vec!["M2".to_string()],
            os_versions: vec!["macOS_15".to_string()],
            opset_versions: vec!["iOS18".to_string()],
        };
        let scope_b = KnowledgeScope {
            device_classes: vec!["M2".to_string(), "M3".to_string()],
            os_versions: vec!["macOS_15".to_string()],
            opset_versions: vec!["iOS18".to_string()],
        };
        let scope_c = KnowledgeScope {
            device_classes: vec!["M4".to_string()],
            os_versions: vec!["macOS_15".to_string()],
            opset_versions: vec!["iOS18".to_string()],
        };

        assert!(scopes_overlap(&scope_a, &scope_b)); // M2 overlap
        assert!(!scopes_overlap(&scope_a, &scope_c)); // No device overlap
    }

    #[test]
    fn test_scopes_overlap_unknown() {
        let scope_with_unknown = KnowledgeScope {
            device_classes: vec!["unknown".to_string()],
            os_versions: vec!["unknown".to_string()],
            opset_versions: vec!["iOS18".to_string()],
        };
        let scope_specific = KnowledgeScope {
            device_classes: vec!["M2".to_string()],
            os_versions: vec!["macOS_15".to_string()],
            opset_versions: vec!["iOS18".to_string()],
        };

        // "unknown" scope should overlap with everything (conservative)
        assert!(scopes_overlap(&scope_with_unknown, &scope_specific));
    }

    #[test]
    fn test_payload_accessors() {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(true));
        payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload.insert("quality_impact".to_string(), serde_json::json!("severe"));
        payload.insert("ane_placed".to_string(), serde_json::json!(false));

        assert_eq!(payload_ane_legal(&payload), Some(true));
        assert_eq!(payload_op_pattern(&payload), Some("mb.matmul"));
        assert_eq!(payload_quality_impact(&payload), Some("severe"));
        assert_eq!(payload_ane_placed(&payload), Some(false));

        // Missing keys
        let empty: HashMap<String, serde_json::Value> = HashMap::new();
        assert_eq!(payload_ane_legal(&empty), None);
        assert_eq!(payload_op_pattern(&empty), None);
        assert_eq!(payload_quality_impact(&empty), None);
        assert_eq!(payload_ane_placed(&empty), None);

        // Wrong types
        let mut wrong = HashMap::new();
        wrong.insert("ane_legal".to_string(), serde_json::json!("not a bool"));
        assert_eq!(payload_ane_legal(&wrong), None);
    }
}
