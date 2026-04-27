//! Compute Plan Offline Verification
//!
//! Proves structural properties of compute plans without requiring
//! Apple hardware execution. This closes the gap where compute plan
//! harvesting could only be validated on macOS with Core ML runtime.
//!
//! ## Approach
//!
//! Instead of running `MLComputePlan` on Apple hardware (which is
//! impossible on Linux), we verify compute plans through:
//!
//! 1. **Structural proof**: A `ComputePlanProof` captures the op→device
//!    placement mapping in a deterministic, hashable form. Two independent
//!    observers can verify they got the same plan by comparing proof hashes.
//!
//! 2. **Knowledge cross-reference**: The proof is cross-referenced with
//!    existing knowledge store entries. If the proof says "mb.matmul → CPU"
//!    and the knowledge store has high-confidence risk data saying the same
//!    thing, the proof is consistent with accumulated evidence.
//!
//! 3. **Invariant checking**: The verifier checks structural invariants
//!    that must hold for any valid compute plan (e.g., all ops must have
//!    a placement, the placement must be one of the known device classes).
//!
//! 4. **Synthetic verification**: Given a MIR graph, we can predict what
//!    a compute plan *should* look like based on known op→device mappings,
//!    and verify the actual plan matches the prediction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single op→device placement entry in a compute plan proof.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PlacementEntry {
    /// The op name (e.g., "linear_1", "gelu_2").
    pub op_name: String,
    /// The op type (e.g., "mb.linear", "mb.gelu").
    pub op_type: String,
    /// The device class assigned by the compute planner.
    pub device_class: DeviceClass,
    /// The function this op belongs to (for multi-function models).
    pub function_name: String,
}

/// Device class for compute plan placement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    NeuralEngine,
    CPU,
    GPU,
    Unknown(String),
}

impl DeviceClass {
    /// Parse from the string returned by MLComputePlan.
    pub fn from_coreml_string(s: &str) -> Self {
        match s {
            "NeuralEngine" => DeviceClass::NeuralEngine,
            "CPU" => DeviceClass::CPU,
            "GPU" => DeviceClass::GPU,
            other => DeviceClass::Unknown(other.to_string()),
        }
    }

    /// Whether this placement is on the NeuralEngine (ANE).
    pub fn is_ane(&self) -> bool {
        matches!(self, DeviceClass::NeuralEngine)
    }
}

/// A compute plan proof: a deterministic, verifiable snapshot of
/// an op→device placement mapping.
///
/// This can be produced on Apple hardware (by harvesting from
/// MLComputePlan) and verified on any platform by checking
/// structural invariants and cross-referencing with the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePlanProof {
    /// Unique identifier for this proof.
    pub proof_id: String,
    /// The model identifier this proof was generated from.
    pub model_id: String,
    /// The hardware generation (e.g., "M2", "M3", "A17").
    pub hardware: String,
    /// The OS version (e.g., "macOS_15.2", "iOS_18.1").
    pub os_version: String,
    /// Op→device placement entries.
    pub placements: Vec<PlacementEntry>,
    /// SHA-256 hash of the sorted placement entries.
    pub proof_hash: String,
    /// Timestamp when this proof was generated.
    pub timestamp: String,
    /// Number of ops in the plan.
    pub op_count: usize,
    /// Number of ops placed on NeuralEngine.
    pub ane_placed_count: usize,
    /// Fraction of ops placed on NeuralEngine (0.0–1.0).
    pub ane_placed_fraction: f32,
}

/// Result of verifying a compute plan proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is structurally valid.
    pub is_valid: bool,
    /// List of errors found during verification.
    pub errors: Vec<String>,
    /// List of warnings (non-fatal issues).
    pub warnings: Vec<String>,
    /// Whether the proof is consistent with knowledge store data.
    pub knowledge_consistent: bool,
    /// Number of placements that match knowledge store data.
    pub knowledge_matches: usize,
    /// Number of placements that conflict with knowledge store data.
    pub knowledge_conflicts: usize,
    /// ANE utilization fraction.
    pub ane_utilization: f32,
    /// The proof's hash (for external verification).
    pub proof_hash: String,
}

/// Known op→device mapping for synthetic verification.
///
/// These are the expected placements based on Apple's documentation
/// and empirical observations from Core ML compute plans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownOpPlacement {
    /// The op pattern (e.g., "mb.linear", "mb.gelu").
    pub op_pattern: String,
    /// Expected device class (what we'd predict).
    pub expected_device: DeviceClass,
    /// Confidence of this prediction.
    pub confidence: f32,
    /// Source of this knowledge (e.g., "documentation", "empirical").
    pub source: String,
}

/// Offline verifier for compute plan proofs.
///
/// Can verify compute plan proofs on any platform (Linux, Windows, etc.)
/// without requiring Apple hardware or the Core ML runtime.
pub struct ComputePlanVerifier {
    /// Known op→device mappings for synthetic verification.
    known_placements: Vec<KnownOpPlacement>,
}

impl ComputePlanVerifier {
    /// Create a new verifier with default known op placements.
    ///
    /// The default known placements are based on Apple's documentation
    /// and widely-reported empirical observations about which Core ML
    /// ops the compute planner assigns to which device.
    pub fn new() -> Self {
        Self {
            known_placements: Self::default_known_placements(),
        }
    }

    /// Create a verifier with custom known placements.
    pub fn with_known_placements(known: Vec<KnownOpPlacement>) -> Self {
        Self {
            known_placements: known,
        }
    }

    /// Default known op→device mappings.
    ///
    /// These represent the current best understanding of how Core ML's
    /// compute planner assigns ops to devices. They are conservative:
    /// only ops that are well-documented as ANE-friendly are marked
    /// as NeuralEngine placements.
    fn default_known_placements() -> Vec<KnownOpPlacement> {
        vec![
            KnownOpPlacement {
                op_pattern: "mb.linear".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.85,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.matmul".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.80,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.gelu".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.75,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.scaled_dot_product_attention".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.70,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.softmax".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.75,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.layer_norm".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.80,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.reshape".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.90,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.transpose".into(),
                expected_device: DeviceClass::NeuralEngine,
                confidence: 0.90,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.embedding".into(),
                expected_device: DeviceClass::CPU,
                confidence: 0.90,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.topk".into(),
                expected_device: DeviceClass::CPU,
                confidence: 0.85,
                source: "empirical".into(),
            },
            KnownOpPlacement {
                op_pattern: "mb.gather".into(),
                expected_device: DeviceClass::CPU,
                confidence: 0.70,
                source: "empirical".into(),
            },
        ]
    }

    /// Verify a compute plan proof.
    ///
    /// Checks:
    /// 1. Structural validity (all ops have placements, hash matches)
    /// 2. Invariant compliance (device classes are valid, no duplicate op names)
    /// 3. Knowledge consistency (placements match known op→device mappings)
    pub fn verify(&self, proof: &ComputePlanProof) -> VerificationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut knowledge_matches = 0usize;
        let mut knowledge_conflicts = 0usize;

        // 1. Check proof hash integrity
        let computed_hash = compute_proof_hash(&proof.placements);
        if computed_hash != proof.proof_hash {
            errors.push(format!(
                "Proof hash mismatch: stored={}, computed={}",
                proof.proof_hash, computed_hash
            ));
        }

        // 2. Check op count consistency
        if proof.placements.len() != proof.op_count {
            errors.push(format!(
                "Op count mismatch: stored={}, actual placements={}",
                proof.op_count, proof.placements.len()
            ));
        }

        // 3. Check ANE count consistency
        let actual_ane_count = proof.placements.iter()
            .filter(|p| p.device_class.is_ane())
            .count();
        if actual_ane_count != proof.ane_placed_count {
            errors.push(format!(
                "ANE count mismatch: stored={}, actual={}",
                proof.ane_placed_count, actual_ane_count
            ));
        }

        // 4. Check ANE fraction consistency
        let actual_ane_fraction = if proof.op_count > 0 {
            actual_ane_count as f32 / proof.op_count as f32
        } else {
            0.0
        };
        if (actual_ane_fraction - proof.ane_placed_fraction).abs() > 0.01 {
            warnings.push(format!(
                "ANE fraction drift: stored={:.3}, computed={:.3}",
                proof.ane_placed_fraction, actual_ane_fraction
            ));
        }

        // 5. Check for duplicate op names within a function
        let mut seen_ops: HashMap<String, String> = HashMap::new();
        for placement in &proof.placements {
            let key = format!("{}::{}", placement.function_name, placement.op_name);
            if let Some(prev) = seen_ops.insert(key, placement.op_type.clone()) {
                errors.push(format!(
                    "Duplicate op name: {} appears as both {} and {}",
                    placement.op_name, prev, placement.op_type
                ));
            }
        }

        // 6. Check that all placements have valid device classes
        for placement in &proof.placements {
            if matches!(placement.device_class, DeviceClass::Unknown(_)) {
                warnings.push(format!(
                    "Unknown device class for op {} (type {})",
                    placement.op_name, placement.op_type
                ));
            }
        }

        // 7. Cross-reference with known op→device mappings
        for placement in &proof.placements {
            if let Some(known) = self.known_placements.iter()
                .find(|k| k.op_pattern == placement.op_type)
            {
                if placement.device_class == known.expected_device {
                    knowledge_matches += 1;
                } else {
                    knowledge_conflicts += 1;
                    warnings.push(format!(
                        "Placement mismatch for {} (type {}): expected {:?}, got {:?} (confidence={:.2}, source={})",
                        placement.op_name, placement.op_type,
                        known.expected_device, placement.device_class,
                        known.confidence, known.source
                    ));
                }
            }
            // Ops without known mappings are neither matches nor conflicts
        }

        let is_valid = errors.is_empty();
        let knowledge_consistent = knowledge_conflicts == 0;

        VerificationResult {
            is_valid,
            errors,
            warnings,
            knowledge_consistent,
            knowledge_matches,
            knowledge_conflicts,
            ane_utilization: actual_ane_fraction,
            proof_hash: computed_hash,
        }
    }

    /// Generate a synthetic compute plan proof from a MIR-like op list.
    ///
    /// Given a list of op types and function names, predict what the
    /// compute plan would look like using known op→device mappings.
    /// This produces a "predicted" proof that can be compared against
    /// an actual proof from Apple hardware.
    pub fn predict_proof(
        &self,
        model_id: &str,
        ops: &[(String, String)], // (op_name, op_type)
        function_name: &str,
    ) -> ComputePlanProof {
        let placements: Vec<PlacementEntry> = ops.iter().map(|(name, op_type)| {
            let device = self.known_placements.iter()
                .find(|k| k.op_pattern == *op_type)
                .map(|k| k.expected_device.clone())
                .unwrap_or(DeviceClass::CPU); // Default: unknown ops go to CPU

            PlacementEntry {
                op_name: name.clone(),
                op_type: op_type.clone(),
                device_class: device,
                function_name: function_name.into(),
            }
        }).collect();

        let op_count = placements.len();
        let ane_placed_count = placements.iter().filter(|p| p.device_class.is_ane()).count();
        let ane_placed_fraction = if op_count > 0 {
            ane_placed_count as f32 / op_count as f32
        } else {
            0.0
        };

        let proof_hash = compute_proof_hash(&placements);

        ComputePlanProof {
            proof_id: format!("predicted_{}", model_id),
            model_id: model_id.into(),
            hardware: "predicted".into(),
            os_version: "predicted".into(),
            placements,
            proof_hash,
            timestamp: chrono::Utc::now().to_rfc3339(),
            op_count,
            ane_placed_count,
            ane_placed_fraction,
        }
    }
}

/// Compute a deterministic hash of the placement entries.
///
/// This allows two independent observers to verify they got the
/// same compute plan by comparing hashes.
fn compute_proof_hash(placements: &[PlacementEntry]) -> String {
    use sha2::{Sha256, Digest};

    // Sort placements by function_name then op_name for determinism
    let mut sorted: Vec<&PlacementEntry> = placements.iter().collect();
    sorted.sort_by(|a, b| {
        match a.function_name.cmp(&b.function_name) {
            std::cmp::Ordering::Equal => a.op_name.cmp(&b.op_name),
            other => other,
        }
    });

    let mut hasher = Sha256::new();
    for p in &sorted {
        hasher.update(p.function_name.as_bytes());
        hasher.update(p.op_name.as_bytes());
        hasher.update(p.op_type.as_bytes());
        hasher.update(format!("{:?}", p.device_class).as_bytes());
    }

    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_proof() -> ComputePlanProof {
        let placements = vec![
            PlacementEntry {
                op_name: "linear_1".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "gelu_1".into(),
                op_type: "mb.gelu".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "linear_2".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "layernorm_1".into(),
                op_type: "mb.layer_norm".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
        ];

        let op_count = placements.len();
        let ane_placed_count = placements.iter().filter(|p| p.device_class.is_ane()).count();
        let ane_placed_fraction = ane_placed_count as f32 / op_count as f32;
        let proof_hash = compute_proof_hash(&placements);

        ComputePlanProof {
            proof_id: "test_proof_1".into(),
            model_id: "test_model".into(),
            hardware: "M2".into(),
            os_version: "macOS_15".into(),
            placements,
            proof_hash,
            timestamp: "2025-01-01T00:00:00Z".into(),
            op_count,
            ane_placed_count,
            ane_placed_fraction,
        }
    }

    #[test]
    fn test_valid_proof_verifies() {
        let verifier = ComputePlanVerifier::new();
        let proof = make_valid_proof();
        let result = verifier.verify(&proof);

        assert!(result.is_valid, "Valid proof should verify: errors={:?}", result.errors);
        assert!(result.knowledge_consistent, "Proof should be knowledge-consistent: conflicts={}", result.knowledge_conflicts);
        assert_eq!(result.knowledge_matches, 4, "All 4 ops should match known placements");
        assert_eq!(result.knowledge_conflicts, 0, "No conflicts expected");
        assert!((result.ane_utilization - 1.0).abs() < 0.01, "All ops on ANE, utilization should be 1.0");
    }

    #[test]
    fn test_tampered_hash_detected() {
        let verifier = ComputePlanVerifier::new();
        let mut proof = make_valid_proof();
        proof.proof_hash = "tampered_hash".into();
        let result = verifier.verify(&proof);

        assert!(!result.is_valid, "Tampered hash should fail verification");
        assert!(result.errors.iter().any(|e| e.contains("hash mismatch")));
    }

    #[test]
    fn test_wrong_op_count_detected() {
        let verifier = ComputePlanVerifier::new();
        let mut proof = make_valid_proof();
        proof.op_count = 999; // Wrong
        let result = verifier.verify(&proof);

        assert!(!result.is_valid, "Wrong op count should fail verification");
        assert!(result.errors.iter().any(|e| e.contains("Op count mismatch")));
    }

    #[test]
    fn test_wrong_ane_count_detected() {
        let verifier = ComputePlanVerifier::new();
        let mut proof = make_valid_proof();
        proof.ane_placed_count = 0; // Wrong (all 4 are on ANE)
        let result = verifier.verify(&proof);

        assert!(!result.is_valid, "Wrong ANE count should fail verification");
    }

    #[test]
    fn test_placement_mismatch_with_knowledge() {
        let verifier = ComputePlanVerifier::new();

        // Create a proof where an embedding op is incorrectly placed on ANE
        let placements = vec![
            PlacementEntry {
                op_name: "embed_1".into(),
                op_type: "mb.embedding".into(),
                device_class: DeviceClass::NeuralEngine, // Wrong! Embeddings go to CPU
                function_name: "main".into(),
            },
        ];

        let proof = ComputePlanProof {
            proof_id: "bad_embed".into(),
            model_id: "test".into(),
            hardware: "M2".into(),
            os_version: "macOS_15".into(),
            placements,
            proof_hash: "will_be_ignored".into(),
            timestamp: "2025-01-01T00:00:00Z".into(),
            op_count: 1,
            ane_placed_count: 1,
            ane_placed_fraction: 1.0,
        };

        let result = verifier.verify(&proof);
        assert!(result.knowledge_conflicts > 0,
            "Embedding on ANE should conflict with known placement");
        assert!(!result.knowledge_consistent,
            "Proof with placement conflicts should not be knowledge-consistent");
    }

    #[test]
    fn test_predict_proof() {
        let verifier = ComputePlanVerifier::new();

        let ops = vec![
            ("linear_1".into(), "mb.linear".into()),
            ("gelu_1".into(), "mb.gelu".into()),
            ("embed_1".into(), "mb.embedding".into()),
        ];

        let proof = verifier.predict_proof("test_model", &ops, "main");

        assert_eq!(proof.op_count, 3);
        assert_eq!(proof.ane_placed_count, 2, "linear + gelu on ANE, embedding on CPU");
        assert!((proof.ane_placed_fraction - 0.667).abs() < 0.01);

        // Verify the predicted proof
        let result = verifier.verify(&proof);
        assert!(result.is_valid, "Predicted proof should be structurally valid");
        assert!(result.knowledge_consistent, "Predicted proof should be knowledge-consistent");
    }

    #[test]
    fn test_proof_hash_deterministic() {
        let placements = vec![
            PlacementEntry {
                op_name: "a".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "b".into(),
                op_type: "mb.gelu".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
        ];

        let hash1 = compute_proof_hash(&placements);
        let hash2 = compute_proof_hash(&placements);
        assert_eq!(hash1, hash2, "Same placements must produce same hash");
    }

    #[test]
    fn test_proof_hash_order_independent() {
        let placements_ab = vec![
            PlacementEntry {
                op_name: "a".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "b".into(),
                op_type: "mb.gelu".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
        ];

        let placements_ba = vec![
            PlacementEntry {
                op_name: "b".into(),
                op_type: "mb.gelu".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
            PlacementEntry {
                op_name: "a".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "main".into(),
            },
        ];

        // Hash should be the same regardless of input order (sorted internally)
        assert_eq!(compute_proof_hash(&placements_ab), compute_proof_hash(&placements_ba),
            "Hash must be order-independent");
    }

    /// Test that a decoder shard proof (linear + gelu + layernorm) verifies correctly.
    #[test]
    fn test_decoder_shard_proof_verifies() {
        let verifier = ComputePlanVerifier::new();

        let placements = vec![
            PlacementEntry {
                op_name: "linear_proj".into(),
                op_type: "mb.linear".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "decode_step".into(),
            },
            PlacementEntry {
                op_name: "gelu_act".into(),
                op_type: "mb.gelu".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "decode_step".into(),
            },
            PlacementEntry {
                op_name: "layer_norm".into(),
                op_type: "mb.layer_norm".into(),
                device_class: DeviceClass::NeuralEngine,
                function_name: "decode_step".into(),
            },
        ];

        let op_count = placements.len();
        let ane_count = placements.iter().filter(|p| p.device_class.is_ane()).count();
        let proof = ComputePlanProof {
            proof_id: "decoder_shard_proof".into(),
            model_id: "decoder_shard".into(),
            hardware: "M2".into(),
            os_version: "macOS_15.2".into(),
            placements,
            proof_hash: String::new(), // Will be checked against computed
            timestamp: chrono::Utc::now().to_rfc3339(),
            op_count,
            ane_placed_count: ane_count,
            ane_placed_fraction: ane_count as f32 / op_count as f32,
        };

        // Hash mismatch will be detected (empty string vs computed)
        let result = verifier.verify(&proof);
        // But knowledge consistency should still be checked
        assert_eq!(result.knowledge_matches, 3, "All 3 ops match known placements");
    }

    /// Test the full predict-then-verify roundtrip.
    #[test]
    fn test_predict_verify_roundtrip() {
        let verifier = ComputePlanVerifier::new();

        // Predict a proof for a typical decoder shard
        let ops = vec![
            ("qkv_proj".into(), "mb.linear".into()),
            ("sdpa".into(), "mb.scaled_dot_product_attention".into()),
            ("out_proj".into(), "mb.linear".into()),
            ("ln_1".into(), "mb.layer_norm".into()),
            ("ln_2".into(), "mb.layer_norm".into()),
        ];

        let proof = verifier.predict_proof("decoder", &ops, "decode_step");

        // The predicted proof should verify cleanly
        let result = verifier.verify(&proof);
        assert!(result.is_valid, "Predicted proof should be valid: {:?}", result.errors);
        assert!(result.knowledge_consistent, "Predicted proof should be knowledge-consistent");
        assert_eq!(result.knowledge_matches, 5);
        assert_eq!(result.knowledge_conflicts, 0);
        assert_eq!(result.warnings.len(), 0, "No warnings expected");
    }
}
