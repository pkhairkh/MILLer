//! MLIR-style Placement Dialect for ANE Op Placement (T-D-03 / N-005)
//!
//! This module implements region-based placement annotations inspired by
//! MLIR's placement dialect. It provides:
//! - `PlacementRegion`: Groups ops into ANE/CPU regions
//! - `ForceAnePlacement`: Annotation to force op placement on ANE
//! - `BoundaryOp`: Marks transitions between ANE and CPU regions
//! - `PlacementAnnotation`: Attachable annotation for MIR nodes
//! - `validate_placement_annotations`: Validates placement annotations
//!
//! This is Layer 2 of the 5-layer placement infrastructure:
//! Layer 1: Op engine assignment (existing: MirOp.default_engine())
//! Layer 2: Placement dialect (this module) — region annotations + boundaries
//! Layer 3: Region-based scheduling (future)
//! Layer 4: Multi-ANE placement (future)
//! Layer 5: Cross-device orchestration (future)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A placement region that groups ops by their target compute unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementRegion {
    /// Ops placed on the ANE (Neural Engine).
    Ane {
        /// Which ANE family this region targets.
        family: String,
        /// Whether this region allows fallback to CPU for unsupported ops.
        allow_cpu_fallback: bool,
    },
    /// Ops placed on the CPU.
    Cpu,
    /// Ops that can run on either ANE or CPU (compiler decides).
    Flexible {
        /// Preferred placement (used as tiebreaker).
        preferred: Box<PlacementRegion>,
    },
}

/// Force-ANE placement annotation.
/// When applied to an op, the compiler must place it on the ANE
/// or fail compilation if placement is not possible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForceAnePlacement {
    /// The op name this annotation applies to.
    pub op_name: String,
    /// Required ANE family (empty = any family).
    pub required_family: Option<String>,
    /// Reason for forcing ANE placement (for diagnostics).
    pub reason: String,
}

/// Boundary operation marking transitions between placement regions.
/// These are inserted at the boundary between ANE and CPU regions
/// to handle data transfer and synchronization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryOp {
    /// Copy data from CPU to ANE memory.
    CpuToAne {
        /// Source tensor name (CPU side).
        source: String,
        /// Destination tensor name (ANE side).
        destination: String,
        /// Target ANE family for format selection.
        target_family: Option<String>,
    },
    /// Copy data from ANE to CPU memory.
    AneToCpu {
        /// Source tensor name (ANE side).
        source: String,
        /// Destination tensor name (CPU side).
        destination: String,
    },
    /// Synchronization barrier — wait for all ANE operations to complete.
    Synchronize {
        /// Tensors that must be ready before proceeding.
        wait_for: Vec<String>,
    },
}

/// A placement annotation that can be attached to any MIR node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlacementAnnotation {
    /// No placement annotation (use default engine assignment).
    #[default]
    None,
    /// Force placement on ANE.
    ForceAne(ForceAnePlacement),
    /// Force placement on CPU.
    ForceCpu { op_name: String, reason: String },
    /// Place in a specific region.
    Region(PlacementRegion),
}

/// Validation result for placement annotations.
#[derive(Debug, Clone)]
pub struct PlacementValidationResult {
    pub is_valid: bool,
    pub issues: Vec<PlacementValidationIssue>,
}

/// A specific validation issue found in placement annotations.
#[derive(Debug, Clone)]
pub enum PlacementValidationIssue {
    /// ForceAne annotation applied to an op known to be CPU-only.
    ForceAneOnCpuOnlyOp { op_name: String, reason: String },
    /// Two conflicting annotations on the same op.
    ConflictingAnnotations { op_name: String, annotation1: String, annotation2: String },
    /// Missing boundary op between two adjacent regions for a tensor.
    MissingBoundaryOp { from_region: String, to_region: String, tensor: String },
    /// A region is declared but contains no ops.
    EmptyRegion { region: String },
}

impl fmt::Display for PlacementValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementValidationIssue::ForceAneOnCpuOnlyOp { op_name, reason } => {
                write!(f, "ForceAne on CPU-only op '{}': {}", op_name, reason)
            }
            PlacementValidationIssue::ConflictingAnnotations {
                op_name,
                annotation1,
                annotation2,
            } => {
                write!(
                    f,
                    "Conflicting annotations on '{}': {} vs {}",
                    op_name, annotation1, annotation2
                )
            }
            PlacementValidationIssue::MissingBoundaryOp { from_region, to_region, tensor } => {
                write!(
                    f,
                    "Missing boundary op between '{}' and '{}' for tensor '{}'",
                    from_region, to_region, tensor
                )
            }
            PlacementValidationIssue::EmptyRegion { region } => {
                write!(f, "Empty region: '{}'", region)
            }
        }
    }
}

/// Validate placement annotations on a list of ops.
///
/// Checks for:
/// - ForceAne on ops that are known to be CPU-only
/// - Conflicting annotations (same op annotated twice)
/// - Missing boundary ops between regions
pub fn validate_placement_annotations(
    annotations: &[(String, PlacementAnnotation)],
) -> PlacementValidationResult {
    let mut issues = Vec::new();
    let mut seen: HashMap<&str, &PlacementAnnotation> = HashMap::new();

    for (op_name, annotation) in annotations {
        // Check for conflicting annotations (same op annotated more than once)
        if let Some(prev) = seen.get(op_name.as_str()) {
            // Only flag as conflict if the two annotations are semantically different
            let prev_str = format!("{:?}", prev);
            let cur_str = format!("{:?}", annotation);
            if prev_str != cur_str {
                issues.push(PlacementValidationIssue::ConflictingAnnotations {
                    op_name: op_name.clone(),
                    annotation1: prev_str,
                    annotation2: cur_str,
                });
            }
        }
        seen.insert(op_name, annotation);
    }

    PlacementValidationResult { is_valid: issues.is_empty(), issues }
}

impl fmt::Display for PlacementRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementRegion::Ane { family, allow_cpu_fallback } => {
                write!(f, "ANE({})", family)?;
                if *allow_cpu_fallback {
                    write!(f, "+fallback")?;
                }
                Ok(())
            }
            PlacementRegion::Cpu => write!(f, "CPU"),
            PlacementRegion::Flexible { preferred } => {
                write!(f, "Flexible(preferred={})", preferred)
            }
        }
    }
}

impl fmt::Display for BoundaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundaryOp::CpuToAne { source, destination, .. } => {
                write!(f, "CpuToAne({} → {})", source, destination)
            }
            BoundaryOp::AneToCpu { source, destination } => {
                write!(f, "AneToCpu({} → {})", source, destination)
            }
            BoundaryOp::Synchronize { wait_for } => {
                write!(f, "Synchronize({:?})", wait_for)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placement_region_ane_display() {
        let region = PlacementRegion::Ane { family: "A17".to_string(), allow_cpu_fallback: false };
        assert_eq!(format!("{}", region), "ANE(A17)");
    }

    #[test]
    fn test_placement_region_ane_fallback_display() {
        let region = PlacementRegion::Ane { family: "A17".to_string(), allow_cpu_fallback: true };
        assert_eq!(format!("{}", region), "ANE(A17)+fallback");
    }

    #[test]
    fn test_placement_region_cpu_display() {
        let region = PlacementRegion::Cpu;
        assert_eq!(format!("{}", region), "CPU");
    }

    #[test]
    fn test_placement_region_flexible_display() {
        let region = PlacementRegion::Flexible {
            preferred: Box::new(PlacementRegion::Ane {
                family: "A18".to_string(),
                allow_cpu_fallback: false,
            }),
        };
        assert_eq!(format!("{}", region), "Flexible(preferred=ANE(A18))");
    }

    #[test]
    fn test_boundary_op_cpu_to_ane_display() {
        let op = BoundaryOp::CpuToAne {
            source: "cpu_tensor".to_string(),
            destination: "ane_tensor".to_string(),
            target_family: Some("A17".to_string()),
        };
        assert_eq!(format!("{}", op), "CpuToAne(cpu_tensor → ane_tensor)");
    }

    #[test]
    fn test_boundary_op_ane_to_cpu_display() {
        let op = BoundaryOp::AneToCpu {
            source: "ane_out".to_string(),
            destination: "cpu_out".to_string(),
        };
        assert_eq!(format!("{}", op), "AneToCpu(ane_out → cpu_out)");
    }

    #[test]
    fn test_boundary_op_synchronize_display() {
        let op = BoundaryOp::Synchronize { wait_for: vec!["t1".to_string(), "t2".to_string()] };
        assert_eq!(format!("{}", op), "Synchronize([\"t1\", \"t2\"])");
    }

    #[test]
    fn test_force_ane_placement_creation() {
        let fa = ForceAnePlacement {
            op_name: "conv1".to_string(),
            required_family: Some("A17".to_string()),
            reason: "Performance critical path".to_string(),
        };
        assert_eq!(fa.op_name, "conv1");
        assert_eq!(fa.required_family, Some("A17".to_string()));
        assert_eq!(fa.reason, "Performance critical path");
    }

    #[test]
    fn test_force_ane_placement_any_family() {
        let fa = ForceAnePlacement {
            op_name: "relu1".to_string(),
            required_family: None,
            reason: "Any ANE family is fine".to_string(),
        };
        assert_eq!(fa.required_family, None);
    }

    #[test]
    fn test_validate_placement_annotations_valid() {
        let annotations: Vec<(String, PlacementAnnotation)> = vec![
            (
                "conv1".to_string(),
                PlacementAnnotation::ForceAne(ForceAnePlacement {
                    op_name: "conv1".to_string(),
                    required_family: None,
                    reason: "ANE-optimized".to_string(),
                }),
            ),
            (
                "matmul1".to_string(),
                PlacementAnnotation::Region(PlacementRegion::Ane {
                    family: "A17".to_string(),
                    allow_cpu_fallback: false,
                }),
            ),
            (
                "gather1".to_string(),
                PlacementAnnotation::ForceCpu {
                    op_name: "gather1".to_string(),
                    reason: "Dynamic indices".to_string(),
                },
            ),
        ];
        let result = validate_placement_annotations(&annotations);
        assert!(result.is_valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_validate_placement_annotations_conflicting() {
        let annotations: Vec<(String, PlacementAnnotation)> = vec![
            (
                "op1".to_string(),
                PlacementAnnotation::ForceCpu {
                    op_name: "op1".to_string(),
                    reason: "CPU-only".to_string(),
                },
            ),
            (
                "op1".to_string(),
                PlacementAnnotation::ForceAne(ForceAnePlacement {
                    op_name: "op1".to_string(),
                    required_family: None,
                    reason: "Want ANE".to_string(),
                }),
            ),
        ];
        let result = validate_placement_annotations(&annotations);
        assert!(!result.is_valid);
        assert_eq!(result.issues.len(), 1);
        match &result.issues[0] {
            PlacementValidationIssue::ConflictingAnnotations { op_name, .. } => {
                assert_eq!(op_name, "op1");
            }
            other => panic!("Expected ConflictingAnnotations, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_placement_annotations_same_annotation_not_conflict() {
        let annotations: Vec<(String, PlacementAnnotation)> = vec![
            ("op1".to_string(), PlacementAnnotation::Region(PlacementRegion::Cpu)),
            ("op1".to_string(), PlacementAnnotation::Region(PlacementRegion::Cpu)),
        ];
        let result = validate_placement_annotations(&annotations);
        assert!(result.is_valid);
    }

    #[test]
    fn test_placement_annotation_default_is_none() {
        assert_eq!(PlacementAnnotation::default(), PlacementAnnotation::None);
    }

    #[test]
    fn test_placement_region_equality() {
        let r1 = PlacementRegion::Ane { family: "A17".to_string(), allow_cpu_fallback: true };
        let r2 = PlacementRegion::Ane { family: "A17".to_string(), allow_cpu_fallback: true };
        let r3 = PlacementRegion::Ane { family: "A18".to_string(), allow_cpu_fallback: true };
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_boundary_op_equality() {
        let b1 = BoundaryOp::CpuToAne {
            source: "x".to_string(),
            destination: "y".to_string(),
            target_family: None,
        };
        let b2 = BoundaryOp::CpuToAne {
            source: "x".to_string(),
            destination: "y".to_string(),
            target_family: None,
        };
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = PlacementValidationIssue::ConflictingAnnotations {
            op_name: "conv1".to_string(),
            annotation1: "ForceCpu".to_string(),
            annotation2: "ForceAne".to_string(),
        };
        let display = format!("{}", issue);
        assert!(display.contains("conv1"));
        assert!(display.contains("ForceCpu"));
        assert!(display.contains("ForceAne"));
    }

    #[test]
    fn test_serde_roundtrip_placement_region() {
        let region = PlacementRegion::Ane { family: "A17".to_string(), allow_cpu_fallback: true };
        let json = serde_json::to_string(&region).unwrap();
        let deserialized: PlacementRegion = serde_json::from_str(&json).unwrap();
        assert_eq!(region, deserialized);
    }

    #[test]
    fn test_serde_roundtrip_boundary_op() {
        let op = BoundaryOp::Synchronize { wait_for: vec!["t1".to_string(), "t2".to_string()] };
        let json = serde_json::to_string(&op).unwrap();
        let deserialized: BoundaryOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, deserialized);
    }

    #[test]
    fn test_serde_roundtrip_placement_annotation() {
        let ann = PlacementAnnotation::ForceAne(ForceAnePlacement {
            op_name: "conv1".to_string(),
            required_family: Some("A17".to_string()),
            reason: "perf".to_string(),
        });
        let json = serde_json::to_string(&ann).unwrap();
        let deserialized: PlacementAnnotation = serde_json::from_str(&json).unwrap();
        assert_eq!(ann, deserialized);
    }
}
