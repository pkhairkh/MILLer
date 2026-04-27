//! Shard Template Seed Loading
//!
//! Typed Rust structs for loading and validating shard template seed
//! entries from JSON files. This module bridges the gap between the
//! freeform JSON seed files in `knowledge/` and the typed PIR types
//! used by the compiler.
//!
//! ## Seed File Format
//!
//! A shard template seed file contains an array of entries, each
//! describing a proven partitioning pattern. The Qwen3 three-shard
//! template is the primary example:
//!
//! ```json
//! {
//!   "version": 1,
//!   "entries": [{
//!     "id": "shard_qwen3_three_shard_v1",
//!     "knowledge_type": "ShardTemplateKnowledge",
//!     "template_id": "qwen3-three-shard-v1",
//!     "partition_spec": [
//!       { "role": "Entry", "layers": [0, 10], "compute_units": "CPU_AND_NE" },
//!       { "role": "Interior", "layers": [11, 19], "compute_units": "CPU_AND_NE" },
//!       { "role": "Exit", "layers": [20, 27], "compute_units": "CPU_AND_NE" }
//!     ],
//!     "io_model": { "compute_units": "CPU_AND_GPU" },
//!     "sampler": { "compute_units": "CPU_AND_GPU" },
//!     "state_config": "per_shard_kv_reverse_ring_buffer",
//!     "context_length": 4096,
//!     "known_good": true,
//!     "quality_delta": { "perplexity_delta": -0.57 },
//!     "confidence": 0.92,
//!     "evidence_source": "RealModelRun",
//!     "evidence_count": 15,
//!     "scope": { "device_classes": ["M2", "M2_Pro", "M3"], ... }
//!   }]
//! }
//! ```
//!
//! ## Validation
//!
//! When loading, each entry is validated:
//! - Template ID must not be empty
//! - Partition specs must use valid role names
//! - Layer ranges must be non-empty (end >= start)
//! - Io and Sampler entries are optional but valid
//! - The seed is scoped and confidence-scored; it is NOT universal truth

use ane_ir::pir::{ShardRole, ComputeUnits, ShardTemplate, ShardPartitionEntry};
use ane_ir::kir::{KnowledgeScope, EvidenceSource};
use anyhow::{Result, bail, Context};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A shard template seed file (top-level structure).
///
/// This matches the JSON format in `knowledge/shard_template_seed.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTemplateSeedFile {
    /// Schema version of the seed file.
    pub version: u64,
    /// Human-readable description of the seed file.
    #[serde(default)]
    pub description: Option<String>,
    /// Array of shard template entries.
    pub entries: Vec<ShardTemplateSeedEntry>,
}

/// A single shard template seed entry.
///
/// Represents one proven partitioning pattern with metadata about
/// its scope, confidence, and quality impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTemplateSeedEntry {
    /// Unique identifier for this seed entry.
    pub id: String,
    /// Knowledge type (always "ShardTemplateKnowledge").
    pub knowledge_type: String,
    /// Template identifier (e.g., "qwen3-three-shard-v1").
    pub template_id: String,
    /// Partition specifications for decoder shards.
    pub partition_spec: Vec<PartitionSpecEntry>,
    /// I/O model configuration (optional).
    pub io_model: Option<IoModelSpec>,
    /// Sampler model configuration (optional).
    pub sampler: Option<SamplerSpec>,
    /// State configuration (e.g., "per_shard_kv_reverse_ring_buffer").
    #[serde(default)]
    pub state_config: Option<String>,
    /// Context length this template supports.
    #[serde(default)]
    pub context_length: usize,
    /// Whether this template is known to produce good results.
    #[serde(default)]
    pub known_good: bool,
    /// Quality delta metrics.
    #[serde(default)]
    pub quality_delta: Option<QualityDelta>,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Evidence source for this template.
    #[serde(default = "default_evidence_source")]
    pub evidence_source: String,
    /// Number of evidence points supporting this template.
    #[serde(default = "default_evidence_count")]
    pub evidence_count: usize,
    /// Scope of applicability.
    #[serde(default)]
    pub scope: Option<SeedScope>,
}

/// A partition specification entry from a seed file.
///
/// Describes one decoder shard: its role, layer range, and
/// target compute units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionSpecEntry {
    /// Shard role (e.g., "Entry", "Interior", "Exit").
    pub role: String,
    /// Layer range [start, end] (inclusive).
    pub layers: Vec<usize>,
    /// Target compute units (e.g., "CPU_AND_NE").
    pub compute_units: String,
}

/// I/O model specification from a seed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoModelSpec {
    /// Target compute units for the I/O model.
    pub compute_units: String,
}

/// Sampler model specification from a seed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerSpec {
    /// Target compute units for the sampler model.
    pub compute_units: String,
}

/// Quality delta metrics for a shard template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDelta {
    /// Change in perplexity (negative = better).
    #[serde(default)]
    pub perplexity_delta: Option<f64>,
}

/// Scope from a seed file (matches knowledge schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedScope {
    /// Device classes this template applies to.
    #[serde(default)]
    pub device_classes: Vec<String>,
    /// OS versions this template applies to.
    #[serde(default)]
    pub os_versions: Vec<String>,
    /// Opset versions this template applies to.
    #[serde(default)]
    pub opset_versions: Vec<String>,
}

fn default_evidence_source() -> String {
    "ManualEntry".to_string()
}

fn default_evidence_count() -> usize {
    1
}

/// A validated and converted shard template ready for use by the compiler.
///
/// This is the product of loading and validating a seed entry. It contains
/// the same information but with typed fields instead of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedShardTemplate {
    /// The original seed entry ID.
    pub seed_id: String,
    /// The PIR-compatible shard template.
    pub template: ShardTemplate,
    /// Whether this template is known to produce good results.
    pub known_good: bool,
    /// Quality delta metrics (if available).
    pub quality_delta: Option<QualityDelta>,
    /// Confidence score.
    pub confidence: f32,
    /// Evidence source.
    pub evidence_source: EvidenceSource,
    /// Evidence count.
    pub evidence_count: usize,
    /// Scope of applicability.
    pub scope: KnowledgeScope,
}

/// Load and validate all shard template seed entries from a directory.
///
/// Reads all `.json` files from the given directory, parses them as
/// `ShardTemplateSeedFile`, validates each entry, and converts them
/// to `ValidatedShardTemplate` instances.
///
/// Invalid entries are logged but do not cause the entire load to fail.
/// This is intentional: seed files may contain entries for knowledge
/// types other than shard templates, and those should be skipped silently.
pub fn load_shard_template_seeds(dir: &str) -> Result<Vec<ValidatedShardTemplate>> {
    let seeds_path = Path::new(dir);
    if !seeds_path.exists() {
        return Ok(vec![]);
    }

    let mut templates = Vec::new();

    for entry in fs::read_dir(seeds_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let json = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read shard template seed: {}", path.display()))?;

        let seed_file: ShardTemplateSeedFile = match serde_json::from_str(&json) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Warning: skipping malformed shard template seed file {}: {}", path.display(), e);
                continue;
            }
        };

        for seed_entry in seed_file.entries {
            // Skip entries that are not shard template knowledge
            if seed_entry.knowledge_type != "ShardTemplateKnowledge" {
                continue;
            }

            match validate_and_convert(&seed_entry) {
                Ok(template) => templates.push(template),
                Err(e) => {
                    eprintln!("Warning: skipping invalid shard template seed entry '{}': {}", seed_entry.id, e);
                }
            }
        }
    }

    Ok(templates)
}

/// Load and validate a single shard template seed file.
pub fn load_shard_template_seed_file(path: &str) -> Result<Vec<ValidatedShardTemplate>> {
    let json = fs::read_to_string(path)
        .with_context(|| format!("Failed to read shard template seed file: {}", path))?;

    let seed_file: ShardTemplateSeedFile = serde_json::from_str(&json)
        .with_context(|| format!("Failed to parse shard template seed file: {}", path))?;

    let mut templates = Vec::new();
    for seed_entry in seed_file.entries {
        if seed_entry.knowledge_type != "ShardTemplateKnowledge" {
            continue;
        }

        let template = validate_and_convert(&seed_entry)
            .with_context(|| format!("Invalid shard template seed entry: {}", seed_entry.id))?;
        templates.push(template);
    }

    Ok(templates)
}

/// Validate a seed entry and convert it to a typed `ValidatedShardTemplate`.
fn validate_and_convert(entry: &ShardTemplateSeedEntry) -> Result<ValidatedShardTemplate> {
    // Validate template ID
    if entry.template_id.is_empty() {
        bail!("Template ID must not be empty");
    }

    // Validate and convert partition specs
    let mut partition_entries = Vec::new();
    for spec in &entry.partition_spec {
        let role = ShardRole::from_str_flexible(&spec.role)
            .ok_or_else(|| anyhow::anyhow!("Invalid shard role: '{}'", spec.role))?;

        // Validate layer range
        if spec.layers.len() != 2 {
            bail!("Partition spec for role '{}' must have exactly 2 layer bounds, got {}", spec.role, spec.layers.len());
        }
        let layer_start = spec.layers[0];
        let layer_end = spec.layers[1];
        if layer_end < layer_start {
            bail!("Partition spec for role '{}' has invalid layer range: [{}, {}]", spec.role, layer_start, layer_end);
        }

        let compute_units = ComputeUnits::from_str_flexible(&spec.compute_units)
            .ok_or_else(|| anyhow::anyhow!("Invalid compute units: '{}'", spec.compute_units))?;

        partition_entries.push(ShardPartitionEntry {
            role,
            layer_start,
            layer_end,
            compute_units,
        });
    }

    // Validate IO model compute units
    let io_compute_units: Option<ComputeUnits> = match entry.io_model.as_ref() {
        Some(io) => Some(ComputeUnits::from_str_flexible(&io.compute_units)
            .ok_or_else(|| anyhow::anyhow!("Invalid IO model compute units: '{}'", io.compute_units))?),
        None => None,
    };

    // Validate sampler compute units
    let sampler_compute_units: Option<ComputeUnits> = match entry.sampler.as_ref() {
        Some(s) => Some(ComputeUnits::from_str_flexible(&s.compute_units)
            .ok_or_else(|| anyhow::anyhow!("Invalid sampler compute units: '{}'", s.compute_units))?),
        None => None,
    };

    // Build PIR ShardTemplate
    let template = ShardTemplate {
        template_id: entry.template_id.clone(),
        partition_spec: partition_entries,
        io_compute_units,
        sampler_compute_units,
        state_config: entry.state_config.clone(),
        context_length: entry.context_length,
    };

    // Parse evidence source
    let evidence_source = parse_evidence_source(&entry.evidence_source);

    // Build scope
    let scope = entry.scope.as_ref().map(|s| KnowledgeScope {
        device_classes: s.device_classes.clone(),
        os_versions: s.os_versions.clone(),
        opset_versions: s.opset_versions.clone(),
    }).unwrap_or_else(|| KnowledgeScope {
        device_classes: vec!["unknown".to_string()],
        os_versions: vec!["unknown".to_string()],
        opset_versions: vec!["iOS18".to_string()],
    });

    // Validate confidence
    if entry.confidence < 0.0 || entry.confidence > 1.0 {
        bail!("Confidence must be in [0.0, 1.0], got {}", entry.confidence);
    }

    Ok(ValidatedShardTemplate {
        seed_id: entry.id.clone(),
        template,
        known_good: entry.known_good,
        quality_delta: entry.quality_delta.clone(),
        confidence: entry.confidence,
        evidence_source,
        evidence_count: entry.evidence_count,
        scope,
    })
}

/// Parse an evidence source string to the typed enum.
fn parse_evidence_source(s: &str) -> EvidenceSource {
    match s {
        "SyntheticRun" => EvidenceSource::SyntheticRun,
        "RealModelRun" => EvidenceSource::RealModelRun,
        "CompileFailure" => EvidenceSource::CompileFailure,
        "LoadFailure" => EvidenceSource::LoadFailure,
        "RuntimeAnomaly" => EvidenceSource::RuntimeAnomaly,
        "ManualEntry" => EvidenceSource::ManualEntry,
        "CrossValidated" => EvidenceSource::CrossValidated,
        _ => EvidenceSource::ManualEntry, // Default for unknown sources
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_shard_template_seeds() {
        // Try workspace-relative path first, then fall back to crate-relative
        let knowledge_dir = if Path::new("knowledge").exists() {
            "knowledge"
        } else if Path::new("../../knowledge").exists() {
            "../../knowledge"
        } else {
            // No knowledge directory available in test environment
            eprintln!("Skipping test_load_shard_template_seeds: no knowledge/ directory found");
            return;
        };

        let templates = load_shard_template_seeds(knowledge_dir).unwrap();
        // There should be at least one entry from shard_template_seed.json
        assert!(!templates.is_empty(), "Expected at least one shard template seed");

        // Directory iteration order is not guaranteed, so find the qwen3 entry by ID
        let qwen3 = templates.iter()
            .find(|t| t.seed_id == "shard_qwen3_three_shard_v1")
            .expect("Expected shard_qwen3_three_shard_v1 entry in seed files");
        assert_eq!(qwen3.seed_id, "shard_qwen3_three_shard_v1");
        assert_eq!(qwen3.template.template_id, "qwen3-three-shard-v1");
        assert_eq!(qwen3.template.partition_spec.len(), 3);
        assert_eq!(qwen3.template.partition_spec[0].role, ShardRole::Entry);
        assert_eq!(qwen3.template.partition_spec[1].role, ShardRole::Interior);
        assert_eq!(qwen3.template.partition_spec[2].role, ShardRole::Exit);
        assert_eq!(qwen3.template.partition_spec[0].layer_start, 0);
        assert_eq!(qwen3.template.partition_spec[0].layer_end, 10);
        assert_eq!(qwen3.template.io_compute_units, Some(ComputeUnits::CPUAndGPU));
        assert_eq!(qwen3.template.sampler_compute_units, Some(ComputeUnits::CPUAndGPU));
        assert_eq!(qwen3.template.state_config, Some("per_shard_kv_reverse_ring_buffer".to_string()));
        assert_eq!(qwen3.template.context_length, 4096);
        assert!(qwen3.known_good);
        assert!((qwen3.confidence - 0.92).abs() < 0.01);
        assert_eq!(qwen3.evidence_source, EvidenceSource::RealModelRun);
        assert_eq!(qwen3.evidence_count, 15);

        // Also verify the decode_step template is loaded (directory order is arbitrary)
        assert!(templates.len() >= 2, "Expected at least 2 shard template seeds, got {}", templates.len());
        let decode_step = templates.iter()
            .find(|t| t.seed_id == "shard_decode_step_three_shard_v1")
            .expect("Expected shard_decode_step_three_shard_v1 entry");
        assert_eq!(decode_step.template.template_id, "decode-step-three-shard-v1");
        assert_eq!(decode_step.template.partition_spec.len(), 3);
        assert!(!decode_step.known_good);
        assert!((decode_step.confidence - 0.5).abs() < 0.01);
        assert_eq!(decode_step.evidence_source, EvidenceSource::SyntheticRun);
    }

    #[test]
    fn test_load_from_nonexistent_directory() {
        let templates = load_shard_template_seeds("/nonexistent/path").unwrap();
        assert!(templates.is_empty());
    }

    #[test]
    fn test_validate_and_convert_valid_entry() {
        let entry = ShardTemplateSeedEntry {
            id: "test_v1".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "test-template".to_string(),
            partition_spec: vec![
                PartitionSpecEntry {
                    role: "Entry".to_string(),
                    layers: vec![0, 5],
                    compute_units: "CPU_AND_NE".to_string(),
                },
                PartitionSpecEntry {
                    role: "Exit".to_string(),
                    layers: vec![6, 10],
                    compute_units: "CPU_AND_NE".to_string(),
                },
            ],
            io_model: Some(IoModelSpec { compute_units: "CPU_AND_GPU".to_string() }),
            sampler: Some(SamplerSpec { compute_units: "CPU_AND_GPU".to_string() }),
            state_config: Some("per_shard_kv".to_string()),
            context_length: 2048,
            known_good: true,
            quality_delta: None,
            confidence: 0.8,
            evidence_source: "SyntheticRun".to_string(),
            evidence_count: 5,
            scope: Some(SeedScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            }),
        };

        let result = validate_and_convert(&entry).unwrap();
        assert_eq!(result.seed_id, "test_v1");
        assert_eq!(result.template.partition_spec.len(), 2);
        assert_eq!(result.template.partition_spec[0].role, ShardRole::Entry);
        assert_eq!(result.template.partition_spec[0].layer_start, 0);
        assert_eq!(result.template.partition_spec[0].layer_end, 5);
        assert_eq!(result.evidence_source, EvidenceSource::SyntheticRun);
    }

    #[test]
    fn test_validate_rejects_empty_template_id() {
        let entry = ShardTemplateSeedEntry {
            id: "test".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "".to_string(),
            partition_spec: vec![],
            io_model: None,
            sampler: None,
            state_config: None,
            context_length: 0,
            known_good: false,
            quality_delta: None,
            confidence: 0.5,
            evidence_source: "ManualEntry".to_string(),
            evidence_count: 1,
            scope: None,
        };

        assert!(validate_and_convert(&entry).is_err());
    }

    #[test]
    fn test_validate_rejects_invalid_role() {
        let entry = ShardTemplateSeedEntry {
            id: "test".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "test-tmpl".to_string(),
            partition_spec: vec![PartitionSpecEntry {
                role: "InvalidRole".to_string(),
                layers: vec![0, 10],
                compute_units: "CPU_AND_NE".to_string(),
            }],
            io_model: None,
            sampler: None,
            state_config: None,
            context_length: 0,
            known_good: false,
            quality_delta: None,
            confidence: 0.5,
            evidence_source: "ManualEntry".to_string(),
            evidence_count: 1,
            scope: None,
        };

        assert!(validate_and_convert(&entry).is_err());
    }

    #[test]
    fn test_validate_rejects_bad_confidence() {
        let entry = ShardTemplateSeedEntry {
            id: "test".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "test-tmpl".to_string(),
            partition_spec: vec![],
            io_model: None,
            sampler: None,
            state_config: None,
            context_length: 0,
            known_good: false,
            quality_delta: None,
            confidence: 1.5,
            evidence_source: "ManualEntry".to_string(),
            evidence_count: 1,
            scope: None,
        };

        assert!(validate_and_convert(&entry).is_err());
    }

    #[test]
    fn test_validate_rejects_inverted_layer_range() {
        let entry = ShardTemplateSeedEntry {
            id: "test".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "test-tmpl".to_string(),
            partition_spec: vec![PartitionSpecEntry {
                role: "Entry".to_string(),
                layers: vec![10, 5], // inverted
                compute_units: "CPU_AND_NE".to_string(),
            }],
            io_model: None,
            sampler: None,
            state_config: None,
            context_length: 0,
            known_good: false,
            quality_delta: None,
            confidence: 0.5,
            evidence_source: "ManualEntry".to_string(),
            evidence_count: 1,
            scope: None,
        };

        assert!(validate_and_convert(&entry).is_err());
    }

    #[test]
    fn test_validate_accepts_io_and_sampler_roles_in_partition() {
        let entry = ShardTemplateSeedEntry {
            id: "test".to_string(),
            knowledge_type: "ShardTemplateKnowledge".to_string(),
            template_id: "test-tmpl".to_string(),
            partition_spec: vec![
                PartitionSpecEntry {
                    role: "Io".to_string(),
                    layers: vec![0, 0],
                    compute_units: "CPU_AND_GPU".to_string(),
                },
                PartitionSpecEntry {
                    role: "Entry".to_string(),
                    layers: vec![0, 5],
                    compute_units: "CPU_AND_NE".to_string(),
                },
                PartitionSpecEntry {
                    role: "Sampler".to_string(),
                    layers: vec![0, 0],
                    compute_units: "CPU_AND_GPU".to_string(),
                },
            ],
            io_model: None,
            sampler: None,
            state_config: None,
            context_length: 0,
            known_good: false,
            quality_delta: None,
            confidence: 0.5,
            evidence_source: "ManualEntry".to_string(),
            evidence_count: 1,
            scope: None,
        };

        let result = validate_and_convert(&entry).unwrap();
        assert_eq!(result.template.partition_spec[0].role, ShardRole::Io);
        assert_eq!(result.template.partition_spec[2].role, ShardRole::Sampler);
    }

    #[test]
    fn test_shard_role_from_str_flexible() {
        assert_eq!(ShardRole::from_str_flexible("Entry"), Some(ShardRole::Entry));
        assert_eq!(ShardRole::from_str_flexible("entry"), Some(ShardRole::Entry));
        assert_eq!(ShardRole::from_str_flexible("Interior"), Some(ShardRole::Interior));
        assert_eq!(ShardRole::from_str_flexible("Exit"), Some(ShardRole::Exit));
        assert_eq!(ShardRole::from_str_flexible("Io"), Some(ShardRole::Io));
        assert_eq!(ShardRole::from_str_flexible("io"), Some(ShardRole::Io));
        assert_eq!(ShardRole::from_str_flexible("IO"), Some(ShardRole::Io));
        assert_eq!(ShardRole::from_str_flexible("Sampler"), Some(ShardRole::Sampler));
        assert_eq!(ShardRole::from_str_flexible("sampler"), Some(ShardRole::Sampler));
        assert_eq!(ShardRole::from_str_flexible("invalid"), None);
    }

    #[test]
    fn test_shard_role_default_compute_units() {
        assert_eq!(ShardRole::Entry.default_compute_units(), ComputeUnits::CPUAndNE);
        assert_eq!(ShardRole::Interior.default_compute_units(), ComputeUnits::CPUAndNE);
        assert_eq!(ShardRole::Exit.default_compute_units(), ComputeUnits::CPUAndNE);
        assert_eq!(ShardRole::Io.default_compute_units(), ComputeUnits::CPUAndGPU);
        assert_eq!(ShardRole::Sampler.default_compute_units(), ComputeUnits::CPUAndGPU);
    }
}
