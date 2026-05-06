//! Snapshot Export/Import
//!
//! Serialize and deserialize the entire knowledge store
//! for backup, sharing, and version control.
//!
//! Snapshots include all seed and observation entries,
//! preserving the distinction between them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::store::{EntrySource, KnowledgeEntry, KnowledgeStore, STORE_SCHEMA_VERSION};
use crate::util::sanitize_id;

/// Snapshot of the knowledge store at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSnapshot {
    /// Schema version of the snapshot format.
    pub schema_version: String,
    /// Timestamp when the snapshot was created.
    pub timestamp: String,
    /// Seed entries (immutable).
    pub seeds: Vec<KnowledgeEntry>,
    /// Observation entries (learned from runs).
    pub observations: Vec<KnowledgeEntry>,
}

/// Export/import operations for knowledge snapshots.
pub struct SnapshotExport;

impl SnapshotExport {
    /// Export a knowledge store to a JSON snapshot file.
    ///
    /// The snapshot includes all entries, preserving the seed/observation
    /// distinction. This is suitable for backup, version control, and
    /// transferring knowledge between environments.
    pub fn export_store(store: &KnowledgeStore, path: &str) -> Result<()> {
        let snapshot = Self::snapshot_from_store(store);

        let json = serde_json::to_string_pretty(&snapshot)
            .with_context(|| "Failed to serialize knowledge snapshot")?;

        std::fs::write(path, json)
            .with_context(|| format!("Failed to write snapshot to: {}", path))?;

        Ok(())
    }

    /// Create a snapshot from a knowledge store (in memory).
    pub fn snapshot_from_store(store: &KnowledgeStore) -> KnowledgeSnapshot {
        let mut seeds = Vec::new();
        let mut observations = Vec::new();

        for entry in store.index_values() {
            match entry.source {
                EntrySource::Seed => seeds.push(entry.clone()),
                EntrySource::Observation => observations.push(entry.clone()),
            }
        }

        KnowledgeSnapshot {
            schema_version: STORE_SCHEMA_VERSION.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            seeds,
            observations,
        }
    }
}

/// Import operations for knowledge snapshots.
pub struct SnapshotImport;

impl SnapshotImport {
    /// Import a knowledge snapshot from a JSON file.
    ///
    /// Returns the snapshot without modifying any store.
    /// Use `import_into_store` to merge the snapshot into an existing store.
    pub fn import_json(path: &str) -> Result<KnowledgeSnapshot> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read snapshot file: {}", path))?;

        let snapshot: KnowledgeSnapshot = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse snapshot file: {}", path))?;

        Ok(snapshot)
    }

    /// Import a snapshot into an existing knowledge store.
    ///
    /// Seed entries are only added if no entry with the same ID exists
    /// (seeds are never overwritten). Observation entries are inserted
    /// via the normal insert_observation path (with conflict detection).
    pub fn import_into_store(
        store: &mut KnowledgeStore,
        snapshot: &KnowledgeSnapshot,
    ) -> Result<ImportStats> {
        let mut stats = ImportStats::default();

        // Validate first
        let warnings = Self::validate(snapshot)?;
        if !warnings.is_empty() {
            log::warn!("Snapshot validation warnings:");
            for w in &warnings {
                log::warn!("  - {}", w);
            }
        }

        // Import seeds (only if not already present)
        for entry in &snapshot.seeds {
            if store.get(&entry.unit.id).is_some() {
                stats.seeds_skipped += 1;
                continue;
            }
            // We can't use insert_observation for seeds, so we need
            // a way to insert seeds directly. For now, we'll write
            // the seed file directly.
            let id = entry.unit.id.clone();
            let seed_path = store.path().join("seeds").join(format!("{}.json", sanitize_id(&id)));
            let json = serde_json::to_string_pretty(entry)
                .with_context(|| format!("Failed to serialize seed: {}", id))?;
            std::fs::write(&seed_path, json)
                .with_context(|| format!("Failed to write seed: {}", seed_path.display()))?;
            store.type_index.entry(entry.unit.knowledge_type).or_default().push(id.clone());
            store
                .source_index
                .entry(entry.unit.evidence_source.to_string())
                .or_default()
                .push(id.clone());
            store.index.insert(id, entry.clone());
            stats.seeds_imported += 1;
        }

        // Import observations
        for entry in &snapshot.observations {
            match store.insert_observation((*entry.unit).clone()) {
                Ok(()) => stats.observations_imported += 1,
                Err(e) => {
                    stats.observations_failed += 1;
                    log::warn!("failed to import observation '{}': {}", entry.unit.id, e);
                }
            }
        }

        Ok(stats)
    }

    /// Validate a snapshot before importing.
    ///
    /// Returns a list of warnings (not errors) for issues found.
    /// A snapshot is still importable with warnings.
    pub fn validate(snapshot: &KnowledgeSnapshot) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check schema version
        if snapshot.schema_version != STORE_SCHEMA_VERSION {
            warnings.push(format!(
                "Snapshot schema version '{}' differs from current '{}'. Import may have issues.",
                snapshot.schema_version, STORE_SCHEMA_VERSION
            ));
        }

        // Check for duplicate IDs within the snapshot
        let mut seen_ids = std::collections::HashSet::new();
        for entry in snapshot.seeds.iter().chain(snapshot.observations.iter()) {
            if seen_ids.contains(&entry.unit.id) {
                warnings.push(format!("Duplicate entry ID '{}' in snapshot", entry.unit.id));
            }
            seen_ids.insert(&entry.unit.id);

            // Check confidence bounds
            if entry.unit.confidence < 0.0 || entry.unit.confidence > 1.0 {
                warnings.push(format!(
                    "Entry '{}' has out-of-bounds confidence: {}",
                    entry.unit.id, entry.unit.confidence
                ));
            }

            // Check evidence count
            if entry.unit.evidence_count == 0 {
                warnings.push(format!("Entry '{}' has zero evidence count", entry.unit.id));
            }
        }

        // Check that seeds don't claim observation origin
        for entry in &snapshot.seeds {
            if entry.provenance.origin == crate::store::EntryOrigin::RunObservation {
                warnings.push(format!(
                    "Seed entry '{}' claims RunObservation origin — this may be a data error",
                    entry.unit.id
                ));
            }
        }

        Ok(warnings)
    }
}

/// Statistics from a snapshot import.
#[derive(Debug, Default)]
pub struct ImportStats {
    pub seeds_imported: usize,
    pub seeds_skipped: usize,
    pub observations_imported: usize,
    pub observations_failed: usize,
}

// sanitize_id is now provided by crate::util

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
    use std::collections::HashMap;

    fn make_unit(id: &str) -> KnowledgeUnit {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(true));

        KnowledgeUnit {
            id: id.to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::LegalityRule,
            confidence: 0.5,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 1,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload,
        }
    }

    #[test]
    fn test_export_and_import_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let unit = make_unit("obs_1");
        store.insert_observation(unit).unwrap();

        let snapshot_path = tmp.path().join("snapshot.json").to_string_lossy().to_string();
        SnapshotExport::export_store(&store, &snapshot_path).unwrap();

        let imported = SnapshotImport::import_json(&snapshot_path).unwrap();
        assert_eq!(imported.observations.len(), 1);
        assert_eq!(imported.observations[0].unit.id, "obs_1");
        assert_eq!(imported.schema_version, STORE_SCHEMA_VERSION);
    }

    #[test]
    fn test_validate_clean_snapshot() {
        let snapshot = KnowledgeSnapshot {
            schema_version: STORE_SCHEMA_VERSION.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            seeds: vec![],
            observations: vec![],
        };

        let warnings = SnapshotImport::validate(&snapshot).unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_schema_mismatch() {
        let snapshot = KnowledgeSnapshot {
            schema_version: "0.0.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            seeds: vec![],
            observations: vec![],
        };

        let warnings = SnapshotImport::validate(&snapshot).unwrap();
        assert!(warnings.iter().any(|w| w.contains("schema version")));
    }
}
