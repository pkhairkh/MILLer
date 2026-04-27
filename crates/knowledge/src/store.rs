//! Knowledge Store
//!
//! File-backed persistent store for knowledge units.
//! Seeds and observations are stored separately:
//!   - Seeds: loaded from static JSON files, immutable
//!   - Observations: learned from runs, stored in append-only JSON files
//!
//! The store is file-based (not SQLite) for this v0 implementation.
//! Each observation is stored as a separate JSON file for atomicity
//! and to avoid write contention. This can be swapped for SQLite later.

use ane_ir::kir::{KnowledgeUnit, KnowledgeType, KnowledgeScope};
use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Schema version for the knowledge store directory layout.
pub const STORE_SCHEMA_VERSION: &str = "1.0.0";

/// A knowledge entry as stored in the knowledge store.
///
/// This extends `KnowledgeUnit` from KIR with store-level metadata:
/// provenance tracking, revision semantics, and conflict metadata.
/// Entries are not arbitrary JSON blobs — they have a fixed schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// The core knowledge unit data.
    pub unit: KnowledgeUnit,

    /// Store-level provenance: how this entry entered the store.
    pub provenance: EntryProvenance,

    /// Entry source: seed (immutable) or observation (append-only).
    pub source: EntrySource,

    /// Conflict metadata: tracks if this entry conflicts with others.
    pub conflict_status: ConflictStatus,

    /// Revision number: incremented on each update (observations only).
    /// Seeds always have revision 0.
    pub revision: u64,
}

/// How this entry entered the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryProvenance {
    /// How the entry was created.
    pub origin: EntryOrigin,
    /// Timestamp when the entry was first inserted.
    pub inserted_at: String,
    /// Timestamp when the entry was last updated (if ever).
    pub updated_at: Option<String>,
    /// The file path or source this entry came from (if applicable).
    pub source_path: Option<String>,
}

/// Origin of a knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntryOrigin {
    /// Loaded from a seed JSON file (immutable).
    SeedFile,
    /// Created from a compile/lab run observation.
    RunObservation,
    /// Imported from an external snapshot.
    Imported,
    /// Manually entered.
    ManualEntry,
}

/// Whether this entry is a seed or an observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntrySource {
    /// Seed entry: loaded from static JSON, immutable.
    Seed,
    /// Observation entry: learned from a run, append-only.
    Observation,
}

/// Conflict status for a knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictStatus {
    /// No known conflicts.
    NoConflict,
    /// Conflicts detected with other entries (listed by ID).
    ConflictedWith(Vec<String>),
    /// Conflict was resolved (resolution note included).
    Resolved { note: String },
}

/// The index file for a knowledge store directory.
/// Tracks all entries, schema version, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreIndex {
    /// Schema version of the store layout.
    pub schema_version: String,
    /// Timestamp when the store was created.
    pub created_at: String,
    /// Number of seed entries.
    pub seed_count: usize,
    /// Number of observation entries.
    pub observation_count: usize,
    /// IDs of all entries (for quick lookup without loading all files).
    pub entry_ids: Vec<String>,
}

/// File-backed knowledge store.
///
/// Directory layout:
/// ```text
/// <store_path>/
///   store_index.json      — Store metadata and entry index
///   seeds/
///     <id>.json           — Seed entries (immutable, loaded from knowledge/*.json)
///   observations/
///     <id>.json           — Observation entries (learned from runs)
/// ```
pub struct KnowledgeStore {
    /// Root directory of the store.
    path: PathBuf,
    /// In-memory index of all entries.
    index: HashMap<String, KnowledgeEntry>,
    /// The store metadata.
    store_index: StoreIndex,
}

impl KnowledgeStore {
    /// Open or create a knowledge store at the given path.
    ///
    /// If the directory exists and contains a store_index.json, loads existing data.
    /// If not, creates a new empty store.
    pub fn open(path: &str) -> Result<Self> {
        let store_path = PathBuf::from(path);

        if store_path.exists() {
            Self::load_existing(&store_path)
        } else {
            Self::create_new(&store_path)
        }
    }

    /// Create a new empty knowledge store.
    fn create_new(store_path: &Path) -> Result<Self> {
        fs::create_dir_all(store_path)
            .with_context(|| format!("Failed to create store directory: {}", store_path.display()))?;
        fs::create_dir_all(store_path.join("seeds"))
            .with_context(|| format!("Failed to create seeds directory: {}", store_path.display()))?;
        fs::create_dir_all(store_path.join("observations"))
            .with_context(|| format!("Failed to create observations directory: {}", store_path.display()))?;

        let now = chrono::Utc::now().to_rfc3339();
        let store_index = StoreIndex {
            schema_version: STORE_SCHEMA_VERSION.to_string(),
            created_at: now,
            seed_count: 0,
            observation_count: 0,
            entry_ids: vec![],
        };

        let store = Self {
            path: store_path.to_path_buf(),
            index: HashMap::new(),
            store_index,
        };

        store.write_store_index()?;
        Ok(store)
    }

    /// Load an existing knowledge store.
    fn load_existing(store_path: &Path) -> Result<Self> {
        let index_path = store_path.join("store_index.json");
        if !index_path.exists() {
            bail!("Store directory exists but has no store_index.json: {}", store_path.display());
        }

        let index_json = fs::read_to_string(&index_path)
            .with_context(|| format!("Failed to read store index: {}", index_path.display()))?;
        let store_index: StoreIndex = serde_json::from_str(&index_json)
            .with_context(|| format!("Failed to parse store index: {}", index_path.display()))?;

        let mut index = HashMap::new();

        // Load seed entries
        let seeds_dir = store_path.join("seeds");
        if seeds_dir.exists() {
            Self::load_entries_from_dir(&seeds_dir, &mut index)?;
        }

        // Load observation entries
        let observations_dir = store_path.join("observations");
        if observations_dir.exists() {
            Self::load_entries_from_dir(&observations_dir, &mut index)?;
        }

        Ok(Self {
            path: store_path.to_path_buf(),
            index,
            store_index,
        })
    }

    /// Load all entries from a directory.
    fn load_entries_from_dir(dir: &Path, index: &mut HashMap<String, KnowledgeEntry>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let json = fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read entry: {}", path.display()))?;
                let entry: KnowledgeEntry = serde_json::from_str(&json)
                    .with_context(|| format!("Failed to parse entry: {}", path.display()))?;
                index.insert(entry.unit.id.clone(), entry);
            }
        }
        Ok(())
    }

    /// Load seed entries from external seed JSON files (e.g., knowledge/*.json).
    ///
    /// Seeds are loaded from the project's knowledge/ directory and stored
    /// as immutable entries. This is the only way seeds enter the store.
    /// Seeds cannot be overwritten by observations.
    pub fn load_seeds_from_directory(&mut self, seeds_dir: &str) -> Result<usize> {
        let seeds_path = Path::new(seeds_dir);
        if !seeds_path.exists() {
            return Ok(0);
        }

        let mut loaded = 0;
        for entry in fs::read_dir(seeds_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }

            let json = fs::read_to_string(&path)
                .with_context(|| format!("Failed to read seed file: {}", path.display()))?;

            // Seed files have a top-level "entries" array
            let seed_file: serde_json::Value = serde_json::from_str(&json)
                .with_context(|| format!("Failed to parse seed file: {}", path.display()))?;

            if let Some(entries) = seed_file.get("entries").and_then(|e| e.as_array()) {
                for entry_val in entries {
                    match serde_json::from_value::<KnowledgeUnit>(entry_val.clone()) {
                        Ok(unit) => {
                            let knowledge_entry = KnowledgeEntry {
                                provenance: EntryProvenance {
                                    origin: EntryOrigin::SeedFile,
                                    inserted_at: chrono::Utc::now().to_rfc3339(),
                                    updated_at: None,
                                    source_path: Some(path.to_string_lossy().to_string()),
                                },
                                source: EntrySource::Seed,
                                conflict_status: ConflictStatus::NoConflict,
                                revision: 0,
                                unit,
                            };
                            let id = knowledge_entry.unit.id.clone();
                            self.index.insert(id, knowledge_entry);
                            loaded += 1;
                        }
                        Err(e) => {
                            // Log but don't fail — seed files may contain extra fields
                            eprintln!("Warning: skipping malformed seed entry: {}", e);
                        }
                    }
                }
            }
        }

        // Persist seed entries to the store
        for (id, entry) in &self.index {
            if entry.source == EntrySource::Seed {
                let seed_path = self.path.join("seeds").join(format!("{}.json", sanitize_id(id)));
                let json = serde_json::to_string_pretty(entry)
                    .with_context(|| format!("Failed to serialize seed entry: {}", id))?;
                fs::write(&seed_path, json)
                    .with_context(|| format!("Failed to write seed entry: {}", seed_path.display()))?;
            }
        }

        self.rebuild_store_index()?;
        Ok(loaded)
    }

    /// Insert a new observation knowledge entry.
    ///
    /// Observations are learned from runs and stored as append-only entries.
    /// If an entry with the same ID already exists and is also an observation,
    /// this increments the revision and updates the entry (confidence update).
    /// If the existing entry is a seed, insertion is rejected (seeds are immutable).
    pub fn insert_observation(&mut self, unit: KnowledgeUnit) -> Result<()> {
        let id = unit.id.clone();

        if let Some(existing) = self.index.get(&id) {
            if existing.source == EntrySource::Seed {
                bail!(
                    "Cannot overwrite seed entry '{}' with an observation. \
                     Seeds are immutable. Use a different ID for the observation.",
                    id
                );
            }
            // Update existing observation: increment revision, update confidence
            let mut updated = KnowledgeEntry {
                provenance: EntryProvenance {
                    origin: EntryOrigin::RunObservation,
                    inserted_at: existing.provenance.inserted_at.clone(),
                    updated_at: Some(chrono::Utc::now().to_rfc3339()),
                    source_path: existing.provenance.source_path.clone(),
                },
                source: EntrySource::Observation,
                conflict_status: existing.conflict_status.clone(),
                revision: existing.revision + 1,
                unit,
            };

            // Check for conflicts: does this contradict an existing entry?
            self.check_conflicts_for_entry(&mut updated);

            self.index.insert(id.clone(), updated);
        } else {
            let mut entry = KnowledgeEntry {
                provenance: EntryProvenance {
                    origin: EntryOrigin::RunObservation,
                    inserted_at: chrono::Utc::now().to_rfc3339(),
                    updated_at: None,
                    source_path: None,
                },
                source: EntrySource::Observation,
                conflict_status: ConflictStatus::NoConflict,
                revision: 0,
                unit,
            };

            // Check for conflicts
            self.check_conflicts_for_entry(&mut entry);

            self.index.insert(id.clone(), entry);
        }

        // Persist the observation entry
        let entry = self.index.get(&id).unwrap();
        let obs_path = self.path.join("observations").join(format!("{}.json", sanitize_id(&id)));
        let json = serde_json::to_string_pretty(entry)
            .with_context(|| format!("Failed to serialize observation: {}", id))?;
        fs::write(&obs_path, json)
            .with_context(|| format!("Failed to write observation: {}", obs_path.display()))?;

        self.rebuild_store_index()?;
        Ok(())
    }

    /// Retrieve a knowledge entry by ID.
    pub fn get(&self, id: &str) -> Option<&KnowledgeEntry> {
        self.index.get(id)
    }

    /// Retrieve a mutable reference to a knowledge entry by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut KnowledgeEntry> {
        self.index.get_mut(id)
    }

    /// List all entry IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// List all seed entry IDs.
    pub fn list_seed_ids(&self) -> Vec<String> {
        self.index.iter()
            .filter(|(_, e)| e.source == EntrySource::Seed)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// List all observation entry IDs.
    pub fn list_observation_ids(&self) -> Vec<String> {
        self.index.iter()
            .filter(|(_, e)| e.source == EntrySource::Observation)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Count entries by source type.
    pub fn counts(&self) -> (usize, usize) {
        let seeds = self.index.values().filter(|e| e.source == EntrySource::Seed).count();
        let observations = self.index.values().filter(|e| e.source == EntrySource::Observation).count();
        (seeds, observations)
    }

    /// Iterate over all entries (for querying).
    pub fn index_values(&self) -> impl Iterator<Item = &KnowledgeEntry> {
        self.index.values()
    }

    /// Get the store's root directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check for conflicts between a new/updated entry and existing entries.
    ///
    /// A conflict exists when two entries of the same knowledge type make
    /// contradictory claims within overlapping scopes. For example:
    /// - One entry says mb.gather is ANE-legal, another says it is not,
    ///   and their scopes overlap.
    fn check_conflicts_for_entry(&self, entry: &mut KnowledgeEntry) {
        let mut conflicts = Vec::new();

        for (existing_id, existing) in &self.index {
            if existing_id == &entry.unit.id {
                continue;
            }
            if existing.unit.knowledge_type != entry.unit.knowledge_type {
                continue;
            }
            if !scopes_overlap(&existing.unit.scope, &entry.unit.scope) {
                continue;
            }

            // Check for contradictory claims
            if claims_contradict(&existing.unit, &entry.unit) {
                conflicts.push(existing_id.clone());
            }
        }

        if !conflicts.is_empty() {
            entry.conflict_status = ConflictStatus::ConflictedWith(conflicts);
        }
    }

    /// Write the store index file.
    fn write_store_index(&self) -> Result<()> {
        let index_path = self.path.join("store_index.json");
        let json = serde_json::to_string_pretty(&self.store_index)
            .with_context(|| "Failed to serialize store index")?;
        fs::write(&index_path, json)
            .with_context(|| format!("Failed to write store index: {}", index_path.display()))?;
        Ok(())
    }

    /// Rebuild the store index from current entries.
    fn rebuild_store_index(&mut self) -> Result<()> {
        let (seed_count, observation_count) = self.counts();
        self.store_index.seed_count = seed_count;
        self.store_index.observation_count = observation_count;
        self.store_index.entry_ids = self.index.keys().cloned().collect();
        self.write_store_index()
    }
}

/// Check if two knowledge scopes overlap (share at least one device class, OS version, and opset version).
fn scopes_overlap(a: &KnowledgeScope, b: &KnowledgeScope) -> bool {
    let devices_overlap = a.device_classes.iter().any(|d| b.device_classes.contains(d))
        || a.device_classes.contains(&"unknown".to_string())
        || b.device_classes.contains(&"unknown".to_string());
    let os_overlap = a.os_versions.iter().any(|v| b.os_versions.contains(v))
        || a.os_versions.contains(&"unknown".to_string())
        || b.os_versions.contains(&"unknown".to_string());
    let opset_overlap = a.opset_versions.iter().any(|v| b.opset_versions.contains(v));

    devices_overlap && os_overlap && opset_overlap
}

/// Check if two knowledge units make contradictory claims.
///
/// This is a heuristic: for LegalityRule entries, contradiction means
/// one says ane_legal=true and the other says ane_legal=false.
/// For other types, we check if the confidence difference is extreme.
fn claims_contradict(a: &KnowledgeUnit, b: &KnowledgeUnit) -> bool {
    match a.knowledge_type {
        KnowledgeType::LegalityRule => {
            // Check if ane_legal claims differ
            let a_legal = a.payload.get("ane_legal").and_then(|v| v.as_bool());
            let b_legal = b.payload.get("ane_legal").and_then(|v| v.as_bool());
            match (a_legal, b_legal) {
                (Some(a_val), Some(b_val)) => a_val != b_val,
                _ => false,
            }
        }
        KnowledgeType::PrecisionHazard => {
            // Check if quality impact claims are opposite
            let a_impact = a.payload.get("quality_impact").and_then(|v| v.as_str());
            let b_impact = b.payload.get("quality_impact").and_then(|v| v.as_str());
            match (a_impact, b_impact) {
                (Some("negligible"), Some("severe")) |
                (Some("severe"), Some("negligible")) => true,
                _ => false,
            }
        }
        _ => {
            // For other types, consider extreme confidence divergence as potential conflict
            (a.confidence - b.confidence).abs() > 0.6
        }
    }
}

/// Sanitize an entry ID for use as a filename.
fn sanitize_id(id: &str) -> String {
    id.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::kir::{KnowledgeType, EvidenceSource, KnowledgeScope};
    use std::collections::HashMap;

    fn make_unit(id: &str, kt: KnowledgeType, ane_legal: bool, confidence: f32) -> KnowledgeUnit {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(ane_legal));
        payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));

        KnowledgeUnit {
            id: id.to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: kt,
            confidence,
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
    fn test_create_and_open_store() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");
        let store_path_str = store_path.to_string_lossy().to_string();

        let store = KnowledgeStore::open(&store_path_str).unwrap();
        assert_eq!(store.list_ids().len(), 0);
        assert!(store_path.join("store_index.json").exists());
        assert!(store_path.join("seeds").exists());
        assert!(store_path.join("observations").exists());
    }

    #[test]
    fn test_insert_observation() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");

        let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

        let unit = make_unit("obs_1", KnowledgeType::LegalityRule, true, 0.3);
        store.insert_observation(unit).unwrap();

        assert_eq!(store.list_ids().len(), 1);
        assert_eq!(store.list_observation_ids(), vec!["obs_1"]);

        let entry = store.get("obs_1").unwrap();
        assert_eq!(entry.source, EntrySource::Observation);
        assert_eq!(entry.revision, 0);
        assert_eq!(entry.provenance.origin, EntryOrigin::RunObservation);

        // Check file was written
        assert!(store_path.join("observations").join("obs_1.json").exists());
    }

    #[test]
    fn test_observation_revision() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");

        let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

        // Insert first version
        let unit = make_unit("obs_1", KnowledgeType::LegalityRule, true, 0.3);
        store.insert_observation(unit).unwrap();

        // Insert updated version
        let unit_v2 = make_unit("obs_1", KnowledgeType::LegalityRule, true, 0.5);
        store.insert_observation(unit_v2).unwrap();

        let entry = store.get("obs_1").unwrap();
        assert_eq!(entry.revision, 1); // Incremented
        assert_eq!(entry.unit.confidence, 0.5); // Updated
        assert!(entry.provenance.updated_at.is_some());
    }

    #[test]
    fn test_cannot_overwrite_seed_with_observation() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");

        let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

        // Manually insert a seed entry
        let seed_unit = make_unit("seed_1", KnowledgeType::LegalityRule, true, 0.7);
        let seed_entry = KnowledgeEntry {
            unit: seed_unit,
            provenance: EntryProvenance {
                origin: EntryOrigin::SeedFile,
                inserted_at: chrono::Utc::now().to_rfc3339(),
                updated_at: None,
                source_path: None,
            },
            source: EntrySource::Seed,
            conflict_status: ConflictStatus::NoConflict,
            revision: 0,
        };
        store.index.insert("seed_1".to_string(), seed_entry);

        // Try to overwrite with observation — should fail
        let obs_unit = make_unit("seed_1", KnowledgeType::LegalityRule, false, 0.3);
        let result = store.insert_observation(obs_unit);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot overwrite seed entry"));
    }

    #[test]
    fn test_conflict_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");

        let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

        // Insert: matmul is legal on M2
        let unit_a = make_unit("obs_a", KnowledgeType::LegalityRule, true, 0.8);
        store.insert_observation(unit_a).unwrap();

        // Insert: matmul is NOT legal on M2 (same scope, opposite claim)
        let unit_b = make_unit("obs_b", KnowledgeType::LegalityRule, false, 0.7);
        store.insert_observation(unit_b).unwrap();

        // obs_b should be marked as conflicted with obs_a
        let entry_b = store.get("obs_b").unwrap();
        if let ConflictStatus::ConflictedWith(ids) = &entry_b.conflict_status {
            assert!(ids.contains(&"obs_a".to_string()));
        } else {
            panic!("Expected ConflictedWith status for obs_b");
        }
    }

    #[test]
    fn test_seeds_and_observations_separated() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");

        let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

        // Insert an observation
        let obs_unit = make_unit("obs_1", KnowledgeType::LegalityRule, true, 0.3);
        store.insert_observation(obs_unit).unwrap();

        let (seeds, observations) = store.counts();
        assert_eq!(seeds, 0);
        assert_eq!(observations, 1);

        // Observation file should be in observations/, not seeds/
        assert!(store_path.join("observations").join("obs_1.json").exists());
        assert!(!store_path.join("seeds").join("obs_1.json").exists());
    }

    #[test]
    fn test_reopen_store() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("test_store");
        let store_path_str = store_path.to_string_lossy().to_string();

        // Create and insert
        {
            let mut store = KnowledgeStore::open(&store_path_str).unwrap();
            let unit = make_unit("obs_persist", KnowledgeType::LegalityRule, true, 0.3);
            store.insert_observation(unit).unwrap();
        }

        // Reopen
        let store = KnowledgeStore::open(&store_path_str).unwrap();
        assert_eq!(store.list_ids().len(), 1);
        let entry = store.get("obs_persist").unwrap();
        assert_eq!(entry.source, EntrySource::Observation);
    }

    #[test]
    fn test_scopes_overlap() {
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
    fn test_unknown_scope_overlaps() {
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
}
