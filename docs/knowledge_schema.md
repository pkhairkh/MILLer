# Knowledge Schema

See [SPEC.md](../SPEC.md) section 6 (Knowledge Adaptation Model) for the design rationale.

This document describes the implemented knowledge store schema.

## Store Layout

The knowledge store uses a file-backed layout (not SQLite for v0):

```
<store_path>/
  store_index.json      — Store metadata and entry index
  seeds/
    <id>.json           — Seed entries (immutable, loaded from knowledge/*.json)
  observations/
    <id>.json           — Observation entries (learned from runs)
```

Schema version: `1.0.0`

## Entry Model

Each entry in the store is a `KnowledgeEntry` with the following structure:

```json
{
  "unit": {
    "id": "string — unique identifier (e.g., 'legality_matmul_fp16')",
    "version": 1,
    "timestamp": "RFC3339 timestamp",
    "knowledge_type": "LegalityRule | MotifCatalog | SurvivalMatrixEntry | ShardTemplateKnowledge | PrecisionHazard | FallbackSignature | DeviceFingerprint | StateTopologyOutcome | SyntheticTransferAnnotation | CpuOnlyOps | AneHwLimits | PalettizationConstraints | AneOpFamilyMatrix",
    "confidence": 0.7,
    "evidence_source": "SyntheticRun | RealModelRun | CompileFailure | LoadFailure | RuntimeAnomaly | ManualEntry | CrossValidated",
    "evidence_count": 5,
    "scope": {
      "device_classes": ["M2"],
      "os_versions": ["macOS_15"],
      "opset_versions": ["iOS18"]
    },
    "conflict_priority": 0,
    "payload": { "ane_legal": true, "op_pattern": "mb.matmul" }
  },
  "provenance": {
    "origin": "SeedFile | RunObservation | Imported | ManualEntry",
    "inserted_at": "RFC3339 timestamp",
    "updated_at": "RFC3339 timestamp or null",
    "source_path": "string or null"
  },
  "source": "Seed | Observation",
  "conflict_status": "NoConflict | ConflictedWith([ids]) | Resolved { note }",
  "revision": 0
}
```

## Seeds vs Observations

- **Seeds**: Loaded from `knowledge/*.json` files. Immutable once loaded. Cannot be overwritten by observations. Always revision 0.
- **Observations**: Created from compile/lab runs or manual entry. Append-only. Revision increments on update. Confidence can be updated.

## Query System

The store implements the `KnowledgeQueryable` trait with the following filters:

- **By knowledge type**: Filter to a specific `KnowledgeType`
- **By confidence**: Minimum confidence threshold
- **By evidence source**: Filter to observations from a specific source
- **By scope**: Filter to entries matching a given scope (device class, OS version, opset version)

Scope matching is conservative: entries with "unknown" scope match any query.

## Conflict Detection

Conflicts are detected automatically when inserting observations:

- **ContradictoryLegality**: Two entries make opposite `ane_legal` claims for overlapping scopes
- **ConfidenceDivergence**: Same claim but very different confidence levels
- **OverlappingScope**: Entries with overlapping scopes that may conflict

High-confidence contradictions (either entry >= 0.8) require manual review.

## Synthetic Transfer

Synthetic knowledge can be transferred to real-model contexts with reduced confidence:

- **Operator-level** (LegalityRule, SurvivalMatrixEntry): confidence × 0.7
- **Pattern-level** (MotifCatalog, FallbackSignature): confidence × 0.65
- **Precision hazards**: confidence × 0.6
- **Topology-level** (ShardTemplateKnowledge, StateTopologyOutcome, DeviceFingerprint): NOT transferable

## Snapshot Export/Import

The entire store can be exported as a single JSON snapshot containing all seeds and observations. Snapshots preserve the seed/observation distinction and include schema versioning for forward compatibility.

## Seed File Formats (Current)

The seed JSON files in `knowledge/` do **not** yet follow the `KnowledgeEntry` schema
defined above. This section documents their actual formats, the discrepancy, and
the planned migration path.

### Why the mismatch exists

The `KnowledgeEntry` schema was designed as the target format for the knowledge
store's internal representation (with `unit`, `provenance`, `source`,
`conflict_status`, and `revision` blocks). The seed files were created earlier
as flat, domain-specific JSON and have not yet been migrated. Changing them now
would break existing load paths (see `load_seeds_from_directory` and
`load_shard_template_seeds`), so the migration is deferred to a future sprint.

### Files using `entries[]` pattern

These files have a top-level `version` and `entries` array. Each entry is a
flat object with knowledge-type-specific fields. They partially overlap with
`KnowledgeUnit` but lack `version`, `timestamp`, `conflict_priority`, and a
structured `payload` field. Extra fields beyond `KnowledgeUnit`'s schema are
silently dropped by `serde` during deserialization.

| File | Knowledge type | Entry count | Notes |
|------|---------------|-------------|-------|
| `legality_seed.json` | LegalityRule | 5 | Fields: `id`, `knowledge_type`, `op_pattern`, `ane_legal`, `confidence`, `evidence_source`, `evidence_count`, `scope`. The `gather` entry also has `note` and `limited_index_range`. |
| `precision_hazard_seed.json` | PrecisionHazard | 4 | Fields: `id`, `knowledge_type`, `op_pattern` (was `op`, renamed in T-129), `weight_type`, `bitwidth`, `granularity`, `layer_range`, `quality_impact`, `confidence`, `evidence_source`, `evidence_count`, `scope`, `note`. |
| `shard_template_seed.json` | ShardTemplateKnowledge | 1 | Loaded via dedicated `ShardTemplateSeedFile` struct (not through generic `load_seeds_from_directory`). Fields: `id`, `knowledge_type`, `template_id`, `partition_spec[]`, `io_model`, `sampler`, `state_config`, `context_length`, `known_good`, `quality_delta`, `confidence`, `evidence_source`, `evidence_count`, `scope`. |
| `decode_step_shard_template_seed.json` | ShardTemplateKnowledge | 1 | Same format as `shard_template_seed.json`. |

**Current load behaviour**: `load_seeds_from_directory()` attempts to
deserialize each entry as `KnowledgeUnit`. Entries that lack required fields
(`version`, `timestamp`, `conflict_priority`, `payload`) are skipped with a
`log::warn`. The shard template files are loaded separately via
`load_shard_template_seeds()` which uses the `ShardTemplateSeedFile` struct.

### Files using flat / domain-specific formats

These files do **not** have an `entries[]` array. They use custom top-level
keys and are **not** loaded by `load_seeds_from_directory()` at all — the
function only looks for an `entries` key and silently ignores files without one.

| File | Top-level key | Knowledge type (not yet in enum) | Notes |
|------|--------------|----------------------------------|-------|
| `cpu_only_ops_seed.json` | `cpu_only_ops[]` | CpuOnlyOps | Each entry: `mil_name`, `reason_code`. Values are hardcoded in `crates/passes/src/cpu_only_ops.rs`. |
| `ane_hw_limits_seed.json` | `hw_limits[]` | AneHwLimits | Each entry: `revision`, `family`, `max_tensor_width`, etc. Values are hardcoded in `crates/ir/src/ane_hw_limits.rs`. |
| `ane_op_family_matrix.json` | `ane_landing_ops[]` | AneOpFamilyMatrix | Each entry: `mil_name`, `anec_dialect`, `engine`, `families{}`, `key_constraints[]`. Not yet loaded by any crate. |
| `palettization_constraints_seed.json` | Named sub-objects | PalettizationConstraints | No array — structured as `conv_palette_minimums`, `palette_upcast`, `vector_palettization_incompatibilities`, `hard_rejections`. Not yet loaded by any crate. |

### Field name fix (T-129)

`precision_hazard_seed.json` previously used `"op"` where the schema defines
`"op_pattern"`. This has been renamed to `"op_pattern"` to align with the
schema and the `payload_op_pattern()` accessor in `util.rs`.

### Migration path

In a future sprint, all seed files will be converted to produce proper
`KnowledgeEntry` objects on load. This requires:

1. **Adding the 4 missing `KnowledgeType` variants** to the Rust enum (done in T-129):
   - `CpuOnlyOps` — CPU-only op catalog entries
   - `AneHwLimits` — per-revision hardware limit parameters
   - `PalettizationConstraints` — palettization/LUT compression constraints
   - `AneOpFamilyMatrix` — per-family ANE op legality matrix entries

2. **Wrapping seed data in `KnowledgeEntry`** during load. Each flat entry will
   be promoted to a full `KnowledgeEntry` with:
   - `unit.payload` containing the domain-specific fields
   - `unit.knowledge_type` set to the appropriate new enum variant
   - `provenance.origin = SeedFile`
   - `source = Seed`, `conflict_status = NoConflict`, `revision = 0`
   - Auto-generated `unit.id`, `unit.version`, `unit.timestamp` where missing

3. **Updating `load_seeds_from_directory()`** to handle all format variants
   (not just `entries[]`), or adding dedicated loader functions for each
   knowledge type.

4. **Updating consumers** that currently read from hardcoded Rust statics
   (e.g., `CPU_ONLY_OPS`, `AneHwLimits::for_revision()`) to optionally
   source from the knowledge store.

Until then, the seed files serve as the human-readable reference/documentation
source, while the Rust code is the runtime source of truth.

## Residuals

- The store is file-backed (JSON), not SQLite. This is adequate for v0 but may need SQLite for larger stores.
- The `UpdatePipeline` does not yet implement automatic confidence decay over time.
- Conflict auto-resolution is limited to low-confidence entries; high-confidence conflicts always require manual review.
- The synthetic transfer confidence scaling factors are initial estimates per SPEC.md section 6.6; they should be validated empirically.
