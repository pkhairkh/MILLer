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
    "knowledge_type": "LegalityRule | MotifCatalog | SurvivalMatrixEntry | ShardTemplateKnowledge | PrecisionHazard | FallbackSignature | DeviceFingerprint | StateTopologyOutcome | SyntheticTransferAnnotation",
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

## Residuals

- The store is file-backed (JSON), not SQLite. This is adequate for v0 but may need SQLite for larger stores.
- The `UpdatePipeline` does not yet implement automatic confidence decay over time.
- Conflict auto-resolution is limited to low-confidence entries; high-confidence conflicts always require manual review.
- The synthetic transfer confidence scaling factors are initial estimates per SPEC.md section 6.6; they should be validated empirically.
