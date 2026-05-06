# ISSUES.md — MILLer Compatibility & Correctness Issue Tracker

> Active issue tracker for the MILLer compiler project.
> Resolved issues have been removed — see git history for the full audit trail.
> Issues are classified by domain, severity, and evidence strength.
> Each issue is independently resolvable by an AI coding agent with repository access.
> Last audit: 2026-05-06 (codebase verification against all tracked issues)

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique issue identifier (`V-NNN`, `M-NNN`, `F-NNN`, `N-NNN`) |
| `title` | Concise, GitHub-style issue title |
| `domain` | Primary concern area |
| `severity` | `critical` / `high` / `medium` / `low` |
| `class` | Violation classification from audit |
| `status` | `open` / `partial` / `deferred` / `resolved` |
| `labels` | GitHub-style labels for filtering |
| `forensic_ref` | Cross-reference to forensic constraint-summary section |
| `affects` | What is impacted (compilation, emission, runtime, documentation) |
| `reproduce` | Steps or conditions that trigger the issue |
| `fix_hint` | Suggested approach for resolution |
| `task_ref` | Reference to TASKS.md task |

---

## Active Issues

(No active issues remain. All tracked issues have been resolved.)

---

## Resolved Issues (Removed)

The following issues were resolved across three audit rounds (2026-05-06):

**Round 1** (22 issues): V-006, V-011, V-014, V-023, V-026, V-037, V-051, V-055, V-056, F-CROSS-01, F-ARCH-01, M-006, M-009, M-012, M-015, N-002, N-003, N-006, N-007, N-008, N-010, N-011, N-012

**Round 2** (17 issues):
- **M-003** (T-P5-04): `compat_output_shape_fallible()` now handles all MirOp variants with explicit errors instead of silently returning empty vec; catch-all logs warnings
- **M-005** (T-P7-01): `slanc_scales.rs` removed entirely — deprecated stub-mimic pass that was never wired into pipeline
- **M-011** (T-P7-05): `validate_legality_status()` gate added to `RiskAnnotatePass`; compilation fails by default if any node has `LegalityStatus::Unknown`; opt-in `new_allow_unknown()` for development
- **M-013** (T-P7-02): CLI now calls `KnowledgeStore::load_seeds_from_directory()` during startup with default `knowledge/` directory fallback
- **M-014** (T-P5-04): MILConv now computes correct output shape `[batch, C_out, out_H, out_W]`; MILLinear documented as requiring weight metadata for output dim (Indeterminate error in fallible API)
- **M-016** (T-P5-09): Name-based dtype heuristic (`ends_with("_ids")`, `contains("mask")`) removed from `mil_lower.rs`; defaults to Fp16 with info log when no dtype_hint set
- **M-017** (T-P5-05): Renamed to `AneLegalityRewritePass` with `CompilationTarget` enum; `LegalityRewritePass` kept as backward-compat type alias; CPU target returns early
- **M-018** (T-P5-09): `compat_input_shape_explicit()` and `compat_output_shape_explicit()` accept `explicit_shape: Option<&[usize]>` to bypass name heuristics; legacy functions log deprecation warnings when heuristic fires
- **M-024** (T-P5-12): Renamed to `StateTopologyValidator` with backward-compat alias `StateTopologyPass`
- **M-025** (T-P7-03): Circular substitution chain now returns `anyhow::bail!()` hard error instead of logging warning and returning partial result
- **M-027** (T-P7-04): Unknown projections in `palettize_weights` now return hard error by default; `allow_unknown_projections: true` opt-in preserves backward compat
- **M-028** (T-P7-07): IOSurface size validation, surface uniformity, and flat buffer layout constraints extracted from `mir_to_proto.rs` into `placement_validate.rs::validate_surface_constraints()`
- **M-029** (T-P5-11): `apply_time_decay()` implements SPEC §553-554 linear decay (1% per 30 days); `KnowledgeStore::apply_confidence_decay()` applies to all entries
- **M-030** (T-P5-11): `KnowledgeStore::prune_below_threshold()` removes entries below confidence threshold, cleans up secondary indexes
- **M-033** (T-P5-04): `resolve_reshape_shape()` now accepts `batch_size: Option<usize>` parameter; when provided, uses known batch size instead of assuming batch=1; None fallback logs M-033 warning
- **N-004** (T-P6-03): All MirOp variants now have explicit match arms in placement_validate.rs; pooling, gather, LRN, constexpr_dequantize wired with constraint checks; 9 new N-004 tests
- **N-009** (T-P7-08): MILTile added to CPU_ONLY_OPS with `NoConverter` reason; safety net for when legality rewrite doesn't fire

**Round 3** (7 issues — all remaining partial + deferred):
- **F-HAL-01** (T-P7-09): HAL sub-variant-specific hardware limits now populated; H14c/H15c/H16c verified with reduced NE counts (1/1/2 vs parent's 2/2/4); H16s verified with expanded pe_reduction_cout_limit (32768) and hw_wa limits (32768); H13g remains unverified (no reliable compact data)
- **F-OPS-01** (T-P7-10): `anec_fused_conv_activate` and `anec_scaled_elementwise` promoted from CPU-only stubs to fully-supported ops with MirOpCompat variants, MIR-to-compat conversion, proto emission paths (decomposition to ANE-legal arithmetic), shape inference rules, and removal from CPU_ONLY_OPS; remaining 14 ANEC stubs stay CPU-only
- **M-019** (T-P7-06): All callers in `mir_to_compat.rs` migrated from legacy `compat_input_shape`/`compat_output_shape` to explicit variants `compat_input_shape_explicit`/`compat_output_shape_explicit`; legacy functions marked `#[deprecated]`; test module uses `#[allow(deprecated)]`
- **M-020** (T-D-01): `BridgeVerifier` added to `subprocess.rs` with strict/lenient/default modes; checks for success-without-output-path, success-without-package-files, missing mlpackage structure, invalid content hashes, and incomplete function descriptors; integrated into `execute_raw_payload()` with warning-level logging; `execute_and_verify()` method for strict callers; 14 new tests
- **M-032** (T-D-02): `pre_emit_verification()` and `verify_mir_spec_compliance()` added to `python/verify.py`; checks for duplicate output names, dangling input references, input/output count/name/shape mismatches; integrated into `mil_emitter.py` emit_mlprogram() and emit_multifunction() as non-blocking logging checks; 9 new Python tests
- **N-005** (T-D-03): `placement_dialect.rs` module added to `ane-ir` crate implementing Layer 2 of 5-layer placement infrastructure; `PlacementRegion` (Ane/Cpu/Flexible), `ForceAnePlacement`, `BoundaryOp` (CpuToAne/AneToCpu/Synchronize), `PlacementAnnotation`, `validate_placement_annotations()` with conflict detection; 19 new tests
- **F-FW-01** (T-D-04): `multi_ane.rs` module added to `ane-ir` crate modeling multi-ANE device enumeration, 4 firmware images per ANE instance (boot/runtime/debug/recovery), `SubTypeDescriptor` for chip-specific firmware matching, `ChainedProgram` with `InterDeviceTransfer` and `TransferMethod` (DirectDma/SharedMemory/OnChipInterconnect), `MultiAneSystem` with single_ane/mac_multi_ane factories and chain validation; 20 new tests

---

## Issue Statistics

| Domain | High | Medium | Low | Total |
|--------|------|--------|-----|-------|
| **All** | **0** | **0** | **0** | **0** |

### Status Breakdown

| Status | Count | Issues |
|--------|-------|--------|
| **Total Active** | **0** | |

### Resolved Since Round 1: 46 issues total
Round 1 (22): V-006, V-011, V-014, V-023, V-026, V-037, V-051, V-055, V-056, F-CROSS-01, F-ARCH-01, M-006, M-009, M-012, M-015, N-002, N-003, N-006, N-007, N-008, N-010, N-011, N-012
Round 2 (17): M-003, M-005, M-011, M-013, M-014, M-016, M-017, M-018, M-024, M-025, M-027, M-028, M-029, M-030, M-033, N-004, N-009
Round 3 (7): F-HAL-01, F-OPS-01, M-019, M-020, M-032, N-005, F-FW-01
