# MILLer — Prioritised Action List

> Ranked by **impact × urgency**. Each task references findings in `ISSUES.md`.
> Estimates assume a single experienced Rust/Python developer.

---

## Top 5 — Do These First ✅ ALL RESOLVED

### T-01 · Fix FFI Validation Path Bug ✅
- **ISSUES ref**: C-11
- **Resolution**: Changed path from `Model/` to `Data/` in `capi.rs`; added test.

### T-02 · Add CI/CD Pipeline ✅
- **ISSUES ref**: C-01
- **Resolution**: Added `.github/workflows/ci.yml` with check, fmt, clippy, test jobs.

### T-03 · Extract Shared Op Definitions Across SIR / AIR / MIR ✅
- **ISSUES ref**: C-04, C-06, S-07, S-08, S-09
- **Resolution**: Created `common.rs` with `MilDtype`, `ComputeUnitHint`, `IrNodeId`, `IrGraph`; genericised `serialize.rs`.

### T-04 · Break Up `linear_slice.rs` God Module ✅
- **ISSUES ref**: C-05, W-29, W-30, S-10, S-11
- **Resolution**: Extracted to `payload.rs`, `shard_desc.rs`; added `#[deprecated]`; replaced hardcoded opset/seed.

### T-05 · Move Lab Orchestration Out of CLI ✅
- **ISSUES ref**: C-12, W-23
- **Resolution**: Created `crates/lab/src/session.rs`; CLI delegates.

---

## High Priority ✅ ALL RESOLVED

### T-06 · Deduplicate `mil_emitter.py` ✅
- **ISSUES ref**: C-13, W-20, W-27
- **Resolution**: Created `python/program_builder.py`; reduced from 2,933 to 1,947 lines (−33.6%); fixed decode_step routing; extracted shared role→op mapping.

### T-07 · Derive KV Cache Dimensions from Shard Spec ✅
- **ISSUES ref**: C-14
- **Resolution**: Added `num_heads`, `head_dim`, `context_length` to `AttentionComputation`; shapes derived from spec.

### T-08 · Centralise Workspace Dependencies ✅
- **ISSUES ref**: C-03
- **Resolution**: Added `safetensors`, `prost`, `prost-types`, `prost-build`, `half` to workspace deps; replaced all hardcoded versions.

### T-09 · Add Integration Tests ✅
- **ISSUES ref**: C-02
- **Resolution**: Added `crates/ir/tests/pipeline.rs`, `crates/knowledge/tests/round_trip.rs`, `crates/cli/tests/cli.rs`.

### T-10 · Fix Knowledge Store Duplications ✅
- **ISSUES ref**: C-07, C-08, C-09, W-04, W-05
- **Resolution**: Created `util.rs` with shared functions; replaced `expect()` with `Result`; reconciled confidence values; added typed accessors.

---

## Medium Priority

### T-11 · Add Tests for Core IR Types ✅ (partial)
- **ISSUES ref**: W-01, W-02
- **Resolution**: Added exhaustive MirOp tests; CLI integration tests. Artifacts and report crate tests still pending.

### T-12 · Slim Down the Bridge Crate ✅
- **ISSUES ref**: C-10, W-07, W-08, S-04
- **Resolution**: Extracted shape inference to `shape_inference.rs`; replaced `f32_to_f16` with `half` crate; documented Qwen3-specific code and boilerplate.

### T-13 · Deduplicate Python Module Constants and Helpers ✅
- **ISSUES ref**: W-17, W-18, W-19, W-26
- **Resolution**: Created `python/common.py` with `COMPUTE_MAP`, `_error_result()`, `_ensure_coremltools()`; `ImportError` now propagates.

### T-14 · Separate Bridge Concerns in `bridge.py` ✅
- **ISSUES ref**: W-21, W-22, W-25
- **Resolution**: `handle_host_inspect` delegates to `model_structure.py`; `handle_profile` delegates to `profiler.generate_inputs()`; merged `convert_stateful_milprogram` into `convert_milprogram`.

### T-15 · Make Emission Deterministic ✅
- **ISSUES ref**: W-10
- **Resolution**: Replaced `Uuid::new_v4()` with `Uuid::new_v5()` using deterministic namespace.

---

## Low Priority

### T-16 — Remove or Deprecate Legacy Proto Format ✅
- **ISSUES ref**: W-11
- **Resolution**: Added `#[deprecated]` to legacy `proto` module with removal deadline (Sprint 60).

### T-17 — Unify `MirOpCompat` with `MirOp` ✅ (lightweight)
- **ISSUES ref**: W-12, W-16
- **Resolution**: Added `From<MirOp> for MirOpCompat` conversion with exhaustive variant test.

### T-18 — Add `#[should_panic]` Tests ✅
- **Resolution**: Added `crates/ir/tests/should_panic.rs` and `crates/knowledge/tests/should_panic.rs`.

### T-19 — Fix A12 Hardware Limits ✅ (documented)
- **ISSUES ref**: S-12
- **Resolution**: Added doc comment noting estimated values; added runtime warning when A12 limits are used.

### T-20 — Update README Test Count ✅
- **ISSUES ref**: S-15
- **Resolution**: Updated from 440 to 637.

---

## Remaining Open Issues

The following issues from ISSUES.md remain open and are not covered by completed tasks:

| Issue | Description | Suggested Action |
|-------|-------------|-----------------|
| W-03 | Knowledge Store O(n) queries | Add indexes by knowledge type/scope |
| W-06 | `eprintln!` in library code | Replace with `log` crate |
| W-09 | `WeightBinBuilder` dead code | Remove unused `total_size` first pass |
| W-13 | `FfiError` missing `std::error::Error` source chain | Implement `#[source]` with thiserror |
| W-14 | Inconsistent error types across crates | Add shared error type or conversion strategy |
| S-01 | `KnowledgeEntry` expensive clones | Use `Arc<KnowledgeUnit>` |
| S-02 | `decay_confidence` dead public API | Remove or deprecate |
| S-03 | `ConflictDetector` O(n²) | Add early termination |
| S-05 | Test writes to `/tmp/` | Use `tempfile` crate |
| S-06 | `FfiModel::Drop` is no-op | Implement on macOS |
| S-13 | Legacy `ElementWise` still in core enums | Add `#[deprecated]` |
| S-14 | Duplicate `MockKnowledge` across tests | Extract to shared test-util module |

---

## Task → Issue Cross-Reference

| Task | Issues | Status |
|------|--------|--------|
| T-01 | C-11 | ✅ |
| T-02 | C-01 | ✅ |
| T-03 | C-04, C-06, S-07, S-08, S-09 | ✅ |
| T-04 | C-05, W-29, W-30, S-10, S-11 | ✅ |
| T-05 | C-12, W-23 | ✅ |
| T-06 | C-13, W-20, W-27 | ✅ |
| T-07 | C-14 | ✅ |
| T-08 | C-03 | ✅ |
| T-09 | C-02 | ✅ |
| T-10 | C-07, C-08, C-09, W-04, W-05 | ✅ |
| T-11 | W-01, W-02 | ✅ (partial) |
| T-12 | C-10, W-07, W-08, S-04 | ✅ |
| T-13 | W-17, W-18, W-19, W-26 | ✅ |
| T-14 | W-21, W-22, W-25 | ✅ |
| T-15 | W-10 | ✅ |
| T-16 | W-11 | ✅ |
| T-17 | W-12, W-16 | ✅ (lightweight) |
| T-18 | — | ✅ |
| T-19 | S-12 | ✅ (documented) |
| T-20 | S-15 | ✅ |
