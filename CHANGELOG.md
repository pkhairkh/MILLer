# CHANGELOG.md — MILLer Compiler

## Pre-Audit Resolved Tasks (Archived from TASKS.md)

These tasks (T-01 through T-21) were completed before the TABULA RASA audit on 2026-05-03.

---

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

### T-16 · Remove or Deprecate Legacy Proto Format ✅
- **ISSUES ref**: W-11
- **Resolution**: Added `#[deprecated]` to legacy `proto` module with removal deadline.

### T-17 · Unify `MirOpCompat` with `MirOp` ✅ (lightweight)
- **ISSUES ref**: W-12, W-16
- **Resolution**: Added `From<MirOp> for MirOpCompat` conversion with exhaustive variant test. Full `ToProto` trait deferred to I-17.

### T-18 · Add `#[should_panic]` Tests ✅
- **Resolution**: Added `crates/ir/tests/should_panic.rs` and `crates/knowledge/tests/should_panic.rs`.

### T-19 · Fix A12 Hardware Limits ✅ (documented)
- **ISSUES ref**: S-12
- **Resolution**: Added doc comment noting estimated values; added runtime warning when A12 limits are used.

### T-20 · Update README Test Count ✅
- **ISSUES ref**: S-15
- **Resolution**: Updated from 440 to 637.

### T-21 · Resolve Remaining 12 Code Review Findings ✅
- **ISSUES ref**: W-03, W-06, W-09, W-13, W-14, S-01, S-02, S-03, S-05, S-06, S-13, S-14
- **Resolution**: All 12 findings addressed (see original TASKS.md for details).

---

## Pre-Audit Resolved Issues (Archived from ISSUES.md)

| Issue | Description | Resolution |
|-------|-------------|------------|
| ISSUE-001 | Mask path uses CPU-only ops | Replaced with precomputed tables + Gather |
| ISSUE-002 | Fill/FillLike survive to emission | Replaced with Const scalar + Add broadcasting |
| ISSUE-003 | Equal/LessEqual on ANE path | Eliminated by ISSUE-001 fix |
| ISSUE-006 | Hardcoded model constants in CLI | Now read from ModelConfig |
| ISSUE-007 | KV mask CPU-only path | Eliminated by ISSUE-001 fix |
| ISSUE-008 | Three mask implementations | Unified in legality_rewrite |
| ISSUE-011 | Where→Select double-rewrite | Rewrite removed; engine classification fixed |
| ISSUE-013 | for_qwen3_0_6b factory | Replaced with from_model_config() |
| ISSUE-014 | Hardcoded shape in role_mir | Derived from spec output_specs |
| ISSUE-015 | kv_cache_rewrite dead code | Deprecated |
| ISSUE-016 | RMSNorm fp16 overflow | Dynamic max-abs stabilization added |
| ISSUE-017 | QK norm not implemented | Full SLaNC implementation added |
| ISSUE-019 | bool→fp16 cast on ANE | Eliminated by ISSUE-001 fix |
| ISSUE-021 | default_engine() misclassifies ops | 8 ops moved to CPU-only |
| ISSUE-022 | Scalar constant resolution | scalar:// protocol added |

---

## TABULA RASA Audit — 2026-05-03

Full-spectrum diagnostic sweep performed. See `AUDIT.md` for complete findings.

**Key metrics:**
- 660 tests passing, 0 failures
- 3 clippy deny errors, 85 warnings
- 55 ops with engine/compat mismatch
- 9 missing constraint validators
- 2 modules with zero test coverage
- IR Cleanliness Score: 78%

**New issues filed:** I-01 through I-20 (see ISSUES.md)
**New tasks created:** T-22 through T-45 (see TASKS.md)
