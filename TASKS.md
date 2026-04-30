# MILLer — Prioritised Action List

> Ranked by **impact × urgency**. Each task references findings in `ISSUES.md`.
> Estimates assume a single experienced Rust/Python developer.

---

## Top 5 — Do These First

### T-01 · Fix FFI Validation Path Bug
- **ISSUES ref**: C-11
- **Impact**: 🔴 Production-correctness — the FFI validator rejects every valid mlpackage
- **Effort**: 15 min
- **Action**: Change `crates/coreml-ffi/src/capi.rs:377` from `Model/com.apple.CoreML/model.mlmodel` to `Data/com.apple.CoreML/model.mlmodel`. Add a test that validates a known-good mlpackage produced by the proto-direct emitter.

### T-02 · Add CI/CD Pipeline
- **ISSUES ref**: C-01
- **Impact**: 🔴 No automated quality gate — any commit can break the build without detection
- **Effort**: 2–3 h
- **Action**: Add `.github/workflows/ci.yml` running `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and `cargo build` on push/PR. Pin the Rust toolchain via `rust-toolchain.toml` (already present). Optionally add the Python `smoke_test.sh` on macOS runners.

### T-03 · Extract Shared Op Definitions Across SIR / AIR / MIR
- **ISSUES ref**: C-04, S-07, S-08, S-09
- **Impact**: 🔴 ~3 000 lines of copy-paste; every new op requires 3-file edits; drift risk
- **Effort**: 3–5 d (refactor + migrate all consumers)
- **Action**:
  1. Create `crates/ir/src/common.rs` with `Dtype`, `ComputeUnitHint`, and a parameterised `Op<N: NodeId>` enum.
  2. Make `SirOp = Op<SirNodeId>`, `AirOp = Op<AirNodeId>`, `MirOp = Op<MirNodeId>` (with MIR's `name` field handled via an extension trait or wrapper).
  3. Move `MilDtype` → `common.rs` to fix the SIR→MIR layering violation (C-06).
  4. Add a shared `trait IrNodeId` and `trait IrGraph` to enable generic algorithms.
  5. Replace `serialize.rs` boilerplate with a single generic function.
  6. Verify all 563 tests still pass after migration.

### T-04 · Break Up `linear_slice.rs` God Module
- **ISSUES ref**: C-05, W-29, W-30, S-10, S-11
- **Impact**: 🔴 1 650+ lines mixing 6+ concerns; payload types duplicate PIR types; deprecated payloads not annotated
- **Effort**: 1–2 d
- **Action**:
  1. Extract bridge payloads → `crates/ir/src/payload.rs`.
  2. Extract MIR lowering → `crates/ir/src/lowering.rs` (or move to `crates/passes/src/`).
  3. Extract shard descriptor logic → `crates/ir/src/shard.rs`.
  4. Remove deprecated family-specific payloads or annotate with `#[deprecated]`.
  5. Replace 12 hardcoded `"iOS18"` with `crate::DEFAULT_OPSET_VERSION`.
  6. Replace 6 hardcoded `seed: 42` with a named constant or parameter.

### T-05 · Move Lab Orchestration Out of CLI
- **ISSUES ref**: C-12, W-23
- **Impact**: 🔴 ~1 000 lines of orchestration logic locked inside a binary crate; `ane_lab` crate exists but is unused for logic
- **Effort**: 2–3 d
- **Action**:
  1. Create `ane_lab::run_lab_session()` that encapsulates the compile→baseline→drift→inspect→knowledge-persist loop.
  2. Refactor `cli/main.rs` to call `run_lab_session()` — it should become a thin dispatch.
  3. Split remaining `main.rs` into `commands/compile.rs`, `commands/lab.rs`, `commands/profile.rs`, etc.
  4. Move `StoreKnowledgeQuery`, `compute_task_hash`, `build_artifact_manifest` into library crates.

---

## High Priority — Address Within 2 Sprints

### T-06 · Deduplicate `mil_emitter.py`
- **ISSUES ref**: C-13, W-20, W-27
- **Effort**: 2–3 d
- **Action**:
  1. Create `python/program_builder.py` with a single `_build_mil_program(template, command)` and a registry of build strategies.
  2. Eliminate the 9× `opset_map`, `np.random.seed/save/restore`, and `_resolve_dtype` duplication.
  3. Fix `emit_decode_step` routing to actually call the stateful path.
  4. Extract shard role→op mapping into a shared config (sync with Rust `RoleMirBuilder`).

### T-07 · Derive KV Cache Dimensions from Shard Spec
- **ISSUES ref**: C-14
- **Effort**: 1 d
- **Action**: Replace hardcoded `[1, 32, 64, 128]` in `crates/passes/src/role_mir.rs:408-434` with values derived from `ShardSpec.num_heads`, `ShardSpec.head_dim`, and `ShardSpec.context_length`.

### T-08 · Centralise Workspace Dependencies
- **ISSUES ref**: C-03
- **Effort**: 1 h
- **Action**: Move `prost`, `prost-types`, `prost-build`, `safetensors`, `toml`, `tempfile`, `uuid` into `[workspace.dependencies]` and update all crate `Cargo.toml` files to use `workspace = true`.

### T-09 · Add Integration Tests
- **ISSUES ref**: C-02
- **Effort**: 3–5 d
- **Action**:
  1. Add `crates/ir/tests/pipeline.rs` — SIR→AIR→MIR→PIR full pipeline.
  2. Add `crates/bridge/tests/dispatch.rs` — Python↔Rust bridge round-trip.
  3. Add `crates/knowledge/tests/round_trip.rs` — store, query, persist, reload.
  4. Add `crates/cli/tests/cli.rs` — end-to-end CLI invocation.

### T-10 · Fix Knowledge Store Duplications and Untyped Payloads
- **ISSUES ref**: C-08, C-09, W-04, W-05
- **Effort**: 2 d
- **Action**:
  1. Merge `sanitize_id` into a single private function in `store.rs`, import from `snapshot.rs`.
  2. Unify `scopes_overlap` — single function parameterised over `KnowledgeScope` (extract scope from `KnowledgeUnit` at call site).
  3. Define typed payload structs per `KnowledgeType` instead of `HashMap<String, serde_json::Value>`.
  4. Reconcile `compute_confidence()` vs `initial_confidence()` — document when each applies or merge.

---

## Medium Priority — Address Within 4 Sprints

### T-11 · Add Tests for Core IR Types
- **ISSUES ref**: W-01, W-02
- **Effort**: 3 d
- **Action**:
  1. `mir.rs`: test that every `MirOp` variant maps to the expected engine in `default_engine()`.
  2. `mir.rs`: test `ComputeUnitHint::from_str_flexible` for all expected strings.
  3. `mir.rs`: test `MilDtype` serialisation round-trip.
  4. `cli/`, `artifacts/`, `report/`: add basic unit tests for public functions.

### T-12 · Slim Down the Bridge Crate
- **ISSUES ref**: C-10, W-07, W-08, S-04
- **Effort**: 2–3 d
- **Action**:
  1. Move shape inference (`compat_output_shape`) to `ane-passes` or a new `ane-inference` crate.
  2. Move Qwen3-specific alias map to a model-configuration file.
  3. Replace hand-rolled `f32_to_f16` with the `half` crate.
  4. Consider a derive macro or visitor pattern for `remap_compat_inputs` / `compat_input_names`.

### T-13 · Deduplicate Python Module Constants and Helpers
- **ISSUES ref**: W-17, W-18, W-19, W-26
- **Effort**: 1 d
- **Action**:
  1. Create `python/common.py` with `COMPUTE_MAP`, `_error_result()`, and coremltools lazy-import helper.
  2. Replace all 5 `compute_map` definitions with an import.
  3. Let `ImportError` propagate instead of returning `None`.

### T-14 — Separate Bridge Concerns in `bridge.py`
- **ISSUES ref**: W-21, W-22, W-25
- **Effort**: 1 d
- **Action**:
  1. `handle_host_inspect` → delegate to `model_structure.py` + `compute_plan.py`.
  2. `handle_profile` → delegate input generation to `profiler.py`.
  3. Merge `convert_stateful_milprogram` into `convert_milprogram` with a `pass_pipeline` parameter.

### T-15 — Make Emission Deterministic
- **ISSUES ref**: W-10
- **Effort**: 2 h
- **Action**: Replace `uuid::Uuid::new_v4()` in `crates/coreml-emit/src/package.rs:151-152` with a deterministic UUID (e.g. `Uuid::new_v5` using a namespace + model hash as input).

---

## Low Priority — Address When Convenient

### T-16 — Remove or Deprecate Legacy Proto Format
- **ISSUES ref**: W-11
- **Effort**: 2 h
- **Action**: Add `#[deprecated]` to legacy proto module. Add a CI check that no new code imports it. Set a removal deadline.

### T-17 — Unify `MirOpCompat` with `MirOp`
- **ISSUES ref**: W-12, W-16
- **Effort**: 3–5 d (significant refactor)
- **Action**: Explore using `MirOp` directly in proto emission via a `ToProto` trait, eliminating the compat layer. If circular deps prevent this, at least add a `From<MirOp> for MirOpCompat` conversion checked by a test.

### T-18 — Add `#[should_panic]` Tests
- **ISSUES ref**: (not a named finding)
- **Effort**: 1 d
- **Action**: Add panic-path tests for invariant violations in IR construction and knowledge store operations.

### T-19 — Fix A12 Hardware Limits
- **ISSUES ref**: S-12
- **Effort**: Research + 1 h
- **Action**: Verify M2 ANE limits independently. Update `ane_hw_limits.rs::a12()` with correct values.

### T-20 — Update README Test Count
- **ISSUES ref**: S-15
- **Effort**: 5 min
- **Action**: Replace "440" with actual count or use a CI badge.

---

## Task → Issue Cross-Reference

| Task | Issues |
|------|--------|
| T-01 | C-11 |
| T-02 | C-01 |
| T-03 | C-04, C-06, S-07, S-08, S-09 |
| T-04 | C-05, W-29, W-30, S-10, S-11 |
| T-05 | C-12, W-23 |
| T-06 | C-13, W-20, W-27 |
| T-07 | C-14 |
| T-08 | C-03 |
| T-09 | C-02 |
| T-10 | C-08, C-09, W-04, W-05 |
| T-11 | W-01, W-02 |
| T-12 | C-10, W-07, W-08, S-04 |
| T-13 | W-17, W-18, W-19, W-26 |
| T-14 | W-21, W-22, W-25 |
| T-15 | W-10 |
| T-16 | W-11 |
| T-17 | W-12, W-16 |
| T-18 | — |
| T-19 | S-12 |
| T-20 | S-15 |
