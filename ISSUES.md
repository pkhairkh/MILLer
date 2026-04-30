# MILLer — Code Review Findings

> **Review scope**: Code Quality, Separation of Concerns, Recycling (Re-usability)
> **Severity key**: 🔴 Critical · 🟡 Warning · 🟢 Suggestion · ✅ Resolved
> **Convention**: Every finding cites concrete `file:line` references.

---

## 🔴 Critical (14 findings — 12 resolved)

### C-01 — No CI/CD Pipeline ✅
- **Category**: Code Quality
- **File**: `.github/workflows/` (does not exist)
- No GitHub Actions, no Makefile, no CI script. The `rust-toolchain.toml` pins Rust 1.95 and `clippy.toml` sets a cognitive complexity threshold, but neither is enforced mechanically. Without CI there is no guarantee tests pass, clippy is clean, or rustfmt is respected.
- **Resolution**: Added `.github/workflows/ci.yml` with check, fmt, clippy, and test jobs (T-02).

### C-02 — Zero Integration Tests ✅
- **Category**: Code Quality
- **File**: All `crates/*/tests/` directories (none exist)
- All 563 `#[test]` functions are inline `#[cfg(test)]` unit tests. No integration tests verify cross-crate behaviour — SIR→AIR→MIR→PIR pipeline, bridge dispatch, knowledge-store round-trip, or CLI end-to-end flows. `scripts/smoke_test.sh` is a partial substitute but requires Python/coremltools and is not gated by CI.
- **Resolution**: Added integration tests in `crates/ir/tests/pipeline.rs`, `crates/knowledge/tests/round_trip.rs`, `crates/cli/tests/cli.rs` (T-09).

### C-03 — Workspace Dependency Versions Not Centralised ✅
- **Category**: Recycling
- **Files & Lines**: `crates/ir/Cargo.toml:10`, `crates/bridge/Cargo.toml:16-17`, `crates/knowledge/Cargo.toml:15`, `crates/coreml-emit/Cargo.toml:14`, `crates/coreml-proto/Cargo.toml:8-9,14`, `crates/coreml-ffi/Cargo.toml:12`
- `prost` appears in 3 crates with hardcoded `"0.12"`. `uuid` is declared in workspace but `ane-coreml-emit` re-declares it with a hardcoded version. `safetensors = "0.4"` is not in `[workspace.dependencies]` at all. Upgrading any of these requires editing 3+ files in lockstep.
- **Resolution**: Added `safetensors`, `prost`, `prost-types`, `prost-build` to workspace dependencies; replaced all hardcoded versions with `workspace = true` (T-08).

### C-04 — Massive Op Enum Duplication Across SIR / AIR / MIR (~3 000 lines of copy-paste) ✅
- **Category**: Recycling / Code Quality
- **Files & Lines**: `crates/ir/src/sir.rs:21-911`, `crates/ir/src/air.rs:12-876`, `crates/ir/src/mir.rs:29-1046`
- `SirOp`, `AirOp`, `MirOp` contain ~70+ structurally identical variants each, differing only in (1) the `NodeId` type and (2) MIR's `MIL` prefix + `name` field.
- **Resolution**: Created `crates/ir/src/common.rs` with shared types; added `IrNodeId` trait and `IrGraph` trait; genericised `serialize.rs` (T-03).

### C-05 — `linear_slice.rs` Is a 1 650+ Line God Module ✅
- **Category**: Separation of Concerns
- **File**: `crates/ir/src/linear_slice.rs` (entire file)
- Mixes 6+ distinct concerns: SIR graph construction, MIR graph lowering, 6 bridge-payload struct definitions, PIR graph construction, shard-descriptor logic, and bridge-payload construction.
- **Resolution**: Extracted payloads to `payload.rs`, shard types to `shard_desc.rs`; added `#[deprecated]` to family-specific payloads; replaced hardcoded opset/seed (T-04).

### C-06 — SIR Depends on MIR — Layering Violation ✅
- **Category**: Separation of Concerns
- **File**: `crates/ir/src/sir.rs:7, 80-81, 340-341, 642-643`
- SIR (highest-level IR) imports `MilDtype` from MIR (lowest-level IR).
- **Resolution**: Moved `MilDtype` to `common.rs`; SIR now imports from `common` instead of `mir` (T-03).

### C-07 — `expect()` on HashMap Lookup After Insert — Library Code Can Panic ✅
- **Category**: Code Quality
- **File**: `crates/knowledge/src/store.rs:345`
- **Resolution**: Replaced `expect()` with `ok_or_else()` returning proper `Result` (T-10).

### C-08 — Duplicated `sanitize_id` Function With Identical Implementation ✅
- **Category**: Recycling
- **Files**: `crates/knowledge/src/store.rs:504`, `crates/knowledge/src/snapshot.rs:200`
- **Resolution**: Extracted to shared `crates/knowledge/src/util.rs` (T-10).

### C-09 — Duplicated `scopes_overlap` With Incompatible Signatures ✅
- **Category**: Recycling / Separation of Concerns
- **Files**: `crates/knowledge/src/store.rs:459-468`, `crates/knowledge/src/conflict.rs:180-190`
- **Resolution**: Canonical version in `util.rs` taking `&KnowledgeScope`; `conflict.rs` delegates (T-10).

### C-10 — Bridge Leaks Compiler Logic ✅
- **Category**: Separation of Concerns
- **File**: `crates/bridge/src/mir_to_compat.rs` (entire file)
- **Resolution**: Extracted shape inference to `crates/bridge/src/shape_inference.rs`; added Qwen3-specific doc comment to `build_input_alias_map`; added boilerplate warning to `remap_compat_inputs` (T-12).

### C-11 — FFI Validation Uses Wrong Directory Path for `model.mlmodel` ✅
- **Category**: Code Quality
- **Files**: `crates/coreml-ffi/src/capi.rs:377`, `crates/bridge/src/proto_direct.rs:254`
- **Resolution**: Changed path from `Model/` to `Data/` (T-01).

### C-12 — CLI Implements Lab Orchestration (~1 000 lines) ✅
- **Category**: Separation of Concerns
- **File**: `crates/cli/src/main.rs:3108-3637`
- **Resolution**: Moved orchestration to `crates/lab/src/session.rs`; CLI now delegates (T-05).

### C-13 — `mil_emitter.py` Is a 2 939-Line Monolith With 9× Boilerplate Duplication ✅
- **Category**: Code Quality / Recycling
- **File**: `python/mil_emitter.py:1-2939`
- **Resolution**: Created `python/program_builder.py` with shared helpers; eliminated 9× `opset_map` duplication; reduced from 2,933 to 1,947 lines (−33.6%) (T-06).

### C-14 — Hardcoded KV Cache Dimensions in `RoleMirBuilder` ✅
- **Category**: Code Quality
- **File**: `crates/passes/src/role_mir.rs:408-416, 428-434`
- **Resolution**: Added `num_heads`, `head_dim`, `context_length` fields to `ShardOpProfile::AttentionComputation`; KV cache shape now derived from spec (T-07).

---

## 🟡 Warning (30 findings — 20 resolved)

### W-01 — Core IR Types Have Zero Tests ✅
- **Category**: Code Quality
- **Resolution**: Added exhaustive `MirOp` engine-assignment tests in `crates/ir/tests/` (T-11).

### W-02 — Three Crates Have Zero Tests (cli, artifacts, report) ✅ (partial)
- **Category**: Code Quality
- **Resolution**: Added CLI integration tests in `crates/cli/tests/cli.rs` (T-11). Artifacts and report tests still pending.

### W-03 — Knowledge Store Is Effectively a Key-Value Store
- **Category**: Separation of Concerns
- **Files**: `crates/knowledge/src/store.rs`, `crates/knowledge/src/query.rs`
- Loads all entries into `HashMap<String, KnowledgeEntry>` and queries by iterating/filtering. No indexes by knowledge type, scope, or device class.

### W-04 — `claims_contradict` / `claims_agree` Use Untyped JSON Payload Access ✅
- **Category**: Code Quality
- **Resolution**: Added typed accessor helpers (`payload_ane_legal`, `payload_op_pattern`, etc.) in `util.rs` (T-10).

### W-05 — Two Confidence Functions With Different Base Values ✅
- **Category**: Code Quality
- **Resolution**: Removed `compute_confidence` from `confidence.rs`; `initial_confidence` in `update.rs` is now sole authoritative source (T-10).

### W-06 — `eprintln!` Used for Logging in Library Code
- **Category**: Code Quality
- **Files**: `crates/knowledge/src/store.rs:266`, `crates/knowledge/src/snapshot.rs:101,103,131`, etc.

### W-07 — `build_input_alias_map` Hardcodes Qwen3-Specific Aliases ✅ (documented)
- **Category**: Separation of Concerns
- **Resolution**: Added prominent doc comment noting Qwen3-specific nature with suggestions for generalization (T-12).

### W-08 — Hand-Rolled `f32_to_f16` Instead of Using the `half` Crate ✅
- **Category**: Recycling
- **Resolution**: Replaced with `half::f16::from_f32()` which correctly handles subnormals and NaN payloads (T-12).

### W-09 — `WeightBinBuilder` Does Double-Pass With Redundant Offset Computation
- **Category**: Code Quality
- **File**: `crates/coreml-emit/src/weights.rs:278-322`

### W-10 — `MlPackageWriter::build_manifest` Generates Non-Deterministic UUIDs ✅
- **Category**: Code Quality
- **Resolution**: Replaced `uuid::Uuid::new_v4()` with `uuid::Uuid::new_v5()` using deterministic namespace (T-15).

### W-11 — `coreml-proto` Has Both Legacy and Apple-Compatible Proto Definitions ✅
- **Category**: Recycling / Separation of Concerns
- **Resolution**: Added `#[deprecated]` to legacy `proto` module with removal plan (T-16).

### W-12 — `mir_compat` Module Duplicates MIR Types ✅ (lightweight)
- **Category**: Recycling
- **Resolution**: Added `From<ane_ir::mir::MirOp> for MirOpCompat` conversion with test ensuring all variants are covered (T-17).

### W-13 — `FfiError` Doesn't Implement `std::error::Error` Source Chain
- **Category**: Code Quality
- **File**: `crates/coreml-ffi/src/error.rs`

### W-14 — Inconsistent Error Types Across Crates
- **Category**: Code Quality / Recycling
- **Files**: All crates

### W-15 — Inconsistent Validation Logic — Bridge vs FFI ✅ (partial)
- **Category**: Separation of Concerns
- **Resolution**: FFI path bug fixed (C-11/T-01); different error granularity remains.

### W-16 — `MirOpCompat` Has 40+ Variants With No Trait-Based Dispatch ✅ (documented)
- **Category**: Recycling
- **Resolution**: Added doc comment recommending derive macro or visitor pattern; added `From<MirOp>` conversion (T-17).

### W-17 — `compute_map` Dict Duplicated 5× Across Python Modules ✅
- **Category**: Recycling
- **Resolution**: Centralised in `python/common.py` as `COMPUTE_MAP` (T-13).

### W-18 — `_error_result()` Defined Independently in `bridge.py` and `mil_emitter.py` ✅
- **Category**: Recycling
- **Resolution**: Centralised in `python/common.py` (T-13).

### W-19 — Module-Level Mutable Global State in Python Modules ✅
- **Category**: Code Quality
- **Resolution**: Centralised `_ensure_coremltools()` in `python/common.py`; per-module globals replaced (T-13).

### W-20 — `emit_decode_step` Misrouted to Stateless Path ✅
- **Category**: Code Quality
- **Resolution**: `emit_decode_step` now routes to stateful path (T-06).

### W-21 — `handle_host_inspect` Reimplements What `model_structure.py` Already Does ✅
- **Category**: Separation of Concerns
- **Resolution**: Refactored to delegate to `model_structure.py` (~170→~80 lines) (T-14).

### W-22 — `handle_profile` Duplicates `profiler.py` Input-Generation Logic ✅
- **Category**: Separation of Concerns
- **Resolution**: Delegated to `profiler.generate_inputs()` (T-14).

### W-23 — CLI `main.rs` Is a 5 627-Line God File ✅
- **Category**: Code Quality
- **Resolution**: Lab orchestration extracted to lab crate (T-05); significant line reduction.

### W-24 — `ShardPlanPass.run()` Hardcodes Shape Placeholders ✅ (partial)
- **Category**: Code Quality
- **Resolution**: KV cache shapes now derived from `AttentionComputation` fields (T-07); some placeholder shapes remain.

### W-25 — `convert_stateful_milprogram` Duplicates `convert_milprogram` ✅
- **Category**: Recycling
- **Resolution**: Merged into `convert_milprogram` with `pass_pipeline` parameter (T-14).

### W-26 — Python Modules Silently Swallow `ImportError` ✅
- **Category**: Code Quality
- **Resolution**: `_import_coremltools()` now raises `ImportError` instead of returning `(None, None, None, None)` (T-13).

### W-27 — `emit_shard_decode_step` Duplicates Rust `RoleMirBuilder` Mapping ✅
- **Category**: Code Quality
- **Resolution**: Extracted shared `SHARD_ROLE_OP_MAP` in `program_builder.py` with doc comment noting Rust source of truth (T-06).

### W-28 — `primary_op_pattern` Always Returns `"mb.matmul"` ✅
- **Category**: Code Quality
- **Resolution**: Now returns `"mb.scaled_dot_product_attention"` for attention shards (T-07).

### W-29 — Duplicate `FunctionDescriptor`/`TensorDescriptor` vs PIR's `FunctionEntry`/`TensorSpec` ✅
- **Category**: Recycling
- **Resolution**: Extracted to `payload.rs` with proper documentation (T-04).

### W-30 — Deprecated Family-Specific Payloads Still Present Without `#[deprecated]` ✅
- **Category**: Recycling / Separation of Concerns
- **Resolution**: Added `#[deprecated]` attributes (T-04).

---

## 🟢 Suggestion (15 findings — 5 resolved)

### S-01 — `KnowledgeEntry` Carries `KnowledgeUnit` Inline Rather Than by Reference
- **Category**: Code Quality
- **File**: `crates/knowledge/src/store.rs:28-44`

### S-02 — `decay_confidence` Defined but Never Called
- **Category**: Code Quality
- **File**: `crates/knowledge/src/confidence.rs:43-45`

### S-03 — `ConflictDetector` Always Runs O(n²) — No Early Termination
- **Category**: Code Quality
- **File**: `crates/knowledge/src/conflict.rs:77-103`

### S-04 — `remap_compat_inputs` Is 180+ Lines of Boilerplate ✅ (documented)
- **Category**: Recycling
- **Resolution**: Added doc comment recommending derive macro or visitor pattern (T-12).

### S-05 — Test in `package.rs` Writes to `/tmp/` Directly
- **Category**: Code Quality

### S-06 — `FfiModel::Drop` Implementation Is a No-Op Comment
- **Category**: Code Quality

### S-07 — No Shared Trait Abstraction for NodeId Types ✅
- **Category**: Recycling
- **Resolution**: Added `IrNodeId` trait in `common.rs` (T-03).

### S-08 — No Shared Graph Trait — Identical Structure 4× Over ✅
- **Category**: Recycling
- **Resolution**: Added `IrGraph` trait in `common.rs` (T-03).

### S-09 — `serialize.rs` Is Pure Copy-Paste Boilerplate ✅
- **Category**: Recycling
- **Resolution**: Replaced with generic `serialize_graph`/`deserialize_graph` (T-03).

### S-10 — `DEFAULT_OPSET_VERSION` Defined but `"iOS18"` Hardcoded 12 Times ✅
- **Category**: Code Quality
- **Resolution**: Replaced with `crate::DEFAULT_OPSET_VERSION` (T-04).

### S-11 — Magic Number `seed: 42` Hardcoded in 6 Payload Constructors ✅
- **Category**: Code Quality
- **Resolution**: Replaced with `DEFAULT_SEED` constant (T-04).

### S-12 — `AneHwLimits::a12()` Is an Unverified Copy of A11 Values ✅ (documented)
- **Category**: Code Quality
- **Resolution**: Added doc comment noting values are estimated; added runtime warning when A12 limits are used (T-19).

### S-13 — `ElementWise` / `ElementWiseOp` Marked Legacy but Still in Core Enums
- **Category**: Code Quality / Separation of Concerns

### S-14 — Duplicate `MockKnowledge` Implementations Across Pass Tests
- **Category**: Recycling

### S-15 — README Test Count Is Stale ✅
- **Category**: Code Quality
- **Resolution**: Updated from 440 to 637 (T-20).

---

## What the Project Does Well

1. **Partial workspace dependency centralisation** — `serde`, `anyhow`, `serde_json`, `sha2`, `chrono`, `clap`, `zip`, `rmp-serde` all use `workspace = true` consistently. ✅ Now fully centralised.
2. **`clippy.toml` cognitive complexity threshold (50)** and `rustfmt.toml` are good practices.
3. **Thorough constraint-validation tests** — `op_constraints`, `dtype_constraints`, `ane_layout`, `ane_hw_limits`, `placement_validate` exercise both `.is_ok()` and `.is_err()` paths.
4. **Honest `smoke_test.sh`** — reports limitations rather than faking success.
5. **Good doc comments throughout IR types** — variant-level documentation is present and helpful.
6. **`MirOpCompat` 27-variant emission path** — comprehensive coverage of ANE-compatible ops.

---

## Severity Summary

| Severity | Total | Resolved | Open |
|----------|-------|----------|------|
| 🔴 Critical | 14 | 14 | 0 |
| 🟡 Warning | 30 | 20 | 10 |
| 🟢 Suggestion | 15 | 5 | 10 |
| **Total** | **59** | **39** | **20** |
