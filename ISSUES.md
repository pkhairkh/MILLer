# MILLer — Code Review Findings

> **Review scope**: Code Quality, Separation of Concerns, Recycling (Re-usability)
> **Severity key**: 🔴 Critical · 🟡 Warning · 🟢 Suggestion
> **Convention**: Every finding cites concrete `file:line` references.

---

## 🔴 Critical (14 findings)

### C-01 — No CI/CD Pipeline
- **Category**: Code Quality
- **File**: `.github/workflows/` (does not exist)
- No GitHub Actions, no Makefile, no CI script. The `rust-toolchain.toml` pins Rust 1.95 and `clippy.toml` sets a cognitive complexity threshold, but neither is enforced mechanically. Without CI there is no guarantee tests pass, clippy is clean, or rustfmt is respected.

### C-02 — Zero Integration Tests
- **Category**: Code Quality
- **File**: All `crates/*/tests/` directories (none exist)
- All 563 `#[test]` functions are inline `#[cfg(test)]` unit tests. No integration tests verify cross-crate behaviour — SIR→AIR→MIR→PIR pipeline, bridge dispatch, knowledge-store round-trip, or CLI end-to-end flows. `scripts/smoke_test.sh` is a partial substitute but requires Python/coremltools and is not gated by CI.

### C-03 — Workspace Dependency Versions Not Centralised
- **Category**: Recycling
- **Files & Lines**: `crates/ir/Cargo.toml:10`, `crates/bridge/Cargo.toml:16-17`, `crates/knowledge/Cargo.toml:15`, `crates/coreml-emit/Cargo.toml:14`, `crates/coreml-proto/Cargo.toml:8-9,14`, `crates/coreml-ffi/Cargo.toml:12`
- `prost` appears in 3 crates with hardcoded `"0.12"`. `uuid` is declared in workspace but `ane-coreml-emit` re-declares it with a hardcoded version. `safetensors = "0.4"` is not in `[workspace.dependencies]` at all. Upgrading any of these requires editing 3+ files in lockstep.

### C-04 — Massive Op Enum Duplication Across SIR / AIR / MIR (~3 000 lines of copy-paste)
- **Category**: Recycling / Code Quality
- **Files & Lines**: `crates/ir/src/sir.rs:21-911`, `crates/ir/src/air.rs:12-876`, `crates/ir/src/mir.rs:29-1046`
- `SirOp`, `AirOp`, `MirOp` contain ~70+ structurally identical variants each, differing only in (1) the `NodeId` type and (2) MIR's `MIL` prefix + `name` field. Example:
  ```rust
  // sir.rs:115-118
  Add { x: SirNodeId, y: SirNodeId },
  // air.rs:65-68
  Add { x: AirNodeId, y: AirNodeId },
  // mir.rs:79-83
  MILAdd { name: String, x: MirNodeId, y: MirNodeId },
  ```
  The reduction ops are worse — 11 variants each, copy-pasted 3×. Adding a new op requires editing three files; a missed edit silently drifts.

### C-05 — `linear_slice.rs` Is a 1 650+ Line God Module
- **Category**: Separation of Concerns
- **File**: `crates/ir/src/linear_slice.rs` (entire file)
- Mixes 6+ distinct concerns: SIR graph construction, MIR graph lowering, 6 bridge-payload struct definitions, PIR graph construction, shard-descriptor logic, and bridge-payload construction. The module docstring says "Linear Projection Slice" but it now handles LUT, decode-step, MLP, attention, and sharded pipelines.

### C-06 — SIR Depends on MIR — Layering Violation
- **Category**: Separation of Concerns
- **File**: `crates/ir/src/sir.rs:7, 80-81, 340-341, 642-643`
- SIR (highest-level IR) imports `MilDtype` from MIR (lowest-level IR): `use super::mir::MilDtype`. This creates a downward dependency in the abstraction hierarchy. A comment at line 913 acknowledges this was a deliberate choice but does not resolve the architectural problem.

### C-07 — `expect()` on HashMap Lookup After Insert — Library Code Can Panic
- **Category**: Code Quality
- **File**: `crates/knowledge/src/store.rs:345`
- ```rust
  let entry = self.index.get(&id).expect("entry must exist after insertion");
  ```
  Library crates must never panic on logic errors — they should return `Result`.

### C-08 — Duplicated `sanitize_id` Function With Identical Implementation
- **Category**: Recycling
- **Files**: `crates/knowledge/src/store.rs:504`, `crates/knowledge/src/snapshot.rs:200`
- Exact same `sanitize_id()` function defined independently in two files. DRY violation with divergence risk.

### C-09 — Duplicated `scopes_overlap` With Incompatible Signatures
- **Category**: Recycling / Separation of Concerns
- **Files**: `crates/knowledge/src/store.rs:459-468`, `crates/knowledge/src/conflict.rs:180-190`
- Two `scopes_overlap()` functions with different type signatures but near-identical logic. The `conflict.rs` version includes an "unknown" wildcard for devices/OS but not for opsets; the `store.rs` version does neither. The same two entries could be considered overlapping by one function but not the other.

### C-10 — Bridge Leaks Compiler Logic — `mir_to_compat.rs` Is 1 000+ Lines of Shape Inference, Alias Mapping, and IR Conversion
- **Category**: Separation of Concerns
- **File**: `crates/bridge/src/mir_to_compat.rs` (entire file)
- The bridge should be a thin dispatcher. Instead it contains: (1) shape inference (`compat_output_shape` lines 462-684, 220 LOC), (2) input alias mapping (`build_input_alias_map` lines 687-745) with hardcoded Qwen3-specific patterns, (3) input remapping (`remap_compat_inputs` lines 761-940+, 180 LOC of boilerplate), (4) weight materialisation (lines 143-211).

### C-11 — FFI Validation Uses Wrong Directory Path for `model.mlmodel`
- **Category**: Code Quality
- **Files**: `crates/coreml-ffi/src/capi.rs:377`, `crates/bridge/src/proto_direct.rs:254`
- ```rust
  // capi.rs:377 — WRONG path
  let mlmodel_path = pkg_path.join("Model/com.apple.CoreML/model.mlmodel");
  // proto_direct.rs:254 — CORRECT path
  let model_path = pkg_path.join("Data/com.apple.CoreML/model.mlmodel");
  ```
  The FFI validator will **always reject valid mlpackages** produced by the proto-direct emitter.

### C-12 — CLI Implements Lab Orchestration (~1 000 lines)
- **Category**: Separation of Concerns
- **File**: `crates/cli/src/main.rs:3108-3637` (`run_lab`) and `:3637-4062` (`run_lab_loop`)
- The `ane_lab` crate exists but is used only for types (`LabRun`, `LabRunBuilder`), not orchestration. All compilation, baseline computation, drift detection, and knowledge-store persistence logic lives in `main.rs`.

### C-13 — `mil_emitter.py` Is a 2 939-Line Monolith With 9× Boilerplate Duplication
- **Category**: Code Quality / Recycling
- **File**: `python/mil_emitter.py:1-2939`
- 9 `build_*_program()` functions and 12 `emit_*()` functions follow an identical pattern. `opset_map` is duplicated 9 times (lines 97, 300, 489, 727, 1025, 1481, 1663, 2175, 2596). A single `_build_mil_program(template, command)` with a strategy/registry pattern would eliminate ~1 500 lines.

### C-14 — Hardcoded KV Cache Dimensions in `RoleMirBuilder`
- **Category**: Code Quality
- **File**: `crates/passes/src/role_mir.rs:408-416, 428-434`
- ```rust
  // TODO: Derive KV cache shape from shard spec instead of hardcoding.
  shape: vec![1, 32, 64, 128],
  ```
  The shard pipeline is NOT generalisable to different model topologies despite documentation claiming "genuinely different MIR graphs" and graph-driven partitioning.

---

## 🟡 Warning (30 findings)

### W-01 — Core IR Types Have Zero Tests
- **Category**: Code Quality
- **Files**: `crates/ir/src/mir.rs`, `crates/ir/src/air.rs`, `crates/ir/src/sir.rs`, `crates/ir/src/kir.rs`, `crates/ir/src/serialize.rs`, `crates/ir/src/prof_ir.rs`
- `MirOp` has 167 variants with a 250-line `default_engine()` match arm. A single missed variant silently produces wrong engine assignments. No test verifies that every `MirOp` variant maps to the expected engine.

### W-02 — Three Crates Have Zero Tests (cli, artifacts, report)
- **Category**: Code Quality
- **Files**: `crates/cli/src/`, `crates/artifacts/src/`, `crates/report/src/`
- The CLI (780+ lines), artifacts (manifest generation, hashing, packaging), and report (JSON + markdown) crates have no tests at all.

### W-03 — Knowledge Store Is Effectively a Key-Value Store
- **Category**: Separation of Concerns
- **Files**: `crates/knowledge/src/store.rs`, `crates/knowledge/src/query.rs`
- Loads all entries into `HashMap<String, KnowledgeEntry>` and queries by iterating/filtering. No indexes by knowledge type, scope, or device class. O(n) queries for every lookup. `KnowledgeQueryable` trait is the only query interface and does linear scans.

### W-04 — `claims_contradict` / `claims_agree` Use Untyped JSON Payload Access
- **Category**: Code Quality
- **Files**: `crates/knowledge/src/store.rs:476-501`, `crates/knowledge/src/transfer.rs:143-155`
- Digs into `KnowledgeUnit.payload` (`HashMap<String, serde_json::Value>`) using string keys like `"ane_legal"`. No compile-time guarantee that keys exist. `claims_agree` in `transfer.rs` defaults to `true` for unknown types — any non-LegalityRule knowledge is assumed to agree with itself.

### W-05 — Two Confidence Functions With Different Base Values for the Same Evidence Sources
- **Category**: Code Quality
- **Files**: `crates/knowledge/src/confidence.rs:9-29`, `crates/knowledge/src/update.rs:90-106`
- `compute_confidence()` and `initial_confidence()` assign different base confidence values: e.g. `SyntheticRun` → 0.6 vs 0.2, `RealModelRun` → 0.9 vs 0.35. Doc comment says "single observations never start above 0.5" but `LoadFailure` starts at 0.8.

### W-06 — `eprintln!` Used for Logging in Library Code
- **Category**: Code Quality
- **Files**: `crates/knowledge/src/store.rs:266`, `crates/knowledge/src/snapshot.rs:101,103,131`, `crates/knowledge/src/shard_template.rs:227-229,245-247`, `crates/bridge/src/mir_to_compat.rs:203-206,237-239,313-315`
- Makes it impossible for consumers to control logging output. Pollutes stderr in library usage.

### W-07 — `build_input_alias_map` Hardcodes Qwen3-Specific Aliases
- **Category**: Separation of Concerns
- **File**: `crates/bridge/src/mir_to_compat.rs:687-745`
- Contains `.self_attn.q_proj.weight`, `.mlp.up_proj.weight`, and specific name patterns (`"mlp_silu"`, `"attn_qk"`). Bridge is non-reusable for any model architecture that isn't Qwen3.

### W-08 — Hand-Rolled `f32_to_f16` Instead of Using the `half` Crate
- **Category**: Recycling
- **File**: `crates/bridge/src/safetensors_resolver.rs:302-342`
- 40 lines of manual bit manipulation. Subnormal f32 values silently become zero (data loss). No NaN payload preservation. The `half` crate handles all edge cases correctly in 3 KB.

### W-09 — `WeightBinBuilder` Does Double-Pass With Redundant Offset Computation
- **Category**: Code Quality
- **File**: `crates/coreml-emit/src/weights.rs:278-322`
- First pass calculates `total_size` but it is never used — dead code. Only `current_pos` from the second pass is returned.

### W-10 — `MlPackageWriter::build_manifest` Generates Non-Deterministic UUIDs
- **Category**: Code Quality
- **File**: `crates/coreml-emit/src/package.rs:151-152`
- `uuid::Uuid::new_v4()` produces random UUIDs. The crate's documentation claims "deterministic emission" (`lib.rs:26`). Two successive builds of the same model produce different `Manifest.json` files.

### W-11 — `coreml-proto` Has Both Legacy and Apple-Compatible Proto Definitions
- **Category**: Recycling / Separation of Concerns
- **Files**: `crates/coreml-proto/build.rs:17-32`, `crates/coreml-proto/src/lib.rs:52-54`
- Compiles TWO proto formats. The legacy one is described as "kept for backward compatibility with existing tests" but is a full public module with no deprecation warnings and no removal plan. Doubles proto compilation time and public API surface.

### W-12 — `mir_compat` Module in `coreml-proto` Duplicates MIR Types
- **Category**: Recycling
- **File**: `crates/coreml-proto/src/lib.rs:303-827`
- `MirOpCompat`, `MilDtypeCompat`, `ComputeUnitHintCompat`, etc. are near-copies of types in `ane-ir::mir`. Adding a new MIR op requires changes in 3 places: `ane-ir`, `mir_compat`, and `mir_to_compat`. The `Unsupported` variant is a catch-all escape hatch that undermines type safety.

### W-13 — `FfiError` Doesn't Implement `std::error::Error` Source Chain
- **Category**: Code Quality
- **File**: `crates/coreml-ffi/src/error.rs`
- Variants with nested reasons use `String` instead of `Box<dyn Error>`. Breaks error chain inspection and prevents using `#[source]` with `thiserror`/`snafu`.

### W-14 — Inconsistent Error Types Across Crates
- **Category**: Code Quality / Recycling
- **Files**: All crates
- `knowledge`/`bridge`/`coreml-emit` use `anyhow::Result`. `coreml-ffi` uses custom `FfiError`. No shared error type or conversion strategy. `FfiError` cannot be converted from `anyhow::Error`, making composition difficult.

### W-15 — Inconsistent Validation Logic — Bridge vs FFI Validate the Same Thing Differently
- **Category**: Separation of Concerns
- **Files**: `crates/bridge/src/proto_direct.rs:196-300`, `crates/coreml-ffi/src/capi.rs:339-403`
- Two functions validate mlpackage structure with different paths (one wrong) and different error granularity.

### W-16 — `MirOpCompat` Has 40+ Variants With No Trait-Based Dispatch
- **Category**: Recycling
- **Files**: `crates/coreml-proto/src/lib.rs:326-786`, `crates/bridge/src/mir_to_compat.rs`
- Every operation (`compat_input_names`, `remap_compat_inputs`, `mir_op_to_proto_op`) requires a full match arm across 40+ variants. No `OpKind` trait or visitor pattern. Every new variant requires touching 3+ functions.

### W-17 — `compute_map` Dict Duplicated 5× Across Python Modules
- **Category**: Recycling
- **Files**: `python/bridge.py:485`, `python/profiler.py:41`, `python/compute_plan.py:60,148`, `python/converter.py:63`
- Same `{"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, ...}` dictionary defined independently in 5 locations.

### W-18 — `_error_result()` Defined Independently in `bridge.py` and `mil_emitter.py`
- **Category**: Recycling
- **Files**: `python/bridge.py:1024`, `python/mil_emitter.py:2912`
- Same helper with slightly different result dict shapes.

### W-19 — Module-Level Mutable Global State in Python Modules
- **Category**: Code Quality
- **Files**: `python/converter.py:23`, `python/profiler.py:9`, `python/compute_plan.py:15`, `python/palettize.py:10`
- Pattern: `ct = None; def _ensure_coremltools(): global ct; ...`. If `_ensure_coremltools()` fails partway, `ct` remains `None` but subsequent calls won't retry.

### W-20 — `emit_decode_step` Misrouted to Stateless Path
- **Category**: Code Quality
- **File**: `python/mil_emitter.py:569-674`
- `emit_decode_step` routes to the stateless build despite being documented as the stateful path. A TODO acknowledges this: `// TODO: Route emit_decode_step to stateful path as documented.`

### W-21 — `handle_host_inspect` Reimplements What `model_structure.py` and `compute_plan.py` Already Do
- **Category**: Separation of Concerns
- **File**: `python/bridge.py:273-445`
- ~170 lines of directory walking, Manifest.json parsing, model loading. Hardcodes `"dtype": "fp16"` as "best effort" while `model_structure.py` extracts actual dtype.

### W-22 — `handle_profile` Duplicates `profiler.py` Input-Generation Logic
- **Category**: Separation of Concerns
- **File**: `python/bridge.py:448-572`
- Generates random inputs and does timing format conversion that should be inside `profiler.py`. Bridge should only dispatch.

### W-23 — CLI `main.rs` Is a 5 627-Line God File
- **Category**: Code Quality
- **File**: `crates/cli/src/main.rs:1-5627`
- 14 `run_*` functions plus `StoreKnowledgeQuery`, `compute_task_hash`, etc. `run_compile_full_sharded` alone is ~860 lines (1784-2642).

### W-24 — `ShardPlanPass.run()` Hardcodes Shape Placeholders Instead of Deriving from SIR
- **Category**: Code Quality
- **File**: `crates/passes/src/shard_plan.rs:361,391,409,415,443`
- Multiple `TensorSpec` entries use hardcoded shapes with comments saying "derived from graph" but values are hardcoded.

### W-25 — `convert_stateful_milprogram` Duplicates `convert_milprogram` With Minimal Changes
- **Category**: Recycling
- **File**: `python/converter.py:137-209`
- Copies entire kwargs construction plus adds pass-pipeline manipulation. Should be a parameter or builder pattern.

### W-26 — Python Modules Silently Swallow `ImportError`
- **Category**: Code Quality
- **File**: `python/mil_emitter.py:55-63`
- `_import_coremltools()` returns `(None, None, None, None)` on failure. Callers check `if ct is None`. Typos or partial installations silently produce no-ops.

### W-27 — `emit_shard_decode_step` Duplicates Rust `RoleMirBuilder` Role→Op Mapping in Python
- **Category**: Code Quality
- **File**: `python/mil_emitter.py:950-1187`
- Hardcodes `Entry→Reshape`, `Interior→GELU`, `Exit→LayerNorm` in Python, duplicating `crates/passes/src/role_mir.rs`.

### W-28 — `primary_op_pattern` Always Returns `"mb.matmul"`
- **Category**: Code Quality
- **File**: `crates/passes/src/shard_plan.rs:158-172`
- All three match arms return the same string. Function gives a false impression of nuanced analysis.

### W-29 — Duplicate `FunctionDescriptor`/`TensorDescriptor` vs PIR's `FunctionEntry`/`TensorSpec`
- **Category**: Recycling
- **Files**: `crates/ir/src/linear_slice.rs:189-203`, `crates/ir/src/pir.rs:68-86`
- Structurally identical types, only names differ.

### W-30 — Deprecated Family-Specific Payloads Still Present Without `#[deprecated]`
- **Category**: Recycling / Separation of Concerns
- **File**: `crates/ir/src/linear_slice.rs:636+`
- Docstring says "deprecated" but no `#[deprecated]` attribute. ~500 lines of dead/duplicate code.

---

## 🟢 Suggestion (15 findings)

### S-01 — `KnowledgeEntry` Carries `KnowledgeUnit` Inline Rather Than by Reference
- **Category**: Code Quality
- **File**: `crates/knowledge/src/store.rs:28-44`
- Cloning entries is expensive due to `HashMap<String, serde_json::Value>` payload. Consider `Arc<KnowledgeUnit>`.

### S-02 — `decay_confidence` Defined but Never Called
- **Category**: Code Quality
- **File**: `crates/knowledge/src/confidence.rs:43-45`
- Public function with no callers. Dead public API.

### S-03 — `ConflictDetector` Always Runs O(n²) — No Early Termination
- **Category**: Code Quality
- **File**: `crates/knowledge/src/conflict.rs:77-103`

### S-04 — `remap_compat_inputs` Is 180+ Lines of Boilerplate
- **Category**: Recycling
- **File**: `crates/bridge/src/mir_to_compat.rs:761-940+`
- Every new `MirOpCompat` variant requires updating this function. A derive macro or `impl`-based approach would eliminate this class of bugs.

### S-05 — Test in `package.rs` Writes to `/tmp/` Directly
- **Category**: Code Quality
- **File**: `crates/coreml-emit/src/package.rs:308`
- Can cause test flakiness in parallel runs.

### S-06 — `FfiModel::Drop` Implementation Is a No-Op Comment
- **Category**: Code Quality
- **File**: `crates/coreml-ffi/src/model.rs:174-178`
- ```rust
  impl Drop for FfiModel {
      fn drop(&mut self) {
          // On macOS, we would call MLModelDestroy(self.handle.unwrap().raw)
      }
  }
  ```

### S-07 — No Shared Trait Abstraction for NodeId Types
- **Category**: Recycling
- **Files**: `crates/ir/src/sir.rs:11`, `crates/ir/src/air.rs:9`, `crates/ir/src/mir.rs:14`
- `SirNodeId(pub String)`, `AirNodeId(pub String)`, `MirNodeId(pub String)` — structurally identical newtypes with no shared trait. A `trait IrNodeId` would enable reusable traversals.

### S-08 — No Shared Graph Trait — Identical Structure 4× Over
- **Category**: Recycling
- **Files**: `crates/ir/src/sir.rs:965-969`, `crates/ir/src/air.rs:891-896`, `crates/ir/src/mir.rs:1301-1307`, `crates/ir/src/pir.rs:878-898`
- All four graph types share `{ nodes, inputs, outputs }` but have no shared trait. No generic topological sort, DCE, or reachability analysis.

### S-09 — `serialize.rs` Is Pure Copy-Paste Boilerplate
- **Category**: Recycling
- **File**: `crates/ir/src/serialize.rs:1-43`
- All 8 functions are identical except for the type name. A single generic function would replace all 8.

### S-10 — `DEFAULT_OPSET_VERSION` Defined but `"iOS18"` Hardcoded 12 Times
- **Category**: Code Quality
- **File**: `crates/ir/src/linear_slice.rs` (12 occurrences)
- Constant defined in `lib.rs:17` and used in `pir.rs` but `linear_slice.rs` hardcodes the string.

### S-11 — Magic Number `seed: 42` Hardcoded in 6 Payload Constructors
- **Category**: Code Quality
- **File**: `crates/ir/src/linear_slice.rs:377, 496, 603, 701, 785, 861`

### S-12 — `AneHwLimits::a12()` Is an Unverified Copy of A11 Values
- **Category**: Code Quality
- **File**: `crates/ir/src/ane_hw_limits.rs:64-69`
- TODO comment says values may be wrong. Code relying on A12 limits may produce incorrect results.

### S-13 — `ElementWise` / `ElementWiseOp` Marked Legacy but Still in Core Enums
- **Category**: Code Quality / Separation of Concerns
- **Files**: `crates/ir/src/sir.rs:906-924`, `crates/ir/src/air.rs:865-875`
- Every match on `SirOp`/`AirOp` must handle legacy variants.

### S-14 — Duplicate `MockKnowledge` Implementations Across Pass Tests
- **Category**: Recycling
- **Files**: `crates/passes/src/risk_annotate.rs:171-250`, `crates/passes/src/shard_plan.rs:775-855`, `crates/passes/src/legality_rewrite.rs:2402-2470`
- Nearly identical mock structures. A shared mock in a test-util module would reduce duplication.

### S-15 — README Test Count Is Stale (440 Claimed, 563 Actual)
- **Category**: Code Quality
- **File**: `README.md:18,23`
- 28% discrepancy. Likely accurate at some sprint boundary but not updated.

---

## What the Project Does Well

1. **Partial workspace dependency centralisation** — `serde`, `anyhow`, `serde_json`, `sha2`, `chrono`, `clap`, `zip`, `rmp-serde` all use `workspace = true` consistently.
2. **`clippy.toml` cognitive complexity threshold (50)** and `rustfmt.toml` are good practices.
3. **Thorough constraint-validation tests** — `op_constraints`, `dtype_constraints`, `ane_layout`, `ane_hw_limits`, `placement_validate` exercise both `.is_ok()` and `.is_err()` paths.
4. **Honest `smoke_test.sh`** — reports limitations rather than faking success.
5. **Good doc comments throughout IR types** — variant-level documentation is present and helpful.
6. **`MirOpCompat` 27-variant emission path** — comprehensive coverage of ANE-compatible ops.

---

## Severity Summary

| Severity | Count |
|----------|-------|
| 🔴 Critical | 14 |
| 🟡 Warning | 30 |
| 🟢 Suggestion | 15 |
| **Total** | **59** |
