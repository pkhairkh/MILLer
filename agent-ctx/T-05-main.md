# T-05 — Move Lab orchestration out of CLI into ane_lab crate

## Task Summary
Extracted lab orchestration logic from `crates/cli/src/main.rs` into a new `session` module in the `ane_lab` crate.

## Changes Made

### 1. Created `crates/lab/src/session.rs`
New module containing:

**Session configuration structs:**
- `LabSession` — holds configuration for a single lab run (input, output, bridge_script, python_path, do_inspect, seed, generated_from)
- `LabLoopSession` — holds configuration for a lab-loop run (same + knowledge_dir)
- `LabResult` — structured output of a lab session (success, task_name, output_dir, manifest_path, error_message)

**Helper functions (moved from CLI):**
- `compute_task_hash()` — deterministic SHA-256 hash for task specs
- `build_artifact_manifest()` — builds artifact manifest from spec + bridge result (now takes `compiler_version` as parameter instead of using `env!` macro internally)
- `build_knowledge_update()` — builds knowledge update JSON from compilation result
- `build_knowledge_update_with_drift()` — builds knowledge update with drift evidence
- `ingest_knowledge_observations()` — ingests observations into the knowledge store
- `StoreKnowledgeQuery` — adapter from `KnowledgeStore` to `PassKnowledgeQuery` trait (with full impl)

**Session run methods:**
- `LabSession::run()` — full orchestration of a lab run (8 steps)
- `LabLoopSession::run()` — full orchestration of a lab-loop run with knowledge ingestion (9 steps)

### 2. Updated `crates/lab/src/lib.rs`
- Added `pub mod session;`

### 3. Updated `crates/lab/Cargo.toml`
- Added dependencies: `ane-knowledge`, `ane-passes`, `ane-artifacts`, `sha2`

### 4. Updated `crates/cli/src/main.rs`
- `run_lab()` now constructs a `LabSession` and calls `.run()` instead of inline logic
- `run_lab_loop()` now constructs a `LabLoopSession` and calls `.run()` instead of inline logic
- `compute_task_hash()` delegates to `ane_lab::session::compute_task_hash`
- `build_artifact_manifest()` delegates to `ane_lab::session::build_artifact_manifest` (with `env!("CARGO_PKG_VERSION")` passed as parameter)
- `build_knowledge_update()` delegates to `ane_lab::session::build_knowledge_update`
- `StoreKnowledgeQuery` now wraps `ane_lab::session::StoreKnowledgeQuery` and delegates trait method calls
- Removed `use sha2::Digest;` (no longer needed directly in CLI)
- Removed dead-code delegate functions (`ingest_knowledge_observations`, `build_knowledge_update_with_drift`)

## Compilation Status
- `cargo check --workspace` passes with no errors and no warnings

## Design Notes
- Used the "strangler fig" pattern: defined the interface in the right crate first
- `build_artifact_manifest` now takes `compiler_version` as a parameter since `env!("CARGO_PKG_VERSION")` would resolve to the lab crate's version, not the CLI's
- `StoreKnowledgeQuery` in CLI wraps the lab crate's version for backward compatibility with existing call sites
- The `compute_baseline()` helper is a private function in `session.rs` that dispatches on op type
