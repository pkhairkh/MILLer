# T-04 — Break up the `linear_slice.rs` god module

## Summary

Successfully refactored the 2097-line `linear_slice.rs` into 3 focused modules:

1. **`payload.rs`** (~680 lines) — All bridge payload types:
   - `DEFAULT_SEED` constant (replaces hardcoded `seed: 42`)
   - `BRIDGE_VERSION` constant
   - `FunctionDescriptor`, `TensorDescriptor`
   - `LinearProjectionPayload`, `LutProjectionPayload`, `DecodeStepPayload`, `MlpBlockPayload`, `AttentionPayload` — all with `#[deprecated(since = "0.2.0", note = "Use FamilyPayload instead")]`
   - `FamilyPayload` (not deprecated)
   - All `"iOS18"` replaced with `crate::DEFAULT_OPSET_VERSION`
   - All `#[allow(deprecated)]` on impl blocks for deprecated types

2. **`shard_desc.rs`** (~340 lines) — Shard-related types:
   - `ShardDesc` struct
   - `sharded_pipeline_shards` function
   - `lower_shard_to_mir` function
   - `ShardedShardPayload` struct + impl
   - `build_sharded_pipeline_pir` function
   - All `"iOS18"` replaced with `crate::DEFAULT_OPSET_VERSION`

3. **`linear_slice.rs`** (~580 lines, was 2097) — SIR construction and MIR lowering:
   - `sir_from_linear_projection` function (core SIR builder)
   - `lower_linear_projection_to_mir` function (core MIR lowerer)
   - `pub use super::payload::*;` and `pub use super::shard_desc::*;` for backward compatibility
   - All `"iOS18"` replaced with `crate::DEFAULT_OPSET_VERSION`
   - Tests remain (with `#[allow(deprecated)]` on test functions using deprecated types)
   - Added `crate::pir::{PackageRole, ShardRole}` import in test module

4. **`lib.rs`** — Added `pub mod payload;` and `pub mod shard_desc;`

## Verification

- `cargo check -p ane-ir --tests` — PASS (0 errors, 0 warnings)
- `cargo test -p ane-ir` — 82 tests PASS
- `cargo check -p ane-passes --tests` — PASS
- `ane-cli` has pre-existing `ane_lab` errors unrelated to this refactoring
- All `ane_ir::linear_slice::X` imports in CLI continue to work via re-exports
