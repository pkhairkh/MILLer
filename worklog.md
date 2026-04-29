---
Task ID: 1
Agent: main
Task: Extract and inspect MILLer.zip, read all docs, audit repo state, select and implement Sprint 56

Work Log:
- Extracted MILLer.zip to /home/z/my-project/work
- Read TASKS.md, STATUS.md, README.md, SPEC.md, ISSUES.md
- Inspected all Rust crates (ir, passes, bridge, lab, knowledge, coreml-proto, coreml-emit, coreml-ffi, artifacts, report, cli)
- Inspected Python bridge files (mil_emitter.py, bridge.py, verify.py)
- Installed Rust toolchain (1.95.0) and protoc (28.3)
- Verified existing test suite: 431 tests passing
- Performed truth audit: identified ISSUES #3 (placeholder shapes in AIR decomposition) as highest-value gap
- Selected Sprint 56: Replace Placeholder AIR Decomposition Shapes with Real Dimensions
- Implemented DecompositionContext struct in legality_rewrite.rs
- Updated decompose_attention_block() to use real dimensions from context
- Updated decompose_decode_step() to use real dimensions from context
- Updated LegalityRewritePass::run() to accept optional DecompositionContext
- Updated CLI callers (compile-full, compile-full-sharded) to construct and pass context from task spec
- Added 6 new tests for Sprint 56 shape propagation
- All 437 workspace tests passing, zero compiler warnings

Stage Summary:
- Sprint 56 core implementation complete: 20 placeholder shapes replaced with real dimensions
- DecompositionContext provides backward-compatible optional parameter (None = old placeholder behavior)
- CLI automatically constructs context from task spec for Attention/DecodeStep/ShardedDecodeStep tasks
- Key finding: The AIR decomposition was structurally correct but semantically incomplete — all SliceByIndex bounds, Reshape target shapes, and StateReadFixed shapes carried zero-filled placeholders instead of real task dimensions
- Test count: 431 → 437 (6 new tests)
- Remaining: Update TASKS.md, STATUS.md, ISSUES.md, repack zip

---
Task ID: 2
Agent: main (continuation)
Task: Finalize Sprint 56 — fix doc duplicates, verify build, create final zip

Work Log:
- Verified Rust toolchain still available (cargo 1.95.0)
- Confirmed 437 workspace tests passing, zero warnings
- Fixed duplicate section in TASKS.md Current Baseline (lines 148-159 were stale copy with "431 tests")
- Verified TASKS.md Sprint 56 section present and complete
- Verified STATUS.md header updated with Sprint 56 and 437 tests
- Verified ISSUES.md item #3 updated with Sprint 56 resolution
- Verified Sprint status snapshot includes Sprint 56 as DONE
- Creating final MILLer.zip

Stage Summary:
- Sprint 56 fully complete: code, tests, and all docs are consistent
- TASKS.md duplicate section removed (stale "431 tests" line replaced)
- All truth-facing docs (TASKS.md, STATUS.md, ISSUES.md) aligned on 437 tests and Sprint 56 completion

---
Task ID: 2
Agent: main
Task: Fix Core ML validation errors: missing outputSchema, wrong input shapes/dtypes, tensor type/storage mismatch

Work Log:
- Traced full shape/dtype propagation pipeline: AIR → mil_lower.rs → MIR → mir_to_compat.rs → MirGraphCompat → mir_to_proto.rs → CoreMlModel → lib.rs (apple proto emission)
- Identified root cause #1: infer_shape() for Identity nodes with input="__placeholder__" returns empty because "__placeholder__" isn't in node_shapes, then overwrites the seeded input shape (e.g., [1, 512]) with [] — this propagates empty shapes throughout the entire graph
- Identified root cause #2: Single-function models were emitted with multi-function schema (functions=[shard_0], top-level I/O empty), causing Core ML to not find outputSchema metadata
- Identified root cause #3: Graph input Identity nodes with x="__placeholder__" were emitted as "identity" MIL ops referencing a non-existent SSA name, instead of the correct "placeholder" MIL op type
- Fix #1 (mil_lower.rs): Added guard to preserve pre-seeded shapes when infer_shape() returns empty — prevents overwriting input_shapes seed
- Fix #2 (lib.rs): Single-function models now use single-function schema pattern: top-level I/O populated, functions empty, MIL program key="main"; multi-function models continue using the multi-function pattern
- Fix #3 (mir_to_compat.rs + lib.rs): Identity nodes with x="__placeholder__" are now converted to MirOpCompat::Placeholder which emits the correct "placeholder" MIL operation with no inputs
- Added Placeholder variant to MirOpCompat enum with proper proto emission in both Apple and legacy formats
- Updated test_apple_proto_model_description_functions to verify single-function schema
- Updated test_apple_proto_state_ops to look for "main" function key
- All 553 workspace tests passing

Stage Summary:
- Three critical Core ML validation fixes implemented:
  1. Shape inference now preserves seeded input shapes (fixes "Tensor storage and type have different number of elements")
  2. Single-function models use correct schema (fixes "missingMetadataField(named: outputSchema)")
  3. Graph inputs use "placeholder" MIL op instead of broken "identity" op (fixes SSA reference errors)
- Files changed: mil_lower.rs, lib.rs (coreml-proto), mir_to_compat.rs (bridge), mir_to_proto.rs (coreml-emit)
