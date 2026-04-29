# STATUS

Last updated: 2026-04-28 (Sprint 60 COMPLETE + ane-trace crate implemented + trace-compile CLI command wired. Per-op constraint validation, CPU_ONLY hard gate, ANE interleave/layout types, and AneHwLimits per-revision enforcement. ane-trace crate provides HuggingFace model tracing via torch.fx, ANE-faithful SIR construction, versioned compilation with per-family constraints (A11–A18), and model architecture registry (GPT-2, LLaMA/Qwen, BERT, Phi). trace-compile CLI command operational. 501 workspace tests passing.)

## Verification Scope Key

Every claim in this document uses these terms precisely:

| Level | Meaning |
|-------|---------|
| **implemented** | Code exists, compiles, type-checks |
| **host-verified** | Runs successfully on this host (Linux x86_64) |
| **Python/Core ML verified** | Python bridge produces correct .mlpackage via coremltools |
| **Rust-integrated verified** | CLI drives the full path: spec -> IR -> bridge -> result -> artifacts |
| **Apple-device verified** | Runs on Apple hardware with real ANE/Core ML runtime |

**Nothing in this repo is Apple-device verified.**

## Vertical Slice

One end-to-end path is implemented and host-verified:

```
TOML task spec -> Rust SIR graph -> Rust MIR graph -> Bridge payload (JSON)
  -> Python coremltools MIL emission -> .mlpackage on disk
  -> Artifact manifest (JSON) -> Backend-knowledge update (JSON)
```

### What Is Implemented

| Step | Code | Status |
|------|------|--------|
| Task spec loading | `crates/ir/src/task_spec.rs` | implemented, host-verified |
| SIR construction | `crates/ir/src/linear_slice.rs` | implemented, host-verified |
| MIR lowering | `crates/ir/src/linear_slice.rs` | implemented, host-verified |
| Bridge payload gen | `crates/ir/src/linear_slice.rs` (`FamilyPayload`) | implemented, host-verified, **versioned** (`bridge_version: 1`) — generic payload replaces 5 family-specific structs; params field carries family-specific fields |
| Bridge subprocess | `crates/bridge/src/subprocess.rs` | implemented, host-verified |
| Bridge result capture | `crates/bridge/src/subprocess.rs` (`BridgeResult`) | implemented, host-verified — captures content_hash, package_files, compute_plan, function_descriptors |
| CLI compile command | `crates/cli/src/main.rs` | implemented, host-verified |
| CLI compile-full command | `crates/cli/src/main.rs` (`run_compile_full`) | implemented, host-verified — drives pass pipeline with optional knowledge store queries via `--knowledge` |
| CLI lab command | `crates/cli/src/main.rs` (`run_lab`) | implemented, host-verified — drives compile + inspect + baseline + drift + structured run record |
| CLI profile command | `crates/cli/src/main.rs` (`run_profile`) | implemented — honest unavailability on non-Apple hardware; works on Apple hardware |
| CLI package command | `crates/cli/src/main.rs` (`run_package`) | implemented, host-verified — deterministic zip packaging of compile artifacts |
| Deterministic task hash | `crates/cli/src/main.rs` (`compute_task_hash`) | implemented, host-verified — SHA-256 of spec identity string |
| Manifest with truth fields | `crates/cli/src/main.rs` (`build_artifact_manifest`) | implemented, host-verified — uses typed `ArtifactManifest` with `implementation_status`, `verification_scope`, `environment_limitations` |
| MIL emission | `python/mil_emitter.py` (`emit_linear_projection`) | implemented, Python/Core ML verified |
| ML program emission | `python/mil_emitter.py` (`emit_mlprogram`) | implemented, Python/Core ML verified |
| MIL program construction | `python/mil_emitter.py` (`build_linear_projection_program`) | implemented, Python/Core ML verified |
| mlpackage save | `python/mil_emitter.py` (`save_mlpackage`) | implemented, Python/Core ML verified |
| Compute plan info | `python/mil_emitter.py` (`compute_plan_info`) | implemented, Python/Core ML verified — reports unavailable on non-Apple platforms |
| Convert command | `python/bridge.py` (`handle_convert`) | implemented, Python/Core ML verified |
| Palettize command | `python/bridge.py` (`handle_palettize`) | implemented, Python/Core ML verified |
| Compute plan command | `python/bridge.py` (`handle_compute_plan`) | implemented, Python/Core ML verified |
| Host inspect command | `python/bridge.py` (`handle_host_inspect`) | implemented, Python/Core ML verified — honest host-side inspection, no ANE claims |
| Profile command | `python/bridge.py` (`handle_profile`) | implemented — requires Apple hardware for predict(); honest error on non-Apple |
| Bridge dispatch | `python/bridge.py` | implemented, Python/Core ML verified — dispatches 15 commands (including `verify`), version-checked; auto-flattens FamilyPayload.params for backward compatibility |
| Unified verification | `python/verify.py` (`verify_model`) | implemented, Python/Core ML verified — four-dimension verification: op graph fidelity, compute-unit placement, state conformance, multi-function conformance. Spec-based fallback on Linux; MLModelStructure/MLComputePlan on macOS. Returns structured `VerificationResult` with weighted overall score |
| Verify bridge command | `python/bridge.py` (`handle_verify`) | implemented, Python/Core ML verified — dispatches `verify` command, persists verification artifacts as JSON |
| Verification artifact persistence | `python/verify.py` (`save_verification_result`) | implemented, Python/Core ML verified — saves full result + summary JSON |
| Knowledge update with drift | `crates/cli/src/main.rs` (`build_knowledge_update_with_drift`) | implemented, host-verified — version 3 with drift evidence and baseline provenance |
| File content hashing | `crates/artifacts/src/hashing.rs` (`hash_file`) | implemented, host-verified |
| Byte content hashing | `crates/artifacts/src/hashing.rs` (`hash_bytes`) | implemented, host-verified |
| MIL conversion | `python/converter.py` (`convert_milprogram`) | implemented, Python/Core ML verified |
| Palettization | `python/palettize.py` (`apply_palettization`) | implemented, Python/Core ML verified |
| Compute plan inspection | `python/compute_plan.py` (`inspect_compute_plan`) | implemented, Python/Core ML verified |
| Profiling | `python/profiler.py` (`profile_model`) | implemented, wired through bridge — requires Apple hardware for timing data |
| Artifact packaging | `crates/artifacts/src/packaging.rs` (`Packager`) | implemented, host-verified (compiles) |
| Markdown reports | `crates/report/src/markdown.rs` (`MarkdownReporter`) | implemented, host-verified (compiles) |
| JSON reports | `crates/report/src/json_report.rs` (`JsonReporter`) | implemented, host-verified (compiles) |
| CLI report command | `crates/cli/src/main.rs` (`run_report`) | implemented, host-verified (compiles) |
| Smoke test | `scripts/smoke_test.sh` | implemented, documented — reports exact limitations |
| Content-hash deduplication | `crates/coreml-emit/src/weights.rs` (`WeightBinBuilder::with_content_dedup`) | implemented, host-verified — SHA-256-based dedup of differently-named weights with identical content and matching shape/dtype (Sprint 45) |
| Manifest mir_ops | `crates/artifacts/src/manifest.rs` (`FunctionDescriptor.mir_ops`) | implemented, host-verified — MIR op type list per function, populated from pass pipeline MIR graphs (Sprint 47) |
| Verify auto-populate mir_ops | `crates/cli/src/main.rs` (`run_verify`) | implemented, host-verified — auto-populates `--mir-ops` from compile manifest when not explicitly provided (Sprint 47) |

### Model Tracing System (ane-trace crate)

| Component | Code | Status |
|-----------|------|--------|
| TracedGraph data structures | `crates/trace/src/graph.rs` | implemented, host-verified — `TracedGraph`, `TracedNode`, `TracedOp`, `TensorShape` |
| Config-driven decomposition | `crates/trace/src/sir_build.rs` | implemented, host-verified — fully ad-hoc decomposition driven by ModelConfig flags (no model registry) |
| Trace configuration | `crates/trace/src/config.rs` | implemented, host-verified — `TraceConfig`, `TraceTarget` (HuggingFace, local, pre-traced), `InputShape` |
| SIR construction from trace | `crates/trace/src/sir_build.rs` | implemented, host-verified — `build_sir_from_trace()` with separate Q/K/V projections, SwiGLU detection, residual connections (explicit on AttentionBlock/MlpBlock nodes as `[normed_hidden, residual]` inputs), QK-norm support (via `has_qk_norm` config flag for Qwen3 and similar architectures), `head_dim` from config override, causal masks, and epsilon validation |
| Versioned compiler | `crates/trace/src/versioned.rs` | implemented, host-verified — `VersionedCompiler` with per-family constraint validation, `AnceFaithfulnessReport` |
| Python subprocess tracing | `crates/trace/src/subprocess.rs` | implemented — `trace_model()` launches torch.fx tracer via Python subprocess |
| Python tracing script | `python/trace_model.py` | implemented — torch.fx symbolic tracing for HuggingFace models |
| CLI trace-compile command | `crates/cli/src/main.rs` (`run_trace_compile`) | implemented, host-verified — `ane-cli trace-compile --model <MODEL> --output <DIR>` |

**Residual:** The ane-trace crate is structurally complete and wired into the CLI. The `trace-compile` command traces a model, builds SIR, validates ANE faithfulness, and writes artifacts (traced graph, SIR, faithfulness report). End-to-end compilation through the full pass pipeline (SIR → AIR → MIR → bridge emission) is not yet wired — the trace-compile path currently stops after SIR + faithfulness report. Integration tests with real HuggingFace models require `torch` and `transformers` Python packages. The Python tracing script requires `torch.fx` which may not support all model architectures (dynamic control flow models may fail symbolic tracing).

### Lab Run System (Sprint 5)

| Component | Code | Status |
|-----------|------|--------|
| LabRun schema | `crates/lab/src/harness.rs` (`LabRun`, `LabRunBuilder`) | implemented, host-verified — schema version 1.0.0 |
| VerificationScope enum | `crates/lab/src/harness.rs` | implemented — structurally distinct: HostOnlyInspection / HostRuntimeExecution / DeviceBackedExecution |
| EnvironmentSummary | `crates/lab/src/harness.rs` | implemented, host-verified — honest detection of host capabilities |
| CompileStepResult | `crates/lab/src/harness.rs` | implemented, host-verified |
| InspectionStepResult | `crates/lab/src/harness.rs` | implemented, host-verified — includes package_present, manifest_readable, model_loadable |
| HostInspector | `crates/lab/src/host_inspect.rs` | implemented, host-verified — Rust-side file checks + Python bridge model load |
| LabRunWriter | `crates/lab/src/run_dir.rs` | implemented, host-verified — canonical directory layout |
| Run directory layout | `crates/lab/src/run_dir.rs` | implemented — run.json, manifest.json, mir.json, mlpackage/, knowledge/, inspection.json |

### Verification Harness System (Sprint 40)

| Component | Code | Status |
|-----------|------|--------|
| verify_model() | `python/verify.py` | implemented, Python/Core ML verified — unified four-dimension verification entry point |
| OpFidelityResult | `python/verify.py` | implemented, Python/Core ML verified — op graph fidelity with spec-based extraction on Linux |
| PlacementResult | `python/verify.py` | implemented — compute-unit placement via MLComputePlan; unavailable on non-Apple platforms |
| StateConformanceResult | `python/verify.py` | implemented, Python/Core ML verified — state declaration detection + read_state/write_state op verification |
| MultifunctionResult | `python/verify.py` | implemented, Python/Core ML verified — multi-function model conformance with function name matching |
| VerificationResult | `python/verify.py` | implemented, Python/Core ML verified — unified result with weighted overall score (40/20/20/20 split) |
| save_verification_result() | `python/verify.py` | implemented, Python/Core ML verified — persists full result + summary JSON artifacts |
| handle_verify() | `python/bridge.py` | implemented, Python/Core ML verified — bridge dispatch for `verify` command |
| MIR-to-MIL short-name mapping | `python/model_structure.py` | implemented — accepts both `MILLinear` and `Linear` MIR op names |

**Residual:** Full-fidelity verification (MLModelStructure-based op graph, MLComputePlan-based placement) requires macOS with Core ML runtime. On Linux, spec-based extraction provides structural verification only. The Rust CLI now dispatches the `verify` bridge command via `ane-cli verify`; the remaining gap is platform fidelity, not CLI wiring.

### Task Generation System (Sprint 12)

| Component | Code | Status |
|-----------|------|--------|
| LinearFamily | `crates/lab/src/families/linear.rs` | implemented, host-verified — generates deterministic linear projection task specs (3+ variants: 64x32, 128x64, 256x128) |
| LinearFamilyConfig | `crates/lab/src/families/linear.rs` | implemented — configurable dimension variants, batch sizes, dtypes, bias, seed |
| TaskGenerator | `crates/lab/src/task_gen.rs` | implemented, host-verified — orchestrates family generators, persists tasks as TOML with manifest |
| TaskFamilyId | `crates/lab/src/task_gen.rs` | implemented — LinearProjection, LutProjection, DecodeStep, MlpBlock, Attention, ShapeHostile, OpRemap, and ShardSurvival; all eight families are now real TaskFamilyTrait implementations |
| GeneratedTasksManifest | `crates/lab/src/task_gen.rs` | implemented, host-verified — JSON manifest of generated tasks with provenance |
| CLI generate-tasks | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli generate-tasks --family linear|lut|decode|mlp|attn|shape|remap|survival --output <dir> --seed <n>` |
| GeneratorProvenance | `crates/lab/src/harness.rs` | implemented — records generator version, family, seed, task_name in LabRun |
| CLI --generated-from | `crates/cli/src/main.rs` | implemented — `ane-cli lab --generated-from LinearProjection,42,1.0.0` attaches provenance |
| Lab crate tests | `crates/lab/` | 131 tests passing (including task generation, family generator, LUT projection, decode step, and sharded baseline tests) |

### LUT Projection Task Family (Sprint 14, Sprint 20)

| Component | Code | Status |
|-----------|------|--------|
| LutProjectionFamily | `crates/lab/src/families/lut_projection.rs` | implemented, host-verified — generates deterministic LUT projection task specs (3+ bitwidth variants: 4/6/8-bit) |
| LutProjectionFamilyConfig | `crates/lab/src/families/lut_projection.rs` | implemented — configurable bitwidths, embed_dims, num_groups, vocab_size, batch sizes, dtypes, seed |
| TaskOp::LutProjection | `crates/ir/src/task_spec.rs` | implemented, host-verified — vocab_size, embed_dim, num_groups, lut_bitwidth, batch_size, dtype |
| LUT projection TOML parsing | `crates/ir/src/task_spec.rs` | implemented, host-verified — `[synthetic.lut_projection]` section with bitwidth validation |
| LUT projection baseline | `crates/lab/src/baseline.rs` | implemented, host-verified — `compute_lut_projection` models LUT-gather pattern |
| CLI generate-tasks --family lut | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli generate-tasks --family lut --output <dir>` |
| LUT projection benchmark | `benchmarks/synthetic/lut_projection.toml` | implemented — 4-bit, 128-dim, 16-group LUT projection |
| LUT family tests | `crates/lab/src/families/lut_projection.rs` | 8 tests passing — generation, determinism, serialization, config, validation |
| LUT baseline tests | `crates/lab/src/baseline.rs` | 3 tests passing — determinism, shape, bitwidth variation |
| LUT task spec tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, invalid bitwidth, missing fields |
| LutProjectionPayload | `crates/ir/src/linear_slice.rs` | implemented — dedicated LUT bridge payload with vocab_size, embed_dim, num_groups, lut_bitwidth fields; command="emit_lut_projection" (Sprint 20) |
| build_lut_projection_program | `python/mil_emitter.py` | implemented — gather-based MIL program construction (Sprint 20) |
| emit_lut_projection | `python/mil_emitter.py` | implemented — dedicated LUT emission path: build gather-based program → convert → save (Sprint 20) |
| emit_lut_projection bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_lut_projection` command (Sprint 20) |
| LUT payload divergence tests | `crates/ir/src/linear_slice.rs` | 6 tests passing — payload creation, rejection of wrong specs, command divergence, deterministic serialization, function descriptors (Sprint 20) |

**Residual:** The LUT emission path is v0. The gather-based program models a simplified LUT pattern using offset indexing to approximate grouped-palette semantics; true per-group independent LUTs with per-group index tensors are not yet implemented. Precision override (dtype adaptation from knowledge) is now wired into the `LutProjectionPayload` path via `from_spec_with_override` (Sprint 30). The `lut_bitwidth` field is carried in the payload but does not yet influence the coremltools conversion parameters. End-to-end validation requires Apple hardware with Core ML runtime.

### MLP Block Task Family (Sprint 26, Sprint 28)

| Component | Code | Status |
|-----------|------|--------|
| MlpBlockFamily | `crates/lab/src/families/mlp_block.rs` | implemented, host-verified — generates deterministic MLP block task specs (4+ variants: 2 input_dims × 2 activations) |
| MlpBlockFamilyConfig | `crates/lab/src/families/mlp_block.rs` | implemented — configurable input_dims, hidden_dims, output_dims, activations, batch sizes, dtypes, seed |
| TaskOp::MlpBlock | `crates/ir/src/task_spec.rs` | implemented, host-verified — input_dim, hidden_dim, output_dim, activation, batch_size, dtype |
| MlpBlock TOML parsing | `crates/ir/src/task_spec.rs` | implemented, host-verified — `[synthetic.mlp_block]` section with activation validation ("gelu" or "relu") |
| MlpBlock baseline | `crates/lab/src/baseline.rs` | implemented, host-verified — `compute_mlp_block` models up-projection + activation + down-projection |
| MlpBlockPayload | `crates/ir/src/linear_slice.rs` | implemented — dedicated MLP block bridge payload with command="emit_mlp_block" (Sprint 28) |
| build_mlp_block_program | `python/mil_emitter.py` | implemented — constructs MIL program modeling up-projection + activation + down-projection (Sprint 28) |
| emit_mlp_block | `python/mil_emitter.py` | implemented — dedicated MLP block emission path: build → convert → save (Sprint 28) |
| emit_mlp_block bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_mlp_block` command (Sprint 28) |
| CLI generate-tasks --family mlp | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli generate-tasks --family mlp --output <dir>` |
| MlpBlock benchmark | `benchmarks/synthetic/mlp_block.toml` | implemented — 128-dim, 512 hidden, GELU, fp16 |
| MlpBlock family tests | `crates/lab/src/families/mlp_block.rs` | 7 tests passing — generation, determinism, serialization, config, output_dim defaults, trait dispatch, variant count |
| MlpBlock baseline tests | `crates/lab/src/baseline.rs` | 3 tests passing — determinism, shape, activation variation |
| MlpBlock task spec tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, invalid activation, missing fields |
| MLP payload divergence tests | `crates/ir/src/linear_slice.rs` | 5 tests passing — payload creation, rejection of wrong specs, command divergence from linear, deterministic serialization, function descriptors (Sprint 28) |

**Residual:** The MLP block emission path now uses native `mb.gelu(mode="TANH_APPROXIMATION")` instead of the hand-rolled 12-op GELU chain (Sprint 31). FC projections now use `mb.linear` instead of `mb.matmul` (Sprint 31). **Post-Sprint 35 correction**: `mb.linear` weight shape was fixed from `[input_dim, output_dim]` to `[output_dim, input_dim]` in the MLP, attention, and decode-step emitters (coremltools 9.0 requires the transposed convention). The `emit_mlp_block` path constructs a fused MIL program with up-projection → native GELU → down-projection in a single MIL function. Dtype override is wired into `MlpBlockPayload::from_spec_with_override` for compile-full (Sprint 30); precision adaptation propagates to MLP block bridge payloads. End-to-end validation requires Apple hardware with Core ML runtime.

### Attention Task Family (Sprint 29)

| Component | Code | Status |
|-----------|------|--------|
| AttentionFamily | `crates/lab/src/families/attention.rs` | implemented, host-verified — generates deterministic attention task specs (2+ variants: 2 embed_dims × 1 num_heads) |
| AttentionFamilyConfig | `crates/lab/src/families/attention.rs` | implemented — configurable embed_dims, num_heads, seq_lens, batch_sizes, dtypes, seed |
| TaskOp::Attention | `crates/ir/src/task_spec.rs` | implemented, host-verified — embed_dim, num_heads, head_dim, seq_len, batch_size, dtype |
| Attention TOML parsing | `crates/ir/src/task_spec.rs` | implemented, host-verified — `[synthetic.attention]` section with divisibility validation |
| Attention baseline | `crates/lab/src/baseline.rs` | implemented, host-verified — `compute_attention` models QKV + scaled dot-product attention + output projection |
| AttentionPayload | `crates/ir/src/linear_slice.rs` | implemented — dedicated attention bridge payload with command="emit_attention" |
| build_attention_program | `python/mil_emitter.py` | implemented — constructs MIL program modeling QKV + attention + output projection |
| emit_attention | `python/mil_emitter.py` | implemented — dedicated attention emission path: build → convert → save |
| emit_attention bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_attention` command |
| CLI generate-tasks --family attn | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli generate-tasks --family attn --output <dir>` |
| Attention benchmark | `benchmarks/synthetic/attention.toml` | implemented — 128-dim, 4 heads, seq_len 32, fp16 |
| Attention family tests | `crates/lab/src/families/attention.rs` | 7 tests passing — generation, determinism, serialization, config, validation, trait dispatch, variant count |
| Attention baseline tests | `crates/lab/src/baseline.rs` | 3 tests passing — determinism, shape, head variation |
| Attention task spec tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, invalid divisibility, missing fields |
| Attention payload divergence tests | `crates/ir/src/linear_slice.rs` | 5+ tests passing — payload creation, rejection of wrong specs, command divergence from linear, deterministic serialization, function descriptors |

**Residual:** The attention emission path now uses `mb.scaled_dot_product_attention` (iOS 18+) for real multi-head attention computation (Sprint 31). **Post-Sprint 35 correction**: `mb.scaled_dot_product_attention` parameter names were fixed from `x/y/z` to `query/key/value` (coremltools 9.0 API). `mb.linear` weight shape was fixed from `[input_dim, output_dim]` to `[output_dim, input_dim]`. **Bug fix (Sprint 39 pass)**: Causal mask parameter was using `mask=` instead of `attn_mask=` (coremltools 9.0 API), which caused MIL construction failure. Fixed to use `attn_mask=`. Causal masking now works correctly for both `causal=True` and `causal=False` modes. The MIL program models QKV projection via `mb.linear`, Q/K/V reshape+transpose for multi-head layout, `mb.scaled_dot_product_attention` with optional causal mask, reshape back, and output projection. The attention emitter is no longer a Q-only placeholder shortcut — it contains real attention semantics. Dtype override is wired into `AttentionPayload::from_spec_with_override` for compile-full (Sprint 30). End-to-end validation requires Apple hardware with Core ML runtime.

### Knowledge Consumption on Active Paths (Sprint 13)

| Component | Code | Status |
|-----------|------|--------|
| compile --knowledge | `crates/cli/src/main.rs` (run_compile) | implemented — loads knowledge store, records consultation status in manifest |
| Knowledge consultation in manifest | `crates/cli/src/main.rs` | implemented — manifest includes knowledge_consulted, knowledge_seed_count, knowledge_observation_count when knowledge was available |
| Knowledge influence test | `crates/passes/src/legality_rewrite.rs` | implemented, host-verified — test_knowledge_influences_legality_pass_output proves legal knowledge increases confidence, illegal knowledge decreases it |
| NoKnowledge defaults test | `crates/passes/src/legality_rewrite.rs` | implemented, host-verified — test_no_knowledge_default_confidence verifies default values |
| Passes crate tests | `crates/passes/` | 2 tests passing (knowledge influence integration tests) |

### Host-Side Evidence Loop (Sprint 21)

| Component | Code | Status |
|-----------|------|--------|
| lab-loop CLI command | `crates/cli/src/main.rs` (`run_lab_loop`) | implemented — single command that executes: task load → compile → baseline → drift → knowledge store ingestion → run artifact emission |
| LabLoop subcommand | `crates/cli/src/main.rs` (Commands::LabLoop) | implemented — `ane-cli lab-loop --input <spec> --output <dir> --knowledge <store-dir>` with required `--knowledge` argument |
| ingest_knowledge_observations | `crates/cli/src/main.rs` | implemented — converts knowledge update JSON observations into KnowledgeUnit structs and ingests them via UpdatePipeline |
| LabRun.adaptation_readiness | `crates/lab/src/harness.rs` | implemented — records whether run produced "artifacts_only", "artifacts_and_observation", or "artifacts_observation_compiler_consumable" |
| LabRunBuilder.adaptation_readiness | `crates/lab/src/harness.rs` | implemented — builder method for adaptation readiness field |
| Manifest adaptation_readiness | `crates/cli/src/main.rs` (`run_lab_loop`) | implemented — manifest includes adaptation_readiness, knowledge_store_path, observations_ingested |
| lab command adaptation_readiness | `crates/cli/src/main.rs` (`run_lab`) | implemented — existing lab command sets adaptation_readiness to "artifacts_only" |

**Residual:** The Rust workspace test run passes on this host, but `lab-loop` itself has not been re-run end-to-end in this audit pass. Observations with evidence_count=0 (from unavailable drift data) are correctly rejected by UpdatePipeline validation. Only LegalityRule observations from successful compiles are ingested with confidence > 0 and evidence_count >= 1, making them compiler-consumable. Full end-to-end validation still benefits from a Python bridge execution environment and, for drift/runtime data, Apple hardware.

### Knowledge-Affecting Compilation (Sprint 16)

| Component | Code | Status |
|-----------|------|--------|
| PrecisionHazardInfo | `crates/passes/src/knowledge_query.rs` | implemented — op_pattern, hazardous_dtype, recommended_dtype, confidence, evidence_count, source_id, description |
| query_precision_hazard() | `crates/passes/src/knowledge_query.rs` | implemented — trait method on PassKnowledgeQuery for precision hazard lookup |
| NoKnowledge.query_precision_hazard() | `crates/passes/src/knowledge_query.rs` | implemented — returns None (no hazard without knowledge) |
| StoreKnowledgeQuery.query_precision_hazard() | `crates/cli/src/main.rs` | implemented, host-verified — queries knowledge store for PrecisionHazard entries |
| PrecisionPolicyPass | `crates/passes/src/precision_policy.rs` | implemented, host-verified — first knowledge-affecting pass: overrides fp16 to fp32 when hazard known |
| PrecisionAdaptation | `crates/passes/src/precision_policy.rs` | implemented — serializable record of dtype override with full provenance |
| SirMetadata.precision_override | `crates/ir/src/sir.rs` | implemented — carries adapted dtype through the pipeline |
| Pipeline wiring (step 4b) | `crates/cli/src/main.rs` | implemented — PrecisionPolicyPass runs between StaticizePass and LegalityRewritePass |
| Manifest precision_adaptations | `crates/cli/src/main.rs` | implemented — manifest includes precision_adaptations array and precision_adapted boolean |
| Precision adaptation tests | `crates/passes/src/precision_policy.rs` | 7 tests passing — hazard changes dtype, NoKnowledge, low-confidence, safe knowledge, provenance, reset, threshold |

**Residual:** The precision adaptation is narrow and v0 — it only affects the dtype decision for ops with known hazards. The adapted dtype now propagates through AIR → MIR → bridge payload to the Python emitter (Sprint 18), closing the previous v0 limitation where `SirMetadata.precision_override` was set but not consumed downstream. The Python emission layer receives the adapted dtype but must map it to `compute_precision` in `converter.convert_milprogram()` for full end-to-end effect on the emitted mlpackage — this is a minor remaining gap that requires Apple hardware to validate.

### Shard Runtime Semantics (Sprint 17)

| Component | Code | Status |
|-----------|------|--------|
| HandoffKind enum | `crates/ir/src/pir.rs` | implemented — `TensorPassThrough`, `StateWriteRead` |
| Concrete Handoff | `crates/ir/src/pir.rs` | implemented — `handoff_kind`, `execution_order`, `source_output_name`, `target_input_name` |
| ShardPlan (extended) | `crates/passes/src/shard_plan.rs` | implemented — `is_multi_shard`, `shard_roles`, `shard_names` |
| ShardPlanPass::build_sharded_plan | `crates/passes/src/shard_plan.rs` | implemented, host-verified — 3-shard Entry/Interior/Exit with concrete handoffs |
| compile-full-sharded CLI | `crates/cli/src/main.rs` | implemented, host-verified — full pass pipeline per shard with knowledge support |
| Per-shard provenance | `crates/cli/src/main.rs` | implemented — `shard_provenance` array with role, precision adaptations, pass pipeline path |
| Concrete handoff manifest | `crates/cli/src/main.rs` | implemented — `concrete_handoffs` array with full handoff semantics |
| Shard plan manifest | `crates/cli/src/main.rs` | implemented — `shard_plan` object in manifest |
| Manifest version 0.6.0 | `crates/cli/src/main.rs` | implemented — shard-aware full-pipeline manifest |
| Shard plan tests | `crates/passes/src/shard_plan.rs` | 4 tests passing — three_shards, pir_structure, dimensions, serialization |
| Compute unit adaptation tests | `crates/passes/src/shard_plan.rs` | 7 tests passing — high_fallback_risk_overrides, no_knowledge_no_adaptation, low_risk_keeps_ane, borderline_risk, adaptations_reset, custom_threshold, provenance |
| Concrete handoff tests | `crates/ir/src/linear_slice.rs` | 3 tests passing — execution_order, source_target_names, pir_concrete_handoffs |

**Residual:** The multi-shard orchestration is at the CLI level — each shard is compiled independently through the pass pipeline. The pass pipeline itself is shard-agnostic (ShardPlanPass always produces single-shard plans when called from within the pass pipeline). Multi-shard planning is done by `ShardPlanPass::build_sharded_plan` which is called at the CLI orchestration level. This is the honest v1: the pass pipeline compiles individual shards, and the CLI orchestrates multi-shard compilation. Inter-shard runtime execution (actual predict() calls across shards) still requires Apple hardware. StateWriteRead handoff kind is defined but not yet used by the active path (no KV-cache state in the synthetic pipeline).

### Precision Adaptation Propagation (Sprint 18)

| Component | Code | Status |
|-----------|------|--------|
| AirNode.precision_override | `crates/ir/src/air.rs` | implemented — carries adapted dtype from SIR to AIR |
| LegalityRewritePass propagation | `crates/passes/src/legality_rewrite.rs` | implemented — propagates `sir_node.metadata.precision_override` to AIR node |
| MilLowerPass dtype from AIR | `crates/passes/src/mil_lower.rs` | implemented — derives `MirNode.dtype` from `air_node.precision_override` |
| MilDtype PartialEq/Eq | `crates/ir/src/mir.rs` | implemented — enables test assertions on dtype equality |
| LinearProjectionPayload::from_spec_with_override | `crates/ir/src/linear_slice.rs` | implemented — bridge payload accepts dtype override |
| ShardedShardPayload::from_shard_with_override | `crates/ir/src/linear_slice.rs` | implemented — per-shard bridge payload accepts dtype override |
| compile-full dtype override wiring | `crates/cli/src/main.rs` | implemented — extracts adapted dtype from precision_policy.adaptations |
| compile-full-sharded dtype override wiring | `crates/cli/src/main.rs` | implemented — per-shard adapted dtype from precision_policy |
| ShardPlan Default impl | `crates/passes/src/shard_plan.rs` | implemented — enables test construction of ShardPlan |
| Precision override MIR tests | `crates/passes/src/mil_lower.rs` | 3 tests passing — override propagates, no override fp16, explicit fp16 |
| Precision override full pipeline test | `crates/passes/src/legality_rewrite.rs` | 1 test passing — SIR→AIR→MIR propagation |
| Bridge payload override tests | `crates/ir/src/linear_slice.rs` | 4 tests passing — payload override, no override, shard override, full pipeline |

**Residual:** The Python bridge `emit_linear_projection` uses the dtype from the payload, which now correctly reflects the adapted dtype. All five active task families (LinearProjection, LutProjection, DecodeStep, MlpBlock, Attention) now support `from_spec_with_override` in their bridge payloads (Sprint 30), so precision adaptation propagates to all family payloads in compile-full. However, the `compute_precision` parameter in `converter.convert_milprogram()` must honor this dtype for the adaptation to affect the actual emitted mlpackage. The bridge already passes dtype to the emitter, so the propagation chain is complete from SIR to Python. The Python emitter needs to use `compute_precision=ct.types.fp32` when the payload dtype is `"fp32"` — this is a minor remaining gap in the Python emission layer that is documented but not yet validated end-to-end on Apple hardware.

### Compute Unit Adaptation — Knowledge-Affecting Shard Planning (Sprint 22)

| Component | Code | Status |
|-----------|------|--------|
| ComputeUnitAdaptation | `crates/passes/src/shard_plan.rs` | implemented — serializable record of compute unit override with full provenance |
| ShardPlanPass knowledge query | `crates/passes/src/shard_plan.rs` | implemented — queries `knowledge_query.query_risk()` for shard's primary op pattern |
| ShardPlanPass adaptation logic | `crates/passes/src/shard_plan.rs` | implemented — overrides CPU_AND_NE to CPU_AND_GPU when fallback_risk >= 0.5 |
| FALLBACK_RISK_THRESHOLD | `crates/passes/src/shard_plan.rs` | implemented — configurable threshold (default 0.5) |
| ShardPlanPass.has_adaptations() | `crates/passes/src/shard_plan.rs` | implemented — reports whether any adaptations were made |
| Manifest compute_unit_adaptations | `crates/cli/src/main.rs` | implemented — manifest includes adaptation records with shard_name, original/adapted compute units, op_pattern, fallback_risk, source_id, confidence, reason |
| Manifest compute_units_adapted | `crates/cli/src/main.rs` | implemented — boolean flag, `false` when no adaptation happened |
| Compute unit adaptation tests | `crates/passes/src/shard_plan.rs` | 7 tests passing — high_fallback_risk_overrides, no_knowledge_no_adaptation, low_risk_keeps_ane, borderline_risk, adaptations_reset, custom_threshold, provenance |

**Residual:** This is the second knowledge-affecting pass in the pipeline (after PrecisionPolicyPass). The adaptation is narrow: only the compute unit assignment is overridden based on fallback risk. The threshold is conservative (0.5) and the override only affects ANE-targeted shards. Without matching knowledge, the pass uses the default CPU_AND_NE assignment, preserving backward compatibility. End-to-end validation requires Rust compilation + a knowledge store containing risk observations. The `build_sharded_plan` static method (used for multi-shard CLI orchestration) does not yet apply per-shard knowledge adaptation — this is expected since multi-shard orchestration happens at the CLI level and per-shard knowledge queries happen when each shard is individually compiled through the pass pipeline.

### Generalized Shard Runtime Semantics (Sprint 23)

| Component | Code | Status |
|-----------|------|--------|
| ShardSpec | `crates/ir/src/pir.rs` | implemented — generalized shard descriptor using TensorSpec for I/O, replacing scalar-dimension ShardDesc for pipeline construction |
| ShardPipelineSpec | `crates/ir/src/pir.rs` | implemented — complete multi-shard pipeline specification independent of any specific task family |
| ShardPipelineSpec::three_shard_linear | `crates/ir/src/pir.rs` | implemented — factory method for 3-shard linear decomposition, backward-compatible with previous build_sharded_plan interface |
| ShardPipelineSpec::three_shard_decode_step | `crates/ir/src/pir.rs` | implemented — factory method for 3-shard decode-step decomposition (QKV/Attention/OutputProjection) |
| ShardPipelineSpec::to_pir_graph | `crates/ir/src/pir.rs` | implemented — generalized PIR construction from any pipeline spec |
| ShardPipelineSpec::to_shard_plan | `crates/passes/src/shard_plan.rs` (build_sharded_plan_from_spec) | implemented — converts pipeline spec to ShardPlan |
| build_sharded_plan_from_spec | `crates/passes/src/shard_plan.rs` | implemented — generalized multi-shard construction from ShardPipelineSpec |
| build_sharded_plan (refactored) | `crates/passes/src/shard_plan.rs` | implemented — backward-compatible method now delegates to build_sharded_plan_from_spec |
| TaskOp::ShardedDecodeStep | `crates/ir/src/task_spec.rs` | implemented — new task op for sharded decode-step with embed_dim, num_heads, head_dim, kv_len, batch_size, dtype |
| parse_sharded_decode_step | `crates/ir/src/task_spec.rs` | implemented — TOML parser for [synthetic.sharded_decode_step] sections |
| CLI compile-sharded (generalized) | `crates/cli/src/main.rs` | implemented — accepts both ShardedLinearPipeline and ShardedDecodeStep tasks |
| CLI compile-full-sharded (generalized) | `crates/cli/src/main.rs` | implemented — accepts both ShardedLinearPipeline and ShardedDecodeStep tasks |
| compute_task_hash (ShardedDecodeStep) | `crates/cli/src/main.rs` | implemented — deterministic hash for ShardedDecodeStep tasks |
| Decode-step shard template seed | `knowledge/decode_step_shard_template_seed.json` | implemented — synthetic-run evidence source, confidence 0.5 |
| Sharded decode-step benchmark | `benchmarks/synthetic/sharded_decode_step.toml` | implemented — 128-dim, 4 heads, 32-token KV cache |
| ShardedDecodeStep TOML parsing tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, invalid divisibility, missing fields |
| Generalized pipeline spec tests | `crates/passes/src/shard_plan.rs` | 3 tests passing — linear-from-spec, decode-step-from-spec, linear-vs-decode-step divergence |
| build_sharded_plan_from_spec_with_knowledge | `crates/passes/src/shard_plan.rs` | implemented — accepts `&[ShardTemplate]` slice; matching templates override compute units; non-matching templates are ignored |
| Shard template seed consumption tests | `crates/passes/src/shard_plan.rs` | 3 tests passing — template overrides compute units, non-matching template is ignored, empty template list same as base |

**Residual:** Shard template seeds are now consumed by both `compile-sharded` and `compile-full-sharded` CLI commands when `--knowledge` is provided (Sprint 27). The `compile-sharded` command now accepts `--knowledge` flag. When templates are available and match the pipeline's shard structure, compute unit assignments are overridden from the template. Decode-step shards now use dedicated `DecodeStepPayload` with `emit_decode_step` bridge command. The `HandoffKind::StateWriteRead` variant is defined but not used by any active path (all handoffs are `TensorPassThrough`). KV cache `StateDeclaration` is carried in the decode-step pipeline spec and PIR graph but not yet consumed by emission. Full end-to-end validation requires Rust compilation + Python bridge execution.

### Generalized Task Family Surface (Sprint 19)

| Component | Code | Status |
|-----------|------|--------|
| TaskFamilyTrait | `crates/lab/src/families/mod.rs` | implemented — uniform trait for family dispatch, eliminating ad hoc branching in TaskGenerator |
| TaskFamilyId::DecodeStep | `crates/lab/src/task_gen.rs` | implemented — third active family variant with create_generator dispatch |
| TaskGenerator trait dispatch | `crates/lab/src/task_gen.rs` | implemented, host-verified — generate() dispatches through TaskFamilyTrait, no family-specific branching in orchestration |
| DecodeStepFamily | `crates/lab/src/families/decode_step.rs` | implemented, host-verified — generates deterministic decode-step task specs (4+ variants: 2 embed_dims × 2 kv_lens with default config) |
| DecodeStepFamilyConfig | `crates/lab/src/families/decode_step.rs` | implemented — configurable embed_dims, num_heads, kv_lens, batch_sizes, dtypes, seed |
| TaskOp::DecodeStep | `crates/ir/src/task_spec.rs` | implemented, host-verified — embed_dim, num_heads, head_dim, kv_len, batch_size, dtype |
| Decode step TOML parsing | `crates/ir/src/task_spec.rs` | implemented, host-verified — `[synthetic.decode_step]` section with divisibility validation |
| CLI generate-tasks --family decode | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli generate-tasks --family decode --output <dir>` |
| decode_step benchmark | `benchmarks/synthetic/decode_step.toml` | implemented — 128-dim, 4 heads, 32-token KV cache |
| TaskFamilyTrait impl for LinearFamily | `crates/lab/src/families/linear.rs` | implemented — family_name, generator_version, generate_tasks |
| TaskFamilyTrait impl for LutProjectionFamily | `crates/lab/src/families/lut_projection.rs` | implemented — family_name, generator_version, generate_tasks |
| TaskFamilyTrait impl for DecodeStepFamily | `crates/lab/src/families/decode_step.rs` | implemented — family_name, generator_version, generate_tasks |
| Decode step baseline | `crates/lab/src/baseline.rs` (`compute_decode_step`) | implemented, host-verified — models QKV projection + simplified attention + output projection |
| DecodeStepPayload | `crates/ir/src/linear_slice.rs` | implemented — dedicated decode-step bridge payload with command="emit_stateful_decode_step" (Sprint 40) |
| build_decode_step_program | `python/mil_emitter.py` | implemented — constructs MIL program modeling QKV projection + simplified attention + output projection |
| emit_decode_step | `python/mil_emitter.py` | implemented — dedicated decode-step emission path: build → convert → save |
| emit_decode_step bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_decode_step` command |
| Decode-step payload divergence tests | `crates/ir/src/linear_slice.rs` | 7 tests passing — payload creation, rejection of wrong specs, command divergence from linear and LUT, deterministic serialization, function descriptors |
| Decode step task spec tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, invalid divisibility, missing fields |
| Decode step family tests | `crates/lab/src/families/decode_step.rs` | 9 tests passing — generation, determinism, serialization, config, validation, trait dispatch |
| Lab crate tests | `crates/lab/` | 128 tests passing (including decode_step, sharded baseline, and trait dispatch tests) |

**Residual:** The decode step emission path now defaults to the stateful variant (Sprint 40). `compile-full` for DecodeStep tasks dispatches `emit_stateful_decode_step` which uses real `mb.read_state` / `mb.coreml_update_state` for KV-cache state semantics (iOS 18+). The stateless path (`emit_stateless_decode_step`, using `mb.const` for KV cache) remains available for single-step testing. The multi-function package's decode_step function also uses the stateful variant (Sprint 40). All emitter paths verified on coremltools 9.0.

### Device Profiling System (Sprint 6)

| Component | Code | Status |
|-----------|------|--------|
| DeviceMetadata | `crates/lab/src/device_meta.rs` | implemented, host-verified — MetadataSource enum makes host-only vs device-backed structurally distinct |
| RunType (warm/cold) | `crates/lab/src/device_meta.rs` | implemented — Cold / Warm { warmup_iterations } |
| ExecutionContext | `crates/lab/src/device_meta.rs` | implemented — only exists for device-backed runs |
| TimingResult | `crates/lab/src/harness.rs` | implemented — p50/p90/p99/min/max/mean/stddev + scope_note |
| FallbackDetector | `crates/lab/src/fallback.rs` | implemented, host-verified — deliberately weak, honest suspicion model |
| FallbackSuspicionLevel | `crates/lab/src/harness.rs` | implemented — Unavailable / LowConfidenceSuspicion / NoConclusion |
| SuspicionEvidence | `crates/lab/src/harness.rs` | implemented — kind, description, strength (0.0-1.0 weak signal) |
| Profile bridge command | `python/bridge.py` (`handle_profile`) | implemented — warmup + measured iterations, timing stats, honest unavailability |

### Numerical Drift System (Sprint 7)

| Component | Code | Status |
|-----------|------|--------|
| BaselineComputer | `crates/lab/src/baseline.rs` | implemented, host-verified — deterministic FP32 reference computation for linear projection |
| BaselineResult | `crates/lab/src/baseline.rs` | implemented, host-verified — versioned (1.0.0), serializable, linked to task_hash |
| DriftDetector | `crates/lab/src/drift.rs` | implemented, host-verified — max_abs, mean_abs, rmse, cosine_distance, relative_error_p99 |
| DriftReport | `crates/lab/src/drift.rs` | implemented, host-verified — versioned (1.0.0), DriftComputationStatus prevents misreading unavailable as computed |
| DriftComputationStatus | `crates/lab/src/drift.rs` | implemented — Computed / Unavailable / LengthMismatch / EmptyInput |
| Knowledge update with drift | `crates/cli/src/main.rs` (`build_knowledge_update_with_drift`) | implemented, host-verified — version 3 with PrecisionHazard observation, baseline_provenance, drift_evidence |
| Baseline artifact in lab runs | `crates/lab/src/run_dir.rs` | implemented — baseline.json written in lab run directory |
| Drift artifact in lab runs | `crates/lab/src/run_dir.rs` | implemented — drift.json written in lab run directory |

### Knowledge Store System (Sprint 8)

| Component | Code | Status |
|-----------|------|--------|
| KnowledgeStore | `crates/knowledge/src/store.rs` | implemented, host-verified — file-backed store with seeds/ and observations/ separation |
| KnowledgeEntry | `crates/knowledge/src/store.rs` | implemented — typed entry with unit, provenance, source, conflict_status, revision |
| EntryProvenance | `crates/knowledge/src/store.rs` | implemented — origin, inserted_at, updated_at, source_path |
| EntrySource enum | `crates/knowledge/src/store.rs` | implemented — Seed / Observation (structurally distinct) |
| ConflictStatus enum | `crates/knowledge/src/store.rs` | implemented — NoConflict / ConflictedWith / Resolved |
| UpdatePipeline | `crates/knowledge/src/update.rs` | implemented, host-verified — validates + ingests observations |
| KnowledgeQuery | `crates/knowledge/src/query.rs` | implemented — type, confidence, evidence_source, scope filters |
| KnowledgeQueryable trait | `crates/knowledge/src/query.rs` | implemented for KnowledgeStore — query + query_best |
| ConflictDetector | `crates/knowledge/src/conflict.rs` | implemented, host-verified — ContradictoryLegality, ConfidenceDivergence detection |
| SyntheticTransfer | `crates/knowledge/src/transfer.rs` | implemented, host-verified — transfer safety + confidence scaling + validation |
| SnapshotExport / SnapshotImport | `crates/knowledge/src/snapshot.rs` | implemented, host-verified — JSON export/import with validation |
| Store schema version | `crates/knowledge/src/store.rs` | 1.0.0 |
| Knowledge crate tests | `crates/knowledge/` | 44 tests passing |
| IR crate tests | `crates/ir/` | 48 tests passing (5 shard pipeline + 3 task spec + 3 concrete handoff + 4 payload dtype override + 1 precision override full pipeline + 1 shard payload roundtrip + 3 decode step task spec + 6 LUT payload divergence tests + 3 ShardedDecodeStep task spec tests + 3 MLP block task spec tests + 5 MLP block payload divergence tests + 1 MlpBlock SIR + 1 MlpBlock MIR + 1 MlpBlock linear projection parse + ...) |
| Passes crate tests | `crates/passes/` | 27 tests passing (2 knowledge influence + 7 precision policy + 4 shard plan + 3 mil_lower precision override + 1 precision override SIR→AIR→MIR + 7 shard plan compute unit adaptation + 3 generalized pipeline spec tests) |

### Shard Role Model (Sprint 9, S9.1)

| Component | Code | Status |
|-----------|------|--------|
| ShardRole (5 variants) | `crates/ir/src/pir.rs` | implemented — Io, Entry, Interior, Exit, Sampler |
| PackageRole | `crates/ir/src/pir.rs` | implemented — IO, DecoderShard(ShardRole), Sampler with from_shard_role/to_shard_role |
| ShardRole helpers | `crates/ir/src/pir.rs` | implemented — from_str_flexible, is_ane_targeted, is_cpu_gpu, is_decoder_shard, canonical_name, default_compute_units |
| ComputeUnits helpers | `crates/ir/src/pir.rs` | implemented — from_str_flexible, to_coreml_string |
| ShardTemplate (extended) | `crates/ir/src/pir.rs` | implemented — includes io_compute_units, sampler_compute_units, state_config, context_length |

### Shard-Aware Compilation Path (Sprint 9, S9.2, Sprint 23)

| Component | Code | Status |
|-----------|------|--------|
| TaskOp::ShardedLinearPipeline | `crates/ir/src/task_spec.rs` | implemented, host-verified — input_dim, hidden_dim, output_dim, batch_size, dtype |
| TaskOp::ShardedDecodeStep | `crates/ir/src/task_spec.rs` | implemented — embed_dim, num_heads, head_dim, kv_len, batch_size, dtype (Sprint 23) |
| ShardedLinearPipeline TOML parsing | `crates/ir/src/task_spec.rs` | implemented, host-verified — [synthetic.sharded_linear_pipeline] section |
| ShardedDecodeStep TOML parsing | `crates/ir/src/task_spec.rs` | implemented — [synthetic.sharded_decode_step] section (Sprint 23) |
| ShardDesc | `crates/ir/src/linear_slice.rs` | implemented — shard descriptor with role, dimensions, compute_units |
| ShardSpec | `crates/ir/src/pir.rs` | implemented — generalized shard descriptor using TensorSpec for I/O (Sprint 23) |
| ShardPipelineSpec | `crates/ir/src/pir.rs` | implemented — generalized multi-shard pipeline specification (Sprint 23) |
| sharded_pipeline_shards() | `crates/ir/src/linear_slice.rs` | implemented, host-verified — produces 3 shards: Entry, Interior, Exit |
| lower_shard_to_mir() | `crates/ir/src/linear_slice.rs` | implemented, host-verified — per-shard MIR lowering |
| ShardedShardPayload | `crates/ir/src/linear_slice.rs` | implemented, host-verified — per-shard bridge payload with shard_role |
| build_sharded_pipeline_pir() | `crates/ir/src/linear_slice.rs` | implemented, host-verified — 3-package PIR with handoffs and shard template |
| CLI compile-sharded subcommand | `crates/cli/src/main.rs` | implemented, host-verified — emits one mlpackage per shard |
| build_sharded_manifest() | `crates/cli/src/main.rs` | implemented — manifest v0.4.0 with role semantics and handoffs |
| Sharded pipeline benchmark | `benchmarks/synthetic/sharded_linear_pipeline.toml` | implemented — 64→48→48→32 fp16 pipeline |
| Shard-aware IR tests | `crates/ir/src/linear_slice.rs` | 5 tests passing — shard count, MIR, PIR, payload, rejection |
| Shard-aware parsing tests | `crates/ir/src/task_spec.rs` | 3 tests passing — parsing, error, backward compat |

**Residual:** The shard-aware path emits one mlpackage per shard but does not include cross-shard orchestration at runtime. IO and sampler models are not part of the synthetic pipeline. Inter-shard handoffs are modeled in PIR but not executed. These are expected v0 limitations.

### Shard Template Seed Loading (Sprint 9, S9.3)

| Component | Code | Status |
|-----------|------|--------|
| ShardTemplateSeedFile | `crates/knowledge/src/shard_template.rs` | implemented — typed JSON deserialization for seed files |
| ShardTemplateSeedEntry | `crates/knowledge/src/shard_template.rs` | implemented — typed seed entry with partition_spec, io_model, sampler, quality_delta, scope |
| ValidatedShardTemplate | `crates/knowledge/src/shard_template.rs` | implemented — validated and converted to PIR ShardTemplate |
| load_shard_template_seeds() | `crates/knowledge/src/shard_template.rs` | implemented, host-verified — loads and validates all seed files from directory |
| load_shard_template_seed_file() | `crates/knowledge/src/shard_template.rs` | implemented, host-verified — loads single seed file |
| Seed validation | `crates/knowledge/src/shard_template.rs` | implemented — rejects empty IDs, invalid roles, inverted ranges, bad confidence |
| CLI query subcommand | `crates/cli/src/main.rs` | implemented, host-verified — queries knowledge store with type/confidence/source filters |
| CLI import subcommand | `crates/cli/src/main.rs` | implemented, host-verified — imports shard template seeds and snapshots |

### Pass Pipeline (compile-full path)

The pass pipeline is now wired into the `compile-full` CLI subcommand. The active
fast-path `compile` command still uses direct SIR→MIR lowering.

| Pass | Code | Status |
|------|------|--------|
| CanonicalizePass | `crates/passes/src/canonicalize.rs` | wired (pass-through for linear projection) |
| StaticizePass | `crates/passes/src/staticize.rs` | wired (pass-through for linear projection) |
| PrecisionPolicyPass | `crates/passes/src/precision_policy.rs` | wired (knowledge-affecting: overrides fp16 to fp32 when precision hazard known) |
| StateTopologyPass | `crates/passes/src/state_topology.rs` | wired (pass-through for linear projection) |
| LegalityRewritePass | `crates/passes/src/legality_rewrite.rs` | wired (SIR→AIR, knowledge-informed legality confidence) |
| RiskAnnotatePass | `crates/passes/src/risk_annotate.rs` | wired (AIR→AIR, knowledge-informed risk scores) |
| ShardPlanPass | `crates/passes/src/shard_plan.rs` | wired (single-shard plan + PIR, knowledge-affecting: overrides CPU_AND_NE to CPU_AND_GPU when fallback risk is high) |
| MilLowerPass | `crates/passes/src/mil_lower.rs` | wired (AIR→MIR, real lowering) |

### What Is Verified in This Environment

**Host-verified (Rust compiles and runs):**
- `cargo build` passes with zero warnings, zero errors
- CLI binary `ane-compile` builds successfully
- Task spec loading, SIR construction, MIR lowering all compile
- LabRun schema serializes/deserializes correctly
- LabRunWriter creates valid directory structures
- HostInspector performs Rust-side file checks
- FallbackDetector assesses suspicion honestly
- BaselineComputer produces deterministic FP32 reference outputs
- DriftDetector computes drift metrics (unavailable on non-Apple HW)
- Task generation produces deterministic linear-family task specs
- Task generation persist produces TOML files and manifest
- Generator provenance flows through LabRun schema
- Precision policy pass adapts dtype based on knowledge (7 tests passing)

**Python/Core ML verified:**
- `mil_emitter.emit_linear_projection()`: builds MIL `mb.matmul + mb.add`, converts via converter.py, saves `.mlpackage`
- `mil_emitter.emit_mlprogram()`: same pipeline using build_linear_projection_program()
- `bridge.py`: dispatches all 8 commands correctly, version-checked
- `converter.py`, `palettize.py`, `compute_plan.py`: all verified
- Emitted `.mlpackage` structure: `Manifest.json`, `model.mlmodel`, `weights/weight.bin`
- Bridge result JSON structure: all fields present and correctly typed
- Function descriptors: bridge returns correct name/inputs/outputs/stateful
- Host inspect command: performs package presence check, manifest reading, model load attempt, compute plan check

**NOT verified in this environment:**
- CLI `compile` end-to-end run (requires Python + coremltools alongside compiled binary)
- BridgeResult deserialization from Python JSON in Rust
- On-device inference / `predict()` (requires Apple hardware runtime)
- Profile timing capture (requires Apple hardware runtime)

### What Is NOT Verified in This Environment

- CLI `compile` end-to-end run (requires compiled binary + Python + coremltools in same environment)
- CLI `report` end-to-end run (requires compiled binary)
- CLI `lab` end-to-end run (requires compiled binary + Python + coremltools)
- CLI `profile` timing results (requires Apple hardware runtime)
- On-device inference / `predict()` (requires Apple hardware runtime)
- Compute plan inspection on Apple hardware (requires Apple runtime)
- ANE placement verification (requires Apple hardware + runtime)
- Numerical drift measurement — baseline computed host-side; full drift requires predict() on Apple hardware
- Fallback detection with real timing data (requires Apple hardware runtime)

### Python Emission Layer

Real emission logic lives in **`python/mil_emitter.py`**.
`python/bridge.py` is thin dispatch only (reads command JSON, dispatches, writes result JSON).
No duplicate or fake implementation surfaces remain.

**`python/converter.py`** is wired through the bridge and used by `emit_linear_projection`, `emit_mlprogram`, and the `convert` command. It encapsulates the `ct.convert()` step.

**`python/palettize.py`** is wired through the bridge and callable via the `palettize` command.

**`python/compute_plan.py`** is a real implementation (not a stub) that calls `MLComputePlan.load_from_path()`. On non-Apple platforms, it reports the reason for unavailability.

**`python/profiler.py`** is wired through bridge.py via the `profile` command. The `handle_profile()` function in bridge.py delegates to `profiler.profile_model()` for the actual warmup and measurement loop, then converts the result format. On non-Apple hardware, the bridge returns an honest error.

### Bridge Versioning

The bridge payload includes a `bridge_version` field (currently `1`). Python checks this on receipt and rejects incompatible versions with a clear error. This prevents silent misinterpretation when Rust and Python are built from different commits. Defined in `crates/ir/src/linear_slice.rs` (`BRIDGE_VERSION`) and checked in `python/bridge.py` (`EXPECTED_BRIDGE_VERSION`).

### Bridge Commands

The Python bridge now dispatches 10 commands:

| Command | Description | Apple HW Required |
|---------|-------------|-------------------|
| `emit_linear_projection` | Build MIL program, convert, save mlpackage | No |
| `emit_lut_projection` | Build LUT gather-based MIL program, convert, save mlpackage | No |
| `emit_decode_step` | Build decode-step MIL program (QKV+attention+output projection), convert, save mlpackage | No |
| `emit_stateful_decode_step` | Build stateful decode-step with real KV-cache state (mb.read_state/mb.coreml_update_state), convert, save mlpackage | No |
| `emit_shard_decode_step` | Build shard-role-aware decode-step (Entry/Interior/Exit produce different dims), convert, save mlpackage | No |
| `emit_palettized_linear_projection` | Build linear projection then apply real coremltools palettization, save mlpackage | No |
| `emit_mlprogram` | Same as emit_linear_projection (explicit mlprogram path) | No |
| `convert` | Build fresh MIL program with specified settings | No |
| `palettize` | Apply palettization to an existing mlpackage | No |
| `compute_plan` | Inspect compute plan for an mlpackage | No (reports unavailable) |
| `inspect_mlpackage` | Inspect mlpackage structure and contents | No |
| `host_inspect` | Host-side inspection of mlpackage artifacts | No |
| `profile` | Profile an mlpackage with timing | Yes |

### Manifest Truth Fields

The emitted artifact manifest now includes truth fields that prevent it from being misread as proving device/runtime success:

| Field | Current Value | Purpose |
|-------|---------------|---------|
| `implementation_status` | `"host_compiled"` | Distinguishes host-compiled from device-verified |
| `verification_scope` | `"host_compile_only"` | Makes explicit that no device verification was performed |
| `environment_limitations` | `["no_apple_hardware", "ane_placement_not_verified", "no_on_device_predict"]` | Lists specific limitations |
| `task_hash` | `"sha256:<hex>"` | Deterministic identity; same spec -> same hash |

The manifest is built using the typed `ArtifactManifest` struct from `manifest.rs` (not raw JSON).

### Lab Run Schema

Every lab run produces a `LabRun` record (JSON) with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | "1.0.0" |
| `run_id` | string | "run_YYYYMMDD_HHMMSS_<hash_prefix>" |
| `verification_scope` | enum | HostOnlyInspection / HostRuntimeExecution / DeviceBackedExecution |
| `environment` | object | Host OS, runtime availability, compiler version |
| `task_id` | string | Deterministic task hash (sha256:<hex>) |
| `compile_result` | object | Success/error, output path, content hash |
| `inspect_result` | object | Package presence, manifest readable, model loadable |
| `timing` | object or null | Only present for execution runs (not host-only) |
| `fallback_suspicion` | object or null | Weak, honest suspicion assessment |
| `warnings` | array | Accumulated warnings |

Host-only runs always have `timing: null` and `fallback_suspicion.suspicion_level: "unavailable"`.

### Multifunction Seam

**Multifunction package support is schema/seam only — no multifunction mlpackage has been emitted or validated.**

| Layer | Status |
|-------|--------|
| PIR `Package.functions: Vec<FunctionEntry>` | Schema present, not yet exercised through bridge |
| Bridge payload `functions` field | Schema present in Rust `LinearProjectionPayload` |
| Bridge result `function_descriptors: Vec<BridgeFunctionDescriptor>` | Captured in Rust `BridgeResult` |
| Manifest `PackageEntry.functions: Vec<FunctionDescriptor>` | Formalized with `emission_status` ("emitted" or "seam_only") |
| Manifest `ArtifactManifest.task_hash` | Deterministic task identity hash |
| coremltools multifunction save | Explicit seam in converter.py — commented placeholder with residual gap documented |
| Python `_resolve_function_descriptors()` | Schema flows through bridge: payload -> emitter -> result -> manifest |

### Core ML Tools Feature Matrix

| Capability | Repo Status | Details |
|------------|-------------|---------|
| **mlprogram emission** | Supported | `converter.convert_milprogram()` produces mlprogram via ct.convert() |
| **mlpackage save** | Supported | `mil_emitter.save_mlpackage()` writes .mlpackage directories |
| **MIL Builder construction** | Supported | `mil_emitter.build_linear_projection_program()` uses mb.program() |
| **Typed execution (FLOAT16/FLOAT32)** | Supported | `converter.convert_milprogram()` accepts compute_precision parameter |
| **Opset versioning (iOS16/17/18)** | Supported | `converter.convert_milprogram()` accepts opset_version parameter |
| **Compute unit hints** | Supported | Bridge payload and converter accept CPU_AND_NE, CPU_AND_GPU, etc. |
| **Palettization (1/2/3/4/6/8-bit LUT)** | Supported | `palettize.apply_palettization()` wired through bridge |
| **Compute plan inspection** | Supported (host-side) | Reports unavailable on non-Apple platforms |
| **Function descriptors** | Supported (schema + emission) | Typed `FunctionDescriptor` in manifest.rs; flows through bridge |
| **Host-side inspection** | Supported | `HostInspector` + `host_inspect` bridge command |
| **Baseline computation (FP32 reference)** | Supported (host-side) | `BaselineComputer` produces deterministic FP32 reference outputs from task spec |
| **Drift detection** | Implemented (requires Apple HW for actual) | `DriftDetector` computes max_abs/mean_abs/rmse; unavailable on non-Apple HW |
| **Device profiling (timing)** | Implemented (requires Apple HW) | Wired through bridge; honest error on non-Apple |
| **Fallback suspicion** | Implemented (v0, weak) | `FallbackDetector` with Unavailable/LowConfidenceSuspicion/NoConclusion |
| **Stateful models** | Schema only | PIR has `FunctionEntry.stateful`; no stateful emission path |
| **Multifunction packages** | Seam only | Schema in PIR and manifest; placeholder in converter.py; NOT IMPLEMENTED |
| **State type inputs (ct.StateType)** | Schema only | IR has StateRead/StateWrite; no ct.StateType usage in Python |
| **MLComputePlan per-op device assignment** | Not available | Requires Apple hardware |
| **Multifunction save API** | Not implemented | Requires save_multifunction(); seam documented in converter.py |
| **Knowledge store (file-backed)** | Supported (v0) | `KnowledgeStore` with seed/observation separation, query, conflict detection, snapshot export/import |
| **Knowledge store (SQLite)** | Not implemented | Spec calls for SQLite; v0 uses JSON files. Can be swapped later. |
| **ANE placement verification** | Not possible | Apple does not expose per-op ANE assignment in public API |

### Task Identity

A deterministic task hash (`compute_task_hash`) is computed from the spec parameters. This hash is:
- Included in the artifact manifest as `task_hash`
- Included in the knowledge update as `task_hash`
- Included in the lab run record as `task_id`
- Deterministic: same spec -> same hash
- Format: `sha256:<hex>`, consistent with the content_hash convention

### Canonical v0 Task

**Name:** `linear_proj_slice` — documented in `docs/ir_reference.md`
**File:** `benchmarks/synthetic/linear_projection_slice.toml`
**Shapes:** [1,64] -> [1,32] fp16

### Residuals

- **Rust compilation verified**: `cargo check` and `cargo build` pass with zero warnings and zero errors. Binary `ane-compile` builds successfully.
- **No on-device profiling executed**: requires Apple hardware. Profile command is wired and will produce timing data when run on Apple hardware.
- **AIR skipped in fast-path vertical slice**: SIR->MIR direct lowering in linear_slice.rs; the `compile-full` command drives the full pass pipeline (SIR→AIR→MIR).
- **Partitioning**: shard-aware path (compile-sharded) produces 3-shard Entry/Interior/Exit decomposition; single-shard path still available via `compile`.
- **Palettization in vertical slice**: `emit_palettized_linear_projection` now applies real coremltools palettization (`palettize_weights()`) to emitted models. The gather-based LUT projection remains as a separate path (`emit_lut_projection`), clearly labeled as an approximation.
- **Manifest version**: 0.3.0 (single-shard), 0.4.0 (shard-aware with role semantics and handoffs), 0.5.0 (pass-pipeline compile-full path)
- **Lab run schema version**: 1.0.0
- **coremltools 9.0 compatibility**: palettize.py updated for `op_name_configs` API change.
- **Bridge versioning**: payload includes `bridge_version: 1`, Python rejects version mismatches.
- **Baseline schema version**: 1.0.0 (versioned, serializable, linked to task_hash)
- **Drift report schema version**: 1.0.0 (DriftComputationStatus prevents misreading unavailable as computed)
- **Knowledge update version**: 3 (includes drift evidence, baseline_provenance, drift_evidence sections)
- **Drift infrastructure complete but metrics unavailable on non-Apple HW**: predict() requires Core ML runtime. On Apple hardware, the DriftDetector.detect() method will compute real FP32-vs-FP16 drift.
- **Knowledge store schema version**: 1.0.0 (file-backed, not SQLite)
- **Knowledge store wired into CLI**: the `query` and `import` CLI subcommands are now implemented. `query` opens a store, loads seeds, executes filtered queries (type/min_conf/source), and displays results in JSON/markdown/table format. `import` supports shard template seed files and snapshot files.

## What Is Still Scaffold

- `crates/lab/src/families/op_remap.rs` — op_remap family generator is `unimplemented!()` (open)
- `crates/lab/src/families/shape_hostile.rs` — shape_hostile family generator is `unimplemented!()` (open)
- `crates/lab/src/families/shard_survival.rs` — shard_survival family generator is `unimplemented!()` (open)
- Higher-dimension feasibility exploration is still workflow-driven; there is no dedicated frontier-search CLI/report.
- `HandoffKind::StateWriteRead` is still declared but inactive on the active path.
- Stateful KV cache updates currently use `mb.slice_update`, but the broader gap is that AIR/MIR do not yet expose a generic compiler mechanism for representing and selecting among semantically equivalent backend-sensitive formulations; state/buffer update strategy is just the clearest current example.
- Precision adaptation dtype propagation — NOW CLOSED (Sprint 18). `SirMetadata.precision_override` propagates through AIR (`AirNode.precision_override`), MIR (`MirNode.dtype`), and bridge payload (`LinearProjectionPayload::from_spec_with_override` / `ShardedShardPayload::from_shard_with_override`) to the Python emitter. The `compile-full` and `compile-full-sharded` CLI paths wire the adapted dtype into the bridge payload when precision policy made adaptations.
- CLI subcommands: `compile`, `compile-full`, `compile-sharded`, `compile-full-sharded`, `lab`, `profile`, `report`, `package`, `query`, `import`, `generate-tasks` are implemented
- Pass pipeline — wired into `compile-full` and `compile-full-sharded` CLI subcommands, knowledge-informed since Sprint 11
- Shard-aware path (S9.2) emits mlpackages but does not orchestrate cross-shard execution
- Multi-shard orchestration (S17) is at CLI level; pass pipeline remains shard-agnostic per shard

### Knowledge Store ↔ Pass Pipeline Integration (Sprint 11)

| Component | Code | Status |
|-----------|------|--------|
| PassKnowledgeQuery trait | `crates/passes/src/knowledge_query.rs` | implemented — trait with `query_legality()`, `query_risk()`, and `query_precision_hazard()` methods |
| LegalityInfo | `crates/passes/src/knowledge_query.rs` | implemented — ane_legal, confidence, evidence_count, source_id |
| RiskInfo | `crates/passes/src/knowledge_query.rs` | implemented — fallback_risk, drift_risk, confidence, evidence_count, source_id |
| PrecisionHazardInfo | `crates/passes/src/knowledge_query.rs` | implemented — op_pattern, hazardous_dtype, recommended_dtype, confidence, evidence_count, source_id, description |
| NoKnowledge | `crates/passes/src/knowledge_query.rs` | implemented — returns `None` for all queries (equivalent to previous `&()`) |
| StoreKnowledgeQuery | `crates/cli/src/main.rs` | implemented, host-verified — wraps KnowledgeStore, implements PassKnowledgeQuery |
| compile-full knowledge wiring | `crates/cli/src/main.rs` | implemented, host-verified — `--knowledge <dir>` loads seeds into passes |
| Dead EmissionCommand/EmissionResult | `crates/bridge/src/command.rs`, `result.rs` | removed — were dead code with no active path |
| Packager CLI subcommand | `crates/cli/src/main.rs` | implemented, host-verified — `ane-cli package` creates deterministic zips |

### Structural Verification via MLModelStructure (Sprint 34)

| Component | Code | Status |
|-----------|------|--------|
| model_structure.py | `python/model_structure.py` | implemented — MLModelStructure.load_from_path() inspection, MIR-vs-structure comparison, fallback file check |
| inspect_model_structure | `python/model_structure.py` | implemented — walks emitted mlpackage structure on Apple hardware; reports unavailability on non-Apple platforms |
| inspect_model_structure_with_mir_comparison | `python/model_structure.py` | implemented — combined structural inspection + MIR comparison |
| compare_mir_vs_structure (Python) | `python/model_structure.py` | implemented — multiset MIR-vs-structure comparison with op fidelity score |
| fallback_file_structure | `python/model_structure.py` | implemented — weaker file-based heuristics when MLModelStructure is unavailable |
| model_structure bridge command | `python/bridge.py` | implemented — dispatches `model_structure` command with optional MIR comparison |
| mir_compare.rs | `crates/lab/src/mir_compare.rs` | implemented — Rust-side MIR-vs-structure comparison with canonical op name mapping |
| compare_mir_vs_structure (Rust) | `crates/lab/src/mir_compare.rs` | implemented — multiset matching, op fidelity score, missing/extra op reporting |
| mir_to_mil_name | `crates/lab/src/mir_compare.rs` | implemented — canonical MIR→MIL op name mapping (29 ops) |
| mir_ops_for_bridge | `crates/lab/src/mir_compare.rs` | implemented — serializes MIR ops as JSON for bridge model_structure command |
| Structural verification fields | `crates/lab/src/harness.rs` | implemented — InspectionStepResult extended with structure_inspection_available, structure_op_names, op_fidelity_score, missing_ops, extra_ops, inspection_method, etc. |
| HostInspector model_structure call | `crates/lab/src/host_inspect.rs` | implemented — calls Python bridge model_structure command, populates structural fields |
| mir_compare tests | `crates/lab/src/mir_compare.rs` | 6 tests — name mapping, extraction, perfect match, missing ops, extra ops, empty MIR |

**Residual:** MLModelStructure requires macOS with Core ML runtime (unavailable on Linux). On non-Apple platforms, the path reports unavailability and falls back to file-based heuristics. Op fidelity comparison is by op type name only (not by input/output signature). Multi-function verification now exists (Sprint 39) with structural validation of function presence and op counts. Current host verification includes `cargo test --workspace --quiet`.

### Multi-Function Package Support (Sprint 39)

| Component | Code | Status |
|-----------|------|--------|
| build_multifunction_program | `python/mil_emitter.py` | implemented, Python/Core ML verified — constructs 2-function MIL program (embedding + decode_step) |
| emit_multifunction | `python/mil_emitter.py` | implemented, Python/Core ML verified — composed emission path (build → convert → save → validate) |
| validate_multifunction_package | `python/mil_emitter.py` | implemented, Python/Core ML verified — structural validation of multi-function mlpackage |
| convert_multifunction_milprogram | `python/converter.py` | implemented, Python/Core ML verified — converts multi-function MIL program to MLModel |
| emit_multifunction bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_multifunction` command |
| validate_multifunction bridge dispatch | `python/bridge.py` | implemented — dispatches `validate_multifunction` command |
| Multi-function emission verification | coremltools 9.0 on Linux | verified — spec.mlProgram.functions contains 2 functions: 'embedding' (3 ops), 'decode_step' (36 ops) |
| Attention emitter bug fix | `python/mil_emitter.py` | implemented, Python/Core ML verified — fixed `mask=` → `attn_mask=` for coremltools 9.0 `scaled_dot_product_attention` |

**Residual:** The multi-function path produces a real two-function mlpackage. A shared-weight variant exists, but coremltools 9.0 does not deduplicate constants across `add_function()` boundaries, so the shared-weight package is not smaller unless proto-direct serialization is used. Runtime callability testing requires Apple hardware. Rust CLI does not yet dispatch the `emit_multifunction` bridge command. The attention emitter `mask` → `attn_mask` bug was a correctness issue that caused MIL construction failure on coremltools 9.0 — it is now fixed and verified for both `causal=True` and `causal=False` modes. Sprint 40: The decode_step function now uses the stateful variant with real KV-cache state semantics (`mb.read_state` / `mb.coreml_update_state`). Multi-function validation confirms 2 functions with decode_step containing 36 ops. The `read_state` op is confirmed present via fallback file structure inspection.

## Lab Run Directory Layout

A lab run produces the following canonical directory structure:

```
<output_dir>/
  run_<timestamp>_<task_hash_prefix>/
    run.json              — LabRun record (primary artifact)
    manifest.json         — Artifact manifest from compilation
    mir.json              — MIR dump from compilation
    mlpackage/            — The compiled .mlpackage directory
    knowledge/
      update_<task>.json  — Knowledge update from this run
    inspection.json       — Host-side inspection result (if performed)
    timing.json           — Timing result (if profiling was performed)
    fallback.json         — Fallback suspicion result (if assessed)
    baseline.json         — FP32 baseline reference output
    drift.json            — Drift report (baseline vs actual comparison)
```

## Knowledge Store Directory Layout

A knowledge store uses the following canonical directory structure:

```
<store_path>/
  store_index.json      — Store metadata and entry index
  seeds/
    <id>.json           — Seed entries (immutable, loaded from knowledge/*.json)
  observations/
    <id>.json           — Observation entries (learned from runs)
```

### Compute Plan Harvesting (Sprint 35)

| Component | Code | Status |
|-----------|------|--------|
| harvest_compute_plan() | `python/compute_plan.py` | implemented, Python-verified on Linux (reports unavailable gracefully) |
| harvest_to_observations() | `python/compute_plan.py` | implemented, Python-verified — converts per-op placement to SurvivalMatrixEntry observations |
| handle_compute_plan_harvest() | `python/bridge.py` | implemented, Python-verified — bridge dispatch for `compute_plan_harvest` command |
| compute_plan_harvest.json | `python/bridge.py` | implemented, Python-verified — artifact persistence at output_path |
| compute_plan_observations.json | `python/bridge.py` | implemented, Python-verified — observation list persistence |
| ComputePlanObservation | `crates/knowledge/src/lib.rs` | implemented — Rust struct for compute plan observations |
| ingest_compute_plan_observations() | `crates/knowledge/src/update.rs` | implemented — validates and inserts compute plan observations into knowledge store |
| EvidenceSource::ComputePlan | `crates/ir/src/kir.rs` | implemented — evidence source variant with confidence 0.9 |
| query_compute_plan_placement() | `crates/passes/src/knowledge_query.rs` | implemented — trait method for compute plan placement queries |
| ComputePlanPlacementInfo | `crates/passes/src/knowledge_query.rs` | implemented — struct for compute plan placement data |
| Risk annotate compute plan integration | `crates/passes/src/risk_annotate.rs` | implemented — ane_placed=False increases fallback_risk by 0.7 |

**Residual:** MLComputePlan requires macOS with Core ML runtime. On Linux, the Python harvesting path reports unavailable. The Rust-side ingestion and risk annotation integration are compiled and exercised by the workspace test run on this host. End-to-end validation of the full harvesting pipeline (emit → harvest → ingest → pass consumption) still requires Apple hardware with Core ML runtime.

### Critique Bug 3 Fix — compute_unit_hint Propagation (Sprint 35)

| Component | Code | Status |
|-----------|------|--------|
| MilLowerPass compute_unit_hint from shard plan | `crates/passes/src/mil_lower.rs` | implemented — derives hint from ShardPlan.compute_units instead of hardcoding CPUAndNE |
| Shard name propagation | `crates/passes/src/mil_lower.rs` | implemented — MirGraph.shard_name from ShardPlan.shard_names |
| compute_unit_hint tests | `crates/passes/src/mil_lower.rs` | 3 new tests — default CPUAndNE, GPU override, shard name propagation |

**Residual:** This fixes critique.pdf Bug 3 where the compute_unit_hint on MirNode was always CPUAndNE regardless of the shard plan's actual compute unit assignment. Knowledge-driven compute unit adaptation (from ShardPlanPass) now propagates through to MIR nodes. The fix is compiled and covered by the workspace test run on this host; Apple-runtime-backed validation remains separate.

### Sprint 36 — Real State Semantics for Decode / KV Cache Paths

| Component | Code | Status |
|-----------|------|--------|
| AirOp::ScaledDotProductAttention | `crates/ir/src/air.rs` | implemented — query, key, value inputs |
| AirOp::SliceByIndex | `crates/ir/src/air.rs` | implemented — input, begin, end vectors |
| AirOp::Gelu | `crates/ir/src/air.rs` | implemented — input, mode string |
| AirOp::Relu | `crates/ir/src/air.rs` | implemented — input |
| LinearProjection → Conv1x1AsLinear (not MatMul) | `crates/passes/src/legality_rewrite.rs` | implemented — Critique Bug 1 fix |
| AttentionBlock decomposition | `crates/passes/src/legality_rewrite.rs` | implemented — 14 AIR ops (base): Conv1x1AsLinear + SliceByIndex×3 + Reshape×3 + Transpose×3 + SDPA + Reshape + Conv1x1AsLinear. Qwen3 and other models with `has_qk_norm=true` add optional QK-norm (RMSNorm on Q and K before SDPA). `head_dim` is read from config when explicitly set (e.g., Qwen3 head_dim=128), otherwise computed as hidden_size/num_heads. |
| DecodeStep decomposition | `crates/passes/src/legality_rewrite.rs` | implemented — 15 AIR ops including StateReadFixed×2 + StateWriteFixed×2 |
| RMSNorm decomposition | `crates/passes/src/legality_rewrite.rs` | implemented — ReduceMean + Rsqrt + ElementWise::Mul×2 |
| RoPETransform decomposition | `crates/passes/src/legality_rewrite.rs` | implemented — Cos + Sin + ElementWise::Mul×2 + ElementWise::Add |
| Sampler decomposition | `crates/passes/src/legality_rewrite.rs` | implemented — Topk + Softmax + Gather |
| StateReadFixed → MILReadState lowering | `crates/passes/src/mil_lower.rs` | implemented — full SIR→AIR→MIR chain |
| StateWriteFixed → MILCoremlUpdateState lowering | `crates/passes/src/mil_lower.rs` | implemented — full SIR→AIR→MIR chain |
| ScaledDotProductAttention → MIRScaledDotProductAttention lowering | `crates/passes/src/mil_lower.rs` | implemented |
| SliceByIndex → MILSliceByIndex lowering | `crates/passes/src/mil_lower.rs` | implemented |
| Gelu → MILGelu lowering | `crates/passes/src/mil_lower.rs` | implemented |
| Relu → MILCast (approximation) lowering | `crates/passes/src/mil_lower.rs` | implemented |
| Split → MILSplit lowering | `crates/passes/src/mil_lower.rs` | implemented |
| Concat → MILConcat lowering | `crates/passes/src/mil_lower.rs` | implemented |
| Risk annotation patterns for new AIR ops | `crates/passes/src/risk_annotate.rs` | implemented — mb.linear, mb.scaled_dot_product_attention, mb.slice_by_index, mb.gelu, mb.relu, mb.read_state, mb.coreml_update_state |
| AIR op coverage table | `crates/ir/src/air.rs` | implemented — module docstring updated |
| MIR coverage table (31 ops) | `crates/ir/src/mir.rs` | implemented — module docstring updated with AIR→MIR lowering coverage |
| Sprint 36 tests (legality_rewrite) | `crates/passes/src/legality_rewrite.rs` | implemented — test_linear_projection_lowers_to_conv1x1aslinear_not_matmul, decomposition tests for AttentionBlock/DecodeStep/RMSNorm/RoPE/Sampler |
| Sprint 36 tests (mil_lower) | `crates/passes/src/mil_lower.rs` | implemented — 8 new tests for SDPA, SliceByIndex, Gelu, StateRead, StateWrite, Split, Concat lowering |
| SIR→AIR decomposition docs | `docs/ir_reference.md` | implemented — Sprint 36 section with decomposition table |
| AIR→MIR lowering docs | `docs/ir_reference.md` | implemented — Sprint 36 section with lowering table |

**Residual:** The SIR→AIR→MIR pipeline now covers all declared SIR ops. The Python emission path now produces stateful MIL programs with real `mb.read_state` / `mb.coreml_update_state` KV-cache state semantics (S36.3 completed). The `convert_stateful_milprogram()` function removes the `common::canonicalize_inplace_pattern` pass to handle coremltools 9.0's limitation with `coreml_update_state`. The stateful model's `spec.description.state` correctly lists `k_state` and `v_state` as `stateType`. Shard-role-aware emission (S37.1) produces structurally different programs for Entry/Interior/Exit shards. Real palettization path (S38.2) applies coremltools `palettize_weights()` to emitted models. On non-Apple platforms, programs construct and convert but predict() is unavailable. End-to-end runtime validation requires Apple hardware with Core ML runtime.

### Sprint 37 — Shard Emission Becomes Role-Sensitive

| Component | Code | Status |
|-----------|------|--------|
| build_shard_decode_step_program | `python/mil_emitter.py` | implemented, Python/Core ML verified — role-aware decode step with Entry/Interior/Exit dimension differences |
| emit_shard_decode_step | `python/mil_emitter.py` | implemented, Python/Core ML verified — composed emission path for shard-role-aware programs |
| emit_shard-decode-step bridge command | `python/bridge.py` | implemented — dispatch entry for shard-role-aware emission |
| Shard role dimension overrides | `python/mil_emitter.py` | implemented — shard_hidden_dim, shard_num_heads, shard_head_dim, shard_output_dim payload fields |
| Unique content hashes per role | Verified | Entry/Interior/Exit produce different hashes when given different dimensions |
| ShardPlanPass + MilLowerPass in compile-full-sharded | `crates/cli/src/main.rs` | implemented, host-verified — per-shard pipeline now includes ShardPlan and MIL lowering, MIR compute_unit_hint matches shard plan |
| build_sharded_plan_from_spec_with_risk_knowledge | `crates/passes/src/shard_plan.rs` | implemented, host-verified — applies both template and risk-based knowledge at multi-shard plan construction |
| Manifest compute_unit_adaptations and effective_compute_units | `crates/cli/src/main.rs` | implemented, host-verified — shard provenance includes adaptation records and effective compute units |
| Risk-knowledge multi-shard plan tests | `crates/passes/src/shard_plan.rs` | implemented, host-verified — 4 tests: high risk overrides, low risk keeps defaults, risk overrides template, NoKnowledge |

**Residual:** The shard-role-aware path produces structurally different programs, and `ShardedDecodeStep` now dispatches to `emit_shard_decode_step` from the CLI. `ShardedLinearPipeline` still creates LinearProjection sub-tasks per shard by design. MIR compute-unit hints are consistent with the shard plan in the sharded path (S37.3). Knowledge adaptation now reaches multi-shard plan construction (S37.4). Manifests include compute_unit_adaptations per shard (S37.5). The remaining gap is architectural convergence between role-specific MIR and the active emission paths. End-to-end runtime validation still requires Apple hardware.

### Sprint 38 — LUT / Palettization Correctness

| Component | Code | Status |
|-----------|------|--------|
| emit_palettized_linear_projection | `python/mil_emitter.py` | implemented, Python/Core ML verified — real coremltools palettization via palettize_weights() |
| emit_palettized-linear-projection bridge command | `python/bridge.py` | implemented — dispatch entry for real palettization path |
| LUT vs palettization distinction | `python/mil_emitter.py`, `python/bridge.py` | implemented — emission_path distinguishes 'lut_projection' (gather-based) from 'palettized_linear_projection' (real palettization) |
| Palettization audit | `python/mil_emitter.py` | implemented — LUT gather-based emitter documented as approximation; real palettization uses OpPalettizerConfig |

**Residual:** The palettization is applied to a linear projection (simplest model). Applying palettization to decode-step or attention models with state is not yet wired. End-to-end numerical validation requires Apple hardware.

### Sprint 40 — Close the Stateful Decode Split-Brain

| Component | Code | Status |
|-----------|------|--------|
| TaskOp::DecodeStep bridge_command | `crates/ir/src/task_spec.rs` | updated — now returns `"emit_stateful_decode_step"` (was `"emit_decode_step"`) |
| DecodeStepPayload command | `crates/ir/src/linear_slice.rs` | updated — command is `"emit_stateful_decode_step"` (Sprint 40) |
| emit_stateless_decode_step | `python/mil_emitter.py` | implemented — explicit stateless path for single-step testing |
| emit_stateless_decode_step bridge dispatch | `python/bridge.py` | implemented — dispatches `emit_stateless_decode_step` command |
| emit_decode_step bridge routing | `python/bridge.py` | updated — `emit_decode_step` now routes to `emit_stateful_decode_step` (Sprint 40) |
| build_multifunction_program | `python/mil_emitter.py` | updated — decode_step function uses stateful variant with mb.read_state/mb.coreml_update_state |
| emit_multifunction | `python/mil_emitter.py` | updated — uses convert_stateful_milprogram for multi-function stateful decode |
| smoke_test_emitters.py | `scripts/smoke_test_emitters.py` | implemented — verifies all 8 emitter paths produce valid mlpackages |

**Residual:** The Rust workspace test run passes on this host. Runtime verification of state persistence across `predict()` calls still requires Apple hardware.

### Role-Specific Sharding — ShardOpProfile + RoleMirBuilder (Sprint 43)

| Component | Code | Status |
|-----------|------|--------|
| ShardOpProfile enum | `crates/ir/src/pir.rs` | implemented, host-verified — 8 variants: EntryLinear, InteriorLinear, ExitLinear, QkvProjection, AttentionComputation, OutputProjection, IoEmbedding, SamplerTopk, LinearOnly |
| ActivationType enum | `crates/ir/src/pir.rs` | implemented — GeluTanh, Relu, None |
| ShardSpec.op_profile | `crates/ir/src/pir.rs` | implemented — each shard carries its op profile, determining the MIR op sequence |
| RoleMirBuilder | `crates/passes/src/role_mir.rs` | implemented, host-verified — produces genuinely different MIR graphs per ShardOpProfile |
| RoleMirBuilder::op_type_signature | `crates/passes/src/role_mir.rs` | implemented — extracts sorted op type names from a MIR graph for structural comparison |
| three_shard_linear (with op_profile) | `crates/ir/src/pir.rs` | implemented — EntryLinear, InteriorLinear(GeluTanh), ExitLinear profiles |
| three_shard_decode_step (with op_profile) | `crates/ir/src/pir.rs` | implemented — QkvProjection, AttentionComputation, OutputProjection profiles |
| Role divergence tests | `crates/passes/src/role_mir.rs` | 7 tests passing — roles_produce_different_op_structures, entry_has_reshape, interior_has_gelu, exit_has_layernorm, linear_only_backward_compat, decode_step_roles_produce_different_structures, io_and_sampler_use_cpu_gpu |

**Residual:** RoleMirBuilder produces structurally different MIR graphs per role, and the Python shard emitters now also produce structurally different programs per role. The remaining gap is that these two layers are still maintained separately; RoleMirBuilder is not yet the single active source of truth for Python/proto-direct shard emission. End-to-end validation still requires Apple hardware with Core ML runtime.

### Compute Plan Offline Verification (Sprint 43)

| Component | Code | Status |
|-----------|------|--------|
| ComputePlanProof | `crates/knowledge/src/compute_plan_verify.rs` | implemented, host-verified — deterministic, hashable snapshot of op-to-device placement |
| PlacementEntry | `crates/knowledge/src/compute_plan_verify.rs` | implemented — op_name, op_type, device_class, function_name |
| DeviceClass enum | `crates/knowledge/src/compute_plan_verify.rs` | implemented — NeuralEngine, CPU, GPU, Unknown |
| VerificationResult | `crates/knowledge/src/compute_plan_verify.rs` | implemented — is_valid, errors, warnings, knowledge_consistent, knowledge_matches, knowledge_conflicts, ane_utilization, proof_hash |
| KnownOpPlacement | `crates/knowledge/src/compute_plan_verify.rs` | implemented — op_pattern, expected_device, confidence, source |
| ComputePlanVerifier | `crates/knowledge/src/compute_plan_verify.rs` | implemented, host-verified — verifies structural validity, invariant compliance, knowledge consistency |
| ComputePlanVerifier::predict_proof | `crates/knowledge/src/compute_plan_verify.rs` | implemented — generates predicted compute plan from op lists using known mappings |
| compute_proof_hash | `crates/knowledge/src/compute_plan_verify.rs` | implemented — SHA-256 hash of sorted placement entries (order-independent, deterministic) |
| Compute plan verification tests | `crates/knowledge/src/compute_plan_verify.rs` | 10 tests passing — valid_proof_verifies, tampered_hash_detected, wrong_op_count_detected, wrong_ane_count_detected, placement_mismatch_with_knowledge, predict_proof, proof_hash_deterministic, proof_hash_order_independent, decoder_shard_proof_verifies, predict_verify_roundtrip |

**Residual:** Compute plan offline verification proves structural properties on any platform, but cannot verify actual runtime placement without Apple hardware. The predicted compute plans use conservative known op-to-device mappings; real placements may differ across hardware generations. Integration with the CLI (e.g., `ane-cli verify-compute-plan` command) is not yet wired.

### Weight Sharing Deduplication Metrics (Sprint 43)

| Component | Code | Status |
|-----------|------|--------|
| WeightBinBuilder.dedup_count | `crates/coreml-emit/src/weights.rs` | implemented, host-verified — counts deduplication events |
| WeightBinBuilder.dedup_bytes_saved | `crates/coreml-emit/src/weights.rs` | implemented, host-verified — total bytes saved by deduplication |
| WeightBinResult.deduplicated_count | `crates/coreml-emit/src/weights.rs` | implemented, host-verified — reported in build() output |
| WeightBinResult.deduplicated_bytes | `crates/coreml-emit/src/weights.rs` | implemented, host-verified — reported in build() output |
| Deduplication metrics test | `crates/coreml-emit/src/weights.rs` | 4 tests passing — deduplication_metrics_tracked, no_deduplication_zero_metrics, content_hash_deduplication, multifunction_weight_sharing_saves_space |

**Residual:** Weight deduplication metrics are now tracked and proven via tests. The proto-direct path can measure and report exactly how much weight data was saved by deduplication — a capability that coremltools 9.0 lacks. Content-hash deduplication (different names, identical content) is documented but not yet implemented; currently deduplication is name-based only.

### External Usability — README Rewrite (Sprint 43)

| Component | Code | Status |
|-----------|------|--------|
| README.md | `README.md` | implemented — major rewrite with architecture diagram, workspace crate table, key capabilities section, verification honesty table, directory layout, design decisions |
