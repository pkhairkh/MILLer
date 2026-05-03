# ♉ AUDIT.md — TABULA RASA Full-Spectrum Diagnostic

**Date:** 2026-05-03  
**Scope:** Full repository sweep — 12 crates, 660 tests, 9 constraint documents  
**Method:** Automated lint + manual source walk + canon cross-reference  

---

## I. EXECUTIVE JUDGEMENT

The MILLer compiler lattice is architecturally sound at its foundation but riddled with faithfulness gaps between the ANE constraint canon and the Rust enforcement code. The core IR family (SIR→AIR→MIR→PIR) is complete and well-structured: 167 MIL ops are represented across all four IRs, the dependency graph is a clean DAG with `ane-ir` as sole root, and all 660 tests pass. The `ane-passes` crate is the largest and most critical, containing 18 compilation passes that collectively implement the constraint enforcement pipeline. The knowledge store, lab, and bridge modules provide supporting infrastructure that is functionally adequate but under-tested in key areas.

However, the audit reveals a systematic disconnection between three sources of truth: `MirOp::default_engine()` (which assigns ANE engines to ops), the `CPU_ONLY_OPS` list (which declares ops CPU-exiled), and the `MirOpCompat` enum (which determines what can actually be emitted). These three have drifted apart catastrophically — 55 ops are assigned ANE engines but have no compat converter, and 4 ops are simultaneously on the CPU-only list AND assigned ANE engines. The placement validator does not consult the CPU-only list at all, relying solely on `default_engine().is_none()`, making the CPU-only list effectively dead code in the validation path. Additionally, interleave constraints and dtype constraints are implemented but never wired into the compilation pipeline, leaving them as dead validation code that looks correct but enforces nothing.

The codebase also carries significant Qwen-3 specific leakage: hardcoded vocabulary sizes (32000 vs Qwen3's 151936), hardcoded head dimensions (128), model-specific weight name patterns in the alias builder, and shape inference fallbacks that assume Qwen3-0.6B dimensions. These leaks are not catastrophic — the compiler still works for Qwen3 — but they will cause silent miscompilation for any other model architecture. Combined with 55 ops that will silently fail at emission time and 9 missing constraint validators, the compiler is correct for its primary target but fragile and hostile to extension. The recommended path forward is to first align the three sources of truth (engine assignment, CPU-only list, compat coverage), then wire the dead constraint validators into the pipeline, and finally parameterize the model-specific constants.

---

## II. ANE-CONSTRAINT VIOLATIONS

### II-A. Ops Absent from CPU-Only List but With No ANEC Converter (55 ops)

These ops are assigned ANE engines in `default_engine()` but map to `MirOpCompat::Unsupported` — they will pass placement validation as ANE-legal but silently fail at emission time.

| # | MirOp Variant | Assigned Engine | Compat Status | Severity |
|---|---|---|---|---|
| 1 | `MILEinsum` | NE | Unsupported | HIGH |
| 2 | `MILConvTranspose` | NE | Unsupported | HIGH |
| 3 | `MILLogicalXor` | PE | Unsupported | HIGH |
| 4 | `MILRelu6` | PE | Unsupported | MEDIUM |
| 5 | `MILSigmoidHard` | PE | Unsupported | MEDIUM |
| 6 | `MILThresholdedRelu` | PE | Unsupported | MEDIUM |
| 7 | `MILClampedRelu` | PE | Unsupported | MEDIUM |
| 8 | `MILLinearActivation` | PE | Unsupported | MEDIUM |
| 9 | `MILScaledTanh` | PE | Unsupported | MEDIUM |
| 10 | `MILElu` | PE | Unsupported | MEDIUM |
| 11 | `MILSoftplusParametric` | PE | Unsupported | MEDIUM |
| 12 | `MILSquare` | PE | Unsupported | MEDIUM |
| 13 | `MILThreshold` | PE | Unsupported | MEDIUM |
| 14 | `MILInverse` | PE | Unsupported | MEDIUM |
| 15 | `MILExp2` | PE | Unsupported | MEDIUM |
| 16 | `MILReduceSumSquare` | PE | Unsupported | MEDIUM |
| 17 | `MILReduceL2Norm` | PE | Unsupported | MEDIUM |
| 18 | `MILReduceL1Norm` | PE | Unsupported | MEDIUM |
| 19 | `MILReduceLogSumExp` | PE | Unsupported | MEDIUM |
| 20 | `MILReduceLogSum` | PE | Unsupported | MEDIUM |
| 21 | `MILReduceArgmax` | PE | Unsupported | HIGH |
| 22 | `MILReduceArgmin` | PE | Unsupported | HIGH |
| 23 | `MILBatchNorm` | PE | Unsupported | HIGH |
| 24 | `MILInstanceNorm` | PE | Unsupported | HIGH |
| 25 | `MILL2Norm` | PE | Unsupported | MEDIUM |
| 26 | `MILLocalResponseNorm` | PE | Unsupported | MEDIUM |
| 27 | `MILMaxPool` | NE | Unsupported | HIGH |
| 28 | `MILAvgPool` | NE | Unsupported | HIGH |
| 29 | `MILL2Pool` | NE | Unsupported | MEDIUM |
| 30 | `MILResize` | NE | Unsupported | MEDIUM |
| 31 | `MILResizeNearestNeighbor` | NE | Unsupported | MEDIUM |
| 32 | `MILResizeBilinear` | NE | Unsupported | MEDIUM |
| 33 | `MILUpsampleNearestNeighbor` | NE | Unsupported | MEDIUM |
| 34 | `MILUpsampleBilinear` | NE | Unsupported | MEDIUM |
| 35 | `MILCropResize` | NE | Unsupported | MEDIUM |
| 36 | `MILAffine` | NE | Unsupported | MEDIUM |
| 37 | `MILResample` | NE | Unsupported | MEDIUM |
| 38 | `MILReshapeLike` | PE | Unsupported | LOW |
| 39 | `MILFlatten2d` | PE | Unsupported | LOW |
| 40 | `MILReverse` | PE | Unsupported | MEDIUM |
| 41 | `MILSliceBySize` | PE | Unsupported | MEDIUM |
| 42 | `MILSlidingWindows` | PE | Unsupported | LOW |
| 43 | `MILDepthToSpace` | NE | Unsupported | MEDIUM |
| 44 | `MILSpaceToDepth` | NE | Unsupported | MEDIUM |
| 45 | `MILPixelShuffle` | NE | Unsupported | MEDIUM |
| 46 | `MILPixelUnshuffle` | NE | Unsupported | MEDIUM |
| 47 | `MILBatchToSpace` | NE | Unsupported | MEDIUM |
| 48 | `MILSpaceToBatch` | NE | Unsupported | MEDIUM |
| 49 | `MILStack` | PE | Unsupported | MEDIUM |
| 50 | `MILArgsort` | PE | Unsupported | MEDIUM |
| 51 | `MILCrop` | PE | Unsupported | LOW |
| 52 | `MILQuantize` | PE | Unsupported | HIGH |
| 53 | `MILDequantize` | PE | Unsupported | HIGH |
| 54 | `MILPrelu` | PE | Unsupported | MEDIUM |
| 55 | `MILSoftsign` | PE | Unsupported | MEDIUM |

### II-B. Ops on CPU-Only List but Still Routable to ANE (4 ops)

These ops are listed in `cpu_only_ops.rs` but `default_engine()` returns a non-None ANE engine, and `placement_validate.rs` does not check the CPU-only list.

| Op | CPU-Only? | `default_engine()` | File | Severity |
|---|---|---|---|---|
| `MILBandPart` | Yes | `Some(PE)` | `mir.rs:1201` | HIGH |
| `MILLogicalAnd` | Yes | `Some(PE)` | `mir.rs:1105` | HIGH |
| `MILLogicalOr` | Yes | `Some(PE)` | `mir.rs:1107` | HIGH |
| `MILLogicalNot` | Yes | `Some(PE)` | `mir.rs:1151` | HIGH |

### II-C. Missing Constraint Validators (9 gap areas)

| Constraint | Canon Reference | Implemented? | File | Severity |
|---|---|---|---|---|
| Padding: no replication/symmetric/negative/batch/channel/depth | §4.13 | No validator exists | `op_constraints.rs` | HIGH |
| Interleave: {1,2,3,4,8}, C-axis only, Int4→8 | §6.3 | Implemented but NOT wired | `ane_layout.rs` | HIGH |
| Dtype: FP16 primary, Int4 per-cout dequant rejected | §5 | Implemented but NOT wired | `dtype_constraints.rs` | HIGH |
| Conv: kernel W/H/D power-of-2, stride=1 B/C, dilation=1 B/C | §4.1 | Partial (range only) | `op_constraints.rs` | MEDIUM |
| MatMul: depth=1 both inputs | §4.3 | No validator exists | — | HIGH |
| Pool: W/H/D multiple of stride | §4.8 | Not checked | `op_constraints.rs` | MEDIUM |
| Resize: alignCorners+centerResult on A14 | §4.10 | No validator exists | — | MEDIUM |
| Broadcast: depth-axis rejection, FP16 A11/A12 | §5 | No validator / soft flag | — | MEDIUM |
| Transpose: interleave C-axis only, TransposeNC C=1 | §4.12 | No validator exists | — | MEDIUM |

### II-D. Missing Family-Version Guards

| Guard | Canon Reference | Status | Severity |
|---|---|---|---|
| ArgMinMax blocked on A18 (LSE_7 no converter) | §4 op-support | NOT implemented | HIGH |
| Elementwise binary A14Plus vs A14Minus split | §4 op-support | NOT implemented | MEDIUM |
| Broadcast FP16-only on A11/A12 | §5 | "Soft constraint" — not enforced | MEDIUM |

### II-E. Shape Vector Placeholder Zeros

| Location | Pattern | Risk | Severity |
|---|---|---|---|
| `legality_rewrite.rs:464-465` | Tile reshape/final shape use `0` placeholders | Zeros survive to Core ML if shape inference fails | HIGH |
| `legality_rewrite.rs:2194,2282` | Attention reshape uses `0` for batch dim | Same risk | MEDIUM |
| `mir_to_compat.rs:2591-2599` | Test asserts zeros survive to emission | Core ML treats 0 as literal zero dimension | HIGH |

---

## III. CODE-QUALITY FINDINGS

| # | Smell | Location | Suggestion | Severity |
|---|---|---|---|---|
| CQ-1 | `clippy::modulo_one` (deny) | `bridge/mir_to_compat.rs:1265,1271` | Replace `% 1` with correct divisor (likely `product_so_far`); remove `/ 1` on lines 1266/1272 | HIGH |
| CQ-2 | `clippy::eq_op` (deny) | `passes/mil_lower.rs:3796` | Remove redundant `perm == &[0usize,2,1,3]` second operand | HIGH |
| CQ-3 | `.unwrap()` on reshape zero-dim search | `passes/mil_lower.rs:220-221` | Return `Result` or use `?` with proper error type | HIGH |
| CQ-4 | `.expect()` on PIR package lookup | `ir/linear_slice.rs:321,326` | Return `Result` — missing package is a user error, not a bug | MEDIUM |
| CQ-5 | 5 `panic!()` in proto validation | `coreml-proto/src/lib.rs:4236,4255,4264,4375,4474` | Return `Result<(), ProtoValidationError>` — dtype/shape mismatches are user-facing errors | MEDIUM |
| CQ-6 | `if_same_then_else` in shard_plan | `passes/shard_plan.rs:312-320` | Three if-branches produce identical results — confirm intentional or fix | MEDIUM |
| CQ-7 | 8 `too_many_arguments` | `passes/legality_rewrite.rs:110,164,927,1373,2176,2311,2754,2867` | Refactor into builder pattern or config struct (worst: 16 args at line 1373) | LOW |
| CQ-8 | 19 `unnecessary_cast` | `passes/legality_rewrite.rs` (12), `bridge/shape_inference.rs` (7) | Remove `as usize` on already-`usize` variables | LOW |
| CQ-9 | 49 files unformatted | All crates except coreml-ffi | Run `cargo fmt` | LOW |
| CQ-10 | 85 clippy warnings total | 6 crates | ~57 auto-fixable via `cargo clippy --fix` | LOW |
| CQ-11 | Deprecated `kv_cache_rewrite` still `pub` | `passes/src/lib.rs:21` | Make `pub(crate)` or remove entirely | LOW |
| CQ-12 | Chip comments wrong | `ir/ane_target.rs:11-22` | A11≠M1, A12≠M2, A14≠M3 — fix comments | LOW |
| CQ-13 | V6 (A12 silicon) mapped to A14 family | `ir/ane_target.rs:68` | A12 gets wrong broadcast/LayerNorm/SDPA gates | HIGH |
| CQ-14 | V17 (M1) mapped to A18 family | `ir/ane_target.rs:71-73` | M1 is A14-class, gets A18's SDPA/LayerNorm gates | MEDIUM |
| CQ-15 | `MilDtype` missing Int4, UInt4, E4M3, E5M2 | `ir/common.rs:15-24` | Cannot enforce Int4-per-cout-dequant or float8 rules | HIGH |
| CQ-16 | Model-specific `vocab_size=32000` default | `passes/role_mir.rs:640-642,1103` | Read from TaskSpec instead | MEDIUM |
| CQ-17 | Model-specific `head_dim=128` fallback | `passes/legality_rewrite.rs:1077,1876,2325` | Error rather than silently using wrong value | MEDIUM |
| CQ-18 | Qwen3 weight name patterns in alias builder | `bridge/mir_to_compat.rs:510` | Parameterize with model-architecture callback | MEDIUM |
| CQ-19 | `vec![1, 512]` shape inference fallback | `bridge/shape_inference.rs:54-58` | Parameterize from TaskSpec | MEDIUM |
| CQ-20 | `MirOpCompat` dual definition (167 vs ~50 variants) | `coreml-proto/src/lib.rs` | Implement `ToProto` trait per T-17 roadmap | HIGH |
| CQ-21 | ~1150 lines per-variant match-arm boilerplate | `bridge/mir_to_compat.rs` | Replace with derive macro or visitor pattern | MEDIUM |
| CQ-22 | Dual shape inference in two crates | `passes/mil_lower.rs`, `bridge/shape_inference.rs` | Extract shared module to `ane-ir` | MEDIUM |
| CQ-23 | SDPA compat missing `attention_mask` and `scale` | `coreml-proto/src/lib.rs` | Add fields to `MirOpCompat::ScaledDotProductAttention` | HIGH |
| CQ-24 | Proto-direct path cannot emit palettized weights | `coreml-proto/src/lib.rs` | Add ConstexprLutToDense etc. to MirOpCompat | HIGH |

---

## IV. BUG REPORT

| # | Symptom | Trigger | Fix Direction | Severity |
|---|---|---|---|---|
| B-1 | Silent emission failure for 55 ops | Any model using ops outside the ~50 MirOpCompat variants (e.g., ConvTranspose, pooling, quantize/dequantize, batch/instance norm) | Align `default_engine()`, CPU_ONLY list, and compat coverage; add `is_cpu_only()` check to `placement_validate.rs` | HIGH |
| B-2 | CPU-only ops routed to ANE | Ops like `band_part`, `logical_and/or/not` on CPU-only list but `default_engine()` returns PE | Fix `default_engine()` to return `None` for CPU-only ops; add `is_cpu_only()` gate in placement validator | HIGH |
| B-3 | A12 silicon gets wrong constraints | V6 mapped to A14 family → non-FP16 broadcast allowed on A12 hardware | Add `A13` family variant or map V6→A12 family | HIGH |
| B-4 | M1 (A14-class) gets A18 constraints | V17 mapped to A18 family → SDPA/LayerNorm allowed on M1 | Add V17→A14 mapping | MEDIUM |
| B-5 | Reshape `.unwrap()` panics | `mil_lower.rs:220-221` — reshapes with no zero-dim inference target | Replace with proper error handling | HIGH |
| B-6 | Zero-dimension shapes in emitted Core ML | Tile/attention reshape placeholders survive when shape inference fails | Add zero-dimension validation before emission; error if any dim is 0 | HIGH |
| B-7 | ArgMinMax silently fails on A18 | No LSE_7 converter exists; no family guard blocks it | Add A18 guard for ArgMinMax in `placement_validate.rs` | HIGH |
| B-8 | Padding constraints not enforced | Replication/symmetric/negative/batch/channel/depth padding passes validation | Add `validate_pad_constraints()` in `op_constraints.rs` | MEDIUM |
| B-9 | MatMul depth checks missing | Depth>1 inputs pass validation; ANE rejects at runtime | Add `validate_matmul_constraints()` | MEDIUM |
| B-10 | `% 1 == 0` always-true condition | `bridge/mir_to_compat.rs:1265,1271` — likely placeholder for real divisor | Fix divisor to correct value (likely `product_so_far`) | HIGH |
| B-11 | SDPA compat missing mask and scale | RoPE-attention models need mask support; proto-direct path cannot emit | Add `attention_mask` and `scale` fields to `MirOpCompat::ScaledDotProductAttention` | HIGH |
| B-12 | Interleave/dtype validators dead code | `ane_layout.rs` and `dtype_constraints.rs` functions never called from pipeline | Wire into `placement_validate.rs` | HIGH |

---

## V. TEST COVERAGE MAP

| Module | # pub fn | # test fn | Est. Coverage | Priority |
|---|---|---|---|---|
| **passes::staticize** | 2 | 0 | 0% | 🔴 Critical |
| **bridge::shape_inference** | 3 | 0 | 0% | 🔴 Critical |
| **coreml-proto** (whole crate) | 23 | 19 | 35% | 🟠 High |
| **lab** (whole crate) | 152 | 123 | 40% | 🟠 High |
| **passes::palettize_weights** | 1 | 2 | ~30% | 🟠 High |
| **passes::legality_rewrite** | 7 | 19 | ~50% | 🟠 High |
| **knowledge::store** | 17 | 9 | 53% | 🟠 High |
| **knowledge::snapshot** | 5 | 3 | 60% | 🟡 Medium |
| **bridge** (whole crate) | 38 | 39 | 50% | 🟡 Medium |
| **ir** (whole crate) | 90 | 98 | 55% | 🟡 Medium |
| **knowledge** (whole crate) | 55 | 77 | 55% | 🟡 Medium |
| **trace** (whole crate) | 13 | 31 | 55% | 🟡 Medium |
| **passes::canonicalize** | 2 | 3 | ~70% | 🟢 Low |
| **passes::mil_lower** | 3 | 41 | ~70% | 🟢 Low |
| **passes::shard_plan** | 8 | 21 | ~70% | 🟢 Low |
| **coreml-emit** (whole crate) | 21 | 39 | 60% | 🟢 Low |
| **artifacts** (whole crate) | 7 | 13 | 70%+ | 🟢 Low |
| **report** (whole crate) | 12 | 17 | 65% | 🟢 Low |

---

## VI. IR CLEANLINESS SCORE

```
┌─────────────────────────────────────────────┐
│  IR CLEANLINESS SCORE                       │
│                                             │
│  SIR ──████████████████████░░░░░  83%       │
│  AIR ──██████████████████░░░░░░░  77%       │
│  MIR ──█████████████████░░░░░░░░  72%       │
│  PIR ──██████████████████████░░░  90%       │
│                                             │
│  OVERALL: ████████████████░░░░░░  78%       │
│                                             │
│  Deductions:                                │
│  - MirOpCompat gap: -12% (55 unsupported)   │
│  - Dead validators: -5% (not wired)         │
│  - Placeholder zeros: -3% (shape inference) │
│  - Model leaks: -2% (hardcoded constants)   │
└─────────────────────────────────────────────┘
```

---

## VII. RECOMMENDED SPRINT BACKLOG

Sorted by **impact × urgency** (highest first):

| Rank | Task | Impact | Urgency | Effort | Issue Ref |
|---|---|---|---|---|---|
| 1 | **Align three sources of truth**: audit `default_engine()`, `CPU_ONLY_OPS`, and `MirOpCompat` — ensure every op is either ANE-legal with a converter or CPU-only with `default_engine() → None` | CRITICAL | NOW | L (3-5d) | I-01 |
| 2 | **Wire `is_cpu_only()` into `placement_validate.rs`** — add `cpu_only_ops::is_cpu_only()` check before allowing ANE placement | CRITICAL | NOW | S (0.5d) | I-02 |
| 3 | **Fix V6→A14 family mapping** — add A13 variant or map V6 to A12 family so A12 silicon gets correct broadcast/SDPA/LayerNorm gates | HIGH | NOW | M (1d) | I-03 |
| 4 | **Wire interleave + dtype validators into pipeline** — call `validate_interleave_constraints()` and `is_dtype_ane_legal()` from `placement_validate.rs` | HIGH | NEXT | S (0.5d) | I-04 |
| 5 | **Add `validate_matmul_constraints()`** — enforce depth=1 for both inputs | HIGH | NEXT | S (0.5d) | I-05 |
| 6 | **Add `validate_pad_constraints()`** — reject replication/symmetric/negative/batch/channel/depth padding | HIGH | NEXT | M (1d) | I-06 |
| 7 | **Fix reshape `.unwrap()` in `mil_lower.rs:220-221`** — return `Result` with proper error type | HIGH | NEXT | S (0.5d) | I-07 |
| 8 | **Add zero-dimension validation before emission** — reject any MIR shape containing 0 dims | HIGH | NEXT | S (0.5d) | I-08 |
| 9 | **Fix `% 1 == 0` logic bug in `mir_to_compat.rs:1265,1271`** — replace with correct divisor | HIGH | NEXT | S (0.5d) | I-09 |
| 10 | **Add `attention_mask` and `scale` to `MirOpCompat::ScaledDotProductAttention`** | HIGH | NEXT | M (1d) | I-10 |
| 11 | **Add A18 guard for ArgMinMax** in `placement_validate.rs` | MEDIUM | NEXT | S (0.5d) | I-11 |
| 12 | **Add tests for `bridge::shape_inference.rs`** — test `compat_output_shape` for every MirOp variant | MEDIUM | NEXT | M (2d) | I-12 |
| 13 | **Add tests for `passes::staticize.rs`** — even pass-through needs a smoke test | MEDIUM | NEXT | S (0.5d) | I-13 |
| 14 | **Expand `MilDtype` with Int4, UInt4, E4M3, E5M2** | MEDIUM | LATER | M (1d) | I-14 |
| 15 | **Parameterize model-specific constants** — vocab_size, head_dim, input shape from TaskSpec | MEDIUM | LATER | M (2d) | I-15 |
| 16 | **Add SIR→AIR roundtrip test** in legality_rewrite with `DecompositionContext::for_decode_step_full()` | MEDIUM | LATER | M (1d) | I-16 |
| 17 | **Implement `ToProto` trait** to unify MirOp and MirOpCompat (T-17) | MEDIUM | LATER | L (3-5d) | I-17 |
| 18 | **Add ConstexprLutToDense etc. to MirOpCompat** — enable palettized weight emission via proto-direct | MEDIUM | LATER | M (2d) | I-18 |
| 19 | **Fix V17 (M1) → A18 mapping** — M1 is A14-class | MEDIUM | LATER | S (0.5d) | I-19 |
| 20 | **Run `cargo fmt` and `cargo clippy --fix`** — clean up 49 unformatted files and ~57 auto-fixable warnings | LOW | LATER | S (0.5d) | I-20 |
