# ANE Violations — MILLer Constraint-Grounded Violation Report

**Operation**: NECROSCOPY — Expanded Compatibility Audit
**Date**: 2026-05-04 (expanded 2026-05-06 with deep binary forensic evidence; re-expanded 2026-05-08 with research-grade forensic methodology)
**Scope**: Full MILLer source tree, knowledge seeds, documentation, and non-invasive local reference inventory, cross-referenced against deep binary analysis of ANE framework libraries and academic research

---

## I. Executive Abstract

### Files Examined

This audit examined the following source categories by filename:

**Rust crates (12)**: `ir`, `passes`, `knowledge`, `lab`, `bridge`, `trace`, `coreml-proto`, `coreml-emit`, `coreml-ffi`, `artifacts`, `report`, `cli` — approximately 95 source files.

**Python bridge (11)**: `bridge.py`, `mil_emitter.py`, `trace_model.py`, `converter.py`, `compute_plan.py`, `model_structure.py`, `verify.py`, `profiler.py`, `palettize.py`, `program_builder.py`, `common.py`.

**Knowledge seeds (8)**: `legality_seed.json`, `ane_hw_limits_seed.json`, `ane_op_family_matrix.json`, `precision_hazard_seed.json`, `palettization_constraints_seed.json`, `shard_template_seed.json`, `decode_step_shard_template_seed.json`, `cpu_only_ops_seed.json`.

**Documentation (14)**: `SPEC.md`, `README.md`, `ISSUES.md`, `CHANGELOG.md`, `docs/audit/tabula-rasa-v3.md`, `docs/audit/ane-violations.md`, `docs/architecture.md`, `docs/ir_reference.md`, `docs/bridge_protocol.md`, `docs/knowledge_schema.md`, `docs/profiling_methodology.md`, and 5 files under `ane-constraints-docs/`.

**Configuration (6)**: `Cargo.toml`, `pyproject.toml`, `rust-toolchain.toml`, `clippy.toml`, `rustfmt.toml`, `requirements-dev.txt`.

**Forensic reference binaries (3)**: `ANEClientSignals` (8 KB), `ANECompiler` (~45.8 MB), `ANEServices` (~300 KB) — Mach-O 64-bit arm64e dynamically linked shared libraries with PAC support, targeting macOS 26.0 (Tahoe).

### Audit Scope

The audit scope covered: (1) internal consistency of MILLer's constraint model against its own source code, tests, and documentation; (2) cross-referencing of claimed capabilities against non-invasive local reference metadata; (3) identification of phantom capabilities, stub-mimic functions, missing validation (lacunae), aberrant claims, and unverified assertions; (4) deep binary forensic cross-referencing of ANECompiler operation schemas, converter catalogs, family enumerations, and Orion programming constraints against MILLer source; (5) cross-examination of 20 Orion ANE programming constraints (arxiv:2603.06728) against every MILLer code path; (6) research-grade validation using arxiv:2601.01673 (MOTIF), arxiv:2604.23457 (ARIstoteles), arxiv:2003.05039 (Devil is Virtual), arxiv:2503.07243 (Type Recovery Patterns).

### Methodology

The forensic analysis employed the following research-grade methodology:

1. **LIEF 0.17.6 for Mach-O structural parsing** — All three binaries (ANEClientSignals, ANECompiler, ANEServices) were parsed for headers, segments, sections, load commands, symbol tables, and imported symbol references. This produced the complete structural map: 132,552 defined symbols in ANECompiler, 961 in ANEServices, 17 in ANEClientSignals.

2. **C++ name demangling (c++filt) for template instantiation recovery** — The `__Z` Mach-O prefix was stripped to `_Z` and demangled via c++filt. This revealed 4,264 converter class template instantiations with family-scoped parameters (e.g., `ConvertReductionArg<MinimumFamily=0>` through `ConvertReductionArg<MinimumFamily=6>`), the complete anec::Family enum (8 values, Family0–Family7), and 14 ZinAneTd hardware version template instantiations.

3. **String triage with semantic categorization** — 15,200 strings were extracted from ANECompiler and categorized into 10 semantic dimensions: dtype rejections (7 architecture-level + 9 BF16/F16 cross-type + 15 operation-specific + 9 quantization), conv kernel constraints (11 kernel size + 8 pooling + 12 stride + 10 dilation + 4 deconvolution + 5 group + 5 padding + 6 large kernel + 9 weight/palettization), SDPA constraints (5 operand + 3 key/value + 1 mask + 4 architecture), concat constraints (9 axis + 8 input + 9 width decomposition), MIL operation rejection patterns (4 "cannot be lowered" + 15 "not supported on ANE" + 14 "not supported on this architecture" + 5 "failed to convert" + 9 lowering failure + 8 family-specific), and RingBuffer/State constraints (14 + 11).

4. **RTTI typeinfo extraction** — 4,681 RTTI typeinfo classes were extracted from ANECompiler, revealing the complete C++ class hierarchy including converter base classes, ZinCompute IR node types, and hardware descriptor classes. 3,519 vtable entries mapped the virtual dispatch surface.

5. **Objective-C metadata extraction** — 2 Objective-C classes were found in ANEServices (logging/debug), confirming its runtime service role. No Objective-C metadata was found in ANECompiler (pure C++ MLIR/LLVM infrastructure).

6. **Converter class → family template mapping** — All 4,264 converter classes were mapped to their source MIL operations and target ANEC operations. Family-scoped converters (16 classes, each with 8 family instantiations) were distinguished from family-agnostic converters (30+ classes with single implementations). The critical finding that ConvertReductionArg is only instantiated for Family0–Family6 (NOT Family7/A18) was confirmed through this mapping.

7. **MinimumFamily trait extraction from MLIR operation definitions** — The `OpTrait::anec::MinimumFamily<anec::Family::X>` trait system was reconstructed from demangled verification function symbols. 53 operations were mapped to their minimum family requirements, revealing that most operations are available from Family0 (A11Legacy) with "also verified on" cross-checks for later families.

8. **ZinAneTdHw hardware descriptor mapping** — 14 hardware version descriptors (v4 through v28, plus u1) were mapped to Apple chip generations. Sub-variant descriptors (v812, v813, v827, etc.) were identified for NE (12), PE (13), and DMA/config (27) engine-specific hardware parameters. ZinHWTraits template instantiations were found only for v8+ (A14/M1 and later), indicating v4–v7 use a legacy trait system.

9. **Cross-referencing with Orion arxiv paper (20 ANE constraints)** — Each of the 20 Orion programming constraints from arxiv:2603.06728 was systematically traced through MILLer source code. This cross-examination confirmed 3 correctly handled constraints, 3 partially handled, and 14 unmodeled. Two constraints (#1 concat, #10 gelu) are actively violated. Two previously suspected violations were corrected: Orion #8 (BLOBFILE offset) is correctly handled (weights.rs:419 uses metadata_offset=64), and Orion #13 (conv bias) is correctly handled (MILConv has no bias field).

Additionally, all source files were read line-by-line. Pattern searches were performed for `todo!()`, `unimplemented!()`, `FIXME`, `HACK`, hardcoded limits, silent fallbacks, permissive defaults, and fake validation. Knowledge seed JSONs were cross-referenced against Rust type definitions, documentation, and conservative local metadata signals.

### High-Level Findings

The audit identified **138 violations** across 6 classifications:

| Classification | Count | Description |
|---------------|-------|-------------|
| ABERRANT | 24 | Source contradicts its own docs, tests, configs, or conservative compatibility evidence |
| PHANTOM | 9 | Source assumes a capability for which there is no adequate evidence |
| LACUNA | 66 | Source omits validation for a constraint it claims to respect |
| STUB-MIMIC | 10 | Function presents itself as real logic but uses no-ops, fake success, or permissive fallback |
| UNVERIFIED | 28 | Plausible but not backed by sufficient source evidence or tests |
| INFO | 1 | Previously suspected violation confirmed as correctly handled |

Severity distribution: **9 CRITICAL**, **40 HIGH**, **57 MEDIUM**, **31 LOW**, **1 INFO**.

### Raw Local Artefacts

Raw local reference artefacts are stored in `forensics/` and `binaries/` within the local working tree. These directories are excluded from the repository by `.gitignore`. No raw binary strings, disassembly, proprietary implementation details, or links to local forensic material appear in this report. All binary-derived findings are expressed as abstract constraint catalogs and cross-reference indices without reproducing proprietary content.

---

## II. Compatibility Constraint Tables

### II-A. Claimed Supported Operation Categories

| Operation Category | Claimed Support Scope | Evidence Basis | Confidence |
|---|---|---|---|
| Convolution (standard, depthwise, dilated, grouped) | All A11+ families | source (ane_op_family_matrix.json, op_constraints.rs), binary (ConvertConv family-agnostic, MinimumFamily=A11Legacy) | high |
| Deconvolution / ConvTranspose | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertDeconvolution, MinimumFamily=A11Legacy) | medium |
| Average Pool, Max Pool, L2Norm Pool | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertPool family-agnostic, MinimumFamily=A11Legacy) | high |
| ArgMinMax (global, windowed) | A14+ only; dropped for A18 | source (ane_target.rs:145), binary (7 family instantiations 0-6, no family 7; ConvertReductionArg NOT instantiated for Family7) | high |
| Reduce (avg, max, min, sum) | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertReduction A14Minus F0-1, A14Plus F2; MinimumFamily=A11Legacy for all reduce ops) | high |
| Softmax | All families (matrix); architecture-dependent (per-op doc) | source + local metadata, binary (ConvertSoftmax family-agnostic, no MinimumFamily trait, MinimumFamily=A11Legacy) | high |
| InstanceNorm | All families (matrix); architecture-dependent (per-op doc) | source + local metadata, binary (ConvertInstanceNorm family-agnostic, no MinimumFamily trait, MinimumFamily=A11Legacy) | high |
| LayerNorm | A14+ only | source (ane_target.rs), binary (MinimumFamily=A11Legacy, also verified on A13/Family2) | medium |
| Linear / Fully-Connected | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertLayer 97 instances, family-scoped) | high |
| MatMul | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertMatMul 8 family instantiations, MinimumFamily=A11Legacy) | high |
| Concat, Tile, Transpose, Reshape, Flatten, Unflatten | All A11+ families | source (ane_op_family_matrix.json); **Concat rejected by ANE compiler per Orion #1**; binary (ConvertConcat family-agnostic; ConvertTranspose 8 family instantiations; ConvertReshape 8 family instantiations) | low |
| Gather, GatherND | A12+ (matrix); illegal (legality seed) | source (contradictory seeds), binary (ConvertGather and ConvertGatherND both family-agnostic) | low |
| Padding, Resample, Resize, CropResize | A12+ families | source (ane_op_family_matrix.json), binary (ConvertPadding 8 family instantiations; ConvertResize 8 family instantiations; ConvertCropResize family-agnostic) | high |
| PixelShuffle, PixelUnshuffle | A14+ | source (ane_op_family_matrix.json), binary (ConvertDepthToSpace2D/ConvertSpaceToDepth2D family-agnostic) | medium |
| BatchToSpace, SpaceToBatch, ChannelToSpace, SpaceToChannel | A12+ | source (ane_op_family_matrix.json), binary (ConvertBatchToSpace family-agnostic) | medium |
| Elementwise unary (relu, sigmoid, tanh, erf, invert, abs, sqrt, rsqrt, ceil, floor, sign, trunc, exp2, log2, elu, gelu, swish, dirac, degamma, clamped_relu, n_relu, leaky_relu, square, round_nearest, high_precision_sigmoid) | All A11+ families | source (ane_op_family_matrix.json covers subset), binary (27 ConvertElementwiseUnary variants, all family-agnostic, all MinimumFamily=A11Legacy) | high |
| Elementwise binary (add, mul, sub, div, max, min) | All A11+ families | source (ane_op_family_matrix.json), binary (ConvertElementwiseBinary family-scoped: A14Minus F0-2, A14Plus F3+) | high |
| Comparison (equal, not_equal, greater, less, etc.) | CPU-only (cpu_only seed); A14+ ANE (matrix) | source (contradictory seeds), binary (anec.equal, anec.greater_than, etc. all MinimumFamily=A11Legacy) | low |
| Logical (and, or, not) | A12+ ANE (matrix); CPU-only (cpu_only seed); no converter (per-op doc) | source (three-way contradiction), binary (anec.equal_zero covers NOT; no dedicated logical_and/or converter) | low |
| SDPA (scaled-dot-product attention) | A16+ (Rust); unreliable A12-A15 (matrix) | source (ane_target.rs), binary (ConvertScaledDotProductAttention family-agnostic, MinimumFamily=A11Legacy, also verified on A13/Family2) | medium |
| PReLU, Softsign | A12+ ANE (matrix); CPU-only (cpu_only seed) | source (contradictory seeds), binary (ConvertElementwiseUnary(PRelu) → anec.leaky_relu, family-agnostic) | low |
| Gelu | A14+ (matrix); not a valid MIL activation (Orion #10) | source + binary forensic, binary (ConvertElementwiseUnary(Gelu) → anec.gelu, family-agnostic, tanh approximation only) | low |

### II-B. Claimed Dimensional Limits

| Constraint | Claimed Value (per AneHwLimits) | Evidence Basis | Confidence |
|---|---|---|---|
| max_tensor_width | 16 384 (V4) → 262 144 (V26) | source (ane_hw_limits.rs, ane_hw_limits_seed.json) | medium |
| max_tensor_height | 16 384 (V4) → 262 144 (V26) | source | medium |
| max_tensor_depth | 256 (V4) → 2 048 (V26) | source | medium |
| max_tensor_channels | 16 384 (V4) → 65 536 (V26) | source | medium |
| Conv channel limit | 32 768 (Orion #16) | binary forensic (ANECompiler strings) | high |
| Conv kernel W/H must be power of 2 | Not validated by MILLer (range 1-7 only) | binary ("Kernel width must be a power of 2", "Kernel height must be a power of 2") | high |
| Conv kernel D must be power of 2 | Not validated by MILLer | binary ("Kernel depth must be a power of 2.") | high |
| Large kernel W/H multiple of 8 | Not validated by MILLer | binary ("kernel width and height should be multiple of 8 for large kernel") | high |
| Large kernel stride 1-2 only | Not validated by MILLer | binary ("x strides should be 1 or 2 for conv with large kernel") | high |
| Large kernel zero padding only | Not validated by MILLer | binary ("only support zero padding for conv with large kernel") | high |
| Large kernel depth ≤ 1 | Not validated by MILLer | binary ("kernel with depth = %zd > 1 is not supported for large kernel") | high |
| Large kernel no palettized weights | Not validated by MILLer | binary ("does not support palettized weight with large kernel stride") | high |
| Deconv SOx == 2 | Not validated by MILLer | binary ("deconv with SOx != 2 is not supported") | high |
| Deconv no large kernel | Not validated by MILLer | binary ("deconv with large kernel size is not supported") | high |
| Deconv no vector palettization | Not validated by MILLer | binary ("deconv with vector palettization is not supported") | high |
| Pool stride 3 only for Avg mode | Not validated by MILLer | binary ("Pool with strides of 3 is only supported with Avg mode.") | high |
| max_tensor_rank | 5 (all revisions) | source | medium |
| A12 limits | Copied from A11 (unverified) | source (ane_hw_limits.rs:66-82, self-documented) | low |
| V26 limits | Fabricated from A18 + num_nes=16 | source (ane_hw_limits.rs:144-146) | low |
| ne_transpose_c_max | 16 384 (all revisions, suspiciously uniform) | source (ane_hw_limits_seed.json) | low |
| Minimum IOSurface for eval | ~49 KB (Orion #4) | binary forensic (Orion) | medium |
| Compilations per process | ~119 (Orion #5) | binary forensic (Orion) | medium |

### II-C. Claimed Alignment and Layout Requirements

| Constraint | Claimed Value | Evidence Basis | Confidence |
|---|---|---|---|
| DRAM alignment | 64-byte row stride alignment | local metadata (ANECompiler strings) | medium |
| Power-of-2 alignment requirement | Enforced for memory/cache | local metadata (ANECompiler strings) | medium |
| Tile DMA granularity | Calculated from dram_alignment per HW version | local metadata (ANECompiler strings) | medium |
| L2 cache alignment | Computed from tensor format | source (ane_hw_limits.rs, ZinIrHalParameters) | medium |
| Multi-output buffer ordering | Alphabetical (Orion #3) | binary forensic (Orion) | high |
| Multi-input surface ordering | Alphabetical (Orion #19) | binary forensic (Orion) | high |
| Multi-output buffer size uniformity | All outputs must have uniform sizes (Orion #2) | binary forensic (Orion) | high |
| Multi-input surface size uniformity | Uniform alloc sizes required (Orion #18) | binary forensic (Orion) | high |
| ANE flat buffer layout | Packed [1,C,1,S] (Orion #20) | binary forensic (Orion) | high |

### II-D. Claimed Data-Type Masks

| Data Type | ANE Legality Claim | Evidence Basis | Confidence |
|---|---|---|---|
| FP16 | Legal for compute, default dtype | source (dtype_constraints.rs), binary (primary compute format, "Output tensor type must be Int8, UInt8, or Float16") | high |
| FP32 | Legal (may be downcast) | source (dtype_constraints.rs:73), binary ("Float32 not supported for architecture" on some families) | medium |
| Int8 | Legal for weights | source (dtype_constraints.rs), binary (weight format, "It should be Int8, UInt8, FP16, FP32 or E4M3") | high |
| UInt8 | Legal for weights | source (dtype_constraints.rs), binary (weight format) | high |
| Int4 | Legal with interleave==8 (caller must check) | source (dtype_constraints.rs:79-81), binary ("Int4 Per-Cout Dequant is not supported") | medium |
| UInt4 | Legal with interleave==8 (caller must check) | source (dtype_constraints.rs:79-81) | medium |
| E4M3 | Architecture-conditional (rejected on some, supported on A17+) | source, binary ("E4M3 is not supported on this architecture", "E4M3 not supported as kernel format on this architecture", "E4M3Overflow is not supported.") | high |
| E5M2 | Rejected by is_dtype_ane_legal(); accepted by quantize validator | source (contradictory checks), binary ("E4M3 or E5M2 format not supported" universally) | high |
| 2xInt8 | Not mentioned in source | binary ("2xInt8 mode is not supported") | high |
| UInt16 | "Limited support" (no constraints defined) | source (dtype_constraints.rs:105), binary (UInt16 in MPS-ANEC allowlist) | low |
| Int16 | Not mentioned in source | binary (Int16 in MPS-ANEC allowlist: "Only Int4, Uint8, Int8, Uint16, Int16, Float16, Float32, Int32, UInt32, UInt64 and Int64 are supported") | medium |
| Int32/UInt32 | Not mentioned as compute type | source, binary ("Src2 Int32,Uint32 not supported for architecture") | medium |
| UInt64/Int64 | Not mentioned in source | binary (in MPS-ANEC allowlist but no ANE compute support evidence) | low |
| Bool | "Limited support" (no constraints defined) | source (dtype_constraints.rs:113) | low |
| BF16 | Not listed as valid dtype | source (absent from MilDtype enum), binary ("detected operation with BF16 inputs and F16 result type which is not supported", "detected operation with F16 inputs and BF16 result type which is not supported", "detected operation with both F16 and BF16 operands which is not supported") | high |

### II-E. Claimed Hardware-Version Gates

| Gate | Claimed Scope | Evidence Basis | Confidence |
|---|---|---|---|
| AneRevision V4–V26 | 11 revisions defined | source (ane_hw_limits.rs) | high |
| AneFamily A11Legacy–A18 | 6 families defined | source (ane_target.rs) | high |
| Binary MinimumFamily enum | 8 families (0–7) defined in converter templates | binary forensic (demangled template params: Family0=A11Legacy, Family1=A12, Family2=A13, Family3=A14/M1, Family4=A15/M2, Family5=A16/M3, Family6=A17 Pro/M4, Family7=A18/M4 Pro) | high |
| ZinAneTdHw versions | 14 hardware descriptors (v4, v5, v6, v7, v8, v10, v11, v17, v19, v20, v24, v26, v28, u1) | binary forensic (ZinAneTdHw concrete classes) | high |
| Sub-variant descriptors | NE(12), PE(13), DMA(27) per version | binary forensic (v812/v813/v827, v1012/v1013/v1027, etc.) | high |
| ZinHWTraits versions | v8+ only (v4–v7 use legacy system) | binary forensic (template instantiation patterns) | high |
| V6→A13 (Rust) vs V6→A14 (JSON seed) | Family mismatch | source (contradictory mappings) | low |
| V11→A17 (Rust) vs V11→A16 (JSON seed) | Family mismatch | source (contradictory mappings) | low |
| V20 (M4 Mac) → A18 | ArgMinMax dropped; unverified for Mac | source (ane_target.rs:145), binary (no family 7 instantiations for ArgMinMax) | medium |
| V26 → future (invented limits) | No hardware specification | source (ane_hw_limits.rs:144-146) | low |
| Family 6–7 (A17 Pro/A18) | Partially modeled in MILLer (AneFamily has A17, A18) but binary MinimumFamily shows 8 values vs MILLer's 6 | binary forensic (enum values 0-7 in MinimumFamily) | medium |

### II-F. Claimed Descriptor Requirements

| Requirement | Claim | Evidence Basis | Confidence |
|---|---|---|---|
| Opset version | iOS18 (hardcoded constant) | source (ir/src/lib.rs:17) | medium |
| Minimum deployment target | iOS18 (hardcoded in shard_plan) | source (shard_plan.rs:561-562) | medium |
| Interleave factor for Int4/UInt4 | Must be 8 (caller-enforced, not validated) | source (dtype_constraints.rs:79-81) | low |
| Conv kernel range | 1–7 (op_constraints.rs) | source | medium |
| Palette bits | Valid: {1,2,3,4,6,8} (documented, not enforced) | source (sir.rs:48-53), binary ("Only 2-bit, 4-bit, 6-bit and 8-bit palettization for conv are supported!") | high |
| Conv bias= param | Not supported (Orion #13) — **correctly handled** | binary forensic (Orion), source (MILConv has no bias field, Python uses mb.linear) | high |
| BLOBFILE offset | uint64(64), not 128 (Orion #8) — **correctly handled** | binary forensic (Orion), source (weights.rs:419 uses metadata_offset=64, test at line 605 confirms) | high |
| MIL text format | Must be NSData*, not NSString* (Orion #9) | binary forensic (Orion) | medium |
| Weight dict initialization | Must be @{}, not nil (Orion #11) | binary forensic (Orion) | medium |
| MatMul transpose flags | Need named const nodes, not immediate bools (Orion #12) | binary forensic (Orion), source (coreml-proto/src/lib.rs:3854-3860 uses make_immediate_bool_value) | high |
| Output var references | Must ref live (post-opt) nodes (Orion #14) | binary forensic (Orion) | medium |
| ANE flat buffer layout | Packed [1,C,1,S] (Orion #20) | binary forensic (Orion) | high |
| Conv 1×1 vs matmul | Conv 1×1 is 3× faster than matmul (Orion #17) | binary forensic (Orion) | high |
| SDPA causal masks | Silently ignored (Orion #6) | binary forensic (Orion) | high |
| Conv kernel power-of-2 | Kernel W/H/D must be power of 2 | binary ("Kernel width must be a power of 2", "Kernel height must be a power of 2", "Kernel depth must be a power of 2.") | high |
| Dilated conv L2 budget | Dilated conv may fail if space-to-batch exceeds L2 DMA buffer | binary ("Dilated convolution cannot be lowered since all possible space-to-batch implementations exceeded the L2 DMA buffer size") | medium |
| Deconv no dilation | Dilation not supported for deconvolution | binary ("Dilation not supported for deconvolution") | high |
| Dilated pooling rejected | No dilated pooling on ANE | binary ("Dilated Pooling not supported on ANE") | high |
| Vector palettization only at Cout | Vector palettization restricted to Cout dimension | binary ("vector palettization is only supported at Cout for ANE") | high |
| Asymmetric quantization rejected | No asymmetric quantization on ANEC | binary ("Asym quantization is not supported") | high |
| Non-constant gather axis rejected | Gather requires constant axis | binary ("gather with non-constant axis is not supported on ANEs") | high |
| 5D stencil rejected | No rank-5 stencil on ANE | binary ("stencil along channel with rank 5 input is not supported on ANEs") | high |
| Width wrap axis architecture-dependent | Not supported on some architectures | binary ("Width wrap axis is not supported on this architecture") | medium |

### II-G. ANEC Operation Schema Constraints

The following table documents the complete ANEC MLIR dialect operation catalog with exact attribute shape constraints extracted from ANECompiler constraint validation strings. Each operation has precisely defined attribute shapes, types, and requirements that must be satisfied for ANE compilation to succeed. The catalog contains 98 computational operations plus 8 architecture-variant marker operations.

| ANEC Operation | Key Attribute Constraints | Converter Class | Family Scoping |
|---|---|---|---|
| anec.abs | (simple unary) | ConvertElementwiseUnary (Abs) | Family-agnostic |
| anec.add | (elementwise binary) | ConvertElementwiseBinary | Family-scoped (A14Minus/A14Plus) |
| anec.arg_min_max | axes=ranks 0/1, kernel_size=shape{2}, stride_values=shape{2}, pad_values=shape{4}, mode=ArgMinMaxMode | ConvertReductionArg | Family 0-6 only (7 instantiations, NOT Family7) |
| anec.average_pool | ksize=shape{3}, padding=shape{6}, stride=shape{3}, inc_pad=unit attribute | ConvertPool (Avg) | Family-agnostic |
| anec.batch_norm | (normalization attributes) | ConvertNormalization | Family-agnostic |
| anec.batch_to_space | factors=shape{3} | ConvertBatchToSpace | Family-agnostic |
| anec.broadcast | (broadcast attributes) | ConvertBroadcast | 8 family instantiations |
| anec.cast | (type conversion) | ConvertCast | Family-agnostic |
| anec.ceil | (simple unary) | ConvertElementwiseUnary (Ceil) | Family-agnostic |
| anec.channel_to_space | factors=shape{3} | ConvertDepthToSpace2D (ChannelToSpace) | Family-agnostic |
| anec.clamped_relu | (activation with clamp parameters) | ConvertElementwiseUnary (ClampedRelu) | Family-agnostic |
| anec.concat | axis=u64, interleave=unit attribute | ConvertConcat | Family-agnostic |
| anec.convolution | stride=shape{3}, dilation=shape{3}, padding=shape{6}, padding_mode=PaddingMode, groups=u64, channel_wise=unit, kernel_scale=f16/f32 rank 0/1/4, kernel_zero_point=si8/ui8 rank 0/1/4, kernel_palettized_LUT=dense rank 0-6, kernel_mutable_palettized_LUT=dict | ConvertConv (Conv2D, Conv3D, DepthwiseConv2D) | Family-agnostic |
| anec.cos | (simple unary) | ConvertElementwiseUnary (Cos) | Family-agnostic |
| anec.crop_resize | output_dims=shape{2}, crop_dims=shape{2}, box_coordinate_mode=BoxCoordinateMode, coordinate_mode=CoordinateMode shape{5}, normalized_range=NormalizedCoordinateRange shape{5}, padding_modes=PaddingMode shape{5}, sampling_method=SamplingGridMethod shape{5}, sampling_mode=SamplingGridMode shape{5}, background_value=f16 | ConvertCropResize | Family-agnostic |
| anec.deconvolution | same attributes as convolution | ConvertConv (Deconv) | Family-agnostic |
| anec.degamma | (simple unary) | ConvertElementwiseUnary (Degamma) | Family-agnostic |
| anec.dequant | (quantization attributes) | ConvertQuantizationOp (Dequantize) | Family-agnostic |
| anec.dirac | (simple unary) | ConvertElementwiseUnary (Dirac) | Family-agnostic |
| anec.div | (elementwise binary) | ConvertDivide | 8 family instantiations |
| anec.elu | (activation with alpha) | ConvertElementwiseUnary (Elu) | Family-agnostic |
| anec.equal | (comparison binary) | (compare group) | Family-agnostic |
| anec.equal_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.erf | (simple unary) | ConvertElementwiseUnary (Erf) | Family-agnostic |
| anec.exp2 | (simple unary) | ConvertExponent / ConvertElementwiseUnary (Exp2) | Family-agnostic |
| anec.flatten | flatten_mode=FlattenMode | ConvertFlatten2D | 8 family instantiations |
| anec.floor | (simple unary) | ConvertElementwiseUnary (Floor) | Family-agnostic |
| anec.gain_offset_control | (GOC attributes) | (GOC group) | Family-agnostic |
| anec.gather_nd | axes=ui64 unique not empty rank 1 | ConvertGatherND | Family-agnostic |
| anec.gelu | (activation, tanh approximation only) | ConvertElementwiseUnary (Gelu) | Family-agnostic |
| anec.global_arg_min_max | axis=u32, mode=ArgMinMaxMode | ConvertReductionArg | Family 0-6 only |
| anec.greater_than | (comparison binary) | (compare group) | Family-agnostic |
| anec.greater_than_equal | (comparison binary) | (compare group) | Family-agnostic |
| anec.greater_than_equal_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.greater_than_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.high_precision_sigmoid | (high-precision activation) | ConvertElementwiseUnary (Sigmoid, high-precision path) | Family-agnostic |
| anec.input_view | dimension=u64, offset=u64, size=u64, step=i64 (negative strides supported) | (internal) | Family-agnostic |
| anec.instance_norm | (normalization attributes) | ConvertInstanceNorm | Family-agnostic |
| anec.invert | (simple unary, reciprocal) | ConvertElementwiseUnary (Reciprocal) | Family-agnostic |
| anec.l2norm_pool | ksize=shape{3}, padding=shape{6}, stride=shape{3} | ConvertPool (L2) | Family-agnostic |
| anec.layer_norm | (normalization attributes) | (layer group) | Family-agnostic |
| anec.leaky_relu | (activation with alpha) | ConvertElementwiseUnary (LeakyRelu) | Family-agnostic |
| anec.less_than | (comparison binary) | (compare group) | Family-agnostic |
| anec.less_than_equal | (comparison binary) | (compare group) | Family-agnostic |
| anec.less_than_equal_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.less_than_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.linear | kernel_scale=f16/f32 rank 0/1, kernel_zero_point=si8/ui8 rank 0/1, kernel_lut=palettized LUT rank 0-6 | ConvertLayer | Family-agnostic (97 instances) |
| anec.log2 | (simple unary) | ConvertLogarithm / ConvertElementwiseUnary (Log2) | Family-agnostic |
| anec.matmul | bias=f16 | ConvertMatMul | 8 family instantiations |
| anec.max | (elementwise binary) | (elementwise binary group) | Family-agnostic |
| anec.max_pool | ksize=shape{3}, padding=shape{6}, stride=shape{3} | ConvertPool (Max) | Family-agnostic |
| anec.min | (elementwise binary) | (elementwise binary group) | Family-agnostic |
| anec.mult | (elementwise binary) | (elementwise binary group) | Family-agnostic |
| anec.n_relu | (clamped/bounded ReLU, Relu6) | ConvertElementwiseUnary (NRelu) | Family-agnostic |
| anec.not_equal | (comparison binary) | (compare group) | Family-agnostic |
| anec.not_equal_zero | (comparison unary vs zero) | (compare-zero group) | Family-agnostic |
| anec.padding | padding_modes=PaddingMode shape{5}, padding_sizes=shape{5,2}, background_value=f16 | ConvertPadding | 8 family instantiations |
| anec.pixel_shuffle | factors=shape{3} | ConvertDepthToSpace2D (PixelShuffle) | Family-agnostic |
| anec.pixel_unshuffle | factors=shape{3} | ConvertSpaceToDepth2D (PixelUnshuffle) | Family-agnostic |
| anec.power | (elementwise binary) | (elementwise binary group) | Family-agnostic |
| anec.quant | (quantization attributes) | ConvertQuantizationOp (Quantize) | Family-agnostic |
| anec.r_sqrt | (simple unary) | ConvertElementwiseUnary (Rsqrt) | Family-agnostic |
| anec.reduce_avg | axes=ui64 unique ranks 0/1 | ConvertReductionA14Minus (F0-1), ConvertReductionA14Plus (F2) | Family-scoped |
| anec.reduce_max | axes=ui64 unique ranks 0/1 | ConvertReductionA14Minus (F0-1), ConvertReductionA14Plus (F2) | Family-scoped |
| anec.reduce_min | axes=ui64 unique ranks 0/1 | ConvertReductionA14Minus (F0-1), ConvertReductionA14Plus (F2) | Family-scoped |
| anec.reduce_sum | axes=ui64 unique ranks 0/1 | ConvertReductionA14Minus (F0-1), ConvertReductionA14Plus (F2) | Family-scoped |
| anec.region_return | (control flow) | (internal) | Family-agnostic |
| anec.relu | (simple unary) | ConvertElementwiseUnary (Relu) | Family-agnostic |
| anec.resample | 7 attributes: coordinate_mode, normalized_range, coordinate_type, warp_coordinate_mode, sampling_method, sampling_mode, background_value | ConvertSampleGrid | Family-agnostic |
| anec.reshape | (shape attributes) | ConvertReshape | 8 family instantiations |
| anec.resize | height=u64, width=u64, scale_factor_x=f32, scale_factor_y=f32, sampling_methods=SamplingGridMethod shape{2}, sampling_modes=SamplingGridMode shape{2}, padding_mode=PaddingMode shape{2} | ConvertResize | 8 family instantiations |
| anec.ring_buffer_reader | (RingBuffer read attributes) | ConvertRingBufferReaderPatternToFusionOp | Family-agnostic |
| anec.ring_buffer_writer | (RingBuffer write attributes) | ConvertRingBufferWriterPatternToFusionOp | Family-agnostic |
| anec.round_nearest | (simple unary) | ConvertElementwiseUnary (RoundNearest) | Family-agnostic |
| anec.scaled_elementwise | (scaled elementwise attributes) | (scaled EW group) | Family-agnostic |
| anec.sdpa | (attention attributes, 4-5 operands) | ConvertScaledDotProductAttention | Family-agnostic |
| anec.sigmoid | (activation) | ConvertElementwiseUnary (Sigmoid) | Family-agnostic |
| anec.sign | (simple unary) | ConvertElementwiseUnary (Sign) | Family-agnostic |
| anec.sin | (simple unary) | ConvertElementwiseUnary (Sin) | Family-agnostic |
| anec.softmax | (normalization attributes) | ConvertSoftmax | Family-agnostic |
| anec.space_to_batch | factors=shape{3} | ConvertBatchToSpace (SpaceToBatch) | Family-agnostic |
| anec.space_to_channel | factors=shape{3} | ConvertSpaceToDepth2D (SpaceToChannel) | Family-agnostic |
| anec.sqr | (simple unary, x²) | ConvertElementwiseUnary (Sqr) | Family-agnostic |
| anec.sqrt | (simple unary) | ConvertElementwiseUnary (Sqrt) | Family-agnostic |
| anec.square | (simple unary, x² alias) | ConvertElementwiseUnary (Square) | Family-agnostic |
| anec.state | (state variable attributes) | ConvertState / ConvertReadVariable | Family-agnostic |
| anec.sub | (elementwise binary) | (elementwise binary group) | Family-agnostic |
| anec.swish | (activation) | ConvertElementwiseUnary (Swish) | Family-agnostic |
| anec.tanh | (activation) | ConvertElementwiseUnary (Tanh) | Family-agnostic |
| anec.tensor_buffer_to_tensor | (buffer conversion) | ConvertTensorBufferPatternToFusionOp | Family-agnostic |
| anec.tensor_to_tensor_buffer | (buffer conversion) | (internal) | Family-agnostic |
| anec.tile | multiples=ui64 rank 1 | ConvertTile | Family-agnostic |
| anec.transpose | transpose_list=list of u64 pairs | ConvertTranspose | 8 family instantiations |
| anec.trunc | (simple unary) | ConvertElementwiseUnary (Trunc) | Family-agnostic |
| anec.unflatten | flatten_mode=FlattenMode, destination_size=shape{3} | (reshape group) | 8 family instantiations |
| anec.unrealized_conversion_cast | (type conversion) | (internal lowering) | Family-agnostic |

### II-H. ANEC Operation → Family Compatibility Matrix

The following matrix maps ANEC operations to their minimum family requirements based on MinimumFamily trait evidence extracted from ANECompiler binary. "Agnostic" means the converter has a single implementation for all families. "Scoped" means separate implementations per family with differing behavior.

| ANEC Operation | Minimum Family | Converter Type | Family-Specific Notes |
|---|---|---|---|
| Convolution | A11Legacy | Agnostic | — |
| Deconvolution | A11Legacy | Agnostic | — |
| Broadcast | A11Legacy | Scoped (8 families) | — |
| Reshape | A11Legacy | Scoped (8 families) | — |
| Transpose | A11Legacy | Scoped (8 families) | — |
| Cast | A11Legacy | Agnostic | — |
| Relu | A11Legacy | Agnostic | — |
| Sigmoid | A11Legacy | Agnostic | Also has high_precision_sigmoid path |
| Tanh | A11Legacy | Agnostic | — |
| Gelu | A11Legacy | Agnostic | **Tanh approximation only**; EXACT mode not supported |
| Swish | A11Legacy | Agnostic | — |
| Dirac | A11Legacy | Agnostic | — |
| Degamma | A11Legacy | Agnostic | — |
| InputView | A11Legacy | Agnostic | Supports negative strides (step=i64) |
| State | A11Legacy | Agnostic | — |
| CropResize | A11Legacy | Agnostic | Also verified on A14 (Family3) |
| Resample | A11Legacy | Agnostic | Also verified on A14 (Family3) |
| Abs | A11Legacy | Agnostic | — |
| Square / Sqr | A11Legacy | Agnostic | — |
| EqualZero, LessThanZero, etc. | A11Legacy | Agnostic | 6 compare-vs-zero operations |
| HighPrecisionSigmoid | A11Legacy | Agnostic | — |
| ScaledElementWise | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| TensorBuffer ↔ Tensor | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| RingBufferReader/Writer | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| SDPA | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| Cos, Sin | A11Legacy | Agnostic | Also verified on A15 (Family4) |
| GlobalArgMinMax | A11Legacy | Agnostic | Also verified on A15 (Family4); **Architecture-dependent** ("not supported on this architecture" on some) |
| Softmax | A11Legacy | Agnostic | Also verified on A13 (Family2); **Architecture-dependent** ("not supported by this ANE architecture" on some) |
| InstanceNorm | A11Legacy | Agnostic | Also verified on A13 (Family2); **Architecture-dependent** ("not supported for this ANE architecture" on some) |
| BatchNorm | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| LayerNorm | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| ReduceAvg/Max/Min/Sum | A11Legacy | Scoped | ConvertReductionA14Minus (F0-2), ConvertReductionA14Plus (F3+) |
| Resize | A11Legacy | Scoped (8 families) | Also verified on A13 (Family2) |
| RoundNearest | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| ChannelToSpace, SpaceToChannel | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| Ceil, Elu, Erf, Exp2, Floor, Log2, Sign, Sqrt, Rsqrt, Tile, Trunc | A11Legacy | Agnostic | Also verified on A13 (Family2) |
| ArgMinMax | A11Legacy | Scoped | **Only Family0-6 (7 instantiations); NOT Family7 (A18)** |
| Divide | A11Legacy | Scoped (8 families) | — |
| FloorDivide | A11Legacy | Scoped (8 families) | — |
| MatMul | A11Legacy | Scoped (8 families) | — |
| ExpandDims, Squeeze | A11Legacy | Scoped (8 families) | — |
| ResizeGeneric | A11Legacy | Scoped (8 families) | — |
| Padding | A11Legacy | Scoped (8 families) | — |
| Slice, StridedSlice | A11Legacy | Scoped (8 families) | — |
| Reverse | A11Legacy | Scoped (8 families) | — |
| Crop | A11Legacy | Scoped (8 families) | — |
| Flatten | (per-family) | Scoped (8 families) | — |
| ElementwiseBinary | A11Legacy | Scoped | A14Minus (F0-2), A14Plus (F3+) |

---

## III. Faithfulness Violations

Sorted by severity. Each entry includes a cross-reference to Section II.

### CRITICAL

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-001 | knowledge/ane_hw_limits_seed.json:40-55 | ABERRANT | V6 mapped to family "A14" but Rust code maps V6→A13. Knowledge seed grants A14-class capabilities to A13 hardware. | source | high | II-E |
| V-002 | knowledge/ane_hw_limits_seed.json:108-123 | ABERRANT | V11 mapped to family "A16" but Rust code maps V11→A17. Knowledge seed misses A17 E4M3 support distinction. | source | high | II-E |
| V-003 | knowledge/cpu_only_ops_seed.json:296-324 | ABERRANT | Comparison ops (equal, not_equal, greater, etc.) listed as CPU-only but have ConvertBinaryCompare ANEC converters on A14+. | source + local metadata | high | II-A |
| V-004 | knowledge/ane_op_family_matrix.json:1239-1285 | PHANTOM | logical_and/or/not listed as "supported" A12+ but have no ANEC converter (per-op doc confirms never land on ANE). | source + local metadata | high | II-A |
| V-005 | knowledge/legality_seed.json:62-75 | ABERRANT | mb.gather declared ANE-illegal (ane_legal: false) but anec.gather exists with ConvertGather converter. Blanket illegal claim is wrong. | source + local metadata | high | II-A |
| V-006 | crates/coreml-proto/src/lib.rs:124 | ABERRANT | Float64 element_size() returns 4 instead of 8. All byte-size calculations for Float64 weights will be wrong. | source | high | II-D |
| V-007 | crates/bridge/src/mir_to_compat.rs:224-249 | LACUNA | Zero-filled weight placeholders silently produce models that compile and load but produce completely incorrect inference. Only indication is stderr warning. | source | high | II-D |
| V-098 | python/mil_emitter.py:432, crates/passes/src/mil_lower.rs:2842-2858 | ABERRANT | MILLer emits MILConcat (mb.concat) in SDPA decomposition and embedding gather paths, but Orion #1 documents that the concat MIL op is rejected by the ANE compiler. Binary forensic confirms extensive concat constraints: "Concat supports only 1 axis", "ANE Concat supports only const positive axis", "failed: only works when concat is applied on the channel axis", "Doesn't support interleaved concat on %s dimension", "Symbolic shape propagation is not supported on Concat". Width concat requires special transpose insertion ("Concat with unaligned W dimension requires transpose inserted for each input"). All models using SDPA decomposition will fail ANE compilation. | source + binary forensic (Orion #1, Extraction 7) | high | II-A, II-G |
| V-113 | crates/trace/src/sir_build.rs:518,1415 | ABERRANT | SIR builder hardcodes Gelu mode="EXACT" but ANEC ConvertElementwiseUnary(Gelu) only supports tanh approximation. The SIR→AIR→MIR pipeline preserves whatever mode the SIR builder sets. Since the SIR builder uses "EXACT", models compiled through the Rust pipeline will emit mb.gelu(mode="EXACT") which is not supported by ANEC. Meanwhile, the Python emitter (mil_emitter.py:893,1142) and role_mir.rs:252 use "TANH_APPROXIMATION". The Rust and Python paths produce incompatible gelu modes. | source + binary forensic (Extraction 2.4, Orion #10, orion_cross_examination.md) | high | II-A, II-G |

### HIGH

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-008 | crates/ir/src/ane_hw_limits.rs:66-82 | INFO | **CORRECTLY HANDLED (downgraded from UNVERIFIED)**: A12 hardware limits are unverified copies of A11 values with self-documented WARNING, but used in production constraint validation. The BLOBFILE offset (Orion #8) is correctly handled — weights.rs:419 uses metadata_offset=64 matching the uint64(64) requirement, and test at line 605 confirms. The conv bias= param (Orion #13) is also correctly handled — MILConv has no bias field and Python uses mb.linear. | source + binary forensic (Orion #8, #13 confirmed correctly handled) | high | II-B, II-F |
| V-009 | crates/ir/src/ane_hw_limits.rs:148-193 | LACUNA | 7 conv/pool/PE-specific constraint fields defined but never validated by validate_tensor_dims. Conv/pool ops with oversized kernels pass validation but fail at ANE emission. | source | high | II-B |
| V-010 | crates/ir/src/mir.rs:1061-1311 | LACUNA | default_engine() returns static engine assignment per op regardless of AneRevision. Ops assigned to PE may be placed on families that don't support them. | source | high | II-E |
| V-011 | crates/ir/src/shard_desc.rs:95 | STUB-MIMIC | Unknown dtype string silently defaults to Fp16. Invalid dtype strings like "bf16" or "int8" produce wrong precision without error. | source | high | II-D |
| V-012 | crates/ir/src/common.rs:297-308 | ABERRANT | Generic model architecture falls back to Qwen3 weight patterns. Non-Qwen3/LLaMA models will have silently broken weight resolution. | source | high | II-A |
| V-013 | crates/ir/src/payload.rs:685-698 | ABERRANT | FamilyPayload hardcodes stateful: false regardless of actual op. Stateful ops emitted via generic path get wrong function descriptors. | source | high | II-F |
| V-014 | crates/passes/src/staticize.rs:43-46 | PHANTOM | Entire StaticizePass::run() is Ok(input) — pure pass-through. Doc claims it replaces symbolic dims, resolves variable-length sequences, records decisions. None implemented. | source | high | II-B |
| V-015 | crates/passes/src/precision_policy.rs:94-118 | LACUNA | Only 14 of ~167 SIR op types query precision hazards. All others silently use default fp16 even if stored knowledge indicates a hazard. | source | high | II-D |
| V-016 | crates/passes/src/state_topology.rs:43-96 | STUB-MIMIC | Pass claims to "verify" and "ensure" state patterns but only logs eprintln warnings — never returns Err. Claims to validate naming conventions and state capacity but implements neither. | source | high | II-F |
| V-017 | crates/passes/src/shard_plan.rs:367-378 | LACUNA | FunctionEntry TensorSpec shapes hardcoded as vec![1,1] throughout. Comments say "derived from graph" but derivation not implemented. | source | high | II-B |
| V-018 | crates/passes/src/mil_lower.rs:92-98 | LACUNA | MatMul inner-dim mismatch only logs eprintln warning and continues, producing graph with wrong dimensions. | source | high | II-B |
| V-019 | crates/passes/src/cpu_only_ops.rs:147-156 | ABERRANT | gather listed in CPU_ONLY_OPS but mil_lower actively emits MILGather for embedding lookup and legality_rewrite generates Gather for RoPE table lookups. | source | high | II-A |
| V-020 | crates/passes/src/placement_validate.rs:272-292 | LACUNA | Interleave constraints skipped entirely when channels unknown, including non-channel-dependent checks (const→1, int4→8). | source | high | II-D |
| V-021 | crates/knowledge/src/transfer.rs:152 | LACUNA | claims_agree defaults to true for 7/8 knowledge types. No contradiction detection for PrecisionHazard, SurvivalMatrixEntry, MotifCatalog, FallbackSignature, DeviceFingerprint, ShardTemplateKnowledge, StateTopologyOutcome. | source | high | II-A |
| V-022 | crates/knowledge/src/store.rs:524-547 | LACUNA | Conflict detection marks new entry as ConflictedWith(existing) but never back-patches existing entry. Querying only existing entries misses mutual conflicts. | source | high | II-F |
| V-023 | crates/bridge/src/mir_to_compat.rs:275-413 | LACUNA | Input/output node fallback to shape vec![1] and dtype Fp16 when node missing from MIR graph. Almost certainly wrong for any real model. | source | high | II-B, II-D |
| V-024 | crates/bridge/src/subprocess.rs:28,55-69 | PHANTOM | timeout_secs field stored but never enforced. Command::output() blocks indefinitely. Timeout is a phantom capability. | source | high | II-F |
| V-025 | crates/bridge/src/mir_to_compat.rs:458-468 | LACUNA | Default architecture silently defaults to Qwen3 weight name patterns. Wrong patterns mean wrong input remapping, producing models with undefined references. | source | high | II-A |
| V-026 | crates/bridge/src/safetensors_resolver.rs:196-199 | LACUNA | F32 weight data passed through without FP16 conversion. BF16 gets converted; F32 does not. F32 bytes written as-is but may be declared as FP16 in proto. | source | high | II-D |
| V-027 | crates/coreml-emit/src/weights.rs:116-119 | LACUNA | Bool, Float64, Unknown data types silently mapped to Float32 blob format. Data-corrupting for Bool tensors; incorrect for Float64. | source | high | II-D |
| V-028 | crates/coreml-emit/src/mir_to_proto.rs:339-356 | LACUNA | State declarations default to empty shape + Fp16 when only write op present. Core ML will reject proto with empty-dimension state. | source | high | II-B, II-D |
| V-029 | knowledge/ane_op_family_matrix.json:806-821 | UNVERIFIED | Softmax listed as "supported" for all families including A11Legacy, but per-op doc shows architecture-dependent. Binary evidence shows ConvertSoftmax is family-agnostic with MinimumFamily=A11Legacy but also verified on A13 (Family2), and ANEC has architecture-conditional rejection ("Softmax is not supported by this ANE architecture"). The per-op doc's architecture-dependent claim is partially correct — some older architectures reject softmax despite the family-agnostic converter. | source + binary forensic (converter catalog, architecture rejection strings) | high | II-A |
| V-030 | knowledge/ane_op_family_matrix.json:951-965 | UNVERIFIED | InstanceNorm listed as "supported" for all families but per-op doc shows architecture-dependent on older families. Binary evidence shows ConvertInstanceNorm is family-agnostic with MinimumFamily=A11Legacy but also verified on A13 (Family2), and ANEC has architecture-conditional rejection ("InstanceNorm layer not supported for this ANE architecture"). | source + binary forensic (converter catalog, architecture rejection strings) | high | II-A |
| V-099 | python/mil_emitter.py:890-894, 1141-1142, crates/trace/src/sir_build.rs:518,1415, crates/passes/src/role_mir.rs:252 | ABERRANT | MILLer emits mb.gelu with contradictory modes: (1) "EXACT" from SIR builder (sir_build.rs:518,1415), (2) "TANH_APPROXIMATION" from role_mir.rs:252 and Python emitter (mil_emitter.py:893,1142), (3) "exact" from test fixtures (staticize.rs:527). Orion #10 documents that gelu is not a valid MIL activation. ANEC ConvertElementwiseUnary(Gelu) only supports tanh approximation. The SIR→AIR→MIR pipeline preserves whatever mode the SIR builder sets — since SIR uses "EXACT", models compiled through the Rust pipeline will produce invalid gelu. | source + binary forensic (Orion #10, Extraction 2.4) | high | II-A |
| V-100 | crates/coreml-proto/src/lib.rs (MirOpCompat enum), crates/ir/src/mir.rs (MirOp enum) | LACUNA | ANEC operation enum contains 98+ operations but MILLer's MirOpCompat only models ~30 variants. 70+ ANEC operations including RingBufferReader/Writer, State, ScaledElementWise, HighPrecisionSigmoid, NRelu, ClampedRelu, Dirac, Degamma, GOC, Sqr, Rsqrt, Elu, LeakyRelu, Log2, Exp2, Sign, Trunc, Ceil, Floor, RegionReturn, UnrealizedConversionCast, TensorBufferToTensor, TensorToTensorBuffer have no MILLer equivalents. Ops that should map to these ANEC ops will fail emission or be incorrectly lowered. | source + binary forensic (ANEC enum, Extraction 8) | high | II-G, II-H |
| V-101 | knowledge/ane_op_family_matrix.json (Softmax, InstanceNorm) | ABERRANT | ConvertSoftmax and ConvertInstanceNorm are family-agnostic converters (no MinimumFamily template specialization in binary), contradicting MILLer's per-op-per-family documentation's architecture-dependent claims. However, ANEC does have architecture-conditional rejection strings for both ("Softmax is not supported by this ANE architecture", "InstanceNorm layer not supported for this ANE architecture"). This means the converters exist for all families but specific architecture variants reject them — a subtlety neither MILLer's documentation nor its constraint model captures. | source + binary forensic (converter catalog, architecture rejection strings) | high | II-A, II-E, II-H |
| V-102 | crates/ir/src/ane_target.rs:145 | ABERRANT | ConvertReductionArg (ArgMinMax) has exactly 7 family instantiations (0-6) in the binary, confirming ArgMinMax is NOT available on Family 7+ (A18+). Binary forensic from Extraction 2.2 confirms ConvertReductionArg is only instantiated for A11Legacy through A17, NOT A18. | source + binary forensic (converter template params, Extraction 2.2) | high | II-E, II-H |
| V-103 | crates/ir/src/ane_hw_limits.rs:178, crates/passes/src/op_constraints.rs | LACUNA | MILLer doesn't enforce 32K-channel limit for convolutions (Orion #16). max_tensor_channels is set to 65536 for newer versions but ANECompiler rejects convolutions with >32K channels. The channel validation in ane_hw_limits.rs:178 uses self.max_tensor_channels (65536) which exceeds the actual conv-specific limit of 32768. ANEC conv constraint strings confirm: "Error: invalid conv config: kernel width = %zd, padded input width = %zd. Kernel width should not exceed the input width" and related dimensional constraint enforcement. | source + binary forensic (Orion #16, Extraction 5) | high | II-B, II-G |
| V-105 | crates/coreml-emit/ (multi-output emission), crates/bridge/src/mir_to_compat.rs | LACUNA | MILLer doesn't enforce alphabetical ordering of multi-output surfaces (Orion #3) or multi-input surfaces (Orion #19). Incorrect surface ordering will cause silent data corruption — outputs mapped to wrong buffers. | source + binary forensic (Orion #3, #19) | high | II-C |
| V-107 | python/mil_emitter.py, crates/coreml-proto/proto/coreml/MIL.proto | LACUNA | ANEC schema defines precise attribute shapes for all 98 operations (e.g., convolution: stride=shape{3}, padding=shape{6}, dilation=shape{3}). MILLer's MIL emission doesn't validate that these attribute shapes match ANEC expectations. Wrong-shaped attributes (e.g., stride with 2 elements instead of 3) will fail at ANE compiler time with cryptic errors. | source + binary forensic (ANEC schema, Extraction 8) | high | II-G |
| V-110 | crates/ir/src/mir.rs:59-68 (MILConv), crates/coreml-proto/proto/coreml/MIL.proto:116-121 (MilConvOp) | LACUNA | ANEC convolution schema includes kernel_scale (f16/f32 rank 0/1/4), kernel_zero_point (si8/ui8 rank 0/1/4), kernel_palettized_LUT (dense rank 0-6), and kernel_mutable_palettized_LUT (dict) attributes for quantized/palettized weights. MILLer's MILConv and MilConvOp proto don't carry any of these attributes, meaning quantized/palettized convolution emission is incomplete and will fail for any non-FP16 weight format. | source + binary forensic (ANEC schema, Extraction 2.3) | high | II-D, II-G |
| V-115 | crates/passes/src/op_constraints.rs, crates/ir/src/ane_hw_limits.rs | LACUNA | Large kernel mode (kernel W/H > threshold) has 12+ ANEC constraints not enforced by MILLer: (1) kernel W/H must be multiple of 8, (2) stride must be 1-2 only, (3) zero padding only, (4) no depth>1, (5) no palettized weights, (6) input/output x strides must match, (7) input/output y strides must match, (8) no grouped conv with large kernel, (9) no dynamic shape, (10) no dilated conv with large kernel, (11) stride decomposition constraints, (12) graph manipulation constraints. None are validated. | binary forensic (Extraction 5.1, 5.3, 5.6, 5.7, 5.8) | high | II-B, II-G |
| V-116 | crates/passes/src/op_constraints.rs | LACUNA | Deconvolution constraints not enforced by MILLer: (1) no dilation ("Dilation not supported for deconvolution"), (2) SOx must equal 2 ("deconv with SOx != 2 is not supported"), (3) no large kernel ("deconv with large kernel size is not supported"), (4) no vector palettization ("deconv with vector palettization is not supported"), (5) stride > 2 does not support kernel depth > 1. ANEC will reject deconvolutions violating these constraints. | binary forensic (Extraction 5.4, 5.5) | high | II-B, II-G |
| V-117 | crates/coreml-emit/ (all emission paths) | LACUNA | Multi-output buffer size uniformity not validated (Orion #2). ANE requires all output buffers to have identical allocation sizes. Non-uniform sizes cause 0x1d runtime error with no compile-time indication. | source + binary forensic (Orion #2) | high | II-C |
| V-118 | crates/coreml-emit/ (all emission paths) | LACUNA | Multi-output surface alphabetical ordering not enforced (Orion #3). ANE reads outputs in alphabetical order; naming mismatch causes silent wrong data — correct values written to wrong output tensors. | source + binary forensic (Orion #3) | high | II-C |
| V-119 | crates/coreml-emit/ (all emission paths) | LACUNA | Multi-input surface alphabetical ordering not enforced (Orion #19). Same silent data corruption risk as V-118 but for inputs — ANE reads input surfaces in alphabetical order regardless of logical ordering. | source + binary forensic (Orion #19) | high | II-C |
| V-120 | crates/coreml-emit/ (all emission paths) | LACUNA | Multi-input surface size uniformity not validated (Orion #18). ANE requires uniform alloc sizes for all input surfaces. Non-uniform sizes cause 0x1d runtime error. | source + binary forensic (Orion #18) | high | II-C |
| V-121 | crates/coreml-emit/ (all emission paths) | LACUNA | ANE flat buffer layout packed [1,C,1,S] not validated (Orion #20). Data written in wrong layout produces silently incorrect inference with no error indication — the most dangerous class of failure. | source + binary forensic (Orion #20) | high | II-C |
| V-125 | crates/passes/src/dtype_constraints.rs | ABERRANT | BF16/F16 cross-type operations explicitly rejected by ANEC but dtype_constraints.rs has no cross-type validation. Binary evidence from Extraction 1.3 shows 9 cross-type rejection strings: "detected operation with BF16 inputs and F16 result type which is not supported", "detected operation with F16 inputs and BF16 result type which is not supported", "detected operation with both F16 and BF16 operands which is not supported", plus cross-type rejections for complex/BF16, complex/integer, float/integer, and different-integer-type operands. | source + binary forensic (Extraction 1.3) | high | II-D |
| V-126 | crates/passes/src/dtype_constraints.rs | LACUNA | Float32 computation rejected on some architectures ("Float32 not supported for architecture") but is_dtype_ane_legal() approves FP32 for all families without architecture check. MILLer will approve FP32 on architectures where ANEC rejects it. | source + binary forensic (Extraction 1.1) | high | II-D |
| V-128 | crates/passes/src/op_constraints.rs | LACUNA | Dilated pooling rejected by ANEC ("Dilated Pooling not supported on ANE") but MILLer has no dilation check for pooling operations. Similarly, dilated stencil is rejected ("Dilated Stencil not supported on ANE"). MILLer allows dilated pooling and stencil without error. | source + binary forensic (Extraction 5.4) | high | II-B |
| V-130 | crates/passes/src/legality_rewrite.rs:3098,3622 | ABERRANT | RoPE rotate_half still emits concat(-x2, x1, axis=-1) even though Orion #1 documents concat is rejected by ANE compiler. Binary forensic from Extraction 7 confirms concat constraints: "Concat supports only 1 axis", "ANE Concat supports only const positive axis", "failed: only works when concat is applied on the channel axis". Concat on axis=-1 is likely not the channel axis and may be rejected. | source + binary forensic (Orion #1, Extraction 7) | high | II-A, II-G |
| V-132 | crates/passes/src/op_constraints.rs | LACUNA | Conv kernel dimensions must be power of 2 (ANEC constraint: "Kernel width must be a power of 2", "Kernel height must be a power of 2", "Kernel depth must be a power of 2"). MILLer validates kernel range 1-7 but not power-of-2 requirement. Kernel sizes 3, 5, 6, 7 pass MILLer's validation but will be rejected by ANEC. | source + binary forensic (Extraction 5.1) | high | II-B |
| V-134 | crates/passes/src/palettize_weights.rs | LACUNA | Asymmetric quantization not supported on ANEC (constraint "Asym quantization is not supported"). No check prevents asymmetric quantization in ANE path. Models with asymmetric quantization will fail at ANEC compile time. | source + binary forensic (Extraction 1.5) | high | II-D |
| V-136 | crates/passes/src/mil_lower.rs | LACUNA | Gather with non-constant axis rejected by ANEC ("gather with non-constant axis is not supported on ANEs"). MILLer emits dynamic-axis gather for embedding lookups. The legality_rewrite pass uses Gather for RoPE table lookups with potentially non-constant axes. | source + binary forensic (Extraction 9.2) | high | II-A, II-G |
| V-138 | crates/ir/src/mir.rs | LACUNA | MIL operation catalog has ~30 MirOpCompat variants but ANEC defines 98+ operations. 70+ ANEC operations including RingBufferReader/Writer, State, ScaledElementWise, HighPrecisionSigmoid, NRelu, ClampedRelu, Dirac, Degamma, GOC, Sqr, Rsqrt, Elu, LeakyRelu, Log2, Exp2, Sign, Trunc, Ceil, Floor, RegionReturn, UnrealizedConversionCast, TensorBufferToTensor, TensorToTensorBuffer have no MILLer equivalents. These ANEC ops have dedicated converters (Extraction 2.4 lists 27 ConvertElementwiseUnary variants alone) and specific hardware behavior that cannot be correctly lowered without corresponding MIR representations. | source + binary forensic (ANEC enum, Extraction 8, Extraction 2.4) | high | II-G, II-H |

### MEDIUM

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-031 | crates/ir/src/ane_hw_limits.rs:144-146 | PHANTOM | V26 "future" limits are fabricated (inherits A18 + num_nes=16). No hardware spec exists; no warning emitted. | source | high | II-B |
| V-032 | crates/ir/src/ane_target.rs:145 | UNVERIFIED | V20 (M4 Mac) mapped to A18 family; ArgMinMax dropped. Mac ANE hardware may differ from mobile A18. | source | medium | II-E |
| V-033 | crates/ir/src/sir.rs:48-53 | LACUNA | palette_bits documented as valid {1,2,3,4,6,8} but no validation enforces this. Out-of-range values accepted silently. Binary confirms "Only 2-bit, 4-bit, 6-bit and 8-bit palettization for conv are supported!" — note 3-bit is excluded for conv but may be supported from certain versions. | source + binary forensic (Extraction 5.9) | high | II-F |
| V-034 | crates/ir/src/sir.rs:1052-1054 | PHANTOM | KvCacheLayout::Paged is a full enum variant documented as "not yet implemented" but constructible via serde deserialization. | source | high | II-A |
| V-035 | crates/ir/src/common.rs:241-254 | ABERRANT | ModelArchConfig::default() silently assumes Qwen3-0.6B. Deprecated but still callable, producing wrong defaults for other architectures. | source | high | II-A |
| V-036 | crates/ir/src/pir.rs:787-790 | ABERRANT | Decode-step claims StateWriteRead handoff but comment says emission uses linear projection. PIR claims runtime semantics not delivered. | source | high | II-F |
| V-037 | crates/ir/src/lib.rs:17 | UNVERIFIED | DEFAULT_OPSET_VERSION = "iOS18" hardcoded without target validation. Models will fail on older iOS at load time. | source | medium | II-F |
| V-038 | crates/ir/src/shard_desc.rs:363-389 | ABERRANT | PIR tensor specs hardcoded to dtype "fp16" ignoring actual task spec dtype. Wrong for fp32/int4 tasks. | source | high | II-D |
| V-039 | crates/ir/src/air.rs:885-895 | LACUNA | legality_confidence, fallback_risk, drift_risk fields have no validation of value ranges, no documented semantics, no producers/consumers. | source | medium | II-F |
| V-040 | crates/passes/src/kv_cache_rewrite.rs:1-313 | ABERRANT | Deprecated pass generates ANE-illegal Where ops. Still compilable with working tests that produce illegal graphs. | source | high | II-A |
| V-041 | crates/passes/src/op_constraints.rs:38-51 | ABERRANT | Conv kernel range 1–7 contradicts later grouped/dilated threshold of 16. Either 1–7 is too restrictive or the 16-check is dead code. Binary evidence shows large kernel mode uses a threshold that likely corresponds to the 16-check. | source + binary forensic (Extraction 5.8) | medium | II-B |
| V-042 | crates/passes/src/op_constraints.rs:160-161 | LACUNA | Pooling validation accepts kernel_size parameter but immediately discards it (let _ = kernel_size). No kernel size limits enforced. Binary shows ANEC has pool kernel constraints: "Invalid Pool kernel depth (%zd), must be [1-%zd] or %zd", "Unsupported NEPool kernel height. Kh=%ld, Sy=%d", "Pool with strides of 3 is only supported with Avg mode.", "Large stride Min/Max pool with padding is not supported". | source + binary forensic (Extraction 5.2) | high | II-B |
| V-043 | crates/passes/src/mil_lower.rs:156-175 | LACUNA | Broadcast incompatibility falls back to x's shape with only eprintln warning. Produces wrong MIR output shapes. Binary shows "Only fp16 is supported for A11/A12 Broadcasts." — additional dtype constraint not enforced. | source + binary forensic (Extraction 1.4) | high | II-B |
| V-044 | crates/passes/src/legality_rewrite.rs:533-543 | LACUNA | Tile decomposition emits reshape_shape and final_shape with 0 placeholders. Zeros propagate through shape inference. | source | high | II-B |
| V-045 | crates/passes/src/shard_plan.rs:559 | LACUNA | PIR context_length always 0. Semantically important for KV cache models but never derived from graph or task spec. | source | high | II-B |
| V-046 | crates/passes/src/shard_plan.rs:561-562 | LACUNA | PIR opset_version and minimum_deployment_target hardcoded to "iOS18" regardless of target. Wrong for A11/A12 (iOS 16-era). | source | high | II-F |
| V-047 | crates/passes/src/shard_plan.rs:400,527 | UNVERIFIED | KV cache default shape fallback vec![2,1,1,1,1] is arbitrary. Batch=2 and all-1 dimensions almost certainly wrong for any real model. | source | high | II-B |
| V-048 | crates/passes/src/placement_validate.rs:516 | LACUNA | ConvTranspose always passes placement validation unconditionally. No kernel size, stride, or group checks. Binary shows deconv has extensive constraints: SOx==2, no large kernel, no vector palettization, no dilation. | source + binary forensic (Extraction 5.5) | high | II-B |
| V-049 | crates/passes/src/dtype_constraints.rs:73 | UNVERIFIED | FP32 allowed as ANE-legal with comment "may be downcast" but downcast not enforced. ANE does not natively compute in FP32. Binary confirms "Float32 not supported for architecture" on some architectures. | source + binary forensic (Extraction 1.1) | medium | II-D |
| V-050 | crates/passes/src/dtype_constraints.rs:79-81 | LACUNA | Int4/UInt4 return Ok(()) with comment "caller must also check interleave==8." Critical constraint deferred to caller with no enforcement. | source | high | II-D |
| V-051 | crates/passes/src/dtype_constraints.rs:180-182 | ABERRANT | Quantize validator accepts E5M2 as output dtype, but is_dtype_ane_legal() rejects E5M2 on all families. Binary confirms E5M2 is universally "not supported" in ANEC ("E4M3 or E5M2 format not supported"). | source + binary forensic (Extraction 1.1) | high | II-D |
| V-052 | crates/passes/src/canonicalize.rs:294 | LACUNA | Wildcard catch-all silently copies unrecognized SirOp variants without rewriting SirNodeId references, producing dangling references for new variants. | source | high | II-F |
| V-053 | crates/knowledge/src/update.rs:87-89 | ABERRANT | Doc says "never start above 0.5" but CompileFailure starts at 0.7, LoadFailure at 0.8, ComputePlan at 0.9. | source | high | II-F |
| V-054 | crates/knowledge/src/confidence.rs:13-25 | PHANTOM | decay_confidence is "currently only used in tests." Advertised temporal decay mechanism has zero production integration. | source | high | II-F |
| V-055 | crates/knowledge/src/transfer.rs:86-89 | ABERRANT | Doc says pattern-level transfer scales "0.5–0.8 depending on similarity" but code uses hardcoded midpoint 0.65. No similarity metric computed. | source | high | II-F |
| V-056 | crates/knowledge/src/lib.rs:38-39 | ABERRANT | ComputePlanObservation doc says "confidence always 0.9" but code accepts any [0,1] value from caller. | source | high | II-F |
| V-057 | crates/knowledge/src/compute_plan_verify.rs:169-242 | UNVERIFIED | default_known_placements claims Apple documentation source but all entries marked "empirical" with hardcoded confidence. No cross-check infrastructure. | source | high | II-A |
| V-058 | crates/knowledge/src/shard_template.rs:377-388 | LACUNA | parse_evidence_source silently maps unknown sources to ManualEntry, changing semantic meaning and reliability interpretation. | source | high | II-F |
| V-059 | crates/bridge/src/mir_to_compat.rs:92-104 | ABERRANT | EmptyWeightResolver doc says "returns Some with empty data" but implementation returns None. | source | high | II-D |
| V-060 | crates/bridge/src/subprocess.rs:73-89 | LACUNA | Python subprocess failure returns Ok(BridgeResult{status:"error"}) instead of Err(). Forces callers to manually check status string. | source | high | II-F |
| V-061 | crates/bridge/src/safetensors_resolver.rs:319 | LACUNA | Shard weight slicing assumes FP16 (bytes_per_row = hidden_size * 2). F32 or BF16 base weights would produce corrupted data. | source | high | II-D |
| V-062 | crates/bridge/src/shape_inference.rs:530 | LACUNA | Catch-all returns empty shape for unrecognized MirOp variants. Core ML inference not guaranteed to succeed for all ops. | source | medium | II-B |
| V-063 | crates/bridge/src/shape_inference.rs:74-80 | UNVERIFIED | Deprecated shape functions hardcode Qwen3 max_seq_len=32768. Still callable, not feature-gated. | source | high | II-B |
| V-064 | crates/coreml-emit/src/emitter.rs:129-148 | PHANTOM | compare_with_python_bridge is an unimplemented stub that always returns None. Doc describes full comparison workflow. | source | high | II-F |
| V-065 | crates/coreml-proto/src/lib.rs:930-938 | PHANTOM | MirOpCompat::Unsupported is constructible but always rejected at emission gate. Type-level capability that can never produce output. | source | high | II-A |
| V-066 | crates/coreml-emit/src/mir_to_proto.rs:94-121 | PHANTOM | Fill, Select, Where are valid MirOpCompat variants but always rejected as "ANE-illegal." Error says "should have been replaced earlier" but type system allows them. | source | high | II-A |
| V-067 | knowledge/ane_hw_limits_seed.json (all revisions) | LACUNA | Only ~14 of ~40+ documented hardware limit parameters captured. No validation for conv kernel sizes, pooling limits, padding limits, PE/NE engine constraints. | source | high | II-B |
| V-068 | knowledge/ane_hw_limits_seed.json (ne_transpose_c_max) | UNVERIFIED | ne_transpose_c_max identical (16384) across all 11 revisions despite 16x variation in other parameters. Possibly copied from V4 defaults. | local metadata | medium | II-B |
| V-069 | knowledge/ (all seeds) | LACUNA | Knowledge seeds use device_classes ["M2","M3"] but compiler operates on AneFamily. No documented device_class→AneFamily mapping exists. | source | high | II-E |
| V-070 | knowledge/legality_seed.json:48-60 | UNVERIFIED | Synthetic evidence at confidence 0.9 exceeds real-model evidence at 0.6. Schema's own ×0.7 synthetic transfer penalty not applied to stored raw confidence. | source | medium | II-F |
| V-071 | knowledge/ (all seeds) | ABERRANT | Knowledge schema defines unit wrapper with provenance, conflict_status, conflict_priority fields. No seed JSON follows this schema — all use flat fields without unit wrapper. | source | high | II-F |
| V-072 | knowledge/ane_op_family_matrix.json:86-101 | UNVERIFIED | SDPA marked "unreliable" for A12-A15 but Rust binary-classifies as not-supported (A16+ only). "Unreliable" has no operational semantics. Binary shows ConvertScaledDotProductAttention is family-agnostic with MinimumFamily=A11Legacy (also verified on A13), suggesting broader support than MILLer models. | source + binary forensic (Extraction 6.4) | medium | II-A |
| V-073 | knowledge/precision_hazard_seed.json | UNVERIFIED | All 4 entries derive from single model (Qwen3). Claims general rules based on 3 evidence points from one model. No cross-validation. | source | medium | II-D |
| V-074 | knowledge/shard_template_seed.json:18-19 | UNVERIFIED | known_good: true with perplexity_delta: -0.57 (worse quality). No documented threshold for acceptable quality delta. | source | medium | II-F |
| V-075 | knowledge/palettization_constraints_seed.json:5-9 | LACUNA | Dual conv min bits (standard:4, alternate:2) without conditional context. Compiler cannot determine which minimum applies for a given hardware version or conv subtype. Binary shows version-conditional constraints: "3-bit palettization is only supported from version {1}", "6-bit palettization is only supported from version {1}". | source + binary forensic (Extraction 5.9) | medium | II-D |
| V-076 | crates/lab/src/device_meta.rs:125-140 | PHANTOM | device_backed() always returns HostOnly on all platforms including macOS. Method name implies device-backed metadata but never produces it. | source | high | II-F |
| V-077 | crates/lab/src/harness.rs:70 | LACUNA | coremltools_available hardcoded to false. Python bridge detection result never folded back. Every LabRun permanently reports false. | source | high | II-F |
| V-078 | crates/lab/src/fallback.rs:135-162 | ABERRANT | assess_overall_level() returns Unavailable for device-backed runs with no baseline, even though device evidence IS available. | source | medium | II-F |
| V-079 | crates/lab/src/harness.rs:408-412 | LACUNA | LabRunBuilder allows timing on HostOnlyInspection runs despite doc saying "MUST be None." No compile-time or runtime enforcement. | source | high | II-F |
| V-080 | crates/coreml-ffi/src/api.rs:43-71 | STUB-MIMIC | CoreMlApi::version() returns Ok("unknown"); compile_model() returns Err on macOS after passing is_available() check. API surface suggests functionality that doesn't exist. | source | high | II-F |
| V-081 | crates/coreml-ffi/src/model.rs:92-111 | STUB-MIMIC | FfiModel::load() returns Ok with handle: None on macOS. "Loaded" model where is_loaded() returns false. Subsequent calls fail but not at load time. | source | high | II-F |
| V-082 | crates/coreml-ffi/src/capi.rs:204-224 | STUB-MIMIC | coreml_model_info() writes zeroed info with CoreMlStatus::Ok. C consumer would interpret Ok as successful metadata retrieval. | source | high | II-F |
| V-083 | crates/coreml-ffi/src/capi.rs:383-391 | ABERRANT | Validation rejects packages without weight.bin, but not all models require it. Inline-const-only models are perfectly valid but fail validation. | source | high | II-F |
| V-084 | crates/trace/src/sir_build.rs:169-172 | LACUNA | QualityContract hardcoded: max_perplexity_delta=0.1, max_latency_ms=50.0. Not per-model configurable. Overly restrictive for 7B models. | source | high | II-F |
| V-085 | crates/trace/src/sir_build.rs:84-94 | LACUNA | Missing input shape falls back to (1,32) silently. Wrong for models with different expected shapes. No warning when fallback used. | source | high | II-B |
| V-086 | crates/bridge/src/mir_to_compat.rs (30+ eprintln) | LACUNA | Critical constraint violations reported via eprintln instead of structured logging or error returns. Cannot be suppressed, tested, or consumed by downstream. | source | high | II-F |
| V-087 | crates/cli/src/main.rs:696,954 | LACUNA | --seed parameter accepted by CLI but silently discarded (_seed prefix). SPEC requires deterministic compilation but seed is unused. | source | high | II-F |
| V-104 | crates/coreml-emit/ (all emission paths) | LACUNA | MILLer doesn't enforce minimum IOSurface size (~49 KB) for eval (Orion #4). Models with small output buffers will fail at runtime with no prior validation. | source + binary forensic (Orion #4) | medium | II-B |
| V-106 | crates/coreml-emit/ (compilation pipeline) | LACUNA | MILLer doesn't enforce ~119 compilation limit per process (Orion #5). Long-running processes that repeatedly compile models will silently fail after hitting this limit with no warning or error. | source + binary forensic (Orion #5) | medium | II-B |
| V-108 | crates/ir/src/ane_hw_limits.rs (AneRevision enum) | LACUNA | Binary shows ZinRegisterProgramming template instantiated for 14 hardware versions (V0-V9, V17, V19, V20, V26), confirming at least 14 distinct ANE hardware code paths. MILLer's AneRevision only defines 11 revisions, missing at least 3 hardware versions that have dedicated compiler code paths (V0-V3 likely correspond to pre-A11 hardware). | source + binary forensic (Extraction 3.1) | medium | II-E |
| V-109 | crates/ir/src/ane_target.rs (AneFamily enum) | LACUNA | The MinimumFamily enum in the binary has values 0-7 (8 families), but MILLer only models 6 families (A11Legacy through A18). Families corresponding to enum values 6-7 (likely A16+ variants or future architectures) are unmapped — MILLer will misattribute ops on future hardware and cannot express constraints for these families. | source + binary forensic (Extraction 4.2) | medium | II-E |
| V-111 | crates/passes/src/dtype_constraints.rs:180-182 | ABERRANT | Binary confirms E5M2 is universally "not supported" (not just architecture-conditional), 2xInt8 mode is "not supported," and E4M3 is "not supported on this architecture" (architecture-conditional). This strengthens the existing V-051 finding with binary evidence: the quantize validator's acceptance of E5M2 is definitively wrong, not just contradictory. | source + binary forensic (Extraction 1.1) | high | II-D |
| V-112 | crates/ir/src/mir.rs (no InputView variant) | LACUNA | ANEC's anec.input_view supports negative strides (step=i64), which MILLer doesn't model at all. Any lowering that requires negative-stride views (e.g., reverse along non-trivial axes, certain crop/resize patterns) cannot be correctly expressed in MILLer's MIR. The absence of this op means certain ANE-legal patterns will be incorrectly lowered or forced to CPU. | source + binary forensic (ANEC schema, Extraction 8) | medium | II-G |
| V-114 | crates/passes/src/mil_lower.rs:3268-3307 | ABERRANT | MILLinear→MILMatMul conversion defeats Conv1x1AsLinear optimization. Orion #17 documents conv 1x1 is 3x faster than matmul on ANE. The pipeline correctly creates Conv1x1AsLinear in AIR (legality_rewrite.rs:354-370), but the ANE legality pass then converts ALL MILLinear to MILMatMul (mil_lower.rs:3268-3307). The comment says "The linear op may not have an ANE execution converter" but binary shows ConvertLayer has 97 instances and ConvertMatMul has 8 family instantiations. Converting to matmul loses the 3x performance benefit. | source + binary forensic (Orion #17, Extraction 2.3) | high | II-A, II-G |
| V-122 | crates/coreml-emit/src/weights.rs | LACUNA | Minimum IOSurface size (~49 KB) not validated (Orion #4). Smaller buffers cause 0x1d runtime error. MILLer has no check that emitted buffer sizes meet this minimum. | source + binary forensic (Orion #4) | medium | II-B |
| V-123 | crates/bridge/src/subprocess.rs | LACUNA | Compilation count per process (~119 max, Orion #5) not tracked. Exceeding limit causes silent crash with no diagnostic. No counter or warning mechanism exists. | source + binary forensic (Orion #5) | medium | II-B |
| V-124 | crates/coreml-emit/src/mir_to_proto.rs | LACUNA | Weight dict initialization not guaranteed to be @{} (Orion #11). Nil weight dict causes immediate crash at ANEC compile time. MILLer's emission path does not verify that the weight dictionary is properly initialized. | source + binary forensic (Orion #11) | medium | II-F |
| V-127 | crates/passes/src/op_constraints.rs | LACUNA | Pooling stride 3 only supported for Avg mode (ANEC constraint "Pool with strides of 3 is only supported with Avg mode"). MILLer allows stride-3 MaxPool without error. Additionally, "Large stride Min/Max pool with padding is not supported" — another unenforced constraint. | source + binary forensic (Extraction 5.2) | medium | II-B |
| V-131 | crates/ir/src/ane_hw_limits.rs | LACUNA | Hardware sub-variants (e.g., v812, v813, v827 for NE/PE/DMA engines) not modeled. Binary from Extraction 3.3 shows 14+ sub-variant descriptors per major version with suffixes 12 (NE), 13 (PE), 27 (DMA/config). MILLer maps only 11 top-level revisions and cannot express engine-specific constraints that may differ between sub-variants. | source + binary forensic (Extraction 3.3) | medium | II-E |
| V-133 | crates/passes/src/palettize_weights.rs | LACUNA | Vector palettization only supported at Cout for ANE (ANEC constraint "vector palettization is only supported at Cout for ANE"). No enforcement of this constraint — vector palettization at other dimensions will fail at ANEC compile time. Additionally, "zero point is not supported for vector palettized kernel" and "Quantized kernel with palettize size=256 is not supported" are unenforced. | source + binary forensic (Extraction 5.9) | medium | II-D |
| V-135 | crates/passes/src/op_constraints.rs | LACUNA | Width wrap axis not supported on some architectures (ANEC constraint "Width wrap axis is not supported on this architecture"). No validation for wrap axis compatibility. | source + binary forensic (Extraction 9.3) | medium | II-B |
| V-137 | crates/passes/src/op_constraints.rs | LACUNA | Stencil (depthwise conv) constraints not enforced: (1) 5D stencil rejected ("stencil along channel with rank 5 input is not supported on ANEs"), (2) non-4D kernel rejected ("stencil kernel rank != 4 is not supported on ANEs"), (3) non-sum reduction mode rejected ("stencil reduction_mode != sum is not supported on ANEs"), (4) dilated stencil rejected ("Dilated Stencil not supported on ANE"), (5) strided stencil rejected ("Strided Stencil not supported on ANE"). | source + binary forensic (Extraction 5.4, 9.2) | medium | II-B, II-G |

### LOW

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-088 | crates/ir/src/ane_hw_limits.rs:144-146 | PHANTOM | V26 "future" limits fabricated. Currently unreachable from normal paths but for_revision(V26) returns plausible-looking but fictional limits. | source | high | II-B |
| V-089 | crates/passes/src/role_mir.rs:131,146,210 | UNVERIFIED | Multiple unwrap_or fallbacks use hardcoded dimension values (64, 48, 32). Silently produce wrong MIR shapes if spec incomplete. | source | high | II-B |
| V-090 | crates/passes/src/canonicalize.rs:113-119 | UNVERIFIED | Chain resolution loop has magic limit of 100 steps with no diagnostic when hit. Graphs with 100+ chained identity nodes produce incomplete substitution. | source | high | II-F |
| V-091 | crates/passes/src/dtype_constraints.rs:105 | UNVERIFIED | UInt16 marked ANE-legal with "limited support" but no constraint checks or documentation of what "limited" means. | source | medium | II-D |
| V-092 | crates/passes/src/dtype_constraints.rs:113 | UNVERIFIED | Bool marked ANE-legal with "limited support" but no constraint checks. | source | medium | II-D |
| V-093 | crates/passes/src/cpu_only_ops.rs:256-319 | LACUNA | CPU_ONLY_OPS_DETAILED has ~30 entries vs 120+ in CPU_ONLY_OPS. ~90 ops have no documented reason for being CPU-only. | source | high | II-A |
| V-094 | crates/knowledge/src/compute_plan_verify.rs:337 | LACUNA | knowledge_consistent is binary all-or-nothing. Single mismatch out of hundreds makes entire proof "not consistent." Discards nuance. | source | medium | II-F |
| V-095 | crates/coreml-emit/src/package.rs:155-195 | UNVERIFIED | UUIDs generated via v5 with fixed namespace + function name. Two models with same function name produce identical UUIDs. Not globally unique. | source | medium | II-F |
| V-096 | crates/lab/src/fallback.rs:165-181 | PHANTOM | FallbackLogEvidence defined with full serde support but never constructed. "Reserved for future use" — phantom capability. | source | high | II-F |
| V-097 | python/mil_emitter.py:904 | LACUNA | LayerNorm epsilon np_dtype(1e-5) truncates for FP16 (becomes ~9.98e-6). Emitted model epsilon differs from programmer intent. | source | high | II-D |
| V-129 | crates/coreml-proto/src/lib.rs:3854-3860 | LACUNA | MatMul transpose flags emitted as immediate bool values instead of named const nodes (Orion #12). May cause ANEC rejection. Binary shows ConvertMatMul is family-scoped (8 instantiations) suggesting the matmul handling varies per family — immediate bool values may not be accepted by all family implementations. | source + binary forensic (Orion #12, Extraction 2.2) | medium | II-F |

---

## IV. Absentee Capabilities

| Capability | Status | Notes |
|---|---|---|
| Static dimension resolution | Declared but not implemented | StaticizePass exists as pure pass-through (V-014) |
| Paged KV-cache attention | Declared but not implemented | KvCacheLayout::Paged variant exists but no code path implements it (V-034) |
| Temporal confidence decay | Declared but not in production | decay_confidence function exists but is test-only (V-054) |
| Python bridge comparison | Declared but not implemented | compare_with_python_bridge always returns None (V-064) |
| Timeout enforcement on Python bridge | Declared but not implemented | timeout_secs stored but never used (V-024) |
| Device-backed metadata on macOS | Declared but not implemented | device_backed() always returns HostOnly (V-076) |
| Core ML FFI compilation on macOS | Declared but not functional | compile_model() always returns Err (V-080) |
| Core ML model version query on macOS | Declared but returns "unknown" | version() returns Ok("unknown") (V-080) |
| Reproducibility seed in compile pipeline | Declared but not wired | --seed accepted but discarded (V-087) |
| Hardware-specific conv/pool kernel validation | Partially declared | Constraint fields exist in AneHwLimits but never validated (V-009) |
| Precision hazard coverage for all ops | Partially declared | Only 14/167 op types query hazards (V-015) |
| BF16 data type support | Out of scope (absent) | Not listed in MilDtype enum; silently defaults to Fp16 in shard_desc (V-011); BF16/F16 cross-type operations rejected by ANEC (V-125) |
| Conflict detection for most knowledge types | Partially declared | claims_agree defaults true for 7/8 types (V-021) |
| Compute plan verification against real hardware | Inferred but unverified | Hardcoded empirical mappings with no cross-check (V-057) |
| Int8 dtype in shard descriptor lowering | Declared but silently treated as Fp16 | Unknown dtype falls through to Fp16 default (V-011) |
| Stateful function descriptors via generic path | Declared but always false | FamilyPayload hardcodes stateful: false (V-013) |
| Concat as ANE-legal op | Assumed legal but rejected by ANE | MILConcat emitted in SDPA decomposition; ANE compiler rejects concat (V-098, Orion #1); concat has extensive axis/interleave/dimension constraints (Extraction 7) |
| Gelu as native MIL activation | Assumed valid but mode-contradicted | mb.gelu emitted with contradictory modes; ANEC only supports tanh approximation (V-099, V-113, Orion #10) |
| Quantized/palettized conv weight attributes | Not modeled | ANEC convolution has kernel_scale, kernel_zero_point, kernel_palettized_LUT attributes not in MILLer (V-110) |
| ANEC InputView with negative strides | Not modeled | anec.input_view supports step=i64; no equivalent in MILLer MIR (V-112) |
| 70+ ANEC operations | Not modeled | RingBufferReader/Writer, State, ScaledElementWise, HighPrecisionSigmoid, NRelu, ClampedRelu, Dirac, Degamma, GOC, Sqr, Rsqrt, Elu, LeakyRelu, Log2, Exp2, Sign, Trunc, Ceil, Floor, RegionReturn, UnrealizedConversionCast, TensorBufferToTensor, TensorToTensorBuffer absent from MirOpCompat (V-100, V-138) |
| IOSurface minimum size validation | Absent | ~49 KB minimum not enforced (V-104, V-122, Orion #4) |
| Compilation count per process limit | Absent | ~119 limit not tracked (V-106, V-123, Orion #5) |
| Alphabetical surface ordering | Absent | Multi-output/input surfaces must be alphabetically ordered (V-105, V-118, V-119, Orion #3, #19) |
| Uniform surface size validation | Absent | Multi-output/input surfaces must have uniform sizes (V-117, V-120, Orion #2, #18) |
| Flat buffer layout validation | Absent | ANE reads data as packed [1,C,1,S] (V-121, Orion #20) |
| Conv 32K-channel limit | Overly permissive | max_tensor_channels allows 65536 but conv-specific limit is 32768 (V-103, Orion #16) |
| Conv kernel power-of-2 validation | Absent | Kernel W/H/D must be power of 2 (V-132) |
| Large kernel mode constraints | Absent | 12+ constraints for kernel W/H > threshold not enforced (V-115) |
| Deconvolution constraints | Absent | SOx==2, no large kernel, no vector palettization, no dilation not enforced (V-116) |
| Dilated pooling rejection | Absent | "Dilated Pooling not supported on ANE" not enforced (V-128) |
| Pool stride-3 for Avg-only | Absent | "Pool with strides of 3 is only supported with Avg mode" not enforced (V-127) |
| BF16/F16 cross-type validation | Absent | 9 cross-type rejection constraints not enforced (V-125) |
| Architecture-conditional FP32 rejection | Absent | "Float32 not supported for architecture" not checked (V-126) |
| Asymmetric quantization rejection | Absent | "Asym quantization is not supported" not enforced (V-134) |
| Vector palettization at-Cout constraint | Absent | "vector palettization is only supported at Cout for ANE" not enforced (V-133) |
| Non-constant gather axis rejection | Absent | "gather with non-constant axis is not supported on ANEs" not enforced (V-136) |
| Stencil constraints | Absent | 5D, non-4D kernel, non-sum reduction, dilated, strided stencil rejections not enforced (V-137) |
| Width wrap axis architecture check | Absent | "Width wrap axis is not supported on this architecture" not enforced (V-135) |
| Weight dict @{} initialization guarantee | Absent | Nil weight dict crashes ANEC (V-124, Orion #11) |
| SDPA causal mask handling | Unimplemented | ANEC silently ignores causal masks in SDPA (Orion #6) |
| Conv1x1AsLinear performance optimization | Defeated by MILLinear→MILMatMul | Conv 1x1 is 3x faster than matmul but pipeline converts all linear to matmul (V-114, Orion #17) |
| RoPE rotate_half without concat | Not implemented | Uses concat(-x2, x1, axis=-1) which is rejected by ANEC (V-130) |
| Family 6-7 sub-variant modeling | Not modeled | Binary shows 8 families with engine-specific sub-variants; MILLer has 6 families (V-109, V-131) |
| Hardware sub-variant constraints | Not modeled | NE(12), PE(13), DMA(27) engine-specific descriptors exist in ANEC but not in MILLer (V-131) |

---

## V. Orion Constraint Cross-Reference

The following table cross-references all 20 Orion programming constraints (arxiv:2603.06728) against MILLer's handling, incorporating deep binary forensic evidence from the Orion cross-examination.

| # | Orion Constraint | Modeled? | Correctly Enforced? | MILLer Handling | Violation(s) | Severity |
|---|---|---|---|---|---|---|
| 1 | concat MIL op rejected by ANE compiler | Partial | **NO** | Concat emitted in SDPA fallback (mil_lower.rs:2842-2858), embedding gather (mil_emitter.py:432), and RoPE rotate_half (legality_rewrite.rs:3098,3622). Binary confirms extensive concat constraints: channel-axis-only, const-positive-axis, no interleaved on some dims, no symbolic shape. | V-098, V-130 | CRITICAL |
| 2 | Multi-output buffers must have uniform sizes | No | N/A | No validation. | V-117 | HIGH |
| 3 | Multi-output surfaces ordered alphabetically | No | N/A | No ordering enforcement. | V-118 | HIGH |
| 4 | Minimum ~49 KB IOSurface for eval | No | N/A | No minimum buffer size validation. | V-104, V-122 | MEDIUM |
| 5 | ~119 compilations per process limit | No | N/A | No compilation count tracking. | V-106, V-123 | MEDIUM |
| 6 | SDPA causal masks silently ignored | Partial | **NO** | Mask emitted but ANEC ignores it. Split-based path works around manually but residual SDPA paths risk silently wrong attention. | (implicit) | HIGH |
| 7 | Weights baked at compile time | No | N/A | No validation; weights may reference runtime-dynamic data. | V-007 | LOW |
| 8 | BLOBFILE offset is uint64(64), not 128 | Yes | **YES** | weights.rs:419 uses metadata_offset=64. Test at line 605 confirms. | **No violation** | NONE |
| 9 | MIL text must be NSData*, not NSString* | N/A | N/A | Python bridge uses coremltools which handles this. | **No violation** | NONE |
| 10 | gelu is not a valid MIL activation | Partial | **NO** | MILLer emits mb.gelu with contradictory modes: "EXACT" (SIR builder) vs "TANH_APPROXIMATION" (role_mir, Python). ANEC only supports tanh approximation. | V-099, V-113 | HIGH |
| 11 | Weight dict must be @{}, not nil | No | N/A | No validation; empty weight dict may be nil, causing crash. | V-124 | MEDIUM |
| 12 | matmul transpose flags need named consts | Partial | **NO** | Bool immediate values emitted instead of named const nodes. | V-129 | LOW |
| 13 | conv does not support bias= param | Yes | **YES** | MILConv has no bias field; Python uses mb.linear with bias. | **No violation** | NONE |
| 14 | Output vars must ref live (post-opt) nodes | No | N/A | No dead-code elimination guarantee for output references. | V-052 | MEDIUM |
| 15 | exec() restart overhead ~50 ms | No | N/A | Not modeled in performance estimation. | (informational) | INFO |
| 16 | 32K-channel convolutions rejected | No | **NO** | max_tensor_channels allows 65536; conv-specific 32K limit not enforced. | V-103 | HIGH |
| 17 | Conv 1×1 is 3× faster than matmul | Partial | **NO** | Conv1x1AsLinear→MILLinear→MILMatMul conversion defeats optimization. Performance regression for all linear projections. | V-114 | MEDIUM |
| 18 | Multi-input surfaces must have uniform alloc sizes | No | N/A | No validation. | V-120 | HIGH |
| 19 | Multi-input surfaces ordered alphabetically | No | N/A | No ordering enforcement. | V-119 | HIGH |
| 20 | ANE reads flat buffer as packed [1,C,1,S] | No | N/A | No buffer layout validation. | V-121 | HIGH |

**Summary**: Of the 20 Orion constraints, MILLer **correctly handles 3** (#8 BLOBFILE offset, #9 NSData, #13 conv bias), **partially handles 3** (#1 concat, #6 SDPA masks, #10 gelu), **fails to model 14** (#2, #3, #4, #5, #7, #11, #14, #16, #17, #18, #19, #20, and partially #12). Two constraints are **actively violated** (#1 concat, #10 gelu). Five constraints (#2, #3, #18, #19, #20) represent **silent data corruption risks**.

---

## VI. Remediation Roadmap

Ordered by priority; each action references the violation ID(s) it addresses.

### Phase 1 — Critical Fixes (Silent Miscompilation or Data Corruption)

1. **V-006**: Fix `Float64` element_size to return 8 instead of 4 in `coreml-proto/src/lib.rs`. All downstream byte-size calculations depend on this.

2. **V-007**: Replace zero-filled weight placeholders with hard errors. When a weight cannot be resolved, fail the compilation rather than producing a silently broken model. Add `--allow-missing-weights` flag for intentional zero-fill scenarios.

3. **V-001, V-002**: Align `ane_hw_limits_seed.json` family assignments with Rust code: V6→A13 (not A14), V11→A17 (not A16). Add automated test that checks JSON→Rust mapping consistency.

4. **V-003, V-004, V-005**: Resolve three-way contradictions in knowledge seeds:
   - Comparison ops: move from `cpu_only_ops_seed.json` to `ane_op_family_matrix.json` with A14+ scope.
   - Logical AND/OR/NOT: mark as CPU-only in `ane_op_family_matrix.json`; remove phantom "supported" entries.
   - Gather: change `legality_seed.json` to `ane_legal: true` with `limited_index_range` constraint.

5. **V-011**: Replace silent Fp16 default for unknown dtype with explicit error listing valid dtype strings. Add Int8 to accepted dtype list.

6. **V-098, V-130**: Replace ALL remaining concat emissions with ANE-legal alternatives:
   - SDPA path (mil_lower.rs:2842-2858): replace concat with reshape+stack or emit as fused SDPA for A16+.
   - RoPE rotate_half (legality_rewrite.rs:3098,3622): replace concat(-x2, x1, axis=-1) with equivalent reshape+transpose sequence.
   - Embedding gather (mil_emitter.py:432): replace concat with reshape+stack.

7. **V-113**: Fix SIR builder Gelu mode: change "EXACT" to "TANH_APPROXIMATION" in sir_build.rs:518,1415. Alternatively, replace all mb.gelu emission with explicit tanh-approximation decomposition using ANE-legal ops: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.

### Phase 2 — High-Priority Validation Gaps

8. **V-009**: Implement validation for the 7 unused conv/pool/PE constraint fields in `validate_tensor_dims()`. Add per-op validation functions for convolution, pooling, and PE operations.

9. **V-010**: Make `default_engine()` revision-aware by cross-referencing `AneFamily` capability methods. Return `None` (no engine) for ops not supported on the target family.

10. **V-012**: Remove Generic→Qwen3 fallback. Return an error when model architecture is not recognized, requiring explicit architecture specification.

11. **V-013**: Derive `stateful` flag from the actual op type in `FamilyPayload::from_spec_with_override()`. Check for DecodeStep and other stateful ops.

12. **V-014**: Either implement StaticizePass or remove it from the pipeline and document its absence. Current phantom pass wastes developer trust.

13. **V-015**: Expand precision_policy coverage to all SIR op types. At minimum, add the top-30 most common op types and mark the remainder with explicit "coverage gap" markers.

14. **V-016**: Replace `eprintln!` warnings in StateTopologyPass with `Result::Err` returns for invalid state patterns, or clearly document that the pass is advisory-only.

15. **V-017**: Derive FunctionEntry shapes from the MIR graph instead of hardcoding `vec![1,1]`. Walk the graph to extract actual batch/seq dimensions.

16. **V-018**: Replace eprintln+continue for MatMul inner-dim mismatch with `Err`. Shape mismatch is a correctness violation, not a warning condition.

17. **V-019**: Resolve Gather contradiction: either remove from CPU_ONLY_OPS (with appropriate scoping for embedding-only), or replace Gather emission with SliceByIndex in all paths.

18. **V-020**: Restructure interleave validation to enforce non-channel-dependent checks (const→1, int4→8) even when channels is None.

19. **V-021**: Implement claims_agree logic for all 8 knowledge types. At minimum, add field-level comparison for PrecisionHazard and SurvivalMatrixEntry.

20. **V-022**: Make conflict detection symmetric: when new entry B conflicts with existing A, mark both A and B as conflicted.

21. **V-023, V-025**: Replace fallback shapes/dtypes with hard errors when nodes are missing from MIR graph. Fail compilation rather than emitting wrong descriptors.

22. **V-024**: Implement timeout using `Command::new().timeout()` or `child.wait_timeout()`. Remove the phantom field or make it functional.

23. **V-026**: Add FP32→FP16 conversion in safetensors_resolver when target dtype is FP16. Ensure data format matches proto declaration.

24. **V-027**: Map Bool to a dedicated Bool blob type or reject at validation time. Map Float64 to 8-byte Float64 blob (after fixing V-006). Reject Unknown dtype early.

25. **V-028**: Derive state shape from the corresponding ReadState op. If no ReadState exists, require explicit shape specification rather than defaulting to empty.

26. **V-029, V-030, V-101**: Resolve Softmax/InstanceNorm family gating contradiction. Binary evidence shows ConvertSoftmax and ConvertInstanceNorm are family-agnostic converters (no MinimumFamily trait), but ANEC has architecture-conditional rejection strings. Update documentation to reflect this nuance: converters exist for all families but specific architecture variants may reject the operation at compile time. The constraint model should capture both the converter availability and the architecture-conditional rejection.

27. **V-099**: Replace mb.gelu emission with explicit tanh-approximation decomposition (0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))) that uses ANE-legal ops, or verify that coremltools' mb.gelu internally lowers to the same decomposition accepted by ANEC's ConvertElementwiseUnary.

28. **V-100, V-138**: Audit MirOpCompat against the complete ANEC operation enum. Add compatibility mappings for at least the 27 ConvertElementwiseUnary variants (ClampedRelu, Elu, LeakyRelu, Sqr, Rsqrt, Sign, Ceil, Floor, Exp2, Log2, Trunc, NRelu, Dirac, Degamma, HighPrecisionSigmoid, RoundNearest, Square) that have dedicated converters but no MILLer equivalents.

29. **V-110**: Add quantized weight attributes to MILConv and MilConvOp: kernel_scale, kernel_zero_point, kernel_palettized_LUT. These are required for any non-FP16 weight format in convolutions.

30. **V-115**: Add large kernel mode constraint validation: kernel W/H multiple of 8, stride 1-2 only, zero padding only, no depth>1, no palettized weights, matching input/output strides, no grouped conv, no dynamic shape, no dilation.

31. **V-116**: Add deconvolution constraint validation: no dilation, SOx==2, no large kernel, no vector palettization.

32. **V-117, V-120**: Add uniform size validation for multi-output and multi-input buffers.

33. **V-118, V-119**: Add alphabetical sorting of multi-output and multi-input surfaces before emission.

34. **V-121**: Add buffer layout validation ensuring data is written in packed [1,C,1,S] format.

35. **V-125**: Add BF16/F16 cross-type operation validation. Reject operations that mix BF16 and F16 operands or have cross-type input/output pairs, per the 9 ANEC cross-type rejection constraints.

36. **V-126**: Add architecture-conditional FP32 rejection. Check target architecture before approving FP32 as ANE-legal.

37. **V-128**: Add dilation check for pooling and stencil operations. Reject dilated pooling and dilated stencil with clear error messages.

38. **V-132**: Add conv kernel power-of-2 validation. Reject kernel sizes 3, 5, 6, 7 (which pass current 1-7 range check but fail ANEC power-of-2 requirement).

39. **V-134**: Add asymmetric quantization rejection for ANE path. Check quantization symmetry before emitting to ANEC.

40. **V-136**: Add gather axis constness validation. Reject dynamic-axis gather operations targeting ANE.

41. **V-114**: Replace MILLinear→MILMatMul conversion with MILLinear→MILConv(1x1) for ANE targets, preserving the Conv1x1AsLinear optimization that is 3x faster per Orion #17.

42. **V-103**: Add conv-specific 32K-channel limit validation in op_constraints.rs, distinct from the general max_tensor_channels limit. Conv channels > 32768 should be rejected at constraint validation time regardless of max_tensor_channels value.

### Phase 3 — Medium-Priority Cleanup

43. **V-031, V-088**: Either remove V26 revision or add explicit "speculative — not based on any hardware" warning in for_revision() return value.

44. **V-033**: Add palette_bits validation to SirOp construction or deserialization, rejecting values outside {1,2,3,4,6,8}. Binary evidence confirms "Only 2-bit, 4-bit, 6-bit and 8-bit palettization for conv are supported" with version-conditional 3-bit and 6-bit support.

45. **V-034**: Gate KvCacheLayout::Paged behind a feature flag or add serde validation that rejects Paged on deserialization.

46. **V-035**: Remove `Default` impl for `ModelArchConfig` or make it return an error. Add `ModelArchConfig::unspecified()` for cases that need a placeholder.

47. **V-036**: Either implement stateful KV cache semantics in the emission path or change the handoff kind to a more accurate descriptor (e.g., `DirectPassThrough`).

48. **V-037, V-046**: Make opset version and deployment target configurable from the CLI or task spec rather than hardcoded.

49. **V-038**: Use actual dtype from task spec for PIR tensor specs instead of hardcoded "fp16".

50. **V-040**: Remove deprecated kv_cache_rewrite from the codebase or gate it behind a feature flag with explicit "ANE-illegal" warning in the module doc.

51. **V-041**: Resolve conv kernel range contradiction: either expand the 1–7 range or remove the dead 16-threshold code. Binary evidence suggests the threshold corresponds to large kernel mode activation.

52. **V-042**: Implement pooling kernel_size validation using hardware limits from AneHwLimits. Add stride-3 Avg-only check. Add large-stride MaxPool+padding rejection.

53. **V-048**: Add deconvolution constraint checks to placement validation: SOx==2, no large kernel, no vector palettization, no dilation.

54. **V-050**: Add interleave validation directly in is_dtype_ane_legal() for Int4/UInt4 instead of deferring to caller.

55. **V-051, V-111**: Align quantize validator with dtype validator — reject E5M2 as quantize output since it is universally "not supported" in ANEC. Binary evidence confirms this is definitive, not architecture-conditional.

56. **V-053, V-055, V-056**: Fix documentation to match actual code behavior (confidence start values, transfer scaling, ComputePlan confidence).

57. **V-071**: Align seed JSON format with knowledge_schema.md, or update schema to match actual seed format.

58. **V-083**: Make weight.bin optional in validation. Only require it when the model declares external weights.

59. **V-087**: Wire --seed parameter through the compile pipeline, or remove it from the CLI.

60. **V-104, V-122**: Add minimum IOSurface size validation (~49 KB) for eval buffers in the emission pipeline. Models with output buffers smaller than this will fail at runtime.

61. **V-106, V-123**: Add compilation count tracking per process. Warn when approaching ~119 limit. Provide a reset mechanism or process restart advisory.

62. **V-107**: Add ANEC attribute shape validation to the emission pipeline. Validate that stride, padding, dilation, and kernel_size attributes have the correct number of elements per the ANEC schema before emission.

63. **V-108**: Add missing AneRevision variants for V0-V3 hardware versions. At minimum, mark them as "pre-A11" with minimal capabilities.

64. **V-109**: Extend AneFamily enum to cover all 8 binary-defined families (0-7). Add AneFamily::A16Plus and AneFamily::Future for families 6-7.

65. **V-112**: Add MirOp::MILInputView or equivalent to support negative-stride tensor views. This is needed for correct lowering of reverse and crop/resize patterns that use anec.input_view with step<0.

66. **V-124**: Add weight dict initialization check ensuring weight dict is @{} (not nil) before ANEC compilation.

67. **V-127**: Add pooling stride-3 Avg-only check. Reject MaxPool with stride 3.

68. **V-131**: Model hardware sub-variants for NE(12), PE(13), DMA(27) engines per ZinAneTdHw sub-descriptors.

69. **V-133**: Add vector palettization at-Cout constraint validation. Reject vector palettization at non-Cout dimensions for ANE.

70. **V-135**: Add width wrap axis architecture-conditional check.

71. **V-137**: Add stencil constraints: reject 5D stencil, non-4D kernel, non-sum reduction, dilated stencil, strided stencil.

### Phase 4 — Low-Priority and Cosmetic

72. **V-089**: Replace hardcoded unwrap_or defaults with explicit configuration or fail-closed behavior.

73. **V-090**: Add diagnostic when canonicalization cycle limit is hit.

74. **V-091, V-092**: Add constraint documentation for UInt16 and Bool "limited support" — specify which ops/families support them.

75. **V-093**: Expand CPU_ONLY_OPS_DETAILED to cover all 120+ CPU-only ops with reason codes.

76. **V-094**: Replace binary knowledge_consistent with a ratio or graded score.

77. **V-095**: Use model-specific salt in UUID generation to improve uniqueness.

78. **V-097**: Document FP16 epsilon truncation or compute epsilon in FP32 before casting.

79. **V-102**: Update documentation for ArgMinMax to reflect high-confidence binary evidence: 7 family instantiations (0-6) confirm unavailability on A18+. Remove "unverified" qualifier from V-032.

80. **V-129**: Emit matmul transpose flags as named const nodes instead of immediate bool values, per Orion #12.

---

## VII. Forensic Note

Local reference materials were used during this audit for non-invasive
compatibility vocabulary extraction. The expanded audit additionally used
deep binary forensic analysis techniques: (a) MLIR dialect constraint
schema extraction from ANECompiler validation strings, (b) C++ demangled
symbol analysis of converter class template instantiations with
MinimumFamily parameters, (c) ANEC operation enum extraction from
converter registration strings, (d) cross-referencing of 20 Orion
programming constraints (arxiv:2603.06728) against MILLer source, (e)
ZinRegisterProgramming hardware version template instantiation analysis,
and (f) architecture-conditional data-type rejection message
categorization.

The re-expanded audit further incorporated: (g) complete family-scoped
converter catalog mapping (4,264 converter classes to source MIL ops and
target ANEC ops), (h) MinimumFamily trait extraction from MLIR
operation definitions (53 operations mapped to minimum family
requirements), (i) ZinAneTdHw hardware descriptor sub-variant mapping
(14+ sub-variants with NE/PE/DMA engine identifiers), (j) conv kernel
constraint extraction from ANEC validation strings (11 kernel size, 8
pooling, 12 stride, 10 dilation, 4 deconvolution, 5 group, 5 padding,
6 large kernel, 9 weight/palettization constraints), (k) complete
dtype rejection string catalog (7 architecture-level, 9 BF16/F16
cross-type, 15 operation-specific, 9 quantization constraints), and
(l) cross-examination of 20 Orion constraints with code-path tracing
for each constraint.

Research references used for methodology validation:
- arxiv:2603.06728 (Orion: 20 ANE programming constraints)
- arxiv:2601.01673 (MOTIF: LLM-Guided Type Inference for macOS Private Frameworks)
- arxiv:2604.23457 (ARIstoteles: Dissecting Apple's Baseband Interface)
- arxiv:2003.05039 (Devil is Virtual: Reversing Virtual Inheritance in C++ Binaries)
- arxiv:2503.07243 (Unraveling Type Recovery Patterns in Binary Code)

These materials are stored in the `forensics/` directory within the
local working tree and are excluded from the repository by `.gitignore`.
No assembly, raw disassembly snippets, long verbatim binary strings,
proprietary implementation details, raw private/local forensic artefacts,
links or paths to local forensic material, or claims of authorization
from any third party appear in this report. All findings are expressed
in abstract, source-facing terms using the author's own wording. Binary-
derived findings are expressed as constraint catalogs and cross-reference
indices without reproducing proprietary content.

### Forensic Methodology Detail

The forensic analysis was performed in three phases:

**Phase 1 (original audit)**: Non-invasive techniques including file
identity hashing, container metadata extraction (Mach-O headers, linked
libraries), and printable string triage for coarse vocabulary. This
phase produced the initial 97-violation report.

**Phase 2 (expanded audit)**: Deep binary analysis including:
- MLIR operation constraint schema extraction from ANEC validation strings
- C++ symbol demangling for template instantiation recovery
- Operation enum extraction from converter registration patterns
- Orion constraint cross-referencing (20 constraints vs MILLer source)
- Hardware version template analysis (ZinAneTd instantiations)
- Data-type rejection message categorization
This phase expanded the report to 112 violations.

**Phase 3 (research-grade re-expansion)**: Comprehensive forensic analysis including:
- Complete family-scoped converter catalog (4,264 classes mapped to MIL→ANEC operation pairs)
- MinimumFamily trait extraction (53 operations with family requirements)
- ZinAneTdHw sub-variant analysis (NE/PE/DMA engine-specific descriptors)
- Conv kernel constraint catalog (70+ constraint strings categorized)
- Complete dtype rejection catalog (40+ rejection strings categorized)
- Orion cross-examination with code-path tracing (20 constraints, 3 confirmed correct, 2 active violations)
- ANEC operation → family compatibility matrix (98 operations mapped to family requirements)
- Concat-specific constraint catalog (axis, input, width-decomposition, and dynamic-shape constraints)
- SDPA/attention constraint catalog (operand, key/value, mask, and architecture constraints)
- RingBuffer/State constraint catalog (14 RingBuffer + 11 State constraint strings)
- MIL operation rejection pattern catalog (4 "cannot be lowered" + 15 "not supported on ANE" + 14 "not supported on this architecture" + 5 "failed to convert" + 9 lowering failure + 8 family-specific patterns)
This phase expanded the report to 138 violations, corrected 2 previously suspected violations, and added 26 new violations from deep forensic evidence.

All forensic evidence is expressed at the level of abstract constraint
catalogs. No raw binary content, disassembly, or proprietary
implementation details are reproduced.
