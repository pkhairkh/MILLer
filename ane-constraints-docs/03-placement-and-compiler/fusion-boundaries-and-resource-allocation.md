# Fusion Boundaries and Resource Allocation

**Target:** Apple Neural Engine (ANE) Private Frameworks — ANECompiler Binary  
**Architecture:** Mach-O 64-bit ARM64e (Apple Silicon M2)  
**Analysis Date:** 2026-04-24  
**Binary Size:** 45,797,376 bytes | 133,164 symbols  
**Internal Codename:** "Zin" Compiler Stack  

---

## 1. Executive Summary

This report provides a deep-dive analysis of how the Apple Neural Engine Compiler (ANEC) fuses operations and what boundaries, constraints, and special circumstances determine whether operations land on the ANE or fall back to other compute units. The analysis is based on comprehensive string extraction, symbol table analysis, and mangled name demangling from the ANECompiler binary — the 44 MB shared library that implements Apple's proprietary "Zin" compiler infrastructure.

The key findings are:

1. **ANEC uses a multi-engine fusion architecture** with three distinct execution engines: NE (Neural Engine convolution/math), PE (Processing Element for element-wise/reduction), and TransposeEngine (for data rearrangement). Each engine has its own fused op categories.

2. **Fusion is pattern-based and atom-driven** — the compiler uses a `ZinFusionPatterns` system organized by `FusionPatternType`, where individual "atoms" (conv atom, GOC atom, activation atom, etc.) are matched against operation graphs. There are separate pattern sets for NE, PE, SNE, and TransposeEngine.

3. **Fusion occurs in multiple pipeline stages** — from initial layer fusion (`ZinMirLayerFusion`) through NE transpose fusion, PE transpose fusion, transpose-engine fusion, and hoisting optimizations that enable further fusion.

4. **The primary fusion boundary mechanism is memory pressure** — when L2 cache or register pressure exceeds hardware limits, the compiler cuts clusters, splits subgraphs, and spatially tiles operations to fit within hardware constraints.

5. **Many operations cannot land on ANE** due to hard constraints on quantization formats, palettization, dilation, kernel sizes, data types, and dimension limitations. These are documented exhaustively in the binary's error strings.

---

## 2. The Three ANE Execution Engines

The ANEC compiler targets three distinct hardware engines within the ANE, each with its own fusion categories and constraints:

### 2.1 NE Engine (Neural Engine — Convolution/Math Pipeline)

The NE engine handles the "heavy compute" operations: convolutions, pooling, matrix multiplication, and their associated transforms. It has the following fused operation categories:

| NE Fused Op Type | Description |
|---|---|
| `NEFUSED_CONV` | Convolution with fused pre/post operations |
| `NEFUSED_POOL` | Pooling with fused activations, scales |
| `NEFUSED_EW` | NE-element-wise (e.g., add after conv) |
| `NEFUSED_DUAL_SOURCE_EW` | Dual-input element-wise on NE |
| `NEFUSED_MATMUL` | Matrix multiplication with fused ops |
| `NEFUSED_CROSS_CORRELATION` | Cross correlation (template matching) |
| `NEFUSED_KERNEL_RASTERIZER` | Kernel rasterization operations |
| `NEFUSED_RCAS` | RCAS (region-based) operations |
| `NEFUSED_BYPASS` | NE bypass (data passthrough with optional transforms) |

### 2.2 PE Engine (Processing Element — Element-Wise/Reduction Pipeline)

The PE engine handles element-wise operations, reductions, scaled element-wise, and GOC (Generic Operation Compute) layers:

| PE Fused Op Type | Description |
|---|---|
| `PEFUSED_ELEMENTWISE` | Element-wise operations with fused pre/post transforms |
| `PEFUSED_GOC` | Generic Operation Compute (flexible compute kernel) |
| `PEFUSED_POOL` | PE-side pooling operations |
| `PEFUSED_SECUREFLUSH` | Secure data flush operations |

### 2.3 TransposeEngine

The TransposeEngine handles data rearrangement operations (transposes, reshapes, channel reordering) and can be fused with adjacent NE/PE operations:

| Component | Description |
|---|---|
| `ZinTransposeEngineLayer` | Standalone transpose engine layer |
| `ZinMirTransposeEngineFusion` | Fusion pass for transpose engine ops |
| `ConvertNEBypassToTransposeEngine` | Converts NE bypass layers to transpose engine when possible |

### 2.4 Engine Layer Constraints

The binary reveals strict rules about which engine layers can interact:

```
Only NE and PE engine layers are supported
Only NE and PE engine layers are supported for Broadcast
Only NE and PE engine layers are supported for DMA perf
Only NE and PE engine layers are supported for Splitting
Only NE and TransposeEngineLayer allowed
Only support PE layer for the prev_engine_layer
Only support engine layer as consumers of L2 cache tensors
Consumer of L2 cached tensor must be engine layer.
```

This means the compiler strictly partitions the world into NE, PE, and TransposeEngine layers. Any operation that cannot be mapped to one of these three engines cannot execute on the ANE.

---

## 3. Fusion Pattern Architecture

### 3.1 The ZinFusionPatterns System

The core fusion mechanism is implemented through the `ZinFusionPatterns` class, which initializes pattern sets for each engine type:

```
ZinFusionPatterns::InitializeNEPatterns()      — 13 cold-path error branches
ZinFusionPatterns::InitializePEPatterns()      — 17 cold-path error branches
ZinFusionPatterns::InitializeSNEPatterns()     — 3 cold-path error branches
ZinFusionPatterns::InitializeTransposeEnginePatterns()
```

The high number of cold-path (error) branches in NE and PE pattern initialization indicates extensive validation logic — many combinations of operations are checked for fusability, and many are rejected.

### 3.2 Fusion Atoms (Building Blocks)

Fusion atoms are the indivisible units that get matched and grouped into fused engine layers. The binary reveals a rich taxonomy organized by engine:

#### NE Atoms (ZinNEAtoms)

| Atom | Purpose |
|---|---|
| `ConvAtom` | Base convolution |
| `NEConvAtom` | NE-specific convolution |
| `ConvGOCAtom` | Convolution + GOC fusion |
| `ConvQuantAtom` | Conv + quantization fusion |
| `MatMulAtom` | Matrix multiplication |
| `MatmulGOCAtom` | MatMul + GOC fusion |
| `MatmulQuantAtom` | MatMul + quantization |
| `PoolAtom` | Pooling base |
| `PoolGOCAtom` | Pool + GOC fusion |
| `PoolQuantAtom` | Pool + quantization |
| `RcasGOCAtom` | RCAS + GOC |
| `ElementWiseAtom` | NE element-wise |
| `EWGOCAtom` | Element-wise + GOC |
| `EWQuantAtom` | Element-wise + quantization |
| `EWAbsAtom` | Element-wise absolute value |
| `NEGOCAtom` | NE-side GOC standalone |
| `BypassGOCAtom` | Bypass + GOC |
| `BypassQuantAtom` | Bypass + quantization |
| `DeQuantAtom` | Dequantization |
| `DeQuantPreScaleAtom` | Dequant + pre-scale |
| `PreScaleAtom` | Pre-scale operation |
| `TextureAtom` | Texture operation |
| `ActivationAtom` | Activation function |
| `InputReLUAtom` | Input-side ReLU |
| `TransposeAtom` | Transpose operation |
| `CopyAtom` | Data copy |
| `RoundAtom` | Rounding operation |
| `KernelRasterizerAtom` | Kernel rasterization |
| `CrossCorrelationAtom` | Cross-correlation |

#### PE Atoms (ZinPEAtoms)

| Atom | Purpose |
|---|---|
| `PEEWGOCAtom` | PE element-wise + GOC |
| `ScaledEWAtom` | Scaled element-wise |
| `ScaledEWAddAtom` | Scaled element-wise add |
| `CommutativeScaledEWAtom` | Commutative scaled EW |
| `UnaryScaledEWAtom` | Unary scaled element-wise |
| `InputDeQuantAtom` | Input dequantization |
| `InputReLUAtom` | Input ReLU |
| `InputTransposeAtom` | Input-side transpose |
| `OutputTransposeAtom` | Output-side transpose |
| `OutputGOCAtom` | Output GOC |
| `OutputReLUAtom` | Output ReLU |
| `PostScaleAtom` | Post-scale operation |
| `PoolPreScaleAtom` | Pool pre-scale |
| `PreScaleSrc1Atom` | Pre-scale for source 1 |
| `PreScaleSrc2Atom` | Pre-scale for source 2 |
| `PerChannelGOCAtom` | Per-channel GOC |
| `PerChannelQuantAtom` | Per-channel quantization |
| `OutputScalarQuantAtom` | Output scalar quantization |
| `ReductionAtom` | Reduction operation |
| `ReductionEpsilonAtom` | Reduction with epsilon |
| `ReductionFinalScaleAtom` | Reduction final scale |
| `AbsOrZeroCompareAtom` | Abs or zero-compare |
| `DynamicGOCAtom` | Dynamic GOC |
| `BinaryPoolPreScaleAtom` | Binary pool pre-scale |
| `DeQuantAtom` | PE dequantization |
| `TextureAtom` | PE texture operation |

### 3.3 Atom Fusability Checks

Each atom has an `IsFusable` method that determines whether it can be fused with adjacent operations. Key fusability checks discovered in the symbol table:

| Check Function | Purpose |
|---|---|
| `GOCAtom::IsFusable()` | Checks if a GOC layer can be fused (takes GOC layer, OpLayer, TextureLayer, TensorFormat, OpCodeType, HalParameters) |
| `DeQuantAtom::IsFusableAsGOC()` | Checks if dequantization can be fused as a GOC (takes vector of DeQuantLayers, TensorFormat, OpLayerGraph, HalParameters) |
| `DeQuantAtom::IsFusableWithInputRelu()` | Checks if dequant can be fused with input ReLU |
| `PreScaleAtom::IsFusable()` | Checks if pre-scale can be fused with a GOC layer |
| `ConvAtom::IsFusableToDequant()` | Checks if conv can fuse into dequantization |
| `ScaledEWAtom::IsFusableEW()` | Checks if scaled EW is fusable as element-wise |
| `PostScaleAtom::IsFusable()` | Checks post-scale fusability with GOC |
| `PoolPreScaleAtom::IsFusablePerChannelPreScale()` | Per-channel pre-scale fusability for pooling |
| `PerChannelGOCAtom::IsFusable()` | Per-channel GOC fusability |
| `InputDeQuantAtom::IsDeQuantFusable()` | Input dequant fusability |
| `IsQuantFusable()` | Checks if quantization operations can be fused |
| `IsFusableBasedOnFormat()` | Tensor format-based fusability check |
| `IsFusableBasedOnFormatOCGSizeAndActiveNE()` | Format + OCG + ActiveNE combined fusability |
| `IsFusableConcat()` | Concat fusability check |

### 3.4 The Fusion Process

The fusion process follows a clear sequence:

```
1. ZinMirLayerFusion::Run()
   ├── ZinMirLayerFusion::Group()   — Groups layers into fusion candidates
   └── ZinMirLayerFusion::Commit()  — Commits fused groups to graph

2. Transpose Fusion Passes
   ├── ZinMirNETransposeFusion       — Fuses transposes into NE layers
   ├── ZinMirPETransposeFusion       — Fuses CW transposes into PE layers
   │   ├── FuseCWTransposeToPEAsInput()
   │   └── FuseCWTransposeToPEAsOutput()
   ├── ZinMirTransposeEngineFusion   — Fuses into transpose engine layers
   └── ZinMirPETransposeFusionWithSinglePatchReduction

3. Hoisting Passes (Enable Further Fusion)
   ├── ZinMirHoistGOCsForEWFusion          — Hoist GOCs for EW fusion
   ├── ZinMirHoistGOCEWAbsForPEEWFusion    — Hoist GOC/EW-Abs for PE-EW fusion
   ├── ZinMirHoistInputTypecastForFusion   — Hoist input typecasts for fusion
   ├── ZinMirHoistOutputTypecastForFusion  — Hoist output typecasts for fusion
   ├── ZinMirHoistGOCorActivationForConvFusion     — Hoist for conv fusion
   ├── ZinMirHoistSEWActivationForGOCConvFusion    — Hoist SEW activation for GOC+Conv
   ├── ZinMirHoistGOCorActivationForMatMultFusion  — Hoist for matmul fusion
   ├── ZinMirHoistInputNEByPassToEnableEngineLayerDMA   — Hoist NE bypass for DMA
   └── ZinMirHoistOutputNEByPassToEnableEngineLayerDMA  — Hoist NE bypass for DMA

4. Post-Fusion
   ├── PostFusionTransposeHoisting
   └── Pre-Fusion Reverse CSE (Common Subexpression Elimination)
```

---

## 4. Specific Fusion Patterns and Boundaries

### 4.1 Convolution Fusion (NEFUSED_CONV)

The NE convolution fusion is the most complex fusion pattern. A fused convolution can absorb:

- **Input-side:** Dequantization, Input ReLU, PreScale, Transpose (CW), Texture operations
- **Output-side:** Activation (ReLU variants, sigmoid, tanh, etc.), GOC, PostScale, Quantization, Bias+Scale fusion

**Key fusion API (from symbol demangling):**

```
ZinIrKernel::FuseScaleBiasWithBottom()      — Fuses scale+bias into conv kernel
ZinIrKernel::FuseScaleWithBottom()          — Fuses scale only
ZinIrKernel::FuseBiasWithBottom()           — Fuses bias only
ZinActivationLayer::FuseIntoPostScale()     — Fuses activation into post-scale
ZinPadLayerUtils::FuseConvWithConsumer()    — Fuses padding into conv consumer
ZinPadLayerUtils::FusePadWithConsumer()     — Fuses pad layer with consumer
```

**Conv fusion boundaries:**

- `grouped conv with large kernel size is not supported` — Group convolutions with large kernels cannot be fused on the NE
- `dilated conv with large kernel size is not supported` — Dilated convs with large kernels are rejected
- `grouped conv` requires `num_groups == out_dims.c` for depthwise, or `num_groups == 1` for normal conv
- Channel dimensions must be divisible by num_groups
- Conv with `input stride > 2` cannot support dynamic shapes
- `large stride conv cannot support dynamic shape`

### 4.2 Pad+Conv Fusion

The `PadAndConv::DecomposeAndFuse()` function handles the fusion of padding operations with convolutions. However, certain padding modes break fusion:

- **Negative padding mode** is not supported for decomp/fusion
- **Replication padding mode** is not supported
- **Symmetric padding mode** is not supported
- **Background padding with non-zero value** requires constant padding mode
- **Multiple padding modes at different axes** cannot be fused
- **Fused pad violates Kernel dims** — if the resulting fused dimensions violate kernel constraints, fusion is rejected

### 4.3 Element-Wise Fusion

Element-wise operations are split between NE and PE engines:

**NE Element-Wise (NEFUSED_EW / NEFUSED_DUAL_SOURCE_EW):**
- Binary EW operations (add, sub, mul, div, max, min)
- `NEElementWise can only have input activation mode as Relu`
- `NEElementWise must contain ew_`
- Input channel must be divisible by programmed output channel
- Dual-source EW requires matching dimensions and interleave

**PE Element-Wise (PEFUSED_ELEMENTWISE):**
- More flexible than NE-EW; supports scaled EW, GOC, and reduction fusion
- ScaledEW (SEW) operations: `ZinScaledElementWiseLayer` with `ZinIrScaledEWInfo`
- PE-EW supports: input dequant, input ReLU, input/output transpose, output GOC, output ReLU, post-scale, per-channel GOC, per-channel quant, reduction
- `PEEW with uint16 input must have only one dma src` — uint16 limits DMA sources

**PE-EW Fusion Hoisting:**
The `ZinMirHoistGOCsForEWFusion` and related passes restructure the graph to enable PE-EW fusion by moving GOC and activation operations to positions where they can be absorbed into the PE-EW engine layer.

**Failed PE-EW fusion conditions:**
- `Error: Unable to fuse GOC, GOC and EW_MAX` — Cannot fuse two GOCs and EW max together
- `Error: Unable to fuse Transpose, ScaledEW, Transpose to GOC` — This specific 3-op pattern cannot be fused
- `Error: failed fusing scales to sew` — Scales cannot be fused into scaled-EW under certain conditions

### 4.4 Activation Fusion

Activations can be fused into both NE and PE layers, but specific conditions apply:

**Successful fusion patterns:**
- ReLU fusing into conv output (via `SetPEOutputReLU`)
- Activation fusing into post-scale (`ZinActivationLayer::FuseIntoPostScale`)
- Hard swish detection via `SwishHardActivationDetection` pattern matching
- Quantization scale fused into activation post-scale (`FuseQuantScaleIntoActivationPostScale`)

**Failed activation fusion conditions:**
- `Error: Unable to fuse clamped relu`
- `Error: Unable to fuse leaky relu`
- `Error: Unable to fuse leaky relu or abs`
- `Error: Unable to fuse swish activation`
- `Error: Unable to fuse swish hard activation`
- `Error: Unable to merge activations`
- `Error: Unable to merge sumsqr`
- `Error: Unable to merge with cast`
- `Log2: post scale can't be fused into LUT` — Log2 activation cannot fuse post-scale into a LUT-based implementation

### 4.5 Quantization/Dequantization Fusion

Quantization fusion is one of the most constrained areas of the compiler:

**Fusion patterns:**
- `ConvQuantAtom` — Conv + quantization
- `PoolQuantAtom` — Pool + quantization
- `EWQuantAtom` — Element-wise + quantization
- `MatmulQuantAtom` — MatMul + quantization
- `DeQuantAtom` — Dequantization as GOC or into conv
- `InputDeQuantAtom` — PE input dequantization
- `FuseQuantScaleIntoActivationPostScale` — Quant scale into activation

**Critical boundary: `ANE doesn't support blockwise scale`**

This is one of the most significant constraints discovered. The ANE hardware does not support blockwise scale quantization. Only per-tensor and per-output-channel (per-cout) quantization are supported.

**Additional quantization boundaries:**
- `Int4 Per-Cout Dequant is not supported`
- `Asym quantization is not supported` — Asymmetric quantization cannot land on ANE
- `mutable kernels are not supported with kernel quantization`
- `Per-channel palettized` kernels have severe limitations (no aliasing, no channel shuffling, no new palette creation)
- `Dequant layer must have fp16 output format`
- `Dequant layer must have int8, uint8, int4 or e4m3 input format`
- `Invalid dequant scale count` — Scale arrays must match expected dimensions

### 4.6 Palettization (LUT) Fusion

Palettized (LUT-based) weight compression has its own fusion rules and constraints:

**Supported palettization ranks:** 0 through 6 (verified by MLIR constraint strings)  
**Valid palette vector sizes:** Powers of 2 from 1 up to architecture maximum  
**Multi-palette mode:** Supported for non-aligned LUT configurations

**Palettization fusion boundaries:**
- `1 and 2 bit palettes not supported`
- `CW conv does not support vector palette`
- `deconv with vector palettization is not supported`
- `dilation is not supported for vector palettized kernels yet`
- `does not support palettized weight with large kernel stride`
- `vector palettized weights are not supported` (on some architectures)
- `per channel cout with vector palettization is not supported yet`
- `Consecutive palette_vector_size channels must make use of the same palette LUT`
- `Invalid OCG size for vector palettization`
- `Convolution layer has been lowered to have both palettization and compression` — Having both is an error
- Palette LUT size must not exceed `ne_palette_lut_size_in_bytes` hardware limit

### 4.7 Transpose Fusion

Three separate transpose fusion passes handle different scenarios:

**ZinMirNETransposeFusion:**
Fuses transposes into NE engine layers. Uses a `ConcatHandler` for transposes adjacent to concat operations:
- `ConcatHandler::NEIncomingHandler()` — Handles transposes incoming to NE
- `ConcatHandler::BuildNewConcat()` — Builds new concat with fused transpose
- `ConcatHandler::UpdateFusedNELayerMirInfo()` — Updates MIR info after fusion
- `IsFusableBasedOnFormatOCGSizeAndActiveNE()` — Checks if transpose can be fused based on tensor format, OCG size, and ActiveNE count

**ZinMirPETransposeFusion:**
Fuses channel-width (CW) transposes into PE layers:
- `FuseCWTransposeToPEAsInput()` — Fuse CW transpose as PE input
- `FuseCWTransposeToPEAsOutput()` — Fuse CW transpose as PE output
- `ConvertToNEBypass()` — Convert unfusable transposes to NE bypass layers
- Also has a `SinglePatchReduction` variant for single-patch scenarios

**ZinMirTransposeEngineFusion:**
- `FuseCWTransposeToEngineLayer()` — Fuses CW transposes into engine layers
- TransposeEngine-based graphs should not need DMA buffers or chain buffers

**Aggressive NE transpose fusion** can be enabled via `EnableAggressiveNETransposeFusion` compiler parameter.

---

## 5. ActiveNE: The Core Resource Allocation Mechanism

### 5.1 What is ActiveNE?

ActiveNE is the number of Neural Engine units (NEs) activated for a given operation. The M2 ANE has multiple NE units, and the compiler must determine how many to activate for each layer based on the operation's channel requirements and the hardware's OCG (Output Channel Group) configuration.

**Key constraints discovered:**

```
(1 << (activene + wu_stack + fat_tile_enable)) <= hal_params.num_nes
(1 << (hw.ne_config.ane_ne_config.kernel_cfg2.palette_group_size) * ... >= (1 << ... kernel_cfg.palette_block_size)
(num_subgroup % (1 << activene)) == 0
(active_ne_times_ocg % logical_intlv) == 0
cout <= (1 << (activene + ocgsize))
cout <= (unicast_cout * (1 << activene))
active_ne_ocg_size % 16 == 0 || relax_active_ne_ocg_constraint
active_ne_ocg_size % 8 == 0 || relax_active_ne_ocg_constraint
ne_info_.active_ne cannot be zero
```

### 5.2 ActiveNE Constraints

- **Active NE count must be a power of 2** (implied by the bit-shift arithmetic)
- **Active NE * OCG size must be divisible by interleave factor** (`active_ne_times_ocg % logical_intlv == 0`)
- **Active NE count must be divisible by num_groups for group-conv** with 1:1 NE-to-LUT ratio
- **Output channels must fit:** `cout <= (1 << (activene + ocgsize))`
- **Total activated NEs cannot exceed hardware NE count:** `(1 << (activene + wu_stack + fat_tile_enable)) <= num_nes`
- **Error: Active NE greater than number of HW NEs** — Hard limit
- **Active NE can be relaxed:** `relax_active_ne_ocg_constraint` flag allows OCG size % 8 instead of % 16

### 5.3 SetActiveNE Process

The `ZinMirSetActiveNE` pass determines the active NE count for each layer in the control flow graph. If it fails, the error message states: `A newly created layer in the latency legalizer could not set active NE`.

For standalone activations (not fused with conv/pool), only the first NE generates data: `For standalone Activations, we only generate the first NE data.`

### 5.4 NEBypass: When Data Doesn't Need Compute

When data needs to pass through the NE without computation (e.g., for format conversion, transpose, or routing), a `ZinNEBypassLayer` is used. NEBypass layers:

- Can optionally contain: Texture, Broadcast, Activation, GOC, Transpose, Quant, and Copy sub-operations
- `NEBYPASS-GOC should not have bias`
- `NEBypass does not support Espresso weight format`
- Can be converted to TransposeEngine when no compute is needed: `ConvertNEBypassToTransposeEngine`
- `NEBypass cannot have tensor kernel`
- Are used to hoist operations to enable DMA: `ZinMirHoistInputNEByPassToEnableEngineLayerDMA`

---

## 6. Memory Pressure and Subgraph Boundaries

### 6.1 The Pressure-Based Partitioning System

The primary mechanism for determining fusion boundaries is memory pressure analysis. The compiler models the L2 cache and register usage of each operation and cuts subgraphs when pressure exceeds hardware capacity.

**Subgraph identification algorithms:**

| Algorithm | Source File |
|---|---|
| `ZinMirSubgraphIdentification` | ZinMirSubgraphIdentification.cpp |
| `ZinMirPressureBasedSubgraphIdentification` | ZinMirPressureBasedSubgraphIdentification.cpp |
| `ZinMirSpatialSplitPressureBasedSubgraphIdentification` | ZinMirSpatialSplitPressureBasedSubgraphIdentification.cpp |
| `BatchOrChannelSplitPressureBasedSubgraphIdentification` | — |
| `BondedSplitSubgraphIdentification` | — |
| `LegalizerSubgraphIdentification` | — |
| `CostBasedSubgraphIdentification` | — |
| `ManualSubgraphIdentification` | — |

### 6.2 Cluster Formation and Cutting

The compiler groups operations into "clusters" and then cuts them when pressure is too high:

```
"Adding %s : %zu to connected cluster"
"Constructing Subgraph for Cluster [%zu,%zu]"
"Cutting Cluster at Partial Input %s : %zu"
"Cutting Cluster at Partial Output %s : %zu"
"Cutting Cluster [%zu,%zu] at Concat With Partial Input"
"Cutting Cluster [%zu,%zu] at Partial Outputs"
"Legalizing Cluster [%zu,%zu] due to high pressure"
"Legalizing Cluster Due to no legal split dimension found"
"Extracting cluster at High Pressure Region [%zu,%zu]"
```

**Pressure model includes:**
- `chain_cost_read0`, `chain_cost_read1`, `chain_cost_write` — Cost of chaining data between TDs
- `chain_time_read0`, `chain_time_read1`, `chain_time_write` — Time for chained transfers
- `nonresident_cost` — Cost when data is not in L2 cache
- `resident_cost` — Cost when data is in L2 cache
- `compute_time_resident`, `compute_time_nonresident` — Compute time based on data residency
- `Pressure attributors` — Individual contributors to memory pressure
- `Total pressure` — Sum of all pressure contributors

### 6.3 Spatial Splitting

When a subgraph is too large for the ANE's L2 cache, the compiler spatially splits operations (divides the spatial dimensions into tiles):

**Spatial split modes:**

| Mode | Description |
|---|---|
| `disabled` | No spatial splitting |
| `test` | Test mode |
| `memory` | Memory-pressure driven |
| `auto` | Automatic selection |
| `manual` | Manual specification |
| `generic-dag` | Generic DAG mode |
| `generic-dag-exp` | Experimental DAG mode |
| `generic-dag-memory` | DAG mode with memory constraints |

**Spatial split constraints:**
- `Can't tile subgraph b/c SIP > budget` — If "Spatial Inward Pressure" exceeds the tile budget, tiling fails
- `OptimizeTileCountByInsertingResetLayers` — Reset layers can be inserted to optimize tiling
- `CutSubgraphAtResetLayers` — Subgraphs can be cut at reset layer boundaries
- `GlobalRefinementInSpatialSplit` — Global refinement pass for spatial split optimization
- `Invalid boundary constraint in Spatial Splitting`
- `Invalid deconv in spatial split`

### 6.4 L2 Legalization

The `ZinMirL2Legalizer` pass ensures that all operations fit within the L2 cache budget:

- `L2-Legalizer: splitting layer: %s` — When a layer must be split
- `L2-Legalizer: failed splitting layer: %s` — When splitting fails
- `Active NE in DRAM Legalizer failed!`
- `ChannelLast in DRAM currently not supported`
- L2 cache can be used for: chained buffers, resident data, circular buffers, inplace operations
- `Allocation decision must be chain` or `must be L2-dep`

### 6.5 L2 Cache and Chaining vs L2-Dependency

Two mechanisms reduce DRAM traffic between consecutive TDs (tile-data programs):

**Chaining:** When consecutive TDs share data, the compiler can "chain" them so the output of one TD is directly consumed by the next without writing back to DRAM. The `chain_buf` and `chain_symbol` represent chained data paths.

**L2-Dependency:** An alternative to chaining where data is kept in L2 cache between TDs. L2-dep is preferred when chaining would create aliasing issues.

**Key constraint:** `Either chain or l2-dep cost should be set, not both.` — A given data path must use one mechanism or the other, never both.

**Chaining boundaries:**
- `Chaining is incorrectly enabled for TD pair with L2 alias` — Cannot chain when L2 aliasing exists
- `NEConv has per-cout-palette-lut, cannot split the kernel of this conv under specified tile sizes in chaining canonicalization`
- `NEConv kind is not normal, cannot split the kernel of this conv under specified tile sizes in chaining canonicalization`
- `Error: We found illegal chaining`
- `An L2 dep pair must be assigned to the same ANE`
- `It must have incoming index for L2-dep pair`

---

## 7. Complete Inventory of Operations That Cannot Land on ANE

Based on exhaustive analysis of error strings and constraint messages in the binary, the following operations or conditions prevent an op from landing on the ANE:

### 7.1 Hard "Cannot Land on ANE" Conditions

| Category | Constraint | Source String |
|---|---|---|
| **Data Types** | E4M3 float not supported on this architecture | `Error: E4M3 is not supported` |
| | E5M2 float not supported | `E4M3 or E5M2 format not supported` |
| | 32-bit format not supported | `32 bit format not supported` |
| | 2xInt8 mode not supported | `2xInt8 mode is not supported` |
| | E4M3Overflow not supported | `E4M3Overflow is not supported` |
| **Quantization** | Blockwise scale quantization | `ANE doesn't support blockwise scale` |
| | Asymmetric quantization | `Asym quantization is not supported` |
| | Int4 per-cout dequant | `Int4 Per-Cout Dequant is not supported` |
| | Mutable kernels with quantization | `mutable kernels are not supported with kernel quantization` |
| **Convolution** | Grouped conv with large kernel | `grouped conv with large kernel size is not supported` |
| | Dilated conv with large kernel | `dilated conv with large kernel size is not supported` |
| | Dilated deconvolution | `dilated deconvolution is not supported!` |
| | Deconv with stride > 1 along depth | `deconv with stride > 1 is not supported along depth axis` |
| | Deconv with SOx != 2 | `deconv with SOx != 2 is not supported` |
| | Deconv with large kernel | `deconv with large kernel size is not supported` |
| | Deconv with vector palettization | `deconv with vector palettization is not supported` |
| | Depth > 1 for MatMult inputs | `depth > 1 is not supported for MatMult inputs` |
| | Kernel depth > 1 for large kernel | `kernel with depth = %zd > 1 is not supported for large kernel` |
| | Conv with large stride + dynamic shape | `Large stride conv cannot support dynamic shape` |
| **Pooling** | Dilated pooling | `Dilated Pooling not supported on ANE` |
| | Avg pool with exclude padding >= kernel | `Cannot support avg pool with exclude padding size equal to or larger than kernel size` |
| **Spatial Operations** | Channel padding | `Channel padding is not supported on ANE` |
| | Cropping on batch/depth/channel | `Cropping on batch / depth / channel dimension not supported` |
| | Batch-to-space on z axis | `batch-to-space on z axis is not supported` |
| | ChannelToSpace in z dimension | `ChannelToSpace in z dimension is not supported` |
| | PixelUnshuffle on z axis | `PixelUnshuffle on z axis is not supported` |
| **Padding** | Replication padding mode | `Replication padding mode is not supported` |
| | Symmetric padding mode | `Symmetric padding mode is not supported` |
| | Negative padding mode | `Negative padding mode is not supported` |
| | Padding on batch dimension | `padding on the batch dimension is not supported yet` |
| | Padding on channel dimension | `padding on the channel dimension is not supported yet` |
| **Palettization** | 1 and 2 bit palettes | `1 and 2 bit palettes not supported` |
| | Vector palettized weights | `vector palettized weights are not supported` (arch-dependent) |
| | Palettized weights with large kernel stride | `does not support palettized weight with large kernel stride` |
| | Dilation with vector palettized | `dilation is not supported for vector palettized kernels yet` |
| | Per-channel cout vector palettized | `per channel cout with vector palettization is not supported yet` |
| **Miscellaneous** | Dilated stencil | `Dilated Stencil not supported on ANE` |
| | 1D Winograd | `1D Winograd is not supported` |
| | Dropout | `Dropout Layer is not supported on this architecture` |
| | Flatten on some architectures | `Flatten is not supported on this architecture` |
| | L2 norm with batch axis | `L2 norm does not support batch axis` |
| | Min/max norm with batch/channel axis | `Min/max norm layer does not support batch/channel axis` |
| | Dynamic shape random ops | `ANE cannot support dynamic shape random op` |
| | NMS for iOS 15/16 | `ANE cannot support NMS for ios15 and ios16` |
| | Linear with input rank >= 5 | `ANE cannot support Linear with input rank >= 5` |
| | Striding in channel dimension | `ANEC does not support striding in channel dimension` |
| | Affine transform | `affine transform is not supported on this architecture` |
| | Circular buffer (arch-dependent) | `Circular buffer is not supported on this architecture` |
| | Condition layer | `Condition layer is not supported` |
| | DynamicGOC | `DynamicGOC is not supported` |
| | Software Conditional layers | `Software Conditional layers are currently unsupported` |
| | Multi layer procedure families | `Multi layer procedure families are not supported` |
| | ElementWise Mult | `ElementWise Mult is not supported` |
| | ElementWise Sqr | `ElementWise Sqr is not supported` |
| | ANE Layer mix of channellast and non-channellast | `ANE Layer cannot mix channellast and non-channellast input/output tensors` |

### 7.2 Tensor Dimension Constraints

The binary reveals hardware limits on tensor dimensions that act as fusion/ANE-placement boundaries:

```
(ox * oy * cout * num_groups) <= hal_params.max_tensor_channels
(sx * sy * cin * num_groups) <= hal_params.max_tensor_channels
(ox * oy * oz * cout) <= hal_params.max_tensor_channels
(sx * sy * sz * cin) <= hal_params.max_tensor_channels
(hout * oy) <= hal_params.max_tensor_height
(wout * ox) <= hal_params.max_tensor_width
```

Operations whose tensor dimensions exceed `max_tensor_channels`, `max_tensor_height`, or `max_tensor_width` cannot be placed on the ANE.

### 7.3 Interleave Constraints

Interleave is a critical parameter for ANE data layout. Valid interleave factors are: `{1, 2, 3, 4, 8}`. Constraints:

- `Const tensor interleave must be 1`
- `ANEC only supports interleave on C axis`
- `ChannelLast does not support non-one channel interleave`
- `(1 << effective_ocgsize) >= interleave_factor * dma_interleave` — OCG size must be large enough
- `no valid interleave factor found but transpose/reshape is not supported on the architecture` — Fatal when no valid interleave exists

---

## 8. Compiler Flags Controlling Fusion and Boundaries

The `ZinIrCompilerParameters` class exposes numerous flags that control fusion behavior. These can be set via the compiler options dictionary:

### 8.1 Fusion Control Flags

| Flag | Effect |
|---|---|
| `DisableMergeActivation` | Disables activation merging into engine layers |
| `DisableMergeConstants` | Disables constant merging |
| `DisableMergeScaleBias` | Disables scale+bias merging into kernels |
| `DisableBondedNetworks` | Disables multi-ANE bonded network support |
| `AggressiveScaleFusion` | Enables aggressive scale fusion |
| `EnableAggressiveNETransposeFusion` | Enables aggressive NE transpose fusion |
| `DumpFusionBoundaryInfo` | Dumps fusion boundary information to JSON |
| `EnableMILConstantCoalescing` | Enables constant coalescing at MIL level |
| `DisableAdjustInterleaveFactor` | Disables interleave factor adjustment |
| `EnableSingleChannelElementwiseOpCopyRemoval` | Removes unnecessary single-channel EW copies |

### 8.2 Memory and Layout Control Flags

| Flag | Effect |
|---|---|
| `EnableDramInplaceAllocation` | Enables in-place DRAM allocation |
| `EnableL2CachedBuffer` | Enables L2 cache buffer usage |
| `L2CacheMode` | Sets L2 cache mode |
| `L2DisableResident` / `L2EnableResident` | Controls L2 residency |
| `EnableCircularBufferInSpatialSplit` | Enables circular buffers during spatial splitting |
| `EnableSpatialSplitInX` | Enables spatial splitting along X dimension |
| `EnableWorkStealingForBondedNetworks` | Work stealing for multi-ANE |
| `DisablePerDmaRdtidForBondedNetworks` | Disables per-DMA read-TID for bonded networks |
| `EnableKernelSplitForMultiPaletteLUT` | Kernel splitting for multi-palette LUTs |
| `CostModelClusterThreshold` | Threshold for cost model clustering |
| `GlobalRefinementInSpatialSplit` | Enables global refinement in spatial split |

### 8.3 Spatial Split Mode

The `spatial_split_transform` flag accepts these values:
- `disabled`, `test`, `memory`, `auto`, `manual`, `generic-dag`, `generic-dag-exp`, `generic-dag-memory`

---

## 9. The Complete Fusion Pipeline (Reconstructed)

Based on all evidence, the complete fusion pipeline in ANEC works as follows:

```
MIL Program
    │
    ▼
MLIR ANEC Dialect Frontend
    │  (Converts MIL ops to ANEC IR ops with FusionType attribute)
    │  (Validates constraints: FusionType, groups, kernel formats)
    ▼
Zin IR Builder (ZinIrOpLayer directed graph)
    │
    ├─── ZinIrOpt Passes (Pre-fusion)
    │    ├── CollapseReshape
    │    ├── CollapseTranspose
    │    ├── WidthConcatCanonicalizer
    │    ├── RemoveRedundantShapeChange
    │    ├── MergeFusableActivationPairs
    │    ├── ScaledEWOrEWWithConstInToGOCFusion
    │    ├── SwishHardActivationDetection
    │    └── Pre-Fusion Reverse CSE
    │
    ├─── MIR Builder (Layer Construction)
    │    ├── Active NE evaluation
    │    ├── NE transpose fusion (or NEBypass)
    │    ├── PE transpose fusion
    │    ├── Transpose engine fusion
    │    └── Layer fusion
    │
    ├─── MIR Prepare (Legalization)
    │    ├── DRAM Legalization (memory fit)
    │    ├── Latency Legalization (timing fit)
    │    ├── L2 Legalization (cache fit)
    │    ├── Multi-segment Legalization
    │    ├── Tensor Dimension Legalization
    │    └── Tensor-based Context Switch Legalization
    │
    ├─── MirOpt Passes (Optimization)
    │    ├── ZinMirActiveNE
    │    ├── ZinMirLayerFusion (Group + Commit)
    │    ├── ZinMirPadOptimization (DecomposePadAndFindFusionCandidates)
    │    ├── ZinMirEwCopyOptimizer
    │    ├── ZinMirOptMergeDeconvConv
    │    ├── ZinMirOptFullyConnectedLayer
    │    ├── PadAndConv / PadAndPool DecomposeAndFuse
    │    ├── ZinMirBatchOrChannelSplitter
    │    ├── ZinMirSubgraphIdentification / PressureBased
    │    ├── ZinMirSpatialSplitter
    │    ├── ZinMirL2Legalizer
    │    ├── ZinMirMultiSegmentLegalizer
    │    ├── Hoisting passes (6 different hoisters for enabling fusion)
    │    ├── MergeConvolutions / MergeFanoutConvolutions
    │    └── PostFusionTransposeHoisting
    │
    ├─── Scheduling + Register Allocation
    │    ├── ZinIrOpLayerGraphScheduler
    │    ├── ZinCpBasedAllocator (copy-based allocation)
    │    ├── ZinIrLocalRegAlloc (with spill support)
    │    ├── ZinL2FootprintCalculator
    │    ├── Chaining vs L2-dep decision
    │    └── Active NE optimization (OptimizePoolActiveNEs)
    │
    └─── Code Generation
         ├── ZinIrCodegen (v1 through v26, plus vu1)
         │    ├── PE Codegen
         │    ├── NE Config + Kernel programming
         │    ├── L2 register programming
         │    ├── Chain/L2-dep buffer allocation
         │    └── Remote dependency handling
         ├── ZinLinker (program linking)
         └── ZinSerial (serialization to binary)
```

---

## 10. Special Circumstances and Edge Cases

### 10.1 Dynamic Shapes

Dynamic shapes create significant constraints on what can be fused or placed on ANE:

- `Dynamic Shapes: One or more network operations are not ANE-resident - Marking all operations as non ANE-resident` — If any op in a dynamic-shape network cannot be ANE-resident, ALL ops are marked non-resident
- `Dynamic shape does not support conv with input or output stride larger than 2`
- `Dynamic shape does not support conv with same or same_lower padding when input stride is larger than 1`
- `Dynamic shape does not support pool with ceil_mode when input stride is larger than 1`
- `Dynamic shape does not support pool with same or same_lower padding when input stride is larger than 1`
- `Dynamic shape cannot support global max/min pool` — Must use reduction instead
- `Dynamic Shapes: memory layout operation is not supported for dynamic shape`
- `Dynamic shape not supported for SPMD procedures`
- `Large stride conv cannot support dynamic shape`

### 10.2 Multi-ANE (Bonded Networks)

When multiple ANE instances are available:

- `ZinBondedAne` manages multi-ANE deployment with `ZinDeploymentComponent` and `ZinPerLayerDeploymentComponentAlgorithm`
- `Bonded networks are not supported on the target` — Not all hardware supports multi-ANE
- `BondedSplitSubgraphIdentification` — Separate subgraph identification for bonded networks
- `Disable Per DMA RDTID for bonded networks`
- `Enable Work Stealing for bonded networks`
- Bonded network test assignments: `disabled`, `random`, `random_non_parallel`

### 10.3 SNE (Soft Neural Engine)

A third engine category, SNE, appears in the symbol table with its own pattern set (`InitializeSNEPatterns`):

- `Only one SNE Layer per group is expected`
- `SNE layer should not be in subgraph` — SNE layers are excluded from spatial splitting
- `ZinSneCodeGeneration` generates SNE-specific condition operations
- SNE supports `ScaledElementWise` via `ZinSNEAtoms::ScaledElementWiseAtom`

### 10.4 Program Chaining (Runtime)

At runtime, programs can be chained together for multi-model pipelines:

- `ANEServicesProgramChainingPrepare` — Prepares program chaining
- `ANEServicesProgramChainingSetActiveProcedure` — Sets active procedure in chain
- This enables zero-CPU-overhead transitions between ANE programs

### 10.5 NEKeepKernel / NEUsePrevKernel

Two hardware optimization registers control kernel caching:

- `SetNEKeepKernel(bool)` — Instructs the NE to keep the current kernel weights loaded
- `SetNEUsePrevKernel(bool)` — Instructs the NE to reuse the previously loaded kernel

These are available on codegen versions v7, v8, v10, and v11, enabling kernel weight reuse between consecutive operations that share the same weights (e.g., depthwise convolution across spatial tiles).

### 10.6 Double Buffering

`SetDoubleBufferingBasedOnOtherRegisters` — The hardware supports double buffering for overlapping compute and DMA, and this is automatically configured based on other register states.

---

## 11. Fusion Boundary Dump

The compiler can dump fusion boundary information for debugging:

- `Dump Fusion Boundary Info: %d` — Enable flag
- `Dumped fusion boundaries after MirOpt and before Spatial Split to JSON`
- `Dumped fusion boundaries after Reg Spill to JSON`
- `.zinir_graph_after_ne_transpose_fusion.dot` — GraphViz DOT output
- `.zinir_graph_final_fusion_info_` — Final fusion info dump
- `.zinir_graph_after_MirOpt_before_Spatial_Split_` — Pre-spatial-split graph
- `.MemoryPressure.debug.txt` — Memory pressure debug
- `.per_sched_pressure.txt` — Per-schedule pressure

---

## 12. Summary of Key Insights

### 12.1 How ANEC Fuses Ops

1. **Atoms are matched** against the operation graph using `ZinFusionPatterns`, organized by `FusionPatternType` (NE, PE, SNE, TransposeEngine)
2. **Matched atoms are grouped** into engine layers by `ZinMirLayerFusion::Group()` 
3. **Hoisting passes** move typecasts, activations, and GOCs to positions that enable further fusion
4. **Transpose fusion** absorbs data rearrangement into engine layers
5. **The fusion result is committed** via `ZinMirLayerFusion::Commit()`
6. **Legalization passes** verify that fused layers fit within hardware constraints (L2, latency, dimensions)

### 12.2 What Creates Fusion Boundaries

| Boundary Type | Mechanism | Example |
|---|---|---|
| **Memory pressure** | L2 cache / register overflow | Cluster cut at high pressure region |
| **Format mismatch** | Tensor format incompatibility | `IsFusableBasedOnFormat()` returns false |
| **ActiveNE constraint** | OCG size / interleave mismatch | `active_ne_times_ocg % logical_intlv != 0` |
| **Quantization incompatibility** | Unsupported quant type | Blockwise scale, asymmetric quant |
| **Architecture limit** | Feature not in hardware | E4M3, dilated pooling, circular buffer |
| **Dimension constraint** | Tensor too large | `ox * oy * cout > max_tensor_channels` |
| **Interleave constraint** | Invalid interleave factor | Factor not in {1,2,3,4,8} |
| **Dynamic shape** | Unsupported with dynamics | Conv stride > 2 with dynamic shape |
| **Palettization conflict** | Incompatible palette format | Vector palette + deconv |
| **Engine mismatch** | Cross-engine data transfer | NE output → PE input requires DMA |
| **L2 alias** | Chaining conflict | Chaining incorrectly enabled with L2 alias |

### 12.3 When Ops Fall Off the ANE

Operations fall off the ANE (back to GPU/CPU) when:
1. The operation type is fundamentally unsupported (dropout, condition layers, affine transform)
2. The data type is unsupported (E4M3, E5M2, 32-bit, 2xInt8)
3. The quantization format is incompatible (blockwise, asymmetric, Int4 per-cout dequant)
4. Dimension constraints are violated (rank > 5, tensor channels exceed max)
5. Memory pressure cannot be resolved (L2 legalizer fails, spatial split fails)
6. Dynamic shapes conflict with static-kernel operations
7. The operation requires a feature not present in the target architecture
8. Fusion boundary analysis determines the op must be isolated (no fusable neighbors)

---

*Analysis performed via static reverse engineering of ANECompiler binary using string extraction, symbol table analysis, and mangled C++ name demangling. No dynamic execution was performed. All findings are derived from embedded diagnostic strings, class/method names, and constraint error messages in the binary.*
