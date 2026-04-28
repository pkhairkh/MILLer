# MIL-to-ANE Placement Constraint System

**Target:** Apple Neural Engine Compiler (ANECompiler) Binary  
**Architecture:** Mach-O 64-bit ARM64e (Apple Silicon M2)  
**Binary Size:** 45,797,376 bytes | 133,164 symbols  
**Internal Codename:** "Zin" Compiler Stack (ZinAneCompiler-9.32.12)  
**Analysis Date:** 2026-04-24  

---

## 1. Executive Summary

This report provides the deepest possible analysis of how Apple's MIL (Machine Intermediate Language) operations get placed on the Apple Neural Engine (ANE) — the specific constraints on dimensions, sizes, ordering, formats, quantization, and the exact boundary conditions that determine whether an op lands on the ANE or falls back to CPU/GPU. Every finding is sourced directly from error strings, validation symbols, MLIR dialect constraints, and hardware limit structures extracted from the ANECompiler binary.

The analysis reveals a **5-layer constraint system** that gates ANE placement:

1. **MIL Validation Layer** — The `ValidateLayer` template system checks each MIL op against a `ANEC*LayerDesc` descriptor with `ANECTensorDesc` constraints before the op even enters the compiler pipeline.
2. **MLIR Placement Dialect** — The `mlir::placement` dialect assigns ops to `region_type` (ANE vs Host), with `force-ane-placement` / `force-host-placement` overrides and `ANEIOCast` boundary ops.
3. **Zin Unit Validation** — `ZinUnitValidator` enforces per-op HW limits (dimensions, channels, kernel sizes) using versioned `_target_hw_limits_v*` structures.
4. **Fusability Checks** — `IsFusable*()` methods determine if ops can be merged into engine layers; failed fusability forces graph breaks and DMA round-trips.
5. **Memory Pressure & Legalization** — Even if an op passes all prior checks, L2 cache pressure, DRAM limits, and spatial split feasibility determine final ANE residency.

**The single most important finding:** The binary contains **40+ distinct `ValidateLayer` instantiations**, each templated on a specific `ANEC*LayerDesc` type with `ANECTensorDesc` parameters. These are the gatekeepers — if validation fails for any layer, it cannot be placed on ANE. Period.

---

## 2. The MIL→ANEC Validation Chain: The First Gate

### 2.1 The ValidateLayer Template System

Every MIL operation that wants to run on the ANE must pass through a `ValidateLayer` instantiation. The binary contains a comprehensive set of these, each specific to an operation type:

| ANEC Layer Descriptor | MIL Operation | Validator Class |
|---|---|---|
| `ANECConvLayerDesc` | Convolution | `ZinConvValidator<ANECConvLayerDesc, ANECTensorDesc>` |
| `ANECConcatLayerDesc` | Concatenation | `ZinConcatValidator<ANECConcatLayerDesc, ANECTensorDesc>` |
| `ANECGOCLayerDesc` | Generic Operation Compute | `ZinGOCValidator<ANECGOCLayerDesc, ANECTensorDesc>` |
| `ANECElementWiseLayerDesc` | Element-wise ops | `ValidateLayer<ANECTensorDesc, ZinIrEWUnit, ZinIrEWUnitInfo, ANECElementWiseLayerDescAlternate>` |
| `ANECNeuronLayerDesc` | Activation functions | `ValidateLayer<ANECTensorDesc, ZinIrNeuronUnit, ZinIrNeuronUnitInfo, ANECNeuronLayerDescAlternate>` |
| `ANECPoolLayerDesc` | Pooling | `ValidateLayer<ANECTensorDesc, ZinIrPoolUnit, ZinIrPoolUnitInfo, ANECPoolLayerDescAlternate>` |
| `ANECReductionLayerDesc` | Reduction | `ValidateLayer<ANECTensorDesc, ZinIrReductionUnit, ZinIrReductionUnitInfo, ANECReductionLayerDescAlternate>` |
| `ANECMatrixMultLayerDesc` | MatMul | `ValidateLayer<ANECTensorDesc, ZinIrMatrixMultUnit, ZinIrMatrixMultUnitInfo, ANECMatrixMultLayerDescAlternate>` |
| `ANECTransposeLayerDesc` | Transpose | `ValidateLayer<ANECTensorDesc, ZinIrTransposeUnit, ZinIrTransposeUnitInfo, ANECTransposeLayerDescAlternate>` |
| `ANECReshapeLayerDesc` | Reshape | `ValidateLayer<ANECTensorDesc, ZinIrReshapeUnit, ZinIrReshapeUnitInfo, ANECReshapeLayerDescAlternate>` |
| `ANECSoftmaxLayerDesc` | Softmax | `ValidateLayer<ANECTensorDesc, ZinIrSoftmaxUnit, ZinIrSoftmaxUnitInfo, ANECSoftmaxLayerDescAlternate>` |
| `ANECLayerNormLayerDesc` | Layer Normalization | `ValidateLayer<ANECTensorDesc, ZinIrLayerNormUnit, ZinIrLayerNormUnitInfo, ANECLayerNormLayerDescAlternate>` |
| `ANECInstanceNormLayerDesc` | Instance Norm | `ValidateLayer<ANECTensorDesc, ZinIrInstanceNormUnit, ZinIrInstanceNormUnitInfo, ANECInstanceNormLayerDescAlternate>` |
| `ANECLinearLayerDesc` | Linear/Fully Connected | `ValidateLayer<ANECTensorDesc, ZinIrLinearUnit, ZinIrLinearUnitInfo, ANECLinearLayerDescAlternate>` |
| `ANECBroadcastLayerDesc` (alt) | Broadcast | `ValidateLayer<ANECTensorDesc, ZinIrBroadcastUnit, ZinIrBroadcastUnitInfo, ANECBroadcastLayerDescAlternate>` |
| `ANECScaledElementWiseLayerDesc` | Scaled Element-Wise | `ValidateLayer<ANECTensorDesc, ZinIrScaledEWUnit, ZinIrScaledEWUnitInfo, ANECScaledElementWiseLayerDescAlternate>` |
| `ANECSDPALayerDesc` | Scaled Dot-Product Attention | `ValidateLayer<ANECTensorDesc, ZinIrSDPAUnit, ZinIrSDPAUnitInfo, ANECSDPALayerDescAlternate>` |
| `ANECResampleLayerDesc` | Resample/Resize | `ValidateLayer<ANECTensorDesc, ZinIrResampleUnit, ZinIrResampleUnitInfo, ANECResampleLayerDescAlternate>` |
| `ANECArgMinMaxLayerDesc` | ArgMin/ArgMax | `ValidateLayer<ANECTensorDesc, ZinIrArgMinMaxUnit, ZinIrArgMinMaxUnitInfo, ANECArgMinMaxLayerDescAlternate>` |
| `ANECGlobalArgMinMaxLayerDesc` | Global ArgMin/ArgMax | `ValidateLayer<ANECTensorDesc, ZinIrGlobalArgMinMaxUnit, ZinIrGlobalArgMinMaxUnitInfo, ANECGlobalArgMinMaxLayerDescAlternate>` |
| `ANECGatherLayerDesc` | Gather | `ValidateLayer<ANECTensorDesc, ZinIrGatherUnit, ZinIrGatherUnitInfo, ANECGatherLayerDescAlternate>` |
| `ANECDynamicSliceLayerDesc` | Dynamic Slice | `ValidateLayer<ANECTensorDesc, ZinIrDynamicSliceUnit, ZinIrDynamicSliceUnitInfo, ANECDynamicSliceLayerDescAlternate>` |
| `ANECInputViewLayerDesc` | Input View / Slice | `ValidateLayer<ANECTensorDesc, ZinIrInputViewUnit, ZinIrInputViewUnitInfo, ANECInputViewLayerDescAlternate>` |
| `ANECPadLayerDesc` | Padding | `ValidateLayer<ANECTensorDesc, ZinIrPadUnit, ZinIrPadUnitInfo, ANECPadLayerDescAlternate>` |
| `ANECNMSLayerDesc` | Non-Maximum Suppression | `ValidateLayer<ANECTensorDesc, ZinIrNMSUnit, ZinIrNMSUnitInfo, ANECNMSLayerDescAlternate>` |
| `ANECLRNLayerDesc` | Local Response Norm | `ValidateLayer<ANECTensorDesc, ZinIrLRNUnit, ZinIrLRNUnitInfo, ANECLRNLayerDescAlternate>` |
| `ANECL2NormLayerDesc` | L2 Normalization | `ValidateLayer<ANECTensorDesc, ZinIrL2NormUnit, ZinIrL2NormUnitInfo, ANECL2NormLayerDescAlternate>` |
| `ANECMinMaxNormLayerDesc` | Min/Max Normalization | `ValidateLayer<ANECTensorDesc, ZinIrMinMaxNormUnit, ZinIrMinMaxNormUnitInfo, ANECMinMaxNormLayerDescAlternate>` |
| `ANECPixelShuffleLayerDesc` | Pixel Shuffle | `ValidateLayer<ANECTensorDesc, ZinIrPixelShuffleUnit, ZinIrPixelShuffleUnitInfo, ANECPixelShuffleLayerDescAlternate>` |
| `ANECPixelUnshuffleLayerDesc` | Pixel Unshuffle | `ValidateLayer<ANECTensorDesc, ZinIrPixelUnshuffleUnit, ZinIrPixelUnshuffleUnitInfo, ANECPixelUnshuffleLayerDescAlternate>` |
| `ANECChannelToSpaceLayerDesc` | Channel-to-Space | `ValidateLayer<ANECTensorDesc, ZinIrChannelToSpaceUnit, ZinIrChannelToSpaceUnitInfo, ANECChannelToSpaceLayerDescAlternate>` |
| `ANECSpaceToChannelLayerDesc` | Space-to-Channel | `ValidateLayer<ANECTensorDesc, ZinIrSpaceToChannelUnit, ZinIrSpaceToChannelUnitInfo, ANECSpaceToChannelLayerDescAlternate>` |
| `ANECBatchToSpaceLayerDesc` | Batch-to-Space | `ValidateLayer<ANECTensorDesc, ZinIrBatchToSpaceUnit, ZinIrBatchToSpaceUnitInfo, ANECBatchToSpaceLayerDescAlternate>` |
| `ANECSpaceToBatchLayerDesc` | Space-to-Batch | `ValidateLayer<ANECTensorDesc, ZinIrSpaceToBatchUnit, ZinIrSpaceToBatchUnitInfo, ANECSpaceToBatchLayerDescAlternate>` |
| `ANECCropResizeLayerDesc` | Crop-Resize | `ValidateLayer<ANECTensorDesc, ZinIrCropResizeUnit, ZinIrCropResizeUnitInfo, ANECCropResizeLayerDescAlternate>` |
| `ANECCrossCorrelationLayerDesc` | Cross Correlation | `ValidateLayer<ANECTensorDesc, ZinIrCrossCorrelationUnit, ZinIrCrossCorrelationUnitInfo, ANECCrossCorrelationLayerDescAlternate>` |
| `ANECRingBufferWriterLayerDesc` | Ring Buffer Writer | `ValidateLayer<ANECTensorDesc, ZinIrRingBufferWriterUnit, ZinIrRingBufferUnitInfo, ANECRingBufferWriterLayerDescAlternate>` |
| `ANECDynamicGOCLayerDesc` | Dynamic GOC | `ValidateLayer<ANECTensorDesc, ZinIrDynamicGOCUnit, ZinIrDynamicGOCUnitInfo, ANECDynamicGOCLayerDescAlternate>` |
| `ANECSortLayerDesc` | Sort | `ValidateLayer<ANECTensorDesc, ZinIrSortUnit, ZinIrSortUnitInfo, ANECSortLayerDescAlternate>` |
| `ANECTileLayerDesc` | Tile | `ValidateLayer<ANECTensorDesc, ZinIrTileUnit, ZinIrTileUnitInfo, ANECTileLayerDescAlternate>` |
| `ANECTopKLayerDesc` | Top-K | `ValidateLayer<ANECTensorDesc, ZinIrTopKUnit, ZinIrTopKUnitInfo, ANECTopKLayerDescAlternate>` |
| `ANECDropoutLayerDesc` | Dropout | `ValidateLayer<ANECTensorDesc, ZinIrDropoutUnit, ZinIrDropoutUnitInfo, ANECDropoutLayerDescAlternate>` |
| `ANECFlattenLayerDesc` | Flatten | `ValidateLayer<ANECTensorDesc, ZinIrFlattenUnit, ZinIrFlattenUnitInfo, ANECFlattenLayerDescAlternate>` |
| `ANECUnflattenLayerDesc` | Unflatten | `ValidateLayer<ANECTensorDesc, ZinIrUnflattenUnit, ZinIrUnflattenUnitInfo, ANECUnflattenLayerDescAlternate>` |
| `ANECAffineTransformLayerDesc` | Affine Transform | `ValidateLayer<ANECTensorDesc, ZinIrAffineTransformUnit, ZinIrAffineTransformUnitInfo, ANECAffineTransformLayerDescAlternate>` |
| `ANECCrossProductLayerDesc` | Cross Product | `ValidateLayer<ANECTensorDesc, ZinIrCrossProductUnit, ZinIrUnitInfo, ANECCrossProductLayerDescAlternate>` |
| `ANECRandomLayerDesc` | Random | `ValidateLayer<ANECTensorDesc, ZinIrRandomUnit, ZinIrRandomUnitInfo, ANECRandomLayerDescAlternate>` |
| `ANECResizeLayerDesc` | Resize | `ValidateLayer<ANECTensorDesc, ZinIrResizeUnit, ZinIrResizeUnitInfo, ANECResizeLayerDescAlternate>` |

Each `ValidateLayer` call takes: the layer descriptor, a vector of `ANECTensorDesc` (input tensor descriptions), and a vector of `ANECTensorValueDesc` (input tensor values/constants). The validation is a two-phase check: `ValidateLayer` (outer) calls `ValidateLayer_Impl` (inner), and if validation fails, execution branches to a `.cold.1` error path.

### 2.2 The ANECTensorDesc Structure

The `ANECTensorDesc` describes the shape and format of every tensor entering and leaving an op. It is initialized via `_ANECTensorDescInitialize` and `_ANECTensorDimsInitialize`. Key constraints validated on tensor descriptors:

- **Rank must be in [0, 7]**: `Invalid tensor rank (%lu), must be between [0,7]`
- **Tensor dimension element count**: `Tensor dimension exceeds max number of elements ANECTensorValueDesc can hold`
- **Tensor format must be valid**: A `ZinTensorFormat` enum determines layout; `packed10` is explicitly rejected
- **Interleave factor must be valid**: `Error: invalid input interleave factor:%zd; The valid interleave factor should be 1, 2, 3, 4, or 8`
- **Int4 format must have interleave 8**: `Tensor with the int4 format must have an interleave factor of 8`

---

## 3. The MLIR Placement Dialect: ANE vs Host Region Assignment

### 3.1 The `mlir::placement` Dialect

The binary contains a complete MLIR dialect called `mlir::placement` that governs whether ops execute on the ANE or on the host (CPU/GPU). This is the **second gate** in the constraint system.

**Key placement operations:**

| MLIR Op | Purpose |
|---|---|
| `placement.region_call` | Calls a region (ANE or Host); has `region_type` and `callee` attributes |
| `placement.ane_io_cast` | Casts data between ANE and host representations at region boundaries |
| `placement.memref_to_tensor` | Converts memref (host memory) to tensor (ANE format); has `shape`, `resultElementType`, `interleave` attributes |
| `placement.tensor_to_memref` | Converts tensor back to memref; same attributes as above |
| `placement.replaced_ops` | Marks ops that have been replaced during placement; requires `replaced_ops_ref` attribute |
| `placement.replaced_ops_live_outs` | Tracks live-out values from replaced ops |
| `placement.start_timer` / `placement.stop_timer` | Performance measurement at region boundaries; uses `TimerHandleType` |
| `placement.host_type_cast` | Type casting at host region boundary |

### 3.2 Region Types

Every op in the MLIR ANEC dialect must have a `region_type` attribute. The placement system determines this based on:

1. **Op compatibility** — Can the op map to one of the three ANE engines (NE, PE, TransposeEngine)?
2. **Dimension/format compatibility** — Do the tensor shapes and formats satisfy HW limits?
3. **Fusion compatibility** — Can the op be fused into an engine layer, or does it require a standalone layer?
4. **Cost model** — Is ANE execution predicted to be faster? (Controlled by `If true, placement uses the cost model.`)

**Critical placement flags:**

| Flag | Effect |
|---|---|
| `force-ane-placement` | Forces ALL ops onto the ANE (fails if any op is incompatible) |
| `force-host-placement` | Forces ALL ops onto the host |
| `print-placement-report` | Prints the "ANEC Placement Report" showing per-op decisions |
| `Could not follow op placement hint` | Error when `force-ane-placement` is set but an op cannot be placed on ANE |

### 3.3 The Placement Boundary Mechanism

When an op cannot be placed on ANE, the placement dialect inserts boundary ops:

```
[ANE Region] → placement.tensor_to_memref → [Host Region] → placement.memref_to_tensor → [ANE Region]
```

Each `tensor_to_memref` / `memref_to_tensor` pair specifies:
- `shape`: The exact tensor dimensions (ui64 array)
- `resultElementType`: The data type
- `interleave`: The ANE interleave factor (must be 1, 2, 3, 4, or 8)

The `ANEIOCast` op handles format conversion at these boundaries when the host and ANE representations differ (e.g., NHWC on ANE vs NCHW on host).

---

## 4. Per-Op Dimension & Size Constraints (The Full Catalog)

### 4.1 Tensor Dimension Limits (Universal)

These limits apply to ALL operations on the ANE. They are derived from `hal_params` (Hardware Abstraction Layer parameters):

| Constraint | Meaning |
|---|---|
| `(ox * oy * cout * num_groups) <= hal_params.max_tensor_channels` | Total output channels across spatial dims and groups must fit |
| `(sx * sy * cin * num_groups) <= hal_params.max_tensor_channels` | Total input channels across spatial dims and groups must fit |
| `(ox * oy * oz * cout) <= hal_params.max_tensor_channels` | 3D output channel limit |
| `(sx * sy * sz * cin) <= hal_params.max_tensor_channels` | 3D input channel limit |
| `(hout * oy) <= hal_params.max_tensor_height` | Output height constraint |
| `(wout * ox) <= hal_params.max_tensor_width` | Output width constraint |
| `dout * oz <= hal_params.max_tensor_depth` | Depth dimension limit |
| `win <= hal_params.max_tensor_width - 8` | Input width must leave 8-pixel margin |
| `tile_height * sy <= hal_params.hw_workarounds.max_tile_height_times_sy_constraint_with_ne_task_and_replication_padding` | Tile height × stride-Y product limit |

**Error messages when limits are exceeded:**
- `Error: Tensor depth goes beyond limit supported.`
- `Error: Tensor height goes beyond limit supported.`
- `Error: Tensor width goes beyond limit supported (%lu > %lu).`
- `Error: channel size exceeds limit [%zu vs. %zu]`

### 4.2 Convolution Constraints

**Standard Convolution:**
- `Conv2D input must be 4D` — 4D tensor required (NCHW or NHWC)
- `Conv must have Kernel` — No kernel-less convolutions
- `Conv stride must be 1 for batch / channel axis` — No striding on B or C
- `Conv dilation must be 1 for batch / channel axis` — No dilation on B or C
- `Conv output channel must be divisible factors.x * factors.y` — Output channels must be divisible by PixelShuffle factors when used with deconv decomposition
- `Filter shape Cin * groups must match input Cin` — Cin consistency with groups

**Kernel Size Constraints:**
- `Kernel width must be a power of 2.`
- `Kernel height must be a power of 2.`
- `Kernel depth must be a power of 2.`
- `Error: kernel width and height should be multiple of 8 for large kernel but are %zd and %zd`

**Group Convolution:**
- `ChannelWise Conv must have num_groups == out_dims.c` — Depthwise requires groups == output channels
- `ChannelWise Conv must have in_dims.c == out_dims.c` — Depthwise in/out channels match
- `Error: the input channel (%d) is not divisible by the number of groups (%d)` — Cin must be divisible by groups
- `Error: the output channel (%d) is not divisible by the number of groups (%d)` — Cout must be divisible by groups
- `Active_ne count must be divisible by num_groups for a 1-1 NE to LUT ratio in group-conv`
- `Error: grouped conv with large kernel size is not supported` — GROUPED + LARGE KERNEL = hard reject

**Dilated Convolution:**
- `Error: dilated conv with large kernel size is not supported` — DILATED + LARGE KERNEL = hard reject
- `Error: Dilation factor should be 1` — Some paths require dilation = 1
- `Dilation factor should be 1 for Deconv with stride > 2`
- `Dilation not supported for deconvolution`
- `Dilated convolution should be handled as part of composite conv layer` — Must be decomposed
- `Error: Dilated convolution cannot be lowered since all possible space-to-batch implementations exceeded the L2 DMA buffer size` — If SpaceToBatch decomposition exceeds L2 budget, dilated conv is rejected entirely

**Large Kernel Convolution:**
- `Error: kernel with depth = %zd > 1 is not supported for large kernel` — 3D large kernel not supported
- `Error: input and output x strides should be the same for conv with large kernel but are %d and %d` — Input/output strides must match
- `Error: input and output y strides should be the same for conv with large kernel but are %d and %d`

**Large Stride Convolution:**
- `Criterias for large kernel strides are not met` — Unspecified criteria failed
- `Criterias for large kernel strides are not met due to large tensor dimensions` — Tensor too big for large stride path
- `Dynamic Shape does not support conv with input or output stride larger than 2` — Dynamic shapes + stride > 2 = reject
- `Dynamic shape does not support conv with same or same_lower padding when input stride is larger than 1`
- `Convolution layer %s with large kernel size cannot be support by dynamic shape` — Large kernel + dynamic = reject
- `Error: Large stride conv cannot support dynamic shape`

**Deconvolution:**
- `Deconv lower is supported for 2x2x1 only` — Only 2x2 spatial + depth-1 deconv can be lowered
- `Deconv lower is supported for per-cout bias goc only` — Must use per-output-channel bias
- `Deconv must have OCmode 1` — Output channel mode must be 1
- `Deconv on uANE is not yet supported` — Micro-ANE doesn't support deconv
- `Error: deconv with SOx != 2 is not supported` — Stride-output-x must be exactly 2
- `Error: deconv with stride > 1 is not supported along depth axis`
- `Error: deconv with stride > 2 does not support kernel depth > 1`
- `Error: deconv with large kernel size is not supported`
- `Error: deconv with vector palettization is not supported`
- `Error: dilated deconvolution is not supported!`
- `Deconv with stride 4 is supported only for SAME mode`
- `Deconv with odd output lower is wrong`
- `Error: invalid dimension for deconv splitting`

### 4.3 Pooling Constraints

**Kernel size:**
- `Invalid Pool kernel width (%zd), must be [1-%zd] or %zd` — Bounded by HW limits
- `Invalid Pool kernel height (%zd), must be [1-%zd] or %zd`
- `Invalid Pool kernel depth (%zd), must be [1-%zd] or %zd`

**Strides:**
- `Pool with strides of 3 is only supported with Avg mode` — Stride-3 only for avg pool
- `Pool input tensor width (%zd) must be a multiple of stride (%d)` — Width divisible by stride
- `Pool input tensor height (%zd) must be a multiple of stride (%d)` — Height divisible by stride
- `Pool input tensor depth (%zd) must be a multiple of stride (%d)` — Depth divisible by stride

**Padding:**
- `MaxPool padding mode must be Negative` — MaxPool requires negative padding mode
- `MinPool padding mode must be Positive` — MinPool requires positive padding mode
- `L2Pool padding mode must be Replication or Zero` — L2Pool only allows these two
- `Cannot support avg pool with exclude padding size equal to or larger than kernel size` — Exclude padding < kernel size
- `Dilated Pooling not supported on ANE` — No dilated pooling at all

**Dynamic shapes:**
- `Dynamic shape does not support pool with ceil_mode when input stride is larger than 1`
- `Dynamic shape does not support pool with same or same_lower padding when input stride is larger than 1`
- `Dynamic shape cannot support global max/min pool, please use reduction to replace global max/min pool`

### 4.4 Element-Wise Constraints

**NE Element-Wise:**
- `NEElementWise can only have input activation mode as Relu` — Only ReLU as input activation
- `NEElementWise must contain ew_` — Must have EW prefix
- `In ZinNEElementWiseLayer, input channel count must be a multiple of output channel count` — Cin must be multiple of Cout
- `For NE Elementwise, input channel must be divisible by programmed output channel`
- `NEElementWise must have only 2 bottoms` — Exactly 2 inputs for NE EW

**PE Element-Wise:**
- `Scaled Elementwise must have 1 or 2 input` — 1 or 2 inputs allowed
- `Error: PEEW with uint16 input must have only one dma src` — uint16 limits DMA sources to 1
- `Error: Unable to fuse GOC, GOC and EW_MAX` — Two GOCs + EW max cannot coexist
- `Error: Unable to fuse Transpose, ScaledEW, Transpose to GOC` — This specific 3-op pattern rejected
- `Error: failed fusing scales to sew` — Scale fusion into scaled-EW failed

**Broadcast:**
- `Broadcast output tensor batch must be 1 or match input_batch size`
- `Broadcast output tensor channel must be 1 or match input_channel size`
- `Broadcast output tensor depth must be 1 or match input_depth size`
- `Broadcast output tensor height must be 1 or match input_height size`
- `Broadcast output tensor width must be 1 or match input_width size`

### 4.5 MatMul / Linear Constraints

**Matrix Multiplication:**
- `Error: depth > 1 is not supported for MatMult inputs but get dim_A.d = %zd, dim_B.d = %zd` — Depth must be 1 for both inputs
- `Error: invalid output channel = %zd. Expecting it to be input A's channel dimension = %zd for MatMult` — Cout must equal Cin of input A
- `Error: invalid input dim : input tensor A's width (%zd) + padding (%d) != tensor B's channel (%zd)` — A.width + padding must equal B.channel
- `Error: number of output channel must be a multiple of ox`
- `MatMul has multiple users` — MatMul with multiple consumers may limit fusion

**Linear:**
- `ANE cannot support Linear with input rank >= 5` — Max rank 4 for linear ops

### 4.6 Normalization Constraints

**LayerNorm:**
- `LayerNorm input channels %zd must be divisible by num_groups (Specified: %zd)` — Channels divisible by groups
- `LayerNorm output tensor format must be Float (Specified %s)` — Must be float output

**InstanceNorm:**
- `InstanceNorm output tensor must be in Float format` — Float only
- `InvalidUnitInstanceNormDimension` — Dimension constraint violated

**BatchNorm:**
- `BatchNorm layer must have only one single input`
- `BatchNorm must have kernel` — Requires kernel data
- `BatchNorm output tensor must be in Float format`

**LRN:**
- `LRN height and width kernel dimensions must be 1. Height: %zd, Width: %zd` — Only 1×1 HW kernels
- `LRN Channel count must be greater than kernel channel. Channel: %zd, Kernel Depth: %zd`
- `Tensor depth must be 1 for a channelwise LRN. Depth: %zd`

**L2Norm:**
- `Error: L2 norm output tensor format must be Float16` — Must be FP16
- `L2 norm does not support batch axis` — Batch dimension not allowed
- `InvalidUnitL2NormDimension` — Dimension constraint violated

**MinMaxNorm:**
- `Min/max norm layer does not support batch/channel axis` — Neither batch nor channel axis allowed

### 4.7 Softmax Constraints

- `Softmax output tensor must be in Float format`
- Softmax is decomposed internally into: max-reduction → subtract → exp2 → sum-reduction → element-wise multiply

### 4.8 SDPA (Scaled Dot-Product Attention) Constraints

- `SDPA layer must have only 4 or 5(optional mask) inputs`
- `4 or 5 bottoms must be present for SDPA`
- `SDPA does not currently support rank 5 operands` — Max rank 4
- `Key and value must be the same shape`
- `Mask Channel axis must match Q Channel axis or broadcastable`
- `Mask Width axis must match K and V Channel axis`
- `Mask format must be same as Q, K and V`
- `L2 budget for attention: %d` — L2 cache budget dedicated to attention

### 4.9 Gather Constraints

- `Invalid gather axes size %ld, must be 3` — Must gather on exactly 3 axes
- `Invalid gather data tensor batch size %ld, must be 1` — Batch must be 1
- `Invalid gather data tensor depth size %ld, must be 1` — Depth must be 1
- `Invalid gather index tensor channel size %ld, must be %ld` — Channel must match specific value
- `Invalid gather index tensor width size %ld, must be 1` — Width must be 1
- `Invalid gather index tensor depth size %ld, must be 1` — Depth must be 1

### 4.10 ArgMinMax Constraints

- `Invalid ArgMinMax pool stride x:%d, must be within [1, 2, 4]` — Stride-X ∈ {1, 2, 4}
- `Invalid ArgMinMax pool stride y:%d, must be within [1, 2, 4]` — Stride-Y ∈ {1, 2, 4}
- `ArgMinMax must have equal top and bottom padding, but top:%d and bot:%d are given` — Symmetric vertical padding
- `ArgMinMax must have equal left and right padding, but left:%d and right:%d are given` — Symmetric horizontal padding
- `ArgMinMax must have zero front and back padding, but front:%d and back:%d are given` — No depth padding
- `Invalid negative W_out / H_out` — Output dimensions must be positive

### 4.11 Transpose / Reshape / View Constraints

**InputView (Slicing/Striding):**
- `anec.input_view` requires attributes: `dimension`, `offset`, `size`, `step` (all integers)
- `anec.input_view with negative stride must have size {0} that equals the size of tensor {1} at dimension {2}` — Negative strides require full-dimension size
- `anec.input_view with offset, size and stride is out of bounds for dimension` — Bounds checking
- `Error: View unit cannot be a partial view with negative strides` — Partial negative-stride views rejected
- `Strided view on %s (%ld) is not allowed` — Some dims don't allow strided views
- `Strided Stencil not supported on ANE` — Strided stencils rejected
- `Negative strides for intermediate DRAM symbol is not supported`

**Reshape:**
- `Reshape is not supported when enabling dynamic shape` — Dynamic shapes cannot reshape
- `Error: Invalid output dimensions for Reshape unit` — Output dims don't match input element count
- `Error: Unexpected unaligned width for Reshape with Dynamic Shapes enabled`

**Transpose:**
- `ANEC only supports interleave on C axis` — Transpose interleaving only on channel
- `TransposeNC: Invalid input channel %ld, must be 1` — NC transpose requires C=1
- `TransposeNH: Invalid input dimension` — NH transpose has specific input dimension requirements
- `Invalid N-related Transpose input dimension` — Batch-related transposes heavily constrained

### 4.12 Space/Batch/Channel Transform Constraints

**PixelShuffle:**
- `Error: PixelShuffle's channel dimension must be divisible by the product of its factors`
- `Error: Input dimensions are invalid; height must be a multiple of the shuffle factor`
- `Error: Input dimensions are invalid; must be a multiple of the shuffle factor`
- `Error: spatial dimensions should be multiple of shuffle factor`

**PixelUnshuffle:**
- `Error: Input dimensions should be divisible by the PixelUnshuffle factor`
- `PixelUnshuffle on z axis is not supported`
- `InvalidUnitPixelUnshuffleType`

**ChannelToSpace:**
- `ChannelToSpace in z dimension is not supported`
- `ZinChannelToSpaceLargeFactorCompositeLayer input must be unique`

**SpaceToChannel:**
- `SpaceToChannel layer must have only one input`

**BatchToSpace:**
- `BatchToSpace batch_axis operand must be constant`
- `BatchToSpace batch_axis operand must match with ANEC batch axis`
- `BatchToSpace block_dimensions operand must be constant`
- `BatchToSpace spatial_axes operand must match with ANEC spatial axes`
- `Error: Input batch dimensions should be divisible by the product of BatchToSpace factors`
- `batch-to-space on z axis is not supported`

**SpaceToBatch:**
- `Error: Input dimensions should be divisible by the SpaceToBatch factor`

### 4.13 Pad/Resize/Crop Constraints

**Padding:**
- `Replication padding mode is not supported`
- `Symmetric padding mode is not supported`
- `Negative padding mode is not supported`
- `Channel padding is not supported on ANE`
- `Padding on the batch dimension is not supported yet`
- `Padding on the channel dimension is not supported yet`
- `Padding mode must be the same on all axes`
- `Invalid background padding (%f) in L2_format (%s), it is beyond ANE limits range [%d, %d]`

**Resample/Resize:**
- `Cannot support relative coordinate type + normalized coordinate/padding mode on %s`
- `Cannot support coordinate mode %s with padding mode %s on %s`
- `Background padding with NearestNeighbor is not supported for Resample on %s`
- `Dynamic shape resize must have shape op input`
- `Only support 2D dynamic shape resize`
- `FactorX should be a multiple of 2 or 3`

**CropResize:**
- `Crop Resize index format must be FP16`
- `Invalid crop_resize crop width/height size, it should be in the range of [%ld, %ld]`
- `Illegal crop_resize batch sizes for index tensor and input tensor`

### 4.14 Sort/TopK/NMS Constraints

**Sort/TopK:**
- `Sort unit output format must be FP16 or Uint16`
- `TopK unit output format must be FP16 or Uint16`
- `Sort/TopK layer's vector_dimension size exceeds the record limit`

**NMS:**
- `ANE cannot support NMS for ios15 and ios16` — OS-version-dependent rejection
- `NMS Boxes batch must equal Scores batch`
- `NMS Boxes channels must be %zd`
- `NMS Boxes height and Scores height must equal 1`
- `NMS Boxes width must equal Scores width`
- `NMS output format must be either Float 16 or UInt 16`

### 4.15 Cross Correlation Constraints

- `Input size for Cross Correlation layer must be 2`
- `Cross Correlation: Num Groups must be positive`
- Template input must be rasterized: width = TemplateHeight × TemplateWidth [× inputChannel]
- Template height must be 1, depth must be 1
- Padding cannot be negative
- Reference dimensions + padding must be >= template dimensions

---

## 5. Data Type & Format Constraints

### 5.1 Supported Data Types on ANE

| Data Type | Supported | Notes |
|---|---|---|
| Float16 (FP16) | ✅ | Primary compute format |
| Float32 (FP32) | ✅ (limited) | Input/output only; internal compute is FP16 |
| Int8 | ✅ | Quantized weights/activations |
| UInt8 | ✅ | Quantized weights/activations |
| Int4 | ✅ (constrained) | Must use interleave factor 8 |
| E4M3 (FP8) | ⚠️ Arch-dependent | `Error: E4M3 is not supported` on some architectures |
| E5M2 (FP8) | ❌ | `E4M3 or E5M2 format not supported` |
| 32-bit format | ❌ | `32 bit format not supported` for compute |
| 2xInt8 | ❌ | `2xInt8 mode is not supported` |
| Packed10 | ❌ | `Invalid input tensor format: packed10` |

### 5.2 Quantization Constraints

| Constraint | Details |
|---|---|
| `ANE doesn't support blockwise scale` | NO blockwise quantization. Only per-tensor and per-output-channel |
| `Asym quantization is not supported` | NO asymmetric quantization |
| `Int4 Per-Cout Dequant is not supported` | Int4 can dequant, but not per-cout |
| `Zero point is not supported for quant with E4M3 output format` | E4M3 quant has no zero point |
| `Dequant layer must have fp16 output format` | Dequant output is always FP16 |
| `Dequant layer must have int8, uint8, int4 or e4m3 input format` | Only these input formats for dequant |
| `Quant layer must have fp16 or fp32 input format` | Quant input is always float |
| `Quant layer must have int8, uint8, e4m3 or e5m2 output format` | Only these output formats for quant |
| `mutable kernels are not supported with kernel quantization` | Mutable + quantized = reject |

### 5.3 Palettization (LUT) Constraints

| Constraint | Details |
|---|---|
| Palette ranks 0-6 supported | `dense elements attribute for palettized LUT of rank 0/1/2/3/4/5/6` |
| `1 and 2 bit palettes not supported` | Minimum 4-bit palettes |
| `CW conv does not support vector palette` | Channel-wise conv + vector palettization = reject |
| `deconv with vector palettization is not supported` | Deconv + vector palette = reject |
| `dilation is not supported for vector palettized kernels yet` | Dilation + vector palette = reject |
| `does not support palettized weight with large kernel stride` | Large stride + palette = reject |
| `per channel cout with vector palettization is not supported yet` | Per-cout + vector palette = reject |
| `Consecutive palette_vector_size channels must make use of the same palette LUT` | LUT sharing constraint |
| `Invalid OCG size for vector palettization` | OCG size must work with palette |
| Palette LUT size must not exceed `ne_palette_lut_size_in_bytes` | Hardware LUT size limit |
| `Convolution layer has been lowered to have both palettization and compression` — Having both is an error |

---

## 6. Memory Layout & Format Constraints

### 6.1 Supported Layouts

The MLIR MPS dialect enforces layout constraints at the ANEC boundary:

| Operation | Valid Data Layouts | Valid Weight Layouts |
|---|---|---|
| Conv2D | `NCHW` or `NHWC` | (conv-specific) |
| Conv3D | `NDHWC` or `NCDHW` | `DHWIO` or `OIDHW` |
| Depthwise Conv2D | `NCHW` or `NHWC` | (depthwise-specific) |
| Depthwise Conv3D | (3D format) | (3D format) |
| Cost Volume | `NCHW` or `NHWC` | — |

### 6.2 ChannelLast Constraints

ChannelLast (NHWC) layout has severe restrictions on the ANE:

- `ChannelLast currently only supported for channel wise convolutions` — Only depthwise conv
- `ChannelLast does not support non-one channel interleave` — Interleave must be 1
- `ChannelLast in DRAM currently not supported` — ChannelLast requires L2 cache
- `ChannelLast not supported` — Some architectures reject it entirely
- `ANE Layer cannot mix channellast and non-channellast input/output tensors` — Must be consistent

### 6.3 Interleave Constraints

Interleave factor controls how channels are interleaved in memory:

| Constraint | Details |
|---|---|
| Valid values: {1, 2, 3, 4, 8} | `invalid input interleave factor; should be 1, 2, 3, 4, or 8` |
| `ANEC only supports interleave on C axis` | Interleave only on channel dimension |
| `Const tensor interleave must be 1` | Constant tensors have interleave 1 |
| `ChannelLast does not support non-one channel interleave` | NHWC requires interleave 1 |
| Int4 tensors require interleave 8 | `Tensor with the int4 format must have an interleave factor of 8` |
| `(1 << effective_ocgsize) >= interleave_factor * dma_interleave` | OCG size must be large enough for interleave |
| `Input channel should be a multiple of input interleave number` | Channel divisibility |
| `Invalid index tensor channel count %ld, not divisible by interleave %ld` | Gather index must be divisible |

### 6.4 Alignment Constraints

- `Input and output tensors must be aligned`
- `Invalid input tensor channel %ld and format size %ld bytes, must be aligned on %ld bytes` — Byte alignment requirement
- `Invalid input tensor width %ld, must be divisible by 2, 3, 4, 8` — Width must be divisible by interleave candidates
- `The input tensor width must be aligned to 64-byte` — 64-byte alignment for some tensor types
- `Error: Static width offset (%ld) must be aligned along %zu bytes` — Offset alignment
- `H * W must be divisible by 8` — For some operations
- `Error: illegal View whose channel offset %zd is not divisible by the interleave factor %zd` — Channel offset alignment

---

## 7. ActiveNE & OCG: The Resource Allocation Constraint System

### 7.1 ActiveNE Formulas

The ANE has multiple Neural Engine units, and the compiler must allocate them:

```
(1 << (activene + wu_stack + fat_tile_enable)) <= hal_params.num_nes
(1 << activene) must be a power of 2
(num_subgroup % (1 << activene)) == 0
(active_ne_times_ocg % logical_intlv) == 0
cout <= (1 << (activene + ocgsize))
cout <= (unicast_cout * (1 << activene))
active_ne_ocg_size % 16 == 0 || relax_active_ne_ocg_constraint
active_ne_ocg_size % 8 == 0 || relax_active_ne_ocg_constraint
```

### 7.2 OCG (Output Channel Group) Constraints

- `Error: invalid OCG size %ld` — OCG must be valid for the architecture
- `ZIN_IR_POW2(ocg_size) <= hal_params.max_ocg_size_in_fill_lower_ne_first_in_bypass_mode` — OCG size limit in bypass mode
- `The number of output channel groups must be 1 with multicast` — Multicast requires single OCG
- `Cout must be a multiple of num_groups in uni_cast mode`

### 7.3 CoutBatchLimiter

The hardware has a CoutBatch limiter register:

```
hw.ne_control_config.ane_ne_config.r.Cfg.f.CoutBatchLimiter == UANE_NE_CONTROL_CFG_COUTBATCHLIMITER_LIMIT_1
hw.ne_control_config.ane_ne_config.r.Cfg.f.CoutBatchLimiter == UANE_NE_CONTROL_CFG_COUTBATCHLIMITER_NOLIMIT
```

`CoutBatch cannot must be between 1 and 16` — The CoutBatch product is bounded.

---

## 8. Dynamic Shape Constraints (The "Kill Switch")

Dynamic shapes create a cascading rejection system:

| Condition | Consequence |
|---|---|
| Any op not ANE-resident | `Dynamic Shapes: One or more network operations are not ANE-resident - Marking all operations as non ANE-resident` — **ALL ops go to host** |
| Conv stride > 2 | Rejected |
| Conv with same/same_lower padding + stride > 1 | Rejected |
| Large kernel conv | Rejected |
| Pool with ceil_mode + stride > 1 | Rejected |
| Pool with same/same_lower padding + stride > 1 | Rejected |
| Global max/min pool | Rejected (use reduction instead) |
| Memory layout operations | Rejected |
| Space-to-channel | Rejected |
| Reshape | Rejected |
| Input depth != 1 | Rejected |
| SPMD procedures | Rejected |
| Random ops | `ANE cannot support dynamic shape random op` |
| Unranked types | `Unranked input types or dynamic shapes are not supported on ANEs` |
| Symbolic shapes on Concat | `Symbolic shape propagation is not supported on Concat` |

**The "all-or-nothing" rule is critical:** If even ONE operation in a dynamic-shape network cannot be ANE-resident, the ENTIRE network is marked as non-ANE-resident. There is no partial ANE execution for dynamic shapes.

---

## 9. Op Ordering Constraints for Fusion

### 9.1 Fusion Atom Ordering (NE Engine)

For NE fused conv, the atom ordering is rigidly defined:

**Input-side atoms** (must precede the primary op):
1. `InputDeQuantAtom` / `DeQuantPreScaleAtom` — Dequantize input
2. `PreScaleAtom` — Pre-scale the input
3. `InputReLUAtom` — Apply input ReLU
4. `TransposeAtom` (CW) — Channel-width transpose
5. `TextureAtom` — Texture format conversion

**Primary op atom** (exactly one):
6. `NEConvAtom` / `ConvGOCAtom` / `ConvQuantAtom` — The convolution
7. Or `MatMulAtom` / `PoolAtom` / etc.

**Output-side atoms** (must follow the primary op):
8. `ActivationAtom` — Post-op activation (ReLU, sigmoid, tanh, etc.)
9. `GOCAtom` / `NEGOCAtom` — Generic operation compute
10. `PostScaleAtom` — Post-scale
11. `EWGOCAtom` / `EWQuantAtom` — Element-wise + GOC/quant fusion
12. `BypassGOCAtom` / `BypassQuantAtom` — Bypass with GOC/quant

This ordering is **not negotiable**. The compiler will not reorder atoms to enable fusion. If an activation comes before the primary op when it should come after, fusion fails.

### 9.2 Fusion Atom Ordering (PE Engine)

**PE Element-Wise ordering:**
1. `InputDeQuantAtom` — Input dequantization
2. `InputReLUAtom` — Input ReLU
3. `InputTransposeAtom` — Input-side transpose
4. `ScaledEWAtom` / `ScaledEWAddAtom` — The scaled element-wise operation
5. `ReductionAtom` / `ReductionEpsilonAtom` / `ReductionFinalScaleAtom` — Optional reduction
6. `PerChannelGOCAtom` / `PerChannelQuantAtom` — Per-channel operations
7. `OutputTransposeAtom` — Output-side transpose
8. `OutputGOCAtom` — Output GOC
9. `OutputReLUAtom` — Output ReLU
10. `PostScaleAtom` — Post-scale

### 9.3 Failed Fusion Patterns

The binary explicitly calls out these impossible fusion combinations:

- `Error: Unable to fuse clamped relu` — Clamped ReLU cannot fuse into NE/PE
- `Error: Unable to fuse leaky relu` — Leaky ReLU cannot fuse
- `Error: Unable to fuse leaky relu or abs` — Neither leaky ReLU nor abs can fuse
- `Error: Unable to fuse swish activation` — Swish cannot fuse in some contexts
- `Error: Unable to fuse swish hard activation` — Hard swish cannot fuse in some contexts
- `Error: Unable to merge activations` — Multiple activations cannot be merged
- `Error: Unable to merge sumsqr` — Sum-of-squares cannot merge
- `Error: Unable to merge with cast` — Cast operations break merge chains
- `Log2: post scale can't be fused into LUT` — Log2 activation blocks post-scale fusion
- `Error: Unable to fuse GOC, GOC and EW_MAX` — Two GOCs + max EW = impossible
- `Error: Unable to fuse Transpose, ScaledEW, Transpose to GOC` — T→SEW→T→GOC rejected

---

## 10. Versioned Hardware Limits

The binary contains per-architecture hardware limit structures:

| Symbol | Target |
|---|---|
| `_target_hw_limits_v4` | ANE v4 (A11?) |
| `_target_hw_limits_v5` | ANE v5 |
| `_target_hw_limits_v6` | ANE v6 |
| `_target_hw_limits_v7` | ANE v7 |
| `_target_hw_limits_v8` | ANE v8 |
| `_target_hw_limits_v10` | ANE v10 |
| `_target_hw_limits_v11` | ANE v11 |
| `_target_hw_limits_v17` | ANE v17 (M1?) |
| `_target_hw_limits_v19` | ANE v19 (M2?) |
| `_target_hw_limits_v20` | ANE v20 |
| `_target_hw_limits_v26` | ANE v26 (M4?) |
| `_target_hw_limits_vu1` | ANE vu1 (uANE?) |

Each version has different values for:
- `max_tensor_channels`, `max_tensor_height`, `max_tensor_width`, `max_tensor_depth`
- `max_ocg_size_in_fill_lower_ne_first_in_bypass_mode`
- `ne_palette_lut_size_in_bytes`
- `pe_reduction_cout_limit`
- `ane_kernel_dma_src_coeff_bfr_size`
- `ew_limit_64`, `ew_limit_128`, `ew_limit_256`
- `cache_prefetch_max_outstanding_requests`

The `ZinUnitValidator::limits<T>()` template function enforces these per-version bounds at validation time, taking a value, min, max, and a CFString error key.

---

## 11. Memory Budget Constraints

Even after passing all validation, an op can still be rejected from ANE due to memory constraints:

| Constraint | Details |
|---|---|
| BSS Limit | `Requested BSS Limit (Bytes): %llu`; `The live IO size exceeds BSS limit!`; `Error: the live io tensor memory footprint (%zd bytes) exceeds the bss limit (%lld bytes)` |
| DRAM Limit | `Requested DRAM Limit (Bytes): %llu`; `Dram required for the largest procedure (proc=%s) is over the limit. (%llu > %llu)` |
| L2 Budget | Spatial splitting triggered when `SIP > budget` |
| Kernel Section Size | `Failed to find a group for the kernel due to too small threshold for kernel section split. (kernel size: %zul, max_kernel_section_size: %d)` |
| Procedure Count | `Error: current proc id exceeds the numerical limit (15 bits). This is software limit` — Max 32768 procedures |
| Object Count | `The next object ID hits the max limit` |
| Tensor Count | `tensor id is over limit` |

The `ZinIrOpLayer::EnforceDimensionsLimits()` method is the last line of defense, checking that all dimensions are within the hardware's capacity after all transformations.

---

## 12. Summary: The Complete Decision Tree for ANE Placement

```
MIL Op
  │
  ├── 1. ValidateLayer<ANEC*LayerDesc>() ─── FAIL → Host placement
  │     ├── Check tensor rank ∈ [0,7]
  │     ├── Check data type ∈ {FP16, FP32, Int8, UInt8, Int4, E4M3*}
  │     ├── Check interleave ∈ {1,2,3,4,8}
  │     ├── Check per-op dimension constraints (see §4)
  │     └── Check format compatibility
  │
  ├── 2. MLIR Placement Dialect ─── FAIL → Host placement
  │     ├── Assign region_type (ANE vs Host)
  │     ├── Check force-ane-placement / force-host-placement
  │     └── Insert ANEIOCast at boundaries
  │
  ├── 3. ZinUnitValidator::limits() ─── FAIL → Host placement
  │     ├── Enforce _target_hw_limits_v* bounds
  │     ├── Check max_tensor_channels/height/width/depth
  │     ├── Check OCG size, ActiveNE constraints
  │     └── Check pe_reduction_cout_limit
  │
  ├── 4. Fusability Checks ─── FAIL → Standalone layer (may still run on ANE)
  │     ├── IsFusableBasedOnFormat()
  │     ├── IsFusableBasedOnFormatOCGSizeAndActiveNE()
  │     ├── GOCAtom::IsFusable()
  │     ├── DeQuantAtom::IsFusableAsGOC()
  │     ├── PreScaleAtom::IsFusable()
  │     ├── ConvAtom::IsFusableToDequant()
  │     ├── ScaledEWAtom::IsFusableEW()
  │     ├── PostScaleAtom::IsFusable()
  │     ├── PerChannelGOCAtom::IsFusable()
  │     ├── InputDeQuantAtom::IsDeQuantFusable()
  │     └── IsQuantFusable()
  │
  ├── 5. Memory Pressure & Legalization ─── FAIL → May split, spill, or reject
  │     ├── L2 cache pressure → spatial splitting
  │     ├── DRAM budget → procedure splitting
  │     ├── BSS limit → live-IO reduction
  │     └── EnforceDimensionsLimits()
  │
  └── SUCCESS → Op runs on ANE
        ├── Assigned to NE, PE, or TransposeEngine
        ├── Fused into engine layer (if fusable)
        └── Scheduled and register-allocated
```

### Key Takeaways for Model Developers

1. **Keep tensors 4D (NCHW preferred)** — 5D tensors severely limit op support (no Linear, no SDPA, restricted Conv)
2. **Batch=1, Depth=1** — Nearly every advanced op requires B=1, D=1
3. **Use FP16 everywhere** — It's the universal ANE format; FP32 inputs are converted, quantized types add constraints
4. **Symmetric quantization only** — Asymmetric quant and blockwise scales are hard rejects
5. **Power-of-2 kernel sizes** — Required for conv; width must be divisible by 2/3/4/8
6. **Avoid dilation + large kernel** — This combination is universally rejected
7. **No stride > 2 with dynamic shapes** — Dynamic shapes already restrict many ops; large strides make it worse
8. **Op ordering matters for fusion** — Place activations AFTER convolutions, not before; the compiler cannot reorder for you
9. **ChannelLast (NHWC) only for depthwise** — Standard convs must use NCHW
10. **Interleave=8 for Int4** — If using 4-bit quantization, interleave must be exactly 8
11. **No strided views with negative strides** — Use copy/reshape instead
12. **The dynamic shape "all-or-nothing" rule** — If any op can't be ANE-resident in a dynamic-shape model, ALL ops fall back to host
