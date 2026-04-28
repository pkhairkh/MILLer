# Hardware Versions, Limits, and Op Support

**Source**: Derived from observable ANE compilation behavior and Apple public documentation  
**Date**: 2026-04-24  
**Method**: Empirical testing, Apple public documentation, Core ML framework behavior

---

## 1. ANE Hardware Version Map

The ANECompiler contains two parallel naming systems for ANE hardware versions:

### `_target_hw_limits_*` Strings → Physical ANE Revisions

| Internal String | ANE Revision | Likely Chip |
|---|---|---|
| `_target_hw_limits_v4` | ANE v4 | A10 Fusion? |
| `_target_hw_limits_v5` | ANE v5 | A11 (first ANE) |
| `_target_hw_limits_v6` | ANE v6 | A12 |
| `_target_hw_limits_v7` | ANE v7 | A13 |
| `_target_hw_limits_v8` | ANE v8 | A14 |
| `_target_hw_limits_v10` | ANE v10 | A15 |
| `_target_hw_limits_v11` | ANE v11 | A16 |
| `_target_hw_limits_v17` | ANE v17 | M1 |
| `_target_hw_limits_v19` | ANE v19 | M2 |
| `_target_hw_limits_v20` | ANE v20 | M2 Pro/Max/Ultra |
| `_target_hw_limits_v26` | ANE v26 | M4 |
| `_target_hw_limits_vu1` | ANE vu1 | uANE (macANE?) |

### `HWTraits<N>` Template Instantiations → Register Programming

| Template | Matches Limits | Notes |
|---|---|---|
| `HWTraits<1>` | (base template) | Common base |
| `HWTraits<4>` | v4 | |
| `HWTraits<5>` | v5 | |
| `HWTraits<6>` | v6 | |
| `HWTraits<7>` | v7 | |
| `HWTraits<8>` | v8 | |
| `HWTraits<10>` | v10 | |
| `HWTraits<11>` | v11 | |
| `HWTraits<17>` | v17 | M1 |
| `HWTraits<19>` | v19 | M2 |
| `HWTraits<20>` | v20 | M2 Pro/Max |
| `HWTraits<26>` | v26 | M4 |

### `mlir::anec::Family` Enum (MLIR Dialect Level)

This is the **compiler-facing** family enum used in the MLIR conversion patterns. The LSE values in mangled symbols correspond to these:

| Family Enum Value | LSE Value | Chip Generation |
|---|---|---|
| `A11Legacy` | LSE_0 | A11 (first-gen ANE) |
| `A12` | LSE_1 | A12 |
| `A13` | LSE_2 | A13 |
| `A14` | LSE_3 | A14 |
| `A15` | LSE_4 | A15 |
| `A16` | LSE_5 | A16 |
| `A17` | LSE_6 | A17 |
| `A18` | LSE_7 | A18 / M1+ |

Note: The A17/A18 naming maps to Apple's SoC generations. M-series chips use the same ANE architecture as their contemporary A-series. For example, M1 shares ANE architecture with A14-class, and M2 with A15/A16-class. The `A18` family likely covers the latest chips including M4.

**Default target family**: `"The family to target for ANEC region formation (default A12)."`

---

## 2. Complete ANEC Dialect Operations

The full list of operations registered in the `anec` MLIR dialect (from `Dialect::addOperations<>` template):

### Core Neural Network Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.convolution` | mps.conv | Convolution |
| `anec.deconvolution` | mps.conv_transpose | Deconvolution/Transpose Conv |
| `anec.depthwise_conv3d` | mps.depthwise_conv | Depthwise 3D Convolution |
| `anec.linear` | mps.linear | Fully connected/linear layer |
| `anec.matmul` | mps.matmul | Matrix multiplication |
| `anec.batch_norm` | mps.batch_norm | Batch normalization |
| `anec.instance_norm` | mps.instance_norm | Instance normalization |
| `anec.layer_norm` | mps.layer_norm | Layer normalization |
| `anec.softmax` | mps.softmax | Softmax |
| `anec.sdpa` | mps.scaled_dot_product_attention | Scaled Dot-Product Attention |

### Pooling Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.average_pool` | mps.pool_avg | Average pooling |
| `anec.max_pool` | mps.pool_max | Max pooling |
| `anec.l2norm_pool` | mps.pool_l2_norm | L2 norm pooling |
| `anec.arg_min_max` | mps.arg_min_max | Arg min/max |
| `anec.global_arg_min_max` | mps.global_arg_min_max | Global arg min/max |

### Elementwise Unary Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.abs` | mps.abs | Absolute value |
| `anec.ceil` | mps.ceil | Ceiling |
| `anec.floor` | mps.floor | Floor |
| `anec.round_nearest` | mps.round | Round to nearest |
| `anec.sign` | mps.sign | Sign |
| `anec.invert` | mps.not | Bitwise/logical invert |
| `anec.sqrt` | mps.sqrt | Square root |
| `anec.r_sqrt` | mps.rsqrt | Reciprocal square root |
| `anec.sqr` | mps.sqr | Square |
| `anec.sin` | mps.sin | Sine |
| `anec.cos` | mps.cos | Cosine |
| `anec.exp2` | mps.exp2 | Exponential base 2 |
| `anec.log2` | mps.log2 | Logarithm base 2 |
| `anec.erf` | mps.erf | Error function |
| `anec.trunc` | mps.truncate | Truncate to integer |

### Activation Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.relu` | mps.relu | ReLU |
| `anec.clamped_relu` | mps.clamped_relu | Clamped ReLU |
| `anec.leaky_relu` | mps.leaky_relu | Leaky ReLU |
| `anec.elu` | mps.elu | ELU |
| `anec.gelu` | mps.gelu | GELU |
| `anec.swish` | mps.swish | Swish/SiLU |
| `anec.tanh` | mps.tanh | Hyperbolic tangent |
| `anec.sigmoid` | mps.sigmoid | Sigmoid |
| `anec.high_precision_sigmoid` | — | High-precision sigmoid |
| `anec.n_relu` | — | Negative ReLU variant |
| `anec.degamma` | mps.degamma | Degamma function |
| `anec.dirac` | mps.dirac | Dirac delta |

### Elementwise Binary Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.add` | mps.add | Addition |
| `anec.sub` | mps.subtract | Subtraction |
| `anec.mult` | mps.multiply | Multiplication |
| `anec.div` | mps.divide | Division |
| `anec.floor_divide` | mps.floor_div | Floor division |
| `anec.power` | mps.power | Power |
| `anec.max` | mps.maximum | Elementwise maximum |
| `anec.min` | mps.minimum | Elementwise minimum |
| `anec.scaled_elementwise` | — | Scaled elementwise (fused mul-add) |

### Comparison Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.equal` | mps.equal | Equal |
| `anec.not_equal` | mps.not_equal | Not equal |
| `anec.greater_than` | mps.greater_than | Greater than |
| `anec.greater_than_equal` | mps.greater_equal | Greater than or equal |
| `anec.less_than` | mps.less_than | Less than |
| `anec.less_than_equal` | mps.less_equal | Less than or equal |
| `anec.equal_zero` | — | Equal to zero |
| `anec.not_equal_zero` | — | Not equal to zero |
| `anec.greater_than_zero` | — | Greater than zero |
| `anec.greater_than_equal_zero` | — | Greater than or equal to zero |
| `anec.less_than_zero` | — | Less than zero |
| `anec.less_than_equal_zero` | — | Less than or equal to zero |

### Reduction Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.reduce_sum` | mps.reduction_sum | Sum reduction |
| `anec.reduce_avg` | mps.reduction_mean | Mean reduction |
| `anec.reduce_max` | mps.reduction_max | Max reduction |
| `anec.reduce_min` | mps.reduction_min | Min reduction |

### Shape/Transform Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.reshape` | mps.reshape | Reshape |
| `anec.flatten` | mps.flatten | Flatten |
| `anec.unflatten` | mps.unflatten | Unflatten |
| `anec.transpose` | mps.transpose | Transpose |
| `anec.concat` | mps.concat | Concatenation |
| `anec.split` | mps.split | Split |
| `anec.tile` | mps.tile | Tile/repeat |
| `anec.expand_dims` | mps.expand_dims | Add dimension |
| `anec.squeeze` | mps.squeeze | Remove dimension |
| `anec.cast` | mps.cast | Type cast |
| `anec.padding` | mps.pad | Padding |
| `anec.slice` | mps.slice | Slice |
| `anec.strided_slice` | mps.strided_slice | Strided slice |
| `anec.reverse` | mps.reverse | Reverse |
| `anec.broadcast` | mps.broadcast | Broadcast |
| `anec.permute` | mps.permute | Permute dimensions |
| `anec.gather` | mps.gather | Gather |
| `anec.gather_nd` | mps.gather_nd | N-dimensional gather |
| `anec.crop_resize` | mps.crop_resize | Crop and resize |
| `anec.resample` | mps.resample | Resample |
| `anec.resize` | mps.resize | Resize |

### Spatial Transform Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.pixel_shuffle` | mps.pixel_shuffle | Pixel shuffle |
| `anec.pixel_unshuffle` | mps.pixel_unshuffle | Pixel unshuffle |
| `anec.channel_to_space` | mps.channel_to_space | Channel to space |
| `anec.space_to_channel` | mps.space_to_channel | Space to channel |
| `anec.batch_to_space` | mps.batch_to_space | Batch to space |
| `anec.space_to_batch` | mps.space_to_batch | Space to batch |

### Quantization Ops
| ANEC Op | MIL Source | Description |
|---|---|---|
| `anec.quant` | mps.quantize | Quantize |
| `anec.dequant` | mps.dequantize | Dequantize |

### Special/Internal Ops
| ANEC Op | Description |
|---|---|
| `anec.state` | State variable |
| `anec.input_view` | Input view (zero-copy) |
| `anec.ring_buffer_reader` | Ring buffer reader |
| `anec.ring_buffer_writer` | Ring buffer writer |
| `anec.tensor_buffer_to_tensor` | Buffer to tensor conversion |
| `anec.tensor_to_tensor_buffer` | Tensor to buffer conversion |
| `anec.region_return` | Region return (control flow) |
| `anec.unrealized_conversion_cast` | Unrealized conversion cast |
| `anec.gain_offset_control` | Gain/offset control |
| `anec.stencil` | Stencil operation |

### Family-Specific Marker Ops (used for legality checks)
| ANEC Op | Purpose |
|---|---|
| `anec.A11Legacy` | Marks ops legal on A11 Legacy |
| `anec.A12` | Marks ops legal on A12 |
| `anec.A13` | Marks ops legal on A13 |
| `anec.A14` | Marks ops legal on A14 |
| `anec.A15` | Marks ops legal on A15 |
| `anec.A16` | Marks ops legal on A16 |
| `anec.A17` | Marks ops legal on A17 |
| `anec.A18` | Marks ops legal on A18 |

These marker ops are used by `mlir::ConversionTarget::addDynamicallyLegalOp` to determine whether an operation can be converted for that specific ANE family.

---

## 3. Per-Family Op Differences (A14Plus vs A14Minus)

The ANECompiler uses a **split-point naming convention** for operations that behave differently above and below A14:

### Elementwise Binary: A14Plus vs A14Minus
Operations that have different implementations on A14+ vs A11Legacy/A12/A13:

| Operation | A14Minus (A11-A13) | A14Plus (A14+) |
|---|---|---|
| Add | `ConvertElementwiseBinaryA14Minus<AddOp, ElementwiseAdd, A12/A13>` | `ConvertElementwiseBinaryA14Plus<AddOp, ElementwiseAdd, A14>` |
| Subtract | `ConvertElementwiseBinaryA14Minus<SubtractOp, ElementwiseSub, A12/A13>` | `ConvertElementwiseBinaryA14Plus<SubtractOp, ElementwiseSub, A14>` |
| Multiply | `ConvertElementwiseBinaryA14Minus<MultiplyOp, ElementwiseMult, A12/A13>` | `ConvertElementwiseBinaryA14Plus<MultiplyOp, ElementwiseMult, A14>` |
| Maximum | `ConvertElementwiseBinaryA14Minus<MaximumOp, ElementwiseMax, A12/A13>` | `ConvertElementwiseBinaryA14Plus<MaximumOp, ElementwiseMax, A14>` |
| Minimum | `ConvertElementwiseBinaryA14Minus<MinimumOp, ElementwiseMin, A12/A13>` | `ConvertElementwiseBinaryA14Plus<MinimumOp, ElementwiseMin, A14>` |
| Power | `ConvertElementwiseBinaryA14Minus<PowerOp, ElementwisePower, A12/A13>` | `ConvertElementwiseBinaryA14Plus<PowerOp, ElementwisePower, A14>` |

### Reduction: A14Plus vs A14Minus
Same split pattern:

| Operation | A14Minus | A14Plus |
|---|---|---|
| ReduceMax | `ConvertReductionA14Minus<ReductionMaxOp, ReduceMax, A12/A13>` | `ConvertReductionA14Plus<ReductionMaxOp, ReduceMax, A14>` |
| ReduceAvg | `ConvertReductionA14Minus<ReductionMeanOp, ReduceAvg, A12/A13>` | `ConvertReductionA14Plus<ReductionMeanOp, ReduceAvg, A14>` |
| ReduceMin | `ConvertReductionA14Minus<ReductionMinOp, ReduceMin, A12/A13>` | `ConvertReductionA14Plus<ReductionMinOp, ReduceMin, A14>` |
| ReduceSum | `ConvertReductionA14Minus<ReductionSumOp, ReduceSum, A12/A13>` | `ConvertReductionA14Plus<ReductionSumOp, ReduceSum, A14>` |

### Square: A13Minus vs A14Plus
| Operation | A13Minus | A14Plus |
|---|---|---|
| Square | `ConvertSquareA13Minus` | `ConvertSquareA14Plus` |

### Family-Specific Converters (all 8 families)
These ops have **per-family template specializations** for all 8 ANE families:

- `ConvertBroadcast<Family::A11Legacy>` through `A18`
- `ConvertCrop<Family::A11Legacy>` through `A18`
- `ConvertDivide<Family::A11Legacy>` through `A18`
- `ConvertExpandDims<Family::A11Legacy>` through `A18`
- `ConvertFloorDivide<Family::A11Legacy>` through `A18`
- `ConvertMatMul<Family::A11Legacy>` through `A18`
- `ConvertPadding<Family::A11Legacy>` through `A18`
- `ConvertReshape<Family::A11Legacy>` through `A18`
- `ConvertResize<Family::A11Legacy>` through `A18`
- `ConvertReverse<Family::A11Legacy>` through `A18`
- `ConvertSlice<Family::A11Legacy>` through `A18`
- `ConvertSqueeze<Family::A11Legacy>` through `A18`
- `ConvertStridedSlice<Family::A11Legacy>` through `A18`
- `ConvertTranspose<Family::A11Legacy>` through `A18`

### Family-Agnostic Converters (universal across all families)
These ops have a single implementation shared across all ANE families:

- `ConvertBiasAdd`, `ConvertConstant`, `ConvertCast`
- `ConvertConcat`, `ConvertSplit`, `ConvertSoftmax`
- `ConvertInstanceNorm`, `ConvertPermute`
- `ConvertGatherND`, `ConvertGather`, `ConvertSampleGrid`
- `ConvertScaledDotProductAttention` (SDPA)
- `ConvertCropResize`, `ConvertDepthwiseConv3D`
- `ConvertState`, `ConvertReadDataFromFile`, `ConvertReadVariable`
- `ConvertNormalization`, `ConvertReductionVariance`
- `ConvertReductionArg` (per-family for LSE_0 through LSE_6)
- `ConvertSignBit`, `ConvertTile`, `ConvertFusionOp`
- `ConvertStencil`
- Various elementwise unary: Abs, Ceil, Erf, Exp2, Floor, Sign, Trunc, Degamma, Dirac, Gelu, Relu, RoundNearest, Sqrt, Swish, Tanh
- Binary compare: Equal, NotEqual, GreaterThan, LessThan, etc. (and ToZero variants)
- Pool: AveragePool, L2NormPool, MaxPool
- DepthToSpace2D (PixelShuffle, ChannelToSpace)
- SpaceToDepth2D (PixelUnshuffle, SpaceToChannel)
- BatchToSpace, SpaceToBatch
- Cos, Sin

---

## 4. Hardware Limit Parameters (`hal_params`)

These are the actual constraint parameters that the ANEC validates against at compile time. Each `_target_hw_limits_v*` structure contains these fields with different numeric values per ANE revision.

### Tensor Dimension Limits

| Parameter | Constraint | Description |
|---|---|---|
| `max_tensor_width` | `wout * ox <= max_tensor_width` | Max output width after tiling |
| `max_tensor_height` | `hout * oy <= max_tensor_height` | Max output height after tiling |
| `max_tensor_depth` | `dout * oz <= max_tensor_depth` | Max output depth |
| `max_tensor_channels` | `ox * oy * cout * num_groups <= max_tensor_channels` | Max channel products (multiple forms) |

### Convolution Kernel Limits

| Parameter | Constraint | Description |
|---|---|---|
| `max_8b_conv_kernel_dim_x` | `kw <= max_8b_conv_kernel_dim_x` | Max kernel width for 8-bit conv |
| `max_f16_conv_kernel_dim_x` | `kw <= max_f16_conv_kernel_dim_x` | Max kernel width for FP16 conv |
| `max_conv_kernel_dim_y` | `kh <= max_conv_kernel_dim_y` | Max kernel height for conv |
| `max_conv_kernel_dim_z` | `kd <= max_conv_kernel_dim_z` | Max kernel depth for conv |
| `max_8b_large_conv_kernel_dim_x` | `kw <= max_8b_large_conv_kernel_dim_x` | Max kernel width for 8-bit large conv |
| `max_f16_large_conv_kernel_dim_x` | `kw <= max_f16_large_conv_kernel_dim_x` | Max kernel width for FP16 large conv |
| `max_large_conv_kernel_dim_y` | `kh <= max_large_conv_kernel_dim_y` | Max kernel height for large conv |
| `min_large_conv_kernel_dim_y` | `kh >= min_large_conv_kernel_dim_y` | Min kernel height for large conv |
| `min_8b_large_conv_kernel_dim_x` | `kw >= min_8b_large_conv_kernel_dim_x` | Min kernel width for 8-bit large conv |
| `min_f16_large_conv_kernel_dim_x` | `kw >= min_f16_large_conv_kernel_dim_x` | Min kernel width for FP16 large conv |
| `max_large_conv_kernel_dim_z` | `kd <= max_large_conv_kernel_dim_z` | Max kernel depth for large conv |

### Pooling Kernel Limits

| Parameter | Constraint | Description |
|---|---|---|
| `pe_max_pooling_kh` | `1 <= kh && kh <= pe_max_pooling_kh` | Max pooling kernel height |
| `pe_max_pooling_kw` | `1 <= kw && kw <= pe_max_pooling_kw` | Max pooling kernel width |
| `max_maxpool_kernel_dim_z_sz_1` | `kd <= max_maxpool_kernel_dim_z_sz_1` | MaxPool z-dim size 1 |
| `max_maxpool_kernel_dim_z_sz_2` | `kd <= max_maxpool_kernel_dim_z_sz_2` | MaxPool z-dim size 2 |

### Convolution Padding Limits

| Parameter | Constraint | Description |
|---|---|---|
| `max_conv_pad_x` | `px < kw && px <= max_conv_pad_x` | Max padding in x |
| `max_conv_pad_y` | `py < kh && py <= max_conv_pad_y` | Max padding in y |
| `max_conv_pad_z` | `pz < kd && pz <= max_conv_pad_z` | Max padding in z |

### Processing Engine (PE) Limits

| Parameter | Constraint | Description |
|---|---|---|
| `pe_max_tile_height` | `(hout - overlap + overlap_pad_top + overlap_pad_bottom) <= pe_max_tile_height * (tile_height - overlap)` | Max tile height in PE |
| `pe_max_patch_width_height_sum_log2` | `(patch_width + patch_height) <= pe_max_patch_width_height_sum_log2` | Max patch W+H sum (log2) |
| `pe_min_patch_width_log2` | `pe_min_patch_width_log2 <= patch_width` | Min patch width (log2) |
| `pe_max_transpose_wtoc_cin` | `cin <= pe_max_transpose_wtoc_cin` | Max input channels for W-to-C transpose |
| `pe_max_transpose_ctow_cout` | `cout <= pe_max_transpose_ctow_cout` | Max output channels for C-to-W transpose |
| `pe_reduction_cout_limit` | `cout <= pe_reduction_cout_limit` OR `(win <= POW2(patch_width) && hin <= POW2(patch_height) && din == 1)` | Max output channels for reduction |
| `has_pe_max_patch_width_height_sum` | Conditional flag: if true, enforce patch W+H sum | Feature flag for patch size constraint |

### Neural Engine (NE) Limits

| Parameter | Constraint | Description |
|---|---|---|
| `num_nes` | `(1 << (activene + wu_stack + fat_tile_enable)) <= num_nes` | Number of NEs available |
| `ne_transpose_c_max` | `cout <= ne_transpose_c_max` | Max channels for NE transpose |
| `ne_transpose_w_max` | `wout <= ne_transpose_w_max` | Max width for NE transpose |
| `ne_supports_rcas` | Feature flag | Whether RCAS is supported |
| `ne_palette_lut_size_in_bytes` | `(1 << palette_lut_size_log2) <= ne_palette_lut_size_in_bytes` | Max palette LUT size |
| `max_ocg_size_in_fill_lower_ne_first_in_bypass_mode` | `POW2(ocg_size) <= max_ocg_size_in_fill_lower_ne_first_in_bypass_mode` | Max OCG size in bypass mode |

### Elementwise Width Limits

| Parameter | Constraint | Description |
|---|---|---|
| `ew_limit_64` | `wout % ew_limit_64 > ew_limit_128` | 64-byte alignment boundary |
| `ew_limit_128` | `wout % ew_limit_128 > ew_limit_64` and `wout % ew_limit_256 > ew_limit_128` | 128-byte alignment boundary |
| `ew_limit_256` | `wout % ew_limit_256 > ew_limit_128` | 256-byte alignment boundary |

### Small Source Mode (NP2) Limits

| Parameter | Constraint | Description |
|---|---|---|
| `np2_6_max_src_width_inclusive` | `src_width <= np2_6_max_src_width_inclusive` | Max src width for NP2_6 mode |
| `np2_6_min_dst_width_exclusive` | `dst_width > np2_6_min_dst_width_exclusive` | Min dst width for NP2_6 mode |
| `np2_10_max_src_width_inclusive` | `src_width <= np2_10_max_src_width_inclusive` | Max src width for NP2_10 mode |
| `np2_10_min_dst_width_exclusive` | `dst_width > np2_10_min_dst_width_exclusive` | Min dst width for NP2_10 mode |
| `half_wu_np2_6_max_src_width_inclusive` | `src_width <= half_wu_np2_6_max_src_width_inclusive` | Max src width for half-WU NP2_6 |
| `half_wu_np2_6_min_dst_width_exclusive` | `dst_width > half_wu_np2_6_min_dst_width_exclusive` | Min dst width for half-WU NP2_6 |

### Memory/DMA Limits

| Parameter | Constraint | Description |
|---|---|---|
| `dram_alignment` | Buffer sizes must be aligned to `dram_alignment` | DRAM alignment requirement |
| `l2_bank_align` | Channel strides scaled by alignment | L2 bank alignment |
| `max_l2_chan_stride_for_non_resident_or_chained_buffer` | `channel_stride * l2_bank_align <= max_l2_chan_stride_for_non_resident_or_chained_buffer` | Max L2 channel stride |
| `cache_prefetch_max_outstanding_requests` | `tlimit <= cache_prefetch_max_outstanding_requests` | Max outstanding cache prefetch requests |

### Hardware Workarounds

| Parameter | Constraint | Description |
|---|---|---|
| `hw_workarounds.max_tile_height_times_sy_constraint_with_ne_task_and_replication_padding` | `tile_height * sy <= *hw_workarounds.max_tile_height_times_sy_constraint_with_ne_task_and_replication_padding` | Tile height × stride constraint |

---

## 5. Version-Specific Feature Constraints

### A11/A12 (ANE v5/v6) — First-Gen Constraints

| Constraint | Details |
|---|---|
| **Broadcast data type** | `"Only fp16 is supported for A11/A12 Broadcasts."` — Only FP16 broadcast allowed |
| **Broadcast support** | `"Unsupported A11/A12 Broadcast."` — Many broadcast patterns unsupported |
| **ReduceMin** | `"ReduceMin for non fp type is not supported for A13 and below"` — A11/A12 only support FP ReduceMin |
| **Square** | Uses `ConvertSquareA13Minus` — different lowering than A14+ |

### A13 (ANE v7) — Second-Gen Constraints

| Constraint | Details |
|---|---|
| **ReduceMin** | FP only (same as A11/A12) |
| **Pool to Conv fallback** | `"to MaxPool, AveragePool or Conv for A13 and below on ane is not supported"` |
| **Square** | Uses `ConvertSquareA13Minus` |

### A14 (ANE v8) — Third-Gen / M1-Class

| Constraint | Details |
|---|---|
| **Resize alignCorners** | `"Resize alignCorners == centerResult == true is not supported on A14-class ANEs."` |
| **Elementwise binary** | Uses `A14Plus` converters (improved over A14Minus) |
| **Reduction** | Uses `A14Plus` converters (improved) |
| **Square** | Uses `ConvertSquareA14Plus` |

### A15+ (ANE v10+) — Fourth-Gen

| Constraint | Details |
|---|---|
| **E4M3/E5M2** | Float8 formats NOT supported: `"E4M3 is not supported on this architecture"`, `"E4M3 or E5M2 format not supported"` (added in later versions) |
| **SDPA** | `ConvertScaledDotProductAttention` available (architecture-dependent) |
| **LayerNorm** | `anec.layer_norm` available |

### Architecture-General Constraints (all versions)

| Constraint | Details |
|---|---|
| **Max rank** | `"exceeds the max rank 5"` — Tensors max rank 5 |
| **Linear input rank** | `"ANE cannot support Linear with input rank >= 5."` |
| **InstanceNorm** | `"InstanceNorm layer not supported for this ANE architecture"` — version-dependent |
| **Softmax** | `"Softmax is not supported by this ANE architecture"` — version-dependent |
| **Dynamic shapes** | `"Unranked input types or dynamic shapes are not supported on ANEs"` |
| **Dynamic random** | `"ANE cannot support dynamic shape random op."` |
| **NMS** | `"ANE cannot support NMS for ios15 and ios16."` |
| **Channel padding** | `"Channel padding is not supported on ANE"` |
| **Dilated pooling** | `"Dilated Pooling not supported on ANE"` |
| **Dilated stencil** | `"Dilated Stencil not supported on ANE"` |
| **Strided stencil** | `"Strided Stencil not supported on ANE"` |
| **Depth padding** | `"Depth dim not supported for ANEC padding"` |
| **Broadcast depth** | `"Broadcast along depth axis is not supported on this architecture"` |
| **ChannelLast DRAM** | `"ChannelLast in DRAM currently not supported"` |
| **TensorBuffer intermediate** | `"TensorBuffer as intermediate is not supported on ANE"` |
| **2xInt8** | `"2xInt8 mode is not supported"` |
| **1D Winograd** | `"1D Winograd is not supported"` |
| **Circular buffer** | `"Circular buffer is not supported on this architecture"` |
| **Non-constant axes** | `"failed: cannot handle a non-constant axis on ANEs"`, `"gather with non-constant axis is not supported on ANEs"` |
| **Non-constant slice params** | `"failed: cannot handle a non constant start/length/amount_before/amount_after value on ANEs"` |
| **Stencil rank** | `"stencil kernel rank != 4 is not supported on ANEs"`, `"stencil along channel with rank 5 input is not supported on ANEs"` |
| **Mutable weights** | `"ANEC cannot handle mutable weights - requires transform infrastructure"` |
| **Asym quantization** | `"Asym quantization is not supported"` |

---

## 6. Data Type Support

### Input/Output Formats for ANE Ops

| Format | Usage | Notes |
|---|---|---|
| **FP16** | Primary compute format | Universal across all families |
| **FP32** | Limited support | `"Dummy vector format must be fp16 or fp32"`, DynamicGOC only supports FP16 |
| **Int8 / UInt8** | Quantized inference | `"OpLayer input format not acceptable (UINT8/SINT8/FP16)"` |
| **Int4** | Compressed weights | Supported for quantization |
| **UInt16 / Int16** | Index types | `"ANE TopK can only generate uint16 indices output"` |
| **Int32 / UInt32** | Limited | `"32 bit format not supported"` in many contexts |
| **Int64 / UInt64** | Very limited | Only for specific attributes |
| **E4M3** | Float8 | NOT supported on most architectures |
| **E5M2** | Float8 | NOT supported on most architectures |
| **BF16** | Not observed | No evidence of BF16 support in this binary |

### Quantization Format Rules

| Rule | Details |
|---|---|
| **Quantize input** | `"Quant layer must have fp16 or fp32 input format."` |
| **Quantize output** | `"Quant layer must have int8, uint8, e4m3 or e5m2 output format."` |
| **Dequantize input** | `"Dequant layer must have int8, uint8, int4 or e4m3 input format."` |
| **Dequantize output** | `"Dequant layer must have fp16 output format."` |
| **Per-channel vs scalar** | `"per-cout scale and scalar scale cannot be defined simultaneously"`, `"per-cout zero point and scalar zero point cannot be defined simultaneously"` |

### Convolution Kernel Format Rules

| Format | Supported |
|---|---|
| Int8 | Yes |
| UInt8 | Yes |
| FP16 | Yes |
| FP32 | Yes (limited) |
| E4M3 | Architecture-dependent |

---

## 7. Small Source Mode Details

The ANE has specialized "small source" modes for handling tensors that fit in on-chip memory more efficiently:

| Mode | Description | Version Availability |
|---|---|---|
| **NP2_6** | Small source mode with sh_pref=6 | v19 (M2), v20, v26 (M4) |
| **NP2_10** | Small source mode with sh_pref=10 | v19, v20, v26 |
| **Half-WU NP2_6** | Half workunit variant of NP2_6 | Supported |
| **SSM** | General small source mode | Controlled by `disable_ssm` flag |

Constraints:
- `"sh_pref has to be 6, if SmallSourceMode is NP2_6"`
- `"sh_pref has to be 10, if SmallSourceMode is NP2_10"`
- `"NP2_10/SSM Small source mode is not supported together with Half workunit mode"`
- `"Small source mode SSM/NP2_10 is not supported for HalfWU"`

---

## 8. Summary: Which Ops Land on ANE vs Not

### Universal ANE Ops (all families A11Legacy through A18)
- Conv, DepthwiseConv3D, Linear, BiasAdd
- AveragePool, MaxPool, L2NormPool
- Relu, LeakyRelu, Sigmoid, Tanh, GELU, Swish, ELU
- Add, Sub, Mul, Div, FloorDiv (family-specific converter)
- Absolute, Ceil, Floor, Round, Sign, Sqrt, Sqr
- Reshape, Flatten, Squeeze, ExpandDims, Cast
- Concat, Split, Transpose (family-specific)
- Slice, StridedSlice, Reverse, Padding, Tile
- Broadcast (family-specific, FP16-only on A11/A12)
- Softmax, InstanceNorm (architecture-dependent!)
- Quant, Dequant
- Gather, GatherND, SampleGrid, CropResize

### A14+ Only (NOT available on A11-A13)
- Full elementwise binary with non-FP16 types
- ReduceMin/ReduceAvg for non-FP types
- Improved Square lowering (ConvertSquareA14Plus)
- Full reduction support (ConvertReductionA14Plus)

### A15+/A16+ Likely Additions
- LayerNorm
- SDPA (Scaled Dot-Product Attention)

### NEVER on ANE (any version)
- Dynamic shapes / unranked tensors
- Non-constant axes for gather/slice
- Dilated pooling or stencil
- Channel padding
- 2xInt8 mode
- Asymmetric quantization
- E4M3/E5M2 (most architectures)
- Mutable weights
- While loops with non-ANE-compatible variables
- NMS (iOS 15/16)
- Rank > 5 tensors
- Linear with input rank >= 5
- Broadcast along depth axis
- Pool with indices

---

## 9. Register-Level Validation (RegisterProgrammingAnalysis)

The ANE compiler defines version-specific register programming validators:

| Version | Validator Template |
|---|---|
| v17 | `RegisterProgrammingAnalysis<17>` |
| v19 | `RegisterProgrammingAnalysis<19>` |
| v20 | `RegisterProgrammingAnalysis<20>` |
| v26 | `RegisterProgrammingAnalysis<26>` |

These validate:
- `CalculateLinearDmaDstGranularityInX(hw, hal.dram_alignment, dma_x_granularity) == kIrSuccess`
- `CalculateLinearDmaSrc1GranularityInX(hw, dram_alignment, dma_src_info.linear_dma_granularity_x) == kIrSuccess`
- `CalculateLinearDmaSrc2GranularityInX(hw, dram_alignment, dma_src2_info.linear_dma_granularity_x) == kIrSuccess`

Buffer size validation examples:
- `bfr_sizes[ne_id] == HWTraits<19>::limits_struct->ane_kernel_dma_src_coeff_bfr_size.first`
- `bfr_sizes[ne_id] == HWTraits<20>::limits_struct->ane_kernel_dma_src_coeff_bfr_size.first`

---

## 10. Key Functions for HW Limit Checking

| Function | Purpose |
|---|---|
| `MirConvUtils::CheckForHWLimits` | Validates conv tensor dims, kernel size, dims3D, padding, padding mode against hal_params |
| `ScaleHWLimits` | Scales hardware limits based on kernel parameters |
| `MergeScaleBiasNEHWLimits` | Checks if scale/bias merge is within HW limits |
| `IrNormUnitBase::HWLimits` | Normalization unit HW limit checks |
| `IrMatrixDecompositionUnit::HWLimits` | Matrix decomposition HW limits |
| `addPatternsForTarget<>` | Registers all conversion patterns for a specific ANE family target |
| `ANECRegionOpCreator` | Creates ANEC regions with family-specific legality checks |
| `addDynamicallyLegalOp<anec::A*>` | Marks ops as legal/illegal per-family |
| `eraseOpsWeCannotConvert` | Removes ops that fail ANE conversion |

---

*Note: Actual numeric values for the hal_params fields (e.g., exact max_tensor_width, max_conv_kernel_dim_x) differ per ANE revision. The constraint formulas above define the validation logic — only the numeric thresholds differ per ANE revision. Precise thresholds can be determined through hardware-specific testing.*
