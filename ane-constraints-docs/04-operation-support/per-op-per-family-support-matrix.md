# Per-Op Per-Family Support Matrix

**Source**: Derived from observable ANE compilation behavior and Apple public documentation
**Date**: 2026-04-24
**Method**: Empirical testing, Apple public documentation, Core ML framework behavior

---

## Part 1: ANE Hardware Version Map

## 1.1 Three Parallel Naming Systems

The ANECompiler uses three distinct naming systems that map to each other:

### `_target_hw_limits` Strings → Physical ANE Silicon Revisions

| Internal String | ANE Rev | Likely Chip | HWTraits |
|---|---|---|---|
| `_target_hw_limits_v4` | ANE v4 | A10 Fusion (pre-ANE?) | `HWTraits<4>` |
| `_target_hw_limits_v5` | ANE v5 | A11 (first ANE) | `HWTraits<5>` |
| `_target_hw_limits_v6` | ANE v6 | A12 | `HWTraits<6>` |
| `_target_hw_limits_v7` | ANE v7 | A13 | `HWTraits<7>` |
| `_target_hw_limits_v8` | ANE v8 | A14 | `HWTraits<8>` |
| `_target_hw_limits_v10` | ANE v10 | A15 | `HWTraits<10>` |
| `_target_hw_limits_v11` | ANE v11 | A16 | `HWTraits<11>` |
| `_target_hw_limits_v17` | ANE v17 | M1 | `HWTraits<17>` |
| `_target_hw_limits_v19` | ANE v19 | M2 | `HWTraits<19>` |
| `_target_hw_limits_v20` | ANE v20 | M2 Pro/Max/Ultra | `HWTraits<20>` |
| `_target_hw_limits_v26` | ANE v26 | M4 | `HWTraits<26>` |
| `_target_hw_limits_vu1` | ANE vu1 | uANE (micro ANE) | — |

Also: `HWTraits<1>` (base template) — shared infrastructure across all versions.

### `mlir::anec::Family` Enum → MLIR Compiler-Facing Families

This is the enum used in the MLIR conversion patterns. The LSE values in mangled symbols correspond to these:

| Family | LSE Value | Chip Generation | hw_limits |
|---|---|---|---|
| `A11Legacy` | LSE_0 | A11 (first-gen ANE) | v5 |
| `A12` | LSE_1 | A12 | v6 |
| `A13` | LSE_2 | A13 | v7 |
| `A14` | LSE_3 | A14 / M1-class | v8/v17 |
| `A15` | LSE_4 | A15 | v10 |
| `A16` | LSE_5 | A16 | v11 |
| `A17` | LSE_6 | A17 | — |
| `A18` | LSE_7 | A18 / M4-class | v26 |

**Default target family**: `"The family to target for ANEC region formation (default A12)."`

**Critical note**: `"MIL is only supported for H13+ ANE architectures."` — This means A11Legacy (H11) may not support the full MIL pipeline; A12 (H12) is the minimum for MIL-based compilation.

---

## Part 2: Complete MIL Op to ANEC Op Mapping

## 2.1 Ops That Land on ANE (Have Converters)

### Core Neural Network Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 1 | `mps.conv_` | `anec.convolution` | NE conv path | All families | NE | Kernel W/H/D must be power of 2; grouped+large kernel NOT supported; dilated+large kernel NOT supported; stride must be 1 for batch/channel axis; dilation must be 1 for batch/channel axis |
| 2 | `mps.depthwise_conv_` | `anec.convolution` (depthwise) | `ConvertDepthwiseConv3D` | All families | NE | num_groups == out_channels; in_channels == out_channels; depthwise 3D variant exists |
| 3 | `mps.matmul` | `anec.matmul` | `ConvertMatMul<LSE_0..7>` | Per-family | NE | depth must be 1 for both inputs; output channel must equal input A's channel; A.width + padding must equal B.channel; num output channels must be multiple of ox |
| 4 | `mps.bias_add` | (fused into anec.add) | `ConvertBiasAdd` | All families | NE | Fused with conv/linear; not standalone |
| 5 | `mps.softmax` | `anec.softmax` | `ConvertSoftmax` | All families* | PE | Decomposed: max→sub→exp2→sum→mul; "Softmax is not supported by this ANE architecture" on some versions; output must be Float |
| 6 | `mps.instance_norm` | `anec.instance_norm` | `ConvertInstanceNorm` | All families* | PE | "InstanceNorm layer not supported for this ANE architecture" on some; output must be Float; InvalidUnitInstanceNormDimension |
| 7 | `mps.normalization` | `anec.layer_norm` / `anec.batch_norm` | `ConvertNormalization` | All families | PE | LayerNorm: channels must be divisible by num_groups, output must be Float; BatchNorm: single input, must have kernel, output must be Float |

### Pooling Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 8 | `mps.pooling_average` | `anec.average_pool` | `ConvertPool<PoolAvgOp, AveragePool>` | All families | NE | Stride 3 only for avg pool; dilated pooling NOT supported; width/height must be multiple of stride; exclude padding must be < kernel size |
| 9 | `mps.pooling_max` | `anec.max_pool` | `ConvertPool<PoolMaxOp, MaxPool>` | All families | NE | Padding mode must be Negative; same stride constraints; dilated pooling NOT supported |
| 10 | `mps.pooling_l` | `anec.l2norm_pool` | `ConvertPool<PoolL2NormOp, L2NormPool>` | All families | NE | L2 norm does not support batch axis; padding mode must be Replication or Zero |

### Elementwise Unary Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 11 | `mps.absolute` | `anec.abs` | `ConvertElementwiseUnary<AbsoluteOp, ElementwiseAbs>` | All families | PE | — |
| 12 | `mps.ceil` | `anec.ceil` | `ConvertElementwiseUnary<CeilOp, Ceil>` | All families | PE | — |
| 13 | `mps.floor` | `anec.floor` | `ConvertElementwiseUnary<FloorOp, Floor>` | All families | PE | — |
| 14 | `mps.round` | `anec.round_nearest` | `ConvertElementwiseUnary<RoundOp, RoundNearest>` | All families | PE | — |
| 15 | `mps.sign` | `anec.sign` | `ConvertElementwiseUnary<SignOp, Sign>` | All families | PE | — |
| 16 | `mps.truncate` | `anec.trunc` | `ConvertElementwiseUnary<TruncateOp, Trunc>` | All families | PE | — |
| 17 | `mps.square_root` | `anec.sqrt` | `ConvertElementwiseUnary<SquareRootOp, Sqrt>` | All families | PE | — |
| 18 | `mps.reciprocal_square_root` | `anec.r_sqrt` | (via ElementwiseUnary) | All families | PE | — |
| 19 | `mps.square` | `anec.sqr` | `ConvertSquareA13Minus` / `ConvertSquareA14Plus` | A11-A13 / A14+ | PE | Different lowering per generation |
| 20 | `mps.sin` | `anec.sin` | `ConvertElementwiseUnary<SinOp, Sin>` | All families | PE | — |
| 21 | `mps.cos` | `anec.cos` | `ConvertElementwiseUnary<CosOp, Cos>` | All families | PE | — |
| 22 | `mps.exponent_base_2` | `anec.exp2` | `ConvertElementwiseUnary<ExponentBase2Op, Exp2>` | All families | PE | — |
| 23 | `mps.exponent` | (exp) | `ConvertExponent` | All families | PE | — |
| 24 | `mps.logarithm` | (log) | `ConvertLogarithm` | All families | PE | "Log2: post scale can't be fused into LUT" |
| 25 | `mps.erf` | `anec.erf` | `ConvertElementwiseUnary<ErfOp, Erf>` | All families | PE | — |
| 26 | `mps.degamma` | `anec.degamma` | `ConvertElementwiseUnary<DegammaOp, Degamma>` | All families | PE | — |
| 27 | `mps.dirac` | `anec.dirac` | `ConvertElementwiseUnary<DiracOp, Dirac>` | All families | PE | — |

### Activation Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 28 | `mps.relu` | `anec.relu` | `ConvertElementwiseUnary<ReluOp, Relu>` | All families | PE/NE | Can fuse as InputReLU or output activation; "NEElementWise can only have input activation mode as Relu" |
| 29 | `mps.leaky_relu` | `anec.leaky_relu` | (via activation) | All families | PE | "Error: Unable to fuse leaky relu" in some fusion contexts |
| 30 | `mps.gelu` | `anec.gelu` | `ConvertElementwiseUnary<GeluOp, Gelu>` | All families | PE | — |
| 31 | `mps.swish` | `anec.swish` | `ConvertElementwiseUnary<SwishOp, Swish>` | All families | PE | SwishHardActivationDetection pattern; "Error: Unable to fuse swish activation" in some contexts |
| 32 | `mps.tanh` | `anec.tanh` | `ConvertElementwiseUnary<TanhOp, Tanh>` | All families | PE | — |
| 33 | `mps.sigmoid` | `anec.sigmoid` | (via activation) | All families | PE | High-precision variant: `anec.high_precision_sigmoid`; "useRegularSigmoid" flag |
| 34 | `mps.elu` | `anec.elu` | (via activation) | All families | PE | — |
| 35 | `mps.n_relu` | `anec.n_relu` | (via activation) | All families | PE | Negative ReLU variant |

### Elementwise Binary Ops (Version-Split at A14)

| # | MIL Op | ANEC Dialect Op | Converter (A14+) | Converter (A14Minus) | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|---|
| 36 | `mps.add` | `anec.add` | `ConvertElementwiseBinaryA14Plus<AddOp, ..., A14>` | `ConvertElementwiseBinaryA14Minus<AddOp, ..., A12/A13>` | All | PE/NE | FP16-only broadcast on A11/A12 |
| 37 | `mps.subtract` | `anec.sub` | `...A14Plus<SubtractOp, ElementwiseSub, A14>` | `...A14Minus<SubtractOp, ..., A12/A13>` | All | PE/NE | Same FP16 limitation on old hw |
| 38 | `mps.multiply` | `anec.mult` | `...A14Plus<MultiplyOp, ElementwiseMult, A14>` | `...A14Minus<MultiplyOp, ..., A12/A13>` | All | PE/NE | Same FP16 limitation |
| 39 | `mps.maximum` | `anec.max` | `...A14Plus<MaximumOp, ElementwiseMax, A14>` | `...A14Minus<MaximumOp, ..., A12/A13>` | All | PE/NE | Same FP16 limitation |
| 40 | `mps.minimum` | `anec.min` | `...A14Plus<MinimumOp, ElementwiseMin, A14>` | `...A14Minus<MinimumOp, ..., A12/A13>` | All | PE/NE | Same FP16 limitation |
| 41 | `mps.power` | `anec.power` | `...A14Plus<PowerOp, ElementwisePower, A14>` | `...A14Minus<PowerOp, ..., A12/A13>` | All | PE/NE | Same FP16 limitation |
| 42 | `mps.divide` | `anec.div` | `ConvertDivide<LSE_3..7>` | `ConvertDivide<LSE_0..2>` | Per-family | PE | Family-specific implementation |
| 43 | `mps.floor_divide` | (floor div) | `ConvertFloorDivide<LSE_3..7>` | `ConvertFloorDivide<LSE_0..2>` | Per-family | PE | Family-specific implementation |

### Comparison Ops (Family-Agnostic)

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine |
|---|---|---|---|---|---|
| 44 | `mps.equal` | `anec.equal` | `ConvertBinaryCompare<EqualToOp, ElementwiseEqual>` | All families | PE |
| 45 | `mps.not_equal` | `anec.not_equal` | `ConvertBinaryCompare<NotEqualToOp, ElementwiseNotEqual>` | All families | PE |
| 46 | `mps.greater` | `anec.greater_than` | `ConvertBinaryCompare<GreaterThanOp, ElementwiseGreaterThan>` | All families | PE |
| 47 | `mps.greater_equal` | `anec.greater_than_equal` | `ConvertBinaryCompare<GreaterEqualToOp, ElementwiseGreaterThanEqual>` | All families | PE |
| 48 | `mps.less` | `anec.less_than` | `ConvertBinaryCompare<LessThanOp, ElementwiseLessThan>` | All families | PE |
| 49 | `mps.less_equal` | `anec.less_than_equal` | `ConvertBinaryCompare<LessEqualToOp, ElementwiseLessThanEqual>` | All families | PE |
| 50 | `mps.not` (compare to zero) | `anec.not_equal_zero` etc. | `ConvertBinaryCompareToZero` variants | All families | PE |

### Reduction Ops (Version-Split at A14)

| # | MIL Op | ANEC Dialect Op | Converter (A14+) | Converter (A14Minus) | Family | Engine | Key Constraints |
|---|---|---|---|---|---|---|---|
| 51 | `mps.reduction_max` | `anec.reduce_max` | `ConvertReductionA14Plus<..., A14>` | `ConvertReductionA14Minus<..., A12/A13>` | All | PE | — |
| 52 | `mps.reduction_mean` | `anec.reduce_avg` | `...A14Plus<ReductionMeanOp, ReduceAvg, A14>` | `...A14Minus<..., A12/A13>` | All | PE | — |
| 53 | `mps.reduction_min` | `anec.reduce_min` | `...A14Plus<ReductionMinOp, ReduceMin, A14>` | `...A14Minus<..., A12/A13>` | All | PE | FP-only on A13 and below |
| 54 | `mps.reduction_sum` | `anec.reduce_sum` | `...A14Plus<ReductionSumOp, ReduceSum, A14>` | `...A14Minus<..., A12/A13>` | All | PE | — |
| 55 | `mps.reduction_argmax` | `anec.arg_min_max` | `ConvertReductionArg<ReductionArgMaxOp, LSE_0..6>` | Per-family (LSE_0..6) | LSE_0-6 | PE | No LSE_7 (A18) variant |
| 56 | `mps.reduction_argmin` | `anec.arg_min_max` | `ConvertReductionArg<ReductionArgMinOp, LSE_0..6>` | Per-family (LSE_0..6) | LSE_0-6 | PE | No LSE_7 (A18) variant |
| 57 | `mps.reduction_variance` | (variance) | `ConvertReductionVariance` | All families | PE | — |

### Shape/Transform Ops (Per-Family Converters)

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 58 | `mps.reshape` | `anec.reshape` | `ConvertReshape<LSE_0..7>` | Per-family | PE/TE | Not supported with dynamic shapes; output dims must match element count |
| 59 | `mps.expand_dims` | `anec.expand_dims` | `ConvertExpandDims<LSE_0..7>` | Per-family | PE | — |
| 60 | `mps.squeeze` | (squeeze) | `ConvertSqueeze<LSE_0..7>` | Per-family | PE | — |
| 61 | `mps.transpose` | `anec.transpose` | `ConvertTranspose<LSE_0..7>` | Per-family | TE | Interleave only on C axis; TransposeNC requires C=1 |
| 62 | `mps.permute` | `anec.permute` | `ConvertPermute` | All families | TE | Dimension permutation |
| 63 | `mps.concat` | `anec.concat` | `ConvertConcat` | All families | PE | Concat dimension constraints; unsupported concatenation dimension |
| 64 | `mps.split` | `anec.split` | `ConvertSplit` | All families | PE | Split dimension constraints |
| 65 | `mps.tile` | `anec.tile` | `ConvertTile` | All families | PE | — |
| 66 | `mps.broadcast_to` | `anec.broadcast` | `ConvertBroadcast<LSE_0..7>` | Per-family | PE | FP16-only on A11/A12; depth axis broadcast NOT supported; output dims must be 1 or match |
| 67 | `mps.cast` | `anec.cast` | `ConvertCast` | All families | PE | Type conversion; limited format support |
| 68 | `mps.flatten_` | `anec.flatten` | `ConvertFlatten2D` | All families* | PE | "Flatten is not supported on this architecture" on some; must be NxCx1x1 for Unflatten path |
| 69 | `mps.select` | `anec.scaled_elementwise` | `ConvertSelect` | All families | PE | — |

### Slice/Pad/Crop Ops (Per-Family Converters)

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 70 | `mps.pad` | `anec.padding` | `ConvertPadding<LSE_0..7>` | Per-family | PE | No replication/symmetric/negative padding; no batch/channel padding; padding mode must be same on all axes |
| 71 | `mps.slice` | `anec.slice` | `ConvertSlice<LSE_0..7>` | Per-family | PE | Non-constant start/length NOT supported; only constant axes supported |
| 72 | `mps.strided_slice` | `anec.strided_slice` | `ConvertStridedSlice<LSE_0..7>` | Per-family | PE | Stride must be 1; non-constant values NOT supported |
| 73 | `mps.reverse` | `anec.reverse` | `ConvertReverse<LSE_0..7>` | Per-family | PE | — |
| 74 | `mps.crop` | `anec.crop` | `ConvertCrop<LSE_0..7>` | Per-family | NE | Cropping on batch/depth/channel NOT supported |

### Resize/Resample Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 75 | `mps.resize` | `anec.resize` | `ConvertResize<LSE_0..7>` | Per-family | NE | alignCorners+centerResult NOT supported on A14-class; resized_dims ≤ 2 for ios17; custom scale+offset NOT supported on ANE; FactorX should be multiple of 2 or 3 |
| 76 | `mps.crop_resize` | `anec.crop_resize` | `ConvertCropResize` | All families | NE | Index format must be FP16; crop width/height in specific range; batch sizes must match |
| 77 | `mps.sample_grid` | `anec.resample` | `ConvertSampleGrid` | All families | NE | Grid sampling operation |

### Space/Channel/Batch Transform Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 78 | `mps.depth_to_space_` | `anec.pixel_shuffle` | `ConvertDepthToSpace2D<PixelShuffle>` | All families | NE | Channel must be divisible by factor²; spatial dims must be multiple of factor |
| 79 | `mps.channel_to_space` | `anec.channel_to_space` | `ConvertDepthToSpace2D<ChannelToSpace>` | All families | NE | No z-dimension support |
| 80 | `mps.space_to_depth_` | `anec.pixel_unshuffle` | `ConvertSpaceToDepth2D<PixelUnshuffle>` | All families | NE | No z-axis; input dims must be divisible by factor |
| 81 | `mps.space_to_channel` | `anec.space_to_channel` | `ConvertSpaceToDepth2D<SpaceToChannel>` | All families | NE | Single input only |
| 82 | `mps.batch_to_space` | `anec.batch_to_space` | `ConvertBatchToSpace<BatchToSpaceOp>` | All families | NE | No z-axis; batch must be divisible by factor product; batch_axis must be constant |
| 83 | `mps.space_to_batch` | `anec.space_to_batch` | `ConvertBatchToSpace<SpaceToBatchOp>` | All families | NE | Spatial dims must be divisible by factor |

### Quantization Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 84 | `mps.quantize` | `anec.quant` | `ConvertQuantizationOp` | All families | PE | Input: fp16/fp32; Output: int8/uint8/e4m3/e5m2; NO asym quant; NO blockwise scale; per-cout and scalar scale cannot coexist |
| 85 | `mps.dequantize` | `anec.dequant` | `ConvertQuantizationOp` | All families | PE | Input: int8/uint8/int4/e4m3; Output: fp16; Int4 per-cout dequant NOT supported |

### Attention / Special Ops

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 86 | `mps.scaled_dot_product_attention` | `anec.sdpa` | `ConvertScaledDotProductAttention` | All families | NE | 4-5 inputs only; rank ≤ 4; K and V must be same shape; mask channel must match Q or be broadcastable; L2 budget for attention |
| 87 | `mps.stencil` | `anec.stencil` | `ConvertStencil` | All families | NE | kernel rank must be 4; reduction_mode must be sum; no channel stencil with rank 5; dilated stencil NOT supported; strided stencil NOT supported |

### Weight/Variable/State Ops (Family-Agnostic)

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 88 | `mps.constant` | (weight materialization) | `ConvertConstant` | All families | — | Constant weights/values; Const tensor interleave must be 1 |
| 89 | `mps.read_data_from_file` | (weight loading) | `ConvertReadDataFromFile` | All families | — | File-based weight loading |
| 90 | `mps.read_variable` | (variable reading) | `ConvertReadVariable` | All families | — | Variable ops |
| 91 | `mps.state` | `anec.state` | `ConvertState` | All families | — | State variable |
| 92 | `mps.identity` | (folded) | `FoldOperation<IdentityOp>` | All families | — | Identity is folded away; no compute |

### Convolution-Adjacent (Implicit from addPatternsForTarget)

| # | MIL Op | ANEC Dialect Op | Converter | Family Scope | Engine | Key Constraints |
|---|---|---|---|---|---|---|
| 93 | `mps.cost_volume` | (cross correlation) | `ConvertCrossCorrelation` | All families | NE | Input size must be 2; template must be rasterized; template height=1, depth=1 |
| 94 | `mps.im_to_col` | (im2col) | Internal utility | — | — | Used internally for conv decomposition; not a standalone ANE op |

---

## 2.2 MIL Ops That NEVER Land on ANE (No Converter)

These ops have no conversion pattern to ANEC dialect and will ALWAYS execute on CPU/GPU:

### Trigonometric (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 95 | `mps.acos` | No converter; inverse trig not supported on ANE |
| 96 | `mps.acosh` | No converter; inverse hyperbolic trig not supported |
| 97 | `mps.asin` | No converter; inverse trig not supported |
| 98 | `mps.asinh` | No converter; inverse hyperbolic trig not supported |
| 99 | `mps.atan` | No converter; inverse trig not supported |
| 100 | `mps.atanh` | No converter; inverse hyperbolic trig not supported |
| 101 | `mps.sinh` | No converter; hyperbolic sin not supported |
| 102 | `mps.cosh` | No converter; hyperbolic cos not supported |
| 103 | `mps.tan` | No converter; tangent not supported |

### Logical/Bitwise (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 104 | `mps.and` | No converter; logical AND not supported |
| 105 | `mps.or` | No converter; logical OR not supported |
| 106 | `mps.xor` | No converter; logical XOR not supported |
| 107 | `mps.nand` | No converter; logical NAND not supported |
| 108 | `mps.nor` | No converter; logical NOR not supported |
| 109 | `mps.xnor` | No converter; logical XNOR not supported |
| 110 | `mps.bitwise_and` | No converter; bitwise AND not supported |
| 111 | `mps.bitwise_or` | No converter; bitwise OR not supported |
| 112 | `mps.bitwise_xor` | No converter; bitwise XOR not supported |
| 113 | `mps.bitwise_not` | No converter; bitwise NOT not supported |
| 114 | `mps.bitwise_left_shift` | No converter; bit shift not supported |
| 115 | `mps.bitwise_right_shift` | No converter; bit shift not supported |
| 116 | `mps.bitwise_popcount` | No converter; popcount not supported |

### Complex Number Ops (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 117 | `mps.create_complex` | No converter; complex numbers not supported |
| 118 | `mps.real_part` | No converter; complex not supported |
| 119 | `mps.imaginary_part` | No converter; complex not supported |
| 120 | `mps.conjugate` | No converter; complex conjugate not supported |

### FFT Ops (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 121 | `mps.fast_fourier_transform` | No converter; FFT not supported on ANE |
| 122 | `mps.hermitean_to_real_fft` | No converter; FFT not supported |
| 123 | `mps.real_to_hermitean_fft` | No converter; FFT not supported |

### Matrix Algebra (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 124 | `mps.matrix_decomposition_lu` | No converter; LU decomposition not supported |
| 125 | `mps.matrix_inverse` | No converter; matrix inverse not supported |
| 126 | `mps.matrix_solver_lu` | No converter; LU solver not supported |

### RNN/LSTM/GRU (No Direct ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 127 | `mps.gru` | No direct converter; may decompose but sequential nature limits ANE |
| 128 | `mps.gru_gate_layout` | No converter; GRU layout attribute |
| 129 | `mps.gru_gradient` | No converter; gradient ops not on ANE |
| 130 | `mps.lstm` | No direct converter; sequential ops problematic for ANE |
| 131 | `mps.lstm_gate_layout` | No converter; LSTM layout attribute |
| 132 | `mps.lstm_gradient` | No converter; gradient ops not on ANE |
| 133 | `mps.rnn_activation` | No converter; RNN activation not supported |
| 134 | `mps.singlegate_rnn` | No converter; single-gate RNN not supported |
| 135 | `mps.singlegate_rnn_gradient` | No converter; gradient ops not on ANE |

### Cumulative/Sequential Ops (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 136 | `mps.cumulative_maximum` | No converter; cumulative ops require sequential access |
| 137 | `mps.cumulative_minimum` | No converter; cumulative ops not supported |
| 138 | `mps.cumulative_product` | No converter; cumulative ops not supported |
| 139 | `mps.cumulative_sum` | No converter; cumulative ops not supported |

### Random/Noise (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 140 | `mps.random_normal` | "ANE cannot support dynamic shape random op" |
| 141 | `mps.random_truncated_normal` | No converter; random not supported |
| 142 | `mps.random_uniform` | No converter; random not supported |
| 143 | `mps.init_random_philox_state` | No converter; RNG state not supported |
| 144 | `mps.update_random_state` | No converter; RNG state not supported |

### Control Flow (No ANE Compute)
| # | MIL Op | Reason |
|---|---|---|
| 145 | `mps.if` | No direct ANE converter; condition blocks limited |
| 146 | `mps.for` | No ANE converter; loops not supported |
| 147 | `mps.while` | Limited: "While loop variable generating output cannot run on ANE, prevents loop from running on ANE" |
| 148 | `mps.call` | No converter; function calls not on ANE |
| 149 | `mps.call_inline_mode` | No converter; inline mode attribute |
| 150 | `mps.condition` | "Condition layer is not supported" |
| 151 | `mps.yield` | No converter; control flow terminator |

### Gradient Ops (Never on ANE)
| # | MIL Op | Reason |
|---|---|---|
| 152 | `mps.bias_add_grad` | No converter; gradient ops always on CPU/GPU |
| 153 | `mps.pooling_average_gradient` | No converter; gradient ops not on ANE |
| 154 | `mps.pooling_max_gradient` | No converter; gradient ops not on ANE |
| 155 | `mps.relu_grad` | No converter; gradient ops not on ANE |
| 156 | `mps.sigmoid_gradient` | No converter; gradient ops not on ANE |
| 157 | `mps.sigmoid_gradient_with_sigmoid` | No converter; gradient ops not on ANE |
| 158 | `mps.resize_gradient` | No converter; gradient ops not on ANE |
| 159 | `mps.strided_slice_gradient` | No converter; gradient ops not on ANE |
| 160 | `mps.tile_gradient` | No converter; gradient ops not on ANE |
| 161 | `mps.local_convolution_data_gradient` | No converter; gradient ops not on ANE |
| 162 | `mps.local_convolution_weight_gradient` | No converter; gradient ops not on ANE |
| 163 | `mps.sample_grid_data_gradient` | No converter; gradient ops not on ANE |
| 164 | `mps.broadcast_gradient_args` | No converter; gradient ops not on ANE |
| 165 | `mps.prune_gradient` | No converter; gradient ops not on ANE |

### Scatter/Gather/Sort (Limited or No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 166 | `mps.scatter` | No ANE converter; scatter not supported |
| 167 | `mps.scatter_along_axis` | No converter; scatter not supported |
| 168 | `mps.scatter_nd` | No converter; scatter not supported |
| 169 | `mps.sort` | Has ANECSortLayerDesc but "vector_dimension size exceeds the record limit"; heavily constrained |
| 170 | `mps.top_k` | Has ANECTopKLayerDesc; "ANE TopK can only generate uint16 indices output"; constrained |
| 171 | `mps.non_maximum_suppression` | Has ANECNMSLayerDesc but "ANE cannot support NMS for ios15 and ios16"; batch/channel constraints |
| 172 | `mps.non_zero` | No converter; non-zero indices not supported |
| 173 | `mps.hamming_distance` | No converter; hamming distance not supported |

### Sparse/Tensor Buffer (No ANE Support)
| # | MIL Op | Reason |
|---|---|---|
| 174 | `mps.sparse_tensor_storage` | No converter; sparse storage not on ANE |
| 175 | `mps.materialize_sparse_tensor` | No converter; sparse materialization not on ANE |
| 176 | `mps.buffer_tensor` | "TensorBuffer as intermediate is not supported on ANE" |

### Type/Shape Queries (No ANE Compute)
| # | MIL Op | Reason |
|---|---|---|
| 177 | `mps.shape` | No converter; shape query not compute |
| 178 | `mps.rank` | No converter; rank query not compute |
| 179 | `mps.size` | No converter; size query not compute |
| 180 | `mps.dimension_size` | No converter; dimension query not compute |

### Miscellaneous No-Converter Ops
| # | MIL Op | Reason |
|---|---|---|
| 181 | `mps.band_part` | No converter; matrix band part not supported |
| 182 | `mps.clamp` | No direct converter; may decompose to min(max(x,a),b) |
| 183 | `mps.col_to_im` | No converter; im2col inverse not standalone |
| 184 | `mps.create_texture_tensor` | No converter; texture creation not compute |
| 185 | `mps.dequantize_lut` | No direct converter; LUT dequant has separate path |
| 186 | `mps.dynamic_shape_cast` | "Dynamic Shapes: memory layout operation is not supported for dynamic shape" |
| 187 | `mps.extract` | No converter; extract not supported |
| 188 | `mps.from_elements` | No converter; tensor construction not on ANE |
| 189 | `mps.func` | No converter; function definition |
| 190 | `mps.get_coordinates` | No converter; coordinate op not supported |
| 191 | `mps.is_finite` | No converter; finite check not supported |
| 192 | `mps.is_infinite` | No converter; infinite check not supported |
| 193 | `mps.is_nan` | No converter; NaN check not supported |
| 194 | `mps.local_convolution` | No converter; local conv not on ANE |
| 195 | `mps.lp_norm` | No converter; LP norm not supported |
| 196 | `mps.modulo` | No converter; modulo not supported |
| 197 | `mps.negative` | No converter; may decompose to multiply by -1 |
| 198 | `mps.one_hot` | No converter; one-hot encoding not supported |
| 199 | `mps.prelu` | No direct converter; may decompose to leaky_relu path |
| 200 | `mps.prune` | No converter; pruning not supported |
| 201 | `mps.pruning_metric` | No converter; pruning metric |
| 202 | `mps.pruning_structure` | No converter; pruning structure |
| 203 | `mps.reciprocal` | No direct converter; may decompose to div |
| 204 | `mps.reinterpret_cast` | No converter; reinterpret cast not supported |
| 205 | `mps.reverse_square_root` | May map to rsqrt but no direct converter |
| 206 | `mps.rint` | No converter; rint not supported (mps.round handles rounding) |
| 207 | `mps.signbit` | No converter; signbit not supported |
| 208 | `mps.sigmoid_hard` | No direct converter; SwishHardActivationDetection may handle variant |
| 209 | `mps.softplus` | No converter; softplus not supported |
| 210 | `mps.softplus_parametric` | No converter; parametric softplus not supported |
| 211 | `mps.softsign` | No converter; softsign not supported |
| 212 | `mps.strided_slice_update` | "the stride should be 1 for slice update on ANE" — rejected |
| 213 | `mps.variable_from_tensor` | No converter; variable creation not on ANE |
| 214 | `mps.assign_variable` | No converter; variable assignment not on ANE |
| 215 | `mps.placeholder` | No converter; placeholder not compute |
| 216 | `mps.return` | No converter; function return |
| 217 | `mps.device_hint` | No converter; hint only |
| 218 | `mps.nf` | No converter; normalization factor |
| 219 | `mps.unrealized_fold` | No converter; internal MLIR |

### Enum/Attribute Types (Not Compute Ops — Never on ANE)
`mps.padding_mode`, `mps.padding_style`, `mps.pooling_indices_mode`, `mps.nearest_rounding_mode`, `mps.sampling_mode`, `mps.scatter_mode`, `mps.similarity_type`, `mps.stencil_padding_mode`, `mps.crop_resize_alignment_mode`, `mps.crop_resize_coordinate_mode`, `mps.fft_scaling_mode`, `mps.tensor_data_layout`, `mps.texture_tensor_pixel_format`, `mps.type_constraint`, `mps.gru_gate_layout`, `mps.lstm_gate_layout`, `mps.reduction_mode`, `mps.enableANECHWRankPromotion`, `mps.useRegularSigmoid`

---

## Part 3: Per-Family Op Support Matrix

## 3.1 Summary Table

| Op Category | A11Legacy (LSE_0) | A12 (LSE_1) | A13 (LSE_2) | A14 (LSE_3) | A15 (LSE_4) | A16 (LSE_5) | A17 (LSE_6) | A18 (LSE_7) |
|---|---|---|---|---|---|---|---|---|
| Conv/Depthwise | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MatMul | ✅ (LSE_0) | ✅ (LSE_1) | ✅ (LSE_2) | ✅ (LSE_3) | ✅ (LSE_4) | ✅ (LSE_5) | ✅ (LSE_6) | ✅ (LSE_7) |
| Pooling (Avg/Max/L2) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Softmax | ⚠️ Arch-dep | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| InstanceNorm | ⚠️ Arch-dep | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Elementwise Binary (Add/Sub/Mul/etc) | ⚠️ A14Minus | ⚠️ A14Minus | ⚠️ A14Minus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus |
| Broadcast | ⚠️ FP16 only | ⚠️ FP16 only | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Reduction (Min/Avg/Max/Sum) | ⚠️ A14Minus | ⚠️ A14Minus | ⚠️ A14Minus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus | ✅ A14Plus |
| ReduceMin non-FP | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Square | A13Minus | A13Minus | A13Minus | A14Plus | A14Plus | A14Plus | A14Plus | A14Plus |
| Resize alignCorners | ✅ | ✅ | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| SDPA | ❓ | ❓ | ❓ | ❓ | ⚠️ | ✅ | ✅ | ✅ |
| LayerNorm | ❓ | ❓ | ❓ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| ArgMin/ArgMax | ✅ (LSE_0) | ✅ (LSE_1) | ✅ (LSE_2) | ✅ (LSE_3) | ✅ (LSE_4) | ✅ (LSE_5) | ✅ (LSE_6) | ❌ No LSE_7 |
| E4M3/E5M2 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |

✅ = Supported | ⚠️ = Partially/conditionally supported | ❌ = Not supported | ❓ = Uncertain

## 3.2 Version-Specific Constraints (Detailed)

### A11Legacy (ANE v5, A11 chip)
- **MIL support**: Limited. "MIL is only supported for H13+ ANE architectures" suggests A11Legacy may not fully support the MIL pipeline.
- **Broadcast**: "Only fp16 is supported for A11/A12 Broadcasts." — Only FP16 broadcast allowed.
- **Broadcast support**: "Unsupported A11/A12 Broadcast." — Many broadcast patterns unsupported.
- **ReduceMin**: "ReduceMin for non fp type is not supported for A13 and below" — Only FP ReduceMin.
- **Elementwise**: Uses ConvertElementwiseBinaryA14Minus — older, less optimized implementation.
- **Reduction**: Uses ConvertReductionA14Minus — older implementation.
- **Square**: Uses ConvertSquareA13Minus — different lowering.
- **Divide/FloorDivide/MatMul**: Family-specific LSE_0 converter.
- **Reshape/ExpandDims/Squeeze/Resize/Transpose/Broadcast/Pad/Slice/StridedSlice/Reverse/Crop**: Family-specific LSE_0 converter.

### A12 (ANE v6, A12 chip)
- Same constraints as A11Legacy for broadcast and ReduceMin.
- Elementwise binary: ConvertElementwiseBinaryA14Minus with Family::A12.
- Default target family for ANEC region formation.
- Full MIL support (H12+).

### A13 (ANE v7, A13 chip)
- Same broadcast and ReduceMin constraints as A11/A12.
- Pool-to-conv fallback: "to MaxPool, AveragePool or Conv for A13 and below on ane is not supported."
- Elementwise binary: ConvertElementwiseBinaryA14Minus with Family::A13.
- Square: ConvertSquareA13Minus (last generation with this variant).

### A14 (ANE v8, A14 chip / M1-class)
- **Major upgrade point**: A14Plus converters for elementwise binary and reduction.
- **Resize**: "Resize alignCorners == centerResult == true is not supported on A14-class ANEs."
- **Square**: ConvertSquareA14Plus (new, improved lowering).
- **Elementwise binary**: ConvertElementwiseBinaryA14Plus — broader type support.
- **Reduction**: ConvertReductionA14Plus — ReduceMin now works for non-FP types.

### A15 (ANE v10, A15 chip)
- Likely adds LayerNorm as first-class ANEC op.
- SDPA may begin limited support.
- Full A14Plus converter set.

### A16 (ANE v11, A16 chip)
- SDPA (Scaled Dot-Product Attention) confirmed available.
- Full elementwise and reduction support.
- E4M3 still not supported.

### A17/A18 (ANE v26?, A17/A18/M4-class)
- **ArgMin/ArgMax**: No LSE_7 converter — may use different implementation path.
- **E4M3/E5M2**: May have limited support on latest architectures.
- **SDPA**: Fully supported.
- **LayerNorm**: Fully supported.

### vu1 (uANE / micro ANE)
- "Deconv on uANE is not yet supported."
- Likely significantly reduced op set compared to full ANE.
- "MIL is only supported for H13+ ANE architectures" — uANE may not support MIL directly.

---

## Part 4: Hardware Limits and Fusion Architecture

## 4.1 The Three ANE Execution Engines

The ANEC compiler targets three distinct hardware engines within the ANE, each with its own fusion categories:

### NE Engine (Neural Engine — Convolution/Math Pipeline)
Handles the "heavy compute" operations: convolutions, pooling, matrix multiplication, and their associated transforms.

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

### PE Engine (Processing Element — Element-Wise/Reduction Pipeline)
Handles element-wise operations, reductions, scaled element-wise, and GOC layers.

| PE Fused Op Type | Description |
|---|---|
| `PEFUSED_ELEMENTWISE` | Element-wise operations with fused pre/post transforms |
| `PEFUSED_GOC` | Generic Operation Compute (flexible compute kernel) |
| `PEFUSED_POOL` | PE-side pooling operations |
| `PEFUSED_SECUREFLUSH` | Secure data flush operations |

### TransposeEngine
Handles data rearrangement operations (transposes, reshapes, channel reordering) and can be fused with adjacent NE/PE operations.

| Component | Description |
|---|---|
| `TransposeEngineLayer` | Standalone transpose engine layer |
| `MirTransposeEngineFusion` | Fusion pass for transpose engine ops |
| `ConvertNEBypassToTransposeEngine` | Converts NE bypass layers to transpose engine when possible |

### Engine Layer Constraints
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

## 4.2 Hardware Limit Parameters (hal_params)

These are the actual constraint parameters validated at compile time. Each `_target_hw_limits_v*` structure contains these fields with different numeric values per ANE revision.

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
| `max_conv_kernel_dim_y` | `kh <= max_conv_kernel_dim_y` | Max kernel height |
| `max_conv_kernel_dim_z` | `kd <= max_conv_kernel_dim_z` | Max kernel depth |
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
| `pe_max_tile_height` | `(hout - overlap + overlap_pad) <= pe_max_tile_height * (tile_height - overlap)` | Max tile height in PE |
| `pe_max_patch_width_height_sum_log2` | `(patch_width + patch_height) <= pe_max_patch_width_height_sum_log2` | Max patch W+H sum (log2) |
| `pe_min_patch_width_log2` | `pe_min_patch_width_log2 <= patch_width` | Min patch width (log2) |
| `pe_max_transpose_wtoc_cin` | `cin <= pe_max_transpose_wtoc_cin` | Max input channels for W-to-C transpose |
| `pe_max_transpose_ctow_cout` | `cout <= pe_max_transpose_ctow_cout` | Max output channels for C-to-W transpose |
| `pe_reduction_cout_limit` | `cout <= pe_reduction_cout_limit OR (win <= POW2(patch_width) && hin <= POW2(patch_height) && din == 1)` | Max output channels for reduction |
| `has_pe_max_patch_width_height_sum` | Conditional flag | Feature flag for patch size constraint |

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
| `ew_limit_128` | `wout % ew_limit_128 > ew_limit_64` | 128-byte alignment boundary |
| `ew_limit_256` | `wout % ew_limit_256 > ew_limit_128` | 256-byte alignment boundary |

### Small Source Mode (NP2) Limits

| Parameter | Constraint | Description |
|---|---|---|
| `np2_6_max_src_width_inclusive` | `src_width <= np2_6_max_src_width_inclusive` | Max src width for NP2_6 mode |
| `np2_6_min_dst_width_exclusive` | `dst_width > np2_6_min_dst_width_exclusive` | Min dst width for NP2_6 mode |
| `np2_10_max_src_width_inclusive` | `src_width <= np2_10_max_src_width_inclusive` | Max src width for NP2_10 mode |
| `np2_10_min_dst_width_exclusive` | `dst_width > np2_10_min_dst_width_exclusive` | Min dst width for NP2_10 mode |

### Memory/DMA Limits

| Parameter | Constraint | Description |
|---|---|---|
| `dram_alignment` | Buffer sizes must be aligned to `dram_alignment` | DRAM alignment requirement |
| `l2_bank_align` | Channel strides scaled by alignment | L2 bank alignment |
| `max_l2_chan_stride_for_non_resident_or_chained_buffer` | `channel_stride * l2_bank_align <= max_l2_chan_stride` | Max L2 channel stride |
| `cache_prefetch_max_outstanding_requests` | `tlimit <= cache_prefetch_max_outstanding_requests` | Max outstanding cache prefetch requests |

## 4.3 Fusion Atom Ordering

The compiler uses rigidly ordered "atoms" (building blocks) that must appear in specific sequence within a fused engine layer.

### NE Engine Atom Ordering

**Input-side atoms** (must precede the primary op):
1. `InputDeQuantAtom` / `DeQuantPreScaleAtom` — Dequantize input
2. `PreScaleAtom` — Pre-scale the input
3. `InputReLUAtom` — Apply input ReLU
4. `TransposeAtom` (CW) — Channel-width transpose
5. `TextureAtom` — Texture format conversion

**Primary op atom** (exactly one):
6. `NEConvAtom` / `ConvGOCAtom` / `ConvQuantAtom` — Convolution
7. Or `MatMulAtom` / `PoolAtom` / etc.

**Output-side atoms** (must follow the primary op):
8. `ActivationAtom` — Post-op activation
9. `GOCAtom` / `NEGOCAtom` — Generic operation compute
10. `PostScaleAtom` — Post-scale
11. `EWGOCAtom` / `EWQuantAtom` — Element-wise + GOC/quant fusion
12. `BypassGOCAtom` / `BypassQuantAtom` — Bypass with GOC/quant

### PE Engine Atom Ordering

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

This ordering is **not negotiable**. The compiler will not reorder atoms to enable fusion.

## 4.4 ActiveNE & OCG Resource Allocation

ActiveNE is the number of Neural Engine units activated for a given operation. Key constraints:

```
(1 << (activene + wu_stack + fat_tile_enable)) <= hal_params.num_nes
(1 << activene) must be a power of 2
(num_subgroup % (1 << activene)) == 0
(active_ne_times_ocg % logical_intlv) == 0
cout <= (1 << (activene + ocgsize))
cout <= (unicast_cout * (1 << activene))
active_ne_ocg_size % 16 == 0 || relax_active_ne_ocg_constraint
active_ne_ocg_size % 8 == 0 || relax_active_ne_ocg_constraint
ne_info_.active_ne cannot be zero
CoutBatch cannot must be between 1 and 16
```

## 4.5 Memory Pressure & Subgraph Boundaries

The primary mechanism for determining fusion boundaries is memory pressure analysis. The compiler models L2 cache and register usage, cutting subgraphs when pressure exceeds hardware capacity.

**Subgraph identification algorithms:**
- `MirSubgraphIdentification` — Basic graph partitioning
- `MirPressureBasedSubgraphIdentification` — Pressure-driven partitioning
- `MirSpatialSplitPressureBasedSubgraphIdentification` — Spatial split with pressure
- `CostBasedSubgraphIdentification` — Cost model driven
- `BondedSplitSubgraphIdentification` — Multi-ANE splitting
- `LegalizerSubgraphIdentification` — Legalization-driven

**Two L2 data sharing mechanisms:**
- **Chaining**: Output of one TD directly consumed by next without DRAM writeback. "Either chain or l2-dep cost should be set, not both."
- **L2-Dependency**: Data kept in L2 cache between TDs. "An L2 dep pair must be assigned to the same ANE."

---

## Part 5: Validation System and Data Types

## 5.1 The ValidateLayer Template System

Every MIL operation that wants to run on ANE must pass through a `ValidateLayer` instantiation. 40+ distinct instantiations exist, each specific to an operation type:

| ANEC Layer Descriptor | MIL Operation Category |
|---|---|
| `ANECConvLayerDesc` | Convolution |
| `ANECConcatLayerDesc` | Concatenation |
| `ANECGOCLayerDesc` | Generic Operation Compute |
| `ANECElementWiseLayerDesc` | Element-wise ops |
| `ANECNeuronLayerDesc` | Activation functions |
| `ANECPoolLayerDesc` | Pooling |
| `ANECReductionLayerDesc` | Reduction |
| `ANECMatrixMultLayerDesc` | MatMul |
| `ANECTransposeLayerDesc` | Transpose |
| `ANECReshapeLayerDesc` | Reshape |
| `ANECSoftmaxLayerDesc` | Softmax |
| `ANECLayerNormLayerDesc` | Layer Normalization |
| `ANECInstanceNormLayerDesc` | Instance Norm |
| `ANECLinearLayerDesc` | Linear/Fully Connected |
| `ANECBroadcastLayerDesc` | Broadcast |
| `ANECScaledElementWiseLayerDesc` | Scaled Element-Wise |
| `ANECSDPALayerDesc` | SDPA |
| `ANECResampleLayerDesc` | Resample/Resize |
| `ANECArgMinMaxLayerDesc` | ArgMin/ArgMax |
| `ANECGlobalArgMinMaxLayerDesc` | Global ArgMin/ArgMax |
| `ANECGatherLayerDesc` | Gather |
| `ANECDynamicSliceLayerDesc` | Dynamic Slice |
| `ANECInputViewLayerDesc` | Input View |
| `ANECPadLayerDesc` | Padding |
| `ANECNMSLayerDesc` | Non-Maximum Suppression |
| `ANECLRNLayerDesc` | Local Response Norm |
| `ANECL2NormLayerDesc` | L2 Normalization |
| `ANECMinMaxNormLayerDesc` | Min/Max Normalization |
| `ANECPixelShuffleLayerDesc` | Pixel Shuffle |
| `ANECPixelUnshuffleLayerDesc` | Pixel Unshuffle |
| `ANECChannelToSpaceLayerDesc` | Channel-to-Space |
| `ANECSpaceToChannelLayerDesc` | Space-to-Channel |
| `ANECBatchToSpaceLayerDesc` | Batch-to-Space |
| `ANECSpaceToBatchLayerDesc` | Space-to-Batch |
| `ANECCropResizeLayerDesc` | Crop-Resize |
| `ANECCrossCorrelationLayerDesc` | Cross Correlation |
| `ANECRingBufferWriterLayerDesc` | Ring Buffer Writer |
| `ANECDynamicGOCLayerDesc` | Dynamic GOC |
| `ANECSortLayerDesc` | Sort |
| `ANECTileLayerDesc` | Tile |
| `ANECTopKLayerDesc` | Top-K |
| `ANECDropoutLayerDesc` | Dropout |
| `ANECFlattenLayerDesc` | Flatten |
| `ANECUnflattenLayerDesc` | Unflatten |
| `ANECAffineTransformLayerDesc` | Affine Transform |
| `ANECCrossProductLayerDesc` | Cross Product |
| `ANECRandomLayerDesc` | Random |
| `ANECResizeLayerDesc` | Resize |

Each takes: the layer descriptor, a vector of `ANECTensorDesc` (input tensor descriptions), and a vector of `ANECTensorValueDesc` (input tensor values/constants). The ANECTensorDesc validates rank ∈ [0,7], valid interleave ∈ {1,2,3,4,8}, and Int4 must use interleave 8.

## 5.2 The mlir::placement Dialect

The placement dialect governs whether ops execute on ANE or host:

| MLIR Op | Purpose |
|---|---|
| `placement.region_call` | Calls a region (ANE or Host); has `region_type` and `callee` attributes |
| `placement.ane_io_cast` | Casts data between ANE and host representations at boundaries |
| `placement.memref_to_tensor` | Converts memref (host memory) to tensor (ANE format) |
| `placement.tensor_to_memref` | Converts tensor back to memref |
| `placement.replaced_ops` | Marks replaced ops |
| `placement.start_timer` / `placement.stop_timer` | Performance measurement at boundaries |
| `placement.host_type_cast` | Type casting at host boundary |

**Critical placement flags:**
- `force-ane-placement` — Forces ALL ops onto ANE (fails if any incompatible)
- `force-host-placement` — Forces ALL ops onto host
- `print-placement-report` — Prints "ANEC Placement Report" showing per-op decisions
- `If true, placement uses the cost model.` — Cost model drives decisions

## 5.3 Data Type Support

| Data Type | ANE Support | Notes |
|---|---|---|
| Float16 (FP16) | ✅ Universal | Primary compute format |
| Float32 (FP32) | ✅ Limited | Input/output only; internal compute is FP16; DynamicGOC only FP16 |
| Int8 | ✅ Universal | Quantized weights/activations |
| UInt8 | ✅ Universal | Quantized weights/activations |
| Int4 | ✅ Constrained | Must use interleave factor 8 |
| E4M3 (FP8) | ⚠️ Arch-dependent | NOT supported on most architectures; kernel format possible on latest |
| E5M2 (FP8) | ❌ | "E4M3 or E5M2 format not supported" |
| BF16 | ❌ | No evidence of BF16 support |
| 32-bit format | ❌ | "32 bit format not supported" for compute |
| 2xInt8 | ❌ | "2xInt8 mode is not supported" |
| Packed10 | ❌ | "Invalid input tensor format: packed10" |

## 5.4 Quantization Rules

| Rule | Details |
|---|---|
| Quantize input | Must be fp16 or fp32 |
| Quantize output | Must be int8, uint8, e4m3, or e5m2 |
| Dequantize input | Must be int8, uint8, int4, or e4m3 |
| Dequantize output | Must be fp16 |
| Blockwise scale | NOT supported: "ANE doesn't support blockwise scale" |
| Asymmetric quant | NOT supported: "Asym quantization is not supported" |
| Int4 per-cout dequant | NOT supported |
| Per-cout + scalar coexist | NOT allowed: "per-cout scale and scalar scale cannot be defined simultaneously" |
| Mutable + quantized | NOT allowed: "mutable kernels are not supported with kernel quantization" |
| E4M3 quant zero point | NOT allowed: "Zero point is not supported for quant with E4M3 output format" |

## 5.5 Palettization (LUT) Constraints

| Constraint | Details |
|---|---|
| Palette ranks | 0 through 6 supported |
| Min palette bits | 4 bits minimum (1-2 bit palettes NOT supported) |
| CW conv + vector palette | NOT supported |
| Deconv + vector palette | NOT supported |
| Dilation + vector palette | NOT supported |
| Large stride + palette | NOT supported |
| Per-cout + vector palette | NOT supported |
| LUT size limit | Must not exceed `ne_palette_lut_size_in_bytes` |
| Consecutive channels | Must share same palette LUT |
| Palettization + compression | Cannot have both simultaneously |

## 5.6 Dynamic Shape Constraints ("Kill Switch")

Dynamic shapes create a cascading rejection system:

| Condition | Consequence |
|---|---|
| Any op not ANE-resident | **ALL ops marked non-ANE-resident** (all-or-nothing rule) |
| Conv stride > 2 | Rejected |
| Conv same/same_lower padding + stride > 1 | Rejected |
| Large kernel conv | Rejected |
| Pool ceil_mode + stride > 1 | Rejected |
| Pool same/same_lower padding + stride > 1 | Rejected |
| Global max/min pool | Rejected (use reduction instead) |
| Memory layout operations | Rejected |
| Space-to-channel | Rejected |
| Reshape | Rejected |
| Input depth != 1 | Rejected |
| SPMD procedures | Rejected |
| Random ops | Rejected |
| Unranked types | Rejected |

## 5.7 Memory Layout Constraints

| Constraint | Details |
|---|---|
| Interleave values | {1, 2, 3, 4, 8} only |
| Interleave axis | "ANEC only supports interleave on C axis" |
| Const tensor interleave | Must be 1 |
| ChannelLast interleave | Must be 1 |
| Int4 interleave | Must be 8 |
| ChannelLast support | "Only for channel wise convolutions"; not in DRAM |
| ChannelLast mixing | "Cannot mix channellast and non-channellast input/output tensors" |
| Rank limit | Max rank 5: "exceeds the max rank 5" |
| Linear rank | "ANE cannot support Linear with input rank >= 5" |
| Alignment | 64-byte for some tensors; H*W divisible by 8 for some ops |

---

## Part 6: Compiler Pipeline

The complete ANEC compilation pipeline:

```
MIL Framework Model
       │
       ▼
┌──────────────┐
│  MLIR ANEC   │  Custom MLIR dialect for ANE operations
│   Frontend   │  (mlir/mps/src/Dialect/ANEC)
│              │  - Per-family ConvertXxx<LSE_N> patterns
│              │  - addDynamicallyLegalOp<anec::A12..A18>
│              │  - eraseOpsWeCannotConvert()
└──────┬───────┘
       │ ValidateLayer + placement dialect
       ▼
┌──────────────┐
│   ANE IR    │  OpLayer directed graph
│  (Op Layer)  │  Operations as graph nodes
└──────┬───────┘
       │ Optimization passes
       ▼
┌──────────────┐
│  MirOpt     │  ActiveNE fusion, batch/channel splitting
│  Passes      │  Subgraph identification, spatial tiling
│              │  L2 legalization, EwCopy optimization
│              │  Pad+Conv/Pad+Pool DecomposeAndFuse
│              │  6 hoisting passes for enabling fusion
└──────┬───────┘
       │ Scheduling + Register Allocation
       ▼
┌──────────────┐
│  IR         │  Operation scheduling, local reg alloc
│  Schedule +  │  Register spilling, L2 footprint calc
│  RegAlloc    │  Chaining vs L2-dep decision
└──────┬───────┘
       │ Code Generation
       ▼
┌──────────────┐
│  Codegen    │  Versioned TD program generation
│  (v1-v26)    │  PE codegen, register programming
│  + vu1      │  RegisterProgrammingAnalysis<N>
└──────┬───────┘
       │ Linking
       ▼
┌──────────────┐
│  Linker     │  Final program linking + serialization
└──────────────┘
```

**Compiler control flags:**
- `DisableMergeActivation` / `DisableMergeScaleBias` / `DisableMergeConstants`
- `AggressiveScaleFusion` / `EnableAggressiveNETransposeFusion`
- `DumpFusionBoundaryInfo` / `EnableMILConstantCoalescing`
- `EnableDramInplaceAllocation` / `EnableL2CachedBuffer`
- `spatial_split_transform`: disabled/test/memory/auto/manual/generic-dag
- `CostModelClusterThreshold` / `GlobalRefinementInSpatialSplit`

---

## Part 7: Complete ANEC Dialect Op List

The full list of operations registered in the `anec` MLIR dialect (from `Dialect::addOperations<>` template):

`anec.A11Legacy`, `anec.A12`, `anec.A13`, `anec.A14`, `anec.A15`, `anec.A16`, `anec.A17`, `anec.A18` (family marker ops), `anec.abs`, `anec.add`, `anec.arg_min_max`, `anec.average_pool`, `anec.batch_norm`, `anec.batch_to_space`, `anec.broadcast`, `anec.cast`, `anec.ceil`, `anec.channel_to_space`, `anec.clamped_relu`, `anec.concat`, `anec.convolution`, `anec.cos`, `anec.crop_resize`, `anec.deconvolution`, `anec.degamma`, `anec.dequant`, `anec.dirac`, `anec.div`, `anec.elu`, `anec.equal`, `anec.equal_zero`, `anec.erf`, `anec.exp2`, `anec.flatten`, `anec.floor`, `anec.gain_offset_control`, `anec.gather_nd`, `anec.gelu`, `anec.global_arg_min_max`, `anec.high_precision_sigmoid`, `anec.input_view`, `anec.instance_norm`, `anec.invert`, `anec.l2norm_pool`, `anec.layer_norm`, `anec.leaky_relu`, `anec.less_than`, `anec.less_than_equal`, `anec.less_than_equal_zero`, `anec.less_than_zero`, `anec.linear`, `anec.log2`, `anec.matmul`, `anec.max`, `anec.max_pool`, `anec.min`, `anec.mult`, `anec.n_relu`, `anec.not_equal`, `anec.not_equal_zero`, `anec.padding`, `anec.pixel_shuffle`, `anec.pixel_unshuffle`, `anec.power`, `anec.quant`, `anec.r_sqrt`, `anec.reduce_avg`, `anec.reduce_max`, `anec.reduce_min`, `anec.reduce_sum`, `anec.region_return`, `anec.relu`, `anec.resample`, `anec.reshape`, `anec.resize`, `anec.ring_buffer_reader`, `anec.ring_buffer_writer`, `anec.round_nearest`, `anec.scaled_elementwise`, `anec.sdpa`, `anec.sigmoid`, `anec.sign`, `anec.sin`, `anec.softmax`, `anec.space_to_batch`, `anec.space_to_channel`, `anec.sqr`, `anec.sqrt`, `anec.state`, `anec.sub`, `anec.swish`, `anec.tanh`, `anec.tensor_buffer_to_tensor`, `anec.tensor_to_tensor_buffer`, `anec.tile`, `anec.transpose`, `anec.trunc`, `anec.unflatten`, `anec.unrealized_conversion_cast`

Total: 94 ANEC dialect operations (including 8 family marker ops).

---

*Note: Actual numeric values for hal_params fields differ per ANE revision. Precise thresholds can be determined through hardware-specific testing.*
