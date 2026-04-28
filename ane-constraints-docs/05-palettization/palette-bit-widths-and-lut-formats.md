# Palette Bit Widths and LUT Formats

## TL;DR

1. **Palette128 (7-bit, 128-entry LUT) does NOT exist** — Not found in coremltools or observable ANE compilation behavior. The supported palette bit widths are strictly **1, 2, 3, 4, 6, 8** (no 5-bit or 7-bit).
2. **"128" in palette context = group_size, NOT LUT entries** — The `128` you see is the `group_size` parameter for **per-grouped-channel palettization** (coremltools default=32, user-specified, common values: 32, 64, 128). This controls how output channels are grouped for separate LUTs, NOT the number of LUT entries. A 4-bit palette with group_size=128 still has 16 LUT entries per group.
3. **bf16 is NOT supported in CoreML** — Neither at the ANE kernel format level, nor in the CoreML model format. `constexpr_lut_to_dense` only supports `fp16` and `fp32` as LUT entry types. bf16 is used only on the PyTorch training side (fake_palettize) and raises `KeyError` during CoreML conversion.
4. **1-bit palettes (Palette2) ARE real** — They work for non-conv ops on all hardware versions. The "4-bit minimum" only applies to convolution kernels specifically.

---

## 1. Complete ANE Kernel Format Palette Enum

The ANE compiler defines these **exact** palette format string literals — no more, no less:

| Format Name | LUT Entries | Index Bit Width | Sparse Variant |
|---|---|---|---|
| **Palette2** | 2 | 1 bit | Palette2Sparse |
| **Palette4** | 4 | 2 bit | Palette4Sparse |
| **Palette8** | 8 | 3 bit | Palette8Sparse |
| **Palette16** | 16 | 4 bit | Palette16Sparse |
| **Palette64** | 64 | 6 bit | Palette64Sparse |
| **Palette256** | 256 | 8 bit | Palette256Sparse |

**Definitively absent:** Palette32 (5-bit), Palette128 (7-bit), Palette512 (9-bit). The palette sizes jump 2→4→8→16→64→256 (powers of 2 that are NOT 32 or 128).

### Evidence from C++ template instantiations

The `IrWeightDataBitStream<N>` template is instantiated for N = **1, 2, 3, 4, 6** only (no 5 or 7):

```
IrWeightDataBitStream1  (1-bit indices)
IrWeightDataBitStream2  (2-bit indices)
IrWeightDataBitStream3  (3-bit indices)
IrWeightDataBitStream4  (4-bit indices)
IrWeightDataBitStream6  (6-bit indices)
```

The `IrCompressedConstData_specialization<T, LhN>` template is instantiated for index widths **1, 2, 3, 4, 6, 8** with LUT entry types `h` (uint8), `a` (int8), `Dh` (bf16), `f` (fp32), `e4m3_t` (fp8):

```
IrCompressedConstData_specializationIhLh1EE   (uint8 LUT entries, 1-bit indices)
IrCompressedConstData_specializationIhLh2EE   (uint8 LUT entries, 2-bit indices)
IrCompressedConstData_specializationIhLh3EE   (uint8 LUT entries, 3-bit indices)
IrCompressedConstData_specializationIhLh4EE   (uint8 LUT entries, 4-bit indices)
IrCompressedConstData_specializationIhLh6EE   (uint8 LUT entries, 6-bit indices)
IrCompressedConstData_specializationIhLh8EE   (uint8 LUT entries, 8-bit indices)
IrCompressedConstData_specializationIDhLh16EE  (bf16 LUT entries, 16-byte aligned)
IrCompressedConstData_specializationIaLh8EE    (int8 LUT entries, 8-byte aligned)
IrCompressedConstData_specializationIfLh32EE   (fp32 LUT entries, 32-byte aligned)
IrCompressedConstData_specializationI6e4m3_tLh8EE  (fp8 e4m3 LUT entries, 8-byte aligned)
```

**No Lh5 or Lh7 specializations exist.** No 5-bit or 7-bit index storage.

### Non-palette ANE kernel format values

`Float16`, `Float32`, `Int4`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Sparse`, `Palettized`, `Quantized`, `QuantizedSparse`, `QuantizedPalettized`, `SparsePalettized`, `QuantizedSparsePalettized`

**No bf16 kernel format.**

---

## 2. coremltools Confirmation: No Palette128 / No bf16

### Supported palette nbits (coremltools source)

**File:** `optimize/coreml/_config.py`, line 723:
```python
_VALID_NBITS = (1, 2, 3, 4, 6, 8)
```

**File:** `optimize/coreml/_quantization_passes.py`, line 624:
```python
_SUPPORTED_NBITS = (1, 2, 3, 4, 6, 8)
```

7-bit is NOT in the tuple. Attempting `nbits=7` raises:
```
Invalid value of "nbits" (7) for palettization. Supported "nbits" are (1, 2, 3, 4, 6, 8)
```

### iOS18 constexpr_lut_to_dense: Index types

**File:** `converters/mil/mil/ops/defs/iOS18/compression.py`, lines 296-298:
```python
type_domains = {
    "IndicesT": (types.uint1, types.uint2, types.uint3, types.uint4, types.uint6, types.uint8),
    "T": (types.int8, types.uint8, types.fp16, types.fp32),
}
```

No `uint7` type exists in MIL. The `NUM_PALETTES` validation enforces `2^nbits` (line 327-331):
```python
num_palettes = lut_shape[-2]
nbits = int(math.log2(num_palettes))
if num_palettes != 2**nbits:
    raise ValueError(...)
```

128 is NOT a power of 2, so it would fail this check. The valid NUM_PALETTES values are: 2, 4, 8, 16, 64, 256.

### bf16 in coremltools: NOT supported for model conversion

**File:** `converters/mil/frontend/torch/test/test_torch_conversion_api.py`:
```python
def test_weight_only_quantization_bfloat16_not_support(self):
    """
    Torchao quant_api.int4_weight_only only supports bfloat16.
    """
    # The conversion of bfloat16 hasn't been supported yet.
    with pytest.raises(KeyError, match="torch.bfloat16"):
        ct.convert(exported_model, minimum_deployment_target=ct.target.iOS17)
```

bf16 is used ONLY in PyTorch-side fake_palettize training simulation:
- `fake_palettize.py`: `self.lut_dtype == "b16"` — casts through bf16 for training, not CoreML conversion
- `palettization_config.py`: `lut_dtype: 'i8', 'u8', 'f16', 'bf16', 'f32'` — PyTorch-side only

The CoreML `constexpr_lut_to_dense` type_domains only accept `(int8, uint8, fp16, fp32)` for LUT entry type T — **no bf16**.

---

## 3. The "group_size 128" Confusion — What 128 Actually Means

### In coremltools: group_size controls channel grouping, NOT LUT size

**File:** `optimize/coreml/_config.py`, line 715:
```python
group_size: int = field(default=32)
```

When `granularity=CompressionGranularity.PER_GROUPED_CHANNEL`, the `group_size` parameter controls how many output channels share the same LUT. For example:

- **4-bit palette, group_size=128**: Each group of 128 output channels shares one 16-entry LUT. The LUT still has 16 entries (4-bit indices), NOT 128 entries.
- **2-bit palette, group_size=64**: Each group of 64 output channels shares one 4-entry LUT. The LUT still has 4 entries (2-bit indices), NOT 64 entries.

The group_size is completely independent of the palette bit width. It's the number of channels per group, not the number of LUT entries.

### In the ANE compiler: palette_group_size is a hardware register field

The ANE hardware has a `palette_group_size` field in the kernel config register:

```
hw.ne_config.ane_ne_config.kernel_cfg2.palette_group_size
```

This is the log2 of the number of output channels per LUT group. It's used in the multi-palette LUT buffer size calculation:

```
palette_lut_bfr_size = num_groups * ceil(
    ceil(cout_fld, (1 << palette_group_size)) * (1 << palette_lut_size_log2),
    dram_alignment
)
```

Where `palette_lut_size_log2` = log2 of the total bytes in one LUT (NUM_PALETTES * sizeof(LUT_entry_type)).

### In PyTorch GPTQ: processing_group_size = 128 (completely unrelated)

**File:** `optimize/torch/layerwise_compression/algorithms.py`:
```python
processing_group_size: int = 128  # GPTQ weight update block size
```

This is the number of weight columns processed together during GPTQ compression. Completely unrelated to palette LUT size.

---

## 4. The MIL-Level Palette Bit Width Constraint

The single most authoritative constraint string:

```
MIL conversion error: unknown data format. Only UInt1, 2, 3, 4, 6, 8 are supported
```

This is the MIL-to-ANE IR conversion gate. No UInt7, no 7-bit indices, no 128-entry LUT.

Additionally, the ANEC MLIR dialect constraint:

```
must be 4D/5D memref of 32-bit float or 16-bit float or 8-bit signed integer
or 8-bit unsigned integer or 2/4/6/8-bit unsigned integer values, but got
```

At the ANEC dialect level, kernel weights can be: fp32, fp16, int8, uint8, or **2/4/6/8-bit unsigned integers**. Note: 1-bit and 3-bit are NOT in this constraint, meaning they must be upcast before reaching the ANEC dialect.

---

## 5. The "1 and 2 bit palettes not supported" Error — FULL CONTEXT

Two separate error paths, BOTH conv-specific:

1. **`Error: 1 and 2 bit palettes not supported.`** — From the ANE Conv Validator `ValidateKernelFormat`. Convolution kernels reject 1-bit and 2-bit palette indices.

2. **`Only 2-bit, 4-bit, 6-bit and 8-bit palettization for conv are supported!`** — From a different conv path that supports 2-bit as minimum.

**Key insight**: Both errors are **conv-specific**. Non-conv ops (linear, gather, etc.) can use 1-bit palettes freely. The "1 and 2 bit" error likely fires for specific hardware versions or conv subtypes.

---

## 6. The 3-bit Palette Upcasting System

Functions found in binary:
- `Is3bitPaletteKernelFormat(ANEPaletteFormat)` — checks if format is 3-bit palette
- `NeedsUpcastingFrom3bitPaletteTo4bitPalette(ANEHalParameters, ANEPaletteFormat)` — checks if current HW needs upcast
- `GetUpcasted4bitPaletteFormatFrom3bitPaletteFormat(ANEPaletteFormat)` — returns the 4-bit equivalent

On hardware that doesn't natively support 3-bit indices:
- Palette8 (3-bit, 8 LUT entries) → Palette16 (4-bit, 16 LUT entries)
- The LUT is expanded: 8 real entries + 8 don't-care entries
- Weight indices are widened from 3 bits to 4 bits
- **Transparent** — MIL level sees Palette8, hardware sees Palette16

---

## 7. Version-Gated Palette Support

```
failed to downgrade: requested target version is {0}, but 3-bit palettization is only supported from version {1}
failed to downgrade: requested target version is {0}, but 6-bit palettization is only supported from version {1}
failed to downgrade: requested target version is {0}, but uint3 data is only supported from version {1}
failed to downgrade: requested target version is {0}, but uint6 data is only supported from version {1}
```

3-bit and 6-bit palettization require minimum hardware versions. On older hardware:
- 3-bit → upcasted to 4-bit transparently
- 6-bit → likely rejected or upcasted to 8-bit

1-bit and 2-bit are NOT version-gated — universally supported (or at least from the earliest palettization-capable version).

---

## 8. Multi-Palette Mode (Per-Channel LUTs) — The Real "128" Connection

### Key functions

```
CanUseMultiPaletteMode(ANEHalParameters, bool, ANEKernel)
EnableKernelSplitForMultiPaletteLUT
SetMultiPaletteEnable()            — v4, v5, v6, v7, v8, v10, v11, v17, v19, v20, v26
SetMultiPaletteSizeOneLut(size_t)  — v4, v5, v6, v7, v8, v10, v11, v17, v19, v20, v26
SetPaletteBlockSize(uint32_t)      — v4, v5, v6, v7, v10, v11, v17, v19, v20, v26
SetPaletteGroupSize(size_t)        — v4, v5, v6, v7, v10, v11, v17, v19, v20, v26
```

### How multi-palette works

In multi-palette mode, different groups of output channels can have different LUTs. The key parameters:

- **palette_vector_size** (also called `cluster_dim` in CoreML): The number of consecutive channels that share one LUT entry. Must be a power of 2. Consecutive `palette_vector_size` channels must use the same LUT.
- **palette_group_size**: The log2 of the number of channels per LUT group. Controls how many separate LUTs exist.
- **num_luts**: The total number of separate LUTs, derived from `cout / (1 << palette_group_size)`.

The per-cout LUT has rank 3: `(num_luts, num_palettes, vector_size)`

### Multi-palette constraints

```
Error: should only have a single LUT in aligned format unless multi-palette mode is used.
Error: Instruction corresponding to oplayer %s failed MultiPaletteMode validation.
Invalid multi-palette LUT configuration.
OCG size must be a multiple of palette_vector_size.
Consecutive palette_vector_size channels must make use of the same palette LUT
MIL conversion error: the rank of per-cout LUT should be 3 (num_luts, num_palettes, vector_size)
```

### Per-channel palette LUT constraints for conv

```
Only per-cout LUT is supported!
Kernel quant folding is not yet supported under per-channel palette LUTs.
Mutable weight for per-channel palette is not supported!
NEConv has per-cout-palette-lut, cannot split the kernel of this conv under specified tile sizes.
```

---

## 9. The Palettized Bits Hardware Register

```
ane_ne_kernel_cfg_palettized_bits_bit8_v8
ane_ne_kernel_cfg_palettized_bits_bit8_v10
ane_ne_kernel_cfg_palettized_bits_bit8_v11
ane_ne_kernel_cfg_palettized_bits_bit8_v17
ane_ne_kernel_cfg_palettized_bits_bit8_v19
ane_ne_kernel_cfg_palettized_bits_bit8_v20
ane_ne_kernel_cfg_palettized_bits_bit8_v26
```

All visible enum values are `bit8` variants. This is the palettized_bits field in the ANE NE config register. The `bit8` naming means "8 possible values for the enum field", NOT that all palettized data is 8-bit.

The formula for one LUT entry size in the DMA transfer:
```
size_one_lut = palettized_bits + palette_block_size - sparse_binary + (kernel_fmt == fp16)
```

Where:
- `palettized_bits` = index bit width (1, 2, 3, 4, 6, or 8)
- `palette_block_size` = log2 of the palette block size
- `sparse_binary` = 1 if sparse mode enabled
- `(kernel_fmt == fp16)` = 1 if LUT entries are fp16

LUT size validation:
```
(1 << palette_lut_size_log2) <= hal_params.ne_palette_lut_size_in_bytes
num_luts_per_ocg << size_one_lut <= hal_params.ne_palette_lut_size_in_bytes
```

---

## 10. LUT Entry Data Types

The palette LUT can store entries in multiple formats:

| Template Type | C++ Type | LUT Entry Type | Size |
|---|---|---|---|
| `h` | `uint8_t` | uint8 | 1 byte |
| `a` | `int8_t` | int8 | 1 byte |
| `Dh` | `__bf16` | BF16 | 2 bytes |
| `f` | `float` | FP32 | 4 bytes |
| `e4m3_t` | custom | FP8 E4M3 | 1 byte |

**IMPORTANT DISTINCTION**: BF16 (`__bf16`) exists as a LUT entry type inside ANECompiler. This means the ANE hardware can read BF16-valued lookup table entries as part of the palette decompression pipeline. But BF16 is NOT a kernel format — you can't have BF16 activations or BF16 weights directly. The BF16 values only exist inside the LUT, and are decompressed to fp16 or fp32 during the palette lookup.

This reconciles the apparent contradiction: BF16 IS used in the ANE palette subsystem (as LUT entry storage), but CoreML does NOT support bf16 as a model data type or kernel format.

---

## 11. ANEC MLIR Dialect — Palette Attributes

| ANEC Op | Palette Attribute | Constraint |
|---|---|---|
| `anec.convolution` | `kernel_palettized_LUT` | Dense elements, rank 0-6 |
| `anec.convolution` | `kernel_mutable_palettized_LUT` | Dictionary of named attrs |
| `anec.deconvolution` | `kernel_palettized_LUT` | Dense elements, rank 0-6 |
| `anec.deconvolution` | `kernel_mutable_palettized_LUT` | Dictionary of named attrs |
| `anec.linear` | `kernel_lut` | Dense elements, rank 0-6 |

---

## 12. ANE-Specific Palette Constraints (coremltools Source)

### Scale reordering for ANE
**File:** `converters/mil/mil/passes/defs/optimize_quantization.py`, lines 1009-1121:

The `reorder_lut_per_channel_scale` graph pass moves per-channel scales to AFTER the linear/matmul/conv op for ANE compatibility:
```
Before (not ANE-friendly):
  weight = constexpr_lut_to_dense()
  weight = constexpr_blockwise_shift_scale(weight)
  output = linear/matmul/conv(x, weight)

After (ANE-friendly):
  weight = constexpr_lut_to_dense()
  unscaled_output = linear/matmul(x, weight)
  output = mul(unscaled_output, scale)
```

### LUT deduplication for ANE
**File:** `converters/mil/mil/passes/defs/cleanup/const_deduplication.py`, lines 99-110:

4-bit and 6-bit palette LUT values (16/64 entries) are small and can cause weight unsharing after ANE pre-compilation if not aggressively deduplicated.

---

## 13. Version-Specific Palette Validation (ValidateTd)

| Version | Chip | ValidatePaletteBlockSize | SetPaletteBlockSize | SetPaletteGroupSize | SetMultiPalette* |
|---|---|---|---|---|---|
| v4 | A11 | ❌ | ✅ | ✅ | ✅ |
| v5 | ? | ❌ | ✅ | ✅ | ✅ |
| v6 | ? | ❌ | ✅ | ✅ | ✅ |
| v7 | A12? | ❌ | ✅ | ✅ | ✅ |
| v8 | A12X/Z | ❌ | ✅ | ✅ | ✅ |
| v10 | A14/M1? | ❌ | ✅ | ✅ | ✅ |
| v11 | A15? | ❌ | ✅ | ✅ | ✅ |
| v17 | M1 Pro/Max? | ❌ | ✅ | ✅ | ✅ |
| **v19** | **M2** | **✅** | ✅ | ✅ | ✅ |
| v20 | M3? | ✅ | ✅ | ✅ | ✅ |
| v26 | M4? | ✅ | ✅ | ✅ | ✅ |

All versions from v4 through v26 have `SetPaletteBlockSize`, `SetPaletteGroupSize`, `SetMultiPaletteEnable`, and `SetMultiPaletteSizeOneLut`. Only v19, v20, v26 additionally have `ValidatePaletteBlockSize`.

---

## 14. Resolving the "Palette128" Claim

You mentioned having "Palette128 which works on even 4-bit." After exhaustive analysis of coremltools source and observable ANE behavior, here are the possible explanations:

### Most likely: Confusion between group_size and LUT entries

When you create a palettized model with `nbits=4, group_size=128`, CoreML creates a 4-bit palette (16 LUT entries) where every group of 128 output channels shares its own LUT. You may be seeing "128" in your model's weight spec and interpreting it as 128 LUT entries, when it actually means 128 channels per group.

In a CoreML model, the per-grouped-channel LUT has shape `[num_groups, 16, vector_size]` where `num_groups = cout / group_size`. The `128` is the `group_size`, not the LUT size.

### Less likely but possible: Newer ANECompiler version

Apple may add Palette128/UInt7 support in future ANE compiler versions. To verify on your system, check the ANE compiler version and palette support through the Core ML framework.

### Unlikely: Palette256 with 128 used entries

CoreML could encode a 128-entry palette as Palette256 (8-bit indices) where only 128 of 256 LUT entries are used and the rest are don't-care. But this would waste index bits (8 instead of 7) and there's no evidence of this optimization in the code.

---

## 15. bf16 — Complete Status

| Context | bf16 Support | Evidence |
|---|---|---|
| **CoreML model format** | ❌ NOT supported | `constexpr_lut_to_dense` type_domains only has `(int8, uint8, fp16, fp32)` |
| **CoreML conversion** | ❌ Raises KeyError | `test_weight_only_quantization_bfloat16_not_support` |
| **ANE kernel format** | ❌ NOT a format | No bf16 in ANE kernel format enum |
| **ANE LUT entry type** | ✅ Supported | `__bf16` / `Dh` in IrCompressedConstData |
| **ANE channel config** | ✅ Present | `ane_common_ch_cfg_in_fmt_bf16` for v17, v19, v20, v26 |
| **PyTorch training** | ✅ Fake palettize | `lut_dtype="b16"` for training simulation |
| **MIL IR type system** | ⚠️ Reserved name | `"bf16"` is reserved but has no type definition |

**Summary**: BF16 exists in the ANE hardware as a LUT entry storage format and channel data format for v17+ hardware, but CoreML does NOT expose it as a model-level data type. You cannot create a CoreML model with bf16 weights, bf16 activations, or bf16 LUT entries. The hardware can internally use bf16 for palette LUT entries, but this is an implementation detail, not a user-facing feature.

---

## Summary: Complete Palette Format Table (CORRECTED)

| Palette Format | Index Bit Width | LUT Entries | MIL Type | Sparse Variant | Conv Support | Version Gating |
|---|---|---|---|---|---|---|
| Palette2 | 1 bit | 2 | MIL::UInt1 | Palette2Sparse | ❌ Rejected by conv validators | None (universal) |
| Palette4 | 2 bit | 4 | MIL::UInt2 | Palette4Sparse | ✅ Minimum for conv | None (universal) |
| Palette8 | 3 bit | 8 | MIL::UInt3 | Palette8Sparse | ✅ (upcast to 4-bit on old HW) | Min version required |
| Palette16 | 4 bit | 16 | MIL::UInt4 | Palette16Sparse | ✅ | None (universal) |
| Palette64 | 6 bit | 64 | MIL::UInt6 | Palette64Sparse | ✅ | Min version required |
| Palette256 | 8 bit | 256 | MIL::UInt8 | Palette256Sparse | ✅ (but not with quantization) | None (universal) |

### Non-palette kernel formats
fp16, fp32, Int4, Int8, UInt8, Sparse, SparsePalettized, Quantized, QuantizedSparse, QuantizedPalettized, QuantizedSparsePalettized

### What "128" means in palette context
- **group_size=128**: 128 output channels per LUT group (per-grouped-channel palettization). LUT still has 2^nbits entries.
- **processing_group_size=128**: GPTQ column processing block size. Completely unrelated to palette LUT size.
- **calibration_nsamples=128**: Number of calibration samples for sensitive k-means. Unrelated.
- **NOT 128 LUT entries**: No 7-bit palette format exists in ANECompiler or coremltools.
