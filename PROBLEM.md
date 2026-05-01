# Why the current `model.mlpackage` still does not compile/open

Date: 2026-05-01

Scope: local analysis of the current package on disk. No web searches were used. No source files were modified.

Artifact inspected:

```text
/Users/pkhairkh/Downloads/MILLer/output/qwen3-0.6b/model.mlpackage
```

Current embedded package files:

```text
output/qwen3-0.6b/model.mlpackage/Data/com.apple.CoreML/model.mlmodel
mtime: May 1 16:17:43 2026
size: 717641 bytes

output/qwen3-0.6b/model.mlpackage/Manifest.json
mtime: May 1 16:17:43 2026
size: 544 bytes
```

## Summary

This is a newer package than the previous report. The failure mode changed again.

The previous blockers are fixed in this package:

```text
unresolved references: 0
slice_by_index mask dtype: fixed as bool[4]
zero-dimension outputs: 0
```

The current hard compiler blocker is a missing required `concat` parameter:

```text
sir_14_layer_0_self_attn_rotated:
  type: concat
  inputs: values, axis
  missing: interleave
```

Core ML rejects the first `concat`:

```text
Required param 'interleave' is missing
```

The first failing concat is the RoPE rotate-half concatenation:

```text
concat(values = [sir_14_layer_0_self_attn_neg_x2, sir_14_layer_0_self_attn_x1], axis = 3)
```

The intended behavior is ordinary non-interleaved concat, so the missing parameter should be explicitly serialized as:

```text
interleave = false
```

All `56` concat operations in this package have only:

```text
axis, values
```

and omit:

```text
interleave
```

## Compiler evidence

Command:

```bash
rm -rf /tmp/miller-problem-coremlc
mkdir -p /tmp/miller-problem-coremlc
xcrun coremlcompiler compile output/qwen3-0.6b/model.mlpackage /tmp/miller-problem-coremlc
```

Result:

```text
coremlcompiler: error: Failed to parse the model specification. Error: Unable to parse ML Program: in operation sir_14_layer_0_self_attn_rotated: Required param 'interleave' is missing
```

## Current model contents

The embedded model is:

```text
coremltools: 9.0
coremltools path: /opt/homebrew/lib/python3.12/site-packages/coremltools/__init__.py
specificationVersion: 10
modelType: mlProgram
top input:  sir_0_input_ids, multiArrayType [1, 512], int32
top output: sir_900_output, multiArrayType [1, 512, 151936], fp16
function: main
block: CoreML9
block inputs: 0
block outputs: sir_900_output
operations: 3245
```

Operation histogram:

```text
mul             902
add             451
const           343
reshape         224
linear          197
abs             113
reduce_max      113
maximum         113
real_div        113
reduce_mean     113
rsqrt           113
transpose       112
slice_by_index  112
concat           56
tile             56
matmul           56
softmax          28
silu             28
gather            1
identity          1
```

Important current counts:

```text
unresolved references: 0
concat ops: 56
concat input keys: ('axis', 'values') on all 56 ops
slice masks encoded as bool[4]
zero-dimension outputs: 0
```

## First failing op

The compiler stops at:

```text
idx 377
type: concat
output: sir_14_layer_0_self_attn_rotated
```

Serialized operation:

```text
type: "concat"
inputs {
  key: "values"
  value {
    arguments {
      name: "sir_14_layer_0_self_attn_neg_x2"
    }
    arguments {
      name: "sir_14_layer_0_self_attn_x1"
    }
  }
}
inputs {
  key: "axis"
  value {
    arguments {
      value {
        type: tensor<int32>
        immediateValue: 3
      }
    }
  }
}
outputs {
  name: "sir_14_layer_0_self_attn_rotated"
  type: tensor<float16, [1,16,512,128]>
}
```

The operation is missing:

```text
interleave: tensor<bool>(false)
```

Core ML reports:

```text
Required param 'interleave' is missing
```

## Immediate upstream context

Layer 0 leading into the first failing RoPE concat:

```text
idx 371 transpose -> sir_11_layer_0_self_attn [1,16,512,128]
idx 372 transpose -> sir_12_layer_0_self_attn [1,8,512,128]
idx 373 transpose -> sir_13_layer_0_self_attn [1,8,512,128]
idx 374 slice     -> sir_14_layer_0_self_attn_x1 [1,16,512,64]
idx 375 slice     -> sir_14_layer_0_self_attn_x2 [1,16,512,64]
idx 376 mul       -> sir_14_layer_0_self_attn_neg_x2 [1,16,512,64]
idx 377 concat    -> sir_14_layer_0_self_attn_rotated [1,16,512,128]
idx 378 mul       -> sir_14_layer_0_self_attn_x_cos [1,16,512,128]
idx 379 mul       -> sir_14_layer_0_self_attn_rotated_sin [1,16,512,128]
idx 380 add       -> sir_14_layer_0_self_attn [1,16,512,128]
```

The rotate-half shapes and slice mask types are now correct:

```text
x1:  [1,16,512,64]
x2:  [1,16,512,64]
rot: [1,16,512,128]

begin_mask: bool[4] [true, true, true, false]
end_mask:   bool[4] [true, true, true, false] or [true, true, true, true]
```

The compiler does not reach later attention because it rejects the missing `concat.interleave` parameter first.

## Repeated pattern

Every concat op currently omits `interleave`:

```text
concat count: 56
concat input keys: ('axis', 'values') count 56
```

For the rotate-half pattern, the intended concat is non-interleaved:

```text
concat(-x2, x1, axis=3, interleave=false)
```

So the repair is likely mechanical: include an explicit boolean scalar `false` for every emitted concat unless an interleaved concat is truly intended.

## Local CoreMLTools schema evidence

Local CoreMLTools defines `concat` with:

```text
values: required
axis: required
interleave: const<bool> optional, default false
```

The local op definition has:

```python
input_spec = InputSpec(
    values=TupleInputType(),
    axis=TensorInputType(const=True, type_domain=types.int32),
    interleave=TensorInputType(const=True, optional=True, type_domain=types.bool)
)

def default_inputs(self):
    return DefaultInputs(
        interleave=False,
    )
```

However, the current serialized ML Program is rejected by `coremlcompiler` with:

```text
Required param 'interleave' is missing
```

So for this raw protobuf writer and target compiler path, relying on the default is not accepted. The package should serialize `interleave` explicitly.

## What is fixed now

This package has moved past the previous hard blockers:

```text
old blocker: unresolved RoPE table names
current status: fixed, unresolved references = 0

old blocker: fill_like(x=..., value=...)
current status: fixed, fill_like no longer appears

old blocker: zero-dimension rotate-half slices
current status: fixed, zero-dimension outputs = 0

old blocker: slice_by_index masks encoded as int32 bitmasks
current status: fixed, masks are bool[4]
```

The current failure is now a missing `concat` parameter.

## Why this prevents opening the package

Core ML validates required operation parameters before compiling the ML Program. The first `concat` has:

```text
values
axis
```

but the compiler requires:

```text
values
axis
interleave
```

Because `interleave` is missing, Core ML rejects the package before it reaches later attention, residual, MLP, final norm, or `lm_head` operations.

## Root cause

The current root cause is incomplete `concat` serialization.

The generator emits:

```text
concat(values=[...], axis=3)
```

but the compiler requires:

```text
concat(values=[...], axis=3, interleave=false)
```

Even though CoreMLTools exposes a default for `interleave`, this manually serialized ML Program does not get that default applied before compiler validation.

## Recommended repair direction

When serializing `concat`, always emit the `interleave` input:

```text
interleave: tensor<bool>(false)
```

unless the graph explicitly needs interleaved concat, in which case emit:

```text
interleave: tensor<bool>(true)
```

Add pre-write operation schema validation:

```text
for every concat:
  require values
  require axis
  require interleave bool scalar
```

The first package rejection should happen before writing:

```text
idx 377 sir_14_layer_0_self_attn_rotated:
  concat is missing required serialized parameter "interleave"
```

## Bottom line

The current package has:

```text
No successful Core ML compile.
No unresolved graph references.
No fill_like x/ref_tensor issue.
No zero-dimension outputs.
No slice_by_index int32 mask issue.
56 concat ops missing interleave.
```

The first compiler blocker is:

```text
op#377 sir_14_layer_0_self_attn_rotated:
  Required param 'interleave' is missing.
```

The model cannot compile/open because `concat` is serialized without the explicit `interleave` boolean parameter required by the Core ML compiler.
