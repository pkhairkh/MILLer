# Document Map

## Canonical Source Set

| Topic | Canonical document | Use for |
|---|---|---|
| Framework analysis | `01-binary-research/ane-framework-binary-analysis.md` | Framework inventory, compiler/runtime roles, and constraint taxonomy. |
| Hardware limits | `02-hardware-and-limits/hardware-versions-limits-and-op-support.md` | ANE version mapping, HAL parameters, dtype support, register checks, op landing summary. |
| Placement constraints | `03-placement-and-compiler/mil-to-ane-placement-constraint-system.md` | MIL validation, placement dialect, unit validation, dynamic-shape and memory constraints. |
| Fusion and boundaries | `03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md` | Fusion patterns, execution engines, ActiveNE/OCG allocation, graph breaks. |
| Op support matrix | `04-operation-support/per-op-per-family-support-matrix.md` | MIL-to-ANEC mappings and family-specific support comparisons. |
| MIL op plannability | `04-operation-support/mil-ops-ane-plannability-analysis.md` | Documented coremltools MIL ops grouped by heuristic ANE plannability score. |
| Palettization | `05-palettization/palette-bit-widths-and-lut-formats.md` | Palette bit widths, LUT formats, sparse/multi-palette behavior, bf16 and Palette128 corrections. |

## Semantic Flow

The documents now follow the compiler stack from evidence to decision:

1. Binary/framework evidence establishes what was inspected.
2. Hardware version and limit notes define the target constraints.
3. Placement notes explain how a MIL op becomes eligible for ANE.
4. Fusion/boundary notes explain why eligible ops may still split or fall back.
5. Op support matrices make the compiler-derived decision surface searchable.
6. MIL op plannability notes provide a docs-derived heuristic view across the full public MIL op set.
7. Palette notes isolate quantization-specific rules that cut across ops.

## Archive Policy

Older palette drafts are kept under `99-archive/palette-revisions/` because they capture the reasoning path, but they are no longer the source of truth. Prefer the v3 canonical palette document unless you are auditing how a conclusion changed.
