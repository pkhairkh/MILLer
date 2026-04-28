# ANE Constraint Research Notes

This directory is organized by research function rather than by capture order.

## Reading Order

1. [Orientation Map](00-orientation/document-map.md) - scope, canonical sources, and where each topic lives.
2. [ANE Framework Analysis](01-binary-research/ane-framework-binary-analysis.md) - source/framework inventory and compilation context.
3. [Hardware Versions, Limits, and Op Support](02-hardware-and-limits/hardware-versions-limits-and-op-support.md) - ANE version map, HAL parameters, dtype support, and register-level constraints.
4. [MIL-to-ANE Placement Constraint System](03-placement-and-compiler/mil-to-ane-placement-constraint-system.md) - validation chain, placement dialect, dynamic-shape kill switches, and memory constraints.
5. [Fusion Boundaries and Resource Allocation](03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md) - execution engines, fusion rules, ActiveNE, OCG, and graph-break causes.
6. [Per-Op Per-Family Support Matrix](04-operation-support/per-op-per-family-support-matrix.md) - consolidated op mapping and family support table.
7. [MIL Ops ANE Plannability Analysis](04-operation-support/mil-ops-ane-plannability-analysis.md) - documented coremltools MIL ops scored by heuristic ANE plannability.
8. [Palette Bit Widths and LUT Formats](05-palettization/palette-bit-widths-and-lut-formats.md) - canonical palette/palettization findings.

## Directory Layout

| Directory | Purpose |
|---|---|
| `00-orientation/` | Indexes, maps, and reading guidance. |
| `01-binary-research/` | ANE framework research and compilation findings. |
| `02-hardware-and-limits/` | Hardware versions, HAL limits, dtype support, and version gates. |
| `03-placement-and-compiler/` | Compiler placement, validation, fusion, boundaries, and resource allocation. |
| `04-operation-support/` | MIL op to ANEC op support and per-family matrices. |
| `05-palettization/` | Palette formats, LUT constraints, and quantization-specific notes. |
| `99-archive/` | Superseded drafts retained for provenance. |

## Canonical Notes

- The v3 palette document in `05-palettization/` supersedes the older palette drafts in `99-archive/palette-revisions/`.
- The per-op support matrix is a compiler-derived consolidated index; the MIL ops plannability analysis is a docs-derived heuristic scoring table.
- This folder is currently gitignored by the parent repository.
