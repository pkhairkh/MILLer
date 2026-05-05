# MILLer Test Kit — M2 Mac + Qwen3-0.6B

## Quick Start

```bash
# 1. Install Python dependencies
pip install torch transformers coremltools numpy

# 2. Set the path to your MILLer repo (if running from outside)
export MILLER_ROOT=/path/to/MILLer

# 3. Run the full test suite
./scripts/test_kit.sh

# Or run individual phases:
./scripts/test_kit.sh --phase prereqs      # Check prerequisites only
./scripts/test_kit.sh --phase build        # Build ane-compile
./scripts/test_kit.sh --phase synthetic    # Compile all synthetic task specs
./scripts/test_kit.sh --phase bridge       # Test Python bridge emitters
./scripts/test_kit.sh --phase qwen3        # trace-compile Qwen3-0.6B
./scripts/test_kit.sh --phase knowledge    # Validate knowledge seeds & store
./scripts/test_kit.sh --phase ir-pipeline  # IR pipeline validation
./scripts/test_kit.sh --phase ane          # ANE fastpath & compute plan

# Skip already-done steps:
./scripts/test_kit.sh --skip-build --skip-download

# Verbose output:
./scripts/test_kit.sh --verbose
```

## Pipeline

```
TOML task spec ──→ SIR ──→ AIR ──→ MIR ──→ Bridge payload JSON ──→ Python bridge ──→ .mlpackage
                    │                                    │
                    └── Knowledge-guided passes ─────────┘

For real models (Qwen3-0.6B):
  HuggingFace model ──→ trace-compile ──→ traced graph ──→ SIR ──→ ... ──→ .mlpackage
```

## Test Phases

| Phase | Name | What it tests |
|-------|------|---------------|
| 1 | Prerequisites | M2 Mac, Rust, Python, coremltools, bridge.py, task specs |
| 2 | Build | `cargo build --release -p ane-cli`, verify subcommands |
| 3 | Synthetic | Compile all 8 TOML task specs, verify mlpackage + manifest |
| 4 | Bridge | Test all 7 Python bridge emitters directly, error handling |
| 5 | Qwen3-0.6B | trace-compile, verify, profile, KV-cache variant |
| 6 | Knowledge | Seed schema validation, import, store index, query |
| 7 | IR Pipeline | SIR/AIR/MIR variant counts, legality rules |
| 8 | ANE | Compute plan inspection, op family matrix, CPU-only ops |

## Key Commands

| Task | Command |
|------|---------|
| Compile a task spec | `ane-compile compile -i spec.toml -o output/ --bridge python/bridge.py` |
| Sharded compile | `ane-compile compile-sharded -i spec.toml -o output/ --bridge python/bridge.py --seed 42` |
| Proto-direct (Rust-only) | `ane-compile compile-sharded -i spec.toml -o output/ --proto-direct` |
| Trace a HF model | `ane-compile trace-compile -m Qwen/Qwen3-0.6B -o output/ --bridge python/bridge.py` |
| Verify an mlpackage | `ane-compile verify -m model.mlpackage -o result.json --bridge python/bridge.py` |
| Profile | `ane-compile profile -m model.mlpackage -o result.json --bridge python/bridge.py` |
| Import knowledge | `ane-compile import -s knowledge/ --store ./store --validate` |
| Query knowledge | `ane-compile query --store ./store --filter "ane_legal"` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILLER_ROOT` | `../..` from script | Path to MILLer repository |
| `TEST_WORKDIR` | `./test_work` | Working directory for artifacts |
| `QWEN3_MODEL_ID` | `Qwen/Qwen3-0.6B` | HuggingFace model identifier |
| `SEED` | `42` | Random seed for reproducibility |
| `VERBOSE` | `0` | Enable debug output |

## Requirements

- **macOS** on Apple Silicon (M2+)
- **Rust** toolchain (per `rust-toolchain.toml`)
- **Python** 3.10+ with: torch, transformers, coremltools, numpy
- **~10 GB** free disk space (Qwen3 model weights + artifacts)
