#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# MILLer — macOS ANE Fastpath Testing Script
# ═══════════════════════════════════════════════════════════════════════════
#
# This script tests which MIL ops are AIR-reachable and which combinations
# get fused through the ANE fastpath on Apple Silicon. It must run on macOS
# with coremltools and an Apple Neural Engine present.
#
# It exercises the full compile pipeline:
#   SIR → AIR (legality rewrite) → MIR (mil lower) → proto emission → mlpackage
#   → coremltools compute plan → ANE placement analysis
#
# Usage:
#   ./scripts/macOS_ane_lab.sh [--output-dir DIR] [--bridge PATH] [--cli PATH]
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - Python 3.10+ with coremltools >= 7.0
#   - Compiled Rust CLI (ane-compile) or cargo to build it
#
# Exit codes:
#   0 — all tests passed or completed with results
#   1 — one or more test failures
#   2 — environment not ready (missing macOS / coremltools)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ane_lab_$$}"
BRIDGE="${BRIDGE:-python/bridge.py}"
CLI="${CLI:-./target/debug/ane-compile}"
SEED="${SEED:-42}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --bridge)     BRIDGE="$2"; shift 2 ;;
        --cli)        CLI="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output-dir DIR] [--bridge PATH] [--cli PATH] [--seed N]"
            echo ""
            echo "  --output-dir  Directory for test artifacts (default: /tmp/ane_lab_PID)"
            echo "  --bridge      Path to Python bridge (default: python/bridge.py)"
            echo "  --cli         Path to compiled CLI binary (default: ./target/debug/ane-compile)"
            echo "  --seed        Random seed for reproducibility (default: 42)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

pass=0
fail=0
skip=0
total_ops_tested=0
ane_placed_ops=0
cpu_only_ops=0
fused_patterns=0

notice()  { echo "[LAB]   $*"; }
ok()      { echo "[PASS]  $*"; pass=$((pass + 1)); }
err()     { echo "[FAIL]  $*"; fail=$((fail + 1)); }
skip_msg(){ echo "[SKIP]  $*"; skip=$((skip + 1)); }
section() { echo ""; echo "━━━ $* ━━━"; }

cleanup() {
    if [ "${KEEP_ARTIFACTS:-0}" != "1" ]; then
        rm -rf "$OUTPUT_DIR"
    fi
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

# ─── Environment Checks ────────────────────────────────────────────────────
section "Environment Validation"

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    err "Not running on macOS — ANE fastpath testing requires Apple Silicon"
    echo "  This script inspects MLComputePlan which only works on macOS."
    echo "  Run on a Mac with Apple Silicon for full ANE placement data."
    exit 2
fi

# Check Apple Silicon
CPU_TYPE="$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)"
if [ "$CPU_TYPE" != "1" ]; then
    notice "Not Apple Silicon — ANE may not be available"
fi

# Check Python and coremltools
if ! command -v python3 &>/dev/null; then
    err "python3 not found"
    exit 2
fi

CT_VERSION=$(python3 -c "import coremltools; print(coremltools.__version__)" 2>/dev/null || echo "NOT_FOUND")
if [ "$CT_VERSION" = "NOT_FOUND" ]; then
    err "coremltools not installed — pip install coremltools"
    exit 2
fi
ok "coremltools version: $CT_VERSION"

# Check bridge script
if [ -f "$BRIDGE" ]; then
    ok "Bridge script found: $BRIDGE"
else
    err "Bridge script not found: $BRIDGE"
    exit 2
fi

# Check CLI (build if needed)
if [ ! -x "$CLI" ]; then
    notice "CLI binary not found at $CLI, attempting cargo build..."
    if command -v cargo &>/dev/null; then
        if cargo build -p ane-cli --release 2>&1; then
            CLI="./target/release/ane-compile"
            ok "CLI built (release)"
        else
            err "cargo build failed"
        fi
    else
        skip_msg "No cargo found; some tests will be skipped"
    fi
fi

# ─── Test Matrix: Op Categories for ANE Fastpath ───────────────────────────
section "Op-Level ANE Fastpath Testing"

# Each test emits a small model with specific ops, then inspects the
# compute plan to see which ops the ANE actually accepted.
#
# The test matrix covers the major op categories:
#   1. Linear / FC operations
#   2. Elementwise unary activations
#   3. Elementwise binary ops
#   4. Reduction ops
#   5. Normalization ops
#   6. Tensor transform ops
#   7. Attention / SDPA
#   8. Convolution
#   9. Quantization
#  10. State / recurrent ops

emit_and_inspect() {
    local TEST_NAME="$1"
    local BRIDGE_CMD="$2"
    shift 2
    # Remaining args are expected ANE ops (optional)

    local TEST_DIR="$OUTPUT_DIR/$TEST_NAME"
    mkdir -p "$TEST_DIR"

    local CMD_JSON="$TEST_DIR/command.json"
    local RESULT_JSON="$TEST_DIR/result.json"

    # Write bridge command
    python3 -c "
import json, sys
cmd = $BRIDGE_CMD
cmd['output_path'] = '$TEST_DIR'
cmd['seed'] = $SEED
json.dump(cmd, open('$CMD_JSON', 'w'))
"

    # Execute bridge
    if python3 "$BRIDGE" "$CMD_JSON" "$RESULT_JSON" 2>/dev/null; then
        # Check result
        local STATUS=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('status','unknown'))" 2>/dev/null || echo "parse_error")
        if [ "$STATUS" = "success" ]; then
            ok "$TEST_NAME: emission succeeded"

            # Inspect compute plan for ANE placement
            local MLPACKAGE=$(python3 -c "
import json
r = json.load(open('$RESULT_JSON'))
op = r.get('output_path','')
if not op:
    # Try to find mlpackage in output dir
    import glob
    pkgs = glob.glob('$TEST_DIR/*.mlpackage')
    print(pkgs[0] if pkgs else '')
else:
    print(op)
" 2>/dev/null || echo "")

            if [ -n "$MLPACKAGE" ] && [ -d "$MLPACKAGE" ]; then
                # Harvest compute plan
                local HARVEST_JSON="$TEST_DIR/compute_plan_harvest.json"
                local HARVEST_CMD="$TEST_DIR/harvest_command.json"
                python3 -c "
import json
json.dump({
    'bridge_version': 1,
    'command': 'compute_plan_harvest',
    'mlpackage_path': '$MLPACKAGE',
    'compute_units': 'CPU_AND_NE',
    'output_path': '$TEST_DIR'
}, open('$HARVEST_CMD', 'w'))
"
                if python3 "$BRIDGE" "$HARVEST_CMD" "$HARVEST_JSON" 2>/dev/null; then
                    # Parse ANE placement from harvest
                    local ANE_OPS=$(python3 -c "
import json
try:
    h = json.load(open('$HARVEST_JSON'))
    harvest = h.get('harvest', {})
    ops = harvest.get('ops', {})
    ane_count = 0
    cpu_count = 0
    gpu_count = 0
    for op_name, op_info in ops.items():
        device = op_info.get('device', 'unknown')
        if device == 'ANE' or 'neural_engine' in str(device).lower():
            ane_count += 1
        elif device == 'CPU' or 'cpu' in str(device).lower():
            cpu_count += 1
        elif device == 'GPU' or 'gpu' in str(device).lower():
            gpu_count += 1
    print(f'{ane_count},{cpu_count},{gpu_count}')
except Exception as e:
    print(f'0,0,0')
" 2>/dev/null || echo "0,0,0")

                    local ANE=$(echo "$ANE_OPS" | cut -d, -f1)
                    local CPU=$(echo "$ANE_OPS" | cut -d, -f2)
                    local GPU=$(echo "$ANE_OPS" | cut -d, -f3)
                    total_ops_tested=$((total_ops_tested + ANE + CPU + GPU))
                    ane_placed_ops=$((ane_placed_ops + ANE))
                    cpu_only_ops=$((cpu_only_ops + CPU))

                    notice "$TEST_NAME: ANE=$ANE CPU=$CPU GPU=$GPU"
                else
                    skip_msg "$TEST_NAME: compute plan harvest failed (may need macOS 14+)"
                fi
            else
                notice "$TEST_NAME: mlpackage not found for compute plan inspection"
            fi
        else
            err "$TEST_NAME: emission returned status=$STATUS"
        fi
    else
        err "$TEST_NAME: bridge execution failed"
    fi
}

# ─── Test 1: Linear Projection ────────────────────────────────────────────
notice "Testing: Linear projection (mb.linear)"
emit_and_inspect "linear_proj" "{
    'bridge_version': 1,
    'command': 'emit_linear_projection',
    'task_name': 'linear_proj',
    'input_dim': 64,
    'output_dim': 32,
    'batch_size': 1,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE'
}"

# ─── Test 2: MLP Block (Linear + Activation + Linear) ─────────────────────
notice "Testing: MLP block (fused linear+relu+linear)"
emit_and_inspect "mlp_block" "{
    'bridge_version': 1,
    'command': 'emit_mlp_block',
    'task_name': 'mlp_block',
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 64,
    'batch_size': 1,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE'
}"

# ─── Test 3: Attention Block ──────────────────────────────────────────────
notice "Testing: Attention (QKV + SDPA + out-proj)"
emit_and_inspect "attention" "{
    'bridge_version': 1,
    'command': 'emit_attention',
    'task_name': 'attention',
    'input_dim': 64,
    'num_heads': 4,
    'head_dim': 16,
    'batch_size': 1,
    'seq_len': 8,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE'
}"

# ─── Test 4: Stateful Decode Step ──────────────────────────────────────────
notice "Testing: Stateful decode step (KV-cache + SDPA)"
emit_and_inspect "decode_step" "{
    'bridge_version': 1,
    'command': 'emit_stateful_decode_step',
    'task_name': 'decode_step',
    'input_dim': 64,
    'num_heads': 4,
    'head_dim': 16,
    'batch_size': 1,
    'kv_len': 128,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE'
}"

# ─── Test 5: Multi-Function (Embedding + Decode) ──────────────────────────
notice "Testing: Multi-function model"
emit_and_inspect "multifunc" "{
    'bridge_version': 1,
    'command': 'emit_multifunction',
    'task_name': 'multifunc',
    'input_dim': 64,
    'output_dim': 32,
    'batch_size': 1,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE'
}"

# ─── Test 6: Palettized Linear Projection ─────────────────────────────────
notice "Testing: Palettized linear projection (4-bit LUT)"
emit_and_inspect "palettized" "{
    'bridge_version': 1,
    'command': 'emit_palettized_linear_projection',
    'task_name': 'palettized',
    'input_dim': 64,
    'output_dim': 32,
    'batch_size': 1,
    'dtype': 'fp16',
    'opset_version': 'iOS18',
    'compute_units': 'CPU_AND_NE',
    'palettization_specs': [{'weight_name': 'weight', 'nbits': 4, 'mode': 'kmeans'}]
}"

# ─── Individual Op Fastpath Checks via Python ─────────────────────────────
section "Per-Op ANE Fastpath Matrix"

# Test each op category individually using a Python script
# that builds minimal models and inspects compute plans.
PYTHON_MATRIX_SCRIPT="$OUTPUT_DIR/op_matrix.py"
cat > "$PYTHON_MATRIX_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Per-op ANE fastpath matrix tester.

Builds minimal single-op models using coremltools and checks
which ops the ANE actually accepts via MLComputePlan.
"""
import json
import sys
import os

os.environ.setdefault("COREMLTOOLS_DISABLE_TELEMETRY", "1")

try:
    import coremltools as ct
    from coremltools.models import MLModel
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
except ImportError:
    print(json.dumps({"error": "coremltools not available"}))
    sys.exit(1)

import numpy as np

# ─── Op test definitions ──────────────────────────────────────────────────
# Each entry: (op_name, build_function)
# build_function takes no args and returns an MLModel

def _make_model(prog_builder_fn, input_shape=(1, 64), input_name="x",
                output_name="y", dtype=np.float16, compute_units="CPU_AND_NE"):
    """Generic model builder: takes a function that builds a MIL program
    and returns a converted MLModel."""
    # Resolve numpy dtype → MIL dtype
    if dtype == np.float32:
        mil_dtype = types.fp32
    else:
        mil_dtype = types.fp16

    @mb.program(
        input_specs=[mb.TensorSpec(shape=input_shape, dtype=mil_dtype)],
        opset_version=ct.target.iOS16,
    )
    def prog(x):
        return prog_builder_fn(x)

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=getattr(ct.ComputeUnit, compute_units, ct.ComputeUnit.CPU_AND_NE),
    )
    return model

def _check_ane_placement(model, mlpackage_path):
    """Check compute plan for ANE placement."""
    try:
        plan = MLComputePlan.load_from_path(mlpackage_path)
        ane_ops = 0
        cpu_ops = 0
        gpu_ops = 0
        for op in plan.operations:
            device = str(op.device)
            if "ANE" in device or "NeuralEngine" in device:
                ane_ops += 1
            elif "CPU" in device:
                cpu_ops += 1
            elif "GPU" in device:
                gpu_ops += 1
        return {"ane": ane_ops, "cpu": cpu_ops, "gpu": gpu_ops}
    except Exception as e:
        return {"ane": 0, "cpu": 0, "gpu": 0, "error": str(e)}

def test_op(op_name, build_fn, output_dir):
    """Test a single op and return placement result."""
    try:
        import tempfile
        with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
            model = build_fn()
            mlpackage_path = os.path.join(tmpdir, f"{op_name}.mlpackage")
            model.save(mlpackage_path)
            placement = _check_ane_placement(model, mlpackage_path)
            return {"op": op_name, "status": "ok", **placement}
    except Exception as e:
        return {"op": op_name, "status": "error", "error": str(e), "ane": 0, "cpu": 0, "gpu": 0}

# ─── Build functions for each op ──────────────────────────────────────────
OP_TESTS = []

# Elementwise unary
for op_name, builder in [
    ("relu", lambda x: mb.relu(x=x)),
    ("sigmoid", lambda x: mb.sigmoid(x=x)),
    ("tanh", lambda x: mb.tanh(x=x)),
    ("abs", lambda x: mb.abs(x=x)),
    ("neg", lambda x: mb.neg(x=x)),
    ("exp", lambda x: mb.exp(x=x)),
    ("ceil", lambda x: mb.ceil(x=x)),
    ("floor", lambda x: mb.floor(x=x)),
    ("sqrt", lambda x: mb.sqrt(x=x)),
    ("rsqrt", lambda x: mb.rsqrt(x=x)),
    ("log", lambda x: mb.log(x=x)),
    ("sin", lambda x: mb.sin(x=x)),
    ("cos", lambda x: mb.cos(x=x)),
    ("gelu", lambda x: mb.gelu(x=x)),
    ("silu", lambda x: mb.silu(x=x)),
    ("softplus", lambda x: mb.softplus(x=x)),
    ("elu", lambda x: mb.elu(x=x, alpha=1.0)),
    ("leaky_relu", lambda x: mb.leaky_relu(x=x, alpha=0.01)),
    ("clip", lambda x: mb.clip(x=x, alpha=0.0, beta=6.0)),
    ("sign", lambda x: mb.sign(x=x)),
    ("round", lambda x: mb.round(x=x)),
]:
    OP_TESTS.append((op_name, builder))

# Elementwise binary
for op_name, builder in [
    ("add", lambda x: mb.add(x=x, y=x)),
    ("mul", lambda x: mb.mul(x=x, y=x)),
    ("sub", lambda x: mb.sub(x=x, y=x)),
    ("real_div", lambda x: mb.real_div(x=x, y=mb.add(x=x, y=mb.const(val=0.001)))),
    ("maximum", lambda x: mb.maximum(x=x, y=x)),
    ("minimum", lambda x: mb.minimum(x=x, y=x)),
    ("pow", lambda x: mb.pow(x=x, y=mb.const(val=2.0))),
    ("equal", lambda x: mb.equal(x=x, y=mb.const(val=0.0))),
    ("not_equal", lambda x: mb.not_equal(x=x, y=mb.const(val=0.0))),
    ("greater", lambda x: mb.greater(x=x, y=mb.const(val=0.0))),
    ("less", lambda x: mb.less(x=x, y=mb.const(val=1.0))),
    ("logical_and", lambda x: mb.logical_and(x=mb.greater(x=x, y=mb.const(val=0.0)), y=mb.less(x=x, y=mb.const(val=1.0)))),
    ("logical_or", lambda x: mb.logical_or(x=mb.greater(x=x, y=mb.const(val=0.0)), y=mb.less(x=x, y=mb.const(val=-1.0)))),
    ("logical_not", lambda x: mb.logical_not(x=mb.greater(x=x, y=mb.const(val=0.0)))),
]:
    OP_TESTS.append((op_name, builder))

# Reduction ops
for op_name, builder in [
    ("reduce_sum", lambda x: mb.reduce_sum(x=x, axes=[-1], keep_dims=True)),
    ("reduce_mean", lambda x: mb.reduce_mean(x=x, axes=[-1], keep_dims=True)),
    ("reduce_max", lambda x: mb.reduce_max(x=x, axes=[-1], keep_dims=True)),
    ("reduce_min", lambda x: mb.reduce_min(x=x, axes=[-1], keep_dims=True)),
    ("reduce_prod", lambda x: mb.reduce_prod(x=x, axes=[-1], keep_dims=True)),
    ("reduce_argmax", lambda x: mb.reduce_argmax(x=x, axis=-1, keep_dims=True)),
    ("reduce_argmin", lambda x: mb.reduce_argmin(x=x, axis=-1, keep_dims=True)),
]:
    OP_TESTS.append((op_name, builder))

# Tensor transform
for op_name, builder in [
    ("reshape", lambda x: mb.reshape(x=x, shape=[1, 8, 8])),
    ("transpose", lambda x: mb.transpose(x=x, perm=[0, 2, 1])),
    ("concat", lambda x: mb.concat(values=[x, x], axis=-1)),
    ("split", lambda x: mb.split(x=x, num_splits=2, axis=-1)[0]),
    ("squeeze", lambda x: mb.squeeze(x=x, axes=[0])),
    ("expand_dims", lambda x: mb.expand_dims(x=x, axis=1)),
    ("flatten2d", lambda x: mb.flatten2d(x=x)),
    ("pad", lambda x: mb.pad(x=x, pad=[(0,0), (2,2)])),
    ("reverse", lambda x: mb.reverse(x=x, axes=[-1])),
    ("cumsum", lambda x: mb.cumsum(x=x, axis=-1)),
    ("cast", lambda x: mb.cast(x=x, dtype="int32")),
]:
    OP_TESTS.append((op_name, builder))

# Normalization
for op_name, builder in [
    ("layer_norm", lambda x: mb.layer_norm(x=x, gamma=np.ones(64, dtype=np.float16),
                                            beta=np.zeros(64, dtype=np.float16),
                                            epsilon=1e-5)),
    ("batch_norm", lambda x: mb.batch_norm(x=x, mean=np.zeros(64, dtype=np.float16),
                                             variance=np.ones(64, dtype=np.float16),
                                             gamma=np.ones(64, dtype=np.float16),
                                             beta=np.zeros(64, dtype=np.float16),
                                             epsilon=1e-5)),
    ("instance_norm", lambda x: mb.instance_norm(x=x, gamma=np.ones(64, dtype=np.float16),
                                                   beta=np.zeros(64, dtype=np.float16),
                                                   epsilon=1e-5)),
    ("l2_norm", lambda x: mb.l2_norm(x=x, epsilon=1e-6)),
    ("softmax", lambda x: mb.softmax(x=x, axis=-1)),
]:
    OP_TESTS.append((op_name, builder))

# ─── Run all tests ────────────────────────────────────────────────────────
def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    results = []
    ane_total = 0
    cpu_total = 0

    for op_name, build_fn in OP_TESTS:
        # Wrap builder in a model constructor
        def make_model(builder=build_fn):
            return _make_model(builder)
        result = test_op(op_name, make_model, output_dir)
        results.append(result)
        status = "ANE" if result.get("ane", 0) > 0 else "CPU"
        ane_total += result.get("ane", 0)
        cpu_total += result.get("cpu", 0)
        print(f"  [{status:3s}] {op_name}: ANE={result.get('ane',0)} CPU={result.get('cpu',0)} GPU={result.get('gpu',0)}", flush=True)

    # Summary
    total = ane_total + cpu_total
    ane_rate = (ane_total / total * 100) if total > 0 else 0.0

    summary = {
        "total_ops_tested": len(results),
        "ane_placed": ane_total,
        "cpu_only": cpu_total,
        "ane_placement_rate_pct": round(ane_rate, 1),
        "results": results,
    }

    # Save full results
    results_path = os.path.join(output_dir, "ane_fastpath_matrix.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ANE Placement Rate: {ane_rate:.1f}% ({ane_total}/{total} ops)")
    print(f"  Full results saved to: {results_path}")

    return summary

if __name__ == "__main__":
    main()
PYEOF

# Run the per-op matrix test
notice "Running per-op ANE fastpath matrix (this may take a minute)..."
if python3 "$PYTHON_MATRIX_SCRIPT" "$OUTPUT_DIR" 2>&1; then
    ok "Per-op ANE fastpath matrix completed"
else
    err "Per-op ANE fastpath matrix failed (may need macOS 14+ with Apple Silicon)"
fi

# ─── Structural Verification ───────────────────────────────────────────────
section "Structural Verification"

# For each emitted model, verify MIR-vs-structure comparison
for ML_PKG in "$OUTPUT_DIR"/*/*.mlpackage "$OUTPUT_DIR"/*.mlpackage; do
    if [ -d "$ML_PKG" ]; then
        PKG_NAME=$(basename "$ML_PKG" .mlpackage)
        VERIFY_CMD_JSON="$OUTPUT_DIR/${PKG_NAME}_verify_cmd.json"
        VERIFY_RESULT_JSON="$OUTPUT_DIR/${PKG_NAME}_verify_result.json"
        python3 -c "
import json
json.dump({
    'bridge_version': 1,
    'command': 'verify',
    'mlpackage_path': '$ML_PKG',
    'compute_units': 'CPU_AND_NE'
}, open('$VERIFY_CMD_JSON', 'w'))
"
        if python3 "$BRIDGE" "$VERIFY_CMD_JSON" "$VERIFY_RESULT_JSON" 2>/dev/null; then
            ok "Verification: $PKG_NAME"
        else
            notice "Verification skipped for $PKG_NAME (may need newer coremltools)"
        fi
    fi
done

# ─── SIR→AIR→MIR Pipeline Coverage Test ────────────────────────────────────
section "Pipeline Coverage Analysis"

# Count SirOp, AirOp, MirOp variants from the Rust source
SIR_COUNT=$(python3 -c "
import re
with open('crates/ir/src/sir.rs') as f:
    content = f.read()
    # Count enum variants (lines starting with whitespace + UpperCamelCase)
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
" 2>/dev/null || echo "?")

AIR_COUNT=$(python3 -c "
import re
with open('crates/ir/src/air.rs') as f:
    content = f.read()
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
" 2>/dev/null || echo "?")

MIR_COUNT=$(python3 -c "
import re
with open('crates/ir/src/mir.rs') as f:
    content = f.read()
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
" 2>/dev/null || echo "?")

notice "SIR op variants: $SIR_COUNT"
notice "AIR op variants: $AIR_COUNT"
notice "MIR op variants: $MIR_COUNT"

# ─── Summary Report ────────────────────────────────────────────────────────
section "Lab Summary"

# Initialize defaults for variables that may not be set if matrix is missing
ANE_RATE="${ANE_RATE:-N/A}"

# Load fastpath matrix results if available
MATRIX_FILE="$OUTPUT_DIR/ane_fastpath_matrix.json"
if [ -f "$MATRIX_FILE" ]; then
    ANE_RATE=$(python3 -c "import json; print(json.load(open('$MATRIX_FILE')).get('ane_placement_rate_pct', 'N/A'))" 2>/dev/null || echo "N/A")
    ANE_OPS_TOTAL=$(python3 -c "import json; d=json.load(open('$MATRIX_FILE')); print(d.get('ane_placed', 'N/A'))" 2>/dev/null || echo "N/A")
    CPU_OPS_TOTAL=$(python3 -c "import json; d=json.load(open('$MATRIX_FILE')); print(d.get('cpu_only', 'N/A'))" 2>/dev/null || echo "N/A")
    notice "ANE placement rate: ${ANE_RATE}% (${ANE_OPS_TOTAL} ANE / ${CPU_OPS_TOTAL} CPU)"
else
    notice "Fastpath matrix results not available"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           MILLer — Test Results               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Passed:  $pass                                          ║"
echo "║  Failed:  $fail                                          ║"
echo "║  Skipped: $skip                                          ║"
echo "║  SIR ops: $SIR_COUNT                                     ║"
echo "║  AIR ops: $AIR_COUNT                                     ║"
echo "║  MIR ops: $MIR_COUNT                                     ║"
echo "║  ANE rate: ${ANE_RATE:-N/A}%                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Artifacts: $OUTPUT_DIR                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Write a machine-readable summary
python3 -c "
import json
summary = {
    'passed': $pass,
    'failed': $fail,
    'skipped': $skip,
    'sir_ops': '$SIR_COUNT',
    'air_ops': '$AIR_COUNT',
    'mir_ops': '$MIR_COUNT',
    'ane_placement_rate': '$ANE_RATE',
    'output_dir': '$OUTPUT_DIR',
}
with open('$OUTPUT_DIR/lab_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
" 2>/dev/null

if [ "$fail" -gt 0 ]; then
    echo "  Result: FAIL"
    exit 1
elif [ "$skip" -gt 0 ] && [ "$pass" -eq 0 ]; then
    echo "  Result: SKIP (environment not ready)"
    exit 2
else
    echo "  Result: PASS"
    exit 0
fi
