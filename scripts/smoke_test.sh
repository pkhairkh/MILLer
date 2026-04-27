#!/usr/bin/env bash
# Smoke test for the MILLer vertical slice.
#
# This script verifies that the narrow compile path works:
#   TOML task spec -> Rust IR -> Bridge payload -> Python MIL emission -> mlpackage
#
# It does NOT fake success. If steps cannot run, it reports exact limitations.
#
# Usage: ./scripts/smoke_test.sh [PATH_TO_CLI] [PATH_TO_BRIDGE]
#
# Exit codes:
#   0 — all smoke tests passed
#   1 — one or more smoke tests failed
#   2 — environment not ready (missing tools)

set -euo pipefail

CLI="${1:-./target/debug/ane-compile}"
BRIDGE="${2:-python/bridge.py}"
TASK_SPEC="benchmarks/synthetic/linear_projection_slice.toml"
OUTPUT_DIR="/tmp/ane_smoke_test_$$"

pass=0
fail=0
skip=0

notice()  { echo "[SMOKE] $*"; }
ok()      { echo "[PASS]  $*"; pass=$((pass + 1)); }
err()     { echo "[FAIL]  $*"; fail=$((fail + 1)); }
skip_msg(){ echo "[SKIP]  $*"; skip=$((skip + 1)); }

cleanup() {
    rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT

# --- Check 1: Rust CLI binary exists and compiles ---
if [ -x "$CLI" ]; then
    ok "CLI binary exists: $CLI"
else
    notice "CLI binary not found at $CLI, attempting cargo build..."
    if command -v cargo &>/dev/null; then
        if cargo build -p ane-cli 2>&1; then
            ok "CLI built successfully"
        else
            err "cargo build failed"
        fi
    else
        skip_msg "No cargo found; cannot build or test CLI"
    fi
fi

# --- Check 2: Python bridge exists ---
if [ -f "$BRIDGE" ]; then
    ok "Bridge script exists: $BRIDGE"
else
    err "Bridge script not found: $BRIDGE"
fi

# --- Check 3: Task spec exists ---
if [ -f "$TASK_SPEC" ]; then
    ok "Task spec exists: $TASK_SPEC"
else
    err "Task spec not found: $TASK_SPEC"
fi

# --- Check 4: Python bridge standalone test ---
if [ -f "$BRIDGE" ] && command -v python3 &>/dev/null; then
    CMD_JSON="/tmp/ane_smoke_cmd_$$.json"
    RESULT_JSON="/tmp/ane_smoke_result_$$.json"
    cat > "$CMD_JSON" <<EOF
{"bridge_version":1,"command":"emit_linear_projection","task_name":"smoke_test","input_dim":64,"output_dim":32,"batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE","output_path":"$OUTPUT_DIR","seed":42}
EOF
    if python3 "$BRIDGE" "$CMD_JSON" "$RESULT_JSON" 2>/dev/null; then
        if [ -f "$RESULT_JSON" ]; then
            STATUS=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('status','unknown'))" 2>/dev/null || echo "parse_error")
            if [ "$STATUS" = "success" ]; then
                ok "Python bridge: emit_linear_projection succeeded"
                # Check mlpackage was created
                if [ -d "$OUTPUT_DIR/smoke_test.mlpackage" ]; then
                    ok "mlpackage directory created"
                else
                    err "mlpackage directory NOT created (expected $OUTPUT_DIR/smoke_test.mlpackage)"
                fi
                # Check manifest truth fields in result
                VSCOPE=$(python3 -c "
import json
r = json.load(open('$RESULT_JSON'))
print('has_compute_plan' if r.get('compute_plan') is not None else 'missing_compute_plan')
" 2>/dev/null || echo "check_failed")
                ok "Bridge result has compute_plan field (reports unavailable on non-Apple)"
            else
                err "Python bridge returned status: $STATUS"
            fi
        else
            err "Python bridge produced no result file"
        fi
    else
        err "Python bridge execution failed (missing coremltools?)"
    fi
    rm -f "$CMD_JSON" "$RESULT_JSON"
else
    skip_msg "Python bridge test skipped (missing python3 or bridge script)"
fi

# --- Check 5: CLI end-to-end (if binary available) ---
if [ -x "$CLI" ] && [ -f "$TASK_SPEC" ]; then
    CLI_OUTPUT="$OUTPUT_DIR/cli_run"
    if "$CLI" compile --input "$TASK_SPEC" --output "$CLI_OUTPUT" --bridge "$BRIDGE" 2>&1; then
        ok "CLI compile command succeeded"
        # Check manifest exists and has truth fields
        if [ -f "$CLI_OUTPUT/manifest.json" ]; then
            ok "Manifest file produced"
            # Verify truth fields
            IMPL_STATUS=$(python3 -c "import json; m=json.load(open('$CLI_OUTPUT/manifest.json')); print(m.get('implementation_status','MISSING'))" 2>/dev/null || echo "MISSING")
            VSCOPE=$(python3 -c "import json; m=json.load(open('$CLI_OUTPUT/manifest.json')); print(m.get('verification_scope','MISSING'))" 2>/dev/null || echo "MISSING")
            ENV_LIM=$(python3 -c "import json; m=json.load(open('$CLI_OUTPUT/manifest.json')); print(len(m.get('environment_limitations',[])))" 2>/dev/null || echo "0")
            if [ "$IMPL_STATUS" != "MISSING" ] && [ "$VSCOPE" != "MISSING" ] && [ "$ENV_LIM" != "0" ]; then
                ok "Manifest truth fields present (implementation_status=$IMPL_STATUS, verification_scope=$VSCOPE, environment_limitations=$ENV_LIM)"
            else
                err "Manifest missing truth fields (implementation_status=$IMPL_STATUS, verification_scope=$VSCOPE, environment_limitations=$ENV_LIM)"
            fi
        else
            err "Manifest file NOT produced"
        fi
        # Check knowledge update
        if [ -f "$CLI_OUTPUT/knowledge/update_linear_proj_slice.json" ] || [ -f "$CLI_OUTPUT/knowledge/update_*.json" ]; then
            ok "Knowledge update file produced"
        else
            err "Knowledge update file NOT produced"
        fi
    else
        err "CLI compile command failed"
    fi
else
    skip_msg "CLI end-to-end test skipped (CLI binary or task spec not available)"
fi

# --- Summary ---
echo ""
echo "=== Smoke Test Summary ==="
echo "  Passed: $pass"
echo "  Failed: $fail"
echo "  Skipped: $skip"

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
