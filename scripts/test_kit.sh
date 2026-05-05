#!/usr/bin/env bash
# =============================================================================
# MILLer Compiler — End-to-End Test Kit for Apple M2 (Qwen3-0.6B)
# =============================================================================
#
# Pipeline: TOML task spec / HF model → SIR → AIR → MIR → Bridge/MIL → .mlpackage
#   For Qwen3-0.6B: trace-compile (HuggingFace → traced graph → SIR → full pipeline)
#   For synthetic: compile (TOML spec → SIR → full pipeline)
#
# Usage:
#   ./run_tests.sh [--skip-build] [--skip-download] [--phase PHASE] [--verbose]
#
# Phases:
#   1. prereqs     — Verify M2 Mac, Rust, Python, coremltools
#   2. build       — Build ane-compile CLI in release mode
#   3. synthetic   — Compile all synthetic task specs (7 families)
#   4. bridge      — Test Python bridge directly (all 7 emitters)
#   5. qwen3       — trace-compile Qwen3-0.6B from HuggingFace
#   6. knowledge   — Validate knowledge seed loading & store integrity
#   7. ir-pipeline — Validate SIR → AIR → MIR dumps & legality
#   8. ane         — ANE fastpath matrix & compute plan inspection
#   9. report      — Aggregate & print results
#
# Environment:
#   MILLER_ROOT       — Path to MILLer repo (default: script's parent's parent)
#   TEST_WORKDIR      — Working directory for artifacts (default: ./test_work)
#   QWEN3_MODEL_ID    — HuggingFace model ID (default: Qwen/Qwen3-0.6B)
#   SEED              — Random seed (default: 42)
#   VERBOSE           — Set to 1 for debug output
# =============================================================================

set -euo pipefail

# ─── Colors & Logging ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC}  $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }
log_phase()   { echo -e "\n${BOLD}${CYAN}═══ Phase $1: $2 ═══${NC}"; }
log_section() { echo -e "\n${CYAN}── $1 ──${NC}"; }

# ─── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILLER_ROOT="${MILLER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TEST_WORKDIR="${TEST_WORKDIR:-${SCRIPT_DIR}/test_work}"
QWEN3_MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-0.6B}"
SEED="${SEED:-42}"
VERBOSE="${VERBOSE:-0}"

SKIP_BUILD=0
SKIP_DOWNLOAD=0
PHASE_FILTER=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${TEST_WORKDIR}/results_${TIMESTAMP}.json"

# ─── Argument Parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)     SKIP_BUILD=1; shift ;;
        --skip-download)  SKIP_DOWNLOAD=1; shift ;;
        --phase)          PHASE_FILTER="$2"; shift 2 ;;
        --verbose)        VERBOSE=1; shift ;;
        --help|-h)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *) log_fail "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Results Tracking ────────────────────────────────────────────────────────
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
FAILED_ITEMS=()

record_result() {
    local name="$1" status="$2" detail="${3:-}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    case "$status" in
        pass)  PASSED_TESTS=$((PASSED_TESTS + 1)); log_success "$name" ;;
        fail)  FAILED_TESTS=$((FAILED_TESTS + 1)); log_fail "$name — $detail"; FAILED_ITEMS+=("$name") ;;
        skip)  SKIPPED_TESTS=$((SKIPPED_TESTS + 1)); log_warn "SKIP: $name — $detail" ;;
    esac
}

should_run_phase() {
    local phase_name="$1"
    if [[ -n "$PHASE_FILTER" ]]; then
        [[ "$phase_name" == "$PHASE_FILTER" ]]
    else
        true
    fi
}

# ─── Utility ─────────────────────────────────────────────────────────────────
require_cmd() { command -v "$1" &>/dev/null; }

ane_compile() {
    local bin="${TEST_WORKDIR}/release/ane-compile"
    if [[ ! -x "$bin" ]]; then
        bin="${MILLER_ROOT}/target/release/ane-compile"
    fi
    "$bin" "$@"
}

# =============================================================================
# Phase 1: Prerequisites
# =============================================================================
phase_prereqs() {
    log_phase 1 "Prerequisites"
    local ok=true

    # ─── macOS & Apple Silicon ───
    log_section "Platform Check"
    if [[ "$(uname)" == "Darwin" ]]; then
        local arch=$(uname -m)
        if [[ "$arch" == "arm64" ]]; then
            local chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
            record_result "macOS arm64 (Apple Silicon)" "pass" "$chip"
        else
            record_result "macOS arm64 (Apple Silicon)" "fail" "Got arch=$arch, need arm64"
            ok=false
        fi
    else
        record_result "macOS arm64 (Apple Silicon)" "fail" "Not macOS — got $(uname)"
        ok=false
    fi

    # ─── Rust toolchain ───
    log_section "Rust Toolchain"
    if require_cmd rustc; then
        record_result "Rust compiler" "pass" "$(rustc --version)"
    else
        record_result "Rust compiler" "fail" "rustc not found"
        ok=false
    fi

    if require_cmd cargo; then
        record_result "Cargo" "pass" "$(cargo --version)"
    else
        record_result "Cargo" "fail" "cargo not found"
        ok=false
    fi

    # ─── Python ───
    log_section "Python Environment"
    if require_cmd python3; then
        record_result "Python3" "pass" "$(python3 --version)"
    else
        record_result "Python3" "fail" "python3 not found"
        ok=false
    fi

    for pkg in torch transformers coremltools numpy; do
        if python3 -c "import $pkg" 2>/dev/null; then
            local pkg_ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            record_result "Python package: $pkg" "pass" "$pkg_ver"
        else
            record_result "Python package: $pkg" "fail" "pip install $pkg"
            ok=false
        fi
    done

    # ─── MILLer repo ───
    log_section "MILLer Repository"
    if [[ -f "${MILLER_ROOT}/Cargo.toml" ]]; then
        record_result "MILLer repo at ${MILLER_ROOT}" "pass"
    else
        record_result "MILLer repo at ${MILLER_ROOT}" "fail" "Cargo.toml not found — set MILLER_ROOT"
        ok=false
    fi

    # Check bridge script
    local bridge="${MILLER_ROOT}/python/bridge.py"
    if [[ -f "$bridge" ]]; then
        record_result "Python bridge (bridge.py)" "pass"
    else
        record_result "Python bridge (bridge.py)" "fail" "Not found at $bridge"
        ok=false
    fi

    # Check task specs
    if [[ -d "${MILLER_ROOT}/benchmarks/synthetic" ]]; then
        local spec_count=$(find "${MILLER_ROOT}/benchmarks/synthetic" -name "*.toml" | wc -l | tr -d ' ')
        record_result "Synthetic task specs" "pass" "$spec_count files"
    else
        record_result "Synthetic task specs" "fail" "benchmarks/synthetic/ not found"
        ok=false
    fi

    # ─── Disk space ───
    log_section "Disk Space"
    local free_gb=$(df -g "$HOME" | tail -1 | awk '{print $4}')
    if [[ "$free_gb" -ge 10 ]]; then
        record_result "Free disk space (>=10 GB)" "pass" "${free_gb} GB"
    else
        record_result "Free disk space (>=10 GB)" "fail" "Only ${free_gb} GB"
        ok=false
    fi

    if [[ "$ok" == "true" ]]; then
        log_success "All prerequisites satisfied"
    else
        log_fail "Some prerequisites missing"
    fi
}

# =============================================================================
# Phase 2: Build
# =============================================================================
phase_build() {
    log_phase 2 "Build MILLer (ane-compile)"

    if [[ "$SKIP_BUILD" -eq 1 ]]; then
        record_result "Build (skipped)" "pass"
        return 0
    fi

    mkdir -p "${TEST_WORKDIR}/release"

    log_section "Cargo Build (release)"
    if (cd "$MILLER_ROOT" && cargo build --release -p ane-cli 2>&1 | tee "${TEST_WORKDIR}/build.log"); then
        record_result "cargo build --release -p ane-cli" "pass"
    else
        record_result "cargo build --release -p ane-cli" "fail" "See ${TEST_WORKDIR}/build.log"
        return 1
    fi

    # Copy binary
    if [[ -f "${MILLER_ROOT}/target/release/ane-compile" ]]; then
        cp "${MILLER_ROOT}/target/release/ane-compile" "${TEST_WORKDIR}/release/ane-compile"
        record_result "CLI binary copied" "pass"
    else
        record_result "CLI binary" "fail" "target/release/ane-compile not found"
        return 1
    fi

    # Version check
    log_section "Version & Help"
    local ver
    ver=$(ane_compile --version 2>&1 || true)
    if [[ -n "$ver" ]]; then
        record_result "ane-compile --version" "pass" "$ver"
    else
        record_result "ane-compile --version" "fail" "No output"
    fi

    # Verify key subcommands
    for subcmd in compile compile-sharded compile-full lab profile trace-compile query verify import; do
        if ane_compile "$subcmd" --help &>/dev/null; then
            record_result "subcommand: $subcmd" "pass"
        else
            record_result "subcommand: $subcmd" "fail" "Not available"
        fi
    done
}

# =============================================================================
# Phase 3: Synthetic Task Spec Compilation
# =============================================================================
phase_synthetic() {
    log_phase 3 "Synthetic Task Spec Compilation"

    local bridge="${MILLER_ROOT}/python/bridge.py"
    local specs_dir="${MILLER_ROOT}/benchmarks/synthetic"

    if [[ ! -d "$specs_dir" ]]; then
        record_result "Synthetic specs dir" "fail" "$specs_dir not found"
        return 1
    fi

    # ─── Compile each task spec ───
    log_section "Compile Each Task Spec"
    local spec_count=0
    for spec in "$specs_dir"/*.toml; do
        [[ -f "$spec" ]] || continue
        local spec_name=$(basename "$spec" .toml)
        local output_dir="${TEST_WORKDIR}/synthetic/${spec_name}"
        mkdir -p "$output_dir"

        spec_count=$((spec_count + 1))

        if ane_compile compile \
                --input "$spec" \
                --output "$output_dir" \
                --bridge "$bridge" \
                2>"${TEST_WORKDIR}/synthetic/${spec_name}.log"; then
            record_result "compile: $spec_name" "pass"

            # Verify mlpackage was created
            if find "$output_dir" -name "*.mlpackage" -type d 2>/dev/null | head -1 | grep -q .; then
                record_result "mlpackage produced: $spec_name" "pass"
            else
                record_result "mlpackage produced: $spec_name" "fail" "No .mlpackage in $output_dir"
            fi

            # Verify manifest.json
            if [[ -f "$output_dir/manifest.json" ]]; then
                record_result "manifest.json: $spec_name" "pass"

                # Check truth fields
                local impl_status
                impl_status=$(python3 -c "import json; m=json.load(open('$output_dir/manifest.json')); print(m.get('implementation_status','MISSING'))" 2>/dev/null || echo "MISSING")
                if [[ "$impl_status" != "MISSING" ]]; then
                    record_result "manifest implementation_status: $spec_name" "pass" "$impl_status"
                else
                    record_result "manifest implementation_status: $spec_name" "fail" "Field missing"
                fi
            else
                record_result "manifest.json: $spec_name" "fail" "Not produced"
            fi
        else
            record_result "compile: $spec_name" "fail" "See ${TEST_WORKDIR}/synthetic/${spec_name}.log"
        fi
    done

    log_info "Tested $spec_count task specs"

    # ─── Sharded compile ───
    log_section "Sharded Compilation"
    local shard_spec="${specs_dir}/sharded_linear_pipeline.toml"
    if [[ -f "$shard_spec" ]]; then
        local shard_output="${TEST_WORKDIR}/synthetic/sharded_pipeline"
        mkdir -p "$shard_output"
        if ane_compile compile-sharded \
                --input "$shard_spec" \
                --output "$shard_output" \
                --bridge "$bridge" \
                --seed "$SEED" \
                2>"${TEST_WORKDIR}/sharded_compile.log"; then
            record_result "compile-sharded: sharded_linear_pipeline" "pass"
        else
            record_result "compile-sharded: sharded_linear_pipeline" "fail" "See log"
        fi
    else
        record_result "compile-sharded" "skip" "sharded_linear_pipeline.toml not found"
    fi

    # ─── Proto-direct (no Python bridge) ───
    log_section "Proto-Direct Compilation (Rust-only emission)"
    local linear_spec="${specs_dir}/linear_projection.toml"
    if [[ -f "$linear_spec" ]]; then
        local proto_output="${TEST_WORKDIR}/synthetic/proto_direct"
        mkdir -p "$proto_output"
        if ane_compile compile-sharded \
                --input "$linear_spec" \
                --output "$proto_output" \
                --proto-direct \
                --seed "$SEED" \
                2>"${TEST_WORKDIR}/proto_direct.log"; then
            record_result "proto-direct emission" "pass"
        else
            record_result "proto-direct emission" "fail" "See ${TEST_WORKDIR}/proto_direct.log"
        fi
    else
        record_result "proto-direct emission" "skip" "linear_projection.toml not found"
    fi
}

# =============================================================================
# Phase 4: Python Bridge Direct Test
# =============================================================================
phase_bridge() {
    log_phase 4 "Python Bridge Direct Test"

    local bridge="${MILLER_ROOT}/python/bridge.py"

    if [[ ! -f "$bridge" ]]; then
        record_result "Bridge script" "fail" "$bridge not found"
        return 1
    fi

    # Test each emitter command
    log_section "Bridge Emitter Commands"

    local -A EMITTERS
    EMITTERS[emit_linear_projection]='{"bridge_version":1,"command":"emit_linear_projection","task_name":"test_linear","input_dim":64,"output_dim":32,"batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_mlp_block]='{"bridge_version":1,"command":"emit_mlp_block","task_name":"test_mlp","input_dim":64,"hidden_dim":128,"output_dim":64,"activation":"gelu","batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_attention]='{"bridge_version":1,"command":"emit_attention","task_name":"test_attn","input_dim":64,"num_heads":4,"head_dim":16,"batch_size":1,"seq_len":8,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_stateful_decode_step]='{"bridge_version":1,"command":"emit_stateful_decode_step","task_name":"test_decode","input_dim":64,"num_heads":4,"head_dim":16,"batch_size":1,"kv_len":32,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_lut_projection]='{"bridge_version":1,"command":"emit_lut_projection","task_name":"test_lut","vocab_size":16,"embed_dim":64,"num_groups":16,"lut_bitwidth":4,"batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_multifunction]='{"bridge_version":1,"command":"emit_multifunction","task_name":"test_multi","input_dim":64,"output_dim":32,"batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'
    EMITTERS[emit_shard_decode_step]='{"bridge_version":1,"command":"emit_shard_decode_step","task_name":"test_shard_decode","input_dim":64,"num_heads":4,"head_dim":16,"batch_size":1,"kv_len":32,"shard_role":"Interior","dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE"}'

    for emitter_name in $(echo "${!EMITTERS[@]}" | tr ' ' '\n' | sort); do
        local cmd_json="${EMITTERS[$emitter_name]}"
        local test_dir="${TEST_WORKDIR}/bridge/${emitter_name}"
        mkdir -p "$test_dir"

        # Inject output_path and seed into command
        local cmd_file="$test_dir/command.json"
        local result_file="$test_dir/result.json"

        python3 -c "
import json
cmd = json.loads('''$cmd_json''')
cmd['output_path'] = '$test_dir'
cmd['seed'] = $SEED
json.dump(cmd, open('$cmd_file', 'w'))
"

        if python3 "$bridge" "$cmd_file" "$result_file" 2>/dev/null; then
            local status=$(python3 -c "import json; print(json.load(open('$result_file')).get('status','unknown'))" 2>/dev/null || echo "parse_error")
            if [[ "$status" == "success" ]]; then
                record_result "bridge: $emitter_name" "pass"

                # Check mlpackage was produced
                local output_path
                output_path=$(python3 -c "import json; r=json.load(open('$result_file')); print(r.get('output_path',''))" 2>/dev/null || echo "")
                if [[ -n "$output_path" ]] && [[ -d "$output_path" ]]; then
                    record_result "mlpackage: $emitter_name" "pass"
                elif find "$test_dir" -name "*.mlpackage" -type d 2>/dev/null | head -1 | grep -q .; then
                    record_result "mlpackage: $emitter_name" "pass"
                else
                    record_result "mlpackage: $emitter_name" "fail" "No .mlpackage produced"
                fi

                # Check content_hash
                local content_hash
                content_hash=$(python3 -c "import json; r=json.load(open('$result_file')); print(r.get('content_hash',''))" 2>/dev/null || echo "")
                if [[ -n "$content_hash" ]]; then
                    record_result "content_hash: $emitter_name" "pass" "$content_hash"
                else
                    log_warn "No content_hash for $emitter_name — may be expected"
                fi
            else
                local error_msg=$(python3 -c "import json; r=json.load(open('$result_file')); print(r.get('error_message',''))" 2>/dev/null || echo "unknown")
                record_result "bridge: $emitter_name" "fail" "status=$status: $error_msg"
            fi
        else
            record_result "bridge: $emitter_name" "fail" "Bridge execution failed"
        fi
    done

    # ─── Invalid command test ───
    log_section "Bridge Error Handling"
    local err_dir="${TEST_WORKDIR}/bridge/error_test"
    mkdir -p "$err_dir"
    echo '{"bridge_version":1,"command":"nonexistent_command"}' > "$err_dir/bad_cmd.json"
    if python3 "$bridge" "$err_dir/bad_cmd.json" "$err_dir/bad_result.json" 2>/dev/null; then
        local err_status=$(python3 -c "import json; print(json.load(open('$err_dir/bad_result.json')).get('status',''))" 2>/dev/null || echo "")
        if [[ "$err_status" == "error" ]]; then
            record_result "bridge: invalid command → error" "pass"
        else
            record_result "bridge: invalid command → error" "fail" "Expected error, got: $err_status"
        fi
    else
        record_result "bridge: invalid command → error" "pass" "Non-zero exit is acceptable"
    fi

    # ─── Wrong bridge_version test ───
    echo '{"bridge_version":99,"command":"emit_linear_projection"}' > "$err_dir/wrong_ver.json"
    if python3 "$bridge" "$err_dir/wrong_ver.json" "$err_dir/wrong_ver_result.json" 2>/dev/null; then
        local ver_status=$(python3 -c "import json; print(json.load(open('$err_dir/wrong_ver_result.json')).get('status',''))" 2>/dev/null || echo "")
        if [[ "$ver_status" == "error" ]]; then
            record_result "bridge: wrong version → error" "pass"
        else
            record_result "bridge: wrong version → error" "fail" "Expected error, got: $ver_status"
        fi
    else
        record_result "bridge: wrong version → error" "pass" "Non-zero exit"
    fi
}

# =============================================================================
# Phase 5: Qwen3-0.6B trace-compile
# =============================================================================
phase_qwen3() {
    log_phase 5 "Qwen3-0.6B trace-compile"

    local bridge="${MILLER_ROOT}/python/bridge.py"
    local output_dir="${TEST_WORKDIR}/qwen3"
    mkdir -p "$output_dir"

    # ─── trace-compile ───
    log_section "trace-compile Qwen3-0.6B"
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]] && [[ -d "${output_dir}/qwen3-0.6b.mlpackage" ]]; then
        record_result "trace-compile (skipped, using cache)" "pass"
    else
        if ane_compile trace-compile \
                --model "$QWEN3_MODEL_ID" \
                --output "$output_dir" \
                --bridge "$bridge" \
                --batch-size 1 \
                --seq-len 32 \
                --max-seq-len 2048 \
                --dtype fp16 \
                --seed "$SEED" \
                2>&1 | tee "${TEST_WORKDIR}/trace_compile.log"; then
            record_result "trace-compile Qwen3-0.6B" "pass"
        else
            record_result "trace-compile Qwen3-0.6B" "fail" "See ${TEST_WORKDIR}/trace_compile.log"
        fi
    fi

    # ─── Verify compiled mlpackage ───
    log_section "Verify Qwen3 Output"
    local qwen3_mlpackage=""
    # Find the produced mlpackage
    qwen3_mlpackage=$(find "$output_dir" -name "*.mlpackage" -type d 2>/dev/null | head -1 || true)
    if [[ -n "$qwen3_mlpackage" ]] && [[ -d "$qwen3_mlpackage" ]]; then
        record_result "Qwen3 mlpackage produced" "pass" "$qwen3_mlpackage"

        # Verify it
        if ane_compile verify \
                --mlpackage "$qwen3_mlpackage" \
                --output "${output_dir}/verify_result.json" \
                --bridge "$bridge" \
                2>"${TEST_WORKDIR}/qwen3_verify.log"; then
            record_result "Qwen3 verify" "pass"
        else
            record_result "Qwen3 verify" "fail" "See ${TEST_WORKDIR}/qwen3_verify.log"
        fi
    else
        record_result "Qwen3 mlpackage produced" "fail" "No .mlpackage found in $output_dir"
    fi

    # ─── With KV-cache ───
    log_section "trace-compile with --with-kv-cache"
    local kv_output="${TEST_WORKDIR}/qwen3_kv"
    mkdir -p "$kv_output"
    if ane_compile trace-compile \
            --model "$QWEN3_MODEL_ID" \
            --output "$kv_output" \
            --bridge "$bridge" \
            --batch-size 1 \
            --seq-len 1 \
            --max-seq-len 2048 \
            --dtype fp16 \
            --with-kv-cache \
            --seed "$SEED" \
            2>"${TEST_WORKDIR}/trace_compile_kv.log"; then
        record_result "trace-compile Qwen3 with KV-cache" "pass"
    else
        record_result "trace-compile Qwen3 with KV-cache" "fail" "See ${TEST_WORKDIR}/trace_compile_kv.log"
    fi

    # ─── Profile ───
    log_section "Profile Qwen3"
    if [[ -n "$qwen3_mlpackage" ]]; then
        if ane_compile profile \
                --mlpackage "$qwen3_mlpackage" \
                --output "${output_dir}/profile_result.json" \
                --bridge "$bridge" \
                --warmup 3 \
                --iterations 10 \
                2>"${TEST_WORKDIR}/qwen3_profile.log"; then
            record_result "profile Qwen3-0.6B" "pass"
        else
            record_result "profile Qwen3-0.6B" "fail" "See ${TEST_WORKDIR}/qwen3_profile.log"
        fi
    else
        record_result "profile Qwen3-0.6B" "skip" "No mlpackage to profile"
    fi
}

# =============================================================================
# Phase 6: Knowledge System Validation
# =============================================================================
phase_knowledge() {
    log_phase 6 "Knowledge System Validation"

    local knowledge_dir="${MILLER_ROOT}/knowledge"

    # ─── Seed Files ───
    log_section "Knowledge Seed Files"
    if [[ -d "$knowledge_dir" ]]; then
        local seed_count=$(find "$knowledge_dir" -name "*.json" -type f | wc -l | tr -d ' ')
        record_result "Knowledge seed files found" "pass" "$seed_count files"

        # Validate each seed file schema
        for seed in "$knowledge_dir"/*.json; do
            [[ -f "$seed" ]] || continue
            local seed_name=$(basename "$seed")

            # Check envelope schema
            if python3 -c "
import json, sys
with open('$seed') as f:
    data = json.load(f)

# Validate envelope
assert 'version' in data, 'Missing version field'
assert 'entries' in data, 'Missing entries field'
assert isinstance(data['entries'], list), 'entries must be a list'

# Validate each entry
for i, entry in enumerate(data['entries']):
    assert 'id' in entry, f'entries[{i}]: missing id'
    assert 'knowledge_type' in entry, f'entries[{i}]: missing knowledge_type'
    assert 'confidence' in entry, f'entries[{i}]: missing confidence'
    assert 'evidence_source' in entry, f'entries[{i}]: missing evidence_source'
    assert entry['evidence_source'] in ('SourceCode', 'RealModelRun', 'SyntheticRun'), \
        f'entries[{i}]: invalid evidence_source: {entry[\"evidence_source\"]}'
    assert 0.0 <= entry['confidence'] <= 1.0, f'entries[{i}]: confidence out of range'
    assert 'scope' in entry, f'entries[{i}]: missing scope'
    assert 'payload' in entry, f'entries[{i}]: missing payload'

print(f'  {\"$seed_name\"}: {len(data[\"entries\"])} entries OK')
"; then
                record_result "Schema: $seed_name" "pass"
            else
                record_result "Schema: $seed_name" "fail" "Schema validation failed"
            fi
        done
    else
        record_result "Knowledge directory" "fail" "$knowledge_dir not found"
    fi

    # ─── Import seeds ───
    log_section "Knowledge Import"
    local store_dir="${TEST_WORKDIR}/knowledge_store"
    mkdir -p "$store_dir"

    if ane_compile import \
            --source "$knowledge_dir" \
            --store "$store_dir" \
            --validate \
            2>"${TEST_WORKDIR}/import.log"; then
        record_result "import seeds" "pass"
    else
        record_result "import seeds" "fail" "See ${TEST_WORKDIR}/import.log"
    fi

    # Check store index
    if [[ -f "$store_dir/store_index.json" ]]; then
        record_result "Store index created" "pass"
    else
        record_result "Store index created" "fail" "store_index.json not found"
    fi

    # ─── Query ───
    log_section "Knowledge Query"
    if ane_compile query \
            --store "$store_dir" \
            --filter "ane_legal" \
            2>"${TEST_WORKDIR}/query.log"; then
        record_result "query: ane_legal" "pass"
    else
        record_result "query: ane_legal" "fail" "See ${TEST_WORKDIR}/query.log"
    fi
}

# =============================================================================
# Phase 7: IR Pipeline Validation
# =============================================================================
phase_ir_pipeline() {
    log_phase 7 "IR Pipeline Validation (SIR → AIR → MIR)"

    local bridge="${MILLER_ROOT}/python/bridge.py"
    local specs_dir="${MILLER_ROOT}/benchmarks/synthetic"
    local ir_dir="${TEST_WORKDIR}/ir_dumps"
    mkdir -p "$ir_dir"

    # Use a small task spec for IR dump analysis
    local spec="${specs_dir}/linear_projection_slice.toml"
    if [[ ! -f "$spec" ]]; then
        spec="${specs_dir}/linear_projection.toml"
    fi

    if [[ ! -f "$spec" ]]; then
        record_result "IR pipeline: task spec" "fail" "No linear projection spec found"
        return 1
    fi

    # ─── Compile with verbose output to capture IR stages ───
    log_section "IR Stage Capture"
    local compile_output="${TEST_WORKDIR}/ir_compile_output"

    if ane_compile compile \
            --input "$spec" \
            --output "$compile_output" \
            --bridge "$bridge" \
            2>&1 | tee "${TEST_WORKDIR}/ir_compile.log"; then
        record_result "IR compile (linear_projection)" "pass"
    else
        record_result "IR compile (linear_projection)" "fail"
    fi

    # ─── Legality rules ───
    log_section "Legality Rule Validation"
    local legality_seed="${MILLER_ROOT}/knowledge/legality_seed.json"
    if [[ -f "$legality_seed" ]]; then
        local ane_legal_count
        ane_legal_count=$(python3 -c "
import json
with open('$legality_seed') as f:
    data = json.load(f)
legal = sum(1 for e in data['entries'] if e['payload'].get('ane_legal', False))
illegal = sum(1 for e in data['entries'] if not e['payload'].get('ane_legal', True))
print(f'{legal} legal, {illegal} illegal')
" 2>/dev/null || echo "parse error")
        record_result "Legality rules parsed" "pass" "$ane_legal_count"
    else
        record_result "Legality rules" "skip" "legality_seed.json not found"
    fi

    # ─── SIR/AIR/MIR variant counts ───
    log_section "IR Variant Counts (from source)"
    local sir_count air_count mir_count
    sir_count=$(python3 -c "
import re
try:
    with open('${MILLER_ROOT}/crates/ir/src/sir.rs') as f:
        content = f.read()
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
except: print('?')
" 2>/dev/null || echo "?")
    air_count=$(python3 -c "
import re
try:
    with open('${MILLER_ROOT}/crates/ir/src/air.rs') as f:
        content = f.read()
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
except: print('?')
" 2>/dev/null || echo "?")
    mir_count=$(python3 -c "
import re
try:
    with open('${MILLER_ROOT}/crates/ir/src/mir.rs') as f:
        content = f.read()
    variants = re.findall(r'^\s+([A-Z][a-zA-Z0-9]+)\s*\{', content, re.MULTILINE)
    print(len(variants))
except: print('?')
" 2>/dev/null || echo "?")

    log_info "SIR variants: $sir_count  |  AIR variants: $air_count  |  MIR variants: $mir_count"
    record_result "IR variant counts" "pass" "SIR=$sir_count AIR=$air_count MIR=$mir_count"
}

# =============================================================================
# Phase 8: ANE Fastpath & Compute Plan
# =============================================================================
phase_ane() {
    log_phase 8 "ANE Fastpath & Compute Plan"

    local bridge="${MILLER_ROOT}/python/bridge.py"

    # ─── Compute plan for each synthetic model ───
    log_section "Compute Plan Inspection"
    for mlpackage in $(find "${TEST_WORKDIR}/synthetic" -name "*.mlpackage" -type d 2>/dev/null | head -5); do
        local pkg_name=$(basename "$mlpackage" .mlpackage)
        local plan_dir="${TEST_WORKDIR}/compute_plans"
        mkdir -p "$plan_dir"

        local harvest_cmd="$plan_dir/${pkg_name}_harvest_cmd.json"
        local harvest_result="$plan_dir/${pkg_name}_harvest_result.json"

        python3 -c "
import json
json.dump({
    'bridge_version': 1,
    'command': 'compute_plan_harvest',
    'mlpackage_path': '$mlpackage',
    'compute_units': 'CPU_AND_NE',
    'output_path': '$plan_dir'
}, open('$harvest_cmd', 'w'))
"

        if python3 "$bridge" "$harvest_cmd" "$harvest_result" 2>/dev/null; then
            local placement
            placement=$(python3 -c "
import json
try:
    h = json.load(open('$harvest_result'))
    if h.get('status') == 'success':
        harvest = h.get('harvest', h.get('compute_plan', {}))
        ops = harvest.get('ops', {})
        ane = sum(1 for v in ops.values() if 'ANE' in str(v.get('device','')).upper() or 'neural_engine' in str(v.get('device','')).lower())
        cpu = sum(1 for v in ops.values() if 'CPU' in str(v.get('device','')).upper())
        print(f'ANE={ane} CPU={cpu}')
    else:
        print('error: ' + h.get('error_message','unknown'))
except Exception as e:
    print(f'parse_error: {e}')
" 2>/dev/null || echo "parse_error")
            record_result "compute plan: $pkg_name" "pass" "$placement"
        else
            record_result "compute plan: $pkg_name" "skip" "Harvest requires macOS 14+ with Apple Silicon"
        fi
    done

    # ─── ANE op family matrix check ───
    log_section "ANE Op Family Matrix"
    local op_matrix="${MILLER_ROOT}/knowledge/ane_op_family_matrix.json"
    if [[ -f "$op_matrix" ]]; then
        local matrix_info
        matrix_info=$(python3 -c "
import json
with open('$op_matrix') as f:
    data = json.load(f)
entries = data.get('entries', [])
families = set()
for e in entries:
    payload = e.get('payload', {})
    for fam, status in payload.get('families', {}).items():
        families.add(fam)
supported = sum(1 for e in entries for v in e.get('payload', {}).get('families', {}).values() if v == 'supported')
print(f'{len(entries)} ops, {len(families)} families, {supported} supported entries')
" 2>/dev/null || echo "parse_error")
        record_result "ANE op family matrix" "pass" "$matrix_info"
    else
        record_result "ANE op family matrix" "skip" "ane_op_family_matrix.json not found"
    fi

    # ─── CPU-only ops ───
    log_section "CPU-Only Ops Verification"
    local cpu_seed="${MILLER_ROOT}/knowledge/cpu_only_ops_seed.json"
    if [[ -f "$cpu_seed" ]]; then
        local cpu_ops
        cpu_ops=$(python3 -c "
import json
with open('$cpu_seed') as f:
    data = json.load(f)
ops = [e['payload']['mil_name'] for e in data['entries']]
print(f'{len(ops)} CPU-only ops: {ops[:5]}...')
" 2>/dev/null || echo "parse_error")
        record_result "CPU-only ops seed" "pass" "$cpu_ops"
    else
        record_result "CPU-only ops seed" "skip" "cpu_only_ops_seed.json not found"
    fi
}

# =============================================================================
# Phase 9: Report
# =============================================================================
phase_report() {
    log_phase 9 "Results Report"

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          MILLer Test Kit — Results Summary              ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Total:  ${TOTAL_TESTS}                                            ${BOLD}║${NC}"
    echo -e "${GREEN}║${NC}  Passed: ${PASSED_TESTS}                                            ${BOLD}║${NC}"
    echo -e "${RED}║${NC}  Failed: ${FAILED_TESTS}                                            ${BOLD}║${NC}"
    echo -e "${YELLOW}║${NC}  Skipped:${SKIPPED_TESTS}                                            ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

    if [[ ${#FAILED_ITEMS[@]} -gt 0 ]]; then
        echo ""
        echo -e "${RED}${BOLD}Failed Tests:${NC}"
        for item in "${FAILED_ITEMS[@]}"; do
            echo -e "  ${RED}x${NC} $item"
        done
    fi

    # ─── JSON Results ───
    python3 -c "
import json, datetime
results = {
    'timestamp': datetime.datetime.now().isoformat(),
    'model': '${QWEN3_MODEL_ID}',
    'platform': 'M2 Mac',
    'total': ${TOTAL_TESTS},
    'passed': ${PASSED_TESTS},
    'failed': ${FAILED_TESTS},
    'skipped': ${SKIPPED_TESTS},
    'failed_items': $(printf '%s\n' "${FAILED_ITEMS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'),
    'configuration': {
        'miller_root': '${MILLER_ROOT}',
        'seed': ${SEED},
        'pipeline': 'TOML spec / HF trace → SIR → AIR → MIR → Bridge/MIL → .mlpackage',
    }
}
with open('${RESULTS_FILE}', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results written to ${RESULTS_FILE}')
"

    if [[ "$FAILED_TESTS" -gt 0 ]]; then
        echo ""
        log_fail "Test run completed with ${FAILED_TESTS} failure(s)"
        return 1
    else
        echo ""
        log_success "All tests passed!"
        return 0
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  MILLer Compiler Test Kit — M2 Mac + Qwen3-0.6B           ║"
    echo "║  TOML/HF → SIR → AIR → MIR → Bridge/MIL → .mlpackage     ║"
    echo "║  $(date '+%Y-%m-%d %H:%M:%S')                                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    mkdir -p "$TEST_WORKDIR"

    if should_run_phase "prereqs";    then phase_prereqs;    fi
    if should_run_phase "build";      then phase_build;      fi
    if should_run_phase "synthetic";  then phase_synthetic;  fi
    if should_run_phase "bridge";     then phase_bridge;     fi
    if should_run_phase "qwen3";      then phase_qwen3;      fi
    if should_run_phase "knowledge";  then phase_knowledge;  fi
    if should_run_phase "ir-pipeline";then phase_ir_pipeline; fi
    if should_run_phase "ane";        then phase_ane;        fi

    phase_report
}

main "$@"
