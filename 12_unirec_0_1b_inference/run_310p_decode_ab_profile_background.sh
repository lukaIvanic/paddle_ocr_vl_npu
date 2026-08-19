#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/text_decode_lab.py"
PARSER="$REPO/03_compiled_single_batch_decode/parse_npu_profile.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

phase() {
  printf 'UNIREC_310P_DECODE_AB_PROFILE_PHASE phase=%s epoch_s=%s\n' \
    "$1" "$(date +%s)"
}

cache_inventory() {
  find "$DECODE_CACHE_PARENT" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %T@\n' | sort >"$1"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc)$' || true
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${DECODE_CACHE_PARENT:?export the completed dual-decode cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  DECODE_CACHE_PARENT="$(readlink -f "$DECODE_CACHE_PARENT")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$DECODE_CACHE_PARENT"
  for shape in \
    decode_selfkv256_cross256_increfa_all_b128 \
    decode_selfkv2048_cross1320_increfa_all_b128
  do
    local cache="$DECODE_CACHE_PARENT/$shape"
    test "$(find "$cache" -type f -name compiled_module | wc -l)" -eq 1
    test "$(find "$cache" -type f -name '*.om' | wc -l)" -eq 1
  done
}

run_lane() {
  local lane="$1" self_len="$2" cross_len="$3" position="$4"
  local lane_root="$RUN_ROOT/$lane"
  local lane_log="$lane_root/run.log"
  local child started status now
  mkdir -p "$lane_root"
  phase "${lane}_begin"
  started="$(date +%s)"
  env PYTHONUNBUFFERED=1 UNIREC_STATIC_CACHE_LEN="$self_len" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$LAB" \
      --model "$MODEL" --device npu:0 --dtype float16 \
      --batch-size 128 --self-cache-length "$self_len" \
      --cross-cache-length "$cross_len" --cache-position "$position" \
      --warmup-steps 8 --measure-steps 20 --validation-steps 1 \
      --profile-steps 0 --profile-compiled-steps 1 --profile-metric pipe \
      --backends increfa_all --compiled-timing-steps 0 --graph-mode ge \
      --reuse-state --cache-dir "$DECODE_CACHE_PARENT" \
      --output "$lane_root/result.json" >"$lane_log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$lane_root/pid.txt"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_310P_DECODE_AB_PROFILE_HEARTBEAT lane=%s elapsed_s=%s compiler_processes=%s last_event=%q\n' \
      "$lane" "$((now - started))" "$(compiler_count)" \
      "$(grep 'UNIREC_DECODE_LAB' "$lane_log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$lane_log"
  printf 'UNIREC_310P_DECODE_AB_PROFILE_LANE_END lane=%s status=%s wall_s=%s\n' \
    "$lane" "$status" "$(( $(date +%s) - started ))"
  test "$status" -eq 0
  local profile_dir="$lane_root/profile_compiled_increfa_all_pipe"
  test -d "$profile_dir"
  "$PYTHON_BIN" "$PARSER" --profile-dir "$profile_dir" --topn 40 \
    --skip-trace --out-json "$lane_root/profile.json" \
    --out-md "$lane_root/profile.md"
  phase "${lane}_end"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  ulimit -n 65536 2>/dev/null || true
  cache_inventory "$RUN_ROOT/om_before.txt"
  run_lane a 256 256 32
  run_lane b 2048 1320 1023
  cache_inventory "$RUN_ROOT/om_after.txt"
  diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
    >"$RUN_ROOT/om.diff"
  ! grep -ERqi \
    'recompil|skip cache|compile graph|start.*compil|Traceback|ERROR' \
    "$RUN_ROOT/a/run.log" "$RUN_ROOT/b/run.log"

  A_RESULT="$RUN_ROOT/a/result.json" B_RESULT="$RUN_ROOT/b/result.json" \
    A_PROFILE="$RUN_ROOT/a/profile.json" B_PROFILE="$RUN_ROOT/b/profile.json" \
    RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' \
      | tee "$RUN_ROOT/final_report.txt"
import json
import os

def load_result(name):
    value = json.load(open(os.environ[name]))
    return value, value["lanes"]["increfa_all"]

def load_profile(name):
    value = json.load(open(os.environ[name]))
    assert len(value["runs"]) == 1
    return value["runs"][0]["kernel_details"]

a_result, a_lane = load_result("A_RESULT")
b_result, b_lane = load_result("B_RESULT")
a_profile = load_profile("A_PROFILE")
b_profile = load_profile("B_PROFILE")
assert (
    a_result["shape"]["batch_size"],
    a_result["shape"]["self_cache_length"],
    a_result["shape"]["cross_cache_length"],
) == (128, 256, 256)
assert (
    b_result["shape"]["batch_size"],
    b_result["shape"]["self_cache_length"],
    b_result["shape"]["cross_cache_length"],
) == (128, 2048, 1320)

def kernel_types(profile):
    return {
        row["name"]: {
            "count": row["count"],
            "duration_us": row["duration_us"],
        }
        for row in profile["top_kernel_types"]
    }

a_types = kernel_types(a_profile)
b_types = kernel_types(b_profile)
important = (
    "IncreFlashAttention",
    "MatMulV2",
    "MatMul",
    "AddLayerNorm",
    "Scatter",
    "Relu",
    "ArgMaxV2",
    "Cast",
    "TransData",
    "Greater",
    "Range",
)
report = {
    "status": "pass",
    "a": {
        "step_ms": a_lane["measure"]["step_ms"],
        "raw_tok_s": a_lane["measure"]["raw_tok_s"],
        "first_call_s": a_lane["first_call_s"],
        "kernel_total_us": a_profile["total_duration_us"],
        "kernel_rows": a_profile["row_count"],
        "kernel_types": {key: a_types.get(key) for key in important},
    },
    "b": {
        "step_ms": b_lane["measure"]["step_ms"],
        "raw_tok_s": b_lane["measure"]["raw_tok_s"],
        "first_call_s": b_lane["first_call_s"],
        "kernel_total_us": b_profile["total_duration_us"],
        "kernel_rows": b_profile["row_count"],
        "kernel_types": {key: b_types.get(key) for key in important},
    },
    "b_minus_a_us": {
        key: (
            (b_types.get(key) or {}).get("duration_us", 0.0)
            - (a_types.get(key) or {}).get("duration_us", 0.0)
        )
        for key in important
    },
    "a_over_b_raw_tok_s": (
        a_lane["measure"]["raw_tok_s"] / b_lane["measure"]["raw_tok_s"]
    ),
    "references_910b": {
        "a_kernel_total_us": 2020.66,
        "b_kernel_total_us": 5930.42,
        "a_increfa_us": 1153.26,
        "b_increfa_us": 4888.68,
    },
    "run_root": os.environ["RUN_ROOT"],
}
print("UNIREC_310P_DECODE_AB_PROFILE: PASS")
print("UNIREC_310P_DECODE_AB_PROFILE_RESULT " + json.dumps(report, sort_keys=True))
PY
  phase complete
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short stamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  stamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_ab_profile_${short}_${stamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    DECODE_CACHE_PARENT="$DECODE_CACHE_PARENT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
