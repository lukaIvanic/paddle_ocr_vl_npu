#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SWEEP="$SCRIPT_DIR/decode_mask_occupancy_sweep.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc)$' || true
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated Python executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${ARTIFACT_DIR:?export the canonical first-128 artifact directory}"
  : "${DECODE_CACHE_PARENT:?export the decode cache parent}"
  : "${BASELINE_A_RESULT:?export the completed B128 mask-sweep a/result.json}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one physical NPU}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  ARTIFACT_DIR="$(readlink -f "$ARTIFACT_DIR")"
  DECODE_CACHE_PARENT="$(readlink -f "$DECODE_CACHE_PARENT")"
  BASELINE_A_RESULT="$(readlink -f "$BASELINE_A_RESULT")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -s "$ARTIFACT_DIR/crops.jsonl"
  test -d "$DECODE_CACHE_PARENT"
  test -s "$BASELINE_A_RESULT"
}

cache_inventory() {
  find "$DECODE_CACHE_PARENT" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %T@\n' | sort >"$1"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  local shape_cache child started now status
  shape_cache="$DECODE_CACHE_PARENT/decode_selfkv256_cross256_increfa_all_b64"
  if [[ -d "$shape_cache" ]]; then
    printf '1\n' >"$RUN_ROOT/b64_cache_preexisted.txt"
  else
    printf '0\n' >"$RUN_ROOT/b64_cache_preexisted.txt"
  fi
  cache_inventory "$RUN_ROOT/om_before.txt"

  env PYTHONUNBUFFERED=1 UNIREC_STATIC_CACHE_LEN=256 \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$SWEEP" --model "$MODEL" \
      --artifact-crops-jsonl "$ARTIFACT_DIR/crops.jsonl" --device npu:0 \
      --batch-size 64 --self-cache-length 256 --cross-cache-length 256 \
      --active-rows 64 --source-modes realistic full --cache-positions 32 \
      --warmup-steps 10 --measure-steps 50 \
      --cache-dir "$DECODE_CACHE_PARENT" --output "$RUN_ROOT/result.json" \
      >"$RUN_ROOT/lane.log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$RUN_ROOT/lane_pid.txt"
  started="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_DECODE_A_B64_HEARTBEAT elapsed_s=%s compiler_processes=%s last_event=%q\n' \
      "$((now - started))" "$(compiler_count)" \
      "$(grep -E 'UNIREC_DECODE_MASK_SWEEP_(PROGRESS|POINT)' "$RUN_ROOT/lane.log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$RUN_ROOT/lane.log"
  printf 'UNIREC_DECODE_A_B64_LANE_END status=%s wall_s=%s\n' \
    "$status" "$(( $(date +%s) - started ))"
  test "$status" -eq 0
  ! grep -Eq 'Skip cache as .*recompiled' "$RUN_ROOT/lane.log"

  cache_inventory "$RUN_ROOT/om_after.txt"
  diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
    >"$RUN_ROOT/om.diff" || true
  test "$(find "$shape_cache" -type f -name compiled_module | wc -l)" -eq 1
  test "$(find "$shape_cache" -type f -name '*.om' | wc -l)" -ge 1

  B64_RESULT="$RUN_ROOT/result.json" B128_RESULT="$BASELINE_A_RESULT" \
    RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/final_report.txt"
import json
import os

b64 = json.load(open(os.environ["B64_RESULT"]))
b128 = json.load(open(os.environ["B128_RESULT"]))

def points(value):
    return {
        (
            int(point["active_rows"]),
            int(point["initial_cache_position"]),
            str(point["source_mode"]),
        ): point
        for point in value["points"]
    }

assert b64["status"] == "ok"
assert b128["status"] == "ok"
assert b64["shape"] == {
    "batch_size": 64,
    "self_cache_length": 256,
    "cross_cache_length": 256,
}
assert b128["shape"] == {
    "batch_size": 128,
    "self_cache_length": 256,
    "cross_cache_length": 256,
}
p64 = points(b64)
p128 = points(b128)
b64_real = p64[(64, 32, "realistic")]
b64_full = p64[(64, 32, "full")]
b128_real = p128[(128, 32, "realistic")]
b128_full = p128[(128, 32, "full")]

report = {
    "status": "pass",
    "b64_cache_preexisted": bool(
        int(open(os.path.join(os.environ["RUN_ROOT"], "b64_cache_preexisted.txt")).read())
    ),
    "b64_first_call_s": b64["first_call_s"],
    "b64_realistic": {
        "step_ms": b64_real["decode_step_ms"],
        "raw_tok_s": b64_real["raw_tok_s"],
        "source_mean": b64_real["source_lengths"]["mean"],
    },
    "b64_full": {
        "step_ms": b64_full["decode_step_ms"],
        "raw_tok_s": b64_full["raw_tok_s"],
    },
    "b128_realistic": {
        "step_ms": b128_real["decode_step_ms"],
        "raw_tok_s": b128_real["raw_tok_s"],
        "source_mean": b128_real["source_lengths"]["mean"],
    },
    "b128_full": {
        "step_ms": b128_full["decode_step_ms"],
        "raw_tok_s": b128_full["raw_tok_s"],
    },
    "b64_over_b128_realistic_raw_tok_s": (
        b64_real["raw_tok_s"] / b128_real["raw_tok_s"]
    ),
    "b64_step_over_half_b128_step": (
        b64_real["decode_step_ms"] / (b128_real["decode_step_ms"] / 2.0)
    ),
    "b64_beats_raw_throughput_break_even": (
        b64_real["decode_step_ms"] < b128_real["decode_step_ms"] / 2.0
    ),
    "run_root": os.environ["RUN_ROOT"],
}
print("UNIREC_DECODE_A_B64: PASS")
print("UNIREC_DECODE_A_B64_RESULT " + json.dumps(report, sort_keys=True))
PY
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/decode_a_b64_${short}_${stamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    ARTIFACT_DIR="$ARTIFACT_DIR" DECODE_CACHE_PARENT="$DECODE_CACHE_PARENT" \
    BASELINE_A_RESULT="$BASELINE_A_RESULT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
