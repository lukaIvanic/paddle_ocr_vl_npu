#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/text_decode_lab.py"

phase() {
  printf 'UNIREC_310P_DUAL_DECODE_LAB_PHASE phase=%s epoch_s=%s\n' \
    "$1" "$(date +%s)"
}

cache_inventory() {
  local directory="$1" output="$2"
  if [[ -d "$directory" ]]; then
    find "$directory" -type f \( -name compiled_module -o -name '*.om' \) \
      -printf '%p %s %T@\n' | sort >"$output"
  else
    : >"$output"
  fi
}

count_cache_files() {
  local directory="$1" pattern="$2"
  if [[ -d "$directory" ]]; then
    find "$directory" -type f -name "$pattern" | wc -l
  else
    printf '0\n'
  fi
}

run_lane() {
  local lane="$1" self_len="$2" cross_len="$3" position="$4"
  local measure_steps="$5" timing_steps="$6" output="$7"
  local lane_log="$RUN_ROOT/${lane}.log"
  local started now child status
  local command=(
    "$PYTHON_BIN" "$LAB"
    --model "$MODEL"
    --device npu:0 --dtype float16
    --batch-size 128
    --self-cache-length "$self_len"
    --cross-cache-length "$cross_len"
    --cache-position "$position"
    --warmup-steps 8
    --measure-steps "$measure_steps"
    --validation-steps 8
    --profile-steps 0 --profile-compiled-steps 0
    --backends increfa_all
    --compiled-timing-steps "$timing_steps"
    --graph-mode ge
    --cache-dir "$DECODE_CACHE_PARENT"
    --output "$output"
  )
  printf '%q ' env "UNIREC_STATIC_CACHE_LEN=$self_len" "${command[@]}" \
    >"$RUN_ROOT/${lane}_command.sh"
  printf '\n' >>"$RUN_ROOT/${lane}_command.sh"
  phase "${lane}_begin"
  started="$(date +%s)"
  env PYTHONUNBUFFERED=1 \
    UNIREC_STATIC_CACHE_LEN="$self_len" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "${command[@]}" >"$lane_log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$RUN_ROOT/${lane}_pid.txt"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_310P_DUAL_DECODE_LAB_HEARTBEAT lane=%s elapsed_s=%s compiled_modules=%s oms=%s compiler_processes=%s last_event=%q\n' \
      "$lane" "$((now - started))" \
      "$(count_cache_files "$DECODE_CACHE_PARENT" compiled_module)" \
      "$(count_cache_files "$DECODE_CACHE_PARENT" '*.om')" \
      "$(pgrep -af 'ccec_compiler|op_compiler|tbe.*compile|atc' \
          | grep -vF 'pgrep -af' | wc -l || true)" \
      "$(grep 'UNIREC_DECODE_LAB' "$lane_log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$lane_log"
  printf 'UNIREC_310P_DUAL_DECODE_LAB_LANE_END lane=%s status=%s wall_s=%s\n' \
    "$lane" "$status" "$(( $(date +%s) - started ))"
  [[ "$status" -eq 0 ]]
}

worker_main() {
  RUN_ROOT="$1"
  : "${PYTHON_BIN:?export the validated venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${DECODE_CACHE_PARENT:?export the exact production decode-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$DECODE_CACHE_PARENT"
  test -f "$LAB"

  local a_cache="$DECODE_CACHE_PARENT/decode_selfkv256_cross256_increfa_all_b128"
  local b_cache="$DECODE_CACHE_PARENT/decode_selfkv2048_cross1320_increfa_all_b128"
  local a_before b_before
  a_before="$(count_cache_files "$a_cache" '*.om')"
  b_before="$(count_cache_files "$b_cache" '*.om')"
  [[ "$a_before" -eq 0 || "$a_before" -eq 1 ]]
  [[ "$b_before" -eq 1 ]]
  test "$(find "$b_cache" -type f -name compiled_module | wc -l)" -eq 1

  ulimit -n 65536 2>/dev/null || true
  printf 'project_commit=%s\nphysical_npu=%s\npython=%s\nmodel=%s\ncache_parent=%s\nsoft_nofile=%s\na_om_before=%s\nb_om_before=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$MODEL" "$DECODE_CACHE_PARENT" "$(ulimit -n)" \
    "$a_before" "$b_before" >"$RUN_ROOT/preflight.txt"
  cache_inventory "$DECODE_CACHE_PARENT" "$RUN_ROOT/om_before.txt"
  cache_inventory "$b_cache" "$RUN_ROOT/b_om_before.txt"
  cache_inventory "$a_cache" "$RUN_ROOT/a_om_before.txt"

  run_lane a 256 256 32 200 100 "$RUN_ROOT/a_result.json"
  run_lane b 2048 1320 1023 300 100 "$RUN_ROOT/b_result.json"

  cache_inventory "$DECODE_CACHE_PARENT" "$RUN_ROOT/om_after.txt"
  cache_inventory "$b_cache" "$RUN_ROOT/b_om_after.txt"
  cache_inventory "$a_cache" "$RUN_ROOT/a_om_after.txt"
  diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
    >"$RUN_ROOT/om.diff" || true
  diff -u "$RUN_ROOT/b_om_before.txt" "$RUN_ROOT/b_om_after.txt" \
    >"$RUN_ROOT/b_om.diff"
  test "$(find "$a_cache" -type f -name compiled_module | wc -l)" -eq 1
  test "$(find "$a_cache" -type f -name '*.om' | wc -l)" -eq 1
  test "$(find "$b_cache" -type f -name compiled_module | wc -l)" -eq 1
  test "$(find "$b_cache" -type f -name '*.om' | wc -l)" -eq 1
  if [[ "$a_before" -eq 1 ]]; then
    diff -u "$RUN_ROOT/a_om_before.txt" "$RUN_ROOT/a_om_after.txt" \
      >"$RUN_ROOT/a_om.diff"
    diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
      >"$RUN_ROOT/om_unexpected.diff"
  else
    : >"$RUN_ROOT/a_om.diff"
    grep -vF "$a_cache/" "$RUN_ROOT/om_after.txt" \
      >"$RUN_ROOT/om_after_without_new_a.txt"
    diff -u "$RUN_ROOT/om_before.txt" \
      "$RUN_ROOT/om_after_without_new_a.txt" \
      >"$RUN_ROOT/om_unexpected.diff"
  fi

  A_RESULT="$RUN_ROOT/a_result.json" B_RESULT="$RUN_ROOT/b_result.json" \
    A_BEFORE="$a_before" RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' \
      | tee "$RUN_ROOT/final_report.txt"
import json
import os

a = json.load(open(os.environ["A_RESULT"]))
b = json.load(open(os.environ["B_RESULT"]))
for payload, shape in (
    (a, (128, 256, 256)),
    (b, (128, 2048, 1320)),
):
    assert payload["kind"] == "unirec_text_decode_lab"
    assert (
        payload["shape"]["batch_size"],
        payload["shape"]["self_cache_length"],
        payload["shape"]["cross_cache_length"],
    ) == shape
    lane = payload["lanes"]["increfa_all"]
    assert lane["compile"]["mask_mode"] == "per_step"
    assert lane["compile"]["batch_size"] == 128

def row(payload):
    lane = payload["lanes"]["increfa_all"]
    clean = lane["measure"]
    d2h = lane["compiled_timing"]["production_like_d2h"]
    return {
        "first_call_s": lane["first_call_s"],
        "clean_step_ms": clean["step_ms"],
        "clean_raw_tok_s": clean["raw_tok_s"],
        "d2h_step_ms": d2h["wall_step_ms"],
        "d2h_raw_tok_s": 128000.0 / d2h["wall_step_ms"],
        "peak_hbm_gib": payload["npu_memory"]["max_allocated_bytes"] / 2**30,
    }

ar = row(a)
br = row(b)
a_iters = 11621
b_iters = 8619
single_b_iters = 19388
dual_clean_s = (a_iters * ar["clean_step_ms"] + b_iters * br["clean_step_ms"]) / 1000
single_clean_s = single_b_iters * br["clean_step_ms"] / 1000
dual_d2h_s = (a_iters * ar["d2h_step_ms"] + b_iters * br["d2h_step_ms"]) / 1000
single_d2h_s = single_b_iters * br["d2h_step_ms"] / 1000
report = {
    "status": "pass",
    "a_cache_was_precompiled": bool(int(os.environ["A_BEFORE"])),
    "a": ar,
    "b": br,
    "workload_iterations": {
        "a": a_iters,
        "b": b_iters,
        "single_b": single_b_iters,
    },
    "projection": {
        "dual_clean_graph_s": dual_clean_s,
        "single_b_clean_graph_s": single_clean_s,
        "clean_speedup": single_clean_s / dual_clean_s,
        "dual_d2h_s": dual_d2h_s,
        "single_b_d2h_s": single_d2h_s,
        "d2h_speedup": single_d2h_s / dual_d2h_s,
    },
    "run_root": os.environ["RUN_ROOT"],
}
print("UNIREC_310P_DUAL_DECODE_LAB: PASS")
print("UNIREC_310P_DUAL_DECODE_RESULT " + json.dumps(report, sort_keys=True))
PY
  phase complete
}

launch_main() {
  : "${PYTHON_BIN:?export the validated venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${DECODE_CACHE_PARENT:?export the exact production decode-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  local short stamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  stamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_dual_decode_lab_${short}_${stamp}}"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    DECODE_CACHE_PARENT="$DECODE_CACHE_PARENT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then
  set +e
  (set -e; worker_main "$2")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$2/exit_code.txt"
  exit "$status"
else
  launch_main
fi
