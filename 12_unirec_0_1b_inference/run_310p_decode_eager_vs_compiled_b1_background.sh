#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/decode_eager_vs_compiled_b1_lab.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then
    command -v "$value"
    return
  fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${CACHE_DIR:?export the warmed vocab57344 B1 cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  CACHE_DIR="$(readlink -f "$CACHE_DIR")"
  case "$ASCEND_RT_VISIBLE_DEVICES" in
    0|1|2|3) ;;
    *) printf '310P_DEVICE_MUST_BE_0_TO_3=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2; exit 1 ;;
  esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  test -x "$PYTHON_BIN"
  test "$(basename "$PYTHON_BIN")" = python_nosym
  test -f "$MODEL/model.pth"
  test -d "$CACHE_DIR"
  test -s "$LAB"
  export PYTHON_BIN MODEL CACHE_DIR
}

cache_inventory() {
  find "$CACHE_DIR" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%P %s %T@\n' 2>/dev/null | sort >"$1"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc|ge_compiler)$' || true
}

worker_main() {
  local run_root="$1" child status started now first_call_begin
  resolve_inputs
  cache_inventory "$run_root/cache_before.txt"
  local om_before module_before
  om_before="$(find "$CACHE_DIR" -type f -name '*.om' | wc -l)"
  module_before="$(find "$CACHE_DIR" -type f -name compiled_module | wc -l)"
  if [[ "$om_before" -lt 1 || "$module_before" -lt 1 ]]; then
    printf 'UNIREC_310P_DECODE_EAGER_COMPILED_B1_CACHE_MISSING oms=%s modules=%s cache=%s\n' \
      "$om_before" "$module_before" "$CACHE_DIR" >&2
    return 1
  fi

  {
    printf 'commit=%s\nphysical_npu=%s\npython=%s\nmodel=%s\ncache_dir=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
      "$PYTHON_BIN" "$MODEL" "$CACHE_DIR"
    printf 'om_before=%s\ncompiled_module_before=%s\n' "$om_before" "$module_before"
    "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
PY
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  local command=(
    "$PYTHON_BIN" "$LAB"
    --model "$MODEL"
    --device npu:0
    --cache-dir "$CACHE_DIR"
    --output "$run_root/result.json"
    --source-length 56
    --warmup-steps 20
    --measure-steps 100
    --repeats 5
    --validation-steps 8
  )
  {
    printf 'UNIREC_STATIC_CACHE_LEN=256 '
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$run_root/command.sh"

  env PYTHONUNBUFFERED=1 TORCH_LOGS=recompiles UNIREC_STATIC_CACHE_LEN=256 \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "${command[@]}" >"$run_root/probe.log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$run_root/probe_pid.txt"
  started="$(date +%s)"
  first_call_begin=""
  while kill -0 "$child" 2>/dev/null; do
    sleep 5
    now="$(date +%s)"
    if grep -q '"event": "compiled_first_call_begin"' "$run_root/probe.log"; then
      [[ -n "$first_call_begin" ]] || first_call_begin="$now"
    fi
    local om_now module_now last_event
    om_now="$(find "$CACHE_DIR" -type f -name '*.om' | wc -l)"
    module_now="$(find "$CACHE_DIR" -type f -name compiled_module | wc -l)"
    last_event="$(grep 'UNIREC_DECODE_EAGER_COMPILED_B1_PROGRESS' "$run_root/probe.log" | tail -n 1 || true)"
    printf 'UNIREC_310P_DECODE_EAGER_COMPILED_B1_HEARTBEAT elapsed_s=%s compiler_processes=%s oms=%s modules=%s last_event=%q\n' \
      "$((now - started))" "$(compiler_count)" "$om_now" "$module_now" "$last_event"
    if [[ "$om_now" != "$om_before" || "$module_now" != "$module_before" ]]; then
      printf 'UNIREC_310P_DECODE_EAGER_COMPILED_B1_UNEXPECTED_CACHE_CHANGE before=%s/%s now=%s/%s\n' \
        "$om_before" "$module_before" "$om_now" "$module_now" >&2
      kill "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
      cat "$run_root/probe.log"
      return 1
    fi
    if [[ -n "$first_call_begin" ]] \
      && ! grep -q '"event": "compiled_first_call_end"' "$run_root/probe.log" \
      && (( now - first_call_begin > 60 )); then
      printf 'UNIREC_310P_DECODE_EAGER_COMPILED_B1_COLD_FIRST_CALL elapsed_s=%s\n' \
        "$((now - first_call_begin))" >&2
      kill "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
      cat "$run_root/probe.log"
      return 1
    fi
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$run_root/probe.log"
  test "$status" -eq 0
  test -s "$run_root/result.json"
  ! grep -Eqi 'Skip cache as .*recompiled|Traceback|ERR[0-9]{5}|AICORE.*timeout' \
    "$run_root/probe.log"
  cache_inventory "$run_root/cache_after.txt"
  diff -u "$run_root/cache_before.txt" "$run_root/cache_after.txt" \
    >"$run_root/cache.diff"

  "$PYTHON_BIN" - "$run_root/result.json" <<'PY' | tee "$run_root/final_report.txt"
import json
import sys

p = json.load(open(sys.argv[1]))
assert p["kind"] == "unirec_decode_eager_vs_compiled_b1_lab"
assert p["status"] == "ok"
assert p["shape"] == {
    "batch_size": 1,
    "self_cache_length": 256,
    "cross_cache_length": 256,
    "cache_position": 32,
    "source_length": 56,
    "attention_backend": "increfa_all",
    "semantic_heads": 6,
    "semantic_vocab_size": 56371,
    "lm_head_rows": 57344,
}
assert p["optimizations"]["decode_weight_format"] == "nz"
assert p["optimizations"]["nz_tensor_count"] == 49
assert p["optimizations"]["npu_jit_compile"] is False
assert p["validation"]["token_exact"]
assert p["compile"]["torchair_ge_cache"] is True

reference = {
    "chip": "Ascend910B2",
    "physical_npu": 7,
    "commit": "52de30b",
    "raw_eager_tok_s": 154.76962087602277,
    "compiled_tok_s": 1983.6599563242323,
    "compiled_speedup": 12.81685608000055,
}
eager = p["lanes"]["raw_eager"]
compiled = p["lanes"]["compiled_torchair"]
report = {
    "schema": "unirec_310p_decode_eager_vs_compiled_b1_v1",
    "status": "pass",
    "shape": p["shape"],
    "optimizations": p["optimizations"],
    "raw_eager": eager,
    "compiled_torchair": compiled,
    "compiled_speedup": p["compiled_speedup"],
    "compiled_first_call_s_excluded": p["compiled_first_call_s_excluded"],
    "validation": p["validation"],
    "reference_910b": reference,
    "raw_eager_ratio_vs_910b": (
        eager["median_raw_tok_s"] / reference["raw_eager_tok_s"]
    ),
    "compiled_ratio_vs_910b": (
        compiled["median_raw_tok_s"] / reference["compiled_tok_s"]
    ),
}
print("UNIREC_310P_DECODE_EAGER_VS_COMPILED_B1: PASS")
print("UNIREC_310P_DECODE_EAGER_VS_COMPILED_B1_RESULT " + json.dumps(report, sort_keys=True))
PY
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_DECODE_EAGER_COMPILED_B1_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short stamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  stamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_eager_vs_compiled_b1_${short}_${stamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    CACHE_DIR="$CACHE_DIR" ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == worker ]]; then
  worker_entry "$2"
else
  launch_main
fi
