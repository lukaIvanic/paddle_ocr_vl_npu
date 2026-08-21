#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/decode_eager_vs_compiled_b1_lab.py"
REFERENCE_DIR="$SCRIPT_DIR/references/unirec_910b_decode_eager_compiled_b8_b32_b128_2241889"
BATCHES=(8 32 128)

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
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  CACHE_DIR="${CACHE_DIR:-$REPO/.runtime_cache/12_unirec_0_1b_inference/decode_eager_compiled_batch_310p_2241889}"
  mkdir -p "$CACHE_DIR"
  CACHE_DIR="$(readlink -f "$CACHE_DIR")"
  case "$ASCEND_RT_VISIBLE_DEVICES" in
    0|1|2|3) ;;
    *)
      printf 'UNIREC_310P_BATCH_MATRIX_DEVICE_MUST_BE_0_TO_3=%s\n' \
        "$ASCEND_RT_VISIBLE_DEVICES" >&2
      exit 1
      ;;
  esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  test -x "$PYTHON_BIN"
  test "$(basename "$PYTHON_BIN")" = python_nosym
  test -f "$MODEL/model.pth"
  test -s "$LAB"
  for batch in "${BATCHES[@]}"; do
    test -s "$REFERENCE_DIR/b${batch}.json"
  done
  test -s "$REFERENCE_DIR/summary.json"
  export PYTHON_BIN MODEL CACHE_DIR
}

cache_inventory() {
  find "$CACHE_DIR" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%P %s %T@\n' 2>/dev/null | sort >"$1"
}

cache_counts() {
  local om_count module_count
  om_count="$(find "$CACHE_DIR" -type f -name '*.om' | wc -l)"
  module_count="$(find "$CACHE_DIR" -type f -name compiled_module | wc -l)"
  printf '%s %s\n' "$om_count" "$module_count"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc|ge_compiler)$' || true
}

run_batch() {
  local run_root="$1" batch="$2"
  local batch_log="$run_root/b${batch}.log"
  local output="$run_root/b${batch}.json"
  local child status started now last_event counts om_count module_count
  local command=(
    "$PYTHON_BIN" "$LAB"
    --model "$MODEL"
    --device npu:0
    --cache-dir "$CACHE_DIR"
    --output "$output"
    --batch-size "$batch"
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
  } >"$run_root/b${batch}_command.sh"

  printf 'UNIREC_310P_BATCH_MATRIX_BEGIN batch=%s epoch=%s\n' \
    "$batch" "$(date +%s)"
  printf '%s\n' "$batch" >"$run_root/current_batch.txt"
  env PYTHONUNBUFFERED=1 TORCH_LOGS=recompiles UNIREC_STATIC_CACHE_LEN=256 \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "${command[@]}" >"$batch_log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$run_root/current_child_pid.txt"
  started="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep 5
    now="$(date +%s)"
    read -r om_count module_count <<<"$(cache_counts)"
    last_event="$(
      grep 'UNIREC_DECODE_EAGER_COMPILED_B1_PROGRESS' "$batch_log" \
        | tail -n 1 || true
    )"
    printf 'UNIREC_310P_BATCH_MATRIX_HEARTBEAT batch=%s batch_elapsed_s=%s compiler_processes=%s oms=%s modules=%s last_event=%q\n' \
      "$batch" "$((now - started))" "$(compiler_count)" \
      "$om_count" "$module_count" "$last_event"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$batch_log"
  if [[ "$status" -ne 0 ]]; then
    printf 'UNIREC_310P_BATCH_MATRIX_BATCH_FAILED batch=%s status=%s log=%s\n' \
      "$batch" "$status" "$batch_log" >&2
    return "$status"
  fi
  test -s "$output"
  ! grep -Eqi 'Skip cache as .*recompiled|Traceback|ERR[0-9]{5}|AICORE.*timeout' \
    "$batch_log"
  cache_inventory "$run_root/cache_after_b${batch}.txt"
  counts="$(cache_counts)"
  printf 'UNIREC_310P_BATCH_MATRIX_END batch=%s batch_wall_s=%s cache_counts=%q epoch=%s\n' \
    "$batch" "$(( $(date +%s) - started ))" "$counts" "$(date +%s)"
}

write_final_report() {
  local run_root="$1"
  "$PYTHON_BIN" - "$run_root" "$REFERENCE_DIR" <<'PY' \
    | tee "$run_root/final_report.txt"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reference_root = pathlib.Path(sys.argv[2])
batches = (8, 32, 128)
rows = []
for batch in batches:
    result = json.loads((root / f"b{batch}.json").read_text())
    reference = json.loads((reference_root / f"b{batch}.json").read_text())
    assert result["kind"] == "unirec_decode_eager_vs_compiled_b1_lab"
    assert result["status"] == "ok"
    assert result["shape"] == {
        "batch_size": batch,
        "self_cache_length": 256,
        "cross_cache_length": 256,
        "cache_position": 32,
        "source_length": 56,
        "attention_backend": "increfa_all",
        "semantic_heads": 6,
        "semantic_vocab_size": 56371,
        "lm_head_rows": 57344,
    }
    assert result["optimizations"]["decode_weight_format"] == "nz"
    assert result["optimizations"]["nz_tensor_count"] == 49
    assert result["optimizations"]["npu_jit_compile"] is False
    assert result["optimizations"]["qkv_projection"] == "separate"
    assert result["compile"]["torchair_ge_cache"] is True
    assert result["validation"]["token_exact"] is True
    assert reference["shape"] == result["shape"]
    eager = result["lanes"]["raw_eager"]
    compiled = result["lanes"]["compiled_torchair"]
    eager_ref = reference["lanes"]["raw_eager"]
    compiled_ref = reference["lanes"]["compiled_torchair"]
    rows.append(
        {
            "batch_size": batch,
            "raw_eager": eager,
            "compiled_torchair": compiled,
            "compiled_speedup": result["compiled_speedup"],
            "compiled_first_call_s_excluded": result[
                "compiled_first_call_s_excluded"
            ],
            "validation": result["validation"],
            "reference_910b": {
                "raw_eager": eager_ref,
                "compiled_torchair": compiled_ref,
                "compiled_speedup": reference["compiled_speedup"],
            },
            "raw_eager_ratio_vs_910b": (
                eager["median_raw_tok_s"] / eager_ref["median_raw_tok_s"]
            ),
            "compiled_ratio_vs_910b": (
                compiled["median_raw_tok_s"]
                / compiled_ref["median_raw_tok_s"]
            ),
        }
    )

report = {
    "schema": "unirec_310p_decode_eager_compiled_batch_matrix_v1",
    "status": "pass",
    "chip": "Ascend 310P",
    "measurement": (
        "100 queued full decode steps then one final device sync; "
        "compile, prefill, scheduling, and per-step D2H excluded"
    ),
    "contract": (
        "C256/S256 pos32 source56 FP16 increfa_all six heads; "
        "49 NZ weights; LM57344 sliced to 56371; separate QKV"
    ),
    "reference_910b_commit": "2241889",
    "rows": rows,
}
(root / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_DECODE_EAGER_COMPILED_BATCH_MATRIX: PASS")
print(
    "UNIREC_310P_DECODE_EAGER_COMPILED_BATCH_MATRIX_RESULT "
    + json.dumps(report, sort_keys=True)
)
PY
}

worker_main() {
  local run_root="$1"
  local counts om_before module_before om_after module_after
  resolve_inputs
  cache_inventory "$run_root/cache_before.txt"
  read -r om_before module_before <<<"$(cache_counts)"
  {
    printf 'commit=%s\nphysical_npu=%s\npython=%s\nmodel=%s\ncache_dir=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
      "$PYTHON_BIN" "$MODEL" "$CACHE_DIR"
    printf 'om_before=%s\ncompiled_module_before=%s\n' \
      "$om_before" "$module_before"
    "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
PY
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  for batch in "${BATCHES[@]}"; do
    run_batch "$run_root" "$batch"
  done
  rm -f "$run_root/current_batch.txt" "$run_root/current_child_pid.txt"
  cache_inventory "$run_root/cache_after.txt"
  diff -u "$run_root/cache_before.txt" "$run_root/cache_after.txt" \
    >"$run_root/cache.diff" || true
  read -r om_after module_after <<<"$(cache_counts)"
  printf 'om_before=%s\nom_after=%s\ncompiled_module_before=%s\ncompiled_module_after=%s\n' \
    "$om_before" "$om_after" "$module_before" "$module_after" \
    >"$run_root/cache_counts.txt"
  write_final_report "$run_root"
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_BATCH_MATRIX_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short stamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  stamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_eager_compiled_batch_matrix_${short}_${stamp}}"
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
