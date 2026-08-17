#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
COMPARE="$SCRIPT_DIR/compare_decode_completion_traces.py"
REFERENCE="$SCRIPT_DIR/references/unirec_910b_decode_diagnostic_first128_039a633/recognition_token_digests.jsonl"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P UniRec venv executable}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${COMPILE_CACHE:?export the production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?reuse the passed decode-cache gate parent}"
  : "${PRIOR_RUN_ROOT:?export the completed 310P decode diagnostic run root}"
  : "${CPUSET:=0-63}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  PRIOR_RUN_ROOT="$(readlink -f "$PRIOR_RUN_ROOT")"
  UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$(readlink -f "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE")"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) echo "310P_DEVICE_MUST_BE_0_TO_3" >&2; exit 1;; esac
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$COMPILE_CACHE"
  test -f "$PRIOR_RUN_ROOT/clean.json"
  test -f "$REFERENCE"
  ARTIFACT_DIR="${ARTIFACT_DIR:-$(jq -r '.config.artifact_dir' "$PRIOR_RUN_ROOT/clean.json")}"
  ARTIFACT_DIR="$(readlink -f "$ARTIFACT_DIR")"
  test -f "$ARTIFACT_DIR/summary.json"
  test -f "$ARTIFACT_DIR/crops.jsonl"
  test -f "$ARTIFACT_DIR/cross_kv.bin"
  EXACT_CACHE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$EXACT_CACHE" -name compiled_module | wc -l)" -eq 1
  test "$(find "$EXACT_CACHE" -name '*.om' | wc -l)" -eq 1
}

cache_om_inventory() {
  find "$EXACT_CACHE" -type f -name '*.om' -exec sha256sum '{}' \; | sort
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  cache_om_inventory >"$run_root/cache_om_before.sha256"
  echo "PRIOR_CLEAN_SUMMARY"
  jq -c '{selected_crops:.workload.selected_crops,generated_length:.workload.generated_length,iterations:.decode.decode_iterations,graph_s:.decode.decode_s,raw_tok_s:.decode.raw_decode_tokens_per_s,effective_tok_s:.decode.effective_decode_tokens_per_s,slot_efficiency:.slot_efficiency,validation:.validation}' "$PRIOR_RUN_ROOT/clean.json"
  taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY" \
    --artifact-dir "$ARTIFACT_DIR" --model-path "$MODEL" --device npu:0 \
    --dtype float16 --batch-size 128 --self-cache-length 2048 \
    --cross-cache-length 1320 --max-length 2048 --limit-crops 256 \
    --decode-warmup-passes 2 --decode-admission-prefetch-depth 0 \
    --compile-cache-dir "$COMPILE_CACHE" --progress-every 16 \
    --reference-trace "$REFERENCE" \
    --completion-trace-jsonl "$run_root/completions.jsonl" \
    --output "$run_root/replay.json" | tee "$run_root/replay.log"
  "$PYTHON_BIN" "$COMPARE" --candidate "$run_root/completions.jsonl" \
    --reference "$REFERENCE" --output "$run_root/parity_report.json" \
    | tee "$run_root/final_report.txt"
  cache_om_inventory >"$run_root/cache_om_after.sha256"
  if diff -u "$run_root/cache_om_before.sha256" "$run_root/cache_om_after.sha256" \
      >"$run_root/cache_om.diff"; then
    echo "DECODE_CACHE_OM_INVENTORY_UNCHANGED"
  else
    echo "DECODE_CACHE_OM_INVENTORY_CHANGED" >&2
    cat "$run_root/cache_om.diff" >&2
    return 1
  fi
}

worker_entry() {
  local run_root="$1" status=0
  set +e
  worker_main "$run_root"
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_output_parity_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    COMPILE_CACHE="$COMPILE_CACHE" PRIOR_RUN_ROOT="$PRIOR_RUN_ROOT" \
    ARTIFACT_DIR="$ARTIFACT_DIR" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
