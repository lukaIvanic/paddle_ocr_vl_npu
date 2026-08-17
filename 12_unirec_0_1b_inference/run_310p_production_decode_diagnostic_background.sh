#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
REPORT="$SCRIPT_DIR/report_310p_decode_diagnostic.py"
REFERENCE="$SCRIPT_DIR/references/unirec_910b_decode_diagnostic_first128_039a633/reference_summary.json"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P UniRec venv executable}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench images directory}"
  : "${COMPILE_CACHE:?export the production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?export config.cache_parent from the preceding passed decode-cache gate}"
  : "${CPUSET:=0-63}"
  : "${ARTIFACT_DIR:=}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$(readlink -f "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -d "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE"
  test -f "$REFERENCE"
  local exact_cache
  exact_cache="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$exact_cache" -name compiled_module | wc -l)" -eq 1
  test "$(find "$exact_cache" -name '*.om' | wc -l)" -eq 1
}

run_prefill() {
  local artifact="$1"
  env UNIREC_VISION_BUCKET_PRESET=310p_k10_l4_all \
    taskset -c "$CPUSET" "$PYTHON_BIN" "$PREFILL" \
    --openocr-root "$OPENOCR_ROOT" --model-path "$MODEL" \
    --layout-model "$LAYOUT_MODEL" --input "$IMAGES_DIR" \
    --output-dir "$artifact" --artifact-storage persistent \
    --offset 0 --limit 128 --workers 4 --warmup-pages 8 --warmup-repeats 1 \
    --layout-threshold 0.5 --layout-execution eager --layout-dtype float32 \
    --layout-reading-order-dtype float32 --layout-weight-format native \
    --layout-depthwise-rewrite native --layout-batch-size 2 \
    --dtype float16 --cross-cache-length 1320 \
    --recognition-cache-dir "$COMPILE_CACHE" --vision-full-batches \
    --vision-focal-depthwise-rewrite constant_grouped_all \
    --vision-weight-format torchair_internal \
    --recognition-input-contract compact_uint8_hwc \
    --recognition-preprocess-threads 8 --vision-page-lookahead 4 \
    --no-retain-shared-images --progress-every-pages 16 \
    --progress-heartbeat-s 15
}

run_replay() {
  local artifact="$1" output="$2" trace_path="$3"
  local command=(
    taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY"
    --artifact-dir "$artifact" --model-path "$MODEL" --device npu:0
    --dtype float16 --batch-size 128 --self-cache-length 2048
    --cross-cache-length 1320 --max-length 2048
    --decode-warmup-passes 2 --decode-admission-prefetch-depth 0
    --compile-cache-dir "$COMPILE_CACHE" --progress-every 100
    --output "$output"
  )
  if [[ -n "$trace_path" ]]; then
    command+=(--step-trace-jsonl "$trace_path")
  fi
  "${command[@]}"
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  local artifact
  if [[ -n "$ARTIFACT_DIR" ]]; then
    artifact="$(readlink -f "$ARTIFACT_DIR")"
  else
    artifact="$run_root/prefill_artifact_first128"
    run_prefill "$artifact" | tee "$run_root/prefill.log"
  fi
  test -f "$artifact/summary.json"
  test -f "$artifact/crops.jsonl"
  test -f "$artifact/cross_kv.bin"
  run_replay "$artifact" "$run_root/clean.json" "" | tee "$run_root/clean.log"
  run_replay "$artifact" "$run_root/trace.json" "$run_root/decode_steps.jsonl" \
    | tee "$run_root/trace.log"
  "$PYTHON_BIN" "$REPORT" --clean "$run_root/clean.json" \
    --trace "$run_root/trace.json" --reference "$REFERENCE" \
    --output "$run_root/comparison.json" \
    | tee "$run_root/final_report.txt"
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_production_decode_diagnostic_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" CPUSET="$CPUSET" \
    ARTIFACT_DIR="$ARTIFACT_DIR" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
