#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
GRAPH_SUITE="$SCRIPT_DIR/profile_prefill_graph_suite.py"
ANALYZER="$SCRIPT_DIR/analyze_vision_bucket_compiled_ab.py"
PREFILL_RUNNER="$SCRIPT_DIR/run_prefill_export.py"
FULL_REPORTER="$SCRIPT_DIR/report_310p_grouped_fz_full1651.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for the passed 310P environment}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${LAYOUT_MODEL:?export LAYOUT_MODEL for PP-DocLayoutV2}"
  : "${OPENOCR_ROOT:?export OPENOCR_ROOT for the passed OpenOCR checkout}"
  : "${IMAGES_DIR:?export IMAGES_DIR for OmniDocBench images}"
  : "${LAYOUT_CACHE:?export the passed optimized-layout cache parent}"
  : "${BASELINE_RECOGNITION_CACHE:?export the passed native five-graph cache}"
  : "${OPT_RECOGNITION_CACHE:?export a dedicated grouped-FZ cache path}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_310P_GROUPED_FZ_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"
  BASELINE_RECOGNITION_CACHE="$(readlink -f "$BASELINE_RECOGNITION_CACHE")"
  OPT_RECOGNITION_CACHE="$(realpath -m "$OPT_RECOGNITION_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -d "$BASELINE_RECOGNITION_CACHE"
  test -f "$GRAPH_SUITE"
  test -f "$ANALYZER"
  test -f "$PREFILL_RUNNER"
  test -f "$FULL_REPORTER"
}

run_bucket_lane() {
  local lane="$1"
  local cache="$2"
  shift 2
  local output="$RUN_ROOT/buckets/$lane"
  local command=(
    "$PYTHON_BIN" "$GRAPH_SUITE"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-cache-dir "$LAYOUT_CACHE"
    --recognition-cache-dir "$cache"
    --output-dir "$output"
    --device npu:0
    --lane vision
    --warmup 2
    --control-repeats 10
    --profile-steps 1
    --parser-topn 20
    --skip-profiler
    "$@"
  )
  mkdir -p "$RUN_ROOT/buckets"
  printf '%q ' "${command[@]}" >"$RUN_ROOT/buckets/${lane}_command.sh"
  printf '\n' >>"$RUN_ROOT/buckets/${lane}_command.sh"
  printf 'UNIREC_310P_GROUPED_FZ_PHASE_BEGIN lane=%s\n' "$lane"
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/buckets/${lane}.log"
  test "${PIPESTATUS[0]}" = 0
  printf 'UNIREC_310P_GROUPED_FZ_PHASE_END lane=%s\n' "$lane"
}

seed_text_prefill_cache() {
  local text_source_hash source destination
  text_source_hash="$(
    cd "$SCRIPT_DIR"
    "$PYTHON_BIN" -c \
      'import text_packed_prefill; print(text_packed_prefill._source_hash())'
  )"
  source="$BASELINE_RECOGNITION_CACHE/text_prefill_packed_b1_s1024_float16_src$text_source_hash"
  test -d "$source"
  test -n "$(find "$source" -type f -print -quit)"
  destination="$OPT_RECOGNITION_CACHE/$(basename "$source")"
  if test ! -e "$destination"; then
    cp -a "$source" "$destination"
  fi
  printf 'UNIREC_310P_GROUPED_FZ_TEXT_CACHE source=%s destination=%s\n' \
    "$source" "$destination"
}

run_full_optimized() {
  local output="$RUN_ROOT/full1651_optimized/output"
  mkdir -p "$output"
  local command=(
    "$PYTHON_BIN" "$PREFILL_RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output-dir "$output"
    --artifact-storage discard
    --offset 0
    --limit 1651
    --workers 8
    --warmup-pages 8
    --warmup-repeats 1
    --layout-execution torchair
    --layout-dtype float16
    --layout-batch-size 1
    --layout-depthwise-rewrite native
    --layout-weight-format torchair_internal
    --layout-preformat-frozen-bn-buffers
    --layout-cache-dir "$LAYOUT_CACHE"
    --dtype float16
    --cross-cache-length 512
    --recognition-cache-dir "$OPT_RECOGNITION_CACHE"
    --vision-full-batches
    --vision-focal-depthwise-rewrite constant_grouped
    --vision-weight-format native
    --recognition-input-contract compact_uint8_hwc
    --recognition-preprocess-threads 8
    --vision-page-lookahead 4
    --no-retain-shared-images
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/full1651_optimized/command.sh"
  printf '\n' >>"$RUN_ROOT/full1651_optimized/command.sh"
  printf 'UNIREC_310P_GROUPED_FZ_PHASE_BEGIN lane=full1651_optimized\n'
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/full1651_optimized/run.log"
  test "${PIPESTATUS[0]}" = 0
  printf 'UNIREC_310P_GROUPED_FZ_PHASE_END lane=full1651_optimized\n'
}

report_full_result() {
  "$PYTHON_BIN" "$FULL_REPORTER" \
    --summary "$RUN_ROOT/full1651_optimized/output/summary.json" \
    --historical-prefill-s "${HISTORICAL_PREFILL_S:-350}" \
    | tee "$RUN_ROOT/full1651_optimized/report.log"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  mkdir -p "$OPT_RECOGNITION_CACHE"
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\nmodel=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN" "$MODEL"
    printf 'layout_cache=%s\nbaseline_recognition_cache=%s\noptimized_recognition_cache=%s\n' \
      "$LAYOUT_CACHE" "$BASELINE_RECOGNITION_CACHE" "$OPT_RECOGNITION_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  run_bucket_lane native "$BASELINE_RECOGNITION_CACHE" \
    --vision-depthwise-rewrite native \
    --vision-weight-format native \
    --save-vision-outputs-dir "$RUN_ROOT/buckets/native_outputs"

  run_bucket_lane grouped_fz "$OPT_RECOGNITION_CACHE" \
    --vision-depthwise-rewrite constant_grouped \
    --vision-weight-format native \
    --allow-vision-parity-drift \
    --reference-vision-outputs-dir "$RUN_ROOT/buckets/native_outputs"

  "$PYTHON_BIN" "$ANALYZER" \
    --native "$RUN_ROOT/buckets/native/profile_suite_summary.json" \
    --optimized "$RUN_ROOT/buckets/grouped_fz/profile_suite_summary.json" \
    --output "$RUN_ROOT/buckets/comparison.json" \
    | tee "$RUN_ROOT/buckets/comparison.log"

  seed_text_prefill_cache
  run_full_optimized
  report_full_result
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1"
  local status=0
  local started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_GROUPED_FZ_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_grouped_fz_buckets_full1651_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    BASELINE_RECOGNITION_CACHE="$BASELINE_RECOGNITION_CACHE" \
    OPT_RECOGNITION_CACHE="$OPT_RECOGNITION_CACHE" \
    HISTORICAL_PREFILL_S="${HISTORICAL_PREFILL_S:-350}" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_GROUPED_FZ_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_entry "$2"
    ;;
  "") launch_main ;;
  *) printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
