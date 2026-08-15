#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"
REPORTER="$SCRIPT_DIR/report_prefill_trace.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec inference Python}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the warmed production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  UNIREC_PREFILL_MODE="${UNIREC_PREFILL_MODE:-trace}"
  case "$UNIREC_PREFILL_MODE" in
    trace|clean) ;;
    *)
      printf 'INVALID_UNIREC_PREFILL_MODE=%s expected=trace_or_clean\n' \
        "$UNIREC_PREFILL_MODE" >&2
      exit 1
      ;;
  esac

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_REP128_W1T1_REQUIRES_ONE_NPU=%s\n' \
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
  mkdir -p "$COMPILE_CACHE"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -f "$RUNNER"
  test -f "$MATERIALIZER"
  test -f "$MANIFEST"
  test -f "$REPORTER"
}

run_trace() {
  local representative_input="$RUN_ROOT/representative_128_v1_images"
  local output="$RUN_ROOT/output"
  "$PYTHON_BIN" "$MATERIALIZER" \
    --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$representative_input"
  mkdir -p "$output"
  local command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-execution eager
    --layout-dtype float32
    --layout-reading-order-dtype float32
    --layout-weight-format native
    --layout-depthwise-rewrite constant_grouped
    --layout-threshold 0.5
    --input "$representative_input"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers 1
    --warmup-pages 8
    --layout-batch-size 2
    --vision-page-lookahead 4
    --vision-focal-depthwise-rewrite native
    --vision-weight-format native
    --recognition-preprocess-threads 1
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --decode-batch-size 128
    --compile-cache-dir "$COMPILE_CACHE"
    --stop-after-prefill
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  if [[ "$UNIREC_PREFILL_MODE" == trace ]]; then
    command+=(--prefill-trace)
  fi
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  "${command[@]}"
}

report_result() {
  if [[ "$UNIREC_PREFILL_MODE" == clean ]]; then
    RUN_SUMMARY="$RUN_ROOT/output/run_summary.json" \
      "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["RUN_SUMMARY"]).read_text())
assert run["status"] == "ok"
assert run["execution"] == "production_two_phase_prefill_only"
assert run["prefill_trace_enabled"] is False
assert (run["page_count"], run["workers"]) == (128, 1)
assert run["recognition_preprocess_threads"] == 1
assert run["layout_batch_size"] == 2
assert run["cross_cache_length"] == 1320
assert run["self_cache_length"] == 2048
print(
    "UNIREC_REP128_W1T1_PREFILL_CLEAN PASS "
    f"wall_s={run['timing_s']['prefill_phase']:.6f} "
    f"pages_s={run['throughput']['prefill_pages_per_s']:.6f} "
    f"crops={run['retained_bank']['crop_count']} "
    f"rejected={run['retained_bank']['rejected_crop_count']}"
)
PY
    return
  fi
  RUN_SUMMARY="$RUN_ROOT/output/run_summary.json" \
    TRACE_SUMMARY="$RUN_ROOT/output/prefill_distributions.json" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["RUN_SUMMARY"]).read_text())
trace = json.loads(Path(os.environ["TRACE_SUMMARY"]).read_text())
assert run["status"] == "ok"
assert run["execution"] == "production_two_phase_prefill_only"
assert (run["page_count"], run["workers"]) == (128, 1)
assert run["recognition_preprocess_threads"] == 1
assert run["layout_batch_size"] == 2
assert run["cross_cache_length"] == 1320
assert run["self_cache_length"] == 2048
assert trace["schema"] == "unirec_production_prefill_trace_v1"
assert trace["page_count"] == 128
assert trace["event_counts"]["recognition_crop_preprocess"] > 0
assert trace["event_counts"]["vision_bucket_call"] > 0
assert trace["event_counts"]["text_prefill_pack"] > 0
print(
    "UNIREC_REP128_W1T1_PREFILL_TRACE PASS "
    f"wall_s={run['timing_s']['prefill_phase']:.6f} "
    f"pages_s={run['throughput']['prefill_pages_per_s']:.6f} "
    f"crops={run['retained_bank']['crop_count']} "
    f"rejected={run['retained_bank']['rejected_crop_count']} "
    f"events={trace['event_count']}"
)
PY
  "$PYTHON_BIN" "$REPORTER" \
    "$RUN_ROOT/output/prefill_distributions.json" \
    --top-shapes 16
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'model=%s\nlayout_model=%s\ncompile_cache=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$COMPILE_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_trace
  report_result | tee "$RUN_ROOT/report.log"
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
  printf 'UNIREC_REP128_W1T1_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_${UNIREC_PREFILL_MODE}_${commit_short}_${timestamp}}"
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
    COMPILE_CACHE="$COMPILE_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PREFILL_MODE="$UNIREC_PREFILL_MODE" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$pid"
}

if [[ "${1:-}" == "--worker" ]]; then
  test "$#" = 2
  worker_entry "$2"
else
  test "$#" = 0
  launch_main
fi
