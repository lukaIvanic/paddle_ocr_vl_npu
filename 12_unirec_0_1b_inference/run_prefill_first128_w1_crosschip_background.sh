#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_prefill_export.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the passed UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export PP-DocLayoutV2_safetensors}"
  : "${OPENOCR_ROOT:?export the passed OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench images directory}"
  : "${LAYOUT_CACHE:?export the warmed optimized-layout cache parent}"
  : "${RECOGNITION_CACHE:?export the warmed all-45 grouped-FZ cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_FIRST128_W1_REQUIRES_ONE_NPU=%s\n' \
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
  RECOGNITION_CACHE="$(readlink -f "$RECOGNITION_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -d "$RECOGNITION_CACHE"
  test -f "$RUNNER"
}

run_first128() {
  local output="$RUN_ROOT/output"
  mkdir -p "$output"
  local command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output-dir "$output"
    --artifact-storage discard
    --offset 0
    --limit 128
    --workers 1
    --warmup-pages 128
    --warmup-repeats 1
    --layout-execution torchair
    --layout-dtype float16
    --layout-batch-size 1
    --layout-depthwise-rewrite group16
    --layout-weight-format torchair_internal
    --layout-preformat-frozen-bn-buffers
    --layout-cache-dir "$LAYOUT_CACHE"
    --dtype float16
    --cross-cache-length 512
    --recognition-cache-dir "$RECOGNITION_CACHE"
    --vision-full-batches
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
    --recognition-input-contract compact_uint8_hwc
    --recognition-preprocess-threads 1
    --vision-page-lookahead 4
    --no-retain-shared-images
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/producer.log"
  test "${PIPESTATUS[0]}" = 0
}

report_result() {
  SUMMARY="$RUN_ROOT/output/summary.json" "$PYTHON_BIN" - <<'PY' \
    | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

p = json.loads(Path(os.environ["SUMMARY"]).read_text())
assert p["status"] == "ok"
assert p["validation"]["passed"] is True
assert (p["offset"], p["limit"], p["workers"]) == (0, 128, 1)
assert p["artifact_storage"] == "discard"
assert p["artifact"]["page_count"] == 128
w = p["worker_summary"]
s = w["stage_s"]
v = w["vision_batching"]
print(
    "UNIREC_PREFILL_FIRST128_W1: PASS "
    f"wall={p['producer_wall_s']:.6f}s "
    f"pages_s={p['throughput']['pages_per_s']:.6f} "
    f"crops_s={p['throughput']['crops_per_s']:.6f} "
    f"tokens_s={p['throughput']['real_source_tokens_per_s']:.6f} "
    f"pages={p['artifact']['page_count']} "
    f"crops={p['artifact']['crop_count']} "
    f"rejected={p['artifact']['rejected_crop_count']} "
    f"tokens={p['artifact']['real_source_tokens']} "
    f"worker_busy={w['worker_busy_s'][0]:.6f}s"
)
print(
    "UNIREC_PREFILL_FIRST128_W1_STAGES: "
    f"file_read={s['worker_file_read_sum_s']:.6f}s "
    f"rgb_decode={s['worker_direct_rgb_decode_sum_s']:.6f}s "
    f"rgb_to_bgr={s['worker_rgb_to_bgr_sum_s']:.6f}s "
    f"layout={s['worker_detector_call_sum_s']:.6f}s "
    f"crop_build={s['worker_recognition_crop_build_sum_s']:.6f}s "
    f"input_prepare={s['worker_recognition_input_prepare_sum_s']:.6f}s "
    f"recognition_prefill={s['worker_recognition_prefill_sum_s']:.6f}s "
    f"cache_d2h={s['worker_recognition_prefill_cache_d2h_sum_s']:.6f}s "
    f"shared_pack={s['worker_shared_pack_sum_s']:.6f}s "
    f"ipc_delivery={w['ipc_delivery_sum_s']:.6f}s"
)
print(
    "UNIREC_PREFILL_FIRST128_W1_VISION: "
    f"calls={v['bucket_calls']} "
    f"real_rows={v['compiled_real_rows']} "
    f"physical_rows={v['compiled_physical_rows']} "
    f"slot_efficiency={v['compiled_slot_efficiency']:.9f} "
    f"fallback={v['fallback_rows']} "
    f"peak_hbm_bytes={v['max_npu_peak_memory_bytes']}"
)
print(
    "UNIREC_PREFILL_FIRST128_W1_SETUP: "
    f"setup={p['setup_s']:.6f}s "
    f"warmup={p['warmup']['wall_s']:.6f}s "
    f"warmup_pages_s={p['warmup']['pages_per_s']:.6f}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'model=%s\nlayout_model=%s\nlayout_cache=%s\nrecognition_cache=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$LAYOUT_CACHE" "$RECOGNITION_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_first128
  report_result
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
  printf 'UNIREC_PREFILL_FIRST128_W1_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/prefill_first128_w1_crosschip_${commit_short}_${timestamp}}"
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
    RECOGNITION_CACHE="$RECOGNITION_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_PREFILL_FIRST128_W1_STARTED pid=%s physical=%s\n' \
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
