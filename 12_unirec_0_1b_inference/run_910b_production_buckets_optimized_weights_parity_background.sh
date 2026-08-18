#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
COMPARE="$SCRIPT_DIR/compare_decode_completion_traces.py"
REFERENCE="$SCRIPT_DIR/references/unirec_910b_decode_diagnostic_first128_039a633/recognition_token_digests.jsonl"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 910B UniRec venv executable}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${COMPILE_CACHE:?export the existing production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup first}"
  : "${CPUSET:=0-127}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  case ",$ASCEND_RT_VISIBLE_DEVICES," in *,5,*|*,6,*) exit 2;; esac
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -s "$REFERENCE"
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  local artifact="$run_root/prefill_production_buckets_optimized_weights"
  env UNIREC_VISION_BUCKET_PRESET=production_v1 \
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
    --progress-heartbeat-s 15 | tee "$run_root/prefill.log"

  taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY" \
    --artifact-dir "$artifact" --model-path "$MODEL" --device npu:0 \
    --dtype float16 --batch-size 128 --self-cache-length 2048 \
    --cross-cache-length 1320 --max-length 2048 \
    --decode-warmup-passes 2 --decode-admission-prefetch-depth 0 \
    --compile-cache-dir "$COMPILE_CACHE" --progress-every 100 --verify-crc \
    --reference-trace "$REFERENCE" \
    --completion-trace-jsonl "$run_root/completions.jsonl" \
    --output "$run_root/replay.json" | tee "$run_root/replay.log"
  "$PYTHON_BIN" "$COMPARE" --candidate "$run_root/completions.jsonl" \
    --reference "$REFERENCE" --output "$run_root/parity.json" \
    | tee "$run_root/final_report.txt"
  "$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); s=json.load(open(sys.argv[3]))
prefill_s=s["producer_wall_s"]; prefill_rate=s["throughput"]["pages_per_s"]
decode=r["decode"]; exact=p["token_exact_count"]; compared=p["compared_count"]
caps=p["long_output_counts"]["ge_2047"]
decode_s=decode["decode_s"]; raw=decode["raw_decode_tokens_per_s"]
effective=decode["effective_decode_tokens_per_s"]
print("UNIREC_910B_PRODUCTION_BUCKETS_OPTIMIZED_WEIGHTS: PASS "
      f"prefill_s={prefill_s:.3f} prefill_pages_s={prefill_rate:.3f} "
      f"decode_s={decode_s:.3f} raw_tok_s={raw:.1f} "
      f"effective_tok_s={effective:.1f} "
      f"token_exact={exact}/{compared} length_caps={caps}")
' "$run_root/parity.json" "$run_root/replay.json" "$artifact/summary.json" \
    | tee -a "$run_root/final_report.txt"
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  worker_main "$run_root"
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/910b_production_buckets_optimized_weights_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:-}" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
