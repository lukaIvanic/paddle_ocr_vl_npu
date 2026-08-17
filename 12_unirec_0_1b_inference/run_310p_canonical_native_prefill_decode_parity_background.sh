#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
COMPARE="$SCRIPT_DIR/compare_decode_completion_traces.py"

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
  : "${COMPILE_CACHE:?export the historical production compile-cache parent}"
  : "${CANONICAL_TRACE:?export the canonical 90.13 run recognition_trace.jsonl}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?reuse the passed B128 decode-cache parent}"
  : "${CPUSET:=0-63}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  CANONICAL_TRACE="$(readlink -f "$CANONICAL_TRACE")"
  UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$(readlink -f "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE")"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) echo "310P_DEVICE_MUST_BE_0_TO_3" >&2; exit 1;; esac
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -s "$CANONICAL_TRACE"
  test "$(wc -l <"$CANONICAL_TRACE")" -gt 30000
  EXACT_DECODE_CACHE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$EXACT_DECODE_CACHE" -name compiled_module | wc -l)" -eq 1
  test "$(find "$EXACT_DECODE_CACHE" -name '*.om' | wc -l)" -eq 1
}

om_inventory() {
  local root="$1"
  find "$root" -type f -name '*.om' -exec sha256sum '{}' \; | sort
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  om_inventory "$EXACT_DECODE_CACHE" >"$run_root/decode_om_before.sha256"
  find "$COMPILE_CACHE" -type f -name '*.om' -print | sort \
    >"$run_root/all_om_paths_before.txt"

  local artifact="$run_root/prefill_artifact_first128_canonical_native"
  echo "UNIREC_CANONICAL_NATIVE_PREFILL_BEGIN"
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
    --vision-focal-depthwise-rewrite native --vision-weight-format native \
    --recognition-input-contract compact_uint8_hwc \
    --recognition-preprocess-threads 8 --vision-page-lookahead 4 \
    --no-retain-shared-images --progress-every-pages 16 \
    --progress-heartbeat-s 15 | tee "$run_root/prefill.log"
  echo "UNIREC_CANONICAL_NATIVE_PREFILL_END"

  test -f "$artifact/summary.json"
  test -f "$artifact/crops.jsonl"
  test -f "$artifact/cross_kv.bin"
  "$PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; rows=[json.loads(x) for x in open(p) if x.strip()]; assert len(rows)==957, len(rows); print(f"CANONICAL_NATIVE_ARTIFACT_CROPS={len(rows)}")' "$artifact/crops.jsonl"

  echo "UNIREC_CANONICAL_NATIVE_DECODE_BEGIN"
  taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY" \
    --artifact-dir "$artifact" --model-path "$MODEL" --device npu:0 \
    --dtype float16 --batch-size 128 --self-cache-length 2048 \
    --cross-cache-length 1320 --max-length 2048 \
    --decode-warmup-passes 0 --decode-admission-prefetch-depth 0 \
    --compile-cache-dir "$COMPILE_CACHE" --progress-every 100 \
    --verify-crc --reference-trace "$CANONICAL_TRACE" \
    --completion-trace-jsonl "$run_root/completions.jsonl" \
    --output "$run_root/replay.json" | tee "$run_root/replay.log"
  echo "UNIREC_CANONICAL_NATIVE_DECODE_END"

  "$PYTHON_BIN" "$COMPARE" --candidate "$run_root/completions.jsonl" \
    --reference "$CANONICAL_TRACE" --output "$run_root/parity_report.json" \
    | tee "$run_root/final_report.txt"

  om_inventory "$EXACT_DECODE_CACHE" >"$run_root/decode_om_after.sha256"
  if diff -u "$run_root/decode_om_before.sha256" "$run_root/decode_om_after.sha256" \
      >"$run_root/decode_om.diff"; then
    echo "DECODE_CACHE_OM_INVENTORY_UNCHANGED"
  else
    echo "DECODE_CACHE_OM_INVENTORY_CHANGED" >&2
    cat "$run_root/decode_om.diff" >&2
    return 1
  fi
  find "$COMPILE_CACHE" -type f -name '*.om' -print | sort \
    >"$run_root/all_om_paths_after.txt"
  if diff -u "$run_root/all_om_paths_before.txt" "$run_root/all_om_paths_after.txt" \
      >"$run_root/all_om_paths.diff"; then
    echo "ALL_COMPILE_CACHE_OM_PATHS_UNCHANGED"
  else
    echo "ALL_COMPILE_CACHE_OM_PATHS_CHANGED"
    cat "$run_root/all_om_paths.diff"
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_canonical_native_prefill_decode_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    CANONICAL_TRACE="$CANONICAL_TRACE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
