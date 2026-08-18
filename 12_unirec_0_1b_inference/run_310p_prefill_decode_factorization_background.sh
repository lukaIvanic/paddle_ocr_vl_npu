#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
SELECT="$SCRIPT_DIR/select_unirec_decode_probe_ids.py"
COMPARE_KV="$SCRIPT_DIR/compare_unirec_prefill_artifacts.py"
COMPARE_TOKENS="$SCRIPT_DIR/compare_decode_completion_traces.py"
REPORT="$SCRIPT_DIR/report_unirec_prefill_decode_factorization.py"

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
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${COMPILE_CACHE:?export the existing production compile-cache parent}"
  : "${CANONICAL_ARTIFACT:?export the passed production_v1/native first128 artifact}"
  : "${OPTIMIZED_ARTIFACT:?export the failed K10/internal/grouped first128 artifact}"
  : "${OPTIMIZED_MISMATCH_REPORT:?export its decode parity_report.json}"
  : "${CANONICAL_TRACE:?export the canonical 90.13 recognition_trace.jsonl}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?export the passed B128 decode-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${CPUSET:=0-63}"
  : "${UNIREC_FACTORIZATION_ALLOWED_DEVICES:=0,1,2,3}"
  : "${UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH:=0}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  case ",$UNIREC_FACTORIZATION_ALLOWED_DEVICES," in
    *,$ASCEND_RT_VISIBLE_DEVICES,*) ;;
    *) exit 2 ;;
  esac
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  for variable in MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE \
    CANONICAL_ARTIFACT OPTIMIZED_ARTIFACT OPTIMIZED_MISMATCH_REPORT \
    CANONICAL_TRACE UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE; do
    printf -v "$variable" '%s' "$(readlink -f "${!variable}")"
  done
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  for artifact in "$CANONICAL_ARTIFACT" "$OPTIMIZED_ARTIFACT"; do
    test -s "$artifact/summary.json"
    test -s "$artifact/crops.jsonl"
    test -s "$artifact/cross_kv.bin"
  done
  test -s "$OPTIMIZED_MISMATCH_REPORT"
  test -s "$CANONICAL_TRACE"
  local exact_decode_cache
  exact_decode_cache="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$exact_decode_cache" -name compiled_module | wc -l)" -eq 1
  test "$(find "$exact_decode_cache" -name '*.om' | wc -l)" -eq 1
}

check_intermediate_vision_cache() {
  local key matches=0 directory found
  for key in 960x64_b16 512x256_b16 960x256_b4 512x512_b8 960x512_b4; do
    found=0
    while IFS= read -r directory; do
      if [[ -n "$(find "$directory" -name compiled_module -print -quit)" ]] \
        && [[ -n "$(find "$directory" -name '*.om' -print -quit)" ]]; then
        found=1
        break
      fi
    done < <(
      find "$COMPILE_CACHE" -type d \
        -name "vision_full_bucket_${key}_float16_*dwconstant_grouped_all*wtorchair_internal*"
    )
    if [[ "$found" -eq 1 ]]; then
      matches=$((matches + 1))
    else
      printf 'UNIREC_FACTORIZATION_CACHE_MISS bucket=%s\n' "$key" >&2
    fi
  done
  if [[ "$matches" -ne 5 ]]; then
    printf 'UNIREC_FACTORIZATION_STOP missing_intermediate_graphs=%s\n' "$((5 - matches))" >&2
    return 1
  fi
  printf 'UNIREC_FACTORIZATION_CACHE_PREFLIGHT: PASS graphs=%s\n' "$matches"
}

om_inventory() {
  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' | sort
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  check_intermediate_vision_cache
  om_inventory >"$run_root/om_before.txt"
  local intermediate="$run_root/prefill_production_buckets_optimized_weights"
  printf 'UNIREC_FACTORIZATION_PHASE_BEGIN phase=intermediate_prefill\n'
  env UNIREC_VISION_BUCKET_PRESET=production_v1 \
    taskset -c "$CPUSET" "$PYTHON_BIN" "$PREFILL" \
    --openocr-root "$OPENOCR_ROOT" --model-path "$MODEL" \
    --layout-model "$LAYOUT_MODEL" --input "$IMAGES_DIR" \
    --output-dir "$intermediate" --artifact-storage persistent \
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
    --progress-heartbeat-s 15 | tee "$run_root/intermediate_prefill.log"
  printf 'UNIREC_FACTORIZATION_PHASE_END phase=intermediate_prefill\n'

  post_prefill_main "$run_root"
}

post_prefill_main() {
  local run_root="$1"
  local intermediate="$run_root/prefill_production_buckets_optimized_weights"
  local -a selector_extra=() reporter_extra=()
  if [[ "$UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH" == 1 ]]; then
    selector_extra+=(--allow-empty-mismatches)
    reporter_extra+=(--allow-no-optimized-mismatch)
  fi
  test -s "$run_root/om_before.txt"
  test -s "$intermediate/summary.json"
  test -s "$intermediate/crops.jsonl"
  test -s "$intermediate/cross_kv.bin"

  "$PYTHON_BIN" "$SELECT" \
    --mismatch-report "$OPTIMIZED_MISMATCH_REPORT" \
    --reference-trace "$CANONICAL_TRACE" \
    --artifact-crops "$CANONICAL_ARTIFACT/crops.jsonl" \
    --cohort-size 128 --control-max-tokens 256 \
    "${selector_extra[@]}" \
    --output "$run_root/probe_request_ids.txt" \
    --summary "$run_root/probe_selection.json"

  "$PYTHON_BIN" "$COMPARE_KV" --reference "$CANONICAL_ARTIFACT" \
    --candidate "$intermediate" \
    --request-ids-file "$run_root/probe_request_ids.txt" \
    --output "$run_root/intermediate_cross_kv.json"
  "$PYTHON_BIN" "$COMPARE_KV" --reference "$CANONICAL_ARTIFACT" \
    --candidate "$OPTIMIZED_ARTIFACT" \
    --request-ids-file "$run_root/probe_request_ids.txt" \
    --output "$run_root/optimized_cross_kv.json"

  printf 'UNIREC_FACTORIZATION_PHASE_BEGIN phase=intermediate_decode\n'
  taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY" \
    --artifact-dir "$intermediate" --model-path "$MODEL" --device npu:0 \
    --dtype float16 --batch-size 128 --self-cache-length 2048 \
    --cross-cache-length 1320 --max-length 2048 \
    --request-ids-file "$run_root/probe_request_ids.txt" \
    --decode-warmup-passes 2 --decode-admission-prefetch-depth 0 \
    --compile-cache-dir "$COMPILE_CACHE" --progress-every 16 --verify-crc \
    --reference-trace "$CANONICAL_TRACE" \
    --completion-trace-jsonl "$run_root/intermediate_completions.jsonl" \
    --output "$run_root/intermediate_replay.json" \
    | tee "$run_root/intermediate_replay.log"
  printf 'UNIREC_FACTORIZATION_PHASE_END phase=intermediate_decode\n'

  "$PYTHON_BIN" "$COMPARE_TOKENS" \
    --candidate "$run_root/intermediate_completions.jsonl" \
    --reference "$CANONICAL_TRACE" \
    --output "$run_root/intermediate_parity.json"
  "$PYTHON_BIN" "$REPORT" \
    --canonical-summary "$CANONICAL_ARTIFACT/summary.json" \
    --intermediate-summary "$intermediate/summary.json" \
    --optimized-summary "$OPTIMIZED_ARTIFACT/summary.json" \
    --optimized-mismatch-report "$OPTIMIZED_MISMATCH_REPORT" \
    --intermediate-replay "$run_root/intermediate_replay.json" \
    --intermediate-parity "$run_root/intermediate_parity.json" \
    --intermediate-cross-kv "$run_root/intermediate_cross_kv.json" \
    --optimized-cross-kv "$run_root/optimized_cross_kv.json" \
    "${reporter_extra[@]}" \
    --output "$run_root/factorization_report.json" \
    | tee "$run_root/final_report.txt"

  om_inventory >"$run_root/om_after.txt"
  if diff -u "$run_root/om_before.txt" "$run_root/om_after.txt" \
      >"$run_root/om.diff"; then
    printf 'UNIREC_FACTORIZATION_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_FACTORIZATION_OM_INVENTORY_CHANGED\n' >&2
    cat "$run_root/om.diff" >&2
    return 1
  fi
}

resume_worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (
    set -e
    resolve_inputs
    post_prefill_main "$run_root"
  )
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/resume_exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/resume_process_wall_s.txt"
  exit "$status"
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (
    set -e
    worker_main "$run_root"
  )
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_prefill_decode_factorization_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    CANONICAL_ARTIFACT="$CANONICAL_ARTIFACT" \
    OPTIMIZED_ARTIFACT="$OPTIMIZED_ARTIFACT" \
    OPTIMIZED_MISMATCH_REPORT="$OPTIMIZED_MISMATCH_REPORT" \
    CANONICAL_TRACE="$CANONICAL_TRACE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_FACTORIZATION_ALLOWED_DEVICES="$UNIREC_FACTORIZATION_ALLOWED_DEVICES" \
    UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH="$UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

launch_resume_main() {
  resolve_inputs
  local run_root
  run_root="$(realpath -m "$1")"
  test -d "$run_root"
  test -s "$run_root/om_before.txt"
  test -s "$run_root/prefill_production_buckets_optimized_weights/summary.json"
  test ! -e "$run_root/resume_pid.txt"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    CANONICAL_ARTIFACT="$CANONICAL_ARTIFACT" \
    OPTIMIZED_ARTIFACT="$OPTIMIZED_ARTIFACT" \
    OPTIMIZED_MISMATCH_REPORT="$OPTIMIZED_MISMATCH_REPORT" \
    CANONICAL_TRACE="$CANONICAL_TRACE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_FACTORIZATION_ALLOWED_DEVICES="$UNIREC_FACTORIZATION_ALLOWED_DEVICES" \
    UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH="$UNIREC_FACTORIZATION_ALLOW_NO_OPTIMIZED_MISMATCH" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" resume-worker "$run_root" >"$run_root/resume.log" 2>&1 &
  printf '%s\n' "$!" >"$run_root/resume_pid.txt"
  printf 'RUN_ROOT=%s\nRESUME_LOG=%s\nPID=%s\n' \
    "$run_root" "$run_root/resume.log" "$(cat "$run_root/resume_pid.txt")"
}

case "${1:-}" in
  worker) worker_entry "$2" ;;
  resume-worker) resume_worker_entry "$2" ;;
  resume) launch_resume_main "$2" ;;
  *) launch_main ;;
esac
