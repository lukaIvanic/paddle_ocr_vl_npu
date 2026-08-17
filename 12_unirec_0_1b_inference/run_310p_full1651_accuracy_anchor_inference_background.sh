#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
DECODE_CACHE_PROBE="$SCRIPT_DIR/probe_production_decode_cache_contract.py"
AUDIT="$SCRIPT_DIR/audit_unirec_decode_run_history.py"
REPORT="$SCRIPT_DIR/report_unirec_full_inference_baseline.py"

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
  : "${COMPILE_CACHE:?export the canonical production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?reuse the passed B128 decode-cache parent}"
  : "${CPUSET:=0-63}"
  : "${LAYOUT_CPU_THREADS:=1}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$(readlink -f "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE")"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) echo "310P_DEVICE_MUST_BE_0_TO_3" >&2; exit 1;; esac
  [[ "$LAYOUT_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -f "$RUNNER"
  test -f "$DECODE_CACHE_PROBE"
  test -f "$AUDIT"
  test -f "$REPORT"
  EXACT_DECODE_CACHE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$EXACT_DECODE_CACHE" -name compiled_module | wc -l)" -eq 1
  test "$(find "$EXACT_DECODE_CACHE" -name '*.om' | wc -l)" -eq 1
}

decode_om_inventory() {
  find "$EXACT_DECODE_CACHE" -type f -name '*.om' -exec sha256sum '{}' \; | sort
}

gate_decode_cache() {
  local gate_root="$RUN_ROOT/decode_cache_gate"
  mkdir -p "$gate_root"
  taskset -c "$CPUSET" "$PYTHON_BIN" "$DECODE_CACHE_PROBE" \
    --model-path "$MODEL" --compile-cache-dir "$COMPILE_CACHE" \
    --device npu:0 --batch-size 128 --self-cache-length 2048 \
    --cross-cache-length 1320 --passes 2 \
    --output "$gate_root/passed.json" | tee "$gate_root/run.log"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\ncpuset=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" "$CPUSET"
    printf 'layout_cpu_threads=%s\n' "$LAYOUT_CPU_THREADS"
    printf 'cann_home=%s\n' "${ASCEND_HOME_PATH:-unavailable}"
    if [[ -f "${ASCEND_HOME_PATH:-}/opp/version.info" ]]; then
      grep -E '^(Version|version_dir|timestamp)=' "$ASCEND_HOME_PATH/opp/version.info"
    fi
    "$PYTHON_BIN" -c 'import os,torch,torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} affinity={sorted(os.sched_getaffinity(0))}")'
    df -h /dev/shm
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  } >"$RUN_ROOT/preflight.log" 2>&1

  decode_om_inventory >"$RUN_ROOT/decode_om_before.sha256"
  gate_decode_cache

  local output="$RUN_ROOT/output"
  mkdir -p "$output"
  command=(
    taskset -c "$CPUSET" "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT" --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL" --layout-execution eager
    --layout-dtype float32 --layout-reading-order-dtype float32
    --layout-weight-format native --layout-depthwise-rewrite native
    --layout-threshold 0.5 --input "$IMAGES_DIR" --output-dir "$output"
    --device npu:0 --dtype float16 --offset 0 --limit 1651
    --workers 4 --warmup-pages 8 --layout-batch-size 2
    --layout-cpu-threads "$LAYOUT_CPU_THREADS" --vision-page-lookahead 4
    --vision-bucket-preset production_v1
    --vision-focal-depthwise-rewrite native --vision-weight-format native
    --recognition-preprocess-threads 8
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320 --self-cache-length 2048 --max-length 2048
    --decode-batch-size 128 --compile-cache-dir "$COMPILE_CACHE"
    --decode-warmup-passes 0 --decode-admission-prefetch-depth 0
    --progress-every-pages 1 --progress-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  echo "UNIREC_310P_FULL1651_BASELINE_BEGIN"
  local started_ns ended_ns
  started_ns="$(date +%s%N)"
  "${command[@]}"
  ended_ns="$(date +%s%N)"
  "$PYTHON_BIN" -c 'import sys; print(f"{(int(sys.argv[2])-int(sys.argv[1]))/1e9:.6f}")' \
    "$started_ns" "$ended_ns" >"$RUN_ROOT/inference_process_wall_s.txt"
  echo "UNIREC_310P_FULL1651_BASELINE_END"

  decode_om_inventory >"$RUN_ROOT/decode_om_after.sha256"
  diff -u "$RUN_ROOT/decode_om_before.sha256" "$RUN_ROOT/decode_om_after.sha256" \
    >"$RUN_ROOT/decode_om.diff"
  echo "DECODE_CACHE_OM_INVENTORY_UNCHANGED"

  "$PYTHON_BIN" "$AUDIT" --search-root "$RUN_ROOT" --min-pages 512 \
    --output "$RUN_ROOT/history.json" | tee "$RUN_ROOT/history.txt"
  "$PYTHON_BIN" "$REPORT" \
    --run-summary "$RUN_ROOT/output/run_summary.json" \
    --process-wall "$RUN_ROOT/inference_process_wall_s.txt" \
    --history "$RUN_ROOT/history.json" \
    --output "$RUN_ROOT/baseline_summary.json" \
    | tee "$RUN_ROOT/final_report.txt"
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_full1651_accuracy_anchor_inference_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" CPUSET="$CPUSET" \
    LAYOUT_CPU_THREADS="$LAYOUT_CPU_THREADS" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
