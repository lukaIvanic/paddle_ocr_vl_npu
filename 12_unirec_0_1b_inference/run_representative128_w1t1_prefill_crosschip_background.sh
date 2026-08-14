#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LANE_RUNNER="$SCRIPT_DIR/run_representative128_w1t1_prefill_trace_background.sh"
COMPARATOR="$SCRIPT_DIR/compare_representative_prefill_traces.py"
REFERENCE_TRACE="$REPO/tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_trace_4cf871c_20260814T184415"
REFERENCE_CLEAN="$REPO/tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_clean_4cf871c_20260814T1850"

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec inference Python}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the warmed production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_REP128_W1T1_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  test -f "$LANE_RUNNER"
  test -f "$COMPARATOR"
  test -f "$REFERENCE_TRACE/output/prefill_distributions.json"
  test -f "$REFERENCE_TRACE/output/prefill_iterations.jsonl"
  test -f "$REFERENCE_TRACE/output/prefill_pages.jsonl"
  test -f "$REFERENCE_TRACE/output/run_summary.json"
  test -f "$REFERENCE_CLEAN/run_summary.json"
}

run_lane() {
  local mode="$1"
  local lane_root="$RUN_ROOT/$mode"
  mkdir -p "$lane_root"
  printf 'UNIREC_PREFILL_CROSSCHIP_LANE_BEGIN mode=%s root=%s\n' \
    "$mode" "$lane_root"
  set +e
  UNIREC_PREFILL_MODE="$mode" \
    bash "$LANE_RUNNER" --worker "$lane_root" \
      2>&1 | tee "$lane_root/run.log"
  local status="${PIPESTATUS[0]}"
  set -e
  printf 'UNIREC_PREFILL_CROSSCHIP_LANE_END mode=%s status=%s root=%s\n' \
    "$mode" "$status" "$lane_root"
  return "$status"
}

compare_lanes() {
  "$PYTHON_BIN" "$COMPARATOR" \
    --reference-trace-summary \
      "$REFERENCE_TRACE/output/prefill_distributions.json" \
    --reference-trace-events \
      "$REFERENCE_TRACE/output/prefill_iterations.jsonl" \
    --reference-trace-pages \
      "$REFERENCE_TRACE/output/prefill_pages.jsonl" \
    --reference-trace-run "$REFERENCE_TRACE/output/run_summary.json" \
    --reference-clean-run "$REFERENCE_CLEAN/run_summary.json" \
    --candidate-trace-summary \
      "$RUN_ROOT/trace/output/prefill_distributions.json" \
    --candidate-trace-events \
      "$RUN_ROOT/trace/output/prefill_iterations.jsonl" \
    --candidate-trace-pages \
      "$RUN_ROOT/trace/output/prefill_pages.jsonl" \
    --candidate-trace-run "$RUN_ROOT/trace/output/run_summary.json" \
    --candidate-clean-run "$RUN_ROOT/clean/output/run_summary.json" \
    --reference-chip 910B2 \
    --candidate-chip 310P \
    --output "$RUN_ROOT/crosschip_comparison.json" \
      | tee "$RUN_ROOT/crosschip_comparison.log"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  printf 'commit=%s\nphysical_device=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
    >"$RUN_ROOT/preflight.log"
  printf 'python=%s\nmodel=%s\nlayout_model=%s\ncompile_cache=%s\n' \
    "$PYTHON_BIN" "$MODEL" "$LAYOUT_MODEL" "$COMPILE_CACHE" \
    >>"$RUN_ROOT/preflight.log"
  "$PYTHON_BIN" -c \
    'import torch, torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)' \
    >>"$RUN_ROOT/preflight.log"
  npu-smi info >>"$RUN_ROOT/preflight.log" 2>&1

  local trace_status clean_status
  if run_lane trace; then
    trace_status=0
  else
    trace_status="$?"
  fi
  if run_lane clean; then
    clean_status=0
  else
    clean_status="$?"
  fi
  printf 'trace_status=%s\nclean_status=%s\n' \
    "$trace_status" "$clean_status" >"$RUN_ROOT/lane_status.txt"
  if [[ "$trace_status" != 0 || "$clean_status" != 0 ]]; then
    printf 'UNIREC_PREFILL_CROSSCHIP_LANES_FAIL trace=%s clean=%s\n' \
      "$trace_status" "$clean_status" >&2
    return 1
  fi
  compare_lanes
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
  printf 'UNIREC_PREFILL_CROSSCHIP_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_crosschip_${commit_short}_${timestamp}}"
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
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$pid" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == "--worker" ]]; then
  test "$#" = 2
  worker_entry "$2"
else
  test "$#" = 0
  launch_main
fi
