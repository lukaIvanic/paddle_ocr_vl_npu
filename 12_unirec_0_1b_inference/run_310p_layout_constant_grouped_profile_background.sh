#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAYOUT_LAB="$SCRIPT_DIR/layout_detector_lab.py"
PROFILE_RUNNER="$SCRIPT_DIR/profile_prefill_graph_suite.py"
ANALYZER="$SCRIPT_DIR/compare_layout_constant_grouped_profile.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${INTERNAL_RUN_ROOT:?export the completed 310P internal-weight run root}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'LAYOUT_CONSTANT_GROUPED_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  INTERNAL_RUN_ROOT="$(readlink -f "$INTERNAL_RUN_ROOT")"
  LAYOUT_CACHE="$(readlink -m "$LAYOUT_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$LAYOUT_MODEL/model.safetensors"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$INTERNAL_RUN_ROOT"
  test -f "$INTERNAL_RUN_ROOT/exit_code.txt"
  test "$(tr -d '[:space:]' <"$INTERNAL_RUN_ROOT/exit_code.txt")" = 0
  test -f "$LAYOUT_LAB"
  test -f "$PROFILE_RUNNER"
  test -f "$ANALYZER"

  BASELINE_FORWARD="$INTERNAL_RUN_ROOT/forward_internal.json"
  BASELINE_PROFILE="$INTERNAL_RUN_ROOT/profile_internal/profile_suite_summary.json"
  test -f "$BASELINE_FORWARD"
  test -f "$BASELINE_PROFILE"

  local page_name
  page_name="jiaocaineedrop_jiaocai_needrop_en_620.jpg"
  LAYOUT_INPUT_IMAGE="$(find "$IMAGES_DIR" -type f -name "$page_name" -print -quit)"
  test -f "$LAYOUT_INPUT_IMAGE"
}

run_phase() {
  local phase="$1"
  local phase_log="$2"
  shift 2
  printf 'UNIREC_LAYOUT_CONSTANT_GROUPED_PHASE_BEGIN phase=%s command=' "$phase"
  printf '%q ' "$@"
  printf '\n'
  local started="$SECONDS"
  local status=0
  set +e
  "$@" > >(tee "$phase_log") 2>&1
  status="$?"
  set -e
  printf 'UNIREC_LAYOUT_CONSTANT_GROUPED_PHASE_END phase=%s status=%s wall_s=%s\n' \
    "$phase" "$status" "$((SECONDS - started))"
  return "$status"
}

run_forward() {
  local output="$RUN_ROOT/forward_constant_grouped.json"
  local command=(
    "$PYTHON_BIN" "$LAYOUT_LAB"
    --contract custom
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$LAYOUT_MODEL"
    --input "$LAYOUT_INPUT_IMAGE"
    --output "$output"
    --device npu:0
    --execution torchair
    --compile-cache-dir "$LAYOUT_CACHE"
    --dtype float16
    --reading-order-dtype float32
    --threshold 0.5
    --weight-format torchair_internal
    --depthwise-rewrite constant_grouped
    --input-color-order rgb
    --limit 1
    --warmup-pages 1
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/forward_command.sh"
  printf '\n' >>"$RUN_ROOT/forward_command.sh"
  run_phase forward "$RUN_ROOT/forward.log" "${command[@]}"
}

run_profile() {
  local output="$RUN_ROOT/profile_constant_grouped"
  local command=(
    "$PYTHON_BIN" "$PROFILE_RUNNER"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-input-image "$LAYOUT_INPUT_IMAGE"
    --layout-cache-dir "$LAYOUT_CACHE"
    --recognition-cache-dir "$LAYOUT_CACHE/unused_recognition"
    --output-dir "$output"
    --device npu:0
    --lane layout
    --layout-execution torchair
    --layout-dtype float16
    --layout-reading-order-dtype float32
    --layout-depthwise-rewrite constant_grouped
    --layout-weight-format torchair_internal
    --warmup 2
    --control-repeats 20
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 120
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/profile_command.sh"
  printf '\n' >>"$RUN_ROOT/profile_command.sh"
  run_phase profile "$RUN_ROOT/profile.log" "${command[@]}"
}

summarize() {
  local command=(
    "$PYTHON_BIN" "$ANALYZER"
    --baseline-forward "$BASELINE_FORWARD"
    --baseline-profile "$BASELINE_PROFILE"
    --candidate-forward "$RUN_ROOT/forward_constant_grouped.json"
    --candidate-profile "$RUN_ROOT/profile_constant_grouped/profile_suite_summary.json"
    --output "$RUN_ROOT/comparison_summary.json"
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/analyze_command.sh"
  printf '\n' >>"$RUN_ROOT/analyze_command.sh"
  run_phase summarize "$RUN_ROOT/report.log" "${command[@]}"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1

  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'internal_run_root=%s\nlayout_cache=%s\n' \
      "$INTERNAL_RUN_ROOT" "$LAYOUT_CACHE"
    printf 'layout_input_image=%s\n' "$LAYOUT_INPUT_IMAGE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  mkdir -p "$LAYOUT_CACHE/unused_recognition"
  run_forward
  run_profile
  summarize
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
  printf 'UNIREC_310P_LAYOUT_CONSTANT_GROUPED_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${INTERNAL_RUN_ROOT:?export the completed 310P internal-weight run root}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_constant_grouped_${commit_short}_${timestamp}}"
  LAYOUT_CACHE="${LAYOUT_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_layout_constant_grouped_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  LAYOUT_CACHE="$(realpath -m "$LAYOUT_CACHE")"
  test ! -e "$RUN_ROOT"
  test ! -e "$LAYOUT_CACHE"
  mkdir -p "$RUN_ROOT" "$LAYOUT_CACHE"
  resolve_inputs

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    INTERNAL_RUN_ROOT="$INTERNAL_RUN_ROOT" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_LAYOUT_CONSTANT_GROUPED_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
  printf 'EXIT_CODE_FILE=%s\n' "$RUN_ROOT/exit_code.txt"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_entry "$2"
    ;;
  "")
    launch_main
    ;;
  *)
    printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2
    exit 2
    ;;
esac
