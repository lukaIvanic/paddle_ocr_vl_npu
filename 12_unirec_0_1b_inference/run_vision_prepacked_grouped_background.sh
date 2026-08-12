#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROFILER="$SCRIPT_DIR/profile_prefill_graph_suite.py"
ANALYZER="$SCRIPT_DIR/analyze_vision_depthwise_matrix.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for this NPU runtime}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${NATIVE_PROFILE:?export NATIVE_PROFILE for the completed native graph suite}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical NPU first}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'VISION_PREPACKED_GROUPED_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  NATIVE_PROFILE="$(readlink -f "$NATIVE_PROFILE")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$NATIVE_PROFILE"
  test -f "$PROFILER"
  test -f "$ANALYZER"
}

worker_main() {
  RUN_ROOT="$1"
  CACHE_ROOT="$2"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\nmodel=%s\nnative=%s\ncache=%s\n' \
      "$PYTHON_BIN" "$MODEL" "$NATIVE_PROFILE" "$CACHE_ROOT"
  } >"$RUN_ROOT/preflight.log"

  command=(
    "$PYTHON_BIN" "$PROFILER"
    --model-path "$MODEL"
    --layout-model "$MODEL"
    --layout-cache-dir "$CACHE_ROOT"
    --recognition-cache-dir "$CACHE_ROOT"
    --output-dir "$RUN_ROOT/output"
    --device npu:0
    --lane vision
    --vision-bucket 960x64_b16
    --vision-depthwise-rewrite constant_grouped
    --vision-weight-format native
    --warmup 3
    --control-repeats 50
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 200
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"

  status=0
  SECONDS=0
  printf 'UNIREC_VISION_PREPACKED_GROUPED_PHASE_BEGIN\n'
  set +e
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/profile.log"
  status="${PIPESTATUS[0]}"
  set -e
  if test "$status" -eq 0; then
    test -f "$RUN_ROOT/output/profile_suite_summary.json"
    "$PYTHON_BIN" "$ANALYZER" \
      --native "$NATIVE_PROFILE" \
      --variant "constant_grouped=$RUN_ROOT/output/profile_suite_summary.json" \
      --output "$RUN_ROOT/comparison_summary.json" \
      2>&1 | tee "$RUN_ROOT/analysis.log" || status="$?"
  fi
  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$SECONDS" >"$RUN_ROOT/process_wall_s.txt"
  printf 'UNIREC_VISION_PREPACKED_GROUPED_WORKER_END status=%s wall_s=%s run_log=%s\n' \
    "$status" "$SECONDS" "$RUN_ROOT/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/vision_prepacked_grouped_${commit_short}_${timestamp}}"
  CACHE_ROOT="${CACHE_ROOT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_prepacked_grouped_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  CACHE_ROOT="$(realpath -m "$CACHE_ROOT")"
  test ! -e "$RUN_ROOT"
  test ! -e "$CACHE_ROOT"
  mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    NATIVE_PROFILE="$NATIVE_PROFILE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" "$CACHE_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_VISION_PREPACKED_GROUPED_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 3
    worker_main "$2" "$3"
    ;;
  "") launch_main ;;
  *) printf 'usage: %s [--worker RUN_ROOT CACHE_ROOT]\n' "$0" >&2; exit 2 ;;
esac
