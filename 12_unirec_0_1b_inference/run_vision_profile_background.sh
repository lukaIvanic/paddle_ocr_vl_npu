#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PRODUCTION_LAB="$SCRIPT_DIR/vision_production_lab.py"
GRAPH_SUITE="$SCRIPT_DIR/profile_prefill_graph_suite.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for this NPU runtime}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${OPENOCR_ROOT:?export OPENOCR_ROOT for the matching OpenOCR checkout}"
  : "${PAGE_MANIFEST:?export PAGE_MANIFEST for the full-run pages.jsonl}"
  : "${CROP_MANIFEST:?export CROP_MANIFEST for the matching crops.jsonl}"
  : "${VISION_CACHE:?export VISION_CACHE for the warm five-graph cache}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'VISION_PROFILE_REQUIRES_ONE_VISIBLE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  PAGE_MANIFEST="$(readlink -f "$PAGE_MANIFEST")"
  CROP_MANIFEST="$(readlink -f "$CROP_MANIFEST")"
  VISION_CACHE="$(readlink -f "$VISION_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -f "$PAGE_MANIFEST"
  test -f "$CROP_MANIFEST"
  test -d "$VISION_CACHE"
  test -f "$PRODUCTION_LAB"
  test -f "$GRAPH_SUITE"
}

run_phase() {
  phase="$1"
  phase_log="$2"
  shift 2
  printf 'UNIREC_VISION_PROFILE_PHASE_BEGIN phase=%s command=' "$phase"
  printf '%q ' "$@"
  printf '\n'
  set +e
  SECONDS=0
  "$@" > >(tee "$phase_log") 2>&1
  phase_status="$?"
  phase_wall_s="$SECONDS"
  set -e
  printf 'UNIREC_VISION_PROFILE_PHASE_END phase=%s status=%s wall_s=%s\n' \
    "$phase" "$phase_status" "$phase_wall_s"
  if test "$phase_status" -ne 0; then
    return "$phase_status"
  fi
}

worker_main() {
  RUN_ROOT="$1"
  RUN_LOG="$RUN_ROOT/run.log"
  resolve_inputs

  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'model=%s\n' "$MODEL"
    printf 'openocr=%s\n' "$OPENOCR_ROOT"
    printf 'page_manifest=%s\n' "$PAGE_MANIFEST"
    printf 'crop_manifest=%s\n' "$CROP_MANIFEST"
    printf 'vision_cache=%s\n' "$VISION_CACHE"
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  find "$VISION_CACHE" -name '*.om' -printf '%P\t%s\t%T@\n' 2>/dev/null \
    | sort >"$RUN_ROOT/om_before.tsv"

  production_command=(
    "$PYTHON_BIN"
    "$PRODUCTION_LAB"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --page-manifest "$PAGE_MANIFEST"
    --crop-manifest "$CROP_MANIFEST"
    --cache-dir "$VISION_CACHE"
    --output-dir "$RUN_ROOT/production"
    --page-offset 0
    --page-limit 32
    --page-lookahead 4
    --warmup-replays 1
    --repeats 5
    --parity-samples-per-route 0
    --profile-scope workload
    --profile-metric pipe
    --parser-topn 100
    --diagnostic-heartbeat-s 15
  )
  printf '%q ' "${production_command[@]}" >"$RUN_ROOT/production_command.sh"
  printf '\n' >>"$RUN_ROOT/production_command.sh"

  graph_command=(
    "$PYTHON_BIN"
    "$GRAPH_SUITE"
    --model-path "$MODEL"
    --layout-model "$MODEL"
    --layout-cache-dir "$VISION_CACHE"
    --recognition-cache-dir "$VISION_CACHE"
    --output-dir "$RUN_ROOT/graph_suite"
    --device npu:0
    --lane vision
    --warmup 2
    --control-repeats 10
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 100
  )
  printf '%q ' "${graph_command[@]}" >"$RUN_ROOT/graph_command.sh"
  printf '\n' >>"$RUN_ROOT/graph_command.sh"

  status=0
  worker_started_epoch="$(date +%s)"
  run_phase production "$RUN_ROOT/production.log" \
    "${production_command[@]}" || status="$?"
  if test "$status" -eq 0; then
    run_phase graph_suite "$RUN_ROOT/graph_suite.log" \
      "${graph_command[@]}" || status="$?"
  fi
  wall_s="$(( $(date +%s) - worker_started_epoch ))"

  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$wall_s" >"$RUN_ROOT/process_wall_s.txt"
  find "$VISION_CACHE" -name '*.om' -printf '%P\t%s\t%T@\n' 2>/dev/null \
    | sort >"$RUN_ROOT/om_after.tsv"
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
  printf 'UNIREC_VISION_PROFILE_WORKER_END status=%s wall_s=%s run_log=%s\n' \
    "$status" "$wall_s" "$RUN_LOG"
  exit "$status"
}

launch_main() {
  resolve_inputs
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/vision_profile_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  RUN_LOG="$RUN_ROOT/run.log"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    PAGE_MANIFEST="$PAGE_MANIFEST" \
    CROP_MANIFEST="$CROP_MANIFEST" \
    VISION_CACHE="$VISION_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_LOG" 2>&1 < /dev/null &
  worker_pid="$!"
  printf '%s\n' "$worker_pid" >"$RUN_ROOT/pid.txt"

  sleep 1
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    printf 'BACKGROUND_START_FAILED\n' >&2
    tail -n 40 "$RUN_LOG" >&2 || true
    exit 1
  fi

  printf 'UNIREC_VISION_PROFILE_STARTED pid=%s\n' "$worker_pid"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  printf 'RUN_LOG=%s\n' "$RUN_LOG"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_LOG"
  printf 'EXIT_CODE_FILE=%s\n' "$RUN_ROOT/exit_code.txt"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_main "$2"
    ;;
  "")
    launch_main
    ;;
  *)
    printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2
    exit 2
    ;;
esac
