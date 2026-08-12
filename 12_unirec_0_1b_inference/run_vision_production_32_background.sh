#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/vision_production_lab.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for the passed 310P environment}"
  : "${MODEL:?export MODEL for the passed UniRec model directory}"
  : "${OPENOCR_ROOT:?export OPENOCR_ROOT for the passed OpenOCR checkout}"
  : "${PAGE_MANIFEST:?export PAGE_MANIFEST for the full-run pages.jsonl}"
  : "${CROP_MANIFEST:?export CROP_MANIFEST for the matching crops.jsonl}"
  : "${VISION_CACHE:?export VISION_CACHE for the passed warm five-graph cache}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  PAGE_LIMIT="${PAGE_LIMIT:-32}"

  if ! [[ "$PAGE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'INVALID_PAGE_LIMIT=%s\n' "$PAGE_LIMIT" >&2
    exit 2
  fi

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n' >&2; exit 1 ;;
  esac

  # Preserve the venv interpreter leaf. Resolving bin/python to the base Python
  # bypasses pyvenv.cfg and drops venv-only packages such as kornia_rs.
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
  test -f "$LAB"
}

worker_main() {
  RUN_ROOT="$1"
  RUN_LOG="$RUN_ROOT/run.log"
  OUTPUT_DIR="$RUN_ROOT/output"
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
    printf 'page_limit=%s\n' "$PAGE_LIMIT"
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  find "$VISION_CACHE" -name '*.om' \
    -printf '%P\t%s\t%T@\n' 2>/dev/null \
    | sort >"$RUN_ROOT/om_before.tsv"

  command=(
    "$PYTHON_BIN"
    "$LAB"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --page-manifest "$PAGE_MANIFEST"
    --crop-manifest "$CROP_MANIFEST"
    --cache-dir "$VISION_CACHE"
    --output-dir "$OUTPUT_DIR"
    --page-offset 0
    --page-limit "$PAGE_LIMIT"
    --page-lookahead 4
    --warmup-replays 1
    --repeats 2
    --parity-samples-per-route 0
    --profile-scope none
    --diagnostic-graph-log
    --diagnostic-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"

  printf 'UNIREC_VISION_BACKGROUND_WORKER_BEGIN pid=%s run_log=%s\n' \
    "$$" "$RUN_LOG"
  set +e
  SECONDS=0
  "${command[@]}"
  status="$?"
  wall_s="$SECONDS"
  set -e

  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$wall_s" >"$RUN_ROOT/process_wall_s.txt"
  find "$VISION_CACHE" -name '*.om' \
    -printf '%P\t%s\t%T@\n' 2>/dev/null \
    | sort >"$RUN_ROOT/om_after.tsv"
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
  printf 'UNIREC_VISION_BACKGROUND_WORKER_END status=%s wall_s=%s run_log=%s\n' \
    "$status" "$wall_s" "$RUN_LOG"
  exit "$status"
}

launch_main() {
  resolve_inputs
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_vision_production_first${PAGE_LIMIT}_${commit_short}_${timestamp}}"
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
    PAGE_LIMIT="$PAGE_LIMIT" \
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

  printf 'UNIREC_VISION_BACKGROUND_STARTED pid=%s\n' "$worker_pid"
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
