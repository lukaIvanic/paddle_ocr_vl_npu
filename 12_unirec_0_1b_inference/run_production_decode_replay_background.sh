#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then
    command -v "$value"
    return
  fi
  local directory basename
  directory="$(dirname "$value")"
  basename="$(basename "$value")"
  printf '%s/%s\n' "$(cd "$directory" && pwd -P)" "$basename"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec Python}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${ARTIFACT_DIR:?export a persistent unirec_cross_kv_v1 artifact}"
  : "${COMPILE_CACHE:?export the production compile-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select exactly one physical NPU}"
  : "${BATCH_SIZE:=128}"
  : "${SELF_CACHE_LENGTH:=2048}"
  : "${CROSS_CACHE_LENGTH:=1320}"
  : "${MAX_LENGTH:=$SELF_CACHE_LENGTH}"
  : "${OFFSET_CROPS:=0}"
  : "${LIMIT_CROPS:=0}"
  : "${OVER_CAPACITY:=error}"
  : "${WARMUP_PASSES:=2}"
  : "${ADMISSION_PREFETCH_DEPTH:=0}"
  : "${PREFAULT_ARTIFACT:=1}"
  : "${REFERENCE_TRACE:=}"
  : "${REFERENCE_RUN_SUMMARY:=}"
  : "${STEP_TRACE:=0}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'REQUIRES_EXACTLY_ONE_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  ARTIFACT_DIR="$(readlink -f "$ARTIFACT_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$ARTIFACT_DIR/summary.json"
  test -f "$ARTIFACT_DIR/crops.jsonl"
  test -f "$ARTIFACT_DIR/cross_kv.bin"
  test -d "$COMPILE_CACHE"
  case "$PREFAULT_ARTIFACT" in 0|1) ;; *) exit 2 ;; esac
  case "$STEP_TRACE" in 0|1) ;; *) exit 2 ;; esac
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  local command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/production_decode_replay.py"
    --artifact-dir "$ARTIFACT_DIR"
    --model-path "$MODEL"
    --device npu:0 --dtype float16
    --batch-size "$BATCH_SIZE"
    --self-cache-length "$SELF_CACHE_LENGTH"
    --cross-cache-length "$CROSS_CACHE_LENGTH"
    --max-length "$MAX_LENGTH"
    --offset-crops "$OFFSET_CROPS"
    --limit-crops "$LIMIT_CROPS"
    --over-capacity "$OVER_CAPACITY"
    --decode-warmup-passes "$WARMUP_PASSES"
    --decode-admission-prefetch-depth "$ADMISSION_PREFETCH_DEPTH"
    --compile-cache-dir "$COMPILE_CACHE"
    --output "$run_root/result.json"
  )
  if [[ "$PREFAULT_ARTIFACT" == 0 ]]; then
    command+=(--no-prefault-artifact)
  fi
  if [[ -n "$REFERENCE_TRACE" ]]; then
    command+=(--reference-trace "$REFERENCE_TRACE")
  fi
  if [[ -n "$REFERENCE_RUN_SUMMARY" ]]; then
    command+=(--reference-run-summary "$REFERENCE_RUN_SUMMARY")
  fi
  if [[ "$STEP_TRACE" == 1 ]]; then
    command+=(--step-trace-jsonl "$run_root/decode_steps.jsonl")
  fi
  printf '%q ' "${command[@]}" >"$run_root/command.sh"
  printf '\n' >>"$run_root/command.sh"
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    df -h /dev/shm
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  } >"$run_root/preflight.log"
  "${command[@]}"
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/production_decode_replay_b${BATCH_SIZE}_self${SELF_CACHE_LENGTH}_cross${CROSS_CACHE_LENGTH}_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" ARTIFACT_DIR="$ARTIFACT_DIR" \
    COMPILE_CACHE="$COMPILE_CACHE" BATCH_SIZE="$BATCH_SIZE" \
    SELF_CACHE_LENGTH="$SELF_CACHE_LENGTH" CROSS_CACHE_LENGTH="$CROSS_CACHE_LENGTH" \
    MAX_LENGTH="$MAX_LENGTH" OFFSET_CROPS="$OFFSET_CROPS" LIMIT_CROPS="$LIMIT_CROPS" \
    OVER_CAPACITY="$OVER_CAPACITY" WARMUP_PASSES="$WARMUP_PASSES" \
    ADMISSION_PREFETCH_DEPTH="$ADMISSION_PREFETCH_DEPTH" \
    PREFAULT_ARTIFACT="$PREFAULT_ARTIFACT" REFERENCE_TRACE="$REFERENCE_TRACE" \
    REFERENCE_RUN_SUMMARY="$REFERENCE_RUN_SUMMARY" STEP_TRACE="$STEP_TRACE" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:-}" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then
  worker_entry "$2"
else
  launch_main
fi
