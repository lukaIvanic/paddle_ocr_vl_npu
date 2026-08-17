#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SWEEP="$SCRIPT_DIR/vision_compiled_shape_batch_sweep.py"
ANALYZER="$SCRIPT_DIR/analyze_vision_shape_batch_sweep.py"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?set exactly one physical NPU before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_VISION_SWEEP_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec Python executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${COMPILE_CACHE:?export the warmed production recognition cache parent}"
  reject_bad_device
  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$COMPILE_CACHE"
  test -f "$SWEEP"
  test -f "$ANALYZER"
}

run_point_set() {
  local run_root="$1" width="$2" height="$3" batches="$4"
  local key="${width}x${height}" output="$run_root/${width}x${height}.json"
  local command=(
    "$PYTHON_BIN" "$SWEEP"
    --model-path "$MODEL"
    --cache-dir "$COMPILE_CACHE"
    --output "$output"
    --device npu:0
    --width "$width"
    --height "$height"
    --batch-sizes "$batches"
    --warmups 2
    --repeats 20
    --focal-depthwise-rewrite constant_grouped_all
    --weight-format torchair_internal
  )
  printf 'UNIREC_VISION_SWEEP_SHAPE_BEGIN shape=%s batches=%s\n' "$key" "$batches"
  printf '%q ' "${command[@]}" >"$run_root/command_${key}.sh"
  printf '\n' >>"$run_root/command_${key}.sh"
  "${command[@]}"
  printf 'UNIREC_VISION_SWEEP_SHAPE_END shape=%s output=%s\n' "$key" "$output"
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\npython=%s\nmodel=%s\ncache=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN" "$MODEL" "$COMPILE_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  run_point_set "$run_root" 960 64 1,4,16
  run_point_set "$run_root" 512 256 1,2,4,8,16
  run_point_set "$run_root" 960 256 1,4
  run_point_set "$run_root" 512 512 1,4,8
  run_point_set "$run_root" 960 512 1,4

  local analyzer_command=(
    "$PYTHON_BIN" "$ANALYZER"
    --input "$run_root/960x64.json"
    --input "$run_root/512x256.json"
    --input "$run_root/960x256.json"
    --input "$run_root/512x512.json"
    --input "$run_root/960x512.json"
    --output "$run_root/combined.json"
  )
  if [[ -n "${REFERENCE_JSON:-}" ]]; then
    analyzer_command+=(--reference "$(readlink -f "$REFERENCE_JSON")")
  fi
  printf '%q ' "${analyzer_command[@]}" >"$run_root/command_analyze.sh"
  printf '\n' >>"$run_root/command_analyze.sh"
  "${analyzer_command[@]}"
  npu-smi info >"$run_root/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_VISION_SWEEP_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/vision_shape_batch_sweep_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    COMPILE_CACHE="$COMPILE_CACHE" \
    REFERENCE_JSON="${REFERENCE_JSON:-}" \
    bash "$0" --worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_entry "${2:?worker run root required}"
else
  launch_main
fi
