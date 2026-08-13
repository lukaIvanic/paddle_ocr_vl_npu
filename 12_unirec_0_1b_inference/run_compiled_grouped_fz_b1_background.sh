#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROBE="$SCRIPT_DIR/profile_compiled_grouped_fz_vision_b1.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for this NPU runtime}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical NPU first}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'COMPILED_GROUPED_FZ_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/config.json"
  test -f "$PROBE"
}

run_lane() {
  lane_name="$1"
  shift
  command=(
    "$PYTHON_BIN" "$PROBE"
    --model-path "$MODEL"
    --cache-root "$CACHE_ROOT"
    --output-dir "$RUN_ROOT/$lane_name"
    --device npu:0
    --warmups 3
    --control-repeats 20
    --parser-topn 200
    "$@"
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/${lane_name}_command.sh"
  printf '\n' >>"$RUN_ROOT/${lane_name}_command.sh"
  printf 'UNIREC_COMPILED_GROUPED_FZ_PHASE_BEGIN lane=%s\n' "$lane_name"
  set +e
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/${lane_name}.log"
  lane_status="${PIPESTATUS[0]}"
  set -e
  printf 'UNIREC_COMPILED_GROUPED_FZ_PHASE_END lane=%s status=%s\n' \
    "$lane_name" "$lane_status"
  return "$lane_status"
}

snapshot_oms() {
  destination="$1"
  find "$CACHE_ROOT" -name '*.om' -printf '%P\t%s\t%T@\n' 2>/dev/null \
    | sort >"$destination"
}

worker_main() {
  RUN_ROOT="$1"
  CACHE_ROOT="$2"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\nmodel=%s\ncache=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN" "$MODEL" "$CACHE_ROOT"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  status=0
  SECONDS=0
  snapshot_oms "$RUN_ROOT/om_before.tsv"
  run_lane native --lane native || status="$?"
  snapshot_oms "$RUN_ROOT/om_after_native.tsv"
  if test "$status" -eq 0; then
    run_lane grouped \
      --lane converter_grouped_fz \
      --reference-output "$RUN_ROOT/native/compiled_output.pt" \
      || status="$?"
  fi
  snapshot_oms "$RUN_ROOT/om_after_grouped.tsv"
  if test "$status" -eq 0; then
    run_lane grouped_warm \
      --lane converter_grouped_fz \
      --reference-output "$RUN_ROOT/native/compiled_output.pt" \
      || status="$?"
  fi
  snapshot_oms "$RUN_ROOT/om_after_warm.tsv"

  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$SECONDS" >"$RUN_ROOT/process_wall_s.txt"
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
  printf 'UNIREC_COMPILED_GROUPED_FZ_WORKER_END status=%s wall_s=%s run_log=%s\n' \
    "$status" "$SECONDS" "$RUN_ROOT/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/compiled_grouped_fz_b1_${commit_short}_${timestamp}}"
  CACHE_ROOT="${CACHE_ROOT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/compiled_grouped_fz_b1_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  CACHE_ROOT="$(realpath -m "$CACHE_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" "$CACHE_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_COMPILED_GROUPED_FZ_STARTED pid=%s physical=%s\n' \
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
