#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
EXTENSION_ROOT="$SCRIPT_DIR/custom_ops/layout_msda_aclnn/pytorch_extension"
PROBE="$SCRIPT_DIR/probe_layout_msda_binding.py"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'MSDA_BINDING_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
}

resolve_python() {
  : "${PYTHON_BIN:?export the existing UniRec Python interpreter}"
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
  test -x "$PYTHON_BIN"
  test -f "$EXTENSION_ROOT/setup.py"
  test -f "$PROBE"
}

worker_main() {
  RUN_ROOT="$1"
  reject_bad_device
  resolve_python

  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_BEGIN phase=environment\n'
  {
    printf 'commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-}"
    printf 'ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-}"
    printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-}"
    uname -a
    "$PYTHON_BIN" -c 'import torch,torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
    npu-smi info
  } | tee "$RUN_ROOT/environment.txt"
  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_END phase=environment\n'

  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_BEGIN phase=build\n'
  mkdir -p "$RUN_ROOT/extension" "$RUN_ROOT/build"
  (
    cd "$EXTENSION_ROOT"
    MAX_JOBS=1 USE_NINJA=0 "$PYTHON_BIN" setup.py build_ext \
      --build-lib "$RUN_ROOT/extension" \
      --build-temp "$RUN_ROOT/build"
  ) 2>&1 | tee "$RUN_ROOT/build.log"
  mapfile -t extension_sos < <(
    find "$RUN_ROOT/extension" -type f -name '_C*.so' -print | sort
  )
  if [[ "${#extension_sos[@]}" != 1 ]]; then
    printf 'ERROR expected exactly one extension SO, found %s\n' \
      "${#extension_sos[@]}" >&2
    printf '%s\n' "${extension_sos[@]}" >&2
    return 2
  fi
  printf '%s\n' "${extension_sos[0]}" >"$RUN_ROOT/extension_so.txt"
  printf 'UNIREC_LAYOUT_MSDA_BINDING_SO=%s\n' "${extension_sos[0]}"
  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_END phase=build\n'

  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_BEGIN phase=runtime\n'
  "$PYTHON_BIN" "$PROBE" \
    --extension-so "${extension_sos[0]}" \
    --output "$RUN_ROOT/binding_probe.json" \
    --warmup 10 \
    --repeats 50
  printf 'UNIREC_LAYOUT_MSDA_BINDING_PHASE_END phase=runtime\n'
  npu-smi info >"$RUN_ROOT/npu_after.txt" 2>&1 || true
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  npu-smi info >"$run_root/npu_after.txt" 2>&1 || true
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_BINDING_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  reject_bad_device
  resolve_python
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_msda_binding_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}" \
    ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-}" \
    ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid" 2>/dev/null || test -f "$RUN_ROOT/exit_code.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_BINDING_STARTED pid=%s physical=%s\n' \
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
