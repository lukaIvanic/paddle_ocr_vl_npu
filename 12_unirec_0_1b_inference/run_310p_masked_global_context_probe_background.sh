#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROBE="$SCRIPT_DIR/standalone_masked_global_context_probe.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

if [[ "${1:-}" != "--worker" ]]; then
  : "${PYTHON_BIN:?validated venv/bin/python_nosym is required}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?one free physical 310P device 0-3 is required}"
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) exit 2 ;; esac
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_masked_global_context_$(date +%Y%m%dT%H%M%S)}"
  CACHE_ROOT="${CACHE_ROOT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/masked_global_context}"
  mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
  export PYTHON_BIN ASCEND_RT_VISIBLE_DEVICES RUN_ROOT CACHE_ROOT
  nohup "$0" --worker >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
  exit 0
fi

status=0
started_epoch="$(date +%s)"
trap 'status=$?; printf "%s\n" "$status" >"$RUN_ROOT/exit_code.txt"; printf "%s\n" "$(($(date +%s) - started_epoch))" >"$RUN_ROOT/process_wall_s.txt"' EXIT

printf 'UNIREC_GLOBAL_CONTEXT_RUN_BEGIN epoch_s=%s expected_graphs=1\n' "$(date +%s)"
printf '%s\n' "$(git -C "$REPO" rev-parse HEAD)" >"$RUN_ROOT/commit.txt"

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" "$PROBE" \
    --cache-root "$CACHE_ROOT" \
    --output "$RUN_ROOT/report.json" \
    --timing-repeats 3

printf 'UNIREC_GLOBAL_CONTEXT_RUN_END epoch_s=%s\n' "$(date +%s)"
