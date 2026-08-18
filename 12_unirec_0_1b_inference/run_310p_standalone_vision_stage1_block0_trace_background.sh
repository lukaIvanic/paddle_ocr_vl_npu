#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROBE="$SCRIPT_DIR/standalone_vision_torchair_divergence.py"

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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_standalone_vision_stage1_block0_trace_$(date +%Y%m%dT%H%M%S)}"
  CACHE_ROOT="${CACHE_ROOT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/standalone_vision_divergence}"
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

printf 'UNIREC_STANDALONE_BLOCK0_TRACE_BEGIN epoch_s=%s expected_graphs=1\n' "$(date +%s)"
printf '%s\n' "$(git -C "$REPO" rev-parse HEAD)" >"$RUN_ROOT/commit.txt"

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" "$PROBE" \
    --cache-root "$CACHE_ROOT" \
    --output "$RUN_ROOT/report.json" \
    --start-stage 1 \
    --trace-stage1-block0-ops \
    --weight-format torchair_internal \
    --timing-repeats 1

"$PYTHON_BIN" - "$RUN_ROOT/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "UNIREC_STANDALONE_BLOCK0_TRACE_SUMMARY "
    f"first_divergent_boundary={report['first_divergent_boundary']} "
    f"compiled_first_ms={report['first_call_ms']['compiled']:.3f} "
    f"compiled_steady_ms={report['steady_p50_ms']['compiled']:.3f} "
    f"cache_changed={str(report['cache_changed']).lower()}"
)
for name, row in report["boundary_comparison"].items():
    if not name.startswith("stage_1_block_0"):
        continue
    valid = row["valid_compact"]
    print(
        "UNIREC_STANDALONE_BLOCK0_TRACE_BOUNDARY "
        f"name={name} max_abs={valid['max_abs']:.9g} "
        f"rmse={valid['rmse']:.9g} cosine={valid['cosine']:.9g}"
    )
PY

printf 'UNIREC_STANDALONE_BLOCK0_TRACE_END epoch_s=%s\n' "$(date +%s)"
