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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_standalone_vision_ladder_$(date +%Y%m%dT%H%M%S)}"
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

printf 'UNIREC_STANDALONE_LADDER_BEGIN epoch_s=%s stages=2,1,0\n' "$(date +%s)"
printf '%s\n' "$(git -C "$REPO" rev-parse HEAD)" >"$RUN_ROOT/commit.txt"

for stage in 2 1 0; do
  stage_root="$RUN_ROOT/stage_$stage"
  mkdir -p "$stage_root"
  printf 'UNIREC_STANDALONE_LADDER_STAGE_BEGIN stage=%s epoch_s=%s\n' \
    "$stage" "$(date +%s)"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" "$PROBE" \
      --cache-root "$CACHE_ROOT" \
      --output "$stage_root/report.json" \
      --start-stage "$stage" \
      --weight-format torchair_internal \
      --timing-repeats 1
  printf 'UNIREC_STANDALONE_LADDER_STAGE_END stage=%s epoch_s=%s\n' \
    "$stage" "$(date +%s)"

  if "$PYTHON_BIN" - "$stage_root/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
row = report["comparison"]["valid_compact"]
diverged = float(row["cosine"]) < 0.999 or float(row["max_abs"]) > 0.5
print(
    "UNIREC_STANDALONE_LADDER_RESULT "
    f"stage={report['start_stage']} cosine={row['cosine']:.9g} "
    f"max_abs={row['max_abs']:.9g} rmse={row['rmse']:.9g} "
    f"diverged={str(diverged).lower()}"
)
raise SystemExit(42 if diverged else 0)
PY
  then
    continue
  else
    comparison_status=$?
    if [[ "$comparison_status" -eq 42 ]]; then
      printf 'UNIREC_STANDALONE_LADDER_STOP first_divergent_stage=%s\n' "$stage"
      printf '%s\n' "$stage" >"$RUN_ROOT/first_divergent_stage.txt"
      exit 0
    fi
    exit "$comparison_status"
  fi
done

printf 'UNIREC_STANDALONE_LADDER_PASS all_stages_clean=true\n'
printf 'none\n' >"$RUN_ROOT/first_divergent_stage.txt"
