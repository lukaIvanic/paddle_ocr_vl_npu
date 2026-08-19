#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/decode_head_padding_b1_lab.py"
PARSER="$REPO/03_compiled_single_batch_decode/parse_npu_profile.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc)$' || true
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  CACHE_PARENT="${CACHE_PARENT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/decode_headpad_b1_310p}"
  CACHE_PARENT="$(realpath -m "$CACHE_PARENT")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -s "$LAB"
  test -s "$PARSER"
  mkdir -p "$CACHE_PARENT"
  export PYTHON_BIN MODEL CACHE_PARENT
}

cache_inventory() {
  find "$CACHE_PARENT" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %T@\n' 2>/dev/null | sort >"$1"
}

run_lane() {
  local lane="$1" heads="$2" lane_root child status started now profile_dir
  lane_root="$RUN_ROOT/$lane"
  mkdir -p "$lane_root"
  env PYTHONUNBUFFERED=1 TORCH_LOGS=recompiles \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$LAB" --model "$MODEL" --device npu:0 \
      --cache-dir "$CACHE_PARENT/$lane" --output "$lane_root/result.json" \
      --physical-heads "$heads" --source-length 56 --warmup-steps 20 \
      --measure-steps 100 --timing-steps 100 \
      >"$lane_root/run.log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$lane_root/pid.txt"
  started="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_310P_DECODE_HEAD_PADDING_B1_HEARTBEAT lane=%s elapsed_s=%s compiler_processes=%s om_count=%s last_event=%q\n' \
      "$lane" "$((now - started))" "$(compiler_count)" \
      "$(find "$CACHE_PARENT" -type f -name '*.om' 2>/dev/null | wc -l)" \
      "$(grep 'UNIREC_DECODE_HEAD_PADDING_B1_PROGRESS' "$lane_root/run.log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$lane_root/run.log"
  printf 'UNIREC_310P_DECODE_HEAD_PADDING_B1_LANE_END lane=%s status=%s wall_s=%s\n' \
    "$lane" "$status" "$(( $(date +%s) - started ))"
  test "$status" -eq 0
  test -s "$lane_root/result.json"
  ! grep -Eqi 'Skip cache as .*recompiled|Traceback|ERR[0-9]{5}|AICORE.*timeout' \
    "$lane_root/run.log"
  profile_dir="$("$PYTHON_BIN" - "$lane_root/result.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["compiled_profile"]["profile_dir"])
PY
)"
  "$PYTHON_BIN" "$PARSER" --profile-dir "$profile_dir" --topn 80 \
    --skip-trace --out-json "$lane_root/profile_summary.json" \
    --out-md "$lane_root/profile_summary.md" >/dev/null
}

write_report() {
  RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/final_report.txt"
import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["RUN_ROOT"])
h6 = json.load(open(root / "heads6/result.json"))
h8 = json.load(open(root / "heads8/result.json"))
a = np.load(root / "heads6/result.validation_logits.npy").astype(np.float64)
b = np.load(root / "heads8/result.validation_logits.npy").astype(np.float64)
assert a.shape == b.shape
delta = np.abs(a - b)
cosine = float(
    np.dot(a.ravel(), b.ravel())
    / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
)

def profile(lane):
    value = json.load(open(root / lane / "profile_summary.json"))["runs"][0]
    ops = {
        row["op_type"]: row
        for row in value["op_statistic"]["top_op_types"]
    }
    def op(name):
        row = ops.get(name, {})
        return {
            "count": int(row.get("count", 0)),
            "total_time_us": float(row.get("total_time_us", 0.0)),
        }
    kernels = value["kernel_details"]
    return {
        "kernel_count": int(kernels["row_count"]),
        "kernel_duration_us": float(kernels["total_duration_us"]),
        "TransData": op("TransData"),
        "MatMulV2": op("MatMulV2"),
        "MatMul": op("MatMul"),
        "IncreFlashAttention": op("IncreFlashAttention"),
    }

def timing(value):
    queued = value["compiled_timing"]["queued"]
    production = value["compiled_timing"]["production_like_d2h"]
    return {
        "step_ms": value["step_ms"],
        "raw_tok_s": value["raw_tok_s"],
        "queued_device_step_ms": queued["device_step_ms"],
        "queued_wall_step_ms": queued["wall_step_ms"],
        "production_like_device_step_ms": production["device_step_ms"],
        "production_like_wall_step_ms": production["wall_step_ms"],
        "sampled_token_d2h_wait_step_ms": production[
            "sampled_token_d2h_wait_step_ms"
        ],
        "first_call_s": value["first_call_s"],
        "same_lane_compiled_vs_eager": value["compiled_vs_same_lane_eager"],
    }

argmax_exact = bool(a.argmax() == b.argmax())
report = {
    "status": "pass",
    "shape_heads6": h6["shape"],
    "shape_heads8": h8["shape"],
    "heads6": {"timing": timing(h6), "profile": profile("heads6")},
    "heads8": {"timing": timing(h8), "profile": profile("heads8")},
    "heads8_over_heads6": {
        "raw_tok_s_speedup": h8["raw_tok_s"] / h6["raw_tok_s"],
        "step_speedup": h6["step_ms"] / h8["step_ms"],
        "production_like_wall_speedup": (
            h6["compiled_timing"]["production_like_d2h"]["wall_step_ms"]
            / h8["compiled_timing"]["production_like_d2h"]["wall_step_ms"]
        ),
    },
    "parity": {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": cosine,
        "argmax_exact": argmax_exact,
    },
    "run_root": str(root),
}
assert np.isfinite(a).all() and np.isfinite(b).all()
assert argmax_exact
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_DECODE_HEAD_PADDING_B1: PASS")
print(
    "UNIREC_310P_DECODE_HEAD_PADDING_B1_RESULT "
    + json.dumps(report, sort_keys=True)
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  printf 'commit=%s\nphysical_npu=%s\npython=%s\nmodel=%s\ncache_parent=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$MODEL" "$CACHE_PARENT" >"$RUN_ROOT/command.txt"
  cache_inventory "$RUN_ROOT/cache_before.txt"
  run_lane heads6 6
  run_lane heads8 8
  cache_inventory "$RUN_ROOT/cache_after.txt"
  diff -u "$RUN_ROOT/cache_before.txt" "$RUN_ROOT/cache_after.txt" \
    >"$RUN_ROOT/cache.diff" || true
  write_report
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short stamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  stamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_headpad_b1_${short}_${stamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    CACHE_PARENT="$CACHE_PARENT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
