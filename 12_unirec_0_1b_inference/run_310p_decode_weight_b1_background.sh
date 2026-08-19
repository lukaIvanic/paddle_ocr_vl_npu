#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB="$SCRIPT_DIR/decode_weight_format_b1_lab.py"
PARSER="$REPO/03_compiled_single_batch_decode/parse_npu_profile.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

compiler_count() {
  ps -eo comm= | grep -Ec '^(ccec_compiler|op_compiler|atc)$' || true
}

cache_inventory() {
  find "$CACHE_PARENT" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %T@\n' 2>/dev/null | sort >"$1"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated venv python_nosym}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  CACHE_PARENT="${CACHE_PARENT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/decode_weight_b1_310p}"
  CACHE_PARENT="$(realpath -m "$CACHE_PARENT")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -s "$LAB"
  test -s "$PARSER"
  mkdir -p "$CACHE_PARENT"
  export PYTHON_BIN MODEL CACHE_PARENT
}

run_lane() {
  local lane="$1" format="$2" lane_root child status started now
  lane_root="$RUN_ROOT/$lane"
  mkdir -p "$lane_root"
  env PYTHONUNBUFFERED=1 TORCH_LOGS=recompiles \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$LAB" \
      --model "$MODEL" --device npu:0 --cache-dir "$CACHE_PARENT" \
      --output "$lane_root/result.json" --weight-formats "$format" \
      --source-length 56 --warmup-steps 20 --measure-steps 1000 \
      --timing-steps 100 >"$lane_root/run.log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$lane_root/pid.txt"
  started="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_310P_DECODE_WEIGHT_B1_HEARTBEAT lane=%s elapsed_s=%s compiler_processes=%s om_count=%s last_event=%q\n' \
      "$lane" "$((now - started))" "$(compiler_count)" \
      "$(find "$CACHE_PARENT" -type f -name '*.om' 2>/dev/null | wc -l)" \
      "$(grep 'UNIREC_DECODE_WEIGHT_B1_PROGRESS' "$lane_root/run.log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$lane_root/run.log"
  printf 'UNIREC_310P_DECODE_WEIGHT_B1_LANE_END lane=%s status=%s wall_s=%s\n' \
    "$lane" "$status" "$(( $(date +%s) - started ))"
  test "$status" -eq 0
  test -s "$lane_root/result.json"
  ! grep -Eqi 'Skip cache as .*recompiled|Traceback|ERR[0-9]{5}|AICORE.*timeout' \
    "$lane_root/run.log"

  local profile_dir
  profile_dir="$("$PYTHON_BIN" - "$lane_root/result.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
entry = next(iter(value["lanes"].values()))
print(entry["compiled_profile"]["profile_dir"])
PY
)"
  "$PYTHON_BIN" "$PARSER" --profile-dir "$profile_dir" --topn 80 \
    --skip-trace --out-json "$lane_root/profile_summary.json" \
    --out-md "$lane_root/profile_summary.md" >/dev/null
}

write_report() {
  RUN_ROOT="$RUN_ROOT" PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" - <<'PY' \
    | tee "$RUN_ROOT/final_report.txt"
import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["RUN_ROOT"])
nd = json.load(open(root / "nd/result.json"))
nz = json.load(open(root / "nz/result.json"))
nd_lane = nd["lanes"]["nd"]
nz_lane = nz["lanes"]["nz"]

nd_logits = np.load(root / "nd/result.validation_logits.npy").astype(np.float64)
nz_logits = np.load(root / "nz/result.validation_logits.npy").astype(np.float64)
assert nd_logits.shape == nz_logits.shape
delta = np.abs(nd_logits - nz_logits)
cosine = float(
    np.dot(nd_logits.ravel(), nz_logits.ravel())
    / (np.linalg.norm(nd_logits.ravel()) * np.linalg.norm(nz_logits.ravel()))
)
argmax_exact = bool(np.argmax(nd_logits) == np.argmax(nz_logits))
assert np.isfinite(nd_logits).all() and np.isfinite(nz_logits).all()

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
        "aicore_time_us": float(kernels["total_aicore_time_us"]),
        "weighted_cube_utilization_pct": float(
            kernels["weighted_cube_utilization_pct"]
        ),
        "TransData": op("TransData"),
        "MatMulV2": op("MatMulV2"),
        "MatMul": op("MatMul"),
        "IncreFlashAttention": op("IncreFlashAttention"),
    }

def timing(value):
    queued = value["compiled_timing"]["queued"]
    d2h = value["compiled_timing"]["production_like_d2h"]
    return {
        "step_ms": value["step_ms"],
        "raw_tok_s": value["raw_tok_s"],
        "queued_device_step_ms": queued["device_step_ms"],
        "queued_wall_step_ms": queued["wall_step_ms"],
        "production_like_device_step_ms": d2h["device_step_ms"],
        "production_like_wall_step_ms": d2h["wall_step_ms"],
        "sampled_token_d2h_wait_step_ms": d2h[
            "sampled_token_d2h_wait_step_ms"
        ],
        "first_call_s": value["first_call_s"],
        "weight_format_s": value["nz_format_s"],
        "nz_tensor_count": value["nz_tensor_count"],
    }

report = {
    "status": "pass",
    "shape": nd["shape"],
    "nd": {"timing": timing(nd_lane), "profile": profile("nd")},
    "nz": {"timing": timing(nz_lane), "profile": profile("nz")},
    "nz_over_nd": {
        "raw_tok_s_speedup": nz_lane["raw_tok_s"] / nd_lane["raw_tok_s"],
        "step_speedup": nd_lane["step_ms"] / nz_lane["step_ms"],
        "production_like_wall_speedup": (
            nd_lane["compiled_timing"]["production_like_d2h"]["wall_step_ms"]
            / nz_lane["compiled_timing"]["production_like_d2h"]["wall_step_ms"]
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
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_DECODE_WEIGHT_B1: PASS")
print("UNIREC_310P_DECODE_WEIGHT_B1_RESULT " + json.dumps(report, sort_keys=True))
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  printf 'commit=%s\nphysical_npu=%s\npython=%s\nmodel=%s\ncache_parent=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$MODEL" "$CACHE_PARENT" >"$RUN_ROOT/command.txt"
  cache_inventory "$RUN_ROOT/cache_before.txt"
  run_lane nd nd
  run_lane nz nz
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_weight_b1_${short}_${stamp}}"
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
