#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SWEEP="$SCRIPT_DIR/decode_mask_occupancy_sweep.py"
REFERENCE_ROOT="$SCRIPT_DIR/references/unirec_decode_mask_profile_910b_20260819"

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
  : "${ARTIFACT_DIR:?export the canonical first-128 persistent artifact}"
  : "${DECODE_CACHE_PARENT:?export the completed dual-decode cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device 0-3}"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  ARTIFACT_DIR="$(readlink -f "$ARTIFACT_DIR")"
  DECODE_CACHE_PARENT="$(readlink -f "$DECODE_CACHE_PARENT")"
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -s "$ARTIFACT_DIR/crops.jsonl"
  test -d "$DECODE_CACHE_PARENT"
  test -s "$REFERENCE_ROOT/a_profile.json"
  test -s "$REFERENCE_ROOT/b_profile.json"
  for shape in \
    decode_selfkv256_cross256_increfa_all_b128 \
    decode_selfkv2048_cross1320_increfa_all_b128
  do
    local cache="$DECODE_CACHE_PARENT/$shape"
    test "$(find "$cache" -type f -name compiled_module | wc -l)" -eq 1
    test "$(find "$cache" -type f -name '*.om' | wc -l)" -eq 1
  done
}

cache_inventory() {
  find "$DECODE_CACHE_PARENT" -type f \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %T@\n' | sort >"$1"
}

run_lane() {
  local lane="$1" self_len="$2" cross_len="$3"
  shift 3
  local lane_root="$RUN_ROOT/$lane" child status started now
  mkdir -p "$lane_root"
  env PYTHONUNBUFFERED=1 UNIREC_STATIC_CACHE_LEN="$self_len" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$SWEEP" --model "$MODEL" \
      --artifact-crops-jsonl "$ARTIFACT_DIR/crops.jsonl" --device npu:0 \
      --batch-size 128 --self-cache-length "$self_len" \
      --cross-cache-length "$cross_len" --active-rows 16 32 64 96 128 \
      --source-modes realistic full --warmup-steps 5 --measure-steps 30 \
      --cache-dir "$DECODE_CACHE_PARENT" --output "$lane_root/result.json" \
      "$@" >"$lane_root/run.log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" >"$lane_root/pid.txt"
  started="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    now="$(date +%s)"
    printf 'UNIREC_310P_DECODE_MASK_SWEEP_HEARTBEAT lane=%s elapsed_s=%s compiler_processes=%s last_point=%q\n' \
      "$lane" "$((now - started))" "$(compiler_count)" \
      "$(grep 'UNIREC_DECODE_MASK_SWEEP_POINT' "$lane_root/run.log" | tail -n 1 || true)"
  done
  set +e
  wait "$child"
  status="$?"
  set -e
  cat "$lane_root/run.log"
  printf 'UNIREC_310P_DECODE_MASK_SWEEP_LANE_END lane=%s status=%s wall_s=%s\n' \
    "$lane" "$status" "$(( $(date +%s) - started ))"
  test "$status" -eq 0
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  cache_inventory "$RUN_ROOT/om_before.txt"
  run_lane a 256 256 --cache-positions 32
  run_lane b 2048 1320 --cache-positions 32 1023
  cache_inventory "$RUN_ROOT/om_after.txt"
  diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
    >"$RUN_ROOT/om.diff"
  ! grep -ERqi \
    'recompil|skip cache|compile graph|start.*compil|Traceback|ERROR' \
    "$RUN_ROOT/a/run.log" "$RUN_ROOT/b/run.log"

  A_RESULT="$RUN_ROOT/a/result.json" B_RESULT="$RUN_ROOT/b/result.json" \
    REF_ROOT="$REFERENCE_ROOT" RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' \
      | tee "$RUN_ROOT/final_report.txt"
import json
import os
from pathlib import Path

a = json.load(open(os.environ["A_RESULT"]))
b = json.load(open(os.environ["B_RESULT"]))
ref_root = Path(os.environ["REF_ROOT"])
for value, shape in ((a, (128, 256, 256)), (b, (128, 2048, 1320))):
    assert value["kind"] == "unirec_decode_mask_occupancy_sweep"
    assert value["status"] == "ok"
    assert (
        value["shape"]["batch_size"],
        value["shape"]["self_cache_length"],
        value["shape"]["cross_cache_length"],
    ) == shape

def key(point):
    return (
        int(point["active_rows"]),
        int(point["initial_cache_position"]),
        str(point.get("source_mode", "realistic")),
    )

def points(value):
    return {key(point): point for point in value["points"]}

def load_reference(lane):
    occupancy = json.load(open(ref_root / f"{lane}_occupancy.json"))
    full = json.load(open(ref_root / f"{lane}_full_control.json"))
    merged = points(occupancy)
    merged.update(points(full))
    return merged

ap = points(a)
bp = points(b)
ar = load_reference("a")
br = load_reference("b")

def rows(measured, reference):
    output = []
    for point_key, point in sorted(measured.items()):
        ref = reference.get(point_key)
        output.append(
            {
                "active_rows": point_key[0],
                "cache_position": point_key[1],
                "source_mode": point_key[2],
                "step_ms": point["decode_step_ms"],
                "raw_tok_s": point["raw_tok_s"],
                "effective_tok_s": point["effective_tok_s"],
                "source_mean": point["source_lengths"]["mean"],
                "reference_910b_step_ms": (
                    ref["decode_step_ms"] if ref is not None else None
                ),
                "slowdown_vs_910b": (
                    point["decode_step_ms"] / ref["decode_step_ms"]
                    if ref is not None else None
                ),
            }
        )
    return output

def full_vs_realistic(measured, position):
    realistic = measured[(128, position, "realistic")]
    full = measured[(128, position, "full")]
    return {
        "realistic_step_ms": realistic["decode_step_ms"],
        "full_step_ms": full["decode_step_ms"],
        "full_over_realistic": (
            full["decode_step_ms"] / realistic["decode_step_ms"]
        ),
    }

report = {
    "status": "pass",
    "a": rows(ap, ar),
    "b": rows(bp, br),
    "mask_sensitivity": {
        "a_position32": full_vs_realistic(ap, 32),
        "b_position32": full_vs_realistic(bp, 32),
        "b_position1023": full_vs_realistic(bp, 1023),
    },
    "reference_kernel_profiles": {
        "a": str(ref_root / "a_profile.json"),
        "b": str(ref_root / "b_profile.json"),
    },
    "run_root": os.environ["RUN_ROOT"],
}
print("UNIREC_310P_DECODE_MASK_OCCUPANCY_SWEEP: PASS")
print("UNIREC_310P_DECODE_MASK_OCCUPANCY_RESULT " + json.dumps(report, sort_keys=True))
PY
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
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_decode_mask_sweep_${short}_${stamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    ARTIFACT_DIR="$ARTIFACT_DIR" DECODE_CACHE_PARENT="$DECODE_CACHE_PARENT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
