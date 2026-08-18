#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
PRESET=310p_k20_l4

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

phase() {
  printf 'UNIREC_K20_CACHE_PHASE phase=%s epoch_s=%s\n' "$1" "$(date +%s)"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym executable}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${COMPILE_CACHE:?export the existing aligned-K10 production cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${CPUSET:=0-63}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  for variable in MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE; do
    printf -v "$variable" '%s' "$(readlink -f "${!variable}")"
  done
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) echo 310P_DEVICE_MUST_BE_0_TO_3 >&2; exit 1;; esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  taskset -c "$CPUSET" "$PYTHON_BIN" -c \
    'import os; n=len(os.sched_getaffinity(0)); print(f"UNIREC_K20_CPU_AFFINITY={n}"); assert n >= 32, n'
  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE CPUSET
}

cache_inventory() {
  local output="$1"
  COMPILE_CACHE="$COMPILE_CACHE" SCRIPT_DIR="$SCRIPT_DIR" \
    "$PYTHON_BIN" - "$output" <<'PY'
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["SCRIPT_DIR"])
import vision_bucket_presets
import vision_full_batch

specs = vision_bucket_presets.VISION_BUCKET_PRESETS["310p_k20_l4"]
slots = vision_bucket_presets.assign_vision_bucket_cache_slots(
    specs,
    slot_count=max(10, len(specs)),
)
flat_keys = set(vision_full_batch.FLAT_GLOBAL_CONTEXT_BUCKET_KEYS)
extended_keys = set(vision_full_batch.EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS)
root = Path(os.environ["COMPILE_CACHE"])
report = {}
for spec, slot in zip(specs, slots):
    key = spec.key
    if key in extended_keys:
        source_hash = vision_full_batch._extended_flat_global_context_source_hash()
        method = f"_forward_flat_bucket_slot_{slot}"
        mode = "direct_2d_extended"
    elif key in flat_keys:
        source_hash = vision_full_batch._flat_global_context_source_hash()
        method = f"_forward_flat_bucket_slot_{slot}"
        mode = "direct_2d_legacy"
    else:
        source_hash = vision_full_batch._source_hash()
        method = f"_forward_bucket_slot_{slot}"
        mode = "legacy_two_stage"
    directories = sorted(root.glob(
        f"vision_full_bucket_{key}_float16_src{source_hash}_"
        "dwconstant_grouped_all*wtorchair_internal*"
    ))
    modules = []
    oms = []
    for directory in directories:
        found = list(directory.glob(f"**/{method}/compiled_module"))
        modules.extend(found)
        for module in found:
            oms.extend(module.parent.glob("*.om"))
    report[key] = {
        "slot": slot,
        "method": method,
        "source_hash": source_hash,
        "global_context_mode": mode,
        "target_compiled_module_count": len(set(modules)),
        "target_om_count": len(set(oms)),
        "target_compiled_modules": [str(p) for p in sorted(set(modules))],
        "target_oms": [str(p) for p in sorted(set(oms))],
    }
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
missing = [key for key, row in report.items() if not row["target_compiled_module_count"]]
print(f"UNIREC_K20_CACHE_INVENTORY present={len(report)-len(missing)} missing={len(missing)} keys={missing}")
PY
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  cache_inventory "$run_root/cache_before.json"
  local legacy_missing new_missing
  read -r legacy_missing new_missing < <(
    "$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1]))
missing=[k for k,r in p.items() if not r["target_compiled_module_count"]]
legacy=[k for k in missing if not p[k]["global_context_mode"].startswith("direct_2d_extended")]
new=[k for k in missing if p[k]["global_context_mode"].startswith("direct_2d_extended")]
print(len(legacy), len(new))' "$run_root/cache_before.json"
  )
  printf 'UNIREC_K20_EXPECTED_COMPILES legacy_missing=%s new_missing=%s\n' "$legacy_missing" "$new_missing"
  if (( legacy_missing != 0 )); then
    echo 'UNIREC_K20_STOP legacy K10 cache identity is unexpectedly missing; inspect cache_before.json' >&2
    return 1
  fi
  if (( new_missing > 14 )); then
    echo "UNIREC_K20_STOP unexpected new-graph miss count=$new_missing" >&2
    return 1
  fi

  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' | sort \
    >"$run_root/om_before.txt"
  phase builder_begin
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    UNIREC_VISION_BUCKET_PRESET="$PRESET" \
    UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG=1 \
    taskset -c "$CPUSET" "$PYTHON_BIN" "$PREFILL" \
      --openocr-root "$OPENOCR_ROOT" --model-path "$MODEL" \
      --layout-model "$LAYOUT_MODEL" --input "$IMAGES_DIR" \
      --output-dir "$run_root/builder" --artifact-storage discard \
      --offset 0 --limit 4 --workers 1 --warmup-pages 0 --warmup-repeats 1 \
      --layout-threshold 0.5 --layout-execution eager \
      --layout-dtype float32 --layout-reading-order-dtype float32 \
      --layout-weight-format native --layout-depthwise-rewrite native \
      --layout-batch-size 2 --dtype float16 --cross-cache-length 1320 \
      --recognition-cache-dir "$COMPILE_CACHE" --vision-full-batches \
      --vision-focal-depthwise-rewrite constant_grouped_all \
      --vision-weight-format torchair_internal \
      --recognition-input-contract compact_uint8_hwc \
      --recognition-preprocess-threads 8 --vision-page-lookahead 4 \
      --no-retain-shared-images --progress-every-pages 1 \
      --progress-heartbeat-s 15 2>&1 | tee "$run_root/builder.log"
  phase builder_end
  cache_inventory "$run_root/cache_after.json"
  "$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); missing=[k for k,r in p.items() if not r["target_compiled_module_count"]]
assert not missing, missing' "$run_root/cache_after.json"
  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' | sort \
    >"$run_root/om_after.txt"

  SUMMARY="$run_root/builder/summary.json" "$PYTHON_BIN" - <<'PY' | tee "$run_root/final_report.txt"
import json, os
s = json.load(open(os.environ["SUMMARY"]))
assert s["status"] == "ok"
times = {}
for worker in s["worker_setup_diagnostics"]:
    for key, row in worker["prefix_graph_warmup"]["graphs"].items():
        times.setdefault(key, []).extend(row["pass_wall_s"])
print("UNIREC_K20_CACHE_BUILDER: PASS")
print(f"UNIREC_K20_CACHE_BUILDER_WALL total_s={s['total_wall_s']:.6f} setup_s={s['setup_s']:.6f}")
print("UNIREC_K20_GRAPH_WARMUP_S " + json.dumps(times, sort_keys=True))
PY
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_K20_CACHE_WORKER_END status=%s run_log=%s\n' "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_k20_cache_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
