#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREFILL="$SCRIPT_DIR/run_prefill_export.py"
REPLAY="$SCRIPT_DIR/production_decode_replay.py"
COMPARE="$SCRIPT_DIR/compare_decode_completion_traces.py"
PRESET=310p_k10_l4_aligned

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

phase() {
  printf 'UNIREC_ALIGNED_K10_PHASE phase=%s epoch_s=%s\n' "$1" "$(date +%s)"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym executable}"
  : "${MODEL:?export the unirec-0.1b model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${COMPILE_CACHE:?export the existing production compile-cache parent}"
  : "${CANONICAL_TRACE:?export the canonical 90.13 recognition_trace.jsonl}"
  : "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?export the passed B128 decode-cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P, 0-3}"
  : "${CPUSET:=0-63}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  for variable in MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE \
    CANONICAL_TRACE UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE; do
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
  test -s "$CANONICAL_TRACE"
  test "$(wc -l <"$CANONICAL_TRACE")" -gt 30000
  local exact_decode_cache
  exact_decode_cache="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
  test "$(find "$exact_decode_cache" -name compiled_module | wc -l)" -eq 1
  test "$(find "$exact_decode_cache" -name '*.om' | wc -l)" -eq 1
  taskset -c "$CPUSET" "$PYTHON_BIN" -c \
    'import os; n=len(os.sched_getaffinity(0)); print(f"UNIREC_ALIGNED_K10_CPU_AFFINITY={n}"); assert n >= 32, n'
  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE
  export CANONICAL_TRACE UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE CPUSET
}

cache_inventory() {
  local output="$1"
  COMPILE_CACHE="$COMPILE_CACHE" "$PYTHON_BIN" - "$output" <<'PY'
import json, os, sys
from pathlib import Path

slots = {
    "448x384_b2": 1,
    "512x64_b4": 2,
    "512x192_b2": 0,
    "960x64_b4": 3,
    "960x128_b2": 4,
    "960x256_b1": 5,
    "960x512_b1": 9,
    "960x1024_b1": 7,
    "1024x704_b1": 8,
    "1024x1408_b1": 6,
}
root = Path(os.environ["COMPILE_CACHE"])
report = {}
for key, slot in slots.items():
    directories = sorted(root.glob(
        f"vision_full_bucket_{key}_float16_*"
        "dwconstant_grouped_all*wtorchair_internal*"
    ))
    target_modules = []
    target_oms = []
    for directory in directories:
        target_modules.extend(directory.glob(
            f"**/_forward_bucket_slot_{slot}/compiled_module"
        ))
        for module in target_modules:
            if directory in module.parents:
                target_oms.extend(module.parent.rglob("*.om"))
    report[key] = {
        "slot": slot,
        "target_compiled_module_count": len(set(target_modules)),
        "target_om_count": len(set(target_oms)),
        "target_compiled_modules": [str(path) for path in sorted(set(target_modules))],
        "target_oms": [str(path) for path in sorted(set(target_oms))],
    }
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
missing = [key for key, row in report.items() if not row["target_compiled_module_count"]]
print(f"UNIREC_ALIGNED_K10_CACHE_INVENTORY missing={len(missing)} keys={missing}")
PY
}

prefill_command() {
  local output="$1" workers="$2" limit="$3" storage="$4" warmup_pages="$5"
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    UNIREC_VISION_BUCKET_PRESET="$PRESET" \
    UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG=1 \
    taskset -c "$CPUSET" "$PYTHON_BIN" "$PREFILL" \
      --openocr-root "$OPENOCR_ROOT" --model-path "$MODEL" \
      --layout-model "$LAYOUT_MODEL" --input "$IMAGES_DIR" \
      --output-dir "$output" --artifact-storage "$storage" \
      --offset 0 --limit "$limit" --workers "$workers" \
      --warmup-pages "$warmup_pages" --warmup-repeats 1 \
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
      --progress-heartbeat-s 15
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  cache_inventory "$run_root/cache_before.json"
  local missing_before
  missing_before="$($PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); print(sum(not r["target_compiled_module_count"] for r in p.values()))' "$run_root/cache_before.json")"
  printf 'UNIREC_ALIGNED_K10_EXPECTED_COMPILES count=%s\n' "$missing_before"
  if (( missing_before > 4 )); then
    echo "UNIREC_ALIGNED_K10_STOP unexpected_cache_misses=$missing_before" >&2
    return 1
  fi

  phase cache_builder_begin
  prefill_command "$run_root/cache_builder" 1 4 discard 0 \
    2>&1 | tee "$run_root/cache_builder.log"
  phase cache_builder_end
  cache_inventory "$run_root/cache_after_builder.json"
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); missing=[k for k,r in p.items() if not r["target_compiled_module_count"]]; assert not missing, missing' "$run_root/cache_after_builder.json"

  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' | sort \
    >"$run_root/om_before_hot.txt"
  phase hot_prefill_begin
  prefill_command "$run_root/hot_prefill" 4 128 persistent 8 \
    2>&1 | tee "$run_root/hot_prefill.log"
  phase hot_prefill_end
  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' | sort \
    >"$run_root/om_after_hot.txt"
  if diff -u "$run_root/om_before_hot.txt" "$run_root/om_after_hot.txt" \
      >"$run_root/hot_om.diff"; then
    printf 'UNIREC_ALIGNED_K10_HOT_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_ALIGNED_K10_HOT_OM_INVENTORY_CHANGED\n' >&2
    cat "$run_root/hot_om.diff" >&2
    return 1
  fi

  test "$(wc -l <"$run_root/hot_prefill/crops.jsonl")" -eq 957
  phase decode_begin
  env PYTHONUNBUFFERED=1 \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    taskset -c "$CPUSET" "$PYTHON_BIN" "$REPLAY" \
      --artifact-dir "$run_root/hot_prefill" --model-path "$MODEL" \
      --device npu:0 --dtype float16 --batch-size 128 \
      --self-cache-length 2048 --cross-cache-length 1320 --max-length 2048 \
      --decode-warmup-passes 2 --decode-admission-prefetch-depth 0 \
      --compile-cache-dir "$COMPILE_CACHE" --progress-every 100 --verify-crc \
      --reference-trace "$CANONICAL_TRACE" \
      --completion-trace-jsonl "$run_root/completions.jsonl" \
      --output "$run_root/decode_result.json" \
      2>&1 | tee "$run_root/decode.log"
  phase decode_end
  "$PYTHON_BIN" "$COMPARE" --candidate "$run_root/completions.jsonl" \
    --reference "$CANONICAL_TRACE" --output "$run_root/parity_report.json" \
    | tee "$run_root/parity.log"

  BUILDER="$run_root/cache_builder/summary.json" \
  HOT="$run_root/hot_prefill/summary.json" \
  DECODE="$run_root/decode_result.json" \
  PARITY="$run_root/parity_report.json" \
    "$PYTHON_BIN" - <<'PY' | tee "$run_root/final_report.txt"
import json, os

builder = json.load(open(os.environ["BUILDER"]))
hot = json.load(open(os.environ["HOT"]))
decode = json.load(open(os.environ["DECODE"]))
parity = json.load(open(os.environ["PARITY"]))
assert builder["status"] == hot["status"] == "ok"
assert hot["artifact"]["page_count"] == 128
assert hot["artifact"]["crop_count"] == 957
assert hot["artifact"]["rejected_crop_count"] == 0
assert parity["compared_count"] == parity["token_exact_count"] == 957
assert not parity["first_mismatches"]

def graph_times(summary):
    result = {}
    for worker in summary["worker_setup_diagnostics"]:
        graphs = worker["prefix_graph_warmup"]["graphs"]
        for key, row in graphs.items():
            result.setdefault(key, []).extend(row["pass_wall_s"])
    return result

builder_graphs = graph_times(builder)
hot_graphs = graph_times(hot)
print("UNIREC_ALIGNED_K10_FIRST128: PASS")
print(
    "UNIREC_ALIGNED_K10_CACHE_BUILDER "
    f"total_wall_s={builder['total_wall_s']:.6f} "
    f"setup_s={builder['setup_s']:.6f} "
    f"graph_s={json.dumps(builder_graphs, sort_keys=True)}"
)
print(
    "UNIREC_ALIGNED_K10_HOT_PREFILL "
    f"producer_wall_s={hot['producer_wall_s']:.6f} "
    f"pages_s={hot['throughput']['pages_per_s']:.6f} "
    f"setup_s={hot['setup_s']:.6f} total_wall_s={hot['total_wall_s']:.6f} "
    f"crops={hot['artifact']['crop_count']} "
    f"slot_eff={hot['worker_summary']['vision_batching']['compiled_slot_efficiency']:.6f} "
    f"fallback={hot['worker_summary']['vision_batching']['fallback_rows']} "
    f"graph_warmup_s={json.dumps(hot_graphs, sort_keys=True)}"
)
print(
    "UNIREC_ALIGNED_K10_DECODE "
    f"wall_s={decode['decode_wall_s']:.6f} "
    f"graph_s={decode['decode']['decode_s']:.6f} "
    f"raw_tok_s={decode['decode']['raw_decode_tokens_per_s']:.6f} "
    f"effective_tok_s={decode['decode']['effective_decode_tokens_per_s']:.6f}"
)
print(
    "UNIREC_ALIGNED_K10_PARITY "
    f"exact={parity['token_exact_count']}/{parity['compared_count']} "
    f"length_exact={parity['length_exact_count']} "
    f"mismatches={len(parity['first_mismatches'])}"
)
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
  printf 'UNIREC_ALIGNED_K10_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_aligned_k10_first128_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" COMPILE_CACHE="$COMPILE_CACHE" \
    CANONICAL_TRACE="$CANONICAL_TRACE" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
