#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the K20 cache parent completed by the builder}"
  : "${LAYOUT_CACHE_ROOT:?export the warmed optimized layout cache}"
  : "${K10_REFERENCE_SUMMARY:?export the prior representative-128 W4 K10 run_summary.json}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device, 0-3}"
  : "${TASKSET_CPUS:?export the known-good 64-CPU taskset mask}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  for variable in MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE \
    LAYOUT_CACHE_ROOT K10_REFERENCE_SUMMARY; do
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
  test -d "$LAYOUT_CACHE_ROOT"
  test -s "$K10_REFERENCE_SUMMARY"
  CPU_AFFINITY_COUNT="$(taskset -c "$TASKSET_CPUS" "$PYTHON_BIN" -c \
    'import os; print(len(os.sched_getaffinity(0)))')"
  if (( CPU_AFFINITY_COUNT < 64 )); then
    printf 'UNIREC_K20_REP128_CPU_AFFINITY_TOO_SMALL count=%s mask=%s\n' \
      "$CPU_AFFINITY_COUNT" "$TASKSET_CPUS" >&2
    exit 1
  fi
  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE
  export LAYOUT_CACHE_ROOT K10_REFERENCE_SUMMARY TASKSET_CPUS CPU_AFFINITY_COUNT
}

om_inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'recognition %p %s %T@\n'
    find "$LAYOUT_CACHE_ROOT" -type f -name '*.om' -printf 'layout %p %s %T@\n'
  } | sort >"$output"
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'taskset=%s\ncpu_affinity_count=%s\n' "$TASKSET_CPUS" "$CPU_AFFINITY_COUNT"
    printf 'model=%s\nlayout_model=%s\ncompile_cache=%s\nlayout_cache=%s\nk10_reference=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$K10_REFERENCE_SUMMARY"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  "$PYTHON_BIN" "$MATERIALIZER" --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" --output-dir "$run_root/representative_128_v1_images"
  om_inventory "$run_root/om_before.txt"

  local output="$run_root/k20/output"
  mkdir -p "$output"
  local command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-execution torchair
    --layout-dtype float16
    --layout-reading-order-dtype float32
    --layout-weight-format torchair_internal
    --layout-depthwise-rewrite constant_grouped
    --layout-preformat-frozen-bn-buffers
    --layout-threshold 0.5
    --layout-cache-dir "$LAYOUT_CACHE_ROOT"
    --input "$run_root/representative_128_v1_images"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers 4
    --warmup-pages 8
    --layout-batch-size 2
    --layout-cpu-threads 16
    --vision-page-lookahead 4
    --vision-bucket-preset 310p_k20_l4
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
    --recognition-preprocess-threads 8
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --decode-batch-size 128
    --compile-cache-dir "$COMPILE_CACHE"
    --stop-after-prefill
    --progress-every-pages 8
    --progress-heartbeat-s 15
  )
  printf '%q ' taskset -c "$TASKSET_CPUS" "${command[@]}" >"$run_root/k20/command.sh"
  printf '\n' >>"$run_root/k20/command.sh"
  printf 'UNIREC_310P_K20_REP128_BEGIN epoch_s=%s\n' "$(date +%s)"
  taskset -c "$TASKSET_CPUS" "${command[@]}" 2>&1 | tee "$run_root/k20/run.log"
  printf 'UNIREC_310P_K20_REP128_END epoch_s=%s\n' "$(date +%s)"

  om_inventory "$run_root/om_after.txt"
  if diff -u "$run_root/om_before.txt" "$run_root/om_after.txt" >"$run_root/hot_om.diff"; then
    printf 'UNIREC_310P_K20_HOT_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_310P_K20_HOT_OM_INVENTORY_CHANGED\n' >&2
    cat "$run_root/hot_om.diff" >&2
    return 1
  fi

  K20_SUMMARY="$output/run_summary.json" RUN_ROOT="$run_root" \
    "$PYTHON_BIN" - <<'PY' | tee "$run_root/final_report.txt"
import json, os
from pathlib import Path

k10 = json.load(open(os.environ["K10_REFERENCE_SUMMARY"]))
k20 = json.load(open(os.environ["K20_SUMMARY"]))
assert k10["status"] == k20["status"] == "ok"
assert k20["execution"] == "production_two_phase_prefill_only"
for key, expected in {
    "page_count": 128,
    "workers": 4,
    "recognition_preprocess_threads": 8,
    "layout_batch_size": 2,
    "layout_cpu_threads": 16,
    "layout_execution": "torchair",
    "layout_dtype": "float16",
    "layout_reading_order_dtype": "float32",
    "layout_threshold": 0.5,
    "vision_page_lookahead": 4,
    "vision_focal_depthwise_rewrite": "constant_grouped_all",
    "vision_weight_format": "torchair_internal",
    "cross_cache_length": 1320,
    "self_cache_length": 2048,
}.items():
    assert k20[key] == expected, (key, k20[key])
    assert k10[key] == expected, ("K10", key, k10[key])
assert k10["vision_bucket_preset"] == "310p_k10_l4_all"
assert k20["vision_bucket_preset"] == "310p_k20_l4"
for key in ("crop_count", "rejected_crop_count", "real_source_tokens"):
    assert k10["retained_bank"][key] == k20["retained_bank"][key], key
assert k20["retained_bank"]["rejected_crop_count"] == 0

def row(run):
    batching = run["prefill_phase_summary"]["vision_batching"]
    stages = run["prefill_phase_summary"]["stage_s"]
    physical_pixels = sum(
        int(rows) * int(key.split("x")[0]) * int(key.split("x")[1].split("_")[0])
        for key, rows in batching["bucket_physical_rows"].items()
    )
    setup = run["prefill_worker_setup_diagnostics"]
    return {
        "wall_s": float(run["timing_s"]["prefill_phase"]),
        "pages_s": float(run["throughput"]["prefill_pages_per_s"]),
        "setup_s": float(run["timing_s"]["prefill_worker_setup"]),
        "crop_count": int(run["retained_bank"]["crop_count"]),
        "vision_calls": sum(int(v) for v in batching["bucket_calls"].values()),
        "physical_pixels": physical_pixels,
        "physical_rows": int(batching["compiled_physical_rows"]),
        "slot_efficiency": float(batching["compiled_slot_efficiency"]),
        "fallback_rows": int(batching["fallback_rows"]),
        "layout_service_s": float(stages["worker_detector_call_sum_s"]),
        "recognition_input_service_s": float(stages["worker_recognition_input_prepare_sum_s"]),
        "recognition_prefill_service_s": float(stages["worker_recognition_prefill_sum_s"]),
        "graph_counts": [int(w["prefix_graph_warmup"]["shape_count"]) for w in setup],
    }

a, b = row(k10), row(k20)
assert b["fallback_rows"] == 0
assert b["graph_counts"] == [20, 20, 20, 20]
report = {
    "schema": "unirec_310p_k20_rep128_w4_v1",
    "k10": a,
    "k20": b,
    "k20_over_k10": {
        "throughput": b["pages_s"] / a["pages_s"],
        "wall": b["wall_s"] / a["wall_s"],
        "recognition_prefill_service": b["recognition_prefill_service_s"] / a["recognition_prefill_service_s"],
        "physical_pixels": b["physical_pixels"] / a["physical_pixels"],
    },
}
out = Path(os.environ["RUN_ROOT"]) / "comparison.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_K20_REP128: PASS")
print("UNIREC_310P_K20_K10 " + json.dumps(a, sort_keys=True))
print("UNIREC_310P_K20_K20 " + json.dumps(b, sort_keys=True))
print("UNIREC_310P_K20_COMPARISON " + json.dumps(report["k20_over_k10"], sort_keys=True))
print(f"UNIREC_310P_K20_COMPARISON_JSON={out}")
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
  printf 'UNIREC_310P_K20_REP128_WORKER_END status=%s run_log=%s\n' "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_k20_rep128_w4_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT/k20"
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" IMAGES_DIR="$IMAGES_DIR" \
    COMPILE_CACHE="$COMPILE_CACHE" LAYOUT_CACHE_ROOT="$LAYOUT_CACHE_ROOT" \
    K10_REFERENCE_SUMMARY="$K10_REFERENCE_SUMMARY" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    TASKSET_CPUS="$TASKSET_CPUS" CPU_AFFINITY_COUNT="$CPU_AFFINITY_COUNT" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
