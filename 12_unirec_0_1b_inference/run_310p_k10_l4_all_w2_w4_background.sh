#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated python_nosym executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the warmed recognition cache parent}"
  : "${LAYOUT_CACHE_ROOT:?export the warmed optimized layout cache}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device}"
  : "${TASKSET_CPUS:?export the known-good 64-CPU taskset mask}"

  case "$ASCEND_RT_VISIBLE_DEVICES" in
    0|1|2|3) ;;
    *)
      printf 'UNIREC_310P_SCALING_INVALID_DEVICE=%s expected=0-3\n' \
        "$ASCEND_RT_VISIBLE_DEVICES" >&2
      exit 1
      ;;
  esac
  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  mkdir -p "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  LAYOUT_CACHE_ROOT="$(readlink -f "$LAYOUT_CACHE_ROOT")"
  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR
  export COMPILE_CACHE LAYOUT_CACHE_ROOT

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -f "$RUNNER"
  test -f "$MATERIALIZER"
  test -f "$MANIFEST"
  CPU_AFFINITY_COUNT="$({
    taskset -c "$TASKSET_CPUS" "$PYTHON_BIN" -c \
      'import os; print(len(os.sched_getaffinity(0)))'
  })"
  if (( CPU_AFFINITY_COUNT < 64 )); then
    printf 'UNIREC_310P_SCALING_CPU_AFFINITY_TOO_SMALL count=%s mask=%s\n' \
      "$CPU_AFFINITY_COUNT" "$TASKSET_CPUS" >&2
    exit 1
  fi
  export CPU_AFFINITY_COUNT
}

run_lane() {
  local lane="$1" workers="$2"
  local output="$RUN_ROOT/$lane/output"
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
    --input "$RUN_ROOT/representative_128_v1_images"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers "$workers"
    --warmup-pages 8
    --layout-batch-size 2
    --layout-cpu-threads 16
    --vision-page-lookahead 4
    --vision-bucket-preset 310p_k10_l4_all
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
    --progress-every-pages 16
    --progress-heartbeat-s 15
  )
  printf '%q ' taskset -c "$TASKSET_CPUS" "${command[@]}" \
    >"$RUN_ROOT/$lane/command.sh"
  printf '\n' >>"$RUN_ROOT/$lane/command.sh"
  printf 'UNIREC_310P_K10_L4_ALL_SCALING_LANE_BEGIN lane=%s workers=%s\n' \
    "$lane" "$workers"
  taskset -c "$TASKSET_CPUS" "${command[@]}" 2>&1 \
    | tee "$RUN_ROOT/$lane/run.log"
  printf 'UNIREC_310P_K10_L4_ALL_SCALING_LANE_END lane=%s workers=%s\n' \
    "$lane" "$workers"
}

report_results() {
  RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
rows = []
expected_buckets = {
    "448x192_b2",
    "448x384_b2",
    "512x64_b4",
    "960x64_b4",
    "960x128_b2",
    "960x256_b1",
    "960x448_b1",
    "960x576_b1",
    "960x896_b1",
    "960x1408_b1",
}
for lane, workers in (("w2", 2), ("w4", 4)):
    path = root / lane / "output" / "run_summary.json"
    run = json.loads(path.read_text())
    assert run["status"] == "ok"
    assert run["execution"] == "production_two_phase_prefill_only"
    assert (run["page_count"], run["workers"]) == (128, workers)
    assert run["recognition_preprocess_threads"] == 8
    assert run["layout_batch_size"] == 2
    assert run["layout_cpu_threads"] == 16
    assert run["layout_execution"] == "torchair"
    assert run["layout_dtype"] == "float16"
    assert run["layout_reading_order_dtype"] == "float32"
    assert run["layout_threshold"] == 0.5
    assert run["vision_page_lookahead"] == 4
    assert run["vision_bucket_preset"] == "310p_k10_l4_all"
    assert run["vision_focal_depthwise_rewrite"] == "constant_grouped_all"
    assert run["vision_weight_format"] == "torchair_internal"
    assert run["cross_cache_length"] == 1320
    assert run["self_cache_length"] == 2048
    assert run["prefill_trace_enabled"] is False
    assert run["retained_bank"]["rejected_crop_count"] == 0
    batching = run["prefill_phase_summary"]["vision_batching"]
    assert batching["fallback_rows"] == 0
    assert set(batching["bucket_calls"]) == expected_buckets
    setup = run["prefill_worker_setup_diagnostics"]
    assert len(setup) == workers
    for worker in setup:
        assert worker["cpu_affinity_count"] >= 64
        assert worker["recognition_preprocess_threads"] == 8
        assert worker["layout_cpu_threads"] == 16
        warmup = worker["prefix_graph_warmup"]
        assert set(warmup["graphs"]) == expected_buckets
        assert warmup["fallback_eager"]["execution"] == (
            "skipped_compiled_bucket_coverage"
        )
    stages = run["prefill_phase_summary"]["stage_s"]
    calls = sum(int(value) for value in batching["bucket_calls"].values())
    row = {
        "lane": lane,
        "workers": workers,
        "wall_s": float(run["timing_s"]["prefill_phase"]),
        "pages_per_s": float(run["throughput"]["prefill_pages_per_s"]),
        "setup_s": float(run["timing_s"]["prefill_worker_setup"]),
        "crop_count": int(run["retained_bank"]["crop_count"]),
        "fallback_rows": int(batching["fallback_rows"]),
        "vision_calls": calls,
        "layout_service_s": float(stages["worker_detector_call_sum_s"]),
        "recognition_input_service_s": float(
            stages["worker_recognition_input_prepare_sum_s"]
        ),
        "recognition_prefill_service_s": float(
            stages["worker_recognition_prefill_sum_s"]
        ),
        "shared_pack_service_s": float(stages["worker_shared_pack_sum_s"]),
        "rgb_decode_service_s": float(stages["worker_direct_rgb_decode_sum_s"]),
        "bucket_calls": batching["bucket_calls"],
        "worker_cpu_affinity_counts": [
            int(worker["cpu_affinity_count"]) for worker in setup
        ],
    }
    rows.append(row)
    print(
        "UNIREC_310P_K10_L4_ALL_SCALING_RESULT "
        f"lane={lane} workers={workers} wall_s={row['wall_s']:.6f} "
        f"pages_s={row['pages_per_s']:.6f} setup_s={row['setup_s']:.6f} "
        f"crops={row['crop_count']} vision_calls={calls} fallback_rows=0 "
        f"layout_service_s={row['layout_service_s']:.6f} "
        f"recognition_input_service_s={row['recognition_input_service_s']:.6f} "
        f"recognition_prefill_service_s={row['recognition_prefill_service_s']:.6f} "
        f"affinity_counts={row['worker_cpu_affinity_counts']}"
    )

w2, w4 = rows
comparison = {
    "schema": "unirec_310p_k10_l4_all_w2_w4_scaling_v1",
    "rows": rows,
    "w4_over_w2": {
        "throughput": w4["pages_per_s"] / w2["pages_per_s"],
        "wall": w4["wall_s"] / w2["wall_s"],
        "additional_worker_scaling_efficiency": (
            w4["pages_per_s"] / w2["pages_per_s"] / 2.0
        ),
    },
}
output = root / "w2_w4_scaling.json"
output.write_text(json.dumps(comparison, indent=2) + "\n")
print(
    "UNIREC_310P_K10_L4_ALL_SCALING_COMPARISON "
    f"w2_pages_s={w2['pages_per_s']:.6f} "
    f"w4_pages_s={w4['pages_per_s']:.6f} "
    f"w4_over_w2={w4['pages_per_s'] / w2['pages_per_s']:.6f}x "
    f"output={output}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'taskset=%s\ncpu_affinity_count=%s\n' \
      "$TASKSET_CPUS" "$CPU_AFFINITY_COUNT"
    printf 'model=%s\nlayout_model=%s\ncompile_cache=%s\nlayout_cache=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  "$PYTHON_BIN" "$MATERIALIZER" \
    --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$RUN_ROOT/representative_128_v1_images"
  run_lane w2 2
  run_lane w4 4
  report_results
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_K10_L4_ALL_W2_W4_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_k10_l4_all_w2_w4_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT/w2" "$RUN_ROOT/w4"
  nohup env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    COMPILE_CACHE="$COMPILE_CACHE" \
    LAYOUT_CACHE_ROOT="$LAYOUT_CACHE_ROOT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    TASKSET_CPUS="$TASKSET_CPUS" \
    CPU_AFFINITY_COUNT="$CPU_AFFINITY_COUNT" \
    bash "$0" --worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_entry "${2:?worker run root required}"
else
  launch_main
fi
