#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"
CHIP_LABEL="${UNIREC_K10_CHIP_LABEL:-910B2}"
ALLOWED_DEVICES="${UNIREC_K10_ALLOWED_DEVICES:-}"
RECOGNITION_THREADS="${UNIREC_K10_THREADS:-1}"

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec inference Python}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the recognition compile-cache parent}"
  : "${LAYOUT_CACHE_ROOT:?export the layout compile-cache directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical NPU}"
  case "$RECOGNITION_THREADS" in
    ''|*[!0-9]*|0) printf 'UNIREC_K10_INVALID_THREADS=%s\n' "$RECOGNITION_THREADS" >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_K10_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  if [[ -n "$ALLOWED_DEVICES" ]]; then
    case ",${ALLOWED_DEVICES}," in
      *,${ASCEND_RT_VISIBLE_DEVICES},*) ;;
      *)
        printf 'UNIREC_K10_REJECTED_DEVICE chip=%s device=%s allowed=%s\n' \
          "$CHIP_LABEL" "$ASCEND_RT_VISIBLE_DEVICES" "$ALLOWED_DEVICES" >&2
        exit 1
        ;;
    esac
  else
    case ",${ASCEND_RT_VISIBLE_DEVICES}," in
      *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
    esac
  fi
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
}

run_lane() {
  local lane="$1" input="$RUN_ROOT/representative_128_v1_images"
  local output="$RUN_ROOT/$lane/output"
  local graph_diagnostic=0
  if [[ "$lane" == trace ]]; then
    graph_diagnostic=1
  fi
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
    --input "$input"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers 1
    --warmup-pages 8
    --layout-batch-size 1
    --vision-page-lookahead 1
    --vision-bucket-preset 310p_k10_l1
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
    --recognition-preprocess-threads "$RECOGNITION_THREADS"
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --decode-batch-size 128
    --compile-cache-dir "$COMPILE_CACHE"
    --stop-after-prefill
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  if [[ "$lane" == trace ]]; then
    command+=(--prefill-trace)
  fi
  printf '%q ' "${command[@]}" >"$RUN_ROOT/$lane/command.sh"
  printf '\n' >>"$RUN_ROOT/$lane/command.sh"
  printf 'UNIREC_K10_LANE_BEGIN lane=%s\n' "$lane"
  UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG="$graph_diagnostic" \
    "${command[@]}" 2>&1 | tee "$RUN_ROOT/$lane/run.log"
  printf 'UNIREC_K10_LANE_END lane=%s\n' "$lane"
}

report_clean_only() {
  CLEAN_SUMMARY="$RUN_ROOT/clean/output/run_summary.json" \
  CHIP_LABEL="$CHIP_LABEL" \
  EXPECTED_THREADS="$RECOGNITION_THREADS" \
    "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["CLEAN_SUMMARY"]).read_text())
assert run["status"] == "ok"
assert run["execution"] == "production_two_phase_prefill_only"
assert (run["page_count"], run["workers"]) == (128, 1)
assert run["recognition_preprocess_threads"] == int(os.environ["EXPECTED_THREADS"])
assert run["layout_batch_size"] == 1
assert run["vision_page_lookahead"] == 1
assert run["vision_bucket_preset"] == "310p_k10_l1"
assert run["prefill_trace_enabled"] is False
prefix = "UNIREC_" + os.environ["CHIP_LABEL"].upper().replace("-", "_") + "_K10_L1"
cache_inventory = {}
graph_rows = run["prefill_worker_setup_diagnostics"][0][
    "prefix_graph_warmup"
]["graphs"]
fallback_warmup = run["prefill_worker_setup_diagnostics"][0][
    "prefix_graph_warmup"
]["fallback_eager"]
assert len(fallback_warmup["pass_wall_s"]) == 2
for bucket, row in graph_rows.items():
    cache_dir = Path(row["cache_dir"])
    compiled_modules = sorted(cache_dir.rglob("compiled_module"))
    om_files = sorted(cache_dir.rglob("*.om"))
    assert len(compiled_modules) == 1, (bucket, compiled_modules)
    assert len(om_files) == 1, (bucket, om_files)
    cache_inventory[bucket] = {
        "compiled_module_count": 1,
        "compiled_module_bytes": compiled_modules[0].stat().st_size,
        "om_count": 1,
        "om_bytes": om_files[0].stat().st_size,
    }
print(
    f"{prefix}_CLEAN_ONLY_RESULT PASS "
    f"clean_wall_s={run['timing_s']['prefill_phase']:.6f} "
    f"clean_pages_s={run['throughput']['prefill_pages_per_s']:.6f} "
    f"threads={run['recognition_preprocess_threads']} "
    f"clean_layout_s={run['prefill_phase_summary']['stage_s']['worker_detector_call_sum_s']:.6f} "
    f"crops={run['retained_bank']['crop_count']} "
    f"real_source_tokens={run['retained_bank']['real_source_tokens']}"
)
print(
    f"{prefix}_FALLBACK_WARMUP "
    f"input_shape={fallback_warmup['input_shape']} "
    f"cold_first_use_s={fallback_warmup['cold_first_use_wall_s']:.6f} "
    f"warm_replay_s={fallback_warmup['warm_replay_wall_s']:.6f}"
)
print(f"{prefix}_CACHE " + json.dumps(cache_inventory, sort_keys=True))
PY
}

report_result() {
  TRACE_SUMMARY="$RUN_ROOT/trace/output/run_summary.json" \
  TRACE_DISTRIBUTIONS="$RUN_ROOT/trace/output/prefill_distributions.json" \
  TRACE_ITERATIONS="$RUN_ROOT/trace/output/prefill_iterations.jsonl" \
  CLEAN_SUMMARY="$RUN_ROOT/clean/output/run_summary.json" \
  CHIP_LABEL="$CHIP_LABEL" \
  EXPECTED_THREADS="$RECOGNITION_THREADS" \
    "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

trace_run = json.loads(Path(os.environ["TRACE_SUMMARY"]).read_text())
clean_run = json.loads(Path(os.environ["CLEAN_SUMMARY"]).read_text())
distributions = json.loads(Path(os.environ["TRACE_DISTRIBUTIONS"]).read_text())
expected = {
    "448x64_b4", "448x256_b2", "448x384_b2", "512x128_b4",
    "960x64_b2", "960x64_b4", "960x128_b1", "960x256_b1",
    "960x384_b1", "960x512_b1",
}
for run, traced in ((trace_run, True), (clean_run, False)):
    assert run["status"] == "ok"
    assert run["execution"] == "production_two_phase_prefill_only"
    assert (run["page_count"], run["workers"]) == (128, 1)
    assert run["recognition_preprocess_threads"] == int(os.environ["EXPECTED_THREADS"])
    assert run["layout_batch_size"] == 1
    assert run["vision_page_lookahead"] == 1
    assert run["vision_bucket_preset"] == "310p_k10_l1"
    assert run["vision_focal_depthwise_rewrite"] == "constant_grouped_all"
    assert run["vision_weight_format"] == "torchair_internal"
    assert run["prefill_trace_enabled"] is traced
assert trace_run["retained_bank"]["crop_count"] == clean_run["retained_bank"]["crop_count"]
assert trace_run["retained_bank"]["real_source_tokens"] == clean_run["retained_bank"]["real_source_tokens"]

bucket_graph = distributions["stage_distributions"][
    "vision_bucket_call.device_stage_s.graph_s"
]
fallback_graph = distributions["stage_distributions"][
    "vision_fallback_call.device_stage_s.graph_s"
]
calls = real_rows = physical_rows = effective_pixels = physical_pixels = 0
bucket_calls = {}
chip_label = os.environ["CHIP_LABEL"]
prefix = "UNIREC_" + chip_label.upper().replace("-", "_") + "_K10_L1"
cache_inventory = {}
fallback_warmups = {}
graph_rows = trace_run["prefill_worker_setup_diagnostics"][0][
    "prefix_graph_warmup"
]["graphs"]
for bucket, row in graph_rows.items():
    cache_dir = Path(row["cache_dir"])
    compiled_modules = sorted(cache_dir.rglob("compiled_module"))
    om_files = sorted(cache_dir.rglob("*.om"))
    assert len(compiled_modules) == 1, (bucket, compiled_modules)
    assert len(om_files) == 1, (bucket, om_files)
    cache_inventory[bucket] = {
        "compiled_module_count": 1,
        "compiled_module_bytes": compiled_modules[0].stat().st_size,
        "om_count": 1,
        "om_bytes": om_files[0].stat().st_size,
    }
for lane, run in (("trace", trace_run), ("clean", clean_run)):
    fallback = run["prefill_worker_setup_diagnostics"][0][
        "prefix_graph_warmup"
    ]["fallback_eager"]
    assert len(fallback["pass_wall_s"]) == 2
    fallback_warmups[lane] = fallback
with Path(os.environ["TRACE_ITERATIONS"]).open() as handle:
    for line in handle:
        event = json.loads(line)
        if event.get("event") != "vision_bucket_call":
            continue
        assert event["bucket"] in expected
        calls += 1
        real_rows += int(event["real_rows"])
        physical_rows += int(event["physical_rows"])
        shape = event["physical_input_shape"]
        physical_pixels += int(shape[0]) * int(shape[2]) * int(shape[3])
        effective_pixels += sum(
            int(member["processed_image_size"][0])
            * int(member["processed_image_size"][1])
            for member in event["members"]
        )
        bucket_calls[event["bucket"]] = bucket_calls.get(event["bucket"], 0) + 1
assert set(bucket_calls) == expected
print(
    f"{prefix}_RESULT PASS "
    f"clean_wall_s={clean_run['timing_s']['prefill_phase']:.6f} "
    f"clean_pages_s={clean_run['throughput']['prefill_pages_per_s']:.6f} "
    f"trace_wall_s={trace_run['timing_s']['prefill_phase']:.6f} "
    f"threads={trace_run['recognition_preprocess_threads']} "
    f"bucket_graph_s={bucket_graph['sum_s']:.6f} "
    f"fallback_graph_s={fallback_graph['sum_s']:.6f} "
    f"vision_graph_s={bucket_graph['sum_s'] + fallback_graph['sum_s']:.6f} "
    f"bucket_calls={calls} slot_eff={real_rows / physical_rows:.6f} "
    f"pixel_eff={effective_pixels / physical_pixels:.6f} "
    f"crops={clean_run['retained_bank']['crop_count']}"
)
print(f"{prefix}_BUCKET_CALLS " + json.dumps(bucket_calls, sort_keys=True))
stage_rows = distributions["stage_distributions"]
text_device_rows = [
    row
    for name, row in stage_rows.items()
    if name.startswith(
        "text_prefill_pack.device_stage_s.compiled_packed_text_prefill_s"
    )
]
assert len(text_device_rows) == 1
print(
    f"{prefix}_STAGES "
    f"layout_s={stage_rows['layout_batch_call.wall_s']['sum_s']:.6f} "
    f"layout_model_forward_s={stage_rows['layout_batch_call.stage_s.model_forward_s']['sum_s']:.6f} "
    f"layout_model_forward_mean_ms={stage_rows['layout_batch_call.stage_s.model_forward_s']['mean_ms']:.6f} "
    f"layout_processor_s={stage_rows['layout_batch_call.stage_s.processor_preprocess_s']['sum_s']:.6f} "
    f"vision_bucket_s={bucket_graph['sum_s']:.6f} "
    f"vision_fallback_s={fallback_graph['sum_s']:.6f} "
    f"vision_total_s={bucket_graph['sum_s'] + fallback_graph['sum_s']:.6f} "
    f"crop_preprocess_s={stage_rows['recognition_crop_preprocess.wall_s']['sum_s']:.6f} "
    f"text_pack_wall_s={stage_rows['text_prefill_pack.wall_s']['sum_s']:.6f} "
    f"text_device_s={text_device_rows[0]['sum_s']:.6f} "
    f"shared_pack_s={stage_rows['page_shared_pack.wall_s']['sum_s']:.6f}"
)
print(
    f"{prefix}_FALLBACK_WARMUP "
    f"input_shape={fallback_warmups['clean']['input_shape']} "
    f"trace_cold_first_use_s={fallback_warmups['trace']['cold_first_use_wall_s']:.6f} "
    f"trace_warm_replay_s={fallback_warmups['trace']['warm_replay_wall_s']:.6f} "
    f"clean_cold_first_use_s={fallback_warmups['clean']['cold_first_use_wall_s']:.6f} "
    f"clean_warm_replay_s={fallback_warmups['clean']['warm_replay_wall_s']:.6f}"
)
print(f"{prefix}_CACHE " + json.dumps(cache_inventory, sort_keys=True))
print(
    f"{prefix}_LAYOUT "
    f"clean_layout_s={clean_run['prefill_phase_summary']['stage_s']['worker_detector_call_sum_s']:.6f} "
    f"trace_layout_s={trace_run['prefill_phase_summary']['stage_s']['worker_detector_call_sum_s']:.6f}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  mkdir -p "$RUN_ROOT/trace" "$RUN_ROOT/clean"
  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
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
  case "${UNIREC_K10_RUN_MODE:-both}" in
    both)
      run_lane trace
      run_lane clean
      report_result
      ;;
    clean_only)
      run_lane clean
      report_clean_only
      ;;
    *)
      printf 'UNIREC_K10_INVALID_RUN_MODE=%s\n' "$UNIREC_K10_RUN_MODE" >&2
      return 1
      ;;
  esac
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
  printf 'UNIREC_910B_K10_L1_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/910b_rep128_k10_l1_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
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
    UNIREC_K10_CHIP_LABEL="$CHIP_LABEL" \
    UNIREC_K10_ALLOWED_DEVICES="$ALLOWED_DEVICES" \
    UNIREC_K10_RUN_MODE="${UNIREC_K10_RUN_MODE:-both}" \
    UNIREC_K10_THREADS="$RECOGNITION_THREADS" \
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
