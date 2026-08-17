#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"
REFERENCE_910B="$SCRIPT_DIR/references/unirec_representative128_layout_optimized_b2_910b_c3559e3.json"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_310P_REP128_LAYOUT_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated UniRec python_nosym executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${COMPILE_CACHE:?export the warmed production recognition cache parent}"
  : "${REFERENCE_RUN_SUMMARY:?export the completed 310P representative-128 clean baseline run_summary.json}"
  : "${LAYOUT_CACHE_ROOT:?export the unique candidate layout cache root}"
  reject_bad_device

  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  REFERENCE_RUN_SUMMARY="$(readlink -f "$REFERENCE_RUN_SUMMARY")"
  LAYOUT_CACHE_ROOT="$(readlink -m "$LAYOUT_CACHE_ROOT")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -f "$REFERENCE_RUN_SUMMARY"
  test -f "$RUNNER"
  test -f "$MATERIALIZER"
  test -f "$MANIFEST"
  test -f "$REFERENCE_910B"
}

validate_reference_310p() {
  "$PYTHON_BIN" - "$REFERENCE_RUN_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
assert p["status"] == "ok"
assert p["execution"] == "production_two_phase_prefill_only"
assert (p["page_count"], p["workers"]) == (128, 1)
assert p["recognition_preprocess_threads"] == 1
assert p["layout_batch_size"] == 2
assert p["layout_execution"] == "eager"
assert p["layout_dtype"] == "float32"
assert p["layout_reading_order_dtype"] == "float32"
assert p["layout_threshold"] == 0.5
assert p["vision_focal_depthwise_rewrite"] == "native"
assert p["vision_weight_format"] == "native"
assert p["cross_cache_length"] == 1320
assert p["self_cache_length"] == 2048
assert p["prefill_trace_enabled"] is False
print(
    "UNIREC_310P_REP128_LAYOUT_REFERENCE: PASS "
    f"path={sys.argv[1]} "
    f"wall_s={p['timing_s']['prefill_phase']:.6f} "
    f"layout_s={p['prefill_phase_summary']['stage_s']['worker_detector_call_sum_s']:.6f} "
    f"crops={p['retained_bank']['crop_count']}"
)
PY
}

run_candidate() {
  local representative_input="$RUN_ROOT/representative_128_v1_images"
  local output="$RUN_ROOT/output"
  "$PYTHON_BIN" "$MATERIALIZER" \
    --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$representative_input"
  mkdir -p "$output" "$LAYOUT_CACHE_ROOT"

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
    --input "$representative_input"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers 1
    --warmup-pages 8
    --layout-batch-size 2
    --vision-page-lookahead 4
    --vision-focal-depthwise-rewrite native
    --vision-weight-format native
    --recognition-preprocess-threads 1
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
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  "${command[@]}"
}

report_result() {
  RUN_SUMMARY="$RUN_ROOT/output/run_summary.json" \
    REFERENCE_RUN_SUMMARY="$REFERENCE_RUN_SUMMARY" \
    REFERENCE_910B="$REFERENCE_910B" \
    "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

c = json.loads(Path(os.environ["RUN_SUMMARY"]).read_text())
b = json.loads(Path(os.environ["REFERENCE_RUN_SUMMARY"]).read_text())
r = json.loads(Path(os.environ["REFERENCE_910B"]).read_text())

assert c["status"] == "ok"
assert c["execution"] == "production_two_phase_prefill_only"
assert (c["page_count"], c["workers"]) == (128, 1)
assert c["recognition_preprocess_threads"] == 1
assert c["layout_batch_size"] == 2
assert c["layout_execution"] == "torchair"
assert c["layout_dtype"] == "float16"
assert c["layout_reading_order_dtype"] == "float32"
assert c["layout_threshold"] == 0.5
assert c["vision_focal_depthwise_rewrite"] == "native"
assert c["vision_weight_format"] == "native"
assert c["cross_cache_length"] == 1320
assert c["self_cache_length"] == 2048
assert c["prefill_trace_enabled"] is False
assert c["retained_bank"]["rejected_crop_count"] == 0

diag = c["prefill_worker_setup_diagnostics"][0]
assert diag["layout_weight_format"] == "torchair_internal"
assert diag["layout_depthwise_rewrite"] == "constant_grouped"
assert diag["layout_preformat_frozen_bn_buffers"] is True
assert diag["layout_depthwise_rewrite_summary"]["rewritten_count"] == 27
assert diag["layout_frozen_bn_buffer_format_summary"]["converted_count"] == 80

def fields(p):
    summary = p["prefill_phase_summary"]
    layout_s = summary["stage_s"]["worker_detector_call_sum_s"]
    calls = summary["layout_batching"]["calls"]
    return {
        "wall_s": p["timing_s"]["prefill_phase"],
        "pages_s": p["throughput"]["prefill_pages_per_s"],
        "layout_s": layout_s,
        "layout_calls": calls,
        "layout_ms_per_b2": 1000.0 * layout_s / calls,
        "layout_pages_s": p["page_count"] / layout_s,
        "crops": p["retained_bank"]["crop_count"],
        "tokens": p["retained_bank"]["real_source_tokens"],
    }

candidate = fields(c)
baseline = fields(b)
ref910 = r["optimized"]
comparison = {
    "schema": "unirec_310p_representative128_layout_optimized_b2_v1",
    "candidate": candidate,
    "baseline_310p": baseline,
    "reference_910b": ref910,
    "speedup_vs_310p_baseline": {
        "prefill_wall": baseline["wall_s"] / candidate["wall_s"],
        "prefill_throughput": candidate["pages_s"] / baseline["pages_s"],
        "layout_section": baseline["layout_s"] / candidate["layout_s"],
    },
    "slowdown_vs_910b_optimized": {
        "prefill_wall": candidate["wall_s"] / ref910["prefill_wall_s"],
        "layout_section": candidate["layout_s"] / ref910["layout_call_sum_s"],
    },
    "workload_delta_vs_310p_baseline": {
        "crops": candidate["crops"] - baseline["crops"],
        "real_source_tokens": candidate["tokens"] - baseline["tokens"],
    },
}
Path(os.environ["RUN_SUMMARY"]).with_name("layout_optimized_comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n"
)

print(
    "UNIREC_310P_REP128_LAYOUT_OPTIMIZED_B2: PASS "
    f"wall_s={candidate['wall_s']:.6f} "
    f"pages_s={candidate['pages_s']:.6f} "
    f"layout_s={candidate['layout_s']:.6f} "
    f"layout_calls={candidate['layout_calls']} "
    f"layout_ms_per_b2={candidate['layout_ms_per_b2']:.6f} "
    f"layout_pages_s={candidate['layout_pages_s']:.6f} "
    f"crops={candidate['crops']} tokens={candidate['tokens']}"
)
print(
    "UNIREC_310P_REP128_LAYOUT_SPEEDUP: "
    f"prefill_wall={baseline['wall_s'] / candidate['wall_s']:.6f}x "
    f"prefill_throughput={candidate['pages_s'] / baseline['pages_s']:.6f}x "
    f"layout={baseline['layout_s'] / candidate['layout_s']:.6f}x "
    f"crop_delta={candidate['crops'] - baseline['crops']} "
    f"token_delta={candidate['tokens'] - baseline['tokens']}"
)
print(
    "UNIREC_310P_REP128_LAYOUT_VS_910B: "
    f"prefill_slowdown={candidate['wall_s'] / ref910['prefill_wall_s']:.6f}x "
    f"layout_slowdown={candidate['layout_s'] / ref910['layout_call_sum_s']:.6f}x "
    f"reference_910b_wall_s={ref910['prefill_wall_s']:.6f} "
    f"reference_910b_layout_s={ref910['layout_call_sum_s']:.6f}"
)
print(
    "UNIREC_310P_REP128_LAYOUT_OUTPUT "
    + str(Path(os.environ["RUN_SUMMARY"]).with_name("layout_optimized_comparison.json"))
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  validate_reference_310p | tee "$RUN_ROOT/reference_validation.log"
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'model=%s\nlayout_model=%s\nopenocr=%s\nimages=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$OPENOCR_ROOT" "$IMAGES_DIR"
    printf 'recognition_cache=%s\nlayout_cache=%s\nreference_310p=%s\n' \
      "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$REFERENCE_RUN_SUMMARY"
    "$PYTHON_BIN" -c \
      'import sys; sys.path.insert(0, sys.argv[1]); import torch, torch_npu; from opendoc_layout_npu import DEFAULT_LAYOUT_MSDA_IMPLEMENTATION, LAYOUT_COGVIEW_ATTENTION; assert DEFAULT_LAYOUT_MSDA_IMPLEMENTATION == "decomposed"; assert LAYOUT_COGVIEW_ATTENTION == "direct_softmax"; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__); print("layout_msda="+DEFAULT_LAYOUT_MSDA_IMPLEMENTATION); print("layout_attention="+LAYOUT_COGVIEW_ATTENTION)' \
      "$SCRIPT_DIR"
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_candidate
  report_result
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
  printf 'UNIREC_310P_REP128_LAYOUT_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_representative128_layout_optimized_b2_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  test ! -e "$LAYOUT_CACHE_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    COMPILE_CACHE="$COMPILE_CACHE" \
    REFERENCE_RUN_SUMMARY="$REFERENCE_RUN_SUMMARY" \
    LAYOUT_CACHE_ROOT="$LAYOUT_CACHE_ROOT" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid" 2>/dev/null || test -f "$RUN_ROOT/exit_code.txt"
  printf 'UNIREC_310P_REP128_LAYOUT_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
  printf 'EXIT_CODE_FILE=%s\n' "$RUN_ROOT/exit_code.txt"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_entry "$2"
    ;;
  "") launch_main ;;
  *) printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
