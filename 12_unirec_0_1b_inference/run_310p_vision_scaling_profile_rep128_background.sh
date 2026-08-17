#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
MATRIX="$SCRIPT_DIR/vision_compile_batch_matrix.py"
PROFILE="$SCRIPT_DIR/profile_prefill_graph_suite.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
PREFILL="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"
REF_910_NATIVE="$SCRIPT_DIR/references/unirec_vision_512x256_batch_scaling_910b_20260817_native.json"
REF_910_OPTIMIZED="$SCRIPT_DIR/references/unirec_vision_512x256_batch_scaling_910b_20260817_optimized.json"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_310P_VISION_REQUIRES_ONE_NPU=%s\n' \
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
  : "${LAYOUT_CACHE_ROOT:?export the warmed optimized-layout B2 cache root}"
  : "${REFERENCE_RUN_SUMMARY:?export the completed optimized-layout/native-vision representative-128 run_summary.json}"
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
  LAYOUT_CACHE_ROOT="$(readlink -f "$LAYOUT_CACHE_ROOT")"
  REFERENCE_RUN_SUMMARY="$(readlink -f "$REFERENCE_RUN_SUMMARY")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -d "$LAYOUT_CACHE_ROOT"
  test -f "$REFERENCE_RUN_SUMMARY"
  test -f "$MATRIX"
  test -f "$PROFILE"
  test -f "$MATERIALIZER"
  test -f "$PREFILL"
  test -f "$MANIFEST"
  test -f "$REF_910_NATIVE"
  test -f "$REF_910_OPTIMIZED"
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
assert p["layout_execution"] == "torchair"
assert p["layout_dtype"] == "float16"
assert p["layout_reading_order_dtype"] == "float32"
assert p["layout_threshold"] == 0.5
assert p["vision_focal_depthwise_rewrite"] == "native"
assert p["vision_weight_format"] == "native"
assert p["cross_cache_length"] == 1320
assert p["self_cache_length"] == 2048
assert p["prefill_trace_enabled"] is False
diag = p["prefill_worker_setup_diagnostics"][0]
assert diag["layout_weight_format"] == "torchair_internal"
assert diag["layout_depthwise_rewrite"] == "constant_grouped"
assert diag["layout_preformat_frozen_bn_buffers"] is True
print(
    "UNIREC_310P_VISION_REP128_REFERENCE: PASS "
    f"path={sys.argv[1]} wall_s={p['timing_s']['prefill_phase']:.6f} "
    f"pages_s={p['throughput']['prefill_pages_per_s']:.6f} "
    f"layout_s={p['prefill_phase_summary']['stage_s']['worker_detector_call_sum_s']:.6f} "
    f"crops={p['retained_bank']['crop_count']}"
)
PY
}

run_matrix() {
  local lane="$1" rewrite="$2" weight_format="$3"
  local output="$RUN_ROOT/matrix_${lane}.json"
  local save_dir="$RUN_ROOT/matrix_${lane}_outputs"
  local command=(
    "$PYTHON_BIN" "$MATRIX"
    --model-path "$MODEL"
    --cache-dir "$COMPILE_CACHE"
    --output "$output"
    --device npu:0
    --width 512
    --height 256
    --batch-sizes 1,4,16
    --warmups 2
    --repeats 20
    --focal-depthwise-rewrite "$rewrite"
    --weight-format "$weight_format"
    --save-compiled-outputs-dir "$save_dir"
  )
  if [[ "$lane" == optimized ]]; then
    command+=(
      --reference-compiled-outputs-dir "$RUN_ROOT/matrix_native_outputs"
    )
  fi
  printf 'UNIREC_310P_VISION_PHASE_BEGIN phase=matrix_%s\n' "$lane"
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command_matrix_${lane}.sh"
  printf '\n' >>"$RUN_ROOT/command_matrix_${lane}.sh"
  "${command[@]}"
  printf 'UNIREC_310P_VISION_PHASE_END phase=matrix_%s\n' "$lane"
}

run_profile() {
  local lane="$1" rewrite="$2" weight_format="$3"
  local output="$RUN_ROOT/profile_${lane}"
  local command=(
    "$PYTHON_BIN" "$PROFILE"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-cache-dir "$LAYOUT_CACHE_ROOT"
    --recognition-cache-dir "$COMPILE_CACHE"
    --output-dir "$output"
    --device npu:0
    --lane vision
    --vision-bucket 512x256_b16
    --vision-depthwise-rewrite "$rewrite"
    --vision-weight-format "$weight_format"
    --warmup 2
    --control-repeats 10
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 100
  )
  printf 'UNIREC_310P_VISION_PHASE_BEGIN phase=profile_%s\n' "$lane"
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command_profile_${lane}.sh"
  printf '\n' >>"$RUN_ROOT/command_profile_${lane}.sh"
  "${command[@]}"
  printf 'UNIREC_310P_VISION_PHASE_END phase=profile_%s\n' "$lane"
}

run_representative_candidate() {
  local input="$RUN_ROOT/representative_128_v1_images"
  local output="$RUN_ROOT/representative128_optimized_vision"
  "$PYTHON_BIN" "$MATERIALIZER" \
    --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$input"
  local command=(
    "$PYTHON_BIN" "$PREFILL"
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
    --layout-batch-size 2
    --vision-page-lookahead 4
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
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
  printf 'UNIREC_310P_VISION_PHASE_BEGIN phase=representative128_optimized\n'
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command_representative128_optimized.sh"
  printf '\n' >>"$RUN_ROOT/command_representative128_optimized.sh"
  "${command[@]}"
  printf 'UNIREC_310P_VISION_PHASE_END phase=representative128_optimized\n'
}

report_results() {
  RUN_ROOT="$RUN_ROOT" \
    REFERENCE_RUN_SUMMARY="$REFERENCE_RUN_SUMMARY" \
    REF_910_NATIVE="$REF_910_NATIVE" \
    REF_910_OPTIMIZED="$REF_910_OPTIMIZED" \
    "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
n310 = json.loads((root / "matrix_native.json").read_text())
o310 = json.loads((root / "matrix_optimized.json").read_text())
n910 = json.loads(Path(os.environ["REF_910_NATIVE"]).read_text())
o910 = json.loads(Path(os.environ["REF_910_OPTIMIZED"]).read_text())
baseline = json.loads(Path(os.environ["REFERENCE_RUN_SUMMARY"]).read_text())
candidate = json.loads(
    (root / "representative128_optimized_vision" / "run_summary.json").read_text()
)

def validate_matrix(p, *, rewrite, weight):
    assert p["status"] == "ok"
    assert p["shape"] == [512, 256]
    assert p["batch_sizes"] == [1, 4, 16]
    assert p["focal_depthwise_rewrite"] == rewrite
    assert p["weight_format"] == weight
    assert p["measurement_scope"].startswith("synchronized NPU events")
    for row in p["rows"]:
        assert all(
            check["allclose_atol_5e_2_rtol_5e_2"]
            for check in row["correctness"].values()
        )

validate_matrix(n310, rewrite="native", weight="native")
validate_matrix(
    o310,
    rewrite="constant_grouped_all",
    weight="torchair_internal",
)
for row in o310["rows"]:
    assert row["cross_lane_reference"]["reference"] != "not_requested"
assert candidate["status"] == "ok"
assert candidate["page_count"] == 128
assert candidate["workers"] == 1
assert candidate["recognition_preprocess_threads"] == 1
assert candidate["layout_batch_size"] == 2
assert candidate["layout_execution"] == "torchair"
assert candidate["vision_focal_depthwise_rewrite"] == "constant_grouped_all"
assert candidate["vision_weight_format"] == "torchair_internal"
assert candidate["retained_bank"]["rejected_crop_count"] == 0
diag = candidate["prefill_worker_setup_diagnostics"][0]
assert diag["vision_focal_depthwise_rewrite_summary"]["rewritten_count"] == 45
assert diag["vision_weight_format_summary"]["requested"] == "torchair_internal"

def matrix_rows(p):
    return {
        int(row["batch_size"]): {
            "median_ms": row["timing"]["compiled"]["median_ms"],
            "crops_per_s": row["timing"]["compiled"]["crops_per_s"],
            "mpix_per_s": (
                int(row["batch_size"])
                * 512
                * 256
                * 1000.0
                / row["timing"]["compiled"]["median_ms"]
                / 1e6
            ),
            "cross_lane_reference": row.get("cross_lane_reference"),
        }
        for row in p["rows"]
    }

n310_rows = matrix_rows(n310)
o310_rows = matrix_rows(o310)
n910_rows = matrix_rows(n910)
o910_rows = matrix_rows(o910)
matrix_comparison = {}
for batch in (1, 4, 16):
    matrix_comparison[str(batch)] = {
        "310p_native": n310_rows[batch],
        "310p_optimized": o310_rows[batch],
        "310p_optimization_speedup": (
            n310_rows[batch]["median_ms"] / o310_rows[batch]["median_ms"]
        ),
        "910b_native": n910_rows[batch],
        "910b_optimized": o910_rows[batch],
        "910b_optimization_speedup": (
            n910_rows[batch]["median_ms"] / o910_rows[batch]["median_ms"]
        ),
        "native_310p_slowdown_vs_910b": (
            n310_rows[batch]["median_ms"] / n910_rows[batch]["median_ms"]
        ),
        "optimized_310p_slowdown_vs_910b": (
            o310_rows[batch]["median_ms"] / o910_rows[batch]["median_ms"]
        ),
    }

def rep_fields(p):
    stage = p["prefill_phase_summary"]["stage_s"]
    return {
        "wall_s": p["timing_s"]["prefill_phase"],
        "pages_s": p["throughput"]["prefill_pages_per_s"],
        "layout_s": stage["worker_detector_call_sum_s"],
        "recognition_prefill_s": stage["worker_recognition_prefill_sum_s"],
        "input_prepare_s": stage["worker_recognition_input_prepare_sum_s"],
        "crop_count": p["retained_bank"]["crop_count"],
        "real_source_tokens": p["retained_bank"]["real_source_tokens"],
    }

b = rep_fields(baseline)
c = rep_fields(candidate)
assert c["crop_count"] == b["crop_count"]
assert c["real_source_tokens"] == b["real_source_tokens"]
report = {
    "schema": "unirec_310p_vision_scaling_profile_rep128_v1",
    "matrix": matrix_comparison,
    "representative128": {
        "native_vision_baseline": b,
        "optimized_vision_candidate": c,
        "prefill_wall_speedup": b["wall_s"] / c["wall_s"],
        "recognition_prefill_speedup": (
            b["recognition_prefill_s"] / c["recognition_prefill_s"]
        ),
        "crop_delta": c["crop_count"] - b["crop_count"],
        "token_delta": c["real_source_tokens"] - b["real_source_tokens"],
    },
    "profile_native": str(root / "profile_native" / "profile_suite_summary.json"),
    "profile_optimized": str(root / "profile_optimized" / "profile_suite_summary.json"),
}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

for batch in (1, 4, 16):
    row = matrix_comparison[str(batch)]
    cross = row["310p_optimized"]["cross_lane_reference"]
    print(
        "UNIREC_310P_VISION_512X256_BATCH "
        f"batch={batch} "
        f"native_ms={row['310p_native']['median_ms']:.6f} "
        f"optimized_ms={row['310p_optimized']['median_ms']:.6f} "
        f"speedup={row['310p_optimization_speedup']:.6f}x "
        f"optimized_crops_s={row['310p_optimized']['crops_per_s']:.6f} "
        f"optimized_mpix_s={row['310p_optimized']['mpix_per_s']:.6f} "
        f"native_slowdown_vs_910b={row['native_310p_slowdown_vs_910b']:.6f}x "
        f"optimized_slowdown_vs_910b={row['optimized_310p_slowdown_vs_910b']:.6f}x "
        f"cross_exact={str(cross['exact']).lower()} "
        f"cross_max_abs={cross['max_abs']:.9g} "
        f"cross_mean_abs={cross['mean_abs']:.9g}"
    )
print(
    "UNIREC_310P_VISION_REP128: PASS "
    f"baseline_wall_s={b['wall_s']:.6f} candidate_wall_s={c['wall_s']:.6f} "
    f"prefill_speedup={b['wall_s'] / c['wall_s']:.6f}x "
    f"baseline_recognition_prefill_s={b['recognition_prefill_s']:.6f} "
    f"candidate_recognition_prefill_s={c['recognition_prefill_s']:.6f} "
    f"recognition_prefill_speedup={b['recognition_prefill_s'] / c['recognition_prefill_s']:.6f}x "
    f"crop_delta={c['crop_count'] - b['crop_count']} "
    f"token_delta={c['real_source_tokens'] - b['real_source_tokens']}"
)
print(f"UNIREC_310P_VISION_OUTPUT {root / 'comparison.json'}")
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
    printf 'model=%s\nlayout_model=%s\ncompile_cache=%s\nlayout_cache=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_matrix native native native
  run_matrix optimized constant_grouped_all torchair_internal
  run_profile native native native
  run_profile optimized constant_grouped_all torchair_internal
  run_representative_candidate
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
  printf 'UNIREC_310P_VISION_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_vision_scaling_profile_rep128_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
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
    LAYOUT_CACHE_ROOT="$LAYOUT_CACHE_ROOT" \
    REFERENCE_RUN_SUMMARY="$REFERENCE_RUN_SUMMARY" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid" 2>/dev/null || test -f "$RUN_ROOT/exit_code.txt"
  printf 'UNIREC_310P_VISION_STARTED pid=%s physical=%s\n' \
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
