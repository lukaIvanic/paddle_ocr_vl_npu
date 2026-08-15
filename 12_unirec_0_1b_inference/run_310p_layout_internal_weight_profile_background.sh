#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAYOUT_LAB="$SCRIPT_DIR/layout_detector_lab.py"
PROFILE_RUNNER="$SCRIPT_DIR/profile_prefill_graph_suite.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${NATIVE_RUN_ROOT:?export the completed 310P four-lane run root}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'LAYOUT_INTERNAL_PROFILE_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  NATIVE_RUN_ROOT="$(readlink -f "$NATIVE_RUN_ROOT")"
  LAYOUT_CACHE="$(readlink -m "$LAYOUT_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$LAYOUT_MODEL/model.safetensors"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$NATIVE_RUN_ROOT"
  test -f "$NATIVE_RUN_ROOT/exit_code.txt"
  test "$(tr -d '[:space:]' <"$NATIVE_RUN_ROOT/exit_code.txt")" = 0
  test -f "$LAYOUT_LAB"
  test -f "$PROFILE_RUNNER"

  NATIVE_FORWARD="$NATIVE_RUN_ROOT/forward_compiled_fp16_body_fp32_ro.json"
  NATIVE_PROFILE="$NATIVE_RUN_ROOT/profile_compiled_fp16_body_fp32_ro/profile_suite_summary.json"
  test -f "$NATIVE_FORWARD"
  test -f "$NATIVE_PROFILE"

  local page_name
  page_name="jiaocaineedrop_jiaocai_needrop_en_620.jpg"
  LAYOUT_INPUT_IMAGE="$(find "$IMAGES_DIR" -type f -name "$page_name" -print -quit)"
  test -f "$LAYOUT_INPUT_IMAGE"
}

run_phase() {
  local phase="$1"
  local phase_log="$2"
  shift 2
  printf 'UNIREC_LAYOUT_INTERNAL_PHASE_BEGIN phase=%s command=' "$phase"
  printf '%q ' "$@"
  printf '\n'
  local started="$SECONDS"
  local status=0
  set +e
  "$@" > >(tee "$phase_log") 2>&1
  status="$?"
  set -e
  printf 'UNIREC_LAYOUT_INTERNAL_PHASE_END phase=%s status=%s wall_s=%s\n' \
    "$phase" "$status" "$((SECONDS - started))"
  return "$status"
}

run_forward() {
  local output="$RUN_ROOT/forward_internal.json"
  local command=(
    "$PYTHON_BIN" "$LAYOUT_LAB"
    --contract custom
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$LAYOUT_MODEL"
    --input "$LAYOUT_INPUT_IMAGE"
    --output "$output"
    --device npu:0
    --execution torchair
    --compile-cache-dir "$LAYOUT_CACHE"
    --dtype float16
    --reading-order-dtype float32
    --threshold 0.5
    --weight-format torchair_internal
    --depthwise-rewrite native
    --input-color-order rgb
    --limit 1
    --warmup-pages 1
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/forward_command.sh"
  printf '\n' >>"$RUN_ROOT/forward_command.sh"
  run_phase forward "$RUN_ROOT/forward.log" "${command[@]}"
}

run_profile() {
  local output="$RUN_ROOT/profile_internal"
  local command=(
    "$PYTHON_BIN" "$PROFILE_RUNNER"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-input-image "$LAYOUT_INPUT_IMAGE"
    --layout-cache-dir "$LAYOUT_CACHE"
    --recognition-cache-dir "$LAYOUT_CACHE/unused_recognition"
    --output-dir "$output"
    --device npu:0
    --lane layout
    --layout-execution torchair
    --layout-dtype float16
    --layout-reading-order-dtype float32
    --layout-depthwise-rewrite native
    --layout-weight-format torchair_internal
    --warmup 2
    --control-repeats 20
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 120
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/profile_command.sh"
  printf '\n' >>"$RUN_ROOT/profile_command.sh"
  run_phase profile "$RUN_ROOT/profile.log" "${command[@]}"
}

summarize() {
  RUN_ROOT="$RUN_ROOT" \
  NATIVE_FORWARD="$NATIVE_FORWARD" \
  NATIVE_PROFILE="$NATIVE_PROFILE" \
  "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
native_forward = json.loads(Path(os.environ["NATIVE_FORWARD"]).read_text())
internal_forward = json.loads((root / "forward_internal.json").read_text())
native_profile = json.loads(Path(os.environ["NATIVE_PROFILE"]).read_text())
internal_profile_path = root / "profile_internal" / "profile_suite_summary.json"
internal_profile = json.loads(internal_profile_path.read_text())

native_page = native_forward["pages"][0]
internal_page = internal_forward["pages"][0]
native_boxes = native_page["result"]["boxes"]
internal_boxes = internal_page["result"]["boxes"]

def paired_iou(first, second):
    ax1, ay1, ax2, ay2 = first["coordinate"]
    bx1, by1, bx2, by2 = second["coordinate"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 1.0

same_count = len(native_boxes) == len(internal_boxes)
paired = list(zip(native_boxes, internal_boxes)) if same_count else []
class_label_match = same_count and all(
    (first["cls_id"], first["label"]) == (second["cls_id"], second["label"])
    for first, second in paired
)
order_changed = sum(
    first["custom_value"] != second["custom_value"] for first, second in paired
)
mean_iou = (
    sum(paired_iou(first, second) for first, second in paired) / len(paired)
    if paired else None
)
coordinate_max_abs = max(
    (
        abs(first_value - second_value)
        for first, second in paired
        for first_value, second_value in zip(
            first["coordinate"], second["coordinate"]
        )
    ),
    default=None,
)
score_max_abs = max(
    (abs(first["score"] - second["score"]) for first, second in paired),
    default=None,
)

def profile_fields(profile):
    lane = profile["lanes"][0]
    parsed = lane["parsed_profile"]["summary"]["runs"][0]
    ops = {
        row["op_type"]: row for row in parsed["op_statistic"]["top_op_types"]
    }
    transdata = ops["TransData"]
    rows = parsed["kernel_details"]["top_transdata_shape_signatures"]
    nchw_fz = [row for row in rows if "NCHW -> FRACTAL_Z" in row["name"]]
    depthwise = [
        row for row in nchw_fz
        if len(row["input_shape_samples"]) == 1
        and len([
            value for value in row["input_shape_samples"][0].strip('"').split(',')
        ]) == 4
        and row["input_shape_samples"][0].strip('"').split(',')[1] == "1"
    ]
    regular = [row for row in nchw_fz if row not in depthwise]
    return {
        "clean_mean_ms": float(lane["control_before"]["device_event"]["mean_ms"]),
        "clean_median_ms": float(lane["control_before"]["device_event"]["median_ms"]),
        "transdata_count": int(transdata["count"]),
        "transdata_ms": float(transdata["total_time_us"]) / 1000.0,
        "nchw_fz_count": sum(int(row["count"]) for row in nchw_fz),
        "nchw_fz_ms": sum(float(row["duration_us"]) for row in nchw_fz) / 1000.0,
        "regular_nchw_fz_count": sum(int(row["count"]) for row in regular),
        "regular_nchw_fz_ms": sum(float(row["duration_us"]) for row in regular) / 1000.0,
        "depthwise_nchw_fz_count": sum(int(row["count"]) for row in depthwise),
        "depthwise_nchw_fz_ms": sum(float(row["duration_us"]) for row in depthwise) / 1000.0,
        "nchw_fz_rows": [
            {"name": row["name"], "count": row["count"], "duration_us": row["duration_us"]}
            for row in nchw_fz
        ],
    }

native = profile_fields(native_profile)
internal = profile_fields(internal_profile)
summary = {
    "format": "unirec_310p_layout_internal_weight_profile_v1",
    "native": native,
    "internal": internal,
    "speedup": native["clean_mean_ms"] / internal["clean_mean_ms"],
    "output_gate": {
        "same_box_count": same_count,
        "native_box_count": len(native_boxes),
        "internal_box_count": len(internal_boxes),
        "class_label_sequence_match": class_label_match,
        "reading_order_changed_count": order_changed,
        "mean_paired_iou": mean_iou,
        "coordinate_max_abs_px": coordinate_max_abs,
        "score_max_abs": score_max_abs,
        "digest_match": native_page["result_digest"] == internal_page["result_digest"],
    },
    "artifacts": {
        "native_forward": os.environ["NATIVE_FORWARD"],
        "native_profile": os.environ["NATIVE_PROFILE"],
        "internal_forward": str(root / "forward_internal.json"),
        "internal_profile": str(internal_profile_path),
    },
}
(root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(
    "UNIREC_310P_LAYOUT_INTERNAL_WEIGHT: PASS "
    f"native_ms={native['clean_mean_ms']:.6f} "
    f"internal_ms={internal['clean_mean_ms']:.6f} "
    f"speedup={summary['speedup']:.6f} "
    f"native_td={native['transdata_count']}/{native['transdata_ms']:.6f}ms "
    f"internal_td={internal['transdata_count']}/{internal['transdata_ms']:.6f}ms "
    f"native_regular_conv_fz={native['regular_nchw_fz_count']}/{native['regular_nchw_fz_ms']:.6f}ms "
    f"internal_regular_conv_fz={internal['regular_nchw_fz_count']}/{internal['regular_nchw_fz_ms']:.6f}ms "
    f"native_depthwise_weight_repack={native['depthwise_nchw_fz_count']}/{native['depthwise_nchw_fz_ms']:.6f}ms "
    f"internal_depthwise_weight_repack={internal['depthwise_nchw_fz_count']}/{internal['depthwise_nchw_fz_ms']:.6f}ms "
    f"boxes={len(native_boxes)}/{len(internal_boxes)} "
    f"class_label_match={str(class_label_match).lower()} "
    f"order_changed={order_changed} "
    f"mean_iou={mean_iou} coord_max={coordinate_max_abs} "
    f"score_max={score_max_abs} digest_match={str(summary['output_gate']['digest_match']).lower()}"
)
print(f"UNIREC_310P_LAYOUT_INTERNAL_WEIGHT_OUTPUT {root / 'comparison_summary.json'}")
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1

  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'native_run_root=%s\nlayout_cache=%s\n' \
      "$NATIVE_RUN_ROOT" "$LAYOUT_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  mkdir -p "$LAYOUT_CACHE/unused_recognition"
  run_forward
  run_profile
  summarize
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1"
  local status=0
  local started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_LAYOUT_INTERNAL_WEIGHT_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${NATIVE_RUN_ROOT:?export the completed 310P four-lane run root}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_internal_weight_${commit_short}_${timestamp}}"
  LAYOUT_CACHE="${LAYOUT_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_layout_internal_weight_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  LAYOUT_CACHE="$(realpath -m "$LAYOUT_CACHE")"
  test ! -e "$RUN_ROOT"
  test ! -e "$LAYOUT_CACHE"
  mkdir -p "$RUN_ROOT" "$LAYOUT_CACHE"
  resolve_inputs

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    NATIVE_RUN_ROOT="$NATIVE_RUN_ROOT" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_LAYOUT_INTERNAL_WEIGHT_STARTED pid=%s physical=%s\n' \
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
  --summarize)
    test "$#" -eq 2
    : "${PYTHON_BIN:?export the UniRec Python interpreter}"
    : "${NATIVE_RUN_ROOT:?export the completed 310P four-lane run root}"
    RUN_ROOT="$(realpath "$2")"
    LAYOUT_CACHE="${LAYOUT_CACHE:-$RUN_ROOT/unused_cache}"
    resolve_inputs
    summarize
    ;;
  "") launch_main ;;
  *)
    printf 'usage: %s [--worker RUN_ROOT | --summarize RUN_ROOT]\n' "$0" >&2
    exit 2
    ;;
esac
