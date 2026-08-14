#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROFILE_RUNNER="$SCRIPT_DIR/profile_prefill_graph_suite.py"
LAYOUT_LAB="$SCRIPT_DIR/layout_detector_lab.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${LAYOUT_CACHE:?export a new layout compile-cache directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'LAYOUT_PRECISION_PROFILE_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
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
  LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$LAYOUT_MODEL/model.safetensors"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -f "$PROFILE_RUNNER"
  test -f "$LAYOUT_LAB"

  local page_name
  page_name="jiaocaineedrop_jiaocai_needrop_en_620.jpg"
  if [[ -n "${LAYOUT_INPUT_IMAGE:-}" ]]; then
    LAYOUT_INPUT_IMAGE="$(readlink -f "$LAYOUT_INPUT_IMAGE")"
  else
    LAYOUT_INPUT_IMAGE="$(find "$IMAGES_DIR" -type f -name "$page_name" -print -quit)"
  fi
  test -f "$LAYOUT_INPUT_IMAGE"
}

run_phase() {
  local phase="$1"
  local phase_log="$2"
  shift 2
  printf 'UNIREC_LAYOUT_4LANE_PHASE_BEGIN phase=%s command=' "$phase"
  printf '%q ' "$@"
  printf '\n'
  local status=0
  local started="$SECONDS"
  set +e
  "$@" > >(tee "$phase_log") 2>&1
  status="$?"
  set -e
  printf 'UNIREC_LAYOUT_4LANE_PHASE_END phase=%s status=%s wall_s=%s\n' \
    "$phase" "$status" "$((SECONDS - started))"
  return "$status"
}

run_forward_lane() {
  local name="$1"
  local execution="$2"
  local dtype="$3"
  local output="$RUN_ROOT/forward_${name}.json"
  local command=(
    "$PYTHON_BIN" "$LAYOUT_LAB"
    --contract custom
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$LAYOUT_MODEL"
    --input "$LAYOUT_INPUT_IMAGE"
    --output "$output"
    --device npu:0
    --execution "$execution"
    --compile-cache-dir "$LAYOUT_CACHE"
    --dtype "$dtype"
    --reading-order-dtype float32
    --threshold 0.5
    --weight-format native
    --depthwise-rewrite native
    --input-color-order rgb
    --limit 1
    --warmup-pages 1
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/forward_${name}_command.sh"
  printf '\n' >>"$RUN_ROOT/forward_${name}_command.sh"
  run_phase "forward_${name}" "$RUN_ROOT/forward_${name}.log" \
    "${command[@]}"
}

run_profile_lane() {
  local name="$1"
  local execution="$2"
  local dtype="$3"
  local output="$RUN_ROOT/profile_${name}"
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
    --layout-execution "$execution"
    --layout-dtype "$dtype"
    --layout-reading-order-dtype float32
    --layout-depthwise-rewrite native
    --layout-weight-format native
    --warmup 2
    --control-repeats 20
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 80
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/profile_${name}_command.sh"
  printf '\n' >>"$RUN_ROOT/profile_${name}_command.sh"
  run_phase "profile_${name}" "$RUN_ROOT/profile_${name}.log" \
    "${command[@]}"
}

summarize_results() {
  RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY' \
    | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
lanes = {
    "eager_fp32": {
        "execution": "eager",
        "dtype": "float32",
        "reference_forward_ms": 56.321,
        "reference_profile_ms": 78.447912,
    },
    "eager_fp16_body_fp32_ro": {
        "execution": "eager",
        "dtype": "float16",
        "reference_forward_ms": 56.848,
        "reference_profile_ms": 76.380231,
    },
    "compiled_fp32": {
        "execution": "torchair",
        "dtype": "float32",
        "reference_forward_ms": 17.957,
        "reference_profile_ms": 18.164489,
    },
    "compiled_fp16_body_fp32_ro": {
        "execution": "torchair",
        "dtype": "float16",
        "reference_forward_ms": 15.613,
        "reference_profile_ms": 14.385262,
    },
}

def layout_signature(boxes):
    return [
        (box["cls_id"], box["label"], box["custom_value"])
        for box in boxes
    ]

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

summary = {
    "format": "unirec_310p_layout_precision_profile_matrix_v1",
    "reference": {
        "chip": "Ascend910B2",
        "physical_npu": 7,
        "commit": "172f209",
        "cpu_threads": {"intraop": 1, "interop": 1},
        "input_image": "jiaocaineedrop_jiaocai_needrop_en_620.jpg",
    },
    "lanes": {},
    "pairs": {},
}

for name, reference in lanes.items():
    forward = json.loads((root / f"forward_{name}.json").read_text())
    profile = json.loads(
        (root / f"profile_{name}" / "profile_suite_summary.json").read_text()
    )
    forward_config = forward["config"]
    cpu = forward_config["cpu_runtime"]
    profile_config = profile["config"]
    assert cpu["torch_intraop_threads"] == 1
    assert cpu["torch_interop_threads"] == 1
    assert profile_config["torch_intraop_threads"] == 1
    assert profile_config["torch_interop_threads"] == 1
    assert profile_config["torch_cpu_threads"] == 1
    assert forward_config["execution"] == reference["execution"]
    assert forward_config["dtype"] == reference["dtype"]
    assert forward_config["reading_order_dtype"] == "float32"

    page = forward["pages"][0]
    profile_lane = profile["lanes"][0]
    parsed = profile_lane["parsed_profile"]["summary"]["runs"][0]
    kernel = parsed["kernel_details"]
    step = parsed["step_trace_time"]["totals_us"]
    ops = {
        row["op_type"]: {
            "count": int(row["count"]),
            "total_ms": float(row["total_time_us"]) / 1000.0,
        }
        for row in parsed["op_statistic"]["top_op_types"]
    }
    forward_ms = float(page["stage_s"]["model_forward_s"]) * 1000.0
    profile_ms = float(profile_lane["steady_device_event_mean_ms"])
    lane_summary = {
        "execution": reference["execution"],
        "body_dtype": reference["dtype"],
        "reading_order_dtype": "float32",
        "box_count": int(page["box_count"]),
        "result_digest": page["result_digest"],
        "forward_ms": forward_ms,
        "reference_910b2_forward_ms": reference["reference_forward_ms"],
        "forward_ratio_vs_910b2": forward_ms / reference["reference_forward_ms"],
        "profile_steady_ms": profile_ms,
        "reference_910b2_profile_ms": reference["reference_profile_ms"],
        "profile_ratio_vs_910b2": profile_ms / reference["reference_profile_ms"],
        "profile_stage_ms": float(step.get("Stage", 0.0)) / 1000.0,
        "profile_computing_ms": float(step.get("Computing", 0.0)) / 1000.0,
        "profile_free_ms": float(step.get("Free", 0.0)) / 1000.0,
        "kernel_count": int(kernel["row_count"]),
        "cube_utilization_pct": float(kernel["weighted_cube_utilization_pct"]),
        "operator_types": ops,
        "profile_summary": str(
            root / f"profile_{name}" / "profile_suite_summary.json"
        ),
        "parsed_profile": str(
            next(
                (root / f"profile_{name}").glob(
                    "*/profile_pipe/profile_parse_summary.json"
                )
            )
        ),
    }
    summary["lanes"][name] = lane_summary
    print(
        "UNIREC_310P_LAYOUT_4LANE "
        f"lane={name} forward_ms={forward_ms:.6f} "
        f"profile_steady_ms={profile_ms:.6f} "
        f"forward_ratio_vs_910b2={lane_summary['forward_ratio_vs_910b2']:.6f} "
        f"profile_ratio_vs_910b2={lane_summary['profile_ratio_vs_910b2']:.6f} "
        f"compute_ms={lane_summary['profile_computing_ms']:.6f} "
        f"free_ms={lane_summary['profile_free_ms']:.6f} "
        f"kernels={lane_summary['kernel_count']} "
        f"cube_pct={lane_summary['cube_utilization_pct']:.3f} "
        f"boxes={lane_summary['box_count']} threads=1/1"
    )

for eager_name, compiled_name in (
    ("eager_fp32", "compiled_fp32"),
    ("eager_fp16_body_fp32_ro", "compiled_fp16_body_fp32_ro"),
):
    eager = json.loads((root / f"forward_{eager_name}.json").read_text())
    compiled = json.loads((root / f"forward_{compiled_name}.json").read_text())
    eager_boxes = eager["pages"][0]["result"]["boxes"]
    compiled_boxes = compiled["pages"][0]["result"]["boxes"]
    same_count = len(eager_boxes) == len(compiled_boxes)
    same_signature = same_count and (
        layout_signature(eager_boxes) == layout_signature(compiled_boxes)
    )
    paired = list(zip(eager_boxes, compiled_boxes)) if same_count else []
    mean_iou = (
        sum(paired_iou(first, second) for first, second in paired) / len(paired)
        if paired
        else None
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
    pair_name = f"{eager_name}_vs_{compiled_name}"
    pair_summary = {
        "same_box_count": same_count,
        "same_class_label_order_signature": same_signature,
        "mean_paired_iou": mean_iou,
        "coordinate_max_abs_px": coordinate_max_abs,
        "score_max_abs": score_max_abs,
    }
    summary["pairs"][pair_name] = pair_summary
    print(
        "UNIREC_310P_LAYOUT_4LANE_PAIR "
        f"pair={pair_name} same_count={str(same_count).lower()} "
        f"same_signature={str(same_signature).lower()} "
        f"mean_iou={mean_iou} coordinate_max_abs_px={coordinate_max_abs} "
        f"score_max_abs={score_max_abs}"
    )

output = root / "comparison_summary.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(
    "UNIREC_310P_LAYOUT_4LANE_PROFILE: PASS "
    f"lanes={len(summary['lanes'])} summary={output}"
)
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
    printf 'model=%s\nlayout_model=%s\nopenocr=%s\nimages=%s\n' \
      "$MODEL" "$LAYOUT_MODEL" "$OPENOCR_ROOT" "$IMAGES_DIR"
    printf 'input_image=%s\nlayout_cache=%s\n' \
      "$LAYOUT_INPUT_IMAGE" "$LAYOUT_CACHE"
    printf 'OMP_NUM_THREADS=%s\nMKL_NUM_THREADS=%s\nOPENBLAS_NUM_THREADS=%s\n' \
      "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS"
    lscpu
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  local spec name execution dtype
  for spec in \
    eager_fp32:eager:float32 \
    eager_fp16_body_fp32_ro:eager:float16 \
    compiled_fp32:torchair:float32 \
    compiled_fp16_body_fp32_ro:torchair:float16
  do
    IFS=: read -r name execution dtype <<<"$spec"
    run_forward_lane "$name" "$execution" "$dtype"
    run_profile_lane "$name" "$execution" "$dtype"
  done
  summarize_results
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
  printf 'UNIREC_310P_LAYOUT_4LANE_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_4lane_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  LAYOUT_CACHE="${LAYOUT_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_layout_4lane_${commit_short}_${timestamp}}"
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
    LAYOUT_INPUT_IMAGE="$LAYOUT_INPUT_IMAGE" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_LAYOUT_4LANE_STARTED pid=%s physical=%s\n' \
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
    RUN_ROOT="$(realpath "$2")"
    summarize_results
    ;;
  "") launch_main ;;
  *)
    printf 'usage: %s [--worker RUN_ROOT | --summarize RUN_ROOT]\n' "$0" >&2
    exit 2
    ;;
esac
