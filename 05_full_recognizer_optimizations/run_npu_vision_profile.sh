#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
CROP_ID="${CROP_ID:-hotswap_002_code_txt_p1474_11}"
VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
PROFILE_METRIC="${PROFILE_METRIC:-pipe}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"
TOPN="${TOPN:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/vision_encoder_profiles}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_RUN_DIR="${PROFILE_RUN_DIR:-${OUTPUT_ROOT}/vision_encoder_${RUN_ID}_${CROP_ID}_${VISION_ATTENTION_IMPL}_${PROFILE_METRIC}}"

mkdir -p "${OUTPUT_ROOT}"
export PADDLE_OCR_VL_VISION_ATTENTION="${VISION_ATTENTION_IMPL}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_vision_encoder.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --crop-id "${CROP_ID}"
  --device "${DEVICE}"
  --dtype fp16
  --npu-jit-compile off
  --profile-run-dir "${PROFILE_RUN_DIR}"
  --profile-metric "${PROFILE_METRIC}"
  --vision-attention "${VISION_ATTENTION_IMPL}"
  --warmup-iters "${WARMUP_ITERS}"
  --profile-iters "${PROFILE_ITERS}"
)

echo "COMMAND ${CMD[*]}"
"${CMD[@]}"

PARSE_JSON="${PROFILE_RUN_DIR}/profile_parse_summary.json"
PARSE_MD="${PROFILE_RUN_DIR}/profile_parse_summary.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/parse_npu_profile.py" \
  --profile-dir "${PROFILE_RUN_DIR}" \
  --topn "${TOPN}" \
  --out-json "${PARSE_JSON}" \
  --out-md "${PARSE_MD}"

"${PYTHON_BIN}" - "${PROFILE_RUN_DIR}" "${PARSE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

profile_dir = Path(sys.argv[1])
parse_json = Path(sys.argv[2])
vision_summary = json.loads((profile_dir / "vision_profile_summary.json").read_text(encoding="utf-8"))
parsed = json.loads(parse_json.read_text(encoding="utf-8"))
run = parsed["runs"][0]

print("VISION_PROFILE_SUMMARY", json.dumps({
    "profile_dir": str(profile_dir),
    "crop_id": vision_summary.get("crop_id"),
    "input_tokens": vision_summary.get("input_tokens"),
    "vision_tokens": vision_summary.get("vision_tokens"),
    "hidden_size": vision_summary.get("hidden_size"),
    "warmup_times_s": vision_summary.get("warmup_times_s"),
    "profile_wall_s": vision_summary.get("profile_wall_s"),
    "profile_iters": vision_summary.get("profile_iters"),
    "profile_metric": vision_summary.get("profile_metric"),
    "vision_attention": vision_summary.get("vision_attention"),
}, sort_keys=True))
print("VISION_ATTENTION_VALIDATION", json.dumps(vision_summary.get("validation", {}), sort_keys=True))

if "step_trace_time" in run:
    print("STEP_TRACE_TOTALS_US", json.dumps(run["step_trace_time"].get("totals_us", {}), sort_keys=True))

kernel = run.get("kernel_details", {})

def print_rows(label, rows, value_key="duration_us", limit=10):
    print(label)
    print("name count total_us")
    for row in rows[:limit]:
        name = str(row.get("name") or row.get("op_type") or "unknown").replace("\n", " ")[:160]
        print(f"{name} {row.get('count')} {row.get(value_key)}")

print_rows("TOP_KERNEL_TYPES", kernel.get("top_kernel_types", []))
print_rows("TOP_MATMUL_SHAPES", kernel.get("top_matmul_shape_signatures", []), limit=8)
print_rows("TOP_TRANSDATA_SHAPES", kernel.get("top_transdata_shape_signatures", []), limit=8)
print_rows("TOP_SUSPECT_KERNELS", kernel.get("suspect_kernel_rows", []), limit=10)

operators = run.get("operator_details", {})
print_rows("TOP_OPERATORS_BY_DEVICE_US", operators.get("top_by_device_total_us", []), value_key="device_total_us", limit=10)

print("PROFILE_PARSE_JSON", parse_json)
print("PROFILE_PARSE_MD", profile_dir / "profile_parse_summary.md")
PY
