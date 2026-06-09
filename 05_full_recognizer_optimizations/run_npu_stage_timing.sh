#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
NUM_ITEMS="${NUM_ITEMS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CACHE_LENGTH="${CACHE_LENGTH:-1269}"
DECODE_BACKEND="${DECODE_BACKEND:-torchair}"
VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
DECODE_STEP_TIMING="${DECODE_STEP_TIMING:-0}"
CROP_IDS="${CROP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/full_recognizer_stage_timing}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/stage_timing_num${NUM_ITEMS}_${DECODE_BACKEND}_${VISION_ATTENTION_IMPL}_${VISION_PROMPT_FA_LAYOUT}_warm${WARMUP_ITEMS}.json}"

mkdir -p "${OUTPUT_DIR}"
export PADDLE_OCR_VL_VISION_ATTENTION="${VISION_ATTENTION_IMPL}"
export PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_stage_timing.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --num-items "${NUM_ITEMS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --cache-length "${CACHE_LENGTH}"
  --device "${DEVICE}"
  --dtype fp16
  --decode-backend "${DECODE_BACKEND}"
  --npu-jit-compile off
  --warmup-items "${WARMUP_ITEMS}"
  --json
)

if [[ -n "${CROP_IDS}" ]]; then
  read -r -a CROP_ID_ARGS <<< "${CROP_IDS}"
  CMD+=(--crop-ids "${CROP_ID_ARGS[@]}")
fi

case "${DECODE_STEP_TIMING}" in
  1|true|TRUE|yes|YES|on|ON)
    CMD+=(--decode-step-timing)
    ;;
esac

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"
"${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if not data.get("correctness", {}).get("all_required_checks_passed", False):
    print(json.dumps(data.get("correctness", {}), indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit(f"correctness failed for {path}")
print("CORRECTNESS", json.dumps(data.get("correctness", {}), sort_keys=True))
print("SETUP_TIMING_S", json.dumps(data.get("setup_timing_s", {}), sort_keys=True))
print("VISION_ATTENTION", data.get("vision_attention"))
print("VISION_PROMPT_FA_LAYOUT", data.get("vision_prompt_fa_layout"))
warmup = data.get("stage_warmup", {})
if warmup.get("count", 0):
    print("STAGE_WARMUP", json.dumps({
        "count": warmup.get("count"),
        "elapsed_s": warmup.get("elapsed_s"),
        "item_ids": warmup.get("item_ids"),
        "mismatch_count": warmup.get("mismatch_count"),
    }, sort_keys=True))

summary = data.get("stage_timing_summary_s", {})
stage_names = [
    "native_resolution_visual_encoder_total",
    "vision_total",
    "vision_encoder",
    "adaptive_mlp_projector",
    "text_prefill",
    "prefill_lm_head",
    "static_decode_total",
    "model_total_excluding_device_transfer",
]
print("STAGE_SUMMARY_S")
print("stage avg p50 p90 sum")
for name in stage_names:
    stats = summary.get(name, {})
    print(
        f"{name} "
        f"{stats.get('avg')} "
        f"{stats.get('p50')} "
        f"{stats.get('p90')} "
        f"{stats.get('sum')}"
    )

print("ITEM_SUMMARY")
print("idx id input_tokens vision_tokens projected_image_tokens static_decode_total model_total tokens_match")
for idx, item in enumerate(data.get("items", [])):
    timing = item.get("timing_s", {})
    print(
        f"{idx} "
        f"{item.get('id')} "
        f"{item.get('input_tokens')} "
        f"{item.get('vision_tokens')} "
        f"{item.get('projected_image_tokens')} "
        f"{timing.get('static_decode_total')} "
        f"{timing.get('model_total_excluding_device_transfer')} "
        f"{item.get('correctness', {}).get('tokens_match')}"
    )
    steps = item.get("decode_step_wall_s") or []
    if steps:
        print(
            f"ITEM_DECODE_STEPS idx={idx} "
            f"count={len(steps)} first={steps[0]} max={max(steps)} sum={sum(steps)}"
        )
PY
echo "WROTE ${OUTPUT_PATH}"
