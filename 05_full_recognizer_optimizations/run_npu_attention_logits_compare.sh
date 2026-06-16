#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
NUM_ITEMS="${NUM_ITEMS:-8}"
CACHE_LENGTH="${CACHE_LENGTH:-2048}"
CROP_IDS="${CROP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/attention_logits_compare_npu}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/logits_${RUN_ID}_n${NUM_ITEMS}.json}"

BASELINE_NAME="${BASELINE_NAME:-manual_fp32}"
BASELINE_VISION_ATTENTION="${BASELINE_VISION_ATTENTION:-manual}"
BASELINE_VISION_PROMPT_FA_LAYOUT="${BASELINE_VISION_PROMPT_FA_LAYOUT:-bnsd}"
BASELINE_TEXT_SOFTMAX_DTYPE="${BASELINE_TEXT_SOFTMAX_DTYPE:-fp32}"
BASELINE_VISION_SOFTMAX_DTYPE="${BASELINE_VISION_SOFTMAX_DTYPE:-fp32}"

CANDIDATE_NAME="${CANDIDATE_NAME:-manual_model}"
CANDIDATE_VISION_ATTENTION="${CANDIDATE_VISION_ATTENTION:-manual}"
CANDIDATE_VISION_PROMPT_FA_LAYOUT="${CANDIDATE_VISION_PROMPT_FA_LAYOUT:-bnsd}"
CANDIDATE_TEXT_SOFTMAX_DTYPE="${CANDIDATE_TEXT_SOFTMAX_DTYPE:-model}"
CANDIDATE_VISION_SOFTMAX_DTYPE="${CANDIDATE_VISION_SOFTMAX_DTYPE:-model}"
FAIL_ON_ARGMAX_MISMATCH="${FAIL_ON_ARGMAX_MISMATCH:-0}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_attention_logits.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --num-items "${NUM_ITEMS}"
  --device "${DEVICE}"
  --dtype fp16
  --cache-length "${CACHE_LENGTH}"
  --npu-jit-compile off
  --baseline-name "${BASELINE_NAME}"
  --baseline-vision-attention "${BASELINE_VISION_ATTENTION}"
  --baseline-vision-prompt-fa-layout "${BASELINE_VISION_PROMPT_FA_LAYOUT}"
  --baseline-text-softmax-dtype "${BASELINE_TEXT_SOFTMAX_DTYPE}"
  --baseline-vision-softmax-dtype "${BASELINE_VISION_SOFTMAX_DTYPE}"
  --candidate-name "${CANDIDATE_NAME}"
  --candidate-vision-attention "${CANDIDATE_VISION_ATTENTION}"
  --candidate-vision-prompt-fa-layout "${CANDIDATE_VISION_PROMPT_FA_LAYOUT}"
  --candidate-text-softmax-dtype "${CANDIDATE_TEXT_SOFTMAX_DTYPE}"
  --candidate-vision-softmax-dtype "${CANDIDATE_VISION_SOFTMAX_DTYPE}"
  --json
)

if [[ -n "${CROP_IDS}" ]]; then
  read -r -a CROP_ID_ARGS <<< "${CROP_IDS}"
  CMD+=(--crop-ids "${CROP_ID_ARGS[@]}")
fi
if [[ "${FAIL_ON_ARGMAX_MISMATCH}" == "1" ]]; then
  CMD+=(--fail-on-argmax-mismatch)
fi

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"
"${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
summary = data.get("summary", {})
print("ATTENTION_LOGITS_COMPARE_SUMMARY", json.dumps({
    "output_path": str(path),
    "num_items": data.get("num_items"),
    "device": data.get("device"),
    "dtype": data.get("dtype"),
    "baseline": data.get("baseline"),
    "candidate": data.get("candidate"),
    "next_token_mismatch_count": summary.get("next_token_mismatch_count"),
    "projected_image_max_abs": summary.get("projected_image_embeddings", {}).get("max_abs_diff", {}),
    "hidden_last_max_abs": summary.get("prefill_hidden_last", {}).get("max_abs_diff", {}),
    "prefill_logits_max_abs": summary.get("prefill_logits", {}).get("max_abs_diff", {}),
    "prefill_logits_mean_abs": summary.get("prefill_logits", {}).get("mean_abs_diff", {}),
    "sample_argmax_mismatches": summary.get("sample_argmax_mismatches", []),
}, sort_keys=True))
PY
echo "WROTE ${OUTPUT_PATH}"
