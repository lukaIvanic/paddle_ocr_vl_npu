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
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/full_recognizer_stage_timing}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/stage_timing_num${NUM_ITEMS}_${DECODE_BACKEND}.json}"

mkdir -p "${OUTPUT_DIR}"

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
  --json
)

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"
"${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
echo "WROTE ${OUTPUT_PATH}"
