#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-PaddlePaddle/PaddleOCR-VL-1.6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ITEMS="${NUM_ITEMS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
CACHE_LENGTH="${CACHE_LENGTH:-2048}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-1}"
VISION_PREFILL_BATCH_SIZE="${VISION_PREFILL_BATCH_SIZE:-1}"
DECODE_SCHEDULE="${DECODE_SCHEDULE:-hotswap}"
DECODE_BACKEND="${DECODE_BACKEND:-raw_eager}"
EOS_MODE="${EOS_MODE:-overlap_event_flags}"
VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
CROP_IDS="${CROP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/attention_softmax_compare_cuda}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${OUTPUT_DIR}"

BASELINE_PATH="${OUTPUT_DIR}/softmax_${RUN_ID}_fp32.json"
CANDIDATE_PATH="${OUTPUT_DIR}/softmax_${RUN_ID}_model.json"

run_one() {
  local text_softmax="$1"
  local vision_softmax="$2"
  local output_path="$3"

  PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE="${text_softmax}" \
  PADDLE_OCR_VL_VISION_SOFTMAX_DTYPE="${vision_softmax}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL="${MODEL}" \
  MANIFEST="${MANIFEST}" \
  DEVICE="${DEVICE}" \
  NUM_ITEMS="${NUM_ITEMS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  CACHE_LENGTH="${CACHE_LENGTH}" \
  ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE}" \
  VISION_PREFILL_BATCH_SIZE="${VISION_PREFILL_BATCH_SIZE}" \
  DECODE_SCHEDULE="${DECODE_SCHEDULE}" \
  DECODE_BACKEND="${DECODE_BACKEND}" \
  EOS_MODE="${EOS_MODE}" \
  VALIDATION_ITEMS="${VALIDATION_ITEMS}" \
  VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL}" \
  VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT}" \
  CROP_IDS="${CROP_IDS}" \
  OUTPUT_PATH="${output_path}" \
  bash "${SCRIPT_DIR}/run_cuda_recognizer_queue_benchmark.sh"
}

echo "ATTENTION_SOFTMAX_COMPARE_RUN fp32 text+vision -> ${BASELINE_PATH}"
run_one fp32 fp32 "${BASELINE_PATH}"

echo "ATTENTION_SOFTMAX_COMPARE_RUN model text+vision -> ${CANDIDATE_PATH}"
run_one model model "${CANDIDATE_PATH}"

"${PYTHON_BIN}" - "${BASELINE_PATH}" "${CANDIDATE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
cand_path = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
cand = json.loads(cand_path.read_text(encoding="utf-8"))

base_items = base.get("items", [])
cand_items = cand.get("items", [])
if len(base_items) != len(cand_items):
    raise SystemExit(f"item count differs: fp32={len(base_items)} model={len(cand_items)}")

mismatches = []
for idx, (left, right) in enumerate(zip(base_items, cand_items)):
    left_ids = left.get("generated_ids_trimmed", [])
    right_ids = right.get("generated_ids_trimmed", [])
    left_text = left.get("generated_text", "")
    right_text = right.get("generated_text", "")
    first_token_diff = None
    for pos, (a, b) in enumerate(zip(left_ids, right_ids)):
        if a != b:
            first_token_diff = {"position": pos, "fp32": a, "model": b}
            break
    if first_token_diff is None and len(left_ids) != len(right_ids):
        first_token_diff = {"position": min(len(left_ids), len(right_ids)), "fp32": None, "model": None}
    if first_token_diff is not None or left_text != right_text:
        mismatches.append({
            "idx": idx,
            "id": left.get("id"),
            "category_type": left.get("category_type"),
            "fp32_tokens": len(left_ids),
            "model_tokens": len(right_ids),
            "first_token_diff": first_token_diff,
            "fp32_text_prefix": left_text[:200],
            "model_text_prefix": right_text[:200],
        })

summary = {
    "fp32_path": str(base_path),
    "model_path": str(cand_path),
    "num_items": len(base_items),
    "device": base.get("device"),
    "dtype": base.get("dtype"),
    "decode_backend": base.get("decode_backend"),
    "decode_schedule": base.get("decode_schedule"),
    "active_batch_size": base.get("active_batch_size"),
    "max_new_tokens": base.get("max_new_tokens"),
    "cache_length": base.get("cache_length"),
    "fp32_modes": {
        "text_softmax_dtype": base.get("text_softmax_dtype"),
        "vision_softmax_dtype": base.get("vision_softmax_dtype"),
    },
    "model_modes": {
        "text_softmax_dtype": cand.get("text_softmax_dtype"),
        "vision_softmax_dtype": cand.get("vision_softmax_dtype"),
    },
    "fp32_correctness": base.get("correctness", {}),
    "model_correctness": cand.get("correctness", {}),
    "fp32_throughput": base.get("throughput", {}),
    "model_throughput": cand.get("throughput", {}),
    "token_sequence_mismatch_count": len(mismatches),
    "text_mismatch_count": sum(
        1
        for left, right in zip(base_items, cand_items)
        if left.get("generated_text", "") != right.get("generated_text", "")
    ),
    "sample_mismatches": mismatches[:8],
}
print("ATTENTION_SOFTMAX_COMPARE_SUMMARY", json.dumps(summary, sort_keys=True))
if mismatches:
    raise SystemExit(f"fp32 vs model/fp16 attention generated different outputs for {len(mismatches)} item(s)")
PY

echo "ATTENTION_SOFTMAX_COMPARE_PASS ${BASELINE_PATH} ${CANDIDATE_PATH}"
