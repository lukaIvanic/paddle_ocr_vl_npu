#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
NUM_ITEMS="${NUM_ITEMS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
CACHE_LENGTH="${CACHE_LENGTH:-2048}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-1}"
VISION_PREFILL_BATCH_SIZE="${VISION_PREFILL_BATCH_SIZE:-1}"
DECODE_SCHEDULE="${DECODE_SCHEDULE:-hotswap}"
DECODE_BACKEND="${DECODE_BACKEND:-raw_eager}"
EOS_MODE="${EOS_MODE:-overlap_event_flags}"
VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
CROP_IDS="${CROP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/attention_generation_compare_npu}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

BASELINE_NAME="${BASELINE_NAME:-manual_fp32}"
BASELINE_VISION_ATTENTION="${BASELINE_VISION_ATTENTION:-manual}"
BASELINE_VISION_PROMPT_FA_LAYOUT="${BASELINE_VISION_PROMPT_FA_LAYOUT:-bnsd}"
BASELINE_TEXT_SOFTMAX_DTYPE="${BASELINE_TEXT_SOFTMAX_DTYPE:-fp32}"
BASELINE_VISION_SOFTMAX_DTYPE="${BASELINE_VISION_SOFTMAX_DTYPE:-fp32}"

CANDIDATE_NAME="${CANDIDATE_NAME:-promptfa}"
CANDIDATE_VISION_ATTENTION="${CANDIDATE_VISION_ATTENTION:-prompt_flash_attention}"
CANDIDATE_VISION_PROMPT_FA_LAYOUT="${CANDIDATE_VISION_PROMPT_FA_LAYOUT:-bnsd}"
CANDIDATE_TEXT_SOFTMAX_DTYPE="${CANDIDATE_TEXT_SOFTMAX_DTYPE:-fp32}"
CANDIDATE_VISION_SOFTMAX_DTYPE="${CANDIDATE_VISION_SOFTMAX_DTYPE:-fp32}"

mkdir -p "${OUTPUT_DIR}"

BASELINE_PATH="${OUTPUT_DIR}/generation_${RUN_ID}_${BASELINE_NAME}.json"
CANDIDATE_PATH="${OUTPUT_DIR}/generation_${RUN_ID}_${CANDIDATE_NAME}.json"

run_one() {
  local name="$1"
  local vision_attention="$2"
  local prompt_fa_layout="$3"
  local text_softmax="$4"
  local vision_softmax="$5"
  local output_path="$6"

  echo "ATTENTION_GENERATION_COMPARE_RUN ${name} -> ${output_path}"
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
  VISION_ATTENTION_IMPL="${vision_attention}" \
  VISION_PROMPT_FA_LAYOUT="${prompt_fa_layout}" \
  CROP_IDS="${CROP_IDS}" \
  OUTPUT_PATH="${output_path}" \
  bash "${SCRIPT_DIR}/run_npu_recognizer_queue_benchmark.sh"
}

run_one \
  "${BASELINE_NAME}" \
  "${BASELINE_VISION_ATTENTION}" \
  "${BASELINE_VISION_PROMPT_FA_LAYOUT}" \
  "${BASELINE_TEXT_SOFTMAX_DTYPE}" \
  "${BASELINE_VISION_SOFTMAX_DTYPE}" \
  "${BASELINE_PATH}"

run_one \
  "${CANDIDATE_NAME}" \
  "${CANDIDATE_VISION_ATTENTION}" \
  "${CANDIDATE_VISION_PROMPT_FA_LAYOUT}" \
  "${CANDIDATE_TEXT_SOFTMAX_DTYPE}" \
  "${CANDIDATE_VISION_SOFTMAX_DTYPE}" \
  "${CANDIDATE_PATH}"

"${PYTHON_BIN}" - "${BASELINE_PATH}" "${CANDIDATE_PATH}" "${BASELINE_NAME}" "${CANDIDATE_NAME}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
cand_path = Path(sys.argv[2])
base_name = sys.argv[3]
cand_name = sys.argv[4]
base = json.loads(base_path.read_text(encoding="utf-8"))
cand = json.loads(cand_path.read_text(encoding="utf-8"))

base_items = base.get("items", [])
cand_items = cand.get("items", [])
if len(base_items) != len(cand_items):
    raise SystemExit(f"item count differs: {base_name}={len(base_items)} {cand_name}={len(cand_items)}")

mismatches = []
length_cap_mismatches = []
for idx, (left, right) in enumerate(zip(base_items, cand_items)):
    left_ids = left.get("generated_ids_trimmed", [])
    right_ids = right.get("generated_ids_trimmed", [])
    left_text = left.get("generated_text", "")
    right_text = right.get("generated_text", "")
    first_token_diff = None
    for pos, (a, b) in enumerate(zip(left_ids, right_ids)):
        if a != b:
            first_token_diff = {"position": pos, base_name: a, cand_name: b}
            break
    if first_token_diff is None and len(left_ids) != len(right_ids):
        first_token_diff = {
            "position": min(len(left_ids), len(right_ids)),
            base_name: None,
            cand_name: None,
        }
    if bool(left.get("length_cap_hit")) or bool(right.get("length_cap_hit")):
        length_cap_mismatches.append({
            "idx": idx,
            "id": left.get("id"),
            base_name: bool(left.get("length_cap_hit")),
            cand_name: bool(right.get("length_cap_hit")),
        })
    if first_token_diff is not None or left_text != right_text:
        mismatches.append({
            "idx": idx,
            "id": left.get("id"),
            "category_type": left.get("category_type"),
            f"{base_name}_tokens": len(left_ids),
            f"{cand_name}_tokens": len(right_ids),
            "first_token_diff": first_token_diff,
            f"{base_name}_text_prefix": left_text[:200],
            f"{cand_name}_text_prefix": right_text[:200],
        })

summary = {
    "baseline_name": base_name,
    "candidate_name": cand_name,
    "baseline_path": str(base_path),
    "candidate_path": str(cand_path),
    "num_items": len(base_items),
    "device": base.get("device"),
    "dtype": base.get("dtype"),
    "decode_backend": base.get("decode_backend"),
    "decode_schedule": base.get("decode_schedule"),
    "active_batch_size": base.get("active_batch_size"),
    "max_new_tokens": base.get("max_new_tokens"),
    "cache_length": base.get("cache_length"),
    "baseline_modes": {
        "vision_attention": base.get("vision_attention"),
        "vision_prompt_fa_layout": base.get("vision_prompt_fa_layout"),
        "text_softmax_dtype": base.get("text_softmax_dtype"),
        "vision_softmax_dtype": base.get("vision_softmax_dtype"),
    },
    "candidate_modes": {
        "vision_attention": cand.get("vision_attention"),
        "vision_prompt_fa_layout": cand.get("vision_prompt_fa_layout"),
        "text_softmax_dtype": cand.get("text_softmax_dtype"),
        "vision_softmax_dtype": cand.get("vision_softmax_dtype"),
    },
    "baseline_correctness": base.get("correctness", {}),
    "candidate_correctness": cand.get("correctness", {}),
    "baseline_throughput": base.get("throughput", {}),
    "candidate_throughput": cand.get("throughput", {}),
    "token_sequence_mismatch_count": len(mismatches),
    "text_mismatch_count": sum(
        1
        for left, right in zip(base_items, cand_items)
        if left.get("generated_text", "") != right.get("generated_text", "")
    ),
    "length_cap_in_either_count": len(length_cap_mismatches),
    "sample_mismatches": mismatches[:8],
    "sample_length_cap_items": length_cap_mismatches[:8],
}
print("ATTENTION_GENERATION_COMPARE_SUMMARY", json.dumps(summary, sort_keys=True))
if mismatches:
    raise SystemExit(f"{base_name} vs {cand_name} generated different outputs for {len(mismatches)} item(s)")
PY

echo "ATTENTION_GENERATION_COMPARE_PASS ${BASELINE_PATH} ${CANDIDATE_PATH}"
