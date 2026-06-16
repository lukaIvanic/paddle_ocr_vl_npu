#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
DATASET_DIR="${DATASET_DIR:-/home/lukaiv/datasets/OmniDocBench_current}"
PAGE_START="${PAGE_START:-0}"
NUM_PAGES="${NUM_PAGES:-16}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-8}"
CROP_CHUNK_SIZE="${CROP_CHUNK_SIZE:-120}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
CACHE_LENGTH="${CACHE_LENGTH:-3072}"
VALIDATION_ITEMS="${VALIDATION_ITEMS:-0}"
VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_attention_compare_${RUN_ID}_p${NUM_PAGES}_b${ACTIVE_BATCH_SIZE}}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/count_omnidocbench_gt_crops.py" \
  --dataset-dir "${DATASET_DIR}" \
  --page-start 0 \
  --num-pages 64 \
  --expect-manifest "${SCRIPT_DIR}/expected_omnidocbench_first64_gt_crops.json" \
  --json

COMMON_ENV=(
  "PYTHON_BIN=${PYTHON_BIN}"
  "DATASET_DIR=${DATASET_DIR}"
  "STRICT_KNOWN_FIRST64_GT_MANIFEST=1"
  "PAGE_START=${PAGE_START}"
  "NUM_PAGES=${NUM_PAGES}"
  "ACTIVE_BATCH_SIZE=${ACTIVE_BATCH_SIZE}"
  "CROP_CHUNK_SIZE=${CROP_CHUNK_SIZE}"
  "PAGE_CHUNK_SIZE=0"
  "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  "CACHE_LENGTH=${CACHE_LENGTH}"
  "VALIDATION_ITEMS=${VALIDATION_ITEMS}"
  "FAIL_ON_MISMATCH=1"
  "FAIL_ON_LENGTH_CAP=0"
  "LAYOUT_SOURCE=omnidocbench_gt"
  "EXPECT_LAYOUT_SOURCE=omnidocbench_gt"
  "DEVICE=${DEVICE:-npu:0}"
  "DECODE_BACKEND=${DECODE_BACKEND:-torchair}"
  "NPU_JIT_COMPILE=off"
  "DOWNLOAD_DATASET=0"
  "CHECK_PADDLE_IMPORT=0"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "TORCHAIR_CACHE_DIR=${TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_page_pipeline_npu}"
)

run_case() {
  local attention_impl="$1"
  local output_path="${OUTPUT_DIR}/page_pipeline_${attention_impl}.json"
  echo "VISION_COMPARE_RUN attention=${attention_impl} output=${output_path}"
  env \
    "${COMMON_ENV[@]}" \
    "VISION_ATTENTION_IMPL=${attention_impl}" \
    "VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}" \
    "RUN_ID=${RUN_ID}_${attention_impl}" \
    "OUTPUT_PATH=${output_path}" \
    bash "${SCRIPT_DIR}/run_npu_page_pipeline_smoke.sh"
}

run_case manual
run_case prompt_flash_attention

"${PYTHON_BIN}" - "${OUTPUT_DIR}/page_pipeline_manual.json" "${OUTPUT_DIR}/page_pipeline_prompt_flash_attention.json" <<'PY'
import json
import sys
from pathlib import Path

manual_path = Path(sys.argv[1])
prompt_path = Path(sys.argv[2])
manual = json.loads(manual_path.read_text(encoding="utf-8"))
prompt = json.loads(prompt_path.read_text(encoding="utf-8"))

manual_rows = manual.get("output_fingerprints") or []
prompt_rows = prompt.get("output_fingerprints") or []
manual_by_id = {row.get("id"): row for row in manual_rows}
prompt_by_id = {row.get("id"): row for row in prompt_rows}
mismatches = []
for item_id in sorted(set(manual_by_id) | set(prompt_by_id)):
    lhs = manual_by_id.get(item_id)
    rhs = prompt_by_id.get(item_id)
    if lhs != rhs:
        mismatches.append({"id": item_id, "manual": lhs, "prompt_flash_attention": rhs})


def pick(data, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


summary = {
    "manual_path": str(manual_path),
    "prompt_flash_attention_path": str(prompt_path),
    "manual_vision_attention": manual.get("vision_attention"),
    "prompt_vision_attention": prompt.get("vision_attention"),
    "item_count_manual": len(manual_rows),
    "item_count_prompt_flash_attention": len(prompt_rows),
    "fingerprints_equal": manual.get("output_fingerprint_summary", {}).get("fingerprints_sha256")
    == prompt.get("output_fingerprint_summary", {}).get("fingerprints_sha256"),
    "mismatch_count": len(mismatches),
    "mismatches_sample": mismatches[:16],
    "manual_seconds_per_page": pick(manual, "throughput", "seconds_per_page_measured_e2e"),
    "prompt_seconds_per_page": pick(prompt, "throughput", "seconds_per_page_measured_e2e"),
    "manual_prefill_s": pick(manual, "phase_timing_s", "recognizer_ready_bank_build"),
    "prompt_prefill_s": pick(prompt, "phase_timing_s", "recognizer_ready_bank_build"),
    "manual_vision_encoder_sum_s": pick(manual, "ready_item_timing_summary_s", "vision_encoder", "sum"),
    "prompt_vision_encoder_sum_s": pick(prompt, "ready_item_timing_summary_s", "vision_encoder", "sum"),
    "manual_non_cdm": pick(manual, "omnidocbench_metrics_without_cdm", "available_non_cdm_component_mean_score_percent"),
    "prompt_non_cdm": pick(prompt, "omnidocbench_metrics_without_cdm", "available_non_cdm_component_mean_score_percent"),
    "manual_text_table_conclusion": pick(manual, "omnidocbench_metrics_without_cdm", "text_table_conclusion_mean_score_percent"),
    "prompt_text_table_conclusion": pick(prompt, "omnidocbench_metrics_without_cdm", "text_table_conclusion_mean_score_percent"),
    "manual_length_cap_hit_count": pick(manual, "decode_summary", "length_cap_hit_count"),
    "prompt_length_cap_hit_count": pick(prompt, "decode_summary", "length_cap_hit_count"),
}
print("VISION_ATTENTION_COMPARE_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
(prompt_path.parent / "vision_attention_compare_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

echo "VISION_ATTENTION_COMPARE_OUTPUT_DIR=${OUTPUT_DIR}"
