#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Multi-crop downstream correctness test for compiled static_visual.
# This does not benchmark the full page pipeline. It asks whether the compiled
# visual feature drift survives the projector/prefill/decode path and changes OCR
# generation/rough GT accuracy across real OmniDocBench crops.

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

if [[ -z "${MODEL:-}" ]]; then
  for candidate in \
    "/home/lukaiv/models/paddle_ocr_0_9b_v_1_6" \
    "/workspace/.hf_home/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e"
  do
    if [[ -f "${candidate}/config.json" && -f "${candidate}/tokenizer.json" ]]; then
      export MODEL="${candidate}"
      break
    fi
  done
fi
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"

if [[ -z "${DATASET_DIR:-}" ]]; then
  for candidate in \
    "/home/lukaiv/datasets/OmniDocBench_current" \
    "/home/lukaiv/data/OmniDocBench_current" \
    "/home/lukaiv/data/OmniDocBench" \
    "/home/lukaiv/datasets/OmniDocBench" \
    "/root/autodl-tmp/glm_ocr_portable_bundle/data/OmniDocBench" \
    "/workspace/data/OmniDocBench"
  do
    if [[ -f "${candidate}/OmniDocBench.json" ]]; then
      export DATASET_DIR="${candidate}"
      break
    fi
  done
fi
export DATASET_DIR="${DATASET_DIR:-/home/lukaiv/datasets/OmniDocBench_current}"

export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compiled_visual_downstream_npu}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/compiled_visual_downstream_${RUN_ID}.json}"
export DEVICE="${DEVICE:-npu:0}"
export PAGE_START="${PAGE_START:-0}"
export NUM_PAGES="${NUM_PAGES:-32}"
export MAX_CROPS="${MAX_CROPS:-0}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-torchair}"
export CROP_SAMPLE="${CROP_SAMPLE:-all}"
export MAX_COMPARE_CROPS="${MAX_COMPARE_CROPS:-32}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export ROUGH_GT_MIN_IOU="${ROUGH_GT_MIN_IOU:-0.5}"
export FAIL_ON_TOKEN_MISMATCH="${FAIL_ON_TOKEN_MISMATCH:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "COMPILED_VISUAL_DOWNSTREAM_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV MODEL=${MODEL}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DATASET_DIR=${DATASET_DIR}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DTYPE=${DTYPE} NPU_JIT_COMPILE=${NPU_JIT_COMPILE}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV VISION_COMPILE_BACKEND=${VISION_COMPILE_BACKEND} CROP_SAMPLE=${CROP_SAMPLE}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV MAX_COMPARE_CROPS=${MAX_COMPARE_CROPS} CACHE_LENGTH=${CACHE_LENGTH} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} ROUGH_GT_MIN_IOU=${ROUGH_GT_MIN_IOU}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_compiled_visual_downstream.py"
  --model "${MODEL}"
  --dataset-dir "${DATASET_DIR}"
  --page-start "${PAGE_START}"
  --num-pages "${NUM_PAGES}"
  --max-crops "${MAX_CROPS}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --npu-jit-compile "${NPU_JIT_COMPILE}"
  --vision-attention "${VISION_ATTENTION_IMPL}"
  --vision-prompt-fa-layout "${VISION_PROMPT_FA_LAYOUT}"
  --vision-compile-backend "${VISION_COMPILE_BACKEND}"
  --crop-sample "${CROP_SAMPLE}"
  --max-compare-crops "${MAX_COMPARE_CROPS}"
  --cache-length "${CACHE_LENGTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --rough-gt-min-iou "${ROUGH_GT_MIN_IOU}"
  --json
)

if [[ "${FAIL_ON_TOKEN_MISMATCH}" == "1" ]]; then
  CMD+=(--fail-on-token-mismatch)
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
summary = {
    "page_start": data.get("page_start"),
    "num_pages": data.get("num_pages"),
    "raw_queue_input_count_before_crop_sample": data.get("raw_queue_input_count_before_crop_sample"),
    "selected_compare_count": data.get("selected_compare_count"),
    "max_compare_crops": data.get("max_compare_crops"),
    "vision_attention": data.get("vision_attention"),
    "vision_compile_backend": data.get("vision_compile_backend"),
    "timing_s": data.get("timing_s"),
    "summary": data.get("summary"),
    "sample_items": [
        {
            "idx": item.get("idx"),
            "id": item.get("id"),
            "category_type": item.get("category_type"),
            "vision_tokens": item.get("vision_tokens"),
            "vision_seq_len_mod_16": item.get("vision_seq_len_mod_16"),
            "generated_match": item.get("tokens", {}).get("generated_trimmed_match"),
            "text_match": item.get("texts", {}).get("match"),
            "compiled_visual_nonfinite": item.get("diffs", {}).get("visual_post_layernorm", {}).get("lhs_nonfinite_count"),
            "compiled_projected_nonfinite": item.get("diffs", {}).get("projected_image_embeddings", {}).get("lhs_nonfinite_count"),
            "visual_max_abs": item.get("diffs", {}).get("visual_post_layernorm", {}).get("max_abs_diff"),
            "projected_max_abs": item.get("diffs", {}).get("projected_image_embeddings", {}).get("max_abs_diff"),
            "prefill_logits_max_abs": item.get("diffs", {}).get("prefill_logits", {}).get("max_abs_diff"),
        }
        for item in data.get("items", [])[:8]
    ],
    "output_path": str(path),
}
print("COMPILED_VISUAL_DOWNSTREAM_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "WROTE ${OUTPUT_PATH}"
