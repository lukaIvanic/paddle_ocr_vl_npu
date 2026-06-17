#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-crop downstream correctness test for compiled static_visual.
# This does not benchmark the full page pipeline. It asks whether the compiled
# visual feature drift survives the projector/prefill/decode path and changes OCR
# generation for one real OmniDocBench crop.

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
export NUM_PAGES="${NUM_PAGES:-8}"
export MAX_CROPS="${MAX_CROPS:-0}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-torchair}"
export CROP_SAMPLE="${CROP_SAMPLE:-small_only}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export FAIL_ON_TOKEN_MISMATCH="${FAIL_ON_TOKEN_MISMATCH:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "COMPILED_VISUAL_DOWNSTREAM_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV MODEL=${MODEL}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DATASET_DIR=${DATASET_DIR}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV DTYPE=${DTYPE} NPU_JIT_COMPILE=${NPU_JIT_COMPILE}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV VISION_COMPILE_BACKEND=${VISION_COMPILE_BACKEND} CROP_SAMPLE=${CROP_SAMPLE}"
echo "COMPILED_VISUAL_DOWNSTREAM_ENV CACHE_LENGTH=${CACHE_LENGTH} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

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
  --cache-length "${CACHE_LENGTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
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
    "item": data.get("item"),
    "vision_attention": data.get("vision_attention"),
    "vision_compile": data.get("vision_compile"),
    "timing_s": data.get("timing_s"),
    "visual_diff": data.get("diffs", {}).get("visual_post_layernorm"),
    "projected_diff": data.get("diffs", {}).get("projected_image_embeddings"),
    "prefill_logits_diff": data.get("diffs", {}).get("prefill_logits"),
    "tokens": data.get("tokens"),
    "texts": data.get("texts"),
    "output_path": str(path),
}
print("COMPILED_VISUAL_DOWNSTREAM_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "WROTE ${OUTPUT_PATH}"
