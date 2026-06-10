#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
export DATASET_DIR="${DATASET_DIR:-/home/lukaiv/data/OmniDocBench}"

export DEVICE="${DEVICE:-npu:0}"
export LAYOUT_DEVICE="${LAYOUT_DEVICE:-cpu}"
export LAYOUT_SOURCE="${LAYOUT_SOURCE:-omnidocbench_gt}"
export ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export DECODE_BACKEND="${DECODE_BACKEND:-torchair}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
export DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-0}"
export CHECK_PADDLE_IMPORT="${CHECK_PADDLE_IMPORT:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/page_pipeline_npu_smoke}"
export TORCHAIR_CACHE_DIR="${TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_page_pipeline_npu}"

echo "NPU_PAGE_PIPELINE_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "NPU_PAGE_PIPELINE_ENV MODEL=${MODEL}"
echo "NPU_PAGE_PIPELINE_ENV DATASET_DIR=${DATASET_DIR}"
echo "NPU_PAGE_PIPELINE_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "NPU_PAGE_PIPELINE_ENV DECODE_BACKEND=${DECODE_BACKEND} NPU_JIT_COMPILE=${NPU_JIT_COMPILE}"
echo "NPU_PAGE_PIPELINE_ENV ACTIVE_BATCH_SIZE=${ACTIVE_BATCH_SIZE} CACHE_LENGTH=${CACHE_LENGTH} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "NPU_PAGE_PIPELINE_ENV TORCHAIR_CACHE_DIR=${TORCHAIR_CACHE_DIR}"

exec bash "${SCRIPT_DIR}/run_cuda_page_pipeline_smoke.sh"
