#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-/home/lukaiv/datasets/OmniDocBench_current}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"

echo "EXP07_NPU_REFERENCE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_NPU_REFERENCE MODEL=${MODEL}"
echo "EXP07_NPU_REFERENCE DATASET_DIR=${DATASET_DIR}"
echo "EXP07_NPU_REFERENCE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_NPU_REFERENCE BASELINE_DIR=${BASELINE_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" make-baseline \
  --model "${MODEL}" \
  --dataset-dir "${DATASET_DIR}" \
  --device "${DEVICE}" \
  --dtype fp16 \
  --npu-jit-compile off \
  --vision-attention prompt_flash_attention \
  --vision-prompt-fa-layout bnsd \
  --vision-prompt-fa-mask-sparse-mode 0 \
  --cache-length "${CACHE_LENGTH:-2048}" \
  --page-start "${PAGE_START:-0}" \
  --num-pages "${NUM_PAGES:-64}" \
  --crop-count "${CROP_COUNT:-64}" \
  --selection-buckets "${SELECTION_BUCKETS:-8}" \
  --baseline-dir "${BASELINE_DIR}" \
  "$@"
