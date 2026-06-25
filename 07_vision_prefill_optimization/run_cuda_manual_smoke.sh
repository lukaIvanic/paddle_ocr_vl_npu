#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DEVICE="${DEVICE:-cuda:0}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/cuda_manual_smoke_8}"
export OUTPUT="${OUTPUT:-${SCRIPT_DIR}/outputs/cuda_manual_smoke_compare.json}"

echo "EXP07_CUDA_SMOKE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_CUDA_SMOKE DEVICE=${DEVICE}"
echo "EXP07_CUDA_SMOKE BASELINE_DIR=${BASELINE_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" make-baseline \
  --model "${MODEL}" \
  --dataset-dir "${DATASET_DIR:-}" \
  --device "${DEVICE}" \
  --dtype fp16 \
  --vision-attention manual \
  --cache-length "${CACHE_LENGTH:-2048}" \
  --page-start "${PAGE_START:-0}" \
  --num-pages "${NUM_PAGES:-2}" \
  --crop-count "${CROP_COUNT:-8}" \
  --selection-buckets 4 \
  --baseline-dir "${BASELINE_DIR}" \
  --force

"${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" compare \
  --model "${MODEL}" \
  --dataset-dir "${DATASET_DIR:-}" \
  --device "${DEVICE}" \
  --dtype fp16 \
  --vision-attention manual \
  --cache-length "${CACHE_LENGTH:-2048}" \
  --baseline "${BASELINE_DIR}" \
  --candidate-name cuda_manual_self_compare \
  --output "${OUTPUT}" \
  --max-items "${MAX_ITEMS:-4}" \
  --repeats 1 \
  --warmup-repeats 0
