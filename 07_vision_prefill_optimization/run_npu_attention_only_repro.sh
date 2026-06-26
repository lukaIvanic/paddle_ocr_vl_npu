#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/attention_only_repro_$(date -u +%Y%m%dT%H%M%SZ)}"

export ITEM_INDEX="${ITEM_INDEX:-0}"
export ATTENTION="${ATTENTION:-prompt_flash_attention}"
export COMPILE_BACKEND="${COMPILE_BACKEND:-torchair}"
export TORCHAIR_RUN_EAGERLY="${TORCHAIR_RUN_EAGERLY:-0}"
export MASK_KIND="${MASK_KIND:-current}"
export MASK_RANK="${MASK_RANK:-4}"
export NO_PADDING="${NO_PADDING:-0}"
export LN_FP32_REDUCE="${LN_FP32_REDUCE:-1}"

mkdir -p "${OUT_ROOT}"

RUN_EAGERLY_ARGS=()
if [[ "${TORCHAIR_RUN_EAGERLY}" == "1" ]]; then
  RUN_EAGERLY_ARGS+=(--torchair-run-eagerly)
fi
NO_PADDING_ARGS=()
if [[ "${NO_PADDING}" == "1" ]]; then
  NO_PADDING_ARGS+=(--no-padding)
fi
LN_REDUCE_ARGS=(--ln-fp32-reduce)
if [[ "${LN_FP32_REDUCE}" == "0" ]]; then
  LN_REDUCE_ARGS=(--no-ln-fp32-reduce)
fi

SAFE_NAME="attn_${ATTENTION}_backend_${COMPILE_BACKEND}_mask_${MASK_KIND}_rank_${MASK_RANK}_nopad_${NO_PADDING}_eagerly_${TORCHAIR_RUN_EAGERLY}"
OUTPUT_JSON="${OUT_ROOT}/${SAFE_NAME}.json"

echo "EXP07_ATTENTION_ONLY PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_ATTENTION_ONLY MODEL=${MODEL}"
echo "EXP07_ATTENTION_ONLY BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_ATTENTION_ONLY DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_ATTENTION_ONLY DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_ATTENTION_ONLY ITEM_INDEX=${ITEM_INDEX}"
echo "EXP07_ATTENTION_ONLY ATTENTION=${ATTENTION}"
echo "EXP07_ATTENTION_ONLY COMPILE_BACKEND=${COMPILE_BACKEND}"
echo "EXP07_ATTENTION_ONLY MASK_KIND=${MASK_KIND} MASK_RANK=${MASK_RANK}"
echo "EXP07_ATTENTION_ONLY NO_PADDING=${NO_PADDING}"
echo "EXP07_ATTENTION_ONLY TORCHAIR_RUN_EAGERLY=${TORCHAIR_RUN_EAGERLY}"
echo "EXP07_ATTENTION_ONLY OUT_ROOT=${OUT_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/repro_attention_only_compile.py" \
  --model "${MODEL}" \
  --baseline "${BASELINE_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --item-index "${ITEM_INDEX}" \
  --device "${DEVICE}" \
  --dtype fp16 \
  --npu-jit-compile off \
  --attention "${ATTENTION}" \
  --compile-backend "${COMPILE_BACKEND}" \
  --torchair-mode default \
  --mask-kind "${MASK_KIND}" \
  --mask-rank "${MASK_RANK}" \
  --promptfa-sparse-mode 1 \
  "${LN_REDUCE_ARGS[@]}" \
  "${RUN_EAGERLY_ARGS[@]}" \
  "${NO_PADDING_ARGS[@]}" \
  --output "${OUTPUT_JSON}"

echo "EXP07_ATTENTION_ONLY OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
