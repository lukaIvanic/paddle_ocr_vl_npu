#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/inline_single_layer_repro_$(date -u +%Y%m%dT%H%M%SZ)}"
export ITEM_INDEX="${ITEM_INDEX:-0}"
export ATTENTION="${ATTENTION:-prompt_flash_attention}"
export PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-0}"
export LN_IMPL="${LN_IMPL:-module}"
export LN_LINEAR_MODE="${LN_LINEAR_MODE:-grouped_qkv_mlp_fc1}"
export PRE_PROMPTFA_BRIDGE="${PRE_PROMPTFA_BRIDGE:-none}"
export TORCHAIR_RUN_EAGERLY="${TORCHAIR_RUN_EAGERLY:-0}"
export NO_PADDING="${NO_PADDING:-0}"

mkdir -p "${OUT_ROOT}"

RUN_EAGERLY_ARGS=()
if [[ "${TORCHAIR_RUN_EAGERLY}" == "1" ]]; then
  RUN_EAGERLY_ARGS+=(--torchair-run-eagerly)
fi
NO_PADDING_ARGS=()
if [[ "${NO_PADDING}" == "1" ]]; then
  NO_PADDING_ARGS+=(--no-padding)
fi

OUTPUT_JSON="${OUT_ROOT}/inline_single_layer_repro.json"

echo "EXP07_INLINE_SINGLE_LAYER PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_INLINE_SINGLE_LAYER MODEL=${MODEL}"
echo "EXP07_INLINE_SINGLE_LAYER BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_INLINE_SINGLE_LAYER DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_INLINE_SINGLE_LAYER DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_INLINE_SINGLE_LAYER ITEM_INDEX=${ITEM_INDEX}"
echo "EXP07_INLINE_SINGLE_LAYER ATTENTION=${ATTENTION}"
echo "EXP07_INLINE_SINGLE_LAYER PROMPTFA_PAD_HEAD_DIM_TO=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_INLINE_SINGLE_LAYER LN_IMPL=${LN_IMPL}"
echo "EXP07_INLINE_SINGLE_LAYER LN_LINEAR_MODE=${LN_LINEAR_MODE}"
echo "EXP07_INLINE_SINGLE_LAYER PRE_PROMPTFA_BRIDGE=${PRE_PROMPTFA_BRIDGE}"
echo "EXP07_INLINE_SINGLE_LAYER TORCHAIR_RUN_EAGERLY=${TORCHAIR_RUN_EAGERLY}"
echo "EXP07_INLINE_SINGLE_LAYER NO_PADDING=${NO_PADDING}"
echo "EXP07_INLINE_SINGLE_LAYER OUT_ROOT=${OUT_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/repro_inline_single_layer_compile.py" \
  --model "${MODEL}" \
  --baseline "${BASELINE_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --item-index "${ITEM_INDEX}" \
  --device "${DEVICE}" \
  --dtype fp16 \
  --npu-jit-compile off \
  --attention "${ATTENTION}" \
  --promptfa-sparse-mode 1 \
  --promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}" \
  --ln-impl "${LN_IMPL}" \
  --ln-linear-mode "${LN_LINEAR_MODE}" \
  --pre-promptfa-bridge "${PRE_PROMPTFA_BRIDGE}" \
  --compile-backend torchair \
  --torchair-mode default \
  "${RUN_EAGERLY_ARGS[@]}" \
  "${NO_PADDING_ARGS[@]}" \
  --output "${OUTPUT_JSON}"

echo "EXP07_INLINE_SINGLE_LAYER OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
