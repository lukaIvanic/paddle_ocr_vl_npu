#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/msit_ge_fx_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_ITEMS="${MAX_ITEMS:-1}"
export REPEATS="${REPEATS:-1}"
export WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
export MSIT_DUMP_MODE="${MSIT_DUMP_MODE:-output}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_MSIT_GE_FX PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_MSIT_GE_FX MODEL=${MODEL}"
echo "EXP07_MSIT_GE_FX BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_MSIT_GE_FX DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_MSIT_GE_FX DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_MSIT_GE_FX OUT_ROOT=${OUT_ROOT}"
echo "EXP07_MSIT_GE_FX MAX_ITEMS=${MAX_ITEMS} REPEATS=${REPEATS} WARMUP_REPEATS=${WARMUP_REPEATS}"
echo "EXP07_MSIT_GE_FX MSIT_DUMP_MODE=${MSIT_DUMP_MODE}"

run_dump() {
  local kind="$1"
  shift
  local dump_dir="${OUT_ROOT}/${kind}"
  local output_json="${OUT_ROOT}/${kind}_compare.json"

  echo "EXP07_MSIT_GE_FX RUN kind=${kind} dump_dir=${dump_dir}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" compare \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --baseline "${BASELINE_DIR}" \
    --candidate-name "static_visual_torchair_msit_${kind}" \
    --output "${output_json}" \
    --device "${DEVICE}" \
    --dtype fp16 \
    --npu-jit-compile off \
    --vision-attention prompt_flash_attention \
    --vision-prompt-fa-layout bnsd \
    --vision-prompt-fa-mask-sparse-mode 1 \
    --cache-length "${CACHE_LENGTH:-2048}" \
    --vision-compile-backend torchair \
    --validate-compiled-against-static-eager \
    --max-items "${MAX_ITEMS}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --torchair-msit-dump-kind "${kind}" \
    --torchair-msit-dump-dir "${dump_dir}" \
    --torchair-msit-dump-mode "${MSIT_DUMP_MODE}" \
    "$@"
}

run_dump ge "$@"
run_dump fx "$@"

GE_PATH="${OUT_ROOT}/ge/msit_ge_dump"
FX_PATH="${OUT_ROOT}/fx/msit_fx_dump"
FX_COMPARE_PATH="${FX_PATH}"
if [[ -d "${FX_PATH}/data_dump" ]]; then
  FX_COMPARE_PATH="${FX_PATH}/data_dump"
elif [[ -d "${OUT_ROOT}/fx/data_dump" ]]; then
  FX_COMPARE_PATH="${OUT_ROOT}/fx/data_dump"
fi

echo "EXP07_MSIT_GE_FX GE_PATH=${GE_PATH}"
echo "EXP07_MSIT_GE_FX FX_COMPARE_PATH=${FX_COMPARE_PATH}"

if [[ ! -d "${GE_PATH}" ]]; then
  echo "EXP07_MSIT_GE_FX ERROR missing GE dump directory: ${GE_PATH}" >&2
  exit 2
fi
if [[ ! -d "${FX_COMPARE_PATH}" ]]; then
  echo "EXP07_MSIT_GE_FX ERROR missing FX dump directory: ${FX_COMPARE_PATH}" >&2
  exit 2
fi

if command -v msit >/dev/null 2>&1; then
  echo "EXP07_MSIT_GE_FX RUN_COMPARE"
  msit llm compare \
    --my-path "${GE_PATH}" \
    --golden-path "${FX_COMPARE_PATH}" \
    --output "${OUT_ROOT}/msit_compare"
else
  echo "EXP07_MSIT_GE_FX MSIT_COMPARE_SKIPPED command_not_found=msit"
  echo "EXP07_MSIT_GE_FX MANUAL_COMPARE_COMMAND=msit llm compare --my-path '${GE_PATH}' --golden-path '${FX_COMPARE_PATH}' --output '${OUT_ROOT}/msit_compare'"
fi

echo "EXP07_MSIT_GE_FX OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 3 -type f | sort | head -n 80
