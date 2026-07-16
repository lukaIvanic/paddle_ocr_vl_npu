#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
MODEL="${MODEL:-/workspace/models/PaddleOCR-VL-1.6}"
DATASET_DIR="${DATASET_DIR:-/workspace/datasets/OmniDocBench}"
DEVICE="${DEVICE:-npu:0}"
CROP_RUN_JSON="${CROP_RUN_JSON:-${REPO_ROOT}/tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/manual_default/run.json}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/tmp/07_vision_prefill_optimization/small_visual_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.runtime_cache/07_small_visual_encoder_torchair}"

# Each entry is min_pixels:lower-exclusive:fixed-S.  The defaults cover the
# small regimes observed in the current five-page workload without launching a
# prohibitively large first compile matrix.
BUCKET_CASES="${BUCKET_CASES:-56448:0:384 112896:0:640 112896:640:768}"
ATTENTIONS="${ATTENTIONS:-manual prompt_flash_attention}"
BACKENDS="${BACKENDS:-none torchair}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP_FORWARDS="${WARMUP_FORWARDS:-5}"
MEASUREMENT_BLOCKS="${MEASUREMENT_BLOCKS:-3}"
FORWARDS_PER_BLOCK="${FORWARDS_PER_BLOCK:-20}"
LN_IMPL="${LN_IMPL:-module}"
LN_LINEAR_MODE="${LN_LINEAR_MODE:-normal}"
PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-0}"
TORCHAIR_MODE="${TORCHAIR_MODE:-default}"
USE_CACHE_COMPILE="${USE_CACHE_COMPILE:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${MODEL}/model.safetensors" || ! -f "${MODEL}/config.json" ]]; then
  echo "ERROR: local PaddleOCR-VL model is incomplete: ${MODEL}" >&2
  exit 2
fi
if [[ ! -f "${CROP_RUN_JSON}" ]]; then
  echo "ERROR: crop source run.json does not exist: ${CROP_RUN_JSON}" >&2
  exit 2
fi
if [[ "${DEVICE}" == npu:* && -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: source npu-setup before running this matrix" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}"
FAILED_CASES_FILE="${OUT_ROOT}/failed_cases.tsv"
printf 'case\texit_status\tlog\n' >"${FAILED_CASES_FILE}"

echo "EXP07_SMALL_ENCODER_MATRIX python=${PYTHON_BIN}"
echo "EXP07_SMALL_ENCODER_MATRIX model=${MODEL} device=${DEVICE} physical_npu=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "EXP07_SMALL_ENCODER_MATRIX crop_run_json=${CROP_RUN_JSON}"
echo "EXP07_SMALL_ENCODER_MATRIX bucket_cases=${BUCKET_CASES}"
echo "EXP07_SMALL_ENCODER_MATRIX attentions=${ATTENTIONS} backends=${BACKENDS} B=${BATCH_SIZE}"
echo "EXP07_SMALL_ENCODER_MATRIX warmup=${WARMUP_FORWARDS} blocks=${MEASUREMENT_BLOCKS} forwards_per_block=${FORWARDS_PER_BLOCK}"
echo "EXP07_SMALL_ENCODER_MATRIX ln_impl=${LN_IMPL} ln_linear_mode=${LN_LINEAR_MODE} promptfa_pad_head_dim_to=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_SMALL_ENCODER_MATRIX out=${OUT_ROOT} cache=${CACHE_ROOT}"

for bucket_case in ${BUCKET_CASES}; do
  IFS=: read -r min_pixels lower_s fixed_s <<<"${bucket_case}"
  if [[ -z "${min_pixels}" || -z "${lower_s}" || -z "${fixed_s}" ]]; then
    echo "ERROR: malformed BUCKET_CASES entry: ${bucket_case}" >&2
    exit 2
  fi
  for attention in ${ATTENTIONS}; do
    for backend in ${BACKENDS}; do
      case_name="min${min_pixels}_S${lower_s}_${fixed_s}_B${BATCH_SIZE}_${attention}_${backend}"
      output_json="${OUT_ROOT}/case_${case_name}.json"
      log_path="${OUT_ROOT}/case_${case_name}.log"
      cache_args=(--vision-use-torchair-cache-compile)
      if [[ "${USE_CACHE_COMPILE}" != "1" ]]; then
        cache_args=(--no-vision-use-torchair-cache-compile)
      fi
      echo "EXP07_SMALL_ENCODER_CASE start name=${case_name} output=${output_json}"
      set +e
      "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_small_visual_encoder.py" \
        --model "${MODEL}" \
        --dataset-dir "${DATASET_DIR}" \
        --crop-run-json "${CROP_RUN_JSON}" \
        --device "${DEVICE}" \
        --dtype fp16 \
        --npu-jit-compile off \
        --vision-attention "${attention}" \
        --vision-prompt-fa-layout bnsd \
        --vision-prompt-fa-mask-sparse-mode 1 \
        --preprocessor-min-pixels "${min_pixels}" \
        --bucket-min-exclusive "${lower_s}" \
        --fixed-physical-seq-len "${fixed_s}" \
        --batch-size "${BATCH_SIZE}" \
        --warmup-forwards "${WARMUP_FORWARDS}" \
        --measurement-blocks "${MEASUREMENT_BLOCKS}" \
        --forwards-per-block "${FORWARDS_PER_BLOCK}" \
        --vision-compile-backend "${backend}" \
        "${cache_args[@]}" \
        --vision-torchair-cache-dir "${CACHE_ROOT}" \
        --static-visual-ln-impl "${LN_IMPL}" \
        --static-visual-ln-linear-mode "${LN_LINEAR_MODE}" \
        --static-visual-promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}" \
        --torchair-mode "${TORCHAIR_MODE}" \
        --output "${output_json}" \
        2>&1 | tee "${log_path}"
      case_status="${PIPESTATUS[0]}"
      set -e
      if [[ "${case_status}" != "0" ]]; then
        printf '%s\t%s\t%s\n' "${case_name}" "${case_status}" "${log_path}" >>"${FAILED_CASES_FILE}"
        echo "EXP07_SMALL_ENCODER_CASE failed name=${case_name} status=${case_status} log=${log_path}" >&2
        continue
      fi
      echo "EXP07_SMALL_ENCODER_CASE done name=${case_name}"
    done
  done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_small_visual_encoder_matrix.py" "${OUT_ROOT}" \
  | tee "${OUT_ROOT}/summary.log"

echo "EXP07_SMALL_ENCODER_MATRIX_DONE out=${OUT_ROOT}"

if [[ "$(wc -l <"${FAILED_CASES_FILE}")" -gt 1 ]]; then
  echo "EXP07_SMALL_ENCODER_MATRIX completed with failed cases; see ${FAILED_CASES_FILE}" >&2
  exit 1
fi
