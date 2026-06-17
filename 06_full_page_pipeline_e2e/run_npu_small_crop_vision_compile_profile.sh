#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One real OCR crop, selected by lowest post-preprocessing vision-token count.
# The compiled graph is shape-specialized to this crop and covers
# model.get_image_features(): native-resolution vision encoder + adaptive MLP projector.
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/small_crop_vision_compile_profile_npu}"
export DEVICE="${DEVICE:-npu:0}"
export NUM_PAGES="${NUM_PAGES:-8}"
export PAGE_START="${PAGE_START:-0}"
export MAX_CROPS="${MAX_CROPS:-0}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-torchair}"
export VISION_COMPILE_VALIDATE="${VISION_COMPILE_VALIDATE:-1}"
export MODES="${MODES:-unsynced_loop}"
export CROP_SAMPLE="${CROP_SAMPLE:-small_only}"
export WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
export PROFILE_MODE="${PROFILE_MODE:-unsynced_loop}"
export PROFILE_METRIC="${PROFILE_METRIC:-pipe}"
export PROFILE_WARMUP_REPEATS="${PROFILE_WARMUP_REPEATS:-3}"
export PROFILE_ACTIVE_REPEATS="${PROFILE_ACTIVE_REPEATS:-10}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-${PROFILE_ACTIVE_REPEATS}}"
export TOPN="${TOPN:-20}"
export PROFILE_SKIP_TRACE="${PROFILE_SKIP_TRACE:-1}"

if [[ "${CROP_SAMPLE}" != "small_only" ]]; then
  echo "ERROR: this compile profile is shape-specialized and expects CROP_SAMPLE=small_only" >&2
  exit 2
fi
if [[ "${VISION_COMPILE_BACKEND}" == "none" ]]; then
  echo "ERROR: this script is for compiled vision profiling; set VISION_COMPILE_BACKEND=torchair/default/aot_eager/inductor" >&2
  exit 2
fi

exec "${SCRIPT_DIR}/run_npu_vision_prefill_profile_compare.sh"
