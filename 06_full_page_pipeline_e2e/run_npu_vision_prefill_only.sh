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

export DEVICE="${DEVICE:-npu:0}"
export PAGE_START="${PAGE_START:-0}"
export NUM_PAGES="${NUM_PAGES:-8}"
export MAX_CROPS="${MAX_CROPS:-0}"
export WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export MODES="${MODES:-sync_per_crop,unsynced_loop}"
export PROFILE_DIR="${PROFILE_DIR:-}"
export PROFILE_MODE="${PROFILE_MODE:-unsynced_loop}"
export PROFILE_METRIC="${PROFILE_METRIC:-pipe}"
export INCLUDE_IGNORED_GT="${INCLUDE_IGNORED_GT:-0}"
export INCLUDE_EMPTY_GT="${INCLUDE_EMPTY_GT:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_prefill_only_npu}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/vision_prefill_only_${RUN_ID}_p${NUM_PAGES}_${DTYPE}_${VISION_ATTENTION_IMPL}.json}"

mkdir -p "${OUTPUT_DIR}"

echo "VISION_PREFILL_ONLY_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "VISION_PREFILL_ONLY_ENV MODEL=${MODEL}"
echo "VISION_PREFILL_ONLY_ENV DATASET_DIR=${DATASET_DIR}"
echo "VISION_PREFILL_ONLY_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "VISION_PREFILL_ONLY_ENV PAGE_START=${PAGE_START} NUM_PAGES=${NUM_PAGES} MAX_CROPS=${MAX_CROPS}"
echo "VISION_PREFILL_ONLY_ENV DTYPE=${DTYPE} NPU_JIT_COMPILE=${NPU_JIT_COMPILE} WARMUP_ITEMS=${WARMUP_ITEMS}"
echo "VISION_PREFILL_ONLY_ENV VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}"
echo "VISION_PREFILL_ONLY_ENV MODES=${MODES}"
echo "VISION_PREFILL_ONLY_ENV PROFILE_DIR=${PROFILE_DIR:-disabled} PROFILE_MODE=${PROFILE_MODE} PROFILE_METRIC=${PROFILE_METRIC}"
echo "VISION_PREFILL_ONLY_ENV INCLUDE_IGNORED_GT=${INCLUDE_IGNORED_GT} INCLUDE_EMPTY_GT=${INCLUDE_EMPTY_GT}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_vision_prefill_only.py"
  --model "${MODEL}"
  --dataset-dir "${DATASET_DIR}"
  --page-start "${PAGE_START}"
  --num-pages "${NUM_PAGES}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --npu-jit-compile "${NPU_JIT_COMPILE}"
  --vision-attention "${VISION_ATTENTION_IMPL}"
  --vision-prompt-fa-layout "${VISION_PROMPT_FA_LAYOUT}"
  --modes "${MODES}"
  --warmup-items "${WARMUP_ITEMS}"
  --json
)

if (( MAX_CROPS > 0 )); then
  CMD+=(--max-crops "${MAX_CROPS}")
fi
if [[ -n "${PROFILE_DIR}" ]]; then
  CMD+=(--profile-dir "${PROFILE_DIR}" --profile-mode "${PROFILE_MODE}" --profile-metric "${PROFILE_METRIC}")
fi
if [[ "${INCLUDE_IGNORED_GT}" == "1" ]]; then
  CMD+=(--include-ignored-gt)
fi
if [[ "${INCLUDE_EMPTY_GT}" == "1" ]]; then
  CMD+=(--include-empty-gt)
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
modes = data.get("modes", {})
comparison = data.get("comparisons", {}).get("unsynced_vs_sync_per_crop", {})
summary = {
    "page_count": data.get("page_count"),
    "recognizer_crop_count": data.get("recognizer_crop_count"),
    "raw_extracted_crop_count_before_max_crops": data.get("raw_extracted_crop_count_before_max_crops"),
    "dtype": data.get("dtype"),
    "vision_attention": data.get("vision_attention"),
    "vision_prompt_fa_layout": data.get("vision_prompt_fa_layout"),
    "warmup": data.get("warmup"),
    "sync_per_crop_total_s": modes.get("sync_per_crop", {}).get("total_s"),
    "sync_per_crop_items_per_s": modes.get("sync_per_crop", {}).get("items_per_s"),
    "sync_per_crop_vision_tokens_per_s": modes.get("sync_per_crop", {}).get("vision_tokens_per_s"),
    "unsynced_loop_total_s": modes.get("unsynced_loop", {}).get("total_s"),
    "unsynced_loop_items_per_s": modes.get("unsynced_loop", {}).get("items_per_s"),
    "unsynced_loop_vision_tokens_per_s": modes.get("unsynced_loop", {}).get("vision_tokens_per_s"),
    "unsynced_speedup_over_sync": comparison.get("speedup"),
    "unsynced_saved_s": comparison.get("saved_s"),
    "profiler": {
        "enabled": (data.get("profiler") or {}).get("enabled"),
        "profile_dir": (data.get("profiler") or {}).get("profile_dir"),
        "profile_mode": (data.get("profiler") or {}).get("profile_mode"),
        "profile_metric": (data.get("profiler") or {}).get("profile_metric"),
        "profile_wall_s": (data.get("profiler") or {}).get("profile_wall_s"),
    },
    "profiled_vs_unprofiled": data.get("comparisons", {}).get(
        f"profiled_vs_unprofiled_{(data.get('profiler') or {}).get('profile_mode', '')}",
    ),
    "output_path": str(path),
}
print("VISION_PREFILL_ONLY_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "WROTE ${OUTPUT_PATH}"
