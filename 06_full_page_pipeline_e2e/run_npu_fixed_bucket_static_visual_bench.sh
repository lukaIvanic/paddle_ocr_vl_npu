#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed-size service-bucket benchmark for compiled PromptFA static_visual.
#
# This benchmark compiles once per (min_pixels, cap_tokens) bucket and then
# runs many real crops through that fixed physical token shape. Timings are
# per-forward sync timings, so the output includes latency distributions rather
# than only one aggregate wall time.

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
export NUM_PAGES="${NUM_PAGES:-64}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_PROMPT_FA_MASK_SPARSE_MODE="${VISION_PROMPT_FA_MASK_SPARSE_MODE:-0}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-torchair}"
export BUCKET_CONFIGS="${BUCKET_CONFIGS:-28224:256,28224:384,50176:512,112896:768}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"
export WARMUP_FORWARDS="${WARMUP_FORWARDS:-4}"
export CORRECTNESS_ITEMS="${CORRECTNESS_ITEMS:-8}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export ROUGH_GT_MIN_IOU="${ROUGH_GT_MIN_IOU:-0.5}"
export MAX_CROPS="${MAX_CROPS:-0}"
export MAX_BENCHMARK_ITEMS="${MAX_BENCHMARK_ITEMS:-0}"
export RUN_DOWNSTREAM_CHECK="${RUN_DOWNSTREAM_CHECK:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/fixed_bucket_static_visual_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/fixed_bucket_static_visual.json}"

mkdir -p "${OUTPUT_DIR}"

echo "FIXED_BUCKET_STATIC_VISUAL_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV MODEL=${MODEL}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV DATASET_DIR=${DATASET_DIR}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV PAGE_START=${PAGE_START} NUM_PAGES=${NUM_PAGES}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV DTYPE=${DTYPE} NPU_JIT_COMPILE=${NPU_JIT_COMPILE}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV VISION_PROMPT_FA_MASK_SPARSE_MODE=${VISION_PROMPT_FA_MASK_SPARSE_MODE}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV VISION_COMPILE_BACKEND=${VISION_COMPILE_BACKEND}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV BUCKET_CONFIGS=${BUCKET_CONFIGS}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV BENCHMARK_REPEATS=${BENCHMARK_REPEATS} WARMUP_FORWARDS=${WARMUP_FORWARDS} CORRECTNESS_ITEMS=${CORRECTNESS_ITEMS}"
echo "FIXED_BUCKET_STATIC_VISUAL_ENV RUN_DOWNSTREAM_CHECK=${RUN_DOWNSTREAM_CHECK} MAX_BENCHMARK_ITEMS=${MAX_BENCHMARK_ITEMS}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_fixed_bucket_static_visual.py"
  --model "${MODEL}"
  --dataset-dir "${DATASET_DIR}"
  --page-start "${PAGE_START}"
  --num-pages "${NUM_PAGES}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --npu-jit-compile "${NPU_JIT_COMPILE}"
  --vision-attention "${VISION_ATTENTION_IMPL}"
  --vision-prompt-fa-layout "${VISION_PROMPT_FA_LAYOUT}"
  --vision-prompt-fa-mask-sparse-mode "${VISION_PROMPT_FA_MASK_SPARSE_MODE}"
  --vision-compile-backend "${VISION_COMPILE_BACKEND}"
  --bucket-configs "${BUCKET_CONFIGS}"
  --benchmark-repeats "${BENCHMARK_REPEATS}"
  --warmup-forwards "${WARMUP_FORWARDS}"
  --correctness-items "${CORRECTNESS_ITEMS}"
  --cache-length "${CACHE_LENGTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --rough-gt-min-iou "${ROUGH_GT_MIN_IOU}"
  --max-crops "${MAX_CROPS}"
  --max-benchmark-items "${MAX_BENCHMARK_ITEMS}"
  --json
)

if [[ "${RUN_DOWNSTREAM_CHECK}" == "0" ]]; then
  CMD+=(--no-run-downstream-check)
fi

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"

"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8")
start = raw.find("{")
if start < 0:
    raise SystemExit(f"No JSON object found in benchmark output: {path}")
data, _ = json.JSONDecoder().raw_decode(raw[start:])
rows = []
mask_sparse_mode = data.get("vision_prompt_fa_mask_sparse_mode")
for bucket in data.get("buckets", []):
    if bucket.get("skipped"):
        rows.append({
            "bucket": bucket.get("bucket"),
            "skipped": True,
            "reason": bucket.get("skip_reason"),
        })
        continue
    timing = bucket.get("timing_s", {}).get("per_forward_s", {})
    throughput = bucket.get("throughput", {})
    correctness = bucket.get("correctness", {}).get("summary", {})
    rows.append({
        "bucket": bucket.get("bucket"),
        "min_pixels": bucket.get("min_pixels"),
        "cap_tokens": bucket.get("cap_tokens"),
        "eligible_count": bucket.get("eligible_count"),
        "excluded_count": bucket.get("excluded_count"),
        "padding_waste_pct": bucket.get("padding", {}).get("padding_waste_pct"),
        "latency_ms_avg": None if timing.get("avg") is None else 1000.0 * float(timing["avg"]),
        "latency_ms_p50": None if timing.get("p50") is None else 1000.0 * float(timing["p50"]),
        "latency_ms_p90": None if timing.get("p90") is None else 1000.0 * float(timing["p90"]),
        "physical_tok_s": throughput.get("physical_vision_tokens_per_s"),
        "effective_tok_s": throughput.get("effective_vision_tokens_per_s"),
        "vision_prompt_fa_mask_sparse_mode": mask_sparse_mode,
        "correctness_passed": correctness.get("all_required_checks_passed"),
        "downstream_text_mismatch_count": correctness.get("downstream_text_mismatch_count"),
        "compiled_vs_eager_allclose_fail_count": correctness.get("compiled_vs_fixed_eager_allclose_fail_count"),
        "compiled_real_output_nonfinite_item_count": correctness.get("compiled_real_output_nonfinite_item_count"),
        "compile_first_call_s": bucket.get("compile", {}).get("compiled_first_call_s"),
    })

print("FIXED_BUCKET_STATIC_VISUAL_SUMMARY", json.dumps(rows, ensure_ascii=False, sort_keys=True))
print(f"FIXED_BUCKET_STATIC_VISUAL_OUTPUT_PATH={path}")
PY
