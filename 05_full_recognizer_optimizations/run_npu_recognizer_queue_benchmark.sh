#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
NUM_ITEMS="${NUM_ITEMS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CACHE_LENGTH="${CACHE_LENGTH:-1024}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-1}"
DECODE_SCHEDULE="${DECODE_SCHEDULE:-hotswap}"
DECODE_BACKEND="${DECODE_BACKEND:-torchair}"
EOS_MODE="${EOS_MODE:-overlap_event_flags}"
VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
CROP_IDS="${CROP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/recognizer_queue_benchmark}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/recognizer_queue_${RUN_ID}_${DECODE_SCHEDULE}_n${NUM_ITEMS}_b${ACTIVE_BATCH_SIZE}_cache${CACHE_LENGTH}_${DECODE_BACKEND}.json}"

mkdir -p "${OUTPUT_DIR}"
export PADDLE_OCR_VL_VISION_ATTENTION="${VISION_ATTENTION_IMPL}"
export PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_recognizer_queue.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --num-items "${NUM_ITEMS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --cache-length "${CACHE_LENGTH}"
  --active-batch-size "${ACTIVE_BATCH_SIZE}"
  --decode-schedule "${DECODE_SCHEDULE}"
  --device "${DEVICE}"
  --dtype fp16
  --decode-backend "${DECODE_BACKEND}"
  --eos-mode "${EOS_MODE}"
  --npu-jit-compile off
  --validation-items "${VALIDATION_ITEMS}"
  --json
)

if [[ -n "${CROP_IDS}" ]]; then
  read -r -a CROP_ID_ARGS <<< "${CROP_IDS}"
  CMD+=(--crop-ids "${CROP_ID_ARGS[@]}")
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
if data.get("error"):
    print("QUEUE_BENCHMARK_ERROR", json.dumps({
        "error": data.get("error"),
        "num_items": data.get("num_items"),
        "overflow_count": data.get("cache_preflight", {}).get("overflow_count"),
        "required_cache_length": data.get("cache_preflight", {}).get("required_cache_length", {}).get("max"),
    }, sort_keys=True))
    print("CACHE_PREFLIGHT", json.dumps(data.get("cache_preflight", {}), sort_keys=True))
    raise SystemExit(2)

correctness = data.get("correctness", {})
print("QUEUE_BENCHMARK_SUMMARY", json.dumps({
    "num_items": data.get("num_items"),
    "active_batch_size": data.get("active_batch_size"),
    "decode_schedule": data.get("decode_schedule"),
    "scheduler": data.get("scheduler"),
    "ready_bank_build_strategy": data.get("ready_bank_build_strategy"),
    "actual_decode_batch_sizes": data.get("actual_decode_batch_sizes"),
    "decode_cohort_count": data.get("decode_cohort_count"),
    "cache_length": data.get("cache_length"),
    "max_new_tokens": data.get("max_new_tokens"),
    "device": data.get("device"),
    "decode_backend": data.get("decode_backend"),
    "decode_attention": data.get("decode_attention"),
    "decode_cache_update": data.get("decode_cache_update"),
    "eos_mode": data.get("eos_mode"),
    "vision_attention": data.get("vision_attention"),
    "vision_prompt_fa_layout": data.get("vision_prompt_fa_layout"),
}, sort_keys=True))
print("CACHE_PREFLIGHT", json.dumps(data.get("cache_preflight", {}), sort_keys=True))
print("SETUP_TIMING_S", json.dumps(data.get("setup_timing_s", {}), sort_keys=True))
print("PHASE_TIMING_S", json.dumps(data.get("phase_timing_s", {}), sort_keys=True))
print("PIPELINE_STAGE_TIMING_SUMMARY_S", json.dumps(data.get("pipeline_stage_timing_summary_s", {}), sort_keys=True))
print("DECODE_QUEUE_DETAILS", json.dumps(data.get("decode_queue_details", {}), sort_keys=True))
print("THROUGHPUT", json.dumps(data.get("throughput", {}), sort_keys=True))
print("DECODE_SUMMARY", json.dumps(data.get("decode_summary", {}), sort_keys=True))
print("CORRECTNESS", json.dumps(correctness, sort_keys=True))

ready = data.get("ready_item_timing_summary_s", {})
print("READY_STAGE_SUMMARY_S")
print("stage avg p50 p90 sum")
for name in [
    "device_transfer",
    "vision_total",
    "native_resolution_visual_encoder_total",
    "vision_encoder",
    "adaptive_mlp_projector",
    "vision_projector_total",
    "mrope_index_cpu",
    "mrope_index_transfer",
    "text_prefill",
    "prefill_lm_head",
    "ready_item_total_excluding_device_transfer",
    "ready_item_total_with_device_transfer",
]:
    stats = ready.get(name, {})
    print(f"{name} {stats.get('avg')} {stats.get('p50')} {stats.get('p90')} {stats.get('sum')}")

print("ITEM_SUMMARY")
print("idx id input_tokens vision_tokens projected_image_tokens generated_trimmed decode_calls eos_hit length_cap_hit")
for item in data.get("items", [])[:20]:
    print(
        f"{item.get('idx')} "
        f"{item.get('id')} "
        f"{item.get('input_tokens')} "
        f"{item.get('vision_tokens')} "
        f"{item.get('projected_image_tokens')} "
        f"{item.get('generated_tokens_trimmed')} "
        f"{item.get('decode_calls')} "
        f"{item.get('eos_hit')} "
        f"{item.get('length_cap_hit')}"
    )
print("TEXT_SAMPLE", repr(data.get("texts", {}).get("sample", [])))
print("OUTPUT_JSON", path)
if not correctness.get("all_required_checks_passed", False):
    raise SystemExit(f"correctness failed for {path}")
PY
echo "WROTE ${OUTPUT_PATH}"
