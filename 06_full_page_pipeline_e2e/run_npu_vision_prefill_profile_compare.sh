#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP5_DIR="$(cd "${SCRIPT_DIR}/../05_full_recognizer_optimizations" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_prefill_profile_compare_npu}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export DEVICE="${DEVICE:-npu:0}"
export NUM_PAGES="${NUM_PAGES:-8}"
export PAGE_START="${PAGE_START:-0}"
export MAX_CROPS="${MAX_CROPS:-0}"
export WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export MODES="${MODES:-unsynced_loop}"
export CROP_SAMPLE="${CROP_SAMPLE:-small_medium_large}"
export PROFILE_MODE="${PROFILE_MODE:-unsynced_loop}"
export PROFILE_METRIC="${PROFILE_METRIC:-pipe}"
export PROFILE_WARMUP_REPEATS="${PROFILE_WARMUP_REPEATS:-2}"
export PROFILE_ACTIVE_REPEATS="${PROFILE_ACTIVE_REPEATS:-5}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-${PROFILE_ACTIVE_REPEATS}}"
export TOPN="${TOPN:-20}"
export PROFILE_SKIP_TRACE="${PROFILE_SKIP_TRACE:-1}"

mkdir -p "${OUTPUT_DIR}"

OFF_OUTPUT="${OUTPUT_DIR}/vision_prefill_profile_compare_${RUN_ID}_off.json"
ON_OUTPUT="${OUTPUT_DIR}/vision_prefill_profile_compare_${RUN_ID}_profiled.json"
PROFILE_ROOT="${OUTPUT_DIR}/profiles/${RUN_ID}"

echo "VISION_PREFILL_PROFILE_COMPARE_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "VISION_PREFILL_PROFILE_COMPARE_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "VISION_PREFILL_PROFILE_COMPARE_ENV PAGE_START=${PAGE_START} NUM_PAGES=${NUM_PAGES} MAX_CROPS=${MAX_CROPS}"
echo "VISION_PREFILL_PROFILE_COMPARE_ENV DTYPE=${DTYPE} VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_PROMPT_FA_LAYOUT=${VISION_PROMPT_FA_LAYOUT}"
echo "VISION_PREFILL_PROFILE_COMPARE_ENV MODES=${MODES} CROP_SAMPLE=${CROP_SAMPLE} BENCHMARK_REPEATS=${BENCHMARK_REPEATS}"
echo "VISION_PREFILL_PROFILE_COMPARE_ENV PROFILE_MODE=${PROFILE_MODE} PROFILE_METRIC=${PROFILE_METRIC} PROFILE_WARMUP_REPEATS=${PROFILE_WARMUP_REPEATS} PROFILE_ACTIVE_REPEATS=${PROFILE_ACTIVE_REPEATS} PROFILE_SKIP_TRACE=${PROFILE_SKIP_TRACE}"

echo "VISION_PREFILL_PROFILE_COMPARE_RUN without_profiler output=${OFF_OUTPUT}"
PROFILE_DIR="" OUTPUT_PATH="${OFF_OUTPUT}" RUN_ID="${RUN_ID}_off" "${SCRIPT_DIR}/run_npu_vision_prefill_only.sh"

echo "VISION_PREFILL_PROFILE_COMPARE_RUN with_profiler output=${ON_OUTPUT} profile_root=${PROFILE_ROOT}"
PROFILE_DIR="${PROFILE_ROOT}" OUTPUT_PATH="${ON_OUTPUT}" RUN_ID="${RUN_ID}_profiled" "${SCRIPT_DIR}/run_npu_vision_prefill_only.sh"

PROFILE_RUN_DIR="$("${PYTHON_BIN}" - "${ON_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
profile_dir = (data.get("profiler") or {}).get("profile_dir")
if not profile_dir:
    raise SystemExit("profile_dir missing from profiled output")
print(profile_dir)
PY
)"

PARSE_JSON="${PROFILE_RUN_DIR}/profile_parse_summary.json"
PARSE_MD="${PROFILE_RUN_DIR}/profile_parse_summary.md"
PARSE_CMD=(
  "${PYTHON_BIN}" "${EXP5_DIR}/parse_npu_profile.py"
  --profile-dir "${PROFILE_RUN_DIR}"
  --topn "${TOPN}"
  --out-json "${PARSE_JSON}"
  --out-md "${PARSE_MD}"
)
if [[ "${PROFILE_SKIP_TRACE}" == "1" ]]; then
  PARSE_CMD+=(--skip-trace)
fi

echo "COMMAND ${PARSE_CMD[*]}"
"${PARSE_CMD[@]}"

"${PYTHON_BIN}" - "${OFF_OUTPUT}" "${ON_OUTPUT}" "${PARSE_JSON}" "${PARSE_MD}" <<'PY'
import json
import sys
from pathlib import Path

off_path = Path(sys.argv[1])
on_path = Path(sys.argv[2])
parse_json = Path(sys.argv[3])
parse_md = Path(sys.argv[4])

off = json.loads(off_path.read_text(encoding="utf-8"))
on = json.loads(on_path.read_text(encoding="utf-8"))
parsed = json.loads(parse_json.read_text(encoding="utf-8"))
run = parsed["runs"][0]

profile = on.get("profiler") or {}
profile_mode = profile.get("profile_mode") or "unsynced_loop"
off_mode = off.get("modes", {}).get(profile_mode, {})
on_baseline = on.get("modes", {}).get(profile_mode, {})
profiled = profile.get("profiled_mode_result") or {}
effect = on.get("comparisons", {}).get(f"profiled_vs_unprofiled_{profile_mode}", {})

summary = {
    "off_output": str(off_path),
    "profiled_output": str(on_path),
    "profile_parse_json": str(parse_json),
    "profile_parse_md": str(parse_md),
    "profile_dir": profile.get("profile_dir"),
    "profile_metric": profile.get("profile_metric"),
    "profile_mode": profile_mode,
    "dtype": on.get("dtype"),
    "vision_attention": on.get("vision_attention"),
    "vision_prompt_fa_layout": on.get("vision_prompt_fa_layout"),
    "recognizer_crop_count": on.get("recognizer_crop_count"),
    "raw_queue_input_count_before_crop_sample": on.get("raw_queue_input_count_before_crop_sample"),
    "crop_sample": on.get("crop_sample"),
    "benchmark_repeats": on.get("benchmark_repeats"),
    "profile_warmup_repeats": profile.get("profile_warmup_repeats"),
    "profile_active_repeats": profile.get("profile_active_repeats"),
    "profile_active_steps": profile.get("profile_active_steps"),
    "profiler_step_contract": profile.get("profiler_step_contract"),
    "off_items_per_s": off_mode.get("items_per_s"),
    "off_vision_tokens_per_s": off_mode.get("vision_tokens_per_s"),
    "same_process_unprofiled_items_per_s": on_baseline.get("items_per_s"),
    "same_process_unprofiled_vision_tokens_per_s": on_baseline.get("vision_tokens_per_s"),
    "profiled_items_per_s": profiled.get("items_per_s"),
    "profiled_vision_tokens_per_s": profiled.get("vision_tokens_per_s"),
    "profiled_total_s_over_same_process_unprofiled": effect.get("profiled_total_s_over_baseline"),
    "profiled_vision_tokens_per_s_over_same_process_unprofiled": effect.get("profiled_vision_tokens_per_s_over_baseline"),
}
print("VISION_PREFILL_PROFILE_COMPARE_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))

if "step_trace_time" in run:
    print("STEP_TRACE_TOTALS_US", json.dumps(run["step_trace_time"].get("totals_us", {}), sort_keys=True))

kernel = run.get("kernel_details", {})

def print_rows(label, rows, value_key="duration_us", limit=10):
    print(label)
    print("name count total_us")
    for row in rows[:limit]:
        name = str(row.get("name") or row.get("op_type") or "unknown").replace("\n", " ")[:180]
        print(f"{name} {row.get('count')} {row.get(value_key)}")

print_rows("TOP_KERNEL_TYPES", kernel.get("top_kernel_types", []))
print_rows("TOP_KERNEL_NAMES", kernel.get("top_kernel_names", []))
print_rows("TOP_MATMUL_SHAPES", kernel.get("top_matmul_shape_signatures", []), limit=8)
print_rows("TOP_TRANSDATA_SHAPES", kernel.get("top_transdata_shape_signatures", []), limit=8)
print_rows("TOP_SUSPECT_KERNELS", kernel.get("suspect_kernel_rows", []), limit=10)

operators = run.get("operator_details", {})
print_rows("TOP_OPERATORS_BY_DEVICE_US", operators.get("top_by_device_total_us", []), value_key="device_total_us", limit=10)
PY

echo "WROTE ${OFF_OUTPUT}"
echo "WROTE ${ON_OUTPUT}"
echo "WROTE ${PARSE_JSON}"
echo "WROTE ${PARSE_MD}"
