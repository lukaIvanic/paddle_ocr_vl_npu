#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CACHE_LENGTH="${CACHE_LENGTH:-1269}"
TORCHAIR_CACHE_DIR="${TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_hotswap_bottleneck_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/hotswap_bottleneck_matrix}"

mkdir -p "${OUTPUT_DIR}"

COMMON=(
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --batch-size "${BATCH_SIZE}"
  --backend torchair
  --device "${DEVICE}"
  --dtype fp16
  --npu-jit-compile off
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --cache-length "${CACHE_LENGTH}"
  --eos-mode overlap_event_flags
  --step-timing both
  --torchair-cache-dir "${TORCHAIR_CACHE_DIR}"
  --report summary
  --json
)

run_case() {
  local name="$1"
  shift
  local output_path="${OUTPUT_DIR}/${name}.json"

  echo "BEGIN ${name}"
  echo "COMMAND ${PYTHON_BIN} ${SCRIPT_DIR}/bench_static_compile.py $* ${COMMON[*]}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_static_compile.py" "$@" "${COMMON[@]}" | tee "${output_path}"
  echo "END ${name}"
  echo "WROTE ${output_path}"
}

echo "NPU hot-swap bottleneck matrix"
echo "python=${PYTHON_BIN}"
echo "model=${MODEL}"
echo "manifest=${MANIFEST}"
echo "device=${DEVICE}"
echo "batch_size=${BATCH_SIZE}"
echo "cache_length=${CACHE_LENGTH}"
echo "torchair_cache_dir=${TORCHAIR_CACHE_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo

run_case 01_fixed_compile_or_warmup --schedule fixed_cohort
run_case 02_fixed_warm_reference --schedule fixed_cohort
run_case 03_hotswap_no_replacement_8 --schedule hotswap --num-items 8
run_case 04_hotswap_one_extra_9 --schedule hotswap --num-items 9
run_case 05_hotswap_one_wave_16 --schedule hotswap --num-items 16
run_case 06_hotswap_several_waves_32 --schedule hotswap --num-items 32
run_case 07_hotswap_full_100 --schedule hotswap --num-items 100

cat <<'EOF'

Matrix complete.

Paste back:
- each JSON's correctness object
- tok_per_s
- speed_debug
- loop.step_timing_summary
- timing_s
- timing_accounting

Do not post-process with inline scripts. If a run fails, paste the failing JSON
or traceback and stop.
EOF
