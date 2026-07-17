#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
export DEVICE="${DEVICE:-npu:0}"
export ROWS="${ROWS:-32,64,128,256,512,1024,2048}"
export WEIGHT_LAYOUTS="${WEIGHT_LAYOUTS:-nd_kn,nz_kn,nz_nk_transposed}"
export WARMUP="${WARMUP:-10}"
export ITERATIONS="${ITERATIONS:-50}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/w8a8_vision_linears_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${OUT_ROOT}"
output_json="${OUT_ROOT}/w8a8_vision_linears.json"

echo "W8A8_VISION_LINEARS PYTHON_BIN=${PYTHON_BIN}"
echo "W8A8_VISION_LINEARS DEVICE=${DEVICE}"
echo "W8A8_VISION_LINEARS ROWS=${ROWS}"
echo "W8A8_VISION_LINEARS WEIGHT_LAYOUTS=${WEIGHT_LAYOUTS}"
echo "W8A8_VISION_LINEARS WARMUP=${WARMUP} ITERATIONS=${ITERATIONS}"
echo "W8A8_VISION_LINEARS OUTPUT=${output_json}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_w8a8_vision_linears.py" \
  --device "${DEVICE}" \
  --rows "${ROWS}" \
  --weight-layouts "${WEIGHT_LAYOUTS}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --output "${output_json}"

echo "W8A8_VISION_LINEARS_DONE OUTPUT=${output_json}"
