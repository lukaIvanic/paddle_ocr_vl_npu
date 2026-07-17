#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
ROWS="${ROWS:-4096}"
BACKEND="${BACKEND:-npu}"
WEIGHT_LAYOUT="${WEIGHT_LAYOUT:-nd_kn}"
QUANTIZATION="${QUANTIZATION:-w8a8_static}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/outputs/compiled_w8a8_fc1_rows_${ROWS}.json}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_compiled_w8a8_fc1.py" \
  --device npu:0 \
  --rows "${ROWS}" \
  --backend "${BACKEND}" \
  --weight-layout "${WEIGHT_LAYOUT}" \
  --quantization "${QUANTIZATION}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --output "${OUTPUT}"
