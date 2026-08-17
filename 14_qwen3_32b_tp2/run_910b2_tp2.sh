#!/usr/bin/env bash

set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the official Qwen3-32B checkpoint directory}"

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp67s0f5}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp67s0f5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/python3.12.13/bin/torchrun}"

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=2 \
  "${SCRIPT_DIR}/probe_torchair_tp.py"

benchmark_args=(
  --model-dir "${MODEL_DIR}"
  --layers "${LAYERS:-1}"
  --cache-length "${CACHE_LENGTH:-4096}"
  --prefix-length "${PREFIX_LENGTH:-512}"
  --warmup-steps "${WARMUP_STEPS:-2}"
  --decode-steps "${DECODE_STEPS:-8}"
  --backend "${BACKEND:-torchair}"
)
if [[ -n "${JSON_OUT:-}" ]]; then
  benchmark_args+=(--json-out "${JSON_OUT}")
fi

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node=2 \
  "${SCRIPT_DIR}/benchmark_qwen3_32b_tp2.py" \
  "${benchmark_args[@]}"
