#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
LIMIT="${LIMIT:-1}"
STATIC_KERNEL="${STATIC_KERNEL:-off}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ "$BLOCK_SIZE" != "128" ]]; then
  echo "310P compatibility ladder requires BLOCK_SIZE=128" >&2
  exit 2
fi
if [[ "$LIMIT" != "1" ]]; then
  echo "310P compatibility ladder is a one-page gate and requires LIMIT=1" >&2
  exit 2
fi
if [[ "$STATIC_KERNEL" != "off" ]]; then
  echo "310P compatibility ladder requires STATIC_KERNEL=off" >&2
  exit 2
fi

export BLOCK_SIZE
export LIMIT
export STATIC_KERNEL
export VLLM_WORKER_MULTIPROC_METHOD

run_gate() {
  local mode="$1"
  printf '[310p-compat] starting mode=%s block_size=%s multiproc=%s\n' \
    "$mode" "$BLOCK_SIZE" "$VLLM_WORKER_MULTIPROC_METHOD"
  MODE="$mode" bash "$SCRIPT_DIR/run_npu_reproduction.sh"
  printf '[310p-compat] passed mode=%s\n' "$mode"
}

run_gate eager_sync
run_gate eager_async
run_gate aclgraph_async

printf 'EXPERIMENT17_310P_COMPATIBILITY_LADDER_COMPLETE block_size=%s multiproc=%s\n' \
  "$BLOCK_SIZE" "$VLLM_WORKER_MULTIPROC_METHOD"
