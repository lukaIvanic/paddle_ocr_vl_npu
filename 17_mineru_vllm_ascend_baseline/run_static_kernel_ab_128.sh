#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set +u
source npu-setup
set -u
if [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" == "5" ]]; then
  echo "physical NPU5 is quarantined; refusing to run" >&2
  exit 1
fi

export EXP17_NPU_SETUP_ALREADY_SOURCED=1

printf '[exp17-ab] physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
printf '[exp17-ab] static-kernel off cache warmup, 1 page\n'
STATIC_KERNEL=off LIMIT=1 bash "$SCRIPT_DIR/run_npu_reproduction.sh"

printf '[exp17-ab] static-kernel off measurement, 128 pages\n'
STATIC_KERNEL=off LIMIT=128 bash "$SCRIPT_DIR/run_npu_reproduction.sh"

printf '[exp17-ab] static-kernel on measurement, 128 pages\n'
STATIC_KERNEL=on LIMIT=128 bash "$SCRIPT_DIR/run_npu_reproduction.sh"

printf '[exp17-ab] complete physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
