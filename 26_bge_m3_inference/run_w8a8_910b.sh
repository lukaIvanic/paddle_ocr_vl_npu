#!/usr/bin/env bash
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${RUN_ROOT:?Set a fresh absolute RUN_ROOT}"
[[ "$RUN_ROOT" = /* ]] || exit 2
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
{
  git rev-parse HEAD
  hostname
  printf 'Physical NPU: %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'Command: RUN_ROOT=%q bash %q\n' "$RUN_ROOT" "$0"
  printf 'PROFILE_KERNELS=%s\n' "${PROFILE_KERNELS:-0}"
  python3 -m pip show torch torch-npu transformers
  npu-smi info
} > "$RUN_ROOT/command.txt"
set +e
profile_args=()
if [[ "${PROFILE_KERNELS:-0}" = 1 ]]; then
  profile_args=(--profile-dir "$RUN_ROOT/profiles")
fi
timeout 1200 python3 -u 26_bge_m3_inference/benchmark_w8a8.py \
  --output "$RUN_ROOT/result.json" --cache "$RUN_ROOT/cache" "${profile_args[@]}" 2>&1 | tee "$RUN_ROOT/run.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
