#!/usr/bin/env bash
# Run inside the active research container; source is pulled through Git.
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${RUN_ROOT:?Set RUN_ROOT to a fresh absolute output directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?No idle NPU selected}"
MODEL_DIR="${MODEL_DIR:-/workspace/model_downloads/bge-m3}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
[[ "$RUN_ROOT" = /* ]] || { echo 'RUN_ROOT must be absolute' >&2; exit 2; }
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
{
  git rev-parse HEAD
  hostname
  printf 'Physical NPU: %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'Command: RUN_ROOT=%q MODEL_DIR=%q PYTHON_BIN=%q bash %q\n' "$RUN_ROOT" "$MODEL_DIR" "$PYTHON_BIN" "$0"
  "$PYTHON_BIN" -m pip show torch torch-npu transformers
  npu-smi info
} > "$RUN_ROOT/command.txt"
set +e
timeout 900 "$PYTHON_BIN" -u 26_bge_m3_inference/validate_910b.py \
  --model-dir "$MODEL_DIR" --compile-cache "$RUN_ROOT/cache" \
  --output "$RUN_ROOT/result.json" 2>&1 | tee "$RUN_ROOT/run.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
