#!/usr/bin/env bash
# Select a currently free NPU; never disturb an existing server.
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${ASCEND_RT_VISIBLE_DEVICES:?No free NPU selected}"
: "${RUN_ROOT:?Choose a fresh run directory}"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
python=/workspace/venvs/decision2_eval_py312/bin/python
{
    git rev-parse HEAD
    hostname
    printf 'Physical NPU: %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'Command: RUN_ROOT=%q bash %q\n' "$RUN_ROOT" "$0"
    "$python" -m pip show torch torch-npu transformers
    npu-smi info
} > "$RUN_ROOT/command.txt"
set +e
timeout 600 "$python" -u 24_kalm_reranker/run_smoke.py \
    --model /workspace/models/KaLM-Reranker-V1-Nano-R2 \
    --dtype bf16 --output "$RUN_ROOT/result.json" 2>&1 | tee "$RUN_ROOT/run.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
