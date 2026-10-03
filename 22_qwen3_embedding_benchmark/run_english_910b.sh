#!/usr/bin/env bash
# Only use after checking that the explicitly reserved devices are still free.
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec /workspace/venvs/qwen3_embedding_eval_py312/bin/python -u \
  22_qwen3_embedding_benchmark/run_english_suite.py "$@"
