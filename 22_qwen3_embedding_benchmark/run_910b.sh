#!/usr/bin/env bash
# Vendor environment scripts are not nounset/errexit-safe.
source npu-setup || exit 1
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?NPU selection did not succeed}"
cd "$(git rev-parse --show-toplevel)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
RUN_ROOT=${RUN_ROOT:?Set a fresh RUN_ROOT}
mkdir -p "$RUN_ROOT"
BASE_PYTHON=/usr/local/python3.12.13/bin/python3
EVAL_PYTHON=/workspace/venvs/qwen3_embedding_eval_py312/bin/python
SERVER_PID=
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
{
  git rev-parse HEAD
  hostname
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'Invocation: %q ' "$0" "$@"
  printf '\n'
  "$BASE_PYTHON" -m pip show vllm vllm-ascend torch torch-npu transformers
} > "$RUN_ROOT/command.txt"
"$BASE_PYTHON" 22_qwen3_embedding_benchmark/verify_checkpoint.py \
  /workspace/models/Qwen3-Embedding-0.6B > "$RUN_ROOT/checkpoint_verified.json"
"$BASE_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-embedding-0.6b --host 127.0.0.1 --port 18222 \
  --runner pooling --convert embed --dtype float16 --max-model-len 8192 \
  --pooler-config '{"pooling_type":"LAST","use_activation":true}' \
  --enforce-eager --gpu-memory-utilization 0.35 --max-num-seqs 32 --block-size 128 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --no-enable-chunked-prefill \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for ((attempt=0;attempt<180;attempt++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -80 "$RUN_ROOT/server.log"
    exit 1
  fi
  if curl --fail --silent http://127.0.0.1:18222/health >/dev/null; then break; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:18222/health >/dev/null
set +e
"$EVAL_PYTHON" -u 22_qwen3_embedding_benchmark/run_evaluation.py \
  --output "$RUN_ROOT/evaluation" "$@" 2>&1 | tee "$RUN_ROOT/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
