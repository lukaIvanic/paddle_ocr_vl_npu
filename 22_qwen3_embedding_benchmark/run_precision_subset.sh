#!/usr/bin/env bash
# Separate service/device from the full-suite FP16 run.
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${RUN_ROOT:?Choose a fresh diagnostics directory}"
: "${BASELINE:?Set the saved subset directory}"
mkdir "$RUN_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/workspace/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
BASE_PYTHON=/usr/local/python3.12.13/bin/python3
EVAL_PYTHON=/workspace/venvs/qwen3_embedding_eval_py312/bin/python
SERVER_PID=
cleanup() {
  code=$?
  printf '%s\n' "$code" > "$RUN_ROOT/exit_code.txt"
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
{ git rev-parse HEAD; hostname; printf 'physical_device=%s\nbaseline=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" "$BASELINE"; } > "$RUN_ROOT/command.txt"
"$BASE_PYTHON" -c 'import torch; from vllm_ascend.attention.attention_v1 import AscendAttentionBackend as B; print({"supported_dtypes": [str(x) for x in B.supported_dtypes], "fp32_supported": B.supports_dtype(torch.float32), "bf16_supported": B.supports_dtype(torch.bfloat16)})' > "$RUN_ROOT/vllm_dtype_support.log" 2>&1
"$BASE_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-embedding-bf16-diagnostic --host 127.0.0.1 --port 18223 \
  --runner pooling --convert embed --dtype bfloat16 --max-model-len 8192 \
  --pooler-config '{"pooling_type":"LAST","use_activation":true}' \
  --enforce-eager --gpu-memory-utilization 0.35 --max-num-seqs 32 --block-size 128 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --no-enable-chunked-prefill \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for ((attempt=0;attempt<180;attempt++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -70 "$RUN_ROOT/server.log"; exit 1; fi
  if curl --fail --silent http://127.0.0.1:18223/health >/dev/null; then break; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:18223/health >/dev/null
"$EVAL_PYTHON" -u 22_qwen3_embedding_benchmark/compare_precision_subset.py \
  --baseline "$BASELINE" --output "$RUN_ROOT/evaluation" 2>&1 | tee "$RUN_ROOT/run.log"
