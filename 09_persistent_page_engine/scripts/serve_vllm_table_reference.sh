#!/usr/bin/env bash
# Run inside the existing vLLM-Ascend container after checking device ownership.
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Set exactly one verified free physical NPU}"
if [[ ! "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-7]$ ]]; then
  echo 'Expected one physical NPU ID, 0 through 7.' >&2
  exit 1
fi
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
exec vllm serve /workspace/models/PaddleOCR-VL-1.6 \
  --served-model-name PaddleOCR-VL-1.6 \
  --trust-remote-code --dtype float16 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --mm-processor-kwargs '{"min_pixels":28224,"max_pixels":802816}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}' \
  --host 127.0.0.1 --port 18081
