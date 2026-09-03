#!/usr/bin/env bash
# Run inside the existing vLLM-Ascend container after checking device ownership.
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Set exactly one verified free physical NPU}"
if [[ ! "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-7]$ ]]; then
  echo 'Expected one physical NPU ID, 0 through 7.' >&2
  exit 1
fi
table_vllm_max_seqs="${TABLE_VLLM_MAX_SEQS:-4}"
case "$table_vllm_max_seqs" in
  4) table_vllm_capture_sizes='[1,2,3,4]' ;;
  8) table_vllm_capture_sizes='[1,2,3,4,5,6,7,8]' ;;
  16) table_vllm_capture_sizes='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]' ;;
  *) echo 'TABLE_VLLM_MAX_SEQS must be 4, 8, or 16.' >&2; exit 1 ;;
esac
printf -v table_vllm_compilation_config \
  '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":%s}' \
  "$table_vllm_capture_sizes"
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
printf 'TABLE_VLLM_CONFIG max_seqs=%s compilation=%s\n' \
  "$table_vllm_max_seqs" "$table_vllm_compilation_config"
exec vllm serve /workspace/models/PaddleOCR-VL-1.6 \
  --served-model-name PaddleOCR-VL-1.6 \
  --trust-remote-code --dtype float16 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs "$table_vllm_max_seqs" \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --mm-processor-kwargs '{"min_pixels":28224,"max_pixels":802816}' \
  --compilation-config "$table_vllm_compilation_config" \
  --host 127.0.0.1 --port 18081
