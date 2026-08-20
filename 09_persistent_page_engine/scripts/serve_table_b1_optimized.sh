#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
MODEL="${MODEL:-/workspace/models/PaddleOCR-VL-1.6}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8767}"
QUEUE_CAPACITY="${QUEUE_CAPACITY:-2048}"
VOCAB_PRESET="${VOCAB_PRESET:-$REPO_ROOT/09_persistent_page_engine/presets/table_compact_vocab/b1_verifier_topfreq_16384.json}"

: "${ASCEND_RT_VISIBLE_DEVICES:?Run 'source npu-setup' before starting the server}"
test -f "$VOCAB_PRESET"

cd "$REPO_ROOT"
echo "Starting optimized B1 table API on $HOST:$PORT logical_npu=$ASCEND_RT_VISIBLE_DEVICES"

exec "$PYTHON" 09_persistent_page_engine/scripts/serve_crop_ocr_api.py \
  --host "$HOST" \
  --port "$PORT" \
  --request-timeout-s 3600 \
  --queue-capacity "$QUEUE_CAPACITY" \
  --model "$MODEL" \
  --decode-backend torchair \
  --decode-optimization combined_apply_complete_layer_prefetch1_rope_lut \
  --decode-vocab-token-ids "$VOCAB_PRESET" \
  --token-selection greedy \
  --decode-batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 4096 \
  --min-pixels 28224 \
  --max-pixels 802816
