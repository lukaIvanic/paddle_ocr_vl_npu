#!/usr/bin/env bash

set -uo pipefail

REPO="$(git rev-parse --show-toplevel)" || exit 1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODEL_06B_DIR="${MODEL_06B_DIR:-}"
DEVICE="${DEVICE:-npu:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-128}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-10}"
COMMIT="$(git rev-parse --short=12 HEAD)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/tmp/13_qwen3_reranker/310p_native_gqa_${COMMIT}_${RUN_STAMP}}"

if [[ -z "$MODEL_06B_DIR" || ! -f "$MODEL_06B_DIR/config.json" ]]; then
  echo "MODEL_06B_DIR must point to a complete Qwen3-Reranker-0.6B directory" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "hostname=$(hostname)"
  echo "python_bin=$PYTHON_BIN"
  echo "device=$DEVICE"
  echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
  echo "model_06b_dir=$MODEL_06B_DIR"
  printf 'command='
  printf '%q ' "$PYTHON_BIN" "$REPO/13_qwen3_reranker/probe_310p_promptfa_native_gqa.py" \
    --model-dir "$MODEL_06B_DIR" --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" --sequence-length "$SEQUENCE_LENGTH" \
    --warmups "$WARMUPS" --repeats "$REPEATS" \
    --compile-cache-dir "$OUTPUT_ROOT/cache" \
    --json-out "$OUTPUT_ROOT/result.json"
  printf '\n'
} > "$OUTPUT_ROOT/command.txt"

echo "PROBE_START output_root=$OUTPUT_ROOT"
set +e
"$PYTHON_BIN" "$REPO/13_qwen3_reranker/probe_310p_promptfa_native_gqa.py" \
  --model-dir "$MODEL_06B_DIR" --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" --sequence-length "$SEQUENCE_LENGTH" \
  --warmups "$WARMUPS" --repeats "$REPEATS" \
  --compile-cache-dir "$OUTPUT_ROOT/cache" \
  --json-out "$OUTPUT_ROOT/result.json" 2>&1 | tee "$OUTPUT_ROOT/run.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$OUTPUT_ROOT/exit_code.txt"
echo "PROBE_END exit_code=$status output_root=$OUTPUT_ROOT"
exit "$status"
