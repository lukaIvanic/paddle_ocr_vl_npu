#!/usr/bin/env bash

set -uo pipefail

REPO="$(git rev-parse --show-toplevel)" || exit 1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODEL_06B_DIR="${MODEL_06B_DIR:-}"
MODEL_4B_DIR="${MODEL_4B_DIR:-}"
MATRIX_MODEL="${MATRIX_MODEL:-}"
MATRIX_MODE="${MATRIX_MODE:-}"
MATRIX_BATCH="${MATRIX_BATCH:-}"
DEVICE="${DEVICE:-npu:0}"
WARMUPS="${WARMUPS:-3}"
REPEATS="${REPEATS:-20}"
LENGTHS="${LENGTHS:-128,256,384,512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
REUSE_RUN_ROOT="${REUSE_RUN_ROOT:-}"

case "$MATRIX_MODEL" in
  06b)
    model_dir="$MODEL_06B_DIR"
    ;;
  4b)
    model_dir="$MODEL_4B_DIR"
    ;;
  *)
    echo "MATRIX_MODEL must be 06b or 4b" >&2
    exit 2
    ;;
esac

case "$MATRIX_MODE" in
  dense)
    ffn_mode="dense"
    ;;
  w8a8)
    ffn_mode="gate_up_w8a8"
    ;;
  *)
    echo "MATRIX_MODE must be dense or w8a8" >&2
    exit 2
    ;;
esac

case "$MATRIX_BATCH" in
  1|2|4|8) ;;
  *)
    echo "MATRIX_BATCH must be one of 1,2,4,8" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT is required so parallel workers share one result tree" >&2
  exit 2
fi
if [[ -z "$model_dir" || ! -f "$model_dir/config.json" ]]; then
  echo "model directory for $MATRIX_MODEL is unset or does not contain config.json" >&2
  exit 2
fi

phase_key="${MATRIX_MODEL}_${MATRIX_MODE}"
worker_dir="$OUTPUT_ROOT/$phase_key/b$MATRIX_BATCH"
if [[ -n "$REUSE_RUN_ROOT" ]]; then
  compile_cache_dir="$REUSE_RUN_ROOT/cache/$phase_key"
else
  compile_cache_dir="$REPO/.runtime_cache/13_qwen3_reranker/310p_matrix/$phase_key"
fi
mkdir -p "$worker_dir" "$compile_cache_dir"

command=(
  "$PYTHON_BIN"
  "$REPO/13_qwen3_reranker/benchmark_prefix_cache_throughput.py"
  --model-dir "$model_dir"
  --device "$DEVICE"
  --batch-sizes "$MATRIX_BATCH"
  --continuation-lengths "$LENGTHS"
  --batch-sweep-continuation 128
  --length-sweep-batch "$MATRIX_BATCH"
  --matrix cross
  --lanes prefix_promptfa_compiled
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --compile-cache-dir "$compile_cache_dir"
  --prefill-optimizations combined_bsnd
  --linear-weight-format fractal_nz
  --enable-internal-format
  --ffn-weight-mode "$ffn_mode"
  --json-out "$worker_dir/result.json"
)

{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "hostname=$(hostname)"
  echo "python_bin=$PYTHON_BIN"
  echo "device=$DEVICE"
  echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
  echo "model=$MATRIX_MODEL"
  echo "mode=$MATRIX_MODE"
  echo "batch=$MATRIX_BATCH"
  echo "lengths=$LENGTHS"
  echo "compile_cache_dir=$compile_cache_dir"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$worker_dir/command.txt"

echo "MATRIX_WORKER_START model=$MATRIX_MODEL mode=$MATRIX_MODE B=$MATRIX_BATCH physical_npu=${ASCEND_RT_VISIBLE_DEVICES:-unset} cache=$compile_cache_dir"
set +e
"${command[@]}" 2>&1 |
  while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == THROUGHPUT\ * ]]; then
      printf 'MATRIX_CELL model=%s mode=%s physical_npu=%s %s\n' \
        "$MATRIX_MODEL" "$MATRIX_MODE" "${ASCEND_RT_VISIBLE_DEVICES:-unset}" "$line"
    fi
  done |
  tee "$worker_dir/run.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$worker_dir/exit_code.txt"
echo "MATRIX_WORKER_END model=$MATRIX_MODEL mode=$MATRIX_MODE B=$MATRIX_BATCH physical_npu=${ASCEND_RT_VISIBLE_DEVICES:-unset} exit_code=$status"
exit "$status"
