#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
OPENOCR_ROOT="${OPENOCR_ROOT:-/workspace/repos/OpenOCR}"
PYTHON="${PYTHON:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
DATASET_DIR="${DATASET_DIR:-/workspace/datasets/OmniDocBench/images}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/tmp/12_unirec_0_1b_inference/opendoc_onnx_v16_full}"
OUTPUT_DIR="$RUN_DIR/output"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

command=(
  "$PYTHON" "$OPENOCR_ROOT/openocr.py"
  --task doc
  --input_path "$DATASET_DIR"
  --output_path "$OUTPUT_DIR"
  --use_gpu false
  --use_layout_detection
  --save_json
  --save_markdown
  --max_parallel_blocks 4
  --max_length 2048
)

{
  printf 'project_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'openocr_commit=%s\n' "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON"
  printf 'dataset_dir=%s\n' "$DATASET_DIR"
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$RUN_DIR/command.txt"

SECONDS=0
set +e
"${command[@]}" 2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
printf '%s\n' "$SECONDS" > "$RUN_DIR/wall_seconds.txt"
exit "$status"
