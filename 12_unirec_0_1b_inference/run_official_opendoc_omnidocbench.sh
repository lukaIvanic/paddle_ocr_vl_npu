#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
OPENOCR_ROOT="${OPENOCR_ROOT:-/workspace/repos/OpenOCR}"
PYTHON="${PYTHON:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
DATASET_DIR="${DATASET_DIR:-/workspace/datasets/OmniDocBench/images}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/tmp/12_unirec_0_1b_inference/opendoc_onnx_v16_full}"
OUTPUT_DIR="$RUN_DIR/output"
INPUT_DIR="$DATASET_DIR"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# OpenOCR writes each page immediately, but its directory runner does not skip
# completed pages. Build a symlink-only input directory on every invocation so
# a restart processes only pages without both final artifacts.
if [[ "${RESUME:-1}" == "1" ]]; then
  PENDING_DIR="$RUN_DIR/pending_input"
  mkdir -p "$PENDING_DIR"
  find "$PENDING_DIR" -mindepth 1 -maxdepth 1 -type l -delete

  pending_count=0
  completed_count=0
  for source_path in "$DATASET_DIR"/*; do
    [[ -f "$source_path" ]] || continue
    base_name="$(basename "$source_path")"
    stem="${base_name%.*}"
    if [[ -s "$OUTPUT_DIR/$stem/$stem.md" && -s "$OUTPUT_DIR/$stem/$stem.json" ]]; then
      ((completed_count += 1))
      continue
    fi
    ln -s "$source_path" "$PENDING_DIR/$base_name"
    ((pending_count += 1))
  done

  printf 'resume_completed=%d resume_pending=%d\n' "$completed_count" "$pending_count"
  if ((pending_count == 0)); then
    printf 'All dataset pages already have complete Markdown and JSON outputs.\n'
    exit 0
  fi
  INPUT_DIR="$PENDING_DIR"
fi

command=(
  "$PYTHON" "$OPENOCR_ROOT/openocr.py"
  --task doc
  --input_path "$INPUT_DIR"
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
  printf 'input_dir=%s\n' "$INPUT_DIR"
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
