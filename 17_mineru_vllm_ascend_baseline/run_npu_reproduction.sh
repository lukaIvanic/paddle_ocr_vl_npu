#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
EXPERIMENT_NAME="17_mineru_vllm_ascend_baseline"
PYTHON="${PYTHON:-/workspace/venvs/mineru_vllm_ascend_exp17_py312/bin/python}"
MODE="${MODE:-compiled_async}"
LIMIT="${LIMIT:-1}"
OFFSET="${OFFSET:-0}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/MinerU2.5-Pro-2605-1.2B}"
DATASET_JSON="${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/datasets/OmniDocBench/images}"
IMAGE_LIST="${IMAGE_LIST:-}"
HASH_MODEL_FILES="${HASH_MODEL_FILES:-0}"

set +u
source npu-setup
set -u
if [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" == "5" ]]; then
  echo "physical NPU5 is quarantined; refusing to run" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "experiment environment is missing: $PYTHON" >&2
  exit 1
fi
export HI_PYTHON="$PYTHON"

"$PYTHON" "$SCRIPT_DIR/verify_environment.py"

COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LIMIT_LABEL="${LIMIT//[^a-zA-Z0-9_-]/_}"
RUN_DIR="$REPO_ROOT/tmp/$EXPERIMENT_NAME/${MODE}_n${LIMIT_LABEL}_${STAMP}_${COMMIT}"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$RUN_DIR"

COMMAND=(
  "$PYTHON"
  "$SCRIPT_DIR/run_omnidocbench.py"
  --mode "$MODE"
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$OUTPUT_DIR"
  --offset "$OFFSET"
)
if [[ "$LIMIT" != "all" ]]; then
  COMMAND+=(--limit "$LIMIT")
fi
if [[ -n "$IMAGE_LIST" ]]; then
  COMMAND+=(--image-list "$IMAGE_LIST")
fi
if [[ "$HASH_MODEL_FILES" == "1" ]]; then
  COMMAND+=(--hash-model-files)
fi

{
  printf 'git_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ascend_rt_visible_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} >"$RUN_DIR/command.txt"

printf '[exp17] run_dir=%s\n' "$RUN_DIR"
set +e
(cd "$RUN_DIR" && "${COMMAND[@]}") 2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" >"$RUN_DIR/exit_code.txt"

if [[ "$STATUS" -ne 0 ]]; then
  echo "EXPERIMENT17_FAILED run_dir=$RUN_DIR status=$STATUS" >&2
  exit "$STATUS"
fi
echo "EXPERIMENT17_COMPLETE run_dir=$RUN_DIR"
