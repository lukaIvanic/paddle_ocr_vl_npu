#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

DATASET_DIR="${DATASET_DIR:-/home/lukaiv/datasets/OmniDocBench_current}"
PAGE_START="${PAGE_START:-0}"
NUM_PAGES="${NUM_PAGES:-64}"
OMNIDOCBENCH_REPO_ID="${OMNIDOCBENCH_REPO_ID:-opendatalab/OmniDocBench}"
OMNIDOCBENCH_REVISION="${OMNIDOCBENCH_REVISION:-main}"
EXPECT_GT_CROP_MANIFEST="${EXPECT_GT_CROP_MANIFEST:-${SCRIPT_DIR}/expected_omnidocbench_first64_gt_crops.json}"

if [[ "${PAGE_START}" != "0" || "${NUM_PAGES}" != "64" ]]; then
  cat >&2 <<MSG
restore_expected_omnidocbench_first64.sh is intentionally only for the known
first-64 OmniDocBench contract. Got PAGE_START=${PAGE_START} NUM_PAGES=${NUM_PAGES}.
Use prepare_omnidocbench_pages.py directly for other slices.
MSG
  exit 2
fi

echo "RESTORE_OMNIDOCBENCH PYTHON_BIN=${PYTHON_BIN}"
echo "RESTORE_OMNIDOCBENCH DATASET_DIR=${DATASET_DIR}"
echo "RESTORE_OMNIDOCBENCH REPO=${OMNIDOCBENCH_REPO_ID} REVISION=${OMNIDOCBENCH_REVISION}"
echo "RESTORE_OMNIDOCBENCH EXPECT_GT_CROP_MANIFEST=${EXPECT_GT_CROP_MANIFEST}"
echo "RESTORE_OMNIDOCBENCH HF_ENDPOINT=${HF_ENDPOINT:-unset}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_omnidocbench_pages.py" \
  --repo-id "${OMNIDOCBENCH_REPO_ID}" \
  --revision "${OMNIDOCBENCH_REVISION}" \
  --out-dir "${DATASET_DIR}" \
  --page-start 0 \
  --num-pages 64

"${PYTHON_BIN}" "${SCRIPT_DIR}/count_omnidocbench_gt_crops.py" \
  --dataset-dir "${DATASET_DIR}" \
  --page-start 0 \
  --num-pages 64 \
  --expect-manifest "${EXPECT_GT_CROP_MANIFEST}" \
  --json

echo "RESTORED_OMNIDOCBENCH_DATASET_DIR=${DATASET_DIR}"
