#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_batching_audit_$(date -u +%Y%m%dT%H%M%SZ)}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export BATCH_SIZES="${BATCH_SIZES:-2,4,8}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT MODEL=${MODEL}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT BATCH_SIZES=${BATCH_SIZES}"
echo "EXP07_STATIC_VISUAL_BATCHING_AUDIT OUT_ROOT=${OUT_ROOT}"

OUTPUT_JSON="${OUT_ROOT}/batching_audit.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_static_visual_batching.py" \
  --model "${MODEL}" \
  --dataset-dir "${DATASET_DIR}" \
  --baseline "${BASELINE_DIR}" \
  --fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
  --batch-sizes "${BATCH_SIZES}" \
  --output "${OUTPUT_JSON}"

"${PYTHON_BIN}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
summary = data.get("summary", {})
print("EXP07_STATIC_VISUAL_BATCHING_AUDIT SUMMARY")
print(
    f"output={path} "
    f"fixedS={data.get('fixed_physical_seq_len')} "
    f"manifest={summary.get('manifest_item_count')} "
    f"eligible={summary.get('eligible_count')} "
    f"excluded={summary.get('excluded_count')} "
    f"same_shape_groups={summary.get('unique_same_shape_groups')} "
    f"group_p50={summary.get('group_size', {}).get('p50')} "
    f"group_max={summary.get('group_size', {}).get('max')}"
)
for row in summary.get("batch_size_summaries", []):
    print(
        "BATCH "
        f"B={row.get('batch_size')} "
        f"eligible={row.get('eligible_item_count')} "
        f"full_batches={row.get('full_batch_count')} "
        f"partial_batches={row.get('partial_batch_count')} "
        f"slot_utilization={row.get('slot_utilization_if_group_padded')} "
        f"singletons={row.get('singleton_group_count')}"
    )
for row in summary.get("top_groups", [])[:12]:
    print(
        "GROUP "
        f"count={row.get('count')} "
        f"grid={row.get('image_grid_thw')} "
        f"pixel_shape={row.get('pixel_values_shape')} "
        f"label={row.get('layout_label')} "
        f"ids={row.get('sample_ids')}"
    )
PY
