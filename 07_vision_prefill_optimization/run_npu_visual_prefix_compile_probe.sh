#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/visual_prefix_compile_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
export STAGES="${STAGES:-patch_conv,patch_flat,patch_pad,patch_pos,ln1}"
export MAX_ITEMS="${MAX_ITEMS:-1}"
export RUN_TORCHAIR_EAGERLY="${RUN_TORCHAIR_EAGERLY:-1}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_VISUAL_PREFIX_PROBE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_VISUAL_PREFIX_PROBE MODEL=${MODEL}"
echo "EXP07_VISUAL_PREFIX_PROBE BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_VISUAL_PREFIX_PROBE DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_VISUAL_PREFIX_PROBE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_VISUAL_PREFIX_PROBE OUT_ROOT=${OUT_ROOT}"
echo "EXP07_VISUAL_PREFIX_PROBE MAX_ITEMS=${MAX_ITEMS} STAGES=${STAGES}"
echo "EXP07_VISUAL_PREFIX_PROBE RUN_TORCHAIR_EAGERLY=${RUN_TORCHAIR_EAGERLY}"

run_probe() {
  local name="$1"
  shift
  local output_json="${OUT_ROOT}/${name}.json"
  echo "EXP07_VISUAL_PREFIX_PROBE RUN name=${name} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" probe-visual-prefix-compile \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --baseline "${BASELINE_DIR}" \
    --device "${DEVICE}" \
    --dtype fp16 \
    --npu-jit-compile off \
    --vision-attention prompt_flash_attention \
    --vision-prompt-fa-layout bnsd \
    --vision-prompt-fa-mask-sparse-mode 1 \
    --vision-compile-backend torchair \
    --torchair-mode default \
    --max-items "${MAX_ITEMS}" \
    --stages "${STAGES}" \
    --output "${output_json}" \
    "$@"
}

run_probe torchair_default
if [[ "${RUN_TORCHAIR_EAGERLY}" != "0" ]]; then
  run_probe torchair_run_eagerly --torchair-run-eagerly
fi

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_VISUAL_PREFIX_PROBE SUMMARY")
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    print(
        f"{path.name}: "
        f"ok={summary.get('ok_cases')}/{summary.get('total_cases')} "
        f"compiled_match={summary.get('compiled_second_matches_eager_count')}/"
        f"{summary.get('compiled_cases')} "
        f"nonfinite_cases={summary.get('compiled_nonfinite_case_count')} "
        f"errors={summary.get('error_cases')}"
    )
    first = summary.get("first_mismatch")
    if first:
        print(
            "  FIRST_MISMATCH "
            f"item={first.get('item_index')} id={first.get('id')} stage={first.get('stage')} "
            f"shape={first.get('shape')} max_abs={first.get('max_abs_diff')} "
            f"mean_abs={first.get('mean_abs_diff')} "
            f"compiled_nonfinite={first.get('compiled_nonfinite_count')}"
        )
    for item in summary.get("failed_case_keys", [])[:8]:
        print(
            "  ERROR "
            f"item={item.get('item_index')} id={item.get('id')} stage={item.get('stage')} "
            f"error={item.get('error')}"
        )
PY

echo "EXP07_VISUAL_PREFIX_PROBE OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
