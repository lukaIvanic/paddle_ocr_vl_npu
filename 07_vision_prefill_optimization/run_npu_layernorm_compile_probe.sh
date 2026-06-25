#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/layernorm_compile_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
export SEQ_LENS="${SEQ_LENS:-580,640,768}"
export IMPLS="${IMPLS:-nn,functional,manual,npu_eval}"
export REAL_ITEM_INDEX="${REAL_ITEM_INDEX:-0}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_LAYERNORM_PROBE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_LAYERNORM_PROBE MODEL=${MODEL}"
echo "EXP07_LAYERNORM_PROBE BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_LAYERNORM_PROBE DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_LAYERNORM_PROBE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_LAYERNORM_PROBE OUT_ROOT=${OUT_ROOT}"
echo "EXP07_LAYERNORM_PROBE SEQ_LENS=${SEQ_LENS} IMPLS=${IMPLS} REAL_ITEM_INDEX=${REAL_ITEM_INDEX}"

run_probe() {
  local name="$1"
  shift
  local output_json="${OUT_ROOT}/${name}.json"
  echo "EXP07_LAYERNORM_PROBE RUN name=${name} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" probe-layernorm-compile \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --baseline "${BASELINE_DIR}" \
    --device "${DEVICE}" \
    --dtype fp16 \
    --npu-jit-compile off \
    --vision-compile-backend torchair \
    --torchair-mode default \
    --seq-lens "${SEQ_LENS}" \
    --impls "${IMPLS}" \
    --include-real-first-crop \
    --real-item-index "${REAL_ITEM_INDEX}" \
    --output "${output_json}" \
    "$@"
}

run_probe torchair_default
run_probe torchair_run_eagerly --torchair-run-eagerly

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_LAYERNORM_PROBE SUMMARY")
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
    for item in summary.get("mismatch_case_keys", [])[:8]:
        print(
            "  MISMATCH "
            f"case={item.get('case_index')} impl={item.get('impl')} source={item.get('source')} "
            f"shape={item.get('shape')} max_abs={item.get('max_abs_diff')} "
            f"compiled_nonfinite={item.get('compiled_nonfinite_count')}"
        )
    for item in summary.get("failed_case_keys", [])[:8]:
        print(
            "  ERROR "
            f"case={item.get('case_index')} impl={item.get('impl')} source={item.get('source')} "
            f"error={item.get('error')}"
        )
PY

echo "EXP07_LAYERNORM_PROBE OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
