#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/qkv_linear_compile_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
export ITEM_INDEX="${ITEM_INDEX:-0}"
export SOURCES="${SOURCES:-patch_pos}"
export BRIDGES="${BRIDGES:-none,format_cast_nd,format_cast_nz_then_nd,transpose_roundtrip}"
export LN_IMPLS="${LN_IMPLS:-module}"
export IMPLS="${IMPLS:-functional_q}"
export NPU_MM_BMM_FORMAT_ND="${NPU_MM_BMM_FORMAT_ND:-default}"
export RUN_TORCHAIR_EAGERLY="${RUN_TORCHAIR_EAGERLY:-1}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_QKV_LINEAR_PROBE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_QKV_LINEAR_PROBE MODEL=${MODEL}"
echo "EXP07_QKV_LINEAR_PROBE BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_QKV_LINEAR_PROBE DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_QKV_LINEAR_PROBE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_QKV_LINEAR_PROBE OUT_ROOT=${OUT_ROOT}"
echo "EXP07_QKV_LINEAR_PROBE ITEM_INDEX=${ITEM_INDEX}"
echo "EXP07_QKV_LINEAR_PROBE SOURCES=${SOURCES}"
echo "EXP07_QKV_LINEAR_PROBE BRIDGES=${BRIDGES}"
echo "EXP07_QKV_LINEAR_PROBE LN_IMPLS=${LN_IMPLS}"
echo "EXP07_QKV_LINEAR_PROBE IMPLS=${IMPLS}"
echo "EXP07_QKV_LINEAR_PROBE NPU_MM_BMM_FORMAT_ND=${NPU_MM_BMM_FORMAT_ND}"
echo "EXP07_QKV_LINEAR_PROBE RUN_TORCHAIR_EAGERLY=${RUN_TORCHAIR_EAGERLY}"

run_probe() {
  local name="$1"
  shift
  local output_json="${OUT_ROOT}/${name}.json"
  echo "EXP07_QKV_LINEAR_PROBE RUN name=${name} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" probe-qkv-linear-compile \
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
    --npu-mm-bmm-format-nd "${NPU_MM_BMM_FORMAT_ND}" \
    --item-index "${ITEM_INDEX}" \
    --sources "${SOURCES}" \
    --bridges "${BRIDGES}" \
    --ln-impls "${LN_IMPLS}" \
    --impls "${IMPLS}" \
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
print("EXP07_QKV_LINEAR_PROBE SUMMARY")
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
            f"source={first.get('source')} ln_impl={first.get('ln_impl')} "
            f"bridge={first.get('bridge')} impl={first.get('impl')} "
            f"shape={first.get('shape')} max_abs={first.get('max_abs_diff')} "
            f"mean_abs={first.get('mean_abs_diff')} "
            f"compiled_nonfinite={first.get('compiled_nonfinite_count')} "
            f"compiled_min={first.get('compiled_finite_min')} "
            f"compiled_max={first.get('compiled_finite_max')}"
        )
    for item in summary.get("by_source_impl", []):
        print(
            "  CASE "
            f"source={item.get('source')} ln_impl={item.get('ln_impl')} "
            f"bridge={item.get('bridge')} impl={item.get('impl')} "
            f"ok={item.get('ok')} match={item.get('compiled_second_matches_eager')} "
            f"max_abs={item.get('max_abs_diff')} "
            f"compiled_min={item.get('compiled_finite_min')} "
            f"compiled_max={item.get('compiled_finite_max')}"
        )
    for item in summary.get("failed_case_keys", [])[:8]:
        print(
            "  ERROR "
            f"source={item.get('source')} ln_impl={item.get('ln_impl')} "
            f"bridge={item.get('bridge')} impl={item.get('impl')} "
            f"error={item.get('error')}"
        )
PY

echo "EXP07_QKV_LINEAR_PROBE OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
