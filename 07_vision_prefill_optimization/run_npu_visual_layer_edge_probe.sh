#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/visual_layer_edge_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_ITEMS="${MAX_ITEMS:-1}"
export LN_LINEAR_MODE="${LN_LINEAR_MODE:-grouped_qkv_mlp_fc1}"
export STAGES="${STAGES:-qkv,qk_rope_v,attn_kernel_out,attn_out_proj,attn_residual,ln2,mlp_fc1,mlp_act,mlp_fc2,layer0_out}"
export TORCHAIR_RUN_EAGERLY="${TORCHAIR_RUN_EAGERLY:-0}"

mkdir -p "${OUT_ROOT}"

OUTPUT_JSON="${OUT_ROOT}/visual_layer_edge_probe_${LN_LINEAR_MODE}.json"
RUN_EAGERLY_ARGS=()
if [[ "${TORCHAIR_RUN_EAGERLY}" == "1" ]]; then
  RUN_EAGERLY_ARGS+=(--torchair-run-eagerly)
fi

echo "EXP07_VISUAL_LAYER_EDGE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_VISUAL_LAYER_EDGE MODEL=${MODEL}"
echo "EXP07_VISUAL_LAYER_EDGE BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_VISUAL_LAYER_EDGE DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_VISUAL_LAYER_EDGE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_VISUAL_LAYER_EDGE MAX_ITEMS=${MAX_ITEMS}"
echo "EXP07_VISUAL_LAYER_EDGE LN_LINEAR_MODE=${LN_LINEAR_MODE}"
echo "EXP07_VISUAL_LAYER_EDGE STAGES=${STAGES}"
echo "EXP07_VISUAL_LAYER_EDGE TORCHAIR_RUN_EAGERLY=${TORCHAIR_RUN_EAGERLY}"
echo "EXP07_VISUAL_LAYER_EDGE OUT_ROOT=${OUT_ROOT}"

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
  --static-visual-ln-linear-mode "${LN_LINEAR_MODE}" \
  --torchair-mode default \
  "${RUN_EAGERLY_ARGS[@]}" \
  --stages "${STAGES}" \
  --max-items "${MAX_ITEMS}" \
  --output "${OUTPUT_JSON}"

"${PYTHON_BIN}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
summary = data.get("summary", {})
print("EXP07_VISUAL_LAYER_EDGE SUMMARY")
print(f"output={path}")
print(f"mode={data.get('static_visual_ln_linear_mode')} backend={data.get('backend')}")
print(
    "compiled_second_matches_eager="
    f"{summary.get('compiled_second_matches_eager_count')}/{summary.get('compiled_cases')} "
    f"all={summary.get('compiled_second_matches_eager_all')}"
)
print(f"first_mismatch={summary.get('first_mismatch')}")
print("EXP07_VISUAL_LAYER_EDGE STAGE_TABLE")
for row in data.get("results", []):
    diff = row.get("compiled_second_vs_eager_before", {})
    print(
        f"item={row.get('item_index')} "
        f"stage={row.get('stage')} "
        f"ok={row.get('ok')} "
        f"match={row.get('compiled_second_matches_eager')} "
        f"shape={diff.get('shape')} "
        f"max_abs={diff.get('max_abs_diff')} "
        f"mean_abs={diff.get('mean_abs_diff')} "
        f"compiled_nonfinite={diff.get('lhs_nonfinite_count')} "
        f"compiled_min={row.get('compiled_second_summary', {}).get('finite_min')} "
        f"compiled_max={row.get('compiled_second_summary', {}).get('finite_max')} "
        f"error={row.get('error')}"
    )
PY

echo "EXP07_VISUAL_LAYER_EDGE OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
