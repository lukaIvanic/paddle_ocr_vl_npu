#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_grouped_compare_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_ITEMS="${MAX_ITEMS:-4}"
export STATIC_VISUAL_LN_IMPL="${STATIC_VISUAL_LN_IMPL:-module}"
export PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-0}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_STATIC_VISUAL_GROUPED PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_STATIC_VISUAL_GROUPED MODEL=${MODEL}"
echo "EXP07_STATIC_VISUAL_GROUPED BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_STATIC_VISUAL_GROUPED DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_STATIC_VISUAL_GROUPED DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_STATIC_VISUAL_GROUPED MAX_ITEMS=${MAX_ITEMS}"
echo "EXP07_STATIC_VISUAL_GROUPED STATIC_VISUAL_LN_IMPL=${STATIC_VISUAL_LN_IMPL}"
echo "EXP07_STATIC_VISUAL_GROUPED PROMPTFA_PAD_HEAD_DIM_TO=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_STATIC_VISUAL_GROUPED OUT_ROOT=${OUT_ROOT}"

run_compare() {
  local name="$1"
  local backend="$2"
  local ln_linear_mode="$3"
  local output_json="${OUT_ROOT}/${name}.json"
  echo "EXP07_STATIC_VISUAL_GROUPED RUN name=${name} backend=${backend} mode=${ln_linear_mode} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/vision_prefill_bench.py" compare \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --baseline "${BASELINE_DIR}" \
    --device "${DEVICE}" \
    --dtype fp16 \
    --npu-jit-compile off \
    --vision-attention prompt_flash_attention \
    --vision-prompt-fa-layout bnsd \
    --vision-prompt-fa-mask-sparse-mode 1 \
    --vision-compile-backend "${backend}" \
    --static-visual-ln-impl "${STATIC_VISUAL_LN_IMPL}" \
    --static-visual-ln-linear-mode "${ln_linear_mode}" \
    --static-visual-promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}" \
    --torchair-mode default \
    --validate-compiled-against-static-eager \
    --candidate-name "${name}" \
    --max-items "${MAX_ITEMS}" \
    --repeats 1 \
    --warmup-repeats 0 \
    --output "${output_json}"
}

run_compare eager_grouped_qkv_mlp_fc1 none grouped_qkv_mlp_fc1
run_compare torchair_grouped_qkv torchair grouped_qkv
run_compare torchair_grouped_qkv_mlp_fc1 torchair grouped_qkv_mlp_fc1

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_STATIC_VISUAL_GROUPED SUMMARY")
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    candidate = data.get("candidate", {})
    summary = data.get("summary", {})
    print(
        f"{path.name}: "
        f"backend={candidate.get('vision_compile_backend')} "
        f"ln_impl={candidate.get('static_visual_ln_impl')} "
        f"mode={candidate.get('static_visual_ln_linear_mode')} "
        f"promptfa_padD={candidate.get('static_visual_promptfa_pad_head_dim_to')} "
        f"argmax={summary.get('argmax_match_count')}/{data.get('compared_count')} "
        f"visual_max={summary.get('visual_features', {}).get('max_abs_diff', {}).get('max')} "
        f"image_max={summary.get('image_embeds', {}).get('max_abs_diff', {}).get('max')} "
        f"logits_max={summary.get('prefill_logits', {}).get('max_abs_diff', {}).get('max')} "
        f"visual_tok_s={summary.get('visual_tower_effective_tokens_per_s')}"
    )
    for row in data.get("items", [])[:4]:
        validation = row.get("vision_compile", {}).get("compiled_vs_static_eager_validation", {})
        real_rows = validation.get("real_rows", {})
        print(
            "  ITEM "
            f"idx={row.get('index')} id={row.get('id')} "
            f"argmax={row.get('argmax_match')} "
            f"visual_max={row.get('diffs', {}).get('visual_features', {}).get('max_abs_diff')} "
            f"logits_max={row.get('diffs', {}).get('prefill_logits', {}).get('max_abs_diff')} "
            f"compiled_vs_static_eager_real_max={real_rows.get('max_abs_diff')} "
            f"compiled_nonfinite={row.get('vision_compile', {}).get('first_real_output_nonfinite_count')} "
            f"callD={row.get('vision_compile', {}).get('static_visual_promptfa_call_head_dim')} "
            f"callD_aligned={row.get('vision_compile', {}).get('static_visual_promptfa_call_head_dim_fp16_32b_aligned')}"
        )
PY

echo "EXP07_STATIC_VISUAL_GROUPED OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
