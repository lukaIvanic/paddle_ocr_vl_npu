#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_512_fullgraph_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_ITEMS="${MAX_ITEMS:-4}"
export REPEATS="${REPEATS:-1}"
export WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-512}"
export STATIC_VISUAL_LN_IMPL="${STATIC_VISUAL_LN_IMPL:-manual_fp32}"
export STATIC_VISUAL_LN_LINEAR_MODE="${STATIC_VISUAL_LN_LINEAR_MODE:-grouped_qkv_mlp_fc1}"
export PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-80}"
export TORCHAIR_MODE="${TORCHAIR_MODE:-default}"
export VISION_USE_TORCHAIR_CACHE_COMPILE="${VISION_USE_TORCHAIR_CACHE_COMPILE:-1}"
export VISION_TORCHAIR_CACHE_DIR="${VISION_TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_static_visual}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_STATIC_VISUAL_512 PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_STATIC_VISUAL_512 MODEL=${MODEL}"
echo "EXP07_STATIC_VISUAL_512 BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_STATIC_VISUAL_512 DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_STATIC_VISUAL_512 DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_STATIC_VISUAL_512 MAX_ITEMS=${MAX_ITEMS} REPEATS=${REPEATS} WARMUP_REPEATS=${WARMUP_REPEATS}"
echo "EXP07_STATIC_VISUAL_512 STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_STATIC_VISUAL_512 STATIC_VISUAL_LN_IMPL=${STATIC_VISUAL_LN_IMPL}"
echo "EXP07_STATIC_VISUAL_512 STATIC_VISUAL_LN_LINEAR_MODE=${STATIC_VISUAL_LN_LINEAR_MODE}"
echo "EXP07_STATIC_VISUAL_512 PROMPTFA_PAD_HEAD_DIM_TO=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_STATIC_VISUAL_512 TORCHAIR_MODE=${TORCHAIR_MODE}"
echo "EXP07_STATIC_VISUAL_512 VISION_USE_TORCHAIR_CACHE_COMPILE=${VISION_USE_TORCHAIR_CACHE_COMPILE}"
echo "EXP07_STATIC_VISUAL_512 VISION_TORCHAIR_CACHE_DIR=${VISION_TORCHAIR_CACHE_DIR}"
echo "EXP07_STATIC_VISUAL_512 OUT_ROOT=${OUT_ROOT}"

run_compare() {
  local name="$1"
  local backend="$2"
  local output_json="${OUT_ROOT}/${name}.json"
  local cache_args=()
  if [[ "${backend}" == "torchair" && "${VISION_USE_TORCHAIR_CACHE_COMPILE}" == "1" ]]; then
    cache_args+=(--vision-use-torchair-cache-compile)
    cache_args+=(--vision-torchair-cache-dir "${VISION_TORCHAIR_CACHE_DIR}")
  fi
  echo "EXP07_STATIC_VISUAL_512 RUN name=${name} backend=${backend} output=${output_json}"
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
    --static-visual-fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
    --static-visual-ln-impl "${STATIC_VISUAL_LN_IMPL}" \
    --static-visual-ln-linear-mode "${STATIC_VISUAL_LN_LINEAR_MODE}" \
    --static-visual-promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}" \
    --torchair-mode "${TORCHAIR_MODE}" \
    "${cache_args[@]}" \
    --validate-compiled-against-static-eager \
    --candidate-name "${name}" \
    --max-items "${MAX_ITEMS}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --output "${output_json}"
}

run_compare static_eager_fixed512 none
run_compare torchair_fullgraph_fixed512 torchair

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_STATIC_VISUAL_512 SUMMARY")
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    candidate = data.get("candidate", {})
    summary = data.get("summary", {})
    bucket = summary.get("bucket_filter", {})
    eff = summary.get("visual_tower_effective_tokens_per_s", {})
    phys = summary.get("visual_tower_physical_tokens_per_s", {})
    print(
        f"{path.name}: "
        f"backend={candidate.get('vision_compile_backend')} "
        f"compile_api={candidate.get('compile_api')} "
        f"cache_compile={candidate.get('uses_torchair_cache_compile')} "
        f"fixedS={candidate.get('static_visual_fixed_physical_seq_len')} "
        f"ln_impl={candidate.get('static_visual_ln_impl')} "
        f"linear_mode={candidate.get('static_visual_ln_linear_mode')} "
        f"promptfa_padD={candidate.get('static_visual_promptfa_pad_head_dim_to')} "
        f"argmax={summary.get('argmax_match_count')}/{data.get('compared_count')} "
        f"visual_max={summary.get('visual_features', {}).get('max_abs_diff', {}).get('max')} "
        f"image_max={summary.get('image_embeds', {}).get('max_abs_diff', {}).get('max')} "
        f"logits_max={summary.get('prefill_logits', {}).get('max_abs_diff', {}).get('max')} "
        f"effective_tok_s={eff.get('tokens_per_s')} "
        f"physical_tok_s={phys.get('tokens_per_s')}"
    )
    print(
        "  BUCKET "
        f"manifest={bucket.get('manifest_item_count')} "
        f"eligible={bucket.get('eligible_count_before_max_items')} "
        f"excluded={bucket.get('excluded_count')} "
        f"selected={bucket.get('selected_count')} "
        f"reasons={bucket.get('excluded_reason_counts')}"
    )
    for row in data.get("items", [])[:8]:
        validation = row.get("vision_compile", {}).get("compiled_vs_static_eager_validation", {})
        real_rows = validation.get("real_rows", {})
        timing = row.get("timing_s", {})
        print(
            "  ITEM "
            f"manifest_idx={row.get('index')} compare_idx={row.get('compare_index')} id={row.get('id')} "
            f"real_tok={row.get('vision_tokens')} physical_tok={row.get('candidate_physical_vision_tokens')} "
            f"argmax={row.get('argmax_match')} "
            f"visual_max={row.get('diffs', {}).get('visual_features', {}).get('max_abs_diff')} "
            f"logits_max={row.get('diffs', {}).get('prefill_logits', {}).get('max_abs_diff')} "
            f"compiled_vs_static_eager_real_max={real_rows.get('max_abs_diff')} "
            f"compiled_nonfinite={row.get('vision_compile', {}).get('first_real_output_nonfinite_count')} "
            f"cache_dir={row.get('vision_compile', {}).get('torchair_cache_dir')} "
            f"callD={row.get('vision_compile', {}).get('static_visual_promptfa_call_head_dim')} "
            f"aligned={row.get('vision_compile', {}).get('static_visual_promptfa_call_head_dim_fp16_32b_aligned')} "
            f"visual_tower_s={timing.get('visual_tower_e2e_s', {}).get('avg')}"
        )
PY

echo "EXP07_STATIC_VISUAL_512 OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
