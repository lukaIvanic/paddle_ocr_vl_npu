#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_generation_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_ITEMS="${MAX_ITEMS:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export STATIC_VISUAL_LN_IMPL="${STATIC_VISUAL_LN_IMPL:-manual_fp32}"
export STATIC_VISUAL_LN_LINEAR_MODE="${STATIC_VISUAL_LN_LINEAR_MODE:-grouped_qkv_mlp_fc1}"
export PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-80}"
export TORCHAIR_MODE="${TORCHAIR_MODE:-default}"
export VISION_USE_TORCHAIR_CACHE_COMPILE="${VISION_USE_TORCHAIR_CACHE_COMPILE:-1}"
export VISION_TORCHAIR_CACHE_DIR="${VISION_TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_static_visual}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_STATIC_VISUAL_GENERATION PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_STATIC_VISUAL_GENERATION MODEL=${MODEL}"
echo "EXP07_STATIC_VISUAL_GENERATION BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_STATIC_VISUAL_GENERATION DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_STATIC_VISUAL_GENERATION DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_STATIC_VISUAL_GENERATION MAX_ITEMS=${MAX_ITEMS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} CACHE_LENGTH=${CACHE_LENGTH}"
echo "EXP07_STATIC_VISUAL_GENERATION STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_STATIC_VISUAL_GENERATION STATIC_VISUAL_LN_IMPL=${STATIC_VISUAL_LN_IMPL}"
echo "EXP07_STATIC_VISUAL_GENERATION STATIC_VISUAL_LN_LINEAR_MODE=${STATIC_VISUAL_LN_LINEAR_MODE}"
echo "EXP07_STATIC_VISUAL_GENERATION PROMPTFA_PAD_HEAD_DIM_TO=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_STATIC_VISUAL_GENERATION TORCHAIR_MODE=${TORCHAIR_MODE}"
echo "EXP07_STATIC_VISUAL_GENERATION VISION_USE_TORCHAIR_CACHE_COMPILE=${VISION_USE_TORCHAIR_CACHE_COMPILE}"
echo "EXP07_STATIC_VISUAL_GENERATION VISION_TORCHAIR_CACHE_DIR=${VISION_TORCHAIR_CACHE_DIR}"
echo "EXP07_STATIC_VISUAL_GENERATION OUT_ROOT=${OUT_ROOT}"

cache_args=()
if [[ "${VISION_USE_TORCHAIR_CACHE_COMPILE}" == "1" ]]; then
  cache_args+=(--vision-use-torchair-cache-compile)
  cache_args+=(--vision-torchair-cache-dir "${VISION_TORCHAIR_CACHE_DIR}")
fi

OUTPUT_JSON="${OUT_ROOT}/generation.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_static_visual_generation.py" \
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
  --static-visual-fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
  --static-visual-ln-impl "${STATIC_VISUAL_LN_IMPL}" \
  --static-visual-ln-linear-mode "${STATIC_VISUAL_LN_LINEAR_MODE}" \
  --static-visual-promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}" \
  --torchair-mode "${TORCHAIR_MODE}" \
  "${cache_args[@]}" \
  --validate-compiled-against-static-eager \
  --candidate-name "torchair_fullgraph_generation_fixedS${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
  --max-items "${MAX_ITEMS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --cache-length "${CACHE_LENGTH}" \
  --output "${OUTPUT_JSON}"

"${PYTHON_BIN}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
summary = data.get("summary", {})
candidate = data.get("candidate", {})
bucket = summary.get("bucket_filter", {})
eff = summary.get("visual_tower_effective_tokens_per_s", {})
phys = summary.get("visual_tower_physical_tokens_per_s", {})
print("EXP07_STATIC_VISUAL_GENERATION SUMMARY")
print(
    f"output={path} "
    f"compile_api={candidate.get('compile_api')} "
    f"cache_compile={candidate.get('uses_torchair_cache_compile')} "
    f"fixedS={candidate.get('static_visual_fixed_physical_seq_len')} "
    f"max_new_tokens={candidate.get('max_new_tokens')} "
    f"argmax={summary.get('argmax_match_count')}/{data.get('compared_count')} "
    f"tokens={summary.get('generated_trimmed_match_count')}/{data.get('compared_count')} "
    f"text={summary.get('text_match_count')}/{data.get('compared_count')} "
    f"invalid_tokens={summary.get('invalid_token_count')} "
    f"length_cap_hits={summary.get('length_cap_hit_count')} "
    f"effective_tok_s={eff.get('tokens_per_s')} "
    f"physical_tok_s={phys.get('tokens_per_s')}"
)
print(
    "BUCKET "
    f"manifest={bucket.get('manifest_item_count')} "
    f"eligible={bucket.get('eligible_count_before_max_items')} "
    f"excluded={bucket.get('excluded_count')} "
    f"selected={bucket.get('selected_count')} "
    f"reasons={bucket.get('excluded_reason_counts')}"
)
for row in data.get("items", [])[:8]:
    gen = row.get("generation", {})
    timing = row.get("timing_s", {})
    vc = row.get("vision_compile", {})
    print(
        "ITEM "
        f"manifest_idx={row.get('index')} id={row.get('id')} "
        f"real_tok={row.get('vision_tokens')} physical_tok={row.get('candidate_physical_vision_tokens')} "
        f"argmax={row.get('argmax_match')} token_match={gen.get('generated_trimmed_match')} "
        f"text_match={row.get('texts', {}).get('match')} "
        f"logits_max={row.get('diffs', {}).get('prefill_logits', {}).get('max_abs_diff')} "
        f"visual_real_max={row.get('diffs', {}).get('visual_features', {}).get('max_abs_diff')} "
        f"compiled_nonfinite={vc.get('first_real_output_nonfinite_count')} "
        f"visual_tower_s={timing.get('visual_tower_e2e_s', {}).get('avg')} "
        f"cache_dir={vc.get('torchair_cache_dir')}"
    )
PY

echo "EXP07_STATIC_VISUAL_GENERATION OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
