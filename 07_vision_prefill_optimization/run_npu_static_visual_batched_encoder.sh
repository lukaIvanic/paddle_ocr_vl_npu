#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_batched_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_ITEMS="${MAX_ITEMS:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export STATIC_VISUAL_LN_IMPL="${STATIC_VISUAL_LN_IMPL:-manual_fp32}"
export STATIC_VISUAL_LN_LINEAR_MODE="${STATIC_VISUAL_LN_LINEAR_MODE:-grouped_qkv_mlp_fc1}"
export PROMPTFA_PAD_HEAD_DIM_TO="${PROMPTFA_PAD_HEAD_DIM_TO:-80}"
export TORCHAIR_MODE="${TORCHAIR_MODE:-default}"
export VISION_USE_TORCHAIR_CACHE_COMPILE="${VISION_USE_TORCHAIR_CACHE_COMPILE:-1}"
export VISION_TORCHAIR_CACHE_DIR="${VISION_TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_static_visual}"
export SKIP_GENERATION="${SKIP_GENERATION:-0}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_BATCHED_ENCODER PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_BATCHED_ENCODER MODEL=${MODEL}"
echo "EXP07_BATCHED_ENCODER BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_BATCHED_ENCODER DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "EXP07_BATCHED_ENCODER DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_BATCHED_ENCODER BATCH_SIZE=${BATCH_SIZE} MAX_ITEMS=${MAX_ITEMS}"
echo "EXP07_BATCHED_ENCODER STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_BATCHED_ENCODER STATIC_VISUAL_LN_IMPL=${STATIC_VISUAL_LN_IMPL}"
echo "EXP07_BATCHED_ENCODER STATIC_VISUAL_LN_LINEAR_MODE=${STATIC_VISUAL_LN_LINEAR_MODE}"
echo "EXP07_BATCHED_ENCODER PROMPTFA_PAD_HEAD_DIM_TO=${PROMPTFA_PAD_HEAD_DIM_TO}"
echo "EXP07_BATCHED_ENCODER TORCHAIR_MODE=${TORCHAIR_MODE}"
echo "EXP07_BATCHED_ENCODER VISION_USE_TORCHAIR_CACHE_COMPILE=${VISION_USE_TORCHAIR_CACHE_COMPILE}"
echo "EXP07_BATCHED_ENCODER VISION_TORCHAIR_CACHE_DIR=${VISION_TORCHAIR_CACHE_DIR}"
echo "EXP07_BATCHED_ENCODER SKIP_GENERATION=${SKIP_GENERATION}"
echo "EXP07_BATCHED_ENCODER OUT_ROOT=${OUT_ROOT}"

common_args=(
  --model "${MODEL}"
  --dataset-dir "${DATASET_DIR}"
  --baseline "${BASELINE_DIR}"
  --device "${DEVICE}"
  --dtype fp16
  --npu-jit-compile off
  --vision-attention prompt_flash_attention
  --vision-prompt-fa-layout bnsd
  --vision-prompt-fa-mask-sparse-mode 1
  --static-visual-fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
  --static-visual-ln-impl "${STATIC_VISUAL_LN_IMPL}"
  --static-visual-ln-linear-mode "${STATIC_VISUAL_LN_LINEAR_MODE}"
  --static-visual-promptfa-pad-head-dim-to "${PROMPTFA_PAD_HEAD_DIM_TO}"
  --torchair-mode "${TORCHAIR_MODE}"
  --batch-size "${BATCH_SIZE}"
  --max-items "${MAX_ITEMS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --cache-length "${CACHE_LENGTH}"
)

if [[ "${SKIP_GENERATION}" == "1" ]]; then
  common_args+=(--skip-generation)
fi

run_case() {
  local name="$1"
  local backend="$2"
  local output_json="${OUT_ROOT}/${name}.json"
  local cache_args=()
  if [[ "${backend}" == "torchair" && "${VISION_USE_TORCHAIR_CACHE_COMPILE}" == "1" ]]; then
    cache_args+=(--vision-use-torchair-cache-compile)
    cache_args+=(--vision-torchair-cache-dir "${VISION_TORCHAIR_CACHE_DIR}")
  fi
  echo "EXP07_BATCHED_ENCODER RUN name=${name} backend=${backend} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_static_visual_batched_encoder.py" \
    "${common_args[@]}" \
    --vision-compile-backend "${backend}" \
    "${cache_args[@]}" \
    --candidate-name "${name}" \
    --output "${output_json}"
}

run_case static_eager_batched_encoder none
run_case torchair_batched_encoder torchair

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_BATCHED_ENCODER SUMMARY")
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    candidate = data.get("candidate", {})
    compile_meta = data.get("compile", {})
    bucket = summary.get("bucket_filter", {})
    print(
        f"{path.name}: "
        f"backend={candidate.get('vision_compile_backend')} "
        f"compile_api={candidate.get('compile_api')} "
        f"cache_compile={candidate.get('uses_torchair_cache_compile')} "
        f"B={candidate.get('batch_size')} fixedS={candidate.get('static_visual_fixed_physical_seq_len')} "
        f"batches={summary.get('batch_count')} selected={summary.get('bucket_filter', {}).get('selected_count')} "
        f"argmax={summary.get('argmax_match_count')}/{data.get('compared_count')} "
        f"tokens={summary.get('generated_trimmed_match_count')}/{data.get('compared_count')} "
        f"text={summary.get('text_match_count')}/{data.get('compared_count')} "
        f"nonfinite_items={summary.get('visual_nonfinite_item_count')} "
        f"encoder_eff_tok_s={summary.get('encoder_effective_tokens_per_s')} "
        f"encoder_phys_tok_s={summary.get('encoder_physical_tokens_per_s')} "
        f"prefix_plus_encoder_eff_tok_s={summary.get('prefix_plus_encoder_effective_tokens_per_s')} "
        f"cache_dir={compile_meta.get('torchair_cache_dir')}"
    )
    print(
        "  BUCKET "
        f"manifest={bucket.get('manifest_item_count')} "
        f"eligible={bucket.get('eligible_count_before_max_items')} "
        f"selected={bucket.get('selected_count')} "
        f"excluded={bucket.get('excluded_count')} "
        f"reasons={bucket.get('excluded_reason_counts')}"
    )
    for batch in data.get("batches", [])[:8]:
        print(
            "  BATCH "
            f"idx={batch.get('batch_index')} ids={batch.get('ids')} "
            f"eff_tok={batch.get('effective_tokens')} phys_tok={batch.get('physical_tokens')} "
            f"prefix_s={batch.get('prefix_build_s')} encoder_s={batch.get('batched_encoder_s')} "
            f"encoder_eff_tok_s={batch.get('encoder_effective_tokens_per_s')} "
            f"nonfinite={batch.get('output_nonfinite_count')}"
        )
    for row in data.get("items", [])[:8]:
        print(
            "  ITEM "
            f"manifest_idx={row.get('index')} batch={row.get('batch_index')} local={row.get('batch_local_index')} "
            f"id={row.get('id')} real_tok={row.get('vision_tokens')} "
            f"argmax={row.get('argmax_match')} "
            f"token_match={row.get('generation', {}).get('generated_trimmed_match')} "
            f"text_match={row.get('texts', {}).get('match')} "
            f"logits_max={row.get('diffs', {}).get('prefill_logits', {}).get('max_abs_diff')} "
            f"visual_max={row.get('diffs', {}).get('visual_features', {}).get('max_abs_diff')} "
            f"visual_nonfinite={row.get('candidate_visual_nonfinite_count')}"
        )
PY

echo "EXP07_BATCHED_ENCODER OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort
