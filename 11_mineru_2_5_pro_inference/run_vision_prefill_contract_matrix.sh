#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
MODEL="${MODEL:-/workspace/models/MinerU2.5-Pro-2605-1.2B}"
IMAGE="${IMAGE:-$REPO_ROOT/crops/crop_01_text_block_en.png}"
BUCKET="${BUCKET:-1024}"
WARMUP="${WARMUP:-0}"
REPEATS="${REPEATS:-1}"
GENERATION_TOKENS="${GENERATION_TOKENS:-8}"
LANE_TIMEOUT_S="${LANE_TIMEOUT_S:-1200}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.runtime_cache/11_mineru_2_5_pro_inference/vision_contract_matrix}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/11_mineru_2_5_pro_inference/vision_contract_matrix_${RUN_TAG}}"

mkdir -p "$CACHE_ROOT" "$OUT_ROOT"
OVERALL_EXIT=0

run_lane() {
  local lane="$1"
  shift
  local lane_root="$OUT_ROOT/$lane"
  mkdir -p "$lane_root"
  printf 'VISION_MATRIX_LANE_START lane=%s bucket=%s\n' "$lane" "$BUCKET"
  set +e
  timeout --signal=TERM --kill-after=30s "${LANE_TIMEOUT_S}s" \
    "$PYTHON" "$SCRIPT_DIR/bench_compiled_vision_prefill.py" \
      --model "$MODEL" \
      --image "$IMAGE" \
      --prompt "Text Recognition:" \
      --bucket "$BUCKET" \
      --cache-dir "$CACHE_ROOT" \
      --output "$lane_root/result.json" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      --generation-tokens "$GENERATION_TOKENS" \
      --eager-reference-attention manual \
      "$@" \
      2>&1 | tee "$lane_root/run.log"
  local lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" >"$lane_root/exit_code.txt"
  printf 'VISION_MATRIX_LANE_FINISH lane=%s exit=%s\n' "$lane" "$lane_exit"
  if [[ "$lane_exit" -ne 0 ]]; then
    return "$lane_exit"
  fi
}

run_lane baseline_nn_layernorm_nn_linear \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl module \
  --projection-impl linear || OVERALL_EXIT=1

run_lane promptfa_d80_to_d96_barrier \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl module \
  --projection-impl linear \
  --promptfa-pad-head-dim-to 96 || OVERALL_EXIT=1

run_lane manual_fp32_layernorm_nn_linear \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl manual_fp32 \
  --projection-impl linear || OVERALL_EXIT=1

run_lane nn_layernorm_grouped_qkv_3d_weight \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl module \
  --projection-impl grouped_qkv || OVERALL_EXIT=1

run_lane nn_layernorm_grouped_qkv_mlp_fc1_3d_weight \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl module \
  --projection-impl grouped_qkv_mlp_fc1 || OVERALL_EXIT=1

run_lane paddle_style_combined \
  --compiled-attention prompt_flash_attention \
  --layer-norm-impl manual_fp32 \
  --projection-impl grouped_qkv_mlp_fc1 \
  --promptfa-pad-head-dim-to 96 || OVERALL_EXIT=1

run_lane manual_attention_control \
  --compiled-attention manual \
  --layer-norm-impl module \
  --projection-impl linear || OVERALL_EXIT=1

printf 'VISION_MATRIX_COMPLETE exit=%s out_root=%s cache_root=%s\n' \
  "$OVERALL_EXIT" "$OUT_ROOT" "$CACHE_ROOT"
exit "$OVERALL_EXIT"
