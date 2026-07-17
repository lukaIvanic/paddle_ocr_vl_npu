#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
export MODEL="${MODEL:-/workspace/models/PaddleOCR-VL-1.6}"
export DATASET_DIR="${DATASET_DIR:-}"
export DEVICE="${DEVICE:-npu:0}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/w8a8_static_visual_batched_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_ITEMS="${MAX_ITEMS:-8}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-none}"
export W8A8_WEIGHT_LAYOUT="${W8A8_WEIGHT_LAYOUT:-auto}"
export W8A8_STATIC_CALIBRATION_BATCHES="${W8A8_STATIC_CALIBRATION_BATCHES:-2}"
export W8A8_STATIC_SCALE_HEADROOM="${W8A8_STATIC_SCALE_HEADROOM:-1.05}"
export QUANTIZATION_CASES="${QUANTIZATION_CASES:-none w8a8_dynamic w8a8_static}"

mkdir -p "${OUT_ROOT}"

echo "W8A8_ENCODER PYTHON_BIN=${PYTHON_BIN}"
echo "W8A8_ENCODER MODEL=${MODEL}"
echo "W8A8_ENCODER DATASET_DIR=${DATASET_DIR:-<manifest default>}"
echo "W8A8_ENCODER DEVICE=${DEVICE}"
echo "W8A8_ENCODER BATCH_SIZE=${BATCH_SIZE} MAX_ITEMS=${MAX_ITEMS}"
echo "W8A8_ENCODER FIXED_S=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "W8A8_ENCODER VISION_COMPILE_BACKEND=${VISION_COMPILE_BACKEND}"
echo "W8A8_ENCODER W8A8_WEIGHT_LAYOUT=${W8A8_WEIGHT_LAYOUT}"
echo "W8A8_ENCODER QUANTIZATION_CASES=${QUANTIZATION_CASES}"
echo "W8A8_ENCODER OUT_ROOT=${OUT_ROOT}"

for quantization in ${QUANTIZATION_CASES}; do
  output_json="${OUT_ROOT}/${quantization}.json"
  echo "W8A8_ENCODER_RUN quantization=${quantization} output=${output_json}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_static_visual_batched_encoder.py" \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --baseline "${BASELINE_DIR}" \
    --device "${DEVICE}" \
    --dtype fp16 \
    --npu-jit-compile off \
    --vision-attention prompt_flash_attention \
    --vision-prompt-fa-layout bnsd \
    --vision-prompt-fa-mask-sparse-mode 1 \
    --static-visual-fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
    --static-visual-ln-impl manual_fp32 \
    --static-visual-ln-linear-mode grouped_qkv_mlp_fc1 \
    --static-visual-promptfa-pad-head-dim-to 80 \
    --vision-compile-backend "${VISION_COMPILE_BACKEND}" \
    --vision-linear-quantization "${quantization}" \
    --w8a8-weight-layout "${W8A8_WEIGHT_LAYOUT}" \
    --w8a8-static-calibration-batches "${W8A8_STATIC_CALIBRATION_BATCHES}" \
    --w8a8-static-scale-headroom "${W8A8_STATIC_SCALE_HEADROOM}" \
    --batch-size "${BATCH_SIZE}" \
    --max-items "${MAX_ITEMS}" \
    --skip-generation \
    --candidate-name "${quantization}" \
    --output "${output_json}"
done

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("W8A8_ENCODER_SUMMARY")
baseline = None
records = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    candidate = data["candidate"]
    summary = data["summary"]
    quantization = candidate["vision_linear_quantization"]
    speed = summary["encoder_physical_tokens_per_s"]
    if quantization == "none":
        baseline = speed
    records.append((path, quantization, speed, summary, data))
for path, quantization, speed, summary, data in records:
    ratio = None if not baseline else speed / baseline
    print(
        f"quantization={quantization} "
        f"encoder_phys_tok_s={speed} speedup={ratio} "
        f"encoder_eff_tok_s={summary['encoder_effective_tokens_per_s']} "
        f"argmax={summary['argmax_match_count']}/{data['compared_count']} "
        f"nonfinite={summary['visual_nonfinite_item_count']} "
        f"visual_max_abs={summary['visual_features']['max_abs_diff']} "
        f"logits_max_abs={summary['prefill_logits']['max_abs_diff']} "
        f"json={path}"
    )
PY

echo "W8A8_ENCODER_DONE OUT_ROOT=${OUT_ROOT}"
