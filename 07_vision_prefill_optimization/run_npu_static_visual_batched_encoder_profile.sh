#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_batched_encoder_profile_$(date -u +%Y%m%dT%H%M%SZ)}"
export PROFILE_BATCH_SIZES="${PROFILE_BATCH_SIZES:-1 2}"
export PROFILE_METRICS="${PROFILE_METRICS:-pipe memory}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-2}"
export PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-5}"
export MAX_ITEMS="${MAX_ITEMS:-16}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export VISION_TORCHAIR_CACHE_DIR="${VISION_TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_static_visual}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_BATCHED_ENCODER_PROFILE PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_BATCHED_ENCODER_PROFILE MODEL=${MODEL}"
echo "EXP07_BATCHED_ENCODER_PROFILE BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_BATCHED_ENCODER_PROFILE DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_BATCHED_ENCODER_PROFILE PROFILE_BATCH_SIZES=${PROFILE_BATCH_SIZES}"
echo "EXP07_BATCHED_ENCODER_PROFILE PROFILE_METRICS=${PROFILE_METRICS}"
echo "EXP07_BATCHED_ENCODER_PROFILE PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS}"
echo "EXP07_BATCHED_ENCODER_PROFILE PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS}"
echo "EXP07_BATCHED_ENCODER_PROFILE MAX_ITEMS=${MAX_ITEMS}"
echo "EXP07_BATCHED_ENCODER_PROFILE STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_BATCHED_ENCODER_PROFILE VISION_TORCHAIR_CACHE_DIR=${VISION_TORCHAIR_CACHE_DIR}"
echo "EXP07_BATCHED_ENCODER_PROFILE OUT_ROOT=${OUT_ROOT}"

for batch_size in ${PROFILE_BATCH_SIZES}; do
  for metric in ${PROFILE_METRICS}; do
    case_name="B${batch_size}_${metric}"
    output_json="${OUT_ROOT}/${case_name}.json"
    profile_root="${OUT_ROOT}/${case_name}_profile"
    echo "EXP07_BATCHED_ENCODER_PROFILE RUN B=${batch_size} metric=${metric} output=${output_json}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_static_visual_batched_encoder.py" \
      --model "${MODEL}" \
      --baseline "${BASELINE_DIR}" \
      --device "${DEVICE}" \
      --dtype fp16 \
      --npu-jit-compile off \
      --vision-attention prompt_flash_attention \
      --vision-prompt-fa-layout bnsd \
      --vision-prompt-fa-mask-sparse-mode 1 \
      --batch-size "${batch_size}" \
      --max-items "${MAX_ITEMS}" \
      --static-visual-fixed-physical-seq-len "${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}" \
      --static-visual-ln-impl manual_fp32 \
      --static-visual-ln-linear-mode grouped_qkv_mlp_fc1 \
      --static-visual-promptfa-pad-head-dim-to 80 \
      --vision-use-torchair-cache-compile \
      --vision-torchair-cache-dir "${VISION_TORCHAIR_CACHE_DIR}" \
      --profile-root "${profile_root}" \
      --profile-metric "${metric}" \
      --profile-warmup-steps "${PROFILE_WARMUP_STEPS}" \
      --profile-active-steps "${PROFILE_ACTIVE_STEPS}" \
      --output "${output_json}"
  done
done

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("EXP07_BATCHED_ENCODER_PROFILE SUMMARY_TABLE")
print(
    "case\tB\tmetric\tphys_tok_s\teff_tok_s\tforward_sync_s\tprofiler_step_s\t"
    "compiled_first_call_s\tprofile_dir\tparsed_md"
)
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    parsed = data.get("parsed_profile") or {}
    parsed_md = parsed.get("parsed_md")
    print(
        f"{path.stem}\t"
        f"{data.get('batch_size')}\t"
        f"{data.get('profile_metric')}\t"
        f"{data.get('encoder_physical_tokens_per_s_forward_sync')}\t"
        f"{data.get('encoder_effective_tokens_per_s_forward_sync')}\t"
        f"{data.get('profile_forward_sync_sum_s')}\t"
        f"{data.get('profile_profiler_step_sum_s')}\t"
        f"{data.get('compiled_first_call_s')}\t"
        f"{data.get('profile_dir')}\t"
        f"{parsed_md}"
    )
    parsed_json = parsed.get("parsed_json")
    if not parsed_json:
        continue
    summary = json.loads(Path(parsed_json).read_text())
    for run in summary.get("runs", []):
        kernel = run.get("kernel_details") or {}
        step = run.get("step_trace_time") or {}
        totals = step.get("totals_us") or {}
        print(f"  STEP_TOTALS {path.stem} {json.dumps(totals, sort_keys=True)}")
        print(f"  TOP_KERNEL_TYPES {path.stem}")
        for row in (kernel.get("top_kernel_types") or [])[:10]:
            print(f"    {row.get('name')}\tcount={row.get('count')}\tduration_us={row.get('duration_us')}")
        print(f"  TOP_SHAPE_FORMATS {path.stem}")
        for row in (kernel.get("top_shape_format_signatures") or [])[:8]:
            print(f"    {row.get('name')}\tcount={row.get('count')}\tduration_us={row.get('duration_us')}")
        print(f"  TOP_SUSPECTS {path.stem}")
        for row in (kernel.get("suspect_kernel_rows") or [])[:8]:
            print(
                f"    {row.get('name')}\t{row.get('type')}\tduration_us={row.get('duration_us')}\t"
                f"shapes={row.get('input_shapes')}\tformats={row.get('input_formats')}"
            )
PY

echo "EXP07_BATCHED_ENCODER_PROFILE OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 5 -type f | sort
