#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
export DEVICE="${DEVICE:-npu:0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines/promptfa_fp16_eager_64}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/static_visual_batched_encoder_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
export SWEEP_BATCH_SIZES="${SWEEP_BATCH_SIZES:-1 2 4 8}"
export MAX_ITEMS="${MAX_ITEMS:-32}"
export STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN="${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN:-1024}"
export SKIP_GENERATION="${SKIP_GENERATION:-1}"
export VISION_USE_TORCHAIR_CACHE_COMPILE="${VISION_USE_TORCHAIR_CACHE_COMPILE:-1}"
export VISION_TORCHAIR_CACHE_DIR="${VISION_TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_static_visual}"

mkdir -p "${OUT_ROOT}"

echo "EXP07_BATCHED_ENCODER_SWEEP PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_BATCHED_ENCODER_SWEEP MODEL=${MODEL}"
echo "EXP07_BATCHED_ENCODER_SWEEP BASELINE_DIR=${BASELINE_DIR}"
echo "EXP07_BATCHED_ENCODER_SWEEP DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "EXP07_BATCHED_ENCODER_SWEEP SWEEP_BATCH_SIZES=${SWEEP_BATCH_SIZES}"
echo "EXP07_BATCHED_ENCODER_SWEEP MAX_ITEMS=${MAX_ITEMS}"
echo "EXP07_BATCHED_ENCODER_SWEEP STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=${STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN}"
echo "EXP07_BATCHED_ENCODER_SWEEP SKIP_GENERATION=${SKIP_GENERATION}"
echo "EXP07_BATCHED_ENCODER_SWEEP VISION_USE_TORCHAIR_CACHE_COMPILE=${VISION_USE_TORCHAIR_CACHE_COMPILE}"
echo "EXP07_BATCHED_ENCODER_SWEEP VISION_TORCHAIR_CACHE_DIR=${VISION_TORCHAIR_CACHE_DIR}"
echo "EXP07_BATCHED_ENCODER_SWEEP OUT_ROOT=${OUT_ROOT}"

for batch_size in ${SWEEP_BATCH_SIZES}; do
  case_root="${OUT_ROOT}/B${batch_size}"
  echo "EXP07_BATCHED_ENCODER_SWEEP RUN BATCH_SIZE=${batch_size} OUT_ROOT=${case_root}"
  BATCH_SIZE="${batch_size}" OUT_ROOT="${case_root}" bash "${SCRIPT_DIR}/run_npu_static_visual_batched_encoder.sh"
done

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("B*/*.json")):
    data = json.loads(path.read_text())
    candidate = data.get("candidate", {})
    summary = data.get("summary", {})
    compile_meta = data.get("compile", {})
    bucket = summary.get("bucket_filter", {})
    records.append({
        "path": str(path),
        "backend": candidate.get("vision_compile_backend"),
        "compile_api": candidate.get("compile_api"),
        "cache_compile": candidate.get("uses_torchair_cache_compile"),
        "B": candidate.get("batch_size"),
        "fixedS": candidate.get("static_visual_fixed_physical_seq_len"),
        "selected": bucket.get("selected_count"),
        "eligible": bucket.get("eligible_count_before_max_items"),
        "batches": summary.get("batch_count"),
        "argmax": summary.get("argmax_match_count"),
        "compared": data.get("compared_count"),
        "tokens": summary.get("generated_trimmed_match_count"),
        "text": summary.get("text_match_count"),
        "nonfinite": summary.get("visual_nonfinite_item_count"),
        "invalid": summary.get("invalid_token_count"),
        "encoder_phys": summary.get("encoder_physical_tokens_per_s"),
        "encoder_eff": summary.get("encoder_effective_tokens_per_s"),
        "prefix_plus_phys": summary.get("prefix_plus_encoder_physical_tokens_per_s"),
        "prefix_plus_eff": summary.get("prefix_plus_encoder_effective_tokens_per_s"),
        "encoder_s": summary.get("total_batched_encoder_s"),
        "prefix_s": summary.get("total_prefix_build_s"),
        "first_call_s": compile_meta.get("compiled_first_call_s"),
        "cache_dir": compile_meta.get("torchair_cache_dir"),
    })

print("EXP07_BATCHED_ENCODER_SWEEP SUMMARY_TABLE")
print(
    "backend\tB\tselected\tbatches\targmax\ttext\tnonfinite\t"
    "encoder_phys_tok_s\tencoder_eff_tok_s\tprefix_plus_phys_tok_s\t"
    "encoder_s\tprefix_s\tfirst_call_s\tcache_compile\tjson"
)
for record in sorted(records, key=lambda value: (str(value["backend"]), int(value["B"] or 0))):
    text = "skip" if record["text"] is None else f"{record['text']}/{record['compared']}"
    print(
        f"{record['backend']}\t"
        f"{record['B']}\t"
        f"{record['selected']}\t"
        f"{record['batches']}\t"
        f"{record['argmax']}/{record['compared']}\t"
        f"{text}\t"
        f"{record['nonfinite']}\t"
        f"{record['encoder_phys']}\t"
        f"{record['encoder_eff']}\t"
        f"{record['prefix_plus_phys']}\t"
        f"{record['encoder_s']}\t"
        f"{record['prefix_s']}\t"
        f"{record['first_call_s']}\t"
        f"{record['cache_compile']}\t"
        f"{record['path']}"
    )

print("EXP07_BATCHED_ENCODER_SWEEP TORCHAIR_PHYSICAL_SPEEDUP")
by_backend_b = {(record["backend"], int(record["B"])): record for record in records if record["B"] is not None}
base = by_backend_b.get(("torchair", 1))
if base and base.get("encoder_phys"):
    base_phys = float(base["encoder_phys"])
    for batch_size in sorted({int(record["B"]) for record in records if record["backend"] == "torchair"}):
        record = by_backend_b[("torchair", batch_size)]
        phys = record.get("encoder_phys")
        ratio = float(phys) / base_phys if phys else None
        print(f"B={batch_size} encoder_phys_tok_s={phys} vs_B1={ratio}")
PY

echo "EXP07_BATCHED_ENCODER_SWEEP OUTPUT_TREE"
find "${OUT_ROOT}" -maxdepth 3 -type f | sort
