#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_prefill_compare_npu}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export CASES="${CASES:-manual_fp16 manual_fp32 promptfa_fp16}"

mkdir -p "${OUTPUT_DIR}"

declare -a OUTPUTS=()

run_case() {
  local case_name="$1"
  local dtype="$2"
  local attention="$3"
  local layout="$4"
  local output_path="${OUTPUT_DIR}/vision_prefill_compare_${RUN_ID}_${case_name}.json"

  echo "VISION_PREFILL_COMPARE_CASE name=${case_name} dtype=${dtype} attention=${attention} layout=${layout} output=${output_path}"
  DTYPE="${dtype}" \
    VISION_ATTENTION_IMPL="${attention}" \
    VISION_PROMPT_FA_LAYOUT="${layout}" \
    OUTPUT_PATH="${output_path}" \
    RUN_ID="${RUN_ID}_${case_name}" \
    "${SCRIPT_DIR}/run_npu_vision_prefill_only.sh"
  OUTPUTS+=("${case_name}=${output_path}")
}

for case_name in ${CASES}; do
  case "${case_name}" in
    manual_fp16)
      run_case "${case_name}" "fp16" "manual" "bnsd"
      ;;
    manual_fp32)
      run_case "${case_name}" "fp32" "manual" "bnsd"
      ;;
    promptfa_fp16)
      run_case "${case_name}" "fp16" "prompt_flash_attention" "${VISION_PROMPT_FA_LAYOUT:-bnsd}"
      ;;
    promptfa_fp32)
      run_case "${case_name}" "fp32" "prompt_flash_attention" "${VISION_PROMPT_FA_LAYOUT:-bnsd}"
      ;;
    *)
      echo "Unsupported CASES entry: ${case_name}" >&2
      echo "Supported: manual_fp16 manual_fp32 promptfa_fp16 promptfa_fp32" >&2
      exit 2
      ;;
  esac
done

"${PYTHON_BIN}" - "${OUTPUTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for spec in sys.argv[1:]:
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    modes = data.get("modes", {})
    sync = modes.get("sync_per_crop", {})
    unsynced = modes.get("unsynced_loop", {})
    comp = data.get("comparisons", {}).get("unsynced_vs_sync_per_crop", {})
    rows.append(
        {
            "case": name,
            "path": str(path),
            "dtype": data.get("dtype"),
            "vision_attention": data.get("vision_attention"),
            "vision_prompt_fa_layout": data.get("vision_prompt_fa_layout"),
            "recognizer_crop_count": data.get("recognizer_crop_count"),
            "raw_extracted_crop_count_before_max_crops": data.get("raw_extracted_crop_count_before_max_crops"),
            "warmup_s": (data.get("warmup") or {}).get("elapsed_s"),
            "sync_per_crop_total_s": sync.get("total_s"),
            "sync_per_crop_items_per_s": sync.get("items_per_s"),
            "sync_per_crop_vision_tokens_per_s": sync.get("vision_tokens_per_s"),
            "unsynced_loop_total_s": unsynced.get("total_s"),
            "unsynced_loop_items_per_s": unsynced.get("items_per_s"),
            "unsynced_loop_vision_tokens_per_s": unsynced.get("vision_tokens_per_s"),
            "unsynced_speedup_over_sync": comp.get("speedup"),
        }
    )

by_case = {row["case"]: row for row in rows}
base = by_case.get("manual_fp16")
for row in rows:
    if base and row is not base:
        b = base.get("unsynced_loop_total_s")
        r = row.get("unsynced_loop_total_s")
        row["unsynced_total_s_vs_manual_fp16_ratio"] = (r / b) if b and r else None
        bips = base.get("unsynced_loop_items_per_s")
        rips = row.get("unsynced_loop_items_per_s")
        row["unsynced_items_per_s_vs_manual_fp16_speedup"] = (rips / bips) if bips and rips else None

print("VISION_PREFILL_COMPARE_SUMMARY", json.dumps(rows, ensure_ascii=False, sort_keys=True))
PY
