#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
CROP_ID="${CROP_ID:-hotswap_002_code_txt_p1474_11}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_prompt_fa_probe}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/probe_${CROP_ID}.json}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/probe_vision_prompt_fa_variants.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --crop-id "${CROP_ID}"
  --device "${DEVICE}"
  --dtype fp16
  --npu-jit-compile off
)

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"
"${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print("VISION_PROMPT_FA_PROBE", json.dumps({
    "crop_id": data.get("crop_id"),
    "length": data.get("length"),
    "num_heads": data.get("num_heads"),
    "head_dim": data.get("head_dim"),
    "manual_elapsed_s": data.get("manual_elapsed_s"),
}, sort_keys=True))
print("VARIANT_RESULTS")
print("name ok elapsed_s max_abs_diff mean_abs_diff allclose_5e_2 allclose_1e_1 error")
for row in data.get("results", []):
    print(
        f"{row.get('name')} "
        f"{row.get('ok')} "
        f"{row.get('elapsed_s')} "
        f"{row.get('max_abs_diff')} "
        f"{row.get('mean_abs_diff')} "
        f"{row.get('allclose_5e_2')} "
        f"{row.get('allclose_1e_1')} "
        f"{row.get('error_type') or ''}:{str(row.get('error') or '')[:160]}"
    )
PY
echo "WROTE ${OUTPUT_PATH}"
