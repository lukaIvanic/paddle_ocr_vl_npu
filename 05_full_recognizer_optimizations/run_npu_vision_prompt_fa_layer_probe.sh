#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/crops/hotswap_100_manifest.json}"
DEVICE="${DEVICE:-npu:0}"
CROP_ID="${CROP_ID:-hotswap_002_code_txt_p1474_11}"
MAX_LAYERS="${MAX_LAYERS:-27}"
PROMPT_FA_LAYOUT="${PROMPT_FA_LAYOUT:-bnsd}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/vision_prompt_fa_layer_probe}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/layer_probe_${CROP_ID}_${PROMPT_FA_LAYOUT}_${MAX_LAYERS}.json}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/probe_vision_prompt_fa_layers.py"
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --crop-id "${CROP_ID}"
  --device "${DEVICE}"
  --dtype fp16
  --npu-jit-compile off
  --max-layers "${MAX_LAYERS}"
  --prompt-fa-layout "${PROMPT_FA_LAYOUT}"
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
print("VISION_PROMPT_FA_LAYER_PROBE", json.dumps({
    "crop_id": data.get("crop_id"),
    "prompt_fa_layout": data.get("prompt_fa_layout"),
    "vision_tokens": data.get("vision_tokens"),
    "hidden_size": data.get("hidden_size"),
    "layers_checked": data.get("layers_checked"),
    "final_propagated": data.get("final_propagated"),
}, sort_keys=True))
print("LAYER_DIFFS")
print("layer raw_max projected_max same_layer_max prop_input_max prop_layer_max prop_layer_mean same_layer_allclose_5e_2 prop_layer_allclose_5e_2")
for row in data.get("layers", []):
    print(
        f"{row.get('layer')} "
        f"{row.get('same_input_raw', {}).get('max_abs_diff')} "
        f"{row.get('same_input_projected', {}).get('max_abs_diff')} "
        f"{row.get('same_input_layer', {}).get('max_abs_diff')} "
        f"{row.get('propagated_input', {}).get('max_abs_diff')} "
        f"{row.get('propagated_layer', {}).get('max_abs_diff')} "
        f"{row.get('propagated_layer', {}).get('mean_abs_diff')} "
        f"{row.get('same_input_layer', {}).get('allclose_5e_2')} "
        f"{row.get('propagated_layer', {}).get('allclose_5e_2')}"
    )
PY
echo "WROTE ${OUTPUT_PATH}"
