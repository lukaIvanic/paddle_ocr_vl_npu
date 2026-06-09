#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAYOUTS="${LAYOUTS:-bnsd bsnd bsh}"
MAX_LAYERS="${MAX_LAYERS:-27}"
CROP_ID="${CROP_ID:-hotswap_002_code_txt_p1474_11}"

for layout in ${LAYOUTS}; do
  echo "PROMPT_FA_LAYOUT_SWEEP_BEGIN ${layout}"
  PROMPT_FA_LAYOUT="${layout}" \
  MAX_LAYERS="${MAX_LAYERS}" \
  CROP_ID="${CROP_ID}" \
    bash "${SCRIPT_DIR}/run_npu_vision_prompt_fa_layer_probe.sh"
  echo "PROMPT_FA_LAYOUT_SWEEP_END ${layout}"
done
