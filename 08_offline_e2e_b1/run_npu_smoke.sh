#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
IMAGE="${IMAGE:-/workspace/datasets/OmniDocBench/images/PPT_The Right Moves_page_024.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/tmp/08_offline_e2e_b1/smoke}"

args=(
  "$HERE/run_offline_e2e.py"
  --image "$IMAGE"
  --layout-model "${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV3_safetensors}"
  --recognizer-model "${RECOGNIZER_MODEL:-PaddlePaddle/PaddleOCR-VL-1.6}"
  --device "${DEVICE:-npu:0}"
  --dtype "${DTYPE:-fp16}"
  --decode-backend "${DECODE_BACKEND:-torchair}"
  --cache-length "${CACHE_LENGTH:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-768}"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${MAX_REGIONS:-}" ]]; then
  args+=(--max-regions "$MAX_REGIONS")
fi

"$PYTHON_BIN" "${args[@]}"
"$PYTHON_BIN" - "$OUTPUT_DIR/run.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
assert data["configuration"]["layout_source"] == "real_pp_doclayout_v3_inference"
assert data["configuration"]["region_execution"] == "strict_sequential_b1"
assert data["aggregate"]["pages"] == 1
assert data["aggregate"]["layout_regions"] > 1
assert data["aggregate"]["recognized_regions"] > 0
print(f"validated={path}")
PY
