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
  --recognizer-model "${RECOGNIZER_MODEL:-/workspace/models/PaddleOCR-VL-1.6}"
  --device "${DEVICE:-npu:0}"
  --dtype "${DTYPE:-fp16}"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${DECODE_BACKEND:-}" ]]; then
  args+=(--decode-backend "$DECODE_BACKEND")
fi
if [[ -n "${VISION_BACKEND:-}" ]]; then
  args+=(--vision-backend "$VISION_BACKEND")
fi
if [[ -n "${TEXT_BACKEND:-}" ]]; then
  args+=(--text-backend "$TEXT_BACKEND")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${CACHE_LENGTH:-}" ]]; then
  args+=(--cache-length "$CACHE_LENGTH")
fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  args+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
if [[ -n "${MAX_REGIONS:-}" ]]; then
  args+=(--max-regions "$MAX_REGIONS")
fi
if [[ -n "${VISION_BUCKETS:-}" ]]; then
  args+=(--vision-compile-buckets "$VISION_BUCKETS")
fi
if [[ -n "${TEXT_BUCKETS:-}" ]]; then
  args+=(--text-compile-buckets "$TEXT_BUCKETS")
fi

"$PYTHON_BIN" "${args[@]}"
"$PYTHON_BIN" - "$OUTPUT_DIR/run.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
assert data["configuration"]["layout_source"] == "real_pp_doclayout_v3_inference"
assert data["configuration"]["region_execution"] == "lazy_sequential_prefill_run_scoped_continuous_decode"
assert data["configuration"]["cross_page_decode"] is True
assert data["decode_schedule"]["requests"] == data["aggregate"]["recognized_regions"]
assert data["aggregate"]["pages"] == 1
assert data["aggregate"]["layout_regions"] > 1
assert data["aggregate"]["recognized_regions"] > 0
aggregate = data["aggregate"]
assert aggregate["raw_decode_token_slots"] == (
    aggregate["effective_decode_tokens"]
    + aggregate["idle_decode_token_slots"]
    + aggregate["lookahead_decode_token_slots"]
)
if aggregate["recognized_regions"] > data["configuration"]["batch_size"]:
    assert aggregate["hot_swap_decode_admissions"] > 0
print(f"validated={path}")
PY
