# 310P UniRec optimized-layout W8/T8, first 128 pages

Run only the page-to-cross-KV producer. This measures whether the 32.1 ms
layout winner improves the complete prefill pipeline. It does not run decode or
prove FP16 layout quality.

Use the exact already-warm caches from the successful 310P experiments:

- `LAYOUT_CACHE`: the parent cache passed to the 32.1 ms `group16 +
  torchair_internal + preformatted FrozenBN buffers` layout run;
- `RECOGNITION_CACHE`: the successful six-graph W8/T8 recognition cache.

Do not create a cold layout cache with eight concurrent writers. Never use
physical NPU 5.

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
set -euo pipefail
git pull --ff-only origin main

PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench image directory}"
LAYOUT_CACHE="${LAYOUT_CACHE:?set the exact warm internal+buffer layout cache parent}"
RECOGNITION_CACHE="${RECOGNITION_CACHE:?set the existing six-graph recognition cache}"

test -x "$PYTHON_BIN"
test -d "$LAYOUT_CACHE"
test -d "$RECOGNITION_CACHE"
test "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P first}" != 5

OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_layout_optimized_w8_t8_$(git rev-parse --short HEAD)"
test ! -e "$OUT"
mkdir -p "$OUT/output"

command=(
  "$PYTHON_BIN" "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output-dir "$OUT/output"
  --dtype float16
  --offset 0
  --limit 128
  --workers 8
  --warmup-pages 8
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution torchair
  --layout-dtype float16
  --layout-batch-size 1
  --layout-depthwise-rewrite group16
  --layout-weight-format torchair_internal
  --layout-preformat-frozen-bn-buffers
  --cross-cache-length 512
  --layout-cache-dir "$LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --recognition-preprocess-threads 8
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)

printf '%q ' "${command[@]}" >"$OUT/command.txt"
printf '\n' >>"$OUT/command.txt"
export PYTHONUNBUFFERED=1
set -o pipefail
"${command[@]}" 2>&1 | tee "$OUT/run.log"
test "${PIPESTATUS[0]}" = 0

SUMMARY="$OUT/output/summary.json" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

REFERENCE_910B_PG_S = 27.3511137111
REFERENCE_910B_WALL_S = 4.6798825581

s = json.loads(Path(os.environ["SUMMARY"]).read_text())
assert s["status"] == "ok"
assert (s["offset"], s["limit"], s["workers"]) == (0, 128, 8)
assert s["artifact_storage"] == "discard"
assert s["validation"]["passed"] is True
assert s["layout_execution"] == "torchair"
assert s["layout_dtype"] == "float16"
assert s["layout_depthwise_rewrite"] == "group16"
assert s["layout_weight_format"] == "torchair_internal"
assert s["layout_preformat_frozen_bn_buffers"] is True
assert s["recognition_preprocess_threads"] == 8
assert s["worker_summary"]["prefix_diagnostics"]["new_first_call_count"] == 0

setup = s["worker_setup_diagnostics"]
assert len(setup) == 8
for row in setup:
    assert row["layout_dtype"] == "float16"
    assert row["layout_depthwise_rewrite"] == "group16"
    assert row["layout_weight_format"] == "torchair_internal"
    assert row["layout_preformat_frozen_bn_buffers"] is True
    assert row["layout_depthwise_rewrite_summary"]["target_count"] == 24
    assert row["layout_depthwise_rewrite_summary"]["rewritten_count"] == 24
    assert row["layout_weight_format_summary"]["requested"] == "torchair_internal"
    assert row["layout_frozen_bn_buffer_format_summary"]["converted_count"] > 0

wall = float(s["producer_wall_s"])
pg_s = float(s["throughput"]["pages_per_s"])
artifact = s["artifact"]
vision = s["worker_summary"]["vision_batching"]
stages = s["worker_summary"]["stage_s"]
print(
    "UNIREC_310P_LAYOUT_OPTIMIZED_W8_T8: PASS "
    f"producer={wall:.3f}s pg_s={pg_s:.3f} "
    f"vs_910b_pg={pg_s / REFERENCE_910B_PG_S:.3f}x "
    f"vs_910b_wall={REFERENCE_910B_WALL_S / wall:.3f}x "
    f"crops={artifact['crop_count']} rejected={artifact['rejected_crop_count']} "
    f"real_tokens={artifact['real_source_tokens']} "
    f"slot_eff={vision['compiled_slot_efficiency']:.3f} "
    f"fallback={vision['fallback_rows']} "
    f"layout_sum={stages['worker_detector_call_sum_s']:.3f}s "
    f"cpu_crop_sum={stages['worker_recognition_input_prepare_sum_s']:.3f}s "
    f"prefill_sum={stages['worker_recognition_prefill_sum_s']:.3f}s "
    f"setup={s['setup_s']:.3f}s warmup={s['warmup']['wall_s']:.3f}s"
)
PY
```

Send only the final `UNIREC_310P_LAYOUT_OPTIMIZED_W8_T8` line back to Luka,
then stop. Do not start decode or a larger run.
