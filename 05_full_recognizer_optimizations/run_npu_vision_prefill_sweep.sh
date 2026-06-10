#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VISION_PREFILL_BATCH_SIZES="${VISION_PREFILL_BATCH_SIZES:-1 2 4 8}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/recognizer_queue_vision_prefill_sweep}"
SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/vision_prefill_sweep_${SWEEP_ID}_summary.txt}"

mkdir -p "${OUTPUT_DIR}"

JSON_PATHS=()
for vp in ${VISION_PREFILL_BATCH_SIZES}; do
  export VISION_PREFILL_BATCH_SIZE="${vp}"
  export RUN_ID="${SWEEP_ID}_vp${vp}"
  export OUTPUT_PATH="${OUTPUT_DIR}/recognizer_queue_${SWEEP_ID}_vp${vp}.json"
  LOG_PATH="${OUTPUT_DIR}/recognizer_queue_${SWEEP_ID}_vp${vp}.log"
  JSON_PATHS+=("${OUTPUT_PATH}")
  echo "RUN_VISION_PREFILL_BATCH_SIZE=${vp}"
  bash "${SCRIPT_DIR}/run_npu_recognizer_queue_benchmark.sh" 2>&1 | tee "${LOG_PATH}"
done

"${PYTHON_BIN}" - "${JSON_PATHS[@]}" <<'PY' | tee "${SUMMARY_PATH}"
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:]]
rows = []
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    correctness = data.get("correctness", {})
    throughput = data.get("throughput", {})
    phase = data.get("phase_timing_s", {})
    prefill = data.get("prefill_measurement_summary_s", {})
    buckets = data.get("vision_shape_bucket_summary", {})
    grids = buckets.get("image_grid_thw", {})
    projected = buckets.get("projected_image_tokens", {})
    ready = data.get("ready_bank_build_details", {})
    rows.append(
        {
            "vp": data.get("vision_prefill_batch_size"),
            "pass": correctness.get("all_required_checks_passed"),
            "cross": data.get("has_cross_crop_vision_batches"),
            "actual_vp": data.get("actual_vision_prefill_batch_sizes"),
            "unique_grids": grids.get("unique_count"),
            "unique_projected": projected.get("unique_count"),
            "ready_s": phase.get("ready_bank_build"),
            "vision_core_s": prefill.get("vision_prefill_core_s"),
            "packed_outer_s": prefill.get("packed_vision_outer_wall_s"),
            "ready_ips": throughput.get("ready_build_items_per_s"),
            "vision_ips": throughput.get("vision_prefill_core_items_per_s"),
            "vision_tok_s": throughput.get("vision_core_tokens_per_s"),
            "projected_tok_s": throughput.get("projected_image_core_tokens_per_s"),
            "packed_outer_tok_s": throughput.get("packed_vision_outer_wall_tokens_per_s"),
            "text_tok_s": throughput.get("text_prefill_input_tokens_per_s"),
            "pipeline_ips": throughput.get("items_per_s_measured_pipeline_excluding_model_compile_load"),
            "decode_s": phase.get("decode_queue"),
            "length_cap": correctness.get("length_cap_hit_count"),
            "mismatch": correctness.get("mismatch_count"),
            "invalid": correctness.get("invalid_token_count"),
            "packed_batches": ready.get("packed_vision_batch_count"),
            "multi_crop_batches": ready.get("packed_vision_multi_crop_batch_count"),
            "path": str(path),
        }
    )

headers = [
    "vp",
    "pass",
    "cross",
    "unique_grids",
    "unique_projected",
    "ready_s",
    "vision_core_s",
    "packed_outer_s",
    "ready_ips",
    "vision_ips",
    "vision_tok_s",
    "projected_tok_s",
    "pipeline_ips",
    "length_cap",
    "mismatch",
    "invalid",
    "packed_batches",
    "multi_crop_batches",
]
print("VISION_PREFILL_SWEEP_SUMMARY")
print("\t".join(headers))
for row in rows:
    print("\t".join(str(row.get(key)) for key in headers))

print("VISION_PREFILL_SWEEP_JSON")
print(json.dumps(rows, indent=2, sort_keys=True))

failed = [
    row
    for row in rows
    if not row["pass"] or row["length_cap"] or row["mismatch"] or row["invalid"]
]
if failed:
    raise SystemExit("one or more sweep runs failed required correctness checks")
PY

echo "WROTE_SWEEP_SUMMARY ${SUMMARY_PATH}"
