#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PYTHON_BIN:?validated venv python_nosym is required}"
: "${FACTOR_ROOT:?completed factorization RUN_ROOT is required}"
: "${ALIGNED_ROOT:?failed aligned-K10 RUN_ROOT is required}"

REFERENCE="$FACTOR_ROOT/prefill_production_buckets_optimized_weights"
CANDIDATE="$ALIGNED_ROOT/hot_prefill"
OUTPUT="$ALIGNED_ROOT/six_cross_kv_comparison.json"
IDS="$ALIGNED_ROOT/six_cross_kv_request_ids.txt"

test -x "$PYTHON_BIN"
test -s "$REFERENCE/summary.json"
test -s "$REFERENCE/crops.jsonl"
test -s "$REFERENCE/cross_kv.bin"
test -s "$CANDIDATE/summary.json"
test -s "$CANDIDATE/crops.jsonl"
test -s "$CANDIDATE/cross_kv.bin"

printf '%s\n' \
  page_000033_crop_0001 \
  page_000037_crop_0000 \
  page_000046_crop_0000 \
  page_000047_crop_0000 \
  page_000117_crop_0000 \
  page_000119_crop_0002 >"$IDS"

started="$(date +%s)"
"$PYTHON_BIN" "$SCRIPT_DIR/compare_unirec_prefill_artifacts.py" \
  --reference "$REFERENCE" \
  --candidate "$CANDIDATE" \
  --request-ids-file "$IDS" \
  --top 6 \
  --output "$OUTPUT"
wall_s="$(($(date +%s) - started))"

"$PYTHON_BIN" - "$OUTPUT" "$wall_s" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "UNIREC_ALIGNED_K10_SIX_CROSS_KV: PASS "
    f"wall_s={sys.argv[2]} rows={report['compared_rows']} "
    f"exact={report['exact_rows']} "
    f"wMAE={report['weighted_mean_abs']:.9g} "
    f"wRMSE={report['weighted_rmse']:.9g} "
    f"max_abs={report['max_abs']:.9g}"
)
for row in report["top_rows_by_rmse"]:
    print(
        "UNIREC_ALIGNED_K10_CROSS_KV_ROW "
        f"request_id={row['request_id']} source_length={row['source_length']} "
        f"exact={str(row['exact']).lower()} "
        f"mean_abs={row['mean_abs']:.9g} rmse={row['rmse']:.9g} "
        f"relative_rmse={row['relative_rmse']:.9g} "
        f"max_abs={row['max_abs']:.9g}"
    )
print(f"OUTPUT={sys.argv[1]}")
PY
