# 310P K10 suspect-crop inspection

## Goal

Inspect the six already identified crops without running inference or touching
the NPU. Compare their saved cross-KV rows and manifest geometry across the
canonical, middle, and K10 artifacts.

Use the same exported artifact paths from the completed factorization run:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

: "${PYTHON_BIN:?}"
: "${CANONICAL_ARTIFACT:?}"
: "${OPTIMIZED_ARTIFACT:?}"
: "${OPTIMIZED_MISMATCH_REPORT:?}"
: "${RUN_ROOT:?set this to the completed factorization RUN_ROOT}"

SUSPECT_ROOT="$RUN_ROOT/suspect_crop_inspection"
mkdir -p "$SUSPECT_ROOT"
printf '%s\n' \
  page_000032_crop_0005 \
  page_000016_crop_0003 \
  page_000032_crop_0002 \
  page_000033_crop_0001 \
  page_000037_crop_0000 \
  page_000018_crop_0002 \
  >"$SUSPECT_ROOT/request_ids.txt"

MIDDLE_ARTIFACT="$RUN_ROOT/prefill_production_buckets_optimized_weights"
COMPARE=12_unirec_0_1b_inference/compare_unirec_prefill_artifacts.py

"$PYTHON_BIN" "$COMPARE" \
  --reference "$CANONICAL_ARTIFACT" --candidate "$MIDDLE_ARTIFACT" \
  --request-ids-file "$SUSPECT_ROOT/request_ids.txt" --top 6 \
  --output "$SUSPECT_ROOT/canonical_vs_middle.json"
"$PYTHON_BIN" "$COMPARE" \
  --reference "$CANONICAL_ARTIFACT" --candidate "$OPTIMIZED_ARTIFACT" \
  --request-ids-file "$SUSPECT_ROOT/request_ids.txt" --top 6 \
  --output "$SUSPECT_ROOT/canonical_vs_k10.json"
"$PYTHON_BIN" "$COMPARE" \
  --reference "$MIDDLE_ARTIFACT" --candidate "$OPTIMIZED_ARTIFACT" \
  --request-ids-file "$SUSPECT_ROOT/request_ids.txt" --top 6 \
  --output "$SUSPECT_ROOT/middle_vs_k10.json"

jq '{compared_rows,exact_rows,weighted_mean_abs,weighted_rmse,max_abs,top_rows_by_rmse}' \
  "$SUSPECT_ROOT/canonical_vs_middle.json" \
  "$SUSPECT_ROOT/canonical_vs_k10.json" \
  "$SUSPECT_ROOT/middle_vs_k10.json"

emit_manifest() {
  local lane="$1" artifact="$2"
  jq --arg lane "$lane" --rawfile ids "$SUSPECT_ROOT/request_ids.txt" -c \
    'select(.request_id as $id | ($ids | split("\n") | index($id))) |
     {lane:$lane,request_id,page_index,crop_index,page_image,crop_label:.label,
      original:.prefill.prep.original_image_size,
      processed:.prefill.prep.processed_image_size,
      source_length:.cross_kv.source_length,
      vision_bucket:.prefill.vision_bucket,
      physical_source:.prefill.text_prefill_physical_source_tokens}' \
    "$artifact/crops.jsonl"
}
{
  emit_manifest canonical "$CANONICAL_ARTIFACT"
  emit_manifest middle "$MIDDLE_ARTIFACT"
  emit_manifest k10 "$OPTIMIZED_ARTIFACT"
} >"$SUSPECT_ROOT/all_lane_manifests.jsonl"

jq '{compared_count,token_exact_count,first_mismatches}' \
  "$OPTIMIZED_MISMATCH_REPORT" \
  >"$SUSPECT_ROOT/token_mismatches.json"
```

This is CPU-only and should finish in seconds. Report these four files; do not
rerun prefill or decode:

```bash
cat "$SUSPECT_ROOT/canonical_vs_middle.json"
cat "$SUSPECT_ROOT/canonical_vs_k10.json"
cat "$SUSPECT_ROOT/middle_vs_k10.json"
cat "$SUSPECT_ROOT/token_mismatches.json"
```
