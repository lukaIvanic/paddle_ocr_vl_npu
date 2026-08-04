#!/usr/bin/env bash
# CPU-only, one-command audit of the completed 310P Phase-57 run against the
# exact committed 910B generation/prediction authority.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

PYTHON_BIN=/usr/local/python3.12.13/bin/python
EVAL_WRAPPER="$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py"
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
COMPARATOR="$REPO/09_persistent_page_engine/scripts/compare_phase57_authority.py"
ATLAS="$REPO/09_persistent_page_engine/scripts/generation_difference_atlas.py"
REFERENCE="$REPO/tmp/09_persistent_page_engine/910b_phase57_authority_898ced7/phase57_910b_authority.gdatlas.zip"

CDM_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase56_cdm"
test -s "$CDM_ROOT/runtime_paths.env"
. "$CDM_ROOT/runtime_paths.env"
EVAL_PYTHON="$OMNIDOCBENCH_EVAL_PYTHON"
EVALUATOR_ROOT="$OMNIDOCBENCH_EVALUATOR_ROOT"
test -x "$EVAL_PYTHON"
test -f "$EVALUATOR_ROOT/pdf_validation.py"
test -s "$REFERENCE"

FULL="$($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
roots = []
for path in Path("tmp/09_persistent_page_engine").glob("310p_phase57_cap4096_b64_pse_*/full"):
    summary = path / "output" / "run_summary.json"
    if not summary.is_file():
        continue
    data = json.loads(summary.read_text())
    if data.get("result_count") == data.get("prediction_count") == 1651:
        roots.append(path)
if not roots:
    raise SystemExit("no completed 1,651-page Phase-57 run found")
print(max(roots, key=lambda path: path.stat().st_mtime))
PY
)"

CANDIDATE_EVAL="$($PYTHON_BIN - "$FULL" <<'PY'
import sys
from pathlib import Path
full = Path(sys.argv[1])
roots = [
    path for path in full.glob("evaluation_line_bounded_*")
    if (path / "work/result/predictions_quick_match_metric_result.json").is_file()
    and (path / "cdm_native/cdm_run_summary.json").is_file()
]
if not roots:
    raise SystemExit("no completed Phase-57G matching+CDM evaluation found")
print(max(roots, key=lambda path: path.stat().st_mtime))
PY
)"

SHORT="$(git rev-parse --short HEAD)"
ROOT="${FULL%/full}/authority_audit_${SHORT}"
test ! -e "$ROOT"
mkdir -p "$ROOT/reference_predictions" "$ROOT/reference_recheck/work"

printf '[phase57-authority] full=%s\n' "$FULL" | tee "$ROOT/progress.log"
printf '[phase57-authority] candidate_eval=%s\n' "$CANDIDATE_EVAL" | tee -a "$ROOT/progress.log"
printf '[phase57-authority] extracting exact 910B Markdown authority\n' | tee -a "$ROOT/progress.log"
"$PYTHON_BIN" - "$REFERENCE" "$ROOT/reference_predictions" <<'PY'
import sys, zipfile
from pathlib import Path
archive_path, output = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(archive_path) as archive:
    members = sorted(
        name for name in archive.namelist()
        if name.startswith("predictions/") and name.endswith(".md")
    )
    assert len(members) == 1651, len(members)
    for name in members:
        (output / Path(name).name).write_bytes(archive.read(name))
print(f"[phase57-authority] extracted_pages={len(members)}")
PY

cat >"$ROOT/reference_recheck/work/config.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist]
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 12
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $(realpath "$FULL/output/OmniDocBench_subset.json")
    prediction:
      data_path: $(realpath "$ROOT/reference_predictions")
    match_method: quick_match
    match_workers: 24
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

printf '[phase57-authority] reevaluating exact 910B Markdown on this evaluator\n' | tee -a "$ROOT/progress.log"
SECONDS=0
(
  cd "$ROOT/reference_recheck/work"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$EVAL_WRAPPER" \
    --config config.yaml --evaluator-root "$EVALUATOR_ROOT" \
    --match-workers 24 --teds-workers 12 \
    --page-timeout-sec 120 --fallback-timeout-sec 180 \
    --fallback-latex-timeout-sec 30 --teds-timeout-sec 120
) 2>&1 | tee "$ROOT/reference_recheck/run.log"
printf '%s\n' "$SECONDS" >"$ROOT/reference_recheck/wall_s.txt"

REF_RESULT="$ROOT/reference_recheck/work/result"
REF_METRIC="$REF_RESULT/reference_predictions_quick_match_metric_result.json"
REF_MATCHED="$REF_RESULT/reference_predictions_quick_match_display_formula_result.json"
test -s "$REF_METRIC"
test -s "$REF_MATCHED"

nproc_count="$(nproc)"
mem_gib="$(awk '/MemAvailable:/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
CDM_WORKERS="$nproc_count"
test "$CDM_WORKERS" -le 96 || CDM_WORKERS=96
mem_workers="$((mem_gib / 2))"
test "$mem_workers" -ge 1 || mem_workers=1
test "$CDM_WORKERS" -le "$mem_workers" || CDM_WORKERS="$mem_workers"

printf '[phase57-authority] running CDM on exact 910B reevaluation workers=%s\n' "$CDM_WORKERS" | tee -a "$ROOT/progress.log"
PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
  --input "$REF_MATCHED" --output-dir "$ROOT/reference_recheck/cdm_native" \
  --evaluator-root "$EVALUATOR_ROOT" --workers "$CDM_WORKERS" \
  2>&1 | tee "$ROOT/reference_recheck/cdm.run.log"

CANDIDATE_RESULT="$CANDIDATE_EVAL/work/result"
CANDIDATE_METRIC="$CANDIDATE_RESULT/predictions_quick_match_metric_result.json"
CANDIDATE_CDM="$CANDIDATE_EVAL/cdm_native/cdm_run_summary.json"
REF_CDM="$ROOT/reference_recheck/cdm_native/cdm_run_summary.json"
test -s "$CANDIDATE_METRIC" && test -s "$CANDIDATE_CDM" && test -s "$REF_CDM"

printf '[phase57-authority] comparing contracts, crops, pages, and scores\n' | tee -a "$ROOT/progress.log"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$COMPARATOR" \
  --reference-bundle "$REFERENCE" \
  --candidate-output "$FULL/output" \
  --candidate-metric "$CANDIDATE_METRIC" \
  --candidate-cdm-summary "$CANDIDATE_CDM" \
  --reference-recheck-metric "$REF_METRIC" \
  --reference-recheck-cdm-summary "$REF_CDM" \
  --output "$ROOT/authority_audit.json" \
  2>&1 | tee "$ROOT/authority_audit.log"

printf '[phase57-authority] attributing exact metric losses to pages/crops\n' | tee -a "$ROOT/progress.log"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ATLAS" \
  --reference-bundle "$REFERENCE" \
  --candidate-output "$FULL/output" \
  --candidate-eval-dir "$CANDIDATE_RESULT" \
  --reference-label 910B2 \
  --candidate-label 310P3 \
  --output-dir "$ROOT/atlas" \
  --expected-pages 1651 \
  --review-limit 100 \
  2>&1 | tee "$ROOT/atlas.log"

"$PYTHON_BIN" - "$ROOT/authority_audit.json" "$ROOT/atlas/report.json" "$ROOT/agent_report.md" <<'PY'
import json, sys
from pathlib import Path
audit_path, atlas_path, output_path = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text())
atlas = json.loads(atlas_path.read_text())
scores = audit["scores"]
lines = [
    "# Phase 57 exact 910B authority audit",
    "",
    "`" + audit["classification"] + "`",
    "",
    f"- Run contract differences: {len(audit['run_contract']['differences'])}",
    f"- Crop input mismatches: {audit['crop_inputs']['mismatch_count']}",
    f"- Crop generation classes: {audit['crop_generations']['counts']}",
    f"- Page Markdown classes: {audit['page_markdown']['counts']}",
    f"- 910B reevaluation exact: {scores['evaluator_reproduction_exact']}",
    f"- 910B authority scores: {scores['reference_authority_910b']}",
    f"- Same-host 910B reevaluation: {scores['reference_recheck_on_work_server']}",
    f"- 310P scores: {scores['candidate_310p']}",
    f"- 310P minus authority: {scores['candidate_minus_reference_authority']}",
    "",
    "## Largest crop length divergences",
    "",
]
for row in audit["crop_generations"]["largest_30_length_divergences"][:15]:
    lines.append(
        f"- {row['source_image_name']} block={row['block_index']} label={row['label']} "
        f"tokens={row['reference_tokens']}->{row['candidate_tokens']} ratio={row['length_ratio']:.2f}"
    )
lines.extend(["", "## Top metric-loss pages", ""])
for metric, rows in atlas.get("top_harmful_pages", {}).items():
    lines.append(f"### {metric}")
    for row in rows[:10]:
        compact = {key: row[key] for key in row if key in {"page", "img_id", "image_name", "delta", "candidate_minus_reference", "contribution_delta"}}
        lines.append(f"- `{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}`")
    lines.append("")
output_path.write_text("\n".join(lines) + "\n")
PY

sentence="$(tail -n 1 "$ROOT/authority_audit.log")"
printf '%s agent_report=%s\n' "$sentence" "$ROOT/agent_report.md" | tee "$ROOT/final_sentence.txt"
