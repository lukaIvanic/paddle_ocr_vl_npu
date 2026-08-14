# 310P UniRec full-1651 CDM page comparison

Compare the already-completed 310P full-1651 evaluation against the complete
910B2 reference outputs. This is a CPU-only analysis. **Do not rerun inference,
page matching, TEDS, or CDM. Do not allocate an NPU.** Do not edit tracked
files, commit, push, or create a branch.

## Included 910B2 evidence

The repository contains the complete textual output bundle from the successful
910B2 W4/T8 run at source commit `470d8a6`:

`12_unirec_0_1b_inference/references/unirec_full1651_910b_470d8a6_text_outputs.tar.gz`

SHA-256:
`ca01bba70959b7a5d645aa7dfe9e7fabac7e82277256b24581673cd54f0537a1`

The archive has 8,295 tar entries and 4,984 regular files. It includes all
1,651 raw page Markdown and JSON outputs, `recognition_trace.jsonl`, all 1,651
image-tag-stripped evaluation
Markdown files, the OmniDocBench matched text/formula/table/reading results,
all 2,352 per-formula CDM results, and evaluation metadata. It excludes only
the 1,890 copied JPEG image assets. Those images are not inputs to this
comparison and would add 167 MB uncompressed.

The reference recomputes to:

```text
formula_pages=313
formula_samples=2352
page_cdm=0.9217916318
official_overall=0.9018433880
```

## Run once

Use the existing completed 310P full-1651 run that reported approximately
`Page CDM=0.9051`, `Page TEDS=0.8375`, `Text Edit=0.0551`, and
`Overall=89.58`. Do not select the earlier first-512 run.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

# Set this to the absolute root of the completed 1,651-page 310P run.
RUN_ROOT="${RUN_ROOT:?set the completed full-1651 310P RUN_ROOT}"
test -f "$RUN_ROOT/exit_code.txt"
test "$(tr -d '[:space:]' < "$RUN_ROOT/exit_code.txt")" = 0
test -f "$RUN_ROOT/evaluation_image_tags_stripped/full_eval_summary.json"
test -f "$RUN_ROOT/evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json"

OUT="$RUN_ROOT/cdm_vs_910b_470d8a6"
python3 12_unirec_0_1b_inference/compare_unirec_cdm_page_by_page.py \
  --candidate-run-root "$RUN_ROOT" \
  --output-dir "$OUT" \
  --top-pages 30 \
  | tee "$RUN_ROOT/cdm_vs_910b_470d8a6.log"

printf 'CDM_COMPARE_OUTPUT=%s\n' "$OUT"
sed -n '1,80p' "$OUT/report.md"
```

Expected wall time is seconds to about one minute. The script only reads JSON,
Markdown, and the compressed reference archive. It independently recomputes
both page-weighted CDM values and fails if either one disagrees with its saved
`full_eval_summary.json`.

## Outputs

The output directory contains:

- `summary.json`: headline score delta and exact-match counts;
- `report.md`: the 30 largest page-CDM regressions;
- `page_cdm_comparison.jsonl`: every formula page, sorted worst delta first;
- `formula_cdm_comparison.jsonl`: union of every aligned formula sample, with
  both predictions, exact `gt_cdm`/`pred_cdm` scorer inputs, CDM scores,
  matching indices, and normalized text;
- `stripped_prediction_markdown_digests.jsonl`: all 1,651 evaluator Markdown
  comparisons;
- `raw_output_markdown_digests.jsonl`: all 1,651 raw page Markdown comparisons;
- `reference_archive_manifest.json`: every supplied 910B2 textual artifact;
- `worst_page_formula_details.json`: both chips' full formula records for the
  30 largest page regressions, ready to inspect without another parser.

## Required report

Return:

1. the complete `UNIREC_CDM_PAGE_COMPARE PASS` line;
2. absolute `RUN_ROOT` and `CDM_COMPARE_OUTPUT`;
3. the complete `summary.json`;
4. the first 30 rows of `report.md`;
5. counts of pages that are better/equal/worse, normalized formula predictions
   that match exactly, byte-identical CDM inputs, CDM scores that changed
   despite identical inputs, formula match-topology changes, and stripped
   Markdown pages that match exactly;
6. the five largest page regressions with their candidate/reference formula
   predictions from `formula_cdm_comparison.jsonl`;
7. a concise classification of the gap: changed recognition output, changed
   formula matching/segmentation, or both.

Do not propose or implement a fix yet. Preserve the comparison directory so we
can inspect any individual page without rerunning evaluation.

If this brief was already run from commit `2002ca2`, pull the later correction
and rerun only the comparator command into the same output directory. This does
not rerun CDM. The corrected report distinguishes normalized-text equality from
byte-identical `gt_cdm` and `pred_cdm` inputs. Only the latter can establish
cross-runtime CDM score drift.
