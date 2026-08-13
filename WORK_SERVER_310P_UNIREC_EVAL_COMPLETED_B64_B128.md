# Score the completed UniRec B64 and B128 full runs

This is evaluation-only. Do not rerun layout, prefill, or decode. Do not source
`npu-setup`; no NPU and no large `/dev/shm` are needed. Use the exact completed
1,651-page B64 and B128 output directories that produced the reported timing
summaries.

## Prepare and compare predictions

Pull the commit named by Luka. Resolve the two existing `output` directories;
each must contain `run_summary.json` and 1,651 page subdirectories with Markdown
predictions.

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"

PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
DATASET_JSON="${DATASET_JSON:?set the full OmniDocBench.json path}"
B64_OUTPUT="${B64_OUTPUT:?set the completed B64 output directory}"
B128_OUTPUT="${B128_OUTPUT:?set the completed B128 output directory}"
ROOT="tmp/12_unirec_0_1b_inference/completed_b64_b128_eval_$(git rev-parse --short HEAD)_$(date +%Y%m%dT%H%M%S)"
ROOT="$(realpath -m "$ROOT")"

"$PYTHON_BIN" 12_unirec_0_1b_inference/prepare_completed_unirec_eval.py \
  --dataset-json "$DATASET_JSON" \
  --b64-output "$B64_OUTPUT" \
  --b128-output "$B128_OUTPUT" \
  --evaluation-root "$ROOT"
cat "$ROOT/prediction_comparison.json"
```

Immediately tell Luka the `identical_count` and `differing_count`. If all 1,651
predictions are byte-identical, evaluate B128 only and state that the exact same
quality result applies to B64. If any prediction differs, evaluate both lanes.

## Guarded official evaluation

Reuse the existing evaluator runtime. Do not install or update it silently.
The evaluator checkout must be commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

```bash
EVALUATOR_ROOT="${EVALUATOR_ROOT:?set the existing OmniDocBench evaluator checkout}"
EVAL_PYTHON="${EVAL_PYTHON:?set the existing OmniDocBench evaluation Python}"
test -x "$EVAL_PYTHON"
test -f "$EVALUATOR_ROOT/pdf_validation.py"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

comparison="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["differing_count"])' \
  "$ROOT/prediction_comparison.json")"
if test "$comparison" -eq 0; then
  lanes="b128"
else
  lanes="b64 b128"
fi

for lane in $lanes; do
  work="$ROOT/$lane/work"
  cd "$work"
  ulimit -n 65536
  SECONDS=0
  set +e
  set -o pipefail
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" \
    "$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py" \
    --config config.yaml \
    --evaluator-root "$EVALUATOR_ROOT" \
    --match-workers 12 --teds-workers 12 \
    --page-timeout-sec 120 \
    --fallback-timeout-sec 180 \
    --fallback-latex-timeout-sec 30 \
    2>&1 | tee evaluation.log
  status="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$status" >../eval_exit_code.txt
  printf '%s\n' "$SECONDS" >../eval_wall_s.txt
  test "$status" -eq 0
  cd "$REPO"
done
```

Report progress at page matching completion and TEDS completion. A timeout
recorded by the guarded evaluator is evidence; do not delete a difficult page
or change the denominator.

## CDM and official Overall

For each evaluated lane, run direct CDM over the evaluator's saved matched
formula artifact. Reuse the already-installed CDM-capable runtime. If that
runtime is absent, report `CDM_RUNTIME_MISSING` after returning all non-CDM
metrics; do not improvise an installation.

```bash
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
CDM_WORKERS="$(nproc)"
test "$CDM_WORKERS" -le 96 || CDM_WORKERS=96

for lane in $lanes; do
  result="$ROOT/$lane/work/result"
  matched="$result/predictions_quick_match_display_formula_result.json"
  cdm_out="$ROOT/$lane/cdm"
  mkdir -p "$cdm_out"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
    --input "$matched" --output-dir "$cdm_out" \
    --evaluator-root "$EVALUATOR_ROOT" --workers "$CDM_WORKERS" \
    2>&1 | tee "$cdm_out/run.log"

  "$EVAL_PYTHON" \
    "$REPO/12_unirec_0_1b_inference/summarize_completed_unirec_eval.py" \
    --lane "$lane" \
    --metric-result "$result/predictions_quick_match_metric_result.json" \
    --stage-execution "$result/predictions_quick_match_stage_execution.json" \
    --cdm-summary "$cdm_out/cdm_run_summary.json" \
    --output "$ROOT/$lane/full_eval_summary.json" \
    | tee "$ROOT/$lane/full_eval_sentence.txt"
done
```

Return each `UNIREC_FULL_EVAL` line and `full_eval_summary.json`. Also report
the evaluator and project commits, both original run-summary paths, comparison
JSON, evaluator wall time, match fallback/error counts, TEDS timeout/error
counts, CDM timeout/error counts, and whether one score was reused because the
Markdown predictions were byte-identical. Then stop.
