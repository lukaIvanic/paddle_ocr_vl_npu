#!/usr/bin/env bash
# Evaluation only: no inference, model loading, or NPU setup.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HYBRID_OUTPUT:?completed inference output directory}"
: "${DATASET_JSON:?full OmniDocBench v1.6 ground truth}"
: "${EVAL_JOB:?new existing absolute directory for logs and exit status}"
: "${EVAL_PYTHON:?existing frozen evaluator Python}"
: "${EVALUATOR_ROOT:?existing frozen evaluator checkout}"
: "${OMNIDOCBENCH_EVAL_TOOLS_ROOT:?existing frozen TeX/ImageMagick root}"
test -d "$EVAL_JOB"
test ! -e "$EVAL_JOB/exit_code.txt"
export EVAL_ROOT="$EVAL_JOB/evaluation"
test ! -e "$EVAL_ROOT"
SECONDS=0
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EVAL_JOB/exit_code.txt"
  printf '%s\n' "$SECONDS" >"$EVAL_JOB/elapsed_s.txt"
  exit "$rc"
}
trap finish EXIT
export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
source "$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
export PYTHONUNBUFFERED=1
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = 2b161d010d2e3aff77a0edef359ea3a6411d23cd
test -z "$(git -C "$EVALUATOR_ROOT" status --porcelain)"
[[ "$("$CDM_PDFLATEX" --version | head -n 1)" == *"1.40.28 (TeX Live 2025)"* ]]
[[ "$("$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 1)" == *"ImageMagick 7.1.1-47"* ]]
echo 'EVAL_PHASE runtime_verify start'
"$EVAL_PYTHON" "$REPO/09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py" \
  --evaluator-root "$EVALUATOR_ROOT" | tee "$EVAL_JOB/runtime_verify.json"
echo 'EVAL_PHASE runtime_verify finish'
"$EVAL_PYTHON" "$REPO/18_unirec_paddle_hybrid_pipeline/prepare_accuracy_eval.py" \
  --output "$HYBRID_OUTPUT" --dataset-json "$DATASET_JSON" --evaluation-root "$EVAL_ROOT"
ulimit -n 65536
cd "$EVAL_ROOT/work"
echo 'EVAL_PHASE matching_teds start'
"$EVAL_PYTHON" "$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py" \
  --config config.yaml --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 12 --teds-workers 12 --page-timeout-sec 120 \
  --fallback-timeout-sec 180 --fallback-latex-timeout-sec 30
echo 'EVAL_PHASE matching_teds finish'
echo 'EVAL_PHASE cdm start'
"$EVAL_PYTHON" "$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py" \
  --input "$EVAL_ROOT/work/result/predictions_quick_match_display_formula_result.json" \
  --output-dir "$EVAL_ROOT/cdm" --evaluator-root "$EVALUATOR_ROOT" --workers 12
echo 'EVAL_PHASE cdm finish'
"$EVAL_PYTHON" "$REPO/12_unirec_0_1b_inference/summarize_completed_unirec_eval.py" \
  --lane hybrid_310p_half_ready_full1651_dd172553a4ef \
  --metric-result "$EVAL_ROOT/work/result/predictions_quick_match_metric_result.json" \
  --stage-execution "$EVAL_ROOT/work/result/predictions_quick_match_stage_execution.json" \
  --cdm-summary "$EVAL_ROOT/cdm/cdm_run_summary.json" \
  --output "$EVAL_ROOT/full_eval_summary.json"
"$EVAL_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['EVAL_ROOT'])
for row in json.loads((root/'prediction_transform_manifest.json').read_text()):
    assert hashlib.sha256(Path(row['source']).read_bytes()).hexdigest() == row['original_sha256'], row['source']
    prediction = root/'predictions'/f"{row['stem']}.md"
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == row['evaluation_sha256'], str(prediction)
print('EVAL_PHASE source_and_prediction_hash_check finish')
PY
echo 'HYBRID_EVAL_COMPLETE: inspect full_eval_summary.json and timeout/error counts'
