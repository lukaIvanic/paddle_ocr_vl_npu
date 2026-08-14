#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ARCHIVE="$SCRIPT_DIR/references/unirec_full1651_910b_470d8a6_text_outputs.tar.gz"
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
FINGERPRINTER="$SCRIPT_DIR/fingerprint_unirec_cdm_runtime.py"
ANALYZER="$SCRIPT_DIR/analyze_unirec_cdm_same_host.py"
REFERENCE_FINGERPRINT="$SCRIPT_DIR/references/unirec_cdm_runtime_910b2_20260814.json"
EVAL_ENV="$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
EVALUATOR_COMMIT=2b161d010d2e3aff77a0edef359ea3a6411d23cd

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve() {
  : "${RUN_ROOT:?set the completed full-1651 310P run root}"
  : "${EVAL_PYTHON:?set the same CDM-capable Python used by the completed run}"
  : "${EVALUATOR_ROOT:?set the same OmniDocBench evaluator root used by the completed run}"
  : "${CDM_WORKERS:=64}"
  RUN_ROOT="$(readlink -f "$RUN_ROOT")"
  EVAL_PYTHON="$(absolute_executable_path "$EVAL_PYTHON")"
  EVALUATOR_ROOT="$(readlink -f "$EVALUATOR_ROOT")"

  export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
  # shellcheck source=../09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
  source "$EVAL_ENV"
  EVAL_PYTHON="$OMNIDOCBENCH_EVAL_PYTHON"
  EVALUATOR_ROOT="$OMNIDOCBENCH_EVALUATOR_ROOT"

  test -x "$EVAL_PYTHON"
  test -f "$EVALUATOR_ROOT/pdf_validation.py"
  test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
  test -f "$RUN_ROOT/evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json"
  test -f "$RUN_ROOT/evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json"
  test -s "$ARCHIVE" && test -s "$REFERENCE_FINGERPRINT"
  PYTHONPATH="$(dirname "$CDM_RUNNER")" "$EVAL_PYTHON" -c \
    'from run_cdm_from_matched_formulas import _configure_cdm_runtime; print(_configure_cdm_runtime())'
}

worker() {
  local root="$1"
  local canonical_evaluator="$root/evaluator_clean"
  mkdir -p "$root/inputs"
  # Use the committed evaluator source without modifying the agent's dirty
  # checkout. A local clone is fast and excludes its unstaged mathcolor patch.
  git clone --quiet --no-checkout "$EVALUATOR_ROOT" "$canonical_evaluator"
  git -C "$canonical_evaluator" checkout --quiet --detach "$EVALUATOR_COMMIT"
  test -z "$(git -C "$canonical_evaluator" status --porcelain=v1 --untracked-files=all)"
  tar -xOf "$ARCHIVE" \
    evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json \
    >"$root/inputs/reference_original_cdm_result.json"
  tar -xOf "$ARCHIVE" \
    evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json \
    >"$root/inputs/reference_matched_formulas.json"

  "$EVAL_PYTHON" "$FINGERPRINTER" \
    --evaluator-root "$canonical_evaluator" --output "$root/candidate_runtime_fingerprint.json"

  printf '[same-host] candidate CDM replay begin workers=%s\n' "$CDM_WORKERS"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
    --input "$RUN_ROOT/evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json" \
    --output-dir "$root/candidate_recheck" --evaluator-root "$canonical_evaluator" \
    --workers "$CDM_WORKERS" >"$root/candidate_recheck.log" 2>&1
  printf '[same-host] candidate CDM replay done\n'

  printf '[same-host] 910B output CDM replay on 310P begin workers=%s\n' "$CDM_WORKERS"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
    --input "$root/inputs/reference_matched_formulas.json" \
    --output-dir "$root/reference_recheck" --evaluator-root "$canonical_evaluator" \
    --workers "$CDM_WORKERS" >"$root/reference_recheck.log" 2>&1
  printf '[same-host] 910B output CDM replay on 310P done\n'

  "$EVAL_PYTHON" "$ANALYZER" \
    --reference-original "$root/inputs/reference_original_cdm_result.json" \
    --candidate-original "$RUN_ROOT/evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json" \
    --reference-recheck "$root/reference_recheck/result/predictions_quick_match_cdm_result.json" \
    --candidate-recheck "$root/candidate_recheck/result/predictions_quick_match_cdm_result.json" \
    --reference-fingerprint "$REFERENCE_FINGERPRINT" \
    --candidate-fingerprint "$root/candidate_runtime_fingerprint.json" \
    --output-dir "$root"
  cat "$root/report.md"
}

entry() {
  local root="$1" status=0
  set +e
  worker "$root"
  status="$?"
  set -e
  printf '%s\n' "$status" >"$root/exit_code.txt"
  exit "$status"
}

launch() {
  resolve
  local short timestamp root
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  root="${AUDIT_ROOT:-$RUN_ROOT/cdm_same_host_${short}_${timestamp}}"
  test ! -e "$root"
  mkdir -p "$root"
  nohup env PYTHONUNBUFFERED=1 RUN_ROOT="$RUN_ROOT" EVAL_PYTHON="$EVAL_PYTHON" \
    EVALUATOR_ROOT="$EVALUATOR_ROOT" CDM_WORKERS="$CDM_WORKERS" \
    OMNIDOCBENCH_EVAL_TOOLS_ROOT="$OMNIDOCBENCH_EVAL_TOOLS_ROOT" \
    OMNIDOCBENCH_TOOL_ROOT="$OMNIDOCBENCH_TOOL_ROOT" \
    CDM_PDFLATEX="$CDM_PDFLATEX" CDM_KPSEWHICH="$CDM_KPSEWHICH" \
    "$0" worker "$root" >"$root/run.log" 2>&1 &
  printf '%s\n' "$!" >"$root/pid.txt"
  printf 'AUDIT_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' "$root" "$root/run.log" "$(cat "$root/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then
  resolve
  entry "$2"
else
  launch
fi
