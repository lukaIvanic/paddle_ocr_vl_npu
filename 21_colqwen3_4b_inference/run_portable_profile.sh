#!/usr/bin/env bash
# Run each execution lane in a fresh process, then summarize automatically.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PYTHON_BIN:?Select the successful ColQwen interpreter}"
: "${COLQWEN_MODEL:?Select the existing checkpoint}"
: "${HR_DATASET:?Select the verified English HR dataset}"
: "${COLQWEN_CACHE:?Select an exclusive local cache}"
: "${PROFILE_ROOT:?Choose a new output directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select one idle physical NPU}"
test -x "$PYTHON_BIN"
test ! -e "$PROFILE_ROOT"
mkdir -p "$PROFILE_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
read -r -a metrics <<< "${PROFILE_METRICS:-pipe memory}"
read -r -a scopes <<< "${PROFILE_SCOPES:-full vision text}"
read -r -a executions <<< "${PROFILE_EXECUTIONS:-torchair raw_eager}"
run_logged() {
    local label="$1"
    shift
    { git rev-parse HEAD; hostname; printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"; printf '%q ' "$@"; printf '\n'; } > "$PROFILE_ROOT/$label.command.txt"
    set +e
    "$@" 2>&1 | tee "$PROFILE_ROOT/$label.log"
    local codes=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${codes[0]}" > "$PROFILE_ROOT/$label.exit_code.txt"
    test "${codes[0]}" -eq 0 && test "${codes[1]}" -eq 0
}
for execution in "${executions[@]}"; do
    case "$execution" in torchair|raw_eager) ;; *) printf 'Invalid execution: %s\n' "$execution"; exit 2;; esac
    run_logged "$execution" "$PYTHON_BIN" -u 21_colqwen3_4b_inference/profile_portable_stages.py \
      --model "$COLQWEN_MODEL" --dataset-root "$HR_DATASET" --cache-root "$COLQWEN_CACHE" \
      --output-dir "$PROFILE_ROOT/$execution" --execution "$execution" \
      --expected-chip "${EXPECTED_CHIP:-310P}" --page-index "${PROFILE_PAGE_INDEX:-5}" \
      --warmups "${PROFILE_WARMUPS:-5}" --repeats "${PROFILE_REPEATS:-20}" \
      --profile-steps "${PROFILE_STEPS:-3}" --metrics "${metrics[@]}" --scopes "${scopes[@]}"
    run_logged "analysis_after_$execution" "$PYTHON_BIN" -u \
      21_colqwen3_4b_inference/analyze_portable_profile.py --run-root "$PROFILE_ROOT"
done
printf 'PROFILE_SUITE_COMPLETE %s/summary.json\n' "$PROFILE_ROOT"
