#!/usr/bin/env bash
# Score one complete full-corpus experiment-17 run without changing predictions.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stock_repo="$PWD"
stock_root="${RUN_ROOT:?Set RUN_ROOT to the completed experiment-17 run directory}"
stock_root="$(cd "$stock_root" && pwd)"
source 09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
stock_eval="$stock_root/evaluation_full"
stock_eval_commit="$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" rev-parse HEAD)"
test "$stock_eval_commit" = 2b161d010d2e3aff77a0edef359ea3a6411d23cd
test -z "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" status --porcelain --untracked-files=no -- . ':!result')"
test "$(cat "$stock_root/exit_code.txt")" = 0
"$OMNIDOCBENCH_EVAL_PYTHON" 17_mineru_vllm_ascend_baseline/prepare_omnidocbench_eval.py \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --run-output "$stock_root/output" --evaluation-root "$stock_eval" \
  --expected-pages "${LIMIT:-1651}" --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"
git rev-parse HEAD > "$stock_eval/authoring_commit.txt"
printf '%s\n' "$stock_eval_commit" > "$stock_eval/evaluator_commit.txt"
sha256sum 09_persistent_page_engine/scripts/run_omnidocbench_eval.py \
  09_persistent_page_engine/scripts/omnidocbench_eval_env.sh \
  17_mineru_vllm_ascend_baseline/prepare_omnidocbench_eval.py > "$stock_eval/source_hashes.txt"
"$OMNIDOCBENCH_EVAL_PYTHON" 09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" > "$stock_eval/runtime_smoke.json"
stock_args=("$stock_repo/09_persistent_page_engine/scripts/run_omnidocbench_eval.py"
  --config "$stock_eval/work/config.yaml" --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"
  --match-workers 24 --teds-workers 12 --page-timeout-sec 420 --fallback-timeout-sec 180)
printf '%q ' "$OMNIDOCBENCH_EVAL_PYTHON" "${stock_args[@]}" > "$stock_eval/command.txt"
printf '\n' >> "$stock_eval/command.txt"
cd "$stock_eval/work"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
/usr/bin/time -f '%e' -o "$stock_eval/wall_s.txt" \
  "$OMNIDOCBENCH_EVAL_PYTHON" "${stock_args[@]}" 2>&1 | tee "$stock_eval/run.log"
stock_exit=${PIPESTATUS[0]}
printf '%s\n' "$stock_exit" > "$stock_eval/exit_code.txt"
exit "$stock_exit"
