#!/usr/bin/env bash
# Score complete custom predictions without changing their text or image tags.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mineru_repo="$PWD"
mineru_root="${RUN_ROOT:?Set RUN_ROOT to the completed run directory}"
mineru_root="$(cd "$mineru_root" && pwd)"
source 09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
mineru_eval="$mineru_root/evaluation"
mineru_eval_commit="$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" rev-parse HEAD)"
test "$mineru_eval_commit" = 2b161d010d2e3aff77a0edef359ea3a6411d23cd
# Existing evaluator result files may be dirty; evaluator source must be clean.
test -z "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" status --porcelain --untracked-files=no -- . ':!result')"
test "$(cat "$mineru_root/exit_code.txt")" = 0
"$OMNIDOCBENCH_EVAL_PYTHON" 11_mineru_2_5_pro_inference/prepare_serving_eval.py \
  --run-output "$mineru_root/output" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --evaluation-root "$mineru_eval" --expected-pages "${LIMIT:-1651}" \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"
git rev-parse HEAD > "$mineru_eval/authoring_commit.txt"
printf '%s\n' "$mineru_eval_commit" > "$mineru_eval/evaluator_commit.txt"
sha256sum 09_persistent_page_engine/scripts/run_omnidocbench_eval.py \
  09_persistent_page_engine/scripts/omnidocbench_eval_env.sh \
  11_mineru_2_5_pro_inference/prepare_serving_eval.py > "$mineru_eval/source_hashes.txt"
"$OMNIDOCBENCH_EVAL_PYTHON" 09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" > "$mineru_eval/runtime_smoke.json"
mineru_args=("$mineru_repo/09_persistent_page_engine/scripts/run_omnidocbench_eval.py"
  --config "$mineru_eval/work/config.yaml" --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"
  --match-workers 24 --teds-workers 12 --page-timeout-sec 420 --fallback-timeout-sec 180)
printf '%q ' "$OMNIDOCBENCH_EVAL_PYTHON" "${mineru_args[@]}" > "$mineru_eval/command.txt"
printf '\n' >> "$mineru_eval/command.txt"
cd "$mineru_eval/work"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
/usr/bin/time -f '%e' -o "$mineru_eval/wall_s.txt" \
  "$OMNIDOCBENCH_EVAL_PYTHON" "${mineru_args[@]}" 2>&1 | tee "$mineru_eval/run.log"
mineru_exit=${PIPESTATUS[0]}
printf '%s\n' "$mineru_exit" > "$mineru_eval/exit_code.txt"
exit "$mineru_exit"
