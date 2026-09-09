# 310P: evaluate the completed half-ready hybrid run, all 1,651 pages

## Task and authority

Luka now authorizes OmniDocBench accuracy evaluation of the successful run at
inference commit `dd172553a4ef`. This is a NEW evaluation-only handoff; it
supersedes the previous brief's instruction to defer evaluation, not its
inference settings. Do not rerun inference, layout, crop recognition, compilation,
or any performance benchmark. No NPU is needed; do not source `npu-setup`.

The completed run reported 1,651 pages and 30,570 crops: 28,136 UniRec text,
1,683 Paddle formula and 751 Paddle table, pipeline wall 1,570.69 s. Its root is:

```text
tmp/18_unirec_paddle_hybrid_pipeline/310p_half_ready_full1651_dd172553a4ef_20260909T114817Z/
```

Use that run's `output/` containing `run_summary.json`, `recognition_trace.jsonl`
and the page Markdown/JSON files. If this location differs, resolve it from
the completed job's saved command and completion gate, not another partial or
newer run. Verify inference exit 0 and the retained `COMPLETION_PASS` evidence.
The expected stop counts include length/repetition/KV-limit stops; do not remove
those predictions. Missing pages are an error, not permission to evaluate a subset.

Read `CLAUDE.md`. The work agent is pull-only: no tracked-file/model/evaluator
edits, package installs, branches, commits or pushes. Generated evaluation
artifacts under a new job directory are allowed. Preserve original predictions,
inference summaries, logs and caches. Neither Luka's Mac nor the 910B server is
accessible here. Report any required code change instead of applying it.

## Pull and resolve the existing environment

In Bash, from the existing checkout:

```bash
set -euo pipefail
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
git pull --ff-only origin main
git merge-base --is-ancestor dd172553a4ef HEAD
export HYBRID_OUTPUT="$WORK_SERVER_REPO/tmp/18_unirec_paddle_hybrid_pipeline/310p_half_ready_full1651_dd172553a4ef_20260909T114817Z/output"
test -s "$HYBRID_OUTPUT/run_summary.json"
test -s "$HYBRID_OUTPUT/recognition_trace.jsonl"
```

Do not reset/stash unrelated local changes. If they overlap the evaluation code,
stop and report them. Repo visibility is managed by Luka, not this agent.

Recover these existing absolute paths from the successful same-host frozen
OmniDocBench evaluation commands/runtime fingerprint. Set them explicitly:

```bash
export DATASET_JSON=/absolute/path/to/OmniDocBench/OmniDocBench.json
export EVAL_PYTHON=/absolute/path/to/existing/eval/python
export EVALUATOR_ROOT=/absolute/path/to/existing/OmniDocBench_eval
export OMNIDOCBENCH_EVAL_TOOLS_ROOT=/absolute/path/to/existing/eval/tools
```

These are placeholders, not paths to create. The established
`candidate_runtime_fingerprint.json` records `platform.python_executable`,
`evaluator_root` and `tex_runtime.texlive_root`; inspect the fingerprint associated
with a successful CDM evaluation. Previous discovery instructions are in
`12_unirec_0_1b_inference/WORK_SERVER_310P_UNIREC_PERSISTENT_RESIDENT_K20_FULL1651.md`,
section "Recover the frozen evaluator". Do not assume `/workspace` paths apply.

Require evaluator commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`, unchanged
checkout, TeX Live 2025/pdfTeX 1.40.28 and ImageMagick 7.1.1-47. No ambient TeX
2022 substitution. The runner sources the existing env selector and verifies
actual formula rendering before full evaluation. Missing/invalid runtime means
report the blocker; do not install, update or change the scoring contract.

## Launch once, detached; monitor until completion

```bash
"$EVAL_PYTHON" -m unittest discover \
  -s 18_unirec_paddle_hybrid_pipeline/tests -p test_prepare_accuracy_eval.py -v
export EVAL_JOB="$(mktemp -d "$WORK_SERVER_REPO/tmp/18_unirec_paddle_hybrid_pipeline/310p_completed1651_accuracy.XXXXXX")"
{
  git rev-parse HEAD
  hostname
  printf 'inference_commit=dd172553a4ef\n'
  declare -p HYBRID_OUTPUT DATASET_JSON EVAL_PYTHON EVALUATOR_ROOT OMNIDOCBENCH_EVAL_TOOLS_ROOT EVAL_JOB
  printf 'bash 18_unirec_paddle_hybrid_pipeline/run_completed_accuracy_eval.sh\n'
} >"$EVAL_JOB/command.txt"
nohup setsid bash "$WORK_SERVER_REPO/18_unirec_paddle_hybrid_pipeline/run_completed_accuracy_eval.sh" \
  >"$EVAL_JOB/run.log" 2>&1 </dev/null &
echo "$!" >"$EVAL_JOB/pid.txt"
printf 'EVAL_JOB=%s\n' "$EVAL_JOB"
```

Keep monitoring this exact job until `exit_code.txt` exists and its process
exits. Use a tool execution timeout above 120 minutes where supported (e.g.
14,400,000 ms), with short 30–60-second progress checks. If a tool times out,
reattach to the same detached process; do not restart it. Report phase changes
and genuine failures to Luka. Quiet logs do not establish completion or a hang.
The runner records phase start/finish, durable exit code and elapsed seconds.

The preparer checks all 1,651 prediction stems against full ground truth. It
copies predictions into a separate flat evaluation directory and strips only
HTML `<img>` tags there, following the existing UniRec evaluation convention.
Original Markdown stays untouched. Both hashes and removed-tag counts are saved.
No crop-level oracle combination or removal of difficult pages is allowed.

The established process-isolated evaluator runs matching/TEDS first, then direct
CDM on the saved matched formulas, then the existing summary helper. Its existing
bounded matching recovery is retained and must be reported; introduce no new
fallbacks or timeout/metric changes. In particular, don't count formula edit
distance as CDM or substitute sample-averaged metrics for page metrics.

## Completion and report to Luka, directly in chat

Require `exit_code.txt = 0`, all phase finishes, source/prediction hash verification,
and `evaluation/full_eval_summary.json`. Verify page matching covered all 1,651
pages. Inspect finite score values plus timeout/error/fallback counts before
calling the score valid. If an evaluator stage fails, retain all logs/artifacts
and report the last completed stage; no repeated inference is necessary.

The reused helper prints `UNIREC_FULL_EVAL`, but the explicit lane is
`hybrid_310p_half_ready_full1651_dd172553a4ef`: this is HYBRID output, not an
all-UniRec accuracy result. Report these four anchors as percentages:

```text
Text accuracy = 100 * (1 - text_block_page_edit)
Page Table TEDS = 100 * table_page_teds
Page Formula CDM = 100 * display_formula_page_cdm
Overall = 100 * official_overall
        = mean(Text accuracy, Page Table TEDS, Page Formula CDM)
```

Also give page table-structure TEDS, formula edit distance, reading-order edit
distance (lower is better), evaluated formula samples/pages, matched table counts,
page-matching fallback/error counts, TEDS timeout/error counts, and CDM render/
timeout/error counts. Retain per-page/per-item metric artifacts for later analysis.
Do not claim a cross-chip accuracy delta without an actual comparable evaluated
910B result. Successful inference/parity records alone are not such a result.

Give project/evaluator/inference commits, resolved runtime and ground-truth hash,
original output path, evaluation job path, exit status and elapsed evaluation time.
The inference throughput remains 1.051 pg/s; evaluation time is separate and must
not be folded into inference throughput. End with a short interesting-findings
summary directly to Luka, not only a file link. This brief does not authorize
the pending prefill profiling or any new performance experiment.
