# 310P UniRec MSDA post-hoc operator delta

Do not run any NPU work. Re-analyze the completed 128-page A/B artifacts with
the updated analyzer and report which operator types made native MSDA slower.

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Do not source `npu-setup` and do not launch a model or profiler.
- Reuse the completed baseline/candidate forward JSON and profile-suite JSON
  files. Do not repeat any phase of the A/B.
- Preserve the original comparison JSON. Write a new post-hoc JSON.

## Run

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

export PYTHON_BIN="${PYTHON_BIN:?set the existing venv python_nosym executable}"
export RUN_ROOT="${RUN_ROOT:?set the completed 128-page MSDA A/B run root}"

test -f "$RUN_ROOT/forward_baseline.json"
test -f "$RUN_ROOT/forward_candidate.json"
test -f "$RUN_ROOT/profile_baseline/profile_suite_summary.json"
test -f "$RUN_ROOT/profile_candidate/profile_suite_summary.json"

"$PYTHON_BIN" 12_unirec_0_1b_inference/compare_layout_msda_ab.py \
  --baseline-forward "$RUN_ROOT/forward_baseline.json" \
  --candidate-forward "$RUN_ROOT/forward_candidate.json" \
  --baseline-profile "$RUN_ROOT/profile_baseline/profile_suite_summary.json" \
  --candidate-profile "$RUN_ROOT/profile_candidate/profile_suite_summary.json" \
  --output "$RUN_ROOT/comparison_summary_op_delta.json" \
  | tee "$RUN_ROOT/posthoc_op_delta.log"
```

## Required report

Return:

- commit and absolute completed `RUN_ROOT`;
- complete `UNIREC_LAYOUT_MSDA_REAL_OP_DELTA` line;
- complete `UNIREC_LAYOUT_MSDA_REAL_OP_REGRESSIONS` line;
- complete `UNIREC_LAYOUT_MSDA_REAL_OP_SAVINGS` line;
- from `comparison_summary_op_delta.json`, the first 20 `top_regressions` and
  first 20 `top_savings`, including counts, milliseconds, delta, and core type;
- specifically report baseline → candidate count/time for `ScatterElements`,
  `MultiScaleDeformableAttnFunction`, `GridSample`, `Transpose`, `Cast`,
  `TransData`, `Add`, `BroadcastTo`, `Cumsum`, and `ReduceProdD`;
- absolute post-hoc JSON and log paths.

Then stop. No NPU run is authorized by this brief.
