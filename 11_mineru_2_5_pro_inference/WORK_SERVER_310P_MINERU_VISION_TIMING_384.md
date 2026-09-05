# 310P: explain the completed full run, then request approval for production timing

This is a new handoff for the pull-only 310P work agent. Luka has completed the
manual-FP32/PSE-sentinel 1651-page run. First explain its saved statistics to
Luka. **Wait for his explicit approval before launching the 384-page run.**
Earlier handoffs that automatically chain smoke/full inference/evaluation do
not apply to this task.

## Environment and source

Read `CLAUDE.md` and `AGENTS.md` for lane rules. You cannot reach the 910B
container or Luka's Mac; they cannot reach you. Use your existing completed
310P run, Python environment, CANN activation, model, dataset and graph caches.
Do not change packages, models, source, caches, or evaluator files. Do not
commit, push, branch, reset, stash, or discard changes. Report an issue directly
to Luka in plain text if blocked; do not propose code edits or write a report
file. Generated benchmark JSON, logs, commands and traces are required artifacts.

Resolve the checkout using `git rev-parse --show-toplevel`. Inspect tracked
changes before pulling; preserve all existing changes. Pull with
`git pull --ff-only origin main` and require
`git merge-base --is-ancestor 9bc93294179af403117fbba92a3de37772e95e85 HEAD`.
Stop and explain if a pull conflict prevents this; do not resolve by discarding
user work.

Resolve and export:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
# Set these to the actual paths from your successful 1651-page run:
export FULL_RUN_ROOT=/absolute/path/to/the/completed/310p/full1651
export REFERENCE_SUMMARY="$FULL_RUN_ROOT/output/run_summary_shard_00.json"
export PYTHON_BIN=/absolute/path/to/the/same/verified/exp11/python
test -s "$REFERENCE_SUMMARY"
test -x "$PYTHON_BIN"
```

These two placeholder paths must be resolved from your own saved command and
artifacts. If several completed runs exist, inspect their timestamps, commit,
settings and exit codes and use the successful manual-FP32/PSE full run Luka
just reported at about 0.17 pg/s. Ask Luka only if it remains ambiguous.
Do not reuse a 910B `/workspace` path. Cache paths will be read from the selected
summary; repo-relative paths resolve against your checkout.

## Before approval: explain the existing results

This command is read-only and needs no NPU:

```bash
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/vision_timing_report.py \
  "$REFERENCE_SUMMARY"
```

Read the summary itself and its matching successful exit file. Explain to Luka:

- completed/failed/skipped pages, hot pg/s and hot wall time; setup and warmup
  separately, so about 0.17 pg/s is interpreted with the correct denominator;
- vision real tokens / vision transformer event seconds, text-prefill tokens /
  text transformer event seconds, effective decode tokens / decode event seconds;
- raw B32 slot tok/s, graph calls, mean graph latency, occupancy and idle slots
  with ready work; occupancy describes decode slots, not overall NPU utilization;
- total request count, real vs physical token padding, vision route distribution,
  eager overflows, CPU preparation wait/work, H2D, KV redistribution, token-copy
  wait and submission times. Event sums and CPU work may overlap: don't sum them
  indiscriminately or call a device-event region pure kernel execution;
- any supported bottleneck conclusions and what cannot be inferred yet. Old
  per-bucket `first_call_s` values are cache-load/compile times, not hot latency.

The 910B full-run reference at matching settings was 0.802859 pg/s and 2056.400 s:
vision 44,143.9 real tok/s (656.928 s), text prefill 17,361.1 real tok/s
(470.785 s), decode effective 7,587.6 tok/s, raw B32 slots 7,716.7 tok/s
(335.410 s), mean decode graph 4.147 ms, occupancy 99.754%.
Use full-run-to-full-run comparisons here. If the 310P denominator differs,
state that before computing a slowdown.

Then ask Luka in plain text: "May I run the first 384 pages with per-call vision
timing, using the same production settings and existing caches?"
Stop here until he explicitly approves. Do not interpret silence as approval.

## After approval: launch the production run

Source the same server-owned CANN activation as the successful full run before
enabling shell nounset. Verify one healthy free 310P and select it with
`ASCEND_RT_VISIBLE_DEVICES`. Do not terminate another user's process. Export
`VLLM_WORKER_MULTIPROC_METHOD=spawn` before any torch-npu import.
Ensure no other job uses these caches concurrently. Reuse the existing cache
root and acquire the existing run's lock if it is still the site's convention;
the launcher also takes a nonblocking lock inside the decode cache directory.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
"$PYTHON_BIN" -m unittest discover -s 11_mineru_2_5_pro_inference -p 'test_*.py'
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_vision_timing_384_$(git rev-parse --short=12 HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN_ROOT"
# The launcher creates RUN_ROOT itself and refuses to overwrite any prior run.
export CONTROL_LOG="${RUN_ROOT}.launcher.log"
nohup setsid "$PYTHON_BIN" \
  11_mineru_2_5_pro_inference/run_vision_timing_production.py \
  --reference-summary "$REFERENCE_SUMMARY" \
  --run-root "$RUN_ROOT" --limit 384 \
  </dev/null >"$CONTROL_LOG" 2>&1 &
```

Send Luka `RUN_ROOT/run.log` and `CONTROL_LOG` immediately. The launcher records
PID, source commit, device, complete command, reference summary, inference exit,
validation exit and process wall time. It enforces FP16, B32/KV4096, 32-page
streaming window, lookahead 32, preparation depth 64, packed-768 vision,
manual-FP32 LN + nn.Linear + native D80 PromptFA, packed text buckets
128/256/512/1024, NZ decode, NPU rotary and
`--local-decode-increfa-length-mode pse_sentinel_310p`.
That mode passes `pse_shift` to IncreFA; it is the same path used for the 910B
validation. The two warmup pages and all bucket warmups precede the hot timer.
The launcher checks asset hashes against your full run after inference.

## Monitor until completion

Keep monitoring live. Prefer a tool execution timeout of at least 10,800,000 ms
(180 minutes); if a tool's maximum is lower, launch detached and keep
reattaching. A tool timeout is not benchmark completion. Every 30–60 seconds,
inspect the child-written exit file, progress count and recent phase markers.
Use the local `run.log` and process tree; do not issue thousands of tiny polls.
Keep Luka informed of meaningful progress or errors. If pages finish slowly,
check request IDs and matched phase start/finish records before calling it a
stall. Do not restart the job merely because a log is quiet.

Require all of these:

- `inference_exit_code.txt` and `exit_code.txt` both equal zero;
- `VISION_TIMING_VALIDATION: PASS` in `CONTROL_LOG`;
- 384 completed pages, zero failed/skipped, matching predictions/content/progress;
- raw timing calls, real/physical tokens, route counts and event seconds all
  reconcile with the original aggregate metrics (the launcher checks these);
- warmup excluded; PSE mode and manual-FP32 vision metadata confirmed.

The launcher prints uncovered routes. Report them honestly. Do not launch extra
pages or synthetic shapes automatically; obtain Luka's approval for more work.
Do not perform a new accuracy evaluation in this task.

## Explain the new timing results to Luka

Run the same report command on `$RUN_ROOT/output/run_summary_shard_00.json`.
Use the preserved 910B results in `references/vision_timing_384_910b/` for the
384-to-384 comparison. Report directly in plain text, with concise tables if
helpful; do not create a separate report file.

Report per-route calls, total device seconds and share of vision time, useful
and physical tok/s, useful-token fraction, latency mean/std/min/p50/p90/p95/p99/max.
Keep `packed_768` separate from `bucket_768`. Report exact sequence lengths and
latency distributions for every eager-overflow shape. Show the slowest calls
and request IDs, and distinguish frequent slow work from rare outliers.
Use weighted throughput `sum(tokens)/sum(seconds)`; percentiles are unweighted
per-call linear-interpolated percentiles. Note low sample counts near p99.

Explain which routes dominate total time, how padding affects useful throughput,
how much overflow costs, and which route has the largest 310P/910B latency ratio.
Compare stage-level decode metrics already collected; no decode-position study
is needed. Equal buckets can contain different member lengths, so compare the
raw records and token distribution before assigning a hardware cause.

Raw records are in `output/vision_timing_shard_00.jsonl`; all exact-shape summaries
and 20 slowest calls are in `vision_timing` within the run summary. Preserve them
for later analysis. These are production event regions; direct routes include
padding preparation while packed mask construction occurs before the event.

If anything fails, report the command, exit status, first causal error, last
unmatched marker, run/control-log paths and reused cache paths to Luka. Stop;
do not edit source or invent a workaround.
