# 310P UniRec W4 prefill tail and NPU-contention analysis

## Objective

Explain the current representative-128 W4 prefill result (reported near
6.87 pages/s) at the level of individual calls and tail latency. Measure:

- p50/p90/p95/p99/max for every recorded stage and substage;
- every layout forward, compiled vision call, eager vision fallback, packed
  text-prefill call, crop preprocess, cross-KV D2H, shared pack, and IPC event;
- the sum of NPU service time across all four workers;
- per-bucket NPU latency and its ratio to the committed 910B2 W4 control;
- clean-run inter-page completion gaps and their relationship to the four
  non-writeable NumPy warnings;
- the exact pages/crops/events responsible for the largest tails.

Analysis comes first. Do not rerun inference if the completed W4 trace and clean
artifacts still exist. Do not run decode or evaluation.

Use the commit containing this brief or later.

## Work-server rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use the already-completed W4 run before starting any new NPU work.
- If a rerun is necessary, use one free physical 310P in `0..3`.
- Do not run `npu-setup`; it is not installed on this server.
- Preserve the validated `python_nosym` path. Do not apply `readlink -f` to it.
- Do not use `nproc`. `os.sched_getaffinity(0)` is authoritative.
- Do not delete or rebuild graph caches.

## Pull and select the completed W4 run

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git rev-parse HEAD

test -x "${PYTHON_BIN:?validated python_nosym executable}"
test "$(basename "$PYTHON_BIN")" = python_nosym
test -f 12_unirec_0_1b_inference/analyze_prefill_tail_contention.py
test -f 12_unirec_0_1b_inference/references/unirec_910b_w4_prefill_tail_contention_20260817.json
```

Set `RUN_ROOT` to the absolute root of the completed W4 layout-T4/crop-T8 run
that produced the reported result. It must contain both lanes:

```bash
export RUN_ROOT=/absolute/path/to/completed/W4/run
test -f "$RUN_ROOT/trace/output/run_summary.json"
test -f "$RUN_ROOT/trace/output/prefill_iterations.jsonl"
test -f "$RUN_ROOT/trace/output/prefill_pages.jsonl"
test -f "$RUN_ROOT/clean/output/run_summary.json"
test -f "$RUN_ROOT/clean/run.log"
```

Validate that this is the right run. Stop and report the mismatch if any assert
fails. Do not silently select another configuration.

```bash
"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
t = json.loads((root / "trace/output/run_summary.json").read_text())
c = json.loads((root / "clean/output/run_summary.json").read_text())
for run, traced in ((t, True), (c, False)):
    assert run["status"] == "ok"
    assert run["execution"] == "production_two_phase_prefill_only"
    assert run["page_count"] == 128
    assert run["workers"] == 4
    assert run["layout_cpu_threads"] == 4
    assert run["recognition_preprocess_threads"] == 8
    assert run["layout_batch_size"] == 1
    assert run["vision_page_lookahead"] == 1
    assert run["vision_bucket_preset"] == "310p_k10_l1"
    assert run["prefill_trace_enabled"] is traced
assert t["retained_bank"]["crop_count"] == 2489
assert c["retained_bank"]["crop_count"] == 2489
assert t["retained_bank"]["rejected_crop_count"] == 0
assert c["retained_bank"]["rejected_crop_count"] == 0
print(
    "UNIREC_310P_W4_TAIL_INPUT: PASS",
    "trace_wall_s=", t["timing_s"]["prefill_phase"],
    "clean_wall_s=", c["timing_s"]["prefill_phase"],
    "clean_pages_s=", c["throughput"]["prefill_pages_per_s"],
)
PY
```

## Run the analysis (CPU only)

This reads completed JSONL and logs. It does not touch the NPU.

```bash
ANALYSIS_JSON="$RUN_ROOT/prefill_tail_contention_310p_w4.json"
ANALYSIS_LOG="$RUN_ROOT/prefill_tail_contention_310p_w4.log"
"$PYTHON_BIN" \
  12_unirec_0_1b_inference/analyze_prefill_tail_contention.py \
  --trace-summary "$RUN_ROOT/trace/output/run_summary.json" \
  --iterations "$RUN_ROOT/trace/output/prefill_iterations.jsonl" \
  --pages "$RUN_ROOT/trace/output/prefill_pages.jsonl" \
  --clean-log "$RUN_ROOT/clean/run.log" \
  --label 310P_W4_L4_C8 \
  --reference-json \
    12_unirec_0_1b_inference/references/unirec_910b_w4_prefill_tail_contention_20260817.json \
  --output-json "$ANALYSIS_JSON" \
  --top 20 | tee "$ANALYSIS_LOG"
```

The full JSON retains each stage distribution and the 20 slowest individual
events with page, crop, shape, bucket, and token metadata where available.

## Required report

Paste the complete analysis log. Also report:

1. Absolute `RUN_ROOT`, `ANALYSIS_JSON`, and `ANALYSIS_LOG`.
2. Trace wall, clean wall, clean pages/s, and exact page/crop/token counts.
3. NPU service sums for layout forward, compiled vision, eager fallback, and
   text prefill. Report their aggregate sum and aggregate-sum/trace-wall ratio.
4. For every NPU component: count, mean, p50, p95, p99, and max.
5. For all ten compiled vision buckets: count, mean, p95, p99, max, slot
   efficiency, and the 310P/910B mean and p99 ratios.
6. For these CPU/transport stages: sum, mean, p50, p95, p99, and max:
   - layout processor preprocess;
   - layout bicubic resize;
   - layout contiguous CHW materialization;
   - recognition crop preprocess and resize;
   - coordinator IPC delivery;
   - page shared pack;
   - cross-KV D2H.
7. The 20 largest clean inter-page completion gaps. Include page index, worker,
   worker-page time, crop count, and `warning_between`.
8. All warning-adjacent gaps. Compare them with non-warning gap p50/p95/p99.
9. The five slowest individual events for each of:
   - layout processor preprocess;
   - layout contiguous CHW materialization;
   - crop resize;
   - compiled vision graph;
   - eager fallback input submit and graph;
   - packed text-prefill wall and device time;
   - coordinator IPC;
   - page shared pack.
10. The five pages with the largest `worker_group_wall_s`, `npu_service_s`,
    crop preprocess service, and fallback graph service.

Do not attribute a completion gap to the warning merely because the warning is
printed between two completions. The warning is emitted once per process at the
first read-only NumPy tensor wrap. Compare the measured fallback input-submit
latency and the page's real work first.

## Only if the completed W4 trace is missing

Rerun the exact existing W4 lane with the same warmed caches and CPU affinity
that produced the 6.87 pages/s result:

```bash
export UNIREC_K10_CHIP_LABEL=310P
export UNIREC_K10_ALLOWED_DEVICES=0,1,2,3
export UNIREC_K10_WORKERS=4
export UNIREC_K10_LAYOUT_CPU_THREADS=4
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=both
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Return the absolute background log path immediately. Wait for exit zero, then
run the analysis above. Confirm that no new OM or `compiled_module` appeared.
Stop after the report.
