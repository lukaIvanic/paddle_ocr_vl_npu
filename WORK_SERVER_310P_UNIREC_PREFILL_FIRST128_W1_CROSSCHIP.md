# 310P versus 910B UniRec W1 first-128 prefill

Pull the commit containing this brief and run this task only. This is the 310P
half of an exact cross-chip page-path comparison. The 910B half uses the same
arguments.

This is deliberately **W1 only**. Do not start W2, W4, W8, decode, or the full
1,651-page benchmark after it finishes.

## Fixed experiment contract

- OmniDocBench offset 0, first 128 pages;
- one process worker;
- one persistent CPU crop-resize thread;
- all 128 pages once as an in-process warmup, then the same 128 measured;
- compiled FP16 B1 optimized layout;
- index-free FPN nearest-neighbor upsampling in the current layout source;
- the five production compiled full-vision buckets;
- `constant_grouped_all` plus `torchair_internal` vision weights;
- compact uint8 HWC crop transfer and NPU normalization;
- cross-KV length 512 and artifact storage `discard`;
- one too-large crop may use the established eager fallback.

Do not change any flag, add profiling, or run decode. The purpose is the actual
W1 page path and its existing stage timers, not another graph microbenchmark.
The layout cache key includes the graph source, so an older cache parent is
safe: the current source must resolve to its own graph-cache child.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Launch in the background and immediately send Luka the absolute log path.
- Reuse the exact model, dataset, OpenOCR, and layout cache paths from the
  latest successful UniRec work on this server.
- Do not point `RECOGNITION_CACHE` at the native-weight all-45 cache built at
  commit `b3d331e`. Use the cache from the earlier passed
  `constant_grouped_all + torchair_internal` single-bucket run if it is
  available; otherwise use a new dedicated cache path. The W1 worker will load
  or build all five exact combined-lane graphs during excluded setup.

## Launch

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench images directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the warmed optimized-layout cache parent}"
export RECOGNITION_CACHE="${RECOGNITION_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_all45_internal_first128_$(git rev-parse --short HEAD)}"
mkdir -p "$RECOGNITION_CACHE"

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_prefill_first128_w1_crosschip_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_PREFILL_FIRST128_W1_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P W1 FIRST128 STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Then follow only that owned process:

```bash
tail -f "$RUN_LOG"
```

Every completed page is printed. A 15-second heartbeat diagnoses a genuine
stall. Setup/cache loading is outside measured `producer_wall_s`; the complete
128-page warmup occurs inside the persistent worker before measurement.

## Completion report

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
cat "$RUN_ROOT/report.log"
```

Return:

1. commit, physical NPU, CANN, torch, and torch_npu versions;
2. all five `UNIREC_PREFILL_FIRST128_W1*` lines, including the matched 910B
   ratios;
3. process wall time;
4. absolute `run.log`, `summary.json`, and recognition-cache paths.

The workload must report 128 pages, 950 accepted crops, 6 rejected crops,
61,596 real source tokens, 1 fallback, 1,424 physical compiled vision rows,
and vision slot efficiency 0.667134831. If any differs, label it a workload
mismatch and do not calculate a cross-chip ratio.

If the run fails, return the first causal error and the last completed page.
Do not retry with changed flags.

## Fixed 910B2 reference

The launcher calculates ratios against this already-completed, same-argument
910B2 run at source commit `69eaf86` (the later commit only documented it):

```text
producer wall       15.334605 s
throughput            8.347134 pages/s
layout stage          4.003283 s
input prepare         3.725054 s
recognition prefill   4.274959 s
cache D2H             0.639309 s
shared pack           1.172008 s
IPC delivery          0.601004 s
RGB decode            1.042776 s
```

The 910B2 evidence is:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/prefill_first128_w1_crosschip_69eaf86_20260813T181612/output/summary.json
```

Do not compare setup or warmup wall time across chips. Compare the measured
`producer_wall_s` and its stage sums. Setup includes cache discovery/load, and
the first complete 128-page pass is an excluded warmup.
