# 310P UniRec full-1651 prefill A/B

Pull the commit named by Luka and run this task only. It compares the current
native production prefill against the combined 310P layout and vision
candidate. It does not run decode.

The launcher performs three sequential phases on one physical NPU:

1. full-1651 native baseline;
2. one-worker seeding of all five optimized vision graphs;
3. full-1651 optimized candidate.

Both measured lanes use eight workers, eight CPU preprocessing threads per
worker, an eight-page in-pool warmup, compact HWC input, cross-KV 512, and
discard-only output. Per-device event timing is intentionally disabled: the
first 310P attempt failed at `NPUEvent.elapsed_time()` on page 161, while the
pool already uses spawned workers and wall/aggregate stage timing does not need
NPU events. The report requires identical page counts. It reports but does not
reject crop/token drift from the FP16 layout candidate; those are quality
inputs, not an arbitrary speed-test tolerance.

Every completed page is printed with its completion count, source page index,
worker, worker time, elapsed time, crop count, and rejected-crop count. If no
page finishes for 15 seconds, the runner prints a heartbeat with every worker's
liveness and exit code. A dead worker now fails immediately instead of leaving
the coordinator silent for its long safety timeout.

## 910B2 reference, not a 310P prediction

At commit `46570ba`, physical 910B2 NPU 1:

```text
native baseline:  80.472723 s, 20.516269 pages/s
vision candidate: 96.423615 s, 17.122362 pages/s
ratio:            0.834x (candidate is 19.8% slower)
contract:         1651 pages, 31686 crops, 586 rejected,
                  1895032 real source tokens, 496 eager vision fallbacks
```

The candidate was faster in isolated 910B2 graph tests but slower with eight
processes sharing the NPU. Therefore, do not infer the 310P production result
from isolated graphs or from the 910B2 result.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Reuse the exact successful native five-graph recognition cache.
- Reuse the exact successful current `constant_grouped + torchair_internal +
  preformatted FrozenBN` layout cache parent.
- Give the optimized vision lane its own cache path. The launcher seeds it with
  one worker before W8 starts.
- Do not add `--profile-prefill-device-stages`. That optional instrumentation
  creates the NPU timing events that failed in the first 310P attempt.
- Launch in the background. Send Luka the absolute log path immediately.
- Do not retry or alter flags after a hard process exit. Report the first error.

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
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench image directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?set the passed optimized-layout cache parent}"
export BASELINE_RECOGNITION_CACHE="${BASELINE_RECOGNITION_CACHE:?set the passed native five-graph cache}"
export OPT_RECOGNITION_CACHE="${OPT_RECOGNITION_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_allfocal_full1651_$(git rev-parse --short HEAD)}"

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_310p_prefill_full1651_ab_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_310P_PREFILL_FULL1651_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P PREFILL FULL1651 A/B STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Follow only this owned process:

```bash
tail -f "$RUN_LOG"
```

## Completion

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
cat "$RUN_ROOT/comparison.log"
```

Return the three `UNIREC_310P_PREFILL_FULL1651_*` lines. Also state the commit,
physical NPU, runtime versions, process wall time, absolute run log, both
summary JSON paths, and all three cache paths. Then stop.

If the run fails, return the last completed phase and the first causal error.
Do not start decode or another lane.
