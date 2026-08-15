# 310P UniRec grouped-FZ buckets plus optimized full prefill

Pull the commit containing this brief and run this task only. The launcher does:

1. a lightweight native-compiled versus grouped-FZ-compiled latency A/B for
   each of the five production vision buckets;
2. one optimized-only full 1,651-page OmniDocBench prefill run.

It does **not** run a full native baseline and does not run decode.

## Exact candidate

Use only:

- vision rewrite: `constant_grouped`;
- vision weight format: `native`;
- the five existing production fixed-shape buckets;
- the already validated optimized 310P layout configuration for the full run.

Do not substitute `constant_grouped_all`, `torchair_internal` vision weights,
an obsolete block-expanded vision lane, eager vision, or an ONNX model. Do not
search for stock OpenOCR or ONNX assets. Reuse the exact paths and caches from
the last successful UniRec production-prefill run on this server. If one cannot
be resolved, stop and name only the missing variable.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Launch in the background and immediately send Luka the absolute log path.
- Do not add device-stage profiling to the full page run. It previously failed
  through `NPUEvent.elapsed_time()` in the process pool.
- Do not retry with changed flags after a failure. Preserve the first error.

## What the bucket gate proves

The bucket phase skips full NPU profiling. It warms each graph and measures 20
clean NPU-event samples, split before and after the omitted profiler window. It
saves each native compiled output and requires the matching grouped-FZ compiled
output to be bit-exact. The grouped cache generated here is reused by the full
run, so the eight workers must load rather than cold-compile the five graphs.

On 910B2 at commit `433ede2`, physical NPU 7, all five outputs were bit-exact:

| bucket | native compiled | grouped-FZ compiled | speedup |
|---|---:|---:|---:|
| `960x64_b16` | 14.376645 ms | 11.915343 ms | 1.207x |
| `512x256_b16` | 20.021389 ms | 17.477865 ms | 1.146x |
| `960x256_b4` | 13.391364 ms | 11.595694 ms | 1.155x |
| `512x512_b8` | 18.414016 ms | 16.987336 ms | 1.084x |
| `960x512_b4` | 18.363485 ms | 16.896091 ms | 1.087x |

The first-128 weighted graph aggregate improved from 2.056269 s to 1.764168 s,
or 1.166x. The controlled full 910B2 W8 run did not improve: 75.947 s native
versus 77.005 s grouped-FZ. This is context only, not a 310P prediction.

## Launch

Use the exact already-passed paths from the previous 310P UniRec run:

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
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench image directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the successful optimized-layout cache parent}"
export BASELINE_RECOGNITION_CACHE="${BASELINE_RECOGNITION_CACHE:?reuse the successful native five-vision-graph plus text-prefill cache parent}"
export OPT_RECOGNITION_CACHE="${OPT_RECOGNITION_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_grouped_fz_22_$(git rev-parse --short HEAD)}"
export HISTORICAL_PREFILL_S=350

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_310p_grouped_fz_buckets_full1651_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_310P_GROUPED_FZ_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P GROUPED-FZ BUCKETS + FULL1651 STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Then follow only the owned background job:

```bash
tail -f "$RUN_LOG"
```

The full run prints every completed page and a heartbeat after 15 seconds with
no completion, so silence is diagnosable.

## Completion report

After the process exits:

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
cat "$RUN_ROOT/buckets/comparison.log"
cat "$RUN_ROOT/full1651_optimized/report.log"
```

Return:

1. commit, physical NPU, CANN, torch, and torch_npu versions;
2. all five `UNIREC_VISION_BUCKET_COMPILED_AB` lines;
3. `UNIREC_VISION_BUCKET_COMPILED_AB_WEIGHTED`;
4. `UNIREC_310P_GROUPED_FZ_FULL1651_OPTIMIZED`;
5. `UNIREC_310P_GROUPED_FZ_FULL1651_STAGES`;
6. `UNIREC_310P_GROUPED_FZ_FULL1651_HISTORICAL_CONTEXT`;
7. process wall time, absolute run log, bucket comparison JSON, full summary
   JSON, and optimized recognition-cache path.

The 350 s comparison is explicitly historical and approximate. Do not call it
a controlled A/B. Then stop; do not run decode.

If a bucket is not bit-exact, stop before the full run and return the bucket and
its exact max/mean difference. If the process fails, return the last completed
phase and the first causal error from `run.log`.
