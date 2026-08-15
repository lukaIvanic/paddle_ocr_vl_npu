# 310P UniRec prepacked grouped focal weights

Pull the commit named by Luka and run this exact focused experiment. Do not edit
tracked files, create a branch, commit, or push. This tests the production
`960x64_b16` UniRec vision bucket only. It does not run layout, cross-KV,
decode, or the page pipeline.

The candidate packs the 22 stage-2/3 5x5 and 7x7 depthwise filters once on the
CPU into their grouped `FRACTAL_Z:<groups>` physical layout. TorchAir receives
the packed tensors as frozen graph inputs. The Conv operators consume them
directly. This is the exact solution to test on 310P; do not substitute an
obsolete block-expanded or aligned-spatial lane.

## 910B2 reference

Commit `3519412`, physical 910B2 NPU 7, CANN 9.0.0:

```text
bucket:                    960x64_b16
candidate steady mean:     11.824868 ms (3 warmups, 50 repeats, warm cache)
native steady mean:        14.459757 ms
candidate/native speedup:  1.2228x
parity:                    true at atol=5e-2, rtol=5e-2
max_abs / mean_abs:        0.0361328125 / 0.0009388924
TransData total:           2.34992 ms / 261 calls
target 5x5/7x7 repacks:    0 calls / 0 ms
```

The four target signatures were all absent:

```text
25,24,16,16 -> 600,1,16,16:   0
49,24,16,16 -> 1176,1,16,16:  0
25,48,16,16 -> 1200,1,16,16:  0
49,48,16,16 -> 2352,1,16,16:  0
```

This 910B2 timing is structural evidence, not a 310P speed prediction. Compare
only with a direct 310P measurement.

## Restrictions

- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Launch in the background. Send Luka the absolute `run.log` immediately.
- Use the runner-created fresh recognition cache. Do not reuse a candidate
  cache from an earlier experiment.
- Reuse the completed 310P native profile as a read-only comparison artifact.
- Do not enable operator-by-operator JIT compilation or fall back to CPU.
- Stop after this one bucket and report. Do not start the five-bucket or page
  pipeline run yet.

## Launch

Use the same activated 310P environment and native profile from the successful
vision comparison. Resolve the checkout instead of assuming its path:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*|*,6,*)
    printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES"
    exit 1
    ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export NATIVE_PROFILE="${NATIVE_PROFILE:-$REPO/tmp/12_unirec_0_1b_inference/vision_profile_47cde9c_20260812T143228/graph_suite/profile_suite_summary.json}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$NATIVE_PROFILE"

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_vision_prepacked_grouped_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_VISION_PREPACKED_GROUPED_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P VISION PREPACKED GROUPED STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Follow only that owned process:

```bash
tail -f "$RUN_LOG"
```

## Completion gate and report

After the worker exits:

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
test -f "$RUN_ROOT/comparison_summary.json"
grep '^UNIREC_VISION_DEPTHWISE lane=' "$RUN_ROOT/analysis.log"
```

Return the two `UNIREC_VISION_DEPTHWISE lane=...` lines. Also state the commit,
physical NPU, runtime versions, absolute run log, candidate profile JSON, and
fresh cache path. Then stop.

Acceptance gates:

- candidate parity is true;
- target repack count is zero;
- candidate time and total TransData time/count are reported;
- native and candidate use the same `960x64_b16` workload.

If the worker fails, report the failed phase and the first causal error from
`run.log`. Do not change tracked code on the work server.

Interpretation boundary: graph parity is necessary but is not full OCR-quality
validation. A 310P win still needs all five production buckets and a real-crop
first-32/first-128 production run before adoption.
