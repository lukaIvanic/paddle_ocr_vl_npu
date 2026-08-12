# 310P UniRec focal-depthwise optimization matrix

Pull the commit named by Luka and run this exact isolated matrix. Do not edit
tracked files, branch, commit, or push. This task tests exact replacements for
the dominant stage-2/3 UniRec vision TransData signatures; it does not run
layout, cross-KV, decode, or a full page benchmark.

The matched 310P profile identified these first-128 weighted signatures:

- `49,24,16,16 -> 1176,1,16,16`: 0.831 s;
- `25,24,16,16 -> 600,1,16,16`: 0.426 s;
- `49,48,16,16 -> 2352,1,16,16`: 0.371 s.

They are stage-2/3 5x5 and 7x7 focal depthwise-convolution weight repacks, not
attention-score reshapes. The committed matrix changes only those 22 filters:

1. `group16`: exact block-diagonal grouped filters, as used for layout;
2. `aligned_spatial`: exact 5x5 to 6x8 and 7x7 to 8x8 zero-padded filters;
3. `group16_internal`: group16 plus the proven TorchAir internal-weight pass.

910B2 structural controls, not 310P predictions:

```text
native:             14.459757 ms, TransData 4.456 ms/305, target 1.588 ms
group16:            15.659852 ms, TransData 4.355 ms/305, target 0.287 ms
aligned_spatial:    15.551551 ms, TransData 5.388 ms/305, target 2.329 ms
group16_internal:   13.647655 ms, TransData 2.781 ms/279, target 0.198 ms
```

All three rewrites passed the same-process graph-output tolerance on 910B2.
Select the winner only from 310P results.

## Restrictions

- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Run in the background and send Luka the absolute `run.log` immediately.
- Use fresh variant cache directories. Reuse the completed native profile only
  as the comparison reference; do not overwrite its cache or artifacts.
- Do not enable per-operator JIT compilation or fall back to CPU.
- The runner profiles only production bucket `960x64_b16`, which has the
  largest weighted first-128 gap. It intentionally does not compile the other
  four production buckets.
- Stop after returning the four analyzer lines. Do not promote a lane into the
  production page runner yet.

## Launch

Use the same activated 310P shell and model that produced the completed vision
profile. Resolve the checkout instead of assuming its absolute path:

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
  bash "$REPO/12_unirec_0_1b_inference/run_vision_depthwise_matrix_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_VISION_DEPTHWISE_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P VISION DEPTHWISE STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

The worker runs the three variants sequentially on the one selected NPU. Check
progress without starting another job:

```bash
tail -f "$RUN_LOG"
```

## Completion gate and report

After the worker exits:

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
test -f "$RUN_ROOT/matrix_summary.json"
grep '^UNIREC_VISION_DEPTHWISE lane=' "$RUN_ROOT/analysis.log"
```

Return exactly the four `UNIREC_VISION_DEPTHWISE lane=...` lines, then stop.
If the worker fails, return one short line containing the failed phase and the
first causal error from `run.log`.

Interpretation boundary: the graph parity check is necessary but not a full
OCR-quality test. A 310P winner still needs all five buckets, real-crop parity,
and a production first-32/first-128 run before adoption.
