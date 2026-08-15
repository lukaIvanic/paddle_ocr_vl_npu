# 310P UniRec all-focal prepacking

Pull the commit named by Luka and run this one focused candidate. Do not edit
tracked files, create a branch, commit, or push. It profiles only production
vision bucket `960x64_b16`.

The candidate combines two compatible optimizations:

1. `constant_grouped_all` pre-packs all 45 focal-depthwise filters once into
   grouped `FRACTAL_Z:<groups>` storage. This includes all 3x3, 5x5, and 7x7
   filters in stages 0–3.
2. `torchair_internal` preformats the remaining ordinary Conv/Linear weights.

Do not rerun native, an obsolete block-expanded lane, or the earlier 22-filter
`constant_grouped` lane. Use their completed profiles as controls.

## Evidence and target

Previous 310P results for the same bucket:

```text
native:             about 73.97 ms
old block-expanded: 55.48 ms
constant_grouped:   58.8 ms; TransData 12 ms / 294 calls
```

The new candidate on physical 910B2 NPU 7, commit `bbd9c25`:

```text
steady mean:        11.096903 ms (3 warmups, 50 repeats)
native:             14.459757 ms
parity:             true at atol=5e-2, rtol=5e-2
max_abs / mean_abs: 0.03955078125 / 0.00098514557
rewritten filters:  45
TransData:          1.4728 ms / 189 calls
inventory:          183 activation-layout + 6 ND-to-NZ
grouped repacks:    0
base-weight-to-FZ:  0
```

The 910B2 timing is not a 310P prediction. The structural target is zero focal
weight repacks and zero remaining base-weight-to-FZ conversions.

## Restrictions

- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Launch in the background and send Luka the absolute log path immediately.
- Use a fresh candidate cache. Reuse the completed native profile read-only.
- Do not enable operator-by-operator JIT compilation or fall back to CPU.
- Stop after this single bucket. Do not start all five buckets or page E2E.

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
export VISION_DEPTHWISE_REWRITE=constant_grouped_all
export VISION_WEIGHT_FORMAT=torchair_internal

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
310P VISION ALL-FOCAL STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Follow only this process:

```bash
tail -f "$RUN_LOG"
```

## Completion and report

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
test -f "$RUN_ROOT/comparison_summary.json"
grep '^UNIREC_VISION_' "$RUN_ROOT/analysis.log"
```

Return the `UNIREC_VISION_DEPTHWISE` and
`UNIREC_VISION_TRANSDATA_INVENTORY` lines. Also state the commit, physical NPU,
runtime versions, absolute run log, candidate profile JSON, and cache path.

Acceptance gates:

- parity is true;
- `rewritten=45`;
- `grouped_focal_weight_repack` is absent;
- `base_weight_to_fz` is absent;
- candidate timing and full TransData time/count are present.

If a gate fails, report the first causal error or the exact failed metric. Do
not change tracked code on the work server.
