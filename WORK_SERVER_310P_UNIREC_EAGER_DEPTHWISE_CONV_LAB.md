# 310P UniRec isolated eager 7x7 depthwise Conv2D lab

## Purpose

Run one eager `torch.nn.functional.conv2d` with the exact dominant UniRec
stage-2 logical contract:

```text
input:   [1,384,4,60] FP16
weight:  [384,1,7,7] FP16
groups:  384
padding: 3
```

The native lane must reproduce this sequence before the second lane starts:

```text
[384,1,7,7] NCHW
  -> [49,24,16,16] FRACTAL_Z:1
  -> [1176,1,16,16] FRACTAL_Z:384
  -> Conv2D
```

The second fresh process applies `torch_npu.npu_format_cast(weight, 4)` once
before timing. It tests whether eager Conv2D can consume that weight without
the repeated group repack and whether doing so preserves semantics.

This is not a compiled graph, model forward, bucket benchmark, or page run.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use exactly one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not enable JIT or TorchAir.
- Do not load the UniRec model or search for artifacts. The lab is synthetic
  and fully specifies the exact operator tensors.
- Run once in the background and return the existing logs. Do not rerun a lane
  automatically.
- A final `candidate_semantics_failed` status is an expected completed result,
  not a harness failure.

## 910B2 matched reference

Commit `9cd8bb5`, physical 910B2 NPU 4, CANN 9.0.0:

```text
native clean before/after:          0.199650 / 0.200890 ms
native profiled call:               0.588920 ms
NCHW -> FZ:1:                       1 call / 0.026600 ms
FZ:1 -> FZ:384 target repack:       1 call / 0.088500 ms
native Conv2D:                      1 call / 0.007560 ms

direct FZ:1 clean before/after:     0.116610 / 0.129680 ms
direct FZ:1 profiled call:          0.869600 ms
weight TransData calls:             0
candidate Conv2D:                   1 call / 0.008420 ms
candidate parity max_abs/mean_abs:  0.382568 / 0.0579834
candidate parity:                   failed
```

The direct cast removes the repack but is numerically invalid. The grouped
Conv2D interprets the FZ:1-packed bytes under a grouped FZ:384 contract. Do not
promote it as an optimization.

## Launch

Use one shell from the work-server checkout:

```bash
set -e
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git merge-base --is-ancestor 9cd8bb5 HEAD

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*|*,6,*)
    printf 'REJECTED_PHYSICAL_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    exit 1
    ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1

PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/eager_depthwise_conv_lab_310p_${COMMIT_SHORT}_${STAMP}"
RUN_LOG="$RUN_ROOT.run.log"
test -x "$PYTHON_BIN"
test ! -e "$RUN_ROOT"
test ! -e "$RUN_LOG"

nohup "$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/eager_depthwise_conv_lab.py" \
  --output-dir "$RUN_ROOT" \
  --lane matrix \
  --device npu:0 \
  --warmups 10 \
  --control-repeats 50 \
  --parser-topn 200 \
  >"$RUN_LOG" 2>&1 < /dev/null &

RUN_PID=$!
printf 'PID=%s\nRUN_LOG=%s\nRUN_ROOT=%s\n' \
  "$RUN_PID" "$RUN_LOG" "$RUN_ROOT"
```

Immediately tell Luka the absolute `RUN_LOG` and the exact `tail -f` command.
Follow only this owned process.

```bash
tail --pid="$RUN_PID" -f "$RUN_LOG"
wait "$RUN_PID"
test -s "$RUN_ROOT/matrix_summary.json"
grep '^UNIREC_EAGER_DEPTHWISE_CONV_' "$RUN_LOG"
```

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, absolute log and summary paths;
2. all three `UNIREC_EAGER_DEPTHWISE_CONV_*` lines;
3. native clean before/profiled/after timings;
4. the three native per-kernel counts and times;
5. direct-FZ timings, weight TransData count, Conv2D time, and parity values;
6. the isolated native target-repack ratio versus the 910B2 0.088500 ms value.

Do not test another format, grouped rewrite, compilation, B16, or page E2E.
