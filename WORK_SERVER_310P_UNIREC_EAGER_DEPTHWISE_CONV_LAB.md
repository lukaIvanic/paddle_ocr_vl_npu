# 310P UniRec eager grouped-FZ depthwise Conv2D lab

## Purpose

Run one eager `torch.nn.functional.conv2d` with the dominant UniRec stage-2
contract:

```text
input:   [1,384,4,60] FP16
weight:  [384,1,7,7] FP16
groups:  384
padding: 3
```

The matrix runs three fresh-process lanes:

1. `native`: logical NCHW weight. It must reproduce both weight repacks.
2. `fractal_z_1`: the known fast but numerically invalid direct format-4 cast.
3. `grouped_fz_384`: host-pack the weight into CANN's exact grouped physical
   shape `[1176,1,16,16]`, copy those bytes once, and attach a logical
   `[384,1,7,7]` NCHW / primary-FRACTAL_Z descriptor before eager Conv2D.

The grouped lane deliberately advertises primary FRACTAL_Z format `4`, not the
GE-style encoded integer `98308`. TorchNPU 2.10 eager rejects `98308` as an
unknown format before Conv2D. The physical storage shape still carries the
exact grouped layout. This is a small descriptor bridge, not a custom CANN
operator or a compiled graph.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use exactly one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not enable JIT or TorchAir.
- Do not load the UniRec model or search for model artifacts. The lab is
  synthetic and fully specifies its tensors.
- Run once in the background and return the existing logs. Do not rerun a lane
  automatically.
- Use the existing Python environment, compiler, Ninja, and TorchNPU headers.
  Do not install or upgrade packages. If one is missing, report the exact
  missing path/tool and stop.

## 910B2 reference

Commit `ec97e8e`, physical 910B2 NPU 4, CANN 9.0.0, 50 clean samples:

```text
native median / mean / p90:         0.187940 / 0.188076 / 0.193170 ms
native NCHW -> FZ:1:                1 call / 0.028520 ms
native FZ:1 -> FZ:384:              1 call / 0.085720 ms
native Conv2D:                       1 call / 0.008380 ms

direct FZ:1 median:                 0.115060 ms
direct FZ:1 parity:                 invalid (known control)

grouped packed median / mean / p90: 0.124220 / 0.125372 / 0.143200 ms
grouped weight TransData calls:      0
grouped Conv2D:                      1 call / 0.008780 ms
grouped descriptor base/storage:     [384,1,7,7] / [1176,1,16,16]
grouped descriptor format/bytes:     4 / 602112
grouped parity:                      exact, max_abs=0, mean_abs=0
native/grouped median speedup:       1.513x
```

The 310P result is independent evidence. Do not infer it from the 910B result.

## Launch

Use one shell from the work-server checkout:

```bash
set -e
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git merge-base --is-ancestor ec97e8e HEAD

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
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/eager_grouped_fz_310p_${COMMIT_SHORT}_${STAMP}"
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

Immediately tell Luka the absolute `RUN_LOG` and this exact command:

```bash
tail -f "$RUN_LOG"
```

Follow only the owned process:

```bash
tail --pid="$RUN_PID" -f "$RUN_LOG"
wait "$RUN_PID"
test -s "$RUN_ROOT/matrix_summary.json"
grep '^UNIREC_EAGER_DEPTHWISE_CONV_' "$RUN_LOG"
```

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, absolute log and summary paths;
2. all lane and matrix `UNIREC_EAGER_DEPTHWISE_CONV_*` lines;
3. native and grouped clean min/median/mean/p90/max;
4. native and grouped logical-to-FZ1, FZ1-to-grouped, and Conv2D counts/times;
5. grouped descriptor base shape, storage shape, format, and physical bytes;
6. native-versus-grouped exact/allclose/max-absolute/mean-absolute results;
7. native/grouped median speedup and the isolated 310P target-repack ratio
   versus the 910B2 `0.085720 ms` reference.

If the grouped lane fails, return the complete exception and the last 100 log
lines. Do not switch formats, compile the model, or run page E2E. Stop after
this matrix.
