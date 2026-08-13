# 310P UniRec full eager-vision grouped-FZ model gate

## Purpose

Test the proven grouped-FZ weight representation inside one complete eager
UniRec vision forward at `B1 x 3 x 64 x 960`.

The harness loads the model once, then:

1. warms, times, and profiles native `model.forward_encoder`;
2. prepackages only the nine stage-2 `384x1x7x7`, groups-384 focal weights;
3. warms, times, and profiles the rewritten full encoder;
4. requires bit-exact full-encoder output;
5. requires native target counts `9 + 9 + 9` and grouped counts `0 + 0 + 9`
   for logical-to-FZ1, FZ1-to-FZ384, and physical Conv2D.

This is eager execution. It does not use TorchAir, bucket graphs, model
compilation, page parsing, or decode.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use the existing Python/compiler/Ninja/TorchNPU installation. Do not install
  or upgrade anything.
- Do not enable JIT or global internal formats. A warning that an output tensor
  is created in base format is expected in the grouped lane.
- Run the command once in the background. Do not automatically rerun a failed
  lane or expand to more focal weights.

## Relevant evidence

The isolated 310P convolution passed before this model gate:

```text
native mean:             about 0.98 ms
grouped packed mean:     about 0.15 ms
target repacks:          1 + 1 -> 0 + 0
physical Conv2D:         1 -> 1
isolated speedup:        about 6.5x
estimated nine-call gap: about 7.5 ms
```

Matched format-local 910B2 full-model reference, commit `d724e33`, physical
NPU 4, CANN 9.0.0:

```text
full output parity:                 bit-exact, max_abs=0, mean_abs=0
native/grouped target counts:       9+9+9 / 0+0+9
native/grouped target TransData:    1.02706 / 0 ms
native/grouped mature clean median: 18.77322 / 18.98103 ms
native/grouped kernel compute:       10.56758 / 9.35792 ms
native/grouped Free:                 17.06212 / 19.00568 ms
native/grouped profiled stage:       27.62950 / 28.36375 ms
```

The 910B result is nearly neutral because `1.21 ms` less kernel compute was
offset by `1.94 ms` more launch/free time. Do not transfer that conclusion to
310P: its measured isolated repack cost is much larger.

## Launch

Use one Bash shell in the work-server checkout:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git merge-base --is-ancestor d724e33 HEAD

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
MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/eager_grouped_fz_model_310p_${COMMIT_SHORT}_${STAMP}"
RUN_LOG="$RUN_ROOT.run.log"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test ! -e "$RUN_ROOT"
test ! -e "$RUN_LOG"

nohup "$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/profile_eager_grouped_fz_vision_b1.py" \
  --model-path "$MODEL" \
  --output-dir "$RUN_ROOT" \
  --device npu:0 \
  --width 960 --height 64 \
  --warmups 3 --control-repeats 20 \
  --profile-metric pipe --parser-topn 200 \
  >"$RUN_LOG" 2>&1 < /dev/null &

RUN_PID=$!
printf 'PID=%s\nRUN_LOG=%s\nRUN_ROOT=%s\n' \
  "$RUN_PID" "$RUN_LOG" "$RUN_ROOT"
```

Immediately tell Luka the absolute log path and:

```bash
tail -f "$RUN_LOG"
```

Follow only the owned process:

```bash
tail --pid="$RUN_PID" -f "$RUN_LOG"
wait "$RUN_PID"
test -s "$RUN_ROOT/result.json"
grep '^UNIREC_EAGER_GROUPED_FZ_VISION_' "$RUN_LOG"
```

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, absolute log and result paths;
2. all three `UNIREC_EAGER_GROUPED_FZ_VISION_*` lines;
3. native and grouped clean control-before and control-after distributions;
4. full parity fields, rewrite count, extension build time, and packed bytes;
5. all six target TransData/Conv2D counts and durations;
6. native/grouped Computing, Free, and Stage times;
7. native/grouped mature clean speedup and saved milliseconds.

If status is not `ok`, return the complete exception and final 100 log lines.
Do not test additional shapes, all 22 focal weights, compiled graphs, B16, or
page E2E. Stop after this model gate.
