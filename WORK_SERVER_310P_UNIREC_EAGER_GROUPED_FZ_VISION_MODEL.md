# 310P UniRec all-22 grouped-FZ eager vision model gate

## Purpose

Test the exact grouped-FZ weight representation for every production-relevant
stage-2/3 5x5 and 7x7 focal convolution inside one complete eager UniRec vision
forward at `B1 x 3 x 64 x 960`.

The harness loads the model once, then:

1. warms, times, and profiles native `model.forward_encoder`;
2. prepackages these four immutable-weight families:
   - 9 x `[384,1,5,5]` -> `[600,1,16,16]`;
   - 9 x `[384,1,7,7]` -> `[1176,1,16,16]`;
   - 2 x `[768,1,5,5]` -> `[1200,1,16,16]`;
   - 2 x `[768,1,7,7]` -> `[2352,1,16,16]`;
3. warms, times, and profiles the rewritten full encoder;
4. requires bit-exact full-encoder output;
5. checks every shape family independently;
6. requires aggregate native counts `22 + 22 + 22` and grouped counts
   `0 + 0 + 22` for logical-to-FZ1, FZ1-to-grouped-FZ, and physical Conv2D.

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
  lane or expand to stages 0-1.

## Prior 310P evidence

The nine-weight stage-2 7x7 model gate passed before this expansion:

```text
full output parity:             passed
native/grouped wall:            about 30 / 25 ms
target repacks:                 9 + 9 -> 0 + 0
physical target Conv2D:         9 -> 9
wall saving:                    about 5 ms
isolated nine-call repack cost: about 7 ms
```

This run tests whether the other 13 weights add further wall-time savings.

## Matched 910B2 reference

Commit `c6af4f5`, physical NPU 4, CANN 9.0.0:

```text
status/parity:                    ok / bit-exact, max_abs=0, mean_abs=0
rewrite inventory:                9 + 9 + 2 + 2 = 22
packed physical bytes:            11,821,056
native/grouped target counts:     22+22+22 / 0+0+22
native/grouped target TransData:  2.210040 / 0 ms
native/grouped mature median:      18.981510 / 20.171820 ms
native/grouped kernel compute:     10.707200 / 7.813240 ms
native/grouped Free:               15.720280 / 21.202220 ms
native/grouped profiled Stage:     26.427750 / 29.015500 ms
```

On 910B2, the rewrite saved `2.894 ms` of kernel compute but added `5.482 ms`
of Free/launch time, so clean wall time regressed by `1.190 ms`. Do not carry
that chip-specific wall result to 310P. The same nine-weight subset already
improved 310P wall time by about 5 ms.

## Launch

Use one Bash shell in the work-server checkout:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git merge-base --is-ancestor c6af4f5 HEAD

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
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/eager_grouped_fz_22_model_310p_${COMMIT_SHORT}_${STAMP}"
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
2. every `UNIREC_EAGER_GROUPED_FZ_VISION_*` line, including four signature
   lines;
3. native and grouped clean control-before and control-after distributions;
4. full parity fields, rewrite inventory, extension build time, and packed
   bytes;
5. aggregate and per-signature TransData/Conv2D counts and durations;
6. native/grouped Computing, Free, and Stage times;
7. native/grouped mature clean speedup and saved milliseconds;
8. the incremental gain versus the previous nine-weight result of about
   `30 -> 25 ms`.

If status is not `ok`, return the complete exception and final 100 log lines.
Do not test stages 0-1, compiled graphs, B16, or page E2E. Stop after this gate.
