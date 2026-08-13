# 310P UniRec stock-eager vision B1 profile and 910B2 comparison

## Purpose

Profile exactly one warmed stock-eager UniRec vision forward at
`B1 x 3 x 64 x 960`, then compare it automatically with the committed matched
910B2 reference.

The run has three sequential timing phases in one process:

1. twenty clean NPU-event and synchronized-wall controls;
2. one Level1 pipe-profiled forward;
3. twenty clean controls after the profiler closes.

This measures both immediate profiler distortion and persistent post-profiler
distortion.  The clean controls are the latency authority.  The profile is for
matched kernel attribution.

## Restrictions

- Pull the commit named by Luka or a descendant.
- Pull only.  Do not edit tracked files, branch, commit, or push.
- Use exactly one free physical 310P.  Never use physical NPU 5 or NPU 6.
- Run stock eager `model.forward_encoder`.  Do not use TorchAir compilation,
  bucket masks, batching, production rewrites, or internal weight formats.
- Do not search for page/crop manifests, OpenOCR, prefill artifacts, or model
  exports.  Synthetic zeros preserve this fixed operator-shape contract.
- Do not add another profiler or rerun automatically.
- Run the committed comparison analyzer after the profile.  Do not manually
  reinterpret unmatched rows before it passes its contract checks.

## Run the 310P profile

Use one Bash shell:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_NPU_%s\n' "$ASCEND_RT_VISIBLE_DEVICES"; exit 1 ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1

PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/stock_eager_vision_b1_profile_310p_$COMMIT_SHORT"
RUN_LOG="$RUN_ROOT.run.log"

test -x "$PYTHON_BIN"
test -d "$MODEL"
test ! -e "$RUN_ROOT"
test ! -e "$RUN_LOG"

nohup "$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/profile_stock_eager_vision_b1.py" \
  --model-path "$MODEL" \
  --output-dir "$RUN_ROOT" \
  --device npu:0 \
  --width 960 --height 64 \
  --warmups 3 --control-repeats 20 \
  --profile-metric pipe --parser-topn 100 \
  >"$RUN_LOG" 2>&1 < /dev/null &

RUN_PID=$!
printf 'PID=%s\nLOG=%s\nRESULT=%s\n' \
  "$RUN_PID" "$RUN_LOG" "$RUN_ROOT/result.json"
tail --pid="$RUN_PID" -f "$RUN_LOG"
wait "$RUN_PID"
test -s "$RUN_ROOT/result.json"
test -s "$RUN_ROOT/comparison_reference.json"
grep -E '^UNIREC_STOCK_EAGER_VISION_B1_' "$RUN_LOG"
```

The profile export/parser can spend several seconds after the forward.  That is
setup and analysis time, not inference latency.

## Compare with the committed 910B2 reference

```bash
REFERENCE="$REPO/12_unirec_0_1b_inference/references/stock_eager_vision_b1_910b_c1beb8c.json"
ANALYSIS="$RUN_ROOT/stock_eager_vision_b1_chip_comparison.json"

test "$(sha256sum "$REFERENCE" | awk '{print $1}')" = \
  26c9895483eef65047e3b5b2ae76a43d195a014c999d357b231872f8ff00c7f9

"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/analyze_stock_eager_vision_profiles.py" \
  --npu310 "$RUN_ROOT/result.json" \
  --npu910 "$REFERENCE" \
  --output "$ANALYSIS" \
  --topn 20 \
  2>&1 | tee "$RUN_ROOT/comparison.log"

test -s "$ANALYSIS"
```

The analyzer compares clean control latency, profiler distortion, post-profile
latency, step-trace compute/free/preparing time, total kernels, cube
utilization, kernel types, exact shapes, MatMul signatures, and TransData
shape/format signatures.

## Matched 910B2 reference

Commit `c1beb8c`, physical Ascend 910B2 NPU 4, CANN 9.0.0, FP16, JIT disabled:

- clean control before: **19.798 ms** NPU-event p50;
- profiled forward: **28.792 ms**;
- clean control after: **19.485 ms**;
- profiled-event distortion: **1.466x** versus combined clean controls;
- persistent post-profile ratio: **0.984x**;
- step Stage / Computing / Free / Preparing:
  **28.792 / 10.139 / 18.653 / 0.848 ms**;
- kernel count / summed duration / weighted cube: **895 / 10.139 ms / 74.40%**;
- top kernel type: TransData, **240 calls / 4.614 ms**;
- all profiled and post-profile outputs were exact against the before control.

Do not compare the 310P profiled event directly with 19.798 ms.  Compare clean
control with clean control, and profiled/parsed fields with their matched
profiled/parsed fields.

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, Python, absolute log/result/analysis;
2. the `UNIREC_STOCK_EAGER_VISION_B1_PROFILE` line and every kernel-type line;
3. the complete analyzer stdout, including step, kernel, exact-shape, MatMul,
   and TransData gap lines;
4. a five-line conclusion: clean chip ratio, profiler overhead on both chips,
   compute versus free contribution, largest kernel-type gap, and largest exact
   signature gap.

Do not test a fix or start B16 after this comparison.
