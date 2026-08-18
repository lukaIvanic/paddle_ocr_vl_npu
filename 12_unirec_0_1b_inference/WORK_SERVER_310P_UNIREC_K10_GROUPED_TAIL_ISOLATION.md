# 310P K10 grouped-FZ tail isolation

## Goal

Test one hypothesis with one new graph. The failed `960x448 B1` graph has a
stage-3 spatial area of `14x30 = 420`, which Conv2D physically rounds to a
16-aligned area of 432. The aligned `960x512 B1` graph has `16x30 = 480` and
passes.

Keep `torchair_internal` regular Conv/Linear weights. Change only the 45 focal
depthwise convolutions from `constant_grouped_all` to `constant`. This retains
the custom constant-Conv2D lowering but removes the prepacked grouped-FZ weight
descriptor. Run only `960x448 B1`.

If this lane matches eager, the bug is in grouped-FZ tail/descriptor handling.
If it still fails, the next single lane is native focal Conv2D.

## Run

Use the exact environment and path exports from
`WORK_SERVER_310P_UNIREC_K10_HEIGHT_AB.md`. Pull commit `HEAD` containing this
brief, then launch:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
: "${PYTHON_BIN:?validated python_nosym}"
: "${MODEL:?validated UniRec model}"
: "${OPENOCR_ROOT:?validated OpenOCR checkout}"
: "${COMPILE_CACHE:?same cache parent as the height A/B}"
: "${FACTOR_ROOT:?completed factorization RUN_ROOT}"
: "${ASCEND_RT_VISIBLE_DEVICES:?one free 310P device 0-3}"

ARTIFACT="$FACTOR_ROOT/prefill_production_buckets_optimized_weights"
RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_k10_grouped_tail_isolation_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RUN_ROOT"

(
  set +e
  started="$(date +%s)"
  "$PYTHON_BIN" 12_unirec_0_1b_inference/probe_vision_k10_height_ab.py \
    --openocr-root "$OPENOCR_ROOT" \
    --model-path "$MODEL" \
    --page-manifest "$ARTIFACT/pages.jsonl" \
    --crop-manifest "$ARTIFACT/crops.jsonl" \
    --cache-dir "$COMPILE_CACHE" \
    --output "$RUN_ROOT/report.json" \
    --focal-depthwise-rewrite constant \
    --current-height-only \
    --warmup-replays 1 --timing-repeats 3
  status=$?
  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$(($(date +%s) - started))" >"$RUN_ROOT/process_wall_s.txt"
) >"$RUN_ROOT/run.log" 2>&1 &
printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"

echo "RUN_ROOT=$RUN_ROOT"
echo "tail -f '$RUN_ROOT/run.log'"
```

Immediately monitor the one possible compilation in a second shell:

```bash
export RUN_ROOT COMPILE_CACHE
export FOCAL_REWRITE=constant
bash 12_unirec_0_1b_inference/monitor_vision_k10_height_ab.sh
```

At most one executable vision graph can compile: `960x448_b1`, slot 6.

## Report

Return `RUN_ROOT`, commit, device, process wall, OM inventory change, all phase
and graph-diagnostic lines, the monitor log, and both crops' `448_vs_eager`
max abs, mean abs, RMSE, cosine, and p50 time. Do not run another lane yet.
