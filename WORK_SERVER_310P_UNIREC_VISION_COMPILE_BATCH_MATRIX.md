# 310P UniRec compiled-stock vision follow-up

## Purpose

Run the exact 910B2 follow-up on one Atlas 310P.  For the dominant `960x64`
full-vision canvas, measure B1, B4, and B16 through two lanes:

1. stock `model.forward_encoder` eager;
2. that unmodified stock encoder through fixed-shape TorchAir `cache_compile`.

The previous matrix compiled our masked bucket implementation.  This follow-up
tests whether compiling the stock encoder directly is materially faster.  It
skips every bucket graph to minimize wall time.

## Restrictions

- Pull the commit named by Luka or a descendant.
- Pull only.  Do not edit tracked files, branch, commit, or push.
- Use exactly one free physical 310P.  Never use physical NPU 5 or NPU 6.
- Pass `--stock-only`.  Do not compile or rerun the bucket lanes.
- Do not use page manifests, crop manifests, OpenOCR, layout, profiling,
  prefill, decode, or artifact references.
- Keep NPU JIT compilation disabled.  The script sets this itself.
- First-call cache compile/load time is setup.  Compare the synchronized,
  warmed NPU-event medians.
- Run once, report, and stop.

## Run

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
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/vision_compile_batch_matrix_310p_$COMMIT_SHORT"
CACHE_ROOT="$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_compile_batch_matrix_native_310p_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -d "$MODEL"
test ! -e "$RUN_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT"

nohup "$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/vision_compile_batch_matrix.py" \
  --model-path "$MODEL" \
  --cache-dir "$CACHE_ROOT" \
  --output "$RUN_ROOT/result.json" \
  --device npu:0 \
  --width 960 --height 64 \
  --batch-sizes 1,4,16 \
  --stock-only \
  --warmups 2 --repeats 20 \
  >"$RUN_ROOT/run.log" 2>&1 < /dev/null &

RUN_PID=$!
printf 'PID=%s\nLOG=%s\nRESULT=%s\n' \
  "$RUN_PID" "$RUN_ROOT/run.log" "$RUN_ROOT/result.json"
tail --pid="$RUN_PID" -f "$RUN_ROOT/run.log"
wait "$RUN_PID"
test -s "$RUN_ROOT/result.json"
grep -E '^UNIREC_VISION_COMPILE_BATCH' "$RUN_ROOT/run.log"
```

The first use of each batch shape can compile or load a graph for tens of
seconds.  This is expected.  A completed B-row prints immediately before the
next shape starts.  Do not infer a stall while compiler output is advancing.

## Matched 910B2 reference

Commit `aa83b0e`, physical Ascend 910B2 NPU 4, CANN 9.0.0, FP16, JIT disabled,
two warmups and twenty repeats.  These are same-process medians from the full
four-lane control:

| Batch | Stock eager | Stock compiled | Speedup | Stock compiled crops/s |
|---:|---:|---:|---:|---:|
| 1 | 20.787 ms | 6.605 ms | 3.147x | 151.40 |
| 4 | 21.726 ms | 8.193 ms | 2.652x | 488.21 |
| 16 | 21.657 ms | 12.128 ms | 1.786x | 1319.24 |

Stock compiled versus stock eager passed at all batches.  Maximum absolute
differences were 0.00586, 0.02051, and 0.03809 for B1/B4/B16.  Stock-compiled
first calls were 32.95 s, 47.77 s, and 46.95 s; those are setup observations,
not 310P expectations or inference measurements.

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, and Python;
2. absolute run log and result JSON;
3. all three `UNIREC_VISION_COMPILE_BATCH` lines and the summary line;
4. for each batch, stock-eager and stock-compiled p50, speedup, and crops/s;
5. stock-compiled versus stock-eager max and mean absolute difference;
6. each stock-compiled first-call wall time;
7. the 310P/910B2 ratio for all six p50 cells;
8. stock-compiled versus the prior 310P masked-bucket compiled p50 at each B.

Do not profile or test another optimization after this matrix.
