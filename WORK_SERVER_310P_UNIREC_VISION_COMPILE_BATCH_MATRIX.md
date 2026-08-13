# 310P UniRec full-vision eager versus compiled batch matrix

## Purpose

Run the exact 910B2 control on one Atlas 310P.  For the dominant `960x64`
full-vision canvas, measure B1, B4, and B16 through three lanes:

1. stock `model.forward_encoder` eager;
2. the exact masked bucket module in raw eager mode;
3. that same masked bucket module through TorchAir `cache_compile`.

This separates raw batching behavior from compiler benefit.  It is a synthetic
fixed-shape compute benchmark.  Pixel values do not affect operator shapes.

## Restrictions

- Pull the commit named by Luka or a descendant.
- Pull only.  Do not edit tracked files, branch, commit, or push.
- Use exactly one free physical 310P.  Never use physical NPU 5 or NPU 6.
- Use the native depthwise and native weight-format defaults for this first
  matched comparison.  Do not add the production grouped/internal rewrites.
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

Commit `8f0d4cf`, physical Ascend 910B2 NPU 4, CANN 9.0.0, FP16, native
depthwise, native weights, JIT disabled, two warmups and twenty repeats:

| Batch | Stock eager | Bucket raw eager | Compiled | Compiled / bucket eager | Compiled crops/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.703 ms | 25.587 ms | 7.846 ms | 3.261x | 127.46 |
| 4 | 19.922 ms | 26.193 ms | 9.744 ms | 2.688x | 410.51 |
| 16 | 18.862 ms | 24.930 ms | 13.797 ms | 1.807x | 1159.63 |

Compiled versus stock eager was 2.511x at B1, 2.045x at B4, and 1.367x at
B16.  Both correctness comparisons passed for all three batches.  The 910B2
first calls were 64.99 s, 77.24 s, and 20.42 s; those are setup observations,
not 310P expectations or measured inference.

## Return and stop

Return:

1. commit, physical NPU, CANN, torch-npu, and Python;
2. absolute run log and result JSON;
3. all three `UNIREC_VISION_COMPILE_BATCH` lines and the summary line;
4. for each batch, the three p50 times, compiled speedups, and crops/s;
5. both correctness comparisons, including max and mean absolute difference;
6. each compiled first-call wall time;
7. the 310P/910B2 ratio for each of the nine p50 cells.

Do not profile or test another optimization after this matrix.
