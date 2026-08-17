# 310P UniRec K10/L4 fallback-covering representative-128 run

## Objective

Run one production-faithful W1/T8 trace on the fixed representative 128 pages.
Use the new ten-bucket, four-page-lookahead vision plan. It compiles every crop
shape seen in this set, including the 62 crops that previously used eager
fallback. Do not run a native or old-bucket baseline again. Compare against the
last completed 310P W1/T8 K10/L1 result already on the server.

The code must be this commit or a descendant. Do not edit tracked files on the
work server.

## Important work-server constraints

- The work server has four 310P devices. It does not have `npu-setup` or eight
  NPUs. Select one free device from 0-3 using the server's normal method.
- Use the same validated `python_nosym` executable as the last UniRec run.
  Do not run `readlink -f` on `PYTHON_BIN`; that resolves the venv launcher to
  `/usr/local/.../python3.12` and breaks the environment.
- Do not use `nproc`. Reuse the same known-good 64-CPU `taskset` mask from the
  last W1/T8 run. Confirm the worker reports at least 64 CPUs in
  `cpu_affinity_count` and exactly eight recognition preprocessing threads.
- Run in the background. Print the absolute run log path so Luka can use
  `tail -f`.
- A cold setup may build ten new vision graphs. The measured prefill starts
  only after every graph has been called once. Compilation time is setup, not
  measured prefill time.

## Inputs

Resolve the checkout without hard-coding it:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
git show -s --oneline HEAD
```

Reuse the validated values from the last successful UniRec run:

```bash
export PYTHON_BIN=/absolute/path/to/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export COMPILE_CACHE=/absolute/path/to/the/warmed/production/cache/parent
export LAYOUT_CACHE_ROOT=/absolute/path/to/the/warmed/optimized/layout/cache
export ASCEND_RT_VISIBLE_DEVICES=<one-free-device-from-0-to-3>
export TASKSET_CPUS=<the-same-known-good-64-cpu-mask>
```

Preserve the venv path this way:

```bash
PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
test -x "$PYTHON_BIN"
```

Materialize the fixed page set:

```bash
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_rep128_k10_l4_all_w1t8_${STAMP}"
INPUT="$RUN_ROOT/representative_128_v1_images"
mkdir -p "$RUN_ROOT/output"
"$PYTHON_BIN" 12_unirec_0_1b_inference/materialize_page_subset.py \
  --manifest 12_unirec_0_1b_inference/references/unirec_representative_128_v1.json \
  --images-dir "$IMAGES_DIR" \
  --output-dir "$INPUT"
```

## Single candidate run

Write the exact command into `command.sh`, then run the same command under the
known-good 64-CPU taskset. Use one process worker, layout B2/T16, recognition
T8, four-page vision lookahead, and the fallback-covering preset.

```bash
COMMAND=(
  "$PYTHON_BIN" 12_unirec_0_1b_inference/run_two_phase_batched_unirec.py
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --layout-execution torchair
  --layout-dtype float16
  --layout-reading-order-dtype float32
  --layout-weight-format torchair_internal
  --layout-depthwise-rewrite constant_grouped
  --layout-preformat-frozen-bn-buffers
  --layout-threshold 0.5
  --layout-cache-dir "$LAYOUT_CACHE_ROOT"
  --input "$INPUT"
  --output-dir "$RUN_ROOT/output"
  --device npu:0
  --dtype float16
  --offset 0
  --limit 128
  --workers 1
  --warmup-pages 8
  --layout-batch-size 2
  --layout-cpu-threads 16
  --vision-page-lookahead 4
  --vision-bucket-preset 310p_k10_l4_all
  --vision-focal-depthwise-rewrite constant_grouped_all
  --vision-weight-format torchair_internal
  --recognition-preprocess-threads 8
  --recognition-input-contract compact_uint8_hwc
  --cross-cache-length 1320
  --self-cache-length 2048
  --max-length 2048
  --decode-batch-size 128
  --compile-cache-dir "$COMPILE_CACHE"
  --stop-after-prefill
  --prefill-trace
  --progress-every-pages 16
  --progress-heartbeat-s 15
)
printf '%q ' taskset -c "$TASKSET_CPUS" "${COMMAND[@]}" >"$RUN_ROOT/command.sh"
printf '\n' >>"$RUN_ROOT/command.sh"
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c "$TASKSET_CPUS" "${COMMAND[@]}" \
  >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
  "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
```

Do not call the run failed while setup heartbeats continue and the worker is
alive. Inspect new compiler lines and cache inventory if setup is long. The
expected hot phase is short once setup completes.

## Required validation and report

The run must exit 0. Confirm from `run_summary.json`:

- `workers=1`
- `recognition_preprocess_threads=8`
- `layout_batch_size=2`
- `layout_cpu_threads=16`
- `vision_page_lookahead=4`
- `vision_bucket_preset=310p_k10_l4_all`
- optimized layout and vision weight/rewrite settings match the command
- `rejected_crop_count=0`
- all ten configured graph keys appear
- no `vision_fallback_call` event appears in `prefill_iterations.jsonl`
- crop and token counts are reported; small cross-chip crop drift is not a hard
  failure because layout inference can differ slightly

Report these exact fields:

1. Setup wall, measured prefill wall, and pages/s.
2. Layout wall and layout model-forward sum.
3. Vision graph sum, vision input-device sum, full vision section wall, graph
   call count, slot efficiency, and pixel efficiency.
4. Fallback count and fallback graph time. Expected: zero and zero.
5. Crop-preprocess service sum and active-window union.
6. Text-pack wall/device time, cross-KV D2H, shared-pack, and IPC delivery.
7. The ten-bucket call histogram and all p50/p95/p99 values for vision graph
   time.
8. Peak HBM and the absolute paths to the log and four output artifacts.
9. Direct comparison with the last completed 310P W1/T8 K10/L1 run. Focus on
   vision section, measured wall, calls, fallbacks, slot efficiency, and pixel
   efficiency. Do not rerun that baseline.

## 910B reference from the same candidate

Committed artifacts:

- `12_unirec_0_1b_inference/references/unirec_910b_k10_l4_all_227b24e/run_summary.json`
- `12_unirec_0_1b_inference/references/unirec_910b_k10_l4_all_227b24e/prefill_distributions.json`

Headline values from physical 910B2 NPU 7:

- measured prefill: 31.247148 s, 4.096374 pages/s
- setup: 37.420321 s, with zero new compiler starts on cache reuse
- crops: 2,485; rejected: 0
- layout wall: 3.640192 s; model forward: 1.208284 s
- full vision section: 8.840288 s
- vision graph: 7.102178 s across 1,001 calls
- vision input-device: 1.614774 s
- eager fallback calls/time: 0 / 0
- text-pack wall: 3.475083 s
- cross-KV D2H: 1.485469 s

Compared with the prior 910B K10/L1 trace, full vision improved
10.229530 -> 8.840288 s (13.6%) and all 62 eager fallbacks disappeared. Whole
prefill wall did not improve in this one repeat because CPU crop service and
IPC/shared-pack/D2H were slower. The 310P result is the adoption gate because
the bucket optimizer used the measured 310P latency curve.
