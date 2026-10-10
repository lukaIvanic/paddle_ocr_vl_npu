# 310P: model-default pixels full1651 for end-to-end pg/s

Updated 2026-10-10. **Single active handoff.** Run warmup64 and then full1651 of
the model-default pixel setup on the 310P, report pg/s, then **stop and wait for
Luka's approval**. Do not start the evaluation on your own.

## Why

The 310P production setup reached **0.38461 pg/s** (1,651 pages) with crop
limits min 25,088 / max 602,112 pixels and a 401,408 cap for `text` crops. On
the 910B, the checkpoint's own limits (min 50,176 / max 1,605,632, no text cap)
with KV8192 scored 95.6154 overall against 95.6343, and raised raw vision
tokens 1.45× (15.69M → 22.76M). This run measures what those limits cost on
the 310P. The estimate from measured 310P bucket latencies is about 0.22 pg/s,
so the full run should take about 2 hours.

## The setup

Identical to the 310P production full run, except:

- `--processor-min-pixels 50176 --processor-max-pixels 1605632`, no
  `--processor-text-max-pixels`;
- vision buckets `384,512,768,896,1024,1280,1536,1792,1920,2048,2560,3072,4096,5120,6144,7168,8192`;
- text-prefill buckets `128,256,384,512,576,832,1024,1536,2048,2176`;
- `--local-compiled-cache-length 8192`.

Everything else stays: approximate vision precision (innerPrecise 4), live
PP-DocLayoutV3, B32, FP16, C1+C2 host settings (CPU grids, pinned nonblocking
transfers), text pack target 384, window text prefill, vision pack target 768
and lookahead 32. The full run uses `--no-local-prefill-metrics`, like the
production full run. The warmup keeps metrics on, like the production warmup.

The committed `model_defaults` phase of `run_prefill_bucket_validation.py`
builds both commands from the same 310P candidate command the production run
used, then replaces only the options above.

**KV8192 on the 310P is new.** The 310P IncreFA tile boundary
(`INCREFA_310P_BOUNDARY.md`) falls every 1,408 effective tokens. KV4096 only
reaches 1408 and 2816; KV8192 adds 4224, 5632 and 7040. `pse_sentinel_310p`
derives the boundaries from head geometry, so it should cover them, but it has
never run past 4096 on the 310P. On the 910B, 25 requests ran to the full 8,192
and 42 exceeded 4,096, so the full run crosses every new boundary.

## Rules

- This checkout is pull-only. Do not edit tracked files, install packages,
  commit, push, create branches, or discard changes. If a change is needed,
  stop and report the exact command, error, log tail and minimal proposed fix.
  Do not substitute another method.
- Use the same device and environment as the production full run.
  `run_lane(..., require_idle_card=False)`; no manual npu-smi steps.
- Do not create a Markdown report file. Report in chat.

## Source

```bash
git status --short  # tracked source must be clean; never discard changes
git fetch origin claude/mineru-model-default-pixels-310p
git checkout --detach FETCH_HEAD
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git rev-parse HEAD
```

## Discover the environment

Locate the 310P production full run: the candidate full1651 run that gave
0.38461 pg/s, under `tmp/11_mineru_2_5_pro_inference/`. It was launched with a
generated script derived from the `prefill_buckets_candidate256_310p_*` run, so
look there first. Read its `command.txt`, receipt
and `output/run_summary_shard_00.json`. Take the interpreter, model, layout,
dataset, images, visible device, CANN activation and the three cache
directories (`--local-torchair-cache-dir`, `--local-vision-torchair-cache-dir`,
`--local-text-torchair-cache-dir`) from that command, so the 384–3072 vision
graphs, the text graphs and the KV4096 decode graph are already warm. The
coordinator copies them into the new run root; the originals are not modified.
Record where each value came from. Do not guess or download replacements.

Source the exact successful CANN activation **before** `set -u`.

```bash
export PYTHON_BIN=/absolute/path/from/the/production/full/run
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export CACHE_DECODE=/absolute/path/used/by/the/production/full/run/decode
export CACHE_VISION=/absolute/path/used/by/the/production/full/run/vision
export CACHE_TEXT=/absolute/path/used/by/the/production/full/run/text
export ASCEND_RT_VISIBLE_DEVICES=SAME_AS_PRODUCTION_FULL_RUN
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/model_default_pixels_310p_$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
mkdir -p "$RUN_ROOT"
```

## Run once

The `model_defaults` phase runs:

1. `model_defaults_warmup64`: first 64 pages with `--local-warm-all-prefill-buckets`,
   compiling every vision and text bucket and the KV8192 decode graph. Not
   performance evidence.
2. `model_defaults_full1651`: all 1,651 pages, same settings, without the warmup
   flag. It audits that no graph artifact changes during the run, then writes
   `model_defaults_checks.json` and prints `MODEL_DEFAULTS_CHECKS`.

Each child has a 14,400-second deadline.

```bash
nohup setsid bash -c '
  set +e
  "$PYTHON_BIN" -u 11_mineru_2_5_pro_inference/run_prefill_bucket_validation.py \
    --chip 310p --phase model_defaults --root "$RUN_ROOT" \
    --model "$MODEL_DIR" --layout-model "$LAYOUT_MODEL" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --cache-decode "$CACHE_DECODE" --cache-vision "$CACHE_VISION" \
    --cache-text "$CACHE_TEXT" > "$RUN_ROOT/coordinator.log" 2>&1
  status=$?
  printf "%s\n" "$status" > "$RUN_ROOT/exit_code.txt"
  exit "$status"
' </dev/null > "$RUN_ROOT/launcher.log" 2>&1 &
echo "$!" > "$RUN_ROOT/pid.txt"
```

Poll every 1–2 minutes. Completion requires root `exit_code.txt` = 0, both
child receipts exit zero, warmup 64/0/0 and full 1651/0/0
completed/failed/skipped, and `model_defaults_full1651/cache_audit.json`
reporting `unchanged: true`.

**Watch for a decode stall.** If the run log shows no new completed pages for
15 minutes while the process is alive, that may be the tile-boundary bug. Do
not kill the process. Report to Luka with the elapsed time, last completed
page count and the log tail, and wait.

## Report to Luka in chat, then stop

Label all values **310P**. Give the run-root basename, commit, interpreter,
visible device and the production run each environment value came from.

1. **Throughput:** pages, pipeline wall s and pg/s, against the production full
   run's 0.38461 pg/s (4,292.6 s). Then prefill s, decode s, CPU preparation
   work/wait s, layout s and sampled-token D2H wait s, next to the production
   full run's values from its summary. These regions overlap; do not add them.
2. **Routes:** vision and text-prefill route counts from the full run's summary,
   next to the 910B model-default run's counts:
   - vision: 384 493, 512 192, 768 3102, packed_768 11252, 896 956, 1024 712,
     1280 931, 1536 572, 1792 422, 1920 219, 2048 120, 2560 357, 3072 237,
     4096 299, 5120 228, 6144 146, 7168 135, 8192 419;
   - text: 128 241, 256 293, 384 10990, 512 885, 576 225, 832 539, 1024 209,
     1536 383, 2048 504, 2176 62;
   - raw vision tokens 22,763,756.
3. **Checks** from `MODEL_DEFAULTS_CHECKS`: vision eager overflow and text-prefill
   overflow (must be 0), stop counts, requests over 4,096, max total length and
   `length_stops_below_8192` (must be empty). The 910B had eos 32,026 /
   length 25, 42 over 4,096 and max 8,192.
4. **Warmup:** warmup64 child wall s and setup s; per new bucket (vision
   4096–8192, text 1536/2048/2176) `first_call_s` and `cache_was_warm`; decode
   wrapper and first-call s. First-call times include compile or cache load plus
   execution, so they are not pure compile time.
5. Warnings, failures and deviations.

Then **stop and wait for Luka's approval**. Do not start the evaluation.

## After approval only: evaluation (target under 5 minutes)

- Run `bash 11_mineru_2_5_pro_inference/run_serving_accuracy.sh` with
  `RUN_ROOT="$RUN_ROOT/model_defaults_full1651"`, `LIMIT=1651` and the evaluator
  exports used by the previous 310P full-run evaluation.
- Set `CDM_WORKERS` explicitly, to roughly the number of free cores (check
  `nproc` and load first). Only `CDM_WORKERS` works: the prepared metric
  config's `cdm_workers` takes precedence over `OMNIDOCBENCH_CDM_WORKERS`.
- Confirm the CDM pool's child-process count equals `CDM_WORKERS` and watch the
  rate. Aim for about 40 formula samples/s (2,352 formulas). If it is far below
  that (under 25/s) after the first minute, stop and report the rate, core
  count and load.
- Report overall, text, table TEDS and formula CDM against the 910B result for
  this setup (95.6154 / 96.2542 / 93.3449 / 97.2470) and the 310P production
  full run (95.6240 overall). Also report CDM samples/s, timeouts/errors and
  evaluation wall time.
