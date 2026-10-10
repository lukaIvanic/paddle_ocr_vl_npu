# 310P: full MinerU pair with text-only cap and window prefill

Written 2026-10-10. **Single active handoff for this task.** Supersedes the
first-256 prefill-timing brief. Run one full 1,651-page pair, with approximate
vision precision and prefill metrics **on in both lanes**. This intentionally
overrides C5's normal metrics-off setting so the requested stage tables exist.

## Constraints and provenance

- The requested candidate is exactly: text-label cap 401408, vision buckets
  384/512/768/896/1024/1280/1536/1792/1920/2048/2560/3072, text buckets
  128/256/384/512/576/832/1024, text pack target 384, window prefill.
- Both lanes use current C1+C2 host production: CPU grids, pinned nonblocking
  main-thread copies, preparation/frontend workers 1/2, uint8 off. The global
  max/min pixels stay 602112/25088. Vision pack target/lookahead stay 768/32.
  Both use live PP-DocLayoutV3, B32, KV4096, FP16, and approximate vision
  innerPrecise=4. No decode KV-length, model, dataset, or precision changes.
- This server is pull-only. Do not edit tracked files, install packages,
  commit, push, create branches, or discard user changes. If a change is
  needed, stop and report the exact command, error, relevant log tail, and
  minimal proposed fix. Do not silently substitute another method.
- Use the previously successful device/environment. As in the first-256
  handoff, `run_lane(..., require_idle_card=False)`; do not add manual npu-smi
  steps. The runner's existing receipt mechanism is retained unchanged.
- The staging implementation reuses experiment 20's ready-KV lease admission.
  Prompt-sized staging and a batched first-token read are implementation choices
  for the requested memory/sync bounds. Same-stream window-owned allocation
  avoids another admission fence; experiment 20's reusable arenas keep theirs.
- Warm every graph before measuring. The new explicit all-bucket warmup flag
  reuses the existing real-page capture/replay helper. Warmup results are not
  performance evidence. Each lane has separate cache copies and output paths.

## Source

Code commit: `9da169cf`. Later documentation-only commits are allowed; the
second check enforces that. Read `CLAUDE.md`, `AGENTS.md`, and experiment 11's
README after pulling.

```bash
git status --short  # tracked source must be clean; never discard changes
git fetch origin codex/mineru-prefill-buckets
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 9da169cf HEAD
git diff --quiet 9da169cf HEAD -- '*.py'
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git rev-parse HEAD
```

## Discover the established environment

Take interpreter, CANN activation, model/layout/dataset paths, device, and warm
cache roots from the successful C1+C2+C5 combined-lane receipt under
`host_overlap_310p_*/combined/receipt/command.json`. If its local name differs,
find the receipt whose argv has CPU grids, pinned-nonblocking copies and
metrics off. If unavailable, use the last successful approximate full run's
`command.txt`, then apply the explicit common settings above. Record which
receipt supplied each path. Do not guess or download replacements.

Source the exact successful CANN activation **before** enabling `set -u`.
`CACHE_VISION` is the parent containing the matching `vision_innerprecise4_*`
directory; preserve that hierarchy. Confirm all directories exist.

```bash
export PYTHON_BIN=/absolute/path/to/the/successful/mineru/python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export CACHE_DECODE=/absolute/path/to/the/successful/decode/cache
export CACHE_TEXT=/absolute/path/to/the/successful/text/cache
export CACHE_VISION=/absolute/path/to/the/successful/vision/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=SAME_AS_SUCCESSFUL_RUN
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/prefill_buckets_310p_$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
mkdir -p "$RUN_ROOT"
```

## Run once

The committed coordinator copies caches, runs baseline warmup64 then baseline
full1651, then candidate warmup64 and candidate full1651. Both warmups replay all
configured prefill buckets after the real pages. Baseline **warmup only** uses
its existing vision buckets through 3072; 4224/5632 cannot occur under the
602112-pixel cap, so this authoring choice avoids compiling unused graphs. The
measured baseline retains its original full bucket list. Full runs omit the
warmup flag.
The coordinator checks complete page accounting, exact source commits, precision
scope, warmup coverage, and unchanged graph artifacts during each measured run.
No retries or alternate settings are automatic. Each child has a 14,400-second
deadline to allow compilation; timeout is a failure, not a truncated result.

```bash
nohup setsid bash -c '
  set +e
  "$PYTHON_BIN" -u 11_mineru_2_5_pro_inference/run_prefill_bucket_validation.py \
    --chip 310p --phase pair --root "$RUN_ROOT" \
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

Poll every 30–60 seconds. Completion requires `exit_code.txt` = 0, both measured
lane receipts exit zero, 1651 completed/0 failed/0 skipped in both summaries,
and both `cache_audit.json` files reporting `unchanged: true`. On any failure,
stop and report; do not rerun with modified settings. A graph artifact changing
during a measured run invalidates its throughput.

## Report to Luka in chat

Label every value **310P**; these are not 910B results. Do not create a Markdown
report file. Give the run-root basename, commit, interpreter, visible device,
source receipt for environment discovery, and warmup coverage first.

Then give this comparison using measured outputs only:

| Lane | Pages | Wall s | pg/s | Prefill wall s | Decode device s | CPU prepare work s | CPU prepare wait s | Layout s | Postprocess/write s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current C1+C2 production, metrics on | | | | | | | | | |
| Plus new prefill set, metrics on | | | | | | | | | |

Use `PREFILL_RESULT` rows and the full run summaries for the stage breakdown.
Report every prefill event stage from `prefill_metrics` separately, including
vision blocks, text prefill, KV redistribution and LM head. CPU work, prefill
wall time and device event regions can overlap; do not add them as a wall total.
Report absent stage fields as unavailable rather than zero.

Paste the coordinator's two per-route tables for **each** measured lane:
vision and text prefill. They include calls, real/physical tokens, device
seconds, median/mean ms excluding each route's first call, and real/physical
tok/s. Rates use total tokens divided by total device seconds; event regions
include host launch gaps. Raw samples are `vision_timing_shard_00.jsonl` and
`text_prefill_timing_shard_00.jsonl`. Warmup tables are excluded from the report.
Finally report the pg/s ratio, cache audits, warnings, failures or deviations.
