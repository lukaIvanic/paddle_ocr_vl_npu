> **Completed.** The active 310P handoff is
> [WORK_SERVER_310P_MODEL_DEFAULT_PIXELS_FULL1651.md](WORK_SERVER_310P_MODEL_DEFAULT_PIXELS_FULL1651.md).

# 310P: candidate first 256 pages with text-only cap and window prefill

Updated 2026-10-10. Run only candidate
warmup64 followed by candidate256 (offset 0), with approximate vision precision
and prefill metrics on. No baseline lane, no full1651, no accuracy evaluation.
Compare with the existing `prefill_timing_first256_310p_20261010T092230Z` run:
vision **321.66 s**, text prefill **75.88 s**. These are user-supplied reference
measurements; do not rerun or overwrite that run.

## Constraints and provenance

- The requested candidate is exactly: text-label cap 401408, vision buckets
  384/512/768/896/1024/1280/1536/1792/1920/2048/2560/3072, text buckets
  128/256/384/512/576/832/1024, text pack target 384, window prefill.
- The candidate uses current C1+C2 host production: CPU grids, pinned nonblocking
  main-thread copies, preparation/frontend workers 1/2, uint8 off. The global
  max/min pixels stay 602112/25088. Vision pack target/lookahead stay 768/32.
  Use live PP-DocLayoutV3, B32, KV4096, FP16, and approximate vision
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
  performance evidence. Copy seed caches into this run root; warmup and measured
  candidate reuse these copies and have separate output paths.

## Source

Code commit: `6a345fa0`. Later documentation-only commits are allowed; the
second check enforces that. Read `CLAUDE.md`, `AGENTS.md`, and experiment 11's
README after pulling.

```bash
git status --short  # tracked source must be clean; never discard changes
git fetch origin codex/mineru-prefill-buckets
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 6a345fa0 HEAD
git diff --quiet 6a345fa0 HEAD -- '*.py'
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git rev-parse HEAD
```

## Discover the established environment

Locate the existing `prefill_timing_first256_310p_20261010T092230Z` under
`tmp/11_mineru_2_5_pro_inference/`. Read its `baseline/receipt/command.json`,
`baseline/output/run_summary_shard_00.json`, and coordinator log. Resolve its
actual path if nested differently. Take interpreter, model/layout/dataset paths,
device and cache roots from this successful receipt. Recover the CANN activation
from that run's launch environment or its successful predecessor's receipt.
If a required value is missing, use the successful C1+C2 combined-lane receipt
under `host_overlap_310p_*/combined/receipt/command.json`; record the source.
Do not guess, install packages, or download replacements.

Verify the reference used the same first 256 dataset pages and approximate
precision. Read its saved per-route timings when comparing; do not execute it.
The older first-256 brief specified NPU grids and blocking transfers, whereas
this candidate explicitly uses C1+C2 CPU grids and pinned nonblocking transfers.
Report the actual reference settings from its receipt, and disclose this host
configuration difference if present; the comparison is not a controlled
isolation of the new prefill changes alone.

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
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/prefill_buckets_candidate256_310p_$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
mkdir -p "$RUN_ROOT"
```

## Run once

The committed `candidate256` phase performs exactly:

1. `candidate_warmup64`: first 64 pages, full production candidate flags,
   plus `--local-warm-all-prefill-buckets` to compile/replay every configured
   vision/text bucket, including buckets absent from those pages.
2. `candidate256`: first 256 pages, identical candidate settings and warmed
   caches, without the all-bucket warmup flag. Metrics remain on.

The coordinator checks page accounting, source commits, approximate precision,
warmup coverage and unchanged graph artifacts during candidate256. It calls
`run_lane(..., require_idle_card=False)`. No retries or alternate settings are
automatic. Each child has a 14,400-second deadline to allow compilation.

```bash
nohup setsid bash -c '
  set +e
  "$PYTHON_BIN" -u 11_mineru_2_5_pro_inference/run_prefill_bucket_validation.py \
    --chip 310p --phase candidate256 --root "$RUN_ROOT" \
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

Poll every 30–60 seconds. Completion requires root `exit_code.txt` = 0,
both child receipts exit zero, warmup 64/0/0 completed/failed/skipped,
candidate 256/0/0, and `candidate256/cache_audit.json` reporting
`unchanged: true`. A graph artifact changing during measurement invalidates
its throughput. On failure, stop and report the command and log tail.

## Report to Luka in chat

Label values **310P**. Do not create a Markdown report file. Give the run-root
basename, commit, interpreter, visible device, environment-source receipt,
reference path/settings, and warmup bucket coverage first.

Report candidate256 pages, pipeline wall s, pg/s, prefill wall s, decode device s,
CPU preparation work/wait s, layout s and postprocess/write s when available.
Use `PREFILL_RESULT` and the measured summary; missing fields are unavailable,
not zero. Give every prefill event stage separately. These regions overlap;
do not add them as a wall total.

Paste the coordinator's **candidate256** vision and text-prefill route tables.
They include calls, multi-request/single-request calls, members, real/physical
tokens, device s, median/mean ms excluding each route's first call, and real/
physical tok/s. Text bucket routes include both single and multi-request calls;
there is no separate packed-text route. Include unused configured buckets with
zero calls and unavailable latency. Rates use all tokens / all device seconds;
event regions include host launch gaps. Raw samples are the measured output's
`vision_timing_shard_00.jsonl` and `text_prefill_timing_shard_00.jsonl`.

Compare sums of per-route `device_s` against the supplied historical values;
`PREFILL_REFERENCE_COMPARISON` and `reference_comparison.json` provide these:

| Stage | Existing first256 device s | Candidate256 device s | Saved s | Reduction % |
|---|---:|---:|---:|---:|
| Vision | 321.66 | | | |
| Text prefill | 75.88 | | | |

Verify the historical log's aggregation scope; disclose any mismatch rather
than silently changing the reference. Saved = reference − candidate; reduction
= 100 × saved / reference. No baseline rerun or extrapolation to full1651.

Report **warm-up/compile overhead separately** using `PREFILL_STARTUP`:

- Warmup64 total process wall (`child_wall_s`), setup s, pipeline wall s.
  Total process wall includes startup, pages, all-bucket replay and teardown;
  it excludes the coordinator's cache-copy/preflight time.
- Per vision/text bucket: `compile_wrapper_s`, `first_call_s`, and
  `cache_was_warm` when recorded; include replay-only routes from
  `all_bucket_warmup`. Also report decode wrapper/first-call s.
- First-call times include compile or cache loading plus execution; they are
  **not pure compiler time**. Pure compile time is unavailable unless explicitly
  recorded in compiler logs. Do not add overlapping wrapper/first-call records
  or count warmup time in candidate256 throughput.
- Candidate256 startup/first-call values and the unchanged-cache audit.

Finish with warnings, failures and deviations. Do not launch any further runs.
