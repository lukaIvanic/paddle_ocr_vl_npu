# Work-server 310P Experiment 09 method ladder

This is the execution brief for the AI agent on Luka's replacement Atlas 310P
server. Read `CLAUDE.md` and `AGENTS.md` first.

## Goal

Establish which Experiment 09 execution methods work correctly on this 310P
software stack and measure their approximate performance on small,
representative workloads.

The endpoint is an eight-page run through the real owned
`run_omnidocbench.py` pipeline with the useful optimizations enabled and
characterized. It is **not** a full OmniDocBench run.

## Explicit non-goals

- Do not run 256 pages.
- Do not run all 1,651 pages.
- Do not run an OmniDocBench accuracy evaluation.
- Do not run a large eager corpus.
- Do not compile the complete default vision/text bucket ladders.
- Do not use the 910B2 profile-guided vision policy as if it were a measured
  310P policy.

The largest OCR workload in this brief is eight pages. The only 32-page task is
the isolated layout lab, which does not load or execute the OCR recognizer and
was previously a few seconds on 310P.

## What the previous 310P server established

The old server reportedly reached this point:

- the real eager Experiment 09 path worked;
- aligned PromptFlashAttention worked;
- eager and TorchAir B4 decode outputs matched;
- the representative eight-page workload measured roughly:
  - eager manual B1: 100 s wall, 54 s decode, 48 raw decode tokens/s;
  - eager B4: 64 s wall, 16 s decode, 164 raw decode tokens/s;
  - TorchAir B4: 51 s wall, 4 s decode, 650 raw decode tokens/s;
  - aligned PromptFA plus TorchAir B4: about 40 s wall;
- isolated layout measured about 4.8 pages/s with one worker and 7.54 pages/s
  with two workers.

Those values were manually relayed and no raw 310P artifacts or graph caches
were committed. They are comparison anchors only. The replacement server must
recreate hardware-specific caches and evidence.

Compiled vision prefill, compiled text prefill, and the actual owned
`run_omnidocbench.py` path were not proven on the old 310P server.

## Why the earlier IndexPut error is not the target path

The historical one-crop helper moves `input_ids` and `attention_mask` to NPU
before calling `get_rope_index()`. Its boolean assignment therefore invokes NPU
`aten::index_put_` / `IndexPutV2`.

Experiment 09's `ContinuousRecognizer._prepare_cpu` computes MRoPE
`position_ids` and `rope_deltas` on CPU and transfers the completed tensors to
NPU. Do not use `paddleocr_vl.model.example` as the compatibility gate. Start
with Phase 1 below.

## Operating rules

- The work-server checkout is pull-only. Do not edit tracked files, commit,
  push, or create branches.
- Run `git pull --ff-only origin main` before testing.
- Do not install or replace PyTorch, torch-npu, TorchAir, CANN, or system
  packages.
- Use one free physical 310P device. Do not terminate another user's process.
- Use absolute discovered paths; do not assume Blue Zone `/workspace` paths.
- Every compiled cache must be created locally on this 310P stack. Never copy a
  910B or old-server cache.
- Preserve the expanded command, Git commit, environment fingerprint, complete
  log, exit code, cache inventory, and output JSON for every lane.
- Put small evidence below:

```text
tmp/09_persistent_page_engine/310p_exp09_ladder/
```

- Put compiler caches below `.runtime_cache/`, with `310p` and the graph shape
  in their names.
- Stop at the first failed dependency chain. Preserve the first causal
  traceback rather than changing random flags.

## Phase 0: environment, artifacts, and workloads

Pull and establish the working variables:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_exp09_ladder"
mkdir -p "$OUTPUT_ROOT"

PYTHON_BIN="<absolute compatible Python>"
RECOGNIZER_MODEL="<absolute PaddleOCR-VL-1.6 model directory>"
LAYOUT_MODEL="<absolute PP-DocLayoutV3_safetensors directory>"
DATASET_JSON="<absolute OmniDocBench.json>"
IMAGES_DIR="<absolute OmniDocBench images directory>"
```

Activate the server's intended NPU environment and confirm one free Atlas 310P
device. Record:

- Git commit;
- hostname and exact NPU product;
- Python executable;
- torch, torch-npu, TorchAir, CANN, driver, and firmware versions;
- resolved CANN and OPP paths;
- model and dataset paths;
- free HBM and cache-filesystem space.

The selected interpreter must import the Experiment 09 dependencies and pass a
basic tensor operation on logical `npu:0`.

### Workload A: one complete reference page

Use:

```sh
REFERENCE_PAGE="$IMAGES_DIR/PPT_The Right Moves_page_024.png"
test -f "$REFERENCE_PAGE"
```

This page contains five retained recognition regions and previously completed
with 81 generated tokens including EOS. It is the quick full-output correctness
gate.

### Workload B: representative uniform eight pages

Use annotation indices:

```text
0, 236, 471, 707, 943, 1179, 1414, 1650
```

Resolve those exact entries from `OmniDocBench.json`, map each annotated
basename beneath `IMAGES_DIR`, and write the eight absolute paths in order to:

```text
$OUTPUT_ROOT/uniform8_paths.txt
```

Verify only those eight files with OpenCV. Do not audit or infer on all 1,651
pages. Construct repeatable `--image` arguments programmatically from this
file; do not retype or reorder the names.

This is the established 164-recognition-region comparison workload.

### Workload C: first eight production pages

The actual production runner uses:

```text
--offset 0 --limit 8
```

This is a different eight-page set from Workload B. Never compare their
fingerprints or wall times directly.

## Phase 1: real Experiment 09 serving-path smoke

Run the real `ContinuousRecognizer` through `run_offline_e2e.py`, not the
one-crop helper:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_offline_e2e.py" \
  --image "$REFERENCE_PAGE" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --dtype fp16 \
  --decode-backend raw_eager \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --text-backend raw_eager \
  --text-padding none \
  --batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 8 \
  --max-regions 2 \
  --no-save-annotated \
  --output-dir "$OUTPUT_ROOT/phase1_real_path_smoke"
```

This deliberately caps two regions and eight output tokens. Require:

- exit code zero and a parseable `run.json`;
- at least one real region produces token IDs and text;
- all three stages record `raw_eager`;
- no TorchAir compile and no CPU/CUDA fallback;
- no NPU IndexPut failure.

If this real serving path fails at IndexPut, run only Phase 4A from
`WORK_SERVER_310P_EAGER_SMOKE.md`, report it, and stop. If it passes, the old
one-crop helper result is irrelevant to the Exp09 ladder.

## Phase 2: one-page correctness gates

Run the same reference page without a region cap and allow natural EOS:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_offline_e2e.py" \
  --image "$REFERENCE_PAGE" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --dtype fp16 \
  --decode-backend raw_eager \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --text-backend raw_eager \
  --text-padding none \
  --batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 2808 \
  --no-save-annotated \
  --output-dir "$OUTPUT_ROOT/phase2_manual_full_output"
```

Require all regions to complete naturally by EOS with no missing result.

Repeat with only:

```text
--vision-attention prompt_flash_attention
--vision-promptfa-align-128
--output-dir "$OUTPUT_ROOT/phase2_aligned_pfa_full_output"
```

The 310P alignment flag is mandatory. It rounds PromptFA physical sequence
lengths to a multiple of 128 and avoids the known unaligned-mask rejection.
Require exact token/text parity with the manual lane. Record real and physical
vision tokens and vision device time.

If aligned PromptFA fails, preserve the exact native-op traceback and continue
later lanes with manual attention. Do not invent attention arguments.

## Phase 3: recreate the old-server decode milestone

Use Workload B, every region, KV4096, and a 32-token cap. This is the only
multi-page eager matrix in the brief.

Run these lanes in order:

| Lane | Vision | Decode | Batch |
|---|---|---|---:|
| A | manual eager | raw eager | 1 |
| B | aligned PromptFA eager | raw eager | 1 |
| C | aligned PromptFA eager | raw eager | 4 |
| D-cold | aligned PromptFA eager | TorchAir | 4 |
| D-warm | aligned PromptFA eager | same TorchAir cache | 4 |

Vision and text execution remain `raw_eager` with padding `none` in every lane.
Only Lane D compiles decode.

Create a new decode cache:

```sh
DECODE_B4_CACHE="$REPO/.runtime_cache/310p_decode_b4_k4096_$(git rev-parse --short HEAD)"
test ! -e "$DECODE_B4_CACHE"
```

All commands share:

```text
--layout-model "$LAYOUT_MODEL"
--recognizer-model "$RECOGNIZER_MODEL"
--dtype fp16
--cache-length 4096
--max-new-tokens 32
--vision-backend raw_eager
--vision-padding none
--text-backend raw_eager
--text-padding none
--no-save-annotated
```

Apply the lane-specific vision attention, decode backend, batch size, cache,
and output directory from the table. Run D-warm in a fresh process with the
exact D-cold command and cache.

Require:

- eight pages, 164 recognized regions, and zero partial pages;
- identical structural accounting in every lane;
- A versus B exact parity;
- C versus D-cold exact parity;
- D-cold versus D-warm exact parity;
- no new graph creation in D-warm.

B1 and B4 may produce different tokens from batch numerics. Report that
difference but do not fail B4 if C and D agree.

Report wall time, vision/text device time, decode wall, raw/effective decode
tokens/s, active-slot fraction, admissions, first-call compile/load time, cache
size, and output fingerprints.

This phase re-establishes the previous 310P stopping point. Do not proceed if
TorchAir B4 does not replay correctly.

## Phase 4: compile vision and text prefill

Keep Workload B, aligned PromptFA, TorchAir B4 decode, KV4096, and max32.
Packing remains off.

Use exactly five workload-covering vision buckets:

```text
640,768,1408,2944,4992
```

Use exactly five workload-covering text buckets:

```text
176,208,384,768,1280
```

Create fresh caches:

```sh
VISION_CACHE="$REPO/.runtime_cache/310p_vision_pfa_align128_compact_$(git rev-parse --short HEAD)"
TEXT_CACHE="$REPO/.runtime_cache/310p_text_prefill_compact_$(git rev-parse --short HEAD)"
test ! -e "$VISION_CACHE"
test ! -e "$TEXT_CACHE"
```

Run three lanes:

1. compiled vision, eager text;
2. compiled vision plus cold compiled text;
3. fresh-process warm replay of lane 2.

The common compiled-vision arguments are:

```text
--decode-backend torchair
--torchair-cache-dir "$DECODE_B4_CACHE"
--batch-size 4
--cache-length 4096
--max-new-tokens 32
--vision-backend torchair
--vision-attention prompt_flash_attention
--vision-promptfa-align-128
--vision-padding bucket
--vision-buckets 640,768,1408,2944,4992
--vision-torchair-cache-dir "$VISION_CACHE"
```

Lane 1 uses:

```text
--text-backend raw_eager
--text-padding none
```

Lanes 2 and 3 use:

```text
--text-backend torchair
--text-padding bucket
--text-buckets 176,208,384,768,1280
--text-torchair-cache-dir "$TEXT_CACHE"
```

Require 164 compiled vision requests with no vision overflow. Lanes 2 and 3
must also have 164 compiled text requests with no text overflow.

Compare ordered request ID, token IDs, text, and stop reason:

- Phase 3 D-warm versus compiled-vision/eager-text;
- compiled-vision/eager-text versus compiled-text cold;
- compiled-text cold versus warm.

Exact parity is expected. Report compile/load setup separately from inference
wall, plus useful and physical vision/text tokens/s.

## Phase 5: actual owned Exp09 production path

Now cross from the diagnostic runner into:

```text
run_omnidocbench.py
  -> OwnedLayoutFrontend
  -> bounded page producer
  -> ContinuousRecognizer
  -> compiled vision/text/decode
  -> page assembly and artifact writer
```

Use Workload C and the warm B4/vision/text caches:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --offset 0 \
  --limit 8 \
  --dtype fp16 \
  --batch-size 4 \
  --cache-length 4096 \
  --max-new-tokens 32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-promptfa-align-128 \
  --vision-padding bucket \
  --vision-buckets 640,768,1408,2944,4992 \
  --vision-packing off \
  --text-padding bucket \
  --text-buckets 176,208,384,768,1280 \
  --text-packing off \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --vision-torchair-cache-dir "$VISION_CACHE" \
  --text-torchair-cache-dir "$TEXT_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase5_production_b4"
```

The production runner intentionally has no `--decode-backend` or
`--text-backend`; both are TorchAir.

Repeat in a fresh process with the same caches and `_warm` output suffix.
Require:

- `result_count == prediction_count == 8`;
- eight Markdown predictions and one result per selected page;
- consistent page/request/decode accounting;
- no PaddleX import or CPU/CUDA fallback;
- cross-page crop scheduling and continuous decode admissions are active;
- the background CPU recognition-preparation worker is active;
- recognition-input H2D uses the dedicated transfer-stream/event path;
- page results are emitted and written incrementally rather than retained until
  the entire eight-page run ends;
- timeline HTML/JSON, recognition trace, `page_regions.jsonl`, and
  `run_summary.json`;
- exact cold/warm output parity;
- no unexpected warm-process compilation.

Report any eager overflow sequence lengths. A tail overflow is not a
compatibility failure, but it must remain visible.

Use the timeline and summary to report whether prefill H2D overlap, background
CPU preparation, decode hot swapping, and the bounded page frontend are
actually exercised. These are built-in production mechanisms rather than CLI
alternatives, so they need observation, not separate synthetic modes.

At this point the actual Experiment 09 production path is proven on 310P.

## Phase 6: optimization compatibility and performance

Every comparison below uses the same first eight production pages and the
Phase 5 warm lane as its baseline. Change one method at a time.

### Phase 6A: larger continuous-decode arena

Approximate fp16 KV-only allocations at KV4096 are:

```text
B4:   1.69 GiB
B16:  6.75 GiB
B32: 13.50 GiB
```

Create a fresh B16 decode cache and run cold/warm Phase 5 variants with
`--batch-size 16`. Require cold/warm exact parity and compare against B4.

Attempt B32 only if B16 passes, improves E2E, and observed HBM leaves enough
headroom for another approximately 6.75 GiB plus compiler transients. Use a
fresh B32 cache and the same cold/warm gate. Do not sweep other batch sizes.

Select the fastest stable batch size, not automatically the largest.

### Phase 6B: greedy compiled vision packing

Keep the selected decode size and all other Phase 5 settings. Add `1920` to the
vision bucket list and allow that one new singleton graph to compile:

```text
--vision-buckets 640,768,1408,1920,2944,4992
--vision-packing greedy
--vision-pack-target 1920
```

Run cold/warm with the same vision cache. Report:

- groups and crops/group;
- real/physical vision tokens and fill fraction;
- vision-transformer calls and device time;
- token/text differences from packing-off;
- E2E change.

Structural page/request accounting and cold/warm parity are mandatory.

### Phase 6C: packed text prefill

Packed text is meaningful only after vision groups exist. Starting from the
greedy-vision lane, add:

```text
--text-packing production_group
--text-pack-buckets 128,256,512,1024
--text-pack-max-members 32
--text-packed-cache-dir "<new 310P packed-text cache>"
```

Run cold/warm. Report text-transformer calls, KV redistribution time,
real/physical text tokens, useful tokens/s, token differences, and E2E change.
Require structural accounting and cold/warm parity.

### Phase 6D: reduced-min-pixels speed point

Run one cumulative candidate only:

```text
--preprocessor-min-pixels 28224
```

Add only these small singleton buckets so the reduced shapes are not all padded
to 640/176:

```text
vision additions: 128,256,384,512
text additions:    32,64,96,128
```

Reuse existing larger buckets and compile only the missing shapes. Keep the
selected decode size, greedy vision packing, and packed text.

Output changes relative to default `min_pixels` are expected. The gate is
structural correctness, cold/warm reproducibility, and measured speed. Report
real/physical token reduction and all stage/E2E changes.

### Phase 6E: profile-guided batched vision status

Do not enable production `--vision-packing profile_guided` in this pass. Its
route-selection timings are pinned to Ascend 910B2 and it requires B2x3072 and
B4x1024 batched graph caches.

The correct 310P conclusion is currently:

```text
batched graph method: available for isolated compilation/profiling
production route policy: not yet 310P-calibrated
```

If the user later asks to pursue it, compile and profile those two shapes in the
vision lab, build a measured 310P routing table through the local authoring
lane, then rerun the same eight-page production comparison. Do not edit the
hard-coded profile on the work server.

## Phase 7: isolated layout method check

This is the only 32-page task. It does not load the OCR recognizer.

Run:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --limit 32 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/phase7_layout_w1"
```

Then:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --limit 32 \
  --workers 2 \
  --no-timeline \
  --reference-requests "$OUTPUT_ROOT/phase7_layout_w1/requests.jsonl" \
  --output-dir "$OUTPUT_ROOT/phase7_layout_w2"
```

The two-worker mode overlaps next-page CPU image decode; it does not batch
pages through the detector and it is not a production-runner `--workers`
switch.

Report pages/s, stage totals, and manifest comparison. Exact equality is ideal;
list any tolerated one-pixel resize difference honestly.

## Stop condition and required report

Stop after Phase 7. Do not start a larger OCR workload.

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

End the agent response with:

```text
310P EXP09 METHOD LADDER: PASS | PARTIAL | FAIL

Git / host / exact NPU:
Python / torch / torch_npu / TorchAir / CANN:
Model and dataset paths:

Real serving-path smoke:
Manual full-output page:
Aligned PromptFA full-output page / parity:

Uniform-eight matrix:
manual eager B1:
aligned-PFA eager B1:
aligned-PFA eager B4:
aligned-PFA TorchAir B4 cold:
aligned-PFA TorchAir B4 warm:
B4 eager-compiled parity:
B4 cache:

Compiled prefill:
vision compiled / text eager:
vision+text compiled cold:
vision+text compiled warm:
vision/text parity and overflows:
vision/text cache sizes:

Owned production eight-page:
B4 cold / warm:
pages / requests / predictions:
stage and token totals:
output parity:
timeline / trace / summary:

Optimization methods:
B16:
B32 or reason skipped:
selected decode size:
greedy vision packing:
packed text:
min_pixels/4:
profile-guided status:

Layout W1 / W2:
Layout manifest comparison:

Best stable production configuration:
Best eight-page wall / pages-s:
Vision useful / physical tokens-s:
Text useful / physical tokens-s:
Decode raw / effective tokens-s:
Peak HBM:

First blocker or warning:
Exact command records:
Artifact paths:
```

Report setup/compile time separately from inference wall. Preserve a baseline
even when an optimization wins. Do not describe eight-page performance as
full-corpus throughput.
