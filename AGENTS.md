# AGENTS.md

## Operating Lanes

`CLAUDE.md` is the current orientation for this repo and holds the full lane
model, verified machine facts, and evidence conventions. Read it first. This file
holds the deeper per-experiment history behind the design.

The short version:

- **Local authoring** — Luka's Mac checkout, no accelerator. Edits tracked files,
  commits, pushes, and drives the 910B container over SSH. It must not present
  unrun local code as validated inference.
- **Blue-zone 910B container** (`ssh blue_zone_npu_container`) — the real
  validation lane, reachable from local. Pull-only for source: edit locally,
  push, `git pull` there, run. Never hand-edit tracked files on the container.
- **310P work server** — Atlas 310P devices, not reachable from local and
  pull-only from GitHub. Driven by a self-contained written handoff brief; its
  agent runs and reports, and Luka relays the report back manually. That agent
  must not edit tracked files, commit, push, or create branches. If a code change
  is needed, report the minimal proposed change, the failing command, and the
  relevant logs instead of applying it.

Historical runbooks below use `$WORK_SERVER_REPO` for the work-server checkout
root (`/home/lukaiv/paddle_ocr_vl_npu`); resolve it with
`WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"` rather than hardcoding a
path. Those runbooks were written on 310P — re-verify before trusting any of
them on 910B.

There is no CUDA lane. The Vast.ai GPU rental described in earlier revisions of
this file is retired, and its runbooks have been removed.

## Domain Distinctions

Current direction lives in `CLAUDE.md`, not here — it changes. These
distinctions do not:

- `PaddleOCR-VL-1.6-0.9B` is the VLM recognition component. Upstream it is
  `PaddlePaddle/PaddleOCR-VL-1.6`, loaded through `AutoProcessor` and
  `AutoModelForImageTextToText` / `PaddleOCRVLForConditionalGeneration`.
- Full `PaddleOCR-VL-1.6` page parsing is layout analysis plus recognition plus
  merge/postprocess. For v1.6 that is `PP-DocLayoutV3` plus the 0.9B recognizer.
- Recognizer-only runs are valid for element-level crops and prompts such as
  `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`,
  `Spotting:`, and `Seal Recognition:`.
- Recognizer-only runs are **not** proof of full page-parser quality or
  throughput.

Known implementation surfaces:

- Official PaddleOCR/PaddleX provides the full parser pipeline and Paddle-facing
  configs. This repo no longer imports either; experiment 09 owns the page
  contract directly.
- Hugging Face Transformers provides the core recognizer/VLM directly.
- Huawei Ascend public guidance currently points to PaddleOCR client/pipeline
  plus vLLM VLM service, or a two-container full API service, rather than direct
  local NPU inference. Treat direct Transformers-on-NPU work as an experiment
  until validated.

## Experiment Ladder

Each numbered directory is a rung, kept as the evidence for how the current
design was reached. **09 is the active one**; 01–08 are historical and are not
maintained against current behavior.

- `01_transformers_recognition_baseline/`: minimal Transformers recognizer smoke.
- `02_local_eager_recognition/`: the recognizer reimplemented in local PyTorch
  with no Transformers imports. Everything since is derived from this.
- `03_compiled_single_batch_decode/`: static KV-cache decode, TorchAir compile.
- `04_batched_fixed_cohort_decode/`: batched decode plus the first vLLM-style
  hot-swap scheduler probe.
- `05_full_recognizer_optimizations/`: whole-recognizer stage timing and the
  100-crop queue benchmark. Where the vision-attention work happened.
- `06_full_page_pipeline_e2e/`: first end-to-end page pipeline.
- `07_vision_prefill_optimization/`: vision prefill bucketing and PromptFA work.
  Has its own `README.md`.
- `08_offline_e2e_b1/`: first persistent offline system — sequential bucketed
  compiled vision/text prefill into one continuous compiled decode schedule.
  Has its own `README.md`; read it before interpreting its throughput or parity.
- `09_persistent_page_engine/`: **active.** The persistent page engine. Has its
  own `README.md`, which is authoritative over anything said here.

Supporting artifacts:

- `crops/`: eight OmniDocBench region crops, not full pages.
- `crops/manifest.json`: source image, category, bbox, suggested prompt, and ground truth for each crop.
- `crops/create_omnidocbench_recognition_crops.py`: reproducible crop generator.
- `crops/create_omnidocbench_hotswap_crops.py`: reproducible generator for the larger queue/hot-swap crop set.
- `crops/hotswap_*.png`, `crops/hotswap_100_manifest.json`, `crops/hotswap_100_summary.json`, and `crops/hotswap_100_contact_sheet.jpg`: 100 additional OmniDocBench region crops for batch sizing and future vLLM-style slot hot-swapping tests. Use the hot-swap manifest explicitly; the default tiny smoke manifest remains `crops/manifest.json`.
- `refs/`: small architecture reference artifacts.
- `refs/PaddleOCR`: ignored sparse reference checkout of the official PaddleOCR repo.

Keep `refs/PaddleOCR/` ignored. It is reference material, not project source.

## Smoke Commands

Crop generation runs anywhere. Everything that touches the model needs an NPU, so
run it on the 910B container after `source npu-setup`.

Regenerate the crops from the parent repo's restored OmniDocBench copy:

```sh
python3 crops/create_omnidocbench_recognition_crops.py
python3 crops/create_omnidocbench_hotswap_crops.py
```

Run the core recognition model on one crop with Transformers:

```sh
python3 01_transformers_recognition_baseline/run_transformers_recognition.py
```

This runner uses the **slow** image processor by default, deliberately: the local
preprocessing path follows the slow PaddleOCR-VL image processor source, and the
HF fast processor has small resize rounding differences, so it is not the parity
target. Pass `--use-fast` only when comparing against the fast processor on
purpose.

For another crop:

```sh
python3 01_transformers_recognition_baseline/run_transformers_recognition.py \
  --crop crops/crop_05_table_rwkv_dims.png \
  --prompt "Table Recognition:"
```

Run the local no-Transformers recognition path:

```sh
python3 02_local_eager_recognition/run_local_recognition.py \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:"
```

All experiment CLIs default to `--dtype fp16`. `bf16` remains an explicit
override, but `fp32` is intentionally not a supported run mode.

In the compiled-decode experiments, the bench/probe scripts request
FRACTAL_NZ text-decoder and `lm_head` weights before compile. Runtimes such as
torch-npu 2.10 that disable internal formats keep the native weight format;
the scripts report `decode_native_fallback` and use a separate compile-cache
key. There is intentionally no user-selectable linear-format option.

`03_compiled_single_batch_decode` static decode is IncreFA-only. It uses masked
`torch_npu.npu_incre_flash_attention` with a bool future-slot mask and does not
use `actual_seq_lengths`. There is intentionally no manual/static attention
option now.

Use `03_compiled_single_batch_decode/bench_static_compile.py --eos-mode` to
compare decode-loop EOS behavior:

- `none`: fixed-step decode with no per-token host EOS check.
- `overlap_event_flags`: GLM-OCR-style queue-depth-1 EOS check using a second
  NPU stream, event wait/record, and pinned CPU bool flags.

Use `04_batched_fixed_cohort_decode/bench_static_compile.py` to benchmark
batched decode over distinct real crops:

```sh
python3 04_batched_fixed_cohort_decode/bench_static_compile.py \
  --batch-size 4 \
  --manifest crops/hotswap_100_manifest.json \
  --backend torchair \
  --device npu:0 \
  --eos-mode overlap_event_flags
```

Experiment 4 deliberately avoids padded text/image prefill. It selects real
entries from `crops/manifest.json`, runs preprocessing and prefill one crop at
a time, then concatenates the B single-row static KV caches and runs decode as
a batch. The shared `cache_length` is static decode capacity, not text padding;
each row keeps its own `cache_position` and future-slot mask.

For experiment 4 throughput, treat `*_decode_steps` as graph calls per second
and `*_raw_batch_tokens` as batch token-slots per second. Effective token
metrics exclude EOS-fill tokens after per-row or per-item completion.

Experiment 4 hot-swap mode is the first vLLM-style decode scheduler probe. It
assumes preprocessing and prefill are solved outside the measured decode loop:
the script prefills all selected crops into an NPU-resident ready bank, then
keeps only `--batch-size` rows in the active compiled decode cache. When one or
more active rows finish, the token-copy path copies the active next-token
vector to pinned CPU memory, the host computes per-slot completion from that
copied token vector, and every finished slot is swapped in one pass. The current
experiment-5 queue synchronizes that copied token row before the next decode so
slot replacement is exact; do not describe it as fully hidden under the next
decode. Do not assume only one slot can finish at a time; check
`swap_events[*].finished_slots`, `swapped_in_item_ids`, and
`immediate_finished_slots`.

Hot-swap requires `--eos-mode overlap_event_flags`; `--eos-mode none` is only a
fixed-step fixed-cohort speed baseline and the script rejects it for hot-swap.
The decode mask is still built on device from each row's `cache_position`, so a
slot swap resets the row cache, next token, `cache_position`, and `rope_deltas`;
no text/image padding is introduced.

Backend vocabulary, repo-wide: `--backend eager` means
`torch.compile(..., backend="eager", fullgraph=True, dynamic=False)`, while
`raw_eager` is true uncompiled Python/PyTorch execution. They are not synonyms,
and `raw_eager` is the correctness control, not a competing production path.

Recommended NPU run order for experiment 4:

For the current hot-swap bottleneck investigation, prefer the committed matrix
runner instead of ad hoc shell snippets:

```sh
cd "$WORK_SERVER_REPO"/04_batched_fixed_cohort_decode
bash run_npu_hotswap_bottleneck_matrix.sh
```

The runner executes the fixed baseline plus hot-swap `num-items=8,9,16,32,100`
matrix once with `--step-timing off` for clean throughput and once with
`--step-timing both` for diagnostics. It writes and validates one summary JSON
per run. See
`04_batched_fixed_cohort_decode/NPU_HOTSWAP_BOTTLENECK_MATRIX.md` for what to
paste back.

## Experiment 5

`05_full_recognizer_optimizations` is derived from the current experiment-4
local model/compiled-decode baseline, but the experiment-4 hot-swap matrix
runner is intentionally not copied forward. Experiment 5 moves the optimization
target from decode scheduling to the full recognition model. Layout detection is
still out of scope. The stage-timing harness focuses on model stages after a
crop has been selected; the queue benchmark also includes CPU crop image
read/decode, preprocessing, patchify, and prompt construction. The first
experiment-5 question is stage cost: how expensive are the native-resolution
vision transformer, adaptive MLP connector, text prefill, LM head, and static
decode on the real OmniDocBench crops in `crops/`.

Run the committed NPU stage-timing harness instead of writing ad hoc snippets:

```sh
cd "$WORK_SERVER_REPO"/05_full_recognizer_optimizations
bash run_npu_stage_timing.sh
```

The runner prints `CORRECTNESS`, `SETUP_TIMING_S`, `VISION_ATTENTION`,
`STAGE_SUMMARY_S`, and `ITEM_SUMMARY` after validating the JSON. Paste those
printed sections back instead of writing a separate parser.

By default, the runner uses `WARMUP_ITEMS=1`. That warmup item is recorded under
`STAGE_WARMUP` and excluded from the measured item summary. This is intentional:
experiment 5 is measuring steady-state recognizer stage latency, while cold
TorchAir/CANN compile and first-use behavior belong in setup/warmup fields. If
you intentionally need the cold first-item behavior again, run with
`WARMUP_ITEMS=0`.

Vision attention defaults to the manual PyTorch path. To stage-time the
experimental prompt flash attention path after the vision-only validation passes,
run:

```sh
VISION_ATTENTION_IMPL=prompt_flash_attention bash run_npu_stage_timing.sh
```

If the first measured item has a huge `static_decode_total` outlier, rerun these
two diagnostics exactly:

```sh
# Isolate the first crop and print per-decode-step synchronized timings.
NUM_ITEMS=1 \
CROP_IDS=hotswap_001_code_txt_p0001_box_id_3 \
DECODE_STEP_TIMING=1 \
bash run_npu_stage_timing.sh

# Move one full staged item through the model before measured timing.
# If the outlier was first-use compile/cache behavior, the measured item 001
# should drop to the normal decode range here.
WARMUP_ITEMS=1 bash run_npu_stage_timing.sh
```

The runner writes and validates one JSON under
`outputs/full_recognizer_stage_timing/`, then fails if staged generation does
not match direct local static fixed-step generation. For the first report, paste
back:

- `correctness`
- `setup_timing_s`
- `stage_timing_summary_s.native_resolution_visual_encoder_total`
- `stage_timing_summary_s.vision_total`
- `stage_timing_summary_s.vision_encoder`
- `stage_timing_summary_s.adaptive_mlp_projector`
- `stage_timing_summary_s.mrope_index_cpu`
- `stage_timing_summary_s.mrope_index_transfer`
- `stage_timing_summary_s.text_prefill`
- `stage_timing_summary_s.prefill_lm_head`
- `stage_timing_summary_s.static_decode_total`
- `stage_timing_summary_s.model_total_excluding_device_transfer`
- each item's `input_tokens`, `vision_tokens`, `projected_image_tokens`,
  `decode_calls`, `decode_mode`, `correctness`, and `timing_s`

Stage timing uses device synchronization around each measured model stage. This
adds measurement overhead, so use it to identify bottleneck proportions before
turning any stage into a throughput benchmark.

For the 100-crop full-recognizer queue benchmark, use the dedicated runner. This
is the experiment-5 serving-shaped pass where all crops are known up front: CPU
preprocessing/prompt construction for real crops, sequential device
vision/projector/text prefill into per-crop static-cache states, then hot-swap
decode through one active compiled decode batch.
Pixel/hidden/embedding tensors run on the selected device; the small
`image_grid_thw` shape metadata intentionally stays on CPU to avoid scalar
device-to-host syncs inside vision/projector shape loops.

For the data and code flow, read
`05_full_recognizer_optimizations/QUEUE_BENCHMARK_FLOW.md` before debugging the
queue benchmark. It documents the object shapes, timing buckets, hard checks,
and the decode-path differences. Do not reverse engineer this
script by writing helper snippets unless the committed runner fails to print the
needed field.

```sh
cd "$WORK_SERVER_REPO"/05_full_recognizer_optimizations
bash run_npu_recognizer_queue_benchmark.sh
```

Defaults are `NUM_ITEMS=100`, `ACTIVE_BATCH_SIZE=1`, `DECODE_SCHEDULE=hotswap`,
`CACHE_LENGTH=1024`, `MAX_NEW_TOKENS=32`, `DECODE_BACKEND=torchair`,
`EOS_MODE=overlap_event_flags`, manual vision attention, fp16, NPU JIT compile
off, and queue-vs-same-local-static-model validation for all items. This
validation is not an independent OCR quality or ground-truth check. The runner
writes one JSON under
`outputs/recognizer_queue_benchmark/` and prints:

- `QUEUE_BENCHMARK_SUMMARY`
- `CACHE_PREFLIGHT`
- `SETUP_TIMING_S`
- `PHASE_TIMING_S`
- `PIPELINE_STAGE_TIMING_SUMMARY_S`
- `DECODE_QUEUE_DETAILS`
- `THROUGHPUT`
- `DECODE_SUMMARY`
- `CORRECTNESS`
- `READY_STAGE_SUMMARY_S`
- `ITEM_SUMMARY`
- `TEXT_SAMPLE`
- `OUTPUT_JSON`

If `CACHE_LENGTH=1024` is too small, the script exits early with a valid JSON
containing `error="cache_length_too_small"`, `CACHE_PREFLIGHT.overflow_count`,
and the maximum `required_cache_length`. In that case, do not edit code and do
not write helper scripts; rerun only by increasing `CACHE_LENGTH`, for example:

```sh
CACHE_LENGTH=1536 bash run_npu_recognizer_queue_benchmark.sh
```

To test higher hot-swap active batch sizes, use real crops and change only
`ACTIVE_BATCH_SIZE`. Hot-swap rejects `ACTIVE_BATCH_SIZE > NUM_ITEMS` instead
of creating fake initial rows. Once the ready bank is exhausted, inactive tail
slots are filled with EOS/cache-position-zero sentinels and still occupy the
static compiled batch until the remaining real rows finish; use effective-token
metrics for useful work.

Recommended NPU run order:

```sh
ACTIVE_BATCH_SIZE=1 CACHE_LENGTH=1536 bash run_npu_recognizer_queue_benchmark.sh
ACTIVE_BATCH_SIZE=4 CACHE_LENGTH=1536 bash run_npu_recognizer_queue_benchmark.sh
ACTIVE_BATCH_SIZE=8 CACHE_LENGTH=1536 bash run_npu_recognizer_queue_benchmark.sh
```

These commands must print `decode_schedule="hotswap"`,
`scheduler="hotswap_ready_state_queue"`, and
`DECODE_QUEUE_DETAILS.diagnostics.slot_control_write_mode="batched_index_copy_for_swaps_and_slot_control"`.
They must also print `decode_attention="increfa"`,
`decode_cache_update="npu_scatter"`, and pass
`CORRECTNESS.all_required_checks_passed=true`. Also report
`DECODE_SUMMARY.length_cap_hit_count`; length-capped outputs are allowed for
short benchmark caps, but they are not full-quality OCR completions. Do not
write inline Python helper scripts to inspect the JSON; the runner prints the
fields needed for the report.

For a fixed-cohort baseline only, opt in explicitly:

```sh
DECODE_SCHEDULE=fixed_cohort ACTIVE_BATCH_SIZE=8 CACHE_LENGTH=1536 \
bash run_npu_recognizer_queue_benchmark.sh
```

The measured `decode_queue` phase includes the hot-swap scheduler work over
already-prefilled states, including sampled-token row copies used for EOS and
slot replacement. Final token-row materialization, EOS trimming, and tokenizer
decode are reported as `decode_output_postprocess`. Validation is separate from
measured throughput.

Use the clearer `PIPELINE_STAGE_TIMING_SUMMARY_S` names when summarizing:
`vision_prefill`, `text_prefill`, and `text_decode`. The older
`READY_STAGE_SUMMARY_S` remains available for detailed substage debugging.

Paste back the printed `QUEUE_BENCHMARK_SUMMARY`, `CACHE_PREFLIGHT`,
`PHASE_TIMING_S`, `PIPELINE_STAGE_TIMING_SUMMARY_S`, `DECODE_QUEUE_DETAILS`,
`THROUGHPUT`, `DECODE_SUMMARY`, `CORRECTNESS`, `READY_STAGE_SUMMARY_S`,
`ITEM_SUMMARY`, `TEXT_SAMPLE`, and `OUTPUT_JSON`. Do not write a separate
parser; the runner already validates the JSON and exits nonzero if correctness
fails. `READY_STAGE_SUMMARY_S` must include `mrope_index_cpu` and
`mrope_index_transfer` when using the current experiment-5 runners.

For native-resolution vision encoder profiling, use the committed profiler
runner. It profiles only `vision_model.encoder`; crop preprocessing, device
transfer, patch/position embeddings, post layernorm, the adaptive MLP projector,
text prefill, and decode are outside the profiler window.

```sh
cd "$WORK_SERVER_REPO"/05_full_recognizer_optimizations
bash run_npu_vision_profile.sh
```

The default crop is `hotswap_002_code_txt_p1474_11`, the large 3036-vision-token
crop from the first stage-timing report. The default profiler metric is
`PROFILE_METRIC=pipe`, with one warmup encoder pass and one profiled encoder
pass. The default vision implementation is now
`VISION_ATTENTION_IMPL=prompt_flash_attention`; the profiler first compares it
against the manual vision encoder in the same process before profiling. Paste
back the printed sections:

- `VISION_PROFILE_SUMMARY`
- `VISION_ATTENTION_VALIDATION`
- `STEP_TRACE_TOTALS_US`
- `TOP_KERNEL_TYPES`
- `TOP_MATMUL_SHAPES`
- `TOP_TRANSDATA_SHAPES`
- `TOP_SUSPECT_KERNELS`
- `TOP_OPERATORS_BY_DEVICE_US`
- `PROFILE_PARSE_MD`

Do not write a separate parser; the runner already calls `parse_npu_profile.py`
and prints the important rows. If the pipe profile points at memory bandwidth or
cache behavior, rerun the same command with `PROFILE_METRIC=memory` or
`PROFILE_METRIC=l2` and label the result clearly.

If `VISION_ATTENTION_VALIDATION.allclose_atol_5e_2_rtol_5e_2` is false or the
prompt flash attention call crashes, stop and paste back the error/validation
block. Do not continue to full stage timing until the vision-only validation is
acceptable.

If prompt flash attention diverges from manual attention, run the committed
single-layer call-contract probe next:

```sh
cd "$WORK_SERVER_REPO"/05_full_recognizer_optimizations
bash run_npu_vision_prompt_fa_probe.sh
```

Paste back `VISION_PROMPT_FA_PROBE` and `VARIANT_RESULTS`. This probe compares
manual attention against several PromptFlashAttention call variants on the first
vision layer only, before the output projection and before 27-layer error
accumulation.

The default model integration should use the minimal BNSD full-attention call on
310P: no `actual_seq_lengths`, no `actual_seq_lengths_kv`, and no explicit
`num_key_value_heads`. Huawei documents those arguments as limited on Atlas
inference-series products, while the no-length/no-GQA variants are both correct
and faster in the single-layer probe. The public `torch_npu.npu_prompt_flash_attention`
Python API currently does not expose the lower-level CANN `innerPrecise`
precision mode, so do not try to fix propagated drift by inventing an
`inner_precise` keyword unless the installed `torch_npu` signature explicitly
shows it.

If the minimal PromptFlashAttention integration still fails full-encoder
validation, run the layout sweep:

```sh
cd "$WORK_SERVER_REPO"/05_full_recognizer_optimizations
bash run_npu_vision_prompt_fa_layout_sweep.sh
```

Paste back each `VISION_PROMPT_FA_LAYER_PROBE` and its `LAYER_DIFFS`. This tests
`bnsd`, `bsnd`, and `bsh` layouts without any inline scripts. If all three
layouts still show small same-input layer differences but large propagated
drift, treat PromptFlashAttention as a lower-precision vision attention
replacement and move to a deliberate hybrid/manual fallback experiment instead
of changing random call arguments.

To profile a specific PromptFlashAttention layout after it passes validation,
set `VISION_PROMPT_FA_LAYOUT`, for example:

```sh
VISION_ATTENTION_IMPL=prompt_flash_attention VISION_PROMPT_FA_LAYOUT=bsh bash run_npu_vision_profile.sh
```

Older manual smoke order:

```sh
# 1. Small hot-swap scheduler smoke. This catches slot-swap and multi-item
# completion bookkeeping without waiting for the full 100-crop run.
python3 04_batched_fixed_cohort_decode/bench_static_compile.py \
  --schedule hotswap \
  --manifest crops/hotswap_100_manifest.json \
  --batch-size 2 \
  --num-items 4 \
  --backend torchair \
  --device npu:0 \
  --eos-mode overlap_event_flags \
  --max-new-tokens 8 \
  --step-timing both \
  --json

# 2. Fixed-cohort baseline at the target batch size. Compare this against
# hot-swap raw batch token-slots; it measures decode without row replacement.
python3 04_batched_fixed_cohort_decode/bench_static_compile.py \
  --schedule fixed_cohort \
  --manifest crops/hotswap_100_manifest.json \
  --batch-size 8 \
  --backend torchair \
  --device npu:0 \
  --eos-mode overlap_event_flags \
  --max-new-tokens 32 \
  --step-timing both \
  --json

# 3. Full hot-swap run over the prepared 100 real crops.
python3 04_batched_fixed_cohort_decode/bench_static_compile.py \
  --schedule hotswap \
  --manifest crops/hotswap_100_manifest.json \
  --batch-size 8 \
  --num-items 100 \
  --backend torchair \
  --device npu:0 \
  --eos-mode overlap_event_flags \
  --max-new-tokens 32 \
  --step-timing both \
  --json
```

Read the timing fields carefully:

- `compile_first_call` is TorchAir cache load/compile warmup; do not count it
  as steady decode. If the cache is warm, this should be much smaller than the
  original cold compile.
- `ready_bank_prefill` exists only for hot-swap and is intentionally outside
  the decode throughput comparison.
- `hotswap_validation` exists only for hot-swap and is also outside the decode
  throughput comparison. It replays every item as a single-item static-eager
  reference from the same ready-bank prefill row. The reference first clones the
  ready-bank row into contiguous B=1 cache tensors; do not validate directly on
  ready-bank row views because those are non-contiguous slices from the larger
  NPU cache bank and can behave differently from a normal single-item cache.
- Fixed-cohort `tok_per_s.compiled_raw_batch_tokens` is the main decode baseline.
- Hot-swap `tok_per_s.hotswap_raw_batch_tokens` is the direct scheduler-overhead
  comparison against the fixed baseline for backwards compatibility, but it is
  a total-window metric.
- Hot-swap `tok_per_s.hotswap_steady_raw_batch_tokens` uses only
  `phase_timing_s.steady_decode_loop_s` and is the preferred steady-state
  scheduler comparison. `hotswap_total_*` includes active-cache setup, initial
  slot loads, the steady loop, final drain, and result materialization. The
  reusable overlap token-copy buffer is allocated before the measured decode
  window and reported separately as `timing_s.hotswap_overlap_buffer_setup` /
  `phase_timing_s.external_overlap_buffer_setup_s`.
- Hot-swap `tok_per_s.hotswap_effective_item_tokens` counts only useful item
  tokens and will drop if EOS/length-cap completions create tail bubbles.
- Before interpreting any throughput, check `matches`. The script sets
  `matches.all_required_checks_passed` and exits nonzero after printing the
  summary if required generation checks fail. For hot-swap,
  `matches.hotswap_vs_single_refs.all_trimmed_match` must be true; if false,
  inspect `first_mismatches` before discussing speed.
- Hot-swap output history is written with `history_write_mode:
  cpu_rows_from_token_copy`. The decode loop copies the active next-token vector
  to CPU through the overlap path, then updates per-item CPU token rows from
  the previous step's copied token vector. This deliberately avoids NPU boolean
  advanced indexing with `active_item_indices.clamp_min(0)` and avoids scalar
  0-D NPU `copy_` into `generated_ids`; inactive slots must never be able to
  target item 0 or any other completed item. If `token_ids.invalid_count` is
  nonzero, debug output-history writes before interpreting OCR text mismatches
  as model failures.
- To isolate a hot-swap mismatch, use the experiment-4 diagnostic flags only for
  isolation, not as serving knobs: `--diagnostic-verify-swap-copies`,
  `--diagnostic-swap-copy-mode clone`, and `--diagnostic-sync-finished-flags`.
  `--backend raw_eager` removes TorchAir compile but still uses the NPU IncreFA
  and scatter-update decode path; there is no `--diagnostic-decode-attention`
  flag in experiment 4.
- `step_timing_summary.swap` versus `step_timing_summary.no_swap` shows whether
  slot replacement is expensive. Watch `npu_swap_ms`, `host_swap_s`, and
  `host_wait_prev_flag_s`. CPU timings show realized cadence and may attribute
  async backlog to the next wait point; NPU timings isolate recorded device
  regions.
- Current hot-swap slot replacement uses
  `slot_control_write_mode: batched_index_copy_for_swaps_and_slot_control`.
  Slots that finish in the same iteration are copied as one group per KV
  layer/control tensor, instead of running the full KV/control copy path once
  per finished slot.
- Current overlap token copying uses a reusable one-row pinned CPU ring buffer
  (`overlap_buffer_source: provided_ring1` in normal NPU summary reports).
  A large `phase_timing_s.overlap_buffer_setup_s` inside the decode loop would
  mean the benchmark fell back to allocating the buffer internally and should
  be treated as a setup bug.
- If hot-swap is much slower than fixed cohort, do not rely on p50 alone. Check
  `step_timing_summary.by_swap_count`, `by_finished_slot_count`, and the
  `top_*` slow-step lists. After the batched-copy change, compare
  `swap_count=1` against `swap_count=2+`; multi-slot swap cost should scale
  much less sharply if the batched path is working. Large `host_wait_prev_flag_s`
  outliers point at hidden synchronization. If
  `timing_accounting.wall_minus_host_iter_sum_s` is large, some time is being
  spent outside the instrumented loop or in final synchronization. Compare
  fixed-cohort and hot-swap `timing_accounting.npu_event_region_sums_s.decode`;
  if decode sums are similar but wall time diverges, the gap is scheduler/copy
  overhead rather than the compiled decode graph.

Run single-batch static-cache decode:

```sh
python3 03_compiled_single_batch_decode/run_local_recognition.py \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:" \
  --static
```

Probe compile compatibility:

```sh
python3 03_compiled_single_batch_decode/probe_static_compile.py \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:" \
  --backend eager
```

Benchmark compiled static decode:

```sh
python3 03_compiled_single_batch_decode/bench_static_compile.py \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:" \
  --backend torchair --device npu
```

A `fullgraph`/static compile pass is only a structural compile-compatibility
filter. It does not prove TorchAir/NPU lowering will succeed, or that NPU
throughput is good.

## Hardware Rules

Always apply the lane rules in `CLAUDE.md` before deciding what to do.

The local authoring checkout has no accelerator. That is fine for editing code, preparing crops, committing, and pushing, but not for claiming inference validation.

Ascend NPU is the only validation target. When an Ascend device is available, do not silently fall back to CPU for an NPU experiment. If the NPU path fails, inspect the environment and summarize the blocker rather than routing around it.

910B and 310P are different chips with different operator constraints. Label every result with the chip it ran on, and never carry a 910B result over to 310P or the reverse without rerunning it. The 310P-specific operator constraints established so far are recorded in the experiment sections below.

For Huawei Ascend NPU, current public PaddleOCR guidance says local direct inference is not the supported path; the official route is PaddleOCR client/pipeline plus a vLLM VLM service, or the two-container full API service. Treat direct Transformers-on-NPU work as an experiment until validated.

## Blue-Zone 910B Setup

Full verified machine facts — device inventory, model and dataset paths,
interpreter table, TorchAir cache locations — are in `CLAUDE.md`. The two things
that bite most often:

Always `source npu-setup` (at `/usr/local/bin/npu-setup`, on PATH) before running
anything. It sources CANN and ATB, sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, and
selects a free device into `ASCEND_RT_VISIBLE_DEVICES` via `npu-status
--last-free`. Non-interactive SSH does not read `~/.bashrc`, so without it
`npu-smi` fails on `libc_sec.so` and `import torch_npu` fails outright:

```sh
ssh blue_zone_npu_container 'cd /workspace/repos/paddle_ocr_vl_npu && source npu-setup && <command>'
```

The box is shared: 8 × Ascend 910B2, one process per device. Let `npu-setup` pick
the device, do not terminate other users' processes, and give concurrent runs
distinct TorchAir cache directories.

Keep bulky model caches out of Git. Run evidence under `tmp/` is the deliberate
exception; see the evidence conventions in `CLAUDE.md`.

## Git / Public Repo Hygiene

This folder is intended to become a public GitHub repo. Avoid committing:

- credentials, SSH keys, tokens, or server metadata that exposes secrets;
- model weights or Hugging Face cache directories;
- generated benchmark dumps, profiler traces, or large logs;
- private parent-repo artifacts outside this subproject.

Small reproducible scripts, notes, crop examples, manifests, and concise result summaries are fine.

Only the local authoring lane commits and pushes. The 910B container and the 310P work server pull from `origin`, run, and report.

Run evidence under `tmp/` is force-added past `.gitignore` on purpose, so that a result stays paired with the exact commit and command that produced it.

## Style

Keep notes short, concrete, and source-backed. When a result is a smoke test, call it a smoke test, and say which chip it ran on.
