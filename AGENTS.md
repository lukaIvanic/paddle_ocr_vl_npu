# AGENTS.md

## Operating Lanes

First classify where you are running from actual machine state, not from memory:

- Work/NPU lane: Ascend NPU tooling is present, such as `npu-smi` or `torch_npu`.
- Vast/CUDA lane: a rented Vast.ai GPU box, usually under `/workspace`, where `nvidia-smi` works but Ascend NPU tooling does not.
- Authoring lane: Luka's local code-editing checkout. It may have no accelerator tools at all.

The work/NPU lane is pull-only. Its job is to set up the environment, pull the repo, run scripts, inspect outputs, debug failures, and summarize exact findings. Do not edit tracked files, commit, push, or create branches from the work/NPU lane. If a code change seems necessary, report the minimal proposed change, the command that failed, and the relevant logs instead of applying it.

The Vast/CUDA lane is for dependency bring-up, CUDA smoke tests, model-loading checks, and quick debugging. It should not be confused with the work/NPU lane. CUDA results are smoke-test evidence only; they are not NPU or Ascend performance evidence. Do not commit or push from Vast unless Luka explicitly designates that specific instance as the authoring lane.

The authoring lane may edit tracked files, prepare scripts, manage crops/docs, commit, and sync with GitHub. If it has no accelerator, it should not present unrun local code as validated inference.

## Project Direction

This folder is a standalone research workspace for PaddleOCR-VL on Ascend/NPU, with a near-term focus on the `PaddleOCR-VL-1.6-0.9B` recognition VLM.

The current target is an offline full-page path built incrementally from the
local recognizer: real PP-DocLayoutV3 inference, explicit page/crop/request
boundaries, and a persistent PaddleOCR-VL engine. Experiment 08 uses sequential
B=1 vision and text prefill with static TorchAir buckets, followed by a
persistent compiled decode arena. Finished requests are hot-swapped between
decode iterations by copying the next ready request's valid KV prefix into the
freed slot. It does not yet overlap prefill with decode. The standalone
`run_offline_e2e.py` page assembler remains diagnostic; the faithful full-page
path uses official PaddleX v1.6 assembly and replaces only its inner recognition
model through `run_omnidocbench_paddlex.py`. The recognition model is available as a
Transformers/PyTorch model at:

```text
PaddlePaddle/PaddleOCR-VL-1.6
```

It loads through `AutoProcessor` and `AutoModelForImageTextToText` / `PaddleOCRVLForConditionalGeneration`. Architecturally, it is a native-resolution vision encoder plus adaptive MLP projector plus ERNIE-4.5-0.3B decoder-only multimodal LM. Visual embeddings replace `<image>` token embeddings before decoder inference; there is no encoder-decoder cross-attention block.

Keep the distinction clear:

- `PaddleOCR-VL-1.6-0.9B` is the VLM recognition component.
- Full `PaddleOCR-VL-1.6` page parsing is layout analysis plus recognition plus merge/postprocess.
- For v1.6, the full PaddleOCR/PaddleX pipeline uses `PP-DocLayoutV3` plus `PaddleOCR-VL-1.6-0.9B`.
- Recognizer-only runs are valid for element-level crops and prompts such as `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`, `Spotting:`, and `Seal Recognition:`.
- Recognizer-only runs are not proof of full page-parser quality or throughput.

Known implementation surfaces:

- Official PaddleOCR/PaddleX provides the full parser pipeline and Paddle-facing configs.
- Hugging Face Transformers provides the core recognizer/VLM directly as `PaddlePaddle/PaddleOCR-VL-1.6`.
- Huawei Ascend public guidance currently points to PaddleOCR client/pipeline plus vLLM VLM service, or a two-container full API service, rather than direct local NPU inference.

## Current Local Artifacts

This folder currently contains:

- `crops/`: eight OmniDocBench region crops, not full pages.
- `crops/manifest.json`: source image, category, bbox, suggested prompt, and ground truth for each crop.
- `crops/create_omnidocbench_recognition_crops.py`: reproducible crop generator.
- `crops/create_omnidocbench_hotswap_crops.py`: reproducible generator for the larger queue/hot-swap crop set.
- `crops/hotswap_*.png`, `crops/hotswap_100_manifest.json`, `crops/hotswap_100_summary.json`, and `crops/hotswap_100_contact_sheet.jpg`: 100 additional OmniDocBench region crops for batch sizing and future vLLM-style slot hot-swapping tests. Use the hot-swap manifest explicitly; the default tiny smoke manifest remains `crops/manifest.json`.
- `01_transformers_recognition_baseline/run_transformers_recognition.py`: minimal Transformers recognizer smoke script. It uses the slow image processor by default so it matches the source-backed local preprocessing path; pass `--use-fast` only when deliberately comparing to the fast HF image processor.
- `02_local_eager_recognition/config.py`: dependency-free dataclass mirror of the PaddleOCR-VL config fields needed for inference.
- `02_local_eager_recognition/local_modeling_paddleocr_vl.py`: local PyTorch implementation of the recognition VLM with no Transformers imports.
- `02_local_eager_recognition/run_local_recognition.py`: local recognizer runner using `tokenizers`, local image preprocessing, and the local model.
- `03_compiled_single_batch_decode/`: single-batch decode optimization lane, copied from the local eager implementation and extended with static KV cache decode, TorchAir cache compile, profiler hooks, and flat `torch.compile(fullgraph=True, dynamic=False)` probes.
- `04_batched_fixed_cohort_decode/`: batch-decode scheduler lane. It keeps prefill sequential and padding-free for real crops, benchmarks true fixed-cohort batched static decode, and can also prefill an NPU-resident ready KV bank then hot-swap finished items into fixed compiled decode slots.
- `08_offline_e2e_b1/`: first persistent offline system. It can use either its
  compact diagnostic page frontend or official PaddleX v1.6 page assembly. It
  sends prepared crops through sequential bucketed compiled vision/text prefill
  and one continuous compiled decode schedule. Stable slots are refilled in
  place without compacting active rows.
  It reports raw slots, effective tokens, idle slots, completion look-ahead,
  active-slot utilization, KV-copy traffic, and setup/page/request/device timing.
  Read its README before interpreting throughput or output parity.
- `refs/`: small architecture reference artifacts.
- `refs/PaddleOCR`: ignored sparse reference checkout of the official PaddleOCR repo.

Keep `refs/PaddleOCR/` ignored. It is reference material, not project source.

## Local Smoke Commands

Regenerate the crops from the parent repo's restored OmniDocBench copy:

```sh
python3 crops/create_omnidocbench_recognition_crops.py
python3 crops/create_omnidocbench_hotswap_crops.py
```

Run the core recognition model on one crop with Transformers:

```sh
python3 01_transformers_recognition_baseline/run_transformers_recognition.py
```

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
override for CUDA parity checks, but `fp32` is intentionally not a supported
run mode.

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

Experiment 4 can also run a CUDA debug version of hot-swap with
`--backend raw_eager --device cuda`. In this repo, `--backend eager` means
`torch.compile(..., backend="eager", fullgraph=True, dynamic=False)`;
`raw_eager` is true uncompiled Python/PyTorch execution. Use `raw_eager` for
CUDA hot-swap debugging because the non-NPU static-cache update contains
host-side per-row indexing that should not be sent through Dynamo. CUDA uses
manual PyTorch static-decode attention and a synchronous token-consume path
instead of NPU IncreFA, TorchAir, and the NPU overlap copy stream. Treat CUDA
hot-swap as a scheduler and `generated_ids` bookkeeping check only; it is not
an Ascend throughput result and does not validate NPU-specific
IncreFA/scatter behavior.

Recommended NPU run order for experiment 4:

For the current hot-swap bottleneck investigation, prefer the committed matrix
runner instead of ad hoc shell snippets:

```sh
cd /home/lukaiv/paddle_ocr_vl_npu/04_batched_fixed_cohort_decode
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
cd /home/lukaiv/paddle_ocr_vl_npu/05_full_recognizer_optimizations
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
and the exact NPU-vs-CUDA decode differences. Do not reverse engineer this
script by writing helper snippets unless the committed runner fails to print the
needed field.

```sh
cd /home/lukaiv/paddle_ocr_vl_npu/05_full_recognizer_optimizations
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

On the Vast/CUDA lane, use the CUDA runner for a smoke/algorithmic benchmark.
This is not an NPU throughput result. The CUDA runner defaults to `NUM_ITEMS=8`,
`DEVICE=cuda:0`, `DECODE_SCHEDULE=hotswap`, and `DECODE_BACKEND=raw_eager`
because CUDA uses the manual attention/per-row-KV fallback instead of TorchAir,
IncreFA, and NPU `scatter_update_`.

```sh
cd /workspace
if [ ! -d /workspace/paddle_ocr_vl_npu_queue_cuda/.git ]; then
  git clone https://github.com/lukaIvanic/paddle_ocr_vl_npu.git /workspace/paddle_ocr_vl_npu_queue_cuda
fi
cd /workspace/paddle_ocr_vl_npu_queue_cuda
git pull --ff-only origin main
cd /workspace/paddle_ocr_vl_npu_queue_cuda/05_full_recognizer_optimizations
PYTHON_BIN=/workspace/venvs/paddle_ocr_vl/bin/python \
bash run_cuda_recognizer_queue_benchmark.sh
```

For a larger CUDA-only smoke run, override only the item count and cache length
if needed:

```sh
NUM_ITEMS=100 CACHE_LENGTH=1536 \
PYTHON_BIN=/workspace/venvs/paddle_ocr_vl/bin/python \
bash run_cuda_recognizer_queue_benchmark.sh
```

To sanity-check CUDA higher hot-swap active batches:

```sh
ACTIVE_BATCH_SIZE=4 NUM_ITEMS=8 CACHE_LENGTH=1024 \
PYTHON_BIN=/workspace/venvs/paddle_ocr_vl/bin/python \
bash run_cuda_recognizer_queue_benchmark.sh

ACTIVE_BATCH_SIZE=8 NUM_ITEMS=8 CACHE_LENGTH=1024 \
PYTHON_BIN=/workspace/venvs/paddle_ocr_vl/bin/python \
bash run_cuda_recognizer_queue_benchmark.sh
```

On the current Vast box, keep the checkout clean by using a fresh clone such as
`/workspace/paddle_ocr_vl_npu_queue_cuda` if an older checkout is dirty. The
runner defaults to `MODEL=PaddlePaddle/PaddleOCR-VL-1.6`, sets
`HF_HOME=/workspace/.hf_home` when that cache exists, and disables Xet so it can
reuse the already-downloaded Hugging Face snapshot.

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
cd /home/lukaiv/paddle_ocr_vl_npu/05_full_recognizer_optimizations
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
cd /home/lukaiv/paddle_ocr_vl_npu/05_full_recognizer_optimizations
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
cd /home/lukaiv/paddle_ocr_vl_npu/05_full_recognizer_optimizations
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
- If CUDA/Vast raw eager passes hot-swap but NPU still mismatches, use the
  experiment-4 diagnostic flags only for isolation, not as serving knobs:
  `--diagnostic-verify-swap-copies`, `--diagnostic-swap-copy-mode clone`, and
  `--diagnostic-sync-finished-flags`. On NPU, `--backend raw_eager` removes
  TorchAir compile but still uses the NPU IncreFA and scatter-update decode
  path; there is no `--diagnostic-decode-attention` flag in experiment 4. Use
  CUDA raw eager as the manual-attention scheduler/bookkeeping comparison, not
  as an NPU attention validation.
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

On the Vast/CUDA smoke box on 2026-06-08, the local model matched Transformers
eager bf16 exactly for next-token logits on all eight crops when using the slow
HF/source-matched processor path. The local processor intentionally follows the
slow PaddleOCR-VL image processor source; the HF fast processor has small resize
rounding differences and is not the exact parity target.

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
  --backend inductor
```

Use `--backend inductor` on CUDA for stronger local codegen smoke. Use
`--backend torchair --device npu` on the work/NPU lane for the real Ascend
check. CUDA fullgraph/static passing is only a
structural compile-compatibility filter; it does not prove TorchAir/NPU lowering
will succeed or that NPU throughput is good.

Vast/CUDA smoke on 2026-06-08 for `crop_01`, 32 generated tokens, 31 measured
decode steps, bf16:

- `backend=eager`: all output IDs matched; static-eager-vs-compiled decode logits
  matched exactly; compiled decode measured about 56 tok/s.
- `backend=inductor`: all output IDs matched; static-eager-vs-compiled decode
  logits differed numerically (`max_abs` about 0.226, `mean_abs` about 0.0338)
  but argmax output matched; compiled decode measured about 139 tok/s after a
  roughly 24.6s first compile call.

## Hardware Rules

Always apply the lane rules at the top of this file before deciding what to do.

The authoring checkout may have no accelerator attached. That is fine for editing code, preparing crops, committing, and pushing, but not for claiming inference validation.

The work/NPU lane is the real validation lane. When an Ascend target is available, do not silently fall back to CPU or CUDA for NPU experiments. If the NPU path fails, inspect the environment and summarize the blocker. Do not patch tracked files from the work/NPU lane.

The Vast/CUDA lane is only for dependency bring-up, model-load smoke tests, and quick debugging. Do not describe Vast results as NPU or Ascend throughput. If a result comes from CUDA, label it as CUDA/Vast.

For Huawei Ascend NPU, current public PaddleOCR guidance says local direct inference is not the supported path; the official route is PaddleOCR client/pipeline plus a vLLM VLM service, or the two-container full API service. Treat direct Transformers-on-NPU work as an experiment until validated.

## Vast/CUDA Notes

This section applies only in the Vast/CUDA lane. Work/NPU agents should not treat these commands as their setup instructions.

The Vast/CUDA lane is useful for checking whether the Transformers recognizer loads, preprocesses, and generates on a crop before sending scripts to the NPU lane.

Keep bulky model caches and generated outputs out of Git.

Known-good CUDA smoke setup on the Vast RTX 3060 instance `40080612`:

```sh
python3 -m venv /workspace/venvs/paddle_ocr_vl
/workspace/venvs/paddle_ocr_vl/bin/python -m pip install -U pip setuptools wheel
/workspace/venvs/paddle_ocr_vl/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  transformers==5.0.0 accelerate==1.13.0 safetensors==0.7.0 \
  sentencepiece==0.2.1 protobuf==7.35.0 tiktoken==0.13.0 \
  einops==0.8.2 opencv-python==4.13.0.92 pillow==12.2.0
```

Run the recognizer with Xet disabled on that box. The default Xet downloader
stalled during the first model-weight download, while normal Hub HTTP completed:

```sh
HF_HOME=/workspace/.hf_home \
HF_HUB_DISABLE_XET=1 \
HF_XET_DISABLE=1 \
/workspace/venvs/paddle_ocr_vl/bin/python 01_transformers_recognition_baseline/run_transformers_recognition.py \
  --crop crops/crop_01_text_block_en.png \
  --max-new-tokens 96
```

On 2026-06-08, `transformers==5.10.2` failed with `torch==2.6.0+cu124`
because it expected `torch.float8_e8m0fnu`. Keep the CUDA smoke environment on
`transformers==5.0.0` unless PyTorch is upgraded deliberately.

## Git / Public Repo Hygiene

This folder is intended to become a public GitHub repo. Avoid committing:

- credentials, SSH keys, tokens, Vast instance metadata that exposes secrets;
- model weights or Hugging Face cache directories;
- generated benchmark dumps, profiler traces, or large logs;
- private parent-repo artifacts outside this subproject.

Small reproducible scripts, notes, crop examples, manifests, and concise result summaries are fine.

Only the authoring lane should commit and push. The work/NPU lane should only pull from `origin`, run, and report. The Vast/CUDA lane should normally run and report too, unless Luka explicitly asks to use it for authoring.

## Style

Keep notes short, concrete, and source-backed. Prefer small scripts that can be run on local CPU/CUDA first and then moved to Ascend. When a result is a smoke test, call it a smoke test.
