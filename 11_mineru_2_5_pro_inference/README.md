# Experiment 11: MinerU2.5-Pro Local Inference

**Current execution task (2026-10-10): model-default pixel accuracy on 910B.**
`run_model_default_pixels_validation.py` reproduces the saved production command
with only min/max pixels 50176/1605632, no text-only cap, extended vision/text
buckets, and KV8192. It runs warmup64, full1651, then frozen evaluation with
`CDM_WORKERS=96`. Results are pending. No 310P work is part of this task.
`analyze_model_default_pixels.py` checks exact saved layouts, output lengths,
per-page accuracy changes and logged CDM throughput against production.

**Previous execution task: opt-in prefill buckets and window scheduling.**
The implementation is based on `2ef68bdd`. 910B validation completed at
`0d6cef73`: the defaults match all 1,020 parent requests exactly; the production
set completed all 1,651 pages and the 96-worker OmniDocBench evaluation.
Machine-readable receipts, summaries, comparisons and timing samples are in
[the validation evidence](references/prefill_buckets_910b_20261010/).
Later Python changes affect tests and validation reporting/warmup only.
310P performance remains unmeasured; run the candidate first-256 handoff below for that result.
The single active [310P handoff](WORK_SERVER_310P_PREFILL_BUCKETS_FIRST256.md)
runs candidate warmup64 then candidate256 and compares against the existing
`prefill_timing_first256_310p_20261010T092230Z`; it includes environment discovery.
The first-256 timing task is superseded; its measurements motivated this work.

Add this complete candidate flag set to the existing C1+C2+C5 invocation:

```sh
--processor-max-pixels 602112 --processor-text-max-pixels 401408 \
--local-vision-buckets 384,512,768,896,1024,1280,1536,1792,1920,2048,2560,3072 \
--local-text-buckets 128,256,384,512,576,832,1024 \
--local-text-pack-target 384 --local-text-prefill-schedule window
```

`prefill_buckets_config.PRODUCTION_PREFILL_OPTIONS` contains the candidate
additions. Keep min pixels 25088, vision packing 768, lookahead 32, and text
max-members 32. C1+C2 uses CPU grids and pinned nonblocking copies; C5 disables
metrics in ordinary production. For the requested candidate256 measurement explicitly
turn `--local-prefill-metrics` on. The historical baseline is not rerun.

All new defaults retain admission prefill, packing to the largest text bucket,
and the global crop cap. `run_page_pipeline.py` defaults are unchanged. The
text-only cap requires continuous streaming so that actual layout labels reach
the processor; titles, headers and every label other than exactly `text` keep
the global limit. An independent copy of the image processor uses the same
pixel-limit setter and the same minimum as the all-crop ablation.

Window mode runs existing packed text prefill after each encoded vision window,
stages exact prompt-length KV prefixes, and uses the experiment-20 ready lease
admission path. Packing is unchanged best-fit decreasing, with a separate
target; prompts over that target run alone in their smallest fitting bucket.
The first token, cache position and RoPE delta travel with each lease. One
batched first-token CPU read fences each window, also resolving its text timing
events. Window-owned KV tensors are allocated, copied and released on the same
owner stream, so admission needs no additional host fence. Experiment-20 shared
arenas retain their existing fence before rows can be reused. Staging is bounded
by one lookahead window and stores only real prompt lengths (12288 bytes/token
at model FP16). Decode KV-length handling and vision packing are unchanged.

Use `--local-warm-all-prefill-buckets` only in a separate cache-population run.
It captures real-page graph inputs and uses the existing resize/replay warmup
machinery for every configured prefill bucket, including routes absent from the
prefix. Its summary is marked `warmup_only`; its timing is not throughput.
Run the measured lane without that flag and audit cache artifacts before/after.

**Previous task (2026-10-09): host overhead only.** The retired
[full-page pair handoff](WORK_SERVER_310P_LIVE_PADDLE_CAP3072_FULL1651.md) compared
baseline with the best measured host configuration, both at approximate vision
precision and the 602,112-pixel cap. Opt-in controls cover CPU vision grids,
pinned transfers, a dedicated H2D thread, exact uint8 staging, prefill metrics,
and preparation/frontend worker counts. Production defaults stay unchanged.
No further profiling/GIL work is part of this task. Earlier vision/kernel
handoffs below are background, not additional steps for this comparison.

**Earlier vision-encoding investigation (background).** See
[VISION_310P_TARGET.md](VISION_310P_TARGET.md) and the self-contained
[vision crop handoff](VISION_CROP_310P_HANDOFF.md). Decode KV-cache benchmarks below
are separate work and do not explain the vision bottleneck.
The [910B real-crop vision results](references/vision_crop_contracts_910b_20261007/RESULTS.md)
measure all 32 blocks, including attention and ND/NZ projection comparisons.
The revised diagnostic handoff targets the unexplained 310P matmul/attention
gap first: kernel tiling/core counts, waits, formats and device state. It adds
an explicitly separate synthetic matmul calibration, exploratory grouped
projections, production-config matching and immutable per-lane telemetry.
The [portable reanalysis](references/vision_crop_contracts_910b_20261007/GENERIC_REANALYSIS.md)
checks all 12 committed 910B lanes without chip-specific kernel assertions.
[New 910B tool validation](references/vision_diagnostics_910b_20261007/RESULTS.md)
retains the 48-case calibration, internal-format checks, eight real-crop vision
lanes and per-launch telemetry, separately from the original results.
The [24-lane matched ND/NZ rerun](references/vision_nz_pairs_910b_20261007/RESULTS.md)
checks the actual kernel weight inputs across D80/D128 PromptFA, eager unpad
and both grouped projection modes. Ordinary linears consume NZ; the existing
grouped helper converts its inputs back to ND, so those results are explicitly
mixed-format. Compiled full-vision runs show no NZ speedup on 910B.

The [KV-cache layout probe](kv_cache_probe/README.md) audits the vLLM-Ascend
310P cache contract and preserves isolated eager attention compatibility checks.
It is not a full-model throughput benchmark.

`bench_full_model_kv.py` / `run_full_model_kv.sh` implement real-crop,
EOS-aware generation with complete static TorchAir decoder graphs. Real
vision/text prefill supplies the KV, and useful post-prefill tokens/s includes
the full 24-layer decoder, cache writes, LM head, sampling and host completion
handling. See the [measurement contract](FULL_MODEL_KV_BENCHMARK.md),
[recorded 910B2 results](references/full_model_kv_910B_20261007/RESULTS.md),
and [self-contained 310P handoff](FULL_MODEL_KV_310P_HANDOFF.md). Candidate
timings and token drift are reported separately; no isolated attention result
is treated as full-model performance.

This experiment is the current custom MinerU2.5-Pro implementation transferred
from the standalone `mineru_25_pro_npu` repository at commit `b08ae14`. The
custom implementation does not import vLLM or vLLM-Ascend. Separate official
baseline lanes use stock Transformers or stock vLLM.

The model itself is implemented locally in PyTorch. Transformers remains only
at the processor boundary through `AutoProcessor`, for tokenization and
Qwen2-VL image preprocessing.

Default checkpoint:

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
```

Pass a local checkpoint directory to every runner; these scripts do not
download model weights.

## Persistent page serving

The `local-continuous-client` page runner now defaults to bounded streaming.
Layout and crop requests share one persistent B32 decode scheduler, and each
page is written after its own last crop completes. `PageInbox` supports live
arrivals and explicit input closure. CPU preparation and page buffering are
bounded; model kernels, prompts and crop/postprocessing policy remain unchanged.

See [SERVING_REFACTOR.md](SERVING_REFACTOR.md) for the API, validation and timing
contract. The checksum-protected pre-refactor 384-page token reference is in
`references/serving_anchor_384_13061fc4/`. Use `--token-trace` for token-level
comparison. Use `--no-streaming-pages` explicitly to retain the older stepping
orchestration described in the historical sections below.

On the 910B validation host, `MODE=streaming LIMIT=384 bash
11_mineru_2_5_pro_inference/run_serving_validation.sh` uses the pinned B32/KV4096
settings and existing graph caches. This is not an HTTP server; the page inbox
and callback API is the serving boundary.

The [full 1,651-page v1.6 result](references/serving_streaming_1651_ae4c947c/RESULTS.md)
on one 910B2 is 95.1131 overall accuracy and 0.81331 hot pages/s, with zero
failed pages and 99.77% decode-slot occupancy. This keeps KV4096; 39 requests
hit that capacity, and their unchanged predictions remain in the score.
The result directory preserves full token traces and evaluation evidence.

## Image-resolution overrides

The locally owned processor accepts `--processor-min-pixels` and
`--processor-max-pixels`. For a recognition-resolution ablation, add:

```sh
--processor-min-pixels 25088 --processor-max-pixels 1103872
```

The maximum is pixel **area**, not edge length: `1103872 / (14 * 14) = 5632`
raw vision tokens, or at most 1408 image tokens after the 2×2 merger. Resize
alignment can produce fewer tokens. This shared processor also handles layout,
but the standard 1036×1036 layout image (5476 raw tokens) remains below this cap.
The production minimum is 25088 pixels (128 raw tokens); the checkpoint minimum
is 50176. Omitting the maximum override preserves the checkpoint's 1605632-pixel
limit (8192 raw tokens). Effective limits are recorded in the run summary.

`run_vision_timing_production.py` also accepts `--processor-max-pixels 1103872`,
keeping the reference run's remaining settings and graph caches. The override
changes recognition resolution and needs empirical quality evaluation; it is
not a new default or a claim of unchanged accuracy.

## Included surfaces

```text
config.py
  Local MinerU/Qwen2-VL configuration dataclasses and config loader.

local_modeling_mineru.py
  Conv3D patch embedding, 32-layer vision transformer, 2x2 patch merger,
  24-layer text decoder, eager generation, static KV-cache decode,
  TorchAir-compatible flat decode module, optional FRACTAL_NZ decoder weights,
  and optional native NPU rotary decode.

run_local_model_two_step_extract.py
  Inspectable two-stage page flow: eager layout detection, block parsing and
  cropping, then eager or static-cache compiled recognition decode.

bench_compiled_batch_decode.py
  Fixed-step compiled decode throughput over distinct real crop inputs.
  Sequential crop preprocessing and prefill are intentionally outside the
  measured decode interval.

parse_npu_profile.py
  Parser for torch-npu profiler output from the compiled decode path.

run_official_transformers_omnidocbench.py
  Fidelity-first corpus runner around the official mineru-vl-utils two-step
  page client. It supports stock Transformers plus synchronous and asynchronous
  vLLM engines, plus two local-model lanes. ``local-correctness`` presents the
  local model through the official Transformers client contract;
  ``local-eager-client`` is the first direct eager custom VLM client while the
  official MinerU frontend and post-processing stay unchanged;
  ``local-compiled-client`` keeps prefill eager and replaces only B1 decode
  with the existing TorchAir static-KV graph; ``local-fixed-batch-client``
  prefills requests independently into request-owned rows of a shared KV arena
  and runs full groups through a fixed compiled decode batch, with B1 tails;
  ``local-continuous-client`` uses the same request-owned arena and compiled
  graph, but immediately refills a completed slot from the pending request
  stream. It intentionally retains the fixed path's synchronous host-visible
  completion check so refill can be measured independently.
  It writes official json2md Markdown, content lists, per-page checkpoints,
  shard progress, and timing.
```

The experiment reuses the repository-level `crops/` corpus. Its manifest and
images were verified byte-identical to the standalone MinerU repository, so no
duplicate assets are stored here.

## Architecture

MinerU2.5-Pro is a compact Qwen2-VL-style VLM:

```text
vision tower: 32 layers, hidden 1280, 16 heads, head dim 80
patch embed:  Conv3D, temporal patch 2, spatial patch 14
patch merger: 4 * 1280 -> 5120 -> 896
text decoder: 24 layers, hidden 896, intermediate 4864
attention:    14 query heads, 2 KV heads, head dim 64
vocabulary:   151936 tokens, tied embedding/LM-head weight
```

## Blue-zone setup

Use the project checkout and standard NPU setup:

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup

PYTHON=/usr/local/python3.12.13/bin/python3
MODEL_DIR=/workspace/models/MinerU2.5-Pro-2605-1.2B
```

Verify `MODEL_DIR` against the actual installed model path before running.

The official Transformers baseline uses a separate environment so its pinned
Transformers 4.x dependency does not disturb vLLM-Ascend's Transformers 5.x
environment:

```sh
bash 11_mineru_2_5_pro_inference/setup_official_transformers_env.sh
OFFICIAL_PYTHON=/workspace/venvs/mineru_pro_transformers_py312/bin/python
```

The official vLLM lane uses the installed vLLM-Ascend 0.21 stack and a small
environment that adds MinerU utilities without changing its dependencies:

```sh
bash 11_mineru_2_5_pro_inference/setup_official_vllm_env.sh
VLLM_PYTHON=/workspace/venvs/mineru_pro_vllm_py312/bin/python
```

The validated version contract is vLLM 0.21.0, vLLM-Ascend 0.21.0rc1,
Transformers 5.5.4, Torch 2.10.0, torch-npu 2.10.0, and CANN/NNAL 9.0.0.

The local vLLM lane applies two required MinerU checkpoint compatibility
corrections.  It forces the tied input-embedding/LM-head contract because the
checkpoint stores only `model.embed_tokens.weight`, and it bypasses
`mineru-vl-utils` 1.0.5's extra `LLM.renderer.render_cmpl` pass.  vLLM 0.21
accepts the original `prompt` plus `multi_modal_data` request directly; the
extra renderer pass preserves text IDs but corrupts the multimodal payload.

## Official Transformers OmniDocBench lane

The fidelity baseline uses FP16 stock
Transformers, the fast processor, eager attention, official
`MinerUClient(backend="transformers")`, `image_analysis=False`, greedy
generation, and official `json2md`. The default client batch size is one.

Run a small prefix:

```sh
$OFFICIAL_PYTHON \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  --model "$MODEL_DIR" \
  --output-dir tmp/11_mineru_2_5_pro_inference/official_transformers_n8 \
  --limit 8 \
  --batch-size 1
```

Large accuracy runs may shard pages across independent NPUs without changing
the per-page B1 inference contract. Every shard receives the same global
`--offset`/`--limit`, plus a common `--shard-count` and distinct
`--shard-index`. Shards safely share one output directory because image names
are unique. `--resume` is enabled by default: a page is skipped only after its
Markdown, content list, and page record all exist.

The evaluator consumes the generated `predictions/` directory. Use the pinned
Experiment-09 OmniDocBench evaluator wrapper after every shard completes; do
not score partial output as a full-corpus result.

Corpus runs warm the selected backend with the first two pages of each shard
before measurement. Warmup outputs are discarded, the same pages are processed
again as part of the measured shard, and generation plus vision/text routing
counters are reset before the pipeline timer starts. Use `--warmup-pages 0` to
measure a cold path or another positive value to change the warmup prefix. The
run manifest records the warmup pages and wall time. For the local compiled
lane, the runner also captures real static-graph tensors from those pages and
replays resized forms through every configured vision and packed-text bucket.
This loads all existing graph shapes without introducing synthetic model
inputs, new buckets, or new cache keys. The manifest records whether each
bucket ran directly on a page or through real-page tensor replay.

## Local continuous compiled-decode lane

This lane leaves the official MinerU page frontend, processor, two-step
protocol, and post-processing in place. Recognition requests are prepared at
B1 and prefilled directly into request-owned rows of a shared static KV arena.
The static B8 decode graph then runs until a request ends; that slot is
prefilled with the next pending request before the next decode iteration.
Draining rows use pad tokens only after the pending stream is exhausted.

```sh
$VLLM_PYTHON \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  --backend local-continuous-client \
  --model "$MODEL_DIR" \
  --output-dir tmp/11_mineru_2_5_pro_inference/local_continuous_b8_k4096_n8 \
  --limit 8 \
  --page-batch-size 8 \
  --batch-size 8 \
  --local-prepare-prefetch-depth 16 \
  --local-compiled-cache-length 4096 \
  --local-torchair-cache-dir \
    .runtime_cache/11_mineru_2_5_pro_inference/native_fixed_b8_k4096_fp16
```

CPU processor work runs one request at a time on a bounded background producer;
`--local-prepare-prefetch-depth` controls its queue. NPU transfer and prefill
remain on the inference thread. This keeps host preparation off the decode
critical path without changing the synchronous decode-completion contract.
The producer also constructs MRoPE position IDs and deltas from the CPU token,
mask, and image-grid tensors. The request moves those exact tensors to the NPU
with the other processor outputs, so text prefill does not run Python-driven
MRoPE preparation on the NPU critical path.

`--page-batch-size` bounds the request stream by default: all recognition
requests produced by that page group can refill one another, but requests from
the next page group are admitted only after the current group is written.
`--global-request-stream` removes those drains for the local continuous backend.
It runs layout for every pending page first, then sends all resulting crops
through one recognition request stream. The runner still writes every page
after the shared group completes.

The validated B8/KV4096 configuration used a 16-request CPU preparation queue.
On the first 32 OmniDocBench pages, one 32-page request stream completed in
297.435 seconds (0.10759 pages/s), with 88.38% active decode slots and 531.79
effective decode tok/s. CPU preparation performed 51.28 seconds of work, but
slot admission waited only 0.55 seconds for it. The corresponding unprefetched
cross-page run took 375.790 seconds; the earlier four-group continuous run took
335.984 seconds. All 32 content lists and Markdown files were byte-identical
across the three scheduling arrangements. The compact run summary is under
`tmp/11_mineru_2_5_pro_inference/native_continuous_prefetch16_b8_n32_pg32_k4096_20260805/`.

Local MinerU backends accept `--local-vision-attention manual` and
`--local-vision-attention prompt_flash_attention`. The PromptFA lane is eager:
it uses one unmasked, full-attention `BNSD` operator call for each image segment
defined by the existing vision `cu_seqlens`. MinerU's vision head dimension is
80, so this path does not pad or slice attention heads. Local backends use
`--local-dtype float16`; this is the common PromptFA dtype supported by both
910B and 310P. The standard experiment paths do not expose a BF16 mode.

The local backends also accept `--local-vision-backend torchair` plus
`--local-vision-buckets`. This is a B=1 static bucket path. Patch embedding and
position construction remain eager at the real image shape. The 32 vision
transformer blocks receive padded hidden states, rotary factors, and a bool
mask that isolates real rows from padding. Real rows are sliced before the
unchanged 2x2 patch merger. Sequences above the largest configured bucket use
the same eager unpadded blocks instead of being truncated or resized.

The compiled 32-block vision path always expands both encoder LayerNorms per
block into explicit FP32 mean, variance, normalization, scale, and bias math.
It then converts the normalized result back to the model dtype before the
unchanged `nn.Linear` projections. This is the production default on every
vision bucket and on both 910B and 310P. It avoids the 310P
`MatmulLayerNormReduce` compilation failure. The stock `nn.LayerNorm` form
remains available only through the isolated diagnostic runner; it is not a
production setting. Native D80 PromptFA and ordinary `nn.Linear` remain the
default attention and projection contracts.

The first 910B validation at commit `9511b2e` used FP16 PromptFA. A fixed layout
request routed 5,476 real tokens to the 5,632 bucket: full vision time changed
from 178.1 ms eager to 153.6 ms compiled (1.16x, 35.6K effective tok/s). A real
recognition crop routed 960 tokens to the 1,024 bucket: 144.2 ms to 39.0 ms
(3.69x, 24.6K effective tok/s). Both cases had exact first-64-token generation
parity. Final image-embedding cosine similarity was at least 0.999998, mean
absolute error was at most 0.0506, and neither result contained nonfinite
values. Cold first calls were about 87-91 seconds and are excluded from warm
timing.

Use the isolated comparison before adding more graph shapes:

```sh
$PYTHON 11_mineru_2_5_pro_inference/bench_compiled_vision_prefill.py \
  --model "$MODEL_DIR" \
  --image crops/crop_01_text_block_en.png \
  --prompt "Text Recognition:" \
  --bucket 1024 \
  --cache-dir .runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16 \
  --output tmp/11_mineru_2_5_pro_inference/compiled_vision_crop/result.json
```

For a 310P compiler failure at the full 32-block vision boundary, use the
committed contract matrix instead of reducing PromptFA to a synthetic operator
probe:

```sh
BUCKET=1024 bash 11_mineru_2_5_pro_inference/run_vision_prefill_contract_matrix.sh
```

Every lane runs in a separate Python process and receives its own TorchAir cache
identity. The matrix holds the real image preparation, RoPE, 32 vision blocks,
padding mask, output projection, and MLPs constant while testing:

- stock `nn.LayerNorm` versus explicit FP32 LayerNorm math;
- stock `nn.Linear` versus `npu_grouped_matmul` with a 3D weight for QKV, and
  optionally for the LayerNorm-fed MLP FC1;
- native D80 PromptFA versus a D80-to-D96 pad/call/slice compiler boundary;
- PromptFA versus a full-graph manual-attention control;
- the combined historical Paddle 310P workarounds.

`MINERU_VISION_COMPILE` markers identify cache-compile and first-call
start/finish boundaries. A timeout or failure in one lane is recorded under its
own output directory and does not suppress the remaining discriminator lanes.
The default eager reference for this matrix is manual attention. Interpret
feature differences and token parity before treating a terminating lane as a
usable replacement.

For corpus-level vision work, use `vision_prefill_lab.py`. It replays real
OmniDocBench pages, keeps CPU image preparation and H2D outside the headline
vision throughput, records patch/position/transformer/merger device times, and
can compare eager against bucketed TorchAir features. The standard MinerU layout
scenario forces each real page to 1036x1036, which produces 5,476 real tower
tokens and routes to the 5,632-token compiled bucket:

```sh
$PYTHON 11_mineru_2_5_pro_inference/vision_prefill_lab.py \
  --model "$MODEL_DIR" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --offset 0 --limit 8 \
  --layout-size 1036 1036 \
  --execution eager,torchair \
  --buckets 384,512,768,1024,1536,2048,3072,4224,5632 \
  --warmup-pages 2 --repeats 3 --parity-pages 2 \
  --cache-dir .runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16 \
  --output tmp/11_mineru_2_5_pro_inference/vision_prefill_lab/layout_1036/result.json
```

Use `effective_tok_s` for real model work and `physical_tok_s` for the padded
compiled workload. `pages_per_s` is vision-only throughput. It excludes model
load, image decode/resize, processor work, H2D, layout text generation, and all
recognition crops.

Profile the same full 1036x1036 compiled vision boundary with
`torch_npu.profiler` as follows. Each AI Core metric family gets an independent
capture. The runner measures unprofiled calls before and after each capture so
profiler distortion is explicit, then parses the emitted CANN CSV files:

```sh
$PYTHON 11_mineru_2_5_pro_inference/profile_vision_prefill_lab.py \
  --model "$MODEL_DIR" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --offset 0 --layout-size 1036 1036 \
  --execution torchair --metrics pipe,memory \
  --warmup-steps 3 --baseline-steps 10 --profile-steps 3 \
  --cache-dir .runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16 \
  --profile-root .runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles/layout_1036 \
  --output-dir tmp/11_mineru_2_5_pro_inference/vision_prefill_profile/layout_1036
```

The profile includes patch embedding, position preparation, all transformer
blocks, and the patch merger. It excludes setup, CPU preparation, H2D, and the
TorchAir cache restore/compile call. Heavy profiler output stays under
`.runtime_cache`; only the parsed summaries and compact run result belong in
`tmp/`.

## Official synchronous vLLM lane

Start with eager vLLM to separate compatibility from graph compilation:

```sh
$VLLM_PYTHON \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  --backend vllm-engine \
  --model "$MODEL_DIR" \
  --output-dir tmp/11_mineru_2_5_pro_inference/official_vllm_eager_n1 \
  --limit 1 \
  --batch-size 0 \
  --page-batch-size 1 \
  --vllm-enforce-eager \
  --vllm-max-model-len 8192 \
  --vllm-max-num-seqs 64
```

`--batch-size 0` keeps the official synchronous vLLM behavior: every request
prepared by the current layout/page group is submitted to one vLLM generate
call. The vLLM scheduler then performs continuous batching within that call.

Use `--vllm-max-num-seqs` to control active scheduler concurrency. Use
`--vllm-max-num-batched-tokens` to override the chunked-prefill token budget.
The asynchronous lane uses the same checkpoint and raw multimodal-prompt
corrections:

```sh
$VLLM_PYTHON \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  --backend vllm-async-engine \
  --model "$MODEL_DIR" \
  --output-dir tmp/11_mineru_2_5_pro_inference/official_vllm_async_eager_n8 \
  --limit 8 \
  --page-batch-size 8 \
  --batch-size 0 \
  --vllm-enforce-eager \
  --vllm-max-num-seqs 128
```

`FULL_DECODE_ONLY` leaves multimodal and text prefill outside ACLGraph and
captures pure decode batches only:

```sh
$VLLM_PYTHON \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  --backend vllm-engine \
  --model "$MODEL_DIR" \
  --output-dir tmp/11_mineru_2_5_pro_inference/official_vllm_full_decode_n8 \
  --limit 8 \
  --page-batch-size 8 \
  --batch-size 0 \
  --no-vllm-enforce-eager \
  --vllm-full-decode-only \
  --vllm-max-num-seqs 128 \
  --vllm-cudagraph-capture-sizes 1,2,4,8,16,32,64,128
```

## Two-step single-page smoke

```sh
$PYTHON 11_mineru_2_5_pro_inference/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --max-new-tokens 128 \
  --recognition-decode eager \
  --cache-length 512 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --benchmark-decode \
  --decode-warmup-steps 4 \
  --decode-measure-steps 32 \
  --hash-model-files \
  --output tmp/11_mineru_2_5_pro_inference/two_step_smoke.json
```

This smoke is fully eager and does not import or invoke TorchAir. Pass
`--recognition-decode compiled` to use the static-cache compiled recognition
decode path; vision, projector, processor work and recognition prefill remain
outside that graph.

## Compiled batch-decode benchmark

```sh
$PYTHON 11_mineru_2_5_pro_inference/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --batch-size 4 \
  --cache-length 512 \
  --measure-steps 64 \
  --warmup-steps 8 \
  --validation-steps 8 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --hash-model-files \
  --output tmp/11_mineru_2_5_pro_inference/batch4_decode.json
```

For batch size `B` and `N` measured steps:

```text
raw batch tokens = B * N
decode calls/s   = N / decode wall time
raw batch tok/s  = (B * N) / decode wall time
```

Throughput excludes model load, processor work, per-crop eager prefill, cache
assembly, compilation and warmup. Do not trust throughput unless
`validation.token_match_all` is true.

## Text-decode lab

`text_decode_lab.py` isolates the complete 24-layer static decode step. It
keeps the real model weights, rotary encoding, KV-cache update, decoder
layers, LM head, token feedback, and TorchAir graph. It excludes image work
and prefill. The default lanes compare the existing manual GQA attention with
`torch_npu.npu_incre_flash_attention` and save a Torch-NPU pipe profile for
both lanes.

```sh
$PYTHON 11_mineru_2_5_pro_inference/text_decode_lab.py \
  --batch-size 1 \
  --cache-length 4096 \
  --profile-position 2048 \
  --warmup-steps 8 \
  --measure-steps 64 \
  --validation-steps 8 \
  --profile \
  --profile-steps 2 \
  --profile-metric pipe \
  --output tmp/11_mineru_2_5_pro_inference/text_decode_lab/result.json
```

Use `parse_npu_profile.py` on each emitted `profile_*` directory for ranked
kernel and operator summaries. Profile wall time includes profiler overhead;
only the separate warmed `measure` section is a throughput measurement.

## Current validation status

This commit transfers the previously developed implementation but does not
claim a fresh Experiment-11 NPU run. Run the two-step smoke first and preserve
its command, log and JSON under `tmp/11_mineru_2_5_pro_inference/` before using
the batch benchmark as current evidence.
