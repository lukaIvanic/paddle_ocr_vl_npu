# Experiment 11: MinerU2.5-Pro Local Inference

This experiment is the current custom MinerU2.5-Pro implementation transferred
from the standalone `mineru_25_pro_npu` repository at commit `b08ae14`. It does
not import vLLM or vLLM-Ascend.

The model itself is implemented locally in PyTorch. Transformers remains only
at the processor boundary through `AutoProcessor`, for tokenization and
Qwen2-VL image preprocessing.

Default checkpoint:

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
```

Pass a local checkpoint directory to every runner; these scripts do not
download model weights.

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
  Fidelity-first corpus runner around the stock Transformers model and the
  official mineru-vl-utils two-step page client. It writes official json2md
  Markdown, content lists, per-page checkpoints, shard progress, and timing.
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

## Official Transformers OmniDocBench lane

The fidelity baseline follows the checkpoint model card: BF16 stock
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

## Current validation status

This commit transfers the previously developed implementation but does not
claim a fresh Experiment-11 NPU run. Run the two-step smoke first and preserve
its command, log and JSON under `tmp/11_mineru_2_5_pro_inference/` before using
the batch benchmark as current evidence.
