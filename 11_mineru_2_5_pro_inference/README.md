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
  cropping, then static-cache compiled recognition decode.

bench_compiled_batch_decode.py
  Fixed-step compiled decode throughput over distinct real crop inputs.
  Sequential crop preprocessing and prefill are intentionally outside the
  measured decode interval.

parse_npu_profile.py
  Parser for torch-npu profiler output from the compiled decode path.
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
  --cache-length 512 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --benchmark-decode \
  --decode-warmup-steps 4 \
  --decode-measure-steps 32 \
  --hash-model-files \
  --output tmp/11_mineru_2_5_pro_inference/two_step_smoke.json
```

The layout stage remains eager and dynamic. Only one-token recognition decode
is compiled; vision, projector, processor work and recognition prefill remain
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
