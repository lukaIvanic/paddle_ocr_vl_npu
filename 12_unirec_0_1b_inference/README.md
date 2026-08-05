# Experiment 12: UniRec 0.1B Inference

This experiment compares the official OpenOCR implementation of UniRec-0.1B
with the local implementation that we can modify and optimize.

## Files

- `modeling_optimized_unirec.py`: self-contained UniRec encoder, decoder, image
  processor, static KV cache, weight loader, and TorchAir decode integration.
- `run_optimized.py`: local eager or compiled inference.
- `run_original_transformers.py`: official OpenOCR/Transformers reference.

The local model implementation is copied without architectural changes from
`unirec_research/03_compiled_decode_single_batch` at commit `4b9a9ab`.

## Verified Blue Zone inputs

```text
Model:       /workspace/models/unirec-0.1b
OpenOCR:     /workspace/repos/OpenOCR
Python:      /workspace/venvs/unirec_npu_py312/bin/python
Transformers: 5.2.0
```

The model directory is the official `topdu/unirec-0.1b` checkpoint. The
installed `model.pth` is 535,901,578 bytes and has SHA-256:

```text
b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef
```

This matches the file metadata published by the official Hugging Face model
repository at revision `a377e00d62c01b6544603e2a90f2cffe2a0388e1`.

## Setup

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
```

The commands below use `crops/crop_01_text_block_en.png`. Without an `--image`
argument, each runner uses the first six repository crop images.

## Local implementation

Eager:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec-0.1b \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16 --decode-mode eager
```

Cached TorchAir decode:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec-0.1b \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16 --decode-mode compiled \
  --compile-backend torchair
```

## Official OpenOCR reference

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_original_transformers.py \
  --model-path /workspace/models/unirec-0.1b \
  --openocr-root /workspace/repos/OpenOCR \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16 \
  --openocr-transformers52-compat check \
  --openocr-npu-device-compat check
```

## Artifacts

- Run JSON: `tmp/12_unirec_0_1b_inference/`
- TorchAir and compatibility caches:
  `.runtime_cache/12_unirec_0_1b_inference/`

`source npu-setup` is mandatory. It selects a free physical NPU and exposes it
as logical `npu:0`.

## Validation

Verified on Ascend 910B2 at commit `5985cf3` with one text crop and eager
decode. Both implementations completed successfully on the same checkpoint.
The official OpenOCR output was an exact prefix of the longer local output.
OpenOCR stopped at its internal default generation length; the local runner
continued to the requested 64-token limit.
