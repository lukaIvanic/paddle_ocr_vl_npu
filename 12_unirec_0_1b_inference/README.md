# Experiment 12: UniRec 0.1B Inference

This experiment compares the official UniRec implementations with the local
implementation that we can modify and optimize. It also contains a thin runner
for the official OpenDoc full-page pipeline.

## Files

- `modeling_optimized_unirec.py`: self-contained UniRec encoder, decoder, image
  processor, static KV cache, weight loader, and TorchAir decode integration.
- `run_optimized.py`: local eager or compiled inference.
- `run_original_transformers.py`: exact Transformers model and processor code
  bundled with the official UniRec-0.1B-1217 checkpoint.
- `run_official_opendoc.py`: OpenOCR's official PP-DocLayoutV2 + UniRec page
  pipeline with explicit checkpoint and recognizer-device selection.

The local model implementation is copied without architectural changes from
`unirec_research/03_compiled_decode_single_batch` at commit `4b9a9ab`.

## Verified Blue Zone inputs

```text
Model:       /workspace/models/unirec_0_1b_1217
Official Python: /workspace/venvs/unirec1217_npu_py312/bin/python
Custom Python:   /workspace/venvs/unirec_npu_py312/bin/python
Official Transformers: 4.49.0
```

The 1217 model directory is the official `topdu/unirec_0_1b` checkpoint at revision
`d2469d0f50992a380240266fe169b982ea940615`. This is the table-capable
UniRec-0.1B-1217 release. The installed `model.safetensors` is 535,797,520
bytes and has SHA-256:

```text
1a080d683731d2bdae5a4b8c538160d2e8b1733f44de25cb75f264406db8d746
```

This matches the official Hugging Face LFS metadata.

OpenOCR's documented OpenDoc score-reproduction path uses the distinct
`topdu/unirec-0.1b` `model.pth` checkpoint. It is also table-capable in the
OpenDoc pipeline and is not numerically identical to the 1217 safetensors
checkpoint. Use `/workspace/models/unirec-0.1b/model.pth` when reproducing the
published OpenDoc OmniDocBench score. Do not interchange the two checkpoints
when comparing exact outputs or reported metrics.

## Setup

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
```

Create the isolated official environment once. It inherits the installed
Torch-NPU runtime but keeps the checkpoint's Transformers version separate
from vLLM:

```sh
/usr/local/python3.12.13/bin/python3 -m venv --system-site-packages \
  /workspace/venvs/unirec1217_npu_py312
/workspace/venvs/unirec1217_npu_py312/bin/python -m pip install \
  -r 12_unirec_0_1b_inference/requirements-official.txt
```

The commands below use `crops/crop_01_text_block_en.png`. Without an `--image`
argument, each runner uses the first six repository crop images.

## Local implementation

Eager:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype bfloat16 --decode-mode eager
```

Cached TorchAir decode:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype bfloat16 --decode-mode compiled \
  --compile-backend torchair
```

## Official bundled Transformers reference

```sh
/workspace/venvs/unirec1217_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_original_transformers.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype bfloat16
```

## Official OpenDoc full-page pipeline

OpenOCR provides two full-page implementations:

- `openocr.py --task doc`: the packaged ONNX path. It runs PP-DocLayoutV2 and
  up to four UniRec crop workers. This path is validated in the Blue Zone ARM
  container.
- `tools/infer_doc.py`: the PaddleX + PyTorch path used by the OpenOCR and
  OmniDocBench source trees. The two repositories contain byte-identical
  scripts. PaddlePaddle 3.2.0 currently segfaults while loading the Paddle
  PP-DocLayoutV2 parameters on this ARM container, before page inference.

The official ONNX UniRec export produced exact text parity with the OpenDoc
`model.pth` checkpoint on the same table crop through the tested 128-token
generation. Use the committed full-run wrapper for OmniDocBench v1.6:

```sh
bash 12_unirec_0_1b_inference/run_official_opendoc_omnidocbench.sh
```

The wrapper runs the official CLI over the complete image directory and saves
the exact command, continuous log, exit code, wall time, JSON, and Markdown.
The current official OmniDocBench v1.6 target is:

```text
Overall:       90.67
Text Edit:      0.049
Formula CDM:   93.02
Table TEDS:    83.88
Reading Edit:   0.140
```

The 90.57 figure in OpenOCR's document is the older OmniDocBench v1.5 result.
The installed 1,651-page dataset is v1.6.

The source-pipeline adapter remains available for environments where the
Paddle predictor loads correctly. It preserves OpenOCR's pipeline and places
only UniRec on the NPU:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/run_official_opendoc.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/unirec-0.1b/model.pth \
  --input /workspace/datasets/OmniDocBench/images \
  --output-dir tmp/12_unirec_0_1b_inference/opendoc_reference \
  --recognizer-device npu:0
```

Use `--limit 1` for a smoke. Omit `--limit` for the complete image directory.
Neither runner replaces the OmniDocBench evaluator. Score the Markdown outputs
with the standard evaluator after inference.

## Artifacts

- Run JSON: `tmp/12_unirec_0_1b_inference/`
- TorchAir and compatibility caches:
  `.runtime_cache/12_unirec_0_1b_inference/`

`source npu-setup` is mandatory. It selects a free physical NPU and exposes it
as logical `npu:0`.

## Validation

Validated on Ascend 910B2 with the table crop
`crops/crop_05_table_rwkv_dims.png`, BF16, JIT compile disabled, and a 64-token
limit. The official bundled Transformers implementation and the local eager
implementation produced exact token and text parity across all 64 returned
tokens. Both generated native HTML table markup. The official lane produced
53.2 generated tokens/s. The local eager decode lane produced 115.5 decode
tokens/s.
