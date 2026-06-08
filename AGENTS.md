# AGENTS.md

## Operating Lanes

First classify where you are running from actual machine state, not from memory:

- Work/NPU lane: Ascend NPU tooling is present, such as `npu-smi` or `torch_npu`.
- Blue-zone/GPU lane: NVIDIA CUDA tooling is present, such as `nvidia-smi`, but no Ascend NPU.
- Local-only lane: neither NPU nor CUDA is available.

The work/NPU lane is pull-only. Its job is to set up the environment, pull the repo, run scripts, inspect outputs, debug failures, and summarize exact findings. Do not edit tracked files, commit, push, or create branches from the work/NPU lane. If a code change seems necessary, report the minimal proposed change, the command that failed, and the relevant logs instead of applying it.

The blue-zone/GPU lane may edit this repo only when Luka is using it as the authoring environment. CUDA results are smoke-test evidence only. They are useful for dependency bring-up and model-loading checks, but they are not NPU or Ascend performance evidence.

The local-only lane should prepare scripts, crops, docs, and GitHub sync. It should not present unrun local code as validated inference.

## Project Direction

This folder is a standalone research workspace for PaddleOCR-VL on Ascend/NPU, with a near-term focus on the `PaddleOCR-VL-1.6-0.9B` recognition VLM.

The immediate target is the core recognition model, not the full document parser. The recognition model is available as a Transformers/PyTorch model at:

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
- `01_run_transformers_recognition.py`: minimal Transformers recognizer smoke script.
- `refs/`: small architecture reference artifacts.
- `refs/PaddleOCR`: ignored sparse reference checkout of the official PaddleOCR repo.

Keep `refs/PaddleOCR/` ignored. It is reference material, not project source.

## Local Smoke Commands

Regenerate the crops from the parent repo's restored OmniDocBench copy:

```sh
python3 crops/create_omnidocbench_recognition_crops.py
```

Run the core recognition model on one crop with Transformers:

```sh
python3 01_run_transformers_recognition.py
```

For another crop:

```sh
python3 01_run_transformers_recognition.py \
  --crop crops/crop_05_table_rwkv_dims.png \
  --prompt "Table Recognition:"
```

## Remote / Hardware Rules

As of 2026-06-08, there is no active Ascend/NPU rental attached to this folder. NPU validation will happen later from a work laptop or a new Ascend environment.

CUDA/Vast.ai runs are allowed for basic bring-up, dependency checks, and algorithmic evidence only. Do not describe Vast results as NPU or Ascend throughput. If a result comes from CUDA, label it as CUDA/Vast.

When an Ascend target is available, do not silently fall back to CPU for NPU experiments. If the NPU path fails, stop and fix or document the blocker. CPU is acceptable for local file checks, JSON/image processing, and other tooling boundaries.

For Huawei Ascend NPU, current public PaddleOCR guidance says local direct inference is not the supported path; the official route is PaddleOCR client/pipeline plus a vLLM VLM service, or the two-container full API service. Treat direct Transformers-on-NPU work as an experiment until validated.

## Vast.ai Setup Intent

Use Vast.ai only as a temporary CUDA validation lane. The likely first goal is:

1. Put this subproject on the Vast box.
2. Create a clean Python environment.
3. Install Transformers/PyTorch/Pillow.
4. Run `01_run_transformers_recognition.py` against one crop.
5. Record whether the Transformers recognizer loads, preprocesses, and generates.

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
/workspace/venvs/paddle_ocr_vl/bin/python 01_run_transformers_recognition.py \
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

Only the authoring lane should commit and push. The work/NPU lane should only pull from `origin`, run, and report.

## Style

Keep notes short, concrete, and source-backed. Prefer small scripts that can be run on local CPU/CUDA first and then moved to Ascend. When a result is a smoke test, call it a smoke test.
