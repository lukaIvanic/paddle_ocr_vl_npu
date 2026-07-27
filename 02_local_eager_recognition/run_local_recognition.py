#!/usr/bin/env python3
"""Run PaddleOCR-VL recognition without importing Transformers."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import LocalPaddleOCRVLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP = REPO_ROOT / "crops" / "crop_01_text_block_en.png"

IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"

DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

# Preprocessing constants from
# https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/preprocessor_config.json
# (verified against the checkpoint copy, 2026-07-27).
PATCH_SIZE = 14
MERGE_SIZE = 2
MIN_PIXELS = 112896
MAX_PIXELS = 1003520
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
RESAMPLE = Image.Resampling.BICUBIC


def smart_resize(height: int, width: int) -> tuple[int, int]:
    factor = PATCH_SIZE * MERGE_SIZE
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(height, width)
    image = image.resize((resized_width, resized_height), resample=RESAMPLE)

    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD

    grid_h = resized_height // PATCH_SIZE
    grid_w = resized_width // PATCH_SIZE
    patches = array.transpose(2, 0, 1)
    patches = patches.reshape(3, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE)
    patches = patches.transpose(1, 3, 0, 2, 4)
    flatten_patches = patches.reshape(grid_h * grid_w, 3, PATCH_SIZE, PATCH_SIZE)
    return torch.from_numpy(flatten_patches), torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)


def build_inputs(tokenizer: Tokenizer, image_grid_thw: torch.Tensor, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors PaddleOCR-VL's chat_template.jinja plus the processor's
    # image-token expansion.
    image_token_count = int(image_grid_thw[0].prod().item()) // (MERGE_SIZE * MERGE_SIZE)
    text = f"{BOS}User: {IMAGE_START}{IMAGE_TOKEN * image_token_count}{IMAGE_END}{prompt}\nAssistant:\n"
    ids = tokenizer.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a local model directory.")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP, help="Path to a recognition crop.")
    parser.add_argument("--prompt", default="OCR:", help="Recognition prompt, e.g. OCR:, Table Recognition:, Formula Recognition:.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    args = parser.parse_args()

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    dtype = DTYPES[args.dtype]

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(args.crop)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt)

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(args.model, dtype=dtype, device=device)
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch.npu.synchronize()
    start = time.perf_counter()
    new_ids = model.generate_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    generated = new_ids[0].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text.strip())
    print(f"\n[local] device={device} dtype={dtype} input_tokens={input_ids.shape[1]} new_tokens={len(generated)} elapsed_s={elapsed:.3f}")


if __name__ == "__main__":
    main()
