#!/usr/bin/env python3
"""Run PaddleOCR-VL-1.6 recognition on one crop with Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP = REPO_ROOT / "crops" / "crop_01_text_block_en.png"

DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to a local model directory. If you do not have the model, download it from huggingface.com")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    image = Image.open(args.crop.expanduser().resolve()).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=DTYPES[args.dtype],
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(device).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    print(processor.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
