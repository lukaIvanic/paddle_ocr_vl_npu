#!/usr/bin/env python3
"""Minimal one-crop PaddleOCR-VL inference example."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]

from .modeling import LocalPaddleOCRVLForConditionalGeneration, _resolve_model_dir
from .preprocessing import (
    build_inputs,
    load_preprocessor_config,
    preprocess_image,
)
from .text_decode import DECODE_ATTENTION
from utils.timing import synchronize


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6", help="Hub model id or local model directory.")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png", help="Path to a recognition crop.")
    parser.add_argument("--prompt", default="OCR:", help="Recognition prompt, e.g. OCR:, Table Recognition:, Formula Recognition:.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--static", action="store_true", help="Use the inherited static KV cache decode path.")
    parser.add_argument("--cache-length", type=int, default=None, help="Static KV cache length; defaults to input length + max new tokens.")
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model)
    crop = Path(args.crop)
    if not crop.exists():
        crop = REPO_ROOT / args.crop
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("The one-crop example requires an available NPU")
    dtype = (
        torch.float16
        if args.dtype in {"fp16", "float16"}
        else torch.bfloat16
    )
    torch.npu.set_compile_mode(jit_compile=False)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(crop, pre_cfg)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt, merge_size=int(pre_cfg["merge_size"]))

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    synchronize(device)
    start = time.perf_counter()
    if args.static:
        new_ids = model.generate_ids_static(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=args.max_new_tokens,
            cache_length=args.cache_length,
        )
    else:
        new_ids = model.generate_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=args.max_new_tokens,
        )
    synchronize(device)
    elapsed = time.perf_counter() - start

    generated = new_ids[0].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    decode_attention = DECODE_ATTENTION if args.static else "dynamic_reference"
    print(text.strip())
    print(
        f"\n[local] device={device} dtype={dtype} static={args.static} "
        f"decode_attention={decode_attention} input_tokens={input_ids.shape[1]} "
        f"new_tokens={len(generated)} elapsed_s={elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
