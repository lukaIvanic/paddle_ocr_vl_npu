#!/usr/bin/env python3
"""Run PaddleOCR-VL recognition without importing Transformers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from device_runtime import (
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    parse_dtype,
    resolve_device,
    synchronize_device,
)
from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from preprocessing import (
    build_inputs,
    load_preprocessor_config,
    preprocess_image,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6", help="Hub model id or local model directory.")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png", help="Path to a recognition crop.")
    parser.add_argument("--prompt", default="OCR:", help="Recognition prompt, e.g. OCR:, Table Recognition:, Formula Recognition:.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--static", action="store_true", help="Use the inherited static KV cache decode path.")
    parser.add_argument("--cache-length", type=int, default=None, help="Static KV cache length; defaults to input length + max new tokens.")
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model)
    crop = Path(args.crop)
    if not crop.exists():
        crop = Path(__file__).resolve().parents[1] / args.crop
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(crop, pre_cfg)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt, merge_size=int(pre_cfg["merge_size"]))

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    synchronize_device(device)
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
    synchronize_device(device)
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
