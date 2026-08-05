#!/usr/bin/env python3
"""Run one eager MinerU recognition request through stock Transformers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image


def _triple(value):
    if isinstance(value, int):
        return [value, value, value]
    if len(value) == 1:
        return [int(value[0])] * 3
    return [int(item) for item in value]


def configure_npu() -> None:
    import torch
    import torch.nn.functional as functional
    import torch_npu

    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    original_conv3d = functional.conv3d

    def npu_conv3d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input_tensor.device.type == "npu":
            return torch_npu.npu_conv3d(
                input_tensor,
                weight,
                bias,
                _triple(stride),
                _triple(padding),
                _triple(dilation),
                int(groups),
            )
        return original_conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)

    functional.conv3d = npu_conv3d
    print("[npu] jit_compile=False; Conv3D NPU inference patch enabled", flush=True)


def synchronize() -> None:
    import torch

    torch.npu.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_npu()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_dir = args.model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    print(f"[setup] loading Transformers model from {model_dir}", flush=True)
    setup_start = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    )
    # MinerU stores only model.embed_tokens.weight and its local implementation
    # deliberately uses that tied matrix for output logits. Current Transformers
    # otherwise reports lm_head.weight missing and randomly initializes it.
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model = model.to("npu:0").eval()
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    synchronize()
    setup_s = time.perf_counter() - setup_start
    print(f"[setup] complete in {setup_s:.3f}s", flush=True)

    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "\nText Recognition:"},
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device="npu:0", dtype=torch.float16)

    print(f"[inference] starting eager generation; input_tokens={inputs.input_ids.shape[1]}", flush=True)
    run_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=int(args.max_new_tokens),
        )
    synchronize()
    run_s = time.perf_counter() - run_start
    generated_ids = generated[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    payload = {
        "model": str(model_dir),
        "image": str(image_path),
        "backend": "transformers_eager",
        "npu_jit_compile": False,
        "attention": "eager",
        "input_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(generated_ids.shape[1]),
        "setup_s": float(setup_s),
        "generate_s": float(run_s),
        "text": text,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"WROTE {output_path}", flush=True)


if __name__ == "__main__":
    main()
