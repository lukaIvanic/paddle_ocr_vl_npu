#!/usr/bin/env python3
"""Run PaddleOCR-VL-1.6 recognition on one crop with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
DEFAULT_CROP = REPO_ROOT / "crops" / "crop_01_text_block_en.png"
MANIFEST = REPO_ROOT / "crops" / "manifest.json"
NPU_JIT_COMPILE_CHOICES = ("default", "off", "on")


def default_prompt_for(crop: Path) -> str:
    if not MANIFEST.exists():
        return "OCR:"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest:
        if item.get("file") == crop.name:
            return item.get("suggested_prompt") or "OCR:"
    return "OCR:"


def parse_dtype(torch, name: str, _device):
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def npu_is_available(torch) -> bool:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        return False
    except Exception as exc:
        raise RuntimeError(f"torch_npu is installed but failed to initialize: {exc.__class__.__name__}: {exc}") from exc
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(torch, device_arg: str | None):
    if device_arg:
        if device_arg.startswith("npu"):
            import torch_npu  # noqa: F401
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if npu_is_available(torch):
        return torch.device("npu:0")
    return torch.device("cpu")


def configure_npu_jit_compile(torch, mode: str, device, *, verbose: bool = True) -> None:
    if mode not in NPU_JIT_COMPILE_CHOICES:
        raise ValueError(f"unsupported npu_jit_compile={mode!r}")
    if mode == "default" or device.type != "npu":
        return
    try:
        import torch_npu  # noqa: F401

        requested = mode == "on"
        torch.npu.set_compile_mode(jit_compile=requested)
        if verbose:
            print(f"[npu] set torch.npu compile mode: jit_compile={requested}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"failed to set NPU jit_compile={mode}: {exc.__class__.__name__}: {exc}") from exc


def move_model_to_device(torch, model, device):
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
        return model.npu().eval()
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="eager", help="Attention backend for Transformers; use 'default' to omit.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use-fast", action="store_true", help="Use the fast HF image processor instead of the source-matched slow path.")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    crop = args.crop.expanduser().resolve()
    prompt = args.prompt or default_prompt_for(crop)
    device = resolve_device(torch, args.device)
    dtype = parse_dtype(torch, args.dtype, device)
    configure_npu_jit_compile(torch, args.npu_jit_compile, device)

    image = Image.open(crop).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=args.use_fast, trust_remote_code=args.trust_remote_code)
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    model = move_model_to_device(torch, model, device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
