#!/usr/bin/env python3
"""Run PaddleOCR-VL recognition without importing Transformers."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import LocalPaddleOCRVLForConditionalGeneration, _resolve_model_dir


IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"
NPU_JIT_COMPILE_CHOICES = ("default", "off", "on")


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
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
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_preprocessor_config(model_dir: Path) -> dict:
    defaults = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_pixels": 1003520,
        "merge_size": 2,
        "min_pixels": 112896,
        "patch_size": 14,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "temporal_patch_size": 1,
    }
    path = model_dir / "preprocessor_config.json"
    if path.exists():
        defaults.update(json.loads(path.read_text(encoding="utf-8")))
    return defaults


def preprocess_image(image_path: Path, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path)
    if cfg["do_convert_rgb"]:
        image = image.convert("RGB")
    width, height = image.size
    patch_size = int(cfg["patch_size"])
    merge_size = int(cfg["merge_size"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    if temporal_patch_size != 1:
        raise ValueError(f"temporal_patch_size must be 1 for this recognizer path, got {temporal_patch_size}")

    resized_height, resized_width = height, width
    if cfg["do_resize"]:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=int(cfg["min_pixels"]),
            max_pixels=int(cfg["max_pixels"]),
        )
        resample = Image.Resampling(int(cfg["resample"]))
        image = image.resize((resized_width, resized_height), resample=resample)

    array = np.asarray(image)
    if cfg["do_rescale"]:
        array = array.astype(np.float32) * float(cfg["rescale_factor"])
    else:
        array = array.astype(np.float32)
    if cfg["do_normalize"]:
        mean = np.array(cfg["image_mean"], dtype=np.float32)
        std = np.array(cfg["image_std"], dtype=np.float32)
        array = (array - mean) / std

    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size)
    return torch.from_numpy(flatten_patches), torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)


def build_inputs(
    tokenizer: Tokenizer,
    image_grid_thw: torch.Tensor,
    prompt: str,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_token_count = int(image_grid_thw[0].prod().item()) // merge_size // merge_size
    text = f"{BOS}User: {IMAGE_START}{IMAGE_TOKEN * image_token_count}{IMAGE_END}{prompt}\nAssistant:\n"
    ids = tokenizer.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def parse_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        return False
    except Exception as exc:
        raise RuntimeError(f"torch_npu is installed but failed to initialize: {exc.__class__.__name__}: {exc}") from exc
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if npu_is_available():
            return torch.device("npu:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU device requested, but torch_npu is not importable in this environment.") from exc
    return torch.device(name)


def configure_npu_jit_compile(mode: str, device: torch.device, *, verbose: bool = True) -> None:
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6", help="Hub model id or local model directory.")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png", help="Path to a recognition crop.")
    parser.add_argument("--prompt", default="OCR:", help="Recognition prompt, e.g. OCR:, Table Recognition:, Formula Recognition:.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--static", action="store_true", help="Use the experiment-3 static KV cache decode path.")
    parser.add_argument("--cache-length", type=int, default=None, help="Static KV cache length; defaults to input length + max new tokens.")
    parser.add_argument("--cache-update", default="index_copy", choices=["index_copy", "scatter_update"], help="Static cache update op.")
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
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.set_static_cache_update_mode(args.cache_update)

    if device.type == "cuda":
        torch.cuda.synchronize()
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = new_ids[0].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text.strip())
    print(
        f"\n[local] device={device} dtype={dtype} static={args.static} "
        f"input_tokens={input_ids.shape[1]} new_tokens={len(generated)} elapsed_s={elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
