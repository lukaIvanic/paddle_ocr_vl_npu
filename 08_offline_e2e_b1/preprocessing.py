"""PaddleOCR-VL crop preprocessing and prompt construction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer


IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"


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
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}"
        )
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


def apply_min_pixels_override(cfg: dict, min_pixels: int | None) -> dict:
    """Return a copied preprocessor config with only ``min_pixels`` changed."""
    effective = dict(cfg)
    if min_pixels is None:
        return effective

    min_pixels = int(min_pixels)
    if min_pixels <= 0:
        raise ValueError("preprocessor min_pixels override must be positive")
    max_pixels = int(effective["max_pixels"])
    if min_pixels > max_pixels:
        raise ValueError(
            "preprocessor min_pixels override must not exceed max_pixels: "
            f"min_pixels={min_pixels}, max_pixels={max_pixels}"
        )
    effective["min_pixels"] = min_pixels
    return effective


def preprocess_pil_image(
    image: Image.Image,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess one in-memory crop with the local PaddleOCR-VL recipe."""
    if cfg["do_convert_rgb"]:
        image = image.convert("RGB")
    width, height = image.size
    patch_size = int(cfg["patch_size"])
    merge_size = int(cfg["merge_size"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    if temporal_patch_size != 1:
        raise ValueError(
            "temporal_patch_size must be 1 for this recognizer path, "
            f"got {temporal_patch_size}"
        )

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
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel,
        patch_size,
        patch_size,
    )
    return (
        torch.from_numpy(flatten_patches),
        torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
    )


def preprocess_image(
    image_path: Path,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(image_path) as image:
        return preprocess_pil_image(image, cfg)


def build_inputs(
    tokenizer: Tokenizer,
    image_grid_thw: torch.Tensor,
    prompt: str,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_token_count = (
        int(image_grid_thw[0].prod().item()) // merge_size // merge_size
    )
    text = build_paddleocr_vl_prompt(
        prompt,
        image_token_count=image_token_count,
    )
    ids = tokenizer.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def build_paddleocr_vl_prompt(prompt: str, *, image_token_count: int) -> str:
    """Mirror the chat template and processor image-token expansion."""
    template = (
        f"{BOS}User: {IMAGE_START}{IMAGE_TOKEN}{IMAGE_END}{prompt}\nAssistant:\n"
    )
    placeholder = "<|placeholder|>"
    return template.replace(
        IMAGE_TOKEN,
        placeholder * int(image_token_count),
        1,
    ).replace(placeholder, IMAGE_TOKEN)
