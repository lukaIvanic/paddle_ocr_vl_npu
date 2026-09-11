"""PaddleOCR-VL model input preprocessing and prompt construction."""

from __future__ import annotations

import json
import html
import re
from collections import Counter
from typing import Any
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer


IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"


# Crop preprocessing


def preprocess_pil_image(
    image: Image.Image,
    cfg: dict,
    *,
    defer_normalization: bool = True,
    resize_backend: str = "kornia_rs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare crop patches; default uint8 output is normalized by the NPU stage.

    Kornia + uint8 is anchored by preprocess_options_20260911's B2/B8 runs.
    Explicit defer_normalization=False retains normalized CPU float32 output.
    """
    if cfg["do_convert_rgb"] and image.mode != "RGB":
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

    grid_t, grid_h, grid_w = image_grid_thw_from_size(
        width,
        height,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=int(cfg["min_pixels"]),
        max_pixels=int(cfg["max_pixels"]),
        do_resize=bool(cfg["do_resize"]),
    )
    resized_height = grid_h * patch_size
    resized_width = grid_w * patch_size
    resized_array: np.ndarray | None = None
    if cfg["do_resize"]:
        if resize_backend == "pillow":
            resample = Image.Resampling(int(cfg["resample"]))
            image = image.resize(
                (resized_width, resized_height),
                resample=resample,
            )
        elif resize_backend == "kornia_rs":
            from kornia_rs.image import Image as KorniaImage

            resized_array = KorniaImage.fromarray(np.asarray(image)).resize(
                resized_width,
                resized_height,
                "bicubic",
            ).data
        else:
            raise ValueError(f"unsupported resize_backend: {resize_backend!r}")

    array = resized_array if resized_array is not None else np.asarray(image)
    if not defer_normalization:
        array = _normalize_image_array(array, cfg)

    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
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


def image_grid_thw_from_size(
    width: int,
    height: int,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    min_pixels: int,
    max_pixels: int,
    do_resize: bool = True,
) -> tuple[int, int, int]:
    """Return the shape-only image grid used by ``preprocess_pil_image``.

    This deliberately mirrors only the resize and patch-grid math. It does not
    allocate an image or perform resampling, normalization, or patchification.
    """
    width = int(width)
    height = int(height)
    patch_size = int(patch_size)
    merge_size = int(merge_size)
    temporal_patch_size = int(temporal_patch_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size and merge_size must be positive")
    if temporal_patch_size != 1:
        raise ValueError(
            "temporal_patch_size must be 1 for this recognizer path, "
            f"got {temporal_patch_size}"
        )

    resized_height, resized_width = height, width
    if do_resize:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
    if resized_height % patch_size or resized_width % patch_size:
        raise ValueError(
            "resized image dimensions must be divisible by patch_size: "
            f"size={(resized_width, resized_height)} patch_size={patch_size}"
        )
    return (
        1,
        resized_height // patch_size,
        resized_width // patch_size,
    )


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


def _normalize_image_array(array: np.ndarray, cfg: dict) -> np.ndarray:
    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std = np.asarray(cfg["image_std"], dtype=np.float32)
    if (array.dtype == np.uint8 and array.ndim == 3 and array.shape[-1] == 3
            and cfg["do_normalize"] and mean.size in (1, 3) and std.size in (1, 3)
            and np.all(mean == mean.flat[0])
            and np.all(std == std.flat[0]) and std.flat[0] != 0):
        table = _uint8_normalization_table(bool(cfg["do_rescale"]), float(cfg["rescale_factor"]),
                                           float(mean.flat[0]), float(std.flat[0]))
        return table[array]
    # Preserve the generic recipe for nonuniform per-channel normalization or
    # non-uint8 images; this optimization changes neither resampling nor shapes.
    array = array.astype(np.float32)
    if cfg["do_rescale"]:
        array = array * float(cfg["rescale_factor"])
    if cfg["do_normalize"]:
        array = (array - mean) / std
    return array


@lru_cache(maxsize=16)
def _uint8_normalization_table(rescale: bool, factor: float, mean: float, std: float) -> np.ndarray:
    # There are only 256 possible channel values after Pillow resampling.
    # Keep exactly the reference float32 operation order, not a fused affine
    # expression, so the table preserves every rounding bit.
    values = np.arange(256, dtype=np.uint8).astype(np.float32)
    if rescale:
        values = values * factor
    values = (values - np.float32(mean)) / np.float32(std)
    values.flags.writeable = False
    return values


# Prompt and token input construction


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


# Output normalization and table formatting


def normalize_recognition_text(label: str, text: str | None) -> str:
    result = truncate_repetitive_content(
        text or "",
        min_count=5000 if label == "table" else 50,
    )
    if (
        ("\\(" in result and "\\)" in result)
        or ("\\[" in result and "\\]" in result)
    ):
        result = result.replace("$", "")
        result = (
            result.replace("\\(", " $ ")
            .replace("\\)", " $")
            .replace("\\[\\[", "\\[")
            .replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ")
            .replace("\\]", " $$ ")
        )
        if label == "formula_number":
            result = result.replace("$", "")
    if label == "table":
        converted = convert_otsl_to_html(result)
        if converted:
            result = converted
    return result


def truncate_repetitive_content(
    content: str,
    *,
    min_count: int,
) -> str:
    if len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content
    if "\n" not in stripped and len(stripped) > 100:
        suffix = _repeating_suffix(stripped)
        if suffix is not None:
            prefix, unit, count = suffix
            if len(unit) * count > len(stripped) * 0.5:
                return prefix
    if "\n" not in stripped and len(stripped) > 10:
        unit = _shortest_repeating_substring(stripped)
        if unit is not None and len(stripped) // len(unit) >= 10:
            return unit
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if len(lines) < 10:
        return content
    common, count = Counter(lines).most_common(1)[0]
    return common if count >= 10 and count / len(lines) >= 0.8 else content


def _repeating_suffix(
    value: str,
    min_length: int = 8,
    min_repeats: int = 5,
) -> tuple[str, str, int] | None:
    for length in range(
        len(value) // min_repeats,
        min_length - 1,
        -1,
    ):
        unit = value[-length:]
        if not value.endswith(unit * min_repeats):
            continue
        count = 0
        prefix = value
        while prefix.endswith(unit):
            prefix = prefix[:-length]
            count += 1
        return prefix, unit, count
    return None


def _shortest_repeating_substring(value: str) -> str | None:
    for length in range(1, len(value) // 2 + 1):
        if len(value) % length == 0:
            candidate = value[:length]
            if candidate * (len(value) // length) == value:
                return candidate
    return None


def convert_otsl_to_html(content: str) -> str:
    """Convert the OTSL cell grammar emitted by PaddleOCR-VL into HTML.

    The implementation covers all six OTSL tags.  For ordinary ``fcel`` and
    ``ecel`` tables it is byte-identical to PaddleX.  Span tags are resolved
    into rowspan/colspan attributes without importing PaddleX's Pydantic table
    model.
    """
    rows = _parse_otsl_rows(content)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    grid = [
        row + [("ecel", "")] * (width - len(row))
        for row in rows
    ]
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    for row_index, row in enumerate(grid):
        for column_index, (token, text) in enumerate(row):
            if token in {"fcel", "ecel"}:
                anchor = (row_index, column_index)
                anchors[anchor] = {
                    "text": text,
                    "rowspan": 1,
                    "colspan": 1,
                }
                owner[anchor] = anchor
                continue
            if token == "lcel":
                anchor = owner.get((row_index, column_index - 1))
            elif token == "ucel":
                anchor = owner.get((row_index - 1, column_index))
            else:
                anchor = owner.get(
                    (row_index, column_index - 1),
                    owner.get((row_index - 1, column_index)),
                )
            if anchor is None:
                anchor = (row_index, column_index)
                anchors[anchor] = {
                    "text": text,
                    "rowspan": 1,
                    "colspan": 1,
                }
            owner[(row_index, column_index)] = anchor
            info = anchors[anchor]
            info["rowspan"] = max(
                info["rowspan"],
                row_index - anchor[0] + 1,
            )
            info["colspan"] = max(
                info["colspan"],
                column_index - anchor[1] + 1,
            )

    pieces = ["<table>"]
    for row_index in range(len(grid)):
        pieces.append("<tr>")
        for column_index in range(width):
            anchor = owner[(row_index, column_index)]
            if anchor != (row_index, column_index):
                continue
            info = anchors[anchor]
            attributes = ""
            if info["rowspan"] > 1:
                attributes += f' rowspan="{info["rowspan"]}"'
            if info["colspan"] > 1:
                attributes += f' colspan="{info["colspan"]}"'
            pieces.append(
                f"<td{attributes}>"
                f"{html.escape(info['text'].strip(), quote=True)}</td>"
            )
        pieces.append("</tr>")
    pieces.append("</table>")
    return "".join(pieces)


_OTSL_TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")


def _parse_otsl_rows(content: str) -> list[list[tuple[str, str]]]:
    matches = list(_OTSL_TOKEN.finditer(content))
    rows: list[list[tuple[str, str]]] = [[]]
    for index, match in enumerate(matches):
        token = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        text = content[match.end() : end]
        if token == "nl":
            if rows[-1]:
                rows.append([])
            continue
        rows[-1].append((token, text))
    return [row for row in rows if row]


# Preprocessor configuration


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


def apply_pixel_overrides(
    cfg: dict,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict:
    """Return a copied preprocessor config with validated pixel overrides."""
    effective = dict(cfg)
    effective_min_pixels = (
        int(effective["min_pixels"])
        if min_pixels is None
        else int(min_pixels)
    )
    effective_max_pixels = (
        int(effective["max_pixels"])
        if max_pixels is None
        else int(max_pixels)
    )
    if effective_min_pixels <= 0:
        raise ValueError("preprocessor min_pixels override must be positive")
    if effective_max_pixels <= 0:
        raise ValueError("preprocessor max_pixels override must be positive")
    if effective_min_pixels > effective_max_pixels:
        raise ValueError(
            "preprocessor min_pixels must not exceed max_pixels: "
            f"min_pixels={effective_min_pixels}, "
            f"max_pixels={effective_max_pixels}"
        )
    effective["min_pixels"] = effective_min_pixels
    effective["max_pixels"] = effective_max_pixels
    return effective
