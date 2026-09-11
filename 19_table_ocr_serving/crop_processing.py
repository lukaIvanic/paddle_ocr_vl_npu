"""PaddleOCR-VL model input preprocessing and prompt construction."""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import Any
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


# Fixed crop input contract; one image, 14x14 patches, 2x2 spatial merging.
PATCH_SIZE = 14
MERGE_SIZE = 2
MIN_PIXELS = 28224
MAX_PIXELS = 802816


# Crop preprocessing


def preprocess_pil_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize with Kornia-RS and produce uint8 patches for NPU normalization."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    grid_t, grid_h, grid_w = image_grid_thw_from_size(width, height)
    from kornia_rs.image import Image as KorniaImage

    array = KorniaImage.fromarray(np.asarray(image)).resize(
        grid_w * PATCH_SIZE,
        grid_h * PATCH_SIZE,
        "bicubic",
    ).data
    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
    patches = patches.reshape(
        grid_t, 1, channel, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel, PATCH_SIZE, PATCH_SIZE,
    )
    return (
        torch.from_numpy(flatten_patches),
        torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
    )


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(image_path) as image:
        return preprocess_pil_image(image)


def image_grid_thw_from_size(width: int, height: int) -> tuple[int, int, int]:
    """Compute the single-image patch grid without allocating pixel tensors."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    resized_height, resized_width = smart_resize(height, width)
    return 1, resized_height // PATCH_SIZE, resized_width // PATCH_SIZE


def smart_resize(
    height: int,
    width: int,
) -> tuple[int, int]:
    factor = PATCH_SIZE * MERGE_SIZE
    min_pixels, max_pixels = MIN_PIXELS, MAX_PIXELS
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
