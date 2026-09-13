"""Crop inputs before inference, and recognition text after inference.

Serving first calls preprocess_pil_image for uint8 patches and an image grid,
then prepare_prompt_tokens for the prompt tokens. Their implementation details follow.
NPU normalization and model execution live in the runtime, not this module.
After inference, the table endpoint normalizes math delimiters, then calls
convert_otsl_to_html to render the decoded table structure; its parser follows.
"""

from __future__ import annotations

import html
import re
from typing import Any
import math

import numpy as np
import torch
from kornia_rs.image import Image as KorniaImage
from PIL import Image
from tokenizers import Tokenizer


IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"



PATCH_SIZE = 14
MERGE_SIZE = 2 # Vision tokens from encoder are 2x2 pooled for decoder. E.g. 4096 tokens -> 1024 tokens.
MIN_PIXELS = 28224
MAX_PIXELS = 802816


# Input preparation


def preprocess_pil_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize with Kornia-RS and produce uint8 patches for "normalization" on NPU."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    grid_h, grid_w = image_grid_hw_from_size(width, height)

    array = KorniaImage.fromarray(np.asarray(image)).resize(
        grid_w * PATCH_SIZE,
        grid_h * PATCH_SIZE,
        "bicubic",
    ).data
    # Split the RGB image into a grid of 14x14 patches.
    patches = array.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3)
    # Order patches by row, then column, with channels first.
    patches = patches.transpose(0, 2, 4, 1, 3)
    # Flatten the grid into the vision encoder's patch sequence.
    patches = patches.reshape(grid_h * grid_w, 3, PATCH_SIZE, PATCH_SIZE)
    return (
        torch.from_numpy(patches),
        torch.tensor([[1, grid_h, grid_w]], dtype=torch.long),
    )


def prepare_prompt_tokens(
    tokenizer: Tokenizer,
    image_grid_thw: torch.Tensor,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Each 14x14 image patch is one input token for the vision encoder.
    vision_input_token_count = int(image_grid_thw[0].prod().item())
    # After vision encoding, the projector merges each 2x2 token group into
    # one image token for the language model: e.g. 4096 / (2 * 2) = 1024.
    # We only calculate that count here; no visual features are processed.
    image_token_count = vision_input_token_count // (MERGE_SIZE * MERGE_SIZE)
    # Reserve one prompt placeholder for each projected image token.
    text = build_paddleocr_vl_prompt(
        prompt,
        image_token_count=image_token_count,
    )
    ids = tokenizer.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


# Input implementation details: image geometry, resizing, then prompt text.


def image_grid_hw_from_size(width: int, height: int) -> tuple[int, int]:
    """Compute the single-image patch grid without allocating pixel tensors."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    resized_height, resized_width = calculate_resized_image_shape(height, width)
    return resized_height // PATCH_SIZE, resized_width // PATCH_SIZE


def calculate_resized_image_shape(
    height: int,
    width: int,
) -> tuple[int, int]:
    """Choose the target image size (height, width)

    One 14x14 image patch becomes one encoder input token. The projector
    later merges tokens in 2x2 groups, so the resized image needs both sides
    to be multiples of 28 pixels. The area must also fit our pixel budget.

    Follow Paddle's resize policy in five steps:
    1. Enlarge sides shorter than 28, scaling the other side proportionally.
    2. Reject aspect ratios above 200:1.
    3. Try rounding both sides to the nearest multiple of 28.
    4. Check that rounded shape against the 28,224 to 802,816 pixel budget.
    5. If outside the budget, scale the PRE-ROUNDING dimensions to the limit,
       then round down when shrinking or up when enlarging. (this follows original PaddleX behavior)

    Step 3 only proposes a shape. Step 5 starts from the dimensions after step 1,
    not that rounded proposal.
    """
    pooled_patch_size = PATCH_SIZE * MERGE_SIZE  # 14 * 2 = 28 pixels

    # 1. Make each side at least 28 pixels, keeping the proportions approximately.
    # Multiply before dividing to preserve Paddle's half-pixel rounding behavior.
    if height < pooled_patch_size:
        width = round((width * pooled_patch_size) / height)
        height = pooled_patch_size

    if width < pooled_patch_size:
        height = round((height * pooled_patch_size) / width)
        width = pooled_patch_size

    # 2. Check the proportions before patch-grid rounding changes them.
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}"
        )

    # 3. Propose a patch-aligned shape; keep height/width unchanged for scaling.
    resized_height = round(height / pooled_patch_size) * pooled_patch_size
    resized_width = round(width / pooled_patch_size) * pooled_patch_size
    # 4. The budget check uses the proposed shape's area, not height * width.
    rounded_pixel_count = resized_height * resized_width

    # 5. Only replace that proposal if it is outside the budget.
    # Scale the pre-rounding height/width together. Scaling both sides changes
    # area by the square of the scale, hence the square root.
    if rounded_pixel_count > MAX_PIXELS:
        scale_divisor = math.sqrt((height * width) / MAX_PIXELS)
        scaled_height = height / scale_divisor
        scaled_width = width / scale_divisor
        # Round DOWN to whole groups so the result does not exceed the maximum.
        resized_height = math.floor(scaled_height / pooled_patch_size) * pooled_patch_size
        resized_width = math.floor(scaled_width / pooled_patch_size) * pooled_patch_size
    elif rounded_pixel_count < MIN_PIXELS:
        scale_multiplier = math.sqrt(MIN_PIXELS / (height * width))
        scaled_height = height * scale_multiplier
        scaled_width = width * scale_multiplier
        # Round UP to whole groups so the result reaches at least the minimum.
        resized_height = math.ceil(scaled_height / pooled_patch_size) * pooled_patch_size
        resized_width = math.ceil(scaled_width / pooled_patch_size) * pooled_patch_size
    return resized_height, resized_width


def build_paddleocr_vl_prompt(prompt: str, *, image_token_count: int) -> str:
    """Build the user message with image placeholders and the recognition (table/formula/text) task."""
    # IMAGE_TOKEN is the literal string "<|IMAGE_PLACEHOLDER|>", not a token ID.
    image_placeholder_text = ""
    for _ in range(image_token_count):
        image_placeholder_text += IMAGE_TOKEN

    return (
        f"{BOS}User: "
        f"{IMAGE_START}{image_placeholder_text}{IMAGE_END}"
        f"{prompt}\n"
        "Assistant:\n"
    )


# Output processing: normalize math notation, then render table structure as HTML.


def normalize_math_delimiters(text: str) -> str:
    r"""Match Paddle's output formatting: \(math\) becomes $ math $, for example.

    This runs after generation; raw_text and native token IDs remain unchanged.
    Preserve the historical replacements, including their spaces and removal
    of existing dollar signs when paired math delimiters are present.
    This does not trim repetitive text.
    """
    if (r"\(" in text and r"\)" in text) or (r"\[" in text and r"\]" in text):
        text = text.replace("$", "")
        text = (
            text.replace(r"\(", " $ ")
            .replace(r"\)", " $")
            .replace(r"\[\[", r"\[")
            .replace(r"\]\]", r"\]")
            .replace(r"\[", " $$ ")
            .replace(r"\]", " $$ ")
        )
    return text


def convert_otsl_to_html(content: str) -> str:
    """Turn the model's table notation (OTSL) into an HTML table.

    The model writes table-cell text, mixed with "markers" for table layout:
    <fcel> starts a non-empty cell, <ecel> an empty cell, and <nl> ends a row.

    For example, "<fcel>Apple<fcel>2<nl>" describes one row with two cells
    which becomes "<table><tr><td>Apple</td><td>2</td></tr></table>".

    A merged table-cell occupies several positions in the table grid. (so not only 1 column
     and 1 row, but can be: 2 columns 1 row, or 1 column 8 rows) <lcel> continues
    the cell to the left, <ucel> the cell above, and <xcel> continues a cell
    spanning both rows and columns. We emit that cell only once, with HTML
    rowspan/colspan attributes telling the browser how many positions it covers.

    Cell text is escaped so characters such as "&" and "<" display as text,
    rather than being interpreted as HTML markup.
    """
    # 1. Parse rows and pad missing positions with empty cells.
    rows = _parse_otsl_rows(content)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    grid = [
        row + [("ecel", "")] * (width - len(row))
        for row in rows
    ]
    # 2. Each position belongs to a cell's origin (its top-left position).
    # Continuation markers extend that cell instead of creating another one.
    cells_by_origin: dict[tuple[int, int], dict[str, Any]] = {}
    origin_by_position: dict[tuple[int, int], tuple[int, int]] = {}
    for row_index, row in enumerate(grid):
        for column_index, (token, text) in enumerate(row):
            origin = None
            if token == "lcel":
                origin = origin_by_position.get((row_index, column_index - 1))
            elif token == "ucel":
                origin = origin_by_position.get((row_index - 1, column_index))
            elif token == "xcel":
                # Preserve the existing left-first, then above fallback.
                origin = origin_by_position.get(
                    (row_index, column_index - 1),
                    origin_by_position.get((row_index - 1, column_index)),
                )
            if origin is None:
                # Filled/empty cells start here. Orphan continuation markers
                # also become standalone cells, as in the existing converter.
                origin = (row_index, column_index)
                cells_by_origin[origin] = {
                    "text": text,
                    "rowspan": 1,
                    "colspan": 1,
                }
            else:
                cell = cells_by_origin[origin]
                cell["rowspan"] = max(cell["rowspan"], row_index - origin[0] + 1)
                cell["colspan"] = max(cell["colspan"], column_index - origin[1] + 1)
            origin_by_position[(row_index, column_index)] = origin

    # 3. Emit each cell once, at its origin; continuation positions emit nothing.
    pieces = ["<table>"]
    for row_index in range(len(grid)):
        pieces.append("<tr>")
        for column_index in range(width):
            origin = origin_by_position[(row_index, column_index)]
            if origin != (row_index, column_index):
                continue
            cell = cells_by_origin[origin]
            attributes = ""
            if cell["rowspan"] > 1:
                attributes += f' rowspan="{cell["rowspan"]}"'
            if cell["colspan"] > 1:
                attributes += f' colspan="{cell["colspan"]}"'
            pieces.append(
                f"<td{attributes}>"
                f"{html.escape(cell['text'].strip(), quote=True)}</td>"
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
