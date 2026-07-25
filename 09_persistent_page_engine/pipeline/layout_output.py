"""Owned PaddleOCR-VL page assembly and Markdown output.

The behavior in this module follows the Apache-2.0 PaddleX PaddleOCR-VL 1.6
result contract, but has no PaddleX runtime dependency.  Experiment 09 only
supports the fixed production policy used by the OmniDocBench runner:
document preprocessing, chart recognition, seal recognition, and OCR inside
image blocks are disabled.
"""

from __future__ import annotations

import html
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .layout_postprocess import IMAGE_LABELS, SKIP_ORDER_LABELS


MARKDOWN_IGNORE_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}
VISIBLE_IMAGE_LABELS = set(IMAGE_LABELS) | {"chart", "seal"}


@dataclass
class OwnedPageBlock:
    label: str
    bbox: list[int]
    content: str = ""
    group_id: int | None = None
    polygon_points: Any = None
    image: dict[str, Any] | None = None


def construct_image_path(label: str, box: Any) -> str:
    x_min, y_min, x_max, y_max = [int(value) for value in box]
    return (
        f"imgs/img_in_{label}_box_"
        f"{x_min}_{y_min}_{x_max}_{y_max}.jpg"
    )


def gather_document_images(
    image_bgr: np.ndarray,
    layout_boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror PaddleX's image inventory from detector boxes."""
    height, width = image_bgr.shape[:2]
    images: list[dict[str, Any]] = []
    for box in layout_boxes:
        if box["label"] not in IMAGE_LABELS:
            continue
        x_min, y_min, x_max, y_max = [
            int(value) for value in box["coordinate"]
        ]
        x_min = max(0, min(x_min, width))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height))
        y_max = max(0, min(y_max, height))
        if x_max <= x_min or y_max <= y_min:
            continue
        crop_rgb = np.ascontiguousarray(
            image_bgr[y_min:y_max, x_min:x_max, :3][:, :, ::-1]
        )
        images.append(
            {
                "path": construct_image_path(
                    box["label"],
                    box["coordinate"],
                ),
                "img": Image.fromarray(crop_rgb),
                "label": box["label"],
                "coordinate": (x_min, y_min, x_max, y_max),
                "score": box["score"],
            }
        )
    return images


def _paint_token(
    image: np.ndarray,
    box: Any,
    token: str,
) -> np.ndarray:
    def optimal_scale(square_size: int) -> tuple[float, int, int]:
        left, right = 0.2, 10.0
        selected = left
        width = height = 0
        while right - left > 1e-2:
            middle = (left + right) / 2
            (width, height), _ = cv2.getTextSize(
                token,
                cv2.FONT_HERSHEY_SIMPLEX,
                middle,
                thickness=1,
            )
            if width < square_size * 0.9 and height < square_size * 0.9:
                selected = middle
                left = middle
            else:
                right = middle
        return selected, width, height

    x_min, y_min, x_max, y_max = [int(value) for value in box]
    box_width = x_max - x_min
    box_height = y_max - y_min
    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        color=(255, 255, 255),
        thickness=-1,
    )
    scale, text_width, text_height = optimal_scale(
        min(box_width, box_height)
    )
    thickness = max(1, math.floor(scale * 4))
    text_x = x_min + (box_width - text_width) // 2
    text_y = y_min + (box_height + text_height) // 2
    cv2.putText(
        image,
        token,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return image


def tokenize_table_figures(
    table_image: np.ndarray,
    table_box: Any,
    document_images: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, str], set[str]]:
    """Replace figures fully contained by a table with deterministic tokens."""
    excluded_digits = {"0", "1", "9"}
    token_numbers: list[int] = []
    candidate = 0
    while len(token_numbers) < len(document_images):
        if not (set(str(candidate)) & excluded_digits):
            token_numbers.append(candidate)
        candidate += 1
    random.Random(1024).shuffle(token_numbers)

    table_x_min, table_y_min, table_x_max, table_y_max = table_box
    output = table_image.copy()
    token_map: dict[str, str] = {}
    dropped: set[str] = set()
    for index, figure in enumerate(document_images):
        x_min, y_min, x_max, y_max = figure["coordinate"]
        if not (
            x_min >= table_x_min
            and y_min >= table_y_min
            and x_max <= table_x_max
            and y_max <= table_y_max
        ):
            continue
        dropped.add(figure["path"])
        if min(x_max - x_min, y_max - y_min) < 25:
            continue
        token = f"[F{token_numbers[index]}]"
        output = _paint_token(
            output,
            [
                x_min - table_x_min,
                y_min - table_y_min,
                x_max - table_x_min,
                y_max - table_y_min,
            ],
            token,
        )
        token_map[token] = figure["path"]
    return output, token_map, dropped


def _shortest_repeating_substring(value: str) -> str | None:
    for length in range(1, len(value) // 2 + 1):
        if len(value) % length == 0:
            candidate = value[:length]
            if candidate * (len(value) // length) == value:
                return candidate
    return None


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


def untokenize_table_figures(
    content: str,
    figure_token_map: dict[str, str],
    image_blocks: dict[str, OwnedPageBlock],
) -> str:
    def replace(match: re.Match[str]) -> str:
        token = f"[F{match.group(1)}]"
        image_path = figure_token_map.get(token)
        block = image_blocks.get(image_path or "")
        if block is None or image_path is None:
            return match.group(0)
        image_tag = (
            f'<img src="{_collapse_soft_newlines(image_path)}" '
            'alt="Image"" />'
        )
        if block.content:
            image_tag += f"\n\n{block.content}\n\n"
        return image_tag

    return re.sub(r"\[F(\d+)\]", replace, content)


def assemble_page_blocks(
    blocks: list[dict[str, Any]],
    recognition: dict[int, str],
    *,
    figure_token_maps: dict[int, dict[str, str]],
    dropped_figure_paths: set[str],
) -> list[OwnedPageBlock]:
    output: list[OwnedPageBlock] = []
    image_blocks: dict[str, OwnedPageBlock] = {}
    tables: list[tuple[OwnedPageBlock, dict[str, str]]] = []
    for index, source in enumerate(blocks):
        label = source["label"]
        block = OwnedPageBlock(
            label=label,
            bbox=[int(value) for value in source["box"]],
            content=normalize_recognition_text(
                label,
                recognition.get(index, ""),
            ),
            group_id=source.get("group_id"),
            polygon_points=(
                np.asarray(source["polygon_points"]).tolist()
                if source.get("polygon_points") is not None
                else None
            ),
        )
        if label == "table":
            tables.append((block, figure_token_maps.get(index, {})))
        if label in VISIBLE_IMAGE_LABELS and source["img"] is not None:
            path = construct_image_path(label, source["box"])
            image_blocks[path] = block
            if path not in dropped_figure_paths:
                rgb = np.ascontiguousarray(source["img"][:, :, ::-1])
                block.image = {"path": path, "img": Image.fromarray(rgb)}
            else:
                continue
        output.append(block)
    for block, token_map in tables:
        block.content = untokenize_table_figures(
            block.content,
            token_map,
            image_blocks,
        )
    return output


def _collapse_soft_newlines(value: str) -> str:
    return value.replace("-\n", "").replace("\n", " ")


_TITLE_PATTERN = re.compile(
    r"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|"
    r"[\(\（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零"
    r"壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|"
    r"[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+"
    r"[、\.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))"
    r"(\s*)(.*)$"
)


def _format_title(block: OwnedPageBlock) -> str:
    title = block.content
    match = _TITLE_PATTERN.match(title)
    if match:
        title = match.group(1).strip() + " " + match.group(3).lstrip()
    title = title.rstrip(".")
    level = title.count(".") + 1 if "." in title else 1
    return _collapse_soft_newlines(f"#{'#' * level} {title}")


def _format_first_line(
    block: OwnedPageBlock,
    templates: list[str],
    format_func: Any,
    splitter: str,
) -> str:
    lines = block.content.split(splitter)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lower() in templates:
            lines[index] = format_func(line)
        break
    return splitter.join(lines)


def _format_centered(content: str, *, collapse: bool = True) -> str:
    if collapse:
        content = _collapse_soft_newlines(content)
    return f'<div style="text-align: center;">{content}</div>\n'


def _format_image(
    block: OwnedPageBlock,
    *,
    page_width: int,
) -> str:
    if block.image is None:
        return ""
    scale = int((block.bbox[2] - block.bbox[0]) / page_width * 100)
    return _format_centered(
        f'<img src="{_collapse_soft_newlines(block.image["path"])}" '
        f'alt="Image" width="{scale}%" />'
    )


def _format_table(block: OwnedPageBlock) -> str:
    content = block.content
    content = content.replace(
        "<table>",
        "<table border=1 style='margin: auto; word-wrap: break-word;'>",
    )
    content = content.replace(
        "<th>",
        "<th style='text-align: center; word-wrap: break-word;'>",
    )
    content = content.replace(
        "<td>",
        "<td style='text-align: center; word-wrap: break-word;'>",
    )
    return "\n" + content


def _handlers(page_width: int) -> dict[str, Any]:
    normalize = lambda block: block.content.replace("\n\n", "\n").replace(
        "\n",
        "\n\n",
    )
    image = lambda block: _format_image(block, page_width=page_width)
    centered_text = lambda block: _format_centered(block.content)
    handlers = {
        "paragraph_title": _format_title,
        "abstract_title": _format_title,
        "reference_title": _format_title,
        "content_title": _format_title,
        "doc_title": lambda block: _collapse_soft_newlines(
            f"# {block.content}"
        ),
        "table_title": centered_text,
        "figure_title": centered_text,
        "chart_title": centered_text,
        "vision_footnote": normalize,
        "text": normalize,
        "ocr": normalize,
        "vertical_text": normalize,
        "reference_content": normalize,
        "abstract": partial(
            _format_first_line,
            templates=["摘要", "abstract"],
            format_func=lambda line: f"## {line}\n",
            splitter=" ",
        ),
        "content": lambda block: block.content.replace("-\n", "  \n").replace(
            "\n",
            "  \n",
        ),
        "image": image,
        "chart": image,
        "formula": lambda block: block.content,
        "display_formula": lambda block: block.content,
        "inline_formula": lambda block: block.content,
        "table": _format_table,
        "reference": partial(
            _format_first_line,
            templates=["参考文献", "references"],
            format_func=lambda line: f"## {line}",
            splitter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": image,
        "spotting": lambda block: block.content,
        "number": lambda block: block.content,
        "footnote": lambda block: block.content,
        "header": lambda block: block.content,
        "header_image": image,
        "footer": lambda block: block.content,
        "footer_image": image,
        "aside_text": lambda block: block.content,
    }
    for label in MARKDOWN_IGNORE_LABELS:
        handlers.pop(label, None)
    return handlers


def page_markdown(
    blocks: list[OwnedPageBlock],
    *,
    page_width: int,
) -> tuple[str, dict[str, Image.Image]]:
    handlers = _handlers(page_width)
    chunks: list[str] = []
    images: dict[str, Image.Image] = {}
    for block in blocks:
        if block.image is not None:
            images[block.image["path"]] = block.image["img"]
        handler = handlers.get(block.label)
        if handler is not None:
            chunks.append(handler(block))
    return "\n\n".join(chunks), images


class OwnedPageResult:
    """Small result object matching the runner-facing PaddleX contract."""

    def __init__(
        self,
        *,
        input_path: Path,
        width: int,
        height: int,
        blocks: list[OwnedPageBlock],
        document_images: list[dict[str, Any]],
    ) -> None:
        self.data = {
            "input_path": str(input_path),
            "page_index": None,
            "page_count": None,
            "width": int(width),
            "height": int(height),
            "parsing_res_list": blocks,
            "imgs_in_doc": document_images,
        }

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def json(self) -> dict[str, Any]:
        skip_order = set(SKIP_ORDER_LABELS) | MARKDOWN_IGNORE_LABELS
        order = 1
        serialized: list[dict[str, Any]] = []
        for index, block in enumerate(self.data["parsing_res_list"]):
            block_order = None
            if block.label not in skip_order:
                block_order = order
                order += 1
            item = {
                "block_label": block.label,
                "block_content": block.content,
                "block_bbox": block.bbox,
                "block_id": index,
                "block_order": block_order,
                "group_id": (
                    block.group_id
                    if block.group_id is not None
                    else index
                ),
            }
            if block.polygon_points is not None:
                item["block_polygon_points"] = block.polygon_points
            serialized.append(item)
        return {
            "res": {
                "input_path": self.data["input_path"],
                "page_index": None,
                "page_count": None,
                "width": self.data["width"],
                "height": self.data["height"],
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_layout_detection": True,
                    "use_chart_recognition": False,
                    "use_seal_recognition": False,
                    "use_ocr_for_image_block": False,
                    "format_block_content": False,
                    "merge_layout_blocks": True,
                    "markdown_ignore_labels": sorted(
                        MARKDOWN_IGNORE_LABELS
                    ),
                },
                "parsing_res_list": serialized,
            }
        }

    def save_to_markdown(self, save_path: str) -> None:
        base = Path(save_path)
        output = (
            base
            if base.suffix.lower() in {".md", ".markdown", ".mdown", ".mkd"}
            else base / f"{Path(self.data['input_path']).stem}.md"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        markdown, images = page_markdown(
            self.data["parsing_res_list"],
            page_width=self.data["width"],
        )
        output.write_text(markdown, encoding="utf-8")
        for relative_path, image in images.items():
            image_path = output.parent / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
        for item in self.data["imgs_in_doc"]:
            image_path = output.parent / item["path"]
            if image_path.exists():
                continue
            image_path.parent.mkdir(parents=True, exist_ok=True)
            item["img"].save(image_path)
