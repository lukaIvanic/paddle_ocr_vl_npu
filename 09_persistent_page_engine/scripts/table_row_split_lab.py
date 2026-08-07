#!/usr/bin/env python3
"""Create visual CPU-only table-row split proposals for review."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SplitProposal:
    name: str
    boundaries: tuple[int, ...]
    diagnostics: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/OmniDocBench.json"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--table-records",
        type=Path,
        default=Path(
            "tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/"
            "client/tables.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/09_persistent_page_engine/table_row_split_lab"),
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--contact-sheet-size", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _nearest_rank(values: list[int], q: float) -> int:
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def representative_records(records: list[dict], count: int) -> list[dict]:
    """Select by output length and geometry; never inspect GT table structure."""

    ordered = sorted(records, key=lambda record: int(record["output_tokens"]))
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0]
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(record: dict) -> None:
        request_id = str(record["request_id"])
        if request_id not in seen:
            seen.add(request_id)
            chosen.append(record)

    for q in quantiles:
        token = _nearest_rank([int(item["output_tokens"]) for item in ordered], q)
        add(min(ordered, key=lambda item: abs(int(item["output_tokens"]) - token)))

    # Add geometry extremes because narrow, wide, and tall tables fail differently.
    geometry = sorted(
        records,
        key=lambda record: record["crop_size"][0] / max(1, record["crop_size"][1]),
    )
    for record in (geometry[0], geometry[-1]):
        add(record)

    area = sorted(records, key=lambda record: record["crop_size"][0] * record["crop_size"][1])
    for record in (area[0], area[-1]):
        add(record)

    if len(chosen) < count:
        stride = max(1, len(ordered) // max(1, count - len(chosen)))
        for record in ordered[::stride]:
            add(record)
            if len(chosen) >= count:
                break
    return chosen[:count]


def load_crop(record: dict, images_dir: Path) -> Image.Image:
    with Image.open(images_dir / record["page_name"]) as page:
        page = page.convert("RGB")
        return page.crop(tuple(record["bbox_xyxy"]))


def trim_blank_margin(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Remove uniform light margins while retaining a small safety border."""

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    foreground = gray < 235
    row_has_content = np.count_nonzero(foreground, axis=1) >= max(
        3, round(image.width * 0.001)
    )
    column_has_content = np.count_nonzero(foreground, axis=0) >= max(
        3, round(image.height * 0.001)
    )
    ys = np.flatnonzero(row_has_content)
    xs = np.flatnonzero(column_has_content)
    if not len(xs) or not len(ys):
        return image, (0, 0, image.width, image.height)
    margin = max(2, round(min(image.size) * 0.01))
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    bottom = min(image.height, int(ys.max()) + margin + 1)
    box = (left, top, right, bottom)
    return image.crop(box), box


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def binarize(gray: np.ndarray) -> np.ndarray:
    block = _odd(min(61, max(21, min(gray.shape) // 20)))
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        13,
    )


def text_binarize(gray: np.ndarray) -> np.ndarray:
    """Extract dark glyphs without turning colored cell fills into foreground."""

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, global_ink = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
    )
    _, local_ink = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return cv2.bitwise_or(global_ink, local_ink)


def line_masks(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = binary.shape
    horizontal = np.zeros_like(binary)
    vertical = np.zeros_like(binary)
    for divisor in (8, 14, 24):
        kernel_width = max(12, width // divisor)
        opened = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
        )
        horizontal = cv2.bitwise_or(horizontal, opened)
    for divisor in (8, 14, 24):
        kernel_height = max(12, height // divisor)
        opened = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height)),
        )
        vertical = cv2.bitwise_or(vertical, opened)
    horizontal = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 120), 1)),
    )
    return horizontal, vertical


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _component_height(text_mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    height, width = text_mask.shape
    heights = [
        int(stats[index, cv2.CC_STAT_HEIGHT])
        for index in range(1, count)
        if 2 <= stats[index, cv2.CC_STAT_HEIGHT] <= max(3, height // 8)
        and 2 <= stats[index, cv2.CC_STAT_WIDTH] <= max(3, width // 3)
        and stats[index, cv2.CC_STAT_AREA] >= 3
    ]
    return float(np.median(heights)) if heights else max(4.0, height / 80.0)


def _nms_boundaries(candidates: Iterable[int], min_distance: int, height: int) -> tuple[int, ...]:
    selected: list[int] = []
    for value in sorted(max(1, min(height - 1, int(item))) for item in candidates):
        if not selected or value - selected[-1] >= min_distance:
            selected.append(value)
        elif abs(value - selected[-1]) < min_distance:
            selected[-1] = (selected[-1] + value) // 2
    return tuple([0, *selected, height])


def _scored_nms_boundaries(
    candidates: Iterable[tuple[int, float]],
    min_distance: int,
    height: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    for value, _ in sorted(candidates, key=lambda item: item[1], reverse=True):
        value = max(1, min(height - 1, int(value)))
        if all(abs(value - existing) >= min_distance for existing in selected):
            selected.append(value)
    return tuple([0, *sorted(selected), height])


def ruled_split(binary: np.ndarray, horizontal: np.ndarray) -> SplitProposal:
    height, width = binary.shape
    coverage = np.count_nonzero(horizontal, axis=1) / max(1, width)
    smooth_window = _odd(max(3, height // 500))
    smoothed = cv2.GaussianBlur(coverage.astype(np.float32), (1, smooth_window), 0).reshape(-1)
    threshold = max(0.10, min(0.45, float(np.percentile(smoothed, 98)) * 0.42))
    peaks = [
        (start + end - 1) // 2
        for start, end in _runs(smoothed >= threshold)
    ]
    char_height = _component_height(cv2.bitwise_and(binary, cv2.bitwise_not(horizontal)))
    edge_margin = max(2, round(char_height * 1.2))
    peaks = [point for point in peaks if edge_margin < point < height - edge_margin]
    boundaries = _nms_boundaries(peaks, max(2, round(char_height * 0.55)), height)
    return SplitProposal(
        name="ruled",
        boundaries=boundaries,
        diagnostics={
            "line_threshold": threshold,
            "character_height": char_height,
            "edge_margin": edge_margin,
            "interior_boundaries": max(0, len(boundaries) - 2),
        },
    )


def _text_component_mask(
    text_binary: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    """Remove rules and large filled regions while retaining character blobs."""

    height, width = text_binary.shape
    all_lines = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    source = cv2.bitwise_and(text_binary, cv2.bitwise_not(all_lines))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not (1 <= component_width <= max(4, width // 3)):
            continue
        if not (2 <= component_height <= max(4, height // 7)):
            continue
        if not (2 <= area <= max(16, width * height // 80)):
            continue
        keep[index] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def whitespace_candidates(
    text_binary: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> tuple[list[int], float, np.ndarray]:
    height, width = text_binary.shape
    text = _text_component_mask(text_binary, horizontal, vertical)
    char_height = _component_height(text)
    connected = cv2.dilate(
        text,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, round(char_height * 0.8)), max(1, round(char_height * 0.10))),
        ),
    )
    occupancy = np.count_nonzero(connected, axis=1) / max(1, width)
    smooth_size = _odd(max(3, round(char_height * 0.35)))
    smooth = cv2.GaussianBlur(
        occupancy.astype(np.float32), (1, smooth_size), 0
    ).reshape(-1)
    nonzero = smooth[smooth > 0]
    baseline = float(np.percentile(nonzero, 35)) if len(nonzero) else 0.0
    threshold = max(0.0015, min(0.025, baseline * 0.40))
    minimum_gap = max(2, round(char_height * 0.40))
    candidates = [
        (start + end - 1) // 2
        for start, end in _runs(smooth <= threshold)
        if end - start >= minimum_gap and start > 0 and end < height
    ]
    return candidates, char_height, smooth


def whitespace_split(
    text_binary: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> SplitProposal:
    height, _ = text_binary.shape
    candidates, char_height, smooth = whitespace_candidates(
        text_binary, horizontal, vertical
    )
    boundaries = _nms_boundaries(candidates, max(3, round(char_height * 1.05)), height)
    return SplitProposal(
        name="whitespace",
        boundaries=boundaries,
        diagnostics={
            "character_height": char_height,
            "minimum_occupancy": float(smooth.min(initial=0.0)),
            "interior_boundaries": max(0, len(boundaries) - 2),
        },
    )


def row_edge_split(rgb: np.ndarray, text_binary: np.ndarray) -> SplitProposal:
    """Detect boundaries that change color or intensity across much of a row."""

    height, width = text_binary.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    delta = np.mean(np.abs(np.diff(lab, axis=0)), axis=2)
    coverage = np.mean(delta >= 6.0, axis=1).astype(np.float32)
    char_height = _component_height(text_binary)
    smooth_size = _odd(max(3, round(char_height * 0.20)))
    smooth = cv2.GaussianBlur(coverage, (1, smooth_size), 0).reshape(-1)
    positive = smooth[smooth > 0]
    high = float(np.percentile(positive, 95)) if len(positive) else 0.0
    threshold = max(0.06, min(0.45, high * 0.42))
    candidates = []
    for start, end in _runs(smooth >= threshold):
        local = smooth[start:end]
        peak = start + int(np.argmax(local))
        candidates.append((peak + 1, float(local.max())))
    edge_margin = max(2, round(char_height * 1.2))
    candidates = [
        (point, score)
        for point, score in candidates
        if edge_margin < point < height - edge_margin
    ]
    boundaries = _scored_nms_boundaries(
        candidates,
        max(3, round(char_height * 1.15)),
        height,
    )
    return SplitProposal(
        name="row_edge",
        boundaries=boundaries,
        diagnostics={
            "character_height": char_height,
            "coverage_threshold": threshold,
            "edge_margin": edge_margin,
            "interior_boundaries": max(0, len(boundaries) - 2),
            "width": width,
        },
    )


def select_split(
    ruled: SplitProposal,
    whitespace: SplitProposal,
    row_edge: SplitProposal,
) -> SplitProposal:
    """Select a natural-row proposal from image-only structural evidence."""

    ruled_rows = len(ruled.boundaries) - 1
    whitespace_rows = len(whitespace.boundaries) - 1
    edge_rows = len(row_edge.boundaries) - 1
    if whitespace_rows <= 1 and edge_rows >= max(4, ruled_rows * 2):
        source = row_edge
        reason = "colored_or_filled_rows"
    elif ruled_rows >= 3 and whitespace_rows <= math.ceil(ruled_rows * 1.35):
        source = ruled
        reason = "consistent_explicit_rules"
    elif whitespace_rows >= max(2, ruled_rows * 2):
        source = whitespace
        reason = "borderless_or_sparse_rules"
    elif abs(edge_rows - whitespace_rows) <= max(2, round(whitespace_rows * 0.20)):
        source = whitespace
        reason = "edge_whitespace_agreement"
    else:
        source = whitespace if whitespace_rows >= ruled_rows else ruled
        reason = "conservative_fallback"
    diagnostics = dict(source.diagnostics)
    diagnostics.update(
        {
            "selected_source": source.name,
            "selection_reason": reason,
            "ruled_rows": ruled_rows,
            "whitespace_rows": whitespace_rows,
            "row_edge_rows": edge_rows,
        }
    )
    return SplitProposal("selected", source.boundaries, diagnostics)


def hybrid_split(
    text_binary: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    ruled: SplitProposal,
) -> SplitProposal:
    height, width = text_binary.shape
    whitespace, char_height, occupancy = whitespace_candidates(
        text_binary, horizontal, vertical
    )
    line_points = list(ruled.boundaries[1:-1])
    candidates = list(line_points)
    line_margin = max(2, round(char_height * 0.7))
    ruled_intervals = np.diff([0, *line_points, height])
    positive_intervals = ruled_intervals[ruled_intervals > 0]
    typical_ruled_height = (
        float(np.percentile(positive_intervals, 40))
        if len(positive_intervals)
        else 0.0
    )

    # A whitespace valley is useful when it is not merely the empty area around
    # an already-detected rule and does not make an implausibly short row.
    for point in whitespace:
        if any(abs(point - line) <= line_margin for line in line_points):
            continue
        if line_points:
            enclosing = next(
                (
                    right - left
                    for left, right in zip(
                        [0, *line_points], [*line_points, height]
                    )
                    if left < point < right
                ),
                0,
            )
            needs_subdivision = enclosing > max(
                typical_ruled_height * 1.75,
                char_height * 4.0,
            )
            if not needs_subdivision:
                continue
        candidates.append(point)

    boundaries = list(
        _nms_boundaries(candidates, max(3, round(char_height * 1.15)), height)
    )
    minimum_row = max(4, round(char_height * 1.35))
    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        for index in range(1, len(boundaries)):
            if boundaries[index] - boundaries[index - 1] >= minimum_row:
                continue
            # Preserve an explicit strong rule when possible. Otherwise remove
            # the boundary that lies in the denser text region.
            left = boundaries[index - 1]
            right = boundaries[index]
            if left == 0:
                del boundaries[index]
            elif right == height:
                del boundaries[index - 1]
            else:
                left_score = occupancy[min(len(occupancy) - 1, left)]
                right_score = occupancy[min(len(occupancy) - 1, right)]
                del boundaries[index - 1 if left_score > right_score else index]
            changed = True
            break

    mode = "ruled+whitespace" if line_points else "whitespace-only"
    return SplitProposal(
        name="hybrid",
        boundaries=tuple(boundaries),
        diagnostics={
            "mode": mode,
            "character_height": char_height,
            "line_boundaries": len(line_points),
            "whitespace_candidates": len(whitespace),
            "typical_ruled_height": typical_ruled_height,
            "rows": max(1, len(boundaries) - 1),
            "width": width,
        },
    )


def _scale_proposal(
    proposal: SplitProposal,
    source_height: int,
    target_height: int,
    scale: float,
) -> SplitProposal:
    if scale == 1.0:
        return proposal
    boundaries = tuple(
        0 if y == 0 else target_height if y == source_height else round(y / scale)
        for y in proposal.boundaries
    )
    diagnostics = dict(proposal.diagnostics)
    diagnostics["detection_scale"] = scale
    return SplitProposal(proposal.name, boundaries, diagnostics)


def analyze(image: Image.Image) -> tuple[SplitProposal, ...]:
    max_detection_dimension = 1800
    scale = min(1.0, max_detection_dimension / max(image.size))
    detection_image = (
        image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        if scale < 1.0
        else image
    )
    rgb = np.asarray(detection_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = binarize(gray)
    text_binary = text_binarize(gray)
    horizontal, vertical = line_masks(binary)
    ruled = ruled_split(binary, horizontal)
    whitespace = whitespace_split(text_binary, horizontal, vertical)
    row_edge = row_edge_split(rgb, text_binary)
    hybrid = hybrid_split(text_binary, horizontal, vertical, ruled)
    selected = select_split(ruled, whitespace, row_edge)
    return tuple(
        _scale_proposal(
            proposal,
            detection_image.height,
            image.height,
            scale,
        )
        for proposal in (ruled, whitespace, row_edge, hybrid, selected)
    )


COLORS = {
    "ruled": (220, 40, 40),
    "whitespace": (35, 110, 220),
    "row_edge": (170, 80, 190),
    "hybrid": (15, 150, 80),
    "selected": (230, 125, 20),
}


def draw_overlay(image: Image.Image, proposal: SplitProposal) -> Image.Image:
    result = image.convert("RGBA")
    tint = Image.new("RGBA", result.size, (0, 0, 0, 0))
    tint_draw = ImageDraw.Draw(tint)
    base = COLORS[proposal.name]
    for index, (top, bottom) in enumerate(zip(proposal.boundaries, proposal.boundaries[1:])):
        if index % 2 == 0:
            tint_draw.rectangle((0, top, result.width, bottom), fill=(*base, 22))
    result = Image.alpha_composite(result, tint)
    draw = ImageDraw.Draw(result)
    width = max(2, round(result.width / 500))
    for index, y in enumerate(proposal.boundaries[1:-1], start=1):
        draw.line((0, y, result.width, y), fill=(*base, 255), width=width)
        draw.rectangle((0, max(0, y - 10), 48, min(result.height, y + 10)), fill=(255, 255, 255, 220))
        draw.text((3, max(0, y - 8)), str(index), fill=(*base, 255), font=ImageFont.load_default())
    return result.convert("RGB")


def fit_panel(image: Image.Image, width: int = 380, height: int = 320) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return panel


def sample_panel(record: dict, image: Image.Image, proposals: tuple[SplitProposal, ...]) -> Image.Image:
    panel_width, panel_height, header = 300, 320, 48
    result = Image.new(
        "RGB", (panel_width * (len(proposals) + 1), panel_height + header), "white"
    )
    draw = ImageDraw.Draw(result)
    gt_rows = len(re.findall(r"<tr\b", str(record.get("gt_html", "")), re.IGNORECASE))
    title = (
        f"{record['request_id']}  crop={image.width}x{image.height}  "
        f"output_tokens={record['output_tokens']}  gt_rows(eval-only)={gt_rows}"
    )
    draw.text((8, 6), title, fill="black", font=ImageFont.load_default())
    panels = [
        ("original", image),
        *[
            (
                f"{proposal.name} rows={len(proposal.boundaries) - 1}",
                draw_overlay(image, proposal),
            )
            for proposal in proposals
        ],
    ]
    for index, (label, panel_image) in enumerate(panels):
        fitted = fit_panel(panel_image, panel_width, panel_height)
        result.paste(fitted, (index * panel_width, header))
        draw.text((index * panel_width + 8, header + 6), label, fill="black", font=ImageFont.load_default())
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def write_contact_sheets(panels: list[tuple[str, Image.Image]], output_dir: Path, count: int) -> list[str]:
    names: list[str] = []
    for start in range(0, len(panels), count):
        group = panels[start : start + count]
        width = max(panel.width for _, panel in group)
        height = sum(panel.height for _, panel in group)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for _, panel in group:
            sheet.paste(panel, (0, y))
            y += panel.height
        name = f"contact_sheet_{start // count + 1:02d}.png"
        sheet.save(output_dir / name)
        names.append(name)
    return names


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.output_dir / "panels"
    rows_dir = args.output_dir / "hybrid_rows"
    panels_dir.mkdir(exist_ok=True)
    rows_dir.mkdir(exist_ok=True)

    records = read_jsonl(args.table_records)
    by_id = {record["request_id"]: record for record in records}
    selected = (
        [by_id[request_id] for request_id in args.request_id]
        if args.request_id
        else representative_records(records, args.samples)
    )

    manifest: list[dict] = []
    panels: list[tuple[str, Image.Image]] = []
    for index, record in enumerate(selected, start=1):
        raw_image = load_crop(record, args.images_dir)
        image, trim_box = trim_blank_margin(raw_image)
        proposals = analyze(image)
        name = f"{index:02d}_{safe_name(record['request_id'])}"
        panel = sample_panel(record, image, proposals)
        panel.save(panels_dir / f"{name}.png")
        panels.append((name, panel))

        selected_proposal = proposals[-1]
        row_output = rows_dir / name
        row_output.mkdir(exist_ok=True)
        for row_index, (top, bottom) in enumerate(
            zip(selected_proposal.boundaries, selected_proposal.boundaries[1:]),
            start=1,
        ):
            image.crop((0, top, image.width, bottom)).save(
                row_output / f"row_{row_index:03d}_y{top}-{bottom}.png"
            )

        manifest.append(
            {
                "request_id": record["request_id"],
                "page_name": record["page_name"],
                "bbox_xyxy": record["bbox_xyxy"],
                "raw_crop_size": list(raw_image.size),
                "trim_box_in_raw_crop": list(trim_box),
                "crop_size": list(image.size),
                "output_tokens": record["output_tokens"],
                "gt_rows_eval_only": len(
                    re.findall(
                        r"<tr\b",
                        str(record.get("gt_html", "")),
                        re.IGNORECASE,
                    )
                ),
                "panel": f"panels/{name}.png",
                "strategies": {
                    proposal.name: {
                        "boundaries": list(proposal.boundaries),
                        "rows": len(proposal.boundaries) - 1,
                        "diagnostics": proposal.diagnostics,
                    }
                    for proposal in proposals
                },
            }
        )
        print(
            f"sample={index}/{len(selected)} request={record['request_id']} "
            + " ".join(
                f"{proposal.name}_rows={len(proposal.boundaries) - 1}"
                for proposal in proposals
            ),
            flush=True,
        )

    sheets = write_contact_sheets(panels, args.output_dir, args.contact_sheet_size)
    payload = {
        "input_contract": {
            "table_source": "OmniDocBench v1.6 ground-truth table boxes",
            "splitter_inputs": "crop pixels only",
            "ground_truth_structure_used_by_splitter": False,
        },
        "samples": manifest,
        "contact_sheets": sheets,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={args.output_dir} contact_sheets={len(sheets)}", flush=True)


if __name__ == "__main__":
    main()
