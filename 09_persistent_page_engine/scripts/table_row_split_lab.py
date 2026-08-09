#!/usr/bin/env python3
"""Create visual CPU-only table-row split proposals for review."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
ROW_DRAFT_ORIENTATION_GROUND_TRUTH = (
    EXPERIMENT_ROOT / "accuracy_lab/table_row_orientation_ground_truth.json"
)


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


@lru_cache(maxsize=1)
def row_draft_rotations_cw() -> dict[str, int]:
    payload = json.loads(
        ROW_DRAFT_ORIENTATION_GROUND_TRUTH.read_text(encoding="utf-8")
    )
    rotations = {
        str(request_id): int(degrees)
        for request_id, degrees in payload["rotations_cw"].items()
    }
    invalid = {
        request_id: degrees
        for request_id, degrees in rotations.items()
        if degrees not in (0, 90, 180, 270)
    }
    if invalid:
        raise ValueError(f"invalid row-draft rotations: {invalid}")
    return rotations


def row_draft_rotation_cw(record: dict) -> int:
    return row_draft_rotations_cw().get(str(record["request_id"]), 0)


def orient_row_draft_image(image: Image.Image, record: dict) -> tuple[Image.Image, int]:
    """Orient only the row-draft copy; the whole-table target stays unchanged."""

    rotation_cw = row_draft_rotation_cw(record)
    transpose = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }[rotation_cw]
    if transpose is None:
        return image, rotation_cw
    return image.transpose(transpose), rotation_cw


def trim_blank_margin(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Remove uniform light margins while retaining a small safety border."""

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    foreground = gray < 235
    row_has_content = np.count_nonzero(foreground, axis=1) >= max(
        8, round(image.width * 0.005)
    )
    column_has_content = np.count_nonzero(foreground, axis=0) >= max(
        8, round(image.height * 0.005)
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


def uniform_split(height: int, rows: int = 8) -> SplitProposal:
    """Divide the crop into equal-height horizontal bands."""

    rows = max(1, min(int(rows), height))
    boundaries = tuple(round(index * height / rows) for index in range(rows + 1))
    return SplitProposal(
        name=f"uniform_{rows}",
        boundaries=boundaries,
        diagnostics={
            "requested_rows": int(rows),
            "rows": rows,
            "height": height,
        },
    )


def group_natural_rows(
    proposal: SplitProposal,
    rows: int = 8,
) -> SplitProposal:
    """Group detected logical rows into at most ``rows`` contiguous bands.

    Unlike bounded uniform snapping, this never cuts between two detected
    natural boundaries.  It selects the detected boundary nearest each ideal
    equal-height cut while preserving increasing, non-empty bands.
    """

    boundaries = tuple(int(value) for value in proposal.boundaries)
    natural_rows = max(1, len(boundaries) - 1)
    requested_rows = max(1, int(rows))
    grouped_rows = min(requested_rows, natural_rows)
    if grouped_rows == natural_rows:
        selected = boundaries
    else:
        interior = list(boundaries[1:-1])
        height = boundaries[-1]
        selected_interior: list[int] = []
        lower_index = 0
        for group_index in range(1, grouped_rows):
            remaining = grouped_rows - group_index
            upper_index = len(interior) - remaining
            ideal = group_index * height / grouped_rows
            candidates = range(lower_index, upper_index + 1)
            chosen_index = min(
                candidates,
                key=lambda index: (abs(interior[index] - ideal), index),
            )
            selected_interior.append(interior[chosen_index])
            lower_index = chosen_index + 1
        selected = (boundaries[0], *selected_interior, boundaries[-1])
    diagnostics = dict(proposal.diagnostics)
    diagnostics.update(
        {
            "source": proposal.name,
            "natural_rows": natural_rows,
            "requested_rows": requested_rows,
            "rows": len(selected) - 1,
        }
    )
    return SplitProposal(
        name=f"{proposal.name}_grouped_{requested_rows}",
        boundaries=selected,
        diagnostics=diagnostics,
    )


def _snap_boundary_feature(ink_mask: np.ndarray, y: int) -> dict[str, float | str]:
    """Classify one horizontal cut using the reviewed snap prototype."""

    height, _width = ink_mask.shape
    y = max(0, min(height - 1, int(y)))
    strip = ink_mask[max(0, y - 2) : min(height, y + 3)]
    dark_fraction = float(strip.mean())
    maximum_row_coverage = float(strip.mean(axis=1).max())
    if maximum_row_coverage >= 0.35:
        kind = "horizontal_rule"
    elif dark_fraction <= 0.012 and maximum_row_coverage <= 0.025:
        kind = "blank_gap"
    else:
        kind = "ink_crossing"
    return {
        "kind": kind,
        "dark_fraction": dark_fraction,
        "maximum_row_coverage": maximum_row_coverage,
    }


def snap_uniform_boundaries(
    image: Image.Image,
    proposal: SplitProposal,
) -> SplitProposal:
    """Move uniform cuts to nearby rules or low-ink separator rows."""

    gray = np.asarray(image.convert("L"))
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    ink_mask = binary > 0
    height, _width = ink_mask.shape
    nominal_band_height = height / max(1, len(proposal.boundaries) - 1)
    radius = max(3, min(18, int(round(nominal_band_height * 0.25))))
    snapped = [0]
    details: list[dict[str, float | int | str]] = []

    for index, old_boundary in enumerate(proposal.boundaries[1:-1], start=1):
        old_boundary = int(old_boundary)
        lower = max(snapped[-1] + 3, old_boundary - radius)
        upper = min(height - 3, old_boundary + radius)
        candidates = []
        for y in range(lower, upper + 1):
            feature = _snap_boundary_feature(ink_mask, y)
            normalized_move = abs(y - old_boundary) / max(radius, 1)
            base_score = (
                -1.0 - float(feature["maximum_row_coverage"])
                if feature["kind"] == "horizontal_rule"
                else float(feature["dark_fraction"])
            )
            score = base_score + 0.012 * normalized_move * normalized_move
            candidates.append((score, abs(y - old_boundary), y, feature))

        old_feature = _snap_boundary_feature(ink_mask, old_boundary)
        if not candidates:
            new_boundary = max(snapped[-1] + 1, min(height - 1, old_boundary))
            new_feature = _snap_boundary_feature(ink_mask, new_boundary)
        else:
            best = min(candidates, key=lambda item: (item[0], item[1]))
            old_base_score = (
                -1.0 - float(old_feature["maximum_row_coverage"])
                if old_feature["kind"] == "horizontal_rule"
                else float(old_feature["dark_fraction"])
            )
            if (
                best[3]["kind"] != "horizontal_rule"
                and old_base_score - best[0] < 0.003
            ):
                new_boundary = old_boundary
                new_feature = old_feature
            else:
                new_boundary = int(best[2])
                new_feature = best[3]

        snapped.append(new_boundary)
        details.append(
            {
                "index": index,
                "old_y": old_boundary,
                "new_y": new_boundary,
                "delta_px": new_boundary - old_boundary,
                "old_kind": str(old_feature["kind"]),
                "new_kind": str(new_feature["kind"]),
                "old_dark_fraction": float(old_feature["dark_fraction"]),
                "new_dark_fraction": float(new_feature["dark_fraction"]),
                "search_radius_px": radius,
            }
        )

    snapped.append(height)
    diagnostics = {
        "source": proposal.name,
        "rows": len(snapped) - 1,
        "search_radius_px": radius,
        "changed_boundaries": sum(detail["delta_px"] != 0 for detail in details),
        "ink_crossings_before": sum(
            detail["old_kind"] == "ink_crossing" for detail in details
        ),
        "ink_crossings_after": sum(
            detail["new_kind"] == "ink_crossing" for detail in details
        ),
        "boundary_details": details,
    }
    return SplitProposal(f"{proposal.name}_snapped", tuple(snapped), diagnostics)


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


def uniform_proposals(
    image: Image.Image,
    rows: int,
) -> tuple[SplitProposal, SplitProposal]:
    """Build uniform and snapped-uniform proposals without other detectors."""

    max_detection_dimension = 1800
    scale = min(1.0, max_detection_dimension / max(image.size))
    detection_height = round(image.height * scale)
    uniform = _scale_proposal(
        uniform_split(detection_height, rows=rows),
        detection_height,
        image.height,
        scale,
    )
    return uniform, snap_uniform_boundaries(image, uniform)


def uniform_eight_proposals(image: Image.Image) -> tuple[SplitProposal, SplitProposal]:
    """Compatibility wrapper for the established eight-band strategy."""

    return uniform_proposals(image, rows=8)


def adaptive_max_snapped_proposal(
    image: Image.Image,
    *,
    max_rows: int = 32,
    minimum_rows: int = 2,
) -> SplitProposal:
    """Use the maximum selected natural-row count, then snap every cut.

    This is intentionally different from the older adaptive row-count policy.
    It does not divide an estimated row-work score by a target number of rows
    per band, and it does not round the result to a power of two.  A table can
    therefore produce U3, U13, U27, and so on.  The recognizer's fixed active
    decode batch is independent from this request count.

    Candidate boundaries are the union of ruled, whitespace, hybrid, and
    row-edge evidence.  Nearby candidates are clustered before every retained
    interior boundary passes through the reviewed snap routine.  Boundaries
    that still cross ink after snapping are removed.  The maximum detector row
    count sets the target U, capped by ``max_rows``.  When safe boundaries are
    insufficient to reach that target, a target-U uniform proposal supplies
    the missing density and every one of its cuts is snapped.  This deliberate
    maximum-row bias lets dense tables reach U32 even when they have no clean
    full-width separator gaps.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    minimum_rows = min(int(minimum_rows), int(max_rows), max(1, image.height))
    max_rows = min(int(max_rows), max(1, image.height))

    proposals = {proposal.name: proposal for proposal in analyze(image)}
    selected = proposals["selected"]
    candidate_sources = ("ruled", "whitespace", "row_edge", "hybrid")
    source_rows = {
        name: max(1, len(proposals[name].boundaries) - 1)
        for name in candidate_sources
    }
    target_rows = max(
        minimum_rows,
        min(max_rows, max(source_rows.values(), default=minimum_rows)),
    )
    character_heights = [
        float(proposals[name].diagnostics["character_height"])
        for name in candidate_sources
        if "character_height" in proposals[name].diagnostics
    ]
    typical_character_height = (
        float(np.median(character_heights)) if character_heights else 4.0
    )
    cluster_distance = max(2, round(typical_character_height * 0.55))
    raw_candidates = sorted(
        (int(boundary), name)
        for name in candidate_sources
        for boundary in proposals[name].boundaries[1:-1]
    )
    clusters: list[list[tuple[int, str]]] = []
    for candidate in raw_candidates:
        if clusters and candidate[0] - clusters[-1][-1][0] <= cluster_distance:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    clustered_candidates = [
        round(float(np.median([value for value, _source in cluster])))
        for cluster in clusters
    ]
    union = SplitProposal(
        name="adaptive_candidate_union",
        boundaries=(0, *clustered_candidates, image.height),
        diagnostics={"source_rows": source_rows},
    )
    if len(union.boundaries) - 1 < minimum_rows:
        union = uniform_split(image.height, rows=minimum_rows)
        pre_snap_mode = "snapped_uniform_minimum_fallback"
    else:
        pre_snap_mode = "union_all_detector_boundaries"

    snapped = snap_uniform_boundaries(image, union)
    details = list(snapped.diagnostics.get("boundary_details", []))
    safe_boundaries = [0]
    removed_ink_crossings: list[int] = []
    for boundary, detail in zip(snapped.boundaries[1:-1], details):
        if detail.get("new_kind") == "ink_crossing":
            removed_ink_crossings.append(int(boundary))
            continue
        safe_boundaries.append(int(boundary))
    safe_boundaries.append(image.height)

    safe_boundaries = list(dict.fromkeys(safe_boundaries))
    safe = SplitProposal(
        name="adaptive_safe_snapped",
        boundaries=tuple(safe_boundaries),
        diagnostics={},
    )
    safe_rows = len(safe.boundaries) - 1
    if safe_rows > target_rows:
        bounded = group_natural_rows(safe, rows=target_rows)
        cap_mode = "grouped_at_existing_safe_snapped_boundaries"
        supplemental_snap = None
    elif safe_rows < target_rows:
        supplemental_snap = snap_uniform_boundaries(
            image,
            uniform_split(image.height, rows=target_rows),
        )
        bounded = supplemental_snap
        cap_mode = "detector_max_uniform_then_snapped"
    else:
        bounded = safe
        cap_mode = "all_safe_snapped_boundaries"
        supplemental_snap = None

    if len(bounded.boundaries) - 1 < minimum_rows:
        fallback = snap_uniform_boundaries(
            image,
            uniform_split(image.height, rows=minimum_rows),
        )
        final_boundaries = fallback.boundaries
        fallback_details = fallback.diagnostics.get("boundary_details", [])
        fallback_used = True
    else:
        final_boundaries = bounded.boundaries
        fallback_details = []
        fallback_used = False

    diagnostics = {
        "source": selected.name,
        "selected_source": selected.diagnostics.get("selected_source"),
        "selection_reason": selected.diagnostics.get("selection_reason"),
        "selected_detected_rows": max(1, len(selected.boundaries) - 1),
        "detector_rows": source_rows,
        "raw_candidate_boundaries": len(raw_candidates),
        "clustered_candidate_boundaries": len(clustered_candidates),
        "candidate_cluster_distance_px": cluster_distance,
        "typical_character_height_px": typical_character_height,
        "max_rows": max_rows,
        "minimum_rows": minimum_rows,
        "target_rows": target_rows,
        "pre_snap_mode": pre_snap_mode,
        "cap_mode": cap_mode,
        "rows_before_snap": len(union.boundaries) - 1,
        "rows_after_snap_safety": len(final_boundaries) - 1,
        "removed_ink_crossing_count": len(removed_ink_crossings),
        "removed_ink_crossing_y": removed_ink_crossings,
        "snapped_changed_boundaries": snapped.diagnostics.get(
            "changed_boundaries", 0
        ),
        "snap_ink_crossings_before": snapped.diagnostics.get(
            "ink_crossings_before", 0
        ),
        "snap_ink_crossings_after": snapped.diagnostics.get(
            "ink_crossings_after", 0
        ),
        "snap_boundary_details": details,
        "supplemental_snap_ink_crossings_before": (
            supplemental_snap.diagnostics.get("ink_crossings_before", 0)
            if supplemental_snap is not None
            else 0
        ),
        "supplemental_snap_ink_crossings_after": (
            supplemental_snap.diagnostics.get("ink_crossings_after", 0)
            if supplemental_snap is not None
            else 0
        ),
        "supplemental_snap_boundary_details": (
            supplemental_snap.diagnostics.get("boundary_details", [])
            if supplemental_snap is not None
            else []
        ),
        "minimum_fallback_used": fallback_used,
        "minimum_fallback_boundary_details": fallback_details,
    }
    return SplitProposal(
        name=f"adaptive_max_{max_rows}_snapped",
        boundaries=tuple(int(value) for value in final_boundaries),
        diagnostics=diagnostics,
    )


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
    uniform_8 = uniform_split(detection_image.height, rows=8)
    proposals = tuple(
        _scale_proposal(
            proposal,
            detection_image.height,
            image.height,
            scale,
        )
        for proposal in (ruled, whitespace, row_edge, hybrid, selected, uniform_8)
    )
    natural = proposals[:5]
    grouped = tuple(group_natural_rows(proposal, rows=8) for proposal in natural)
    return (*proposals, *grouped, snap_uniform_boundaries(image, proposals[-1]))


COLORS = {
    "ruled": (220, 40, 40),
    "whitespace": (35, 110, 220),
    "row_edge": (170, 80, 190),
    "hybrid": (15, 150, 80),
    "selected": (230, 125, 20),
    "uniform_8": (30, 155, 145),
    "uniform_8_snapped": (20, 120, 105),
}


def draw_overlay(image: Image.Image, proposal: SplitProposal) -> Image.Image:
    result = image.convert("RGBA")
    tint = Image.new("RGBA", result.size, (0, 0, 0, 0))
    tint_draw = ImageDraw.Draw(tint)
    base = COLORS.get(
        proposal.name,
        COLORS.get(proposal.name.rsplit("_grouped_", 1)[0], (80, 80, 80)),
    )
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
        row_draft_source, rotation_cw = orient_row_draft_image(raw_image, record)
        image, trim_box = trim_blank_margin(row_draft_source)
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
                "row_draft_rotation_cw": rotation_cw,
                "row_draft_source_size": list(row_draft_source.size),
                "trim_box_in_row_draft_source": list(trim_box),
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
