#!/usr/bin/env python3
"""Verify the complete OmniDocBench image set used by Experiment 09."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_PAGE_COUNT = 1651
UNIFORM_SAMPLE_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def image_name(page: Any) -> str:
    if not isinstance(page, dict):
        raise TypeError("page entry is not an object")
    page_info = page.get("page_info")
    if not isinstance(page_info, dict):
        raise TypeError("page_info is missing or is not an object")
    raw_path = page_info.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TypeError("page_info.image_path is missing or is not a string")
    name = Path(raw_path).name
    if not name:
        raise ValueError("page_info.image_path has no filename")
    return name


def decode_image(path: Path) -> tuple[int, int, int]:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unexpected decoded shape {image.shape}")
    height, width, channels = image.shape
    return int(width), int(height), int(channels)


def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    started = time.perf_counter()
    annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise TypeError("OmniDocBench JSON root must be a list")

    schema_errors: list[dict[str, Any]] = []
    names: list[str | None] = []
    for index, page in enumerate(annotations):
        try:
            names.append(image_name(page))
        except (TypeError, ValueError) as error:
            names.append(None)
            schema_errors.append({"index": index, "error": str(error)})

    valid_names = [name for name in names if name is not None]
    name_counts = Counter(valid_names)
    duplicate_names = sorted(
        name for name, count in name_counts.items() if count > 1
    )

    missing: list[str] = []
    zero_byte: list[str] = []
    unreadable: list[dict[str, str]] = []
    decoded_dimensions: list[tuple[int, int]] = []
    total_referenced_bytes = 0

    for name in dict.fromkeys(valid_names):
        path = images_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        total_referenced_bytes += size
        if size == 0:
            zero_byte.append(name)
            continue
        try:
            width, height, _channels = decode_image(path)
            decoded_dimensions.append((width, height))
        except (OSError, ValueError) as error:
            unreadable.append({"image": name, "error": str(error)})

    sample = []
    for index in uniform_indices(len(annotations), UNIFORM_SAMPLE_COUNT):
        name = names[index]
        sample.append(
            {
                "index": index,
                "image_name": name,
                "absolute_path": (
                    str((images_dir / name).resolve()) if name is not None else None
                ),
            }
        )

    exact_page_count = len(annotations) == EXPECTED_PAGE_COUNT
    valid = all(
        (
            exact_page_count,
            not schema_errors,
            not duplicate_names,
            not missing,
            not zero_byte,
            not unreadable,
            len(decoded_dimensions) == len(valid_names),
        )
    )

    report = {
        "valid": valid,
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "expected_page_count": EXPECTED_PAGE_COUNT,
        "annotation_count": len(annotations),
        "exact_page_count": exact_page_count,
        "referenced_image_count": len(valid_names),
        "unique_referenced_image_count": len(set(valid_names)),
        "decoded_image_count": len(decoded_dimensions),
        "total_referenced_bytes": total_referenced_bytes,
        "schema_error_count": len(schema_errors),
        "schema_errors_first_20": schema_errors[:20],
        "duplicate_name_count": len(duplicate_names),
        "duplicate_names_first_20": duplicate_names[:20],
        "missing_image_count": len(missing),
        "missing_images_first_20": missing[:20],
        "zero_byte_image_count": len(zero_byte),
        "zero_byte_images_first_20": zero_byte[:20],
        "unreadable_image_count": len(unreadable),
        "unreadable_images_first_20": unreadable[:20],
        "decoded_width": {
            "min": min((width for width, _height in decoded_dimensions), default=None),
            "max": max((width for width, _height in decoded_dimensions), default=None),
        },
        "decoded_height": {
            "min": min((height for _width, height in decoded_dimensions), default=None),
            "max": max((height for _width, height in decoded_dimensions), default=None),
        },
        "uniform_sample": sample,
        "audit_wall_s": time.perf_counter() - started,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
