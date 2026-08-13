#!/usr/bin/env python3
"""Compare canonical-RGB crop construction with the prior BGR frontend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from layout_page_input import decode_page_rgb, materialize_layout_bgr, materialize_layout_rgb
from layout_process_pool import _base_label, _prepare_frontend_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-result", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _legacy_bgr_crops(
    *,
    rgb: np.ndarray,
    layout_result: dict[str, Any],
    crop_margin: Any,
    tokenize_figure_of_table: Any,
) -> dict[str, Any]:
    bgr = materialize_layout_bgr(rgb)
    image_labels = ["image", "header_image", "footer_image", "seal"]
    blocks = []
    block_images = []
    for box in layout_result["boxes"]:
        x1, y1, x2, y2 = map(int, box["coordinate"])
        cropped = bgr[y1:y2, x1:x2]
        block_image = None if cropped.size == 0 else cropped
        block_images.append(block_image)
        blocks.append({"box": box["coordinate"], "label": box["label"]})
    figures = []
    for block, image in zip(blocks, block_images):
        if _base_label(block["label"]) in image_labels and image is not None:
            x1, y1, x2, y2 = map(int, block["box"])
            figures.append(
                {
                    "coordinate": block["box"],
                    "path": (
                        f"imgs/img_in_{_base_label(block['label'])}_box_"
                        f"{x1}_{y1}_{x2}_{y2}.jpg"
                    ),
                }
            )
    crops = []
    dropped: set[str] = set()
    for block, image in zip(blocks, block_images):
        label = block["label"]
        if _base_label(label) in image_labels or image is None:
            continue
        token_map = {}
        drop_figures = []
        if "table" in label:
            image, token_map, drop_figures = tokenize_figure_of_table(
                image,
                block["box"],
                figures,
            )
        elif "formula" in label and label != "formula_number":
            image = crop_margin(image)
        crops.append(
            {
                "label": label,
                "image_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                "figure_token_map": token_map,
            }
        )
        dropped.update(drop_figures)
    return {"crops": crops, "drop_figures_set": sorted(dropped)}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    layout_report = json.loads(args.layout_result.read_text())
    pages = layout_report["pages"][: args.limit]
    old_wall_s = 0.0
    new_wall_s = 0.0
    compared_crops = 0
    for index, page in enumerate(pages):
        path = Path(page["image"])
        rgb, _ = decode_page_rgb(path)

        started = time.perf_counter()
        legacy = _legacy_bgr_crops(
            rgb=rgb,
            layout_result=page["result"],
            crop_margin=crop_margin,
            tokenize_figure_of_table=tokenize_figure_of_table,
        )
        old_wall_s += time.perf_counter() - started

        started = time.perf_counter()
        direct, _ = _prepare_frontend_payload(
            page_index=index,
            path=path,
            rgb=materialize_layout_rgb(rgb),
            layout_result=page["result"],
            use_chart_recognition=True,
            tokenize_figure_of_table=tokenize_figure_of_table,
        )
        new_wall_s += time.perf_counter() - started

        if legacy["drop_figures_set"] != direct["drop_figures_set"]:
            raise RuntimeError(f"drop-figure mismatch on page {index}")
        if len(legacy["crops"]) != len(direct["crops"]):
            raise RuntimeError(f"crop-count mismatch on page {index}")
        for crop_index, (old_crop, new_crop) in enumerate(
            zip(legacy["crops"], direct["crops"])
        ):
            for field in ("label", "figure_token_map"):
                if old_crop[field] != new_crop[field]:
                    raise RuntimeError(
                        f"{field} mismatch on page {index} crop {crop_index}"
                    )
            if not np.array_equal(old_crop["image_rgb"], new_crop["image_rgb"]):
                difference = np.abs(
                    old_crop["image_rgb"].astype(np.int16)
                    - new_crop["image_rgb"].astype(np.int16)
                )
                raise RuntimeError(
                    f"pixel mismatch on page {index} crop {crop_index}: "
                    f"max_abs={difference.max()}"
                )
            compared_crops += 1
        print(
            f"LAYOUT_RGB_PARITY page={index + 1}/{len(pages)} "
            f"crops={len(direct['crops'])}",
            flush=True,
        )

    report = {
        "status": "pass",
        "pages": len(pages),
        "crops": compared_crops,
        "legacy_bgr_frontend_s": old_wall_s,
        "direct_rgb_frontend_s": new_wall_s,
        "speedup_x": old_wall_s / new_wall_s if new_wall_s else None,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("LAYOUT_RGB_PARITY PASS " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
