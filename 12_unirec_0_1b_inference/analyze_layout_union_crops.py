"""Compare OpenDoc block stitching with direct page-union crops.

This is a preprocessing-only lab. It runs layout detection, constructs the
same block crops as the production UniRec runner, and does not load or execute
the recognition model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--save-worst", type=int, default=12)
    return parser.parse_args()


def base_label(label: str) -> str:
    head, separator, tail = label.rpartition("_")
    if separator and tail.isdigit():
        return head
    return label


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def make_blocks(image: np.ndarray, boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    blocks: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = clamp_box(box["coordinate"], width, height)
        crop = image[y1:y2, x1:x2]
        blocks.append(
            {
                "img": None if crop.size == 0 else crop,
                "box": [x1, y1, x2, y2],
                "label": box["label"],
                "score": box.get("score", 1.0),
                "_source_index": index,
            }
        )
    return blocks


def group_members(outputs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(outputs):
        block = outputs[index]
        aligns = block.get("merge_aligns")
        member_count = len(aligns) + 1 if aligns else 1
        if member_count > 1:
            members = [block]
            cursor = index + 1
            while cursor < len(outputs) and len(members) < member_count:
                candidate = outputs[cursor]
                if candidate.get("img") is None:
                    members.append(candidate)
                cursor += 1
            if len(members) != member_count:
                raise RuntimeError(
                    f"Could not recover {member_count}-member merge group at output {index}"
                )
            groups.append(members)
        index += 1
    return groups


def intersection_area(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def analyze_group(
    *,
    page_index: int,
    image_path: Path,
    image: np.ndarray,
    all_blocks: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    height, width = image.shape[:2]
    source_indices = [int(member["_source_index"]) for member in members]
    boxes = [clamp_box(all_blocks[index]["box"], width, height) for index in source_indices]
    union = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    ux1, uy1, ux2, uy2 = union
    union_crop = image[uy1:uy2, ux1:ux2]
    stitched = members[0]["img"]
    if stitched is None:
        raise RuntimeError("Merged group head has no stitched image")

    union_height, union_width = union_crop.shape[:2]
    coverage = np.zeros((union_height, union_width), dtype=np.bool_)
    for x1, y1, x2, y2 in boxes:
        coverage[y1 - uy1 : y2 - uy1, x1 - ux1 : x2 - ux1] = True
    union_area = int(coverage.size)
    uncovered = ~coverage
    uncovered_pixels = int(uncovered.sum())
    if union_crop.size:
        grayscale = cv2.cvtColor(union_crop, cv2.COLOR_BGR2GRAY)
        uncovered_ink_pixels = int(np.logical_and(uncovered, grayscale < 245).sum())
    else:
        uncovered_ink_pixels = 0

    member_index_set = set(source_indices)
    foreign_labels: Counter[str] = Counter()
    foreign_overlap_area = 0
    for other_index, other in enumerate(all_blocks):
        if other_index in member_index_set:
            continue
        other_box = clamp_box(other["box"], width, height)
        overlap = intersection_area(union, other_box)
        if overlap:
            foreign_overlap_area += overlap
            foreign_labels[base_label(other["label"])] += 1

    same_shape = stitched.shape == union_crop.shape
    pixel_exact = bool(same_shape and np.array_equal(stitched, union_crop))
    result = {
        "page_index": page_index,
        "image": image_path.name,
        "source_indices": source_indices,
        "labels": [base_label(all_blocks[index]["label"]) for index in source_indices],
        "boxes": [list(box) for box in boxes],
        "aligns": members[0].get("merge_aligns"),
        "union_box": list(union),
        "stitched_shape": list(stitched.shape),
        "union_shape": list(union_crop.shape),
        "same_shape": same_shape,
        "pixel_exact": pixel_exact,
        "union_area": union_area,
        "stitched_area": int(stitched.shape[0] * stitched.shape[1]),
        "uncovered_fraction": uncovered_pixels / union_area if union_area else 0.0,
        "uncovered_ink_fraction": uncovered_ink_pixels / union_area if union_area else 0.0,
        "foreign_overlap_fraction": foreign_overlap_area / union_area if union_area else 0.0,
        "foreign_labels": dict(foreign_labels),
    }
    return result, union_crop


def save_comparison(
    path: Path,
    stitched: np.ndarray,
    union_crop: np.ndarray,
    title: str,
) -> None:
    stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    union_rgb = cv2.cvtColor(union_crop, cv2.COLOR_BGR2RGB)
    left = Image.fromarray(stitched_rgb)
    right = Image.fromarray(union_rgb)
    top = 34
    canvas = Image.new(
        "RGB",
        (left.width + right.width + 12, max(left.height, right.height) + top),
        "white",
    )
    canvas.paste(left, (0, top))
    canvas.paste(right, (left.width + 12, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"PIL stitch | union crop -- {title}", fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools import infer_doc_onnx
    from tools.utils.utility import get_image_file_list

    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    image_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(args.input.expanduser().resolve())))
    ][args.offset : args.offset + args.limit]
    detector = PPDocLayoutV2NpuAdapter(
        model_path=args.layout_model,
        device=args.device,
        dtype=args.dtype,
        threshold=args.threshold,
    )
    image_labels = infer_doc_onnx.IMAGE_LABELS + ["chart"]

    page_records: list[dict[str, Any]] = []
    intended_groups: list[dict[str, Any]] = []
    examples: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray]] = []
    current_merge_s = 0.0
    intended_merge_s = 0.0
    layout_s = 0.0
    total_blocks = 0
    current_multi_groups = 0
    current_changed_images = 0

    for page_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {image_path}")
        started = time.perf_counter()
        layout = detector([image], threshold=args.threshold)[0]
        layout_s += time.perf_counter() - started
        blocks = make_blocks(image, layout["boxes"])
        total_blocks += len(blocks)

        started = time.perf_counter()
        current_outputs = infer_doc_onnx.merge_blocks(
            blocks,
            non_merge_labels=image_labels + ["table"],
        )
        current_merge_s += time.perf_counter() - started
        current_groups = group_members(current_outputs)
        current_multi_groups += len(current_groups)
        for original, output in zip(blocks, current_outputs):
            first = original.get("img")
            second = output.get("img")
            if first is None or second is None:
                if first is not second:
                    current_changed_images += 1
            elif first.shape != second.shape or not np.array_equal(first, second):
                current_changed_images += 1

        normalized_blocks = []
        for block in blocks:
            normalized = block.copy()
            normalized["label"] = base_label(block["label"])
            normalized_blocks.append(normalized)
        started = time.perf_counter()
        intended_outputs = infer_doc_onnx.merge_blocks(
            normalized_blocks,
            non_merge_labels=image_labels + ["table"],
        )
        intended_merge_s += time.perf_counter() - started
        groups = group_members(intended_outputs)
        page_group_records = []
        for members in groups:
            record, union_crop = analyze_group(
                page_index=page_index,
                image_path=image_path,
                image=image,
                all_blocks=normalized_blocks,
                members=members,
            )
            page_group_records.append(record)
            intended_groups.append(record)
            risk = (
                record["uncovered_ink_fraction"]
                + record["foreign_overlap_fraction"]
                + (0.25 if not record["same_shape"] else 0.0)
            )
            examples.append((risk, record, members[0]["img"], union_crop))
        page_records.append(
            {
                "page_index": page_index,
                "image": image_path.name,
                "block_count": len(blocks),
                "current_multi_group_count": len(current_groups),
                "intended_multi_group_count": len(groups),
                "intended_groups": page_group_records,
            }
        )
        print(
            f"UNION_CROP_PAGE {page_index + 1}/{len(image_paths)} "
            f"blocks={len(blocks)} current_groups={len(current_groups)} "
            f"intended_groups={len(groups)}",
            flush=True,
        )

    exact_groups = sum(bool(group["pixel_exact"]) for group in intended_groups)
    same_shape_groups = sum(bool(group["same_shape"]) for group in intended_groups)
    groups_with_foreign = sum(group["foreign_overlap_fraction"] > 0 for group in intended_groups)
    groups_with_uncovered_ink = sum(
        group["uncovered_ink_fraction"] > 0.001 for group in intended_groups
    )
    member_count_histogram = Counter(len(group["source_indices"]) for group in intended_groups)
    summary = {
        "page_count": len(image_paths),
        "total_blocks": total_blocks,
        "layout_s": layout_s,
        "current_merge_s": current_merge_s,
        "current_multi_group_count": current_multi_groups,
        "current_changed_image_count": current_changed_images,
        "intended_merge_s": intended_merge_s,
        "intended_multi_group_count": len(intended_groups),
        "intended_member_count_histogram": dict(sorted(member_count_histogram.items())),
        "union_pixel_exact_group_count": exact_groups,
        "union_same_shape_group_count": same_shape_groups,
        "union_groups_with_foreign_overlap": groups_with_foreign,
        "union_groups_with_uncovered_ink_gt_0_1pct": groups_with_uncovered_ink,
        "mean_union_uncovered_fraction": float(
            np.mean([group["uncovered_fraction"] for group in intended_groups])
        )
        if intended_groups
        else 0.0,
        "mean_union_uncovered_ink_fraction": float(
            np.mean([group["uncovered_ink_fraction"] for group in intended_groups])
        )
        if intended_groups
        else 0.0,
        "mean_union_foreign_overlap_fraction": float(
            np.mean([group["foreign_overlap_fraction"] for group in intended_groups])
        )
        if intended_groups
        else 0.0,
    }
    payload = {"summary": summary, "pages": page_records, "groups": intended_groups}
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    examples.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, record, stitched, union_crop) in enumerate(
        examples[: args.save_worst], start=1
    ):
        save_comparison(
            args.output_dir / f"worst_{rank:02d}_page_{record['page_index']:03d}.jpg",
            stitched,
            union_crop,
            (
                f"page={record['page_index']} members={record['source_indices']} "
                f"uncovered_ink={record['uncovered_ink_fraction']:.3f} "
                f"foreign={record['foreign_overlap_fraction']:.3f}"
            ),
        )
    print("UNION_CROP_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
