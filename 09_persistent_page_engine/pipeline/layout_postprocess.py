"""Owned PP-DocLayoutV3 geometry and PaddleOCR-VL crop preparation.

This module implements the narrow production contract used by Experiment 09.
It intentionally has no PaddleX dependency. The behavior was ported from the
Apache-2.0 PaddleX PP-DocLayoutV3 and PaddleOCR-VL postprocessors, then reduced
to the fixed v1.6 configuration exercised by this project.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np
from PIL import Image


SKIP_ORDER_LABELS = {
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
}

IMAGE_LABELS = ["image", "header_image", "footer_image"]


def _polygon_overlap_ratio(
    polygon1: Any,
    polygon2: Any,
    mode: str = "union",
) -> float:
    from shapely.geometry import Polygon

    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    intersection = poly1.intersection(poly2).area
    if mode == "union":
        reference = poly1.union(poly2).area
    elif mode == "small":
        reference = min(poly1.area, poly2.area)
    elif mode == "large":
        reference = max(poly1.area, poly2.area)
    else:
        raise ValueError(f"unsupported polygon overlap mode: {mode}")
    return intersection / reference if reference else 0.0


def _box_overlap_ratio(
    box1: Any,
    box2: Any,
    mode: str = "union",
) -> float:
    first = np.asarray(box1, dtype=np.float64)
    second = np.asarray(box2, dtype=np.float64)
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = abs((first[2] - first[0]) * (first[3] - first[1]))
    second_area = abs((second[2] - second[0]) * (second[3] - second[1]))
    if mode == "union":
        reference = first_area + second_area - intersection
    elif mode == "small":
        reference = min(first_area, second_area)
    elif mode == "large":
        reference = max(first_area, second_area)
    else:
        raise ValueError(f"unsupported box overlap mode: {mode}")
    return intersection / reference if reference else 0.0


def _projection_overlap_ratio(
    box1: Any,
    box2: Any,
    direction: str = "horizontal",
    mode: str = "union",
) -> float:
    start, end = (0, 2) if direction == "horizontal" else (1, 3)
    overlap = min(box1[end], box2[end]) - max(box1[start], box2[start])
    if overlap <= 0:
        return 0.0
    if mode == "union":
        reference = max(box1[end], box2[end]) - min(
            box1[start],
            box2[start],
        )
    elif mode == "small":
        reference = min(
            box1[end] - box1[start],
            box2[end] - box2[start],
        )
    elif mode == "large":
        reference = max(
            box1[end] - box1[start],
            box2[end] - box2[start],
        )
    else:
        raise ValueError(f"unsupported projection overlap mode: {mode}")
    return overlap / reference if reference > 0 else 0.0


def _rect_from_box(box: Any) -> np.ndarray:
    x_min, y_min, x_max, y_max = np.asarray(box).astype(np.int32)
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def _axis_aligned_rect_fast_path(
    box: Any,
    polygon: Any,
) -> np.ndarray | None:
    points = np.asarray(polygon, dtype=np.float32)
    if points.size != 8:
        return None
    points = points.reshape(4, 2)
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {
        (float(xs[0]), float(ys[0])),
        (float(xs[1]), float(ys[0])),
        (float(xs[1]), float(ys[1])),
        (float(xs[0]), float(ys[1])),
    }
    if {tuple(point) for point in points.tolist()} != expected:
        return None

    x_min, y_min, x_max, y_max = np.asarray(box).astype(np.int32)
    if not (
        xs[0] >= x_min
        and ys[0] >= y_min
        and xs[1] <= x_max
        and ys[1] <= y_max
    ):
        return None
    box_area = float((x_max - x_min) * (y_max - y_min))
    polygon_area = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    if box_area <= 0 or polygon_area / box_area < 0.95:
        return None
    return _rect_from_box(box)


def _polygon_to_quad(polygon: Any) -> np.ndarray | None:
    if polygon is None or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    quad = cv2.boxPoints(cv2.minAreaRect(points))
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    top_left = np.argmin(quad[:, 0] + quad[:, 1])
    return np.roll(quad, -top_left, axis=0)


def _normalize_polygon(
    box: Any,
    polygon: Any,
    previous_polygon: np.ndarray | None,
) -> np.ndarray:
    rect = _rect_from_box(box)
    if polygon is None:
        return rect
    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)
    if len(polygon) < 4:
        return rect

    fast_rect = _axis_aligned_rect_fast_path(box, polygon)
    if fast_rect is not None:
        return fast_rect

    quad = _polygon_to_quad(polygon)
    if quad is not None:
        rect_list = rect.tolist()
        quad_list = quad.tolist()
        if _polygon_overlap_ratio(rect_list, quad_list, "union") >= 0.95:
            return rect
        polygon_list = polygon.tolist()
        polygon_quad_iou = _polygon_overlap_ratio(
            polygon_list,
            quad_list,
            "union",
        )
        previous_iou = 0.0
        if previous_polygon is not None:
            previous_iou = _polygon_overlap_ratio(
                previous_polygon.tolist(),
                rect_list,
                "small",
            )
        if polygon_quad_iou >= 0.8 and previous_iou < 0.01:
            return quad
    return polygon


class LayoutPostprocessor:
    """Fixed PP-DocLayoutV3 v1.6 postprocessor."""

    def __init__(self, labels: list[str], threshold: float = 0.5) -> None:
        self.labels = labels
        self.threshold = float(threshold)

    def __call__(
        self,
        prediction: dict[str, Any],
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        order = prediction["order_seq"].detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if len(boxes) == 0:
            return []
        formatted = np.concatenate(
            [
                labels[:, None].astype(np.float32, copy=False),
                scores[:, None].astype(np.float32, copy=False),
                boxes.astype(np.float32, copy=False),
                order[:, None],
            ],
            axis=1,
        )
        polygons = [
            np.asarray(points) for points in prediction["polygon_points"]
        ]
        return self._apply(formatted, polygons, image_size)

    def _apply(
        self,
        boxes: np.ndarray,
        polygons: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(int)
        keep = (boxes[:, 1] > self.threshold) & (boxes[:, 0] > -1)
        boxes = boxes[keep]
        polygons = [
            polygon
            for polygon, selected in zip(polygons, keep)
            if selected
        ]

        if len(boxes) > 1:
            width, height = image_size
            area_threshold = 0.82 if width > height else 0.93
            image_index = self.labels.index("image")
            page_area = width * height
            filtered_boxes: list[np.ndarray] = []
            filtered_polygons: list[np.ndarray] = []
            for box, polygon in zip(boxes, polygons):
                if int(box[0]) == image_index:
                    x_min = max(0, box[2])
                    y_min = max(0, box[3])
                    x_max = min(width, box[4])
                    y_max = min(height, box[5])
                    if (
                        (x_max - x_min) * (y_max - y_min)
                        > area_threshold * page_area
                    ):
                        continue
                filtered_boxes.append(box)
                filtered_polygons.append(polygon)
            if filtered_boxes:
                boxes = np.asarray(filtered_boxes)
                polygons = filtered_polygons

        # The shipped v1.6 model uses no layout_merge_bboxes_mode, so its
        # containment filter is intentionally absent from this fixed path.
        sorted_indices = np.argsort(boxes[:, 6])
        boxes = boxes[sorted_indices, :6]
        polygons = [polygons[index] for index in sorted_indices]

        normalized: list[np.ndarray] = []
        for box, polygon in zip(boxes, polygons):
            normalized.append(
                _normalize_polygon(
                    box[2:6],
                    polygon,
                    normalized[-1] if normalized else None,
                )
            )

        width, height = image_size
        results: list[dict[str, Any]] = []
        for index, (box, polygon) in enumerate(zip(boxes, normalized)):
            x_min = int(max(0, box[2]))
            y_min = int(max(0, box[3]))
            x_max = int(min(width, box[4]))
            y_max = int(min(height, box[5]))
            if x_max <= x_min or y_max <= y_min:
                continue
            class_id = int(box[0])
            results.append(
                {
                    "cls_id": class_id,
                    "label": self.labels[class_id],
                    "score": float(box[1]),
                    "coordinate": [x_min, y_min, x_max, y_max],
                    "order": index + 1,
                    "polygon_points": polygon,
                }
            )
        results = _filter_detector_boxes(results)
        next_order = 1
        for result in results:
            if result["label"] in SKIP_ORDER_LABELS:
                result["order"] = None
            else:
                result["order"] = next_order
                next_order += 1
        return results


def _filter_detector_boxes(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boxes = [box for box in source if box["label"] != "reference"]
    dropped: set[int] = set()
    for first_index, first in enumerate(boxes):
        x1, y1, x2, y2 = first["coordinate"]
        if x2 - x1 < 6 or y2 - y1 < 6:
            dropped.add(first_index)
        for second_index in range(first_index + 1, len(boxes)):
            if first_index in dropped or second_index in dropped:
                continue
            second = boxes[second_index]
            overlap = _box_overlap_ratio(
                first["coordinate"],
                second["coordinate"],
                "small",
            )
            if (
                first["label"] == "inline_formula"
                or second["label"] == "inline_formula"
            ) and overlap > 0.5:
                if first["label"] == "inline_formula":
                    dropped.add(first_index)
                if second["label"] == "inline_formula":
                    dropped.add(second_index)
                continue
            if overlap <= 0.7:
                continue
            if (
                _polygon_overlap_ratio(
                    first["polygon_points"],
                    second["polygon_points"],
                    "small",
                )
                < 0.7
            ):
                continue
            first_area = (x2 - x1) * (y2 - y1)
            sx1, sy1, sx2, sy2 = second["coordinate"]
            second_area = (sx2 - sx1) * (sy2 - sy1)
            labels = {first["label"], second["label"]}
            if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                if "table" not in labels or labels <= {
                    "table",
                    "image",
                    "seal",
                    "chart",
                }:
                    continue
            dropped.add(second_index if first_area >= second_area else first_index)
    return [
        box for index, box in enumerate(boxes) if index not in dropped
    ]


def filter_overlap_boxes(
    layout_result: dict[str, Any],
) -> dict[str, Any]:
    filtered = deepcopy(layout_result)
    boxes = [
        box
        for box in filtered["boxes"]
        if box["label"] != "reference"
    ]
    if not boxes:
        filtered["boxes"] = boxes
        return filtered
    coords = np.asarray(
        [box["coordinate"] for box in boxes],
        dtype=np.float64,
    )
    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    areas = widths * heights
    x1 = np.maximum(coords[:, None, 0], coords[None, :, 0])
    y1 = np.maximum(coords[:, None, 1], coords[None, :, 1])
    x2 = np.minimum(coords[:, None, 2], coords[None, :, 2])
    y2 = np.minimum(coords[:, None, 3], coords[None, :, 3])
    intersections = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    small_areas = np.minimum(areas[:, None], areas[None, :])
    overlaps = np.divide(
        intersections,
        small_areas,
        out=np.zeros_like(intersections),
        where=small_areas > 0,
    )

    dropped: set[int] = set()
    for first_index, first in enumerate(boxes):
        if widths[first_index] < 6 or heights[first_index] < 6:
            dropped.add(first_index)
        for second_index in range(first_index + 1, len(boxes)):
            if first_index in dropped or second_index in dropped:
                continue
            second = boxes[second_index]
            overlap = overlaps[first_index, second_index]
            if (
                first["label"] == "inline_formula"
                or second["label"] == "inline_formula"
            ) and overlap > 0.5:
                if first["label"] == "inline_formula":
                    dropped.add(first_index)
                if second["label"] == "inline_formula":
                    dropped.add(second_index)
                continue
            if overlap <= 0.7:
                continue
            if (
                _polygon_overlap_ratio(
                    first["polygon_points"],
                    second["polygon_points"],
                    "small",
                )
                < 0.7
            ):
                continue
            labels = {first["label"], second["label"]}
            if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                if "table" not in labels or labels <= {
                    "table",
                    "image",
                    "seal",
                    "chart",
                }:
                    continue
            dropped.add(
                second_index
                if areas[first_index] >= areas[second_index]
                else first_index
            )
    filtered["boxes"] = [
        box for index, box in enumerate(boxes) if index not in dropped
    ]
    return filtered


def _is_full_crop_rectangle(box_info: dict[str, Any]) -> bool:
    polygon = np.asarray(box_info["polygon_points"], dtype=np.int32)
    if polygon.size != 8:
        return False
    polygon = polygon.reshape(4, 2)
    x_min, y_min, x_max, y_max = [
        int(value) for value in box_info["coordinate"]
    ]
    return {tuple(point) for point in polygon.tolist()} == {
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    }


def crop_layout_regions(
    image: np.ndarray,
    boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for box_info in boxes:
        box = box_info["coordinate"]
        x_min, y_min, x_max, y_max = [int(value) for value in box]
        crop = image[y_min:y_max, x_min:x_max].copy()
        output = {
            "img": crop,
            "box": box,
            "label": box_info["label"],
            "polygon_points": box_info["polygon_points"],
        }
        if not _is_full_crop_rectangle(box_info):
            mask = np.zeros(crop.shape[:2], dtype=np.int32)
            polygon = np.asarray(
                box_info["polygon_points"],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            polygon = polygon - np.array([x_min, y_min])
            cv2.fillPoly(mask, [polygon], 1)
            crop[~mask.astype(bool)] = 255
        outputs.append(output)
    return outputs


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.fromarray(image)


def _merge_images(
    images: list[np.ndarray],
    aligns: list[str],
) -> np.ndarray:
    pil_images = [_to_pil(image) for image in images]
    x_offsets = [0] * len(pil_images)
    merged_width = pil_images[0].width
    for index in range(1, len(pil_images)):
        image_width = pil_images[index].width
        step_width = max(merged_width, image_width)
        align = aligns[index - 1]
        if align == "center":
            first_x = (step_width - merged_width) // 2
            second_x = (step_width - image_width) // 2
        elif align == "right":
            first_x = step_width - merged_width
            second_x = step_width - image_width
        else:
            first_x = 0
            second_x = 0
        for previous in range(index):
            x_offsets[previous] += first_x
        x_offsets[index] = second_x
        merged_width = step_width
    total_height = sum(image.height for image in pil_images)
    canvas = Image.new("RGB", (merged_width, total_height), (255, 255, 255))
    y_offset = 0
    for image, x_offset in zip(pil_images, x_offsets):
        canvas.paste(image, (x_offset, y_offset))
        y_offset += image.height
    return np.asarray(canvas)


def merge_blocks(
    blocks: list[dict[str, Any]],
    non_merge_labels: list[str],
) -> list[dict[str, Any]]:
    mergeable = [
        (index, block)
        for index, block in enumerate(blocks)
        if block["label"] not in non_merge_labels
    ]
    non_merge = {
        index: block
        for index, block in enumerate(blocks)
        if block["label"] in non_merge_labels
    }
    groups: list[tuple[list[int], list[str]]] = []
    current_indices: list[int] = []
    current_aligns: list[str] = []

    def aligned(first: float, second: float) -> bool:
        return abs(first - second) <= 5

    def overlaps_non_merge(first_index: int, second_index: int) -> bool:
        first = blocks[first_index]["box"]
        second = blocks[second_index]["box"]
        combined = [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        ]
        return any(
            index not in {first_index, second_index}
            and block["label"] in non_merge_labels
            and _box_overlap_ratio(combined, block["box"]) > 0
            for index, block in enumerate(blocks)
        )

    for position, (index, block) in enumerate(mergeable):
        if not current_indices:
            current_indices = [index]
            continue
        previous_index, previous = mergeable[position - 1]
        box = block["box"]
        previous_box = previous["box"]
        horizontal_overlap = _projection_overlap_ratio(
            box,
            previous_box,
            "horizontal",
        )
        crosses = (
            horizontal_overlap == 0
            and block["label"] == "text"
            and block["label"] == previous["label"]
            and box[0] > previous_box[2]
            and box[1] < previous_box[3]
            and box[0] - previous_box[2]
            < max(
                previous_box[2] - previous_box[0],
                box[2] - box[0],
            )
            * 0.3
        )
        stacked = (
            horizontal_overlap > 0
            and block["label"] == "text"
            and block["label"] == previous["label"]
            and box[3] >= previous_box[1]
            and abs(box[1] - previous_box[3])
            < max(
                previous_box[3] - previous_box[1],
                box[3] - box[1],
            )
            * 0.5
            and (
                aligned(box[0], previous_box[0])
                ^ aligned(box[2], previous_box[2])
            )
            and overlaps_non_merge(index, previous_index)
        )
        if crosses or stacked:
            if crosses:
                align = "center"
            elif aligned(box[0], previous_box[0]):
                align = "left"
            elif aligned(box[2], previous_box[2]):
                align = "right"
            else:
                align = "center"
            current_indices.append(index)
            current_aligns.append(align)
        else:
            groups.append((current_indices, current_aligns))
            current_indices = [index]
            current_aligns = []
    if current_indices:
        groups.append((current_indices, current_aligns))

    ranges = [
        (min(indices), max(indices), indices, aligns)
        for indices, aligns in groups
    ]
    outputs: list[dict[str, Any]] = []
    used: set[int] = set()
    index = 0
    while index < len(blocks):
        matched = False
        for start, end, indices, aligns in ranges:
            if index != start or any(item in used for item in indices):
                continue
            matched = True
            images = [blocks[item]["img"] for item in indices]
            width = max(image.shape[1] for image in images)
            height = sum(image.shape[0] for image in images)
            if height / width >= 3:
                for item in indices:
                    block = blocks[item].copy()
                    block["img"] = blocks[item]["img"]
                    block["merge_aligns"] = None
                    outputs.append(block)
                    used.add(item)
            else:
                merged = _merge_images(images, aligns)
                for position, item in enumerate(indices):
                    block = blocks[item].copy()
                    block["img"] = merged if position == 0 else None
                    block["merge_aligns"] = aligns if position == 0 else None
                    block["group_id"] = indices[0]
                    outputs.append(block)
                    used.add(item)
            for item in range(start + 1, end):
                if item in non_merge:
                    outputs.append(non_merge[item])
                    used.add(item)
            index = end + 1
            break
        if matched:
            continue
        if index in non_merge and index not in used:
            outputs.append(non_merge[index])
            used.add(index)
        index += 1
    return outputs


def crop_formula_margin(image: np.ndarray) -> np.ndarray:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    minimum = int(gray.min())
    maximum = int(gray.max())
    if minimum == maximum:
        return image
    lookup = np.zeros(256, dtype=np.uint8)
    for value in range(minimum, maximum + 1):
        lookup[value] = int(
            (value - minimum) / (maximum - minimum) * 255
        )
    normalized = cv2.LUT(gray, lookup)
    _, binary = cv2.threshold(
        normalized,
        200,
        255,
        cv2.THRESH_BINARY_INV,
    )
    coordinates = cv2.findNonZero(binary)
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    return image[y : y + height, x : x + width]
