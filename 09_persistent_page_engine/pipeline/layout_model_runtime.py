"""Self-contained PP-DocLayoutV3 model and mask runtime optimizations."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class _CropRectangleFastPath:
    """Skip polygon masking when the mask is provably the whole crop."""

    def __init__(self, original: Any) -> None:
        self.original = original
        self._lock = threading.Lock()
        self._calls = 0
        self._boxes = 0
        self._polygon_boxes = 0
        self._rectangle_fast_paths = 0
        self._fallbacks = 0
        self._wall_ns = 0
        self._rectangle_ns = 0
        self._fallback_ns = 0

    @staticmethod
    def _is_full_crop_rectangle(box_info: dict[str, Any]) -> bool:
        polygon_points = box_info.get("polygon_points")
        if polygon_points is None:
            return False

        xmin, ymin, xmax, ymax = [
            int(value) for value in box_info["coordinate"]
        ]
        if xmin >= xmax or ymin >= ymax:
            return False

        # CropByBoxes casts the polygon to int32 before cv2.fillPoly. Match that
        # cast before proving that the four points cover the entire sliced
        # [ymin:ymax, xmin:xmax] image.
        polygon = np.asarray(polygon_points, dtype=np.int32)
        if polygon.size != 8:
            return False
        polygon = polygon.reshape(4, 2)
        expected = {
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        }
        return {tuple(point) for point in polygon.tolist()} == expected

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[dict[str, Any]],
        layout_shape_mode: str = "auto",
    ) -> list[dict[str, Any]]:
        started_ns = time.perf_counter_ns()
        outputs: list[dict[str, Any]] = []
        polygon_boxes = 0
        rectangle_fast_paths = 0
        fallbacks = 0
        rectangle_ns = 0
        fallback_ns = 0

        for box_info in boxes:
            has_polygon = (
                layout_shape_mode != "rect"
                and "polygon_points" in box_info
            )
            polygon_boxes += int(has_polygon)
            if has_polygon and self._is_full_crop_rectangle(box_info):
                rectangle_started_ns = time.perf_counter_ns()
                label_id = box_info["cls_id"]
                box = box_info["coordinate"]
                label = box_info.get("label", label_id)
                xmin, ymin, xmax, ymax = [int(value) for value in box]
                outputs.append(
                    {
                        "img": image[ymin:ymax, xmin:xmax].copy(),
                        "box": box,
                        "label": label,
                        "polygon_points": box_info["polygon_points"],
                    }
                )
                rectangle_ns += (
                    time.perf_counter_ns() - rectangle_started_ns
                )
                rectangle_fast_paths += 1
                continue

            fallback_started_ns = time.perf_counter_ns()
            single = self.original(
                image,
                [box_info],
                layout_shape_mode,
            )
            fallback_ns += time.perf_counter_ns() - fallback_started_ns
            if len(single) != 1:
                raise RuntimeError(
                    "PaddleX CropByBoxes returned "
                    f"{len(single)} crops for one layout box"
                )
            outputs.append(single[0])
            fallbacks += 1

        finished_ns = time.perf_counter_ns()
        with self._lock:
            self._calls += 1
            self._boxes += len(boxes)
            self._polygon_boxes += polygon_boxes
            self._rectangle_fast_paths += rectangle_fast_paths
            self._fallbacks += fallbacks
            self._wall_ns += finished_ns - started_ns
            self._rectangle_ns += rectangle_ns
            self._fallback_ns += fallback_ns
        return outputs

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "boxes": self._boxes,
                "polygon_boxes": self._polygon_boxes,
                "rectangle_fast_paths": self._rectangle_fast_paths,
                "fallbacks": self._fallbacks,
                "coverage": (
                    self._rectangle_fast_paths / self._polygon_boxes
                    if self._polygon_boxes
                    else 0.0
                ),
                "wall_s": self._wall_ns / 1_000_000_000,
                "rectangle_s": self._rectangle_ns / 1_000_000_000,
                "fallback_s": self._fallback_ns / 1_000_000_000,
            }


class _PolygonNormalizationFastPath:
    """Resolve contained axis-aligned rectangles without OpenCV or Shapely."""

    def __init__(self, original: Any) -> None:
        self.original = original
        self._lock = threading.Lock()
        self._calls = 0
        self._rectangle_fast_paths = 0
        self._fallbacks = 0
        self._wall_ns = 0
        self._predicate_ns = 0
        self._fallback_ns = 0

    @staticmethod
    def _contained_rectangle_result(
        box: Any,
        polygon: Any,
        layout_shape_mode: str,
    ) -> np.ndarray | None:
        if layout_shape_mode != "auto" or polygon is None:
            return None

        points = np.asarray(polygon, dtype=np.float32)
        if points.size != 8:
            return None
        points = points.reshape(4, 2)
        x_values = np.unique(points[:, 0])
        y_values = np.unique(points[:, 1])
        if len(x_values) != 2 or len(y_values) != 2:
            return None
        expected = {
            (float(x_values[0]), float(y_values[0])),
            (float(x_values[1]), float(y_values[0])),
            (float(x_values[1]), float(y_values[1])),
            (float(x_values[0]), float(y_values[1])),
        }
        if {tuple(point) for point in points.tolist()} != expected:
            return None

        # The original normalizer converts the detected box to int32 before
        # constructing its rectangle.
        x_min, y_min, x_max, y_max = np.asarray(box).astype(np.int32)
        polygon_x_min = float(x_values[0])
        polygon_x_max = float(x_values[1])
        polygon_y_min = float(y_values[0])
        polygon_y_max = float(y_values[1])
        if not (
            polygon_x_min >= x_min
            and polygon_y_min >= y_min
            and polygon_x_max <= x_max
            and polygon_y_max <= y_max
        ):
            return None

        rect_area = float((x_max - x_min) * (y_max - y_min))
        polygon_area = (
            (polygon_x_max - polygon_x_min)
            * (polygon_y_max - polygon_y_min)
        )
        if rect_area <= 0 or polygon_area / rect_area < 0.95:
            return None

        # For a contained axis-aligned rectangle this is exactly the same
        # union IoU tested by the original Shapely path.
        return np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )

    def __call__(
        self,
        box: Any,
        polygon: Any,
        layout_shape_mode: str,
        previous_polygon: Any = None,
    ) -> Any:
        started_ns = time.perf_counter_ns()
        predicate_started_ns = time.perf_counter_ns()
        rectangle = self._contained_rectangle_result(
            box,
            polygon,
            layout_shape_mode,
        )
        predicate_ns = time.perf_counter_ns() - predicate_started_ns

        fallback_ns = 0
        if rectangle is not None:
            result = rectangle
            rectangle_fast_paths = 1
            fallbacks = 0
        else:
            fallback_started_ns = time.perf_counter_ns()
            result = self.original(
                box,
                polygon,
                layout_shape_mode,
                previous_polygon,
            )
            fallback_ns = time.perf_counter_ns() - fallback_started_ns
            rectangle_fast_paths = 0
            fallbacks = 1

        finished_ns = time.perf_counter_ns()
        with self._lock:
            self._calls += 1
            self._rectangle_fast_paths += rectangle_fast_paths
            self._fallbacks += fallbacks
            self._wall_ns += finished_ns - started_ns
            self._predicate_ns += predicate_ns
            self._fallback_ns += fallback_ns
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "rectangle_fast_paths": self._rectangle_fast_paths,
                "fallbacks": self._fallbacks,
                "coverage": (
                    self._rectangle_fast_paths / self._calls
                    if self._calls
                    else 0.0
                ),
                "wall_s": self._wall_ns / 1_000_000_000,
                "predicate_s": self._predicate_ns / 1_000_000_000,
                "fallback_s": self._fallback_ns / 1_000_000_000,
            }


def _full_border_contour(box: Any) -> np.ndarray:
    """Return the contour produced by a full resized binary mask."""

    x_min, y_min, x_max, y_max = (int(value) for value in box)
    return np.array(
        [
            [x_min, y_min],
            [x_min, y_max - 1],
            [x_max - 1, y_max - 1],
            [x_max - 1, y_min],
        ],
        dtype=np.int32,
    )


def _extract_custom_vertices_vectorized(
    _processor: Any,
    polygon: Any,
    sharp_angle_thresh: float = 45,
) -> list[tuple[Any, Any]]:
    """Preserve PP-DocLayoutV3's vertex rule without its Python point loop."""

    points = np.asarray(polygon)
    if len(points) == 0:
        return []
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    vector_1 = previous - points
    vector_2 = following - points
    cross_products = (
        vector_1[:, 1] * vector_2[:, 0]
        - vector_1[:, 0] * vector_2[:, 1]
    )
    if (
        points.shape == (4, 2)
        and np.all(cross_products < 0)
        and np.all(np.sum(vector_1 * vector_2, axis=1) == 0)
    ):
        return [tuple(point) for point in points.astype(np.float64)]

    selected = cross_products < 0
    if not np.any(selected):
        return []

    output = points[selected].astype(np.float64, copy=True)
    selected_vector_1 = vector_1[selected]
    selected_vector_2 = vector_2[selected]
    norm_1 = np.linalg.norm(selected_vector_1, axis=1)
    norm_2 = np.linalg.norm(selected_vector_2, axis=1)
    angle_cos = np.clip(
        np.sum(selected_vector_1 * selected_vector_2, axis=1)
        / (norm_1 * norm_2),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(angle_cos))
    sharp = np.abs(angles - sharp_angle_thresh) < 1
    if np.any(sharp):
        direction = (
            selected_vector_1[sharp] / norm_1[sharp, None]
            + selected_vector_2[sharp] / norm_2[sharp, None]
        )
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        step_size = (norm_1[sharp] + norm_2[sharp]) / 2
        output[sharp] += direction * step_size[:, None]
    return [tuple(point) for point in output]


def _extract_polygon_points_by_masks_owned(
    processor: Any,
    boxes: Any,
    masks: Any,
    scale_ratio: Any,
) -> list[Any]:
    """Preserve the Transformers mask contract without redundant copies."""

    boxes_np = np.asarray(boxes)
    masks_np = np.asarray(masks)
    scale_width = float(scale_ratio[0]) / 4.0
    scale_height = float(scale_ratio[1]) / 4.0
    mask_height, mask_width = masks_np.shape[1:]
    polygon_points: list[Any] = []

    for box, mask in zip(boxes_np, masks_np):
        x_min, y_min, x_max, y_max = (int(value) for value in box)
        box_width = x_max - x_min
        box_height = y_max - y_min
        rect = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )
        if box_width <= 0 or box_height <= 0:
            polygon_points.append(rect)
            continue

        x_start = max(
            0,
            min(mask_width, int(round(x_min * scale_width))),
        )
        x_end = max(
            0,
            min(mask_width, int(round(x_max * scale_width))),
        )
        y_start = max(
            0,
            min(mask_height, int(round(y_min * scale_height))),
        )
        y_end = max(
            0,
            min(mask_height, int(round(y_max * scale_height))),
        )
        cropped_mask = mask[y_start:y_end, x_start:x_end]
        if cropped_mask.dtype != np.uint8:
            cropped_mask = cropped_mask.astype(np.uint8)
        resized_mask = cv2.resize(
            cropped_mask,
            (box_width, box_height),
            interpolation=cv2.INTER_NEAREST,
        )
        polygon = processor._mask2polygon(resized_mask)
        if polygon is not None and len(polygon) < 4:
            polygon_points.append(rect)
            continue
        if polygon is not None and len(polygon) > 0:
            polygon = polygon + np.array([x_min, y_min])
        polygon_points.append(polygon)

    return polygon_points


class _MaskRectangleFastPath:
    """Skip contour extraction when it is guaranteed to normalize to the box."""

    def __init__(self, original: Any) -> None:
        self.original = original
        self._fallback_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="layout-mask",
        )
        self._lock = threading.Lock()
        self._calls = 0
        self._detections = 0
        self._rectangles = 0
        self._fallbacks = 0
        self._wall_ns = 0
        self._predicate_ns = 0
        self._fallback_ns = 0

    @staticmethod
    def _is_full_external_rectangle(
        box: np.ndarray,
        mask: np.ndarray,
        scale_ratio: Any,
    ) -> bool:
        x_min, y_min, x_max, y_max = (int(value) for value in box)
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 0 or box_height <= 0:
            return False
        if box_width == 1 or box_height == 1:
            return False

        # approxPolyDP must retain all four corners. Its epsilon is 0.004 times
        # the closed contour perimeter.
        contour_width = box_width - 1
        contour_height = box_height - 1
        epsilon = 0.008 * (contour_width + contour_height)
        if epsilon >= min(contour_width, contour_height):
            return False

        # A full external border survives nearest-neighbour resize as a full
        # rectangle. RETR_EXTERNAL therefore sees the four box corners even if
        # there are holes inside the mask.
        scale_width = float(scale_ratio[0]) / 4.0
        scale_height = float(scale_ratio[1]) / 4.0
        mask_height, mask_width = mask.shape
        x_start = max(
            0,
            min(mask_width, int(round(x_min * scale_width))),
        )
        x_end = max(
            0,
            min(mask_width, int(round(x_max * scale_width))),
        )
        y_start = max(
            0,
            min(mask_height, int(round(y_min * scale_height))),
        )
        y_end = max(
            0,
            min(mask_height, int(round(y_max * scale_height))),
        )
        if x_start >= x_end or y_start >= y_end:
            return False
        cropped = mask[y_start:y_end, x_start:x_end]
        if not (
            cropped[0].all()
            and cropped[-1].all()
            and cropped[:, 0].all()
            and cropped[:, -1].all()
        ):
            return False

        return True

    def __call__(
        self,
        boxes: Any,
        masks: Any,
        scale_ratio: Any,
    ) -> list[Any]:
        started_ns = time.perf_counter_ns()
        boxes_np = np.asarray(boxes)
        masks_np = np.asarray(masks)
        polygons: list[Any] = [None] * len(boxes_np)
        fallback_indices: list[int] = []
        rectangles = 0
        predicate_ns = 0

        for index, box in enumerate(boxes_np):
            predicate_started_ns = time.perf_counter_ns()
            is_candidate = self._is_full_external_rectangle(
                box,
                masks_np[index],
                scale_ratio,
            )
            predicate_ns += time.perf_counter_ns() - predicate_started_ns

            if is_candidate:
                polygons[index] = _full_border_contour(box)
                rectangles += 1
            else:
                fallback_indices.append(index)

        fallback_started_ns = time.perf_counter_ns()
        if fallback_indices:
            fallback_boxes = boxes_np[fallback_indices]
            fallback_masks = masks_np[fallback_indices]
            if len(fallback_indices) < 4:
                fallback_polygons = self.original(
                    fallback_boxes,
                    fallback_masks,
                    scale_ratio,
                )
            else:
                worker_count = min(4, len(fallback_indices))
                chunk_size = (
                    len(fallback_indices) + worker_count - 1
                ) // worker_count
                futures = [
                    self._fallback_executor.submit(
                        self.original,
                        fallback_boxes[start : start + chunk_size],
                        fallback_masks[start : start + chunk_size],
                        scale_ratio,
                    )
                    for start in range(
                        0,
                        len(fallback_indices),
                        chunk_size,
                    )
                ]
                fallback_polygons = [
                    polygon
                    for future in futures
                    for polygon in future.result()
                ]
            if len(fallback_polygons) != len(fallback_indices):
                raise RuntimeError(
                    "PP-DocLayout mask extractor returned "
                    f"{len(fallback_polygons)} polygons for "
                    f"{len(fallback_indices)} detections"
                )
            for index, polygon in zip(
                fallback_indices,
                fallback_polygons,
            ):
                polygons[index] = polygon
        fallback_ns = time.perf_counter_ns() - fallback_started_ns
        fallbacks = len(fallback_indices)

        finished_ns = time.perf_counter_ns()
        with self._lock:
            self._calls += 1
            self._detections += len(boxes_np)
            self._rectangles += rectangles
            self._fallbacks += fallbacks
            self._wall_ns += finished_ns - started_ns
            self._predicate_ns += predicate_ns
            self._fallback_ns += fallback_ns
        return polygons

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "detections": self._detections,
                "rectangle_fast_paths": self._rectangles,
                "fallbacks": self._fallbacks,
                "coverage": (
                    self._rectangles / self._detections
                    if self._detections
                    else 0.0
                ),
                "wall_s": self._wall_ns / 1_000_000_000,
                "predicate_s": self._predicate_ns / 1_000_000_000,
                "fallback_s": self._fallback_ns / 1_000_000_000,
            }


def _pairwise_containment(boxes: np.ndarray) -> np.ndarray:
    """Return matrix[i, j] when box i is at least 90% inside box j."""

    coords = boxes[:, 2:6]
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    intersection_width = np.maximum(
        0,
        np.minimum(x2[:, None], x2[None, :])
        - np.maximum(x1[:, None], x1[None, :]),
    )
    intersection_height = np.maximum(
        0,
        np.minimum(y2[:, None], y2[None, :])
        - np.maximum(y1[:, None], y1[None, :]),
    )
    intersection_area = intersection_width * intersection_height
    source_area = (x2 - x1) * (y2 - y1)

    ratios = np.divide(
        intersection_area,
        source_area[:, None],
        out=np.zeros_like(intersection_area),
        where=source_area[:, None] > 0,
    )
    contained = ratios >= 0.9
    np.fill_diagonal(contained, False)
    return contained


def _vectorized_check_containment(
    boxes: Any,
    formula_index: int | None = None,
    category_index: int | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of PaddleX's pairwise containment loop."""

    boxes_array = np.asarray(boxes)
    count = len(boxes_array)
    contains_other = np.zeros(count, dtype=int)
    contained_by_other = np.zeros(count, dtype=int)
    if count == 0:
        return contains_other, contained_by_other

    contained = _pairwise_containment(boxes_array)
    classes = boxes_array[:, 0]

    if formula_index is not None:
        contained &= ~(
            (classes[:, None] == formula_index)
            & (classes[None, :] != formula_index)
        )

    if category_index is not None and mode is not None:
        if mode == "large":
            contained &= classes[None, :] == category_index
        elif mode == "small":
            contained &= classes[:, None] == category_index
        else:
            contained.fill(False)

    contained_by_other[contained.any(axis=1)] = 1
    contains_other[contained.any(axis=0)] = 1
    return contains_other, contained_by_other


def _pairwise_iou(coords: np.ndarray) -> np.ndarray:
    """Match PaddleX's inclusive-coordinate IoU for every box pair."""

    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    intersection_width = np.maximum(
        0,
        np.minimum(x2[:, None], x2[None, :])
        - np.maximum(x1[:, None], x1[None, :])
        + 1,
    )
    intersection_height = np.maximum(
        0,
        np.minimum(y2[:, None], y2[None, :])
        - np.maximum(y1[:, None], y1[None, :])
        + 1,
    )
    intersection_area = intersection_width * intersection_height
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return intersection_area / (
        area[:, None] + area[None, :] - intersection_area
    )


def _vectorized_nms(
    boxes: Any,
    iou_same: float = 0.6,
    iou_diff: float = 0.95,
) -> list[int]:
    """Preserve PaddleX NMS selection order without its Python inner loop."""

    boxes_array = np.asarray(boxes)
    if len(boxes_array) == 0:
        return []

    classes = boxes_array[:, 0]
    ious = _pairwise_iou(boxes_array[:, 2:6])
    remaining = np.argsort(boxes_array[:, 1])[::-1]
    selected: list[int] = []

    while remaining.size:
        current = int(remaining[0])
        selected.append(current)
        candidates = remaining[1:]
        thresholds = np.where(
            classes[candidates] == classes[current],
            iou_same,
            iou_diff,
        )
        remaining = candidates[ious[current, candidates] < thresholds]

    return selected


def _post_process_selected_masks_only(
    processor: Any,
    outputs: Any,
    threshold: float = 0.5,
    target_sizes: Any = None,
) -> list[dict[str, Any]]:
    """Preserve detector results while processing masks after selection."""

    boxes = outputs.pred_boxes
    logits = outputs.logits
    order_logits = outputs.order_logits
    use_polygons = getattr(processor, "_layout_polygon_mode", "mask") == "mask"
    masks = outputs.out_masks if use_polygons else None
    if target_sizes is None:
        raise ValueError("layout postprocessing requires target image sizes")
    if len(logits) != len(target_sizes):
        raise ValueError(
            "target size count must match detector batch dimension"
        )

    order_seqs = processor._get_order_seqs(order_logits)
    box_centers, box_dimensions = torch.split(boxes, 2, dim=-1)
    boxes = torch.cat(
        [
            box_centers - 0.5 * box_dimensions,
            box_centers + 0.5 * box_dimensions,
        ],
        dim=-1,
    )
    if isinstance(target_sizes, list):
        image_height, image_width = torch.as_tensor(target_sizes).unbind(1)
    else:
        image_height, image_width = target_sizes.unbind(1)
    scale_factor = torch.stack(
        [image_width, image_height, image_width, image_height],
        dim=1,
    ).to(boxes.device)
    boxes = boxes * scale_factor[:, None, :]

    query_count = logits.shape[1]
    class_count = logits.shape[2]
    scores = torch.sigmoid(logits)
    scores, flattened_indices = torch.topk(
        scores.flatten(1),
        query_count,
        dim=-1,
    )
    labels = flattened_indices % class_count
    query_indices = flattened_indices // class_count

    results: list[dict[str, Any]] = []
    for batch_index, target_size in enumerate(target_sizes):
        kept_positions = torch.nonzero(
            scores[batch_index] >= threshold,
            as_tuple=False,
        ).squeeze(-1)
        kept_order = order_seqs[batch_index].gather(
            0,
            query_indices[batch_index].gather(0, kept_positions),
        )
        kept_order, order_indices = torch.sort(kept_order)
        selected_positions = kept_positions.gather(0, order_indices)
        selected_queries = query_indices[batch_index].gather(
            0,
            selected_positions,
        )

        selected_scores = scores[batch_index].gather(
            0,
            selected_positions,
        )
        selected_labels = labels[batch_index].gather(
            0,
            selected_positions,
        )
        selected_boxes = boxes[batch_index].index_select(
            0,
            selected_queries,
        )
        cpu_boxes = selected_boxes.detach().cpu()
        cpu_scores = selected_scores.detach().cpu()
        cpu_labels = selected_labels.detach().cpu()
        cpu_order = kept_order.detach().cpu()
        result = {
            "scores": cpu_scores,
            "labels": cpu_labels,
            "boxes": cpu_boxes,
            "order_seq": cpu_order,
        }
        if use_polygons:
            if masks is None:
                raise AssertionError("layout mask output is unavailable")
            selected_masks = masks[batch_index].index_select(
                0,
                selected_queries,
            )
            selected_masks = selected_masks.sigmoid() > threshold
            cpu_masks = selected_masks.detach().cpu()
            result["polygon_points"] = (
                processor._extract_polygon_points_by_masks(
                    cpu_boxes.numpy(),
                    cpu_masks.numpy(),
                    [
                        processor.size["width"] / target_size[1],
                        processor.size["height"] / target_size[0],
                    ],
                )
            )
        results.append(result)

    return results


def _process_with_tensor_inputs(
    predictor: Any,
    batch_data: Any,
    threshold: float | dict[int, float] | None = None,
    layout_nms: bool = False,
    layout_unclip_ratio: float | tuple[float, float] | dict[int, Any] | None = None,
    layout_merge_bboxes_mode: str | dict[int, str] | None = None,
    layout_shape_mode: str = "auto",
    filter_overlap_boxes: bool = True,
    skip_order_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror PaddleX processing while avoiding NumPy -> PIL -> Torch copies."""

    if not hasattr(predictor.image_processor, "post_process_object_detection"):
        raise RuntimeError(
            f"{type(predictor.image_processor).__name__} does not support "
            "`post_process_object_detection`."
        )

    datas = predictor.read_op(batch_data.instances)
    images = [
        torch.from_numpy(data["img"]).permute(2, 0, 1)
        for data in datas
    ]
    effective_threshold, hf_threshold = predictor._get_hf_threshold(threshold)

    model_inputs = predictor.preprocess_images(images=images)
    outputs = predictor.forward(model_inputs)
    predictions = predictor.postprocess(
        outputs,
        datas=datas,
        threshold=hf_threshold,
    )

    batch_outputs = [
        predictor._format_layout_transformers_output(prediction)
        for prediction in predictions
    ]
    boxes = predictor.layout_postprocess(
        batch_outputs,
        datas,
        threshold=effective_threshold,
        layout_nms=layout_nms or predictor.layout_nms,
        layout_unclip_ratio=(
            layout_unclip_ratio or predictor.layout_unclip_ratio
        ),
        layout_merge_bboxes_mode=(
            layout_merge_bboxes_mode or predictor.layout_merge_bboxes_mode
        ),
        layout_shape_mode=layout_shape_mode,
        filter_overlap_boxes=filter_overlap_boxes,
        skip_order_labels=skip_order_labels,
    )

    return {
        "input_path": batch_data.input_paths,
        "page_index": batch_data.page_indexes,
        "input_img": [data["ori_img"] for data in datas],
        "boxes": boxes,
    }


def _decoder_forward_final_heads_only(
    decoder: Any,
    inputs_embeds: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    reference_points: torch.Tensor | None = None,
    spatial_shapes: torch.Tensor | None = None,
    spatial_shapes_list: Any = None,
    level_start_index: torch.Tensor | None = None,
    order_head: Any = None,
    global_pointer: Any = None,
    mask_query_head: Any = None,
    norm: Any = None,
    mask_feat: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    """Run every decoder layer but materialize inference heads only once."""

    from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import (
        PPDocLayoutV3DecoderOutput,
        inverse_sigmoid,
    )

    if inputs_embeds is None or reference_points is None:
        raise RuntimeError(
            "PP-DocLayoutV3 inference requires query embeddings and "
            "reference points"
        )

    hidden_states = inputs_embeds
    reference_points = F.sigmoid(reference_points)

    for index, decoder_layer in enumerate(decoder.layers):
        reference_points_input = reference_points.unsqueeze(2)
        object_queries_position_embeddings = decoder.query_pos_head(
            reference_points
        )
        hidden_states = decoder_layer(
            hidden_states,
            object_queries_position_embeddings=(
                object_queries_position_embeddings
            ),
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points_input,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

        if decoder.bbox_embed is not None:
            predicted_corners = decoder.bbox_embed(hidden_states)
            new_reference_points = F.sigmoid(
                predicted_corners + inverse_sigmoid(reference_points)
            )
            reference_points = new_reference_points.detach()

    out_query = norm(hidden_states)
    if getattr(decoder, "_layout_emit_masks", True):
        mask_query_embed = mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        _, _, mask_height, mask_width = mask_feat.shape
        out_mask = torch.bmm(
            mask_query_embed,
            mask_feat.flatten(start_dim=2),
        ).reshape(batch_size, mask_dim, mask_height, mask_width)
    else:
        out_mask = hidden_states[..., :1].unsqueeze(-1)

    logits = (
        decoder.class_embed(out_query)
        if decoder.class_embed is not None
        else None
    )
    order_logits = None
    if order_head is not None and global_pointer is not None:
        valid_query = (
            out_query[:, -decoder.num_queries :]
            if decoder.num_queries is not None
            else out_query
        )
        order_logits = global_pointer(order_head[index](valid_query))

    return PPDocLayoutV3DecoderOutput(
        last_hidden_state=hidden_states,
        intermediate_hidden_states=hidden_states.unsqueeze(1),
        intermediate_logits=(
            logits.unsqueeze(1) if logits is not None else None
        ),
        intermediate_reference_points=reference_points.unsqueeze(1),
        decoder_out_order_logits=(
            order_logits.unsqueeze(1) if order_logits is not None else None
        ),
        decoder_out_masks=out_mask.unsqueeze(1),
    )


def _mask_to_box_capture_friendly(
    mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Original mask bounds without host-to-device constants during capture."""

    mask = mask.bool()
    height, width = mask.shape[-2:]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=mask.device),
        torch.arange(width, device=mask.device),
        indexing="ij",
    )
    x_coords = x_coords.to(dtype)
    y_coords = y_coords.to(dtype)

    x_coords_masked = x_coords * mask
    x_max = x_coords_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    x_min = torch.where(
        mask,
        x_coords_masked,
        torch.full_like(x_coords_masked, torch.finfo(dtype).max),
    ).flatten(start_dim=-2).min(dim=-1).values

    y_coords_masked = y_coords * mask
    y_max = y_coords_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    y_min = torch.where(
        mask,
        y_coords_masked,
        torch.full_like(y_coords_masked, torch.finfo(dtype).max),
    ).flatten(start_dim=-2).min(dim=-1).values

    unnormalized_bbox = torch.stack(
        [x_min, y_min, x_max, y_max],
        dim=-1,
    )
    is_mask_non_empty = torch.any(mask, dim=(-2, -1)).unsqueeze(-1)
    unnormalized_bbox = unnormalized_bbox * is_mask_non_empty
    device_width = x_coords[-1, -1] + 1
    device_height = y_coords[-1, -1] + 1
    normalized_bbox = unnormalized_bbox / torch.stack(
        [device_width, device_height, device_width, device_height]
    )
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalized_bbox.unbind(
        dim=-1
    )
    return torch.stack(
        [
            (x_min_norm + x_max_norm) / 2,
            (y_min_norm + y_max_norm) / 2,
            x_max_norm - x_min_norm,
            y_max_norm - y_min_norm,
        ],
        dim=-1,
    )


class _NpuGraphCoreForward:
    """Capture the fixed-shape core model once and replay it per page."""

    def __init__(self, eager_forward: Any) -> None:
        self.eager_forward = eager_forward
        self.graph: Any = None
        self.output: Any = None
        self.pixel_values: torch.Tensor | None = None
        self.pixel_mask: torch.Tensor | None = None
        self.constant_kwargs: dict[str, Any] | None = None

    def _validate_constants(
        self,
        *,
        encoder_outputs: Any,
        labels: Any,
        kwargs: dict[str, Any],
    ) -> None:
        if encoder_outputs is not None or labels is not None:
            raise RuntimeError(
                "layout NPU graph only supports inference without supplied "
                "encoder outputs or labels"
            )
        if self.constant_kwargs is not None and kwargs != self.constant_kwargs:
            raise RuntimeError(
                "layout NPU graph received different non-tensor arguments"
            )

    def __call__(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        encoder_outputs: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        self._validate_constants(
            encoder_outputs=encoder_outputs,
            labels=labels,
            kwargs=kwargs,
        )
        if self.graph is None:
            self.pixel_values = torch.empty_like(pixel_values)
            self.pixel_values.copy_(pixel_values)
            if pixel_mask is not None:
                self.pixel_mask = torch.empty_like(pixel_mask)
                self.pixel_mask.copy_(pixel_mask)
            self.constant_kwargs = dict(kwargs)

            self.eager_forward(
                self.pixel_values,
                pixel_mask=self.pixel_mask,
                **self.constant_kwargs,
            )
            torch.npu.synchronize()

            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph):
                self.output = self.eager_forward(
                    self.pixel_values,
                    pixel_mask=self.pixel_mask,
                    **self.constant_kwargs,
                )
            torch.npu.synchronize()
        else:
            if self.pixel_values is None:
                raise AssertionError("captured layout input is missing")
            if pixel_values.shape != self.pixel_values.shape:
                raise RuntimeError(
                    "layout NPU graph input shape changed from "
                    f"{tuple(self.pixel_values.shape)} to "
                    f"{tuple(pixel_values.shape)}"
                )
            self.pixel_values.copy_(pixel_values)
            if pixel_mask is None:
                if self.pixel_mask is not None:
                    raise RuntimeError("layout NPU graph pixel mask disappeared")
            else:
                if self.pixel_mask is None:
                    raise RuntimeError("layout NPU graph pixel mask appeared")
                self.pixel_mask.copy_(pixel_mask)

        self.graph.replay()
        return self.output
