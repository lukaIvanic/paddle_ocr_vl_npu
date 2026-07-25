"""Faithful runtime optimizations for the PaddleX PP-DocLayoutV3 path."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import torch


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


def install_layout_runtime_optimizations(paddlex_pipeline: Any) -> dict[str, Any]:
    """Install output-preserving optimizations on the concrete PaddleX pipeline."""

    predictor = getattr(paddlex_pipeline, "layout_det_model", None)
    required = (
        "read_op",
        "image_processor",
        "preprocess_images",
        "forward",
        "postprocess",
        "layout_postprocess",
        "_format_layout_transformers_output",
    )
    missing = [name for name in required if not hasattr(predictor, name)]
    if missing:
        raise RuntimeError(
            "unsupported PaddleX layout predictor; missing "
            + ", ".join(sorted(missing))
        )

    predictor.process = MethodType(_process_with_tensor_inputs, predictor)

    from paddlex.inference.models.layout_analysis import (
        processors as layout_processors,
    )
    from paddlex.inference.models.object_detection import (
        processors as detection_processors,
    )

    for module in (layout_processors, detection_processors):
        module.nms = _vectorized_nms
        module.check_containment = _vectorized_check_containment

    return {
        "direct_tensor_preprocessing": True,
        "vectorized_geometry": True,
    }
