"""Faithful runtime optimizations for the PaddleX PP-DocLayoutV3 path."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch


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
    return {
        "direct_tensor_preprocessing": True,
    }
