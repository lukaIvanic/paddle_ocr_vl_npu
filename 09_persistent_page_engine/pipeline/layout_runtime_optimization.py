"""Faithful runtime optimizations for the PaddleX PP-DocLayoutV3 path."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
    mask_query_embed = mask_query_head(out_query)
    batch_size, mask_dim, _ = mask_query_embed.shape
    _, _, mask_height, mask_width = mask_feat.shape
    out_mask = torch.bmm(
        mask_query_embed,
        mask_feat.flatten(start_dim=2),
    ).reshape(batch_size, mask_dim, mask_height, mask_width)

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


def install_layout_runtime_optimizations(
    paddlex_pipeline: Any,
    *,
    backend: str = "eager",
    torchair_cache_dir: Path | None = None,
) -> dict[str, Any]:
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

    model = getattr(getattr(predictor, "infer", None), "model", None)
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        decoder = getattr(
            getattr(getattr(model, "model", None), "decoder", None),
            "decoder",
            None,
        )
    if decoder is None:
        decoder = getattr(
            getattr(model, "model", None),
            "decoder",
            None,
        )
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError(
            "unsupported PP-DocLayoutV3 model; decoder was not found"
        )
    decoder.forward = MethodType(_decoder_forward_final_heads_only, decoder)

    compiled = False
    if backend == "torchair":
        if torchair_cache_dir is None:
            raise ValueError(
                "TorchAir layout execution requires a cache directory"
            )
        from paddleocr_vl.model.compile_utils import import_torchair
        from transformers.models.pp_doclayout_v3 import (
            modeling_pp_doclayout_v3,
        )

        torchair, CompilerConfig = import_torchair()
        torchair_cache_dir.mkdir(parents=True, exist_ok=True)
        modeling_pp_doclayout_v3.torch_compilable_check = (
            lambda *args, **kwargs: None
        )
        model.forward = torchair.inference.cache_compile(
            model.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(torchair_cache_dir),
            ge_cache=True,
        )
        compiled = True
    elif backend != "eager":
        raise ValueError(
            f"layout backend must be 'eager' or 'torchair', got {backend!r}"
        )

    return {
        "direct_tensor_preprocessing": True,
        "final_decoder_heads_only": True,
        "model_backend": backend,
        "model_compiled": compiled,
        "transformers_runtime_shape_checks": (
            "outside_static_graph" if compiled else "enabled"
        ),
        "torchair_cache_dir": (
            str(torchair_cache_dir) if torchair_cache_dir is not None else None
        ),
        "vectorized_geometry": True,
    }
