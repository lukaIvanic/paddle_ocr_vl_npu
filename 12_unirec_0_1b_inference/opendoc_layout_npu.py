"""Eager NPU PP-DocLayoutV2 adapter for the official OpenDoc pipeline."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from vision_focal_depthwise import (
    ConstantDepthwiseConv2d,
    register_focal_depthwise_constant_converter,
)


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}

LAYOUT_WEIGHT_FORMAT_CHOICES = (
    "native",
    "torchair_internal",
)
DEFAULT_LAYOUT_WEIGHT_FORMAT = "torchair_internal"

LAYOUT_DEPTHWISE_REWRITE_CHOICES = (
    "native",
    "constant_grouped",
)
DEFAULT_LAYOUT_DEPTHWISE_REWRITE = "constant_grouped"

LAYOUT_COGVIEW_ATTENTION_IMPL_CHOICES = (
    "stabilized",
    "direct_softmax",
)
DEFAULT_LAYOUT_COGVIEW_ATTENTION_IMPL = "stabilized"


class _PrecomputedLayoutAffine2d(torch.nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale + self.bias


def _rewrite_layout_depthwise_convs(
    model: torch.nn.Module,
    *,
    requested: str,
) -> dict[str, Any]:
    """Rewrite native layout depthwise filters without changing their math."""
    if requested not in LAYOUT_DEPTHWISE_REWRITE_CHOICES:
        raise ValueError(f"Unsupported layout depthwise rewrite: {requested}")
    if requested == "native":
        return {
            "requested": requested,
            "target_count": 0,
            "rewritten_count": 0,
            "modules": [],
        }
    rewritten: list[dict[str, Any]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Conv2d):
            continue
        channels = int(module.in_channels)
        is_native_depthwise = (
            module.groups == channels
            and module.out_channels == channels
            and tuple(module.weight.shape[:2]) == (channels, 1)
        )
        is_target = is_native_depthwise and module.bias is None
        if not is_target:
            continue
        replacement = ConstantDepthwiseConv2d(
            module,
            prepack_grouped=True,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = replacement
        rewritten.append(
            {
                "module": name,
                "channels": channels,
                "kernel": [int(value) for value in module.kernel_size],
                "stride": [int(value) for value in module.stride],
                "padding": [int(value) for value in module.padding],
                "original_groups": int(module.groups),
                "groups": channels,
                "group_width": 1,
                "weight_id": replacement.weight_id,
                "weight_shape": [
                    int(value) for value in module.weight.shape
                ],
                "weight_storage_shape": [
                    int(value) for value in replacement.packed_weight.shape
                ],
                "weight_binding": "frozen_prepacked_fractal_z_grouped",
            }
        )
    summary = {
        "requested": requested,
        "target_count": len(rewritten),
        "rewritten_count": len(rewritten),
        "modules": rewritten,
    }
    register_focal_depthwise_constant_converter()
    return summary


def _fuse_layout_frozen_batch_norms(model: torch.nn.Module) -> dict[str, Any]:
    """Fold inference-only FrozenBatchNorm2d modules into their Conv2d."""
    fused: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            convolution = getattr(module, "convolution", None)
            normalization = getattr(module, "normalization", None)
            if not isinstance(convolution, torch.nn.Conv2d):
                continue
            if type(normalization).__name__ != "PPDocLayoutV2FrozenBatchNorm2d":
                continue
            required = ("weight", "bias", "running_mean", "running_var")
            if not all(hasattr(normalization, field) for field in required):
                raise RuntimeError(f"unexpected frozen BN contract at {name}")
            scale = normalization.weight * (
                normalization.running_var + 1e-5
            ).rsqrt()
            source_bias = convolution.bias
            if source_bias is None:
                source_bias = torch.zeros_like(normalization.running_mean)
            fused_weight = convolution.weight * scale.reshape(-1, 1, 1, 1)
            fused_bias = (
                source_bias * scale
                + normalization.bias
                - normalization.running_mean * scale
            )
            convolution.weight = torch.nn.Parameter(
                fused_weight, requires_grad=False
            )
            convolution.bias = torch.nn.Parameter(fused_bias, requires_grad=False)
            module.normalization = torch.nn.Identity()
            fused.append(
                {
                    "module": name,
                    "channels": int(convolution.out_channels),
                    "weight_shape": [
                        int(value) for value in convolution.weight.shape
                    ],
                }
            )
    return {
        "fused_count": len(fused),
        "modules": fused,
    }


def _precompute_layout_frozen_bn_affines(
    model: torch.nn.Module,
    *,
    preformat_nc1hwc0: bool,
) -> dict[str, Any]:
    """Precompute FrozenBatchNorm affine tensors without Conv-BN reassociation."""
    torch_npu = None
    if preformat_nc1hwc0:
        import torch_npu as imported_torch_npu

        torch_npu = imported_torch_npu
    replaced: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            normalization = getattr(module, "normalization", None)
            if type(normalization).__name__ != "PPDocLayoutV2FrozenBatchNorm2d":
                continue
            weight = normalization.weight.reshape(1, -1, 1, 1)
            source_bias = normalization.bias.reshape(1, -1, 1, 1)
            running_var = normalization.running_var.reshape(1, -1, 1, 1)
            running_mean = normalization.running_mean.reshape(1, -1, 1, 1)
            scale = weight * (running_var + 1e-5).rsqrt()
            bias = source_bias - running_mean * scale
            if torch_npu is not None:
                scale = torch_npu.npu_format_cast(scale, 3)
                bias = torch_npu.npu_format_cast(bias, 3)
            module.normalization = _PrecomputedLayoutAffine2d(scale, bias)
            replaced.append(
                {
                    "module": name,
                    "channels": int(scale.shape[1]),
                    "scale_format": (
                        None
                        if torch_npu is None
                        else int(torch_npu.get_npu_format(scale))
                    ),
                    "bias_format": (
                        None
                        if torch_npu is None
                        else int(torch_npu.get_npu_format(bias))
                    ),
                }
            )
    return {"replaced_count": len(replaced), "modules": replaced}


def _preformat_layout_frozen_bn_buffers(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Store original FrozenBN buffers as NC1HWC0 without changing its math."""
    import torch_npu

    converted: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, normalization in list(model.named_modules()):
            if type(normalization).__name__ != "PPDocLayoutV2FrozenBatchNorm2d":
                continue
            formats: dict[str, int] = {}
            for field in ("weight", "bias", "running_var", "running_mean"):
                source = getattr(normalization, field)
                shaped = source.reshape(1, -1, 1, 1)
                formatted = torch_npu.npu_format_cast(shaped, 3)
                setattr(normalization, field, formatted)
                formats[field] = int(torch_npu.get_npu_format(formatted))
            converted.append(
                {
                    "module": name,
                    "channels": int(normalization.weight.shape[1]),
                    "formats": formats,
                }
            )
    return {"converted_count": len(converted), "modules": converted}


def _fuse_layout_eval_batch_norms(model: torch.nn.Module) -> dict[str, Any]:
    """Fold evaluation BatchNorm2d modules in ConvNormLayer into Conv2d."""
    fused: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            convolution = getattr(module, "conv", None)
            normalization = getattr(module, "norm", None)
            if not isinstance(convolution, torch.nn.Conv2d):
                continue
            if not isinstance(normalization, torch.nn.BatchNorm2d):
                continue
            if normalization.running_mean is None or normalization.running_var is None:
                raise RuntimeError(f"BatchNorm2d has no running stats at {name}")
            if normalization.affine:
                norm_weight = normalization.weight
                norm_bias = normalization.bias
            else:
                norm_weight = torch.ones_like(normalization.running_mean)
                norm_bias = torch.zeros_like(normalization.running_mean)
            scale = norm_weight * (
                normalization.running_var + normalization.eps
            ).rsqrt()
            source_bias = convolution.bias
            if source_bias is None:
                source_bias = torch.zeros_like(normalization.running_mean)
            convolution.weight = torch.nn.Parameter(
                convolution.weight * scale.reshape(-1, 1, 1, 1),
                requires_grad=False,
            )
            convolution.bias = torch.nn.Parameter(
                source_bias * scale
                + norm_bias
                - normalization.running_mean * scale,
                requires_grad=False,
            )
            module.norm = torch.nn.Identity()
            fused.append(
                {
                    "module": name,
                    "channels": int(convolution.out_channels),
                    "weight_shape": [
                        int(value) for value in convolution.weight.shape
                    ],
                }
            )
    return {"fused_count": len(fused), "modules": fused}


def _prepare_layout_weight_formats(
    model: torch.nn.Module,
    *,
    requested: str,
) -> dict[str, Any]:
    """Preformat layout weights while recording the exact NPU formats used."""
    if requested not in LAYOUT_WEIGHT_FORMAT_CHOICES:
        raise ValueError(f"Unsupported layout weight format: {requested}")
    import torch_npu

    tracked: list[tuple[str, str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            tracked.append((name, "linear", module))
        elif isinstance(module, torch.nn.Conv2d):
            kind = "depthwise_conv2d" if module.groups > 1 else "conv2d"
            tracked.append((name, kind, module))

    def histogram() -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for _name, kind, module in tracked:
            code = str(int(torch_npu.get_npu_format(module.weight)))
            bucket = result.setdefault(kind, {})
            bucket[code] = bucket.get(code, 0) + 1
        return {kind: dict(sorted(values.items())) for kind, values in sorted(result.items())}

    before = histogram()
    if requested == "torchair_internal":
        try:
            from torch_npu.dynamo.torchair import use_internal_format_weight
        except ImportError:
            from torchair import use_internal_format_weight
        use_internal_format_weight(model)

    return {
        "requested": requested,
        "before_histogram": before,
        "after_histogram": histogram(),
    }


def _layout_outputs_for_cpu_postprocess(outputs: Any) -> SimpleNamespace:
    """Copy only the tensors consumed by the variable-size postprocessor.

    Transformers postprocessing uses scatter plus boolean/tensor indexing to
    construct a variable-size result. Keep the fixed-shape detector forward on
    NPU, then execute this small selection/ranking tail on CPU so Atlas 310P
    does not dispatch those operations through its failing AICPU Index path.
    """
    return SimpleNamespace(
        logits=outputs.logits.detach().cpu(),
        pred_boxes=outputs.pred_boxes.detach().cpu(),
        order_logits=outputs.order_logits.detach().cpu(),
    )


def prepare_layout_resized_uint8_exact(
    images: list[np.ndarray],
    *,
    timing_s: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Resize fixed PP-DocLayoutV2 inputs without expanding them to float32.

    The checkpoint contract is a uint8 RGB input resized directly to 800x800
    with torchvision bicubic interpolation and ``antialias=False``.  The NPU
    performs the subsequent float32 divide and model-dtype cast after copying
    this tensor, reducing host-transfer bytes by 4x.  The initial CHW copy is
    required because the compiled graph consumes contiguous BCHW strides.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tv_functional

    prepared: list[torch.Tensor] = []
    for image in images:
        started = time.perf_counter()
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "PP-DocLayoutV2 preprocessing requires uint8 HWC RGB, got "
                f"dtype={image.dtype} shape={image.shape}"
            )
        _record_profile_elapsed(timing_s, "processor_validate_s", started)

        started = time.perf_counter()
        image_tensor = torch.from_numpy(image)
        _record_profile_elapsed(
            timing_s, "processor_numpy_to_tensor_s", started
        )

        started = time.perf_counter()
        channels_first = image_tensor.permute(2, 0, 1)
        _record_profile_elapsed(
            timing_s, "processor_hwc_to_chw_view_s", started
        )

        started = time.perf_counter()
        channels_first = channels_first.contiguous()
        _record_profile_elapsed(
            timing_s, "processor_chw_contiguous_s", started
        )

        started = time.perf_counter()
        channels_first = channels_first.unsqueeze(0)
        _record_profile_elapsed(
            timing_s, "processor_batch_view_s", started
        )

        started = time.perf_counter()
        resized = tv_functional.resize(
            channels_first,
            [800, 800],
            interpolation=InterpolationMode.BICUBIC,
            antialias=False,
        )
        _record_profile_elapsed(
            timing_s, "processor_bicubic_resize_s", started
        )
        prepared.append(resized)
    if not prepared:
        raise ValueError("PP-DocLayoutV2 preprocessing requires at least one image")
    # Production uses layout B1. Avoid copying the already contiguous 800x800
    # tensor through torch.cat for that common case.
    started = time.perf_counter()
    pixel_values = (
        prepared[0]
        if len(prepared) == 1
        else torch.cat(prepared, dim=0)
    )
    _record_profile_elapsed(
        timing_s, "processor_batch_assemble_s", started
    )
    return {"pixel_values": pixel_values}


def prepare_layout_pixel_values_exact(
    images: list[np.ndarray],
) -> dict[str, torch.Tensor]:
    """Produce the historical float32 input for parity checks and tools."""
    resized = prepare_layout_resized_uint8_exact(images)["pixel_values"]
    return {"pixel_values": resized.to(dtype=torch.float32).div_(255.0)}


def _record_profile_elapsed(
    timing_s: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    if timing_s is not None:
        timing_s[name] = timing_s.get(name, 0.0) + time.perf_counter() - started


def _layout_order_sequences_cumsum(
    order_logits: torch.Tensor,
    *,
    timing_s: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute the HF reading-order vote exactly without triangular copies."""
    started = time.perf_counter()
    order_scores = torch.sigmoid(order_logits)
    _record_profile_elapsed(timing_s, "box_order_sigmoid_s", started)
    batch_size, sequence_length, _ = order_scores.shape

    # For query j, the first term is sum(scores[i, j], i < j).
    started = time.perf_counter()
    column_prefix = order_scores.cumsum(dim=1)
    earlier_votes = torch.cat(
        [
            torch.zeros(
                (batch_size, 1),
                dtype=order_scores.dtype,
                device=order_scores.device,
            ),
            torch.diagonal(
                column_prefix[:, :-1, 1:],
                dim1=1,
                dim2=2,
            ),
        ],
        dim=1,
    )

    # The second term is sum(1 - scores[j, i], i > j).
    row_prefix = order_scores.cumsum(dim=2)
    through_diagonal = torch.diagonal(row_prefix, dim1=1, dim2=2)
    later_counts = torch.arange(
        sequence_length - 1,
        -1,
        -1,
        dtype=order_scores.dtype,
        device=order_scores.device,
    ).unsqueeze(0)
    later_votes = later_counts - (row_prefix[:, :, -1] - through_diagonal)
    _record_profile_elapsed(timing_s, "box_order_votes_s", started)

    started = time.perf_counter()
    order_pointers = torch.argsort(earlier_votes + later_votes, dim=1)
    order_sequences = torch.empty_like(order_pointers)
    ranks = torch.arange(
        sequence_length,
        device=order_pointers.device,
        dtype=order_pointers.dtype,
    ).expand(batch_size, -1)
    order_sequences.scatter_(1, order_pointers, ranks)
    _record_profile_elapsed(timing_s, "box_order_rank_s", started)
    return order_sequences


def post_process_layout_object_detection_exact(
    outputs: Any,
    *,
    threshold: float,
    target_sizes: list[tuple[int, int]] | torch.Tensor,
    timing_s: dict[str, float] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Match the HF PP-DocLayoutV2 box decoder with less order work."""
    boxes = outputs.pred_boxes
    logits = outputs.logits
    order_sequences = _layout_order_sequences_cumsum(
        outputs.order_logits,
        timing_s=timing_s,
    )

    started = time.perf_counter()
    box_centers, box_dimensions = torch.split(boxes, 2, dim=-1)
    boxes = torch.cat(
        [
            box_centers - 0.5 * box_dimensions,
            box_centers + 0.5 * box_dimensions,
        ],
        dim=-1,
    )

    if len(logits) != len(target_sizes):
        raise ValueError(
            "target size count must equal the layout logits batch size"
        )
    if isinstance(target_sizes, list):
        image_heights, image_widths = torch.as_tensor(target_sizes).unbind(1)
    else:
        image_heights, image_widths = target_sizes.unbind(1)
    scale_factor = torch.stack(
        [image_widths, image_heights, image_widths, image_heights],
        dim=1,
    ).to(boxes.device)
    boxes = boxes * scale_factor[:, None, :]
    _record_profile_elapsed(timing_s, "box_xyxy_scale_s", started)

    started = time.perf_counter()
    query_count = logits.shape[1]
    class_count = logits.shape[2]
    scores, flat_indices = torch.topk(
        torch.sigmoid(logits).flatten(1),
        query_count,
        dim=-1,
    )
    labels = flat_indices % class_count
    query_indices = flat_indices // class_count
    _record_profile_elapsed(timing_s, "box_class_topk_s", started)

    started = time.perf_counter()
    boxes = boxes.gather(
        dim=1,
        index=query_indices.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]),
    )
    order_sequences = order_sequences.gather(dim=1, index=query_indices)
    _record_profile_elapsed(timing_s, "box_gather_s", started)

    started = time.perf_counter()
    results: list[dict[str, torch.Tensor]] = []
    for score, label, box, order_sequence in zip(
        scores,
        labels,
        boxes,
        order_sequences,
    ):
        keep = score >= threshold
        selected_order = order_sequence[keep]
        selected_order, indices = torch.sort(selected_order)
        results.append(
            {
                "scores": score[keep][indices],
                "labels": label[keep][indices],
                "boxes": box[keep][indices],
                "order_seq": selected_order,
            }
        )
    _record_profile_elapsed(timing_s, "box_filter_sort_s", started)
    return results


def filter_overlap_boxes_vectorized(
    layout_result: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Match OpenOCR's overlap decisions with vectorized box geometry.

    OpenOCR calculates the same axis-aligned intersection and area values in
    a nested Python loop.  Precomputing those values keeps the original
    order-dependent drop loop and label exception intact while removing the
    expensive per-pair NumPy allocations and repeated area calculations.
    """
    boxes = [
        box.copy()
        for box in layout_result["boxes"]
        if box["label"] != "reference"
    ]
    if len(boxes) < 2:
        return {**layout_result, "boxes": boxes}

    coordinates = np.asarray(
        [box["coordinate"] for box in boxes],
        dtype=np.float64,
    )
    widths = coordinates[:, 2] - coordinates[:, 0]
    heights = coordinates[:, 3] - coordinates[:, 1]
    areas = np.abs(widths * heights)

    intersection_widths = np.maximum(
        0.0,
        np.minimum(coordinates[:, None, 2], coordinates[None, :, 2])
        - np.maximum(coordinates[:, None, 0], coordinates[None, :, 0]),
    )
    intersection_heights = np.maximum(
        0.0,
        np.minimum(coordinates[:, None, 3], coordinates[None, :, 3])
        - np.maximum(coordinates[:, None, 1], coordinates[None, :, 1]),
    )
    intersections = intersection_widths * intersection_heights
    smaller_areas = np.minimum(areas[:, None], areas[None, :])
    overlap_ratios = np.divide(
        intersections,
        smaller_areas,
        out=np.zeros_like(intersections),
        where=smaller_areas != 0.0,
    )

    dropped: set[int] = set()
    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            if first_index in dropped or second_index in dropped:
                continue
            if overlap_ratios[first_index, second_index] <= 0.7:
                continue
            second = boxes[second_index]
            if (
                first["label"] == "image" or second["label"] == "image"
            ) and first["label"] != second["label"]:
                continue
            dropped.add(
                second_index
                if areas[first_index] >= areas[second_index]
                else first_index
            )

    return {
        **layout_result,
        "boxes": [
            box for index, box in enumerate(boxes) if index not in dropped
        ],
    }


class PPDocLayoutV2NpuAdapter:
    """Match ``LayoutDetectorONNX`` while running Transformers on NPU."""

    LABEL_MAP = {
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "display_formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer_image",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header_image",
        14: "image",
        15: "inline_formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "vertical_text",
        24: "vision_footnote",
    }

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "npu:0",
        dtype: str = "float32",
        reading_order_dtype: str | None = None,
        threshold: float = 0.5,
        profile_stages: bool = False,
        execution: str = "eager",
        compile_cache_dir: str | Path | None = None,
        batch_size: int = 1,
        weight_format: str = DEFAULT_LAYOUT_WEIGHT_FORMAT,
        freeze_parameters: bool = False,
        depthwise_rewrite: str = DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
        fuse_frozen_bn: bool = False,
        fuse_eval_bn: bool = False,
        precompute_frozen_bn_affine: bool = False,
        preformat_frozen_bn_buffers: bool = False,
        cogview_attention_impl: str = DEFAULT_LAYOUT_COGVIEW_ATTENTION_IMPL,
        input_color_order: str = "bgr",
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported layout dtype: {dtype}")
        if reading_order_dtype is not None and reading_order_dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported reading-order dtype: {reading_order_dtype}"
            )
        if not str(device).startswith("npu"):
            raise ValueError("PPDocLayoutV2NpuAdapter requires an NPU device")
        if execution not in {"eager", "torchair"}:
            raise ValueError(f"Unsupported layout execution: {execution}")
        if execution == "torchair" and compile_cache_dir is None:
            raise ValueError("TorchAir layout execution requires compile_cache_dir")
        if batch_size < 1:
            raise ValueError("layout batch size must be >= 1")
        if weight_format not in LAYOUT_WEIGHT_FORMAT_CHOICES:
            raise ValueError(f"Unsupported layout weight format: {weight_format}")
        if depthwise_rewrite not in LAYOUT_DEPTHWISE_REWRITE_CHOICES:
            raise ValueError(
                f"Unsupported layout depthwise rewrite: {depthwise_rewrite}"
            )
        if cogview_attention_impl not in LAYOUT_COGVIEW_ATTENTION_IMPL_CHOICES:
            raise ValueError(
                "Unsupported layout CogView attention implementation: "
                f"{cogview_attention_impl}"
            )
        if input_color_order not in {"bgr", "rgb"}:
            raise ValueError(
                f"Unsupported layout input color order: {input_color_order}"
            )

        import torch_npu  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.reading_order_dtype = (
            self.dtype
            if reading_order_dtype is None
            else DTYPE_MAP[reading_order_dtype]
        )
        self.threshold = float(threshold)
        self.profile_stages = bool(profile_stages)
        self.execution = execution
        self.batch_size = int(batch_size)
        self.weight_format = str(weight_format)
        self.freeze_parameters = bool(freeze_parameters)
        self.depthwise_rewrite = str(depthwise_rewrite)
        self.fuse_frozen_bn = bool(fuse_frozen_bn)
        self.fuse_eval_bn = bool(fuse_eval_bn)
        self.precompute_frozen_bn_affine = bool(precompute_frozen_bn_affine)
        self.preformat_frozen_bn_buffers = bool(preformat_frozen_bn_buffers)
        self.cogview_attention_impl = str(cogview_attention_impl)
        self.input_color_order = str(input_color_order)
        frozen_bn_rewrite_count = sum(
            (
                self.fuse_frozen_bn,
                self.precompute_frozen_bn_affine,
                self.preformat_frozen_bn_buffers,
            )
        )
        if frozen_bn_rewrite_count > 1:
            raise ValueError(
                "FrozenBN fusion, affine precompute, and buffer preformat are exclusive"
            )
        self._filter_overlap_boxes = filter_overlap_boxes_vectorized

        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        # Transformers 5.5.4 uses forms rejected by eager Ascend 310P: a 0-D
        # torch.where branch in anchor generation, data-dependent indexed writes
        # in reading-order input construction, and broadcast advanced indexing
        # in its reading-order mask helper. Bind only their shape-explicit
        # equivalents here. TorchAir-only attention/linear rewrites remain
        # confined to LayoutFullGraphRuntime.
        from layout_torchair import make_eager_npu_compatible

        make_eager_npu_compatible(self.model)
        self.frozen_bn_fusion_summary = (
            _fuse_layout_frozen_batch_norms(self.model)
            if self.fuse_frozen_bn
            else {"fused_count": 0, "modules": []}
        )
        self.eval_bn_fusion_summary = (
            _fuse_layout_eval_batch_norms(self.model)
            if self.fuse_eval_bn
            else {"fused_count": 0, "modules": []}
        )
        defer_constant_grouped = self.depthwise_rewrite == "constant_grouped"
        self.depthwise_rewrite_summary = (
            {
                "requested": self.depthwise_rewrite,
                "target_count": 0,
                "rewritten_count": 0,
                "modules": [],
                "deferred_until_after_device_transfer": True,
            }
            if defer_constant_grouped
            else _rewrite_layout_depthwise_convs(
                self.model,
                requested=self.depthwise_rewrite,
            )
        )
        if (
            self.weight_format != "native"
            or defer_constant_grouped
            or self.precompute_frozen_bn_affine
            or self.preformat_frozen_bn_buffers
        ):
            # This gate must precede the process's first NPU allocation.
            torch.npu.config.allow_internal_format = True
        self.model.eval().to(device=self.device, dtype=self.dtype)
        if self.reading_order_dtype != self.dtype:
            self.model.reading_order.to(dtype=self.reading_order_dtype)
        if defer_constant_grouped:
            self.depthwise_rewrite_summary = _rewrite_layout_depthwise_convs(
                self.model,
                requested=self.depthwise_rewrite,
            )
        self.frozen_bn_affine_summary = (
            _precompute_layout_frozen_bn_affines(
                self.model,
                preformat_nc1hwc0=True,
            )
            if self.precompute_frozen_bn_affine
            else {"replaced_count": 0, "modules": []}
        )
        self.frozen_bn_buffer_format_summary = (
            _preformat_layout_frozen_bn_buffers(self.model)
            if self.preformat_frozen_bn_buffers
            else {"converted_count": 0, "modules": []}
        )
        self.weight_format_summary = _prepare_layout_weight_formats(
            self.model,
            requested=self.weight_format,
        )
        torch.npu.synchronize()
        self.compiled_runtime = None
        self.graph_warmup = None
        if execution == "torchair":
            from layout_torchair import LayoutFullGraphRuntime

            self.compiled_runtime = LayoutFullGraphRuntime(
                self.model,
                cache_root=Path(compile_cache_dir) / (
                    f"depthwise_{self.depthwise_rewrite}_"
                    f"weightformat_{self.weight_format}_"
                    f"readingorder_{str(self.reading_order_dtype).removeprefix('torch.')}_"
                    f"frozenbn{int(self.fuse_frozen_bn)}_"
                    f"evalbn{int(self.fuse_eval_bn)}_"
                    f"precomputedfrozenbn{int(self.precompute_frozen_bn_affine)}_"
                    f"formattedfrozenbnbuffers{int(self.preformat_frozen_bn_buffers)}_"
                    f"cogview_{self.cogview_attention_impl}"
                ),
                dtype=self.dtype,
                device=self.device,
                batch_size=self.batch_size,
                freeze_parameters=self.freeze_parameters,
                cogview_attention_impl=self.cogview_attention_impl,
            )
        self.setup_s = time.perf_counter() - started
        self.page_count = 0
        self.forward_s = 0.0
        self.postprocess_s = 0.0
        self.stage_s: dict[str, float] = defaultdict(float)

    def _record_stage(self, name: str, started: float) -> None:
        if self.profile_stages:
            self.stage_s[name] += time.perf_counter() - started

    def reset_timing(self) -> None:
        """Reset measured page work while retaining the loaded model."""
        self.page_count = 0
        self.forward_s = 0.0
        self.postprocess_s = 0.0
        self.stage_s.clear()

    def warmup_graph(
        self, image: np.ndarray, *, passes: int = 2
    ) -> dict[str, Any] | None:
        if self.compiled_runtime is None:
            return None
        pass_wall_s = []
        for index in range(passes):
            started = time.perf_counter()
            self._predict_batch([image] * self.batch_size, self.threshold)
            elapsed = time.perf_counter() - started
            pass_wall_s.append(elapsed)
            print(
                f"LAYOUT_GRAPH_WARMUP pass={index + 1}/{passes} "
                f"wall_s={elapsed:.3f}",
                flush=True,
            )
        self.reset_timing()
        self.graph_warmup = {
            "passes": passes,
            "pass_wall_s": pass_wall_s,
            "cache_dir": str(self.compiled_runtime.cache_dir),
            "dynamic": False,
            "fullgraph": True,
            "input": "first_benchmark_page",
            "batch_size": self.batch_size,
        }
        return self.graph_warmup

    @torch.inference_mode()
    def _predict_batch(
        self,
        images: list[np.ndarray],
        threshold: float,
    ) -> list[dict[str, Any]]:
        if not images:
            return []
        if len(images) > self.batch_size:
            raise ValueError(
                f"layout batch has {len(images)} pages, maximum is {self.batch_size}"
            )
        for image in images:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    "OpenDoc layout input must be an HxWx3 image, got "
                    f"shape={image.shape}"
                )
        real_page_count = len(images)
        padded_images = list(images)
        padded_images.extend([images[-1]] * (self.batch_size - real_page_count))
        target_sizes = [image.shape[:2] for image in padded_images]

        total_started = time.perf_counter()
        started = time.perf_counter()
        if self.input_color_order == "rgb":
            rgbs = padded_images
        else:
            rgbs = [
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for image in padded_images
            ]
        self._record_stage("input_to_rgb_s", started)

        started = time.perf_counter()
        inputs = prepare_layout_resized_uint8_exact(
            rgbs,
            timing_s=self.stage_s if self.profile_stages else None,
        )
        self._record_stage("processor_preprocess_s", started)

        started = time.perf_counter()
        device_pixels = inputs["pixel_values"].to(device=self.device)
        device_pixels = device_pixels.to(dtype=torch.float32).div_(255.0)
        if self.dtype != torch.float32:
            device_pixels = device_pixels.to(dtype=self.dtype)
        moved = {"pixel_values": device_pixels}
        torch.npu.synchronize()
        self._record_stage("inputs_h2d_s", started)

        started = time.perf_counter()
        if self.compiled_runtime is None:
            outputs = self.model(**moved)
        else:
            from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import (
                PPDocLayoutV2ForObjectDetectionOutput,
            )

            logits, pred_boxes, order_logits = self.compiled_runtime(
                moved["pixel_values"]
            )
            outputs = PPDocLayoutV2ForObjectDetectionOutput(
                logits=logits,
                pred_boxes=pred_boxes,
                order_logits=order_logits,
            )
        torch.npu.synchronize()
        forward_s = time.perf_counter() - started
        self.forward_s += forward_s
        if self.profile_stages:
            self.stage_s["model_forward_s"] += forward_s

        postprocess_started = time.perf_counter()
        started = postprocess_started
        cpu_outputs = _layout_outputs_for_cpu_postprocess(outputs)
        self._record_stage("outputs_d2h_s", started)

        started = time.perf_counter()
        prediction = post_process_layout_object_detection_exact(
            cpu_outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            timing_s=self.stage_s if self.profile_stages else None,
        )
        self._record_stage("box_decode_s", started)

        started = time.perf_counter()
        cpu_predictions = []
        for item in prediction[:real_page_count]:
            cpu_predictions.append(
                {
                    "scores": item["scores"].detach().cpu().tolist(),
                    "labels": item["labels"].detach().cpu().tolist(),
                    "boxes": item["boxes"].detach().cpu().tolist(),
                    "order_seq": item["order_seq"].detach().cpu().tolist(),
                }
            )
        self._record_stage("postprocess_tensor_to_list_s", started)

        started = time.perf_counter()
        results = []
        for image, item in zip(images, cpu_predictions):
            height, width = image.shape[:2]
            result_boxes: list[dict[str, Any]] = []
            for score, label_id, box, order in zip(
                item["scores"],
                item["labels"],
                item["boxes"],
                item["order_seq"],
            ):
                class_id = int(label_id)
                x1, y1, x2, y2 = box
                result_boxes.append(
                    {
                        "cls_id": class_id,
                        "label": self.LABEL_MAP.get(class_id, f"class_{class_id}"),
                        "score": float(score),
                        "coordinate": [
                            float(np.clip(x1, 0, width)),
                            float(np.clip(y1, 0, height)),
                            float(np.clip(x2, 0, width)),
                            float(np.clip(y2, 0, height)),
                        ],
                        "custom_value": float(order),
                    }
                )
            results.append({"boxes": result_boxes})
        self._record_stage("result_box_build_s", started)

        started = time.perf_counter()
        results = [self._filter_overlap_boxes(result) for result in results]
        self._record_stage("overlap_filter_s", started)

        started = time.perf_counter()
        for result in results:
            result["boxes"] = sorted(
                result["boxes"],
                key=lambda box: box["custom_value"],
            )
            for index, box in enumerate(result["boxes"], start=1):
                box["label"] = f"{box['label']}_{index:02d}"
        self._record_stage("order_and_label_s", started)
        self.postprocess_s += time.perf_counter() - postprocess_started
        self._record_stage("detector_total_s", total_started)
        self.page_count += real_page_count
        return results

    def _predict_one(
        self,
        image: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        return self._predict_batch([image], threshold)[0]

    def __call__(
        self,
        images: np.ndarray | list[np.ndarray],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(images, np.ndarray):
            images = [images]
        active_threshold = self.threshold if threshold is None else float(threshold)
        results = []
        for start in range(0, len(images), self.batch_size):
            results.extend(
                self._predict_batch(
                    images[start : start + self.batch_size],
                    active_threshold,
                )
            )
        return results

    def timing_summary(self) -> dict[str, Any]:
        stage_s = dict(self.stage_s)
        return {
            "setup_s": self.setup_s,
            "page_count": self.page_count,
            "forward_s": self.forward_s,
            "postprocess_s": self.postprocess_s,
            "execution": self.execution,
            "batch_size": self.batch_size,
            "graph_warmup": self.graph_warmup,
            "stage_s": stage_s,
            "stage_mean_ms": {
                name: seconds * 1000.0 / self.page_count
                for name, seconds in stage_s.items()
                if self.page_count
            },
        }
