"""Exact alternatives for UniRec stage-2/3 focal depthwise convolutions."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES = (
    "native",
    "group16",
    "aligned_spatial",
)


class AlignedSpatialDepthwiseConv(nn.Module):
    """Zero-pad a focal depthwise filter to a 16-aligned spatial area."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        kernel = tuple(int(value) for value in source.kernel_size)
        if kernel == (5, 5):
            target_kernel = (6, 8)
            weight_offset = (0, 1)
            input_padding = (3, 4, 2, 3)
        elif kernel == (7, 7):
            target_kernel = (8, 8)
            weight_offset = (0, 0)
            input_padding = (3, 4, 3, 4)
        else:
            raise ValueError(f"unsupported aligned focal kernel: {kernel}")
        channels = int(source.in_channels)
        if not (
            source.out_channels == channels
            and source.groups == channels
            and tuple(source.stride) == (1, 1)
            and tuple(source.dilation) == (1, 1)
            and source.bias is None
        ):
            raise ValueError(
                "aligned focal rewrite requires a bias-free depthwise conv"
            )
        expanded = source.weight.detach().new_zeros(
            (channels, 1, target_kernel[0], target_kernel[1])
        )
        top, left = weight_offset
        expanded[
            :,
            :,
            top : top + kernel[0],
            left : left + kernel[1],
        ] = source.weight.detach()
        self.weight = nn.Parameter(expanded, requires_grad=False)
        self.groups = channels
        self.input_padding = input_padding
        self.source_kernel = kernel
        self.target_kernel = target_kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            F.pad(inputs, self.input_padding),
            self.weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.groups,
        )


def rewrite_vision_focal_depthwise_convs(
    vision_encoder: nn.Module,
    *,
    requested: str,
) -> dict[str, Any]:
    """Rewrite the dominant stage-2/3 5x5 and 7x7 focal convolutions exactly."""
    if requested not in VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES:
        raise ValueError(f"unsupported vision focal rewrite: {requested}")
    targets: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(vision_encoder.layers):
        if stage_index < 2:
            continue
        for block_index, block in enumerate(stage.blocks):
            for focal_index, focal_layer in enumerate(
                block.modulation.focal_layers
            ):
                convolution = focal_layer[0]
                if not isinstance(convolution, nn.Conv2d):
                    raise TypeError(
                        "focal rewrite expected Conv2d at "
                        f"stage={stage_index} block={block_index} "
                        f"focal={focal_index}"
                    )
                kernel = tuple(int(value) for value in convolution.kernel_size)
                if kernel not in {(5, 5), (7, 7)}:
                    continue
                channels = int(convolution.in_channels)
                row: dict[str, Any] = {
                    "module": (
                        f"layers.{stage_index}.blocks.{block_index}."
                        f"modulation.focal_layers.{focal_index}.0"
                    ),
                    "stage": stage_index,
                    "channels": channels,
                    "source_kernel": list(kernel),
                    "source_groups": int(convolution.groups),
                }
                if requested == "group16":
                    group_width = 16
                    if channels % group_width:
                        raise ValueError(
                            f"group16 does not divide focal channels: {channels}"
                        )
                    original = convolution.weight.detach()
                    expanded = original.new_zeros(
                        (channels, group_width, kernel[0], kernel[1])
                    )
                    indices = torch.arange(channels, device=original.device)
                    expanded[
                        indices,
                        indices.remainder(group_width),
                    ] = original[:, 0]
                    convolution.weight = nn.Parameter(
                        expanded,
                        requires_grad=False,
                    )
                    convolution.groups = channels // group_width
                    row.update(
                        {
                            "target_kernel": list(kernel),
                            "target_groups": int(convolution.groups),
                            "group_width": group_width,
                            "weight_shape": list(convolution.weight.shape),
                        }
                    )
                elif requested == "aligned_spatial":
                    aligned = AlignedSpatialDepthwiseConv(convolution)
                    focal_layer[0] = aligned
                    row.update(
                        {
                            "target_kernel": list(aligned.target_kernel),
                            "target_groups": int(aligned.groups),
                            "group_width": 1,
                            "input_padding": list(aligned.input_padding),
                            "weight_shape": list(aligned.weight.shape),
                        }
                    )
                else:
                    row.update(
                        {
                            "target_kernel": list(kernel),
                            "target_groups": int(convolution.groups),
                            "group_width": 1,
                            "weight_shape": list(convolution.weight.shape),
                        }
                    )
                targets.append(row)
    return {
        "requested": requested,
        "target_count": len(targets),
        "rewritten_count": 0 if requested == "native" else len(targets),
        "modules": targets,
    }


def vision_rewrite_source_hash(requested: str) -> str:
    if requested == "native":
        return ""
    payload = inspect.getsource(rewrite_vision_focal_depthwise_convs).encode(
        "utf-8"
    )
    payload += inspect.getsource(AlignedSpatialDepthwiseConv).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
