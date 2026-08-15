"""Exact alternatives for UniRec focal depthwise convolutions."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES = (
    "native",
    "constant",
    "constant_grouped",
    "constant_grouped_all",
    "aligned_spatial",
)


_CONSTANT_WEIGHTS: dict[int, dict[str, Any]] = {}
_CONSTANT_CONVERTER_REGISTERED = False
_GROUPED_RUNTIME_FORMAT_PATCHED = False
_NEXT_CONSTANT_WEIGHT_ID = 0
_PREPACK_CONVERTER_REGISTERED = False
_EAGER_GROUPED_FZ_BRIDGE: Any | None = None


def grouped_fz_storage_shape(
    logical_shape: tuple[int, int, int, int] | list[int],
    *,
    groups: int,
) -> tuple[int, int, int, int]:
    """Return CANN's physical FRACTAL_Z:<groups> filter shape."""
    output_channels, input_channels_per_group, kernel_h, kernel_w = (
        int(value) for value in logical_shape
    )
    groups = int(groups)
    if groups < 1 or output_channels % groups:
        raise ValueError(
            "grouped FZ requires output channels divisible by groups, got "
            f"shape={tuple(logical_shape)} groups={groups}"
        )
    output_channels_per_group = output_channels // groups
    return (
        ((groups * input_channels_per_group + 15) // 16)
        * kernel_h
        * kernel_w,
        (output_channels_per_group + 15) // 16,
        16,
        16,
    )


def pack_grouped_fz_host(weight: np.ndarray, *, groups: int) -> np.ndarray:
    """Pack an NCHW grouped-convolution filter into CANN FRACTAL_Z:G."""
    if weight.ndim != 4:
        raise ValueError(f"grouped FZ requires a 4D filter, got {weight.shape}")
    output_channels, input_channels_per_group, kernel_h, kernel_w = (
        int(value) for value in weight.shape
    )
    groups = int(groups)
    storage_shape = grouped_fz_storage_shape(weight.shape, groups=groups)
    output_channels_per_group = output_channels // groups
    packed = np.zeros(storage_shape, dtype=weight.dtype)
    kernel_area = kernel_h * kernel_w
    for output_channel in range(output_channels):
        group = output_channel // output_channels_per_group
        output_in_group = output_channel % output_channels_per_group
        storage_output_c0 = output_channel % 16
        for input_channel in range(input_channels_per_group):
            grouped_input = group * input_channels_per_group + input_channel
            storage_row_base = (grouped_input // 16) * kernel_area
            storage_c0 = grouped_input % 16
            for kernel_h_index in range(kernel_h):
                for kernel_w_index in range(kernel_w):
                    kernel_index = kernel_h_index * kernel_w + kernel_w_index
                    packed[
                        storage_row_base + kernel_index,
                        output_in_group // 16,
                        storage_output_c0,
                        storage_c0,
                    ] = weight[
                        output_channel,
                        input_channel,
                        kernel_h_index,
                        kernel_w_index,
                    ]
    return packed


def _load_eager_grouped_fz_bridge() -> tuple[Any, float]:
    """Build or load the tiny TorchNPU storage-descriptor bridge."""
    global _EAGER_GROUPED_FZ_BRIDGE
    if _EAGER_GROUPED_FZ_BRIDGE is not None:
        return _EAGER_GROUPED_FZ_BRIDGE, 0.0

    import torch_npu
    from torch.utils.cpp_extension import load

    if shutil.which("ninja") is None:
        interpreter_ninja = Path(sys.executable).resolve().parent / "ninja"
        if interpreter_ninja.is_file() and os.access(interpreter_ninja, os.X_OK):
            os.environ["PATH"] = (
                f"{interpreter_ninja.parent}{os.pathsep}"
                f"{os.environ.get('PATH', '')}"
            )
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ninja is required to build the grouped-FZ descriptor bridge; "
            "put an existing ninja executable on PATH"
        )

    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    library_dir = torch_npu_root / "lib"
    started = time.perf_counter()
    _EAGER_GROUPED_FZ_BRIDGE = load(
        name="unirec_grouped_fz_descriptor_bridge_v3",
        sources=[str(Path(__file__).with_name("grouped_fz_descriptor_bridge.cpp"))],
        extra_include_paths=[str(torch_npu_root / "include")],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{library_dir}",
            "-ltorch_npu",
            f"-Wl,-rpath,{library_dir}",
        ],
        verbose=True,
    )
    return _EAGER_GROUPED_FZ_BRIDGE, time.perf_counter() - started


def rewrite_eager_stage23_5x5_7x7_grouped_fz(
    vision_encoder: nn.Module,
) -> dict[str, Any]:
    """Prepack all 22 stage-2/3 5x5 and 7x7 focal weights for eager NPU."""
    bridge, extension_build_s = _load_eager_grouped_fz_bridge()
    rows: list[dict[str, Any]] = []
    for stage_index in (2, 3):
        stage = vision_encoder.layers[stage_index]
        expected_channels = 384 if stage_index == 2 else 768
        for block_index, block in enumerate(stage.blocks):
            for focal_index, focal_layer in enumerate(
                block.modulation.focal_layers
            ):
                convolution = focal_layer[0]
                if not isinstance(convolution, nn.Conv2d):
                    raise TypeError(
                        "eager grouped-FZ rewrite expected native Conv2d at "
                        f"stage={stage_index} block={block_index} "
                        f"focal={focal_index}"
                    )
                kernel = tuple(int(value) for value in convolution.kernel_size)
                if kernel not in {(5, 5), (7, 7)}:
                    continue
                channels = int(convolution.in_channels)
                if not (
                    channels == expected_channels
                    and convolution.out_channels == channels
                    and convolution.groups == channels
                    and tuple(convolution.stride) == (1, 1)
                    and tuple(convolution.dilation) == (1, 1)
                    and convolution.bias is None
                    and convolution.weight.dtype == torch.float16
                    and convolution.weight.device.type == "npu"
                ):
                    raise ValueError(
                        "unexpected stage-2/3 focal convolution contract: "
                        f"{convolution} weight={convolution.weight.shape}/"
                        f"{convolution.weight.dtype}/{convolution.weight.device}"
                    )

                logical_shape = tuple(
                    int(value) for value in convolution.weight.shape
                )
                host_weight = np.ascontiguousarray(
                    convolution.weight.detach().to(device="cpu").numpy()
                )
                packed_host = pack_grouped_fz_host(
                    host_weight, groups=channels
                )
                packed_storage = torch.from_numpy(packed_host).to(
                    device=convolution.weight.device
                )
                wrapped = bridge.wrap_grouped_fz(
                    packed_storage,
                    list(logical_shape),
                    channels,
                )
                parameter = nn.Parameter(wrapped, requires_grad=False)
                origin_format, storage_format, base_shape, storage_shape = (
                    bridge.describe_npu_storage(parameter)
                )
                descriptor = {
                    "origin_format": int(origin_format),
                    "storage_format": int(storage_format),
                    "base_shape": [int(value) for value in base_shape],
                    "storage_shape": [int(value) for value in storage_shape],
                    "physical_bytes": int(packed_host.nbytes),
                }
                expected_descriptor = {
                    "origin_format": 0,
                    "storage_format": 4,
                    "base_shape": list(logical_shape),
                    "storage_shape": list(packed_host.shape),
                    "physical_bytes": int(packed_host.nbytes),
                }
                if descriptor != expected_descriptor:
                    raise RuntimeError(
                        "grouped-FZ descriptor changed while binding Parameter: "
                        f"actual={descriptor} expected={expected_descriptor}"
                    )
                convolution.weight = parameter
                rows.append(
                    {
                        "module": (
                            f"layers.{stage_index}.blocks.{block_index}."
                            f"modulation.focal_layers.{focal_index}.0"
                        ),
                        "stage": stage_index,
                        "groups": channels,
                        "kernel": list(kernel),
                        "descriptor": descriptor,
                    }
                )

    signature_counts: dict[str, int] = {}
    for row in rows:
        descriptor = row["descriptor"]
        key = (
            f"{descriptor['base_shape']}->{descriptor['storage_shape']}"
        )
        signature_counts[key] = signature_counts.get(key, 0) + 1
    expected_signature_counts = {
        "[384, 1, 5, 5]->[600, 1, 16, 16]": 9,
        "[384, 1, 7, 7]->[1176, 1, 16, 16]": 9,
        "[768, 1, 5, 5]->[1200, 1, 16, 16]": 2,
        "[768, 1, 7, 7]->[2352, 1, 16, 16]": 2,
    }
    if signature_counts != expected_signature_counts:
        raise RuntimeError(
            "unexpected stage-2/3 grouped-FZ rewrite inventory: "
            f"actual={signature_counts} expected={expected_signature_counts}"
        )
    return {
        "requested": "eager_stage23_5x5_7x7_grouped_fz",
        "rewritten_count": len(rows),
        "signature_counts": signature_counts,
        "extension_build_s": extension_build_s,
        "physical_bytes": sum(
            int(row["descriptor"]["physical_bytes"]) for row in rows
        ),
        "modules": rows,
    }


@torch.library.custom_op("unirec::focal_group_prepack_v1", mutates_args=())
def _focal_group_prepack(
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    del groups
    return weight.clone()


@_focal_group_prepack.register_fake
def _focal_group_prepack_fake(
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    del groups
    return torch.empty_like(weight)


@torch.library.custom_op("unirec::focal_depthwise_const_v1", mutates_args=())
def _focal_depthwise_const(
    inputs: torch.Tensor,
    weight_id: int,
) -> torch.Tensor:
    row = _CONSTANT_WEIGHTS[int(weight_id)]
    return F.conv2d(
        inputs,
        row["device_weight"],
        bias=None,
        stride=row["stride"],
        padding=row["padding"],
        dilation=row["dilation"],
        groups=int(row["groups"]),
    )


@_focal_depthwise_const.register_fake
def _focal_depthwise_const_fake(
    inputs: torch.Tensor,
    weight_id: int,
) -> torch.Tensor:
    row = _CONSTANT_WEIGHTS[int(weight_id)]
    return inputs.new_empty(_constant_depthwise_output_shape(inputs, row))


@torch.library.custom_op(
    "unirec::focal_depthwise_prepacked_v1",
    mutates_args=(),
)
def _focal_depthwise_prepacked(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_id: int,
) -> torch.Tensor:
    del packed_weight
    row = _CONSTANT_WEIGHTS[int(weight_id)]
    return F.conv2d(
        inputs,
        row["device_weight"],
        bias=None,
        stride=row["stride"],
        padding=row["padding"],
        dilation=row["dilation"],
        groups=int(row["groups"]),
    )


@_focal_depthwise_prepacked.register_fake
def _focal_depthwise_prepacked_fake(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_id: int,
) -> torch.Tensor:
    del packed_weight
    row = _CONSTANT_WEIGHTS[int(weight_id)]
    return inputs.new_empty(_constant_depthwise_output_shape(inputs, row))


def _constant_depthwise_output_shape(
    inputs: torch.Tensor,
    row: dict[str, Any],
) -> tuple[Any, int, Any, Any]:
    input_h, input_w = inputs.shape[-2], inputs.shape[-1]
    kernel_h, kernel_w = (int(value) for value in row["kernel"])
    stride_h, stride_w = (int(value) for value in row["stride"])
    padding_h, padding_w = (int(value) for value in row["padding"])
    dilation_h, dilation_w = (int(value) for value in row["dilation"])
    output_h = (
        input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    return (
        inputs.shape[0],
        int(row["groups"]),
        output_h,
        output_w,
    )


def _import_torchair() -> Any:
    try:
        return importlib.import_module("torch_npu.dynamo.torchair")
    except ImportError:
        return importlib.import_module("torchair")


def _patch_grouped_runtime_input_formats(converter_module: Any) -> None:
    """Preserve marked grouped-FZ buffer descriptors after runtime binding."""
    global _GROUPED_RUNTIME_FORMAT_PATCHED
    if _GROUPED_RUNTIME_FORMAT_PATCHED:
        return
    modules = [converter_module]
    generated_module = importlib.import_module(
        "torchair._ge_concrete_graph.fx2ge_converter"
    )
    if generated_module is not converter_module:
        modules.append(generated_module)
    for module in modules:
        original = module._update_internal_format_from_inputs

        def _update_internal_format_from_inputs(
            graph: Any,
            runtime_inputs: Any,
            *,
            _original: Any = original,
        ) -> None:
            _original(graph, runtime_inputs)
            producers = {op.name: op for op in graph.op}
            for convolution in graph.op:
                marker = convolution.attr.get("_unirec_grouped_fz_format")
                if marker is None:
                    continue
                storage_format = int(marker.i)
                logical_shape = list(
                    convolution.attr[
                        "_unirec_grouped_fz_logical_shape"
                    ].list.i
                )
                producer_name = convolution.input[1].split(":", 1)[0]
                producer = producers[producer_name]
                producer.attr["_enable_storage_format_spread"].b = False
                if producer.type == "ConstPlaceHolder":
                    producer.attr["origin_shape"].list.i[:] = logical_shape
                    producer.attr["origin_format"].i = 0
                    producer.attr["storage_format"].i = storage_format
                for descriptor in producer.output_desc:
                    descriptor.layout = "FRACTAL_Z"
                    descriptor.shape.dim[:] = []
                    descriptor.attr["format_for_int"].i = storage_format
                    descriptor.attr["origin_format_for_int"].i = 0
                    descriptor.attr["origin_shape"].list.val_type = 2
                    descriptor.attr["origin_shape"].list.i[:] = logical_shape
                    descriptor.attr["origin_shape_initialized"].b = True
                    descriptor.attr["origin_format_is_set"].b = True
                for descriptor in producer.input_desc:
                    descriptor.CopyFrom(producer.output_desc[0])

        module._update_internal_format_from_inputs = (
            _update_internal_format_from_inputs
        )
    _GROUPED_RUNTIME_FORMAT_PATCHED = True


def register_focal_depthwise_constant_converter() -> None:
    """Lower a native focal convolution with its immutable weight as GE Const."""
    global _CONSTANT_CONVERTER_REGISTERED
    if _CONSTANT_CONVERTER_REGISTERED:
        return
    torchair = _import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    converter_utils = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_converter.converter_utils"
    )
    ge = converter_utils.ge
    register_converter = converter_module.register_fx_node_ge_converter
    specific_input_layout = converter_utils.specific_op_input_layout
    specific_output_layout = converter_utils.specific_op_output_layout
    _patch_grouped_runtime_input_formats(converter_module)

    @register_converter(torch.ops.unirec.focal_depthwise_const_v1.default)
    def _convert_focal_depthwise_const(
        inputs: Any,
        weight_id: int,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        row = _CONSTANT_WEIGHTS[int(weight_id)]
        weight = ge.Const(row["host_weight"])
        if row["prepacked_grouped"]:
            logical_shape = row["shape"]
            storage_format = (int(row["groups"]) << 8) | 4
            weight.desc.layout = "FRACTAL_Z"
            weight.desc.shape.dim[:] = []
            weight.desc.attr["format_for_int"].i = storage_format
            weight.desc.attr["origin_format_for_int"].i = 0
            weight.desc.attr["origin_shape"].list.val_type = 2
            weight.desc.attr["origin_shape"].list.i.extend(logical_shape)
            weight.desc.attr["origin_shape_initialized"].b = True
            weight.desc.attr["origin_format_is_set"].b = True
            specific_output_layout(weight, indices=0, layout="FRACTAL_Z")
            weight.node.attr["_enable_storage_format_spread"].b = False
        stride_h, stride_w = row["stride"]
        padding_h, padding_w = row["padding"]
        dilation_h, dilation_w = row["dilation"]
        output = ge.Conv2D(
            inputs,
            weight,
            None,
            None,
            strides=[1, 1, int(stride_h), int(stride_w)],
            pads=[
                int(padding_h),
                int(padding_h),
                int(padding_w),
                int(padding_w),
            ],
            dilations=[1, 1, int(dilation_h), int(dilation_w)],
            groups=int(row["groups"]),
            data_format="NCHW",
        )
        specific_input_layout(output, indices=[0, 1], layout="NCHW")
        specific_output_layout(output, indices=0, layout="NCHW")
        if row["prepacked_grouped"]:
            logical_shape = row["shape"]
            storage_format = (int(row["groups"]) << 8) | 4
            filter_desc = output.node.input_desc[1]
            filter_desc.shape.dim[:] = []
            filter_desc.attr["format_for_int"].i = storage_format
            filter_desc.attr["origin_format_for_int"].i = 0
            filter_desc.attr["origin_shape"].list.val_type = 2
            filter_desc.attr["origin_shape"].list.i.extend(logical_shape)
            filter_desc.attr["origin_shape_initialized"].b = True
            filter_desc.attr["origin_format_is_set"].b = True
        return output

    @register_converter(torch.ops.unirec.focal_depthwise_prepacked_v1.default)
    def _convert_focal_depthwise_prepacked(
        inputs: Any,
        packed_weight: Any,
        weight_id: int,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        row = _CONSTANT_WEIGHTS[int(weight_id)]
        stride_h, stride_w = row["stride"]
        padding_h, padding_w = row["padding"]
        dilation_h, dilation_w = row["dilation"]
        output = ge.Conv2D(
            inputs,
            packed_weight,
            None,
            None,
            strides=[1, 1, int(stride_h), int(stride_w)],
            pads=[
                int(padding_h),
                int(padding_h),
                int(padding_w),
                int(padding_w),
            ],
            dilations=[1, 1, int(dilation_h), int(dilation_w)],
            groups=int(row["groups"]),
            data_format="NCHW",
        )
        specific_input_layout(output, indices=[0, 1], layout="NCHW")
        specific_output_layout(output, indices=0, layout="NCHW")
        logical_shape = row["shape"]
        storage_format = (int(row["groups"]) << 8) | 4
        output.node.attr["_unirec_grouped_fz_format"].i = storage_format
        output.node.attr[
            "_unirec_grouped_fz_logical_shape"
        ].list.val_type = 2
        output.node.attr[
            "_unirec_grouped_fz_logical_shape"
        ].list.i.extend(logical_shape)
        filter_desc = output.node.input_desc[1]
        filter_desc.attr["format_for_int"].i = storage_format
        filter_desc.attr["origin_format_for_int"].i = 0
        filter_desc.attr["origin_shape"].list.val_type = 2
        filter_desc.attr["origin_shape"].list.i.extend(logical_shape)
        filter_desc.attr["origin_shape_initialized"].b = True
        filter_desc.attr["origin_format_is_set"].b = True
        return output

    _CONSTANT_CONVERTER_REGISTERED = True


def register_focal_group_prepack_converter() -> None:
    """Lower a setup-only call to the grouped Conv2D weight representation."""
    global _PREPACK_CONVERTER_REGISTERED
    if _PREPACK_CONVERTER_REGISTERED:
        return
    torchair = _import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    converter_utils = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_converter.converter_utils"
    )
    register_converter = converter_module.register_fx_node_ge_converter
    ge = converter_utils.ge
    specific_input_layout = converter_utils.specific_op_input_layout
    specific_output_layout = converter_utils.specific_op_output_layout

    @register_converter(torch.ops.unirec.focal_group_prepack_v1.default)
    def _convert_focal_group_prepack(
        weight: Any,
        groups: int,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        logical_shape = tuple(int(value) for value in weight.symsize)
        packed = ge.TransData(
            weight,
            src_format="NCHW",
            dst_format="FRACTAL_Z",
            src_subformat=0,
            dst_subformat=int(groups),
            groups=int(groups),
        )
        specific_input_layout(packed, indices=0, layout="NCHW")
        specific_output_layout(packed, indices=0, layout="FRACTAL_Z")
        packed.desc.shape.dim[:] = logical_shape
        packed.desc.attr["format_for_int"].i = (int(groups) << 8) | 4
        return packed

    _PREPACK_CONVERTER_REGISTERED = True


def focal_group_prepack(
    weight: torch.Tensor,
    *,
    groups: int,
) -> torch.Tensor:
    return _focal_group_prepack(weight, int(groups))


class ConstantDepthwiseConv2d(nn.Module):
    """Depthwise Conv2d backed by an immutable optional grouped-FZ weight."""

    def __init__(
        self,
        source: nn.Conv2d,
        *,
        weight_id: int | None = None,
        prepack_grouped: bool = False,
        host_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        global _NEXT_CONSTANT_WEIGHT_ID
        kernel = tuple(int(value) for value in source.kernel_size)
        channels = int(source.in_channels)
        if not (
            source.out_channels == channels
            and source.groups == channels
            and tuple(source.weight.shape) == (channels, 1, *kernel)
            and source.bias is None
            and source.padding_mode == "zeros"
        ):
            raise ValueError(
                "constant grouped rewrite requires a bias-free native "
                "depthwise convolution with zero padding"
            )
        if weight_id is None:
            weight_id = _NEXT_CONSTANT_WEIGHT_ID
            _NEXT_CONSTANT_WEIGHT_ID += 1
        weight = source.weight.detach().contiguous()
        host_weight = np.ascontiguousarray(
            weight.to(device="cpu", dtype=host_dtype).numpy()
        )
        if prepack_grouped:
            host_weight = pack_grouped_fz_host(host_weight, groups=channels)
        _CONSTANT_WEIGHTS[int(weight_id)] = {
            "device_weight": weight,
            "host_weight": host_weight,
            "groups": channels,
            "kernel": kernel,
            "stride": tuple(int(value) for value in source.stride),
            "padding": tuple(int(value) for value in source.padding),
            "dilation": tuple(int(value) for value in source.dilation),
            "shape": tuple(int(value) for value in weight.shape),
            "storage_shape": tuple(int(value) for value in host_weight.shape),
            "prepacked_grouped": bool(prepack_grouped),
        }
        self.prepack_grouped = bool(prepack_grouped)
        if self.prepack_grouped:
            self.packed_weight = nn.Parameter(
                torch.from_numpy(host_weight).to(weight.device),
                requires_grad=False,
            )
        self.weight_id = int(weight_id)
        self.groups = channels
        self.kernel_size = kernel
        self.stride = tuple(int(value) for value in source.stride)
        self.padding = tuple(int(value) for value in source.padding)
        self.dilation = tuple(int(value) for value in source.dilation)

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        row = _CONSTANT_WEIGHTS[self.weight_id]
        row["device_weight"] = fn(row["device_weight"])
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.prepack_grouped:
            return _focal_depthwise_prepacked(
                inputs,
                self.packed_weight,
                self.weight_id,
            )
        return _focal_depthwise_const(inputs, self.weight_id)


class ConstantFocalDepthwiseConv(ConstantDepthwiseConv2d):
    """Focal depthwise Conv2d whose immutable weight is embedded in the OM."""

    def __init__(
        self,
        source: nn.Conv2d,
        *,
        weight_id: int,
        prepack_grouped: bool = False,
    ) -> None:
        kernel = tuple(int(value) for value in source.kernel_size)
        if not (
            kernel in {(3, 3), (5, 5), (7, 7)}
            and tuple(source.stride) == (1, 1)
            and tuple(source.dilation) == (1, 1)
        ):
            raise ValueError(
                "constant focal rewrite requires a 3x3/5x5/7x7 "
                "stride-1 dilation-1 convolution"
            )
        super().__init__(
            source,
            weight_id=weight_id,
            prepack_grouped=prepack_grouped,
            host_dtype=torch.float16,
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
    """Rewrite selected focal depthwise convolutions exactly."""
    global _NEXT_CONSTANT_WEIGHT_ID
    if requested not in VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES:
        raise ValueError(f"unsupported vision focal rewrite: {requested}")
    all_focal = requested == "constant_grouped_all"
    targets: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(vision_encoder.layers):
        if not all_focal and stage_index < 2:
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
                target_kernels = (
                    {(3, 3), (5, 5), (7, 7)}
                    if all_focal
                    else {(5, 5), (7, 7)}
                )
                if kernel not in target_kernels:
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
                if requested in {
                    "constant",
                    "constant_grouped",
                    "constant_grouped_all",
                }:
                    weight_id = _NEXT_CONSTANT_WEIGHT_ID
                    _NEXT_CONSTANT_WEIGHT_ID += 1
                    constant = ConstantFocalDepthwiseConv(
                        convolution,
                        weight_id=weight_id,
                        prepack_grouped=requested
                        in {"constant_grouped", "constant_grouped_all"},
                    )
                    focal_layer[0] = constant
                    row.update(
                        {
                            "target_kernel": list(kernel),
                            "target_groups": channels,
                            "group_width": 1,
                            "weight_id": weight_id,
                            "weight_shape": list(
                                _CONSTANT_WEIGHTS[weight_id]["shape"]
                            ),
                            "weight_storage_shape": list(
                                _CONSTANT_WEIGHTS[weight_id]["storage_shape"]
                            ),
                            "weight_binding": (
                                "frozen_prepacked_fractal_z_grouped"
                                if requested
                                in {"constant_grouped", "constant_grouped_all"}
                                else "ge_const_not_runtime_input"
                            ),
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
    if requested in {"constant", "constant_grouped", "constant_grouped_all"}:
        register_focal_depthwise_constant_converter()
    constant_digest = ""
    if requested in {"constant", "constant_grouped", "constant_grouped_all"}:
        digest = hashlib.sha256()
        for row in targets:
            digest.update(
                _CONSTANT_WEIGHTS[int(row["weight_id"])]["host_weight"].tobytes()
            )
        constant_digest = digest.hexdigest()[:16]
    return {
        "requested": requested,
        "target_count": len(targets),
        "rewritten_count": 0 if requested == "native" else len(targets),
        "constant_weight_digest": constant_digest,
        "modules": targets,
    }


def vision_rewrite_source_hash(
    requested: str,
    *,
    constant_weight_digest: str = "",
) -> str:
    if requested == "native":
        return ""
    payload = Path(__file__).read_bytes() + constant_weight_digest.encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:12]
