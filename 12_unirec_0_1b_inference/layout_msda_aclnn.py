"""Native ACLNN multi-scale deformable attention for PP-DocLayoutV2."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


GE_OP_NAME = "MultiScaleDeformableAttnFunction"
PYTORCH_OP_NAME = "unirec_layout::msda_aclnn"
EXPECTED_MODULE_COUNT = 6

_LOADED_EXTENSION: Path | None = None
_CONVERTER_REGISTERED = False
_CONVERTER_DEVICE_NAME: str | None = None
_CONVERTER_USES_310P_INTERNAL_LAYOUT = False
_HOST_INFER_REFRESH: Any | None = None
_HOST_INFER_REFRESHED = False


def _uses_310p_internal_layout(device_name: str) -> bool:
    return "310P" in device_name.upper()


def load_layout_msda_extension(path: str | Path) -> Path:
    """Load the small host binding around the installed ACLNN operator."""
    global _LOADED_EXTENSION
    global _HOST_INFER_REFRESH
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if _LOADED_EXTENSION is not None:
        if _LOADED_EXTENSION != resolved:
            raise RuntimeError(
                "layout MSDA extension already loaded from "
                f"{_LOADED_EXTENSION}, cannot replace it with {resolved}"
            )
        return resolved
    torch.ops.load_library(str(resolved))
    if not hasattr(torch.ops.unirec_layout, "msda_aclnn"):
        raise RuntimeError("layout MSDA extension did not register its torch op")
    host_library = ctypes.CDLL(str(resolved))
    try:
        refresh = host_library.unirec_layout_msda_refresh_host_infer
    except AttributeError as exc:
        raise RuntimeError(
            "layout MSDA extension lacks the host-infer refresh entrypoint"
        ) from exc
    refresh.argtypes = []
    refresh.restype = ctypes.c_int
    _HOST_INFER_REFRESH = refresh
    _LOADED_EXTENSION = resolved
    return resolved


def _refresh_layout_msda_host_infer() -> None:
    """Register the corrected callback after CANN loads its stock callback."""
    global _HOST_INFER_REFRESHED
    if _HOST_INFER_REFRESHED:
        return
    if _HOST_INFER_REFRESH is None:
        raise RuntimeError("layout MSDA extension is not loaded")
    status = int(_HOST_INFER_REFRESH())
    if status != 0:
        raise RuntimeError(f"layout MSDA host-infer refresh failed: {status}")
    _HOST_INFER_REFRESHED = True


def _import_torchair() -> Any:
    try:
        return importlib.import_module("torch_npu.dynamo.torchair")
    except ImportError:
        return importlib.import_module("torchair")


def register_layout_msda_converter() -> None:
    """Lower the custom torch identity to the installed GE MSDA operator."""
    global _CONVERTER_DEVICE_NAME
    global _CONVERTER_REGISTERED
    global _CONVERTER_USES_310P_INTERNAL_LAYOUT
    if _CONVERTER_REGISTERED:
        return
    device_name = torch.npu.get_device_name()
    use_310p_internal_layout = _uses_310p_internal_layout(device_name)
    force_host_infer_probe = (
        os.environ.get("UNIREC_LAYOUT_MSDA_FORCE_HOST_INFER_PROBE") == "1"
    )
    torchair = _import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    converter_utils = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_converter.converter_utils"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    ge_apis_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_apis"
    )
    ge_graph_module = importlib.import_module(
        f"{torchair.__name__}.ge._ge_graph"
    )
    ge_data_type = ge_graph_module.DataType
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op
    specific_input_layout = converter_utils.specific_op_input_layout
    specific_output_layout = converter_utils.specific_op_output_layout

    @register_converter(torch.ops.unirec_layout.msda_aclnn.default)
    def _convert_layout_msda(
        value: Any,
        value_spatial_shapes: Any,
        value_level_start_index: Any,
        sampling_locations: Any,
        attention_weights: Any,
        meta_outputs: Any = None,
    ) -> Any:
        if meta_outputs is None:
            raise RuntimeError("layout MSDA converter requires output metadata")
        value_spatial_shapes = ge_apis_module.Cast(
            value_spatial_shapes,
            dst_type=ge_data_type.DT_INT32,
        )
        value_level_start_index = ge_apis_module.Cast(
            value_level_start_index,
            dst_type=ge_data_type.DT_INT32,
        )

        # The public ACLNN API accepts the logical PyTorch shapes on every SoC,
        # but its 310P implementation transposes and promotes the three floating
        # inputs before it calls this lower-level GE operator. A direct custom
        # GE node bypasses that wrapper, so reproduce its exact 310P contract:
        #   value   [B,K,H,D]     -> [B,H,K,D]
        #   loc     [B,Q,H,L,P,2] -> [L,B,H,Q,P,2]
        #   weights [B,Q,H,L,P]   -> [L,B,H,Q,P]
        # and run the internal kernel in FP32. The internal result is
        # [B,H*D,Q], which is transposed and cast back below.
        if use_310p_internal_layout:
            value = ge_apis_module.Transpose(value, [0, 2, 1, 3])
            sampling_locations = ge_apis_module.Transpose(
                sampling_locations,
                [3, 0, 2, 1, 4, 5],
            )
            attention_weights = ge_apis_module.Transpose(
                attention_weights,
                [3, 0, 2, 1, 4],
            )
            value = ge_apis_module.Cast(value, dst_type=ge_data_type.DT_FLOAT)
            sampling_locations = ge_apis_module.Cast(
                sampling_locations,
                dst_type=ge_data_type.DT_FLOAT,
            )
            attention_weights = ge_apis_module.Cast(
                attention_weights,
                dst_type=ge_data_type.DT_FLOAT,
            )

        output = ge_custom_op(
            GE_OP_NAME,
            inputs={
                "value": value,
                "value_spatial_shapes": value_spatial_shapes,
                "value_level_start_index": value_level_start_index,
                "sampling_locations": sampling_locations,
                "attention_weights": attention_weights,
            },
            outputs=["output"],
        )
        # CANN declares every internal MSDA tensor as ND. Without these
        # annotations TorchAir can assign NCHW to low-rank metadata tensors.
        specific_input_layout(
            output,
            indices=[0, 1, 2, 3, 4],
            layout="ND",
        )
        specific_output_layout(output, indices=0, layout="ND")

        if use_310p_internal_layout:
            _refresh_layout_msda_host_infer()
            # Keep the custom node intermediate. This makes GE invoke the
            # corrected host infer callback registered by the extension,
            # which supplies the 310P internal [B,H*D,Q] contract.
            output = ge_apis_module.Transpose(output, [0, 2, 1])
            output = ge_apis_module.Cast(
                output,
                dst_type=meta_outputs.dtype,
            )
        elif force_host_infer_probe:
            _refresh_layout_msda_host_infer()
            # Diagnostic only: make the custom node intermediate on 910B so
            # CANN must invoke a registered host infer callback, matching the
            # graph topology of the 310P wrapper path. A same-dtype Cast is
            # expected to disappear after graph construction.
            output = ge_apis_module.Cast(
                output,
                dst_type=meta_outputs.dtype,
            )
        return output

    # The stock layout model computes level_start_index from its three-row
    # spatial-shape tensor with prod(dim=1). That result was dead in the old
    # GridSample decomposition, but the native operator consumes it. TorchAir
    # 2.10 ships only a NotImplemented stub for aten.prod.dim_int, so replace
    # that stub for this exact live contract while the ACLNN lane is active.
    @register_converter(torch.ops.aten.prod.dim_int)
    def _convert_layout_spatial_prod(
        inputs: Any,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: int | None = None,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        if int(dim) != 1 or bool(keepdim) or dtype is not None:
            raise NotImplementedError(
                "layout MSDA only lowers prod(dim=1, keepdim=False, dtype=None)"
            )
        inputs_i32 = ge_apis_module.Cast(inputs, dst_type=ge_data_type.DT_INT32)
        product_i32 = ge_apis_module.ReduceProdD(
            inputs_i32,
            axes=[1],
            keep_dims=False,
        )
        return ge_apis_module.Cast(product_i32, dst_type=ge_data_type.DT_INT64)

    _CONVERTER_DEVICE_NAME = device_name
    _CONVERTER_USES_310P_INTERNAL_LAYOUT = use_310p_internal_layout
    _CONVERTER_REGISTERED = True


class LayoutMsdaAclnn(nn.Module):
    """Match Transformers' MSDA module signature with one installed ACLNN op."""

    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ) -> torch.Tensor:
        del value_spatial_shapes_list, im2col_step
        return torch.ops.unirec_layout.msda_aclnn(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )


def rewrite_layout_msda(
    model: nn.Module,
    *,
    implementation: str,
    extension_so: str | Path | None,
    register_converter: bool,
) -> dict[str, Any]:
    """Replace all six parameter-free decompositions with the ACLNN call."""
    if implementation == "decomposed":
        if extension_so is not None:
            raise ValueError(
                "msda_extension_so is only valid with implementation='aclnn'"
            )
        return {
            "implementation": implementation,
            "target_count": 0,
            "rewritten_count": 0,
            "modules": [],
            "extension_so": None,
            "converter_registered": False,
        }
    if implementation != "aclnn":
        raise ValueError(f"unsupported layout MSDA implementation: {implementation}")
    if extension_so is None:
        raise ValueError("implementation='aclnn' requires msda_extension_so")
    resolved_extension = load_layout_msda_extension(extension_so)
    if register_converter:
        register_layout_msda_converter()

    from transformers.models.pp_doclayout_v2 import (
        modeling_pp_doclayout_v2 as layout_mod,
    )

    rewritten: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, layout_mod.MultiScaleDeformableAttention):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = LayoutMsdaAclnn()
        rewritten.append(name)
    if len(rewritten) != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            "unexpected PP-DocLayoutV2 MSDA module count: "
            f"expected {EXPECTED_MODULE_COUNT}, rewrote {len(rewritten)}"
        )
    return {
        "implementation": implementation,
        "target_count": EXPECTED_MODULE_COUNT,
        "rewritten_count": len(rewritten),
        "modules": rewritten,
        "extension_so": str(resolved_extension),
        "converter_registered": bool(register_converter),
        "converter_device_name": _CONVERTER_DEVICE_NAME,
        "converter_uses_310p_internal_layout": (
            _CONVERTER_USES_310P_INTERNAL_LAYOUT
        ),
        "ge_op": GE_OP_NAME,
        "torch_op": PYTORCH_OP_NAME,
    }
