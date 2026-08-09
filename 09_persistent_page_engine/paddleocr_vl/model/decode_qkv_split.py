"""B1 packed-QKV layout backed by three single-output AIV AscendC ops."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


GE_QUERY_OP_NAME = "PaddleDecodeQuerySliceV2"
GE_KEY_OP_NAME = "PaddleDecodeKeySliceV2"
GE_VALUE_OP_NAME = "PaddleDecodeValueSliceV2"
GE_OP_NAME = ",".join(
    (GE_QUERY_OP_NAME, GE_KEY_OP_NAME, GE_VALUE_OP_NAME)
)
PYTORCH_QUERY_OP_NAME = "paddleocr_vl::decode_query_slice_v2"
PYTORCH_KEY_OP_NAME = "paddleocr_vl::decode_key_slice_v2"
PYTORCH_VALUE_OP_NAME = "paddleocr_vl::decode_value_slice_v2"
PYTORCH_OP_NAME = ",".join(
    (PYTORCH_QUERY_OP_NAME, PYTORCH_KEY_OP_NAME, PYTORCH_VALUE_OP_NAME)
)


def _reference(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = qkv.split((2048, 256, 256), dim=-1)
    return (
        query.view(1, 1, 16, 128).transpose(1, 2),
        key.view(1, 1, 2, 128).transpose(1, 2),
        value.view(1, 1, 2, 128).transpose(1, 2),
    )


@torch.library.custom_op(PYTORCH_QUERY_OP_NAME, mutates_args=())
def _decode_query_slice(qkv: torch.Tensor) -> torch.Tensor:
    return _reference(qkv)[0]


@_decode_query_slice.register_fake
def _decode_query_slice_fake(qkv: torch.Tensor) -> torch.Tensor:
    return torch.empty((1, 16, 1, 128), dtype=qkv.dtype, device=qkv.device)


@torch.library.custom_op(PYTORCH_KEY_OP_NAME, mutates_args=())
def _decode_key_slice(qkv: torch.Tensor) -> torch.Tensor:
    return _reference(qkv)[1]


@_decode_key_slice.register_fake
def _decode_key_slice_fake(qkv: torch.Tensor) -> torch.Tensor:
    return torch.empty((1, 2, 1, 128), dtype=qkv.dtype, device=qkv.device)


@torch.library.custom_op(PYTORCH_VALUE_OP_NAME, mutates_args=())
def _decode_value_slice(qkv: torch.Tensor) -> torch.Tensor:
    return _reference(qkv)[2]


@_decode_value_slice.register_fake
def _decode_value_slice_fake(qkv: torch.Tensor) -> torch.Tensor:
    return torch.empty((1, 2, 1, 128), dtype=qkv.dtype, device=qkv.device)


_CONVERTER_REGISTERED = False


def register_decode_qkv_split_converter() -> None:
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op

    @register_converter(torch.ops.paddleocr_vl.decode_query_slice_v2.default)
    def _convert_query(qkv: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_QUERY_OP_NAME,
            inputs={"qkv": qkv},
            outputs=["query"],
        )

    @register_converter(torch.ops.paddleocr_vl.decode_key_slice_v2.default)
    def _convert_key(qkv: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_KEY_OP_NAME,
            inputs={"qkv": qkv},
            outputs=["key"],
        )

    @register_converter(torch.ops.paddleocr_vl.decode_value_slice_v2.default)
    def _convert_value(qkv: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_VALUE_OP_NAME,
            inputs={"qkv": qkv},
            outputs=["value"],
        )

    _CONVERTER_REGISTERED = True


def decode_qkv_split(
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if qkv.shape != (1, 1, 2560) or qkv.dtype != torch.float16:
        raise ValueError("decode_qkv_split requires FP16 qkv[1,1,2560]")
    return (
        _decode_query_slice(qkv),
        _decode_key_slice(qkv),
        _decode_value_slice(qkv),
    )
