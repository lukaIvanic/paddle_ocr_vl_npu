"""B1 packed-QKV split backed by an independent AIV-only AscendC op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


GE_OP_NAME = "PaddleDecodeQkvSplitV2"
PYTORCH_OP_NAME = "paddleocr_vl::decode_qkv_split"


def _reference(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = qkv.split((2048, 256, 256), dim=-1)
    return (
        query.view(1, 1, 16, 128).transpose(1, 2),
        key.view(1, 1, 2, 128).transpose(1, 2),
        value.view(1, 1, 2, 128).transpose(1, 2),
    )


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_qkv_split(
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _reference(qkv)


@_decode_qkv_split.register_fake
def _decode_qkv_split_fake(
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((1, 16, 1, 128), dtype=qkv.dtype, device=qkv.device),
        torch.empty((1, 2, 1, 128), dtype=qkv.dtype, device=qkv.device),
        torch.empty((1, 2, 1, 128), dtype=qkv.dtype, device=qkv.device),
    )


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
    op = torch.ops.paddleocr_vl.decode_qkv_split.default

    @register_converter(op)
    def _convert_decode_qkv_split(
        qkv: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={"qkv": qkv},
            outputs=["query", "key", "value"],
        )

    _CONVERTER_REGISTERED = True


def decode_qkv_split(
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if qkv.shape != (1, 1, 2560) or qkv.dtype != torch.float16:
        raise ValueError("decode_qkv_split requires FP16 qkv[1,1,2560]")
    return _decode_qkv_split(qkv)
