"""B1 device-scalar decode position backed by an independent AscendC op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


GE_OP_NAME = "PaddleDecodePositionAddV1"
PYTORCH_OP_NAME = "paddleocr_vl::decode_position_add_v1"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_position_add(
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> torch.Tensor:
    return cache_position + rope_delta


@_decode_position_add.register_fake
def _decode_position_add_fake(
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> torch.Tensor:
    del rope_delta
    return torch.empty_like(cache_position)


_CONVERTER_REGISTERED = False


def register_decode_position_add_converter() -> None:
    """Lower the graph identity to the independent scalar AscendC op."""
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

    @register_converter(torch.ops.paddleocr_vl.decode_position_add_v1.default)
    def _convert_decode_position_add(
        cache_position: Any,
        rope_delta: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={
                "cache_position": cache_position,
                "rope_delta": rope_delta,
            },
            outputs=["decode_position"],
        )

    _CONVERTER_REGISTERED = True


def decode_position_add(
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> torch.Tensor:
    """Add the specialized B1/S1 INT64 decode-position scalars."""
    if cache_position.shape != (1, 1) or rope_delta.shape != (1, 1):
        raise ValueError("decode_position_add requires two B1/S1 tensors")
    if cache_position.dtype != torch.int64 or rope_delta.dtype != torch.int64:
        raise ValueError("decode_position_add requires INT64 inputs")
    return _decode_position_add(cache_position, rope_delta)
