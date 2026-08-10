"""B1 decode RoPE factors backed by one independent AscendC lookup op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


GE_OP_NAME = "PaddleDecodeRopeLookupV1"
PYTORCH_OP_NAME = "paddleocr_vl::decode_rope_lookup_v1"


def _reference(
    factor_lut: torch.Tensor,
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = (cache_position + rope_delta).reshape(-1)
    selected = torch.index_select(factor_lut, 1, position)
    cos, sin = selected.unbind(dim=0)
    return (
        cos.unsqueeze(1).unsqueeze(1).clone(),
        sin.unsqueeze(1).unsqueeze(1).clone(),
    )


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_rope_lookup(
    factor_lut: torch.Tensor,
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reference(factor_lut, cache_position, rope_delta)


@_decode_rope_lookup.register_fake
def _decode_rope_lookup_fake(
    factor_lut: torch.Tensor,
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cache_position, rope_delta
    output = torch.empty(
        (1, 1, 1, 128), dtype=factor_lut.dtype, device=factor_lut.device
    )
    return output, torch.empty_like(output)


_CONVERTER_REGISTERED = False


def register_decode_rope_lookup_converter() -> None:
    """Lower the factor lookup identity to the independent AscendC op."""
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

    @register_converter(torch.ops.paddleocr_vl.decode_rope_lookup_v1.default)
    def _convert_decode_rope_lookup(
        factor_lut: Any,
        cache_position: Any,
        rope_delta: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={
                "factor_lut": factor_lut,
                "cache_position": cache_position,
                "rope_delta": rope_delta,
            },
            outputs=["cos", "sin"],
        )

    _CONVERTER_REGISTERED = True


def decode_rope_lookup(
    factor_lut: torch.Tensor,
    cache_position: torch.Tensor,
    rope_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final FP16 B1/S1/D128 cosine and sine factors."""
    if factor_lut.shape != (2, 1024, 128):
        raise ValueError("decode_rope_lookup requires factor_lut[2,1024,128]")
    if cache_position.shape != (1, 1) or rope_delta.shape != (1, 1):
        raise ValueError("decode_rope_lookup requires B1/S1 position tensors")
    if factor_lut.dtype != torch.float16:
        raise ValueError("decode_rope_lookup requires an FP16 factor LUT")
    if cache_position.dtype != torch.int64 or rope_delta.dtype != torch.int64:
        raise ValueError("decode_rope_lookup requires INT64 position tensors")
    return _decode_rope_lookup(factor_lut, cache_position, rope_delta)
