"""Alias-free functional B1 K/V preparation for a strict SuperKernel."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_kv_prepare_functional_mixed24"
GE_OP_NAME = "PaddleDecodeKvPrepareFunctionalMixed24"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_kv_prepare_functional_mixed24(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch_npu

    positions = cache_position.reshape(-1).contiguous()
    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()
    torch_npu.scatter_update_(key_cache_out, positions, key_state, 2)
    torch_npu.scatter_update_(value_cache_out, positions, value_state, 2)
    kv_positions = torch.arange(
        1024,
        device=cache_position.device,
        dtype=torch.int64,
    )
    attention_mask = (
        kv_positions > cache_position.reshape(-1)[0]
    ).view(1, 1, 1, 1024)
    return query.clone(), attention_mask, key_cache_out, value_cache_out


@_decode_kv_prepare_functional_mixed24.register_fake
def _decode_kv_prepare_functional_mixed24_fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del cache_position, key_state, value_state
    return (
        torch.empty_like(query),
        torch.empty(
            (1, 1, 1, 1024),
            dtype=torch.bool,
            device=query.device,
        ),
        torch.empty_like(key_cache),
        torch.empty_like(value_cache),
    )


_CONVERTER_REGISTERED = False


def register_decode_kv_prepare_functional_mixed24_converter() -> None:
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

    @register_converter(
        torch.ops.paddleocr_vl.decode_kv_prepare_functional_mixed24.default
    )
    def _convert_decode_kv_prepare_functional_mixed24(
        query: Any,
        key_cache: Any,
        value_cache: Any,
        cache_position: Any,
        key_state: Any,
        value_state: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "cache_position": cache_position,
                "key_state": key_state,
                "value_state": value_state,
            },
            outputs=[
                "ordered_query",
                "attention_mask",
                "key_cache_out",
                "value_cache_out",
            ],
        )

    _CONVERTER_REGISTERED = True


def decode_kv_prepare_functional_mixed24(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.shape != (1, 16, 1, 128):
        raise ValueError("functional KV prepare requires Q[1,16,1,128]")
    if key_cache.shape != (1, 2, 1024, 128):
        raise ValueError("functional KV prepare requires K[1,2,1024,128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("functional KV prepare requires matching K/V caches")
    if key_state.shape != (1, 2, 1, 128) or value_state.shape != key_state.shape:
        raise ValueError("functional KV prepare requires state[1,2,1,128]")
    if cache_position.shape != (1,) or cache_position.dtype != torch.int64:
        raise ValueError("functional KV prepare requires position int64[1]")
    tensors = (query, key_cache, value_cache, key_state, value_state)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("functional KV prepare requires FP16 tensors")
    return _decode_kv_prepare_functional_mixed24(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        cache_position.contiguous(),
        key_state.contiguous(),
        value_state.contiguous(),
    )
