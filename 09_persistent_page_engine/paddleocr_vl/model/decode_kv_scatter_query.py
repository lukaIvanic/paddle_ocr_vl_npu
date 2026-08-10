"""Ordering-token binding for the B1 AscendC K/V scatter side effect."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_kv_scatter_query_v1"
GE_OP_NAME = "PaddleDecodeKvScatterQueryV1"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_kv_scatter_query(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> torch.Tensor:
    # Eager is only a trace/reference surface. The installed AscendC kernel
    # performs the persistent cache side effect; its ordered query output keeps
    # that side effect live and ordered before attention in the GE graph.
    del key_cache, value_cache, cache_position, key_state, value_state
    return query.clone()


@_decode_kv_scatter_query.register_fake
def _decode_kv_scatter_query_fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> torch.Tensor:
    del key_cache, value_cache, cache_position, key_state, value_state
    return torch.empty_like(query)


_CONVERTER_REGISTERED = False


def register_decode_kv_scatter_query_converter() -> None:
    """Lower the explicit ordering identity to the independent AscendC op."""
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

    @register_converter(torch.ops.paddleocr_vl.decode_kv_scatter_query_v1.default)
    def _convert_decode_kv_scatter_query(
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
            outputs=["ordered_query"],
        )

    _CONVERTER_REGISTERED = True


def decode_kv_scatter_query(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> torch.Tensor:
    """Write BNSD K/V state and return the query ordering dependency."""
    if query.shape != (1, 16, 1, 128):
        raise ValueError("specialized KV scatter query requires Q[1,16,1,128]")
    if key_cache.shape != (1, 2, 1024, 128):
        raise ValueError("specialized KV scatter query requires K[1,2,1024,128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("specialized KV scatter query requires matching K/V caches")
    if key_state.shape != (1, 2, 1, 128) or value_state.shape != key_state.shape:
        raise ValueError("specialized KV scatter query requires state[1,2,1,128]")
    if cache_position.shape != (1,) or cache_position.dtype != torch.int64:
        raise ValueError("specialized KV scatter query requires position int64[1]")
    tensors = (query, key_cache, value_cache, key_state, value_state)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("specialized KV scatter query requires FP16 tensors")
    return _decode_kv_scatter_query(
        query.contiguous(),
        key_cache,
        value_cache,
        cache_position.contiguous(),
        key_state.contiguous(),
        value_state.contiguous(),
    )
