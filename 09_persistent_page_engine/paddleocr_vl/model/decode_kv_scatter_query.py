"""Mutable B1 K/V scatter plus query ordering token for TorchAir."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_kv_scatter_query_v2"
GE_OP_NAME = "PaddleDecodeKvScatterQueryV2"


@torch.library.custom_op(
    PYTORCH_OP_NAME,
    mutates_args=("key_cache", "value_cache"),
)
def _decode_kv_scatter_query(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
) -> torch.Tensor:
    # This eager implementation is also the correctness reference. TorchAir
    # auto-functionalizes the two mutable arguments and lowers the call to the
    # independent AscendC reference operator below.
    import torch_npu

    positions = cache_position.reshape(-1).contiguous()
    torch_npu.scatter_update_(key_cache, positions, key_state, 2)
    torch_npu.scatter_update_(value_cache, positions, value_state, 2)
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
_REF_PASS_PATCHED = False


def _patch_torchair_ref_mapping(torchair: Any) -> None:
    """Teach this installed TorchAir release the V2 reference-output ABI."""
    global _REF_PASS_PATCHED
    if _REF_PASS_PATCHED:
        return
    graph_pass = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.graph_pass"
    )
    original = graph_pass._get_output_to_input_ref_idx

    def _get_output_to_input_ref_idx(op: Any) -> dict[int, int]:
        mapping = dict(original(op))
        if op.type == GE_OP_NAME:
            mapping[1] = 1
            mapping[2] = 2
        return mapping

    graph_pass._get_output_to_input_ref_idx = _get_output_to_input_ref_idx
    _REF_PASS_PATCHED = True


def register_decode_kv_scatter_query_converter() -> None:
    """Lower the mutable custom op to the independent AscendC ref op."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    torchair, _CompilerConfig = import_torchair()
    _patch_torchair_ref_mapping(torchair)
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op

    @register_converter(torch.ops.paddleocr_vl.decode_kv_scatter_query_v2.default)
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
        outputs = ge_custom_op(
            GE_OP_NAME,
            inputs={
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "cache_position": cache_position,
                "key_state": key_state,
                "value_state": value_state,
            },
            outputs=["ordered_query", "key_cache", "value_cache"],
        )
        # The PyTorch schema returns only the query. TorchAir's mutable-op
        # functionalization discovers the two cache ref outputs by their names.
        return outputs[0]

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
