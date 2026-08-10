"""Mixed-task B1 KV update, future-mask build, and all-vector GQA."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_gqa_incre_flash_attention_mixed"
GE_OP_NAME = "PaddleDecodeGqaIncreFlashAttentionMixed"
INPUT_LAYOUT = "BNSD"


@torch.library.custom_op(
    PYTORCH_OP_NAME,
    mutates_args=("key_cache", "value_cache", "attention_mask"),
)
def _decode_gqa_incre_flash_attention_mixed(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    """Stock eager reference for the independently named mixed GE op."""
    import torch_npu

    positions = cache_position.reshape(-1).contiguous()
    torch_npu.scatter_update_(key_cache, positions, key_state, 2)
    torch_npu.scatter_update_(value_cache, positions, value_state, 2)
    future_mask = (
        torch.arange(1024, dtype=torch.int64, device=query.device)
        > positions[0]
    ).view(1, 1, 1, 1024)
    attention_mask.copy_(future_mask)
    return torch_npu.npu_incre_flash_attention(
        query,
        key_cache,
        value_cache,
        atten_mask=attention_mask,
        actual_seq_lengths=None,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout=INPUT_LAYOUT,
        scale_value=scale_value,
        inner_precise=inner_precise,
    )


@_decode_gqa_incre_flash_attention_mixed.register_fake
def _decode_gqa_incre_flash_attention_mixed_fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    del key_cache, value_cache, attention_mask, cache_position
    del key_state, value_state, num_heads, num_key_value_heads
    del scale_value, inner_precise, vector_core_count
    return torch.empty_like(query)


_CONVERTER_REGISTERED = False
_REF_PASS_PATCHED = False


def _patch_torchair_ref_mapping(torchair: Any) -> None:
    """Map the compact GE op's trailing outputs back to mutable inputs."""
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
            mapping[3] = 3
        return mapping

    graph_pass._get_output_to_input_ref_idx = _get_output_to_input_ref_idx
    _REF_PASS_PATCHED = True


def register_decode_gqa_increfa_mixed_converter() -> None:
    """Lower the unique PyTorch identity to the mixed-task fused GE op."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    torchair, _CompilerConfig = import_torchair()
    _patch_torchair_ref_mapping(torchair)
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    ge_attr = importlib.import_module("torchair.ge.attr")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op

    @register_converter(
        torch.ops.paddleocr_vl.decode_gqa_incre_flash_attention_mixed.default
    )
    def _convert_decode_gqa_incre_flash_attention_mixed(
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attention_mask: Any,
        cache_position: Any,
        key_state: Any,
        value_state: Any,
        num_heads: int,
        num_key_value_heads: int,
        scale_value: float,
        inner_precise: int,
        vector_core_count: int,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        outputs = ge_custom_op(
            GE_OP_NAME,
            inputs={
                "query": query,
                "key": key_cache,
                "value": value_cache,
                "atten_mask": attention_mask,
                "cache_position": cache_position,
                "key_state": key_state,
                "value_state": value_state,
            },
            attrs={
                "num_heads": ge_attr.Int(num_heads),
                "scale_value": ge_attr.Float(scale_value),
                "input_layout": ge_attr.Str(INPUT_LAYOUT),
                "num_key_value_heads": ge_attr.Int(num_key_value_heads),
                "block_size": ge_attr.Int(0),
                "inner_precise": ge_attr.Int(inner_precise),
                "vector_core_count": ge_attr.Int(vector_core_count),
            },
            outputs=["attention_out", "key", "value", "atten_mask"],
        )
        return outputs[0]

    _CONVERTER_REGISTERED = True


def decode_gqa_incre_flash_attention_mixed(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    key_state: torch.Tensor,
    value_state: torch.Tensor,
    *,
    num_heads: int = 16,
    num_key_value_heads: int = 2,
    scale_value: float,
    inner_precise: int = 1,
    vector_core_count: int = 16,
) -> torch.Tensor:
    """Execute the KV1024 all-vector algorithm with mixed task geometry."""
    if query.shape != (1, 16, 1, 128):
        raise ValueError("mixed fused GQA requires query[1,16,1,128]")
    if key_cache.shape != (1, 2, 1024, 128):
        raise ValueError("mixed fused GQA requires key cache[1,2,1024,128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("mixed fused GQA requires matching K/V caches")
    if key_state.shape != (1, 2, 1, 128) or value_state.shape != key_state.shape:
        raise ValueError("mixed fused GQA requires state[1,2,1,128]")
    if attention_mask.shape != (1, 1, 1, 1024):
        raise ValueError("mixed fused GQA requires mask scratch[1,1,1,1024]")
    if attention_mask.dtype != torch.bool:
        raise ValueError("mixed fused GQA requires a bool mask scratch")
    if cache_position.shape != (1,) or cache_position.dtype != torch.int64:
        raise ValueError("mixed fused GQA requires int64 cache position[1]")
    if any(
        tensor.dtype != torch.float16
        for tensor in (query, key_cache, value_cache, key_state, value_state)
    ):
        raise ValueError("mixed fused GQA requires FP16 Q/K/V/state tensors")
    if num_heads != 16 or num_key_value_heads != 2:
        raise ValueError("mixed fused GQA fixes 16 query and 2 KV heads")
    if inner_precise != 1 or vector_core_count != 16:
        raise ValueError("mixed fused GQA fixes inner_precise=1 and 16 AIV cores")
    return _decode_gqa_incre_flash_attention_mixed(
        query.contiguous(),
        key_cache,
        value_cache,
        attention_mask,
        cache_position.contiguous(),
        key_state.contiguous(),
        value_state.contiguous(),
        num_heads,
        num_key_value_heads,
        scale_value,
        inner_precise,
        vector_core_count,
    )
