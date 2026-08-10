"""Packed B1 QKV, RoPE, KV update, mask build, and GQA AscendC op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_packed_qkv_rope_gqa_mixed24"
GE_OP_NAME = "PaddleDecodePackedQkvRopeGqaMixed24"
INPUT_LAYOUT = "BNSD"


def _rotary_half(
    states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return (states * cosine) + (rotated * sine)


@torch.library.custom_op(
    PYTORCH_OP_NAME,
    mutates_args=("qkv", "key_cache", "value_cache", "attention_mask"),
)
def _decode_packed_qkv_rope_gqa_mixed24(
    qkv: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    factor_lut: torch.Tensor,
    rope_delta: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    """Stock eager reference for the independent packed GE identity."""
    import torch_npu

    query_raw, key_raw, value_raw = qkv.split((2048, 256, 256), dim=-1)
    query = query_raw.view(1, 16, 1, 128)
    key_state = key_raw.view(1, 2, 1, 128)
    value_state = value_raw.view(1, 2, 1, 128)
    rope_position = (cache_position.reshape(-1) + rope_delta.reshape(-1))[0]
    factors = factor_lut[:, rope_position, :]
    cosine = factors[0].view(1, 1, 1, 128)
    sine = factors[1].view(1, 1, 1, 128)
    query = _rotary_half(query, cosine, sine)
    key_state = _rotary_half(key_state, cosine, sine)
    qkv.copy_(
        torch.cat(
            (
                query.reshape(1, 1, 2048),
                key_state.reshape(1, 1, 256),
                value_state.reshape(1, 1, 256),
            ),
            dim=-1,
        )
    )
    positions = cache_position.reshape(-1).contiguous()
    torch_npu.scatter_update_(key_cache, positions, key_state, 2)
    torch_npu.scatter_update_(value_cache, positions, value_state, 2)
    future_mask = (
        torch.arange(1024, dtype=torch.int64, device=qkv.device) > positions[0]
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


@_decode_packed_qkv_rope_gqa_mixed24.register_fake
def _decode_packed_qkv_rope_gqa_mixed24_fake(
    qkv: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    factor_lut: torch.Tensor,
    rope_delta: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    del key_cache, value_cache, attention_mask, cache_position
    del factor_lut, rope_delta, num_heads, num_key_value_heads
    del scale_value, inner_precise, vector_core_count
    return torch.empty((1, 16, 1, 128), dtype=qkv.dtype, device=qkv.device)


_CONVERTER_REGISTERED = False
_REF_PASS_PATCHED = False


def _patch_torchair_ref_mapping(torchair: Any) -> None:
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
            mapping[4] = 0
        return mapping

    graph_pass._get_output_to_input_ref_idx = _get_output_to_input_ref_idx
    _REF_PASS_PATCHED = True


def register_decode_packed_qkv_rope_gqa_mixed24_converter() -> None:
    """Lower the unique PyTorch identity to the packed mixed24 GE op."""
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
        torch.ops.paddleocr_vl.decode_packed_qkv_rope_gqa_mixed24.default
    )
    def _convert_decode_packed_qkv_rope_gqa_mixed24(
        qkv: Any,
        key_cache: Any,
        value_cache: Any,
        attention_mask: Any,
        cache_position: Any,
        factor_lut: Any,
        rope_delta: Any,
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
                "query": qkv,
                "key": key_cache,
                "value": value_cache,
                "atten_mask": attention_mask,
                "cache_position": cache_position,
                "factor_lut": factor_lut,
                "rope_delta": rope_delta,
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
            outputs=["attention_out", "key", "value", "atten_mask", "qkv"],
        )
        return outputs[0]

    _CONVERTER_REGISTERED = True


def decode_packed_qkv_rope_gqa_mixed24(
    qkv: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    factor_lut: torch.Tensor,
    rope_delta: torch.Tensor,
    *,
    num_heads: int = 16,
    num_key_value_heads: int = 2,
    scale_value: float,
    inner_precise: int = 1,
    vector_core_count: int = 16,
) -> torch.Tensor:
    if qkv.shape != (1, 1, 2560) or qkv.dtype != torch.float16:
        raise ValueError("packed mixed24 GQA requires FP16 qkv[1,1,2560]")
    if key_cache.shape != (1, 2, 1024, 128):
        raise ValueError("packed mixed24 GQA requires key cache[1,2,1024,128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("packed mixed24 GQA requires matching K/V caches")
    if attention_mask.shape != (1, 1, 1, 1024):
        raise ValueError("packed mixed24 GQA requires mask[1,1,1,1024]")
    if attention_mask.dtype != torch.bool:
        raise ValueError("packed mixed24 GQA requires a bool mask")
    if cache_position.shape != (1,) or cache_position.dtype != torch.int64:
        raise ValueError("packed mixed24 GQA requires INT64 cache_position[1]")
    if factor_lut.shape != (2, 1024, 128) or factor_lut.dtype != torch.float16:
        raise ValueError("packed mixed24 GQA requires FP16 factor_lut[2,1024,128]")
    if rope_delta.shape not in ((1,), (1, 1)) or rope_delta.dtype != torch.int64:
        raise ValueError("packed mixed24 GQA requires INT64 rope_delta[1] or [1,1]")
    if any(t.dtype != torch.float16 for t in (key_cache, value_cache)):
        raise ValueError("packed mixed24 GQA requires FP16 caches")
    if num_heads != 16 or num_key_value_heads != 2:
        raise ValueError("packed mixed24 GQA is fixed to 16Q:2KV")
    if vector_core_count != 16:
        raise ValueError("packed mixed24 GQA uses 16 attention workers")
    return _decode_packed_qkv_rope_gqa_mixed24(
        qkv,
        key_cache,
        value_cache,
        attention_mask,
        cache_position,
        factor_lut,
        rope_delta,
        num_heads,
        num_key_value_heads,
        scale_value,
        inner_precise,
        vector_core_count,
    )
