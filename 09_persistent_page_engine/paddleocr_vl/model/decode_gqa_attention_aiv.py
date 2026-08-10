"""SuperKernel-safe attention-only B1/KV1024 GQA AIV graph operator."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_gqa_attention_aiv"
GE_OP_NAME = "PaddleDecodeGqaAttentionAiv"
INPUT_LAYOUT = "BNSD"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_gqa_attention_aiv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    """Use stock IncreFA as the eager numerical reference."""
    import torch_npu

    return torch_npu.npu_incre_flash_attention(
        query,
        key,
        value,
        atten_mask=atten_mask,
        actual_seq_lengths=None,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout=INPUT_LAYOUT,
        scale_value=scale_value,
        inner_precise=inner_precise,
    )


@_decode_gqa_attention_aiv.register_fake
def _decode_gqa_attention_aiv_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    inner_precise: int,
    vector_core_count: int,
) -> torch.Tensor:
    del key, value, atten_mask, num_heads, num_key_value_heads
    del scale_value, inner_precise, vector_core_count
    return torch.empty_like(query)


_CONVERTER_REGISTERED = False


def register_decode_gqa_attention_aiv_converter() -> None:
    """Lower the graph op to the attention-only custom AscendC entry."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    ge_attr = importlib.import_module("torchair.ge.attr")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op

    @register_converter(torch.ops.paddleocr_vl.decode_gqa_attention_aiv.default)
    def _convert_decode_gqa_attention_aiv(
        query: Any,
        key: Any,
        value: Any,
        atten_mask: Any,
        num_heads: int,
        num_key_value_heads: int,
        scale_value: float,
        inner_precise: int,
        vector_core_count: int,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={
                "query": query,
                "key": key,
                "value": value,
                "atten_mask": atten_mask,
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
            outputs=["attention_out"],
        )

    _CONVERTER_REGISTERED = True


def decode_gqa_attention_aiv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    num_heads: int = 16,
    num_key_value_heads: int = 2,
    scale_value: float,
    inner_precise: int = 1,
    vector_core_count: int = 16,
) -> torch.Tensor:
    """Execute the fixed B1/16Q/2KV/D128/KV1024 non-split boundary."""
    if query.shape != (1, 16, 1, 128):
        raise ValueError("decode GQA attention requires query[1,16,1,128]")
    if key.shape != (1, 2, 1024, 128) or value.shape != key.shape:
        raise ValueError("decode GQA attention requires K/V[1,2,1024,128]")
    if atten_mask.shape != (1, 1, 1, 1024) or atten_mask.dtype != torch.bool:
        raise ValueError("decode GQA attention requires bool mask[1,1,1,1024]")
    if any(tensor.dtype != torch.float16 for tensor in (query, key, value)):
        raise ValueError("decode GQA attention requires FP16 Q/K/V")
    if num_heads != 16 or num_key_value_heads != 2:
        raise ValueError("decode GQA attention fixes 16 query and 2 KV heads")
    if inner_precise != 1 or vector_core_count != 16:
        raise ValueError("decode GQA attention fixes inner_precise=1 and 16 AIV cores")
    return _decode_gqa_attention_aiv(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        atten_mask.contiguous(),
        num_heads,
        num_key_value_heads,
        scale_value,
        inner_precise,
        vector_core_count,
    )
