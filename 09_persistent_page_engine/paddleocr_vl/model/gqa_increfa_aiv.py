"""Explicit TorchAir binding for the independent Paddle GQA AIV operator."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair

GE_OP_NAME = "PaddleGqaIncreFlashAttentionAiv"
PYTORCH_OP_NAME = "paddleocr_vl::gqa_incre_flash_attention_aiv"
INPUT_LAYOUT = "BNSD"
EXPECTED_QUERY_HEADS = 16
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_DIM = 128
MIN_KV_LENGTH_FOR_48_CORES = 1536


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _gqa_incre_flash_attention_aiv(
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
    """Stock eager reference. TorchAir lowers this to the separate GE op."""
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


@_gqa_incre_flash_attention_aiv.register_fake
def _gqa_incre_flash_attention_aiv_fake(
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


def register_gqa_increfa_aiv_converter() -> None:
    """Lower the explicit graph op to PaddleGqaIncreFlashAttentionAiv."""
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
    op = torch.ops.paddleocr_vl.gqa_incre_flash_attention_aiv.default

    @register_converter(op)
    def _convert_gqa_incre_flash_attention_aiv(
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
                "key": [key],
                "value": [value],
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


def gqa_incre_flash_attention_aiv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    num_heads: int = EXPECTED_QUERY_HEADS,
    num_key_value_heads: int = EXPECTED_KV_HEADS,
    scale_value: float,
    inner_precise: int = 1,
    vector_core_count: int = 48,
) -> torch.Tensor:
    """Call the B1 FP16 16Q:2KV GQA AIV operator explicitly."""
    tensors = (query, key, value)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("Paddle GQA IncreFA AIV requires FP16 Q/K/V")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Paddle GQA IncreFA AIV requires rank-4 BNSD Q/K/V")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("Paddle GQA IncreFA AIV supports B1 only")
    if query.shape[2] != 1 or key.shape != value.shape:
        raise ValueError("Paddle GQA IncreFA AIV requires S_q=1 and matching K/V")
    if num_heads != EXPECTED_QUERY_HEADS or query.shape[1] != EXPECTED_QUERY_HEADS:
        raise ValueError("Paddle GQA IncreFA AIV requires 16 query heads")
    if num_key_value_heads != EXPECTED_KV_HEADS or key.shape[1] != EXPECTED_KV_HEADS:
        raise ValueError("Paddle GQA IncreFA AIV requires 2 KV heads")
    if query.shape[3] != EXPECTED_HEAD_DIM or key.shape[3] != EXPECTED_HEAD_DIM:
        raise ValueError("Paddle GQA IncreFA AIV requires head_dim=128")
    if atten_mask.dtype != torch.bool or atten_mask.ndim != 4:
        raise ValueError("Paddle GQA IncreFA AIV requires a rank-4 bool mask")
    if inner_precise != 1:
        raise ValueError("Paddle GQA IncreFA AIV fixes inner_precise=1")
    if vector_core_count < EXPECTED_QUERY_HEADS or vector_core_count > 48:
        raise ValueError(
            "vector_core_count must be in [16, 48]; this kernel assigns one "
            "AIV work item to each GQA query head"
        )
    if (
        vector_core_count == 48
        and int(key.shape[2]) < MIN_KV_LENGTH_FOR_48_CORES
    ):
        raise ValueError(
            "vector_core_count=48 requires KV length >= "
            f"{MIN_KV_LENGTH_FOR_48_CORES}; the existing three-way split-K "
            "kernel stalls when a partition is shorter than 512 tokens"
        )
    return _gqa_incre_flash_attention_aiv(
        query, key, value, atten_mask, num_heads, num_key_value_heads,
        scale_value, inner_precise, vector_core_count
    )
