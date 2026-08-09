"""Explicit PyTorch binding for the separate Paddle MHA AIV operator.

The eager custom-op body calls stock ``torch_npu`` IncreFA as its reference.
TorchAir compilation never lowers that body.  The registered converter emits
the separately packaged ``PaddleMhaIncreFlashAttentionAiv`` GE operator.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


GE_OP_NAME = "PaddleMhaIncreFlashAttentionAiv"
PYTORCH_OP_NAME = "paddleocr_vl::mha_incre_flash_attention_aiv"
INPUT_LAYOUT = "BNSD"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _mha_incre_flash_attention_aiv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    num_heads: int,
    scale_value: float,
    inner_precise: int,
) -> torch.Tensor:
    """Eager reference; the compiled converter targets the distinct GE op."""
    import torch_npu

    return torch_npu.npu_incre_flash_attention(
        query,
        key,
        value,
        atten_mask=atten_mask,
        actual_seq_lengths=None,
        num_heads=num_heads,
        num_key_value_heads=0,
        input_layout=INPUT_LAYOUT,
        scale_value=scale_value,
        inner_precise=inner_precise,
    )


@_mha_incre_flash_attention_aiv.register_fake
def _mha_incre_flash_attention_aiv_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    num_heads: int,
    scale_value: float,
    inner_precise: int,
) -> torch.Tensor:
    del key, value, atten_mask, num_heads, scale_value, inner_precise
    return torch.empty_like(query)


_CONVERTER_REGISTERED = False


def register_mha_increfa_aiv_converter() -> None:
    """Lower the explicit PyTorch op to its separately named CANN operator."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return

    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    ge_attr = importlib.import_module(f"{torchair.__name__}.ge.attr")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op
    op = torch.ops.paddleocr_vl.mha_incre_flash_attention_aiv.default

    @register_converter(op)
    def _convert_mha_incre_flash_attention_aiv(
        query: Any,
        key: Any,
        value: Any,
        atten_mask: Any,
        num_heads: int,
        scale_value: float,
        inner_precise: int,
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
                "num_key_value_heads": ge_attr.Int(0),
                "block_size": ge_attr.Int(0),
                "inner_precise": ge_attr.Int(inner_precise),
            },
            outputs=["attention_out"],
        )

    _CONVERTER_REGISTERED = True


def mha_incre_flash_attention_aiv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    num_heads: int,
    scale_value: float,
    inner_precise: int = 1,
) -> torch.Tensor:
    """Call the separate FP16/BNSD/B1/MHA AIV operator explicitly.

    These checks are deliberately narrower than stock IncreFA.  GQA, other
    layouts, other dtypes, and batches larger than one must use another op.
    """
    tensors = (query, key, value)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("Paddle MHA IncreFA AIV requires FP16 Q/K/V")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Paddle MHA IncreFA AIV requires rank-4 BNSD Q/K/V")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("Paddle MHA IncreFA AIV currently supports B1 only")
    if query.shape[2] != 1:
        raise ValueError("Paddle MHA IncreFA AIV requires one query token")
    if key.shape != value.shape:
        raise ValueError("Paddle MHA IncreFA AIV requires matching K/V shapes")
    if query.shape[1] != num_heads or key.shape[1] != num_heads:
        raise ValueError(
            "Paddle MHA IncreFA AIV requires equal query and KV head counts"
        )
    if query.shape[3] != key.shape[3]:
        raise ValueError("Paddle MHA IncreFA AIV requires matching head dims")
    if atten_mask.dtype != torch.bool:
        raise ValueError("Paddle MHA IncreFA AIV requires a bool mask")
    if inner_precise != 1:
        raise ValueError("Paddle MHA IncreFA AIV currently fixes inner_precise=1")
    return _mha_incre_flash_attention_aiv(
        query,
        key,
        value,
        atten_mask,
        num_heads,
        scale_value,
        inner_precise,
    )
