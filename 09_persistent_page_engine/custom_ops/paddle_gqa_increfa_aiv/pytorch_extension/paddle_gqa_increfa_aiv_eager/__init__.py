from __future__ import annotations

from pathlib import Path

import torch

PYTORCH_OP_NAME = "paddleocr_vl_npu::paddle_gqa_incre_flash_attention_aiv_eager"
ACLNN_OP_NAME = "aclnnPaddleGqaIncreFlashAttentionAiv"


def _load_extension() -> Path:
    candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if len(candidates) != 1:
        raise ImportError("expected one built _C*.so; run pytorch_extension/build.sh")
    torch.ops.load_library(str(candidates[0]))
    return candidates[0]


EXTENSION_PATH = _load_extension()


def paddle_gqa_incre_flash_attention_aiv_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    num_heads: int = 16,
    num_key_value_heads: int = 2,
    scale_value: float,
    inner_precise: int = 1,
    vector_core_count: int = 48,
) -> torch.Tensor:
    return torch.ops.paddleocr_vl_npu.paddle_gqa_incre_flash_attention_aiv_eager(
        query, key, value, atten_mask, num_heads, num_key_value_heads,
        scale_value, inner_precise, vector_core_count
    )


__all__ = [
    "ACLNN_OP_NAME", "EXTENSION_PATH", "PYTORCH_OP_NAME",
    "paddle_gqa_incre_flash_attention_aiv_eager",
]
