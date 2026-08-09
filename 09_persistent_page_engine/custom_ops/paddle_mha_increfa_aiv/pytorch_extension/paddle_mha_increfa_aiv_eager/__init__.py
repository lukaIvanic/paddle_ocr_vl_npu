"""Direct-eager PyTorch API for the separate Paddle MHA AIV operator."""

from __future__ import annotations

from pathlib import Path

import torch


PYTORCH_OP_NAME = (
    "paddleocr_vl_npu::paddle_mha_incre_flash_attention_aiv_eager"
)
ACLNN_OP_NAME = "aclnnPaddleMhaIncreFlashAttentionAiv"


def _load_extension() -> Path:
    candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if len(candidates) != 1:
        raise ImportError(
            "expected one built _C*.so beside paddle_mha_increfa_aiv_eager; "
            "run pytorch_extension/build.sh"
        )
    torch.ops.load_library(str(candidates[0]))
    return candidates[0]


EXTENSION_PATH = _load_extension()


def paddle_mha_incre_flash_attention_aiv_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    num_heads: int,
    scale_value: float,
    inner_precise: int = 1,
) -> torch.Tensor:
    """Execute the separate vendor op directly through PyTorch eager."""
    return torch.ops.paddleocr_vl_npu.paddle_mha_incre_flash_attention_aiv_eager(
        query,
        key,
        value,
        atten_mask,
        num_heads,
        scale_value,
        inner_precise,
    )


__all__ = [
    "ACLNN_OP_NAME",
    "EXTENSION_PATH",
    "PYTORCH_OP_NAME",
    "paddle_mha_incre_flash_attention_aiv_eager",
]
