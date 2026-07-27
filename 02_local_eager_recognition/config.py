#!/usr/bin/env python3
"""Hardcoded PaddleOCR-VL-1.6 architecture config.

Values verified against the checkpoint's config.json on the blue-zone box
(2026-07-27). The checkpoint is pinned; nothing is read from disk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaddleOCRVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 384
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    spatial_merge_size: int = 2


@dataclass(frozen=True)
class PaddleOCRTextConfig:
    vocab_size: int = 103424
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    use_bias: bool = False
    head_dim: int = 128
    rope_theta: float = 500000.0
    mrope_section: tuple[int, int, int] = (16, 24, 24)


@dataclass(frozen=True)
class PaddleOCRVLConfig:
    text_config: PaddleOCRTextConfig = PaddleOCRTextConfig()
    vision_config: PaddleOCRVisionConfig = PaddleOCRVisionConfig()
    image_token_id: int = 100295
    vision_start_token_id: int = 101305
    eos_token_id: int = 2
