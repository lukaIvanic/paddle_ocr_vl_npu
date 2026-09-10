#!/usr/bin/env python3
"""Dependency-free PaddleOCR-VL model config dataclasses.

These mirror the Transformers PaddleOCR-VL config fields that matter for
inference, without importing transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    attention_dropout: float = 0.0
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PaddleOCRVisionConfig":
        raw = dict(raw or {})
        return cls(
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(raw.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(raw.get("num_attention_heads", cls.num_attention_heads)),
            num_channels=int(raw.get("num_channels", cls.num_channels)),
            image_size=int(raw.get("image_size", cls.image_size)),
            patch_size=int(raw.get("patch_size", cls.patch_size)),
            hidden_act=str(raw.get("hidden_act", cls.hidden_act)),
            layer_norm_eps=float(raw.get("layer_norm_eps", cls.layer_norm_eps)),
            attention_dropout=float(raw.get("attention_dropout", cls.attention_dropout)),
            spatial_merge_size=int(raw.get("spatial_merge_size", cls.spatial_merge_size)),
            temporal_patch_size=int(raw.get("temporal_patch_size", cls.temporal_patch_size)),
        )


@dataclass(frozen=True)
class PaddleOCRTextConfig:
    vocab_size: int = 103424
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    use_bias: bool = False
    head_dim: int = 128
    rope_parameters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PaddleOCRTextConfig":
        raw = dict(raw or {})
        rope_parameters = dict(raw.get("rope_parameters") or raw.get("rope_scaling") or {})
        rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
        rope_parameters.setdefault("rope_theta", raw.get("rope_theta", 500000.0))
        rope_parameters.setdefault("mrope_section", [16, 24, 24])
        hidden_size = int(raw.get("hidden_size", cls.hidden_size))
        num_heads = int(raw.get("num_attention_heads", cls.num_attention_heads))
        return cls(
            vocab_size=int(raw.get("vocab_size", cls.vocab_size)),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(raw.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=num_heads,
            num_key_value_heads=int(raw.get("num_key_value_heads", cls.num_key_value_heads)),
            hidden_act=str(raw.get("hidden_act", cls.hidden_act)),
            max_position_embeddings=int(raw.get("max_position_embeddings", cls.max_position_embeddings)),
            rms_norm_eps=float(raw.get("rms_norm_eps", cls.rms_norm_eps)),
            pad_token_id=int(raw.get("pad_token_id", cls.pad_token_id)),
            bos_token_id=int(raw.get("bos_token_id", cls.bos_token_id)),
            eos_token_id=int(raw.get("eos_token_id", cls.eos_token_id)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", cls.tie_word_embeddings)),
            use_bias=bool(raw.get("use_bias", cls.use_bias)),
            head_dim=int(raw.get("head_dim") or hidden_size // num_heads),
            rope_parameters=rope_parameters,
        )


@dataclass(frozen=True)
class PaddleOCRVLConfig:
    text_config: PaddleOCRTextConfig
    vision_config: PaddleOCRVisionConfig
    image_token_id: int = 100295
    video_token_id: int = 101307
    vision_start_token_id: int = 101305
    vision_end_token_id: int = 101306
    tie_word_embeddings: bool = False
    eos_token_id: int = 2
    pad_token_id: int = 0

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def rope_parameters(self) -> dict[str, Any]:
        return self.text_config.rope_parameters or {}

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PaddleOCRVLConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "PaddleOCRVLConfig":
        return cls.from_json_file(Path(model_dir) / "config.json")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PaddleOCRVLConfig":
        raw = dict(raw)
        vision_config = PaddleOCRVisionConfig.from_dict(raw.get("vision_config"))
        if "text_config" in raw and raw["text_config"] is not None:
            text_raw = dict(raw["text_config"])
            for key in ("rope_scaling", "rope_theta", "tie_word_embeddings"):
                text_raw.setdefault(key, raw.get(key))
        else:
            text_raw = raw
        text_config = PaddleOCRTextConfig.from_dict(text_raw)
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=int(raw.get("image_token_id", cls.image_token_id)),
            video_token_id=int(raw.get("video_token_id", cls.video_token_id)),
            vision_start_token_id=int(raw.get("vision_start_token_id", cls.vision_start_token_id)),
            vision_end_token_id=int(raw.get("vision_end_token_id", cls.vision_end_token_id)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", text_config.tie_word_embeddings)),
            eos_token_id=int(raw.get("eos_token_id", text_config.eos_token_id)),
            pad_token_id=int(raw.get("pad_token_id", text_config.pad_token_id)),
        )
