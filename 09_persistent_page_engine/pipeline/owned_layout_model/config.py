"""Configuration objects for the owned PP-DocLayoutV3 implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HGNetV2Config:
    """The exact HGNetV2-L backbone used by PP-DocLayoutV3."""

    num_channels: int = 3
    embedding_size: int = 64
    hidden_act: str = "relu"
    stem_channels: tuple[int, ...] = (3, 32, 48)
    stem_strides: tuple[int, ...] = (2, 1, 1, 2, 1)
    stage_in_channels: tuple[int, ...] = (48, 128, 512, 1024)
    stage_mid_channels: tuple[int, ...] = (48, 96, 192, 384)
    stage_out_channels: tuple[int, ...] = (128, 512, 1024, 2048)
    stage_num_blocks: tuple[int, ...] = (1, 1, 3, 1)
    stage_downsample: tuple[bool, ...] = (False, True, True, True)
    stage_downsample_strides: tuple[int, ...] = (2, 2, 2, 2)
    stage_light_block: tuple[bool, ...] = (False, False, True, True)
    stage_kernel_size: tuple[int, ...] = (3, 3, 5, 5)
    stage_numb_of_layers: tuple[int, ...] = (6, 6, 6, 6)
    use_learnable_affine_block: bool = False
    out_features: tuple[str, ...] = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return ("stem", "stage1", "stage2", "stage3", "stage4")

    @property
    def channels(self) -> tuple[int, ...]:
        # This intentionally follows the checkpoint's HGNet BackboneMixin
        # metadata. Stage1 is the x4 feature and is removed before these
        # channel declarations are used by the detector projections.
        return (256, 512, 1024, 2048)


@dataclass
class PPDocLayoutV3Config:
    """Inference-relevant PP-DocLayoutV3 configuration."""

    num_labels: int = 25
    id2label: dict[int, str] = field(default_factory=dict)
    batch_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    activation_dropout: float = 0.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "silu"
    encoder_activation_function: str = "gelu"
    decoder_activation_function: str = "relu"
    encoder_hidden_dim: int = 256
    encoder_in_channels: tuple[int, ...] = (512, 1024, 2048)
    feat_strides: tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    encode_proj_layers: tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10_000
    eval_size: tuple[int, int] | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    mask_feature_channels: tuple[int, ...] = (64, 64)
    x4_feat_dim: int = 128
    d_model: int = 256
    num_prototypes: int = 32
    mask_enhanced: bool = True
    num_queries: int = 300
    decoder_in_channels: tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    global_pointer_head_size: int = 64
    gp_dropout_value: float = 0.1
    anchor_image_size: tuple[int, int] | None = None
    learn_initial_query: bool = False
    freeze_backbone_batch_norms: bool = True
    backbone_config: HGNetV2Config = field(default_factory=HGNetV2Config)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "PPDocLayoutV3Config":
        path = model_dir / "config.json"
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        backbone = raw.get("backbone_config") or {}
        if backbone.get("model_type") != "hgnet_v2":
            raise ValueError(
                "owned PP-DocLayoutV3 supports only the HGNetV2 backbone"
            )
        if backbone.get("arch", "L") != "L":
            raise ValueError(
                "owned PP-DocLayoutV3 supports only the HGNetV2-L backbone"
            )

        aliases = {
            "feature_strides": "feat_strides",
        }
        values: dict[str, Any] = {}
        supported = set(cls.__dataclass_fields__) - {"backbone_config"}
        for key, value in raw.items():
            key = aliases.get(key, key)
            if key not in supported:
                continue
            if key == "id2label":
                value = {int(index): str(label) for index, label in value.items()}
            elif isinstance(getattr(cls(), key), tuple) and value is not None:
                value = tuple(value)
            values[key] = value
        values["num_labels"] = len(raw["id2label"])
        values["backbone_config"] = HGNetV2Config(
            out_features=tuple(
                backbone.get(
                    "out_features",
                    ("stage1", "stage2", "stage3", "stage4"),
                )
            )
        )
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        expected = {
            "d_model": 256,
            "encoder_hidden_dim": 256,
            "num_feature_levels": 3,
            "decoder_layers": 6,
            "num_queries": 300,
            "num_labels": 25,
        }
        mismatches = {
            name: (getattr(self, name), value)
            for name, value in expected.items()
            if getattr(self, name) != value
        }
        if mismatches:
            raise ValueError(
                "unsupported PP-DocLayoutV3 checkpoint configuration: "
                f"{mismatches}"
            )
