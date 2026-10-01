"""Dependency-free configuration for the image/text Ops-Colqwen3-4B baseline."""
from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    hidden_act: str = 'gelu_pytorch_tanh'
    deepstack_visual_indexes: tuple = (5, 11, 17)


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = 'silu'


@dataclass(frozen=True)
class ColQwenConfig:
    vision_config: VisionConfig = VisionConfig()
    text_config: TextConfig = TextConfig()
    dims: int = 2560
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    mask_non_image_embeddings: bool = False
    mrope_section: tuple = (24, 20, 20)

    @classmethod
    def from_dict(cls, raw):
        # Deliberately support one architecture, not silently approximate others.
        if raw.get('model_type') != 'ops_colqwen3':
            raise ValueError('Expected the Ops-Colqwen3 checkpoint')
        for section, expected in [('vision_config', VisionConfig()), ('text_config', TextConfig())]:
            actual = raw[section]
            for field in fields(expected):
                value = actual[field.name]
                target = getattr(expected, field.name)
                if isinstance(target, tuple):
                    value = tuple(value)
                if value != target:
                    raise ValueError(f'Unsupported {section}.{field.name}: {value!r} != {target!r}')
        rope = raw['text_config']['rope_scaling']
        if (rope.get('rope_type') != 'default' or rope.get('mrope_interleaved') is not True
                or tuple(rope['mrope_section']) != cls.mrope_section):
            raise ValueError('Unsupported rotary configuration')
        for name in ('dims', 'image_token_id', 'video_token_id', 'vision_start_token_id'):
            if raw[name] != getattr(cls, name):
                raise ValueError(f'Unsupported {name}: {raw[name]!r}')
        mask = raw.get('mask_non_image_embeddings', False)
        if not isinstance(mask, bool):
            raise ValueError('mask_non_image_embeddings must be boolean')
        return cls(mask_non_image_embeddings=mask)

    @classmethod
    def from_model_dir(cls, directory):
        return cls.from_dict(json.loads((Path(directory) / 'config.json').read_text()))
