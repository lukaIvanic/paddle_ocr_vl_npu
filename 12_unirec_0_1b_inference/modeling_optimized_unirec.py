from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import PreTrainedTokenizerFast

try:
    import torch_npu
except Exception:
    torch_npu = None

LOCAL_UNIREC_STATIC_CACHE_LEN = 256
LOCAL_UNIREC_SELF_ATTN_BACKENDS = {"eager", "increfa"}
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def import_torchair_cache_compile() -> tuple[Any, str]:
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile

        return cache_compile, "torch_npu.dynamo.torchair.inference.cache_compile"
    except Exception as first_exc:
        try:
            from torchair.inference import cache_compile

            return cache_compile, "torchair.inference.cache_compile"
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to import TorchAir cache_compile. Tried "
                "torch_npu.dynamo.torchair.inference.cache_compile first, then "
                "torchair.inference.cache_compile. "
                f"first_error={first_exc!r}; second_error={second_exc!r}"
            ) from second_exc


@dataclass
class UniRecConfig:
    vocab_size: int = 50000
    max_position_embeddings: int = 3072
    decoder_layers: int = 6
    decoder_ffn_dim: int = 1536
    decoder_attention_heads: int = 6
    d_model: int = 384
    decoder_start_token_id: int = 0
    scale_embedding: bool = True
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "UniRecConfig":
        allowed = cls.__dataclass_fields__.keys()
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "UniRecConfig":
        config_path = Path(model_path) / "config.json"
        return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))


class LocalMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LocalConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(inputs)))


class LocalFocalModulation(nn.Module):
    def __init__(self, dim: int, focal_window: int, focal_level: int):
        super().__init__()
        self.focal_level = focal_level

        self.f = nn.Linear(dim, 2 * dim + (focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.focal_layers = nn.ModuleList()

        for level in range(focal_level):
            kernel_size = 2 * level + focal_window
            padding = kernel_size // 2
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=padding,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[-1]
        projected = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(projected, (channels, channels, self.focal_level + 1), 1)

        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level : level + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level :]

        modulator = self.h(ctx_all)
        x_out = q * modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        return self.proj(x_out)


class LocalFocalNetBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, focal_level: int, focal_window: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.modulation = LocalFocalModulation(dim=dim, focal_window=focal_window, focal_level=focal_level)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LocalMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
        )
        self.H = None
        self.W = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.H, self.W
        batch, _, channels = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(batch, h, w, channels)
        x = self.modulation(x).view(batch, h * w, channels)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class LocalPatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class LocalBasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int | None,
        depth: int,
        downsample_kernel: tuple[int, int],
        focal_level: int,
        focal_window: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LocalFocalNetBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    focal_level=focal_level,
                    focal_window=focal_window,
                )
                for _ in range(depth)
            ]
        )
        self.downsample = (
            LocalPatchEmbed(
                patch_size=downsample_kernel,
                in_chans=dim,
                embed_dim=out_dim,
            )
            if out_dim is not None
            else None
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            block.H, block.W = h, w
            x = block(x)

        if self.downsample is None:
            return x, h, w

        x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return self.downsample(x)


class LocalFocalSVTR(nn.Module):
    DEPTHS = (2, 2, 9, 2)
    SUB_K = ((2, 2), (2, 2), (2, 2), (-1, -1))
    FOCAL_LEVELS = (3, 3, 3, 3)
    FOCAL_WINDOWS = (3, 3, 3, 3)

    def __init__(self):
        super().__init__()
        embed_dim = 96
        mlp_ratio = 4.0
        embed_dims = [embed_dim * (2**i) for i in range(len(self.DEPTHS))]

        self.num_features = embed_dims[-1]
        self.patch_embed = nn.Sequential(
            LocalConvBNLayer(
                in_channels=3,
                out_channels=embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            LocalConvBNLayer(
                in_channels=embed_dims[0] // 2,
                out_channels=embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.pos_drop = nn.Identity()

        self.layers = nn.ModuleList()
        for index, depth in enumerate(self.DEPTHS):
            out_dim = embed_dims[index + 1] if index < len(self.DEPTHS) - 1 else None
            self.layers.append(
                LocalBasicLayer(
                    dim=embed_dims[index],
                    out_dim=out_dim,
                    depth=depth,
                    downsample_kernel=self.SUB_K[index],
                    focal_level=self.FOCAL_LEVELS[index],
                    focal_window=self.FOCAL_WINDOWS[index],
                    mlp_ratio=mlp_ratio,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, h, w = layer(x, h, w)
        return x


class LocalUniRecEncoder(nn.Module):
    def __init__(self, config: UniRecConfig):
        super().__init__()
        self.vision_encoder = LocalFocalSVTR()
        self.vision_fc = nn.Linear(config.d_model, config.d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_fc(self.vision_encoder(pixel_values))


@dataclass
class LocalUniRecStaticCache:
    key_cache: tuple[torch.Tensor, ...]
    value_cache: tuple[torch.Tensor, ...]
    cache_len: int
    cross_key_cache: tuple[torch.Tensor, ...] | None = None
    cross_value_cache: tuple[torch.Tensor, ...] | None = None
    cross_attention_mask: torch.Tensor | None = None
    actual_cross_attention_length: int | None = None

    @classmethod
    def from_prefill(
        cls,
        layer_keys: tuple[torch.Tensor, ...],
        layer_values: tuple[torch.Tensor, ...],
        cross_key_cache: tuple[torch.Tensor, ...],
        cross_value_cache: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
        cache_len: int,
        cross_cache_len: int | None = None,
    ) -> "LocalUniRecStaticCache":
        static_keys = []
        static_values = []
        for key_states, value_states in zip(layer_keys, layer_values):
            batch_size, num_heads, prompt_len, head_dim = key_states.shape
            key_cache = key_states.new_zeros((batch_size, num_heads, cache_len, head_dim))
            value_cache = value_states.new_zeros((batch_size, num_heads, cache_len, head_dim))
            key_cache[:, :, :prompt_len, :] = key_states
            value_cache[:, :, :prompt_len, :] = value_states
            static_keys.append(key_cache)
            static_values.append(value_cache)

        actual_cross_attention_length = int(cross_attention_mask.shape[-1])
        if cross_cache_len is None:
            padded_cross_key_cache = tuple(cross_key_cache)
            padded_cross_value_cache = tuple(cross_value_cache)
            padded_cross_attention_mask = cross_attention_mask
        else:
            if cross_cache_len < actual_cross_attention_length:
                raise ValueError(
                    "cross_cache_len must be >= actual cross-attention length, "
                    f"got cross_cache_len={cross_cache_len} actual={actual_cross_attention_length}"
                )

            padded_cross_key_cache = []
            padded_cross_value_cache = []
            for key_states, value_states in zip(cross_key_cache, cross_value_cache):
                batch_size, num_heads, source_len, head_dim = key_states.shape
                padded_keys = key_states.new_zeros((batch_size, num_heads, cross_cache_len, head_dim))
                padded_values = value_states.new_zeros((batch_size, num_heads, cross_cache_len, head_dim))
                padded_keys[:, :, :source_len, :] = key_states
                padded_values[:, :, :source_len, :] = value_states
                padded_cross_key_cache.append(padded_keys)
                padded_cross_value_cache.append(padded_values)

            mask_shape = list(cross_attention_mask.shape)
            mask_shape[-1] = cross_cache_len
            negative_inf = torch.finfo(cross_attention_mask.dtype).min
            padded_cross_attention_mask = cross_attention_mask.new_full(mask_shape, negative_inf)
            padded_cross_attention_mask[..., :actual_cross_attention_length] = cross_attention_mask
        return cls(
            key_cache=tuple(static_keys),
            value_cache=tuple(static_values),
            cache_len=cache_len,
            cross_key_cache=tuple(padded_cross_key_cache),
            cross_value_cache=tuple(padded_cross_value_cache),
            cross_attention_mask=padded_cross_attention_mask,
            actual_cross_attention_length=actual_cross_attention_length,
        )


@dataclass
class UniRecPrefilledItem:
    """One exact B1 prefill, ready to join a fixed decode cohort."""

    prep: dict[str, Any]
    kv_cache: LocalUniRecStaticCache
    generated_ids: torch.Tensor
    next_token: torch.Tensor
    prefill_s: float


class LocalScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class LocalSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.offset = 2
        self.register_buffer(
            "weights",
            self._build_weights(num_positions + self.offset, embedding_dim, padding_idx),
            persistent=False,
        )

    @staticmethod
    def _build_weights(num_embeddings: int, embedding_dim: int, padding_idx: int) -> torch.Tensor:
        half_dim = embedding_dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32)
        exponent = torch.exp(exponent * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        positions = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1)
        embeddings = positions * exponent.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(num_embeddings, 1, dtype=torch.float32)], dim=1)
        embeddings[padding_idx, :] = 0
        return embeddings

    def forward_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        gathered = self.weights.index_select(0, position_ids.reshape(-1))
        return gathered.view(*position_ids.shape, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        max_position = self.padding_idx + 1 + sequence_length + past_key_values_length
        if max_position > self.weights.shape[0]:
            self.weights = self._build_weights(max_position + self.offset, self.embedding_dim, self.padding_idx).to(
                device=self.weights.device,
                dtype=self.weights.dtype,
            )
        position_ids = torch.arange(
            self.padding_idx + 1 + past_key_values_length,
            self.padding_idx + 1 + sequence_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, sequence_length)
        return self.forward_position_ids(position_ids)


class LocalDecoderAttention(nn.Module):
    def __init__(self, config: UniRecConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    @staticmethod
    def apply_linear_3d(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a decoder linear through an explicit rank-2 matrix shape."""
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Decoder linear expects [batch, sequence, hidden], got {tuple(hidden_states.shape)}"
            )
        batch_size, sequence_length, hidden_size = hidden_states.shape
        output = linear(hidden_states.reshape(batch_size * sequence_length, hidden_size))
        return output.reshape(batch_size, sequence_length, linear.out_features)

    def project_states(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, target_length, _ = hidden_states.shape
        source_states = hidden_states if key_value_states is None else key_value_states
        source_length = source_states.shape[1]

        query_states = self.apply_linear_3d(self.q_proj, hidden_states).view(
            batch_size, target_length, self.num_heads, self.head_dim
        )
        key_states = self.apply_linear_3d(self.k_proj, source_states).view(
            batch_size, source_length, self.num_heads, self.head_dim
        )
        value_states = self.apply_linear_3d(self.v_proj, source_states).view(
            batch_size, source_length, self.num_heads, self.head_dim
        )

        return (
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
        )

    def project_query_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, target_length, _ = hidden_states.shape
        query_states = self.apply_linear_3d(self.q_proj, hidden_states).view(
            batch_size, target_length, self.num_heads, self.head_dim
        )
        return query_states.transpose(1, 2).contiguous()

    def project_key_value_states(self, key_value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, source_length, _ = key_value_states.shape
        key_states = self.apply_linear_3d(self.k_proj, key_value_states).view(
            batch_size, source_length, self.num_heads, self.head_dim
        )
        value_states = self.apply_linear_3d(self.v_proj, key_value_states).view(
            batch_size, source_length, self.num_heads, self.head_dim
        )
        return key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()

    def attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, num_heads, target_length, head_dim = query_states.shape
        source_length = key_states.shape[2]
        query_bmm = (query_states * self.scaling).reshape(
            batch_size * num_heads, target_length, head_dim
        )
        key_bmm = key_states.reshape(
            batch_size * num_heads, source_length, head_dim
        )
        attention_scores = torch.bmm(query_bmm, key_bmm.transpose(1, 2)).view(
            batch_size, num_heads, target_length, source_length
        )
        attention_scores = attention_scores.to(dtype=torch.float32) + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1).to(dtype=output_dtype)
        attention_output = torch.bmm(
            attention_probs.reshape(
                batch_size * num_heads, target_length, source_length
            ),
            value_states.reshape(
                batch_size * num_heads, source_length, head_dim
            ),
        ).view(batch_size, num_heads, target_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, target_length, self.embed_dim)
        return self.apply_linear_3d(self.out_proj, attention_output)

    def attend_increfa(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        actual_seq_lengths: list[int],
        kv_padding_size: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, _, target_length, _ = query_states.shape
        if batch_size != 1:
            raise ValueError(f"Local UniRec IncreFA path supports only batch size 1, got {batch_size}")
        if target_length != 1:
            raise ValueError(f"Local UniRec IncreFA path expects q_len == 1, got {target_length}")
        if attention_mask.shape != (batch_size, 1, 1, key_states.shape[2]):
            raise ValueError(
                f"Local UniRec IncreFA path expects attention_mask with shape ({batch_size}, 1, 1, {key_states.shape[2]}), got {tuple(attention_mask.shape)}"
            )
        if len(actual_seq_lengths) != batch_size:
            raise ValueError(
                f"Local UniRec IncreFA path expects actual_seq_lengths with length {batch_size}, got {len(actual_seq_lengths)}"
            )
        if kv_padding_size.numel() != 1:
            raise ValueError(
                f"Local UniRec IncreFA path expects scalar kv_padding_size tensor, got {tuple(kv_padding_size.shape)}"
            )
        if torch_npu is None:
            raise RuntimeError("IncreFA requires torch_npu, but torch_npu is not importable in this environment")

        attention_output = torch_npu.npu_incre_flash_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            atten_mask=attention_mask.to(dtype=torch.bool).contiguous(),
            actual_seq_lengths=[int(v) for v in actual_seq_lengths],
            kv_padding_size=kv_padding_size.to(dtype=torch.int64, device=query_states.device).contiguous(),
            num_heads=int(self.num_heads),
            num_key_value_heads=int(self.num_heads),
            input_layout="BNSD",
            scale_value=float(self.scaling),
        )
        attention_output = attention_output.to(dtype=output_dtype)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, target_length, self.embed_dim)
        return self.apply_linear_3d(self.out_proj, attention_output)

    @staticmethod
    def build_static_cache_attention_mask(
        cache_position: torch.Tensor,
        cache_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = cache_position.shape[0]
        kv_positions = torch.arange(cache_len, device=device, dtype=cache_position.dtype).view(1, 1, 1, cache_len)
        valid_positions = kv_positions <= cache_position.view(batch_size, 1, 1, 1)
        attention_mask = torch.zeros((batch_size, 1, 1, cache_len), device=device, dtype=torch.float32)
        return attention_mask.masked_fill(~valid_positions, torch.finfo(torch.float32).min)

    @staticmethod
    def build_static_cache_attention_mask_bool(
        cache_position: torch.Tensor,
        cache_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = cache_position.shape[0]
        kv_positions = torch.arange(cache_len, device=device, dtype=cache_position.dtype).view(1, 1, 1, cache_len)
        return kv_positions > cache_position.view(batch_size, 1, 1, 1)

    @staticmethod
    def build_increfa_decode_metadata(
        *,
        active_length: int,
        cache_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        if active_length <= 0:
            raise ValueError(f"Local UniRec IncreFA metadata expects active_length > 0, got {active_length}")
        if active_length > cache_len:
            raise ValueError(
                f"Local UniRec IncreFA metadata expects active_length <= cache_len, got active_length={active_length} cache_len={cache_len}"
            )
        cache_position = torch.tensor([active_length - 1], dtype=torch.int64, device=device)
        attention_mask = LocalDecoderAttention.build_static_cache_attention_mask_bool(
            cache_position=cache_position,
            cache_len=cache_len,
            device=device,
        )
        kv_padding_size = torch.tensor(cache_len - active_length, dtype=torch.int64, device=device)
        return attention_mask, [active_length], kv_padding_size

    @staticmethod
    def scatter_update_static_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        # Ascend's Scatter KV-cache tiling contract takes one position per
        # batch row. Fixed cohorts happen to carry the same value in every row,
        # but the B-entry vector must still be preserved.
        updated_positions = (
            cache_position.reshape(-1)
            .to(dtype=torch.int64, device=key_cache.device)
            .contiguous()
        )
        if key_cache.device.type == "npu":
            if torch_npu is None:
                raise RuntimeError("NPU scatter_update_ cache updates require torch_npu")
            torch_npu.scatter_update_(key_cache, updated_positions, key_states.contiguous(), 2)
            torch_npu.scatter_update_(value_cache, updated_positions, value_states.contiguous(), 2)
            return
        if updated_positions.numel() == 1:
            key_cache.index_copy_(2, updated_positions, key_states.contiguous())
            value_cache.index_copy_(2, updated_positions, value_states.contiguous())
            return
        batch_indices = torch.arange(
            key_cache.shape[0],
            dtype=torch.long,
            device=key_cache.device,
        )
        key_cache[batch_indices, :, updated_positions, :] = key_states[:, :, 0, :]
        value_cache[batch_indices, :, updated_positions, :] = value_states[:, :, 0, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.project_states(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
        )
        return self.attend(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_dtype=hidden_states.dtype,
        )


class LocalDecoderLayer(nn.Module):
    def __init__(self, config: UniRecConfig):
        super().__init__()
        self.self_attn = LocalDecoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = LocalDecoderAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def apply_cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return residual + hidden_states

    def project_cross_attention_states(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_attn.project_key_value_states(encoder_hidden_states)

    def apply_cross_attention_with_cache(
        self,
        hidden_states: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        query_states = self.encoder_attn.project_query_states(hidden_states)
        hidden_states = self.encoder_attn.attend(
            query_states=query_states,
            key_states=cross_key_cache,
            value_states=cross_value_cache,
            attention_mask=cross_attention_mask,
            output_dtype=residual.dtype,
        )
        return residual + hidden_states

    def apply_ffn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = LocalDecoderAttention.apply_linear_3d(self.fc1, hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = LocalDecoderAttention.apply_linear_3d(self.fc2, hidden_states)
        return residual + hidden_states

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        query_states, key_states, value_states = self.self_attn.project_states(hidden_states)
        hidden_states = self.self_attn.attend(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=self_attention_mask,
            output_dtype=residual.dtype,
        )
        self_attn_output = residual + hidden_states

        hidden_states = self.apply_cross_attention(
            hidden_states=self_attn_output,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = self.apply_ffn(hidden_states)
        return hidden_states, key_states, value_states

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        active_length: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        self_attention_backend: str = "eager",
    ) -> torch.Tensor:
        if self_attention_backend not in LOCAL_UNIREC_SELF_ATTN_BACKENDS:
            raise ValueError(f"Unsupported Local UniRec self-attention backend: {self_attention_backend}")

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        query_states, key_states, value_states = self.self_attn.project_states(hidden_states)
        self.self_attn.scatter_update_static_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            key_states=key_states,
            value_states=value_states,
        )
        self_attention_mask = self.self_attn.build_static_cache_attention_mask(
            cache_position=cache_position,
            cache_len=key_cache.shape[2],
            device=hidden_states.device,
        )
        if self_attention_backend == "eager":
            hidden_states = self.self_attn.attend(
                query_states=query_states,
                key_states=key_cache,
                value_states=value_cache,
                attention_mask=self_attention_mask,
                output_dtype=residual.dtype,
            )
        else:
            self_attention_mask_bool, actual_seq_lengths, kv_padding_size = self.self_attn.build_increfa_decode_metadata(
                active_length=active_length,
                cache_len=key_cache.shape[2],
                device=hidden_states.device,
            )
            hidden_states = self.self_attn.attend_increfa(
                query_states=query_states,
                key_states=key_cache,
                value_states=value_cache,
                attention_mask=self_attention_mask_bool,
                actual_seq_lengths=actual_seq_lengths,
                kv_padding_size=kv_padding_size,
                output_dtype=residual.dtype,
            )
        hidden_states = residual + hidden_states
        hidden_states = self.apply_cross_attention_with_cache(
            hidden_states=hidden_states,
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            cross_attention_mask=cross_attention_mask,
        )
        return self.apply_ffn(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _, _ = self.forward_prefill(
            hidden_states=hidden_states,
            self_attention_mask=self_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return hidden_states


class LocalUniRecDecoder(nn.Module):
    def __init__(self, config: UniRecConfig):
        super().__init__()
        embed_scale = float(config.d_model**0.5) if config.scale_embedding else 1.0
        self.embed_tokens = LocalScaledWordEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.embed_positions = LocalSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList([LocalDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def build_decoder_attention_mask(decoder_input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, target_length = decoder_input_ids.shape
        negative_inf = torch.finfo(torch.float32).min
        attention_mask = torch.full(
            (target_length, target_length),
            negative_inf,
            dtype=torch.float32,
            device=decoder_input_ids.device,
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)
        return attention_mask.view(1, 1, target_length, target_length).expand(batch_size, 1, target_length, target_length)

    @staticmethod
    def build_cross_attention_mask(encoder_attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
        negative_inf = torch.finfo(torch.float32).min
        expanded = encoder_attention_mask[:, None, None, :].expand(
            encoder_attention_mask.shape[0],
            1,
            target_length,
            encoder_attention_mask.shape[1],
        )
        zeros = torch.zeros_like(expanded, dtype=torch.float32)
        negative = torch.full_like(expanded, negative_inf, dtype=torch.float32)
        return torch.where(expanded.bool(), zeros, negative)

    def build_decoder_input_hidden_states(
        self,
        decoder_input_ids: torch.Tensor,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        return self.embed_tokens(decoder_input_ids) + self.embed_positions(
            decoder_input_ids,
            past_key_values_length=past_key_values_length,
        )

    def build_cached_decoder_input_hidden_states(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = cache_position.reshape(-1, 1).to(dtype=torch.long, device=decoder_input_ids.device)
        position_ids = position_ids + self.embed_positions.padding_idx + 1
        return self.embed_tokens(decoder_input_ids) + self.embed_positions.forward_position_ids(position_ids)

    def build_cross_attention_cache(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        cross_key_cache = []
        cross_value_cache = []
        for layer in self.layers:
            key_states, value_states = layer.project_cross_attention_states(encoder_hidden_states)
            cross_key_cache.append(key_states)
            cross_value_cache.append(value_states)
        return tuple(cross_key_cache), tuple(cross_value_cache)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.build_decoder_input_hidden_states(decoder_input_ids)
        self_attention_mask = self.build_decoder_attention_mask(decoder_input_ids)
        cross_attention_mask = self.build_cross_attention_mask(
            encoder_attention_mask=encoder_attention_mask,
            target_length=decoder_input_ids.shape[1],
        )

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                self_attention_mask=self_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attention_mask,
            )
        return self.layer_norm(hidden_states)

    def prefill_with_cache(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        cache_len: int = LOCAL_UNIREC_STATIC_CACHE_LEN,
        cross_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, LocalUniRecStaticCache]:
        hidden_states = self.build_decoder_input_hidden_states(decoder_input_ids)
        self_attention_mask = self.build_decoder_attention_mask(decoder_input_ids)
        full_cross_attention_mask = self.build_cross_attention_mask(
            encoder_attention_mask=encoder_attention_mask,
            target_length=decoder_input_ids.shape[1],
        )
        decode_cross_attention_mask = self.build_cross_attention_mask(
            encoder_attention_mask=encoder_attention_mask,
            target_length=1,
        )
        cross_key_cache, cross_value_cache = self.build_cross_attention_cache(encoder_hidden_states)

        layer_keys = []
        layer_values = []
        for layer in self.layers:
            hidden_states, key_states, value_states = layer.forward_prefill(
                hidden_states=hidden_states,
                self_attention_mask=self_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=full_cross_attention_mask,
            )
            layer_keys.append(key_states)
            layer_values.append(value_states)

        return self.layer_norm(hidden_states), LocalUniRecStaticCache.from_prefill(
            layer_keys=tuple(layer_keys),
            layer_values=tuple(layer_values),
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            cross_attention_mask=decode_cross_attention_mask,
            cache_len=cache_len,
            cross_cache_len=cross_cache_len,
        )

    def forward_cached(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        active_length: int,
        key_cache: tuple[torch.Tensor, ...],
        value_cache: tuple[torch.Tensor, ...],
        cross_key_cache: tuple[torch.Tensor, ...],
        cross_value_cache: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
        self_attention_backend: str = "eager",
    ) -> torch.Tensor:
        hidden_states = self.build_cached_decoder_input_hidden_states(decoder_input_ids, cache_position)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_cached(
                hidden_states=hidden_states,
                cache_position=cache_position,
                active_length=active_length,
                key_cache=key_cache[layer_idx],
                value_cache=value_cache[layer_idx],
                cross_key_cache=cross_key_cache[layer_idx],
                cross_value_cache=cross_value_cache[layer_idx],
                cross_attention_mask=cross_attention_mask,
                self_attention_backend=self_attention_backend,
            )

        return self.layer_norm(hidden_states)


class LocalUniRecCachedDecodeStepModule(nn.Module):
    def __init__(self, model: "LocalUniRecModel", *, self_attention_backend: str = "eager"):
        super().__init__()
        if self_attention_backend not in LOCAL_UNIREC_SELF_ATTN_BACKENDS:
            raise ValueError(f"Unsupported Local UniRec self-attention backend: {self_attention_backend}")
        self.model = model
        self.self_attention_backend = self_attention_backend

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        active_length: int,
        self_key_cache: tuple[torch.Tensor, ...],
        self_value_cache: tuple[torch.Tensor, ...],
        cross_key_cache: tuple[torch.Tensor, ...],
        cross_value_cache: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_cached_logits(
            decoder_input_ids=decoder_input_ids,
            cache_position=cache_position,
            active_length=active_length,
            key_cache=self_key_cache,
            value_cache=self_value_cache,
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            cross_attention_mask=cross_attention_mask,
            self_attention_backend=self.self_attention_backend,
        )


class LocalUniRecModel(nn.Module):
    def __init__(self, config: UniRecConfig):
        super().__init__()
        self.config = config
        self.encoder = LocalUniRecEncoder(config)
        self.decoder = LocalUniRecDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def decoder_start_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, 1),
            int(self.config.decoder_start_token_id),
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def select_next_token(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits[:, -1, :].to(dtype=torch.float32), dim=-1, keepdim=True).to(dtype=torch.long)

    @staticmethod
    def build_encoder_attention_mask(encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

    def forward_encoder(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)

    def forward(self, pixel_values: torch.Tensor, decoder_input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_hidden_states = self.forward_encoder(pixel_values)
        encoder_attention_mask = self.build_encoder_attention_mask(encoder_hidden_states)
        decoder_output = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = LocalDecoderAttention.apply_linear_3d(self.lm_head, decoder_output)
        return {
            "encoder_last_hidden_state": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_output": decoder_output,
            "logits": logits,
        }

    def prefill_with_cache(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        cross_cache_len: int | None = None,
    ) -> dict[str, object]:
        if pixel_values.shape[0] != 1:
            raise ValueError(
                f"Local UniRec cached decode currently supports only batch size 1, got {pixel_values.shape[0]}"
            )

        encoder_hidden_states = self.forward_encoder(pixel_values)
        encoder_attention_mask = self.build_encoder_attention_mask(encoder_hidden_states)
        decoder_output, kv_cache = self.decoder.prefill_with_cache(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cache_len=LOCAL_UNIREC_STATIC_CACHE_LEN,
            cross_cache_len=cross_cache_len,
        )
        logits = LocalDecoderAttention.apply_linear_3d(self.lm_head, decoder_output)
        return {
            "logits": logits,
            "kv_cache": kv_cache,
        }

    def forward_cached_logits(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        active_length: int,
        key_cache: tuple[torch.Tensor, ...],
        value_cache: tuple[torch.Tensor, ...],
        cross_key_cache: tuple[torch.Tensor, ...],
        cross_value_cache: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
        self_attention_backend: str = "eager",
    ) -> torch.Tensor:
        decoder_output = self.decoder.forward_cached(
            decoder_input_ids=decoder_input_ids,
            cache_position=cache_position,
            active_length=active_length,
            key_cache=key_cache,
            value_cache=value_cache,
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            cross_attention_mask=cross_attention_mask,
            self_attention_backend=self_attention_backend,
        )
        return LocalDecoderAttention.apply_linear_3d(self.lm_head, decoder_output)

    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int,
        decode_step_module: nn.Module | None = None,
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        if batch_size != 1:
            raise ValueError(f"Local UniRec cached decode currently supports only batch size 1, got {batch_size}")
        if max_length > LOCAL_UNIREC_STATIC_CACHE_LEN:
            raise ValueError(
                f"Local UniRec cached decode requires max_length <= {LOCAL_UNIREC_STATIC_CACHE_LEN}, got {max_length}"
            )

        generated_ids = self.decoder_start_ids(batch_size=batch_size, device=pixel_values.device)
        eos_token_id = int(self.config.eos_token_id)

        with torch.inference_mode():
            if int(generated_ids.shape[1]) >= max_length:
                return generated_ids

            prefill_outputs = self.prefill_with_cache(
                pixel_values=pixel_values,
                decoder_input_ids=generated_ids,
            )
            kv_cache = prefill_outputs["kv_cache"]
            if kv_cache.cross_key_cache is None or kv_cache.cross_value_cache is None or kv_cache.cross_attention_mask is None:
                raise RuntimeError("Cached generate requires cross-attention cache to be initialized")

            next_token = self.select_next_token(prefill_outputs["logits"])
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if torch.all(next_token == eos_token_id).item():
                return generated_ids

            cache_position = torch.tensor(
                [generated_ids.shape[1] - 1],
                dtype=torch.int64,
                device=pixel_values.device,
            )
            while int(generated_ids.shape[1]) < max_length:
                if decode_step_module is None:
                    logits = self.forward_cached_logits(
                        decoder_input_ids=next_token,
                        cache_position=cache_position,
                        active_length=int(generated_ids.shape[1]),
                        key_cache=kv_cache.key_cache,
                        value_cache=kv_cache.value_cache,
                        cross_key_cache=kv_cache.cross_key_cache,
                        cross_value_cache=kv_cache.cross_value_cache,
                        cross_attention_mask=kv_cache.cross_attention_mask,
                    )
                else:
                    logits = decode_step_module(
                        next_token,
                        cache_position,
                        int(generated_ids.shape[1]),
                        kv_cache.key_cache,
                        kv_cache.value_cache,
                        kv_cache.cross_key_cache,
                        kv_cache.cross_value_cache,
                        kv_cache.cross_attention_mask,
                    )

                next_token = self.select_next_token(logits)
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                if torch.all(next_token == eos_token_id).item():
                    break
                cache_position = cache_position + 1

        return generated_ids


RULES = [
    (r"-<\|sn\|>", ""),
    (r" <\|sn\|>", " "),
    (r"<\|sn\|>", " "),
    (r"<\|unk\|>", ""),
    (r"<s>", ""),
    (r"</s>", ""),
    (r"\uffff", ""),
    (r"_{4,}", "___"),
    (r"\.{4,}", "..."),
]


def clean_special_tokens(text: str) -> str:
    text = text.replace(" ", "").replace("Ġ", " ").replace("Ċ", "\n")
    text = text.replace("<|bos|>", "").replace("<|eos|>", "").replace("<|pad|>", "")
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    text = text.replace("<tdcolspan=", "<td colspan=")
    text = text.replace("<tdrowspan=", "<td rowspan=")
    text = text.replace('"colspan=', '" colspan=')
    return text


class UniRecImageProcessor:
    def __init__(
        self,
        max_side: list[int] | None = None,
        divided_factor: list[int] | None = None,
        rescale_factor: float = 1 / 255.0,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        resample: int = Image.BICUBIC,
    ) -> None:
        self.max_side = max_side or [64 * 15, 64 * 22]
        self.divided_factor = divided_factor or [64, 64]
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.resample = resample

    def _calculate_target_size(self, original_width: int, original_height: int) -> tuple[int, int]:
        max_width, max_height = self.max_side
        aspect_ratio = original_width / original_height
        if original_width > max_width or original_height > max_height:
            if (max_width / max_height) >= aspect_ratio:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
        else:
            new_width, new_height = original_width, original_height
        div_w, div_h = self.divided_factor
        final_width = max(int(new_width // div_w * div_w), 64)
        final_height = max(int(new_height // div_h * div_h), 64)
        return final_width, final_height

    def get_processed_size(self, original_width: int, original_height: int) -> tuple[int, int]:
        return self._calculate_target_size(original_width, original_height)

    @staticmethod
    def _estimate_encoder_axis_tokens(axis_size: int) -> int:
        token_axis = int(axis_size)
        token_axis = (token_axis + 1) // 2
        token_axis = (token_axis + 1) // 2
        token_axis = token_axis // 2
        token_axis = token_axis // 2
        token_axis = token_axis // 2
        if token_axis <= 0:
            raise ValueError(f"Input image axis is too small after downsampling: {axis_size}")
        return token_axis

    def estimate_encoder_token_count_from_processed_size(
        self,
        *,
        processed_width: int,
        processed_height: int,
    ) -> int:
        return int(
            self._estimate_encoder_axis_tokens(processed_height)
            * self._estimate_encoder_axis_tokens(processed_width)
        )

    def estimate_encoder_token_count_for_image_size(self, original_width: int, original_height: int) -> int:
        processed_width, processed_height = self.get_processed_size(original_width, original_height)
        return self.estimate_encoder_token_count_from_processed_size(
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        image = image.convert("RGB")
        original_width, original_height = image.size
        target_size = self.get_processed_size(original_width, original_height)
        image = image.resize(target_size, resample=self.resample)
        array = np.asarray(image, dtype=np.float32)
        if self.do_rescale:
            array *= self.rescale_factor
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        return {"pixel_values": torch.from_numpy(array).unsqueeze(0)}


def synchronize_device(device: str | torch.device | None = None) -> None:
    device_text = "" if device is None else str(device)
    if device_text.startswith("cuda"):
        torch.cuda.synchronize()
        return
    if device_text.startswith("npu"):
        if torch_npu is None:
            raise RuntimeError("Cannot synchronize NPU because torch_npu is not importable")
        torch.npu.synchronize()


def build_tokenizer(model_path: Path) -> PreTrainedTokenizerFast:
    tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token=tokenizer_config.get("bos_token", "<s>"),
        eos_token=tokenizer_config.get("eos_token", "</s>"),
        pad_token=tokenizer_config.get("pad_token", "<pad>"),
        unk_token=tokenizer_config.get("unk_token", "<unk>"),
        clean_up_tokenization_spaces=tokenizer_config.get("clean_up_tokenization_spaces", False),
        model_max_length=tokenizer_config.get("model_max_length", 2048),
    )


class OptimizedUniRecRunner:
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str,
        dtype: str,
        compile_cache_dir: str | Path | None = None,
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = str(device)
        self.dtype_name = dtype
        self.dtype = DTYPE_MAP[dtype]
        self.config = UniRecConfig.from_pretrained(self.model_path)
        self.tokenizer = build_tokenizer(self.model_path)
        self.processor = UniRecImageProcessor()
        self._compiled_decode_modules: dict[str, nn.Module] = {}
        self._compiled_decode_metadata: dict[str, dict[str, Any]] = {}
        self._static_cross_cache_len_by_processor_max_side: dict[tuple[int, int], int] = {}
        self.compile_cache_dir = Path(compile_cache_dir).expanduser().resolve() if compile_cache_dir else None

        load_start = time.perf_counter()
        if self.device.startswith("npu"):
            if torch_npu is None:
                raise RuntimeError("NPU execution requires torch_npu")
            torch_npu.npu.config.allow_internal_format = False
        self.model = self._load_model().to(self.device)
        self.model.eval()
        synchronize_device(self.device)
        self.model_load_s = time.perf_counter() - load_start

    def _load_model(self) -> LocalUniRecModel:
        model = LocalUniRecModel(self.config).to(dtype=self.dtype)
        safetensors_path = self.model_path / "model.safetensors"
        pytorch_path = self.model_path / "model.pth"
        if safetensors_path.is_file():
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError(
                    "Loading model.safetensors requires the safetensors package"
                ) from exc
            state_dict = load_file(str(safetensors_path), device="cpu")
        elif pytorch_path.is_file():
            checkpoint = torch.load(pytorch_path, map_location=torch.device("cpu"))
            state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.pth checkpoint found under {self.model_path}"
            )
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized = key[len("module.") :] if key.startswith("module.") else key
            if normalized.startswith("model.encoder."):
                normalized = "encoder." + normalized[len("model.encoder.") :]
            elif normalized.startswith("model.decoder."):
                normalized = "decoder." + normalized[len("model.decoder.") :]
            elif normalized.startswith("model.lm_head."):
                normalized = "lm_head." + normalized[len("model.lm_head.") :]
            elif normalized == "model.shared.weight":
                remapped["decoder.embed_tokens.weight"] = value
                remapped["lm_head.weight"] = value
                continue
            elif normalized.startswith(("encoder.", "decoder.", "lm_head.")):
                pass
            else:
                continue
            remapped[normalized] = value
        missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Local UniRec checkpoint load mismatch after remap: "
                f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
            )
        return model

    def _prepare_pil_image(
        self,
        image: Image.Image,
        *,
        image_source: str,
        image_load_s: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        image = image.convert("RGB")
        t0 = time.perf_counter()
        processed_width, processed_height = self.processor.get_processed_size(image.width, image.height)
        encoder_seq_len_hint = self.processor.estimate_encoder_token_count_for_image_size(image.width, image.height)
        inputs = self.processor(image)
        t1 = time.perf_counter()
        pixel_values = inputs["pixel_values"].to(self.device).to(dtype=self.dtype)
        synchronize_device(self.device)
        t2 = time.perf_counter()
        prepare_total_s = float(image_load_s) + (t2 - t0)
        return {"pixel_values": pixel_values}, {
            "image": image_source,
            "original_image_size": [int(image.width), int(image.height)],
            "processed_image_size": [int(processed_width), int(processed_height)],
            "encoder_seq_len_hint": int(encoder_seq_len_hint),
            "pixel_values_shape": list(pixel_values.shape),
            "image_load_s": float(image_load_s),
            "image_preprocess_s": t1 - t0,
            "move_to_device_s": t2 - t1,
            "prepare_total_s": prepare_total_s,
        }

    def prepare_image(self, image_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        synchronize_device(self.device)
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        t1 = time.perf_counter()
        return self._prepare_pil_image(
            image,
            image_source=str(Path(image_path).resolve()),
            image_load_s=t1 - t0,
        )

    def prepare_pil_image(
        self,
        image: Image.Image,
        *,
        image_source: str = "<pil_image>",
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Prepare an in-memory crop without a file encode/decode round trip."""
        synchronize_device(self.device)
        return self._prepare_pil_image(
            image,
            image_source=image_source,
            image_load_s=0.0,
        )

    def _get_static_cross_cache_len(self) -> int:
        processor_max_side = tuple(int(value) for value in self.processor.max_side)
        cached = self._static_cross_cache_len_by_processor_max_side.get(processor_max_side)
        if cached is not None:
            return cached
        max_width, max_height = processor_max_side
        dummy_pixel_values = torch.zeros((1, 3, max_height, max_width), dtype=self.dtype, device=self.device)
        with torch.inference_mode():
            encoder_hidden_states = self.model.forward_encoder(dummy_pixel_values)
        static_cross_cache_len = int(encoder_hidden_states.shape[1])
        self._static_cross_cache_len_by_processor_max_side[processor_max_side] = static_cross_cache_len
        return static_cross_cache_len

    def _compile_decode_module(
        self,
        *,
        backend: str,
        self_attention_backend: str,
        compile_dynamic: bool,
        cross_cache_len: int | None,
        batch_size: int,
    ) -> tuple[nn.Module, dict[str, Any]]:
        normalized_cross_cache_len = "none" if cross_cache_len is None else str(int(cross_cache_len))
        cache_key = (
            f"{backend}:self_attn={self_attention_backend}:dynamic={int(compile_dynamic)}:"
            f"self_kv={LOCAL_UNIREC_STATIC_CACHE_LEN}:cross_kv={normalized_cross_cache_len}:"
            f"batch={int(batch_size)}"
        )
        cached = self._compiled_decode_modules.get(cache_key)
        if cached is not None:
            return cached, self._compiled_decode_metadata[cache_key]
        module = LocalUniRecCachedDecodeStepModule(self.model, self_attention_backend=self_attention_backend)
        if backend == "torchair":
            if not self.device.startswith("npu"):
                raise ValueError("backend=torchair requires an NPU device")
            if torch_npu is None:
                raise RuntimeError("backend=torchair requires torch_npu")
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

            cfg = CompilerConfig()
            cfg.mode.value = "max-autotune"
            if self.compile_cache_dir is None:
                compiled = torch.compile(
                    module,
                    backend=torchair.get_npu_backend(compiler_config=cfg),
                    dynamic=compile_dynamic,
                    fullgraph=True,
                )
                compile_meta = {
                    "compile_api": "torch.compile",
                    "backend": "torchair",
                    "fullgraph": True,
                    "dynamic": bool(compile_dynamic),
                    "torchair_ge_cache": False,
                    "torchair_cache_dir": None,
                }
            else:
                shape_name = (
                    f"decode_selfkv{LOCAL_UNIREC_STATIC_CACHE_LEN}_cross"
                    f"{normalized_cross_cache_len}_{self_attention_backend}"
                )
                if batch_size != 1:
                    shape_name += f"_b{int(batch_size)}"
                shape_cache_dir = self.compile_cache_dir / shape_name
                shape_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_compile, cache_compile_import_path = import_torchair_cache_compile()
                compiled = cache_compile(
                    module.forward,
                    config=cfg,
                    dynamic=compile_dynamic,
                    cache_dir=str(shape_cache_dir),
                    ge_cache=True,
                    fullgraph=True,
                )
                compile_meta = {
                    "compile_api": cache_compile_import_path,
                    "compile_api_import_path": cache_compile_import_path,
                    "backend": "torchair",
                    "fullgraph": True,
                    "dynamic": bool(compile_dynamic),
                    "torchair_ge_cache": True,
                    "torchair_cache_dir": str(shape_cache_dir),
                }
        elif backend == "inductor":
            compiled = torch.compile(module, dynamic=compile_dynamic, fullgraph=True)
            compile_meta = {
                "compile_api": "torch.compile",
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": bool(compile_dynamic),
            }
        else:
            raise ValueError(f"Unsupported optimized compile backend: {backend}")
        compile_meta.update(
            {
                "static_self_kv_len": int(LOCAL_UNIREC_STATIC_CACHE_LEN),
                "static_cross_kv_len": None if cross_cache_len is None else int(cross_cache_len),
                "self_attention_backend": self_attention_backend,
                "batch_size": int(batch_size),
            }
        )
        self._compiled_decode_modules[cache_key] = compiled
        self._compiled_decode_metadata[cache_key] = compile_meta
        return compiled, compile_meta

    def _decode_text(self, generated_ids: torch.Tensor) -> str:
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return clean_special_tokens(decoded[0] if decoded else "")

    def _decode_text_batch(self, generated_ids: list[list[int]]) -> list[str]:
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )
        return [clean_special_tokens(text) for text in decoded]

    def generate_one(
        self,
        image_path: str | Path,
        *,
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool = False,
    ) -> dict[str, Any]:
        inputs, prep = self.prepare_image(image_path)
        return self._generate_prepared(
            pixel_values=inputs["pixel_values"],
            prep=prep,
            max_length=max_length,
            decode_mode=decode_mode,
            compile_backend=compile_backend,
            compile_dynamic=compile_dynamic,
        )

    def generate_image(
        self,
        image: Image.Image,
        *,
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool = False,
        image_source: str = "<pil_image>",
    ) -> dict[str, Any]:
        """Generate from an already-cropped PIL image using the normal model path."""
        inputs, prep = self.prepare_pil_image(image, image_source=image_source)
        return self._generate_prepared(
            pixel_values=inputs["pixel_values"],
            prep=prep,
            max_length=max_length,
            decode_mode=decode_mode,
            compile_backend=compile_backend,
            compile_dynamic=compile_dynamic,
        )

    def prefill_image_for_cohort(
        self,
        image: Image.Image,
        *,
        image_source: str = "<pil_image>",
    ) -> UniRecPrefilledItem:
        """Run the exact B1 image and decoder prefill used by normal generation."""
        inputs, prep = self.prepare_pil_image(image, image_source=image_source)
        pixel_values = inputs["pixel_values"]
        cross_cache_len = self._get_static_cross_cache_len()
        decoder_input_ids = self.model.decoder_start_ids(
            batch_size=1,
            device=pixel_values.device,
        )
        with torch.inference_mode():
            synchronize_device(self.device)
            started = time.perf_counter()
            prefill_outputs = self.model.prefill_with_cache(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                cross_cache_len=cross_cache_len,
            )
            next_token = self.model.select_next_token(prefill_outputs["logits"])
            generated_ids = torch.cat((decoder_input_ids, next_token), dim=1)
            synchronize_device(self.device)
            prefill_s = time.perf_counter() - started
        return UniRecPrefilledItem(
            prep=prep,
            kv_cache=prefill_outputs["kv_cache"],
            generated_ids=generated_ids,
            next_token=next_token,
            prefill_s=prefill_s,
        )

    @staticmethod
    def _stack_prefilled_caches(
        items: list[UniRecPrefilledItem],
    ) -> LocalUniRecStaticCache:
        if not items:
            raise ValueError("Cannot stack an empty UniRec prefill cohort")
        caches = [item.kv_cache for item in items]
        if any(
            cache.cross_key_cache is None
            or cache.cross_value_cache is None
            or cache.cross_attention_mask is None
            for cache in caches
        ):
            raise RuntimeError("Every cohort item must have a static cross-attention cache")
        layer_count = len(caches[0].key_cache)
        if any(len(cache.key_cache) != layer_count for cache in caches):
            raise ValueError("All cohort items must have the same decoder layer count")
        return LocalUniRecStaticCache(
            key_cache=tuple(
                torch.cat([cache.key_cache[layer] for cache in caches], dim=0)
                for layer in range(layer_count)
            ),
            value_cache=tuple(
                torch.cat([cache.value_cache[layer] for cache in caches], dim=0)
                for layer in range(layer_count)
            ),
            cache_len=caches[0].cache_len,
            cross_key_cache=tuple(
                torch.cat(
                    [cache.cross_key_cache[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layer_count)
            ),
            cross_value_cache=tuple(
                torch.cat(
                    [cache.cross_value_cache[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layer_count)
            ),
            cross_attention_mask=torch.cat(
                [cache.cross_attention_mask for cache in caches],
                dim=0,
            ),
            actual_cross_attention_length=None,
        )

    def generate_prefilled_cohort(
        self,
        items: list[UniRecPrefilledItem],
        *,
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool = False,
        pad_to_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Decode a fixed cohort; finished and dummy rows emit EOS padding."""
        if not items:
            raise ValueError("Cannot decode an empty UniRec cohort")
        if max_length > LOCAL_UNIREC_STATIC_CACHE_LEN:
            raise ValueError(
                f"max_length must be <= {LOCAL_UNIREC_STATIC_CACHE_LEN}, got {max_length}"
            )
        if decode_mode not in {"eager", "compiled", "compiled_ifa"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        real_batch_size = len(items)
        target_batch_size = (
            real_batch_size if pad_to_batch_size is None else int(pad_to_batch_size)
        )
        if target_batch_size < real_batch_size:
            raise ValueError(
                "pad_to_batch_size must be >= the number of real items, "
                f"got target={target_batch_size} real={real_batch_size}"
            )
        padded_items = list(items)
        while len(padded_items) < target_batch_size:
            padded_items.append(items[-1])

        batch_size = len(padded_items)
        self_attention_backend = "increfa" if decode_mode == "compiled_ifa" else "eager"
        decode_module = None
        compile_wrap_s = None
        compile_meta = None
        cross_cache_len = int(
            padded_items[0].kv_cache.cross_attention_mask.shape[-1]
        )
        if decode_mode.startswith("compiled"):
            compile_started = time.perf_counter()
            decode_module, compile_meta = self._compile_decode_module(
                backend=compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=compile_dynamic,
                cross_cache_len=cross_cache_len,
                batch_size=batch_size,
            )
            compile_wrap_s = time.perf_counter() - compile_started

        cache = self._stack_prefilled_caches(padded_items)
        generated_ids = torch.cat(
            [item.generated_ids for item in padded_items],
            dim=0,
        )
        next_token = torch.cat([item.next_token for item in padded_items], dim=0)
        eos_token_id = int(self.config.eos_token_id)
        finished = next_token.eq(eos_token_id)
        if target_batch_size > real_batch_size:
            finished[real_batch_size:] = True
            next_token[real_batch_size:] = eos_token_id
        cache_position = torch.full(
            (batch_size,),
            int(generated_ids.shape[1] - 1),
            dtype=torch.int64,
            device=next_token.device,
        )

        raw_decode_iterations = 0
        with torch.inference_mode():
            synchronize_device(self.device)
            decode_started = time.perf_counter()
            all_finished = bool(torch.all(finished).item())
            while int(generated_ids.shape[1]) < max_length and not all_finished:
                active_length = int(generated_ids.shape[1])
                if decode_module is None:
                    logits = self.model.forward_cached_logits(
                        decoder_input_ids=next_token,
                        cache_position=cache_position,
                        active_length=active_length,
                        key_cache=cache.key_cache,
                        value_cache=cache.value_cache,
                        cross_key_cache=cache.cross_key_cache,
                        cross_value_cache=cache.cross_value_cache,
                        cross_attention_mask=cache.cross_attention_mask,
                        self_attention_backend="eager",
                    )
                else:
                    logits = decode_module(
                        next_token,
                        cache_position,
                        active_length,
                        cache.key_cache,
                        cache.value_cache,
                        cache.cross_key_cache,
                        cache.cross_value_cache,
                        cache.cross_attention_mask,
                    )
                predicted = self.model.select_next_token(logits)
                predicted = torch.where(
                    finished,
                    torch.full_like(predicted, eos_token_id),
                    predicted,
                )
                generated_ids = torch.cat((generated_ids, predicted), dim=1)
                raw_decode_iterations += 1
                finished = torch.logical_or(finished, predicted.eq(eos_token_id))
                next_token = predicted
                cache_position = cache_position + 1
                all_finished = bool(torch.all(finished).item())
            synchronize_device(self.device)
            decode_s = time.perf_counter() - decode_started

        rows = generated_ids[:real_batch_size].detach().cpu().tolist()
        trimmed_rows: list[list[int]] = []
        for row in rows:
            try:
                eos_index = row.index(eos_token_id, 1)
            except ValueError:
                trimmed_rows.append([int(token) for token in row])
            else:
                trimmed_rows.append([int(token) for token in row[: eos_index + 1]])
        texts = self._decode_text_batch(trimmed_rows)
        effective_decode_tokens = sum(max(0, len(row) - 2) for row in trimmed_rows)
        raw_decode_token_slots = raw_decode_iterations * batch_size
        results = []
        for item, token_ids, text in zip(items, trimmed_rows, texts):
            decode_tokens = max(0, len(token_ids) - 2)
            results.append(
                {
                    "image": str(item.prep["image"]),
                    "text": text,
                    "generated_ids": token_ids,
                    "generated_token_count": max(0, len(token_ids) - 1),
                    "prefill_generated_token_count": 1 if len(token_ids) > 1 else 0,
                    "decode_generated_token_count": decode_tokens,
                    "ttft_s": item.prefill_s,
                    "decode_s": decode_s,
                    "total_latency_s": (
                        float(item.prep["prepare_total_s"])
                        + item.prefill_s
                        + decode_s
                    ),
                    "decode_tokens_per_s": (
                        float(decode_tokens) / decode_s if decode_s > 0 else None
                    ),
                    "compile_wrap_s": compile_wrap_s,
                    "compile": compile_meta,
                    "cross_cache_len": cross_cache_len,
                    "static_self_kv_len": int(LOCAL_UNIREC_STATIC_CACHE_LEN),
                    "device": self.device,
                    "dtype": self.dtype_name,
                    "decode_mode": decode_mode,
                    "compile_backend": (
                        compile_backend if decode_mode.startswith("compiled") else None
                    ),
                    "prep": item.prep,
                }
            )
        return {
            "items": results,
            "batch_size": batch_size,
            "real_batch_size": real_batch_size,
            "dummy_batch_size": batch_size - real_batch_size,
            "decode_iterations": raw_decode_iterations,
            "raw_decode_token_slots": raw_decode_token_slots,
            "effective_decode_tokens": effective_decode_tokens,
            "padding_decode_token_slots": (
                raw_decode_token_slots - effective_decode_tokens
            ),
            "decode_s": decode_s,
            "raw_decode_tokens_per_s": (
                float(raw_decode_token_slots) / decode_s if decode_s > 0 else None
            ),
            "effective_decode_tokens_per_s": (
                float(effective_decode_tokens) / decode_s if decode_s > 0 else None
            ),
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
        }

    def _generate_prepared(
        self,
        *,
        pixel_values: torch.Tensor,
        prep: dict[str, Any],
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool,
    ) -> dict[str, Any]:
        if max_length > LOCAL_UNIREC_STATIC_CACHE_LEN:
            raise ValueError(f"max_length must be <= {LOCAL_UNIREC_STATIC_CACHE_LEN}, got {max_length}")
        if decode_mode not in {"eager", "compiled", "compiled_ifa"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")

        batch_size = int(pixel_values.shape[0])
        decoder_input_ids = self.model.decoder_start_ids(batch_size=batch_size, device=pixel_values.device)
        generated_ids = decoder_input_ids
        eos_token_id = int(self.config.eos_token_id)
        self_attention_backend = "increfa" if decode_mode == "compiled_ifa" else "eager"
        decode_module = None
        compile_wrap_s = None
        compile_meta = None
        cross_cache_len = self._get_static_cross_cache_len() if decode_mode.startswith("compiled") else None
        if decode_mode.startswith("compiled"):
            compile_start = time.perf_counter()
            decode_module, compile_meta = self._compile_decode_module(
                backend=compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=compile_dynamic,
                cross_cache_len=cross_cache_len,
                batch_size=batch_size,
            )
            compile_wrap_s = time.perf_counter() - compile_start

        with torch.inference_mode():
            synchronize_device(self.device)
            prefill_start = time.perf_counter()
            prefill_outputs = self.model.prefill_with_cache(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                cross_cache_len=cross_cache_len,
            )
            kv_cache = prefill_outputs["kv_cache"]
            next_token = self.model.select_next_token(prefill_outputs["logits"])
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            synchronize_device(self.device)
            prefill_end = time.perf_counter()

            finished = bool(torch.all(next_token == eos_token_id).item())
            cache_position = torch.tensor([generated_ids.shape[1] - 1], dtype=torch.int64, device=pixel_values.device)
            decode_generated_token_count = 0
            synchronize_device(self.device)
            decode_start = time.perf_counter()
            while int(generated_ids.shape[1]) < max_length and not finished:
                active_length = int(generated_ids.shape[1])
                if decode_module is None:
                    logits = self.model.forward_cached_logits(
                        decoder_input_ids=next_token,
                        cache_position=cache_position,
                        active_length=active_length,
                        key_cache=kv_cache.key_cache,
                        value_cache=kv_cache.value_cache,
                        cross_key_cache=kv_cache.cross_key_cache,
                        cross_value_cache=kv_cache.cross_value_cache,
                        cross_attention_mask=kv_cache.cross_attention_mask,
                        self_attention_backend="eager",
                    )
                else:
                    logits = decode_module(
                        next_token,
                        cache_position,
                        active_length,
                        kv_cache.key_cache,
                        kv_cache.value_cache,
                        kv_cache.cross_key_cache,
                        kv_cache.cross_value_cache,
                        kv_cache.cross_attention_mask,
                    )
                next_token = self.model.select_next_token(logits)
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                decode_generated_token_count += 1
                if torch.all(next_token == eos_token_id).item():
                    finished = True
                else:
                    cache_position = cache_position + 1
            synchronize_device(self.device)
            decode_end = time.perf_counter()

        text = self._decode_text(generated_ids)
        prefill_s = prefill_end - prefill_start
        decode_s = decode_end - decode_start
        generated_token_count = max(0, int(generated_ids.shape[1]) - 1)
        total_latency_s = float(prep["prepare_total_s"]) + prefill_s + decode_s
        return {
            "image": str(prep["image"]),
            "text": text,
            "generated_ids": generated_ids.detach().cpu().tolist()[0],
            "generated_token_count": generated_token_count,
            "prefill_generated_token_count": 1 if generated_token_count > 0 else 0,
            "decode_generated_token_count": int(decode_generated_token_count),
            "ttft_s": prefill_s,
            "decode_s": decode_s,
            "total_latency_s": total_latency_s,
            "decode_tokens_per_s": (
                float(decode_generated_token_count) / decode_s if decode_s > 0 and decode_generated_token_count > 0 else None
            ),
            "overall_tokens_per_s": float(generated_token_count) / total_latency_s if total_latency_s > 0 else None,
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
            "cross_cache_len": cross_cache_len,
            "static_self_kv_len": int(LOCAL_UNIREC_STATIC_CACHE_LEN),
            "device": self.device,
            "dtype": self.dtype_name,
            "decode_mode": decode_mode,
            "compile_backend": compile_backend if decode_mode.startswith("compiled") else None,
            "prep": prep,
        }
