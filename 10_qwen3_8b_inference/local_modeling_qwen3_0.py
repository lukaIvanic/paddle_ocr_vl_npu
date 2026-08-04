#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

try:
    import torch_npu
except Exception:
    pass


@dataclass(frozen=True)
class LocalQwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LocalQwen3Config":
        with open(path) as handle:
            raw = json.load(handle)
        return cls.from_hf_config(raw)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "LocalQwen3Config":
        return cls.from_json_file(Path(model_dir) / "config.json")

    @classmethod
    def from_hf_config(cls, raw: dict) -> "LocalQwen3Config":
        hidden_size = int(raw["hidden_size"])
        num_attention_heads = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim") or hidden_size // num_attention_heads)
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1000000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 40960)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )


class LocalQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LocalQwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: LocalQwen3Config):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)[None, :, None].float()
        position_ids = position_ids.to(device=device, dtype=torch.float32)[:, None, :]
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states, key_states


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, repeats, sequence_length, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * repeats, sequence_length, head_dim)


def scatter_update_tensor_(
    cache: torch.Tensor,
    cache_position: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    positions = cache_position.reshape(-1).to(dtype=torch.int64, device=cache.device).contiguous()
    torch_npu.scatter_update_(cache, positions, updates.contiguous(), 2)


def build_static_decode_mask(
    cache_position: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(cache_length, device=cache_position.device, dtype=torch.int64)
    return kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)


class LocalQwen3StaticCache:
    def __init__(
        self,
        config: LocalQwen3Config,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key_caches = []
        value_caches = []
        for _layer_idx in range(config.num_hidden_layers):
            key_cache = torch.zeros(
                (batch_size, config.num_key_value_heads, cache_length, config.head_dim),
                device=device,
                dtype=dtype,
            )
            key_caches.append(key_cache)
            value_caches.append(torch.zeros_like(key_cache))
        self.key_caches = tuple(key_caches)
        self.value_caches = tuple(value_caches)

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[layer_idx], self.value_caches[layer_idx]


class LocalQwen3MLP(nn.Module):
    def __init__(self, config: LocalQwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LocalQwen3Attention(nn.Module):
    def __init__(self, config: LocalQwen3Config, *, decode_increfa_mode: str):
        super().__init__()
        if decode_increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported decode_increfa_mode={decode_increfa_mode!r}")
        self.decode_increfa_mode = decode_increfa_mode
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = LocalQwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = LocalQwen3RMSNorm(config.head_dim, config.rms_norm_eps)

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, _hidden = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch, sequence_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch, sequence_length, self.num_key_value_heads, self.head_dim)
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        return query_states, key_states, value_states

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_cache[:, :, : key_states.shape[2], :].copy_(key_states)
        value_cache[:, :, : value_states.shape[2], :].copy_(value_states)

        full_key_states = repeat_kv(key_states, self.num_key_value_groups)
        full_value_states = repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, full_key_states.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(probs, full_value_states)
        batch, _heads, sequence_length, _dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).reshape(batch, sequence_length, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        actual_seq_length: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        scatter_update_tensor_(key_cache, cache_position, key_states)
        scatter_update_tensor_(value_cache, cache_position, value_states)

        if self.decode_increfa_mode == "actual_seq_lengths":
            if actual_seq_length is None:
                raise ValueError("decode_increfa_mode='actual_seq_lengths' requires actual_seq_length")
            increfa_mask = None
            actual_seq_lengths = [actual_seq_length]
        else:
            increfa_mask = attention_mask
            actual_seq_lengths = None
        attn_output = torch_npu.npu_incre_flash_attention(
            query_states.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            atten_mask=None if increfa_mask is None else increfa_mask.contiguous(),
            actual_seq_lengths=actual_seq_lengths,
            num_heads=int(self.num_heads),
            num_key_value_heads=int(self.num_key_value_heads),
            input_layout="BNSD",
            scale_value=float(self.scaling),
        ).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(hidden_states.shape[0], 1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), key_cache, value_cache


class LocalQwen3DecoderLayer(nn.Module):
    def __init__(self, config: LocalQwen3Config, *, decode_increfa_mode: str):
        super().__init__()
        self.self_attn = LocalQwen3Attention(config, decode_increfa_mode=decode_increfa_mode)
        self.mlp = LocalQwen3MLP(config)
        self.input_layernorm = LocalQwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LocalQwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_output = self.self_attn.forward_prefill(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
            key_cache,
            value_cache,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        actual_seq_length: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, key_cache, value_cache = self.self_attn.forward_decode(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            key_cache,
            value_cache,
            cache_position,
            attention_mask,
            actual_seq_length,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key_cache, value_cache


class LocalQwen3ForCausalLM(nn.Module):
    def __init__(self, config: LocalQwen3Config, *, decode_increfa_mode: str = "mask"):
        super().__init__()
        if decode_increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported decode_increfa_mode={decode_increfa_mode!r}")
        self.config = config
        self.decode_increfa_mode = decode_increfa_mode
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LocalQwen3DecoderLayer(config, decode_increfa_mode=decode_increfa_mode)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LocalQwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = LocalQwen3RotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def make_causal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, sequence_length = input_ids.shape
        mask = torch.full(
            (sequence_length, sequence_length),
            torch.finfo(self.embed_tokens.weight.dtype).min,
            dtype=self.embed_tokens.weight.dtype,
            device=input_ids.device,
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.reshape(1, 1, sequence_length, sequence_length).expand(batch, 1, sequence_length, sequence_length)

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        static_kv_cache_len: int,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if input_ids.shape[1] > static_kv_cache_len:
            raise ValueError(f"prompt length {input_ids.shape[1]} exceeds static_kv_cache_len={static_kv_cache_len}")
        cache = LocalQwen3StaticCache(
            self.config,
            batch_size=input_ids.shape[0],
            cache_length=static_kv_cache_len,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).view(1, -1)
        cos, sin = self.rotary_emb(position_ids, dtype=hidden_states.dtype, device=hidden_states.device)
        attention_mask = self.make_causal_mask(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            layer_key_cache, layer_value_cache = cache.layer(layer_idx)
            hidden_states = layer.forward_prefill(
                hidden_states,
                cos,
                sin,
                attention_mask,
                layer_key_cache,
                layer_value_cache,
            )
        return cache.key_caches, cache.value_caches

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        actual_seq_length: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids, dtype=hidden_states.dtype, device=hidden_states.device)
        attention_mask = None
        if self.decode_increfa_mode == "mask":
            attention_mask = build_static_decode_mask(cache_position, key_caches[0].shape[2])
        next_key_caches = []
        next_value_caches = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, layer_key_cache, layer_value_cache = layer.forward_decode(
                hidden_states,
                cos,
                sin,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                attention_mask,
                actual_seq_length,
            )
            next_key_caches.append(layer_key_cache)
            next_value_caches.append(layer_value_cache)
        logits = self.lm_head(self.norm(hidden_states))
        return logits, tuple(next_key_caches), tuple(next_value_caches)

    def decode(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        actual_seq_length: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        position_ids = cache_position.view(1, 1)
        logits, key_caches, value_caches = self.forward_decode(
            input_ids,
            position_ids,
            cache_position,
            key_caches,
            value_caches,
            actual_seq_length,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_id, key_caches, value_caches
