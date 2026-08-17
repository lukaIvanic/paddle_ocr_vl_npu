#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    import torch_npu
except ModuleNotFoundError:
    torch_npu = None


@dataclass(frozen=True)
class Qwen3TPConfig:
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
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3TPConfig":
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        hidden_size = int(raw["hidden_size"])
        num_attention_heads = int(raw["num_attention_heads"])
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(
                raw.get("head_dim") or hidden_size // num_attention_heads
            ),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 40_960)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )

    def with_num_hidden_layers(self, num_hidden_layers: int) -> "Qwen3TPConfig":
        if not 1 <= num_hidden_layers <= self.num_hidden_layers:
            raise ValueError(
                f"num_hidden_layers must be in [1, {self.num_hidden_layers}], "
                f"got {num_hidden_layers}"
            )
        return replace(self, num_hidden_layers=int(num_hidden_layers))

    def validate_tp(self, tp_size: int) -> None:
        for name, size in (
            ("vocab_size", self.vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("intermediate_size", self.intermediate_size),
        ):
            if size % tp_size:
                raise ValueError(f"{name}={size} is not divisible by tp_size={tp_size}")


def shard_bounds(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size % world_size:
        raise ValueError(f"size={size} is not divisible by world_size={world_size}")
    width = size // world_size
    return rank * width, (rank + 1) * width


def linear_tokenwise(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    leading_shape = hidden_states.shape[:-1]
    output = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def tp_all_reduce_sum(tensor: torch.Tensor, tp_size: int) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def tp_local_argmax(
    local_logits: torch.Tensor,
    *,
    vocab_start: int,
    tp_size: int,
) -> torch.Tensor:
    local_values, local_indices = local_logits.max(dim=-1)
    global_indices = local_indices + int(vocab_start)
    if tp_size == 1:
        return global_indices

    # This is the same communication shape used by vLLM's local-argmax path:
    # gather one (value, global token id) pair per rank, not the full vocabulary.
    local_pair = torch.stack(
        (local_values.float(), global_indices.float()), dim=-1
    ).reshape(-1, 2).contiguous()
    gathered = torch.empty(
        (tp_size * local_pair.shape[0], 2),
        device=local_pair.device,
        dtype=local_pair.dtype,
    )
    dist.all_gather_into_tensor(gathered, local_pair)
    gathered = gathered.view(tp_size, local_pair.shape[0], 2).transpose(0, 1)
    winning_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    next_token = gathered[:, :, 1].gather(dim=-1, index=winning_rank)
    return next_token.to(torch.int64)


def build_static_decode_mask(
    cache_position: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(
        cache_length, device=cache_position.device, dtype=torch.int64
    )
    return kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)


def scatter_update_tensor_(
    cache: torch.Tensor,
    cache_position: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for the Qwen3 TP2 decode path")
    positions = cache_position.reshape(-1).to(
        dtype=torch.int64, device=cache.device
    ).contiguous()
    torch_npu.scatter_update_(cache, positions, updates.contiguous(), 2)


def npu_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for the Qwen3 TP2 decode path")
    return torch_npu.npu_rms_norm(hidden_states, weight, float(eps))[0]


def npu_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for the Qwen3 TP2 decode path")
    normalized, _rstd, summed = torch_npu.npu_add_rms_norm(
        hidden_states,
        residual,
        weight,
        float(eps),
    )
    return normalized, summed


class Qwen3TPRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)


class Qwen3TPVocabEmbedding(nn.Module):
    def __init__(self, config: Qwen3TPConfig, *, tp_rank: int, tp_size: int):
        super().__init__()
        self.tp_size = int(tp_size)
        self.vocab_start, self.vocab_end = shard_bounds(
            config.vocab_size, tp_rank, tp_size
        )
        self.weight = nn.Parameter(
            torch.empty(self.vocab_end - self.vocab_start, config.hidden_size)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out_of_range = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(
            min=0, max=self.weight.shape[0] - 1
        )
        hidden_states = F.embedding(local_ids, self.weight)
        hidden_states = hidden_states.masked_fill(out_of_range.unsqueeze(-1), 0)
        return tp_all_reduce_sum(hidden_states, self.tp_size)


class Qwen3TPAttention(nn.Module):
    def __init__(self, config: Qwen3TPConfig, *, tp_size: int):
        super().__init__()
        self.tp_size = int(tp_size)
        self.num_heads = config.num_attention_heads // tp_size
        self.num_key_value_heads = config.num_key_value_heads // tp_size
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = Qwen3TPRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3TPRMSNorm(config.head_dim, config.rms_norm_eps)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence_length, _hidden = hidden_states.shape
        packed = linear_tokenwise(self.qkv_proj, hidden_states)
        query_states, key_states, value_states = packed.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query_states = query_states.view(
            batch, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states = npu_rms_norm(
            query_states, self.q_norm.weight, self.q_norm.variance_epsilon
        )
        key_states = npu_rms_norm(
            key_states, self.k_norm.weight, self.k_norm.variance_epsilon
        )
        query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos.unsqueeze(1),
            sin.unsqueeze(1),
            layout="BSND",
            rotary_mode="half",
        )
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        scatter_update_tensor_(key_cache, cache_position, key_states)
        scatter_update_tensor_(value_cache, cache_position, value_states)

        attention_output = torch_npu.npu_incre_flash_attention(
            query_states,
            key_cache,
            value_cache,
            atten_mask=attention_mask.contiguous(),
            num_heads=int(self.num_heads),
            num_key_value_heads=int(self.num_key_value_heads),
            input_layout="BNSD",
            scale_value=float(self.scaling),
        )
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, self.q_size
        )
        output = linear_tokenwise(self.o_proj, attention_output)
        return tp_all_reduce_sum(output, self.tp_size)


class Qwen3TPMLP(nn.Module):
    def __init__(self, config: Qwen3TPConfig, *, tp_size: int):
        super().__init__()
        self.tp_size = int(tp_size)
        self.intermediate_size = config.intermediate_size // tp_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = linear_tokenwise(self.gate_up_proj, hidden_states)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        output = linear_tokenwise(self.down_proj, F.silu(gate) * up)
        return tp_all_reduce_sum(output, self.tp_size)


class Qwen3TPDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3TPConfig, *, tp_size: int):
        super().__init__()
        self.self_attn = Qwen3TPAttention(config, tp_size=tp_size)
        self.mlp = Qwen3TPMLP(config, tp_size=tp_size)
        self.input_layernorm = Qwen3TPRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3TPRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            attention_input = npu_rms_norm(
                hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
            residual = hidden_states
        else:
            attention_input, residual = npu_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
        attention_output = self.self_attn.forward_decode(
            attention_input,
            cos,
            sin,
            key_cache,
            value_cache,
            cache_position,
            attention_mask,
        )
        mlp_input, residual = npu_add_rms_norm(
            attention_output,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        return self.mlp(mlp_input), residual


class Qwen3TPStaticCache:
    def __init__(
        self,
        config: Qwen3TPConfig,
        *,
        tp_size: int,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        local_kv_heads = config.num_key_value_heads // tp_size
        shape = (
            batch_size,
            local_kv_heads,
            cache_length,
            config.head_dim,
        )
        self.key_caches = tuple(
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        )
        self.value_caches = tuple(
            torch.zeros_like(key_cache) for key_cache in self.key_caches
        )


class Qwen3TPForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3TPConfig,
        *,
        tp_rank: int,
        tp_size: int,
    ):
        super().__init__()
        config.validate_tp(tp_size)
        self.config = config
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.embed_tokens = Qwen3TPVocabEmbedding(
            config, tp_rank=tp_rank, tp_size=tp_size
        )
        self.layers = nn.ModuleList(
            Qwen3TPDecoderLayer(config, tp_size=tp_size)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3TPRMSNorm(config.hidden_size, config.rms_norm_eps)
        vocab_start, vocab_end = shard_bounds(config.vocab_size, tp_rank, tp_size)
        self.vocab_start = int(vocab_start)
        self.lm_head = nn.Linear(
            config.hidden_size, vocab_end - vocab_start, bias=False
        )
        self.register_buffer("decode_factor_lut", None, persistent=False)

    def prepare_decode(self, *, cache_length: int) -> None:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.embed_tokens.weight.device,
                )
                / self.config.head_dim
            )
        )
        positions = torch.arange(
            cache_length,
            dtype=torch.float32,
            device=self.embed_tokens.weight.device,
        ).view(-1, 1)
        freqs = positions * inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.decode_factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(
            dtype=self.embed_tokens.weight.dtype
        )

    def decode(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        selected = torch.index_select(
            self.decode_factor_lut,
            1,
            cache_position.reshape(-1).to(dtype=torch.int64),
        )
        cos, sin = selected.unbind(dim=0)
        cos = cos.view(input_ids.shape[0], 1, self.config.head_dim)
        sin = sin.view(input_ids.shape[0], 1, self.config.head_dim)
        attention_mask = build_static_decode_mask(
            cache_position, key_caches[0].shape[2]
        )

        residual: torch.Tensor | None = None
        for layer_index, layer in enumerate(self.layers):
            hidden_states, residual = layer.forward_decode(
                hidden_states,
                residual,
                cos,
                sin,
                key_caches[layer_index],
                value_caches[layer_index],
                cache_position,
                attention_mask,
            )
        hidden_states, _residual = npu_add_rms_norm(
            hidden_states,
            residual,
            self.norm.weight,
            self.norm.variance_epsilon,
        )
        local_logits = linear_tokenwise(self.lm_head, hidden_states)[:, -1, :]
        return tp_local_argmax(
            local_logits,
            vocab_start=self.vocab_start,
            tp_size=self.tp_size,
        ).view(input_ids.shape[0], 1)
