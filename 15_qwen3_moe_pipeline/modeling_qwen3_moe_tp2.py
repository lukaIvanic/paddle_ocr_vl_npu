#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

import torch_npu

from modeling_qwen3_moe_pipeline import (
    Qwen3MoeConfig,
    Qwen3MoeRMSNorm,
    build_static_decode_mask,
    linear_tokenwise,
    npu_rms_norm,
    scatter_update_tensor_,
    selected_expert_bmm,
)


def shard_bounds(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size % world_size:
        raise ValueError(f"size={size} is not divisible by world_size={world_size}")
    width = size // world_size
    return rank * width, (rank + 1) * width


def tp_all_reduce_sum(tensor: torch.Tensor, tp_size: int) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class Qwen3MoeTPAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, *, tp_size: int):
        super().__init__()
        self.tp_size = int(tp_size)
        self.num_heads = config.num_attention_heads // tp_size
        self.num_key_value_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = Qwen3MoeRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(config.head_dim, config.rms_norm_eps)

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
        batch_size, sequence_length, _hidden = hidden_states.shape
        packed = linear_tokenwise(self.qkv_proj, hidden_states)
        query_states, key_states, value_states = packed.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query_states = query_states.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
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
            batch_size, sequence_length, self.q_size
        )
        local_output = linear_tokenwise(self.o_proj, attention_output)
        return tp_all_reduce_sum(local_output, self.tp_size)


class Qwen3MoeTPSparseBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, *, tp_size: int):
        super().__init__()
        self.tp_size = int(tp_size)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size // tp_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.intermediate_size,
            )
        )

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        router_logits = linear_tokenwise(self.gate, flat_hidden)
        router_probs = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        return routing_weights.to(router_logits.dtype), selected_experts

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_weights, selected_experts = self.route(hidden_states)
        local_output = selected_expert_bmm(
            hidden_states.reshape(-1, self.hidden_size),
            selected_experts,
            routing_weights,
            self.gate_up_proj,
            self.down_proj,
        ).view_as(hidden_states)
        return (
            tp_all_reduce_sum(local_output, self.tp_size),
            selected_experts,
            routing_weights,
        )


class Qwen3MoeTPDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, *, tp_size: int):
        super().__init__()
        self.self_attn = Qwen3MoeTPAttention(config, tp_size=tp_size)
        self.mlp = Qwen3MoeTPSparseBlock(config, tp_size=tp_size)
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attention_input = npu_rms_norm(
            hidden_states,
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
        hidden_states = residual + attention_output
        residual = hidden_states
        mlp_input = npu_rms_norm(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        mlp_output, selected_experts, routing_weights = self.mlp(mlp_input)
        return residual + mlp_output, selected_experts, routing_weights


class Qwen3MoeTPStageCache:
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        tp_rank: int,
        tp_size: int,
        num_layers: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        del tp_rank
        shape = (
            1,
            config.num_key_value_heads // tp_size,
            cache_length,
            config.head_dim,
        )
        self.key_caches = tuple(
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.value_caches = tuple(
            torch.zeros_like(cache) for cache in self.key_caches
        )

    def restore_full_prefix(
        self,
        snapshot: dict[str, tuple[torch.Tensor, ...]],
        *,
        tp_rank: int,
        tp_size: int,
    ) -> int:
        full_heads = int(snapshot["key_caches"][0].shape[1])
        head_start, head_end = shard_bounds(full_heads, tp_rank, tp_size)
        prefix_length = int(snapshot["key_caches"][0].shape[2])
        for target, source in zip(self.key_caches, snapshot["key_caches"]):
            target[:, :, :prefix_length, :].copy_(
                source[:, head_start:head_end, :, :].to(
                    device=target.device, dtype=target.dtype
                )
            )
        for target, source in zip(self.value_caches, snapshot["value_caches"]):
            target[:, :, :prefix_length, :].copy_(
                source[:, head_start:head_end, :, :].to(
                    device=target.device, dtype=target.dtype
                )
            )
        return prefix_length


class Qwen3MoeTPStage(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        tp_rank: int,
        tp_size: int,
        layer_start: int,
        layer_end: int,
        with_lm_head: bool,
        with_embedding: bool = False,
    ):
        super().__init__()
        config.validate_qwen3_30b_a3b()
        if config.num_attention_heads % tp_size:
            raise ValueError("Query heads are not divisible by TP size")
        if config.num_key_value_heads % tp_size:
            raise ValueError("KV heads are not divisible by TP size")
        if config.moe_intermediate_size % tp_size:
            raise ValueError("MoE intermediate size is not divisible by TP size")
        self.config = config
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.layers = nn.ModuleList(
            Qwen3MoeTPDecoderLayer(config, tp_size=tp_size)
            for _ in range(layer_end - layer_start)
        )
        self.norm = (
            Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
            if with_lm_head
            else None
        )
        vocab_start, vocab_end = shard_bounds(
            config.vocab_size, tp_rank, tp_size
        )
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.embed_tokens = (
            nn.Embedding(vocab_end - vocab_start, config.hidden_size)
            if with_embedding
            else None
        )
        self.lm_head = (
            nn.Linear(config.hidden_size, vocab_end - vocab_start, bias=False)
            if with_lm_head
            else None
        )
        self.register_buffer("decode_factor_lut", None, persistent=False)

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def prepare_decode(self, *, cache_length: int) -> None:
        parameter = next(self.parameters())
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.head_dim,
                    2,
                    dtype=torch.float32,
                    device=parameter.device,
                )
                / self.config.head_dim
            )
        )
        positions = torch.arange(
            cache_length, dtype=torch.float32, device=parameter.device
        ).view(-1, 1)
        freqs = positions * inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.decode_factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(
            dtype=parameter.dtype
        )

    def make_cache(self, *, cache_length: int) -> Qwen3MoeTPStageCache:
        parameter = next(self.parameters())
        return Qwen3MoeTPStageCache(
            self.config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            num_layers=self.num_layers,
            cache_length=cache_length,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        *,
        capture_router: bool,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        selected = torch.index_select(
            self.decode_factor_lut,
            1,
            cache_position.reshape(-1).to(dtype=torch.int64),
        )
        cos, sin = selected.unbind(dim=0)
        cos = cos.view(1, 1, self.config.head_dim)
        sin = sin.view(1, 1, self.config.head_dim)
        attention_mask = build_static_decode_mask(
            cache_position, key_caches[0].shape[2]
        )
        router_indices = []
        router_weights = []
        for layer_index, layer in enumerate(self.layers):
            hidden_states, indices, weights = layer.forward_decode(
                hidden_states,
                cos,
                sin,
                key_caches[layer_index],
                value_caches[layer_index],
                cache_position,
                attention_mask,
            )
            if capture_router:
                router_indices.append(indices)
                router_weights.append(weights)
        return hidden_states, tuple(router_indices), tuple(router_weights)

    def decode_local_output(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_states, _indices, _weights = self._decode(
            hidden_states,
            cache_position,
            key_caches,
            value_caches,
            capture_router=False,
        )
        if self.norm is None or self.lm_head is None:
            return hidden_states
        hidden_states = npu_rms_norm(
            hidden_states, self.norm.weight, self.norm.variance_epsilon
        )
        return linear_tokenwise(self.lm_head, hidden_states)[:, -1, :]

    def decode_input_ids_local_output(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("This TP stage has no token embedding")
        local_ids = input_ids - int(self.vocab_start)
        local_mask = (local_ids >= 0) & (
            local_ids < self.embed_tokens.num_embeddings
        )
        safe_local_ids = local_ids.clamp(
            min=0, max=self.embed_tokens.num_embeddings - 1
        )
        hidden_states = self.embed_tokens(safe_local_ids)
        hidden_states = hidden_states * local_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )
        hidden_states = tp_all_reduce_sum(hidden_states, self.tp_size)
        return self.decode_local_output(
            hidden_states,
            cache_position,
            key_caches,
            value_caches,
        )

    def decode_debug(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        output, indices, weights = self._decode(
            hidden_states,
            cache_position,
            key_caches,
            value_caches,
            capture_router=True,
        )
        if self.norm is not None and self.lm_head is not None:
            output = npu_rms_norm(
                output, self.norm.weight, self.norm.variance_epsilon
            )
            output = linear_tokenwise(self.lm_head, output)[:, -1, :]
        return output, indices, weights
