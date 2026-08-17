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
except ModuleNotFoundError:
    torch_npu = None


@dataclass(frozen=True)
class Qwen3MoeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3MoeConfig":
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            moe_intermediate_size=int(raw["moe_intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            num_experts=int(raw["num_experts"]),
            num_experts_per_tok=int(raw["num_experts_per_tok"]),
            norm_topk_prob=bool(raw["norm_topk_prob"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 40_960)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )

    def validate_qwen3_30b_a3b(self) -> None:
        actual = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.moe_intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.num_experts,
            self.num_experts_per_tok,
            self.norm_topk_prob,
        )
        expected = (151936, 2048, 6144, 768, 48, 32, 4, 128, 128, 8, True)
        if actual != expected:
            raise ValueError(
                "Experiment 15 is fixed to Qwen3-30B-A3B. "
                f"Expected {expected}, got {actual}."
            )


def require_torch_npu() -> None:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for Experiment 15 NPU inference")


def linear_tokenwise(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    leading_shape = hidden_states.shape[:-1]
    output = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def build_static_decode_mask(
    cache_position: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(
        cache_length, device=position.device, dtype=torch.int64
    )
    return kv_positions.unsqueeze(0) > position.unsqueeze(1)


def scatter_update_tensor_(
    cache: torch.Tensor,
    cache_position: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    require_torch_npu()
    positions = cache_position.reshape(-1).to(
        dtype=torch.int64, device=cache.device
    ).contiguous()
    torch_npu.scatter_update_(cache, positions, updates.contiguous(), 2)


def npu_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    require_torch_npu()
    return torch_npu.npu_rms_norm(hidden_states, weight, float(eps))[0]


def selected_expert_bmm(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Execute selected experts without host routing or token re-encoding.

    Shapes:
      hidden_states: [tokens, hidden]
      selected_experts/routing_weights: [tokens, top_k]
      gate_up_proj: [experts, 2 * intermediate, hidden]
      down_proj: [experts, hidden, intermediate]
    """
    tokens, hidden_size = hidden_states.shape
    top_k = selected_experts.shape[1]
    intermediate_size = gate_up_proj.shape[1] // 2
    flat_experts = selected_experts.reshape(-1)

    selected_gate_up = torch.index_select(gate_up_proj, 0, flat_experts)
    expanded_hidden = (
        hidden_states.unsqueeze(1)
        .expand(tokens, top_k, hidden_size)
        .reshape(tokens * top_k, hidden_size, 1)
    )
    gate_up = torch.bmm(selected_gate_up, expanded_hidden).squeeze(-1)
    gate, up = gate_up.split(intermediate_size, dim=-1)
    intermediate = F.silu(gate) * up

    selected_down = torch.index_select(down_proj, 0, flat_experts)
    expert_output = torch.bmm(selected_down, intermediate.unsqueeze(-1)).squeeze(-1)
    expert_output = expert_output.view(tokens, top_k, hidden_size)
    return (expert_output * routing_weights.unsqueeze(-1)).sum(dim=1)


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
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
        require_torch_npu()
        batch_size, sequence_length, _hidden = hidden_states.shape
        packed = linear_tokenwise(self.qkv_proj, hidden_states)
        query_states, key_states, value_states = packed.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query_states = query_states.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
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
        return linear_tokenwise(self.o_proj, attention_output)


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        )

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        routing_weights = routing_weights.to(router_logits.dtype)
        return router_logits, routing_weights, selected_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _router_logits, routing_weights, selected_experts = self.route(hidden_states)
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        output = selected_expert_bmm(
            flat_hidden,
            selected_experts,
            routing_weights,
            self.gate_up_proj,
            self.down_proj,
        )
        return (
            output.view_as(hidden_states),
            selected_experts,
            routing_weights,
        )


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config)
        self.mlp = Qwen3SparseMoeBlock(config)
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


class Qwen3MoeStageCache:
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        num_layers: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (
            1,
            config.num_key_value_heads,
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

    def snapshot_prefix(self, prefix_length: int) -> dict[str, tuple[torch.Tensor, ...]]:
        return {
            "key_caches": tuple(
                cache[:, :, :prefix_length, :].cpu() for cache in self.key_caches
            ),
            "value_caches": tuple(
                cache[:, :, :prefix_length, :].cpu() for cache in self.value_caches
            ),
        }

    def restore_prefix(self, snapshot: dict[str, tuple[torch.Tensor, ...]]) -> int:
        prefix_length = int(snapshot["key_caches"][0].shape[2])
        for target, source in zip(self.key_caches, snapshot["key_caches"]):
            target[:, :, :prefix_length, :].copy_(
                source.to(device=target.device, dtype=target.dtype)
            )
        for target, source in zip(self.value_caches, snapshot["value_caches"]):
            target[:, :, :prefix_length, :].copy_(
                source.to(device=target.device, dtype=target.dtype)
            )
        return prefix_length


class Qwen3MoePipelineStage(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        layer_start: int,
        layer_end: int,
        with_embedding: bool,
        with_lm_head: bool,
    ):
        super().__init__()
        if not 0 <= layer_start < layer_end <= config.num_hidden_layers:
            raise ValueError(
                f"Invalid layer range [{layer_start}, {layer_end}) for "
                f"{config.num_hidden_layers} layers"
            )
        self.config = config
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size)
            if with_embedding
            else None
        )
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config)
            for _ in range(layer_end - layer_start)
        )
        self.norm = (
            Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
            if with_lm_head
            else None
        )
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
            cache_length,
            dtype=torch.float32,
            device=parameter.device,
        ).view(-1, 1)
        freqs = positions * inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.decode_factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(
            dtype=parameter.dtype
        )

    def make_cache(self, *, cache_length: int) -> Qwen3MoeStageCache:
        parameter = next(self.parameters())
        return Qwen3MoeStageCache(
            self.config,
            num_layers=self.num_layers,
            cache_length=cache_length,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _decode_hidden(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Qwen3MoeStageCache,
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
            cache_position, cache.key_caches[0].shape[2]
        )
        router_indices = []
        router_weights = []
        for layer_index, layer in enumerate(self.layers):
            hidden_states, indices, weights = layer.forward_decode(
                hidden_states,
                cos,
                sin,
                cache.key_caches[layer_index],
                cache.value_caches[layer_index],
                cache_position,
                attention_mask,
            )
            if capture_router:
                router_indices.append(indices)
                router_weights.append(weights)
        return hidden_states, tuple(router_indices), tuple(router_weights)

    def decode_input_ids(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Qwen3MoeStageCache,
        *,
        capture_router: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        if self.embed_tokens is None:
            raise RuntimeError("This pipeline stage does not own the embedding")
        hidden_states = self.embed_tokens(input_ids)
        return self._decode_hidden(
            hidden_states,
            cache_position,
            cache,
            capture_router=capture_router,
        )

    def decode_hidden_states(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Qwen3MoeStageCache,
        *,
        capture_router: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        return self._decode_hidden(
            hidden_states,
            cache_position,
            cache,
            capture_router=capture_router,
        )

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is None or self.lm_head is None:
            raise RuntimeError("This pipeline stage does not own the LM head")
        hidden_states = npu_rms_norm(
            hidden_states, self.norm.weight, self.norm.variance_epsilon
        )
        return linear_tokenwise(self.lm_head, hidden_states)
