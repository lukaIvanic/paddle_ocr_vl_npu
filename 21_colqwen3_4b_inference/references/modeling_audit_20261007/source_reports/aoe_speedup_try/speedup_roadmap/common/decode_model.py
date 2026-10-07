from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import torch_npu


def linear_last_dim(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim <= 2:
        return linear(hidden_states)
    original_shape = hidden_states.shape[:-1]
    projected = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return projected.reshape(*original_shape, projected.shape[-1])


@dataclass(frozen=True)
class GlmOcrTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any]

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, *, num_layers: int | None = None) -> "GlmOcrTextConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["text_config"]
        hidden_size = int(raw["hidden_size"])
        num_attention_heads = int(raw["num_attention_heads"])
        rope_scaling = dict(raw.get("rope_scaling") or raw.get("rope_parameters") or {})
        rope_theta = float(rope_scaling.pop("rope_theta", raw.get("rope_theta", 10000.0)))
        rope_scaling.pop("partial_rotary_factor", None)
        total_layers = int(raw["num_hidden_layers"])
        if num_layers is not None:
            total_layers = min(total_layers, int(num_layers))
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=total_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim") or hidden_size // num_attention_heads),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )


class GlmOcrRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, *, norm_impl: str = "manual"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)
        if norm_impl not in {"manual", "npu"}:
            raise ValueError(f"Unsupported norm_impl={norm_impl}")
        self.norm_impl = norm_impl

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm_impl == "npu":
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def apply_llm_pairwise_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_embed_even = (x_even * cos) - (x_odd * sin)
    x_embed_odd = (x_odd * cos) + (x_even * sin)
    return torch.stack((x_embed_even, x_embed_odd), dim=-1).flatten(-2)


def apply_llm_pairwise_rotary_view(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]
    x_embed_even = (x_even * cos) - (x_odd * sin)
    x_embed_odd = (x_odd * cos) + (x_even * sin)
    return torch.stack((x_embed_even, x_embed_odd), dim=-1).flatten(-2)


def apply_llm_half_layout_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even, x_odd = x.chunk(2, dim=-1)
    x_embed_even = (x_even * cos) - (x_odd * sin)
    x_embed_odd = (x_odd * cos) + (x_even * sin)
    return torch.cat((x_embed_even, x_embed_odd), dim=-1)


def prepare_multimodal_rotary_factors(
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    *,
    interleaved_full_dim: bool = False,
    half_layout_full_dim: bool = False,
    attention_layout: str = "bnsd",
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_layout not in {"bnsd", "bsnd"}:
        raise ValueError(f"Unsupported attention_layout={attention_layout}")
    if interleaved_full_dim and half_layout_full_dim:
        raise ValueError("interleaved_full_dim and half_layout_full_dim are mutually exclusive")
    mrope_section = scale_mrope_section(mrope_section, int(cos.shape[-1])) * 2
    unsqueeze_dim = 1 if attention_layout == "bnsd" else 2
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2]
    if interleaved_full_dim:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    elif half_layout_full_dim:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    return cos, sin


def apply_prepared_interleaved_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = torch_npu.npu_rotary_mul(q_rot.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="interleave")
    k_out = torch_npu.npu_rotary_mul(k_rot.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="interleave")
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def apply_prepared_manual_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = apply_llm_pairwise_rotary(q_rot, cos, sin)
    k_out = apply_llm_pairwise_rotary(k_rot, cos, sin)
    if rotary_dim == query_states.shape[-1]:
        return q_out, k_out
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def apply_prepared_manual_rotary_no_full_slice(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == query_states.shape[-1]:
        return apply_llm_pairwise_rotary(query_states, cos, sin), apply_llm_pairwise_rotary(key_states, cos, sin)
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = apply_llm_pairwise_rotary(q_rot, cos, sin)
    k_out = apply_llm_pairwise_rotary(k_rot, cos, sin)
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def apply_prepared_manual_rotary_view(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == query_states.shape[-1]:
        return apply_llm_pairwise_rotary_view(query_states, cos, sin), apply_llm_pairwise_rotary_view(key_states, cos, sin)
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = apply_llm_pairwise_rotary_view(q_rot, cos, sin)
    k_out = apply_llm_pairwise_rotary_view(k_rot, cos, sin)
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def apply_prepared_manual_rotary_half_layout(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == query_states.shape[-1]:
        return apply_llm_half_layout_rotary(query_states, cos, sin), apply_llm_half_layout_rotary(key_states, cos, sin)
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = apply_llm_half_layout_rotary(q_rot, cos, sin)
    k_out = apply_llm_half_layout_rotary(k_rot, cos, sin)
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def apply_prepared_npu_rotary_half_layout(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    if rotary_dim == query_states.shape[-1]:
        return (
            torch_npu.npu_rotary_mul(query_states.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="half"),
            torch_npu.npu_rotary_mul(key_states.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="half"),
        )
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    q_out = torch_npu.npu_rotary_mul(q_rot.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="half")
    k_out = torch_npu.npu_rotary_mul(k_rot.contiguous(), cos.contiguous(), sin.contiguous(), rotary_mode="half")
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def interleaved_to_half_layout_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat((tensor[..., 0::2], tensor[..., 1::2]), dim=-1)


def projection_weight_to_half_layout(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    shaped = weight.view(int(num_heads), int(head_dim), weight.shape[-1])
    return torch.cat((shaped[:, 0::2, :], shaped[:, 1::2, :]), dim=1).reshape(weight.shape)


def scale_mrope_section(mrope_section: list[int], target_head_dim: int) -> list[int]:
    base_head_dim = 2 * sum(int(value) for value in mrope_section)
    if target_head_dim == base_head_dim:
        return [int(value) for value in mrope_section]
    scaled = []
    for value in mrope_section:
        numerator = int(value) * int(target_head_dim)
        if numerator % base_head_dim != 0:
            raise ValueError(f"Cannot scale mrope_section={mrope_section} to head_dim={target_head_dim}")
        scaled.append(numerator // base_head_dim)
    return scaled


def apply_multimodal_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = prepare_multimodal_rotary_factors(cos, sin, mrope_section)
    rotary_dim = cos.shape[-1] * 2
    q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    return (
        torch.cat((apply_llm_pairwise_rotary(q_rot, cos, sin), q_pass), dim=-1),
        torch.cat((apply_llm_pairwise_rotary(k_rot, cos, sin), k_pass), dim=-1),
    )


class GlmOcrTextRotaryEmbedding(nn.Module):
    def __init__(self, config: GlmOcrTextConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[:, :, :, None].float() * self.inv_freq[None, None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=hidden_states.dtype), emb.sin().to(dtype=hidden_states.dtype)


class GlmOcrTextMLP(nn.Module):
    def __init__(self, config: GlmOcrTextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up_states = linear_last_dim(self.gate_up_proj, hidden_states).chunk(2, dim=-1)
        return linear_last_dim(self.down_proj, F.silu(gate) * up_states)


class GlmOcrTextAttention(nn.Module):
    def __init__(
        self,
        config: GlmOcrTextConfig,
        *,
        rotary_impl: str = "manual",
        qkv_impl: str = "separate",
        attention_layout: str = "bnsd",
        increfa_mode: str = "mask",
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.mrope_section = list(config.rope_scaling["mrope_section"])
        if rotary_impl not in {
            "manual",
            "manual_hoisted",
            "manual_hoisted_noslice",
            "manual_hoisted_view",
            "manual_hoisted_half_layout",
            "npu_rotary_mul_half_layout",
            "npu_rotary_mul_interleave",
        }:
            raise ValueError(f"Unsupported rotary_impl={rotary_impl}")
        if qkv_impl not in {"separate", "fused"}:
            raise ValueError(f"Unsupported qkv_impl={qkv_impl}")
        if attention_layout not in {"bnsd", "bsnd"}:
            raise ValueError(f"Unsupported attention_layout={attention_layout}")
        if increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported increfa_mode={increfa_mode}")
        self.rotary_impl = rotary_impl
        self.qkv_impl = qkv_impl
        self.attention_layout = attention_layout
        self.increfa_mode = increfa_mode
        self.q_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        if qkv_impl == "fused":
            self.qkv_proj = nn.Linear(config.hidden_size, self.q_width + 2 * self.kv_width, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.q_width, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, self.kv_width, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.kv_width, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prepared_rotary_factors: tuple[torch.Tensor, torch.Tensor] | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        decode_control: Any,
    ) -> torch.Tensor:
        batch, sequence_length, _hidden_size = hidden_states.shape
        if self.qkv_impl == "fused":
            qkv_states = linear_last_dim(self.qkv_proj, hidden_states)
            query_states, key_states, value_states = qkv_states.split((self.q_width, self.kv_width, self.kv_width), dim=-1)
        else:
            query_states = linear_last_dim(self.q_proj, hidden_states)
            key_states = linear_last_dim(self.k_proj, hidden_states)
            value_states = linear_last_dim(self.v_proj, hidden_states)
        query_states = query_states.view(batch, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch, sequence_length, self.num_key_value_heads, self.head_dim)
        if self.attention_layout == "bnsd":
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
        if self.rotary_impl == "npu_rotary_mul_interleave":
            rotary_cos, rotary_sin = prepare_multimodal_rotary_factors(
                position_embeddings[0],
                position_embeddings[1],
                self.mrope_section,
                interleaved_full_dim=True,
                attention_layout=self.attention_layout,
            )
            query_states, key_states = apply_prepared_interleaved_rotary(query_states, key_states, rotary_cos, rotary_sin)
        elif self.rotary_impl == "manual_hoisted":
            if prepared_rotary_factors is None:
                raise RuntimeError("manual_hoisted rotary requires prepared factors")
            query_states, key_states = apply_prepared_manual_rotary(
                query_states,
                key_states,
                prepared_rotary_factors[0],
                prepared_rotary_factors[1],
            )
        elif self.rotary_impl == "manual_hoisted_noslice":
            if prepared_rotary_factors is None:
                raise RuntimeError("manual_hoisted_noslice rotary requires prepared factors")
            query_states, key_states = apply_prepared_manual_rotary_no_full_slice(
                query_states,
                key_states,
                prepared_rotary_factors[0],
                prepared_rotary_factors[1],
            )
        elif self.rotary_impl == "manual_hoisted_view":
            if prepared_rotary_factors is None:
                raise RuntimeError("manual_hoisted_view rotary requires prepared factors")
            query_states, key_states = apply_prepared_manual_rotary_view(
                query_states,
                key_states,
                prepared_rotary_factors[0],
                prepared_rotary_factors[1],
            )
        elif self.rotary_impl == "manual_hoisted_half_layout":
            if prepared_rotary_factors is None:
                raise RuntimeError("manual_hoisted_half_layout rotary requires prepared factors")
            query_states, key_states = apply_prepared_manual_rotary_half_layout(
                query_states,
                key_states,
                prepared_rotary_factors[0],
                prepared_rotary_factors[1],
            )
        elif self.rotary_impl == "npu_rotary_mul_half_layout":
            if prepared_rotary_factors is None:
                raise RuntimeError("npu_rotary_mul_half_layout rotary requires prepared factors")
            query_states, key_states = apply_prepared_npu_rotary_half_layout(
                query_states,
                key_states,
                prepared_rotary_factors[0],
                prepared_rotary_factors[1],
            )
        else:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                position_embeddings[0],
                position_embeddings[1],
                self.mrope_section,
            )
        positions = cache_position.reshape(-1).to(device=key_cache.device, dtype=torch.int64).contiguous()
        scatter_axis = 2 if self.attention_layout == "bnsd" else 1
        torch_npu.scatter_update_(key_cache, positions, key_states.contiguous(), scatter_axis)
        torch_npu.scatter_update_(value_cache, positions, value_states.contiguous(), scatter_axis)
        if self.increfa_mode == "mask":
            atten_mask = decode_control.contiguous()
            actual_seq_lengths = None
        else:
            atten_mask = None
            actual_seq_lengths = decode_control
        attn_output = torch_npu.npu_incre_flash_attention(
            query_states.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            num_heads=int(self.num_heads),
            num_key_value_heads=int(self.num_key_value_heads),
            input_layout="BNSD" if self.attention_layout == "bnsd" else "BSND",
            scale_value=float(self.scaling),
        )
        if self.attention_layout == "bnsd":
            attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().reshape(batch, sequence_length, self.num_heads * self.head_dim)
        return linear_last_dim(self.o_proj, attn_output)


class GlmOcrTextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GlmOcrTextConfig,
        *,
        rotary_impl: str = "manual",
        qkv_impl: str = "separate",
        norm_impl: str = "manual",
        attention_layout: str = "bnsd",
        increfa_mode: str = "mask",
    ):
        super().__init__()
        self.self_attn = GlmOcrTextAttention(
            config,
            rotary_impl=rotary_impl,
            qkv_impl=qkv_impl,
            attention_layout=attention_layout,
            increfa_mode=increfa_mode,
        )
        self.mlp = GlmOcrTextMLP(config)
        self.input_layernorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_impl=norm_impl)
        self.post_attention_layernorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_impl=norm_impl)
        self.post_self_attn_layernorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_impl=norm_impl)
        self.post_mlp_layernorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_impl=norm_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prepared_rotary_factors: tuple[torch.Tensor, torch.Tensor] | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        decode_control: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            normalized_hidden_states,
            position_embeddings,
            prepared_rotary_factors,
            key_cache,
            value_cache,
            cache_position,
            decode_control,
        )
        hidden_states = self.post_self_attn_layernorm(attn_output)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return residual + hidden_states


class GlmOcrDecodeOnlyModel(nn.Module):
    def __init__(
        self,
        config: GlmOcrTextConfig,
        *,
        rotary_impl: str = "manual",
        qkv_impl: str = "separate",
        norm_impl: str = "manual",
        attention_layout: str = "bnsd",
        increfa_mode: str = "mask",
    ):
        super().__init__()
        if increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported increfa_mode={increfa_mode}")
        self.config = config
        self.rotary_impl = rotary_impl
        self.qkv_impl = qkv_impl
        self.norm_impl = norm_impl
        self.attention_layout = attention_layout
        self.increfa_mode = increfa_mode
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                GlmOcrTextDecoderLayer(
                    config,
                    rotary_impl=rotary_impl,
                    qkv_impl=qkv_impl,
                    norm_impl=norm_impl,
                    attention_layout=attention_layout,
                    increfa_mode=increfa_mode,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_impl=norm_impl)
        self.rotary_emb = GlmOcrTextRotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *flat_cache_and_control: torch.Tensor,
    ) -> torch.Tensor:
        layer_count = self.config.num_hidden_layers
        expected_inputs = 2 * layer_count + 3
        if len(flat_cache_and_control) != expected_inputs:
            raise ValueError(f"expected {expected_inputs} cache/control tensors, got {len(flat_cache_and_control)}")
        key_caches = flat_cache_and_control[:layer_count]
        value_caches = flat_cache_and_control[layer_count : 2 * layer_count]
        cache_position = flat_cache_and_control[-3]
        rope_deltas = flat_cache_and_control[-2]
        decode_control = flat_cache_and_control[-1]
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length, _hidden_size = hidden_states.shape
        if sequence_length != 1:
            raise ValueError(f"decode expects sequence_length=1, got {sequence_length}")
        cache_position = cache_position.reshape(-1).to(device=hidden_states.device, dtype=torch.int64)
        position_ids = cache_position.view(batch_size, 1) + rope_deltas.to(device=hidden_states.device, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        prepared_rotary_factors = None
        if self.rotary_impl in {
            "manual_hoisted",
            "manual_hoisted_noslice",
            "manual_hoisted_view",
            "manual_hoisted_half_layout",
            "npu_rotary_mul_half_layout",
        }:
            prepared_rotary_factors = prepare_multimodal_rotary_factors(
                position_embeddings[0],
                position_embeddings[1],
                self.layers[0].self_attn.mrope_section,
                half_layout_full_dim=self.rotary_impl == "npu_rotary_mul_half_layout",
                attention_layout=self.attention_layout,
            )
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                prepared_rotary_factors,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                decode_control,
            )
        hidden_states = self.norm(hidden_states)
        return linear_last_dim(self.lm_head, hidden_states)


def load_decode_model(
    model_dir: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int | None = None,
    rotary_impl: str = "manual",
    qkv_impl: str = "separate",
    norm_impl: str = "manual",
    attention_layout: str = "bnsd",
    increfa_mode: str = "mask",
) -> GlmOcrDecodeOnlyModel:
    from safetensors.torch import load_file

    config = GlmOcrTextConfig.from_model_dir(model_dir, num_layers=num_layers)
    model = GlmOcrDecodeOnlyModel(
        config,
        rotary_impl=rotary_impl,
        qkv_impl=qkv_impl,
        norm_impl=norm_impl,
        attention_layout=attention_layout,
        increfa_mode=increfa_mode,
    )
    state = load_file(str(Path(model_dir) / "model.safetensors"), device="cpu")
    mapped = {}
    qk_half_layout = rotary_impl in {"manual_hoisted_half_layout", "npu_rotary_mul_half_layout"}
    for key, value in state.items():
        if key.startswith("model.language_model.layers."):
            new_key = key.replace("model.language_model.layers.", "layers.", 1)
            layer_idx = int(new_key.split(".")[1])
            if layer_idx < config.num_hidden_layers:
                if qkv_impl == "fused" and any(
                    new_key.endswith(f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")
                ):
                    continue
                if qk_half_layout and new_key.endswith("self_attn.q_proj.weight"):
                    value = projection_weight_to_half_layout(value, config.num_attention_heads, config.head_dim)
                elif qk_half_layout and new_key.endswith("self_attn.k_proj.weight"):
                    value = projection_weight_to_half_layout(value, config.num_key_value_heads, config.head_dim)
                mapped[new_key] = value
        elif key == "model.language_model.norm.weight":
            mapped["norm.weight"] = value
        elif key in {"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"}:
            mapped["embed_tokens.weight"] = value
        elif key in {"lm_head.weight", "model.lm_head.weight"}:
            mapped["lm_head.weight"] = value
    if qkv_impl == "fused":
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}.self_attn"
            q_weight = state[f"{prefix}.q_proj.weight"]
            k_weight = state[f"{prefix}.k_proj.weight"]
            if qk_half_layout:
                q_weight = projection_weight_to_half_layout(q_weight, config.num_attention_heads, config.head_dim)
                k_weight = projection_weight_to_half_layout(k_weight, config.num_key_value_heads, config.head_dim)
            mapped[f"layers.{layer_idx}.self_attn.qkv_proj.weight"] = torch.cat(
                (
                    q_weight,
                    k_weight,
                    state[f"{prefix}.v_proj.weight"],
                ),
                dim=0,
            )
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"decode model state mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def build_decode_mask(cache_position: torch.Tensor, cache_length: int) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(cache_length, device=cache_position.device, dtype=torch.int64)
    return (kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)).view(cache_position.shape[0], 1, 1, cache_length)
