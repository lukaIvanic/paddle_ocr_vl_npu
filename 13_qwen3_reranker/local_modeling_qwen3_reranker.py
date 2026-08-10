#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


PROMPT_FA_FULL_ATTENTION_TOKENS = (1 << 31) - 1


@dataclass(frozen=True)
class LocalQwen3RerankerConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "LocalQwen3RerankerConfig":
        with open(Path(model_dir) / "config.json") as handle:
            raw = json.load(handle)
        return cls.from_hf_config(raw)

    @classmethod
    def from_hf_config(cls, raw: dict) -> "LocalQwen3RerankerConfig":
        hidden_size = int(raw["hidden_size"])
        num_attention_heads = int(raw["num_attention_heads"])
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
        )


class LocalQwen3RerankerRMSNorm(nn.Module):
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


class LocalQwen3RerankerRotaryEmbedding(nn.Module):
    def __init__(self, config: LocalQwen3RerankerConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Elementwise broadcasting is equivalent to the usual outer product,
        # but avoids a rank-3 broadcasted MatMul that GE mis-infers when B > 1.
        inv_freq = self.inv_freq.to(device=device).view(1, 1, -1).float()
        position_ids = position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


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


def linear_tokenwise(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply Linear through the 2-D token matrix expected by GE."""
    leading_shape = hidden_states.shape[:-1]
    output = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def prompt_flash_attention_bnsd_310p_compatible(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    num_heads: int,
    scale: float,
) -> torch.Tensor:
    """Run the Atlas inference-series-safe PromptFA contract.

    Atlas 310P does not support the PromptFA actual-sequence-length inputs or a
    non-default ``num_key_value_heads``. Expand GQA key/value heads explicitly,
    encode left padding and causality in a bool mask, and omit those unsupported
    optional arguments from the operator call.
    """
    if query_states.dtype != torch.float16:
        raise ValueError("310P-compatible prompt_flash_attention requires float16 Q/K/V")
    if key_states.dtype != query_states.dtype or value_states.dtype != query_states.dtype:
        raise ValueError("prompt_flash_attention requires matching Q/K/V dtypes")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("BNSD prompt_flash_attention requires rank-4 Q/K/V tensors")
    if int(query_states.shape[1]) != int(num_heads):
        raise ValueError("num_heads must match the query N dimension")
    if key_states.shape != value_states.shape:
        raise ValueError("key and value shapes must match")

    num_key_value_heads = int(key_states.shape[1])
    if int(num_heads) % num_key_value_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    num_key_value_groups = int(num_heads) // num_key_value_heads
    key_states = repeat_kv(key_states, num_key_value_groups).contiguous()
    value_states = repeat_kv(value_states, num_key_value_groups).contiguous()

    try:
        import torch_npu
    except Exception as exc:
        raise RuntimeError(f"torch_npu import failed for prompt flash attention: {exc}") from exc

    return torch_npu.npu_prompt_flash_attention(
        query_states.contiguous(),
        key_states,
        value_states,
        atten_mask=attention_mask.to(dtype=torch.bool).contiguous(),
        num_heads=int(num_heads),
        input_layout="BNSD",
        scale_value=float(scale),
        pre_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
        next_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
        sparse_mode=0,
    )


def build_left_padded_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch, sequence_length = attention_mask.shape
    device = attention_mask.device
    query_positions = torch.arange(sequence_length, device=device, dtype=torch.int64).view(
        1,
        1,
        sequence_length,
        1,
    )
    key_positions = torch.arange(sequence_length, device=device, dtype=torch.int64).view(
        1,
        1,
        1,
        sequence_length,
    )
    causal = query_positions >= key_positions
    key_valid = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, sequence_length)
    query_is_pad = ~attention_mask.to(dtype=torch.bool).view(batch, 1, sequence_length, 1)
    self_only_for_pad_rows = query_positions == key_positions
    allowed = (causal & key_valid) | (query_is_pad & self_only_for_pad_rows)
    zeros = torch.zeros((batch, 1, sequence_length, sequence_length), device=device, dtype=dtype)
    return zeros.masked_fill(~allowed, torch.finfo(dtype).min)


def build_left_padded_causal_bool_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    batch, sequence_length = attention_mask.shape
    device = attention_mask.device
    query_positions = torch.arange(sequence_length, device=device, dtype=torch.int64).view(
        1,
        1,
        sequence_length,
        1,
    )
    key_positions = torch.arange(sequence_length, device=device, dtype=torch.int64).view(
        1,
        1,
        1,
        sequence_length,
    )
    causal = query_positions >= key_positions
    key_valid = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, sequence_length)
    query_is_pad = ~attention_mask.to(dtype=torch.bool).view(batch, 1, sequence_length, 1)
    self_only_for_pad_rows = query_positions == key_positions
    allowed = (causal & key_valid) | (query_is_pad & self_only_for_pad_rows)
    return ~allowed


def build_left_padded_causal_bool_mask_chunk(
    attention_mask: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
) -> torch.Tensor:
    """Build one [B,1,Q,K] causal block without materializing an S-by-S mask."""
    batch, sequence_length = attention_mask.shape
    if not 0 <= int(query_start) < int(query_end) <= int(sequence_length):
        raise ValueError("chunk query bounds must be inside the padded sequence")
    device = attention_mask.device
    query_positions = torch.arange(
        int(query_start), int(query_end), device=device, dtype=torch.int64
    ).view(1, 1, -1, 1)
    key_positions = torch.arange(int(query_end), device=device, dtype=torch.int64).view(
        1, 1, 1, -1
    )
    causal = query_positions >= key_positions
    key_valid = attention_mask[:, : int(query_end)].to(dtype=torch.bool).view(
        batch, 1, 1, int(query_end)
    )
    query_is_pad = ~attention_mask[:, int(query_start) : int(query_end)].to(
        dtype=torch.bool
    ).view(batch, 1, int(query_end) - int(query_start), 1)
    self_only_for_pad_rows = query_positions == key_positions
    allowed = (causal & key_valid) | (query_is_pad & self_only_for_pad_rows)
    return ~allowed


class LocalQwen3RerankerMLP(nn.Module):
    def __init__(self, config: LocalQwen3RerankerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = linear_tokenwise(self.gate_proj, hidden_states)
        up = linear_tokenwise(self.up_proj, hidden_states)
        return linear_tokenwise(self.down_proj, F.silu(gate) * up)


class LocalQwen3RerankerAttention(nn.Module):
    def __init__(self, config: LocalQwen3RerankerConfig, *, attention_impl: str):
        super().__init__()
        if attention_impl not in {"eager", "prompt_flash_attention"}:
            raise ValueError(f"unsupported attention_impl={attention_impl!r}")
        self.attention_impl = attention_impl
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = LocalQwen3RerankerRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = LocalQwen3RerankerRMSNorm(config.head_dim, config.rms_norm_eps)

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, _hidden = hidden_states.shape
        query_states = linear_tokenwise(self.q_proj, hidden_states).view(
            batch, sequence_length, self.num_heads, self.head_dim
        )
        key_states = linear_tokenwise(self.k_proj, hidden_states).view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = linear_tokenwise(self.v_proj, hidden_states).view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()

    def forward_eager(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence_length, _hidden = hidden_states.shape
        query_states, key_states, value_states = self.project_qkv(hidden_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch, sequence_length, self.num_heads * self.head_dim)
        return linear_tokenwise(self.o_proj, attn_output)

    def forward_prompt_flash_attention(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.device.type != "npu":
            raise RuntimeError("prompt_flash_attention requires NPU tensors")
        batch, sequence_length, _hidden = hidden_states.shape
        query_states, key_states, value_states = self.project_qkv(hidden_states, cos, sin)

        attn_output = prompt_flash_attention_bnsd_310p_compatible(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            num_heads=int(self.num_heads),
            scale=float(self.scaling),
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, sequence_length, self.num_heads * self.head_dim)
        return linear_tokenwise(self.o_proj, attn_output)

    def forward_prompt_flash_attention_chunk(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_states: torch.Tensor | None,
        past_value_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.device.type != "npu":
            raise RuntimeError("prompt_flash_attention requires NPU tensors")
        batch, query_length, _hidden = hidden_states.shape
        query_states, current_key_states, current_value_states = self.project_qkv(
            hidden_states, cos, sin
        )
        if (past_key_states is None) != (past_value_states is None):
            raise ValueError("past key and value states must both be present or absent")
        if past_key_states is None:
            key_states = current_key_states
            value_states = current_value_states
        else:
            key_states = torch.cat((past_key_states, current_key_states), dim=2).contiguous()
            value_states = torch.cat((past_value_states, current_value_states), dim=2).contiguous()

        attn_output = prompt_flash_attention_bnsd_310p_compatible(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            num_heads=int(self.num_heads),
            scale=float(self.scaling),
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch, query_length, self.num_heads * self.head_dim
        )
        return (
            linear_tokenwise(self.o_proj, attn_output),
            key_states,
            value_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.attention_impl == "prompt_flash_attention":
            return self.forward_prompt_flash_attention(hidden_states, cos, sin, attention_mask)
        return self.forward_eager(hidden_states, cos, sin, attention_mask)


class LocalQwen3RerankerDecoderLayer(nn.Module):
    def __init__(self, config: LocalQwen3RerankerConfig, *, attention_impl: str):
        super().__init__()
        self.self_attn = LocalQwen3RerankerAttention(config, attention_impl=attention_impl)
        self.mlp = LocalQwen3RerankerMLP(config)
        self.input_layernorm = LocalQwen3RerankerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LocalQwen3RerankerRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward_prompt_flash_attention_chunk(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_states: torch.Tensor | None,
        past_value_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_output, updated_key_states, updated_value_states = (
            self.self_attn.forward_prompt_flash_attention_chunk(
                self.input_layernorm(hidden_states),
                cos,
                sin,
                attention_mask,
                past_key_states,
                past_value_states,
            )
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, updated_key_states, updated_value_states


class LocalQwen3RerankerForCausalLM(nn.Module):
    def __init__(self, config: LocalQwen3RerankerConfig, *, attention_impl: str = "eager"):
        super().__init__()
        if attention_impl not in {"eager", "prompt_flash_attention"}:
            raise ValueError(f"unsupported attention_impl={attention_impl!r}")
        self.config = config
        self.attention_impl = attention_impl
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LocalQwen3RerankerDecoderLayer(config, attention_impl=attention_impl)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LocalQwen3RerankerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = LocalQwen3RerankerRotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward_hidden_states_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        layer_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids, dtype=hidden_states.dtype, device=hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, layer_attention_mask)
        return self.norm(hidden_states)

    def forward_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        additive_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.forward_hidden_states_prepared(input_ids, position_ids, additive_mask)
        return self.lm_head(hidden_states[:, -1])

    def forward_prepared_yes_no(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        additive_mask: torch.Tensor,
        yes_no_weight: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.forward_hidden_states_prepared(input_ids, position_ids, additive_mask)
        return F.linear(hidden_states[:, -1], yes_no_weight)

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        layer_attention_mask = (
            build_left_padded_causal_bool_mask(attention_mask)
            if self.attention_impl == "prompt_flash_attention"
            else build_left_padded_causal_mask(attention_mask, self.embed_tokens.weight.dtype)
        )
        return self.forward_hidden_states_prepared(input_ids, position_ids, layer_attention_mask)

    def forward_hidden_states_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """Run sequential PromptFA prefill blocks and return the final block states."""
        if self.attention_impl != "prompt_flash_attention":
            raise ValueError("chunked prefill requires prompt_flash_attention")
        sequence_length = int(input_ids.shape[1])
        chunk_size = int(chunk_size)
        if chunk_size <= 0 or chunk_size % 128 != 0:
            raise ValueError("310P-compatible chunk_size must be a positive multiple of 128")
        if sequence_length % chunk_size != 0:
            raise ValueError("padded sequence length must be divisible by chunk_size")

        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        embedded_inputs = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            position_ids,
            dtype=embedded_inputs.dtype,
            device=embedded_inputs.device,
        )
        key_caches: list[torch.Tensor | None] = [None] * len(self.layers)
        value_caches: list[torch.Tensor | None] = [None] * len(self.layers)
        final_hidden_states: torch.Tensor | None = None

        for query_start in range(0, sequence_length, chunk_size):
            query_end = query_start + chunk_size
            hidden_states = embedded_inputs[:, query_start:query_end]
            chunk_mask = build_left_padded_causal_bool_mask_chunk(
                attention_mask,
                query_start=query_start,
                query_end=query_end,
            )
            for layer_index, layer in enumerate(self.layers):
                hidden_states, updated_key_states, updated_value_states = (
                    layer.forward_prompt_flash_attention_chunk(
                        hidden_states,
                        cos[:, query_start:query_end],
                        sin[:, query_start:query_end],
                        chunk_mask,
                        key_caches[layer_index],
                        value_caches[layer_index],
                    )
                )
                key_caches[layer_index] = updated_key_states
                value_caches[layer_index] = updated_value_states
            final_hidden_states = self.norm(hidden_states)

        if final_hidden_states is None:
            raise RuntimeError("chunked prefill produced no chunks")
        return final_hidden_states

    def _forward_chunk_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...] | None,
        value_caches: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            position_ids,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        updated_key_caches: list[torch.Tensor] = []
        updated_value_caches: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            past_key_states = None if key_caches is None else key_caches[layer_index]
            past_value_states = None if value_caches is None else value_caches[layer_index]
            hidden_states, updated_key_states, updated_value_states = (
                layer.forward_prompt_flash_attention_chunk(
                    hidden_states,
                    cos,
                    sin,
                    chunk_mask,
                    past_key_states,
                    past_value_states,
                )
            )
            updated_key_caches.append(updated_key_states)
            updated_value_caches.append(updated_value_states)
        return self.norm(hidden_states), tuple(updated_key_caches), tuple(updated_value_caches)

    def forward_first_chunk_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return self._forward_chunk_prepared(
            input_ids,
            position_ids,
            chunk_mask,
            None,
            None,
        )

    def forward_next_chunk_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return self._forward_chunk_prepared(
            input_ids,
            position_ids,
            chunk_mask,
            key_caches,
            value_caches,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.forward_hidden_states(input_ids, attention_mask)
        return self.lm_head(hidden_states[:, -1])

    def score_yes_no(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        false_token_id: int,
        true_token_id: int,
    ) -> torch.Tensor:
        yes_no_ids = torch.tensor([false_token_id, true_token_id], device=input_ids.device, dtype=torch.long)
        yes_no_logits = F.linear(self.forward_hidden_states(input_ids, attention_mask)[:, -1], self.lm_head.weight[yes_no_ids])
        return F.softmax(yes_no_logits, dim=-1)[:, 1]
