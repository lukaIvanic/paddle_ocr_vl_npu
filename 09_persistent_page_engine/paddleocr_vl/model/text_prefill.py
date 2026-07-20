"""Unified eager and compiled model execution for text prefill.

Token embedding, multimodal embedding scatter, the LM head, and greedy argmax
remain outside this stage. This module prepares exact-shape or bucket-padded
multimodal embeddings and runs the same text transformer plus in-place KV-cache
population either eagerly or through TorchAir. It returns the hidden state at
the last real prompt token, so padded query rows never become observable.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .config import PaddleOCRTextConfig
from .text_decode import LocalPaddleOCRVLStaticCache
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


DEFAULT_TEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
TEXT_BACKEND_CHOICES = ("raw_eager", "torchair")
TEXT_PADDING_CHOICES = ("auto", "none", "bucket")
TEXT_SOFTMAX_DTYPE_ENV = "PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE"
SOFTMAX_DTYPE_CHOICES = ("fp32", "model")


def get_text_softmax_dtype_mode() -> str:
    mode = (
        os.environ.get(TEXT_SOFTMAX_DTYPE_ENV, "fp32").strip().lower()
        or "fp32"
    )
    if mode not in SOFTMAX_DTYPE_CHOICES:
        raise ValueError(
            f"{TEXT_SOFTMAX_DTYPE_ENV} must be one of "
            f"{SOFTMAX_DTYPE_CHOICES}, got {mode!r}"
        )
    return mode


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"unsupported activation: {name!r}")


def _linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    output = linear(x.reshape(-1, x.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def attention_softmax(
    scores: torch.Tensor,
    *,
    dim: int,
    output_dtype: torch.dtype,
    mode: str,
) -> torch.Tensor:
    if mode == "fp32":
        return F.softmax(scores, dim=dim, dtype=torch.float32).to(output_dtype)
    if mode == "model":
        return F.softmax(scores, dim=dim, dtype=output_dtype).to(output_dtype)
    raise ValueError(f"unsupported attention softmax dtype mode: {mode!r}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, seq_len, head_dim
    )


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    mrope_section = [int(value) for value in mrope_section] * 2
    cos = torch.cat(
        [
            part[i % 3]
            for i, part in enumerate(cos.split(mrope_section, dim=-1))
        ],
        dim=-1,
    )
    sin = torch.cat(
        [
            part[i % 3]
            for i, part in enumerate(sin.split(mrope_section, dim=-1))
        ],
        dim=-1,
    )
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def build_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_length: int = 0,
) -> torch.Tensor:
    batch_size, query_length = inputs_embeds.shape[:2]
    if attention_mask is None:
        kv_length = int(past_length + query_length)
        attention_mask = torch.ones(
            batch_size,
            kv_length,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
    else:
        kv_length = int(attention_mask.shape[-1])
    kv_positions = torch.arange(
        kv_length,
        device=inputs_embeds.device,
        dtype=cache_position.dtype,
    )
    allowed = kv_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
    allowed = allowed.reshape(1, 1, query_length, kv_length).expand(
        batch_size, 1, query_length, kv_length
    )
    padding_allowed = attention_mask[:, None, None, :kv_length].to(
        device=inputs_embeds.device, dtype=torch.bool
    )
    allowed = allowed & padding_allowed
    mask = torch.zeros(
        (batch_size, 1, query_length, kv_length),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    return mask.masked_fill(
        ~allowed, torch.finfo(inputs_embeds.dtype).min
    )


def update_prefill_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    sequence_length = int(key_states.shape[2])
    key_cache[:, :, :sequence_length, :].copy_(key_states.contiguous())
    value_cache[:, :, :sequence_length, :].copy_(value_states.contiguous())


class PaddleOCRRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class PaddleOCRRotaryEmbedding(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        rope = config.rope_parameters or {}
        self.base = float(rope.get("rope_theta", 500000.0))
        self.dim = int(config.head_dim)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)
        self.attention_scaling = 1.0

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq().to(device=device),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids = position_ids[:, :, None, :].float()
        freqs = (inv_freq * position_ids).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PaddleOCRMLP(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _linear_tokenwise(self.gate_proj, x)
        up = _linear_tokenwise(self.up_proj, x)
        return _linear_tokenwise(
            self.down_proj, _activation(self.hidden_act, gate) * up
        )


class PaddleOCRAttention(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.head_dim**-0.5
        self.mrope_section = list(
            (config.rope_parameters or {})["mrope_section"]
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, _hidden = hidden_states.shape
        query_states = _linear_tokenwise(
            self.q_proj, hidden_states
        ).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = _linear_tokenwise(
            self.k_proj, hidden_states
        ).view(
            batch,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = _linear_tokenwise(
            self.v_proj, hidden_states
        ).view(
            batch,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        return query_states, key_states, value_states

    def apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            position_embeddings[0],
            position_embeddings[1],
            self.mrope_section,
        )

    def attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _heads, query_length, _dim = query_states.shape
        key_for_attn = repeat_kv(
            key_states, self.num_key_value_groups
        )
        value_for_attn = repeat_kv(
            value_states, self.num_key_value_groups
        )
        scores = (
            torch.matmul(query_states, key_for_attn.transpose(2, 3))
            * self.scaling
        )
        if attention_mask is not None:
            scores = scores + attention_mask[
                :, :, :, : key_for_attn.shape[-2]
            ]
        probs = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=get_text_softmax_dtype_mode(),
        )
        attention_output = torch.matmul(probs, value_for_attn)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch, query_length, -1)
        )
        return _linear_tokenwise(self.o_proj, attention_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(
            query_states, key_states, position_embeddings
        )
        if past_key_values is not None:
            past_key, past_value = past_key_values[self.layer_idx]
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        new_past = (key_states, value_states) if use_cache else None
        return (
            self.attend(
                query_states, key_states, value_states, attention_mask
            ),
            new_past,
        )

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(
            query_states, key_states, position_embeddings
        )
        return (
            self.attend(
                query_states, key_states, value_states, attention_mask
            ),
            key_states,
            value_states,
        )


class PaddleOCRDecoderLayer(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = PaddleOCRAttention(config, layer_idx)
        self.mlp = PaddleOCRMLP(config)
        self.input_layernorm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past = self.self_attn(
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, new_past

    def apply_blocks(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: LocalPaddleOCRVLStaticCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, key_states, value_states = (
            self.self_attn.forward_prefill_static(
                hidden_states,
                attention_mask,
                position_embeddings,
            )
        )
        if cache is not None:
            key_cache, value_cache = cache.layer(self.layer_idx)
            update_prefill_kv_cache_(
                key_cache,
                value_cache,
                key_states,
                value_states,
            )
        return self.apply_blocks(residual, attention_output)


class PaddleOCRTextModel(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                PaddleOCRDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = PaddleOCRRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        past_length = (
            0
            if past_key_values is None
            else int(past_key_values[0][0].shape[2])
        )
        cache_position = torch.arange(
            past_length,
            past_length + inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        causal_mask = build_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
            past_length=past_length,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None
        for layer in self.layers:
            hidden_states, new_past = layer(
                hidden_states,
                causal_mask,
                position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_past)
        return self.norm(hidden_states), new_past_key_values

    def forward_prefill_static(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache: LocalPaddleOCRVLStaticCache | None = None,
    ) -> torch.Tensor:
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        causal_mask = build_causal_mask(
            inputs_embeds, attention_mask, cache_position
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer.forward_prefill_static(
                hidden_states,
                causal_mask,
                position_embeddings,
                cache=cache,
            )
        return self.norm(hidden_states)


def parse_text_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("text buckets cannot be empty")
        try:
            buckets = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"invalid text buckets: {value!r}") from exc
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets:
        raise ValueError("text buckets cannot be empty")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("every text bucket must be positive")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("text buckets must be unique and strictly increasing")
    return buckets


def select_text_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    real_seq_len = int(real_seq_len)
    if real_seq_len <= 0:
        raise ValueError("real text sequence length must be positive")
    for bucket in buckets:
        if real_seq_len <= int(bucket):
            return int(bucket)
    return None


class TextPrefillStage(torch.nn.Module):
    """Text prefill with flat mutable cache inputs for eager or compiled use."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration):
        super().__init__()
        self.text_model = model.model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.softmax_dtype_mode = get_text_softmax_dtype_mode()

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = attention.project_qkv(hidden_states)
        query_states, key_states = attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
        update_prefill_kv_cache_(
            key_cache,
            value_cache,
            key_states,
            value_states,
        )
        key_for_attn = repeat_kv(key_states, int(attention.num_key_value_groups))
        value_for_attn = repeat_kv(value_states, int(attention.num_key_value_groups))
        batch, num_heads, seq_length, head_dim = query_states.shape

        # GE mis-infers the broadcast axes of the stock 4-D matmul. Flattening
        # B and H produces the same arithmetic while presenting two ordinary
        # 3-D batched matrix multiplications to the compiler.
        query_bh = query_states.reshape(batch * num_heads, seq_length, head_dim)
        key_bh = key_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        value_bh = value_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
            batch,
            num_heads,
            seq_length,
            seq_length,
        ) * attention.scaling
        scores = scores + causal_mask
        probabilities = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=self.softmax_dtype_mode,
        )
        attention_output = torch.bmm(
            probabilities.reshape(batch * num_heads, seq_length, seq_length),
            value_bh,
        ).view(batch, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch,
            seq_length,
            num_heads * head_dim,
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_index: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        causal_mask = build_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
        )
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                causal_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        hidden_states = self.text_model.norm(hidden_states)
        return torch.index_select(hidden_states, 1, last_token_index)


def unique_bucket_forward(
    module: TextPrefillStage,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Give each static bucket a distinct Dynamo code object."""

    original = module.forward.__func__
    name = f"text_prefill_bucket_{int(bucket)}"
    code = original.__code__.replace(co_name=name)
    function = types.FunctionType(
        code,
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


@dataclass(frozen=True)
class PreparedTextPrefill:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    last_token_index: torch.Tensor
    real_seq_len: int
    physical_seq_len: int
    execution: str


def prepare_text_prefill(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    physical_seq_len: int,
    execution: str,
) -> PreparedTextPrefill:
    if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
        raise ValueError(
            "text prefill expects B=1 embeddings shaped [1, S, H], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    real_seq_len = int(inputs_embeds.shape[1])
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"real text sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )
    if tuple(attention_mask.shape) != (1, real_seq_len):
        raise ValueError(
            f"attention_mask must have shape {(1, real_seq_len)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if tuple(position_ids.shape) != (3, 1, real_seq_len):
        raise ValueError(
            f"position_ids must have shape {(3, 1, real_seq_len)}, "
            f"got {tuple(position_ids.shape)}"
        )

    pad_tokens = physical_seq_len - real_seq_len
    padded_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_tokens)).contiguous()
    padded_mask = F.pad(attention_mask, (0, pad_tokens), value=0).contiguous()
    # get_rope_index uses position 1 for masked/padded rows. The padded query
    # results are discarded, but preserving that convention keeps the graph's
    # unused rows well defined.
    padded_positions = F.pad(position_ids, (0, pad_tokens), value=1).contiguous()
    last_token_index = torch.tensor(
        [real_seq_len - 1],
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    return PreparedTextPrefill(
        inputs_embeds=padded_embeds,
        attention_mask=padded_mask,
        position_ids=padded_positions,
        last_token_index=last_token_index,
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
        execution=str(execution),
    )


def text_source_hash() -> str:
    return short_file_hash(Path(__file__).resolve())


def text_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        [
            "text_transformer_prefill",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"cache{int(cache_length)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{text_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class TextPrefillRuntime:
    """Run one text-prefill stage eagerly or through static bucket graphs."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        buckets: Iterable[int],
        cache_root: Path,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        padding: str = "auto",
    ):
        self.model = model
        self.backend = str(backend)
        self.buckets = parse_text_buckets(buckets)
        self.requested_padding = str(padding)
        if self.requested_padding not in TEXT_PADDING_CHOICES:
            raise ValueError(
                f"text padding must be one of {TEXT_PADDING_CHOICES}, got {padding!r}"
            )
        self.padding = (
            "bucket"
            if self.requested_padding == "auto" and self.backend == "torchair"
            else "none"
            if self.requested_padding == "auto"
            else self.requested_padding
        )
        if self.backend == "torchair" and self.padding != "bucket":
            raise ValueError("compiled text prefill requires bucket padding")
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_length = int(cache_length)
        self.device = device
        self.dtype = dtype
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.eager_stage = TextPrefillStage(model).eval()
        self.modules: dict[int, TextPrefillStage] = {}
        self.metadata: dict[str, Any] = {
            "backend": self.backend,
            "enabled": self.backend == "torchair",
            "boundary": "text_transformer_plus_in_place_prefill_kv_writes",
            "buckets": list(self.buckets),
            "requested_padding": self.requested_padding,
            "padding": self.padding,
            "overflow": (
                "eager_same_stage_unpadded"
                if self.padding == "bucket"
                else None
            ),
        }
        if self.backend not in TEXT_BACKEND_CHOICES:
            raise ValueError(
                f"text backend must be one of {TEXT_BACKEND_CHOICES}, got {backend!r}"
            )
        if self.padding == "bucket" and self.buckets[-1] > self.cache_length:
            raise ValueError(
                f"largest text bucket {self.buckets[-1]} exceeds cache length "
                f"{self.cache_length}"
            )
        if self.backend == "raw_eager":
            return
        if self.device.type != "npu":
            raise ValueError("compiled text backend torchair requires an NPU device")

        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.text_config.hidden_size)
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = TextPrefillStage(model).eval()
            cache_dir = text_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                cache_length=self.cache_length,
                dtype=self.dtype,
                device=self.device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = CompilerConfig()
            entrypoint = unique_bucket_forward(module, bucket)
            synchronize(self.device)
            started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(self.device)
            wrapper_s = time.perf_counter() - started

            warm_inputs = torch.zeros(
                (1, bucket, hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            warm_mask = torch.ones(
                (1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_positions = torch.zeros(
                (3, 1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_last_index = torch.tensor(
                [bucket - 1],
                device=self.device,
                dtype=torch.int64,
            )
            warm_cache = model.allocate_static_cache(
                batch_size=1,
                cache_length=self.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="zeros",
            )
            synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(
                warm_inputs,
                warm_mask,
                warm_positions,
                warm_last_index,
                *warm_cache.flat_tensors(),
            )
            synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_inputs, warm_mask, warm_positions, warm_last_index, warm_cache

            self.modules[bucket] = module
            self.entrypoints[bucket] = entrypoint
            self.compiled[bucket] = compiled
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            per_bucket[str(bucket)] = {
                "compile_wrapper_s": float(wrapper_s),
                "compile_first_call_s": float(first_call_s),
                "torchair_cache_dir": str(cache_dir),
            }
        self.metadata.update(
            {
                "compile_api": "torchair.inference.cache_compile",
                "dynamic": False,
                "fullgraph": True,
                "torchair_ge_cache": True,
                "compile_wrapper_total_s": float(wrapper_total_s),
                "compile_first_call_total_s": float(first_call_total_s),
                "per_bucket": per_bucket,
                "cache_key_fields": {
                    "cache_length": self.cache_length,
                    "dtype": str(dtype),
                    "linear_weight_format": linear_weight_format,
                    "model_config_hash": short_file_hash(model_dir / "config.json"),
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "text_source_hash": text_source_hash(),
                    "attention": "manual_causal",
                    "softmax_dtype": get_text_softmax_dtype_mode(),
                    "execution_mode": TORCHAIR_EXECUTION_MODE,
                },
            }
        )

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        bucket = (
            select_text_bucket(real_seq_len, self.buckets)
            if self.padding == "bucket"
            else None
        )
        if bucket is None:
            return {
                "execution": (
                    "eager_overflow"
                    if self.padding == "bucket"
                    else "eager"
                ),
                "real_text_tokens": real_seq_len,
                "physical_text_tokens": real_seq_len,
                "padding_text_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        return {
            "execution": (
                "compiled" if self.backend == "torchair" else "eager_padded"
            ),
            "real_text_tokens": real_seq_len,
            "physical_text_tokens": bucket,
            "padding_text_tokens": bucket - real_seq_len,
            "useful_token_fraction": float(real_seq_len) / float(bucket),
            "bucket": bucket,
        }

    def prepare(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        route: dict[str, Any],
    ) -> PreparedTextPrefill:
        return prepare_text_prefill(
            inputs_embeds,
            attention_mask,
            position_ids,
            physical_seq_len=int(route["physical_text_tokens"]),
            execution=str(route["execution"]),
        )

    def run_prepared(
        self,
        prepared: PreparedTextPrefill,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> torch.Tensor:
        run = (
            self.compiled[prepared.physical_seq_len]
            if prepared.execution == "compiled"
            else self.eager_stage
        )
        return run(
            prepared.inputs_embeds,
            prepared.attention_mask,
            prepared.position_ids,
            prepared.last_token_index,
            *cache.flat_tensors(),
        )
