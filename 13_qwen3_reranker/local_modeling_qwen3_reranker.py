#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


PROMPT_FA_FULL_ATTENTION_TOKENS = (1 << 31) - 1
FRACTAL_NZ = 29
RERANKER_LINEAR_WEIGHT_FORMAT_CHOICES = ("native", "fractal_nz")


@dataclass(frozen=True)
class RerankerPrefillOptimizationConfig:
    name: str
    native_rms_norm: bool = False
    native_rotary: bool = False
    prebuilt_square_mask: bool = False
    expanded_prefix_kv: bool = False
    prompt_fa_layout: str = "BNSD"


PREFILL_OPTIMIZATION_PRESETS: dict[str, RerankerPrefillOptimizationConfig] = {
    "baseline": RerankerPrefillOptimizationConfig(name="baseline"),
    "native_rms": RerankerPrefillOptimizationConfig(
        name="native_rms",
        native_rms_norm=True,
    ),
    "native_rotary": RerankerPrefillOptimizationConfig(
        name="native_rotary",
        native_rotary=True,
    ),
    "prebuilt_square_mask": RerankerPrefillOptimizationConfig(
        name="prebuilt_square_mask",
        prebuilt_square_mask=True,
    ),
    "expanded_prefix_kv": RerankerPrefillOptimizationConfig(
        name="expanded_prefix_kv",
        expanded_prefix_kv=True,
    ),
    "native_rms_rotary": RerankerPrefillOptimizationConfig(
        name="native_rms_rotary",
        native_rms_norm=True,
        native_rotary=True,
    ),
    "native_rms_rotary_mask": RerankerPrefillOptimizationConfig(
        name="native_rms_rotary_mask",
        native_rms_norm=True,
        native_rotary=True,
        prebuilt_square_mask=True,
    ),
    "combined": RerankerPrefillOptimizationConfig(
        name="combined",
        native_rms_norm=True,
        native_rotary=True,
        prebuilt_square_mask=True,
    ),
    "combined_bsnd": RerankerPrefillOptimizationConfig(
        name="combined_bsnd",
        native_rms_norm=True,
        native_rotary=True,
        prebuilt_square_mask=True,
        prompt_fa_layout="BSND",
    ),
}


def resolve_prefill_optimization(
    optimization: str | RerankerPrefillOptimizationConfig,
) -> RerankerPrefillOptimizationConfig:
    if isinstance(optimization, RerankerPrefillOptimizationConfig):
        return optimization
    try:
        return PREFILL_OPTIMIZATION_PRESETS[str(optimization)]
    except KeyError as exc:
        raise ValueError(
            f"unknown prefill optimization {optimization!r}; expected one of "
            f"{tuple(PREFILL_OPTIMIZATION_PRESETS)}"
        ) from exc


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
        self.native_npu = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.native_npu and hidden_states.device.type == "npu":
            import torch_npu

            return torch_npu.npu_rms_norm(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )[0]
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


def apply_native_rotary_pos_emb_bsnd(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one native half-RoPE call to BSND Q/K tensors."""
    import torch_npu

    return torch_npu.npu_apply_rotary_pos_emb(
        query_states.contiguous(),
        key_states.contiguous(),
        cos.unsqueeze(2).contiguous(),
        sin.unsqueeze(2).contiguous(),
        layout="BSND",
        rotary_mode="half",
    )


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, repeats, sequence_length, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * repeats, sequence_length, head_dim)


def repeat_kv_bsnd(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, sequence_length, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch,
        sequence_length,
        num_kv_heads,
        repeats,
        head_dim,
    )
    return hidden_states.reshape(batch, sequence_length, num_kv_heads * repeats, head_dim)


def repeat_kv_for_layout(
    hidden_states: torch.Tensor,
    repeats: int,
    input_layout: str,
) -> torch.Tensor:
    if input_layout == "BNSD":
        return repeat_kv(hidden_states, repeats)
    if input_layout == "BSND":
        return repeat_kv_bsnd(hidden_states, repeats)
    raise ValueError(f"unsupported PromptFA input layout {input_layout!r}")


def build_310p_square_promptfa_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Prepend valid dummy rows so masked PromptFA receives square Q/K."""
    if attention_mask.ndim != 4 or int(attention_mask.shape[1]) != 1:
        raise ValueError("PromptFA attention mask must have shape [B,1,Q,K]")
    batch, _one, query_length, key_length = attention_mask.shape
    if int(query_length) > int(key_length):
        raise ValueError("PromptFA square-mask preparation requires Q length <= K length")
    padded_query_rows = int(key_length) - int(query_length)
    if padded_query_rows == 0:
        return attention_mask.to(dtype=torch.bool).contiguous()
    dummy_query_positions = torch.arange(
        padded_query_rows,
        device=attention_mask.device,
        dtype=torch.int64,
    ).view(1, 1, padded_query_rows, 1)
    key_positions = torch.arange(
        int(key_length),
        device=attention_mask.device,
        dtype=torch.int64,
    ).view(1, 1, 1, int(key_length))
    dummy_mask = (dummy_query_positions != key_positions).expand(
        int(batch),
        1,
        padded_query_rows,
        int(key_length),
    )
    return torch.cat(
        (dummy_mask, attention_mask.to(dtype=torch.bool)),
        dim=2,
    ).contiguous()


def linear_tokenwise(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply Linear through the 2-D token matrix expected by GE."""
    leading_shape = hidden_states.shape[:-1]
    output = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def prompt_flash_attention_310p_compatible(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    num_heads: int,
    scale: float,
    input_layout: str,
    attention_mask_is_square: bool = False,
) -> torch.Tensor:
    """Run the Atlas inference-series PromptFA contract with native GQA.

    Atlas 310P does not support the PromptFA actual-sequence-length inputs or a
    rectangular masked Q/K shape. Keep compact GQA key/value heads and pass
    their real count to PromptFA. Square-pad projected Q with disposable rows
    when Q is shorter than K, encode left padding and causality in a bool mask,
    and omit actual-sequence-length arguments.
    """
    if query_states.dtype != torch.float16:
        raise ValueError("310P-compatible prompt_flash_attention requires float16 Q/K/V")
    if key_states.dtype != query_states.dtype or value_states.dtype != query_states.dtype:
        raise ValueError("prompt_flash_attention requires matching Q/K/V dtypes")
    if input_layout not in {"BNSD", "BSND"}:
        raise ValueError(f"unsupported PromptFA input layout {input_layout!r}")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError(f"{input_layout} prompt_flash_attention requires rank-4 Q/K/V tensors")
    head_axis = 1 if input_layout == "BNSD" else 2
    sequence_axis = 2 if input_layout == "BNSD" else 1
    if int(query_states.shape[head_axis]) != int(num_heads):
        raise ValueError("num_heads must match the query N dimension")
    if key_states.shape != value_states.shape:
        raise ValueError("key and value shapes must match")

    num_key_value_heads = int(key_states.shape[head_axis])
    if int(num_heads) % num_key_value_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    query_length = int(query_states.shape[sequence_axis])
    key_length = int(key_states.shape[sequence_axis])
    if query_length > key_length:
        raise ValueError("310P-compatible masked PromptFA requires Q length <= K length")
    mask_query_length = key_length if attention_mask_is_square else query_length
    expected_mask_shape = (
        int(query_states.shape[0]),
        1,
        mask_query_length,
        key_length,
    )
    if tuple(attention_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"expected attention mask shape {expected_mask_shape}, got {tuple(attention_mask.shape)}"
        )

    padded_query_rows = key_length - query_length
    if padded_query_rows:
        if input_layout == "BNSD":
            dummy_shape = (
                int(query_states.shape[0]),
                int(query_states.shape[1]),
                padded_query_rows,
                int(query_states.shape[3]),
            )
        else:
            dummy_shape = (
                int(query_states.shape[0]),
                padded_query_rows,
                int(query_states.shape[2]),
                int(query_states.shape[3]),
            )
        dummy_queries = query_states.new_zeros(dummy_shape)
        query_states = torch.cat(
            (dummy_queries, query_states),
            dim=sequence_axis,
        ).contiguous()

        if not attention_mask_is_square:
            attention_mask = build_310p_square_promptfa_mask(attention_mask)

    try:
        import torch_npu
    except Exception as exc:
        raise RuntimeError(f"torch_npu import failed for prompt flash attention: {exc}") from exc

    attention_output = torch_npu.npu_prompt_flash_attention(
        query_states.contiguous(),
        key_states,
        value_states,
        atten_mask=attention_mask.to(dtype=torch.bool).contiguous(),
        num_heads=int(num_heads),
        num_key_value_heads=num_key_value_heads,
        input_layout=input_layout,
        scale_value=float(scale),
        pre_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
        next_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
        sparse_mode=0,
    )
    if padded_query_rows:
        if input_layout == "BNSD":
            attention_output = attention_output[:, :, -query_length:, :]
        else:
            attention_output = attention_output[:, -query_length:, :, :]
    return attention_output


def prompt_flash_attention_bnsd_310p_compatible(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    num_heads: int,
    scale: float,
    attention_mask_is_square: bool = False,
) -> torch.Tensor:
    return prompt_flash_attention_310p_compatible(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        num_heads=num_heads,
        scale=scale,
        input_layout="BNSD",
        attention_mask_is_square=attention_mask_is_square,
    )


def prompt_flash_attention_bsnd_310p_compatible(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    num_heads: int,
    scale: float,
    attention_mask_is_square: bool = False,
) -> torch.Tensor:
    return prompt_flash_attention_310p_compatible(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        num_heads=num_heads,
        scale=scale,
        input_layout="BSND",
        attention_mask_is_square=attention_mask_is_square,
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
        self.qkv_proj: nn.Linear | None = None
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = LocalQwen3RerankerRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = LocalQwen3RerankerRMSNorm(config.head_dim, config.rms_norm_eps)
        self.native_rotary = False
        self.prebuilt_square_mask = False
        self.expanded_prefix_kv = False
        self.prompt_fa_layout = "BNSD"

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        output_layout: str = "BNSD",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if output_layout not in {"BNSD", "BSND"}:
            raise ValueError(f"unsupported projected QKV layout {output_layout!r}")
        batch, sequence_length, _hidden = hidden_states.shape
        if self.qkv_proj is None:
            query_states = linear_tokenwise(self.q_proj, hidden_states)
            key_states = linear_tokenwise(self.k_proj, hidden_states)
            value_states = linear_tokenwise(self.v_proj, hidden_states)
        elif getattr(self.qkv_proj, "returns_separate_qkv", False):
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            projected = linear_tokenwise(self.qkv_proj, hidden_states)
            query_width = self.num_heads * self.head_dim
            kv_width = self.num_key_value_heads * self.head_dim
            query_states = projected[..., :query_width]
            key_states = projected[..., query_width : query_width + kv_width]
            value_states = projected[..., query_width + kv_width :]
        query_states = query_states.view(
            batch, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        if self.native_rotary and query_states.device.type == "npu":
            query_states, key_states = apply_native_rotary_pos_emb_bsnd(
                query_states,
                key_states,
                cos,
                sin,
            )
            if output_layout == "BNSD":
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            if output_layout == "BSND":
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
        if output_layout == "BNSD":
            value_states = value_states.transpose(1, 2)
        return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()

    def merge_attention_heads(
        self,
        attention_output: torch.Tensor,
        *,
        batch: int,
        sequence_length: int,
    ) -> torch.Tensor:
        if self.prompt_fa_layout == "BNSD":
            attention_output = attention_output.transpose(1, 2)
        elif self.prompt_fa_layout != "BSND":
            raise ValueError(f"unsupported PromptFA layout {self.prompt_fa_layout!r}")
        return attention_output.reshape(
            batch,
            sequence_length,
            self.num_heads * self.head_dim,
        )

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
        query_states, key_states, value_states = self.project_qkv(
            hidden_states,
            cos,
            sin,
            output_layout=self.prompt_fa_layout,
        )

        attn_output = prompt_flash_attention_310p_compatible(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            num_heads=int(self.num_heads),
            scale=float(self.scaling),
            input_layout=self.prompt_fa_layout,
            attention_mask_is_square=self.prebuilt_square_mask,
        )
        attn_output = self.merge_attention_heads(
            attn_output,
            batch=batch,
            sequence_length=sequence_length,
        )
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
            hidden_states,
            cos,
            sin,
            output_layout=self.prompt_fa_layout,
        )
        if (past_key_states is None) != (past_value_states is None):
            raise ValueError("past key and value states must both be present or absent")
        if past_key_states is None:
            key_states = current_key_states
            value_states = current_value_states
        else:
            sequence_axis = 2 if self.prompt_fa_layout == "BNSD" else 1
            key_states = torch.cat(
                (past_key_states, current_key_states),
                dim=sequence_axis,
            ).contiguous()
            value_states = torch.cat(
                (past_value_states, current_value_states),
                dim=sequence_axis,
            ).contiguous()

        attn_output = prompt_flash_attention_310p_compatible(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            num_heads=int(self.num_heads),
            scale=float(self.scaling),
            input_layout=self.prompt_fa_layout,
        )
        attn_output = self.merge_attention_heads(
            attn_output,
            batch=batch,
            sequence_length=query_length,
        )
        return (
            linear_tokenwise(self.o_proj, attn_output),
            key_states,
            value_states,
        )

    def forward_eager_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, _hidden = hidden_states.shape
        query_states, key_states, value_states = self.project_qkv(hidden_states, cos, sin)
        key_for_attention = repeat_kv(key_states, self.num_key_value_groups)
        value_for_attention = repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_for_attention.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.matmul(probabilities, value_for_attention)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch, sequence_length, self.num_heads * self.head_dim
        )
        return linear_tokenwise(self.o_proj, attention_output), key_states, value_states

    def forward_prompt_flash_attention_cached_prefix(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_key_states: torch.Tensor,
        prefix_value_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, _hidden = hidden_states.shape
        query_states, current_key_states, current_value_states = self.project_qkv(
            hidden_states,
            cos,
            sin,
            output_layout=self.prompt_fa_layout,
        )
        if self.expanded_prefix_kv:
            head_axis = 1 if self.prompt_fa_layout == "BNSD" else 2
            if int(prefix_key_states.shape[head_axis]) != int(self.num_heads):
                raise ValueError("expanded prefix K/V must use the query-head count")
            current_key_states = repeat_kv_for_layout(
                current_key_states,
                self.num_key_value_groups,
                self.prompt_fa_layout,
            ).contiguous()
            current_value_states = repeat_kv_for_layout(
                current_value_states,
                self.num_key_value_groups,
                self.prompt_fa_layout,
            ).contiguous()
        sequence_axis = 2 if self.prompt_fa_layout == "BNSD" else 1
        key_states = torch.cat(
            (prefix_key_states, current_key_states),
            dim=sequence_axis,
        ).contiguous()
        value_states = torch.cat(
            (prefix_value_states, current_value_states),
            dim=sequence_axis,
        ).contiguous()
        attention_output = prompt_flash_attention_310p_compatible(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            num_heads=int(self.num_heads),
            scale=float(self.scaling),
            input_layout=self.prompt_fa_layout,
            attention_mask_is_square=self.prebuilt_square_mask,
        )
        attention_output = self.merge_attention_heads(
            attention_output,
            batch=batch,
            sequence_length=query_length,
        )
        return linear_tokenwise(self.o_proj, attention_output)

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

    def forward_eager_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_output, key_states, value_states = self.self_attn.forward_eager_prefix_cache(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key_states, value_states

    def forward_prompt_flash_attention_cached_prefix(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_key_states: torch.Tensor,
        prefix_value_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.forward_prompt_flash_attention_cached_prefix(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
            prefix_key_states,
            prefix_value_states,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


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
        self.prefill_optimization = PREFILL_OPTIMIZATION_PRESETS["baseline"]

    def set_prefill_optimization(
        self,
        optimization: str | RerankerPrefillOptimizationConfig,
    ) -> RerankerPrefillOptimizationConfig:
        config = resolve_prefill_optimization(optimization)
        if config.prompt_fa_layout not in {"BNSD", "BSND"}:
            raise ValueError(
                f"unsupported PromptFA input layout {config.prompt_fa_layout!r}"
            )
        self.prefill_optimization = config
        for module in self.modules():
            if isinstance(module, LocalQwen3RerankerRMSNorm):
                module.native_npu = config.native_rms_norm
        for layer in self.layers:
            attention = layer.self_attn
            attention.native_rotary = config.native_rotary
            attention.prebuilt_square_mask = config.prebuilt_square_mask
            attention.expanded_prefix_kv = config.expanded_prefix_kv
            attention.prompt_fa_layout = config.prompt_fa_layout
        return config

    def prepare_prefix_caches(
        self,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        prepared_keys = key_caches
        prepared_values = value_caches
        if self.prefill_optimization.expanded_prefix_kv:
            repeats = int(self.config.num_attention_heads // self.config.num_key_value_heads)
            prepared_keys = tuple(
                repeat_kv(cache, repeats).contiguous() for cache in prepared_keys
            )
            prepared_values = tuple(
                repeat_kv(cache, repeats).contiguous() for cache in prepared_values
            )
        if self.prefill_optimization.prompt_fa_layout == "BSND":
            prepared_keys = tuple(
                cache.transpose(1, 2).contiguous() for cache in prepared_keys
            )
            prepared_values = tuple(
                cache.transpose(1, 2).contiguous() for cache in prepared_values
            )
        return prepared_keys, prepared_values

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

    def build_prefix_cache_eager(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            position_ids,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        key_caches: list[torch.Tensor] = []
        value_caches: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, key_states, value_states = layer.forward_eager_prefix_cache(
                hidden_states,
                cos,
                sin,
                prefix_mask,
            )
            key_caches.append(key_states)
            value_caches.append(value_states)
        return tuple(key_caches), tuple(value_caches)

    def forward_cached_suffix_prepared(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        continuation_mask: torch.Tensor,
        prefix_key_caches: tuple[torch.Tensor, ...],
        prefix_value_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            position_ids,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.forward_prompt_flash_attention_cached_prefix(
                hidden_states,
                cos,
                sin,
                continuation_mask,
                prefix_key_caches[layer_index],
                prefix_value_caches[layer_index],
            )
        return self.norm(hidden_states)

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


def reranker_transformer_linears(
    model: LocalQwen3RerankerForCausalLM,
) -> tuple[tuple[str, nn.Linear], ...]:
    """Return the remaining dense transformer projections.

    Quantized wrappers replace some projections with non-Linear modules, so
    discover the dense remainder instead of assuming seven Linears per layer.
    """
    projection_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("layers.")
        and name.rsplit(".", 1)[-1] in projection_names
        and isinstance(module, nn.Linear)
    ]
    return tuple(modules)


def fuse_reranker_qkv_projections_inplace(
    model: LocalQwen3RerankerForCausalLM,
) -> int:
    """Replace each attention layer's three FP16 Q/K/V linears with one."""
    fused_count = 0
    for layer in model.layers:
        attention = layer.self_attn
        if attention.qkv_proj is not None:
            continue
        q_proj = attention.q_proj
        k_proj = attention.k_proj
        v_proj = attention.v_proj
        if not all(isinstance(proj, nn.Linear) for proj in (q_proj, k_proj, v_proj)):
            raise TypeError("QKV fusion requires dense nn.Linear projections")
        fused = nn.Linear(
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            bias=False,
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat((q_proj.weight, k_proj.weight, v_proj.weight), dim=0)
            )
        attention.qkv_proj = fused
        attention.q_proj = None
        attention.k_proj = None
        attention.v_proj = None
        fused_count += 1
    return fused_count


def prepare_reranker_linear_weight_format(
    model: LocalQwen3RerankerForCausalLM,
    *,
    requested: str,
) -> dict[str, object]:
    """Precast the timed transformer Linear weights to FRACTAL_NZ.

    The checkpoint ties ``lm_head`` to ``embed_tokens``. Keep that shared table
    in ND for embedding lookup and for the tiny yes/no projection outside the
    compiled prefill graph.
    """
    requested = str(requested)
    if requested not in RERANKER_LINEAR_WEIGHT_FORMAT_CHOICES:
        raise ValueError(
            "reranker linear weight format must be one of "
            f"{RERANKER_LINEAR_WEIGHT_FORMAT_CHOICES}, got {requested!r}"
        )
    modules = reranker_transformer_linears(model)
    non_npu = [
        (name, str(module.weight.device))
        for name, module in modules
        if module.weight.device.type != "npu"
    ]
    if non_npu:
        raise RuntimeError(
            "reranker Linear format preparation requires NPU-resident weights: "
            f"{non_npu[:4]}"
        )

    import torch_npu

    def histogram() -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    str(int(torch_npu.get_npu_format(module.weight)))
                    for _name, module in modules
                ).items()
            )
        )

    before = histogram()
    converted: list[str] = []
    if requested == "fractal_nz":
        for name, module in modules:
            before_code = int(torch_npu.get_npu_format(module.weight))
            if before_code == FRACTAL_NZ:
                continue
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data,
                FRACTAL_NZ,
            )
            after_code = int(torch_npu.get_npu_format(module.weight))
            if after_code != FRACTAL_NZ:
                raise RuntimeError(
                    "reranker Linear format cast did not produce FRACTAL_NZ: "
                    f"module={name} before={before_code} after={after_code}"
                )
            converted.append(name)
    after = histogram()
    all_after_are_nz = all(
        int(torch_npu.get_npu_format(module.weight)) == FRACTAL_NZ
        for _name, module in modules
    )
    if requested == "fractal_nz" and not all_after_are_nz:
        raise RuntimeError(
            f"not all reranker Linear weights are FRACTAL_NZ after conversion: {after}"
        )
    return {
        "requested": requested,
        "effective_mode": requested,
        "target_format": "FRACTAL_NZ" if requested == "fractal_nz" else "unchanged",
        "target_format_code": FRACTAL_NZ if requested == "fractal_nz" else None,
        "linear_weight_count": len(modules),
        "converted_count": len(converted),
        "converted_modules_sample": converted[:16],
        "before_format_histogram": before,
        "after_format_histogram": after,
        "all_after_are_nz": all_after_are_nz,
        "excluded_shared_weight": {
            "name": "embed_tokens.weight/lm_head.weight",
            "reason": "embedding lookup and yes/no projection stay outside the compiled prefill graph",
        },
    }
