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


FRACTAL_NZ = 29


@dataclass(frozen=True)
class DecodeOptimizationConfig:
    name: str
    npu_rms_norm: bool = False
    fused_add_rms_norm: bool = False
    packed_qkv: bool = False
    npu_rotary: bool = False
    rope_lookup: bool = False
    weight_prefetch: bool = False
    packed_mlp: bool = False
    npu_swiglu: bool = False


DECODE_OPTIMIZATION_PRESETS: dict[str, DecodeOptimizationConfig] = {
    "baseline": DecodeOptimizationConfig(name="baseline"),
    "npu_rms_norm": DecodeOptimizationConfig(
        name="npu_rms_norm",
        npu_rms_norm=True,
    ),
    "combined_apply": DecodeOptimizationConfig(
        name="combined_apply",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
    ),
    "combined_apply_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_rope_lut",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
        rope_lookup=True,
    ),
    "combined_apply_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
        rope_lookup=True,
        weight_prefetch=True,
    ),
    "combined_apply_packed_mlp": DecodeOptimizationConfig(
        name="combined_apply_packed_mlp",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
        packed_mlp=True,
    ),
    "combined_apply_packed_mlp_swiglu": DecodeOptimizationConfig(
        name="combined_apply_packed_mlp_swiglu",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
        packed_mlp=True,
        npu_swiglu=True,
    ),
    "combined_apply_packed_mlp_prefetch": DecodeOptimizationConfig(
        name="combined_apply_packed_mlp_prefetch",
        npu_rms_norm=True,
        fused_add_rms_norm=True,
        packed_qkv=True,
        npu_rotary=True,
        rope_lookup=True,
        weight_prefetch=True,
        packed_mlp=True,
    ),
}


def resolve_decode_optimization(
    name: str | DecodeOptimizationConfig,
) -> DecodeOptimizationConfig:
    if isinstance(name, DecodeOptimizationConfig):
        return name
    try:
        return DECODE_OPTIMIZATION_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported decode optimization {name!r}; expected "
            f"{tuple(DECODE_OPTIMIZATION_PRESETS)}"
        ) from exc


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
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32).view(1, 1, -1)
        position_ids = position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1)
        # This is an outer product over positions and inverse frequencies. Use
        # explicit broadcasting rather than a K=1 MatMul: the math is identical,
        # while TorchAir can infer the fixed decode shape without interpreting
        # the head dimension as the reduction axis.
        freqs = position_ids * inv_freq
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


def linear_tokenwise(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through TorchAir's unambiguous 2-D MatMul contract."""
    leading_shape = hidden_states.shape[:-1]
    output = linear(hidden_states.reshape(-1, hidden_states.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def _packed_linear(modules: tuple[nn.Linear, ...]) -> nn.Linear:
    """Create a decode-only projection with concatenated checkpoint weights."""
    first = modules[0]
    if any(module.in_features != first.in_features for module in modules):
        raise ValueError("packed Linear inputs must share in_features")
    if any(module.bias is not None for module in modules):
        raise ValueError("Qwen3 packed decode projections must be bias-free")
    packed = nn.Linear(
        first.in_features,
        sum(module.out_features for module in modules),
        bias=False,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        packed.weight.copy_(
            torch.cat([module.weight for module in modules], dim=0)
        )
    return packed


def _decode_rms_norm(
    norm: LocalQwen3RMSNorm,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if not optimization.npu_rms_norm or hidden_states.device.type != "npu":
        return norm(hidden_states)
    return torch_npu.npu_rms_norm(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
    )[0]


def _decode_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: LocalQwen3RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type != "npu":
        summed = x + residual
        return norm(summed), summed
    normalized, _rstd, summed = torch_npu.npu_add_rms_norm(
        x,
        residual,
        norm.weight,
        norm.variance_epsilon,
    )
    return normalized, summed


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
        gate = linear_tokenwise(self.gate_proj, hidden_states)
        up = linear_tokenwise(self.up_proj, hidden_states)
        return linear_tokenwise(self.down_proj, F.silu(gate) * up)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        optimization: DecodeOptimizationConfig,
    ) -> torch.Tensor:
        if optimization.packed_mlp:
            gate_up = linear_tokenwise(self.decode_gate_up_proj, hidden_states)
            if optimization.npu_swiglu and gate_up.device.type == "npu":
                activated = torch_npu.npu_swiglu(gate_up, dim=-1)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
            output = linear_tokenwise(self.down_proj, activated)
        else:
            output = self(hidden_states)
        if optimization.weight_prefetch and output.device.type == "npu":
            for weight in self._decode_prefetch_next_attention:
                torch_npu.npu_prefetch(
                    weight,
                    output,
                    int(weight.numel() * weight.element_size()),
                )
        return output


class LocalQwen3Attention(nn.Module):
    def __init__(
        self,
        config: LocalQwen3Config,
        *,
        decode_increfa_mode: str,
    ):
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
        return query_states, key_states, value_states

    def project_qkv_decode(
        self,
        hidden_states: torch.Tensor,
        optimization: DecodeOptimizationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, _hidden = hidden_states.shape
        if optimization.packed_qkv:
            packed = linear_tokenwise(self.decode_qkv_proj, hidden_states)
            query_size = self.num_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            query_states, key_states, value_states = packed.split(
                (query_size, kv_size, kv_size),
                dim=-1,
            )
        else:
            query_states = linear_tokenwise(self.q_proj, hidden_states)
            key_states = linear_tokenwise(self.k_proj, hidden_states)
            value_states = linear_tokenwise(self.v_proj, hidden_states)
        query_states = query_states.view(
            batch, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states = _decode_rms_norm(
            self.q_norm,
            query_states,
            optimization,
        ).transpose(1, 2)
        key_states = _decode_rms_norm(
            self.k_norm,
            key_states,
            optimization,
        ).transpose(1, 2)
        return query_states, key_states, value_states.transpose(1, 2)

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
        return linear_tokenwise(self.o_proj, attn_output)

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
        optimization: DecodeOptimizationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if optimization.weight_prefetch and hidden_states.device.type == "npu":
            for weight in self._decode_prefetch_current_mlp:
                torch_npu.npu_prefetch(
                    weight,
                    hidden_states,
                    int(weight.numel() * weight.element_size()),
                )
        query_states, key_states, value_states = self.project_qkv_decode(
            hidden_states,
            optimization,
        )
        if optimization.npu_rotary and query_states.device.type == "npu":
            query_bsnd = query_states.transpose(1, 2).contiguous()
            key_bsnd = key_states.transpose(1, 2).contiguous()
            query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
                query_bsnd,
                key_bsnd,
                cos.unsqueeze(1),
                sin.unsqueeze(1),
                layout="BSND",
                rotary_mode="half",
            )
            query_states = query_bsnd.transpose(1, 2)
            key_states = key_bsnd.transpose(1, 2)
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

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
        return linear_tokenwise(self.o_proj, attn_output), key_cache, value_cache


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
        optimization: DecodeOptimizationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, key_cache, value_cache = self.self_attn.forward_decode(
            _decode_rms_norm(
                self.input_layernorm,
                hidden_states,
                optimization,
            ),
            cos,
            sin,
            key_cache,
            value_cache,
            cache_position,
            attention_mask,
            actual_seq_length,
            optimization,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp.forward_decode(
            _decode_rms_norm(
                self.post_attention_layernorm,
                hidden_states,
                optimization,
            ),
            optimization,
        )
        return hidden_states, key_cache, value_cache

    def forward_decode_fused_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        actual_seq_length: int | None,
        optimization: DecodeOptimizationConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if residual is None:
            attention_input = _decode_rms_norm(
                self.input_layernorm,
                hidden_states,
                optimization,
            )
            residual = hidden_states
        else:
            attention_input, residual = _decode_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm,
            )
        attention_output, key_cache, value_cache = self.self_attn.forward_decode(
            attention_input,
            cos,
            sin,
            key_cache,
            value_cache,
            cache_position,
            attention_mask,
            actual_seq_length,
            optimization,
        )
        mlp_input, residual = _decode_add_rms_norm(
            attention_output,
            residual,
            self.post_attention_layernorm,
        )
        hidden_states = self.mlp.forward_decode(mlp_input, optimization)
        return hidden_states, residual, key_cache, value_cache


class LocalQwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: LocalQwen3Config,
        *,
        decode_increfa_mode: str = "mask",
        decode_optimization: str = "baseline",
    ):
        super().__init__()
        if decode_increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported decode_increfa_mode={decode_increfa_mode!r}")
        self.config = config
        self.decode_increfa_mode = decode_increfa_mode
        self.decode_optimization = resolve_decode_optimization(
            decode_optimization
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LocalQwen3DecoderLayer(config, decode_increfa_mode=decode_increfa_mode)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LocalQwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = LocalQwen3RotaryEmbedding(config)
        self.rotary_emb.register_buffer(
            "decode_factor_lut",
            None,
            persistent=False,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def prepare_decode_optimizations(self, *, cache_length: int) -> dict[str, object]:
        optimization = self.decode_optimization
        packed_qkv_count = 0
        packed_mlp_count = 0
        if optimization.packed_qkv:
            for layer in self.layers:
                attention = layer.self_attn
                if not hasattr(attention, "decode_qkv_proj"):
                    attention.decode_qkv_proj = _packed_linear(
                        (
                            attention.q_proj,
                            attention.k_proj,
                            attention.v_proj,
                        )
                    )
                    packed_qkv_count += 1
        if optimization.packed_mlp:
            for layer in self.layers:
                mlp = layer.mlp
                if not hasattr(mlp, "decode_gate_up_proj"):
                    mlp.decode_gate_up_proj = _packed_linear(
                        (mlp.gate_proj, mlp.up_proj)
                    )
                    packed_mlp_count += 1
        if optimization.rope_lookup:
            positions = torch.arange(
                int(cache_length),
                device=self.rotary_emb.inv_freq.device,
                dtype=torch.float32,
            ).view(-1, 1)
            freqs = positions * self.rotary_emb.inv_freq.float().view(1, -1)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.rotary_emb.decode_factor_lut = torch.stack(
                (emb.cos(), emb.sin()),
                dim=0,
            ).to(dtype=self.embed_tokens.weight.dtype)
        if optimization.weight_prefetch:
            for index, layer in enumerate(self.layers):
                layer.self_attn._decode_prefetch_current_mlp = (
                    *(
                        (layer.mlp.decode_gate_up_proj.weight,)
                        if optimization.packed_mlp
                        else (
                            layer.mlp.gate_proj.weight,
                            layer.mlp.up_proj.weight,
                        )
                    ),
                    layer.mlp.down_proj.weight,
                )
                if index + 1 < len(self.layers):
                    next_attention = self.layers[index + 1].self_attn
                    next_qkv = (
                        next_attention.decode_qkv_proj.weight,
                    ) if optimization.packed_qkv else (
                        next_attention.q_proj.weight,
                        next_attention.k_proj.weight,
                        next_attention.v_proj.weight,
                    )
                    layer.mlp._decode_prefetch_next_attention = (
                        *next_qkv,
                        next_attention.o_proj.weight,
                    )
                else:
                    layer.mlp._decode_prefetch_next_attention = (
                        self.lm_head.weight,
                    )
        return {
            "name": optimization.name,
            "packed_qkv_count": packed_qkv_count,
            "packed_mlp_count": packed_mlp_count,
            "npu_swiglu": optimization.npu_swiglu,
            "rope_lookup": optimization.rope_lookup,
            "rope_lookup_shape": (
                None
                if self.rotary_emb.decode_factor_lut is None
                else list(self.rotary_emb.decode_factor_lut.shape)
            ),
            "weight_prefetch": optimization.weight_prefetch,
        }

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
        optimization = self.decode_optimization
        hidden_states = self.embed_tokens(input_ids)
        if optimization.rope_lookup:
            selected = torch.index_select(
                self.rotary_emb.decode_factor_lut,
                1,
                position_ids.reshape(-1).to(dtype=torch.int64),
            )
            cos, sin = selected.unbind(dim=0)
            cos = cos.view(input_ids.shape[0], 1, self.config.head_dim)
            sin = sin.view(input_ids.shape[0], 1, self.config.head_dim)
        else:
            cos, sin = self.rotary_emb(
                position_ids,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        attention_mask = None
        if self.decode_increfa_mode == "mask":
            attention_mask = build_static_decode_mask(cache_position, key_caches[0].shape[2])
        next_key_caches = []
        next_value_caches = []
        if optimization.fused_add_rms_norm:
            residual: torch.Tensor | None = None
            for layer_idx, layer in enumerate(self.layers):
                (
                    hidden_states,
                    residual,
                    layer_key_cache,
                    layer_value_cache,
                ) = layer.forward_decode_fused_residual(
                    hidden_states,
                    residual,
                    cos,
                    sin,
                    key_caches[layer_idx],
                    value_caches[layer_idx],
                    cache_position,
                    attention_mask,
                    actual_seq_length,
                    optimization,
                )
                next_key_caches.append(layer_key_cache)
                next_value_caches.append(layer_value_cache)
            hidden_states, _residual = _decode_add_rms_norm(
                hidden_states,
                residual,
                self.norm,
            )
        else:
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
                    optimization,
                )
                next_key_caches.append(layer_key_cache)
                next_value_caches.append(layer_value_cache)
            hidden_states = _decode_rms_norm(
                self.norm,
                hidden_states,
                optimization,
            )
        output_head = getattr(self, "decode_lm_head", self.lm_head)
        logits = linear_tokenwise(output_head, hidden_states)
        return logits, tuple(next_key_caches), tuple(next_value_caches)

    def decode(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        actual_seq_length: int | None = None,
    ) -> torch.Tensor:
        position_ids = cache_position.reshape(-1, 1)
        logits, _key_caches, _value_caches = self.forward_decode(
            input_ids,
            position_ids,
            cache_position,
            key_caches,
            value_caches,
            actual_seq_length,
        )
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def prepare_decode_lm_head_copy(
    model: LocalQwen3ForCausalLM,
) -> nn.Linear:
    """Create an independent decode head when the checkpoint ties embeddings."""
    if hasattr(model, "decode_lm_head"):
        return model.decode_lm_head
    decode_lm_head = nn.Linear(
        model.config.hidden_size,
        model.config.vocab_size,
        bias=False,
        device=model.lm_head.weight.device,
        dtype=model.lm_head.weight.dtype,
    )
    with torch.no_grad():
        decode_lm_head.weight.copy_(model.lm_head.weight)
    decode_lm_head.requires_grad_(False)
    model.decode_lm_head = decode_lm_head
    return decode_lm_head


def _cast_linear_modules_to_nz(
    modules: list[tuple[str, nn.Linear]],
) -> dict[str, object]:
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    converted: list[str] = []
    for name, module in modules:
        before_format = int(torch_npu.get_npu_format(module.weight))
        before[name] = before_format
        if before_format != FRACTAL_NZ:
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data,
                FRACTAL_NZ,
            )
            converted.append(name)
        after_format = int(torch_npu.get_npu_format(module.weight))
        after[name] = after_format
        if after_format != FRACTAL_NZ:
            raise RuntimeError(
                "decode Linear weight did not retain FRACTAL_NZ: "
                f"module={name} before={before_format} after={after_format}"
            )
    return {
        "target_format": "FRACTAL_NZ",
        "target_format_code": FRACTAL_NZ,
        "linear_count": len(modules),
        "converted_count": len(converted),
        "already_nz_count": len(modules) - len(converted),
        "all_after_are_nz": all(value == FRACTAL_NZ for value in after.values()),
        "before_format_histogram": {
            str(code): list(before.values()).count(code)
            for code in sorted(set(before.values()))
        },
        "after_format_histogram": {
            str(code): list(after.values()).count(code)
            for code in sorted(set(after.values()))
        },
        "converted_modules_sample": converted[:16],
    }


def cast_decode_lm_head_to_nz(
    model: LocalQwen3ForCausalLM,
) -> dict[str, object]:
    """Convert only an independent decode LM head to FRACTAL_NZ."""
    decode_lm_head = prepare_decode_lm_head_copy(model)
    return _cast_linear_modules_to_nz(
        [("decode_lm_head", decode_lm_head)]
    )


def cast_decode_linear_weights_to_nz(
    model: LocalQwen3ForCausalLM,
) -> dict[str, object]:
    """Convert only the linears consumed by one-token decode to FRACTAL_NZ."""
    modules: list[tuple[str, nn.Linear]] = []
    optimization = model.decode_optimization
    for layer_index, layer in enumerate(model.layers):
        attention = layer.self_attn
        if optimization.packed_qkv:
            modules.append(
                (
                    f"layers.{layer_index}.self_attn.decode_qkv_proj",
                    attention.decode_qkv_proj,
                )
            )
        else:
            modules.extend(
                (
                    (f"layers.{layer_index}.self_attn.q_proj", attention.q_proj),
                    (f"layers.{layer_index}.self_attn.k_proj", attention.k_proj),
                    (f"layers.{layer_index}.self_attn.v_proj", attention.v_proj),
                )
            )
        modules.append(
            (f"layers.{layer_index}.self_attn.o_proj", attention.o_proj)
        )
        if optimization.packed_mlp:
            modules.append(
                (
                    f"layers.{layer_index}.mlp.decode_gate_up_proj",
                    layer.mlp.decode_gate_up_proj,
                )
            )
        else:
            modules.extend(
                (
                    (f"layers.{layer_index}.mlp.gate_proj", layer.mlp.gate_proj),
                    (f"layers.{layer_index}.mlp.up_proj", layer.mlp.up_proj),
                )
            )
        modules.append(
            (f"layers.{layer_index}.mlp.down_proj", layer.mlp.down_proj)
        )
    output_head = (
        prepare_decode_lm_head_copy(model)
        if model.config.tie_word_embeddings
        else model.lm_head
    )
    output_head_name = (
        "decode_lm_head" if output_head is not model.lm_head else "lm_head"
    )
    modules.append((output_head_name, output_head))
    return _cast_linear_modules_to_nz(modules)
