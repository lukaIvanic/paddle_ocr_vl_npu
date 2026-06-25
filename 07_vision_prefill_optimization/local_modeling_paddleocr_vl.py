#!/usr/bin/env python3
"""Local PaddleOCR-VL recognizer implementation without Transformers imports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import PaddleOCRTextConfig, PaddleOCRVLConfig, PaddleOCRVisionConfig


FRACTAL_NZ = 29
DECODE_LINEAR_WEIGHT_FORMAT = "decode_nz"
DECODE_ATTENTION = "increfa"
DECODE_CACHE_UPDATE = "npu_scatter"
VISION_ATTENTION_ENV = "PADDLE_OCR_VL_VISION_ATTENTION"
VISION_ATTENTION_CHOICES = ("manual", "prompt_flash_attention")
VISION_PROMPT_FA_LAYOUT_ENV = "PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT"
VISION_PROMPT_FA_LAYOUT_CHOICES = ("bnsd", "bsnd", "bsh")
VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV = "PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE"
TEXT_SOFTMAX_DTYPE_ENV = "PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE"
VISION_SOFTMAX_DTYPE_ENV = "PADDLE_OCR_VL_VISION_SOFTMAX_DTYPE"
SOFTMAX_DTYPE_CHOICES = ("fp32", "model")


def get_vision_attention_impl() -> str:
    mode = os.environ.get(VISION_ATTENTION_ENV, "manual").strip() or "manual"
    if mode not in VISION_ATTENTION_CHOICES:
        raise ValueError(f"{VISION_ATTENTION_ENV} must be one of {VISION_ATTENTION_CHOICES}, got {mode!r}")
    return mode


def get_vision_prompt_fa_layout() -> str:
    layout = os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd").strip().lower() or "bnsd"
    if layout not in VISION_PROMPT_FA_LAYOUT_CHOICES:
        raise ValueError(f"{VISION_PROMPT_FA_LAYOUT_ENV} must be one of {VISION_PROMPT_FA_LAYOUT_CHOICES}, got {layout!r}")
    return layout


def get_vision_prompt_fa_mask_sparse_mode() -> int:
    raw = os.environ.get(VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV, "1").strip() or "1"
    try:
        mode = int(raw)
    except ValueError as exc:
        raise ValueError(f"{VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV} must be an integer, got {raw!r}") from exc
    if mode not in (0, 1):
        raise ValueError(f"{VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV} must be 0 or 1 for vision padding masks, got {mode}")
    return mode


def get_text_softmax_dtype_mode() -> str:
    mode = os.environ.get(TEXT_SOFTMAX_DTYPE_ENV, "fp32").strip().lower() or "fp32"
    if mode not in SOFTMAX_DTYPE_CHOICES:
        raise ValueError(f"{TEXT_SOFTMAX_DTYPE_ENV} must be one of {SOFTMAX_DTYPE_CHOICES}, got {mode!r}")
    return mode


def get_vision_softmax_dtype_mode() -> str:
    mode = os.environ.get(VISION_SOFTMAX_DTYPE_ENV, "fp32").strip().lower() or "fp32"
    if mode not in SOFTMAX_DTYPE_CHOICES:
        raise ValueError(f"{VISION_SOFTMAX_DTYPE_ENV} must be one of {SOFTMAX_DTYPE_CHOICES}, got {mode!r}")
    return mode


def attention_softmax(scores: torch.Tensor, *, dim: int, output_dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return F.softmax(scores, dim=dim, dtype=torch.float32).to(output_dtype)
    if mode == "model":
        return F.softmax(scores, dim=dim, dtype=output_dtype).to(output_dtype)
    raise ValueError(f"unsupported attention softmax dtype mode: {mode!r}")


def vision_prompt_flash_attention_bnsd(
    q_bnsd: torch.Tensor,
    k_bnsd: torch.Tensor,
    v_bnsd: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    layout: str | None = None,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run PromptFA with a selectable public layout and return BNSD output."""
    if q_bnsd.device.type != "npu":
        raise RuntimeError("vision prompt_flash_attention requires NPU tensors plus torch_npu.")
    import torch_npu

    selected_layout = get_vision_prompt_fa_layout() if layout is None else layout.strip().lower()
    if selected_layout not in VISION_PROMPT_FA_LAYOUT_CHOICES:
        raise ValueError(f"unsupported vision PromptFA layout: {selected_layout!r}")

    mask_kwargs = {}
    sparse_mode = 0
    if atten_mask is not None:
        mask_kwargs["atten_mask"] = atten_mask.to(torch.bool).contiguous()
        sparse_mode = get_vision_prompt_fa_mask_sparse_mode()

    if selected_layout == "bnsd":
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd.contiguous(),
            k_bnsd.contiguous(),
            v_bnsd.contiguous(),
            num_heads=int(num_heads),
            input_layout="BNSD",
            scale_value=float(scale),
            sparse_mode=sparse_mode,
            **mask_kwargs,
        )

    if selected_layout == "bsnd":
        out_bsnd = torch_npu.npu_prompt_flash_attention(
            q_bnsd.transpose(1, 2).contiguous(),
            k_bnsd.transpose(1, 2).contiguous(),
            v_bnsd.transpose(1, 2).contiguous(),
            num_heads=int(num_heads),
            input_layout="BSND",
            scale_value=float(scale),
            sparse_mode=sparse_mode,
            **mask_kwargs,
        )
        return out_bsnd.transpose(1, 2).contiguous()

    batch, heads, seq_len, head_dim = q_bnsd.shape
    out_bsh = torch_npu.npu_prompt_flash_attention(
        q_bnsd.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim),
        k_bnsd.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim),
        v_bnsd.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim),
        num_heads=int(num_heads),
        input_layout="BSH",
        scale_value=float(scale),
        sparse_mode=sparse_mode,
        **mask_kwargs,
    )
    return out_bsh.view(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()


def _resolve_model_dir(model_id_or_path: str | Path) -> Path:
    raw = str(model_id_or_path).strip()
    if not raw:
        raise FileNotFoundError(
            "Experiment 07 requires --model to point to a local PaddleOCR-VL model directory; "
            "Hugging Face download is disabled."
        )
    path = Path(raw).expanduser()
    if path.exists():
        path = path.resolve()
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"--model {str(path)!r} exists but is missing required PaddleOCR-VL files: {missing}"
            )
        return path
    raise FileNotFoundError(
        "Experiment 07 requires a local PaddleOCR-VL model directory; Hugging Face download is "
        f"disabled for this benchmark. Got --model {str(model_id_or_path)!r}. On the NPU work box, "
        "use --model /home/lukaiv/models/paddle_ocr_0_9b_v_1_6 or set MODEL to that path."
    )


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"unsupported activation: {name!r}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    mrope_section = [int(value) for value in mrope_section] * 2
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def build_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_length: int = 0,
) -> torch.Tensor:
    batch_size, query_length = inputs_embeds.shape[:2]
    if attention_mask is None:
        kv_length = int(past_length + query_length)
        attention_mask = torch.ones(batch_size, kv_length, device=inputs_embeds.device, dtype=torch.long)
    else:
        kv_length = int(attention_mask.shape[-1])
    kv_positions = torch.arange(kv_length, device=inputs_embeds.device, dtype=cache_position.dtype)
    allowed = kv_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
    allowed = allowed.reshape(1, 1, query_length, kv_length).expand(batch_size, 1, query_length, kv_length)
    padding_allowed = attention_mask[:, None, None, :kv_length].to(device=inputs_embeds.device, dtype=torch.bool)
    allowed = allowed & padding_allowed
    mask = torch.zeros((batch_size, 1, query_length, kv_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return mask.masked_fill(~allowed, torch.finfo(inputs_embeds.dtype).min)


def build_static_decode_bool_mask(cache_position: torch.Tensor, cache_length: int) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(int(cache_length), device=cache_position.device, dtype=torch.int64)
    return (kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)).view(cache_position.shape[0], 1, 1, int(cache_length))


def update_prefill_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    sequence_length = int(key_states.shape[2])
    key_cache[:, :, :sequence_length, :].copy_(key_states.contiguous())
    value_cache[:, :, :sequence_length, :].copy_(value_states.contiguous())


def update_decode_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    positions = cache_position.reshape(-1).to(device=key_cache.device, dtype=torch.int64).contiguous()
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(key_cache, positions, key_states.contiguous(), 2)
        torch_npu.scatter_update_(value_cache, positions, value_states.contiguous(), 2)
        return
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    batch_indices = torch.arange(int(key_cache.shape[0]), device=key_cache.device, dtype=torch.int64)
    key_cache[batch_indices, :, positions, :] = key_states.squeeze(2)
    value_cache[batch_indices, :, positions, :] = value_states.squeeze(2)


def cast_decode_linear_weights_to_nz(
    model: "LocalPaddleOCRVLForConditionalGeneration",
) -> dict[str, object]:
    modules = [
        (f"model.{name}", module)
        for name, module in model.model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules.append(("lm_head", model.lm_head))
    non_npu_modules = [(name, str(module.weight.device)) for name, module in modules if module.weight.device.type != "npu"]
    if non_npu_modules:
        return {
            "mode": DECODE_LINEAR_WEIGHT_FORMAT,
            "target_format": "FRACTAL_NZ",
            "target_format_code": FRACTAL_NZ,
            "target_count": len(modules),
            "cast_count": 0,
            "converted_count": 0,
            "already_nz_count": 0,
            "skipped": True,
            "skip_reason": "requires_npu_resident_weights",
            "non_npu_modules_sample": non_npu_modules[:16],
            "all_after_are_nz": False,
        }

    import torch_npu

    before_formats: dict[str, int] = {}
    after_formats: dict[str, int] = {}
    converted: list[str] = []
    already_nz: list[str] = []
    for name, module in modules:
        before = int(torch_npu.get_npu_format(module.weight))
        before_formats[name] = before
        if before == FRACTAL_NZ:
            already_nz.append(name)
        module.weight.data = torch_npu.npu_format_cast(module.weight.data, FRACTAL_NZ)
        after = int(torch_npu.get_npu_format(module.weight))
        after_formats[name] = after
        if before != FRACTAL_NZ and after == FRACTAL_NZ:
            converted.append(name)
    return {
        "mode": DECODE_LINEAR_WEIGHT_FORMAT,
        "target_format": "FRACTAL_NZ",
        "target_format_code": FRACTAL_NZ,
        "target_count": len(modules),
        "cast_count": len(modules),
        "converted_count": len(converted),
        "already_nz_count": len(already_nz),
        "converted_modules_sample": converted[:16],
        "before_formats_sample": dict(list(before_formats.items())[:16]),
        "after_formats_sample": dict(list(after_formats.items())[:16]),
        "all_after_are_nz": bool(after_formats) and all(value == FRACTAL_NZ for value in after_formats.values()),
    }


@dataclass
class LocalModelOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    rope_deltas: torch.Tensor | None = None


@dataclass
class LocalStaticModelOutput:
    logits: torch.Tensor
    cache: "LocalPaddleOCRVLStaticCache"
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor


@dataclass
class LocalPaddleOCRVLStaticCache:
    key_caches: tuple[torch.Tensor, ...]
    value_caches: tuple[torch.Tensor, ...]
    cache_length: int

    @classmethod
    def allocate(
        cls,
        config: PaddleOCRTextConfig,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> "LocalPaddleOCRVLStaticCache":
        cache_shape = (
            int(batch_size),
            int(config.num_key_value_heads),
            int(cache_length),
            int(config.head_dim),
        )
        key_caches = []
        value_caches = []
        for _layer_idx in range(config.num_hidden_layers):
            if init_mode == "zeros":
                key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
                value_cache = torch.zeros_like(key_cache)
            elif init_mode == "empty":
                key_cache = torch.empty(cache_shape, device=device, dtype=dtype)
                value_cache = torch.empty_like(key_cache)
            else:
                raise ValueError(f"unknown static cache init_mode: {init_mode!r}")
            key_caches.append(key_cache)
            value_caches.append(value_cache)
        return cls(tuple(key_caches), tuple(value_caches), int(cache_length))

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[int(layer_idx)], self.value_caches[int(layer_idx)]

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.key_caches, *self.value_caches)


class PaddleOCRRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
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
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer("inv_freq", self._compute_inv_freq().to(device=device), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids = position_ids[:, :, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PaddleOCRMLP(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_activation(self.hidden_act, self.gate_proj(x)) * self.up_proj(x))


class PaddleOCRAttention(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.mrope_section = list((config.rope_parameters or {})["mrope_section"])
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.use_bias)

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, _hidden = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
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
        key_for_attn = repeat_kv(key_states, self.num_key_value_groups)
        value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_for_attn.shape[-2]]
        probs = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=get_text_softmax_dtype_mode(),
        )
        attn_output = torch.matmul(probs, value_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, query_length, -1)
        return self.o_proj(attn_output)

    def attend_decode_increfa(
        self,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_states.device.type != "npu":
            raise RuntimeError("static decode uses IncreFA and requires NPU tensors plus torch_npu.")
        import torch_npu

        batch = query_states.shape[0]
        attn_output = torch_npu.npu_incre_flash_attention(
            query_states.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            atten_mask=None if attention_mask is None else attention_mask.contiguous(),
            actual_seq_lengths=None,
            num_heads=int(self.num_heads),
            num_key_value_heads=int(self.num_key_value_heads),
            input_layout="BNSD",
            scale_value=float(self.scaling),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, 1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def attend_decode_manual(
        self,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        additive_mask = attention_mask
        if additive_mask is not None and additive_mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(additive_mask, dtype=query_states.dtype).masked_fill(
                additive_mask,
                torch.finfo(query_states.dtype).min,
            )
        return self.attend(query_states, key_cache, value_cache, additive_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        if past_key_values is not None:
            past_key, past_value = past_key_values[self.layer_idx]
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        new_past = (key_states, value_states) if use_cache else None
        return self.attend(query_states, key_states, value_states, attention_mask), new_past

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        return self.attend(query_states, key_states, value_states, attention_mask), key_states, value_states

    def forward_decode_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        update_decode_kv_cache_(
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
        )
        if query_states.device.type != "npu":
            return self.attend_decode_manual(query_states, key_cache, value_cache, attention_mask)
        return self.attend_decode_increfa(query_states, key_cache, value_cache, attention_mask)


class PaddleOCRDecoderLayer(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = PaddleOCRAttention(config, layer_idx)
        self.mlp = PaddleOCRMLP(config)
        self.input_layernorm = PaddleOCRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PaddleOCRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

    def apply_blocks(self, residual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        hidden_states = residual + attn_output
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
        attn_output, key_states, value_states = self.self_attn.forward_prefill_static(
            hidden_states,
            attention_mask,
            position_embeddings,
        )
        if cache is not None:
            key_cache, value_cache = cache.layer(self.layer_idx)
            update_prefill_kv_cache_(
                key_cache,
                value_cache,
                key_states,
                value_states,
            )
        return self.apply_blocks(residual, attn_output)

    def forward_decode_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn.forward_decode_static(
            hidden_states,
            attention_mask,
            position_embeddings,
            key_cache,
            value_cache,
            cache_position,
        )
        return self.apply_blocks(residual, attn_output)


class PaddleOCRTextModel(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([PaddleOCRDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = PaddleOCRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PaddleOCRRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        past_length = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        cache_position = torch.arange(
            past_length,
            past_length + inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        causal_mask = build_causal_mask(inputs_embeds, attention_mask, cache_position, past_length=past_length)
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
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.int64)
        causal_mask = build_causal_mask(inputs_embeds, attention_mask, cache_position)
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

    def forward_decode_static(
        self,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        cache_length: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = inputs_embeds.shape
        if seq_length != 1:
            raise ValueError(f"static decode expects exactly one token, got seq_length={seq_length}")
        cache_position = cache_position.reshape(-1).to(device=inputs_embeds.device, dtype=torch.int64)
        if cache_position.numel() == 1:
            cache_position = cache_position.expand(batch_size)
        if cache_position.numel() != batch_size:
            raise ValueError(f"cache_position must be scalar or batch-shaped, got {tuple(cache_position.shape)}")
        if attention_mask is None:
            attention_mask = build_static_decode_bool_mask(cache_position, cache_length)
        position_ids = cache_position.view(batch_size, 1) + rope_deltas.to(device=inputs_embeds.device, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode_static(
                hidden_states,
                attention_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
            )
        return self.norm(hidden_states)


class PaddleOCRProjector(nn.Module):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        merge = config.vision_config.spatial_merge_size
        hidden_size = config.vision_config.hidden_size * merge * merge
        self.merge_kernel_size = (merge, merge)
        self.pre_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-5)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        chunks = image_features.split(image_grid_thw.prod(dim=1).tolist(), dim=0)
        m1, m2 = self.merge_kernel_size
        processed = []
        for image_feature, image_grid in zip(chunks, image_grid_thw):
            image_feature = self.pre_norm(image_feature)
            t, h, w = [int(v.item()) for v in image_grid]
            d = image_feature.shape[-1]
            h_block = h // m1
            w_block = w // m2
            image_feature = image_feature.reshape(t, h_block, m1, w_block, m2, d)
            image_feature = image_feature.transpose(2, 3)
            image_feature = image_feature.reshape(t * h_block * w_block, m1 * m2 * d)
            hidden_states = self.linear_1(image_feature)
            hidden_states = F.gelu(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed.append(hidden_states)
        return torch.cat(processed, dim=0)


class PaddleOCRVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer("inv_freq", self._compute_inv_freq().to(device=device), persistent=False)

    def forward(self, seqlen: int | torch.Tensor) -> torch.Tensor:
        seq = torch.arange(int(seqlen), device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class PaddleOCRVisionEmbeddings(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        dim = embeddings.shape[-1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(height, width), mode="bilinear", align_corners=False)
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.reshape(batch_size * sequence_len, channel, height, width)
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_size, sequence_len, -1).squeeze(0)
        start = 0
        tmp_embeddings = []
        for image_grid in image_grid_thw:
            t, h, w = [int(v.item()) for v in image_grid]
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            pos = self.interpolate_pos_encoding(image_embeddings, h, w).squeeze(0).repeat(t, 1)
            tmp_embeddings.append(image_embeddings + pos)
            start = end
        return torch.cat(tmp_embeddings, dim=0)


class PaddleOCRVisionAttention(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        if int(cu_seqlens.numel()) == 2:
            q_splits = (query_states,)
            k_splits = (key_states,)
            v_splits = (value_states,)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
            q_splits, k_splits, v_splits = [
                torch.split(tensor, lengths, dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
        outputs = []
        attention_impl = get_vision_attention_impl()
        for q, k, v in zip(q_splits, k_splits, v_splits):
            if attention_impl == "prompt_flash_attention":
                outputs.append(
                    vision_prompt_flash_attention_bnsd(
                        q,
                        k,
                        v,
                        num_heads=int(self.num_heads),
                        scale=float(self.scaling),
                    )
                )
            elif attention_impl == "manual":
                scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
                probs = attention_softmax(
                    scores,
                    dim=-1,
                    output_dtype=q.dtype,
                    mode=get_vision_softmax_dtype_mode(),
                )
                outputs.append(torch.matmul(probs, v))
            else:
                raise ValueError(f"unknown vision attention implementation: {attention_impl!r}")
        attn_output = torch.cat(outputs, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_length, -1)
        return self.out_proj(attn_output)


class PaddleOCRVisionMLP(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(_activation(self.hidden_act, self.fc1(hidden_states)))


class PaddleOCRVisionEncoderLayer(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = PaddleOCRVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PaddleOCRVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class PaddleOCRVisionEncoder(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([PaddleOCRVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = PaddleOCRVisionRotaryEmbedding(head_dim // 2)

    def forward(self, inputs_embeds: torch.Tensor, cu_seqlens: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        device = inputs_embeds.device
        split_hids = []
        split_wids = []
        for t, h, w in image_grid_thw:
            image_pids = torch.arange(int(t * h * w), device=device) % int(h * w)
            split_hids.append(image_pids // int(w))
            split_wids.append(image_pids % int(w))
        pids = torch.stack([torch.cat(split_hids), torch.cat(split_wids)], dim=-1)
        max_grid_size = max(max(int(h), int(w)) for _, h, w in image_grid_thw)
        rotary_max = self.rotary_pos_emb(max_grid_size)
        rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
        position_embeddings = (rotary_embeddings.cos(), rotary_embeddings.sin())
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, position_embeddings)
        return hidden_states


class PaddleOCRVisionTransformer(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.embeddings = PaddleOCRVisionEmbeddings(config)
        self.encoder = PaddleOCRVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, cu_seqlens: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, image_grid_thw=image_grid_thw)
        hidden_states = self.encoder(hidden_states, cu_seqlens=cu_seqlens, image_grid_thw=image_grid_thw)
        return self.post_layernorm(hidden_states)


class PaddleOCRVisionModel(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.vision_model = PaddleOCRVisionTransformer(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    def forward(self, pixel_values: torch.Tensor, cu_seqlens: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values=pixel_values, cu_seqlens=cu_seqlens, image_grid_thw=image_grid_thw)


class LocalPaddleOCRVLForConditionalGeneration(nn.Module):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.config = config
        self.visual = PaddleOCRVisionModel(config.vision_config)
        self.mlp_AR = PaddleOCRProjector(config)
        self.model = PaddleOCRTextModel(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.rope_deltas: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        dtype: torch.dtype | None = torch.float16,
        device: str | torch.device | None = None,
    ) -> "LocalPaddleOCRVLForConditionalGeneration":
        model_dir = _resolve_model_dir(model_id_or_path)
        config = PaddleOCRVLConfig.from_model_dir(model_dir)
        model = cls(config)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        from safetensors.torch import load_file

        state_dict = load_file(model_dir / "model.safetensors", device=str(device or "cpu"))
        ignored = {"visual.vision_model.embeddings.packing_position_embedding.weight"}
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ignored and not key.startswith("visual.vision_model.head.")
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
        if missing:
            raise RuntimeError(f"missing checkpoint keys: {missing}")
        model._reset_rope_buffers()
        return model.eval()

    def _reset_rope_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, (PaddleOCRRotaryEmbedding, PaddleOCRVisionRotaryEmbedding)):
                module.reset_inv_freq(device=module.inv_freq.device)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.type(self.visual.dtype).unsqueeze(0)
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        image_embeds = self.visual(pixel_values=pixel_values, image_grid_thw=image_grid_thw, cu_seqlens=cu_seqlens)
        return self.mlp_AR(image_embeds, image_grid_thw)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        if image_grid_thw is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            mrope_position_deltas = []
            image_index = 0
            for batch_idx, sample_input_ids in enumerate(input_ids):
                visible_input_ids = sample_input_ids[attention_mask[batch_idx].to(sample_input_ids.device) == 1]
                vision_start_indices = torch.argwhere(visible_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = visible_input_ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == image_token_id).sum().item())
                input_tokens = visible_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums
                for _ in range(image_nums):
                    ed = input_tokens.index(image_token_id, st) if remain_images > 0 else len(input_tokens) + 1
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[:, batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[batch_idx]))
            return position_ids, torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            return position_ids, max_position_ids + 1 - attention_mask.shape[-1]
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return position_ids, torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[image_mask].numel() != image_embeds.numel():
            raise ValueError(
                "image features and image tokens do not match: "
                f"tokens={int((input_ids == self.config.image_token_id).sum().item())} "
                f"features={int(image_embeds.shape[0])}"
            )
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def allocate_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> LocalPaddleOCRVLStaticCache:
        return LocalPaddleOCRVLStaticCache.allocate(
            self.config.text_config,
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode=init_mode,
        )

    def forward_static_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        *,
        cache_length: int,
        cache: LocalPaddleOCRVLStaticCache | None = None,
        cache_init_mode: str = "zeros",
        logits_to_keep: int = 0,
    ) -> LocalStaticModelOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        batch_size, sequence_length, _hidden = inputs_embeds.shape
        if int(sequence_length) > int(cache_length):
            raise ValueError(f"prefill sequence length {sequence_length} exceeds static cache length {cache_length}")
        position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        if cache is None:
            cache = self.allocate_static_cache(
                batch_size=int(batch_size),
                cache_length=int(cache_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                init_mode=cache_init_mode,
            )
        hidden_states = self.model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        next_cache_position = torch.full((int(batch_size),), int(sequence_length), device=inputs_embeds.device, dtype=torch.int64)
        self.rope_deltas = rope_deltas
        return LocalStaticModelOutput(
            logits=logits,
            cache=cache,
            rope_deltas=rope_deltas,
            next_cache_position=next_cache_position,
        )

    def forward_static_decode(
        self,
        input_ids: torch.Tensor,
        cache: LocalPaddleOCRVLStaticCache,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalModelOutput:
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = self.model.forward_decode_static(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=cache.key_caches,
            value_caches=cache.value_caches,
            cache_length=cache.cache_length,
            attention_mask=attention_mask,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return LocalModelOutput(logits=logits, rope_deltas=rope_deltas)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        rope_deltas: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalModelOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        if position_ids is None:
            if past_key_values is None:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas
            else:
                past_length = int(past_key_values[0][0].shape[2])
                batch_size, seq_length, _hidden = inputs_embeds.shape
                delta = rope_deltas if rope_deltas is not None else self.rope_deltas
                if delta is None:
                    raise ValueError("rope_deltas are required for cached decode")
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                position_ids = position_ids + (past_length + delta.to(inputs_embeds.device)).view(1, batch_size, 1)
        hidden_states, past = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return LocalModelOutput(logits=logits, past_key_values=past, rope_deltas=rope_deltas if rope_deltas is not None else self.rope_deltas)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        rope_deltas = outputs.rope_deltas
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == eos_token_id
        current_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            outputs = self.forward(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                pixel_values=None,
                image_grid_thw=None,
                past_key_values=past,
                use_cache=True,
                rope_deltas=rope_deltas,
                logits_to_keep=1,
            )
            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, eos_token_id), next_token)
            generated.append(next_token)
            finished |= next_token.squeeze(1) == eos_token_id
            current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)
        return torch.cat(generated, dim=1)

    @torch.inference_mode()
    def generate_ids_static(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 128,
        cache_length: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        prompt_length = int(input_ids.shape[1])
        min_cache_length = prompt_length + max(0, int(max_new_tokens) - 1)
        cache_length = int(
            cache_length
            if cache_length is not None
            else (prompt_length + int(max_new_tokens))
        )
        if cache_length < min_cache_length:
            raise ValueError(
                f"cache_length={cache_length} is too small for prompt length {prompt_length} "
                f"and max_new_tokens={max_new_tokens}; need at least {min_cache_length}"
            )
        outputs = self.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        cache = outputs.cache
        rope_deltas = outputs.rope_deltas
        cache_position = outputs.next_cache_position
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == eos_token_id
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            outputs_decode = self.forward_static_decode(
                input_ids=next_token,
                cache=cache,
                cache_position=cache_position,
                rope_deltas=rope_deltas,
                logits_to_keep=1,
            )
            next_token = torch.argmax(outputs_decode.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, eos_token_id), next_token)
            generated.append(next_token)
            finished |= next_token.squeeze(1) == eos_token_id
            cache_position = cache_position + 1
        return torch.cat(generated, dim=1)

    def make_flat_static_decode_module(self) -> "PaddleOCRFlatStaticDecodeModule":
        return PaddleOCRFlatStaticDecodeModule(self)


class PaddleOCRFlatStaticDecodeModule(nn.Module):
    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = flat_cache_tensors[: self.num_layers]
        value_caches = flat_cache_tensors[self.num_layers :]
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = self.model.model.forward_decode_static(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            cache_length=int(key_caches[0].shape[2]),
            attention_mask=None,
        )
        return self.model.lm_head(hidden_states[:, -1:, :])
