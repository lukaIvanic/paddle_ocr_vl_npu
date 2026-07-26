"""Unified eager and compiled model execution for vision prefill.

Patch embedding and absolute-position interpolation remain eager and operate at
the crop's real shape. This module prepares zero-padded or exact-shape inputs,
runs one compiler-safe vision stage either directly or through TorchAir, and
slices the real rows before the existing projector consumes them.
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
from .config import PaddleOCRVLConfig, PaddleOCRVisionConfig
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


DEFAULT_VISION_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048)
VISION_BACKEND_CHOICES = ("raw_eager", "torchair")
VISION_PADDING_CHOICES = ("auto", "none", "bucket")
VISION_ATTENTION_ENV = "PADDLE_OCR_VL_VISION_ATTENTION"
VISION_ATTENTION_CHOICES = ("manual", "prompt_flash_attention")
VISION_PROMPT_FA_LAYOUT_ENV = "PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT"
VISION_PROMPT_FA_LAYOUT_CHOICES = ("bnsd", "bsnd", "bsh")
VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV = (
    "PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE"
)
VISION_PROMPT_FA_310P_SEQ_ALIGNMENT = 128
VISION_SOFTMAX_DTYPE_ENV = "PADDLE_OCR_VL_VISION_SOFTMAX_DTYPE"
SOFTMAX_DTYPE_CHOICES = ("fp32", "model")


def get_vision_attention_impl() -> str:
    mode = os.environ.get(VISION_ATTENTION_ENV, "manual").strip() or "manual"
    if mode not in VISION_ATTENTION_CHOICES:
        raise ValueError(
            f"{VISION_ATTENTION_ENV} must be one of "
            f"{VISION_ATTENTION_CHOICES}, got {mode!r}"
        )
    return mode


def get_vision_prompt_fa_layout() -> str:
    layout = (
        os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd").strip().lower()
        or "bnsd"
    )
    if layout not in VISION_PROMPT_FA_LAYOUT_CHOICES:
        raise ValueError(
            f"{VISION_PROMPT_FA_LAYOUT_ENV} must be one of "
            f"{VISION_PROMPT_FA_LAYOUT_CHOICES}, got {layout!r}"
        )
    return layout


def get_vision_prompt_fa_mask_sparse_mode() -> int:
    raw = (
        os.environ.get(VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV, "1").strip()
        or "1"
    )
    try:
        mode = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV} must be an integer, "
            f"got {raw!r}"
        ) from exc
    if mode not in (0, 1):
        raise ValueError(
            f"{VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV} must be 0 or 1 for "
            f"vision padding masks, got {mode}"
        )
    return mode


def prompt_flash_attention_call_head_dim(head_dim: int) -> int:
    """Return the smallest PromptFA-compatible head dimension."""
    head_dim = int(head_dim)
    return ((head_dim + 15) // 16) * 16


def align_vision_seq_len(seq_len: int, alignment: int) -> int:
    """Round a physical vision sequence up to its runtime alignment."""
    seq_len = int(seq_len)
    alignment = int(alignment)
    if seq_len <= 0:
        raise ValueError(f"vision sequence length must be positive, got {seq_len}")
    if alignment <= 0:
        raise ValueError(f"vision sequence alignment must be positive, got {alignment}")
    return ((seq_len + alignment - 1) // alignment) * alignment


def align_vision_buckets(
    buckets: str | Iterable[int],
    alignment: int,
) -> tuple[int, ...]:
    """Normalize configured buckets to the physical PromptFA alignment."""
    parsed = parse_vision_buckets(buckets)
    return tuple(
        sorted({align_vision_seq_len(bucket, alignment) for bucket in parsed})
    )


def get_vision_softmax_dtype_mode() -> str:
    mode = (
        os.environ.get(VISION_SOFTMAX_DTYPE_ENV, "fp32").strip().lower()
        or "fp32"
    )
    if mode not in SOFTMAX_DTYPE_CHOICES:
        raise ValueError(
            f"{VISION_SOFTMAX_DTYPE_ENV} must be one of "
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
        raise RuntimeError(
            "vision prompt_flash_attention requires NPU tensors plus torch_npu."
        )
    import torch_npu

    selected_layout = (
        get_vision_prompt_fa_layout()
        if layout is None
        else layout.strip().lower()
    )
    if selected_layout not in VISION_PROMPT_FA_LAYOUT_CHOICES:
        raise ValueError(
            f"unsupported vision PromptFA layout: {selected_layout!r}"
        )

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
        q_bnsd.transpose(1, 2).contiguous().view(
            batch, seq_len, heads * head_dim
        ),
        k_bnsd.transpose(1, 2).contiguous().view(
            batch, seq_len, heads * head_dim
        ),
        v_bnsd.transpose(1, 2).contiguous().view(
            batch, seq_len, heads * head_dim
        ),
        num_heads=int(num_heads),
        input_layout="BSH",
        scale_value=float(scale),
        sparse_mode=sparse_mode,
        **mask_kwargs,
    )
    return (
        out_bsh.view(batch, seq_len, heads, head_dim)
        .transpose(1, 2)
        .contiguous()
    )


class PaddleOCRProjector(nn.Module):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        merge = config.vision_config.spatial_merge_size
        hidden_size = config.vision_config.hidden_size * merge * merge
        self.merge_kernel_size = (merge, merge)
        self.pre_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-5)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(
            hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        chunks = image_features.split(image_grid_thw.prod(dim=1).tolist(), dim=0)
        m1, m2 = self.merge_kernel_size
        processed = []
        for image_feature, image_grid in zip(chunks, image_grid_thw):
            image_feature = self.pre_norm(image_feature)
            t, h, w = [int(v.item()) for v in image_grid]
            d = image_feature.shape[-1]
            h_block = h // m1
            w_block = w // m2
            image_feature = image_feature.reshape(
                t, h_block, m1, w_block, m2, d
            )
            image_feature = image_feature.transpose(2, 3)
            image_feature = image_feature.reshape(
                t * h_block * w_block, m1 * m2 * d
            )
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
        return 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq().to(device=device),
            persistent=False,
        )

    def forward(self, seqlen: int | torch.Tensor) -> torch.Tensor:
        seq = torch.arange(
            int(seqlen),
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
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

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        dim = embeddings.shape[-1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        ).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.reshape(
            batch_size * sequence_len, channel, height, width
        )
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_size, sequence_len, -1).squeeze(0)
        start = 0
        tmp_embeddings = []
        for image_grid in image_grid_thw:
            t, h, w = [int(v.item()) for v in image_grid]
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            pos = (
                self.interpolate_pos_encoding(image_embeddings, h, w)
                .squeeze(0)
                .repeat(t, 1)
            )
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
        query_states = self.q_proj(hidden_states).view(
            seq_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            seq_length, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            seq_length, self.num_heads, self.head_dim
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        if int(cu_seqlens.numel()) == 2:
            q_splits = (query_states,)
            k_splits = (key_states,)
            v_splits = (value_states,)
        else:
            lengths = (
                (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
            )
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
                raise ValueError(
                    f"unknown vision attention implementation: "
                    f"{attention_impl!r}"
                )
        attn_output = torch.cat(outputs, dim=2)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(seq_length, -1)
        )
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
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = PaddleOCRVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = PaddleOCRVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens,
            position_embeddings,
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class PaddleOCRVisionEncoder(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                PaddleOCRVisionEncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = PaddleOCRVisionRotaryEmbedding(head_dim // 2)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        device = inputs_embeds.device
        split_hids = []
        split_wids = []
        for t, h, w in image_grid_thw:
            image_pids = torch.arange(
                int(t * h * w), device=device
            ) % int(h * w)
            split_hids.append(image_pids // int(w))
            split_wids.append(image_pids % int(w))
        pids = torch.stack(
            [torch.cat(split_hids), torch.cat(split_wids)], dim=-1
        )
        max_grid_size = max(
            max(int(h), int(w)) for _, h, w in image_grid_thw
        )
        rotary_max = self.rotary_pos_emb(max_grid_size)
        rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
        position_embeddings = (
            rotary_embeddings.cos(),
            rotary_embeddings.sin(),
        )
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, cu_seqlens, position_embeddings
            )
        return hidden_states


class PaddleOCRVisionTransformer(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.embeddings = PaddleOCRVisionEmbeddings(config)
        self.encoder = PaddleOCRVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values, image_grid_thw=image_grid_thw
        )
        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
        )
        return self.post_layernorm(hidden_states)


class PaddleOCRVisionModel(nn.Module):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        self.vision_model = PaddleOCRVisionTransformer(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
        )


def parse_vision_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("vision buckets cannot be empty")
        try:
            buckets = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"invalid vision buckets: {value!r}") from exc
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets:
        raise ValueError("vision buckets cannot be empty")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("every vision bucket must be positive")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("vision buckets must be unique and strictly increasing")
    return buckets


def select_vision_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    real_seq_len = int(real_seq_len)
    if real_seq_len <= 0:
        raise ValueError("real vision sequence length must be positive")
    for bucket in buckets:
        if real_seq_len <= int(bucket):
            return int(bucket)
    return None


class VisionPrefillStage(torch.nn.Module):
    """Vision encoder plus post LayerNorm for eager or compiled use."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        attention_impl: str,
    ):
        super().__init__()
        self.transformer = model.visual.vision_model
        self.attention_impl = str(attention_impl)
        if self.attention_impl not in VISION_ATTENTION_CHOICES:
            raise ValueError(
                "vision attention must be one of "
                f"{VISION_ATTENTION_CHOICES}, got {attention_impl!r}"
            )
        self.softmax_dtype_mode = get_vision_softmax_dtype_mode()

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = hidden_states.shape
        qkv = torch.cat(
            [
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
                attention.v_proj(hidden_states),
            ],
            dim=-1,
        )
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        num_heads = int(attention.num_heads)
        head_dim = int(attention.head_dim)
        query_states = query_states.view(batch_size, seq_length, num_heads, head_dim)
        key_states = key_states.view(batch_size, seq_length, num_heads, head_dim)
        value_states = value_states.view(batch_size, seq_length, num_heads, head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            rope_cos,
            rope_sin,
        )
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        if self.attention_impl == "prompt_flash_attention":
            call_head_dim = prompt_flash_attention_call_head_dim(head_dim)
            if call_head_dim != head_dim:
                padding = (0, call_head_dim - head_dim)
                query_states = F.pad(query_states, padding).contiguous()
                key_states = F.pad(key_states, padding).contiguous()
                value_states = F.pad(value_states, padding).contiguous()
            attention_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=num_heads,
                scale=float(attention.scaling),
                atten_mask=attention_mask,
            )
            if call_head_dim != head_dim:
                attention_output = attention_output[..., :head_dim].contiguous()
        else:
            # Explicit [B*H, S, D] bmm is equivalent to the stock 4-D matmul
            # but prevents GE from treating the attention head as a broadcast
            # axis.
            query_bh = query_states.reshape(
                batch_size * num_heads, seq_length, head_dim
            )
            key_bh = key_states.reshape(
                batch_size * num_heads, seq_length, head_dim
            )
            value_bh = value_states.reshape(
                batch_size * num_heads, seq_length, head_dim
            )
            scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
                batch_size,
                num_heads,
                seq_length,
                seq_length,
            ) * attention.scaling
            scores = scores.masked_fill(
                attention_mask, torch.finfo(scores.dtype).min
            )
            probs = attention_softmax(
                scores,
                dim=-1,
                output_dtype=query_states.dtype,
                mode=self.softmax_dtype_mode,
            )
            attention_output = torch.bmm(
                probs.reshape(
                    batch_size * num_heads, seq_length, seq_length
                ),
                value_bh,
            ).view(batch_size, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_length,
            -1,
        )
        return attention.out_proj(attention_output)

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = prefix_hidden_states
        for encoder_layer in self.transformer.encoder.layers:
            attention_input = encoder_layer.layer_norm1(hidden_states)
            hidden_states = hidden_states + self._attention(
                encoder_layer.self_attn,
                attention_input,
                rope_cos,
                rope_sin,
                attention_mask,
            )
            mlp_input = encoder_layer.layer_norm2(hidden_states)
            hidden_states = hidden_states + encoder_layer.mlp.fc2(
                _activation(
                    encoder_layer.mlp.hidden_act,
                    encoder_layer.mlp.fc1(mlp_input),
                )
            )
        return self.transformer.post_layernorm(hidden_states)


def unique_bucket_forward(
    module: VisionPrefillStage,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Clone ``forward``'s code object so Dynamo caches shapes independently.

    TorchDynamo keys recompilation state by Python code object. Passing the same
    class method to eight ``cache_compile`` wrappers therefore makes later
    static shapes look like recompilations and TorchAir skips their persistent
    caches. Each bucket needs a semantically identical but distinct entry code
    object.
    """

    original = module.forward.__func__
    name = f"vision_encoder_bucket_{int(bucket)}"
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
class PreparedVisionPrefill:
    prefix_hidden_states: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    attention_mask: torch.Tensor
    real_seq_len: int
    physical_seq_len: int
    execution: str


@dataclass(frozen=True)
class PreparedPackedVisionPrefill:
    prepared: PreparedVisionPrefill
    segment_lengths: tuple[int, ...]


def build_vision_rope(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    real_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    if int(grid.shape[0]) != 1:
        raise ValueError(f"compiled B=1 vision expects one grid row, got {tuple(grid.shape)}")
    t, h, w = (int(value) for value in grid[0].tolist())
    if t * h * w != int(real_seq_len):
        raise ValueError(
            f"image grid has {t * h * w} tokens but embeddings have {int(real_seq_len)} rows"
        )
    encoder = model.visual.vision_model.encoder
    image_pids = torch.arange(int(real_seq_len), device=device, dtype=torch.int64) % int(h * w)
    pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
    rotary_max = encoder.rotary_pos_emb(max(h, w))
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return rotary_embeddings.cos().contiguous(), rotary_embeddings.sin().contiguous()


def prepare_vision_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    prefix_hidden_states: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    physical_seq_len: int,
    execution: str,
) -> PreparedVisionPrefill:
    if prefix_hidden_states.ndim != 2:
        raise ValueError(
            f"vision prefix must have shape [S, H], got {tuple(prefix_hidden_states.shape)}"
        )
    real_seq_len = int(prefix_hidden_states.shape[0])
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"real vision sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )
    rope_cos, rope_sin = build_vision_rope(
        model,
        image_grid_thw,
        real_seq_len=real_seq_len,
        device=prefix_hidden_states.device,
    )
    pad_tokens = physical_seq_len - real_seq_len
    prefix = F.pad(prefix_hidden_states, (0, 0, 0, pad_tokens)).unsqueeze(0).contiguous()
    if pad_tokens:
        rope_cos = torch.cat(
            [
                rope_cos,
                torch.ones(
                    (pad_tokens, rope_cos.shape[-1]),
                    device=rope_cos.device,
                    dtype=rope_cos.dtype,
                ),
            ],
            dim=0,
        )
        rope_sin = torch.cat(
            [
                rope_sin,
                torch.zeros(
                    (pad_tokens, rope_sin.shape[-1]),
                    device=rope_sin.device,
                    dtype=rope_sin.dtype,
                ),
            ],
            dim=0,
        )
    indices = torch.arange(physical_seq_len, device=prefix_hidden_states.device)
    is_real = indices < real_seq_len
    attention_mask = (is_real[:, None] != is_real[None, :]).view(
        1,
        1,
        physical_seq_len,
        physical_seq_len,
    )
    return PreparedVisionPrefill(
        prefix_hidden_states=prefix,
        rope_cos=rope_cos.unsqueeze(0).contiguous(),
        rope_sin=rope_sin.unsqueeze(0).contiguous(),
        attention_mask=attention_mask.contiguous(),
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
        execution=str(execution),
    )


def prepare_packed_vision_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    hidden_states: Iterable[torch.Tensor],
    image_grids_thw: Iterable[torch.Tensor],
    *,
    physical_seq_len: int,
    execution: str,
) -> PreparedPackedVisionPrefill:
    hidden = list(hidden_states)
    grids = list(image_grids_thw)
    if not hidden or len(hidden) != len(grids):
        raise ValueError(
            "packed vision inputs require equal non-empty hidden/grid lists"
        )
    lengths = tuple(int(value.shape[0]) for value in hidden)
    real_seq_len = sum(lengths)
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"packed vision sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )

    ropes = [
        build_vision_rope(
            model,
            grid,
            real_seq_len=length,
            device=hidden[0].device,
        )
        for grid, length in zip(grids, lengths)
    ]
    prefix = torch.cat(hidden, dim=0)
    rope_cos = torch.cat([pair[0] for pair in ropes], dim=0)
    rope_sin = torch.cat([pair[1] for pair in ropes], dim=0)
    pad_tokens = physical_seq_len - real_seq_len
    if pad_tokens:
        prefix = F.pad(prefix, (0, 0, 0, pad_tokens))
        rope_cos = torch.cat(
            [
                rope_cos,
                torch.ones(
                    (pad_tokens, rope_cos.shape[-1]),
                    device=rope_cos.device,
                    dtype=rope_cos.dtype,
                ),
            ],
            dim=0,
        )
        rope_sin = torch.cat(
            [
                rope_sin,
                torch.zeros(
                    (pad_tokens, rope_sin.shape[-1]),
                    device=rope_sin.device,
                    dtype=rope_sin.dtype,
                ),
            ],
            dim=0,
        )

    segment_ids = torch.cat(
        [
            torch.full(
                (length,),
                index,
                device=prefix.device,
                dtype=torch.int32,
            )
            for index, length in enumerate(lengths)
        ]
        + (
            [
                torch.full(
                    (pad_tokens,),
                    -1,
                    device=prefix.device,
                    dtype=torch.int32,
                )
            ]
            if pad_tokens
            else []
        )
    )
    attention_mask = (segment_ids[:, None] != segment_ids[None, :]).view(
        1,
        1,
        physical_seq_len,
        physical_seq_len,
    )
    return PreparedPackedVisionPrefill(
        prepared=PreparedVisionPrefill(
            prefix_hidden_states=prefix.unsqueeze(0).contiguous(),
            rope_cos=rope_cos.unsqueeze(0).contiguous(),
            rope_sin=rope_sin.unsqueeze(0).contiguous(),
            attention_mask=attention_mask.contiguous(),
            real_seq_len=real_seq_len,
            physical_seq_len=physical_seq_len,
            execution=str(execution),
        ),
        segment_lengths=lengths,
    )


def vision_source_hash() -> str:
    return short_file_hash(Path(__file__).resolve())


def vision_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    attention_impl: str,
    head_dim: int,
) -> Path:
    attention_key = (
        "manual_bmm"
        if attention_impl == "manual"
        else "promptfa_"
        f"d{prompt_flash_attention_call_head_dim(head_dim)}_"
        f"{cache_key_part(get_vision_prompt_fa_layout())}_"
        f"sparse{get_vision_prompt_fa_mask_sparse_mode()}"
    )
    key = "_".join(
        [
            f"encoder_postln_{attention_key}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_vision_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{vision_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class VisionPrefillRuntime:
    """Run one vision-prefill stage eagerly or through static bucket graphs."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        buckets: Iterable[int],
        cache_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        attention_impl: str = "manual",
        padding: str = "auto",
        seq_alignment: int = 1,
    ):
        self.model = model
        self.backend = str(backend)
        self.attention_impl = str(attention_impl)
        if self.attention_impl not in VISION_ATTENTION_CHOICES:
            raise ValueError(
                "vision attention must be one of "
                f"{VISION_ATTENTION_CHOICES}, got {attention_impl!r}"
            )
        self.seq_alignment = int(seq_alignment)
        if self.seq_alignment <= 0:
            raise ValueError(
                "vision sequence alignment must be positive, "
                f"got {self.seq_alignment}"
            )
        if self.seq_alignment != 1 and self.attention_impl != "prompt_flash_attention":
            raise ValueError(
                "vision sequence alignment is only supported with "
                "prompt_flash_attention"
            )
        self.buckets = align_vision_buckets(buckets, self.seq_alignment)
        self.requested_padding = str(padding)
        if self.requested_padding not in VISION_PADDING_CHOICES:
            raise ValueError(
                "vision padding must be one of "
                f"{VISION_PADDING_CHOICES}, got {padding!r}"
            )
        self.padding = (
            "bucket"
            if self.requested_padding == "auto" and self.backend == "torchair"
            else "none"
            if self.requested_padding == "auto"
            else self.requested_padding
        )
        if self.backend == "torchair" and self.padding != "bucket":
            raise ValueError("compiled vision execution requires bucket padding")
        self.device = device
        self.dtype = dtype
        self.cache_root = cache_root.expanduser().resolve()
        hidden_size = int(model.config.vision_config.hidden_size)
        head_dim = hidden_size // int(
            model.config.vision_config.num_attention_heads
        )
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.eager_stage = VisionPrefillStage(
            model,
            attention_impl=self.attention_impl,
        ).eval()
        self.modules: dict[int, VisionPrefillStage] = {}
        self.metadata: dict[str, Any] = {
            "backend": self.backend,
            "enabled": self.backend == "torchair",
            "boundary": "vision_encoder_layers_plus_post_layernorm",
            "attention": self.attention_impl,
            "prompt_flash_attention_layout": (
                get_vision_prompt_fa_layout()
                if self.attention_impl == "prompt_flash_attention"
                else None
            ),
            "prompt_flash_attention_mask_sparse_mode": (
                get_vision_prompt_fa_mask_sparse_mode()
                if self.attention_impl == "prompt_flash_attention"
                else None
            ),
            "vision_head_dim": head_dim,
            "prompt_flash_attention_call_head_dim": (
                prompt_flash_attention_call_head_dim(head_dim)
                if self.attention_impl == "prompt_flash_attention"
                else None
            ),
            "buckets": list(self.buckets),
            "sequence_alignment": self.seq_alignment,
            "requested_padding": self.requested_padding,
            "padding": self.padding,
            "overflow": (
                "eager_same_stage_unpadded"
                if self.padding == "bucket"
                else None
            ),
        }
        if self.backend not in VISION_BACKEND_CHOICES:
            raise ValueError(f"vision backend must be one of {VISION_BACKEND_CHOICES}, got {backend!r}")
        if self.backend == "raw_eager":
            return
        if self.device.type != "npu":
            raise ValueError("compiled vision backend torchair requires an NPU device")

        torchair, CompilerConfig = import_torchair()
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = VisionPrefillStage(
                model,
                attention_impl=self.attention_impl,
            ).eval()
            cache_dir = vision_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                dtype=self.dtype,
                device=self.device,
                model_dir=model_dir,
                attention_impl=self.attention_impl,
                head_dim=head_dim,
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

            warm_prefix = torch.zeros(
                (1, bucket, hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            warm_cos = torch.ones(
                (1, bucket, head_dim),
                device=self.device,
                # The stock rotary table is derived from an fp32 inv_freq, so
                # real calls supply fp32 cos/sin even when hidden states are fp16.
                dtype=torch.float32,
            )
            warm_sin = torch.zeros_like(warm_cos)
            warm_mask = torch.zeros(
                (1, 1, bucket, bucket),
                device=self.device,
                dtype=torch.bool,
            )
            synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(warm_prefix, warm_cos, warm_sin, warm_mask)
            synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_prefix, warm_cos, warm_sin, warm_mask

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
                    "dtype": str(dtype),
                    "model_config_hash": short_file_hash(model_dir / "config.json"),
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "vision_source_hash": vision_source_hash(),
                    "attention": self.attention_impl,
                    "prompt_flash_attention_layout": (
                        get_vision_prompt_fa_layout()
                        if self.attention_impl == "prompt_flash_attention"
                        else None
                    ),
                    "prompt_flash_attention_mask_sparse_mode": (
                        get_vision_prompt_fa_mask_sparse_mode()
                        if self.attention_impl == "prompt_flash_attention"
                        else None
                    ),
                    "softmax_dtype": get_vision_softmax_dtype_mode(),
                    "execution_mode": TORCHAIR_EXECUTION_MODE,
                },
            }
        )

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        bucket = (
            select_vision_bucket(real_seq_len, self.buckets)
            if self.padding == "bucket"
            else None
        )
        if bucket is None:
            physical_seq_len = align_vision_seq_len(
                real_seq_len,
                self.seq_alignment,
            )
            return {
                "execution": (
                    "eager_overflow"
                    if self.padding == "bucket"
                    else (
                        "eager_padded"
                        if physical_seq_len != real_seq_len
                        else "eager"
                    )
                ),
                "real_vision_tokens": real_seq_len,
                "physical_vision_tokens": physical_seq_len,
                "padding_vision_tokens": physical_seq_len - real_seq_len,
                "useful_token_fraction": (
                    float(real_seq_len) / float(physical_seq_len)
                ),
                "bucket": None,
            }
        return {
            "execution": (
                "compiled" if self.backend == "torchair" else "eager_padded"
            ),
            "real_vision_tokens": real_seq_len,
            "physical_vision_tokens": bucket,
            "padding_vision_tokens": bucket - real_seq_len,
            "useful_token_fraction": float(real_seq_len) / float(bucket),
            "bucket": bucket,
        }

    def prepare(
        self,
        prefix_hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        route: dict[str, Any],
    ) -> PreparedVisionPrefill:
        return prepare_vision_prefill(
            self.model,
            prefix_hidden_states,
            image_grid_thw,
            physical_seq_len=int(route["physical_vision_tokens"]),
            execution=str(route["execution"]),
        )

    def prepare_packed(
        self,
        hidden_states: Iterable[torch.Tensor],
        image_grids_thw: Iterable[torch.Tensor],
        *,
        route: dict[str, Any],
    ) -> PreparedPackedVisionPrefill:
        return prepare_packed_vision_prefill(
            self.model,
            hidden_states,
            image_grids_thw,
            physical_seq_len=int(route["physical_vision_tokens"]),
            execution=str(route["execution"]),
        )

    def run_prepared(self, prepared: PreparedVisionPrefill) -> torch.Tensor:
        run = (
            self.compiled[prepared.physical_seq_len]
            if prepared.execution == "compiled"
            else self.eager_stage
        )
        output = run(
            prepared.prefix_hidden_states,
            prepared.rope_cos,
            prepared.rope_sin,
            prepared.attention_mask,
        )
        return output[0, : prepared.real_seq_len].contiguous()
