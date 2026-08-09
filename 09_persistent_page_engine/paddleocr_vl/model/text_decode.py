"""Unified eager and compiled model execution for text decode."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    compile_backend,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .config import PaddleOCRTextConfig
from .gqa_increfa_aiv import (
    gqa_incre_flash_attention_aiv,
    register_gqa_increfa_aiv_converter,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


FRACTAL_NZ = 29
DECODE_LINEAR_WEIGHT_FORMAT = "decode_nz"
DECODE_LINEAR_WEIGHT_FALLBACK = "decode_native_fallback"
DECODE_ATTENTION = "increfa"
DECODE_CACHE_UPDATE = "npu_scatter"


@dataclass(frozen=True)
class DecodeOptimizationConfig:
    """Experimental implementation choices for the text-decode lab.

    Production callers keep the baseline defaults.  The lab selects one
    named preset so every compiled graph has an explicit, reproducible
    implementation contract.
    """

    name: str
    hoist_mrope: bool = False
    packed_qkv: bool = False
    rms_norm: str = "manual"
    rotary: str = "manual"
    rotary_factors: str = "mrope"
    packed_mlp: bool = False
    npu_swiglu: bool = False
    add_rms_norm: bool = False
    attention: str = "gqa"
    increfa_length_mode: str = "mask"
    stage_aware_weight_prefetch: bool = False
    post_scatter_kv_prefetch: bool = False
    vector_add_rms_norm: bool = False
    gqa_aiv_vector_core_count: int = 0


DECODE_OPTIMIZATION_PRESETS: dict[str, DecodeOptimizationConfig] = {
    "baseline": DecodeOptimizationConfig(name="baseline"),
    "mrope_hoist": DecodeOptimizationConfig(
        name="mrope_hoist",
        hoist_mrope=True,
    ),
    "packed_qkv": DecodeOptimizationConfig(
        name="packed_qkv",
        packed_qkv=True,
    ),
    "npu_rms_norm": DecodeOptimizationConfig(
        name="npu_rms_norm",
        rms_norm="npu",
    ),
    "npu_apply_rotary": DecodeOptimizationConfig(
        name="npu_apply_rotary",
        hoist_mrope=True,
        rotary="npu_apply",
    ),
    "npu_rotary_mul": DecodeOptimizationConfig(
        name="npu_rotary_mul",
        hoist_mrope=True,
        rotary="npu_rotary_mul",
    ),
    "packed_mlp": DecodeOptimizationConfig(
        name="packed_mlp",
        packed_mlp=True,
    ),
    "packed_mlp_swiglu": DecodeOptimizationConfig(
        name="packed_mlp_swiglu",
        packed_mlp=True,
        npu_swiglu=True,
    ),
    "npu_add_rms_norm": DecodeOptimizationConfig(
        name="npu_add_rms_norm",
        rms_norm="npu",
        add_rms_norm=True,
    ),
    "combined_apply": DecodeOptimizationConfig(
        name="combined_apply",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
    ),
    "combined_apply_pse_sentinel": DecodeOptimizationConfig(
        name="combined_apply_pse_sentinel",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        increfa_length_mode="pse_sentinel",
    ),
    "combined_apply_static_actual": DecodeOptimizationConfig(
        name="combined_apply_static_actual",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        increfa_length_mode="static_actual",
    ),
    "combined_apply_mha_repeat": DecodeOptimizationConfig(
        name="combined_apply_mha_repeat",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_repeat",
    ),
    "combined_apply_gqa_aiv_b1": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=16,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_pairwise_sync_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_pairwise_sync_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_two_way_reduce_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_two_way_reduce_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_local_partial_reduce_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_local_partial_reduce_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k48_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k48_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=48,
    ),
    "combined_apply_mha_cache": DecodeOptimizationConfig(
        name="combined_apply_mha_cache",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_cache",
    ),
    "combined_apply_mha_cache_prefetch_kv": DecodeOptimizationConfig(
        name="combined_apply_mha_cache_prefetch_kv",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_cache",
        post_scatter_kv_prefetch=True,
    ),
    "combined_apply_manual_attention": DecodeOptimizationConfig(
        name="combined_apply_manual_attention",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="manual",
    ),
    "combined_apply_manual_attention_unscaled": DecodeOptimizationConfig(
        name="combined_apply_manual_attention_unscaled",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="manual_unscaled",
    ),
    "combined_apply_prefetch": DecodeOptimizationConfig(
        name="combined_apply_prefetch",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_scalar_rope": DecodeOptimizationConfig(
        name="combined_apply_prefetch_scalar_rope",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="scalar",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut_no_norm": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut_no_norm",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="identity",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut_vector_norm": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut_vector_norm",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        vector_add_rms_norm=True,
    ),
    "combined_apply_prefetch_no_rope": DecodeOptimizationConfig(
        name="combined_apply_prefetch_no_rope",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="identity",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_no_increfa": DecodeOptimizationConfig(
        name="combined_apply_prefetch_no_increfa",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="no_increfa",
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_all": DecodeOptimizationConfig(
        name="combined_apply_all",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        packed_mlp=True,
        npu_swiglu=True,
        add_rms_norm=True,
    ),
    "combined_rotary_mul": DecodeOptimizationConfig(
        name="combined_rotary_mul",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_rotary_mul",
        add_rms_norm=True,
    ),
}


def decode_optimization_names() -> tuple[str, ...]:
    return tuple(DECODE_OPTIMIZATION_PRESETS)


def resolve_decode_optimization(
    optimization: str | DecodeOptimizationConfig,
) -> DecodeOptimizationConfig:
    if isinstance(optimization, DecodeOptimizationConfig):
        return optimization
    try:
        return DECODE_OPTIMIZATION_PRESETS[str(optimization)]
    except KeyError as exc:
        raise ValueError(
            f"unknown decode optimization {optimization!r}; expected one of "
            f"{decode_optimization_names()}"
        ) from exc


@dataclass
class LocalPaddleOCRVLStaticCache:
    """Fixed-shape KV tensors shared by prefill and continuous decode."""

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
        num_key_value_heads: int | None = None,
    ) -> "LocalPaddleOCRVLStaticCache":
        cache_heads = (
            int(config.num_key_value_heads)
            if num_key_value_heads is None
            else int(num_key_value_heads)
        )
        if cache_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        cache_shape = (
            int(batch_size),
            cache_heads,
            int(cache_length),
            int(config.head_dim),
        )
        key_caches = []
        value_caches = []
        for _layer_idx in range(config.num_hidden_layers):
            if init_mode == "zeros":
                key_cache = torch.zeros(
                    cache_shape, device=device, dtype=dtype
                )
                value_cache = torch.zeros_like(key_cache)
            elif init_mode == "empty":
                key_cache = torch.empty(
                    cache_shape, device=device, dtype=dtype
                )
                value_cache = torch.empty_like(key_cache)
            else:
                raise ValueError(
                    f"unknown static cache init_mode: {init_mode!r}"
                )
            key_caches.append(key_cache)
            value_caches.append(value_cache)
        return cls(
            tuple(key_caches), tuple(value_caches), int(cache_length)
        )

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.key_caches[int(layer_idx)],
            self.value_caches[int(layer_idx)],
        )

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.key_caches, *self.value_caches)


def _linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    output = linear(x.reshape(-1, x.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def _packed_linear(
    modules: tuple[nn.Linear, ...],
) -> nn.Linear:
    first = modules[0]
    if any(module.in_features != first.in_features for module in modules):
        raise ValueError("packed Linear inputs must share in_features")
    biases = tuple(module.bias for module in modules)
    if any(bias is None for bias in biases) and not all(
        bias is None for bias in biases
    ):
        raise ValueError("packed Linear inputs must use the same bias contract")
    packed = nn.Linear(
        first.in_features,
        sum(module.out_features for module in modules),
        bias=biases[0] is not None,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        packed.weight.copy_(
            torch.cat([module.weight for module in modules], dim=0)
        )
        if packed.bias is not None:
            packed.bias.copy_(
                torch.cat(
                    [bias for bias in biases if bias is not None],
                    dim=0,
                )
            )
    return packed


def prepare_decode_optimization_modules(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
) -> DecodeOptimizationConfig:
    """Create packed projections once, before weight-format conversion."""
    config = resolve_decode_optimization(optimization)
    for layer in model.model.layers:
        attention = layer.self_attn
        if config.packed_qkv and not hasattr(
            attention, "decode_qkv_proj"
        ):
            attention.decode_qkv_proj = _packed_linear(
                (attention.q_proj, attention.k_proj, attention.v_proj)
            )
        mlp = layer.mlp
        if config.packed_mlp and not hasattr(
            mlp, "decode_gate_up_proj"
        ):
            mlp.decode_gate_up_proj = _packed_linear(
                (mlp.gate_proj, mlp.up_proj)
            )
    return config


def prepare_decode_weight_prefetch(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
) -> None:
    """Install the proven stage-aware decode weight-prefetch schedule."""
    config = resolve_decode_optimization(optimization)
    if not config.stage_aware_weight_prefetch:
        return
    layers = model.model.layers
    for index, layer in enumerate(layers):
        layer.self_attn._decode_prefetch_current_mlp = (
            layer.mlp.gate_proj.weight,
            layer.mlp.up_proj.weight,
            layer.mlp.down_proj.weight,
        )
        layer.mlp._decode_prefetch_next_attention = (
            (
                layers[index + 1].self_attn.decode_qkv_proj.weight,
                layers[index + 1].self_attn.o_proj.weight,
            )
            if index + 1 < len(layers)
            else (model.lm_head.weight,)
        )


def build_static_decode_bool_mask(
    cache_position: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    kv_positions = torch.arange(
        int(cache_length),
        device=cache_position.device,
        dtype=torch.int64,
    )
    return (
        kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)
    ).view(cache_position.shape[0], 1, 1, int(cache_length))


def update_decode_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    positions = (
        cache_position.reshape(-1)
        .to(device=key_cache.device, dtype=torch.int64)
        .contiguous()
    )
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(
            key_cache, positions, key_states.contiguous(), 2
        )
        torch_npu.scatter_update_(
            value_cache, positions, value_states.contiguous(), 2
        )
        return
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    batch_indices = torch.arange(
        int(key_cache.shape[0]),
        device=key_cache.device,
        dtype=torch.int64,
    )
    key_cache[batch_indices, :, positions, :] = key_states.squeeze(2)
    value_cache[batch_indices, :, positions, :] = value_states.squeeze(2)


def _prepare_multimodal_rotary_factors(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform the MRoPE section selection once per decode step."""
    section = [int(value) for value in mrope_section] * 2
    prepared = []
    for factors in position_embeddings:
        prepared.append(
            torch.cat(
                [
                    part[index % 3]
                    for index, part in enumerate(
                        factors.split(section, dim=-1)
                    )
                ],
                dim=-1,
            )
            .unsqueeze(1)
            .contiguous()
        )
    return prepared[0], prepared[1]


def _prepare_scalar_rotary_factors(
    rotary_emb: nn.Module,
    inputs_embeds: torch.Tensor,
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decode RoPE factors directly from the one shared MRoPE axis."""
    freqs = (
        position.reshape(-1, 1).float()
        * rotary_emb.inv_freq.reshape(1, -1).float()
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        emb.cos().to(dtype=inputs_embeds.dtype).view(
            inputs_embeds.shape[0], 1, 1, -1
        ),
        emb.sin().to(dtype=inputs_embeds.dtype).view(
            inputs_embeds.shape[0], 1, 1, -1
        ),
    )


def _lookup_scalar_rotary_factors(
    rotary_emb: nn.Module,
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select packed cosine/sine rows from the persistent decode RoPE LUT."""
    selected = torch.index_select(
        rotary_emb.decode_rope_factor_lut,
        1,
        position.reshape(-1).to(dtype=torch.int64),
    )
    cos, sin = selected.unbind(dim=0)
    cos = cos.unsqueeze(1).unsqueeze(1)
    sin = sin.unsqueeze(1).unsqueeze(1)
    return cos, sin


def prepare_decode_rope_factor_lut(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
    *,
    cache_length: int,
    dtype: torch.dtype,
) -> None:
    """Create the final decode cos/sin table once, outside the graph."""
    config = resolve_decode_optimization(optimization)
    if config.rotary_factors != "lookup":
        return
    rotary_emb = model.model.rotary_emb
    positions = torch.arange(
        int(cache_length),
        device=rotary_emb.inv_freq.device,
        dtype=torch.float32,
    )
    freqs = positions.reshape(-1, 1) * rotary_emb.inv_freq.reshape(1, -1).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(dtype=dtype)
    rotary_emb.register_buffer(
        "decode_rope_factor_lut",
        factor_lut.contiguous(),
        persistent=False,
    )


def _project_decode_qkv(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not optimization.packed_qkv:
        return attention.project_qkv(hidden_states)
    batch, query_length, _hidden = hidden_states.shape
    qkv = _linear_tokenwise(
        attention.decode_qkv_proj,
        hidden_states,
    )
    q_size = int(attention.num_heads * attention.head_dim)
    kv_size = int(attention.num_key_value_heads * attention.head_dim)
    query_states, key_states, value_states = qkv.split(
        (q_size, kv_size, kv_size),
        dim=-1,
    )
    query_states = query_states.view(
        batch,
        query_length,
        attention.num_heads,
        attention.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    return query_states, key_states, value_states


def _apply_decode_rotary(
    attention: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor] | None,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prepared_factors is None:
        return attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
    if optimization.rotary == "identity":
        # Lab-only full-graph ablation. Keep dynamic Q/K tensors and remove
        # only the rotary operation so unprofiled timing measures its actual
        # marginal contribution to the compiled decode step.
        return query_states, key_states
    cos, sin = prepared_factors
    if optimization.rotary == "manual":
        half = query_states.shape[-1] // 2
        query_rotated = torch.cat(
            (-query_states[..., half:], query_states[..., :half]),
            dim=-1,
        )
        key_rotated = torch.cat(
            (-key_states[..., half:], key_states[..., :half]),
            dim=-1,
        )
        return (
            (query_states * cos) + (query_rotated * sin),
            (key_states * cos) + (key_rotated * sin),
        )

    import torch_npu

    if optimization.rotary == "npu_rotary_mul":
        return (
            torch_npu.npu_rotary_mul(
                query_states.contiguous(),
                cos,
                sin,
                rotary_mode="half",
            ),
            torch_npu.npu_rotary_mul(
                key_states.contiguous(),
                cos,
                sin,
                rotary_mode="half",
            ),
        )
    if optimization.rotary == "npu_apply":
        query_bsnd = query_states.transpose(1, 2).contiguous()
        key_bsnd = key_states.transpose(1, 2).contiguous()
        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd,
            key_bsnd,
            cos,
            sin,
            layout="BSND",
            rotary_mode="half",
        )
        return (
            query_bsnd.transpose(1, 2),
            key_bsnd.transpose(1, 2),
        )
    raise ValueError(
        f"unsupported decode rotary implementation: "
        f"{optimization.rotary!r}"
    )


def _decode_rms_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if optimization.rms_norm == "identity":
        return hidden_states
    if optimization.rms_norm == "manual":
        return norm(hidden_states)
    if optimization.rms_norm != "npu":
        raise ValueError(
            f"unsupported decode RMSNorm implementation: "
            f"{optimization.rms_norm!r}"
        )
    import torch_npu

    return torch_npu.npu_rms_norm(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
    )[0]


def _decode_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    normalized, _rstd, summed = torch_npu.npu_add_rms_norm(
        x,
        residual,
        norm.weight,
        norm.variance_epsilon,
    )
    return normalized, summed


@torch.library.custom_op(
    "paddleocr_vl::vector_add_rms_norm",
    mutates_args=(),
)
def _vector_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager reference for the lab-only compiled VectorAddRmsNorm op."""
    import torch_npu

    return torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)


@_vector_add_rms_norm.register_fake
def _vector_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, epsilon
    rstd_shape = (*x.shape[:-1], 1)
    return (
        torch.empty_like(x),
        torch.empty(rstd_shape, dtype=torch.float32, device=x.device),
        torch.empty_like(x),
    )


_VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED = False


def _register_vector_add_rms_norm_converter() -> None:
    """Lower the lab custom op to its installed CANN GE operator."""
    global _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED
    if _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED:
        return

    import importlib

    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op
    op = torch.ops.paddleocr_vl.vector_add_rms_norm.default

    @register_converter(op)
    def _convert_vector_add_rms_norm(
        x: Any,
        residual: Any,
        weight: Any,
        epsilon: float,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            "VectorAddRmsNorm",
            x,
            residual,
            weight,
            float(epsilon),
        )

    _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED = True


def _decode_add_with_optional_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if optimization.rms_norm == "identity":
        summed = x + residual
        return summed, summed
    if optimization.vector_add_rms_norm:
        normalized, _rstd, summed = _vector_add_rms_norm(
            x,
            residual,
            norm.weight,
            norm.variance_epsilon,
        )
        return normalized, summed
    return _decode_add_rms_norm(x, residual, norm)


def _decode_mlp(
    mlp: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if not optimization.packed_mlp:
        output = mlp(hidden_states)
    else:
        gate_up = _linear_tokenwise(
            mlp.decode_gate_up_proj,
            hidden_states,
        )
        if optimization.npu_swiglu:
            import torch_npu

            activated = torch_npu.npu_swiglu(gate_up, dim=-1)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        output = _linear_tokenwise(mlp.down_proj, activated)
    if optimization.stage_aware_weight_prefetch:
        import torch_npu

        for weight in mlp._decode_prefetch_next_attention:
            torch_npu.npu_prefetch(
                weight,
                output,
                int(weight.numel() * weight.element_size()),
            )
    return output


def _decode_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor] | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pse_shift: torch.Tensor | None,
    actual_seq_lengths: list[int] | None,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if optimization.stage_aware_weight_prefetch:
        import torch_npu

        for weight in attention._decode_prefetch_current_mlp:
            torch_npu.npu_prefetch(
                weight,
                hidden_states,
                int(weight.numel() * weight.element_size()),
            )
    query_states, key_states, value_states = _project_decode_qkv(
        attention,
        hidden_states,
        optimization,
    )
    query_states, key_states = _apply_decode_rotary(
        attention,
        query_states,
        key_states,
        position_embeddings,
        prepared_factors,
        optimization,
    )
    if optimization.attention == "mha_cache":
        if query_states.device.type != "npu":
            raise ValueError("mha_cache is an NPU-only decode lab path")
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, token_count, head_dim = key_states.shape
        expected_heads = int(attention.num_heads)
        if int(key_cache.shape[1]) != expected_heads:
            raise ValueError(
                "mha_cache requires a fully expanded decode arena: "
                f"expected {expected_heads} heads, got {int(key_cache.shape[1])}"
            )
        key_states = (
            key_states[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, token_count, head_dim)
            .reshape(batch_size, expected_heads, token_count, head_dim)
            .contiguous()
        )
        value_states = (
            value_states[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, token_count, head_dim)
            .reshape(batch_size, expected_heads, token_count, head_dim)
            .contiguous()
        )
    update_decode_kv_cache_(
        key_cache,
        value_cache,
        cache_position,
        key_states,
        value_states,
    )
    if optimization.post_scatter_kv_prefetch:
        if key_cache.device.type != "npu":
            raise ValueError(
                "post-scatter K/V prefetch is an NPU-only decode lab path"
            )
        import torch_npu

        torch_npu.npu_prefetch(
            key_cache,
            key_states,
            int(key_cache.numel() * key_cache.element_size()),
        )
        torch_npu.npu_prefetch(
            value_cache,
            value_states,
            int(value_cache.numel() * value_cache.element_size()),
        )
    if query_states.device.type != "npu":
        additive_mask = attention_mask
        if additive_mask is not None and additive_mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(
                additive_mask, dtype=query_states.dtype
            ).masked_fill(
                additive_mask,
                torch.finfo(query_states.dtype).min,
            )
        return attention.attend(
            query_states,
            key_cache,
            value_cache,
            additive_mask,
        )

    if optimization.attention == "no_increfa":
        # Lab-only full-graph ablation.  Keep QKV projection, RoPE, cache
        # writes, and output projection, but substitute dynamic query data for
        # the IncreFA result.  A dynamic substitute prevents TorchAir from
        # constant-folding the downstream output projection.
        batch = query_states.shape[0]
        attention_output = (
            query_states.transpose(1, 2)
            .contiguous()
            .reshape(batch, 1, attention.num_heads * attention.head_dim)
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    if optimization.attention in ("manual", "manual_unscaled"):
        # Lab-only decomposition of one-token GQA decode.  It deliberately
        # exposes QK, score scaling, masking, softmax, and PV as separate graph
        # operations so their kernels can be profiled against IncreFA.  The
        # unscaled lane omits only the numerical 1/sqrt(head_dim) multiply; it
        # is intentionally incorrect and exists only to measure that Vector
        # operation's cost.
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, kv_length, head_dim = key_cache.shape
        query_heads = int(attention.num_heads)
        key_manual = (
            key_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size * query_heads, kv_length, head_dim)
        )
        value_manual = (
            value_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size * query_heads, kv_length, head_dim)
        )
        query_manual = query_states.reshape(
            batch_size * query_heads, 1, head_dim
        )
        scores = torch.bmm(
            query_manual,
            key_manual.transpose(1, 2),
        ).view(batch_size, query_heads, 1, kv_length)
        if optimization.attention == "manual":
            scores = scores * float(attention.scaling)
        if pse_shift is not None:
            scores = scores + pse_shift
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask,
                torch.finfo(scores.dtype).min,
            )
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            dtype=query_states.dtype
        )
        attention_output = torch.bmm(
            probabilities.reshape(
                batch_size * query_heads, 1, kv_length
            ),
            value_manual,
        ).view(batch_size, query_heads, 1, head_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, 1, query_heads * head_dim)
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    import torch_npu

    batch = query_states.shape[0]
    key_for_attention = key_cache
    value_for_attention = value_cache
    num_key_value_heads = int(attention.num_key_value_heads)
    if optimization.attention == "mha_repeat":
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, kv_length, head_dim = key_cache.shape
        key_for_attention = (
            key_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size, kv_heads * groups, kv_length, head_dim)
            .contiguous()
        )
        value_for_attention = (
            value_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size, kv_heads * groups, kv_length, head_dim)
            .contiguous()
        )
        num_key_value_heads = 0
    elif optimization.attention == "mha_cache":
        if int(key_cache.shape[1]) != int(attention.num_heads):
            raise ValueError("mha_cache received a non-expanded decode arena")
        num_key_value_heads = 0
    elif optimization.attention not in ("gqa", "gqa_aiv"):
        raise ValueError(
            f"unsupported decode attention implementation: "
            f"{optimization.attention!r}"
        )

    if optimization.attention == "gqa_aiv":
        if pse_shift is not None or actual_seq_lengths is not None:
            raise ValueError(
                "gqa_aiv requires masked IncreFA with no PSE or actual lengths"
            )
        if attention_mask is None:
            raise ValueError("gqa_aiv requires the static bool attention mask")
        attention_output = gqa_incre_flash_attention_aiv(
            query_states.contiguous(),
            key_for_attention.contiguous(),
            value_for_attention.contiguous(),
            attention_mask.contiguous(),
            num_heads=int(attention.num_heads),
            num_key_value_heads=num_key_value_heads,
            scale_value=float(attention.scaling),
            inner_precise=1,
            vector_core_count=optimization.gqa_aiv_vector_core_count,
        )
    else:
        attention_output = torch_npu.npu_incre_flash_attention(
            query_states.contiguous(),
            key_for_attention.contiguous(),
            value_for_attention.contiguous(),
            pse_shift=pse_shift,
            atten_mask=(
                None if attention_mask is None else attention_mask.contiguous()
            ),
            actual_seq_lengths=actual_seq_lengths,
            num_heads=int(attention.num_heads),
            num_key_value_heads=num_key_value_heads,
            input_layout="BNSD",
            scale_value=float(attention.scaling),
        )
    attention_output = (
        attention_output.transpose(1, 2)
        .contiguous()
        .reshape(batch, 1, attention.num_heads * attention.head_dim)
    )
    return _linear_tokenwise(attention.o_proj, attention_output)


def run_text_decode_transformer(
    text_model: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    rope_deltas: torch.Tensor,
    key_caches: tuple[torch.Tensor, ...],
    value_caches: tuple[torch.Tensor, ...],
    cache_length: int,
    attention_mask: torch.Tensor | None = None,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> torch.Tensor:
    """Execute the complete one-token transformer decode stage."""
    optimization = resolve_decode_optimization(optimization)
    batch_size, seq_length, _hidden = inputs_embeds.shape
    if seq_length != 1:
        raise ValueError(
            f"static decode expects exactly one token, got "
            f"seq_length={seq_length}"
        )
    cache_position = cache_position.reshape(-1).to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    if cache_position.numel() == 1:
        cache_position = cache_position.expand(batch_size)
    if cache_position.numel() != batch_size:
        raise ValueError(
            "cache_position must be scalar or batch-shaped, got "
            f"{tuple(cache_position.shape)}"
        )
    if attention_mask is None:
        attention_mask = build_static_decode_bool_mask(
            cache_position, cache_length
        )
    pse_shift: torch.Tensor | None = None
    actual_seq_lengths: list[int] | None = None
    if optimization.increfa_length_mode == "pse_sentinel":
        # The 310P masked-GQA kernel can deadlock when the valid prefix is an
        # exact 1280-token internal tile. Keep one always-present PSE graph:
        # expose one otherwise-masked cache position only at those boundaries,
        # then suppress it additively. The PSE is zero at all other positions.
        effective_lengths = cache_position.view(batch_size, 1, 1, 1) + 1
        physical_positions = torch.arange(
            int(cache_length),
            device=inputs_embeds.device,
            dtype=torch.int64,
        ).view(1, 1, 1, int(cache_length))
        boundary = (
            (effective_lengths.remainder(1280) == 0)
            & (effective_lengths < int(cache_length))
        )
        sentinel = boundary & (physical_positions == effective_lengths)
        attention_mask = attention_mask & ~sentinel
        pse_shift = torch.zeros(
            (
                batch_size,
                int(text_model.config.num_attention_heads),
                1,
                int(cache_length),
            ),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(
            sentinel.expand(
                batch_size,
                int(text_model.config.num_attention_heads),
                1,
                int(cache_length),
            ),
            torch.finfo(inputs_embeds.dtype).min,
        )
    elif optimization.increfa_length_mode == "static_actual":
        # Deliberately constant for the static BxKV graph. The boolean mask
        # still carries each row's logical prefix length.
        actual_seq_lengths = [int(cache_length)] * int(batch_size)
    elif optimization.increfa_length_mode != "mask":
        raise ValueError(
            "unsupported IncreFA length mode: "
            f"{optimization.increfa_length_mode!r}"
        )
    decode_position = cache_position.view(batch_size, 1) + rope_deltas.to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    if optimization.rotary_factors == "mrope":
        position_ids = decode_position.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
        prepared_factors = (
            _prepare_multimodal_rotary_factors(
                position_embeddings,
                text_model.layers[0].self_attn.mrope_section,
            )
            if optimization.hoist_mrope
            else None
        )
    elif optimization.rotary_factors == "scalar":
        prepared_factors = _prepare_scalar_rotary_factors(
            text_model.rotary_emb,
            inputs_embeds,
            decode_position,
        )
        position_embeddings = prepared_factors
    elif optimization.rotary_factors == "lookup":
        prepared_factors = _lookup_scalar_rotary_factors(
            text_model.rotary_emb,
            decode_position,
        )
        position_embeddings = prepared_factors
    else:
        raise ValueError(
            "unsupported decode rotary-factor mode: "
            f"{optimization.rotary_factors!r}"
        )
    hidden_states = inputs_embeds
    if optimization.name == "baseline":
        for layer_idx, layer in enumerate(text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = _decode_attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                None,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                attention_mask,
                pse_shift,
                actual_seq_lengths,
                optimization,
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        return text_model.norm(hidden_states)

    if optimization.add_rms_norm:
        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(text_model.layers):
            if residual is None:
                attention_input = _decode_rms_norm(
                    layer.input_layernorm,
                    hidden_states,
                    optimization,
                )
                residual = hidden_states
            else:
                attention_input, residual = _decode_add_with_optional_rms_norm(
                    hidden_states,
                    residual,
                    layer.input_layernorm,
                    optimization,
                )
            attention_output = _decode_attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                prepared_factors,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                attention_mask,
                pse_shift,
                actual_seq_lengths,
                optimization,
            )
            mlp_input, residual = _decode_add_with_optional_rms_norm(
                attention_output,
                residual,
                layer.post_attention_layernorm,
                optimization,
            )
            hidden_states = _decode_mlp(
                layer.mlp,
                mlp_input,
                optimization,
            )
        hidden_states, _residual = _decode_add_with_optional_rms_norm(
            hidden_states,
            residual,
            text_model.norm,
            optimization,
        )
        return hidden_states

    for layer_idx, layer in enumerate(text_model.layers):
        residual = hidden_states
        attention_input = _decode_rms_norm(
            layer.input_layernorm,
            hidden_states,
            optimization,
        )
        attention_output = _decode_attention(
            layer.self_attn,
            attention_input,
            position_embeddings,
            prepared_factors,
            key_caches[layer_idx],
            value_caches[layer_idx],
            cache_position,
            attention_mask,
            pse_shift,
            actual_seq_lengths,
            optimization,
        )
        if (
            optimization.rms_norm == "manual"
            and not optimization.packed_mlp
        ):
            hidden_states = layer.apply_blocks(residual, attention_output)
            continue
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = _decode_rms_norm(
            layer.post_attention_layernorm,
            hidden_states,
            optimization,
        )
        hidden_states = _decode_mlp(
            layer.mlp,
            hidden_states,
            optimization,
        )
        hidden_states = residual + hidden_states
    return _decode_rms_norm(text_model.norm, hidden_states, optimization)


def cast_decode_linear_weights_to_nz(
    model: "LocalPaddleOCRVLForConditionalGeneration",
) -> dict[str, object]:
    """Prepare all text-decode Linear weights in NPU FRACTAL_NZ format."""
    modules = [
        (f"model.{name}", module)
        for name, module in model.model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules.append(("lm_head", model.lm_head))
    non_npu_modules = [
        (name, str(module.weight.device))
        for name, module in modules
        if module.weight.device.type != "npu"
    ]
    if non_npu_modules:
        return {
            "requested_mode": DECODE_LINEAR_WEIGHT_FORMAT,
            "mode": DECODE_LINEAR_WEIGHT_FALLBACK,
            "effective_mode": DECODE_LINEAR_WEIGHT_FALLBACK,
            "target_format": "FRACTAL_NZ",
            "target_format_code": FRACTAL_NZ,
            "target_count": len(modules),
            "cast_count": 0,
            "converted_count": 0,
            "already_nz_count": 0,
            "skipped": True,
            "skip_reason": "requires_npu_resident_weights",
            "fallback_reason": "requires_npu_resident_weights",
            "non_npu_modules_sample": non_npu_modules[:16],
            "all_after_are_nz": False,
        }

    import torch_npu

    before_formats: dict[str, int] = {}
    after_formats: dict[str, int] = {}
    converted: list[str] = []
    already_nz: list[str] = []
    failures: list[dict[str, object]] = []
    cast_count = 0
    for name, module in modules:
        before = int(torch_npu.get_npu_format(module.weight))
        before_formats[name] = before
        if before == FRACTAL_NZ:
            already_nz.append(name)
            after_formats[name] = before
            continue
        cast_count += 1
        try:
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data, FRACTAL_NZ
            )
        except Exception as exc:
            failures.append(
                {
                    "module": name,
                    "before_format": before,
                    "error": repr(exc),
                }
            )
            break
        after = int(torch_npu.get_npu_format(module.weight))
        after_formats[name] = after
        if before != FRACTAL_NZ and after == FRACTAL_NZ:
            converted.append(name)
        else:
            failures.append(
                {
                    "module": name,
                    "before_format": before,
                    "after_format": after,
                    "error": "npu_format_cast_did_not_produce_fractal_nz",
                }
            )
            break
    all_after_are_nz = len(after_formats) == len(modules) and all(
        value == FRACTAL_NZ for value in after_formats.values()
    )
    if all_after_are_nz:
        effective_mode = DECODE_LINEAR_WEIGHT_FORMAT
    elif converted:
        effective_mode = "decode_mixed_format"
    else:
        effective_mode = DECODE_LINEAR_WEIGHT_FALLBACK
    return {
        "requested_mode": DECODE_LINEAR_WEIGHT_FORMAT,
        "mode": effective_mode,
        "effective_mode": effective_mode,
        "target_format": "FRACTAL_NZ",
        "target_format_code": FRACTAL_NZ,
        "target_count": len(modules),
        "cast_count": cast_count,
        "converted_count": len(converted),
        "already_nz_count": len(already_nz),
        "converted_modules_sample": converted[:16],
        "before_formats_sample": dict(list(before_formats.items())[:16]),
        "after_formats_sample": dict(list(after_formats.items())[:16]),
        "all_after_are_nz": all_after_are_nz,
        "fallback_reason": failures[0]["error"] if failures else None,
        "failures_sample": failures[:16],
    }

class TextDecodeStage(torch.nn.Module):
    """One fixed-shape autoregressive text step.

    The same module is called directly for eager execution or wrapped by the
    selected compiler. Cache tensors stay flat at the boundary so the compiled
    graph can mutate the persistent decode arena in place.
    """

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        optimization: str | DecodeOptimizationConfig = "baseline",
    ):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.optimization = resolve_decode_optimization(optimization)

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
        hidden_states = run_text_decode_transformer(
            self.model.model,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            cache_length=int(key_caches[0].shape[2]),
            attention_mask=None,
            optimization=self.optimization,
        )
        return _linear_tokenwise(self.model.lm_head, hidden_states[:, -1:, :])


def decode_attention_label(
    device: torch.device,
    optimization: DecodeOptimizationConfig | None = None,
) -> str:
    if device.type != "npu":
        return "manual"
    if optimization is not None and optimization.attention == "gqa_aiv":
        return "paddle_gqa_increfa_aiv"
    return DECODE_ATTENTION


def decode_cache_update_label(device: torch.device) -> str:
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


def decode_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    # Decode owns its graph, while the shared text layer methods it calls are
    # defined by the prefill stage.
    for name in ("text_prefill.py", "text_decode.py", "gqa_increfa_aiv.py"):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def torchair_cache_dir_for_shape(
    cache_root: Path,
    *,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    model_dir: Path | None = None,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> Path:
    optimization = resolve_decode_optimization(optimization)
    model_hash = (
        short_file_hash(model_dir / "config.json")
        if model_dir is not None
        else "model_unknown"
    )
    shape_key = "_".join(
        [
            linear_weight_format,
            DECODE_ATTENTION,
            DECODE_CACHE_UPDATE,
            f"opt{cache_key_part(optimization.name)}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"dtype{cache_key_part(dtype or 'unknown')}",
            f"bs{int(batch_size)}",
            f"cache{int(cache_length)}",
            f"model{model_hash}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device or torch.device('cpu'))}",
            f"torchair{torchair_version_label(device or torch.device('cpu'))}",
            f"src{decode_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / shape_key


def compile_text_decode_stage(
    stage: TextDecodeStage,
    *,
    backend_name: str,
    device: torch.device,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype | None = None,
    model_dir: Path | None = None,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> tuple[Any, dict[str, Any]]:
    optimization = resolve_decode_optimization(optimization)
    if optimization.attention == "gqa_aiv":
        if backend_name != "torchair":
            raise ValueError("gqa_aiv is an independent TorchAir-only operator")
        if batch_size != 1:
            raise ValueError("gqa_aiv currently supports only batch_size=1")
    common_metadata = {
        "backend": backend_name,
        "enabled": backend_name != "raw_eager",
        "boundary": "token_embedding_text_transformer_lm_head_static_step",
        "linear_weight_format": linear_weight_format,
        "decode_attention": decode_attention_label(device, optimization),
        "decode_cache_update": decode_cache_update_label(device),
        "decode_optimization": optimization.name,
        "decode_optimization_config": {
            "hoist_mrope": optimization.hoist_mrope,
            "packed_qkv": optimization.packed_qkv,
            "rms_norm": optimization.rms_norm,
            "rotary": optimization.rotary,
            "rotary_factors": optimization.rotary_factors,
            "packed_mlp": optimization.packed_mlp,
            "npu_swiglu": optimization.npu_swiglu,
            "add_rms_norm": optimization.add_rms_norm,
            "attention": optimization.attention,
            "increfa_length_mode": optimization.increfa_length_mode,
            "stage_aware_weight_prefetch": (
                optimization.stage_aware_weight_prefetch
            ),
            "post_scatter_kv_prefetch": (
                optimization.post_scatter_kv_prefetch
            ),
            "vector_add_rms_norm": optimization.vector_add_rms_norm,
            "gqa_aiv_vector_core_count": (
                optimization.gqa_aiv_vector_core_count
            ),
        },
    }
    if backend_name == "raw_eager":
        return stage, {**common_metadata, "compile_api": "none"}

    if backend_name == "torchair":
        if device.type != "npu":
            raise ValueError("--backend torchair requires an NPU device.")
        if optimization.vector_add_rms_norm:
            _register_vector_add_rms_norm_converter()
        torchair, CompilerConfig = import_torchair()
        if optimization.attention == "gqa_aiv":
            register_gqa_increfa_aiv_converter()
        shape_cache_dir = torchair_cache_dir_for_shape(
            cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=optimization,
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        compiled_decode = torchair.inference.cache_compile(
            stage.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
        )
        return compiled_decode, {
            **common_metadata,
            "torchair_cache_dir": str(shape_cache_dir),
            "torchair_ge_cache": True,
            "compile_api": "torchair.inference.cache_compile",
            "cache_key_fields": {
                "batch_size": int(batch_size),
                "cache_length": int(cache_length),
                "dtype": str(dtype),
                "model_config_hash": (
                    short_file_hash(model_dir / "config.json")
                    if model_dir is not None
                    else None
                ),
                "torch": str(torch.__version__),
                "torch_npu": torch_npu_version_label(device),
                "torchair": torchair_version_label(device),
                "decode_source_hash": decode_source_hash(),
                "linear_weight_format": linear_weight_format,
                "decode_attention": decode_attention_label(
                    device, optimization
                ),
                "decode_cache_update": decode_cache_update_label(device),
                "execution_mode": TORCHAIR_EXECUTION_MODE,
                "decode_optimization": optimization.name,
            },
        }

    backend = compile_backend(backend_name)
    compile_kwargs = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    return torch.compile(stage, **compile_kwargs), {
        **common_metadata,
        "compile_api": "torch.compile",
    }


class TextDecodeRuntime:
    """Own the shared decode stage, its execution wrapper, and warm arena."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        device: torch.device,
        cache_root: Path,
        batch_size: int,
        cache_length: int,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        optimization: str | DecodeOptimizationConfig = "baseline",
    ):
        self.optimization = prepare_decode_optimization_modules(
            model,
            optimization,
        )
        prepare_decode_rope_factor_lut(
            model,
            self.optimization,
            cache_length=cache_length,
            dtype=dtype,
        )
        prepare_decode_weight_prefetch(model, self.optimization)
        self.stage = TextDecodeStage(
            model,
            optimization=self.optimization,
        ).eval()
        self.cache_num_key_value_heads = (
            int(model.config.text_config.num_attention_heads)
            if self.optimization.attention == "mha_cache"
            else int(model.config.text_config.num_key_value_heads)
        )
        synchronize(device)
        started = time.perf_counter()
        self.fn, self.metadata = compile_text_decode_stage(
            self.stage,
            backend_name=backend,
            device=device,
            cache_root=cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=self.optimization,
        )
        synchronize(device)
        compile_wrapper_s = time.perf_counter() - started

        self.warm_cache: LocalPaddleOCRVLStaticCache = model.allocate_static_cache(
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode="zeros",
            num_key_value_heads=self.cache_num_key_value_heads,
        )
        self.metadata["cache_num_key_value_heads"] = (
            self.cache_num_key_value_heads
        )
        self.metadata["cache_allocated_bytes"] = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.warm_cache.flat_tensors()
        )
        warm_input = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        warm_position = torch.ones((batch_size,), device=device, dtype=torch.int64)
        warm_rope = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        synchronize(device)
        started = time.perf_counter()
        self.fn(
            warm_input,
            warm_position,
            warm_rope,
            *self.warm_cache.flat_tensors(),
        )
        synchronize(device)
        compile_first_call_s = time.perf_counter() - started
        del warm_input, warm_position, warm_rope
        self.setup_timing_s = {
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
        }
