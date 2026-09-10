"""Unified eager and compiled model execution for text decode."""

from __future__ import annotations

import hashlib
import json
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from _support.model.compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    compile_backend,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from _support.model.config import PaddleOCRTextConfig
from _support.utils.timing import synchronize

if TYPE_CHECKING:
    from paddle_ocr_vl_1_6_modeling import LocalPaddleOCRVLForConditionalGeneration


FRACTAL_NZ = 29
DECODE_LINEAR_WEIGHT_FORMAT = "decode_nz"
DECODE_LINEAR_WEIGHT_FALLBACK = "decode_native_fallback"
DECODE_ATTENTION = "increfa"
DECODE_CACHE_UPDATE = "npu_scatter"



@dataclass
class LocalPaddleOCRVLStaticCache:
    """Zero-initialized, separate K/V storage for the persistent serving engine."""

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
    ):
        cache_shape = (
            int(batch_size),
            int(config.num_key_value_heads),
            int(cache_length),
            int(config.head_dim),
        )
        key_caches = []
        value_caches = []
        for _ in range(config.num_hidden_layers):
            key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
            value_cache = torch.zeros_like(key_cache)
            key_caches.append(key_cache)
            value_caches.append(value_cache)
        return cls(tuple(key_caches), tuple(value_caches), int(cache_length))

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[int(layer_idx)], self.value_caches[int(layer_idx)]

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.key_caches, *self.value_caches)


def _linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    token_matrix = x.reshape(-1, x.shape[-1])
    output = linear(token_matrix)
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


def prepare_decode_projections(
    model: "LocalPaddleOCRVLForConditionalGeneration",
) -> None:
    """Create packed projections once, before weight-format conversion."""
    for layer in model.model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "decode_qkv_proj"):
            attention.decode_qkv_proj = _packed_linear(
                (attention.q_proj, attention.k_proj, attention.v_proj)
            )
        mlp = layer.mlp
        if not hasattr(mlp, "decode_gate_up_proj"):
            mlp.decode_gate_up_proj = _packed_linear((mlp.gate_proj, mlp.up_proj))


def load_decode_vocab_token_ids(
    path: Path,
    *,
    full_vocab_size: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Load an explicit native-token decode vocabulary.

    The file contains token IDs, not text.  This deliberately has no tokenizer
    dependency: generated text must never be decoded and re-encoded to build a
    compact output head.
    """
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    raw_ids = payload.get("token_ids") if isinstance(payload, dict) else payload
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"decode vocabulary {resolved} has no token_ids list")
    token_ids = tuple(int(value) for value in raw_ids)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"decode vocabulary {resolved} contains duplicate IDs")
    invalid = [
        token_id
        for token_id in token_ids
        if not 0 <= token_id < int(full_vocab_size)
    ]
    if invalid:
        raise ValueError(
            f"decode vocabulary {resolved} has IDs outside [0, "
            f"{int(full_vocab_size)}): {invalid[:16]}"
        )
    digest_payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()
    declared_digest = (
        payload.get("token_ids_sha256")
        if isinstance(payload, dict)
        else None
    )
    if declared_digest is not None and str(declared_digest) != digest:
        raise ValueError(
            f"decode vocabulary {resolved} digest mismatch: "
            f"declared={declared_digest} actual={digest}"
        )
    metadata = {
        "enabled": True,
        "path": str(resolved),
        "full_vocab_size": int(full_vocab_size),
        "selected_vocab_size": len(token_ids),
        "token_ids_sha256": digest,
        "source": payload.get("source") if isinstance(payload, dict) else None,
        "selection": (
            payload.get("selection")
            if isinstance(payload, dict)
            else None
        ),
    }
    return token_ids, metadata


def prepare_decode_compact_lm_head(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    token_ids: tuple[int, ...],
) -> nn.Linear:
    """Gather selected full-head rows into a decode-only LM head."""
    if hasattr(model, "decode_lm_head"):
        raise ValueError("model already has a decode-only LM head")
    full_head = model.lm_head
    index = torch.tensor(
        token_ids,
        device=full_head.weight.device,
        dtype=torch.int64,
    )
    compact = nn.Linear(
        int(full_head.in_features),
        len(token_ids),
        bias=False,
        device=full_head.weight.device,
        dtype=full_head.weight.dtype,
    )
    with torch.no_grad():
        compact.weight.copy_(full_head.weight.index_select(0, index))
    compact.weight.requires_grad_(False)
    model.decode_lm_head = compact
    model.register_buffer(
        "decode_token_id_map",
        index,
        persistent=False,
    )
    return compact


def prepare_decode_weight_prefetch(
    model: "LocalPaddleOCRVLForConditionalGeneration",
) -> None:
    """Install the proven stage-aware decode weight-prefetch schedule."""
    layers = model.model.layers
    decode_lm_head = getattr(model, "decode_lm_head", model.lm_head)

    def mlp_weights(mlp: nn.Module) -> tuple[torch.Tensor, ...]:
        return (mlp.decode_gate_up_proj.weight, mlp.down_proj.weight)

    def complete_layer_weights(layer: nn.Module) -> tuple[torch.Tensor, ...]:
        return (
            layer.self_attn.decode_qkv_proj.weight,
            layer.self_attn.o_proj.weight,
            *mlp_weights(layer.mlp),
        )

    for index, layer in enumerate(layers):
        future_weights: list[torch.Tensor] = []
        future_index = index + 1
        if future_index < len(layers):
            future_weights.extend(complete_layer_weights(layers[future_index]))
        if index + 1 >= len(layers):
            future_weights.append(decode_lm_head.weight)
        layer._decode_prefetch_future_layers = tuple(future_weights)


def build_static_decode_bool_mask(
    cache_position: torch.Tensor,
    cache_length: int,
    kv_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    if kv_positions is None:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = (
        cache_position.reshape(-1)
        .to(device=key_cache.device, dtype=torch.int64)
        .contiguous()
    )
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(key_cache, positions, key_states.contiguous(), 2)
        torch_npu.scatter_update_(value_cache, positions, value_states.contiguous(), 2)
        return key_cache, value_cache
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    batch_indices = torch.arange(
        int(key_cache.shape[0]),
        device=key_cache.device,
        dtype=torch.int64,
    )
    key_cache[batch_indices, :, positions, :] = key_states.squeeze(2)
    value_cache[batch_indices, :, positions, :] = value_states.squeeze(2)
    return key_cache, value_cache


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
    *,
    cache_length: int,
    dtype: torch.dtype,
) -> None:
    """Create the final decode cos/sin table once, outside the graph."""
    rotary_emb = model.model.rotary_emb
    positions = torch.arange(
        int(cache_length), device=rotary_emb.inv_freq.device, dtype=torch.float32
    )
    freqs = positions.reshape(-1, 1) * rotary_emb.inv_freq.reshape(1, -1).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(dtype=dtype)
    rotary_emb.register_buffer(
        "decode_rope_factor_lut", factor_lut.contiguous(), persistent=False
    )


def _project_decode_qkv(
    attention: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (batch, query_length, _hidden) = hidden_states.shape
    qkv = _linear_tokenwise(attention.decode_qkv_proj, hidden_states)
    q_size = int(attention.num_heads * attention.head_dim)
    kv_size = int(attention.num_key_value_heads * attention.head_dim)
    (query_states, key_states, value_states) = qkv.split(
        (q_size, kv_size, kv_size), dim=-1
    )
    query_states = query_states.view(
        batch, query_length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch, query_length, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch, query_length, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
    return (query_states, key_states, value_states)


def _apply_decode_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    prepared_factors: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    (cos, sin) = prepared_factors
    import torch_npu

    query_bsnd = query_states.transpose(1, 2).contiguous()
    key_bsnd = key_states.transpose(1, 2).contiguous()
    (query_bsnd, key_bsnd) = torch_npu.npu_apply_rotary_pos_emb(
        query_bsnd, key_bsnd, cos, sin, layout="BSND", rotary_mode="half"
    )
    return (query_bsnd.transpose(1, 2), key_bsnd.transpose(1, 2))


def _decode_rms_norm(norm: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_rms_norm(hidden_states, norm.weight, norm.variance_epsilon)[0]


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


def _decode_mlp(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_up = _linear_tokenwise(mlp.decode_gate_up_proj, hidden_states)
    import torch_npu

    activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = _linear_tokenwise(mlp.down_proj, activated)
    return output


def _decode_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    prepared_factors: tuple[torch.Tensor, torch.Tensor] | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    (query_states, key_states, value_states) = _project_decode_qkv(
        attention, hidden_states
    )
    (query_states, key_states) = _apply_decode_rotary(query_states, key_states, prepared_factors)
    (key_cache, value_cache) = update_decode_kv_cache_(
        key_cache,
        value_cache,
        cache_position,
        key_states,
        value_states,
    )
    if key_cache.device.type != "npu":
        raise ValueError("post-scatter K/V prefetch is an NPU-only decode lab path")
    import torch_npu

    torch_npu.npu_prefetch(
        key_cache, key_states, int(key_cache.numel() * key_cache.element_size())
    )
    torch_npu.npu_prefetch(
        value_cache, value_states, int(value_cache.numel() * value_cache.element_size())
    )
    batch = query_states.shape[0]
    attention_output = torch_npu.npu_incre_flash_attention(
        query_states.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        pse_shift=None,
        atten_mask=None if attention_mask is None else attention_mask.contiguous(),
        actual_seq_lengths=None,
        num_heads=int(attention.num_heads),
        num_key_value_heads=int(attention.num_key_value_heads),
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
    static_kv_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the complete one-token transformer decode stage."""
    (batch_size, seq_length, _hidden) = inputs_embeds.shape
    if seq_length != 1:
        raise ValueError(
            f"static decode expects exactly one token, got seq_length={seq_length}"
        )
    cache_position = cache_position.reshape(-1).to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    if cache_position.numel() == 1:
        cache_position = cache_position.expand(batch_size)
    if cache_position.numel() != batch_size:
        raise ValueError(
            f"cache_position must be scalar or batch-shaped, got {tuple(cache_position.shape)}"
        )
    if attention_mask is None:
        attention_mask = build_static_decode_bool_mask(
            cache_position, cache_length, kv_positions=static_kv_positions
        )
    cache_position_2d = cache_position.view(batch_size, 1)
    rope_deltas_i64 = rope_deltas.to(device=inputs_embeds.device, dtype=torch.int64)
    decode_position = cache_position_2d + rope_deltas_i64
    prepared_factors = _lookup_scalar_rotary_factors(
        text_model.rotary_emb, decode_position
    )
    hidden_states = inputs_embeds
    residual: torch.Tensor | None = None
    for layer_idx, layer in enumerate(text_model.layers):
        import torch_npu

        for weight in layer._decode_prefetch_future_layers:
            torch_npu.npu_prefetch(
                weight, hidden_states, int(weight.numel() * weight.element_size())
            )
        if residual is None:
            attention_input = _decode_rms_norm(layer.input_layernorm, hidden_states)
            residual = hidden_states
        else:
            (attention_input, residual) = _decode_add_rms_norm(
                hidden_states, residual, layer.input_layernorm
            )
        attention_output = _decode_attention(
            layer.self_attn,
            attention_input,
            prepared_factors,
            key_caches[layer_idx],
            value_caches[layer_idx],
            cache_position,
            attention_mask,
        )
        (mlp_input, residual) = _decode_add_rms_norm(
            attention_output, residual, layer.post_attention_layernorm
        )
        hidden_states = _decode_mlp(layer.mlp, mlp_input)
    (hidden_states, _residual) = _decode_add_rms_norm(
        hidden_states, residual, text_model.norm
    )
    return hidden_states


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
    if hasattr(model, "decode_lm_head"):
        modules.append(("decode_lm_head", model.decode_lm_head))
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
    ):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def _forward_impl(
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
            static_kv_positions=None,
        )
        output_head = getattr(self.model, "decode_lm_head", self.model.lm_head)
        logits = _linear_tokenwise(output_head, hidden_states[:, -1:, :])
        if hasattr(self.model, "decode_token_id_map"):
            compact_ids = torch.argmax(logits[:, -1, :].float(), dim=-1)
            return self.model.decode_token_id_map.index_select(0, compact_ids).view(
                -1, 1
            )
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(
            input_ids, cache_position, rope_deltas, *flat_cache_tensors
        )


def decode_attention_label(device: torch.device) -> str:
    if device.type != "npu":
        return "manual"
    return DECODE_ATTENTION


def decode_cache_update_label(device: torch.device) -> str:
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


def decode_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    # Decode owns its graph, while the shared text layer methods it calls are
    # defined by the prefill stage.
    for name in (
        "text_prefill_and_decode.py",
        "text_prefill_and_decode.py",
        "_support/model/gqa_increfa_aiv.py",
        "_support/model/decode_gqa_increfa_aiv.py",
        "_support/model/decode_gqa_increfa_mixed.py",
        "_support/model/decode_packed_qkv_rope_gqa_mixed24.py",
        "_support/model/decode_token_embedding.py",
        "_support/model/decode_linear_matmul_v3.py",
        "_support/model/decode_qkv_split.py",
        "_support/model/decode_kv_scatter_query.py",
        "_support/model/decode_gqa_attention_aiv.py",
        "_support/model/decode_gqa_attention_mixed24.py",
        "_support/model/decode_swiglu.py",
        "_support/model/decode_position_add.py",
        "_support/model/decode_rope_lookup.py",
        "_support/model/decode_kv_scatter.py",
    ):
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
) -> Path:
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
            "optcombined_apply_complete_layer_prefetch1_rope_lut_packed_mlp",
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
) -> tuple[Any, dict[str, Any]]:
    common_metadata = {
        "backend": backend_name,
        "enabled": backend_name != "raw_eager",
        "boundary": "token_embedding_text_transformer_lm_head_static_step",
        "linear_weight_format": linear_weight_format,
        "decode_attention": decode_attention_label(device),
        "decode_cache_update": decode_cache_update_label(device),
    }
    if backend_name == "raw_eager":
        return (stage, {**common_metadata, "compile_api": "none"})
    if backend_name == "torchair":
        if device.type != "npu":
            raise ValueError("--backend torchair requires an NPU device.")
        (torchair, CompilerConfig) = import_torchair()
        shape_cache_dir = torchair_cache_dir_for_shape(
            cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        original = stage.forward.__func__
        name = f"text_decode_b{int(batch_size)}_kv{int(cache_length)}"
        function = types.FunctionType(
            original.__code__.replace(co_name=name),
            original.__globals__,
            name,
            original.__defaults__,
            original.__closure__,
        )
        function.__annotations__ = dict(original.__annotations__)
        function.__kwdefaults__ = original.__kwdefaults__
        entrypoint = types.MethodType(function, stage)
        compiled_decode = torchair.inference.cache_compile(
            entrypoint,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
        )
        return (
            compiled_decode,
            {
                **common_metadata,
                "torchair_cache_dir": str(shape_cache_dir),
                "torchair_ge_cache": True,
                "compile_api": "torchair.inference.cache_compile",
                "cache_key_fields": {
                    "batch_size": int(batch_size),
                    "cache_length": int(cache_length),
                    "dtype": str(dtype),
                    "model_config_hash": short_file_hash(model_dir / "config.json")
                    if model_dir is not None
                    else None,
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "decode_source_hash": decode_source_hash(),
                    "linear_weight_format": linear_weight_format,
                    "decode_attention": decode_attention_label(device),
                    "decode_cache_update": decode_cache_update_label(device),
                    "execution_mode": TORCHAIR_EXECUTION_MODE,
                },
            },
        )
    backend = compile_backend(backend_name)
    compile_kwargs = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    return (
        torch.compile(stage, **compile_kwargs),
        {**common_metadata, "compile_api": "torch.compile"},
    )


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
    ):
        prepare_decode_projections(model)
        prepare_decode_rope_factor_lut(model, cache_length=cache_length, dtype=dtype)
        prepare_decode_weight_prefetch(model)
        self.stage = TextDecodeStage(model).eval()
        self.cache_num_key_value_heads = int(
            model.config.text_config.num_key_value_heads
        )
        synchronize(device)
        started = time.perf_counter()
        (self.fn, self.metadata) = compile_text_decode_stage(
            self.stage,
            backend_name=backend,
            device=device,
            cache_root=cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
        )
        synchronize(device)
        compile_wrapper_s = time.perf_counter() - started
        self.warm_cache: LocalPaddleOCRVLStaticCache = model.allocate_static_cache(
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
        )
        self.metadata["cache_num_key_value_heads"] = self.cache_num_key_value_heads
        self.metadata["cache_allocated_bytes"] = sum(
            (
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in self.warm_cache.flat_tensors()
            )
        )
        warm_input = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        warm_position = torch.ones((batch_size,), device=device, dtype=torch.int64)
        warm_rope = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        synchronize(device)
        started = time.perf_counter()
        self.fn(warm_input, warm_position, warm_rope, *self.warm_cache.flat_tensors())
        synchronize(device)
        compile_first_call_s = time.perf_counter() - started
        del warm_input, warm_position, warm_rope
        self.setup_timing_s = {
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
        }


# Text prefill uses its validated causal attention computation.

DEFAULT_TEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
TEXT_BACKEND_CHOICES = ("raw_eager", "torchair")
TEXT_PADDING_CHOICES = ("auto", "none", "bucket")


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"unsupported activation: {name!r}")


def _prefill_linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    output = linear(x.reshape(-1, x.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


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
        gate = _prefill_linear_tokenwise(self.gate_proj, x)
        up = _prefill_linear_tokenwise(self.up_proj, x)
        return _prefill_linear_tokenwise(
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
        query_states = _prefill_linear_tokenwise(
            self.q_proj, hidden_states
        ).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = _prefill_linear_tokenwise(
            self.k_proj, hidden_states
        ).view(
            batch,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = _prefill_linear_tokenwise(
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
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.bmm(
            probabilities.reshape(batch * num_heads, seq_length, seq_length),
            value_bh,
        ).view(batch, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch,
            seq_length,
            num_heads * head_dim,
        )
        return _prefill_linear_tokenwise(attention.o_proj, attention_output)

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
            "softmaxfp32",
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
                    "softmax_dtype": 'fp32',
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
