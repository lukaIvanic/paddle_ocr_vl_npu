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
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


FRACTAL_NZ = 29
DECODE_LINEAR_WEIGHT_FORMAT = "decode_nz"
DECODE_LINEAR_WEIGHT_FALLBACK = "decode_native_fallback"
DECODE_ATTENTION = "increfa"
DECODE_CACHE_UPDATE = "npu_scatter"


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


def _decode_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    query_states, key_states, value_states = attention.project_qkv(hidden_states)
    query_states, key_states = attention.apply_rotary(
        query_states,
        key_states,
        position_embeddings,
    )
    update_decode_kv_cache_(
        key_cache,
        value_cache,
        cache_position,
        key_states,
        value_states,
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

    import torch_npu

    batch = query_states.shape[0]
    attention_output = torch_npu.npu_incre_flash_attention(
        query_states.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        atten_mask=(
            None if attention_mask is None else attention_mask.contiguous()
        ),
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
) -> torch.Tensor:
    """Execute the complete one-token transformer decode stage."""
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
    position_ids = cache_position.view(batch_size, 1) + rope_deltas.to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    for layer_idx, layer in enumerate(text_model.layers):
        residual = hidden_states
        attention_input = layer.input_layernorm(hidden_states)
        attention_output = _decode_attention(
            layer.self_attn,
            attention_input,
            position_embeddings,
            key_caches[layer_idx],
            value_caches[layer_idx],
            cache_position,
            attention_mask,
        )
        hidden_states = layer.apply_blocks(residual, attention_output)
    return text_model.norm(hidden_states)


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
        hidden_states = run_text_decode_transformer(
            self.model.model,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            cache_length=int(key_caches[0].shape[2]),
            attention_mask=None,
        )
        return _linear_tokenwise(self.model.lm_head, hidden_states[:, -1:, :])


def decode_attention_label(device: torch.device) -> str:
    return DECODE_ATTENTION if device.type == "npu" else "manual"


def decode_cache_update_label(device: torch.device) -> str:
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


def decode_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    # Decode owns its graph, while the shared text layer methods it calls are
    # defined by the prefill stage.
    for name in ("text_prefill.py", "text_decode.py"):
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
        return stage, {**common_metadata, "compile_api": "none"}

    if backend_name == "torchair":
        if device.type != "npu":
            raise ValueError("--backend torchair requires an NPU device.")
        torchair, CompilerConfig = import_torchair()
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
                "decode_attention": decode_attention_label(device),
                "decode_cache_update": decode_cache_update_label(device),
                "execution_mode": TORCHAIR_EXECUTION_MODE,
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
    ):
        self.stage = TextDecodeStage(model).eval()
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
        )
        synchronize(device)
        compile_wrapper_s = time.perf_counter() - started

        self.warm_cache: LocalPaddleOCRVLStaticCache = model.allocate_static_cache(
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode="zeros",
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
