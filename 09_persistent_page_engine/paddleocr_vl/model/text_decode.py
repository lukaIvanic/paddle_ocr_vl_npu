"""Unified eager and compiled model execution for text decode."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import torch

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    compile_backend,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .modeling import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    DECODE_LINEAR_WEIGHT_FORMAT,
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    _linear_tokenwise,
)
from utils.timing import synchronize


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
        hidden_states = self.model.model.forward_decode_static(
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
    for name in ("modeling.py", "text_decode.py"):
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
