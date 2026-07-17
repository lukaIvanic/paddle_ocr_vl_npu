"""Compile and cache the fixed-shape continuous-decode graph."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    compile_backend,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    DECODE_LINEAR_WEIGHT_FORMAT,
)


def decode_attention_label(device: torch.device) -> str:
    return DECODE_ATTENTION if device.type == "npu" else "manual"


def decode_cache_update_label(device: torch.device) -> str:
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


def decode_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in ("local_modeling_paddleocr_vl.py", "decode_compile.py"):
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


def compile_decode_module(
    flat_decode: torch.nn.Module,
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
        "linear_weight_format": linear_weight_format,
        "decode_attention": decode_attention_label(device),
        "decode_cache_update": decode_cache_update_label(device),
    }
    if backend_name == "raw_eager":
        return flat_decode, {**common_metadata, "compile_api": "none"}

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
            flat_decode.forward,
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
    return torch.compile(flat_decode, **compile_kwargs), {
        **common_metadata,
        "compile_api": "torch.compile",
    }
