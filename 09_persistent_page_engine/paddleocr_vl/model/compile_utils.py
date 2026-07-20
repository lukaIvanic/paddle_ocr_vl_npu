"""TorchAir import, version, and cache-key helpers for model stages."""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import torch


TORCHAIR_EXECUTION_MODE = "inference"


def import_torchair():
    try:
        import torchair

        CompilerConfig = torchair.CompilerConfig
    except Exception as direct_error:
        try:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import (
                CompilerConfig,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "TorchAir is unavailable: direct `import torchair` failed with "
                f"{direct_error!r}, and `from torch_npu.dynamo import torchair` "
                f"failed with {fallback_error!r}."
            ) from fallback_error

    if not hasattr(torchair, "inference"):
        torchair.inference = importlib.import_module(
            f"{torchair.__name__}.inference"
        )
    return torchair, CompilerConfig


def compile_backend(name: str):
    if name == "default":
        return None
    if name == "torchair":
        torchair, CompilerConfig = import_torchair()
        return torchair.get_npu_backend(compiler_config=CompilerConfig())
    return name


def cache_key_part(value: object) -> str:
    text = str(value).replace("torch.", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def short_file_hash(path: Path) -> str:
    if not path.exists():
        return "nohash"
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def torch_npu_version_label(device: torch.device) -> str:
    if device.type != "npu":
        return "non_npu"
    try:
        import torch_npu

        return cache_key_part(getattr(torch_npu, "__version__", "unknown"))
    except Exception:
        return "unknown"


def torchair_version_label(device: torch.device) -> str:
    if device.type != "npu":
        return "non_npu"
    try:
        torchair, _ = import_torchair()
        return cache_key_part(getattr(torchair, "__version__", "unknown"))
    except Exception:
        return "unknown"
