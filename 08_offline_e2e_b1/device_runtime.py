"""Device selection, dtype parsing, and accelerator synchronization."""

from __future__ import annotations

import sys

import torch


NPU_JIT_COMPILE_CHOICES = ("default", "off", "on")


def parse_dtype(name: str, _device: torch.device) -> torch.dtype:
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        return False
    except Exception as exc:
        raise RuntimeError(
            "torch_npu is installed but failed to initialize: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if npu_is_available():
            return torch.device("npu:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NPU device requested, but torch_npu is not importable in this "
                "environment."
            ) from exc
    return torch.device(name)


def configure_npu_jit_compile(
    mode: str,
    device: torch.device,
    *,
    verbose: bool = True,
) -> None:
    if mode not in NPU_JIT_COMPILE_CHOICES:
        raise ValueError(f"unsupported npu_jit_compile={mode!r}")
    if mode == "default" or device.type != "npu":
        return
    try:
        import torch_npu  # noqa: F401

        requested = mode == "on"
        torch.npu.set_compile_mode(jit_compile=requested)
        if verbose:
            print(
                f"[npu] set torch.npu compile mode: jit_compile={requested}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:
        raise RuntimeError(
            f"failed to set NPU jit_compile={mode}: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()
