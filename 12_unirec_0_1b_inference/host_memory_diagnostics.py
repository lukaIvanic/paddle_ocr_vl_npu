"""Opt-in process-local host-memory diagnostics for UniRec serving."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any


def enabled() -> bool:
    return os.environ.get("UNIREC_HOST_MEMORY_DIAGNOSTICS", "0") == "1"


def _proc_memory_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        name, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if len(fields) >= 2 and fields[1] == "kB":
            values[name] = int(fields[0]) * 1024
    return {
        "rss": values.get("Rss", 0),
        "pss": values.get("Pss", 0),
        "private": values.get("Private_Clean", 0)
        + values.get("Private_Dirty", 0),
        "shared": values.get("Shared_Clean", 0)
        + values.get("Shared_Dirty", 0),
        "anonymous": values.get("Anonymous", 0),
        "file_pss": values.get("Pss_File", 0),
        "anon_pss": values.get("Pss_Anon", 0),
        "shmem_pss": values.get("Pss_Shmem", 0),
    }


def _jemalloc_stats_bytes() -> dict[str, int] | None:
    library = ctypes.CDLL(None)
    if not hasattr(library, "mallctl"):
        return None
    mallctl = library.mallctl
    mallctl.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    mallctl.restype = ctypes.c_int
    epoch = ctypes.c_size_t(1)
    epoch_size = ctypes.c_size_t(ctypes.sizeof(epoch))
    if mallctl(
        b"epoch",
        ctypes.byref(epoch),
        ctypes.byref(epoch_size),
        ctypes.byref(epoch),
        epoch_size.value,
    ):
        return None
    result: dict[str, int] = {}
    for short_name in (
        "allocated",
        "active",
        "metadata",
        "resident",
        "mapped",
        "retained",
    ):
        value = ctypes.c_size_t()
        value_size = ctypes.c_size_t(ctypes.sizeof(value))
        status = mallctl(
            f"stats.{short_name}".encode(),
            ctypes.byref(value),
            ctypes.byref(value_size),
            None,
            0,
        )
        if status:
            return None
        result[short_name] = int(value.value)
    return result


def _module_tensor_bytes(module: Any) -> dict[str, int]:
    if module is None:
        return {}
    totals: dict[str, int] = {}
    seen: set[tuple[str, int, int]] = set()
    tensors = list(module.parameters()) + list(module.buffers())
    for tensor in tensors:
        device = str(tensor.device)
        logical_bytes = int(tensor.numel()) * int(tensor.element_size())
        try:
            data_ptr = int(tensor.data_ptr())
        except RuntimeError:
            data_ptr = id(tensor)
        key = (device, data_ptr, logical_bytes)
        if key in seen:
            continue
        seen.add(key)
        totals[device] = totals.get(device, 0) + logical_bytes
    return totals


def process_snapshot() -> dict[str, Any]:
    """Return process and allocator memory without requiring diagnostics output."""
    return {
        "proc_bytes": _proc_memory_bytes(),
        "jemalloc_bytes": _jemalloc_stats_bytes(),
    }


def emit(
    label: str,
    *,
    modules: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not enabled():
        return
    report: dict[str, Any] = {
        "label": label,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        **process_snapshot(),
    }
    if modules:
        report["module_tensor_bytes"] = {
            name: _module_tensor_bytes(module)
            for name, module in modules.items()
        }
    if extra:
        report["extra"] = extra
    print(
        "UNIREC_HOST_MEMORY " + json.dumps(report, sort_keys=True),
        flush=True,
    )
