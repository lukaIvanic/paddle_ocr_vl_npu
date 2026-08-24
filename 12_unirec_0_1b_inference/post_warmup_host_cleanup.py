"""Optional host allocator cleanup after every serving graph is resident."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from typing import Any

from host_memory_diagnostics import process_snapshot


_CLEANED = False


def enabled() -> bool:
    return os.environ.get("UNIREC_PURGE_HOST_AFTER_WARMUP", "0") == "1"


def _purge_all_jemalloc_arenas() -> int | None:
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
    # 4096 is jemalloc's MALLCTL_ARENAS_ALL sentinel.
    return int(mallctl(b"arena.4096.purge", None, None, None, 0))


def purge_host_allocator_pages() -> int | None:
    """Return currently unused jemalloc pages without the one-shot gate."""
    if not enabled():
        return None
    return _purge_all_jemalloc_arenas()


def cleanup_after_warmup(label: str) -> dict[str, Any] | None:
    """Collect dead Python objects and return unused jemalloc pages to Linux."""
    global _CLEANED
    if not enabled() or _CLEANED:
        return None

    before = process_snapshot()
    started = time.perf_counter()
    collected = gc.collect()
    purge_status = _purge_all_jemalloc_arenas()
    after = process_snapshot()
    _CLEANED = True

    before_pss = int(before["proc_bytes"]["pss"])
    after_pss = int(after["proc_bytes"]["pss"])
    report = {
        "label": label,
        "pid": os.getpid(),
        "gc_collected": int(collected),
        "jemalloc_purge_status": purge_status,
        "before": before,
        "after": after,
        "pss_reclaimed_bytes": max(0, before_pss - after_pss),
        "wall_s": time.perf_counter() - started,
    }
    print(
        "UNIREC_POST_WARMUP_HOST_CLEANUP "
        + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return report
