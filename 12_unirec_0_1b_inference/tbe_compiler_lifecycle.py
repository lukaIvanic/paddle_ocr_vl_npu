"""Release CANN TBE compiler subprocesses after all serving graphs are warm."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


_DEINITIALIZED = False


def enabled() -> bool:
    return os.environ.get("UNIREC_DEINIT_TBE_AFTER_WARMUP", "0") == "1"


def _process_pss_bytes(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        for line in path.read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def deinitialize_after_warmup(label: str) -> dict[str, Any] | None:
    """Destroy the process-local TBE compiler pool once serving is replay-only."""
    global _DEINITIALIZED
    if not enabled() or _DEINITIALIZED:
        return None

    from te_fusion import parallel_compilation

    compiler = parallel_compilation.OpCompiler.compiler
    worker_pids = []
    if compiler is not None:
        worker_pids = [
            int(process.pid)
            for process in getattr(compiler, "_worker_list", ())
            if process.pid is not None and process.is_alive()
        ]
    pss_by_pid = {
        str(pid): pss
        for pid in worker_pids
        if (pss := _process_pss_bytes(pid)) is not None
    }

    started = time.perf_counter()
    parallel_compilation.deinit_multi_process_env()
    for pid in worker_pids:
        deadline = time.monotonic() + 5.0
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    _DEINITIALIZED = True

    report = {
        "label": label,
        "pid": os.getpid(),
        "compiler_worker_pids": worker_pids,
        "compiler_worker_pss_bytes": pss_by_pid,
        "compiler_worker_pss_total_bytes": sum(pss_by_pid.values()),
        "workers_still_alive": [
            pid for pid in worker_pids if Path(f"/proc/{pid}").exists()
        ],
        "wall_s": time.perf_counter() - started,
    }
    print(
        "UNIREC_TBE_COMPILER_DEINIT " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return report
