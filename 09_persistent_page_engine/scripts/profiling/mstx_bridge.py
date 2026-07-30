"""Build and call a tiny bridge to CANN's public MSTX C API.

``torch_npu.npu.mstx`` returned range ID zero under the installed CANN 9
``msprof op`` launcher. The CANN MSTX headers expose the intended injection
protocol directly: on first call they load ``MSTX_INJECTION_PATH`` and request
the profiler function table. This helper preserves that public mechanism
while keeping the generated shared object in the ignored runtime cache.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


SOURCE = Path(__file__).with_name("mstx_bridge.c")


def _cann_root() -> Path:
    values = (
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/cann-9.0.0",
    )
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            return path
    raise RuntimeError(
        "CANN root was not found through ASCEND_HOME_PATH, "
        "ASCEND_TOOLKIT_HOME, or the installed CANN 9 path"
    )


def _include_dir(root: Path) -> Path:
    candidates = (
        root / "include",
        root / "aarch64-linux/include",
        root / "arm64-linux/include",
    )
    for path in candidates:
        if (path / "mstx/ms_tools_ext.h").is_file():
            return path
    raise RuntimeError(f"CANN MSTX headers were not found under {root}")


def build_bridge(cache_dir: Path) -> Path:
    """Build the bridge once for this exact source and CANN header tree."""
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("gcc is required to build the MSTX bridge")
    root = _cann_root()
    include_dir = _include_dir(root)
    digest = hashlib.sha256(
        SOURCE.read_bytes()
        + b"\0"
        + str(include_dir).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"mstx_bridge_{digest}.so"
    if output.is_file():
        return output

    with tempfile.TemporaryDirectory(
        prefix="mstx_bridge_",
        dir=cache_dir,
    ) as temporary:
        candidate = Path(temporary) / output.name
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c11",
            f"-I{include_dir}",
            str(SOURCE),
            "-ldl",
            "-o",
            str(candidate),
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not candidate.is_file():
            raise RuntimeError(
                "failed to build the CANN MSTX bridge:\n"
                f"command: {' '.join(command)}\n"
                f"stdout: {completed.stdout}\n"
                f"stderr: {completed.stderr}"
            )
        os.replace(candidate, output)
    return output


class MstxBridge:
    """Typed ctypes wrapper for the two MSTX range calls we need."""

    def __init__(self, cache_dir: Path) -> None:
        self.path = build_bridge(cache_dir)
        self._library = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        self._start = self._library.vision_mstx_range_start
        self._start.argtypes = (ctypes.c_char_p, ctypes.c_void_p)
        self._start.restype = ctypes.c_uint64
        self._end = self._library.vision_mstx_range_end
        self._end.argtypes = (ctypes.c_uint64,)
        self._end.restype = None

    def range_start(self, message: str, stream: int | None = None) -> int:
        if not message:
            raise ValueError("MSTX message must be non-empty")
        stream_pointer = ctypes.c_void_p(0 if stream is None else stream)
        return int(self._start(message.encode("ascii"), stream_pointer))

    def range_end(self, range_id: int) -> None:
        if range_id <= 0:
            raise ValueError(f"invalid MSTX range ID: {range_id}")
        self._end(range_id)
