from __future__ import annotations

from pathlib import Path

import torch


def _load_extension() -> Path:
    candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if len(candidates) != 1:
        raise ImportError("expected one built _C*.so; run pytorch_extension/build.sh")
    torch.ops.load_library(str(candidates[0]))
    return candidates[0]


EXTENSION_PATH = _load_extension()
