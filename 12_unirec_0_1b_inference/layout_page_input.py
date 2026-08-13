"""Shared production page decoding and layout-input materialization.

The UniRec worker and layout lab import these exact helpers. Keep image codec
selection and RGB-to-BGR storage semantics here so the lab cannot drift from
the production layout boundary.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_page_rgb(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Decode one production page as uint8 RGB HWC."""
    started = time.perf_counter()
    encoded = path.read_bytes()
    read_s = time.perf_counter() - started

    started = time.perf_counter()
    if encoded.startswith(PNG_SIGNATURE):
        from kornia_rs.image import Image as KorniaImage

        rgb = KorniaImage.decode(encoded, "RGB").data
    else:
        import torch
        from torchvision.io import ImageReadMode, decode_image

        encoded_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        rgb = (
            decode_image(encoded_tensor, mode=ImageReadMode.RGB)
            .permute(1, 2, 0)
            .numpy()
        )
    decode_s = time.perf_counter() - started
    _validate_rgb(rgb)
    return rgb, {"file_read_s": read_s, "direct_rgb_decode_s": decode_s}


def materialize_layout_bgr(rgb: np.ndarray) -> np.ndarray:
    """Create the exact contiguous BGR page passed to the layout adapter."""
    _validate_rgb(rgb)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    if not bgr.flags.c_contiguous or bgr.dtype != np.uint8:
        raise RuntimeError("layout BGR materialization must be contiguous uint8")
    return bgr


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"unsupported decoded image: {rgb.shape} {rgb.dtype}")
