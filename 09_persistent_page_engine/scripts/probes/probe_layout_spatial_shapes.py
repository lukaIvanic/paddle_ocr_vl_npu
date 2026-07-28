#!/usr/bin/env python3
"""Probe PP-DocLayoutV3 spatial-shape construction without loading a model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import torch
import torch_npu  # noqa: F401


SPATIAL_SHAPES = ((100, 100), (50, 50), (25, 25))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("legacy_indexput", "single_constructor"),
        required=True,
    )
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("this probe requires an NPU device")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")
    torch.npu.set_compile_mode(jit_compile=False)

    if args.mode == "legacy_indexput":
        result = torch.empty(
            (len(SPATIAL_SHAPES), 2),
            device=device,
            dtype=torch.long,
        )
        for level, (height, width) in enumerate(SPATIAL_SHAPES):
            result[level, 0] = height
            result[level, 1] = width
    else:
        result = torch.tensor(
            SPATIAL_SHAPES,
            device=device,
            dtype=torch.long,
        )

    torch.npu.synchronize()
    actual = result.cpu().tolist()
    expected = [list(shape) for shape in SPATIAL_SHAPES]
    if actual != expected:
        raise AssertionError(
            f"spatial-shape mismatch: actual={actual}, expected={expected}"
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "torch": torch.__version__,
                "torch_npu": getattr(torch_npu, "__version__", "<missing>"),
                "spatial_shapes": actual,
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
