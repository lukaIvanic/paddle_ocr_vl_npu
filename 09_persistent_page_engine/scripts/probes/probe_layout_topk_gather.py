#!/usr/bin/env python3
"""Probe PP-DocLayoutV3 top-k row selection without loading the model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import torch
import torch_npu  # noqa: F401


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "legacy_advanced_index",
            "gather",
            "capture_gather",
        ),
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

    output_memory = torch.arange(
        2 * 8 * 4,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 8, 4)
    topk_indices = torch.tensor(
        ((7, 3, 1), (0, 4, 6)),
        device=device,
        dtype=torch.long,
    )

    def gather_rows() -> torch.Tensor:
        return output_memory.gather(
            dim=1,
            index=topk_indices.unsqueeze(-1).repeat(
                1,
                1,
                output_memory.shape[-1],
            ),
        )

    if args.mode == "legacy_advanced_index":
        batch_indices = torch.arange(
            output_memory.shape[0],
            device=device,
        ).unsqueeze(1)
        result = output_memory[batch_indices, topk_indices]
    elif args.mode == "gather":
        result = gather_rows()
    else:
        warmup = gather_rows()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = gather_rows()
        torch.npu.synchronize()
        graph.replay()
        del warmup

    torch.npu.synchronize()
    actual = result.cpu().tolist()
    expected = [
        [[28.0, 29.0, 30.0, 31.0],
         [12.0, 13.0, 14.0, 15.0],
         [4.0, 5.0, 6.0, 7.0]],
        [[32.0, 33.0, 34.0, 35.0],
         [48.0, 49.0, 50.0, 51.0],
         [56.0, 57.0, 58.0, 59.0]],
    ]
    if actual != expected:
        raise AssertionError(
            f"top-k gather mismatch: actual={actual}, expected={expected}"
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "torch": torch.__version__,
                "torch_npu": getattr(torch_npu, "__version__", "<missing>"),
                "shape": list(result.shape),
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
