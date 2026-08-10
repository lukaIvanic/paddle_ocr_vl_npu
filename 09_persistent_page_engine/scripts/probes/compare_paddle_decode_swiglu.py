#!/usr/bin/env python3
"""Validate PaddleDecodeSwiGluV1 through TorchAir on Ascend 910B2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_swiglu import (
    PYTORCH_OP_NAME,
    decode_swiglu,
    register_decode_swiglu_converter,
)


class DecodeSwiGlu(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            self.scope = __import__(
                "torchair.scope", fromlist=["super_kernel"]
            ).super_kernel

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.scope is None:
            return decode_swiglu(gate, up)
        with self.scope(
            "paddle_decode_swiglu_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return decode_swiglu(gate, up)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_swiglu_converter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    gate_cpu = torch.randn((1, 1, 3072), generator=generator, dtype=torch.float16)
    up_cpu = torch.randn((1, 1, 3072), generator=generator, dtype=torch.float16)
    reference = (torch.nn.functional.silu(gate_cpu.float()) * up_cpu.float()).half()
    gate = gate_cpu.to("npu:0")
    up = up_cpu.to("npu:0")

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        DecodeSwiGlu(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    first = step(gate, up)
    torch.npu.synchronize()
    actual = first.cpu()
    diff = (actual.float() - reference.float()).abs()
    for _ in range(args.warmup):
        step(gate, up)
    torch.npu.synchronize()
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        step(gate, up)
        torch.npu.synchronize()
        durations.append((time.perf_counter() - started) * 1e6)
    result = {
        "kind": "paddle_decode_swiglu_torchair_probe",
        "operator": {
            "pytorch": PYTORCH_OP_NAME,
            "ge": "PaddleDecodeSwiGluV1",
            "kernel": "paddle_decode_swi_glu_v1",
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "contract": {
            "shape": list(gate.shape),
            "dtype": str(gate.dtype),
            "strict_scope": args.strict_scope,
            "block_dim": 1,
            "core_type": "AIV_ONLY",
        },
        "correctness": {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "allclose_atol_2e_3_rtol_2e_3": bool(
                torch.allclose(actual, reference, atol=2e-3, rtol=2e-3)
            ),
        },
        "timing_us": {
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "repeats": len(durations),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["correctness"]["allclose_atol_2e_3_rtol_2e_3"]:
        raise RuntimeError("PaddleDecodeSwiGluV1 failed tolerance parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
