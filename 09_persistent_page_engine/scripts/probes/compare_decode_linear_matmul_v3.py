#!/usr/bin/env python3
"""Compare stock FP16 Linear with installed AscendC MatMulV3 on B1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_linear_matmul_v3 import (
    GE_OP_NAME,
    PYTORCH_OP_NAME,
    decode_linear_matmul_v3,
    register_decode_linear_matmul_v3_converter,
)


FRACTAL_NZ = 29


class DecodeLinearMatMulV3(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.strict_scope = strict_scope
        self.scope = None
        if strict_scope:
            scope_module = __import__("torchair.scope", fromlist=["super_kernel"])
            self.scope = scope_module.super_kernel

    def _forward_impl(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return decode_linear_matmul_v3(x, weight)

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        if self.scope is None:
            return self._forward_impl(x, weight)
        with self.scope(
            "paddle_decode_linear_matmul_v3_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return self._forward_impl(x, weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_linear_matmul_v3_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    x = torch.randn((1, args.k), generator=generator, dtype=torch.float16).to(
        "npu:0"
    )
    weight = torch.randn(
        (args.n, args.k), generator=generator, dtype=torch.float16
    ).to("npu:0")
    weight = torch_npu.npu_format_cast(weight, FRACTAL_NZ)
    reference = F.linear(x, weight)

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        DecodeLinearMatMulV3(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    started = time.perf_counter()
    output = step(x, weight)
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - started

    output_cpu = output.float().cpu()
    reference_cpu = reference.float().cpu()
    diff = (output_cpu - reference_cpu).abs()
    result = {
        "kind": "decode_linear_matmul_v3_probe",
        "operator": {"pytorch": PYTORCH_OP_NAME, "ge": GE_OP_NAME},
        "contract": {
            "x_shape": list(x.shape),
            "weight_shape": list(weight.shape),
            "output_shape": list(output.shape),
            "dtype": str(x.dtype),
            "weight_npu_format": int(torch_npu.get_npu_format(weight)),
            "transpose_x2": True,
            "strict_scope": args.strict_scope,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "exact": bool(torch.equal(output_cpu, reference_cpu)),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "argmax_match": bool(
                output_cpu.argmax(dim=-1).item()
                == reference_cpu.argmax(dim=-1).item()
            ),
        },
        "timing": {"first_call_s": first_call_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["correctness"]["exact"]:
        raise RuntimeError(
            "MatMulV3 differs from stock Linear: "
            f"max_abs={result['correctness']['max_abs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
