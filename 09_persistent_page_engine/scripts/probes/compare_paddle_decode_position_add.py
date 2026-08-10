#!/usr/bin/env python3
"""Validate the specialized B1 decode-position add through TorchAir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_position_add import (
    GE_OP_NAME,
    PYTORCH_OP_NAME,
    decode_position_add,
    register_decode_position_add_converter,
)


class CustomPositionAdd(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            scope_module = __import__("torchair.scope", fromlist=["super_kernel"])
            self.scope = scope_module.super_kernel

    def _forward_impl(
        self, cache_position: torch.Tensor, rope_delta: torch.Tensor
    ) -> torch.Tensor:
        return decode_position_add(cache_position, rope_delta)

    def forward(
        self, cache_position: torch.Tensor, rope_delta: torch.Tensor
    ) -> torch.Tensor:
        if self.scope is None:
            return self._forward_impl(cache_position, rope_delta)
        with self.scope(
            "paddle_decode_position_add_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return self._forward_impl(cache_position, rope_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    parser.add_argument("--cache-position", type=int, default=128)
    parser.add_argument("--rope-delta", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_position_add_converter()

    cache_position = torch.tensor(
        [[args.cache_position]], dtype=torch.int64, device="npu:0"
    )
    rope_delta = torch.tensor(
        [[args.rope_delta]], dtype=torch.int64, device="npu:0"
    )
    reference = cache_position + rope_delta
    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        CustomPositionAdd(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    started = time.perf_counter()
    output = step(cache_position, rope_delta)
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - started

    output_cpu = output.cpu()
    reference_cpu = reference.cpu()
    exact = bool(torch.equal(output_cpu, reference_cpu))
    result = {
        "kind": "paddle_decode_position_add_torchair_probe",
        "operator": {"pytorch": PYTORCH_OP_NAME, "ge": GE_OP_NAME},
        "contract": {
            "input_shape": list(cache_position.shape),
            "dtype": str(cache_position.dtype),
            "strict_scope": args.strict_scope,
            "cache_position": args.cache_position,
            "rope_delta": args.rope_delta,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "exact": exact,
            "actual": output_cpu.tolist(),
            "expected": reference_cpu.tolist(),
        },
        "timing": {"first_call_s": first_call_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not exact:
        raise RuntimeError("custom decode-position add mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
