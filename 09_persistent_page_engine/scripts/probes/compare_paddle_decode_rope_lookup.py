#!/usr/bin/env python3
"""Validate the specialized B1 decode RoPE lookup through TorchAir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_rope_lookup import (
    GE_OP_NAME,
    PYTORCH_OP_NAME,
    decode_rope_lookup,
    register_decode_rope_lookup_converter,
)


class CustomRopeLookup(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            scope_module = __import__("torchair.scope", fromlist=["super_kernel"])
            self.scope = scope_module.super_kernel

    def _forward_impl(
        self,
        factor_lut: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_rope_lookup(factor_lut, cache_position, rope_delta)

    def forward(
        self,
        factor_lut: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scope is None:
            return self._forward_impl(factor_lut, cache_position, rope_delta)
        with self.scope(
            "paddle_decode_rope_lookup_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return self._forward_impl(factor_lut, cache_position, rope_delta)


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

    position = args.cache_position + args.rope_delta
    if not 0 <= position < 1024:
        raise ValueError("cache position plus RoPE delta must be in [0,1024)")
    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_rope_lookup_converter()

    inv_freq = 1.0 / (
        500000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
    )
    positions = torch.arange(1024, dtype=torch.float32)
    freqs = positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(torch.float16)
    factor_lut = factor_lut.contiguous().to("npu:0")
    cache_position = torch.tensor(
        [[args.cache_position]], dtype=torch.int64, device="npu:0"
    )
    rope_delta = torch.tensor(
        [[args.rope_delta]], dtype=torch.int64, device="npu:0"
    )
    reference = (
        factor_lut[0, position].view(1, 1, 1, 128),
        factor_lut[1, position].view(1, 1, 1, 128),
    )
    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        CustomRopeLookup(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    started = time.perf_counter()
    output = step(factor_lut, cache_position, rope_delta)
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - started

    comparisons = {}
    all_exact = True
    for name, actual, expected in zip(
        ("cos", "sin"), output, reference, strict=True
    ):
        actual_cpu = actual.float().cpu()
        expected_cpu = expected.float().cpu()
        exact = bool(torch.equal(actual_cpu, expected_cpu))
        comparisons[name] = {
            "shape": list(actual.shape),
            "exact": exact,
            "max_abs": float((actual_cpu - expected_cpu).abs().max().item()),
            "actual_first_8": actual_cpu.flatten()[:8].tolist(),
            "expected_first_8": expected_cpu.flatten()[:8].tolist(),
        }
        all_exact = all_exact and exact
    result = {
        "kind": "paddle_decode_rope_lookup_torchair_probe",
        "operator": {"pytorch": PYTORCH_OP_NAME, "ge": GE_OP_NAME},
        "contract": {
            "factor_lut_shape": list(factor_lut.shape),
            "output_shape": list(output[0].shape),
            "dtype": str(factor_lut.dtype),
            "strict_scope": args.strict_scope,
            "cache_position": args.cache_position,
            "rope_delta": args.rope_delta,
            "selected_position": position,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {"all_exact": all_exact, "outputs": comparisons},
        "timing": {"first_call_s": first_call_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_exact:
        raise RuntimeError("custom decode RoPE lookup mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
