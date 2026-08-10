#!/usr/bin/env python3
"""Validate CANN's AscendC ScatterPaKvCache for the B1 ND decode arena."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_kv_scatter import (
    decode_kv_scatter,
    register_decode_kv_scatter_converter,
)


class DecodeKvScatter(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            scope_module = __import__("torchair.scope", fromlist=["super_kernel"])
            self.scope = scope_module.super_kernel

    def _forward_impl(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_kv_scatter(
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
        )

    def forward(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scope is None:
            return self._forward_impl(
                key_cache,
                value_cache,
                cache_position,
                key_states,
                value_states,
            )
        with self.scope(
            "paddle_decode_kv_scatter_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return self._forward_impl(
                key_cache,
                value_cache,
                cache_position,
                key_states,
                value_states,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    parser.add_argument("--cache-position", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not 0 <= args.cache_position < 1024:
        raise ValueError("cache position must be in [0,1024)")
    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_kv_scatter_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    key_states = torch.randn(
        (1, 2, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    value_states = torch.randn(
        (1, 2, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    key_cache = torch.zeros(
        (1, 1024, 2, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    cache_position = torch.tensor(
        [args.cache_position], dtype=torch.int64, device="npu:0"
    )
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    expected_key[:, args.cache_position : args.cache_position + 1, :, :].copy_(
        key_states.transpose(1, 2)
    )
    expected_value[
        :, args.cache_position : args.cache_position + 1, :, :
    ].copy_(value_states.transpose(1, 2))

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        DecodeKvScatter(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    started = time.perf_counter()
    output_key, output_value = step(
        key_cache,
        value_cache,
        cache_position,
        key_states,
        value_states,
    )
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - started

    comparisons = {}
    all_exact = True
    for name, actual, expected in (
        ("key", output_key, expected_key),
        ("value", output_value, expected_value),
    ):
        actual_cpu = actual.float().cpu()
        expected_cpu = expected.float().cpu()
        exact = bool(torch.equal(actual_cpu, expected_cpu))
        comparisons[name] = {
            "exact": exact,
            "max_abs": float((actual_cpu - expected_cpu).abs().max().item()),
            "nonzero_count": int(torch.count_nonzero(actual_cpu).item()),
            "selected_first_8": actual_cpu[
                0, args.cache_position, 0, :8
            ].tolist(),
        }
        all_exact = all_exact and exact
    input_alias = {
        "key": int(output_key.data_ptr()) == int(key_cache.data_ptr()),
        "value": int(output_value.data_ptr()) == int(value_cache.data_ptr()),
    }
    result = {
        "kind": "paddle_decode_kv_scatter_pa_torchair_probe",
        "operator": {"pytorch": "npu::npu_scatter_pa_kv_cache", "ge": "ScatterPaKvCache"},
        "contract": {
            "cache_shape": list(key_cache.shape),
            "state_shape": list(key_states.shape),
            "cache_position": args.cache_position,
            "cache_mode": "Norm",
            "strict_scope": args.strict_scope,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_exact": all_exact,
            "outputs": comparisons,
            "output_aliases_input": input_alias,
        },
        "timing": {"first_call_s": first_call_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_exact:
        raise RuntimeError("ScatterPaKvCache does not match the ND cache update")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
