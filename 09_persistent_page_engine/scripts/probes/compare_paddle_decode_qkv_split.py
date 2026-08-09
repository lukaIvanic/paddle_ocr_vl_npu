#!/usr/bin/env python3
"""Validate the specialized B1 packed-QKV split through TorchAir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_qkv_split import (
    GE_OP_NAME,
    PYTORCH_OP_NAME,
    decode_qkv_split,
    register_decode_qkv_split_converter,
)


class CustomQkvSplit(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            scope_module = __import__("torchair.scope", fromlist=["super_kernel"])
            self.scope = scope_module.super_kernel

    def _forward_impl(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return decode_qkv_split(qkv)

    def forward(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scope is None:
            return self._forward_impl(qkv)
        with self.scope(
            "paddle_decode_qkv_split_probe",
            "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort",
        ):
            return self._forward_impl(qkv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_qkv_split_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    qkv = torch.randn((1, 1, 2560), generator=generator, dtype=torch.float16)
    qkv = qkv.to("npu:0")
    query, key, value = qkv.split((2048, 256, 256), dim=-1)
    reference = (
        query.view(1, 1, 16, 128).transpose(1, 2),
        key.view(1, 1, 2, 128).transpose(1, 2),
        value.view(1, 1, 2, 128).transpose(1, 2),
    )

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        CustomQkvSplit(args.strict_scope).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    started = time.perf_counter()
    output = step(qkv)
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - started

    comparisons = {}
    all_exact = True
    output_layouts = {}
    for name, actual, expected in zip(
        ("query", "key", "value"), output, reference, strict=True
    ):
        actual_cpu = actual.float().cpu()
        expected_cpu = expected.float().cpu()
        exact = bool(torch.equal(actual_cpu, expected_cpu))
        comparisons[name] = {
            "shape": list(actual.shape),
            "exact": exact,
            "max_abs": float((actual_cpu - expected_cpu).abs().max().item()),
            "actual_abs_max": float(actual_cpu.abs().max().item()),
            "actual_nonzero_count": int(torch.count_nonzero(actual_cpu).item()),
            "actual_first_8": actual_cpu.flatten()[:8].tolist(),
            "expected_first_8": expected_cpu.flatten()[:8].tolist(),
        }
        output_layouts[name] = {
            "data_ptr": int(actual.data_ptr()),
            "storage_nbytes": int(actual.untyped_storage().nbytes()),
            "storage_offset": int(actual.storage_offset()),
            "stride": list(actual.stride()),
            "npu_format": int(torch_npu.get_npu_format(actual)),
        }
        all_exact = all_exact and exact

    actual_query = output[0].float().cpu().flatten()
    expected_query = reference[0].float().cpu().flatten()
    expected_key = reference[1].float().cpu().flatten()
    expected_value = reference[2].float().cpu().flatten()
    region_diagnostics = {
        "query_prefix_256_equals_expected_key": bool(
            torch.equal(actual_query[:256], expected_key)
        ),
        "query_prefix_256_equals_expected_value": bool(
            torch.equal(actual_query[:256], expected_value)
        ),
        "query_prefix_256_equals_expected_query": bool(
            torch.equal(actual_query[:256], expected_query[:256])
        ),
        "query_tail_1792_equals_expected_query": bool(
            torch.equal(actual_query[256:], expected_query[256:])
        ),
        "pointer_deltas_bytes": {
            "key_minus_query": (
                output_layouts["key"]["data_ptr"]
                - output_layouts["query"]["data_ptr"]
            ),
            "value_minus_query": (
                output_layouts["value"]["data_ptr"]
                - output_layouts["query"]["data_ptr"]
            ),
            "value_minus_key": (
                output_layouts["value"]["data_ptr"]
                - output_layouts["key"]["data_ptr"]
            ),
        },
    }

    result = {
        "kind": "paddle_decode_qkv_split_torchair_probe",
        "operator": {"pytorch": PYTORCH_OP_NAME, "ge": GE_OP_NAME},
        "contract": {
            "qkv_shape": list(qkv.shape),
            "dtype": str(qkv.dtype),
            "strict_scope": args.strict_scope,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_exact": all_exact,
            "outputs": comparisons,
            "output_layouts": output_layouts,
            "region_diagnostics": region_diagnostics,
        },
        "timing": {"first_call_s": first_call_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_exact:
        raise RuntimeError("custom QKV split does not match the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
