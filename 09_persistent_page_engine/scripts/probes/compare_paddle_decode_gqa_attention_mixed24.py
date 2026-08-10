#!/usr/bin/env python3
"""Validate the independent mixed24 attention-only operator through TorchAir."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_gqa_attention_mixed24 import (
    decode_gqa_attention_mixed24,
    register_decode_gqa_attention_mixed24_converter,
)


class DecodeGqaAttentionMixed24(torch.nn.Module):
    def __init__(self, strict_scope: bool, super_kernel_options: str) -> None:
        super().__init__()
        self.super_kernel_options = super_kernel_options
        self.scope = None
        if strict_scope:
            self.scope = __import__(
                "torchair.scope", fromlist=["super_kernel"]
            ).super_kernel

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return decode_gqa_attention_mixed24(
            query,
            key,
            value,
            attention_mask,
            num_heads=16,
            num_key_value_heads=2,
            scale_value=1.0 / math.sqrt(128.0),
            inner_precise=1,
            vector_core_count=16,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        args = (query, key, value, attention_mask)
        if self.scope is None:
            return self._forward_impl(*args)
        with self.scope(
            "paddle_decode_gqa_attention_mixed24_probe",
            self.super_kernel_options,
        ):
            return self._forward_impl(*args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    parser.add_argument(
        "--super-kernel-options",
        default=(
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=per-func:early-start=1:split-mode=4"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_gqa_attention_mixed24_converter()

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    compiler_config = CompilerConfig()
    graph_dump_dir = args.cache_dir / "graph_dump"
    graph_dump_dir.mkdir(parents=True, exist_ok=True)
    compiler_config.debug.graph_dump.type = "pbtxt"
    compiler_config.debug.graph_dump.path = str(graph_dump_dir)
    step = torchair.inference.cache_compile(
        DecodeGqaAttentionMixed24(
            args.strict_scope,
            args.super_kernel_options,
        ).forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    query = torch.randn(
        (1, 16, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    key = torch.randn(
        (1, 2, 1024, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    value = torch.randn(
        (1, 2, 1024, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")

    checks = []
    timings = []
    for position in (128, 129, 1023):
        attention_mask = (
            torch.arange(1024, device="npu:0", dtype=torch.int64) > position
        ).view(1, 1, 1, 1024)
        reference = torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            atten_mask=attention_mask,
            actual_seq_lengths=None,
            num_heads=16,
            num_key_value_heads=2,
            input_layout="BNSD",
            scale_value=1.0 / math.sqrt(128.0),
            inner_precise=1,
        )
        torch.npu.synchronize()
        started = time.perf_counter()
        actual = step(query, key, value, attention_mask)
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)
        actual_cpu = actual.cpu()
        reference_cpu = reference.cpu()
        max_abs = float((actual_cpu.float() - reference_cpu.float()).abs().max())
        checks.append(
            {
                "position": position,
                "exact": bool(torch.equal(actual_cpu, reference_cpu)),
                "allclose_atol_1e_3_rtol_1e_3": bool(
                    torch.allclose(
                        actual_cpu,
                        reference_cpu,
                        atol=1e-3,
                        rtol=1e-3,
                    )
                ),
                "max_abs": max_abs,
            }
        )

    summary = {
        "kind": "paddle_decode_gqa_attention_mixed24_torchair_probe",
        "operator": {
            "pytorch": "paddleocr_vl::decode_gqa_attention_mixed24",
            "ge": "PaddleDecodeGqaAttentionMixed24",
            "kernel": "paddle_decode_gqa_attention_aiv",
        },
        "contract": {
            "query_shape": [1, 16, 1, 128],
            "cache_shape": [1, 2, 1024, 128],
            "strict_scope": args.strict_scope,
            "super_kernel_options": args.super_kernel_options,
            "outer_vector_core_count": 24,
            "attention_worker_count": 16,
        },
        "environment": {
            "device": torch_npu.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_close": all(
                check["allclose_atol_1e_3_rtol_1e_3"] for check in checks
            ),
            "all_exact": all(check["exact"] for check in checks),
            "checks": checks,
        },
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["correctness"]["all_close"]:
        raise RuntimeError("mixed24 attention-only parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
