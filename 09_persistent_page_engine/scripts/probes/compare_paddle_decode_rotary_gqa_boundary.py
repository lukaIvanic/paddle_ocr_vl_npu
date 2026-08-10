#!/usr/bin/env python3
"""Validate the real Paddle B1 rotary-to-fused-GQA SuperKernel boundary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_gqa_increfa_aiv import (
    decode_gqa_incre_flash_attention_aiv,
    register_decode_gqa_increfa_aiv_converter,
)


class DecodeRotaryGqaBoundary(torch.nn.Module):
    def __init__(self, super_kernel_options: str) -> None:
        super().__init__()
        self.super_kernel_options = super_kernel_options
        self.scope = __import__(
            "torchair.scope", fromlist=["super_kernel"]
        ).super_kernel

    def _forward_impl(
        self,
        query_bsnd: torch.Tensor,
        key_bsnd: torch.Tensor,
        value_state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd,
            key_bsnd,
            cos,
            sin,
            layout="BSND",
            rotary_mode="half",
        )
        return decode_gqa_incre_flash_attention_aiv(
            query_bsnd.transpose(1, 2),
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            key_bsnd.transpose(1, 2),
            value_state,
            scale_value=1.0 / math.sqrt(128.0),
            vector_core_count=16,
        )

    def forward(
        self,
        query_bsnd: torch.Tensor,
        key_bsnd: torch.Tensor,
        value_state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        with self.scope(
            "paddle_decode_rotary_gqa_boundary",
            self.super_kernel_options,
        ):
            return self._forward_impl(
                query_bsnd,
                key_bsnd,
                value_state,
                cos,
                sin,
                key_cache,
                value_cache,
                attention_mask,
                cache_position,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--super-kernel-options",
        default=(
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=per-func:early-start=0:split-mode=4"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_gqa_increfa_aiv_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    base_query_bsnd = torch.randn(
        (1, 1, 16, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    base_key_bsnd = torch.randn(
        (1, 1, 2, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    value_states = [
        torch.randn(
            (1, 2, 1, 128), generator=generator, dtype=torch.float16
        ).to("npu:0")
        for _ in range(2)
    ]
    angles = torch.randn(
        (1, 1, 1, 128), generator=generator, dtype=torch.float16
    )
    cos = torch.cos(angles.float()).to(torch.float16).to("npu:0")
    sin = torch.sin(angles.float()).to(torch.float16).to("npu:0")
    key_cache = torch.zeros(
        (1, 2, 1024, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    attention_mask = torch.zeros(
        (1, 1, 1, 1024), dtype=torch.bool, device="npu:0"
    )
    ref_key_cache = torch.zeros_like(key_cache)
    ref_value_cache = torch.zeros_like(value_cache)

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        DecodeRotaryGqaBoundary(args.super_kernel_options).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    checks: list[dict[str, object]] = []
    timings: list[float] = []
    all_close = True
    for position, value_state in zip((128, 129), value_states, strict=True):
        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device="npu:0"
        )
        reference_query_bsnd, reference_key_bsnd = (
            torch_npu.npu_apply_rotary_pos_emb(
                base_query_bsnd.clone(),
                base_key_bsnd.clone(),
                cos,
                sin,
                layout="BSND",
                rotary_mode="half",
            )
        )
        query_bsnd = base_query_bsnd.clone()
        key_bsnd = base_key_bsnd.clone()
        started = time.perf_counter()
        output = step(
            query_bsnd,
            key_bsnd,
            value_state,
            cos,
            sin,
            key_cache,
            value_cache,
            attention_mask,
            position_tensor,
        )
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)

        ref_key_state = reference_key_bsnd.transpose(1, 2)
        torch_npu.scatter_update_(
            ref_key_cache, position_tensor, ref_key_state, 2
        )
        torch_npu.scatter_update_(
            ref_value_cache, position_tensor, value_state, 2
        )
        expected_mask = (
            torch.arange(1024, dtype=torch.int64, device="npu:0") > position
        ).view(1, 1, 1, 1024)
        reference = torch_npu.npu_incre_flash_attention(
            reference_query_bsnd.transpose(1, 2),
            ref_key_cache,
            ref_value_cache,
            atten_mask=expected_mask,
            actual_seq_lengths=None,
            num_heads=16,
            num_key_value_heads=2,
            input_layout="BNSD",
            scale_value=1.0 / math.sqrt(128.0),
            inner_precise=1,
        )
        torch.npu.synchronize()

        output_cpu = output.float().cpu()
        reference_cpu = reference.float().cpu()
        output_diff = (output_cpu - reference_cpu).abs()
        output_exact = bool(torch.equal(output_cpu, reference_cpu))
        key_exact = bool(torch.equal(key_cache.cpu(), ref_key_cache.cpu()))
        value_exact = bool(
            torch.equal(value_cache.cpu(), ref_value_cache.cpu())
        )
        mask_exact = bool(torch.equal(attention_mask.cpu(), expected_mask.cpu()))
        passed = output_exact and key_exact and value_exact and mask_exact
        all_close = all_close and passed
        checks.append(
            {
                "position": position,
                "output_exact": output_exact,
                "output_max_abs": float(output_diff.max().item()),
                "key_cache_exact": key_exact,
                "value_cache_exact": value_exact,
                "attention_mask_exact": mask_exact,
            }
        )

    result = {
        "kind": "paddle_decode_rotary_gqa_boundary_probe",
        "contract": {
            "query_bsnd_shape": list(base_query_bsnd.shape),
            "key_bsnd_shape": list(base_key_bsnd.shape),
            "cache_shape": list(key_cache.shape),
            "factor_shape": list(cos.shape),
            "positions": [128, 129],
            "vector_core_count": 16,
            "super_kernel_options": args.super_kernel_options,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_required_checks_passed": all_close,
            "steps": checks,
        },
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_close:
        raise RuntimeError("rotary-to-fused-GQA boundary failed parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
