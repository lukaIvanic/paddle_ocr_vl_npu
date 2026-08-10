#!/usr/bin/env python3
"""Validate cube-first MatMul/QKV to the mixed-task fused GQA."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_gqa_increfa_mixed import (
    decode_gqa_incre_flash_attention_mixed,
    decode_gqa_incre_flash_attention_mixed24,
    register_decode_gqa_increfa_mixed_converter,
    register_decode_gqa_increfa_mixed24_converter,
)
from paddleocr_vl.model.decode_linear_matmul_v3 import (
    decode_linear_matmul_v3,
    register_decode_linear_matmul_v3_converter,
)
from paddleocr_vl.model.decode_qkv_split import (
    decode_qkv_split,
    register_decode_qkv_split_converter,
)


FRACTAL_NZ = 29


class DecodeMatMulQkvGqaMixedBoundary(torch.nn.Module):
    def __init__(self, geometry: str, super_kernel_options: str) -> None:
        super().__init__()
        self.geometry = geometry
        self.super_kernel_options = super_kernel_options
        self.scope = __import__(
            "torchair.scope", fromlist=["super_kernel"]
        ).super_kernel

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        packed_qkv = decode_linear_matmul_v3(hidden_states, qkv_weight)
        query, key_state, value_state = decode_qkv_split(
            packed_qkv.reshape(1, 1, 2560)
        )
        fused_attention = (
            decode_gqa_incre_flash_attention_mixed24
            if self.geometry == "mixed24"
            else decode_gqa_incre_flash_attention_mixed
        )
        return fused_attention(
            query,
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            key_state,
            value_state,
            scale_value=1.0 / math.sqrt(128.0),
            vector_core_count=16,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        with self.scope(
            f"paddle_decode_matmul_qkv_gqa_{self.geometry}_boundary",
            self.super_kernel_options,
        ):
            return self._forward_impl(
                hidden_states,
                qkv_weight,
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
        "--geometry",
        choices=("mixed", "mixed24"),
        default="mixed",
        help="Select the independent fused GQA operator identity.",
    )
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
    register_decode_linear_matmul_v3_converter()
    register_decode_qkv_split_converter()
    if args.geometry == "mixed24":
        register_decode_gqa_increfa_mixed24_converter()
    else:
        register_decode_gqa_increfa_mixed_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    hidden_states = torch.randn(
        (1, 1024), generator=generator, dtype=torch.float16
    ).to("npu:0")
    qkv_weight = torch.randn(
        (2560, 1024), generator=generator, dtype=torch.float16
    ).to("npu:0")
    qkv_weight = torch_npu.npu_format_cast(qkv_weight, FRACTAL_NZ)
    key_cache = torch.zeros(
        (1, 2, 1024, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    attention_mask = torch.zeros(
        (1, 1, 1, 1024), dtype=torch.bool, device="npu:0"
    )
    ref_key_cache = torch.zeros_like(key_cache)
    ref_value_cache = torch.zeros_like(value_cache)

    reference_qkv = F.linear(hidden_states, qkv_weight).reshape(1, 1, 2560)
    reference_query_raw, reference_key_raw, reference_value_raw = (
        reference_qkv.split((2048, 256, 256), dim=-1)
    )
    reference_query = reference_query_raw.view(1, 1, 16, 128).transpose(1, 2)
    reference_key_state = reference_key_raw.view(1, 1, 2, 128).transpose(1, 2)
    reference_value_state = reference_value_raw.view(
        1, 1, 2, 128
    ).transpose(1, 2)

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        DecodeMatMulQkvGqaMixedBoundary(
            args.geometry, args.super_kernel_options
        ).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    checks: list[dict[str, object]] = []
    timings: list[float] = []
    all_close = True
    for position in (128, 129):
        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device="npu:0"
        )
        started = time.perf_counter()
        output = step(
            hidden_states,
            qkv_weight,
            key_cache,
            value_cache,
            attention_mask,
            position_tensor,
        )
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)

        torch_npu.scatter_update_(
            ref_key_cache, position_tensor, reference_key_state, 2
        )
        torch_npu.scatter_update_(
            ref_value_cache, position_tensor, reference_value_state, 2
        )
        expected_mask = (
            torch.arange(1024, dtype=torch.int64, device="npu:0") > position
        ).view(1, 1, 1, 1024)
        reference = torch_npu.npu_incre_flash_attention(
            reference_query,
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
        key_cache_cpu = key_cache.float().cpu()
        ref_key_cache_cpu = ref_key_cache.float().cpu()
        value_cache_cpu = value_cache.float().cpu()
        ref_value_cache_cpu = ref_value_cache.float().cpu()
        output_diff = (output_cpu - reference_cpu).abs()
        key_diff = (key_cache_cpu - ref_key_cache_cpu).abs()
        value_diff = (value_cache_cpu - ref_value_cache_cpu).abs()
        output_close = bool(
            torch.allclose(output_cpu, reference_cpu, atol=2.0e-3, rtol=2.0e-3)
        )
        key_close = bool(
            torch.allclose(
                key_cache_cpu, ref_key_cache_cpu, atol=2.0e-3, rtol=2.0e-3
            )
        )
        value_close = bool(
            torch.allclose(
                value_cache_cpu,
                ref_value_cache_cpu,
                atol=2.0e-3,
                rtol=2.0e-3,
            )
        )
        mask_exact = bool(torch.equal(attention_mask.cpu(), expected_mask.cpu()))
        passed = output_close and key_close and value_close and mask_exact
        all_close = all_close and passed
        checks.append(
            {
                "position": position,
                "output_allclose_atol_2e_3_rtol_2e_3": output_close,
                "output_max_abs": float(output_diff.max().item()),
                "key_cache_allclose_atol_2e_3_rtol_2e_3": key_close,
                "key_cache_max_abs": float(key_diff.max().item()),
                "value_cache_allclose_atol_2e_3_rtol_2e_3": value_close,
                "value_cache_max_abs": float(value_diff.max().item()),
                "attention_mask_exact": mask_exact,
            }
        )

    result = {
        "kind": f"paddle_decode_matmul_qkv_gqa_{args.geometry}_boundary_probe",
        "operator": {
            "pytorch": (
                "paddleocr_vl::decode_gqa_incre_flash_attention_mixed24"
                if args.geometry == "mixed24"
                else "paddleocr_vl::decode_gqa_incre_flash_attention_mixed"
            ),
            "ge": (
                "PaddleDecodeGqaIncreFlashAttentionMixed24"
                if args.geometry == "mixed24"
                else "PaddleDecodeGqaIncreFlashAttentionMixed"
            ),
        },
        "contract": {
            "hidden_shape": list(hidden_states.shape),
            "qkv_weight_shape": list(qkv_weight.shape),
            "qkv_weight_npu_format": int(torch_npu.get_npu_format(qkv_weight)),
            "cache_shape": list(key_cache.shape),
            "positions": [128, 129],
            "vector_core_count": 16,
            "launch_aiv_count": 24 if args.geometry == "mixed24" else 16,
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
        raise RuntimeError("cube-first MatMul/QKV/mixed-GQA boundary failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
