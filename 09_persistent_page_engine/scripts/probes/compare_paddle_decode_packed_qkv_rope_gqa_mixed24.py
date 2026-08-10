#!/usr/bin/env python3
"""Validate packed QKV/RoPE/GQA standalone and after Cube-first MatMul."""

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
from paddleocr_vl.model.decode_linear_matmul_v3 import (
    decode_linear_matmul_v3,
    register_decode_linear_matmul_v3_converter,
)
from paddleocr_vl.model.decode_packed_qkv_rope_gqa_mixed24 import (
    decode_packed_qkv_rope_gqa_mixed24,
    register_decode_packed_qkv_rope_gqa_mixed24_converter,
)


FRACTAL_NZ = 29


class PackedStep(torch.nn.Module):
    def __init__(self, mode: str, super_kernel_options: str) -> None:
        super().__init__()
        self.mode = mode
        self.super_kernel_options = super_kernel_options
        self.scope = __import__(
            "torchair.scope", fromlist=["super_kernel"]
        ).super_kernel

    def _run(
        self,
        qkv: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        factor_lut: torch.Tensor,
        rope_delta: torch.Tensor,
    ) -> torch.Tensor:
        return decode_packed_qkv_rope_gqa_mixed24(
            qkv.reshape(1, 1, 2560),
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            factor_lut,
            rope_delta,
            scale_value=1.0 / math.sqrt(128.0),
        )

    def forward(
        self,
        source: torch.Tensor,
        qkv_weight: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        factor_lut: torch.Tensor,
        rope_delta: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "standalone":
            return self._run(
                source,
                key_cache,
                value_cache,
                attention_mask,
                cache_position,
                factor_lut,
                rope_delta,
            )
        with self.scope("paddle_decode_packed_qkv_rope_gqa_mixed24_boundary", self.super_kernel_options):
            qkv = decode_linear_matmul_v3(source, qkv_weight)
            return self._run(
                qkv,
                key_cache,
                value_cache,
                attention_mask,
                cache_position,
                factor_lut,
                rope_delta,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standalone", "boundary"), required=True)
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


def rotate_half(states: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return (states * cosine) + (rotated * sine)


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_packed_qkv_rope_gqa_mixed24_converter()
    register_decode_linear_matmul_v3_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    hidden = torch.randn((1, 1024), generator=generator, dtype=torch.float16).to("npu:0")
    qkv_weight = torch.randn(
        (2560, 1024), generator=generator, dtype=torch.float16
    ).to("npu:0")
    qkv_weight = torch_npu.npu_format_cast(qkv_weight, FRACTAL_NZ)
    base_qkv = F.linear(hidden, qkv_weight).reshape(1, 1, 2560)
    source_template = base_qkv if args.mode == "standalone" else hidden
    angles = (
        torch.arange(1024, dtype=torch.float32).view(1024, 1)
        * torch.linspace(0.0001, 0.01, 64, dtype=torch.float32).view(1, 64)
    )
    factor_lut = torch.stack(
        (torch.cat((angles, angles), dim=-1).cos(),
         torch.cat((angles, angles), dim=-1).sin()),
        dim=0,
    ).to(dtype=torch.float16, device="npu:0")
    rope_delta = torch.tensor([[7]], dtype=torch.int64, device="npu:0")
    key_cache = torch.zeros((1, 2, 1024, 128), dtype=torch.float16, device="npu:0")
    value_cache = torch.zeros_like(key_cache)
    attention_mask = torch.zeros((1, 1, 1, 1024), dtype=torch.bool, device="npu:0")
    ref_key_cache = torch.zeros_like(key_cache)
    ref_value_cache = torch.zeros_like(value_cache)

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        PackedStep(args.mode, args.super_kernel_options).forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    checks: list[dict[str, object]] = []
    timings: list[float] = []
    all_passed = True
    for position in (128, 129):
        source = source_template.clone()
        position_tensor = torch.tensor([position], dtype=torch.int64, device="npu:0")
        started = time.perf_counter()
        output = step(
            source,
            qkv_weight,
            key_cache,
            value_cache,
            attention_mask,
            position_tensor,
            factor_lut,
            rope_delta,
        )
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)

        ref_qkv = base_qkv.clone()
        query_raw, key_raw, value_raw = ref_qkv.split((2048, 256, 256), dim=-1)
        cosine = factor_lut[0, position + 7].view(1, 1, 1, 128)
        sine = factor_lut[1, position + 7].view(1, 1, 1, 128)
        query = rotate_half(query_raw.view(1, 16, 1, 128), cosine, sine)
        key_state = rotate_half(key_raw.view(1, 2, 1, 128), cosine, sine)
        value_state = value_raw.view(1, 2, 1, 128)
        expected_qkv = torch.cat(
            (
                query.reshape(1, 1, 2048),
                key_state.reshape(1, 1, 256),
                value_state.reshape(1, 1, 256),
            ),
            dim=-1,
        )
        torch_npu.scatter_update_(ref_key_cache, position_tensor, key_state, 2)
        torch_npu.scatter_update_(ref_value_cache, position_tensor, value_state, 2)
        expected_mask = (
            torch.arange(1024, dtype=torch.int64, device="npu:0") > position
        ).view(1, 1, 1, 1024)
        reference = torch_npu.npu_incre_flash_attention(
            query,
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

        actual_qkv = source.reshape(1, 1, 2560) if args.mode == "standalone" else None
        output_diff = (output.float() - reference.float()).abs().max().item()
        key_diff = (key_cache.float() - ref_key_cache.float()).abs().max().item()
        value_diff = (value_cache.float() - ref_value_cache.float()).abs().max().item()
        qkv_diff = (
            (actual_qkv.float() - expected_qkv.float()).abs().max().item()
            if actual_qkv is not None
            else None
        )
        passed = (
            output_diff <= 2.0e-3
            and key_diff <= 2.0e-3
            and value_diff <= 2.0e-3
            and torch.equal(attention_mask, expected_mask)
            and (qkv_diff is None or qkv_diff <= 2.0e-3)
        )
        all_passed = all_passed and bool(passed)
        checks.append(
            {
                "position": position,
                "output_max_abs": float(output_diff),
                "key_cache_max_abs": float(key_diff),
                "value_cache_max_abs": float(value_diff),
                "qkv_max_abs": None if qkv_diff is None else float(qkv_diff),
                "attention_mask_exact": bool(torch.equal(attention_mask, expected_mask)),
                "passed": bool(passed),
            }
        )

    result = {
        "kind": f"paddle_decode_packed_qkv_rope_gqa_mixed24_{args.mode}_probe",
        "operator": {
            "pytorch": "paddleocr_vl::decode_packed_qkv_rope_gqa_mixed24",
            "ge": "PaddleDecodePackedQkvRopeGqaMixed24",
        },
        "contract": {
            "mode": args.mode,
            "qkv_shape": [1, 1, 2560],
            "cache_shape": [1, 2, 1024, 128],
            "factor_lut_shape": [2, 1024, 128],
            "rope_delta": 7,
            "positions": [128, 129],
            "launch_aiv_count": 24,
            "attention_worker_count": 16,
            "super_kernel_options": args.super_kernel_options,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_required_checks_passed": bool(all_passed),
            "steps": checks,
        },
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_passed:
        raise RuntimeError("packed QKV/RoPE/GQA parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
