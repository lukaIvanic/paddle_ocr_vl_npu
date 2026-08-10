#!/usr/bin/env python3
"""Validate a fused Paddle B1 cache-update and GQA op through TorchAir."""

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
from paddleocr_vl.model.decode_gqa_increfa_mixed import (
    decode_gqa_incre_flash_attention_mixed,
    register_decode_gqa_increfa_mixed_converter,
)


class DecodeGqaIncrefaAiv(torch.nn.Module):
    def __init__(
        self,
        geometry: str,
        strict_scope: bool,
        super_kernel_options: str,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.super_kernel_options = super_kernel_options
        self.scope = None
        if strict_scope:
            self.scope = __import__(
                "torchair.scope", fromlist=["super_kernel"]
            ).super_kernel

    def _forward_impl(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
    ) -> torch.Tensor:
        operator = (
            decode_gqa_incre_flash_attention_mixed
            if self.geometry == "mixed"
            else decode_gqa_incre_flash_attention_aiv
        )
        return operator(
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
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
    ) -> torch.Tensor:
        args = (
            query,
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            key_state,
            value_state,
        )
        if self.scope is None:
            return self._forward_impl(*args)
        with self.scope(
            f"paddle_decode_gqa_increfa_{self.geometry}_probe",
            self.super_kernel_options,
        ):
            return self._forward_impl(*args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        choices=("aiv", "mixed"),
        default="aiv",
        help="Compile the zero-cube AIV op or its 1:1 mixed-task control.",
    )
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
    if args.geometry == "mixed":
        register_decode_gqa_increfa_mixed_converter()
    else:
        register_decode_gqa_increfa_aiv_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    query = torch.randn(
        (1, 16, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    key_cache = torch.zeros(
        (1, 2, 1024, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    attention_mask = torch.zeros(
        (1, 1, 1, 1024), dtype=torch.bool, device="npu:0"
    )
    ref_key_cache = torch.zeros_like(key_cache)
    ref_value_cache = torch.zeros_like(value_cache)
    key_states = [
        torch.randn(
            (1, 2, 1, 128), generator=generator, dtype=torch.float16
        ).to("npu:0")
        for _ in range(2)
    ]
    value_states = [
        torch.randn(
            (1, 2, 1, 128), generator=generator, dtype=torch.float16
        ).to("npu:0")
        for _ in range(2)
    ]
    positions = [128, 129]

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    compiler_config = CompilerConfig()
    graph_dump_dir = args.cache_dir / "graph_dump"
    graph_dump_dir.mkdir(parents=True, exist_ok=True)
    compiler_config.debug.graph_dump.type = "pbtxt"
    compiler_config.debug.graph_dump.path = str(graph_dump_dir)
    step = torchair.inference.cache_compile(
        DecodeGqaIncrefaAiv(
            args.geometry,
            args.strict_scope,
            args.super_kernel_options,
        ).forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    checks: list[dict[str, object]] = []
    timings: list[float] = []
    all_close = True
    for position, key_state, value_state in zip(
        positions, key_states, value_states, strict=True
    ):
        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device="npu:0"
        )
        started = time.perf_counter()
        attention = step(
            query,
            key_cache,
            value_cache,
            attention_mask,
            position_tensor,
            key_state,
            value_state,
        )
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)

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

        attention_cpu = attention.cpu()
        reference_cpu = reference.cpu()
        attention_max_abs = float(
            (attention_cpu.float() - reference_cpu.float()).abs().max()
        )
        attention_close = bool(
            torch.allclose(
                attention_cpu, reference_cpu, atol=3e-4, rtol=3e-4
            )
        )
        key_exact = bool(torch.equal(key_cache.cpu(), ref_key_cache.cpu()))
        value_exact = bool(
            torch.equal(value_cache.cpu(), ref_value_cache.cpu())
        )
        mask_exact = bool(
            torch.equal(attention_mask.cpu(), expected_mask.cpu())
        )
        all_close = (
            all_close and attention_close and key_exact and value_exact and mask_exact
        )
        checks.append(
            {
                "position": position,
                "attention_vs_stock_close": attention_close,
                "attention_vs_stock_max_abs": attention_max_abs,
                "key_cache_exact": key_exact,
                "value_cache_exact": value_exact,
                "attention_mask_exact": mask_exact,
                "masked_count": int(torch.count_nonzero(attention_mask).cpu()),
            }
        )

    result = {
        "kind": f"paddle_decode_gqa_increfa_{args.geometry}_torchair_probe",
        "operator": {
            "pytorch": (
                "paddleocr_vl::decode_gqa_incre_flash_attention_mixed"
                if args.geometry == "mixed"
                else "paddleocr_vl::decode_gqa_incre_flash_attention_aiv"
            ),
            "ge": (
                "PaddleDecodeGqaIncreFlashAttentionMixed"
                if args.geometry == "mixed"
                else "PaddleDecodeGqaIncreFlashAttentionAiv"
            ),
            "kernel": (
                "paddle_decode_gqa_incre_flash_attention_mixed"
                if args.geometry == "mixed"
                else "paddle_decode_gqa_incre_flash_attention_aiv"
            ),
        },
        "contract": {
            "query_shape": list(query.shape),
            "cache_shape": list(key_cache.shape),
            "state_shape": list(key_states[0].shape),
            "mask_shape": list(attention_mask.shape),
            "positions": positions,
            "strict_scope": args.strict_scope,
            "super_kernel_options": args.super_kernel_options,
            "vector_core_count": 16,
            "core_type": (
                "MIX_AIC_1_1_WITH_NOOP_AIC"
                if args.geometry == "mixed"
                else "MIX_AIV_ZERO_CUBE"
            ),
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {"all_required_checks_passed": all_close, "steps": checks},
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_close:
        raise RuntimeError("fused decode GQA failed parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
