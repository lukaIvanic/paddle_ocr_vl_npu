#!/usr/bin/env python3
"""Validate functional mixed24 K/V preparation fused into mixed24 GQA."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch_npu

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_gqa_attention_mixed24 import (
    decode_gqa_attention_mixed24,
    register_decode_gqa_attention_mixed24_converter,
)
from paddleocr_vl.model.decode_kv_prepare_functional_mixed24 import (
    decode_kv_prepare_functional_mixed24,
    register_decode_kv_prepare_functional_mixed24_converter,
)


class FunctionalKvAttention(torch.nn.Module):
    def __init__(
        self,
        options: str,
        strict_scope: bool,
        include_gqa: bool,
        attention_impl: str,
    ) -> None:
        super().__init__()
        self.options = options
        self.include_gqa = include_gqa
        self.attention_impl = attention_impl
        self.scope = None
        if strict_scope:
            self.scope = __import__(
                "torchair.scope", fromlist=["super_kernel"]
            ).super_kernel

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        def run() -> tuple[torch.Tensor, ...]:
            ordered_query, mask, key_out, value_out = (
                decode_kv_prepare_functional_mixed24(
                    query,
                    key_cache,
                    value_cache,
                    cache_position,
                    key_state,
                    value_state,
                )
            )
            if not self.include_gqa:
                return ordered_query, mask, key_out, value_out
            if self.attention_impl == "stock":
                attention = torch_npu.npu_incre_flash_attention(
                    ordered_query,
                    key_out,
                    value_out,
                    atten_mask=mask,
                    actual_seq_lengths=None,
                    num_heads=16,
                    num_key_value_heads=2,
                    input_layout="BNSD",
                    scale_value=1.0 / math.sqrt(128.0),
                    inner_precise=1,
                )
            else:
                attention = decode_gqa_attention_mixed24(
                    ordered_query,
                    key_out,
                    value_out,
                    mask,
                    num_heads=16,
                    num_key_value_heads=2,
                    scale_value=1.0 / math.sqrt(128.0),
                    inner_precise=1,
                    vector_core_count=16,
                )
            return ordered_query, mask, key_out, value_out, attention
        if self.scope is None:
            return run()
        with self.scope("paddle_functional_kv_attention", self.options):
            return run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-super-kernel", action="store_true")
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument(
        "--attention-impl",
        choices=("custom", "stock"),
        default="custom",
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
    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_kv_prepare_functional_mixed24_converter()
    if not args.functional_only and args.attention_impl == "custom":
        register_decode_gqa_attention_mixed24_converter()

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = CompilerConfig()
    dump_dir = args.cache_dir / "graph_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    config.debug.graph_dump.type = "pbtxt"
    config.debug.graph_dump.path = str(dump_dir)
    step = torchair.inference.cache_compile(
        FunctionalKvAttention(
            args.super_kernel_options,
            strict_scope=not args.no_super_kernel,
            include_gqa=not args.functional_only,
            attention_impl=args.attention_impl,
        ).forward,
        config=config,
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    query = torch.randn(
        (1, 16, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    key_cache = torch.zeros(
        (1, 2, 1024, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    key_states = [
        torch.randn(
            (1, 2, 1, 128), generator=generator, dtype=torch.float16
        ).to("npu:0")
        for _ in range(2)
    ]
    value_states = [torch.randn_like(state) for state in key_states]
    positions = [128, 129]
    timings = []
    checks = []
    all_exact = True

    for position, key_state, value_state in zip(
        positions, key_states, value_states, strict=True
    ):
        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device="npu:0"
        )
        started = time.perf_counter()
        outputs = step(
            query,
            key_cache,
            value_cache,
            position_tensor,
            key_state,
            value_state,
        )
        if args.functional_only:
            ordered, mask, key_cache, value_cache = outputs
            attention = None
        else:
            ordered, mask, key_cache, value_cache, attention = outputs
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)

        reference = None
        if attention is not None:
            reference = torch_npu.npu_incre_flash_attention(
                ordered,
                key_cache,
                value_cache,
                atten_mask=mask,
                actual_seq_lengths=None,
                num_heads=16,
                num_key_value_heads=2,
                input_layout="BNSD",
                scale_value=1.0 / math.sqrt(128.0),
                inner_precise=1,
            )
            torch.npu.synchronize()
        query_exact = bool(torch.equal(ordered.cpu(), query.cpu()))
        mask_expected = (
            torch.arange(1024, dtype=torch.int64) > position
        ).view(1, 1, 1, 1024)
        mask_exact = bool(torch.equal(mask.cpu(), mask_expected))
        key_actual = key_cache[:, :, position : position + 1, :].cpu()
        value_actual = value_cache[:, :, position : position + 1, :].cpu()
        key_exact = bool(torch.equal(key_actual, key_state.cpu()))
        value_exact = bool(torch.equal(value_actual, value_state.cpu()))
        attention_max_abs = None
        attention_close = None
        attention_output_max_abs = None
        if attention is not None and reference is not None:
            attention_max_abs = float(
                (attention.float() - reference.float()).abs().max().cpu()
            )
            attention_close = bool(
                torch.allclose(attention, reference, atol=3e-4, rtol=3e-4)
            )
            attention_output_max_abs = float(
                attention.float().abs().max().cpu()
            )
        step_exact = (
            query_exact
            and mask_exact
            and key_exact
            and value_exact
            and (attention_close is not False)
        )
        all_exact = all_exact and step_exact
        checks.append(
            {
                "position": position,
                "query_exact": query_exact,
                "mask_exact": mask_exact,
                "key_exact": key_exact,
                "value_exact": value_exact,
                "attention_vs_stock_close": attention_close,
                "attention_vs_stock_max_abs": attention_max_abs,
                "attention_output_max_abs": attention_output_max_abs,
            }
        )

    result = {
        "kind": "paddle_decode_functional_kv_attention_superkernel_probe",
        "operator": {
            "pytorch": (
                "paddleocr_vl::decode_kv_prepare_functional_mixed24"
            ),
            "ge": "PaddleDecodeKvPrepareFunctionalMixed24",
            "kernel": "paddle_decode_kv_prepare_functional_mixed24",
            "attention_ge": "PaddleDecodeGqaAttentionMixed24",
        },
        "contract": {
            "cache_semantics": "functional_explicit_outputs",
            "strict_scope": not args.no_super_kernel,
            "include_gqa": not args.functional_only,
            "attention_impl": args.attention_impl,
            "cache_shape": list(key_cache.shape),
            "positions": positions,
            "super_kernel_options": args.super_kernel_options,
            "block_dim": 24,
            "core_type": "MIX_AIC_1_1",
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {"all_exact": all_exact, "steps": checks},
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_exact:
        raise RuntimeError("functional KV-attention SuperKernel failed parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
