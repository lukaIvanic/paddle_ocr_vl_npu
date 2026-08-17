#!/usr/bin/env python3
"""Screen static TorchAir attention contracts at the Qwen3-MoE B1/KV4096 shape."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--compile-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


class AttentionVariant(nn.Module):
    def __init__(self, variant: str, cache_length: int):
        super().__init__()
        self.variant = variant
        self.cache_length = int(cache_length)
        self.scale = 1.0 / math.sqrt(128.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        common = {
            "num_heads": 32,
            "num_key_value_heads": 4,
            "input_layout": "BNSD",
        }
        if self.variant == "increfa_bnsd_mask":
            return torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                atten_mask=mask,
                scale_value=self.scale,
                inner_precise=1,
                **common,
            )
        if self.variant == "increfa_bnsd_nomask":
            return torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                atten_mask=None,
                scale_value=self.scale,
                inner_precise=1,
                **common,
            )
        if self.variant == "increfa_bnsd_actual_full":
            return torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                atten_mask=None,
                actual_seq_lengths=[self.cache_length],
                scale_value=self.scale,
                inner_precise=1,
                **common,
            )
        if self.variant == "increfa_bsnd_mask":
            return torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                atten_mask=mask,
                num_heads=32,
                num_key_value_heads=4,
                input_layout="BSND",
                scale_value=self.scale,
                inner_precise=1,
            )
        if self.variant == "fia_v1_bnsd_mask":
            return torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                atten_mask=mask,
                num_heads=32,
                num_key_value_heads=4,
                input_layout="BNSD",
                scale=self.scale,
                sparse_mode=0,
                inner_precise=1,
            )[0]
        if self.variant == "fia_v2_bnsd_mask":
            return torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                atten_mask=mask,
                num_query_heads=32,
                num_key_value_heads=4,
                input_layout="BNSD",
                softmax_scale=self.scale,
                sparse_mode=0,
                inner_precise=1,
            )[0]
        raise ValueError(f"Unsupported variant: {self.variant}")


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.cache_length < 1 or args.warmup_steps < 0 or args.steps < 1:
        raise ValueError("Invalid cache length or iteration count")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(17)
    query_bnsd = torch.randn(
        1, 32, 1, 128, dtype=torch.bfloat16, device=device
    ).contiguous()
    key_bnsd = torch.randn(
        1, 4, args.cache_length, 128, dtype=torch.bfloat16, device=device
    ).contiguous()
    value_bnsd = torch.randn_like(key_bnsd).contiguous()
    query_bsnd = query_bnsd.transpose(1, 2).contiguous()
    key_bsnd = key_bnsd.transpose(1, 2).contiguous()
    value_bsnd = value_bnsd.transpose(1, 2).contiguous()
    mask = torch.zeros(
        1, args.cache_length, dtype=torch.bool, device=device
    ).contiguous()
    reference = torch_npu.npu_incre_flash_attention(
        query_bnsd,
        key_bnsd,
        value_bnsd,
        atten_mask=mask,
        num_heads=32,
        num_key_value_heads=4,
        input_layout="BNSD",
        scale_value=1.0 / math.sqrt(128.0),
        inner_precise=1,
    )
    synchronize(device)

    variants = (
        "increfa_bnsd_mask",
        "increfa_bnsd_nomask",
        "increfa_bnsd_actual_full",
        "increfa_bsnd_mask",
        "fia_v1_bnsd_mask",
        "fia_v2_bnsd_mask",
    )
    results: dict[str, object] = {}
    cache_compile = import_cache_compile()
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    for variant in variants:
        module = AttentionVariant(variant, args.cache_length).to(device).eval()
        cache_dir = args.compile_cache_dir.resolve() / variant
        cache_dir.mkdir(parents=True, exist_ok=True)
        compiled = cache_compile(
            module.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        if variant == "increfa_bsnd_mask":
            inputs = (query_bsnd, key_bsnd, value_bsnd, mask)
        else:
            inputs = (query_bnsd, key_bnsd, value_bnsd, mask)
        first_started = time.perf_counter()
        output = compiled(*inputs)
        synchronize(device)
        first_call_s = time.perf_counter() - first_started
        for _ in range(args.warmup_steps):
            output = compiled(*inputs)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            output = compiled(*inputs)
        synchronize(device)
        elapsed_s = time.perf_counter() - started
        if variant == "increfa_bsnd_mask":
            output_bnsd = output.transpose(1, 2)
        else:
            output_bnsd = output
        difference = (output_bnsd.float() - reference.float()).abs()
        result = {
            "first_call_s": first_call_s,
            "elapsed_s": elapsed_s,
            "steps": args.steps,
            "mean_us": elapsed_s * 1e6 / args.steps,
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        }
        results[variant] = result
        print(f"INCREFA_VARIANT {variant} {json.dumps(result, sort_keys=True)}", flush=True)

    payload = {
        "shape": {
            "batch": 1,
            "query_heads": 32,
            "key_value_heads": 4,
            "head_dim": 128,
            "cache_length": args.cache_length,
            "dtype": "bfloat16",
        },
        "contract": "all KV slots valid; mask is all false",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("INCREFA_VARIANT_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
