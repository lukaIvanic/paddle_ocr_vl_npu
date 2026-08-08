#!/usr/bin/env python3
"""Launch the production-shape IncreFA operator for low-level profiling."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import torch
import torch_npu


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-length", type=int, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if not 0 <= args.position < args.cache_length:
        parser.error("--position must be inside --cache-length")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.num_key_value_heads <= 0 or 16 % args.num_key_value_heads != 0:
        parser.error("--num-key-value-heads must be a positive divisor of 16")
    return args


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")

    device = torch.device(f"npu:{args.device_index}")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    query = torch.randn(
        (1, 16, 1, 128), device=device, dtype=dtype
    ).contiguous()
    key = torch.randn(
        (1, args.num_key_value_heads, args.cache_length, 128),
        device=device,
        dtype=dtype,
    ).contiguous()
    value = torch.randn_like(key).contiguous()
    positions = torch.arange(
        args.cache_length, device=device, dtype=torch.int64
    )
    mask = (positions > args.position).view(
        1, 1, 1, args.cache_length
    ).contiguous()

    def step() -> torch.Tensor:
        return torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            atten_mask=mask,
            actual_seq_lengths=None,
            num_heads=16,
            num_key_value_heads=args.num_key_value_heads,
            input_layout="BNSD",
            scale_value=1.0 / math.sqrt(128.0),
            inner_precise=1,
        )

    output = None
    for _ in range(args.warmup):
        output = step()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(args.repeats):
        output = step()
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "kind": "direct_increfa_operator_profile",
        "configuration": {
            "batch_size": 1,
            "query_heads": 16,
            "key_value_heads": args.num_key_value_heads,
            "head_dim": 128,
            "cache_length": args.cache_length,
            "position": args.position,
            "input_layout": "BNSD",
            "dtype": "fp16",
            "inner_precise": 1,
            "device_index": args.device_index,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "host_bounded_timing": {
            "elapsed_s": elapsed_s,
            "mean_us": elapsed_s * 1e6 / args.repeats,
            "is_throughput_authority": False,
        },
        "output": {
            "shape": list(output.shape) if output is not None else None,
            "dtype": str(output.dtype) if output is not None else None,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
