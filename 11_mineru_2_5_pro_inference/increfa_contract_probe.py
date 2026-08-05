#!/usr/bin/env python3
"""Compare batched IncreFA length and mask contracts against manual GQA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--valid-lengths",
        help="Comma-separated per-row valid KV lengths. Defaults to a varied range.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(torch, device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    import torch_npu

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    batch = int(args.batch_size)
    cache_length = int(args.cache_length)
    num_heads = int(args.num_heads)
    num_kv_heads = int(args.num_kv_heads)
    head_dim = int(args.head_dim)
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if args.valid_lengths:
        valid_lengths = [int(value) for value in args.valid_lengths.split(",")]
    else:
        valid_lengths = [
            128 + (cache_length - 129) * row // max(1, batch - 1)
            for row in range(batch)
        ]
    if len(valid_lengths) != batch:
        raise ValueError("valid-length count must equal batch-size")
    if not all(1 <= length <= cache_length for length in valid_lengths):
        raise ValueError("valid lengths must be within the physical cache")

    torch.manual_seed(int(args.seed))
    query = torch.randn(
        batch, num_heads, 1, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(
        batch, num_kv_heads, cache_length, head_dim,
        device=device, dtype=torch.float16,
    )
    value = torch.randn_like(key)
    positions = torch.arange(cache_length, device=device).view(1, 1, 1, -1)
    lengths_tensor = torch.tensor(valid_lengths, device=device).view(batch, 1, 1, 1)
    mask = (positions >= lengths_tensor).contiguous()
    expanded_mask = mask.expand(batch, num_heads, 1, cache_length).contiguous()
    scale = 1.0 / math.sqrt(head_dim)

    repeated_key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    repeated_value = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * scale
    scores = scores.masked_fill(expanded_mask, torch.finfo(scores.dtype).min)
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    reference = torch.matmul(probabilities, repeated_value)
    synchronize(torch, device)

    variants = {
        "mask_b1": {"atten_mask": mask},
        "mask_bn": {"atten_mask": expanded_mask},
        "mask_b1_actual_full": {
            "atten_mask": mask,
            "actual_seq_lengths": [cache_length] * batch,
        },
        "mask_bn_actual_full": {
            "atten_mask": expanded_mask,
            "actual_seq_lengths": [cache_length] * batch,
        },
        "actual_per_row": {"actual_seq_lengths": valid_lengths},
        "mask_b1_inner_precise_0": {"atten_mask": mask, "inner_precise": 0},
        "mask_bn_inner_precise_0": {
            "atten_mask": expanded_mask,
            "inner_precise": 0,
        },
    }
    results = {}
    for name, options in variants.items():
        started = time.perf_counter()
        output = torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BNSD",
            scale_value=scale,
            **options,
        )
        synchronize(torch, device)
        delta = (output.float() - reference.float()).abs()
        results[name] = {
            "wall_s": time.perf_counter() - started,
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "cosine": float(
                F.cosine_similarity(
                    output.float().flatten(), reference.float().flatten(), dim=0
                ).item()
            ),
        }
        print(name, json.dumps(results[name], sort_keys=True), flush=True)

    payload = {
        "shape": {
            "batch_size": batch,
            "cache_length": cache_length,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "valid_lengths": valid_lengths,
        "mask_shape": list(mask.shape),
        "expanded_mask_shape": list(expanded_mask.shape),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
