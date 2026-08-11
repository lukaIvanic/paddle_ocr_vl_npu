#!/usr/bin/env python3
"""Benchmark direct CPU cross-K/V admission policies without loading UniRec."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


LAYERS = 6
HEADS = 6
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-cache-length", type=int, default=1024)
    parser.add_argument("--cross-cache-length", type=int, default=512)
    parser.add_argument("--warmup-admissions", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def load_lengths(path: Path, cross_cache_length: int) -> list[int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows.sort(key=lambda row: int(row["admission_index"]))
    lengths = [int(row["text_prefill_real_source_tokens"]) for row in rows]
    if not lengths:
        raise ValueError("trace contains no admissions")
    if max(lengths) > cross_cache_length:
        raise ValueError(
            f"trace source length {max(lengths)} exceeds {cross_cache_length}"
        )
    return lengths


def allocate_arena(
    *,
    device: str,
    batch_size: int,
    self_cache_length: int,
    cross_cache_length: int,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    torch.Tensor,
]:
    self_shape = (batch_size, HEADS, self_cache_length, HEAD_DIM)
    cross_shape = (batch_size, HEADS, cross_cache_length, HEAD_DIM)
    negative_inf = torch.finfo(torch.float32).min
    with torch.inference_mode():
        self_keys = tuple(
            torch.zeros(self_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        self_values = tuple(
            torch.zeros(self_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        cross_keys = tuple(
            torch.zeros(cross_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        cross_values = tuple(
            torch.zeros(cross_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        cross_mask = torch.full(
            (batch_size, 1, 1, cross_cache_length),
            negative_inf,
            dtype=torch.float32,
            device=device,
        )
    return self_keys, self_values, cross_keys, cross_values, cross_mask


def run_policy(
    policy: str,
    *,
    lengths: list[int],
    host_by_length: dict[int, np.ndarray],
    arena: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ],
    mask_templates: dict[int, torch.Tensor],
    device: str,
    batch_size: int,
) -> float:
    self_keys, self_values, cross_keys, cross_values, cross_mask = arena
    negative_inf = torch.finfo(cross_mask.dtype).min
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for admission, source_len in enumerate(lengths):
            slot = admission % batch_size
            packed = host_by_length[source_len]
            if policy == "full_reset":
                cross_mask[slot : slot + 1].fill_(negative_inf)
                cross_mask[slot : slot + 1, :, :, :source_len].zero_()
            elif policy == "no_self_reset":
                cross_mask[slot : slot + 1].fill_(negative_inf)
                cross_mask[slot : slot + 1, :, :, :source_len].zero_()
            elif policy == "masked_reuse":
                cross_mask[slot : slot + 1].copy_(mask_templates[source_len])
            else:
                raise ValueError(f"unknown policy: {policy}")
            for layer in range(LAYERS):
                if policy == "full_reset":
                    self_keys[layer][slot : slot + 1].zero_()
                    self_values[layer][slot : slot + 1].zero_()
                if policy in {"full_reset", "no_self_reset"}:
                    cross_keys[layer][
                        slot : slot + 1, :, source_len:, :
                    ].zero_()
                    cross_values[layer][
                        slot : slot + 1, :, source_len:, :
                    ].zero_()
                cross_keys[layer][slot : slot + 1, :, :source_len, :].copy_(
                    torch.from_numpy(packed[layer])
                )
                cross_values[layer][slot : slot + 1, :, :source_len, :].copy_(
                    torch.from_numpy(packed[LAYERS + layer])
                )
    synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.warmup_admissions < 0:
        raise ValueError("rounds must be positive and warmup-admissions nonnegative")
    lengths = load_lengths(args.trace, args.cross_cache_length)
    unique_lengths = sorted(set(lengths))
    host_by_length = {
        length: np.zeros(
            (2 * LAYERS, 1, HEADS, length, HEAD_DIM),
            dtype=np.float16,
        )
        for length in unique_lengths
    }
    negative_inf = torch.finfo(torch.float32).min
    mask_templates = {}
    with torch.inference_mode():
        for length in unique_lengths:
            template = torch.full(
                (1, 1, 1, args.cross_cache_length),
                negative_inf,
                dtype=torch.float32,
                device=args.device,
            )
            template[..., :length].zero_()
            mask_templates[length] = template
    arena = allocate_arena(
        device=args.device,
        batch_size=args.batch_size,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
    )
    warmup_lengths = lengths[: args.warmup_admissions]
    results = {}
    for policy in ("full_reset", "no_self_reset", "masked_reuse"):
        if warmup_lengths:
            run_policy(
                policy,
                lengths=warmup_lengths,
                host_by_length=host_by_length,
                arena=arena,
                mask_templates=mask_templates,
                device=args.device,
                batch_size=args.batch_size,
            )
        samples = [
            run_policy(
                policy,
                lengths=lengths,
                host_by_length=host_by_length,
                arena=arena,
                mask_templates=mask_templates,
                device=args.device,
                batch_size=args.batch_size,
            )
            for _ in range(args.rounds)
        ]
        results[policy] = {
            "samples_s": samples,
            "median_s": statistics.median(samples),
            "admissions_per_s": len(lengths) / statistics.median(samples),
        }
    baseline = results["full_reset"]["median_s"]
    for value in results.values():
        value["speedup_vs_full_reset"] = baseline / value["median_s"]
    summary = {
        "device": args.device,
        "admissions": len(lengths),
        "batch_size": args.batch_size,
        "self_cache_length": args.self_cache_length,
        "cross_cache_length": args.cross_cache_length,
        "source_length_histogram": dict(sorted(Counter(lengths).items())),
        "host_payload_bytes": sum(host_by_length[length].nbytes for length in lengths),
        "results": results,
    }
    print("UNIREC_ARENA_ADMISSION_BENCH " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
