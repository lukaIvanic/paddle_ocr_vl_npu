#!/usr/bin/env python3
"""Compare PromptFA mask/sparse-mode paths with identical attention semantics.

This is intentionally an operator-level probe. It answers which public
PromptFlashAttention invocation is fastest before spending time compiling the
complete 27-layer vision stage.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_npu


INT_MAX = 2_147_483_647


@dataclass(frozen=True)
class Measurement:
    case: str
    lane: str
    batch: int
    sequence_length: int
    physical_tokens: int
    median_device_us: float
    median_wall_us: float
    physical_tokens_per_second: float
    device_us_samples: list[float]
    wall_us_samples: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronize() -> None:
    torch_npu.npu.synchronize()


def make_qkv(
    *,
    batch: int,
    heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch, heads, sequence_length, head_dim)
    return tuple(
        torch.randn(shape, dtype=torch.float16, device=device)
        for _ in range(3)
    )


def all_false_mask(
    *, batch: int, sequence_length: int, device: torch.device
) -> torch.Tensor:
    return torch.zeros(
        (batch, 1, sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    )


def block_mask(
    *, segment_lengths: list[int], device: torch.device
) -> torch.Tensor:
    segment_ids = torch.repeat_interleave(
        torch.arange(len(segment_lengths), device=device),
        torch.tensor(segment_lengths, device=device),
    )
    return (
        segment_ids.view(1, 1, -1, 1)
        != segment_ids.view(1, 1, 1, -1)
    )


def promptfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    atten_mask: torch.Tensor | None,
    sparse_mode: int,
    actual_seq_lengths: list[int] | None = None,
) -> torch.Tensor:
    return torch_npu.npu_prompt_flash_attention(
        q,
        k,
        v,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
        num_heads=heads,
        scale_value=1.0 / math.sqrt(q.shape[-1]),
        pre_tokens=INT_MAX,
        next_tokens=INT_MAX,
        input_layout="BNSD",
        sparse_mode=sparse_mode,
    )


def measure(
    *,
    case_name: str,
    lane_name: str,
    batch: int,
    sequence_length: int,
    call: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[Measurement, torch.Tensor]:
    print(f"[run] {case_name}/{lane_name}", flush=True)
    output = call()
    for _ in range(warmup - 1):
        output = call()
    synchronize()

    device_samples: list[float] = []
    wall_samples: list[float] = []
    for repeat in range(repeats):
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_event.record()
        for _ in range(iterations):
            output = call()
        end_event.record()
        synchronize()
        wall_us = (time.perf_counter() - start_wall) * 1_000_000 / iterations
        device_us = start_event.elapsed_time(end_event) * 1_000 / iterations
        device_samples.append(device_us)
        wall_samples.append(wall_us)
        print(
            f"  repeat {repeat + 1}/{repeats}: "
            f"device={device_us:.3f} us wall={wall_us:.3f} us",
            flush=True,
        )

    median_device_us = statistics.median(device_samples)
    physical_tokens = batch * sequence_length
    measurement = Measurement(
        case=case_name,
        lane=lane_name,
        batch=batch,
        sequence_length=sequence_length,
        physical_tokens=physical_tokens,
        median_device_us=median_device_us,
        median_wall_us=statistics.median(wall_samples),
        physical_tokens_per_second=physical_tokens
        / (median_device_us / 1_000_000),
        device_us_samples=device_samples,
        wall_us_samples=wall_samples,
    )
    return measurement, output.detach()


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = (candidate.float() - reference.float()).abs()
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 1:
        raise ValueError("warmup, iterations, and repeats must all be positive")

    torch.manual_seed(args.seed)
    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)

    measurements: list[Measurement] = []
    comparisons: dict[str, dict[str, float | bool]] = {}

    # Dense B4xS512: no mask and an all-false mask are mathematically equal.
    dense_batch, dense_s = 4, 512
    dense_qkv = make_qkv(
        batch=dense_batch,
        heads=args.heads,
        sequence_length=dense_s,
        head_dim=args.head_dim,
        device=device,
    )
    dense_mask = all_false_mask(
        batch=dense_batch, sequence_length=dense_s, device=device
    )
    dense_outputs: dict[str, torch.Tensor] = {}
    dense_lanes = {
        "sparse0_no_mask": lambda: promptfa(
            *dense_qkv,
            heads=args.heads,
            atten_mask=None,
            sparse_mode=0,
        ),
        "sparse0_all_false_mask": lambda: promptfa(
            *dense_qkv,
            heads=args.heads,
            atten_mask=dense_mask,
            sparse_mode=0,
        ),
        "sparse1_all_false_mask": lambda: promptfa(
            *dense_qkv,
            heads=args.heads,
            atten_mask=dense_mask,
            sparse_mode=1,
        ),
    }
    for lane_name, call in dense_lanes.items():
        measurement, output = measure(
            case_name="dense_b4_s512",
            lane_name=lane_name,
            batch=dense_batch,
            sequence_length=dense_s,
            call=call,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        measurements.append(measurement)
        dense_outputs[lane_name] = output
    for lane_name in dense_outputs:
        comparisons[f"dense/{lane_name}_vs_sparse0_no_mask"] = compare(
            dense_outputs["sparse0_no_mask"], dense_outputs[lane_name]
        )

    # Packed B1: sparse mode 0 and 1 receive the identical complete block mask.
    packed_cases = {
        "packed_b1_s512": [64, 96, 128, 224],
        "packed_b1_s2048": [64] * 32,
    }
    for case_name, segment_lengths in packed_cases.items():
        packed_s = sum(segment_lengths)
        packed_qkv = make_qkv(
            batch=1,
            heads=args.heads,
            sequence_length=packed_s,
            head_dim=args.head_dim,
            device=device,
        )
        packed_mask = block_mask(
            segment_lengths=segment_lengths, device=device
        )
        packed_outputs: dict[str, torch.Tensor] = {}
        for sparse_mode in (0, 1):
            lane_name = f"sparse{sparse_mode}_block_mask"
            call = lambda sparse_mode=sparse_mode: promptfa(
                *packed_qkv,
                heads=args.heads,
                atten_mask=packed_mask,
                sparse_mode=sparse_mode,
            )
            measurement, output = measure(
                case_name=case_name,
                lane_name=lane_name,
                batch=1,
                sequence_length=packed_s,
                call=call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            measurements.append(measurement)
            packed_outputs[lane_name] = output
        comparisons[f"{case_name}/sparse1_vs_sparse0"] = compare(
            packed_outputs["sparse0_block_mask"],
            packed_outputs["sparse1_block_mask"],
        )

    # Ragged B4: CANN 9 documents per-batch actual sequence lengths for 310P.
    # This lane checks whether it actually avoids work relative to padded B4.
    ragged_lengths = [64, 100, 256, 512]
    ragged_qkv = make_qkv(
        batch=4,
        heads=args.heads,
        sequence_length=512,
        head_dim=args.head_dim,
        device=device,
    )
    ragged_lanes = {
        "sparse0_padded": lambda: promptfa(
            *ragged_qkv,
            heads=args.heads,
            atten_mask=None,
            sparse_mode=0,
        ),
        "sparse0_actual_lengths": lambda: promptfa(
            *ragged_qkv,
            heads=args.heads,
            atten_mask=None,
            sparse_mode=0,
            actual_seq_lengths=ragged_lengths,
        ),
    }
    ragged_outputs: dict[str, torch.Tensor] = {}
    for lane_name, call in ragged_lanes.items():
        measurement, output = measure(
            case_name="ragged_b4_s512",
            lane_name=lane_name,
            batch=4,
            sequence_length=512,
            call=call,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        measurements.append(measurement)
        ragged_outputs[lane_name] = output
    valid_comparisons = []
    for batch_index, valid_length in enumerate(ragged_lengths):
        valid_comparisons.append(
            compare(
                ragged_outputs["sparse0_padded"][
                    batch_index, :, :valid_length
                ],
                ragged_outputs["sparse0_actual_lengths"][
                    batch_index, :, :valid_length
                ],
            )
        )
    comparisons["ragged/actual_lengths_vs_padded_valid_prefixes"] = {
        "exact": all(bool(item["exact"]) for item in valid_comparisons),
        "max_abs": max(float(item["max_abs"]) for item in valid_comparisons),
        "mean_abs": statistics.mean(
            float(item["mean_abs"]) for item in valid_comparisons
        ),
    }

    payload = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": args.device,
            "device_name": torch_npu.npu.get_device_name(),
            "heads": args.heads,
            "head_dim": args.head_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "measurements": [asdict(item) for item in measurements],
        "comparisons": comparisons,
        "ragged_lengths": ragged_lengths,
        "packed_segments": packed_cases,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"[saved] {args.output}", flush=True)
    print(encoded)


if __name__ == "__main__":
    main()
