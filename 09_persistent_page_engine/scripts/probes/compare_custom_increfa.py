#!/usr/bin/env python3
"""Run and compare stock versus custom B1 FP16/MHA IncreFA.

The ``run`` command must execute in a fresh process.  Select the custom
operator by sourcing its installed ``set_env.bash`` before launching Python;
leave ``ASCEND_CUSTOM_OPP_PATH`` unset for the stock control.  The ``compare``
command is CPU-only and checks the saved full output tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


QUERY_HEADS = 16
KV_HEADS = 16
HEAD_DIM = 128
SCALE_VALUE = 1.0 / math.sqrt(HEAD_DIM)
DEFAULT_KV_LENGTHS = (128, 512, 2048)
CUSTOM_KERNEL_BASENAME = (
    "IncreFlashAttention_7b761bdde53e2d667f3cdc458400fc8e"
)


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("KV lengths must be positive integers")
    return lengths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument(
        "--kv-lengths",
        type=parse_lengths,
        default=DEFAULT_KV_LENGTHS,
    )
    run_parser.add_argument("--warmup", type=int, default=20)
    run_parser.add_argument("--blocks", type=int, default=7)
    run_parser.add_argument("--repeats-per-block", type=int, default=200)
    run_parser.add_argument("--seed", type=int, default=20260809)
    run_parser.add_argument("--device-index", type=int, default=0)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--stock", type=Path, required=True)
    compare_parser.add_argument("--custom", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, default=0.0)
    compare_parser.add_argument("--rtol", type=float, default=0.0)

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.warmup < 0:
            parser.error("--warmup must be non-negative")
        if args.blocks <= 0 or args.repeats_per_block <= 0:
            parser.error("timing blocks and repeats must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def custom_package_evidence() -> dict[str, Any]:
    custom_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    roots = [Path(item) for item in custom_path.split(":") if item]
    matches: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in (".o", ".json"):
            for path in root.rglob(CUSTOM_KERNEL_BASENAME + suffix):
                matches.append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "ascend_custom_opp_path": custom_path or None,
        "kernel_files": sorted(matches, key=lambda item: item["path"]),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "p90": percentile(values, 0.90),
    }


def cpu_inputs(torch: Any, kv_length: int, seed: int) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + kv_length)
    query = torch.randn(
        (1, QUERY_HEADS, 1, HEAD_DIM),
        generator=generator,
        dtype=torch.float16,
    )
    key = torch.randn(
        (1, KV_HEADS, kv_length, HEAD_DIM),
        generator=generator,
        dtype=torch.float16,
    )
    value = torch.randn(
        (1, KV_HEADS, kv_length, HEAD_DIM),
        generator=generator,
        dtype=torch.float16,
    )
    return query, key, value


def run_case(
    torch: Any,
    torch_npu: Any,
    *,
    device: Any,
    kv_length: int,
    seed: int,
    warmup: int,
    blocks: int,
    repeats_per_block: int,
) -> dict[str, Any]:
    query_cpu, key_cpu, value_cpu = cpu_inputs(torch, kv_length, seed)
    query = query_cpu.to(device).contiguous()
    key = key_cpu.to(device).contiguous()
    value = value_cpu.to(device).contiguous()
    # Keep the production masked-MHA call contract.  At the last valid cache
    # position the mask is all false, so KV length is also the effective length.
    mask = torch.zeros((1, 1, 1, kv_length), device=device, dtype=torch.bool)

    def step() -> Any:
        return torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            atten_mask=mask,
            actual_seq_lengths=None,
            num_heads=QUERY_HEADS,
            num_key_value_heads=0,
            input_layout="BNSD",
            scale_value=SCALE_VALUE,
            inner_precise=1,
        )

    output = None
    for _ in range(warmup):
        output = step()
    torch.npu.synchronize()

    event_us_per_call: list[float] = []
    host_us_per_call: list[float] = []
    for _ in range(blocks):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        host_started = time.perf_counter()
        start_event.record()
        for _ in range(repeats_per_block):
            output = step()
        end_event.record()
        end_event.synchronize()
        host_elapsed_s = time.perf_counter() - host_started
        event_us_per_call.append(
            float(start_event.elapsed_time(end_event)) * 1000.0 / repeats_per_block
        )
        host_us_per_call.append(host_elapsed_s * 1e6 / repeats_per_block)

    assert output is not None
    output_cpu = output.detach().to("cpu").float().contiguous()
    return {
        "kv_length": kv_length,
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "mask_shape": list(mask.shape),
        "output_shape": list(output_cpu.shape),
        "output_values_fp32": output_cpu.flatten().tolist(),
        "timing": {
            "profiled": False,
            "warmup_calls": warmup,
            "blocks": blocks,
            "repeats_per_block": repeats_per_block,
            "npu_event_us_per_call": event_us_per_call,
            "host_wall_us_per_call": host_us_per_call,
            "npu_event_us_per_call_summary": timing_summary(event_us_per_call),
            "host_wall_us_per_call_summary": timing_summary(host_us_per_call),
        },
    }


def run_command(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{args.device_index}")
    result = {
        "schema_version": 1,
        "kind": "custom_increfa_reproduction_run",
        "label": args.label,
        "configuration": {
            "batch_size": 1,
            "query_heads": QUERY_HEADS,
            "key_value_heads": KV_HEADS,
            "op_num_key_value_heads": 0,
            "head_dim": HEAD_DIM,
            "input_layout": "BNSD",
            "dtype": "fp16",
            "masked": True,
            "actual_seq_lengths": None,
            "inner_precise": 1,
            "scale_value": SCALE_VALUE,
            "kv_lengths": list(args.kv_lengths),
            "seed": args.seed,
            "device_index": args.device_index,
        },
        "environment": {
            "hostname": os.uname().nodename,
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "device_name": torch.npu.get_device_name(args.device_index),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
        },
        "custom_package": custom_package_evidence(),
        "cases": [],
    }
    for kv_length in args.kv_lengths:
        result["cases"].append(
            run_case(
                torch,
                torch_npu,
                device=device,
                kv_length=kv_length,
                seed=args.seed,
                warmup=args.warmup,
                blocks=args.blocks,
                repeats_per_block=args.repeats_per_block,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "label": args.label,
        "output": str(args.output),
        "custom_package": result["custom_package"],
        "timing": {
            str(case["kv_length"]): case["timing"][
                "npu_event_us_per_call_summary"
            ]
            for case in result["cases"]
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def compare_vectors(
    stock: Sequence[float],
    custom: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(stock) != len(custom):
        raise ValueError(f"output length mismatch: {len(stock)} != {len(custom)}")
    absolute = [abs(left - right) for left, right in zip(stock, custom)]
    tolerance = [atol + rtol * abs(left) for left in stock]
    stock_l2 = math.sqrt(sum(value * value for value in stock))
    diff_l2 = math.sqrt(sum(value * value for value in absolute))
    dot = sum(left * right for left, right in zip(stock, custom))
    custom_l2 = math.sqrt(sum(value * value for value in custom))
    return {
        "element_count": len(stock),
        "exact_count": sum(left == right for left, right in zip(stock, custom)),
        "allclose": all(diff <= limit for diff, limit in zip(absolute, tolerance)),
        "max_abs": max(absolute, default=0.0),
        "mean_abs": statistics.fmean(absolute) if absolute else 0.0,
        "relative_l2": diff_l2 / stock_l2 if stock_l2 else 0.0,
        "cosine_similarity": (
            dot / (stock_l2 * custom_l2) if stock_l2 and custom_l2 else 1.0
        ),
    }


def compare_command(args: argparse.Namespace) -> int:
    stock = json.loads(args.stock.read_text(encoding="utf-8"))
    custom = json.loads(args.custom.read_text(encoding="utf-8"))
    stock_cases = {case["kv_length"]: case for case in stock["cases"]}
    custom_cases = {case["kv_length"]: case for case in custom["cases"]}
    if stock_cases.keys() != custom_cases.keys():
        raise ValueError("stock and custom KV-length sets differ")
    comparisons = []
    for kv_length in sorted(stock_cases):
        comparison = compare_vectors(
            stock_cases[kv_length]["output_values_fp32"],
            custom_cases[kv_length]["output_values_fp32"],
            atol=args.atol,
            rtol=args.rtol,
        )
        comparison["kv_length"] = kv_length
        comparison["stock_npu_event_us_per_call"] = stock_cases[kv_length][
            "timing"
        ]["npu_event_us_per_call_summary"]
        comparison["custom_npu_event_us_per_call"] = custom_cases[kv_length][
            "timing"
        ]["npu_event_us_per_call_summary"]
        custom_mean = comparison["custom_npu_event_us_per_call"]["mean"]
        stock_mean = comparison["stock_npu_event_us_per_call"]["mean"]
        comparison["stock_over_custom_speed_ratio"] = stock_mean / custom_mean
        comparisons.append(comparison)
    result = {
        "schema_version": 1,
        "kind": "custom_increfa_reproduction_comparison",
        "stock": str(args.stock),
        "custom": str(args.custom),
        "atol": args.atol,
        "rtol": args.rtol,
        "all_cases_allclose": all(item["allclose"] for item in comparisons),
        "all_cases_exact": all(
            item["exact_count"] == item["element_count"] for item in comparisons
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["all_cases_allclose"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "compare":
        return compare_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
