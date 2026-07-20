#!/usr/bin/env python3
"""Compare compiled FP16 Linear with compiled static W8A8 vision fc1."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch

from w8a8_vision import PackedW8A8Linear


def synchronize() -> None:
    torch.npu.synchronize()


def timed(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    output = fn(x)
    for _ in range(max(0, warmup - 1)):
        output = fn(x)
    synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        output = fn(x)
    synchronize()
    return float((time.perf_counter() - started) / iterations), output


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    diff = (reference - candidate).abs()
    cosine = torch.dot(reference, candidate) / (
        torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(candidate)
    ).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(cosine.item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--backend", default="npu")
    parser.add_argument("--weight-layout", default="nd_kn")
    parser.add_argument(
        "--quantization",
        default="w8a8_static",
        choices=("w8a8_static", "w8a8_static_pad64"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch_npu

    device = torch.device(args.device)
    dtype = torch.float16
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234 + int(args.rows))
    x = torch.randn((args.rows, 1152), generator=generator, dtype=dtype).to(device)
    weight = (
        torch.randn((4304, 1152), generator=generator, dtype=torch.float32)
        / math.sqrt(1152)
    ).to(dtype=dtype, device=device)
    bias = torch.zeros((4304,), dtype=dtype, device=device)

    fp16 = torch.nn.Linear(1152, 4304, bias=True, device=device, dtype=dtype).eval()
    with torch.no_grad():
        fp16.weight.copy_(weight)
        fp16.bias.copy_(bias)
    input_scale = float(x.float().abs().max().item()) * 1.05 / 127.0
    w8a8 = PackedW8A8Linear(
        weight,
        bias,
        mode=str(args.quantization),
        weight_layout=str(args.weight_layout),
        static_input_scale=input_scale,
    ).eval()

    synchronize()
    fp16_compile_started = time.perf_counter()
    compiled_fp16 = torch.compile(
        fp16,
        backend=str(args.backend),
        fullgraph=True,
        dynamic=False,
    )
    fp16_first = compiled_fp16(x)
    synchronize()
    fp16_first_call_s = float(time.perf_counter() - fp16_compile_started)

    synchronize()
    w8a8_compile_started = time.perf_counter()
    compiled_w8a8 = torch.compile(
        w8a8,
        backend=str(args.backend),
        fullgraph=True,
        dynamic=False,
    )
    w8a8_first = compiled_w8a8(x)
    synchronize()
    w8a8_first_call_s = float(time.perf_counter() - w8a8_compile_started)

    fp16_s, fp16_output = timed(
        compiled_fp16,
        x,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    w8a8_s, w8a8_output = timed(
        compiled_w8a8,
        x,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    output = {
        "schema_version": 1,
        "kind": "compiled_static_w8a8_vision_fc1_benchmark",
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "device": str(device),
        "device_name": str(torch.npu.get_device_name(torch.npu.current_device())),
        "rows": int(args.rows),
        "in_features": 1152,
        "out_features": 4304,
        "backend": str(args.backend),
        "weight_layout": str(args.weight_layout),
        "quantization": str(args.quantization),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "fp16_first_call_s": fp16_first_call_s,
        "w8a8_first_call_s": w8a8_first_call_s,
        "fp16_s": fp16_s,
        "w8a8_s": w8a8_s,
        "w8a8_speedup": float(fp16_s / w8a8_s),
        "first_call_diff": diff_stats(fp16_first, w8a8_first),
        "steady_state_diff": diff_stats(fp16_output, w8a8_output),
        "w8a8_nonfinite_count": int((~torch.isfinite(w8a8_output.float())).sum().item()),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    print(f"COMPILED_W8A8_OUTPUT={output_path}", flush=True)


if __name__ == "__main__":
    main()
