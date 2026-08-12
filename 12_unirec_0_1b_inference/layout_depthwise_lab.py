#!/usr/bin/env python3
"""Benchmark exact alternatives to PP-DocLayoutV2 depthwise 5x5 convolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--execution", choices=("eager", "torchair"), default="torchair"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".runtime_cache/12_unirec_0_1b_inference/layout_depthwise_lab"),
    )
    parser.add_argument(
        "--implementation",
        choices=("all", "stock_depthwise", "grouped16", "shift_sum"),
        default="all",
    )
    parser.add_argument("--channels", type=int, choices=(192, 384))
    parser.add_argument("--profile-dir", type=Path)
    return parser.parse_args()


def grouped16_weight(weight: torch.Tensor) -> torch.Tensor:
    channels = weight.shape[0]
    if channels % 16:
        raise ValueError(f"channels must be divisible by 16, got {channels}")
    channel_in_group = torch.arange(channels, device=weight.device).remainder(16)
    selector = F.one_hot(channel_in_group, num_classes=16).to(weight.dtype)
    return weight * selector.view(channels, 16, 1, 1)


class StockDepthwise(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(inputs, self.weight, padding=2, groups=inputs.shape[1])


class Grouped16(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(grouped16_weight(weight), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(inputs, self.weight, padding=2, groups=inputs.shape[1] // 16)


class ShiftSum(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight[:, 0], requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        padded = F.pad(inputs, (2, 2, 2, 2))
        output = inputs.new_zeros(inputs.shape)
        for row in range(5):
            for column in range(5):
                output = output + padded[
                    :, :, row : row + height, column : column + width
                ] * self.weight[:, row, column].view(1, -1, 1, 1)
        return output


def event_ms(
    call: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            call(inputs)
        torch.npu.synchronize()
        started = torch.npu.Event(enable_timing=True)
        ended = torch.npu.Event(enable_timing=True)
        started.record()
        for _ in range(repeats):
            call(inputs)
        ended.record()
        ended.synchronize()
    return float(started.elapsed_time(ended)) / repeats


def compile_module(
    module: torch.nn.Module,
    *,
    cache_dir: Path,
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_compile(
        module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )


def profile_call(
    call: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    profile_dir: Path,
) -> None:
    import torch_npu.profiler as npu_prof

    profile_dir = profile_dir.expanduser().resolve()
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    experimental = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )
    torch.npu.synchronize()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=experimental,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("unirec.layout_depthwise_lab"):
            call(inputs)
        torch.npu.synchronize()
        profiler.step()


def main() -> None:
    args = parse_args()
    import torch_npu

    torch.npu.set_device(torch.device(args.device))
    torch.npu.set_compile_mode(jit_compile=False)
    cases = ((192, 50, 50, 18), (384, 25, 25, 6))
    if args.channels is not None:
        cases = tuple(case for case in cases if case[0] == args.channels)
    if args.profile_dir is not None and (
        args.implementation == "all" or args.channels is None
    ):
        raise ValueError(
            "--profile-dir requires one --implementation and one --channels"
        )
    report = []
    for channels, height, width, production_calls in cases:
        torch.manual_seed(channels)
        inputs = torch.randn(
            (1, channels, height, width), device=args.device, dtype=torch.float16
        )
        weight = torch.randn(
            (channels, 1, 5, 5), device=args.device, dtype=torch.float16
        )
        modules = {
            "stock_depthwise": StockDepthwise(weight.clone()).to(args.device).eval(),
            "grouped16": Grouped16(weight.clone()).to(args.device).eval(),
            "shift_sum": ShiftSum(weight.clone()).to(args.device).eval(),
        }
        with torch.inference_mode():
            reference = modules["stock_depthwise"](inputs)
            torch.npu.synchronize()
        if args.implementation != "all":
            modules = {args.implementation: modules[args.implementation]}
        for name, module in modules.items():
            call: Callable[[torch.Tensor], torch.Tensor] = module
            cache_dir = None
            if args.execution == "torchair":
                source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
                cache_dir = args.cache_dir.expanduser().resolve() / (
                    f"{name}_c{channels}_{height}x{width}_src{source_hash}"
                )
                call = compile_module(module, cache_dir=cache_dir)
            with torch.inference_mode():
                first_call_started = time.perf_counter()
                output = call(inputs)
                torch.npu.synchronize()
                first_call_s = time.perf_counter() - first_call_started
                difference = (output.float() - reference.float()).abs()
            mean_ms = event_ms(call, inputs, args.warmup, args.repeats)
            if args.profile_dir is not None:
                profile_call(call, inputs, profile_dir=args.profile_dir)
            row = {
                "channels": channels,
                "height": height,
                "width": width,
                "production_calls": production_calls,
                "implementation": name,
                "execution": args.execution,
                "first_call_s": first_call_s,
                "mean_ms": mean_ms,
                "weighted_production_ms": mean_ms * production_calls,
                "mean_abs_diff": float(difference.mean().cpu()),
                "max_abs_diff": float(difference.max().cpu()),
                "weight_format": int(torch_npu.get_npu_format(module.weight)),
                "cache_dir": None if cache_dir is None else str(cache_dir),
            }
            report.append(row)
            print("LAYOUT_DEPTHWISE " + json.dumps(row, sort_keys=True), flush=True)
    print("LAYOUT_DEPTHWISE_SUMMARY " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
