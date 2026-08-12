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
        choices=(
            "all",
            "stock_depthwise",
            "grouped16",
            "grouped16_fz",
            "grouped32_fz",
            "grouped64_fz",
            "dense_native",
            "dense_internal",
            "dense_fz",
            "shift_sum",
        ),
        default="all",
    )
    parser.add_argument("--channels", type=int, choices=(192, 384))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--freeze-parameters", action="store_true")
    return parser.parse_args()


def grouped_weight(weight: torch.Tensor, group_width: int) -> torch.Tensor:
    channels = weight.shape[0]
    if channels % group_width:
        raise ValueError(
            f"channels must be divisible by group_width, got {channels=} "
            f"{group_width=}"
        )
    channel_in_group = torch.arange(channels, device=weight.device).remainder(
        group_width
    )
    selector = F.one_hot(channel_in_group, num_classes=group_width).to(weight.dtype)
    return weight * selector.view(channels, group_width, 1, 1)


class StockDepthwise(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(inputs, self.weight, padding=2, groups=inputs.shape[1])


class ExactGrouped(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        group_width: int,
        preformat_fz: bool,
    ) -> None:
        super().__init__()
        grouped = grouped_weight(weight, group_width)
        if preformat_fz:
            import torch_npu

            grouped = torch_npu.npu_format_cast(grouped, 4)
        channels = weight.shape[0]
        self.conv = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels // group_width,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.conv.weight = torch.nn.Parameter(grouped, requires_grad=False)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.conv.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


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
    freeze_parameters: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    config.experimental_config.frozen_parameter.value = freeze_parameters
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
            "grouped16": ExactGrouped(
                weight.clone(), group_width=16, preformat_fz=False
            )
            .to(args.device)
            .eval(),
            "grouped16_fz": ExactGrouped(
                weight.clone(), group_width=16, preformat_fz=True
            )
            .to(args.device)
            .eval(),
            "grouped32_fz": ExactGrouped(
                weight.clone(), group_width=32, preformat_fz=True
            )
            .to(args.device)
            .eval(),
            "grouped64_fz": ExactGrouped(
                weight.clone(), group_width=64, preformat_fz=True
            )
            .to(args.device)
            .eval(),
            "dense_fz": ExactGrouped(
                weight.clone(), group_width=channels, preformat_fz=True
            )
            .to(args.device)
            .eval(),
            "dense_native": ExactGrouped(
                weight.clone(), group_width=channels, preformat_fz=False
            )
            .to(args.device)
            .eval(),
            "dense_internal": ExactGrouped(
                weight.clone(), group_width=channels, preformat_fz=False
            )
            .to(args.device)
            .eval(),
            "shift_sum": ShiftSum(weight.clone()).to(args.device).eval(),
        }
        try:
            from torch_npu.dynamo.torchair import use_internal_format_weight
        except ImportError:
            from torchair import use_internal_format_weight
        use_internal_format_weight(modules["dense_internal"])
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
                    f"{name}_c{channels}_{height}x{width}_"
                    f"frozen{int(args.freeze_parameters)}_src{source_hash}"
                )
                call = compile_module(
                    module,
                    cache_dir=cache_dir,
                    freeze_parameters=args.freeze_parameters,
                )
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
