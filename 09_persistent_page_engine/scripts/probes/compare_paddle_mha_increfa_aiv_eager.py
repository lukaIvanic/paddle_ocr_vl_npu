#!/usr/bin/env python3
"""Compare stock IncreFA with the separate AIV op through PyTorch eager."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Callable, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
DEFAULT_EXTENSION_ROOT = (
    EXPERIMENT_ROOT
    / "custom_ops/paddle_mha_increfa_aiv/pytorch_extension"
)
QUERY_HEADS = 16
HEAD_DIM = 128
SCALE_VALUE = 1.0 / math.sqrt(HEAD_DIM)


def parse_lengths(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("KV lengths must be positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument(
        "--lanes",
        choices=("both", "stock", "custom"),
        default="both",
        help="Run both parity lanes or isolate one eager dispatch path.",
    )
    parser.add_argument("--kv-lengths", type=parse_lengths, default=(128, 512, 2048))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--repeats-per-block", type=int, default=200)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help=(
            "Optionally capture one torch_npu profile per KV length. Profiling "
            "requires an isolated --lanes stock or --lanes custom process."
        ),
    )
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    if (
        args.warmup < 0
        or args.blocks <= 0
        or args.repeats_per_block <= 0
        or args.profile_iters <= 0
    ):
        parser.error(
            "warmup must be nonnegative; blocks, repeats, and profile-iters "
            "must be positive"
        )
    if args.profile_dir is not None and args.lanes == "both":
        parser.error("--profile-dir requires isolated --lanes stock or custom")
    return args


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


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def time_step(
    step: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    blocks: int,
    repeats_per_block: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output = None
    for _ in range(warmup):
        output = step(*inputs)
    torch.npu.synchronize()

    event_us: list[float] = []
    host_us: list[float] = []
    for _ in range(blocks):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host_started = time.perf_counter()
        start.record()
        for _ in range(repeats_per_block):
            output = step(*inputs)
        end.record()
        end.synchronize()
        event_us.append(float(start.elapsed_time(end)) * 1000.0 / repeats_per_block)
        host_us.append(
            (time.perf_counter() - host_started) * 1e6 / repeats_per_block
        )
    assert output is not None
    return output, {
        "backend": "pytorch_raw_eager",
        "warmup_calls": warmup,
        "blocks": blocks,
        "repeats_per_block": repeats_per_block,
        "npu_event_us_per_call": event_us,
        "host_wall_us_per_call": host_us,
        "npu_event_us_per_call_summary": timing_summary(event_us),
        "host_wall_us_per_call_summary": timing_summary(host_us),
    }


def profiler_config(metric: str) -> Any:
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def profile_step(
    step: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    lane: str,
    kv_length: int,
    profile_dir: Path,
    metric: str,
    warmup: int,
    profile_iters: int,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    case_dir = profile_dir / f"{lane}_kv{kv_length}_{metric}"
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        step(*inputs)
    torch.npu.synchronize()

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    wall_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(case_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
    ) as profiler:
        with torch.profiler.record_function(
            f"paddleocr_vl.mha_increfa_eager.{lane}.kv{kv_length}"
        ):
            for _ in range(profile_iters):
                step(*inputs)
        torch.npu.synchronize()
        profiler.step()
    torch.npu.synchronize()
    return {
        "profile_dir": str(case_dir),
        "metric": metric,
        "profiler_level": "Level1",
        "warmup_calls_outside_profiler": warmup,
        "captured_calls": profile_iters,
        "profile_wall_s": time.perf_counter() - wall_started,
        "profile_wall_is_throughput_measurement": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch_npu

    extension_root = args.extension_root.expanduser().resolve()
    sys.path.insert(0, str(extension_root))
    from paddle_mha_increfa_aiv_eager import (
        ACLNN_OP_NAME,
        EXTENSION_PATH,
        PYTORCH_OP_NAME,
        paddle_mha_incre_flash_attention_aiv_eager,
    )

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def stock(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def custom(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return paddle_mha_incre_flash_attention_aiv_eager(
            query,
            key,
            value,
            mask,
            num_heads=QUERY_HEADS,
            scale_value=SCALE_VALUE,
            inner_precise=1,
        )

    dispatch_table = torch._C._dispatch_dump_table(PYTORCH_OP_NAME)
    if "PrivateUse1" not in dispatch_table:
        raise RuntimeError("custom eager op has no PrivateUse1 implementation")
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "separate_paddle_mha_increfa_aiv_eager_comparison",
        "operator_identity": {
            "pytorch": PYTORCH_OP_NAME,
            "aclnn": ACLNN_OP_NAME,
            "extension_so": str(EXTENSION_PATH),
            "stock_reference": "torch_npu.npu_incre_flash_attention",
            "backend": "pytorch_raw_eager",
            "torchair_used": False,
            "python_stock_fallback": False,
            "same_name_override": False,
            "dispatch_table": dispatch_table,
        },
        "lanes": args.lanes,
        "contract": {
            "batch_size": 1,
            "query_heads": QUERY_HEADS,
            "key_value_heads": QUERY_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "fp16",
            "layout": "BNSD",
            "masked": True,
            "actual_seq_lengths": None,
            "inner_precise": 1,
            "scale_value": SCALE_VALUE,
            "kv_lengths": list(args.kv_lengths),
        },
        "environment": {
            "hostname": os.uname().nodename,
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "device_name": torch.npu.get_device_name(0),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
        },
        "cases": [],
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    with torch.inference_mode():
        for kv_length in args.kv_lengths:
            query = torch.randn(
                (1, QUERY_HEADS, 1, HEAD_DIM),
                generator=generator,
                dtype=torch.float16,
            ).to(device)
            key = torch.randn(
                (1, QUERY_HEADS, kv_length, HEAD_DIM),
                generator=generator,
                dtype=torch.float16,
            ).to(device)
            value = torch.randn(
                (1, QUERY_HEADS, kv_length, HEAD_DIM),
                generator=generator,
                dtype=torch.float16,
            ).to(device)
            mask = torch.zeros(
                (1, 1, 1, kv_length), dtype=torch.bool, device=device
            )
            inputs = (query, key, value, mask)

            case: dict[str, Any] = {
                "kv_length": kv_length,
                "first_call_s": {},
                "timing": {},
                "profile": {},
                "output_sha256": {},
                "parity": None,
            }
            outputs: dict[str, torch.Tensor] = {}
            lane_steps = (
                (("stock", stock), ("custom", custom))
                if args.lanes == "both"
                else ((args.lanes, stock if args.lanes == "stock" else custom),)
            )
            for lane_name, lane_step in lane_steps:
                started = time.perf_counter()
                output = lane_step(*inputs)
                torch.npu.synchronize()
                case["first_call_s"][lane_name] = time.perf_counter() - started
                output, timing = time_step(
                    lane_step,
                    inputs,
                    warmup=args.warmup,
                    blocks=args.blocks,
                    repeats_per_block=args.repeats_per_block,
                )
                output_cpu = output.float().cpu().contiguous()
                outputs[lane_name] = output_cpu
                case["timing"][lane_name] = timing
                case["output_sha256"][lane_name] = tensor_sha256(output_cpu)
                if args.profile_dir is not None:
                    case["profile"][lane_name] = profile_step(
                        lane_step,
                        inputs,
                        lane=lane_name,
                        kv_length=kv_length,
                        profile_dir=args.profile_dir.expanduser().resolve(),
                        metric=args.profile_metric,
                        warmup=args.warmup,
                        profile_iters=args.profile_iters,
                    )

            if args.lanes == "both":
                stock_cpu = outputs["stock"]
                custom_cpu = outputs["custom"]
                difference = (stock_cpu - custom_cpu).abs()
                case["parity"] = {
                    "exact": bool(torch.equal(stock_cpu, custom_cpu)),
                    "allclose_atol_0_rtol_0": bool(
                        torch.allclose(stock_cpu, custom_cpu, atol=0.0, rtol=0.0)
                    ),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "stock_sha256": case["output_sha256"]["stock"],
                    "custom_sha256": case["output_sha256"]["custom"],
                }
            result["cases"].append(case)

    result["all_exact"] = (
        all(case["parity"]["exact"] for case in result["cases"])
        if args.lanes == "both"
        else None
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.lanes == "both" and not result["all_exact"]:
        raise SystemExit("separate eager operator did not match stock exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
