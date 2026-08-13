#!/usr/bin/env python3
"""Profile one exact-shape stock-eager UniRec vision B1 forward."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch_npu


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--parser-topn", type=int, default=100)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if args.width % 32 or args.height % 32:
        parser.error("width and height must be divisible by 32")
    if args.warmups < 1 or args.control_repeats < 2:
        parser.error("use at least one warmup and two control repeats")
    if args.parser_topn < 1:
        parser.error("parser topn must be positive")
    return args


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible physical NPU, got {devices}")
    if devices[0] in {5, 6}:
        raise RuntimeError(f"physical NPU {devices[0]} is excluded")
    return devices


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": _percentile(samples, 0.90),
        "max_ms": max(samples),
    }


def _measure_controls(
    run: Callable[[], torch.Tensor],
    *,
    device: str,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    event_ms = []
    wall_ms = []
    output = None
    for _ in range(repeats):
        synchronize_device(device)
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        output = run()
        end.record()
        end.synchronize()
        event_ms.append(float(start.elapsed_time(end)))
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
    if output is None:
        raise RuntimeError("control forward produced no output")
    return {
        "device_event": _summary(event_ms),
        "synchronized_wall": _summary(wall_ms),
    }, output


def _profiler_config(metric: str) -> Any:
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


def _parse_profile(profile_dir: Path, *, topn: int) -> dict[str, Any]:
    parser_path = REPO_ROOT / "11_mineru_2_5_pro_inference/parse_npu_profile.py"
    parsed_json = profile_dir / "profile_parse_summary.json"
    parsed_md = profile_dir / "profile_parse_summary.md"
    command = [
        sys.executable,
        str(parser_path),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--out-json",
        str(parsed_json),
        "--out-md",
        str(parsed_md),
        "--skip-trace",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "summary": json.loads(parsed_json.read_text(encoding="utf-8")),
        "markdown": str(parsed_md),
    }


def _profile_once(
    run: Callable[[], torch.Tensor],
    *,
    device: str,
    profile_dir: Path,
    metric: str,
) -> tuple[dict[str, float], torch.Tensor]:
    import torch_npu.profiler as npu_prof

    profile_dir.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    synchronize_device(device)
    event_start = torch_npu.npu.Event(enable_timing=True)
    event_end = torch_npu.npu.Event(enable_timing=True)
    context_started = time.perf_counter()
    profiled_output = None
    forward_wall_ms = 0.0
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        synchronize_device(device)
        forward_started = time.perf_counter()
        event_start.record()
        with torch.profiler.record_function("unirec.stock_eager_vision_b1"):
            profiled_output = run()
        event_end.record()
        event_end.synchronize()
        forward_wall_ms = (time.perf_counter() - forward_started) * 1000.0
        profiler.step()
    if profiled_output is None:
        raise RuntimeError("profiled forward produced no output")
    return {
        "device_event_ms": float(event_start.elapsed_time(event_end)),
        "synchronized_forward_wall_ms": forward_wall_ms,
        "profile_context_wall_s": time.perf_counter() - context_started,
    }, profiled_output


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = (left - right).abs()
    return {
        "allclose_atol_5e_2_rtol_5e_2": bool(
            torch.allclose(left, right, atol=5e-2, rtol=5e-2)
        ),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _compact_comparison_reference(report: dict[str, Any]) -> dict[str, Any]:
    parsed_run = report["parsed_profile"]["summary"]["runs"][0]
    kernel = parsed_run["kernel_details"]
    row_fields = ("name", "count", "duration_us")

    def compact_rows(name: str) -> list[dict[str, Any]]:
        return [
            {field: row[field] for field in row_fields}
            for row in kernel.get(name, [])
        ]

    compact_kernel = {
        "row_count": kernel["row_count"],
        "total_duration_us": kernel["total_duration_us"],
        "weighted_cube_utilization_pct": kernel[
            "weighted_cube_utilization_pct"
        ],
        "top_kernel_types": compact_rows("top_kernel_types"),
        "top_shape_signatures": compact_rows("top_shape_signatures"),
        "top_matmul_shape_signatures": compact_rows(
            "top_matmul_shape_signatures"
        ),
        "top_transdata_shape_signatures": compact_rows(
            "top_transdata_shape_signatures"
        ),
    }
    return {
        "status": report["status"],
        "device_name": report["device_name"],
        "dtype": report["dtype"],
        "npu_jit_compile": report["npu_jit_compile"],
        "input_shape": report["input_shape"],
        "execution": report["execution"],
        "control_before": report["control_before"],
        "profile_timing": report["profile_timing"],
        "control_after": report["control_after"],
        "profiler_overhead": report["profiler_overhead"],
        "correctness": report["correctness"],
        "profile_metric": report["profile_metric"],
        "parsed_profile": {
            "summary": {
                "runs": [
                    {
                        "step_trace_time": {
                            "row_count": parsed_run["step_trace_time"][
                                "row_count"
                            ],
                            "totals_us": parsed_run["step_trace_time"][
                                "totals_us"
                            ],
                        },
                        "kernel_details": compact_kernel,
                    }
                ]
            }
        },
    }


def main() -> None:
    args = parse_args()
    devices = _physical_devices()
    if not args.device.startswith("npu"):
        raise ValueError("this profiler requires an NPU device")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    setup_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
    )
    pixels = torch.zeros(
        (1, 3, args.height, args.width),
        device=args.device,
        dtype=runner.dtype,
    )
    setup_s = time.perf_counter() - setup_started

    def run() -> torch.Tensor:
        return runner.model.forward_encoder(pixels)

    with torch.inference_mode():
        warmup_wall_ms = []
        for _ in range(args.warmups):
            synchronize_device(args.device)
            started = time.perf_counter()
            run()
            synchronize_device(args.device)
            warmup_wall_ms.append((time.perf_counter() - started) * 1000.0)
        control_before, before_output = _measure_controls(
            run,
            device=args.device,
            repeats=args.control_repeats,
        )
        profile_dir = output_dir / f"profile_{args.profile_metric}"
        profile_timing, profiled_output = _profile_once(
            run,
            device=args.device,
            profile_dir=profile_dir,
            metric=args.profile_metric,
        )
        control_after, after_output = _measure_controls(
            run,
            device=args.device,
            repeats=args.control_repeats,
        )

    parsed = _parse_profile(profile_dir, topn=args.parser_topn)
    before_event = float(control_before["device_event"]["median_ms"])
    after_event = float(control_after["device_event"]["median_ms"])
    control_combined = (before_event + after_event) / 2.0
    profile_event = float(profile_timing["device_event_ms"])
    correctness = {
        "profiled_vs_control_before": _difference(profiled_output, before_output),
        "control_after_vs_control_before": _difference(after_output, before_output),
    }
    allclose = all(
        row["allclose_atol_5e_2_rtol_5e_2"] for row in correctness.values()
    )
    report = {
        "status": "ok" if allclose else "correctness_failed",
        "device": args.device,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "input_shape": [1, 3, args.height, args.width],
        "execution": "stock_eager_model_forward_encoder",
        "model_setup_s": setup_s,
        "warmup_wall_ms": warmup_wall_ms,
        "control_repeats": args.control_repeats,
        "control_before": control_before,
        "profile_timing": profile_timing,
        "control_after": control_after,
        "profiler_overhead": {
            "profiled_event_vs_control_before": profile_event / before_event,
            "profiled_event_vs_combined_control": profile_event / control_combined,
            "control_after_vs_before": after_event / before_event,
        },
        "correctness": correctness,
        "profile_metric": args.profile_metric,
        "profile_dir": str(profile_dir),
        "parsed_profile": parsed,
    }
    output_json = output_dir / "result.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    comparison_reference = output_dir / "comparison_reference.json"
    comparison_reference.write_text(
        json.dumps(_compact_comparison_reference(report), indent=2) + "\n",
        encoding="utf-8",
    )

    run_summary = parsed["summary"]["runs"][0]
    kernels = run_summary.get("kernel_details", {})
    print(
        "UNIREC_STOCK_EAGER_VISION_B1_PROFILE "
        f"device={report['device_name']} "
        f"control_before_ms={before_event:.3f} "
        f"profiled_event_ms={profile_event:.3f} "
        f"control_after_ms={after_event:.3f} "
        f"profile_overhead={profile_event / control_combined:.3f} "
        f"post_profile_ratio={after_event / before_event:.3f} "
        f"kernel_count={int(kernels.get('row_count', 0))} "
        f"kernel_total_ms={float(kernels.get('total_duration_us', 0.0)) / 1000.0:.3f} "
        f"cube_pct={float(kernels.get('weighted_cube_utilization_pct', 0.0)):.2f}",
        flush=True,
    )
    for row in kernels.get("top_kernel_types", [])[:15]:
        print(
            "UNIREC_STOCK_EAGER_VISION_B1_KERNEL_TYPE "
            f"name={json.dumps(row['name'])} count={int(row['count'])} "
            f"duration_ms={float(row['duration_us']) / 1000.0:.3f}",
            flush=True,
        )
    print(f"PROFILE_PARSE_JSON={profile_dir / 'profile_parse_summary.json'}")
    print(f"COMPARISON_REFERENCE_JSON={comparison_reference}")
    print(f"OUTPUT_JSON={output_json}", flush=True)
    if not allclose:
        raise RuntimeError("profiled stock-eager output failed control parity")


if __name__ == "__main__":
    main()
