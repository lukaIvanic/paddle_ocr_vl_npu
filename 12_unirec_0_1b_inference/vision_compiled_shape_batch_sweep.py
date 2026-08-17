#!/usr/bin/env python3
"""Measure one UniRec compiled vision canvas across physical batch sizes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)
import vision_full_batch  # noqa: E402
from vision_full_batch import (  # noqa: E402
    BucketedFullVisionRuntime,
    VisionBucketSpec,
    _MaskedFullVisionEncoder,
    _make_host_masks,
)


def _static_forward_module_factory(
    runner: OptimizedUniRecRunner,
    _spec: VisionBucketSpec,
) -> _MaskedFullVisionEncoder:
    """Use the ordinary class-defined forward for cache-persistence diagnosis."""
    return _MaskedFullVisionEncoder(runner).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--batch-sizes", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--module-identity",
        default="dynamic_bucket",
        choices=("dynamic_bucket", "static_class"),
        help=(
            "Use production's dynamic per-bucket forward identity or the "
            "ordinary class-defined forward for cache-persistence diagnosis."
        ),
    )
    parser.add_argument(
        "--focal-depthwise-rewrite",
        default="constant_grouped_all",
        choices=("constant_grouped_all",),
    )
    parser.add_argument(
        "--weight-format",
        default="torchair_internal",
        choices=("torchair_internal",),
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if args.width % 32 or args.height % 32:
        parser.error("width and height must be divisible by 32")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    return args


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not raw:
        raise RuntimeError(
            "set ASCEND_RT_VISIBLE_DEVICES to exactly one physical NPU before "
            "launching the vision sweep"
        )
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible physical NPU, got {devices}")
    if devices[0] in {5, 6}:
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    return devices


def _parse_batch_sizes(raw: str) -> list[int]:
    batches = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not batches or batches[0] < 1:
        raise ValueError("batch sizes must be positive")
    if batches[0] != 1:
        raise ValueError("the sweep requires B1 as the cross-batch reference")
    return batches


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p90_ms": _percentile(values, 0.90),
        "max_ms": max(values),
    }


def _measure_ms(function: Callable[[], torch.Tensor]) -> float:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = (left - right).abs()
    return {
        "allclose_atol_5e_2_rtol_5e_2": bool(
            torch.allclose(left, right, atol=5e-2, rtol=5e-2)
        ),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def main() -> None:
    args = parse_args()
    devices = _physical_devices()
    if not args.device.startswith("npu"):
        raise ValueError("the vision sweep requires an NPU device")
    batches = _parse_batch_sizes(args.batch_sizes)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=cache_dir,
    )
    if args.module_identity == "static_class":
        vision_full_batch._new_masked_full_encoder_module = (
            _static_forward_module_factory
        )
    specs = tuple(
        VisionBucketSpec(args.width, args.height, batch) for batch in batches
    )
    print(
        "UNIREC_VISION_SWEEP_RUNTIME_BEGIN "
        f"shape={args.width}x{args.height} batches={batches}",
        flush=True,
    )
    runtime_started = time.perf_counter()
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=specs,
        diagnostic_graph_log=True,
        focal_depthwise_rewrite=args.focal_depthwise_rewrite,
        weight_format=args.weight_format,
    )
    runtime_init_wall_s = time.perf_counter() - runtime_started
    print(
        "UNIREC_VISION_SWEEP_RUNTIME_END "
        f"shape={args.width}x{args.height} wall_s={runtime_init_wall_s:.6f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    b1_output: torch.Tensor | None = None
    warning_count = 0
    device = torch.device(args.device)
    with torch.inference_mode():
        for spec in specs:
            print(
                "UNIREC_VISION_SWEEP_POINT_BEGIN "
                f"bucket={spec.key}",
                flush=True,
            )
            pixels = torch.zeros(
                (spec.batch_size, 3, spec.height, spec.width),
                dtype=runner.dtype,
                device=device,
            )
            host_masks = _make_host_masks(
                [(spec.width, spec.height)] * spec.batch_size,
                spec=spec,
            )
            masks = tuple(torch.from_numpy(mask).to(device) for mask in host_masks)
            module = runtime.modules[spec.key]
            compiled = runtime.compiled[spec.key]
            run = lambda: compiled(pixels, *masks)

            first_started = time.perf_counter()
            compiled_output = run()
            synchronize_device(device)
            first_call_wall_s = time.perf_counter() - first_started
            compiled_output_cpu = compiled_output.detach().cpu()

            eager_output = module(pixels, *masks)
            synchronize_device(device)
            eager_output_cpu = eager_output.detach().cpu()
            compiled_vs_eager = _difference(compiled_output_cpu, eager_output_cpu)
            if not compiled_vs_eager["allclose_atol_5e_2_rtol_5e_2"]:
                warning_count += 1
                print(
                    "UNIREC_VISION_SWEEP_WARNING "
                    f"bucket={spec.key} comparison=compiled_vs_eager "
                    f"max_abs={compiled_vs_eager['max_abs']:.9g} "
                    f"mean_abs={compiled_vs_eager['mean_abs']:.9g}",
                    flush=True,
                )

            row0 = compiled_output_cpu[0:1].contiguous()
            if b1_output is None:
                b1_output = row0
                row0_vs_b1 = {
                    "reference": "self",
                    "allclose_atol_5e_2_rtol_5e_2": True,
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                }
            else:
                row0_vs_b1 = {
                    "reference": f"{args.width}x{args.height}_b1.row0",
                    **_difference(row0, b1_output),
                }
                if not row0_vs_b1["allclose_atol_5e_2_rtol_5e_2"]:
                    warning_count += 1
                    print(
                        "UNIREC_VISION_SWEEP_WARNING "
                        f"bucket={spec.key} comparison=row0_vs_b1 "
                        f"max_abs={row0_vs_b1['max_abs']:.9g} "
                        f"mean_abs={row0_vs_b1['mean_abs']:.9g}",
                        flush=True,
                    )

            for _ in range(args.warmups):
                run()
            synchronize_device(device)
            torch.npu.reset_peak_memory_stats(device)
            allocated_before = int(torch.npu.memory_allocated(device))
            samples = [_measure_ms(run) for _ in range(args.repeats)]
            synchronize_device(device)
            allocated_after = int(torch.npu.memory_allocated(device))
            peak_allocated = int(torch.npu.max_memory_allocated(device))
            timing = _summary(samples)
            physical_pixels = spec.batch_size * spec.width * spec.height
            tokens_per_crop = (spec.width // 32) * (spec.height // 32)
            timing.update(
                {
                    "per_crop_median_ms": timing["median_ms"] / spec.batch_size,
                    "crops_per_s": spec.batch_size * 1000.0 / timing["median_ms"],
                    "mpix_per_s": physical_pixels / timing["median_ms"] / 1000.0,
                }
            )
            row = {
                "bucket": spec.key,
                "width": spec.width,
                "height": spec.height,
                "batch_size": spec.batch_size,
                "physical_pixels": physical_pixels,
                "tokens_per_crop": tokens_per_crop,
                "physical_input_shape": [
                    spec.batch_size,
                    3,
                    spec.height,
                    spec.width,
                ],
                "physical_output_shape": list(compiled_output_cpu.shape),
                "first_call_wall_s": first_call_wall_s,
                "timing": timing,
                "memory": {
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_increment_bytes": max(0, peak_allocated - allocated_before),
                },
                "warnings": {
                    "compiled_vs_eager": compiled_vs_eager,
                    "compiled_row0_vs_b1": row0_vs_b1,
                },
            }
            rows.append(row)
            print(
                "UNIREC_VISION_SWEEP_POINT_END "
                f"bucket={spec.key} median_ms={timing['median_ms']:.6f} "
                f"crops_s={timing['crops_per_s']:.6f} "
                f"mpix_s={timing['mpix_per_s']:.6f} "
                f"warning_count={warning_count} "
                f"peak_increment_bytes={row['memory']['peak_increment_bytes']}",
                flush=True,
            )
            del eager_output, eager_output_cpu, compiled_output, compiled_output_cpu
            del row0, pixels, masks, host_masks

    baseline = rows[0]
    baseline_ms = float(baseline["timing"]["median_ms"])
    for row in rows:
        batch = int(row["batch_size"])
        median_ms = float(row["timing"]["median_ms"])
        row["scaling_vs_b1"] = {
            "throughput_speedup": batch * baseline_ms / median_ms,
            "batch_efficiency": baseline_ms / median_ms,
            "latency_ratio": median_ms / baseline_ms,
        }

    report = {
        "schema": "unirec_vision_compiled_shape_batch_sweep_v1",
        "status": "ok_with_warnings" if warning_count else "ok",
        "correctness_policy": "warning_only",
        "warning_count": warning_count,
        "device": args.device,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "shape": [args.width, args.height],
        "batch_sizes": batches,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "focal_depthwise_rewrite": args.focal_depthwise_rewrite,
        "weight_format": args.weight_format,
        "module_identity": args.module_identity,
        "runtime_init_wall_s": runtime_init_wall_s,
        "compile_api": runtime.compile_api,
        "cache_inventory": runtime.cache_inventory(),
        "measurement_scope": (
            "synchronized NPU events around optimized compiled full vision "
            "encoder graph compute only; no H2D, layout, text prefill, or decode"
        ),
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_VISION_SWEEP_SUMMARY "
        f"status={report['status']} shape={args.width}x{args.height} "
        f"points={len(rows)} warnings={warning_count} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
