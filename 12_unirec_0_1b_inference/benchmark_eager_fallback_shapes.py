#!/usr/bin/env python3
"""Benchmark production UniRec eager-fallback vision shapes on one NPU.

The crop manifest supplies the exact processed-shape distribution from a
completed production run.  Pixel contents do not affect the operator shapes,
so the timed lane uses reusable synthetic compact uint8 HWC inputs and the
same H2D/normalization helper plus ``model.forward_encoder`` call as
``BucketedFullVisionRuntime._run_fallback``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=12,
        help="Top shapes ranked by fallback count times processed pixel area; 0 means all.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Write the selected workload without loading torch or the model.",
    )
    args = parser.parse_args()
    if args.max_shapes < 0:
        parser.error("--max-shapes must be non-negative")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats positive")
    return args


def fallback_shape_histogram(path: Path) -> Counter[tuple[int, int]]:
    histogram: Counter[tuple[int, int]] = Counter()
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            prefill = row.get("prefill") or {}
            if prefill.get("vision_bucket") is not None:
                continue
            size = (prefill.get("prep") or {}).get("processed_image_size")
            if not isinstance(size, list) or len(size) != 2:
                raise ValueError(
                    f"fallback row {line_number} has no processed_image_size"
                )
            width, height = (int(size[0]), int(size[1]))
            if width < 1 or height < 1:
                raise ValueError(f"invalid fallback shape at row {line_number}: {size}")
            histogram[(width, height)] += 1
    if not histogram:
        raise ValueError(f"no eager fallback rows found in {path}")
    return histogram


def select_shapes(
    histogram: Counter[tuple[int, int]],
    max_shapes: int,
) -> list[tuple[int, int]]:
    ranked = sorted(
        histogram,
        key=lambda shape: (
            -(histogram[shape] * shape[0] * shape[1]),
            -histogram[shape],
            -shape[0] * shape[1],
            shape,
        ),
    )
    return ranked if max_shapes == 0 else ranked[:max_shapes]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def workload_report(
    histogram: Counter[tuple[int, int]],
    selected: list[tuple[int, int]],
) -> dict[str, Any]:
    total_calls = sum(histogram.values())
    total_weighted_pixels = sum(
        count * width * height for (width, height), count in histogram.items()
    )
    selected_calls = sum(histogram[shape] for shape in selected)
    selected_weighted_pixels = sum(
        histogram[(width, height)] * width * height for width, height in selected
    )
    return {
        "fallback_calls": total_calls,
        "unique_shapes": len(histogram),
        "selected_shape_count": len(selected),
        "selected_calls": selected_calls,
        "selected_call_fraction": selected_calls / total_calls,
        "selected_weighted_pixel_fraction": (
            selected_weighted_pixels / total_weighted_pixels
        ),
        "selected_shapes": [
            {
                "width": width,
                "height": height,
                "count": histogram[(width, height)],
                "pixels": width * height,
                "weighted_pixels": histogram[(width, height)] * width * height,
            }
            for width, height in selected
        ],
    }


def main() -> None:
    args = parse_args()
    histogram = fallback_shape_histogram(args.crop_manifest)
    selected = select_shapes(histogram, args.max_shapes)
    workload = workload_report(histogram, selected)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        "UNIREC_EAGER_FALLBACK_WORKLOAD "
        f"calls={workload['fallback_calls']} unique={workload['unique_shapes']} "
        f"selected={workload['selected_shape_count']} "
        f"call_coverage={workload['selected_call_fraction']:.3f} "
        f"pixel_coverage={workload['selected_weighted_pixel_fraction']:.3f}",
        flush=True,
    )
    if args.list_only:
        report = {"status": "listed", "workload": workload}
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return

    import torch
    import torch_npu

    from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
    from vision_full_batch import _compact_uint8_hwc_to_device

    if not args.device.startswith("npu"):
        raise ValueError("this benchmark requires an NPU device")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device_name = torch.npu.get_device_name(0)
    setup_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )
    setup_s = time.perf_counter() - setup_started
    rows = []
    benchmark_started = time.perf_counter()
    for shape_index, (width, height) in enumerate(selected, start=1):
        host_pixels = np.zeros((height, width, 3), dtype=np.uint8)

        def run_once() -> tuple[float, float, float, list[int]]:
            transfer_start = torch_npu.npu.Event(enable_timing=True)
            transfer_end = torch_npu.npu.Event(enable_timing=True)
            encoder_start = torch_npu.npu.Event(enable_timing=True)
            encoder_end = torch_npu.npu.Event(enable_timing=True)
            synchronize_device(args.device)
            wall_started = time.perf_counter()
            transfer_start.record()
            pixel_values = _compact_uint8_hwc_to_device(
                host_pixels,
                device=args.device,
                dtype=runner.dtype,
            )
            transfer_end.record()
            encoder_start.record()
            with torch.inference_mode():
                hidden = runner.model.forward_encoder(pixel_values)
            encoder_end.record()
            encoder_end.synchronize()
            return (
                (time.perf_counter() - wall_started) * 1000.0,
                float(transfer_start.elapsed_time(transfer_end)),
                float(encoder_start.elapsed_time(encoder_end)),
                [int(value) for value in hidden.shape],
            )

        for _ in range(args.warmups):
            run_once()
        wall_samples = []
        transfer_samples = []
        encoder_samples = []
        output_shape = None
        for _ in range(args.repeats):
            wall_ms, transfer_ms, encoder_ms, output_shape = run_once()
            wall_samples.append(wall_ms)
            transfer_samples.append(transfer_ms)
            encoder_samples.append(encoder_ms)
        row = {
            "width": width,
            "height": height,
            "count": histogram[(width, height)],
            "output_shape": output_shape,
            "production_wall": summarize(wall_samples),
            "compact_h2d_normalize": summarize(transfer_samples),
            "eager_encoder": summarize(encoder_samples),
        }
        rows.append(row)
        print(
            "UNIREC_EAGER_FALLBACK_SHAPE "
            f"index={shape_index}/{len(selected)} shape={width}x{height} "
            f"count={row['count']} wall_p50_ms={row['production_wall']['median_ms']:.3f} "
            f"encoder_p50_ms={row['eager_encoder']['median_ms']:.3f}",
            flush=True,
        )

    selected_weighted_encoder_s = sum(
        row["count"] * row["eager_encoder"]["median_ms"] / 1000.0
        for row in rows
    )
    selected_weighted_wall_s = sum(
        row["count"] * row["production_wall"]["median_ms"] / 1000.0
        for row in rows
    )
    report = {
        "status": "ok",
        "device": args.device,
        "device_name": device_name,
        "dtype": args.dtype,
        "npu_jit_compile": False,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "model_setup_s": setup_s,
        "benchmark_wall_s": time.perf_counter() - benchmark_started,
        "workload": workload,
        "selected_weighted_encoder_s": selected_weighted_encoder_s,
        "selected_weighted_production_wall_s": selected_weighted_wall_s,
        "rows": rows,
        "measurement_scope": {
            "included": [
                "production compact uint8 HWC H2D and NPU normalization",
                "native-shape eager model.forward_encoder",
                "synchronized NPU event timing",
            ],
            "excluded": [
                "crop decode and resize",
                "compiled vision buckets",
                "cross-KV prefill and export",
            ],
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_EAGER_FALLBACK_SUMMARY "
        f"device={report['device_name']} shapes={len(rows)} "
        f"call_coverage={workload['selected_call_fraction']:.3f} "
        f"pixel_coverage={workload['selected_weighted_pixel_fraction']:.3f} "
        f"weighted_encoder_s={selected_weighted_encoder_s:.3f} "
        f"weighted_wall_s={selected_weighted_wall_s:.3f} "
        f"benchmark_wall_s={report['benchmark_wall_s']:.3f}",
        flush=True,
    )
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
