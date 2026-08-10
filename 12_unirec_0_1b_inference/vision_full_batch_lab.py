#!/usr/bin/env python3
"""Validate one full-encoder vision bucket against the current vision path."""

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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_atlas import CropShape, _pack_shapes  # noqa: E402
from vision_full_batch import (  # noqa: E402
    BucketedFullVisionRuntime,
    VisionBucketSpec,
    _make_host_masks,
)
from vision_prefix_crop_lab import _reconstruct_crops  # noqa: E402
from vision_static_shape import (  # noqa: E402
    PerShapeCompiledPrefixUniRecVisionRuntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-limit", type=int, default=32)
    parser.add_argument("--bucket-width", type=int, default=960)
    parser.add_argument("--bucket-height", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.page_limit < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("page limit and repeats must be positive; warmup cannot be negative")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the full-vision lab")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _measure_wall_ms(fn: Callable[[], Any]) -> float:
    synchronize_device("npu:0")
    started = time.perf_counter()
    fn()
    synchronize_device("npu:0")
    return (time.perf_counter() - started) * 1000.0


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    bucket_width: int,
    bucket_height: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    fitting = [
        row
        for row in rows
        if int(row["prefill"]["prep"]["processed_image_size"][0]) <= bucket_width
        and int(row["prefill"]["prep"]["processed_image_size"][1]) <= bucket_height
    ]
    by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in fitting:
        shape = tuple(
            int(value)
            for value in row["prefill"]["prep"]["processed_image_size"]
        )
        by_shape.setdefault(shape, []).append(row)
    selected = [by_shape[shape][0] for shape in sorted(by_shape)]
    if len(selected) > batch_size:
        raise RuntimeError(
            f"bucket contains {len(selected)} shapes but batch size is {batch_size}"
        )
    remaining = [row for row in fitting if row not in selected]
    remaining.sort(key=lambda row: (int(row["page_index"]), int(row["crop_index"])))
    selected.extend(remaining[: batch_size - len(selected)])
    if len(selected) != batch_size:
        raise RuntimeError(f"cannot form a full real batch of {batch_size}")
    return selected


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    spec = VisionBucketSpec(
        width=args.bucket_width,
        height=args.bucket_height,
        batch_size=args.batch_size,
    )
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    page_rows = read_jsonl(args.page_manifest.expanduser().resolve())
    page_indices = {int(row["page_index"]) for row in page_rows[: args.page_limit]}
    rows = [
        row
        for row in read_jsonl(args.crop_manifest.expanduser().resolve())
        if int(row["page_index"]) in page_indices
    ]
    selected_rows = _select_rows(
        rows,
        bucket_width=spec.width,
        bucket_height=spec.height,
        batch_size=spec.batch_size,
    )
    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=selected_rows,
        crop_margin=crop_margin,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    prepared = []
    dimensions = []
    host_pixels = np.zeros(
        (spec.batch_size, 3, spec.height, spec.width),
        dtype=np.float32,
    )
    for row_index, row in enumerate(selected_rows):
        request_id = str(row["request_id"])
        inputs, metadata = runner.prepare_pil_image(
            images[request_id], image_source=request_id
        )
        width, height = (
            int(value)
            for value in row["prefill"]["prep"]["processed_image_size"]
        )
        if metadata["processed_image_size"] != [width, height]:
            raise RuntimeError(f"processed shape mismatch for {request_id}")
        prepared.append(inputs["pixel_values"])
        dimensions.append((width, height))
        cpu_pixels = runner.processor(images[request_id])["pixel_values"].numpy()
        host_pixels[row_index, :, :height, :width] = cpu_pixels[0]

    bucket_runtime = BucketedFullVisionRuntime(runner, specs=(spec,))
    current_runtime = PerShapeCompiledPrefixUniRecVisionRuntime(
        runner,
        shapes=sorted(set(dimensions), key=lambda shape: (shape[1], shape[0])),
    )
    host_masks = _make_host_masks(dimensions, spec=spec)
    with torch.inference_mode(False):
        bucket_pixels = torch.from_numpy(host_pixels).to(
            "npu:0", dtype=runner.dtype
        )
        bucket_masks = tuple(torch.from_numpy(mask).to("npu:0") for mask in host_masks)

    def run_bucket() -> torch.Tensor:
        return bucket_runtime.compiled[spec.key](bucket_pixels, *bucket_masks)

    def run_eager() -> list[torch.Tensor]:
        return [runner.model.forward_encoder(pixels) for pixels in prepared]

    def run_current() -> list[torch.Tensor]:
        prefix_states = {}
        shapes = []
        for source_index, pixels in enumerate(prepared):
            x, height, width = current_runtime._run_prefix(pixels)
            prefix_states[source_index] = x
            shapes.append(CropShape(source_index, height, width))
        packs, overflow = _pack_shapes(shapes)
        atlas_outputs = current_runtime._run_atlas_packs(packs, prefix_states)
        for crop in overflow:
            atlas_outputs[crop.source_index] = current_runtime._run_stage2_eager(
                prefix_states[crop.source_index], crop.height, crop.width
            )
        return [
            current_runtime._run_suffix(
                atlas_outputs[crop.source_index], crop.height, crop.width
            )
            for crop in sorted(shapes, key=lambda item: item.source_index)
        ]

    with torch.inference_mode():
        first_call_started = time.perf_counter()
        bucket_output = run_bucket()
        synchronize_device("npu:0")
        first_call_wall_s = time.perf_counter() - first_call_started
        eager_outputs = run_eager()
        current_outputs = run_current()
        synchronize_device("npu:0")

        bucket_grid = bucket_output.reshape(
            spec.batch_size,
            spec.height // 32,
            spec.width // 32,
            bucket_output.shape[-1],
        )
        correctness_rows = []
        for row_index, ((width, height), eager, current, row) in enumerate(
            zip(dimensions, eager_outputs, current_outputs, selected_rows)
        ):
            actual = bucket_grid[
                row_index : row_index + 1,
                : height // 32,
                : width // 32,
            ].reshape(1, -1, bucket_output.shape[-1]).contiguous()
            eager_difference = (actual - eager).abs()
            current_difference = (actual - current).abs()
            correctness_rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "processed_size": [width, height],
                    "bucket_vs_eager_allclose": bool(
                        torch.allclose(actual, eager, atol=5e-2, rtol=5e-2)
                    ),
                    "bucket_vs_eager_max_abs": float(eager_difference.max().item()),
                    "bucket_vs_current_allclose": bool(
                        torch.allclose(actual, current, atol=5e-2, rtol=5e-2)
                    ),
                    "bucket_vs_current_max_abs": float(
                        current_difference.max().item()
                    ),
                }
            )

        for _ in range(args.warmup):
            run_bucket()
            run_current()
            run_eager()
        synchronize_device("npu:0")
        samples = {"bucket": [], "current": [], "eager": []}
        lanes = (
            ("bucket", run_bucket),
            ("current", run_current),
            ("eager", run_eager),
        )
        for repeat_index in range(args.repeats):
            ordered = lanes if repeat_index % 2 == 0 else tuple(reversed(lanes))
            for name, function in ordered:
                samples[name].append(_measure_wall_ms(function))

    timing_ms = {name: _summary_ms(values) for name, values in samples.items()}
    allclose = all(
        row["bucket_vs_eager_allclose"] and row["bucket_vs_current_allclose"]
        for row in correctness_rows
    )
    report = {
        "status": "ok" if allclose else "correctness_failed",
        "physical_devices": physical_devices,
        "page_limit": args.page_limit,
        "bucket": {
            "key": spec.key,
            "first_call_wall_s": first_call_wall_s,
            "selected_dimensions": [list(value) for value in dimensions],
            "cache_dir": str(bucket_runtime.cache_dirs[spec.key]),
        },
        "correctness": {
            "all_rows_close": allclose,
            "max_abs_vs_eager": max(
                row["bucket_vs_eager_max_abs"] for row in correctness_rows
            ),
            "max_abs_vs_current": max(
                row["bucket_vs_current_max_abs"] for row in correctness_rows
            ),
            "rows": correctness_rows,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_ms": timing_ms,
        "speed": {
            "bucket_vs_current": timing_ms["current"]["p50"]
            / timing_ms["bucket"]["p50"],
            "bucket_vs_eager": timing_ms["eager"]["p50"]
            / timing_ms["bucket"]["p50"],
            "bucket_crops_per_s": spec.batch_size
            * 1000.0
            / timing_ms["bucket"]["p50"],
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_FULL_VISION_BUCKET_LAB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)
    if not allclose:
        raise RuntimeError("full vision bucket failed validation")


if __name__ == "__main__":
    main()
