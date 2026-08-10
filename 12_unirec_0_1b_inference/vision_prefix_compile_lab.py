#!/usr/bin/env python3
"""Compare eager and per-shape compiled UniRec vision-prefix execution."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)
from vision_atlas import UniRecVisionAtlasRuntime  # noqa: E402
from vision_prefix_crop_lab import (  # noqa: E402
    _load_selected_rows,
    _reconstruct_crops,
)
from vision_static_shape import StaticShapeUniRecVisionRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    if len(set(args.request_id)) != len(args.request_id):
        parser.error("--request-id values must be unique")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the compile lab")
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


def _measure_ms(fn: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    selected_rows = _load_selected_rows(
        args.crop_manifest.expanduser().resolve(),
        args.request_id,
    )
    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=selected_rows,
        crop_margin=crop_margin,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )
    eager_runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    compiled_runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    eager_runtime = UniRecVisionAtlasRuntime(eager_runner)

    eager_prepared: dict[
        str,
        tuple[dict[str, torch.Tensor], dict[str, Any]],
    ] = {}
    compiled_prepared: dict[
        str,
        tuple[dict[str, torch.Tensor], dict[str, Any]],
    ] = {}
    rows_by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in selected_rows:
        request_id = str(row["request_id"])
        eager_prepared[request_id] = eager_runner.prepare_pil_image(
            images[request_id],
            image_source=request_id,
        )
        compiled_prepared[request_id] = compiled_runner.prepare_pil_image(
            images[request_id],
            image_source=request_id,
        )
        actual_size = eager_prepared[request_id][1]["processed_image_size"]
        expected_size = row["prefill"]["prep"]["processed_image_size"]
        if actual_size != expected_size:
            raise RuntimeError(
                f"processed-size mismatch for {request_id}: "
                f"{actual_size} != {expected_size}"
            )
        shape = (int(expected_size[0]), int(expected_size[1]))
        rows_by_shape.setdefault(shape, []).append(row)

    shape_results = []
    with torch.inference_mode():
        for (width, height), rows in rows_by_shape.items():
            compiled_runtime = StaticShapeUniRecVisionRuntime(
                compiled_runner,
                input_width=width,
                input_height=height,
            )
            first_request_id = str(rows[0]["request_id"])
            first_pixels = compiled_prepared[first_request_id][0]["pixel_values"]

            # The first call includes graph compilation or cache loading and is
            # deliberately excluded from steady-state timings.
            first_call_started = time.perf_counter()
            compiled_first = compiled_runtime._run_prefix(first_pixels)[0]
            synchronize_device("npu:0")
            first_call_wall_s = time.perf_counter() - first_call_started

            correctness = []
            for row in rows:
                request_id = str(row["request_id"])
                eager_pixels = eager_prepared[request_id][0]["pixel_values"]
                compiled_pixels = compiled_prepared[request_id][0]["pixel_values"]
                eager_output = eager_runtime._run_prefix(eager_pixels)[0]
                compiled_output = compiled_runtime._run_prefix(compiled_pixels)[0]
                synchronize_device("npu:0")
                difference = (eager_output - compiled_output).abs()
                correctness.append(
                    {
                        "request_id": request_id,
                        "allclose_atol_5e_2_rtol_5e_2": bool(
                            torch.allclose(
                                eager_output,
                                compiled_output,
                                atol=5e-2,
                                rtol=5e-2,
                            )
                        ),
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                    }
                )

            del compiled_first
            for _ in range(args.warmup):
                for row in rows:
                    request_id = str(row["request_id"])
                    eager_runtime._run_prefix(
                        eager_prepared[request_id][0]["pixel_values"]
                    )
                    compiled_runtime._run_prefix(
                        compiled_prepared[request_id][0]["pixel_values"]
                    )
            synchronize_device("npu:0")

            per_crop = []
            all_eager_ms: list[float] = []
            all_compiled_ms: list[float] = []
            for row in rows:
                request_id = str(row["request_id"])
                eager_pixels = eager_prepared[request_id][0]["pixel_values"]
                compiled_pixels = compiled_prepared[request_id][0]["pixel_values"]
                eager_ms: list[float] = []
                compiled_ms: list[float] = []
                for repeat in range(args.repeats):
                    lanes = (
                        (
                            "eager",
                            lambda: eager_runtime._run_prefix(eager_pixels)[0],
                        ),
                        (
                            "compiled",
                            lambda: compiled_runtime._run_prefix(compiled_pixels)[0],
                        ),
                    )
                    if repeat % 2:
                        lanes = tuple(reversed(lanes))
                    for name, fn in lanes:
                        elapsed_ms, _output = _measure_ms(fn)
                        if name == "eager":
                            eager_ms.append(elapsed_ms)
                        else:
                            compiled_ms.append(elapsed_ms)
                eager_summary = _summary_ms(eager_ms)
                compiled_summary = _summary_ms(compiled_ms)
                all_eager_ms.extend(eager_ms)
                all_compiled_ms.extend(compiled_ms)
                per_crop.append(
                    {
                        "request_id": request_id,
                        "label": row["label"],
                        "original_size": row["prefill"]["prep"]["original_image_size"],
                        "encoder_tokens": int(row["cross_kv"]["source_length"]),
                        "eager_ms": eager_summary,
                        "compiled_ms": compiled_summary,
                        "p50_speedup": (
                            eager_summary["p50"] / compiled_summary["p50"]
                        ),
                    }
                )

            eager_shape_summary = _summary_ms(all_eager_ms)
            compiled_shape_summary = _summary_ms(all_compiled_ms)
            shape_results.append(
                {
                    "processed_size": [width, height],
                    "crop_count": len(rows),
                    "first_call_wall_s": first_call_wall_s,
                    "prefix_cache_dir": str(compiled_runtime.prefix_cache_dir),
                    "correctness": correctness,
                    "eager_ms": eager_shape_summary,
                    "compiled_ms": compiled_shape_summary,
                    "p50_speedup": (
                        eager_shape_summary["p50"]
                        / compiled_shape_summary["p50"]
                    ),
                    "per_crop": per_crop,
                }
            )
            del compiled_runtime

    if not all(
        item["allclose_atol_5e_2_rtol_5e_2"]
        for shape in shape_results
        for item in shape["correctness"]
    ):
        raise RuntimeError("compiled prefix failed eager-output validation")

    report = {
        "status": "ok",
        "physical_devices": physical_devices,
        "worker_count": 1,
        "execution": {
            "eager": "crop-local Python/PyTorch NPU",
            "compiled": "one static TorchAir graph per processed HxW",
        },
        "warmup": args.warmup,
        "repeats_per_crop": args.repeats,
        "timing": "one synchronized NPU event interval per full prefix call",
        "shapes": shape_results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_PREFIX_COMPILE_LAB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
