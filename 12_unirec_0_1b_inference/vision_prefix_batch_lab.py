#!/usr/bin/env python3
"""Benchmark same-shape batching for the compiled UniRec vision prefix."""

from __future__ import annotations

import argparse
import hashlib
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
    import_torchair_cache_compile,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_atlas import UniRecVisionAtlasRuntime  # noqa: E402
from vision_prefix_crop_lab import _reconstruct_crops  # noqa: E402
from vision_static_shape import _StaticVisionPrefix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--processed-shape", action="append", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the batch lab")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _parse_shapes(values: list[str]) -> list[tuple[int, int]]:
    shapes = []
    for value in values:
        try:
            width, height = (int(part) for part in value.lower().split("x", 1))
        except (TypeError, ValueError) as exception:
            raise ValueError(f"invalid processed shape {value!r}; expected WIDTHxHEIGHT") from exception
        if width < 1 or height < 1:
            raise ValueError(f"processed shape must be positive: {value!r}")
        shapes.append((width, height))
    if len(set(shapes)) != len(shapes):
        raise ValueError("processed shapes must be unique")
    return shapes


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


def _new_batched_module(
    runner: OptimizedUniRecRunner,
    *,
    width: int,
    height: int,
    batch_size: int,
) -> _StaticVisionPrefix:
    filename = f"<unirec_static_prefix_b{batch_size}_{width}x{height}>"
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "def forward(self, pixel_values):\n"
            "    return self._forward_fixed(pixel_values)\n",
            filename,
            "exec",
        ),
        namespace,
    )
    module_type = type(
        f"StaticVisionPrefix_b{batch_size}_{width}x{height}",
        (_StaticVisionPrefix,),
        {"forward": namespace["forward"]},
    )
    return module_type(
        runner,
        input_height=height,
        input_width=width,
    ).eval()


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    shapes = _parse_shapes(args.processed_shape)
    batch_sizes = sorted({int(value) for value in args.batch_sizes.split(",")})
    if not batch_sizes or batch_sizes[0] < 1:
        raise ValueError("batch sizes must be positive")

    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    rows = read_jsonl(args.crop_manifest.expanduser().resolve())
    selected_by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    maximum_batch = max(batch_sizes)
    for shape in shapes:
        matching = [
            row
            for row in rows
            if tuple(row["prefill"]["prep"]["processed_image_size"]) == shape
        ]
        if not matching:
            raise RuntimeError(f"crop manifest contains no {shape[0]}x{shape[1]} crops")
        selected_by_shape[shape] = matching[:maximum_batch]

    selected_rows = [
        row for shape in shapes for row in selected_by_shape[shape]
    ]
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
    cache_compile, compile_api = import_torchair_cache_compile()
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    shape_reports = []
    with torch.inference_mode():
        for width, height in shapes:
            rows_for_shape = selected_by_shape[(width, height)]
            prepared = []
            for row in rows_for_shape:
                request_id = str(row["request_id"])
                inputs, metadata = compiled_runner.prepare_pil_image(
                    images[request_id],
                    image_source=request_id,
                )
                if tuple(metadata["processed_image_size"]) != (width, height):
                    raise RuntimeError(
                        f"processed-size mismatch for {request_id}: "
                        f"{metadata['processed_image_size']} != {(width, height)}"
                    )
                prepared.append(inputs["pixel_values"])

            batch_reports = []
            for batch_size in batch_sizes:
                if len(prepared) < batch_size:
                    continue
                compiled_inputs = torch.cat(prepared[:batch_size], dim=0)
                eager_inputs = compiled_inputs.clone()
                module = _new_batched_module(
                    compiled_runner,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                )
                graph_cache_dir = args.cache_dir.expanduser().resolve() / (
                    f"vision_static_prefix_batch_b{batch_size}_{width}x{height}_"
                    f"float16_src{source_hash}"
                )
                graph_cache_dir.mkdir(parents=True, exist_ok=True)
                compiled = cache_compile(
                    module.forward,
                    config=config,
                    dynamic=False,
                    cache_dir=str(graph_cache_dir),
                    ge_cache=True,
                    fullgraph=True,
                )

                first_call_started = time.perf_counter()
                compiled_output = compiled(compiled_inputs)
                synchronize_device("npu:0")
                first_call_wall_s = time.perf_counter() - first_call_started
                eager_output = eager_runtime._run_prefix(eager_inputs)[0]
                synchronize_device("npu:0")
                difference = (compiled_output - eager_output).abs()
                correctness = {
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(
                            compiled_output,
                            eager_output,
                            atol=5e-2,
                            rtol=5e-2,
                        )
                    ),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }

                for _ in range(args.warmup):
                    eager_runtime._run_prefix(eager_inputs)
                    compiled(compiled_inputs)
                synchronize_device("npu:0")

                eager_ms = []
                compiled_ms = []
                for repeat_index in range(args.repeats):
                    lanes = (
                        ("eager", lambda: eager_runtime._run_prefix(eager_inputs)[0]),
                        ("compiled", lambda: compiled(compiled_inputs)),
                    )
                    if repeat_index % 2:
                        lanes = tuple(reversed(lanes))
                    for name, fn in lanes:
                        elapsed_ms, _ = _measure_ms(fn)
                        (eager_ms if name == "eager" else compiled_ms).append(
                            elapsed_ms
                        )
                eager_summary = _summary_ms(eager_ms)
                compiled_summary = _summary_ms(compiled_ms)
                batch_reports.append(
                    {
                        "batch_size": batch_size,
                        "real_crop_count": batch_size,
                        "first_call_wall_s": first_call_wall_s,
                        "cache_dir": str(graph_cache_dir),
                        "correctness": correctness,
                        "eager_ms": eager_summary,
                        "compiled_ms": compiled_summary,
                        "compiled_crops_per_s": (
                            batch_size * 1000.0 / compiled_summary["p50"]
                        ),
                        "compiled_per_crop_ms": (
                            compiled_summary["p50"] / batch_size
                        ),
                        "compiled_vs_eager_speedup": (
                            eager_summary["p50"] / compiled_summary["p50"]
                        ),
                    }
                )
            baseline = next(
                item for item in batch_reports if item["batch_size"] == 1
            )
            baseline_ms = float(baseline["compiled_ms"]["p50"])
            for item in batch_reports:
                item["speedup_vs_sequential_compiled_b1"] = (
                    int(item["batch_size"])
                    * baseline_ms
                    / float(item["compiled_ms"]["p50"])
                )
            shape_reports.append(
                {
                    "processed_size": [width, height],
                    "available_real_crops": len(rows_for_shape),
                    "batches": batch_reports,
                }
            )

    if not all(
        batch["correctness"]["allclose_atol_5e_2_rtol_5e_2"]
        for shape in shape_reports
        for batch in shape["batches"]
    ):
        raise RuntimeError("a compiled batched prefix failed eager validation")

    report = {
        "status": "ok",
        "physical_devices": physical_devices,
        "compile_api": compile_api,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "synchronized NPU events around one complete prefix batch",
        "shapes": shape_reports,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_PREFIX_BATCH_LAB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
