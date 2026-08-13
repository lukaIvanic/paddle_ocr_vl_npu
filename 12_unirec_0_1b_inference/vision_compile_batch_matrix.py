#!/usr/bin/env python3
"""Compare stock and bucket eager/compiled full vision batches."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
    import_torchair_cache_compile,
    synchronize_device,
)
from vision_full_batch import (  # noqa: E402
    BucketedFullVisionRuntime,
    VISION_WEIGHT_FORMAT_CHOICES,
    VisionBucketSpec,
    _make_host_masks,
)
from vision_focal_depthwise import (  # noqa: E402
    VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--stock-only",
        action="store_true",
        help="Compile and time only stock eager/compiled; skip all bucket graphs.",
    )
    parser.add_argument(
        "--focal-depthwise-rewrite",
        choices=VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES,
        default="native",
    )
    parser.add_argument(
        "--weight-format",
        choices=VISION_WEIGHT_FORMAT_CHOICES,
        default="native",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if args.width % 32 or args.height % 32:
        parser.error("width and height must be divisible by 32")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats positive")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the vision matrix")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(device in {5, 6} for device in devices):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible physical NPU, got {devices}")
    return devices


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or sizes[0] < 1:
        raise ValueError("batch sizes must be positive")
    return sizes


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


def _measure_ms(fn: Callable[[], torch.Tensor]) -> float:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    fn()
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


class _StockVisionEncoder(torch.nn.Module):
    """Expose the unmodified UniRec encoder as one fixed-shape graph."""

    def __init__(self, runner: OptimizedUniRecRunner) -> None:
        super().__init__()
        self.encoder = runner.model.encoder

    def _forward_fixed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._forward_fixed(pixel_values)


def _new_stock_encoder_module(
    runner: OptimizedUniRecRunner,
    spec: VisionBucketSpec,
) -> _StockVisionEncoder:
    """Give every stock fixed batch a deterministic graph identity."""
    filename = f"<unirec_stock_vision_{spec.key}>"
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
        f"StockVisionEncoder_{spec.width}x{spec.height}_b{spec.batch_size}",
        (_StockVisionEncoder,),
        {"forward": namespace["forward"]},
    )
    return module_type(runner).eval()


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    if not args.device.startswith("npu"):
        raise ValueError("the vision matrix requires an NPU device")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=cache_dir,
    )
    specs = tuple(
        VisionBucketSpec(
            width=args.width,
            height=args.height,
            batch_size=batch_size,
        )
        for batch_size in batch_sizes
    )
    runtime = None
    if not args.stock_only:
        runtime = BucketedFullVisionRuntime(
            runner,
            specs=specs,
            focal_depthwise_rewrite=args.focal_depthwise_rewrite,
            weight_format=args.weight_format,
        )
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    stock_config = CompilerConfig()
    stock_config.mode.value = "max-autotune"
    cache_compile, stock_compile_api = import_torchair_cache_compile()
    stock_source_hash = hashlib.sha256(
        inspect.getsource(_StockVisionEncoder).encode("utf-8")
        + inspect.getsource(_new_stock_encoder_module).encode("utf-8")
        + Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    ).hexdigest()[:12]

    rows = []
    with torch.inference_mode():
        for spec in specs:
            pixels = torch.zeros(
                (spec.batch_size, 3, spec.height, spec.width),
                device=args.device,
                dtype=runner.dtype,
            )
            stock_module = _new_stock_encoder_module(runner, spec)
            stock_cache_dir = cache_dir / (
                f"vision_stock_fixed_{spec.key}_float16_src{stock_source_hash}"
            )
            stock_cache_dir.mkdir(parents=True, exist_ok=True)
            stock_compiled = cache_compile(
                stock_module.forward,
                config=stock_config,
                dynamic=False,
                cache_dir=str(stock_cache_dir),
                ge_cache=True,
                fullgraph=True,
            )

            lane_list: list[tuple[str, Callable[[], torch.Tensor]]] = [
                ("stock_eager", lambda: runner.model.forward_encoder(pixels)),
                ("stock_compiled", lambda: stock_compiled(pixels)),
            ]
            if runtime is not None:
                host_masks = _make_host_masks(
                    [(spec.width, spec.height)] * spec.batch_size,
                    spec=spec,
                )
                masks = tuple(
                    torch.from_numpy(mask).to(args.device) for mask in host_masks
                )
                module = runtime.modules[spec.key]
                compiled = runtime.compiled[spec.key]
                lane_list.extend(
                    (
                        ("bucket_module_eager", lambda: module(pixels, *masks)),
                        ("compiled", lambda: compiled(pixels, *masks)),
                    )
                )
            lanes = tuple(lane_list)

            bucket_compiled_first_call_wall_s = None
            compiled_output = None
            if runtime is not None:
                bucket_first_call_started = time.perf_counter()
                compiled_output = compiled(pixels, *masks)
                synchronize_device(args.device)
                bucket_compiled_first_call_wall_s = (
                    time.perf_counter() - bucket_first_call_started
                )
            stock_first_call_started = time.perf_counter()
            stock_compiled_output = stock_compiled(pixels)
            synchronize_device(args.device)
            stock_compiled_first_call_wall_s = (
                time.perf_counter() - stock_first_call_started
            )
            stock_output = lanes[0][1]()
            synchronize_device(args.device)
            correctness = {
                "stock_compiled_vs_stock_eager": _difference(
                    stock_compiled_output, stock_output
                ),
            }
            if runtime is not None:
                bucket_eager_output = lanes[2][1]()
                synchronize_device(args.device)
                correctness.update(
                    {
                        "compiled_vs_bucket_module_eager": _difference(
                            compiled_output, bucket_eager_output
                        ),
                        "bucket_module_eager_vs_stock_eager": _difference(
                            bucket_eager_output, stock_output
                        ),
                    }
                )

            for _ in range(args.warmups):
                for _name, function in lanes:
                    function()
            synchronize_device(args.device)

            samples = {name: [] for name, _function in lanes}
            for repeat_index in range(args.repeats):
                ordered = lanes if repeat_index % 2 == 0 else tuple(reversed(lanes))
                for name, function in ordered:
                    samples[name].append(_measure_ms(function))
            timing = {name: _summary(values) for name, values in samples.items()}
            for name, summary in timing.items():
                summary["per_crop_median_ms"] = (
                    summary["median_ms"] / spec.batch_size
                )
                summary["crops_per_s"] = (
                    spec.batch_size * 1000.0 / summary["median_ms"]
                )
            row = {
                "batch_size": spec.batch_size,
                "shape": [spec.width, spec.height],
                "stock_compiled_first_call_wall_s": (
                    stock_compiled_first_call_wall_s
                ),
                "stock_compiled_cache_dir": str(stock_cache_dir),
                "bucket_compiled_first_call_wall_s": (
                    bucket_compiled_first_call_wall_s
                ),
                "correctness": correctness,
                "timing": timing,
                "speedup": {
                    "stock_compiled_vs_stock_eager": (
                        timing["stock_eager"]["median_ms"]
                        / timing["stock_compiled"]["median_ms"]
                    ),
                },
            }
            if runtime is not None:
                row["speedup"].update(
                    {
                        "compiled_vs_bucket_module_eager": (
                            timing["bucket_module_eager"]["median_ms"]
                            / timing["compiled"]["median_ms"]
                        ),
                        "compiled_vs_stock_eager": (
                            timing["stock_eager"]["median_ms"]
                            / timing["compiled"]["median_ms"]
                        ),
                    }
                )
            rows.append(row)
            message = (
                "UNIREC_VISION_COMPILE_BATCH "
                f"batch={spec.batch_size} "
                f"stock_eager_ms={timing['stock_eager']['median_ms']:.3f} "
                f"stock_compiled_ms={timing['stock_compiled']['median_ms']:.3f} "
                f"stock_compile_speedup={row['speedup']['stock_compiled_vs_stock_eager']:.3f} "
                f"stock_compiled_crops_s={timing['stock_compiled']['crops_per_s']:.2f}"
            )
            if runtime is not None:
                message += (
                    f" bucket_eager_ms={timing['bucket_module_eager']['median_ms']:.3f}"
                    f" compiled_ms={timing['compiled']['median_ms']:.3f}"
                    f" compile_speedup={row['speedup']['compiled_vs_bucket_module_eager']:.3f}"
                    f" compiled_crops_s={timing['compiled']['crops_per_s']:.2f}"
                )
            print(message, flush=True)

    baseline = rows[0]
    for row in rows:
        batch_size = int(row["batch_size"])
        row["throughput_speedup_vs_b1"] = {
            lane: (
                batch_size
                * baseline["timing"][lane]["median_ms"]
                / row["timing"][lane]["median_ms"]
            )
            for lane in row["timing"]
        }

    allclose = all(
        comparison["allclose_atol_5e_2_rtol_5e_2"]
        for row in rows
        for comparison in row["correctness"].values()
    )
    report = {
        "status": "ok" if allclose else "correctness_failed",
        "device": args.device,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": physical_devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "shape": [args.width, args.height],
        "batch_sizes": batch_sizes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "focal_depthwise_rewrite": args.focal_depthwise_rewrite,
        "weight_format": args.weight_format,
        "stock_only": args.stock_only,
        "compile_api": runtime.compile_api if runtime is not None else None,
        "stock_compile_api": stock_compile_api,
        "cache_inventory": (
            runtime.cache_inventory() if runtime is not None else None
        ),
        "rows": rows,
        "measurement_scope": (
            "synchronized NPU events around vision encoder compute only; "
            "no H2D, preprocessing, layout, text prefill, or decode"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_VISION_COMPILE_BATCH_SUMMARY "
        f"status={report['status']} device={report['device_name']} "
        f"shape={args.width}x{args.height} batches={batch_sizes}",
        flush=True,
    )
    print(f"OUTPUT_JSON={output}", flush=True)
    if not allclose:
        raise RuntimeError("vision compile batch matrix failed correctness")


if __name__ == "__main__":
    main()
