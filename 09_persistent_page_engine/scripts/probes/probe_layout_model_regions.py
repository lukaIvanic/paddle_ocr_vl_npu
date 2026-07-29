#!/usr/bin/env python3
"""Measure the fixed-shape owned layout model's major NPU regions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.layout_frontend import OwnedLayoutFrontend, _decode_rgb


DEFAULT_IMAGE = Path(
    "/workspace/datasets/OmniDocBench/images/"
    "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
)
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=DEFAULT_LAYOUT_MODEL,
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _summary(milliseconds: list[float]) -> dict[str, float | int]:
    seconds = [value / 1000.0 for value in milliseconds]
    return {
        "calls": len(seconds),
        "mean_s": statistics.fmean(seconds),
        "median_s": statistics.median(seconds),
        "min_s": min(seconds),
        "max_s": max(seconds),
        "total_s": sum(seconds),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")

    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("layout region probe requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    model_dir = args.layout_model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        graph_capture=False,
        model_backend="owned",
    )
    image_rgb, _decode_timing = _decode_rgb(image_path)
    pixel_values = frontend._prepare_pixel_values(image_rgb)

    for _ in range(args.warmup_iters):
        frontend.model(pixel_values=pixel_values)
    torch_npu.npu.synchronize(device)

    core = frontend.model.model
    regions = {
        "detector_core": core,
        "backbone": core.backbone,
        "hybrid_encoder": core.encoder,
        "aifi_transformer": core.encoder.encoder[0],
        "decoder": core.decoder,
    }
    event_pairs: defaultdict[str, list[tuple[Any, Any]]] = defaultdict(list)
    handles = []

    for name, module in regions.items():
        starts: list[Any] = []

        def before(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            *,
            region_name: str = name,
            pending_starts: list[Any] = starts,
        ) -> None:
            event = torch_npu.npu.Event(enable_timing=True)
            event.record()
            pending_starts.append(event)

        def after(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            _output: Any,
            *,
            region_name: str = name,
            pending_starts: list[Any] = starts,
        ) -> None:
            if not pending_starts:
                raise RuntimeError(
                    f"missing start event for layout region {region_name}"
                )
            event = torch_npu.npu.Event(enable_timing=True)
            event.record()
            event_pairs[region_name].append(
                (pending_starts.pop(), event)
            )

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))

    full_pairs: list[tuple[Any, Any]] = []
    try:
        for _ in range(args.repeats):
            start = torch_npu.npu.Event(enable_timing=True)
            end = torch_npu.npu.Event(enable_timing=True)
            start.record()
            frontend.model(pixel_values=pixel_values)
            end.record()
            full_pairs.append((start, end))
        full_pairs[-1][1].synchronize()
    finally:
        for handle in handles:
            handle.remove()

    timings_ms = {
        "full_wrapper": [
            float(start.elapsed_time(end)) for start, end in full_pairs
        ],
        **{
            name: [
                float(start.elapsed_time(end))
                for start, end in pairs
            ]
            for name, pairs in event_pairs.items()
        },
    }
    summaries = {
        name: _summary(values) for name, values in timings_ms.items()
    }
    full_mean = float(summaries["full_wrapper"]["mean_s"])
    for name, summary in summaries.items():
        summary["share_of_full"] = (
            float(summary["mean_s"]) / full_mean
            if full_mean
            else 0.0
        )
    result = {
        "model": str(model_dir),
        "image": str(image_path),
        "input_shape": list(pixel_values.shape),
        "input_dtype": str(pixel_values.dtype),
        "warmup_iters": args.warmup_iters,
        "repeats": args.repeats,
        "regions": summaries,
        "notes": {
            "hybrid_encoder_includes_aifi": True,
            "decoder_layers": int(frontend.model.config.decoder_layers),
            "nested_regions_are_not_additive": True,
        },
    }
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
