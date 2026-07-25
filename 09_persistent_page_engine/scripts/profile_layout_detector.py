#!/usr/bin/env python3
"""Profile one steady PP-DocLayoutV3 NPU-graph replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.layout_frontend import (
    LAYOUT_CONV_WEIGHT_FORMATS,
    OwnedLayoutFrontend,
    _decode_rgb,
)


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
    parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument(
        "--conv-weight-format",
        choices=LAYOUT_CONV_WEIGHT_FORMATS,
        default="native",
    )
    return parser.parse_args()


def profiler_config():
    import torch_npu.profiler as npu_prof

    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative")

    import torch_npu
    import torch_npu.profiler as npu_prof

    if not torch.npu.is_available():
        raise RuntimeError("layout detector profiling requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    image_path = args.image.expanduser().resolve()
    model_dir = args.layout_model.expanduser().resolve()
    profile_dir = args.profile_dir.expanduser().resolve()
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        graph_capture=True,
        conv_weight_format=args.conv_weight_format,
    )
    setup_s = time.perf_counter() - setup_started
    image_rgb, decode_timing = _decode_rgb(image_path)
    pixel_values = frontend._prepare_pixel_values(image_rgb)

    warmup_times_s: list[float] = []
    for _ in range(args.warmup_iters):
        torch_npu.npu.synchronize(device)
        started = time.perf_counter()
        frontend.model(pixel_values=pixel_values)
        torch_npu.npu.synchronize(device)
        warmup_times_s.append(time.perf_counter() - started)

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    torch_npu.npu.synchronize(device)
    profile_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=profiler_config(),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(
            "paddle_ocr_vl.layout_detector_graph_replay"
        ):
            frontend.model(pixel_values=pixel_values)
        torch_npu.npu.synchronize(device)
        profiler.step()
    profile_wall_s = time.perf_counter() - profile_started

    summary = {
        "profile_kind": "steady_layout_detector_graph_replay",
        "profile_dir": str(profile_dir),
        "model": str(model_dir),
        "image": str(image_path),
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "input_shape": [int(value) for value in pixel_values.shape],
        "input_dtype": str(pixel_values.dtype),
        "conv_weight_format": args.conv_weight_format,
        "fractal_z_conv_weight_count": (
            frontend.fractal_z_conv_weight_count
        ),
        "setup_s": setup_s,
        "decode_timing": decode_timing,
        "warmup_iters": args.warmup_iters,
        "warmup_times_s": warmup_times_s,
        "profile_wall_s": profile_wall_s,
    }
    summary_path = profile_dir / "layout_detector_profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
