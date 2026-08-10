#!/usr/bin/env python3
"""Compile or load every static UniRec stages-0/1 vision-prefix graph."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from modeling_optimized_unirec import OptimizedUniRecRunner
from vision_static_shape import (
    PerShapeCompiledPrefixUniRecVisionRuntime,
    load_static_vision_shapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shapes-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")
    return args


def main() -> None:
    args = parse_args()
    physical = [
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if not physical:
        raise RuntimeError("source npu-setup before warming vision graphs")
    if 5 in physical:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")

    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    shapes = load_static_vision_shapes(args.shapes_manifest)
    total_started = time.perf_counter()
    setup_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    runtime = PerShapeCompiledPrefixUniRecVisionRuntime(runner, shapes=shapes)
    setup_s = time.perf_counter() - setup_started
    warmup_started = time.perf_counter()
    graph_report = runtime.warmup_all_prefix_graphs(passes=args.passes)
    warmup_s = time.perf_counter() - warmup_started
    first_pass = [value["pass_wall_s"][0] for value in graph_report.values()]
    result = {
        "status": "ok",
        "physical_devices": physical,
        "shape_count": len(shapes),
        "shapes": [list(shape) for shape in shapes],
        "passes": args.passes,
        "setup_s": setup_s,
        "warmup_s": warmup_s,
        "total_s": time.perf_counter() - total_started,
        "first_pass_s": {
            "min": min(first_pass),
            "median": statistics.median(first_pass),
            "mean": statistics.fmean(first_pass),
            "max": max(first_pass),
            "sum": sum(first_pass),
        },
        "graphs": graph_report,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_PREFIX_CACHE_END " + json.dumps(result), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
