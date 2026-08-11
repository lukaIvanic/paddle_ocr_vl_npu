#!/usr/bin/env python3
"""Overlap exact CPU resize/packing with compact NPU input preparation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401

from benchmark_cpu_recognition_bucket_pack import (
    build_canvases_and_destinations,
)
from benchmark_cpu_recognition_preprocess import (
    build_lanes,
    load_exact_crops,
    read_jsonl,
)
from modeling_optimized_unirec import UniRecImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--page-lookahead", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-crops", type=int, default=64)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if min(args.workers, args.page_lookahead, args.rounds) < 1:
        parser.error("workers, page lookahead, and rounds must be positive")
    physical_device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not physical_device:
        parser.error("source npu-setup before running this benchmark")
    if physical_device == "5":
        parser.error("physical NPU 5 is excluded")
    return args


def prepare_canvas_on_npu(
    canvas: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    pixels = torch.from_numpy(canvas).to(device)
    pixels = pixels.permute(0, 3, 1, 2).to(torch.float32)
    pixels.mul_(np.float32(2.0 / 255.0))
    pixels.sub_(np.float32(1.0))
    return pixels.to(torch.float16).contiguous()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    processor = UniRecImageProcessor()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    crops = load_exact_crops(
        artifact_dir,
        args.openocr_root.expanduser().resolve(),
        processor,
    )
    pages = sorted(
        read_jsonl(artifact_dir / "pages.jsonl"),
        key=lambda value: int(value["page_index"]),
    )
    page_crop_counts = [len(page["crops"]) for page in pages]
    lane = build_lanes(processor)["pillow_no_convert_uint8_hwc"]
    canvases, destinations, canvas_indices, call_counts = (
        build_canvases_and_destinations(
            crops,
            page_crop_counts,
            processor,
            page_lookahead=args.page_lookahead,
        )
    )
    device = torch.device(args.device)
    member_counts = [0] * len(canvases)
    for canvas_index in canvas_indices:
        member_counts[canvas_index] += 1

    def resize_and_pack(index: int) -> tuple[int, float]:
        output = lane(crops[index])
        np.copyto(destinations[index], output)
        return canvas_indices[index], float(output.reshape(-1)[0])

    unique_shapes = {}
    for canvas in canvases:
        unique_shapes.setdefault(canvas.shape, canvas)
    with torch.inference_mode():
        for canvas in unique_shapes.values():
            warmup = prepare_canvas_on_npu(canvas, device=device)
            del warmup
        torch.npu.synchronize()

    checksum = 0.0
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        checksum += sum(
            value
            for _canvas_index, value in executor.map(
                resize_and_pack,
                range(min(args.warmup_crops, len(crops))),
            )
        )
        for _round_index in range(args.rounds):
            gc.collect()
            started = time.perf_counter()
            for canvas in canvases:
                canvas.fill(0)
            zeroed = time.perf_counter()
            remaining = member_counts.copy()
            futures = [
                executor.submit(resize_and_pack, index)
                for index in range(len(crops))
            ]
            npu_call_count = 0
            with torch.inference_mode():
                for future in as_completed(futures):
                    canvas_index, value = future.result()
                    checksum += value
                    remaining[canvas_index] -= 1
                    if remaining[canvas_index] == 0:
                        prepared = prepare_canvas_on_npu(
                            canvases[canvas_index],
                            device=device,
                        )
                        del prepared
                        npu_call_count += 1
            cpu_and_launch_done = time.perf_counter()
            torch.npu.synchronize()
            finished = time.perf_counter()
            if npu_call_count != len(canvases) or any(remaining):
                raise RuntimeError("streamed input preparation lost a canvas")
            results.append(
                {
                    "zero_s": zeroed - started,
                    "cpu_resize_pack_and_npu_launch_s": (
                        cpu_and_launch_done - zeroed
                    ),
                    "final_npu_drain_s": finished - cpu_and_launch_done,
                    "wall_s": finished - started,
                    "npu_call_count": npu_call_count,
                }
            )
    report = {
        "status": "ok",
        "physical_npu": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "device": str(device),
        "workers": args.workers,
        "crop_count": len(crops),
        "canvas_count": len(canvases),
        "canvas_bytes": sum(canvas.nbytes for canvas in canvases),
        "call_counts": call_counts,
        "rounds": results,
        "median_s": statistics.median(result["wall_s"] for result in results),
        "checksum": checksum,
        "parity_basis": (
            "full exact Pillow uint8/model-input audits plus exact 256-value "
            "NPU normalization audit in the component labs"
        ),
    }
    print("UNIREC_STREAMED_RECOGNITION_INPUT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
