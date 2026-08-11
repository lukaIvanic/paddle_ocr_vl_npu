#!/usr/bin/env python3
"""Resize hard-page crops directly into retained uint8 HWC vision canvases."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from benchmark_cpu_recognition_preprocess import (
    build_lanes,
    load_exact_crops,
    read_jsonl,
)
from modeling_optimized_unirec import UniRecImageProcessor
from vision_full_batch import DEFAULT_VISION_BUCKETS, VisionBucketSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--workers", default="8,16,32")
    parser.add_argument("--page-lookahead", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-crops", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.workers = [int(value) for value in args.workers.split(",")]
    if (
        any(value < 1 for value in args.workers)
        or args.page_lookahead < 1
        or args.rounds < 1
    ):
        parser.error("workers, page lookahead, and rounds must be positive")
    return args


def select_bucket(width: int, height: int) -> VisionBucketSpec | None:
    candidates = [
        spec for spec in DEFAULT_VISION_BUCKETS if spec.accepts(width, height)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda spec: (
            spec.width * spec.height,
            spec.batch_size,
            spec.height,
            spec.width,
        ),
    )


def build_canvases_and_destinations(
    crops: list[np.ndarray],
    page_crop_counts: list[int],
    processor: UniRecImageProcessor,
    *,
    page_lookahead: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, int]]:
    crop_groups = []
    offset = 0
    for start in range(0, len(page_crop_counts), page_lookahead):
        count = sum(page_crop_counts[start : start + page_lookahead])
        crop_groups.append(list(range(offset, offset + count)))
        offset += count
    if offset != len(crops):
        raise ValueError(f"page crop total {offset} != corpus total {len(crops)}")

    canvases = []
    destinations: list[np.ndarray | None] = [None] * len(crops)
    destination_canvas_indices: list[int | None] = [None] * len(crops)
    call_counts: dict[str, int] = {}
    for crop_indices in crop_groups:
        grouped: dict[str, list[int]] = {
            spec.key: [] for spec in DEFAULT_VISION_BUCKETS
        }
        fallbacks = []
        for crop_index in crop_indices:
            height, width = crops[crop_index].shape[:2]
            processed_width, processed_height = processor.get_processed_size(
                width,
                height,
            )
            spec = select_bucket(processed_width, processed_height)
            if spec is None:
                fallbacks.append((crop_index, processed_width, processed_height))
            else:
                grouped[spec.key].append(crop_index)
        for spec in DEFAULT_VISION_BUCKETS:
            pending = grouped[spec.key]
            for start in range(0, len(pending), spec.batch_size):
                members = pending[start : start + spec.batch_size]
                canvas = np.zeros(
                    (spec.batch_size, spec.height, spec.width, 3),
                    dtype=np.uint8,
                )
                canvases.append(canvas)
                canvas_index = len(canvases) - 1
                call_counts[spec.key] = call_counts.get(spec.key, 0) + 1
                for row, crop_index in enumerate(members):
                    height, width = crops[crop_index].shape[:2]
                    processed_width, processed_height = processor.get_processed_size(
                        width,
                        height,
                    )
                    destinations[crop_index] = canvas[
                        row,
                        :processed_height,
                        :processed_width,
                        :,
                    ]
                    destination_canvas_indices[crop_index] = canvas_index
        for crop_index, processed_width, processed_height in fallbacks:
            canvas = np.zeros(
                (1, processed_height, processed_width, 3),
                dtype=np.uint8,
            )
            canvases.append(canvas)
            canvas_index = len(canvases) - 1
            call_counts["fallback_eager"] = (
                call_counts.get("fallback_eager", 0) + 1
            )
            destinations[crop_index] = canvas[0]
            destination_canvas_indices[crop_index] = canvas_index
    if any(destination is None for destination in destinations) or any(
        index is None for index in destination_canvas_indices
    ):
        raise RuntimeError("one or more crops have no canvas destination")
    return (
        canvases,
        list(destinations),
        list(destination_canvas_indices),
        call_counts,
    )


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
    canvases, destinations, _canvas_indices, call_counts = (
        build_canvases_and_destinations(
            crops,
            page_crop_counts,
            processor,
            page_lookahead=args.page_lookahead,
        )
    )

    def resize_and_pack(index: int) -> float:
        output = lane(crops[index])
        np.copyto(destinations[index], output)
        return float(output.reshape(-1)[0])

    results = []
    for workers in args.workers:
        checksum = 0.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            checksum += sum(
                executor.map(
                    resize_and_pack,
                    range(min(args.warmup_crops, len(crops))),
                )
            )
            round_zero_s = []
            round_resize_pack_s = []
            round_wall_s = []
            for _round_index in range(args.rounds):
                gc.collect()
                started = time.perf_counter()
                for canvas in canvases:
                    canvas.fill(0)
                zeroed = time.perf_counter()
                checksum += sum(executor.map(resize_and_pack, range(len(crops))))
                finished = time.perf_counter()
                round_zero_s.append(zeroed - started)
                round_resize_pack_s.append(finished - zeroed)
                round_wall_s.append(finished - started)
        result = {
            "workers": workers,
            "round_zero_s": round_zero_s,
            "round_resize_pack_s": round_resize_pack_s,
            "round_wall_s": round_wall_s,
            "median_zero_s": statistics.median(round_zero_s),
            "median_resize_pack_s": statistics.median(round_resize_pack_s),
            "median_s": statistics.median(round_wall_s),
            "checksum": checksum,
        }
        results.append(result)
        print("UNIREC_CPU_BUCKET_PACK_RESULT " + json.dumps(result, sort_keys=True))

    verification = None
    if args.verify:
        started = time.perf_counter()
        exact_crops = 0
        different_values = 0
        for index, crop in enumerate(crops):
            expected = lane(crop)
            actual = destinations[index]
            crop_different = int(np.count_nonzero(actual != expected))
            exact_crops += int(crop_different == 0)
            different_values += crop_different
        verification = {
            "all_exact": different_values == 0,
            "exact_crops": exact_crops,
            "different_values": different_values,
            "total_values": sum(destination.size for destination in destinations),
            "wall_s_outside_benchmark": time.perf_counter() - started,
        }
    report = {
        "status": "ok",
        "crop_count": len(crops),
        "page_count": len(pages),
        "page_lookahead": args.page_lookahead,
        "call_counts": call_counts,
        "canvas_count": len(canvases),
        "canvas_bytes": sum(canvas.nbytes for canvas in canvases),
        "real_output_bytes": sum(destination.nbytes for destination in destinations),
        "output_contract": "retained_uint8_hwc_bucket_canvases",
        "verification": verification,
        "results": results,
    }
    print("UNIREC_CPU_BUCKET_PACK_SUMMARY " + json.dumps(report, sort_keys=True))
    if verification is not None and not verification["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
