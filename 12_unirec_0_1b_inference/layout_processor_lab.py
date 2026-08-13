#!/usr/bin/env python3
"""Isolate the exact CPU processor used by production UniRec layout B1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from layout_page_input import decode_page_rgb, materialize_layout_rgb
from opendoc_layout_npu import prepare_layout_resized_uint8_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument(
        "--thread-counts",
        default="native,1,2,4,8",
        help="Comma-separated intra-op thread lanes; native preserves default",
    )
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.warmup_calls < 0:
        parser.error("offset/warmup must be non-negative and limit positive")
    lanes: list[int | None] = []
    for value in args.thread_counts.split(","):
        value = value.strip().lower()
        if value == "native":
            lanes.append(None)
        else:
            count = int(value)
            if count < 1:
                parser.error("thread counts must be positive")
            lanes.append(count)
    args.thread_counts = lanes
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "total_s": sum(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p90_ms": percentile(values, 0.90) * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def run_lane(
    paths: list[Path],
    *,
    lane: str,
    threads: int,
    warmup_calls: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    warmup_rgb, _ = decode_page_rgb(paths[0])
    warmup_image = materialize_layout_rgb(warmup_rgb)
    for _ in range(warmup_calls):
        prepare_layout_resized_uint8_exact([warmup_image])

    records = []
    fingerprints = []
    for index, path in enumerate(paths):
        rgb, _decode_timing = decode_page_rgb(path)
        image = materialize_layout_rgb(rgb)
        detail: dict[str, float] = defaultdict(float)
        started = time.perf_counter()
        pixels = prepare_layout_resized_uint8_exact(
            [image], timing_s=detail
        )["pixel_values"]
        total_s = time.perf_counter() - started
        byte_view = memoryview(pixels.numpy()).cast("B")
        fingerprints.append(f"{zlib.crc32(byte_view) & 0xFFFFFFFF:08x}")
        records.append(
            {
                "page_index": index,
                "image": str(path),
                "processor_preprocess_s": total_s,
                **detail,
            }
        )
    names = sorted({name for row in records for name in row if name.endswith("_s")})
    stages = {
        name: summarize([float(row[name]) for row in records])
        for name in names
    }
    return {
        "lane": lane,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "page_count": len(records),
        "fingerprints": fingerprints,
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    native_threads = torch.get_num_threads()
    openocr_root = args.openocr_root.expanduser().resolve()
    sys.path.insert(0, str(openocr_root))
    from tools.utils.utility import get_image_file_list

    paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(args.input.resolve())))
    ][args.offset : args.offset + args.limit]
    if len(paths) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} pages but selected {len(paths)}"
        )

    lanes = []
    reference_fingerprints = None
    for requested in args.thread_counts:
        threads = native_threads if requested is None else requested
        lane = "native" if requested is None else f"threads_{threads}"
        print(
            f"LAYOUT_PROCESSOR_LAB lane={lane} threads={threads} begin",
            flush=True,
        )
        result = run_lane(
            paths,
            lane=lane,
            threads=threads,
            warmup_calls=args.warmup_calls,
        )
        if reference_fingerprints is None:
            reference_fingerprints = result["fingerprints"]
        result["exact_vs_native"] = (
            result["fingerprints"] == reference_fingerprints
        )
        stages = result["stages"]
        print(
            "LAYOUT_PROCESSOR_LAB_RESULT "
            f"lane={lane} total_mean_ms="
            f"{stages['processor_preprocess_s']['mean_ms']:.3f} "
            f"contiguous_mean_ms="
            f"{stages['processor_chw_contiguous_s']['mean_ms']:.3f} "
            f"resize_mean_ms="
            f"{stages['processor_bicubic_resize_s']['mean_ms']:.3f} "
            f"exact_vs_native={result['exact_vs_native']}",
            flush=True,
        )
        lanes.append(result)

    report = {
        "config": {
            "input": str(args.input.resolve()),
            "offset": args.offset,
            "limit": args.limit,
            "warmup_calls": args.warmup_calls,
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "native_torch_intraop_threads": native_threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "lanes": lanes,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"LAYOUT_PROCESSOR_LAB_DONE output={output}", flush=True)


if __name__ == "__main__":
    main()
