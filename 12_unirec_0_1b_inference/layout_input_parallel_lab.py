#!/usr/bin/env python3
"""Measure direct-RGB layout-input preparation with CPU worker threads.

Each task reads one page and returns the CPU ``[1, 3, 800, 800]`` float32
tensor expected by PP-DocLayoutV2.  No H2D copy or model execution occurs.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import torch
from transformers import AutoImageProcessor

from layout_input_ab_lab import optimized_prepare, read_bytes, validate_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument("--offset", type=int, default=769)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument(
        "--workers",
        default="1,2,4,8",
        help="Comma-separated ThreadPoolExecutor worker counts",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-pages", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.workers = tuple(int(value) for value in args.workers.split(","))
    if not args.workers or any(value < 1 for value in args.workers):
        parser.error("--workers must contain positive integers")
    if len(set(args.workers)) != len(args.workers):
        parser.error("--workers must not contain duplicates")
    if args.offset < 0 or args.limit < 1 or args.rounds < 1:
        parser.error("--offset must be non-negative; --limit/--rounds positive")
    if args.warmup_pages < 0:
        parser.error("--warmup-pages must be non-negative")
    return args


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def check_contract_without_reduction(pixel_values: torch.Tensor) -> None:
    if pixel_values.shape != (1, 3, 800, 800):
        raise RuntimeError(f"unexpected layout input shape: {pixel_values.shape}")
    if pixel_values.dtype != torch.float32 or pixel_values.device.type != "cpu":
        raise RuntimeError(
            "layout input must be CPU float32, got "
            f"{pixel_values.dtype} on {pixel_values.device}"
        )


def prepare_path(
    processor: Any,
    path: Path,
) -> dict[str, float]:
    task_started = time.perf_counter()
    encoded, file_read_s = read_bytes(path)
    pixel_values, timing = optimized_prepare(processor, encoded)
    check_contract_without_reduction(pixel_values)
    return {
        "file_read_s": file_read_s,
        **timing,
        "task_wall_s": time.perf_counter() - task_started,
    }


def run_once(
    processor: Any,
    paths: list[Path],
    *,
    workers: int,
    round_index: int,
) -> dict[str, Any]:
    stage_s: dict[str, float] = defaultdict(float)
    started_cpu_s = cpu_seconds()
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="unirec-layout-input",
    ) as executor:
        results = executor.map(lambda path: prepare_path(processor, path), paths)
        for index, timing in enumerate(results):
            for stage, seconds in timing.items():
                stage_s[stage] += float(seconds)
            if (index + 1) % 32 == 0 or index + 1 == len(paths):
                print(
                    f"LAYOUT_INPUT_PARALLEL round={round_index + 1} "
                    f"workers={workers} pages={index + 1}/{len(paths)}",
                    flush=True,
                )
    wall_s = time.perf_counter() - started
    process_cpu_s = cpu_seconds() - started_cpu_s
    return {
        "round": round_index,
        "workers": workers,
        "pages": len(paths),
        "wall_s": wall_s,
        "pages_per_s": len(paths) / wall_s,
        "process_cpu_s": process_cpu_s,
        "average_cpu_cores": process_cpu_s / wall_s,
        "summed_stage_s": dict(stage_s),
    }


def summarize(records: list[dict[str, Any]], workers: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for worker_count in workers:
        rows = [row for row in records if row["workers"] == worker_count]
        walls = [float(row["wall_s"]) for row in rows]
        rates = [float(row["pages_per_s"]) for row in rows]
        cpu_cores = [float(row["average_cpu_cores"]) for row in rows]
        summary[str(worker_count)] = {
            "wall_s_rounds": walls,
            "wall_s_median": statistics.median(walls),
            "pages_per_s_rounds": rates,
            "pages_per_s_median": statistics.median(rates),
            "average_cpu_cores_rounds": cpu_cores,
            "average_cpu_cores_median": statistics.median(cpu_cores),
        }
    baseline = summary[str(workers[0])]["wall_s_median"]
    for worker_count in workers:
        row = summary[str(worker_count)]
        row["speedup_vs_first_x"] = baseline / row["wall_s_median"]
        row["parallel_efficiency_vs_first"] = (
            row["speedup_vs_first_x"] * workers[0] / worker_count
        )
    return summary


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.utility import get_image_file_list

    paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(args.input.expanduser().resolve())))
    ][args.offset : args.offset + args.limit]
    if len(paths) != args.limit:
        raise RuntimeError(f"requested {args.limit} pages, found {len(paths)}")
    processor = AutoImageProcessor.from_pretrained(
        args.layout_model.expanduser().resolve()
    )
    print(
        f"LAYOUT_INPUT_PARALLEL setup pages={len(paths)} "
        f"workers={args.workers} rounds={args.rounds} "
        f"torch_threads={torch.get_num_threads()} cv2_threads={cv2.getNumThreads()}",
        flush=True,
    )

    for path in paths[: args.warmup_pages]:
        encoded = path.read_bytes()
        pixel_values, _ = optimized_prepare(processor, encoded)
        validate_contract(pixel_values)
    print("LAYOUT_INPUT_PARALLEL warmup done", flush=True)

    records = []
    for round_index in range(args.rounds):
        order = args.workers if round_index % 2 == 0 else tuple(reversed(args.workers))
        for workers in order:
            record = run_once(
                processor,
                paths,
                workers=workers,
                round_index=round_index,
            )
            records.append(record)
            print("LAYOUT_INPUT_PARALLEL result " + json.dumps(record), flush=True)

    report = {
        "config": {
            "layout_model": str(args.layout_model.expanduser().resolve()),
            "input": str(args.input.expanduser().resolve()),
            "offset": args.offset,
            "limit": args.limit,
            "workers": list(args.workers),
            "rounds": args.rounds,
            "warmup_pages": args.warmup_pages,
            "device_work": False,
            "output_contract": [1, 3, 800, 800],
            "dtype": "torch.float32",
            "torch_num_threads": torch.get_num_threads(),
            "opencv_num_threads": cv2.getNumThreads(),
        },
        "summary": summarize(records, args.workers),
        "rounds": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("LAYOUT_INPUT_PARALLEL summary " + json.dumps(report["summary"]), flush=True)
    print(f"LAYOUT_INPUT_PARALLEL done output={output}", flush=True)


if __name__ == "__main__":
    main()
