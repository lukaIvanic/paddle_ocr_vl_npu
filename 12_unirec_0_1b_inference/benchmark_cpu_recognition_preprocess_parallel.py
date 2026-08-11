#!/usr/bin/env python3
"""Scale the bit-exact fused Pillow lane across persistent CPU workers."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from benchmark_cpu_recognition_preprocess import build_lanes, load_exact_crops
from modeling_optimized_unirec import UniRecImageProcessor


ArrayLane = Callable[[np.ndarray], np.ndarray]
_PROCESS_CROPS: list[np.ndarray] | None = None
_PROCESS_LANE: ArrayLane | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--workers", default="1,2,4,8,16,32")
    parser.add_argument("--modes", default="threads,processes")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-crops", type=int, default=64)
    args = parser.parse_args()
    args.workers = [int(value) for value in args.workers.split(",")]
    args.modes = [value.strip() for value in args.modes.split(",")]
    if any(value < 1 for value in args.workers) or args.rounds < 1:
        parser.error("worker counts and --rounds must be positive")
    if not set(args.modes) <= {"threads", "processes"}:
        parser.error("--modes accepts only threads and processes")
    return args


def _checksum(lane: ArrayLane, crop: np.ndarray) -> float:
    return float(lane(crop).reshape(-1)[0])


def _process_checksum(index: int) -> float:
    if _PROCESS_CROPS is None or _PROCESS_LANE is None:
        raise RuntimeError("process benchmark globals are not initialized")
    return _checksum(_PROCESS_LANE, _PROCESS_CROPS[index])


def _finish_result(
    *,
    mode: str,
    workers: int,
    crop_count: int,
    wall_times: list[float],
    checksum: float,
) -> dict[str, object]:
    median_s = statistics.median(wall_times)
    return {
        "mode": mode,
        "workers": workers,
        "round_wall_s": wall_times,
        "median_s": median_s,
        "crops_per_s": crop_count / median_s,
        "checksum": checksum,
    }


def benchmark_threads(
    crops: list[np.ndarray],
    lane: ArrayLane,
    *,
    workers: int,
    rounds: int,
    warmup_crops: int,
) -> dict[str, object]:
    checksum = 0.0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        checksum += sum(
            executor.map(
                lambda crop: _checksum(lane, crop),
                crops[:warmup_crops],
            )
        )
        wall_times = []
        for _round_index in range(rounds):
            gc.collect()
            started = time.perf_counter()
            checksum += sum(
                executor.map(lambda crop: _checksum(lane, crop), crops)
            )
            wall_times.append(time.perf_counter() - started)
    return _finish_result(
        mode="threads",
        workers=workers,
        crop_count=len(crops),
        wall_times=wall_times,
        checksum=checksum,
    )


def benchmark_processes(
    crops: list[np.ndarray],
    lane: ArrayLane,
    *,
    workers: int,
    rounds: int,
    warmup_crops: int,
) -> dict[str, object]:
    global _PROCESS_CROPS, _PROCESS_LANE
    _PROCESS_CROPS = crops
    _PROCESS_LANE = lane
    checksum = 0.0
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        checksum += sum(
            pool.imap_unordered(
                _process_checksum,
                range(min(warmup_crops, len(crops))),
                chunksize=1,
            )
        )
        wall_times = []
        for _round_index in range(rounds):
            gc.collect()
            started = time.perf_counter()
            checksum += sum(
                pool.imap_unordered(
                    _process_checksum,
                    range(len(crops)),
                    chunksize=1,
                )
            )
            wall_times.append(time.perf_counter() - started)
    _PROCESS_CROPS = None
    _PROCESS_LANE = None
    return _finish_result(
        mode="processes",
        workers=workers,
        crop_count=len(crops),
        wall_times=wall_times,
        checksum=checksum,
    )


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    processor = UniRecImageProcessor()
    crops = load_exact_crops(
        args.artifact_dir.expanduser().resolve(),
        args.openocr_root.expanduser().resolve(),
        processor,
    )
    lane = build_lanes(processor)["pillow_chw_fused_formula"]
    results = []
    for mode in args.modes:
        for workers in args.workers:
            print(
                f"UNIREC_CPU_PARALLEL_BEGIN mode={mode} workers={workers}",
                flush=True,
            )
            benchmark = benchmark_threads if mode == "threads" else benchmark_processes
            result = benchmark(
                crops,
                lane,
                workers=workers,
                rounds=args.rounds,
                warmup_crops=args.warmup_crops,
            )
            results.append(result)
            print(
                "UNIREC_CPU_PARALLEL_END " + json.dumps(result, sort_keys=True),
                flush=True,
            )
    baseline = next(
        value
        for value in results
        if value["mode"] == args.modes[0] and value["workers"] == 1
    )
    baseline_s = float(baseline["median_s"])
    for result in results:
        result["speedup_vs_one_worker"] = baseline_s / float(result["median_s"])
    print(
        "UNIREC_CPU_PARALLEL_SUMMARY "
        + json.dumps(
            {
                "status": "ok",
                "crop_count": len(crops),
                "lane": "pillow_chw_fused_formula",
                "parity": "bit_exact_in_full_sequential_lab",
                "opencv_threads_per_worker": cv2.getNumThreads(),
                "torch_threads_per_worker": torch.get_num_threads(),
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
