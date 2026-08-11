#!/usr/bin/env python3
"""Scale the bit-exact fused Pillow lane across persistent CPU workers."""

from __future__ import annotations

import argparse
import gc
import json
import mmap
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
_PROCESS_OUTPUT: mmap.mmap | None = None
_PROCESS_OUTPUT_SPECS: list[tuple[int, tuple[int, ...]]] | None = None
_PROCESS_INPUT: mmap.mmap | None = None
_PROCESS_INPUT_SPECS: list[tuple[int, tuple[int, ...]]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--workers", default="1,2,4,8,16,32")
    parser.add_argument("--modes", default="threads,processes")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-crops", type=int, default=64)
    parser.add_argument("--verify-shared-outputs", action="store_true")
    args = parser.parse_args()
    args.workers = [int(value) for value in args.workers.split(",")]
    args.modes = [value.strip() for value in args.modes.split(",")]
    if any(value < 1 for value in args.workers) or args.rounds < 1:
        parser.error("worker counts and --rounds must be positive")
    if not set(args.modes) <= {
        "threads",
        "processes",
        "processes_shared",
        "processes_shared_io",
        "processes_streamed_shared_io",
    }:
        parser.error(
            "--modes accepts only threads, processes, processes_shared, "
            "processes_shared_io, and processes_streamed_shared_io"
        )
    return args


def _checksum(lane: ArrayLane, crop: np.ndarray) -> float:
    return float(lane(crop).reshape(-1)[0])


def _process_checksum(index: int) -> float:
    if _PROCESS_CROPS is None or _PROCESS_LANE is None:
        raise RuntimeError("process benchmark globals are not initialized")
    return _checksum(_PROCESS_LANE, _PROCESS_CROPS[index])


def _process_shared_output(index: int) -> float:
    if (
        _PROCESS_CROPS is None
        or _PROCESS_LANE is None
        or _PROCESS_OUTPUT is None
        or _PROCESS_OUTPUT_SPECS is None
    ):
        raise RuntimeError("shared-output process globals are not initialized")
    output = _PROCESS_LANE(_PROCESS_CROPS[index])
    offset, shape = _PROCESS_OUTPUT_SPECS[index]
    destination = np.ndarray(
        shape,
        dtype=np.float32,
        buffer=_PROCESS_OUTPUT,
        offset=offset,
    )
    np.copyto(destination, output)
    return float(output.reshape(-1)[0])


def _process_shared_io(index: int) -> float:
    if (
        _PROCESS_LANE is None
        or _PROCESS_INPUT is None
        or _PROCESS_INPUT_SPECS is None
        or _PROCESS_OUTPUT is None
        or _PROCESS_OUTPUT_SPECS is None
    ):
        raise RuntimeError("shared-input/output process globals are not initialized")
    input_offset, input_shape = _PROCESS_INPUT_SPECS[index]
    crop = np.ndarray(
        input_shape,
        dtype=np.uint8,
        buffer=_PROCESS_INPUT,
        offset=input_offset,
    )
    output = _PROCESS_LANE(crop)
    output_offset, output_shape = _PROCESS_OUTPUT_SPECS[index]
    destination = np.ndarray(
        output_shape,
        dtype=np.float32,
        buffer=_PROCESS_OUTPUT,
        offset=output_offset,
    )
    np.copyto(destination, output)
    return float(output.reshape(-1)[0])


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


def _verify_shared_outputs(
    crops: list[np.ndarray],
    lane: ArrayLane,
    output: mmap.mmap,
    output_specs: list[tuple[int, tuple[int, ...]]],
) -> dict[str, object]:
    started = time.perf_counter()
    exact_crops = 0
    different_values = 0
    total_values = 0
    max_absolute = 0.0
    for index, crop in enumerate(crops):
        expected = lane(crop)
        offset, shape = output_specs[index]
        actual = np.ndarray(
            shape,
            dtype=np.float32,
            buffer=output,
            offset=offset,
        )
        difference = actual - expected
        crop_different = int(np.count_nonzero(difference))
        exact_crops += int(crop_different == 0)
        different_values += crop_different
        total_values += int(difference.size)
        max_absolute = max(
            max_absolute,
            float(np.max(np.abs(difference), initial=0.0)),
        )
    return {
        "reference": "direct_pillow_chw_fused_formula",
        "all_exact": different_values == 0,
        "exact_crops": exact_crops,
        "crop_count": len(crops),
        "different_values": different_values,
        "total_values": total_values,
        "max_absolute": max_absolute,
        "wall_s_outside_benchmark": time.perf_counter() - started,
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


def benchmark_processes_shared(
    crops: list[np.ndarray],
    lane: ArrayLane,
    *,
    workers: int,
    rounds: int,
    warmup_crops: int,
    output_specs: list[tuple[int, tuple[int, ...]]],
    output_bytes: int,
) -> dict[str, object]:
    global _PROCESS_CROPS, _PROCESS_LANE, _PROCESS_OUTPUT, _PROCESS_OUTPUT_SPECS
    _PROCESS_CROPS = crops
    _PROCESS_LANE = lane
    _PROCESS_OUTPUT_SPECS = output_specs
    _PROCESS_OUTPUT = mmap.mmap(
        -1,
        output_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    checksum = 0.0
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        checksum += sum(
            pool.imap_unordered(
                _process_shared_output,
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
                    _process_shared_output,
                    range(len(crops)),
                    chunksize=1,
                )
            )
            wall_times.append(time.perf_counter() - started)
    # Prove that the completed output mapping is readable by the owner without
    # adding a full 1.6 GB consumer scan to the preprocessing timing window.
    boundary_checksum = 0.0
    for index in (0, len(crops) // 2, len(crops) - 1):
        offset, shape = output_specs[index]
        output = np.ndarray(
            shape,
            dtype=np.float32,
            buffer=_PROCESS_OUTPUT,
            offset=offset,
        )
        boundary_checksum += float(output.reshape(-1)[0])
    _PROCESS_OUTPUT.close()
    _PROCESS_CROPS = None
    _PROCESS_LANE = None
    _PROCESS_OUTPUT = None
    _PROCESS_OUTPUT_SPECS = None
    result = _finish_result(
        mode="processes_shared",
        workers=workers,
        crop_count=len(crops),
        wall_times=wall_times,
        checksum=checksum,
    )
    result["output_bytes"] = output_bytes
    result["boundary_checksum"] = boundary_checksum
    return result


def benchmark_processes_shared_io(
    crops: list[np.ndarray],
    lane: ArrayLane,
    *,
    workers: int,
    rounds: int,
    warmup_crops: int,
    input_specs: list[tuple[int, tuple[int, ...]]],
    input_bytes: int,
    output_specs: list[tuple[int, tuple[int, ...]]],
    output_bytes: int,
    stream_inputs: bool = False,
    verify_outputs: bool = False,
) -> dict[str, object]:
    global _PROCESS_LANE, _PROCESS_INPUT, _PROCESS_INPUT_SPECS
    global _PROCESS_OUTPUT, _PROCESS_OUTPUT_SPECS
    _PROCESS_LANE = lane
    _PROCESS_INPUT_SPECS = input_specs
    _PROCESS_OUTPUT_SPECS = output_specs
    _PROCESS_INPUT = mmap.mmap(
        -1,
        input_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    _PROCESS_OUTPUT = mmap.mmap(
        -1,
        output_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )

    def pack_inputs(count: int) -> None:
        for index, crop in enumerate(crops[:count]):
            offset, shape = input_specs[index]
            destination = np.ndarray(
                shape,
                dtype=np.uint8,
                buffer=_PROCESS_INPUT,
                offset=offset,
            )
            np.copyto(destination, crop)

    checksum = 0.0
    mode = (
        "processes_streamed_shared_io"
        if stream_inputs
        else "processes_shared_io"
    )
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        warmup_count = min(warmup_crops, len(crops))
        pack_inputs(warmup_count)
        checksum += sum(
            pool.imap_unordered(
                _process_shared_io,
                range(warmup_count),
                chunksize=1,
            )
        )
        wall_times = []
        input_pack_times = []
        compute_output_times = []
        for _round_index in range(rounds):
            gc.collect()
            started = time.perf_counter()
            if stream_inputs:
                input_pack_s = 0.0

                def pack_and_yield_indices():
                    nonlocal input_pack_s
                    for index, crop in enumerate(crops):
                        copy_started = time.perf_counter()
                        offset, shape = input_specs[index]
                        destination = np.ndarray(
                            shape,
                            dtype=np.uint8,
                            buffer=_PROCESS_INPUT,
                            offset=offset,
                        )
                        np.copyto(destination, crop)
                        input_pack_s += time.perf_counter() - copy_started
                        yield index

                checksum += sum(
                    pool.imap_unordered(
                        _process_shared_io,
                        pack_and_yield_indices(),
                        chunksize=1,
                    )
                )
                packed = started + input_pack_s
            else:
                pack_inputs(len(crops))
                packed = time.perf_counter()
                checksum += sum(
                    pool.imap_unordered(
                        _process_shared_io,
                        range(len(crops)),
                        chunksize=1,
                    )
                )
            finished = time.perf_counter()
            input_pack_times.append(packed - started)
            compute_output_times.append(finished - packed)
            wall_times.append(finished - started)

    boundary_checksum = 0.0
    for index in (0, len(crops) // 2, len(crops) - 1):
        offset, shape = output_specs[index]
        output = np.ndarray(
            shape,
            dtype=np.float32,
            buffer=_PROCESS_OUTPUT,
            offset=offset,
        )
        boundary_checksum += float(output.reshape(-1)[0])
    verification = (
        _verify_shared_outputs(
            crops,
            lane,
            _PROCESS_OUTPUT,
            output_specs,
        )
        if verify_outputs
        else None
    )
    _PROCESS_INPUT.close()
    _PROCESS_OUTPUT.close()
    _PROCESS_LANE = None
    _PROCESS_INPUT = None
    _PROCESS_INPUT_SPECS = None
    _PROCESS_OUTPUT = None
    _PROCESS_OUTPUT_SPECS = None
    result = _finish_result(
        mode=mode,
        workers=workers,
        crop_count=len(crops),
        wall_times=wall_times,
        checksum=checksum,
    )
    result.update(
        {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "boundary_checksum": boundary_checksum,
            "input_delivery": (
                "streamed_and_overlapped" if stream_inputs else "pack_then_compute"
            ),
            "shared_output_verification": verification,
        }
    )
    if stream_inputs:
        result.update(
            {
                "round_input_copy_cpu_s": input_pack_times,
                "round_wall_minus_input_copy_cpu_s": compute_output_times,
                "median_input_copy_cpu_s": statistics.median(input_pack_times),
                "median_wall_minus_input_copy_cpu_s": statistics.median(
                    compute_output_times
                ),
            }
        )
    else:
        result.update(
            {
                "round_input_pack_s": input_pack_times,
                "round_compute_and_output_s": compute_output_times,
                "median_input_pack_s": statistics.median(input_pack_times),
                "median_compute_and_output_s": statistics.median(
                    compute_output_times
                ),
            }
        )
    return result


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
    input_specs = []
    input_bytes = 0
    output_specs = []
    output_bytes = 0
    for crop in crops:
        input_bytes = (input_bytes + 63) // 64 * 64
        input_specs.append((input_bytes, crop.shape))
        input_bytes += crop.nbytes
        width, height = processor.get_processed_size(
            crop.shape[1],
            crop.shape[0],
        )
        output_bytes = (output_bytes + 63) // 64 * 64
        shape = (1, 3, height, width)
        output_specs.append((output_bytes, shape))
        output_bytes += int(np.prod(shape, dtype=np.int64)) * 4
    results = []
    for mode in args.modes:
        for workers in args.workers:
            print(
                f"UNIREC_CPU_PARALLEL_BEGIN mode={mode} workers={workers}",
                flush=True,
            )
            if mode == "threads":
                result = benchmark_threads(
                    crops,
                    lane,
                    workers=workers,
                    rounds=args.rounds,
                    warmup_crops=args.warmup_crops,
                )
            elif mode == "processes":
                result = benchmark_processes(
                    crops,
                    lane,
                    workers=workers,
                    rounds=args.rounds,
                    warmup_crops=args.warmup_crops,
                )
            elif mode == "processes_shared":
                result = benchmark_processes_shared(
                    crops,
                    lane,
                    workers=workers,
                    rounds=args.rounds,
                    warmup_crops=args.warmup_crops,
                    output_specs=output_specs,
                    output_bytes=output_bytes,
                )
            elif mode == "processes_shared_io":
                result = benchmark_processes_shared_io(
                    crops,
                    lane,
                    workers=workers,
                    rounds=args.rounds,
                    warmup_crops=args.warmup_crops,
                    input_specs=input_specs,
                    input_bytes=input_bytes,
                    output_specs=output_specs,
                    output_bytes=output_bytes,
                    verify_outputs=args.verify_shared_outputs,
                )
            else:
                result = benchmark_processes_shared_io(
                    crops,
                    lane,
                    workers=workers,
                    rounds=args.rounds,
                    warmup_crops=args.warmup_crops,
                    input_specs=input_specs,
                    input_bytes=input_bytes,
                    output_specs=output_specs,
                    output_bytes=output_bytes,
                    stream_inputs=True,
                    verify_outputs=args.verify_shared_outputs,
                )
            results.append(result)
            print(
                "UNIREC_CPU_PARALLEL_END " + json.dumps(result, sort_keys=True),
                flush=True,
            )
    baseline = min(
        (value for value in results if value["mode"] == args.modes[0]),
        key=lambda value: int(value["workers"]),
    )
    baseline_s = float(baseline["median_s"])
    for result in results:
        result["speedup_vs_baseline_configuration"] = baseline_s / float(
            result["median_s"]
        )
    print(
        "UNIREC_CPU_PARALLEL_SUMMARY "
        + json.dumps(
            {
                "status": "ok",
                "crop_count": len(crops),
                "lane": "pillow_chw_fused_formula",
                "parity": "bit_exact_in_full_sequential_lab",
                "baseline_mode": baseline["mode"],
                "baseline_workers": baseline["workers"],
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
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
