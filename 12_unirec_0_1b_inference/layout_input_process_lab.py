#!/usr/bin/env python3
"""Measure sharded direct-RGB layout preparation in spawned processes.

Each process owns a disjoint set of image paths, loads its own image processor,
and keeps prepared tensors local.  Only aggregate timing counters return to the
parent.  No image/tensor IPC, H2D copy, or model execution occurs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import resource
import statistics
import sys
import time
import traceback
from collections import defaultdict
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
    parser.add_argument("--processes", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-pages-per-process", type=int, default=1)
    parser.add_argument("--worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.processes = tuple(int(value) for value in args.processes.split(","))
    if not args.processes or any(value < 1 for value in args.processes):
        parser.error("--processes must contain positive integers")
    if len(set(args.processes)) != len(args.processes):
        parser.error("--processes must not contain duplicates")
    if args.offset < 0 or args.limit < 1 or args.rounds < 1:
        parser.error("--offset must be non-negative; --limit/--rounds positive")
    if args.warmup_pages_per_process < 0 or args.worker_timeout_s <= 0:
        parser.error("warmup must be non-negative and timeout positive")
    return args


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def worker_main(
    worker_index: int,
    paths: list[str],
    layout_model: str,
    warmup_pages: int,
    start_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    try:
        processor = AutoImageProcessor.from_pretrained(layout_model)
        for path_string in paths[:warmup_pages]:
            encoded = Path(path_string).read_bytes()
            pixel_values, _ = optimized_prepare(processor, encoded)
            validate_contract(pixel_values)
        ready_queue.put(
            {
                "status": "ready",
                "worker": worker_index,
                "torch_threads": torch.get_num_threads(),
                "opencv_threads": cv2.getNumThreads(),
            }
        )
        if not start_event.wait(timeout=300.0):
            raise TimeoutError("timed out waiting for parent start event")

        stage_s: dict[str, float] = defaultdict(float)
        started_cpu_s = cpu_seconds()
        started = time.perf_counter()
        for path_string in paths:
            encoded, file_read_s = read_bytes(Path(path_string))
            pixel_values, timing = optimized_prepare(processor, encoded)
            if (
                pixel_values.shape != (1, 3, 800, 800)
                or pixel_values.dtype != torch.float32
                or pixel_values.device.type != "cpu"
            ):
                raise RuntimeError("prepared tensor violated the CPU input contract")
            stage_s["file_read_s"] += file_read_s
            for stage, seconds in timing.items():
                stage_s[stage] += float(seconds)
        wall_s = time.perf_counter() - started
        process_cpu_s = cpu_seconds() - started_cpu_s
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result_queue.put(
            {
                "status": "ok",
                "worker": worker_index,
                "pages": len(paths),
                "wall_s": wall_s,
                "process_cpu_s": process_cpu_s,
                "average_cpu_cores": process_cpu_s / wall_s,
                "max_rss_kb": int(usage.ru_maxrss),
                "stage_s": dict(stage_s),
            }
        )
    except BaseException as exception:
        message = {
            "status": "error",
            "worker": worker_index,
            "error": repr(exception),
            "traceback": traceback.format_exc(),
        }
        try:
            ready_queue.put(message)
            result_queue.put(message)
        finally:
            return


def receive(queue_object: Any, *, timeout_s: float) -> dict[str, Any]:
    try:
        return queue_object.get(timeout=timeout_s)
    except queue.Empty as exception:
        raise TimeoutError(f"worker queue was silent for {timeout_s}s") from exception


def run_once(
    paths: list[Path],
    *,
    layout_model: Path,
    process_count: int,
    warmup_pages: int,
    timeout_s: float,
    round_index: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    start_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    shards = [
        [str(path) for path in paths[worker_index::process_count]]
        for worker_index in range(process_count)
    ]
    processes = [
        context.Process(
            target=worker_main,
            args=(
                worker_index,
                shard,
                str(layout_model),
                warmup_pages,
                start_event,
                ready_queue,
                result_queue,
            ),
            name=f"unirec-layout-input-{worker_index}",
        )
        for worker_index, shard in enumerate(shards)
    ]
    launch_started = time.perf_counter()
    for process in processes:
        process.start()

    try:
        ready = [receive(ready_queue, timeout_s=timeout_s) for _ in processes]
        errors = [message for message in ready if message["status"] != "ready"]
        if errors:
            raise RuntimeError(f"worker setup failed: {errors}")
        setup_wall_s = time.perf_counter() - launch_started
        measured_started = time.perf_counter()
        start_event.set()
        worker_results = [
            receive(result_queue, timeout_s=timeout_s) for _ in processes
        ]
        measured_wall_s = time.perf_counter() - measured_started
        errors = [message for message in worker_results if message["status"] != "ok"]
        if errors:
            raise RuntimeError(f"worker measurement failed: {errors}")
    finally:
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    total_pages = sum(int(result["pages"]) for result in worker_results)
    return {
        "round": round_index,
        "processes": process_count,
        "pages": total_pages,
        "shard_sizes": [len(shard) for shard in shards],
        "process_setup_wall_s": setup_wall_s,
        "measured_wall_s": measured_wall_s,
        "pages_per_s": total_pages / measured_wall_s,
        "total_wall_including_setup_s": setup_wall_s + measured_wall_s,
        "total_process_cpu_s": sum(
            float(result["process_cpu_s"]) for result in worker_results
        ),
        "average_cpu_cores": sum(
            float(result["process_cpu_s"]) for result in worker_results
        )
        / measured_wall_s,
        "summed_max_rss_kb": sum(int(result["max_rss_kb"]) for result in worker_results),
        "worker_setup": sorted(ready, key=lambda message: message["worker"]),
        "worker_results": sorted(
            worker_results, key=lambda result: result["worker"]
        ),
    }


def summarize(records: list[dict[str, Any]], process_counts: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for process_count in process_counts:
        rows = [row for row in records if row["processes"] == process_count]
        walls = [float(row["measured_wall_s"]) for row in rows]
        rates = [float(row["pages_per_s"]) for row in rows]
        setups = [float(row["process_setup_wall_s"]) for row in rows]
        cores = [float(row["average_cpu_cores"]) for row in rows]
        rss = [int(row["summed_max_rss_kb"]) for row in rows]
        summary[str(process_count)] = {
            "measured_wall_s_rounds": walls,
            "measured_wall_s_median": statistics.median(walls),
            "pages_per_s_rounds": rates,
            "pages_per_s_median": statistics.median(rates),
            "process_setup_wall_s_rounds": setups,
            "process_setup_wall_s_median": statistics.median(setups),
            "average_cpu_cores_median": statistics.median(cores),
            "summed_max_rss_kb_median": statistics.median(rss),
        }
    baseline = summary[str(process_counts[0])]["measured_wall_s_median"]
    for process_count in process_counts:
        row = summary[str(process_count)]
        row["speedup_vs_first_x"] = baseline / row["measured_wall_s_median"]
        row["parallel_efficiency_vs_first"] = (
            row["speedup_vs_first_x"] * process_counts[0] / process_count
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
    layout_model = args.layout_model.expanduser().resolve()
    print(
        f"LAYOUT_INPUT_PROCESS setup pages={len(paths)} "
        f"processes={args.processes} rounds={args.rounds}",
        flush=True,
    )

    records = []
    for round_index in range(args.rounds):
        order = (
            args.processes
            if round_index % 2 == 0
            else tuple(reversed(args.processes))
        )
        for process_count in order:
            print(
                f"LAYOUT_INPUT_PROCESS starting round={round_index + 1} "
                f"processes={process_count}",
                flush=True,
            )
            record = run_once(
                paths,
                layout_model=layout_model,
                process_count=process_count,
                warmup_pages=args.warmup_pages_per_process,
                timeout_s=args.worker_timeout_s,
                round_index=round_index,
            )
            records.append(record)
            print("LAYOUT_INPUT_PROCESS result " + json.dumps(record), flush=True)

    report = {
        "config": {
            "layout_model": str(layout_model),
            "input": str(args.input.expanduser().resolve()),
            "offset": args.offset,
            "limit": args.limit,
            "processes": list(args.processes),
            "rounds": args.rounds,
            "warmup_pages_per_process": args.warmup_pages_per_process,
            "start_method": "spawn",
            "tensor_or_image_ipc": False,
            "device_work": False,
            "output_contract": [1, 3, 800, 800],
        },
        "summary": summarize(records, args.processes),
        "rounds": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("LAYOUT_INPUT_PROCESS summary " + json.dumps(report["summary"]), flush=True)
    print(f"LAYOUT_INPUT_PROCESS done output={output}", flush=True)


if __name__ == "__main__":
    main()
