#!/usr/bin/env python3
"""Isolate full-page frontend transport without loading any NPU model.

The lab replays page and crop array shapes from a recorded recognition trace.
Workers construct resident uint8 payloads and return them through one of four
transport lanes:

* metadata: descriptors only; coordinator and queue control lower bound.
* pickle_arrays: the current production shape, one ndarray per page/crop.
* pickle_arena: one packed ndarray per page, still copied through the queue.
* shared_memory_arena: one parent-owned shared-memory arena per page; only its
  name and descriptors cross the queue.

No layout or recognition model is loaded.  Model execution cannot hide or
distort transport time.  Setup and worker shutdown are reported separately.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import resource
import statistics
import time
import traceback
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODES = (
    "metadata",
    "pickle_arrays",
    "pickle_arena",
    "shared_memory_arena",
)


@dataclass(frozen=True)
class PageSpec:
    page_index: int
    page_name: str
    page_shape: tuple[int, int, int] | None
    crop_shapes: tuple[tuple[int, int, int], ...]

    @property
    def segment_shapes(self) -> tuple[tuple[int, int, int], ...]:
        if self.page_shape is None:
            return self.crop_shapes
        return (self.page_shape, *self.crop_shapes)

    @property
    def segment_nbytes(self) -> tuple[int, ...]:
        return tuple(int(np.prod(shape)) for shape in self.segment_shapes)

    @property
    def payload_nbytes(self) -> int:
        return sum(self.segment_nbytes)

    @property
    def sentinel(self) -> int:
        return (self.page_index % 251) + 1


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "tmp/12_unirec_0_1b_inference/"
            "hard128_fullfrontend_2f70266/output/recognition_trace.jsonl"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
        help="Image root used only to read full-page dimensions from headers.",
    )
    parser.add_argument("--limit-pages", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=16,
        help="Maximum page payloads resident between workers and coordinator.",
    )
    parser.add_argument("--modes", type=parse_csv, default=MODES)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--consumer",
        choices=("sample", "copy"),
        default="sample",
        help=(
            "sample verifies three bytes per segment; copy additionally copies "
            "every segment into coordinator-owned memory before releasing it"
        ),
    )
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Exclude the full-page image and replay only crop arrays.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    invalid_modes = sorted(set(args.modes) - set(MODES))
    if invalid_modes:
        parser.error(f"unknown modes: {invalid_modes}; choices={MODES}")
    if not args.modes:
        parser.error("--modes must not be empty")
    if args.limit_pages is not None and args.limit_pages < 1:
        parser.error("--limit-pages must be positive")
    if args.workers < 1 or args.max_inflight < 1 or args.rounds < 1:
        parser.error("--workers, --max-inflight, and --rounds must be positive")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return args


def load_page_specs(
    trace_path: Path,
    input_root: Path,
    *,
    limit_pages: int | None,
    include_page_image: bool,
) -> list[PageSpec]:
    grouped: dict[int, dict[str, Any]] = {}
    with trace_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            page_index = int(record["page_index"])
            page = grouped.setdefault(
                page_index,
                {"page": str(record["page"]), "crops": []},
            )
            if page["page"] != str(record["page"]):
                raise RuntimeError(f"page name changed at index {page_index}")
            width, height = map(int, record["crop_size"])
            page["crops"].append((height, width, 3))

    page_indices = sorted(grouped)
    if limit_pages is not None:
        page_indices = page_indices[:limit_pages]
    specs = []
    for page_index in page_indices:
        record = grouped[page_index]
        page_name = record["page"]
        page_shape = None
        if include_page_image:
            with Image.open(input_root / page_name) as image:
                width, height = image.size
            page_shape = (int(height), int(width), 3)
        specs.append(
            PageSpec(
                page_index=page_index,
                page_name=page_name,
                page_shape=page_shape,
                crop_shapes=tuple(record["crops"]),
            )
        )
    if not specs:
        raise RuntimeError(f"no pages found in {trace_path}")
    return specs


def allocate_arrays(spec: PageSpec) -> list[np.ndarray]:
    return [
        np.full(shape, spec.sentinel, dtype=np.uint8)
        for shape in spec.segment_shapes
    ]


def allocate_arena(spec: PageSpec) -> np.ndarray:
    return np.full(spec.payload_nbytes, spec.sentinel, dtype=np.uint8)


def arena_views(buffer: Any, spec: PageSpec) -> list[np.ndarray]:
    views = []
    offset = 0
    for shape, nbytes in zip(spec.segment_shapes, spec.segment_nbytes):
        views.append(
            np.ndarray(shape, dtype=np.uint8, buffer=buffer, offset=offset)
        )
        offset += nbytes
    if offset != spec.payload_nbytes:
        raise RuntimeError(f"arena accounting mismatch for page {spec.page_index}")
    return views


def worker_main(
    worker_index: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        result_queue.put({"status": "ready", "worker": worker_index})
        while True:
            task = task_queue.get()
            if task is None:
                return
            mode: str = task["mode"]
            run_id = int(task["run_id"])
            spec: PageSpec = task["spec"]
            build_started = time.perf_counter()
            payload: Any = None
            shared: SharedMemory | None = None
            if mode == "metadata":
                payload = None
            elif mode == "pickle_arrays":
                payload = allocate_arrays(spec)
            elif mode == "pickle_arena":
                payload = allocate_arena(spec)
            elif mode == "shared_memory_arena":
                shared = SharedMemory(name=str(task["shared_memory_name"]))
                arena = np.ndarray(
                    (spec.payload_nbytes,),
                    dtype=np.uint8,
                    buffer=shared.buf,
                )
                arena.fill(spec.sentinel)
                del arena
            else:
                raise RuntimeError(f"unsupported mode: {mode}")
            build_s = time.perf_counter() - build_started
            # Use the system wall clock only for cross-process timestamps.
            # Python 3.9 on macOS can give perf_counter()/monotonic() a
            # process-local epoch.  Short lab runs are not sensitive to clock
            # adjustment; all local durations still use perf_counter().
            ready_at = time.time()
            result_queue.put(
                {
                    "status": "ok",
                    "worker": worker_index,
                    "run_id": run_id,
                    "mode": mode,
                    "page_index": spec.page_index,
                    "payload": payload,
                    "payload_nbytes": spec.payload_nbytes,
                    "build_s": build_s,
                    "ready_at": ready_at,
                    "worker_rss_kb": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ),
                }
            )
            if shared is not None:
                shared.close()
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "worker": worker_index,
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


def receive(result_queue: Any, timeout_s: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        message = result_queue.get(timeout=timeout_s)
    except queue.Empty as exception:
        raise TimeoutError(f"IPC lab was silent for {timeout_s}s") from exception
    return message, time.perf_counter() - started


def verify_and_consume(
    arrays: list[np.ndarray],
    spec: PageSpec,
    *,
    consumer: str,
) -> tuple[int, float]:
    mismatches = 0
    consume_started = time.perf_counter()
    if len(arrays) != len(spec.segment_shapes):
        return 1, time.perf_counter() - consume_started
    for array, expected_shape in zip(arrays, spec.segment_shapes):
        if array.shape != expected_shape or array.dtype != np.uint8:
            mismatches += 1
            continue
        flat = array.reshape(-1)
        indices = (0, len(flat) // 2, len(flat) - 1)
        if any(int(flat[index]) != spec.sentinel for index in indices):
            mismatches += 1
        if consumer == "copy":
            copied = np.array(array, copy=True, order="C")
            if int(copied.reshape(-1)[-1]) != spec.sentinel:
                mismatches += 1
            del copied
    return mismatches, time.perf_counter() - consume_started


def run_lane(
    specs: list[PageSpec],
    *,
    mode: str,
    run_id: int,
    workers: int,
    max_inflight: int,
    consumer: str,
    timeout_s: float,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=max_inflight)
    result_queue = context.Queue(maxsize=max_inflight)
    processes = [
        context.Process(
            target=worker_main,
            args=(worker_index, task_queue, result_queue),
            name=f"unirec-frontend-ipc-{mode}-{worker_index}",
        )
        for worker_index in range(workers)
    ]
    setup_started = time.perf_counter()
    for process in processes:
        process.start()
    shared_blocks: dict[int, SharedMemory] = {}
    try:
        ready = [receive(result_queue, timeout_s)[0] for _ in processes]
        errors = [message for message in ready if message["status"] != "ready"]
        if errors:
            raise RuntimeError(f"worker setup failed: {errors}")
        setup_s = time.perf_counter() - setup_started
        lane_started = time.perf_counter()
        lane_started_system = time.time()
        shm_allocate_s = 0.0
        next_spec_index = 0
        completed = 0
        receive_wait_s = 0.0
        consume_s = 0.0
        delivery_values = []
        build_values = []
        worker_pages = [0] * workers
        worker_peak_rss_kb = [0] * workers
        mismatches = 0
        first_ready_s = None
        last_ready_s = None
        progress_step = max(1, len(specs) // 10)

        def dispatch(spec: PageSpec) -> None:
            nonlocal shm_allocate_s
            task: dict[str, Any] = {
                "mode": mode,
                "run_id": run_id,
                "spec": spec,
            }
            if mode == "shared_memory_arena":
                allocate_started = time.perf_counter()
                shared = SharedMemory(create=True, size=spec.payload_nbytes)
                shm_allocate_s += time.perf_counter() - allocate_started
                shared_blocks[spec.page_index] = shared
                task["shared_memory_name"] = shared.name
            task_queue.put(task)

        initial = min(max_inflight, len(specs))
        for _ in range(initial):
            dispatch(specs[next_spec_index])
            next_spec_index += 1

        specs_by_index = {spec.page_index: spec for spec in specs}
        while completed < len(specs):
            message, wait_s = receive(result_queue, timeout_s)
            receive_wait_s += wait_s
            if message["status"] != "ok":
                raise RuntimeError(f"worker execution failed: {message}")
            if int(message["run_id"]) != run_id or message["mode"] != mode:
                raise RuntimeError(f"unexpected result: {message}")
            arrived_at = time.time()
            ready_at = float(message["ready_at"])
            if first_ready_s is None:
                first_ready_s = ready_at - lane_started_system
            last_ready_s = ready_at - lane_started_system
            delivery_values.append(arrived_at - ready_at)
            build_values.append(float(message["build_s"]))
            page_index = int(message["page_index"])
            spec = specs_by_index[page_index]
            if int(message["payload_nbytes"]) != spec.payload_nbytes:
                mismatches += 1
            if mode == "metadata":
                arrays: list[np.ndarray] = []
                consume_started = time.perf_counter()
                if spec.segment_shapes and message["payload"] is not None:
                    mismatches += 1
                page_consume_s = time.perf_counter() - consume_started
            elif mode == "pickle_arrays":
                arrays = message["payload"]
                page_mismatches, page_consume_s = verify_and_consume(
                    arrays, spec, consumer=consumer
                )
                mismatches += page_mismatches
            elif mode == "pickle_arena":
                arena = message["payload"]
                arrays = arena_views(arena, spec)
                page_mismatches, page_consume_s = verify_and_consume(
                    arrays, spec, consumer=consumer
                )
                mismatches += page_mismatches
                del arrays, arena
            else:
                shared = shared_blocks.pop(page_index)
                arrays = arena_views(shared.buf, spec)
                page_mismatches, page_consume_s = verify_and_consume(
                    arrays, spec, consumer=consumer
                )
                mismatches += page_mismatches
                del arrays
                shared.close()
                shared.unlink()
            consume_s += page_consume_s
            worker_index = int(message["worker"])
            worker_pages[worker_index] += 1
            worker_peak_rss_kb[worker_index] = max(
                worker_peak_rss_kb[worker_index],
                int(message["worker_rss_kb"]),
            )
            completed += 1
            if next_spec_index < len(specs):
                dispatch(specs[next_spec_index])
                next_spec_index += 1
            if completed % progress_step == 0 or completed == len(specs):
                print(
                    f"UNIREC_FRONTEND_IPC_PROGRESS mode={mode} "
                    f"pages={completed}/{len(specs)}",
                    flush=True,
                )
        wall_s = time.perf_counter() - lane_started
    finally:
        for _ in processes:
            task_queue.put(None)
        shutdown_started = time.perf_counter()
        for process in processes:
            process.join(timeout=10.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        shutdown_s = time.perf_counter() - shutdown_started
        for shared in shared_blocks.values():
            shared.close()
            shared.unlink()

    total_bytes = sum(spec.payload_nbytes for spec in specs)
    delivery_sorted = sorted(delivery_values)
    return {
        "mode": mode,
        "run_id": run_id,
        "consumer": consumer,
        "workers": workers,
        "max_inflight": max_inflight,
        "pages": len(specs),
        "segments": sum(len(spec.segment_shapes) for spec in specs),
        "payload_bytes": total_bytes,
        "queue_array_bytes": total_bytes if mode.startswith("pickle_") else 0,
        "shared_memory_bytes": total_bytes if mode == "shared_memory_arena" else 0,
        "setup_s": setup_s,
        "wall_s": wall_s,
        "pages_per_s": len(specs) / wall_s,
        "payload_gib_per_s": total_bytes / (2**30) / wall_s,
        "producer_build_sum_s": sum(build_values),
        "producer_build_mean_s": statistics.fmean(build_values),
        "producer_build_max_s": max(build_values),
        "first_ready_s": first_ready_s,
        "last_ready_s": last_ready_s,
        "receive_wait_s": receive_wait_s,
        "ipc_delivery_sum_s": sum(delivery_values),
        "ipc_delivery_mean_s": statistics.fmean(delivery_values),
        "ipc_delivery_p50_s": delivery_sorted[len(delivery_sorted) // 2],
        "ipc_delivery_p95_s": delivery_sorted[
            min(len(delivery_sorted) - 1, int(len(delivery_sorted) * 0.95))
        ],
        "ipc_delivery_max_s": max(delivery_values),
        "consumer_s": consume_s,
        "parent_shm_allocate_s": shm_allocate_s,
        "worker_page_counts": worker_pages,
        "worker_peak_rss_kb": worker_peak_rss_kb,
        "shutdown_s": shutdown_s,
        "mismatches": mismatches,
        "parity": "PASS" if mismatches == 0 else "FAIL",
    }


def workload_summary(specs: list[PageSpec]) -> dict[str, Any]:
    page_bytes = sum(
        int(np.prod(spec.page_shape))
        for spec in specs
        if spec.page_shape is not None
    )
    crop_bytes = sum(
        sum(int(np.prod(shape)) for shape in spec.crop_shapes)
        for spec in specs
    )
    per_page = [spec.payload_nbytes for spec in specs]
    return {
        "pages": len(specs),
        "crops": sum(len(spec.crop_shapes) for spec in specs),
        "segments": sum(len(spec.segment_shapes) for spec in specs),
        "page_image_bytes": page_bytes,
        "crop_bytes": crop_bytes,
        "total_bytes": page_bytes + crop_bytes,
        "total_gib": (page_bytes + crop_bytes) / (2**30),
        "per_page_mean_mib": statistics.fmean(per_page) / (2**20),
        "per_page_max_mib": max(per_page) / (2**20),
    }


def main() -> None:
    args = parse_args()
    specs = load_page_specs(
        args.trace.expanduser().resolve(),
        args.input.expanduser().resolve(),
        limit_pages=args.limit_pages,
        include_page_image=not args.crop_only,
    )
    workload = workload_summary(specs)
    print("UNIREC_FRONTEND_IPC_WORKLOAD " + json.dumps(workload), flush=True)
    records = []
    run_id = 0
    for round_index in range(args.rounds):
        for mode in args.modes:
            print(
                f"UNIREC_FRONTEND_IPC_LANE_BEGIN mode={mode} "
                f"round={round_index}",
                flush=True,
            )
            record = run_lane(
                specs,
                mode=mode,
                run_id=run_id,
                workers=args.workers,
                max_inflight=args.max_inflight,
                consumer=args.consumer,
                timeout_s=args.timeout_s,
            )
            record["round"] = round_index
            records.append(record)
            print(
                "UNIREC_FRONTEND_IPC_LANE_END " + json.dumps(record),
                flush=True,
            )
            run_id += 1
    output = {
        "trace": str(args.trace.expanduser().resolve()),
        "input": str(args.input.expanduser().resolve()),
        "crop_only": args.crop_only,
        "workload": workload,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if any(record["parity"] != "PASS" for record in records):
        raise RuntimeError(f"one or more IPC lanes failed parity; see {args.output}")
    print(f"UNIREC_FRONTEND_IPC_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
