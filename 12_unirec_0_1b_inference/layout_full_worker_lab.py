#!/usr/bin/env python3
"""Compare persistent shared-thread and isolated-process B1 layout workers.

The coordinator sends only file paths.  Thread workers share one complete
PP-DocLayoutV2 processor/model/runtime.  Process workers each own one complete
runtime.  Every worker performs direct-RGB page decoding, B1 layout inference,
and postprocessing, then returns only compact timing and output hashes.  Worker
setup and graph warmup are excluded from measured wall.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import queue
import resource
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kornia_rs.image import Image as KorniaImage
from torchvision.io import ImageReadMode, decode_image

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_detector_torchair"
        ),
    )
    parser.add_argument("--offset", type=int, default=769)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--worker-counts", default="1,2,4,8")
    parser.add_argument("--modes", default="threads,processes")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.worker_counts = tuple(
        int(value) for value in args.worker_counts.split(",")
    )
    args.modes = tuple(value.strip() for value in args.modes.split(","))
    if not args.worker_counts or any(value < 1 for value in args.worker_counts):
        parser.error("--worker-counts must contain positive integers")
    if len(set(args.worker_counts)) != len(args.worker_counts):
        parser.error("--worker-counts must not contain duplicates")
    if not args.modes or any(
        value not in {"threads", "processes"} for value in args.modes
    ):
        parser.error("--modes accepts threads and/or processes")
    if args.offset < 0 or args.limit < 1 or args.rounds < 1:
        parser.error("--offset must be non-negative; --limit/--rounds positive")
    if args.worker_timeout_s <= 0:
        parser.error("--worker-timeout-s must be positive")
    return args


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def decode_rgb(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    encoded = path.read_bytes()
    read_s = time.perf_counter() - started
    started = time.perf_counter()
    if encoded.startswith(PNG_SIGNATURE):
        rgb = KorniaImage.decode(encoded, "RGB").data
    else:
        encoded_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        rgb = (
            decode_image(encoded_tensor, mode=ImageReadMode.RGB)
            .permute(1, 2, 0)
            .numpy()
        )
    decode_s = time.perf_counter() - started
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"unsupported decoded image: {rgb.shape} {rgb.dtype}")
    return rgb, {"file_read_s": read_s, "direct_rgb_decode_s": decode_s}


def result_digest(result: dict[str, Any]) -> str:
    payload = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_runtime(
    *,
    model_path: str | Path,
    cache_dir: str | Path,
    threshold: float,
) -> PPDocLayoutV2NpuAdapter:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    return PPDocLayoutV2NpuAdapter(
        model_path=model_path,
        device="npu:0",
        dtype="float32",
        threshold=threshold,
        profile_stages=True,
        execution="torchair",
        compile_cache_dir=cache_dir,
        batch_size=1,
    )


def warm_runtime(
    runtime: PPDocLayoutV2NpuAdapter,
    path: Path,
    *,
    threshold: float,
) -> None:
    rgb, _ = decode_rgb(path)
    # The adapter's historical contract is BGR.  A negative-stride channel
    # view avoids a full contiguous RGB->BGR copy; the adapter converts the
    # view back to RGB before its processor.
    runtime([rgb[..., ::-1]], threshold=threshold)
    runtime.reset_timing()


def run_shard(
    runtime: PPDocLayoutV2NpuAdapter,
    indexed_paths: list[tuple[int, str]],
    *,
    threshold: float,
    reset_runtime_timing: bool = True,
    collect_runtime_timing: bool = True,
) -> dict[str, Any]:
    if reset_runtime_timing:
        runtime.reset_timing()
    decode_timing = {"file_read_s": 0.0, "direct_rgb_decode_s": 0.0}
    outputs = []
    started_cpu_s = cpu_seconds()
    started = time.perf_counter()
    for page_index, path_string in indexed_paths:
        rgb, timing = decode_rgb(Path(path_string))
        for name, seconds in timing.items():
            decode_timing[name] += seconds
        result = runtime([rgb[..., ::-1]], threshold=threshold)[0]
        outputs.append(
            {
                "page_index": page_index,
                "box_count": len(result["boxes"]),
                "digest": result_digest(result),
            }
        )
    wall_s = time.perf_counter() - started
    process_cpu_s = cpu_seconds() - started_cpu_s
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "pages": len(indexed_paths),
        "wall_s": wall_s,
        "process_cpu_s": process_cpu_s,
        "average_cpu_cores": process_cpu_s / wall_s,
        "max_rss_kb": int(usage.ru_maxrss),
        "decode_stage_s": decode_timing,
        "outputs": outputs,
    }
    if collect_runtime_timing:
        result["layout_timing"] = runtime.timing_summary()
    return result


def make_shards(paths: list[Path], worker_count: int) -> list[list[tuple[int, str]]]:
    indexed = [(index, str(path)) for index, path in enumerate(paths)]
    return [indexed[worker_index::worker_count] for worker_index in range(worker_count)]


def aggregate_round(
    *,
    mode: str,
    worker_count: int,
    round_index: int,
    setup_wall_s: float,
    measured_wall_s: float,
    worker_results: list[dict[str, Any]],
    shared_runtime_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = sorted(
        [output for result in worker_results for output in result["outputs"]],
        key=lambda output: output["page_index"],
    )
    page_count = sum(int(result["pages"]) for result in worker_results)
    # RUSAGE_SELF is process-wide.  Every thread observes nearly the same
    # process CPU interval, so summing thread samples would count it once per
    # thread.  Independent process samples are additive.
    process_cpu_s = (
        max(float(result["process_cpu_s"]) for result in worker_results)
        if mode == "threads"
        else sum(float(result["process_cpu_s"]) for result in worker_results)
    )
    result = {
        "mode": mode,
        "workers": worker_count,
        "round": round_index,
        "pages": page_count,
        "setup_wall_s": setup_wall_s,
        "measured_wall_s": measured_wall_s,
        "pages_per_s": page_count / measured_wall_s,
        "worker_wall_s": [float(result["wall_s"]) for result in worker_results],
        "total_process_cpu_s": process_cpu_s,
        "average_cpu_cores": process_cpu_s / measured_wall_s,
        "summed_max_rss_kb": (
            max(int(result["max_rss_kb"]) for result in worker_results)
            if mode == "threads"
            else sum(int(result["max_rss_kb"]) for result in worker_results)
        ),
        "box_count": sum(int(output["box_count"]) for output in outputs),
        "output_digests": [output["digest"] for output in outputs],
        "worker_results": worker_results,
    }
    if shared_runtime_timing is not None:
        result["shared_runtime_timing"] = shared_runtime_timing
    return result


def run_thread_pool(
    paths: list[Path],
    *,
    model_path: Path,
    cache_dir: Path,
    threshold: float,
    worker_count: int,
    rounds: int,
) -> tuple[float, list[dict[str, Any]]]:
    setup_started = time.perf_counter()
    shards = make_shards(paths, worker_count)
    runtime = create_runtime(
        model_path=model_path,
        cache_dir=cache_dir,
        threshold=threshold,
    )
    warm_runtime(runtime, paths[0], threshold=threshold)
    print(
        f"LAYOUT_FULL_WORKER setup mode=threads shared_runtime=1 "
        f"workers={worker_count}",
        flush=True,
    )
    setup_wall_s = time.perf_counter() - setup_started

    records = []
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="unirec-layout-full",
    ) as executor:
        for round_index in range(rounds):
            runtime.reset_timing()
            measured_started = time.perf_counter()
            futures = [
                executor.submit(
                    run_shard,
                    runtime,
                    shard,
                    threshold=threshold,
                    reset_runtime_timing=False,
                    collect_runtime_timing=False,
                )
                for shard in shards
            ]
            worker_results = [future.result() for future in futures]
            measured_wall_s = time.perf_counter() - measured_started
            records.append(
                aggregate_round(
                    mode="threads",
                    worker_count=worker_count,
                    round_index=round_index,
                    setup_wall_s=setup_wall_s,
                    measured_wall_s=measured_wall_s,
                    worker_results=worker_results,
                    shared_runtime_timing=runtime.timing_summary(),
                )
            )
    del runtime
    gc.collect()
    torch.npu.empty_cache()
    return setup_wall_s, records


def process_worker(
    worker_index: int,
    model_path: str,
    cache_dir: str,
    threshold: float,
    warmup_path: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        runtime = create_runtime(
            model_path=model_path,
            cache_dir=cache_dir,
            threshold=threshold,
        )
        warm_runtime(runtime, Path(warmup_path), threshold=threshold)
        result_queue.put({"status": "ready", "worker": worker_index})
        while True:
            task = task_queue.get()
            if task is None:
                break
            round_index, page_index, path_string = task
            result = run_shard(
                runtime,
                [(page_index, path_string)],
                threshold=threshold,
                reset_runtime_timing=False,
                collect_runtime_timing=False,
            )
            result_queue.put(
                {
                    "status": "ok",
                    "worker": worker_index,
                    "round": round_index,
                    "page_index": page_index,
                    "result": result,
                }
            )
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "worker": worker_index,
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


def receive(result_queue: Any, timeout_s: float) -> dict[str, Any]:
    try:
        return result_queue.get(timeout=timeout_s)
    except queue.Empty as exception:
        raise TimeoutError(f"worker pool was silent for {timeout_s}s") from exception


def aggregate_dynamic_worker_results(
    messages: list[dict[str, Any]],
    worker_count: int,
) -> list[dict[str, Any]]:
    """Combine page-sized results without losing per-worker load balance."""
    worker_results = [
        {
            "pages": 0,
            "wall_s": 0.0,
            "process_cpu_s": 0.0,
            "average_cpu_cores": 0.0,
            "max_rss_kb": 0,
            "decode_stage_s": {
                "file_read_s": 0.0,
                "direct_rgb_decode_s": 0.0,
            },
            "outputs": [],
        }
        for _ in range(worker_count)
    ]
    for message in messages:
        target = worker_results[int(message["worker"])]
        page_result = message["result"]
        target["pages"] += int(page_result["pages"])
        target["wall_s"] += float(page_result["wall_s"])
        target["process_cpu_s"] += float(page_result["process_cpu_s"])
        target["max_rss_kb"] = max(
            int(target["max_rss_kb"]),
            int(page_result["max_rss_kb"]),
        )
        for name, seconds in page_result["decode_stage_s"].items():
            target["decode_stage_s"][name] += float(seconds)
        target["outputs"].extend(page_result["outputs"])
    for target in worker_results:
        target["average_cpu_cores"] = (
            target["process_cpu_s"] / target["wall_s"]
            if target["wall_s"]
            else 0.0
        )
    return worker_results


def run_process_pool(
    paths: list[Path],
    *,
    model_path: Path,
    cache_dir: Path,
    threshold: float,
    worker_count: int,
    rounds: int,
    timeout_s: float,
) -> tuple[float, list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queue = context.Queue()
    processes = [
        context.Process(
            target=process_worker,
            args=(
                worker_index,
                str(model_path),
                str(cache_dir),
                threshold,
                str(paths[worker_index % len(paths)]),
                task_queue,
                result_queue,
            ),
            name=f"unirec-layout-full-{worker_index}",
        )
        for worker_index in range(worker_count)
    ]
    setup_started = time.perf_counter()
    for process in processes:
        process.start()
    try:
        ready = [receive(result_queue, timeout_s) for _ in processes]
        errors = [message for message in ready if message["status"] != "ready"]
        if errors:
            raise RuntimeError(f"process setup failed: {errors}")
        setup_wall_s = time.perf_counter() - setup_started
        records = []
        for round_index in range(rounds):
            measured_started = time.perf_counter()
            for page_index, path in enumerate(paths):
                task_queue.put((round_index, page_index, str(path)))
            messages = []
            progress_step = max(1, len(paths) // 10)
            while len(messages) < len(paths):
                message = receive(result_queue, timeout_s)
                if message["status"] != "ok":
                    raise RuntimeError(f"process measurement failed: {message}")
                if int(message["round"]) != round_index:
                    raise RuntimeError(
                        "unexpected dynamic worker round: "
                        f"{message['round']} != {round_index}"
                    )
                messages.append(message)
                if len(messages) % progress_step == 0 or len(messages) == len(paths):
                    print(
                        f"LAYOUT_FULL_WORKER progress mode=processes "
                        f"workers={worker_count} round={round_index} "
                        f"pages={len(messages)}/{len(paths)}",
                        flush=True,
                    )
            measured_wall_s = time.perf_counter() - measured_started
            worker_results = aggregate_dynamic_worker_results(
                messages,
                worker_count,
            )
            records.append(
                aggregate_round(
                    mode="processes",
                    worker_count=worker_count,
                    round_index=round_index,
                    setup_wall_s=setup_wall_s,
                    measured_wall_s=measured_wall_s,
                    worker_results=worker_results,
                )
            )
    finally:
        for _ in processes:
            task_queue.put(None)
        for process in processes:
            process.join(timeout=10.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
    return setup_wall_s, records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    baseline_digests = records[0]["output_digests"]
    for mode in sorted({record["mode"] for record in records}):
        summary[mode] = {}
        for workers in sorted(
            {record["workers"] for record in records if record["mode"] == mode}
        ):
            rows = [
                record
                for record in records
                if record["mode"] == mode and record["workers"] == workers
            ]
            walls = [float(row["measured_wall_s"]) for row in rows]
            rates = [float(row["pages_per_s"]) for row in rows]
            summary[mode][str(workers)] = {
                "setup_wall_s": rows[0]["setup_wall_s"],
                "wall_s_rounds": walls,
                "wall_s_median": statistics.median(walls),
                "pages_per_s_rounds": rates,
                "pages_per_s_median": statistics.median(rates),
                "average_cpu_cores_median": statistics.median(
                    float(row["average_cpu_cores"]) for row in rows
                ),
                "summed_max_rss_kb_median": statistics.median(
                    int(row["summed_max_rss_kb"]) for row in rows
                ),
                "box_count": rows[0]["box_count"],
                "all_rounds_match_baseline": all(
                    row["output_digests"] == baseline_digests for row in rows
                ),
            }
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
    model_path = args.model_path.expanduser().resolve()
    cache_dir = args.compile_cache_dir.expanduser().resolve()
    print(
        f"LAYOUT_FULL_WORKER setup pages={len(paths)} modes={args.modes} "
        f"workers={args.worker_counts} rounds={args.rounds}",
        flush=True,
    )

    records = []
    for mode in args.modes:
        for worker_count in args.worker_counts:
            print(
                f"LAYOUT_FULL_WORKER starting mode={mode} workers={worker_count}",
                flush=True,
            )
            if mode == "threads":
                _, current = run_thread_pool(
                    paths,
                    model_path=model_path,
                    cache_dir=cache_dir,
                    threshold=args.threshold,
                    worker_count=worker_count,
                    rounds=args.rounds,
                )
            else:
                _, current = run_process_pool(
                    paths,
                    model_path=model_path,
                    cache_dir=cache_dir,
                    threshold=args.threshold,
                    worker_count=worker_count,
                    rounds=args.rounds,
                    timeout_s=args.worker_timeout_s,
                )
            records.extend(current)
            for record in current:
                print(
                    "LAYOUT_FULL_WORKER result "
                    + json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {"worker_results", "output_digests"}
                        }
                    ),
                    flush=True,
                )

    report = {
        "config": {
            "model_path": str(model_path),
            "input": str(args.input.expanduser().resolve()),
            "compile_cache_dir": str(cache_dir),
            "offset": args.offset,
            "limit": args.limit,
            "worker_counts": list(args.worker_counts),
            "modes": list(args.modes),
            "rounds": args.rounds,
            "threshold": args.threshold,
            "layout_batch_size": 1,
            "execution": "torchair",
            "worker_ownership": {
                "threads": "one_shared_runtime_for_all_threads",
                "processes": "one_complete_runtime_per_process",
            },
            "process_scheduling": "dynamic_shared_filepath_queue",
            "coordinator_payload": "file_path_in_boxes_and_timing_out",
        },
        "summary": summarize(records),
        "rounds": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("LAYOUT_FULL_WORKER summary " + json.dumps(report["summary"]), flush=True)
    print(f"LAYOUT_FULL_WORKER done output={output}", flush=True)


if __name__ == "__main__":
    main()
