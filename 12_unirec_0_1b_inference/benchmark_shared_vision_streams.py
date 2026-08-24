#!/usr/bin/env python3
"""Measure whether one K20 runtime can preserve four-lane NPU concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")

import torch
import torch_npu  # noqa: F401

from host_memory_diagnostics import process_snapshot
from modeling_optimized_unirec import OptimizedUniRecRunner
from tbe_compiler_lifecycle import deinitialize_after_warmup
from vision_bucket_presets import resolve_vision_bucket_specs
from vision_full_batch import BucketedFullVisionRuntime


REPRESENTATIVE128_BUCKET_CALLS = {
    "192x64_b4": 142,
    "512x64_b4": 149,
    "960x64_b4": 156,
    "960x128_b2": 79,
    "960x256_b1": 44,
    "960x384_b1": 37,
    "512x128_b4": 55,
    "960x192_b1": 50,
    "960x512_b1": 37,
    "448x192_b2": 48,
    "576x256_b2": 37,
    "320x320_b2": 20,
    "448x384_b2": 32,
    "448x576_b1": 19,
    "512x768_b1": 8,
    "960x896_b1": 9,
    "960x704_b1": 28,
    "960x1152_b1": 7,
    "960x1344_b1": 4,
    "128x1408_b1": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bucket-preset", default="310p_k20_l4")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def build_inputs(
    runtime: BucketedFullVisionRuntime,
    *,
    lanes: int,
) -> list[dict[str, tuple[torch.Tensor, ...]]]:
    device = torch.device(runtime.runner.device)
    inputs = []
    with torch.inference_mode(False):
        for lane in range(lanes):
            lane_inputs = {}
            for spec in runtime.specs:
                pixels = torch.full(
                    (spec.batch_size, 3, spec.height, spec.width),
                    fill_value=float(lane + 1) / float(lanes + 1),
                    dtype=runtime.runner.dtype,
                    device=device,
                )
                masks = tuple(
                    torch.ones(
                        (
                            spec.batch_size,
                            1,
                            spec.height // factor,
                            spec.width // factor,
                        ),
                        dtype=runtime.runner.dtype,
                        device=device,
                    )
                    for factor in (2, 4, 8, 16, 32)
                )
                lane_inputs[spec.key] = (pixels, *masks)
            inputs.append(lane_inputs)
    torch.npu.synchronize()
    return inputs


def run_sequential(
    runtime: BucketedFullVisionRuntime,
    calls_by_lane: list[list[str]],
    inputs_by_lane: list[dict[str, tuple[torch.Tensor, ...]]],
    streams: list[Any],
) -> float:
    started = time.perf_counter()
    for rank, stream in enumerate(streams):
        with torch.inference_mode(), torch.npu.stream(stream):
            for key in calls_by_lane[rank]:
                runtime.compiled[key](*inputs_by_lane[rank][key])
        stream.synchronize()
    return time.perf_counter() - started


def run_concurrent(
    runtime: BucketedFullVisionRuntime,
    calls_by_lane: list[list[str]],
    inputs_by_lane: list[dict[str, tuple[torch.Tensor, ...]]],
    streams: list[Any],
) -> float:
    barrier = threading.Barrier(len(streams) + 1)

    def lane(rank: int) -> None:
        torch.npu.set_device(torch.device(runtime.runner.device))
        stream = streams[rank]
        with torch.inference_mode(), torch.npu.stream(stream):
            barrier.wait()
            for key in calls_by_lane[rank]:
                runtime.compiled[key](*inputs_by_lane[rank][key])
        stream.synchronize()

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(lane, rank) for rank in range(len(streams))]
        barrier.wait()
        started = time.perf_counter()
        for future in futures:
            future.result()
    return time.perf_counter() - started


def parity_probe(
    runtime: BucketedFullVisionRuntime,
    *,
    keys_by_lane: list[str],
    inputs_by_lane: list[dict[str, tuple[torch.Tensor, ...]]],
    streams: list[Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    references = []
    for rank, (key, stream) in enumerate(zip(keys_by_lane, streams)):
        with torch.inference_mode(), torch.npu.stream(stream):
            output = runtime.compiled[key](*inputs_by_lane[rank][key])
        stream.synchronize()
        references.append(output.cpu())

    barrier = threading.Barrier(len(streams) + 1)

    def lane(rank: int) -> torch.Tensor:
        key = keys_by_lane[rank]
        stream = streams[rank]
        with torch.inference_mode(), torch.npu.stream(stream):
            barrier.wait()
            output = runtime.compiled[key](*inputs_by_lane[rank][key])
        stream.synchronize()
        return output.cpu()

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(lane, rank) for rank in range(len(streams))]
        barrier.wait()
        concurrent = [future.result() for future in futures]
    for key, reference, candidate in zip(
        keys_by_lane,
        references,
        concurrent,
    ):
        diff = (candidate.float() - reference.float()).abs()
        report[key] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "exact": torch.equal(reference, candidate),
        }
    return report


def assign_keys_to_lanes(
    runtime: BucketedFullVisionRuntime,
    *,
    lanes: int,
) -> list[list[str]]:
    """Greedily balance physical pixel work while keeping each graph on one stream."""
    specs = {spec.key: spec for spec in runtime.specs}
    work = {
        key: count
        * specs[key].batch_size
        * specs[key].width
        * specs[key].height
        for key, count in REPRESENTATIVE128_BUCKET_CALLS.items()
    }
    assignments = [[] for _ in range(lanes)]
    totals = [0 for _ in range(lanes)]
    for key in sorted(work, key=work.get, reverse=True):
        lane = min(range(lanes), key=totals.__getitem__)
        assignments[lane].append(key)
        totals[lane] += work[key]
    return assignments


def main() -> None:
    args = parse_args()
    if args.lanes < 1 or args.repeats < 1:
        raise ValueError("lanes and repeats must be positive")
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)

    runner = OptimizedUniRecRunner(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=args.cache_dir,
    )
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=resolve_vision_bucket_specs(args.bucket_preset),
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name=args.bucket_preset,
    )
    inputs_by_lane = build_inputs(runtime, lanes=args.lanes)
    streams = [torch.npu.Stream() for _ in range(args.lanes)]
    keys = {spec.key for spec in runtime.specs}
    if set(REPRESENTATIVE128_BUCKET_CALLS) != keys:
        raise RuntimeError(
            "representative histogram does not match bucket preset: "
            f"missing={sorted(keys - set(REPRESENTATIVE128_BUCKET_CALLS))} "
            f"extra={sorted(set(REPRESENTATIVE128_BUCKET_CALLS) - keys)}"
        )

    keys_by_lane = assign_keys_to_lanes(runtime, lanes=args.lanes)
    # Bind each graph once to its permanent stream and establish runtime state.
    run_concurrent(
        runtime,
        keys_by_lane,
        inputs_by_lane,
        streams,
    )
    deinitialize_after_warmup("shared_vision_streams_warmup_complete")

    calls = [
        key
        for key, count in REPRESENTATIVE128_BUCKET_CALLS.items()
        for _ in range(count)
    ]
    calls *= args.repeats
    calls_by_lane = []
    for rank, lane_keys in enumerate(keys_by_lane):
        lane_calls = [
            key
            for key in lane_keys
            for _ in range(REPRESENTATIVE128_BUCKET_CALLS[key] * args.repeats)
        ]
        random.Random(rank).shuffle(lane_calls)
        calls_by_lane.append(lane_calls)

    sequential_s = run_sequential(
        runtime,
        calls_by_lane,
        inputs_by_lane,
        streams,
    )
    concurrent_s = run_concurrent(
        runtime,
        calls_by_lane,
        inputs_by_lane,
        streams,
    )
    probe_keys = [lane_keys[0] for lane_keys in keys_by_lane]
    parity = parity_probe(
        runtime,
        keys_by_lane=probe_keys,
        inputs_by_lane=inputs_by_lane,
        streams=streams,
    )
    report = {
        "status": "pass" if all(row["exact"] for row in parity.values()) else "fail",
        "chip": torch_npu.npu.get_device_name(0),
        "bucket_preset": args.bucket_preset,
        "lanes": args.lanes,
        "call_count": len(calls),
        "sequential_s": sequential_s,
        "concurrent_s": concurrent_s,
        "speedup": sequential_s / concurrent_s,
        "sequential_calls_per_s": len(calls) / sequential_s,
        "concurrent_calls_per_s": len(calls) / concurrent_s,
        "lane_call_counts": [len(lane_calls) for lane_calls in calls_by_lane],
        "keys_by_lane": keys_by_lane,
        "parity": parity,
        "host_memory": process_snapshot(),
    }
    print("UNIREC_SHARED_VISION_STREAMS " + json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
