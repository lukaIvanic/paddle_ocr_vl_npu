#!/usr/bin/env python3
"""Compare production CPU-mask H2D with eager NPU mask construction.

The lab replays the exact compiled vision-bucket calls recorded in a production
prefill trace.  It does not load or compile the UniRec vision encoder.  Both
lanes produce the six tensors consumed by that unchanged encoder graph:

* pixel mask at the bucket canvas resolution;
* masks at factors 2, 4, 8, 16, and 32.

The control transfers the current CPU-built masks.  The candidate transfers a
small table of valid per-level extents and constructs the masks with eager NPU
comparisons outside the encoder graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch_npu


SCHEMA = "unirec_vision_mask_h2d_lab_v1"
FACTORS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class CallSpec:
    bucket: str
    batch_size: int
    height: int
    width: int
    dimensions: tuple[tuple[int, int], ...]  # (width, height), padded to B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup-replays", type=int, default=2)
    parser.add_argument("--measured-replays", type=int, default=5)
    return parser.parse_args()


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "sum": sum(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "min": min(samples),
        "max": max(samples),
    }


def read_calls(path: Path) -> list[CallSpec]:
    calls = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "vision_bucket_call":
                continue
            batch_size, channels, height, width = map(
                int, event["physical_input_shape"]
            )
            if channels != 3:
                raise ValueError(f"unexpected vision input shape: {event}")
            dimensions = [
                tuple(map(int, member["processed_image_size"]))
                for member in event["members"]
            ]
            dimensions.extend([(0, 0)] * (batch_size - len(dimensions)))
            calls.append(
                CallSpec(
                    bucket=str(event["bucket"]),
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    dimensions=tuple(dimensions),
                )
            )
    if not calls:
        raise ValueError(f"no vision_bucket_call events in {path}")
    return calls


def make_host_masks(spec: CallSpec) -> tuple[np.ndarray, ...]:
    masks = []
    for factor in FACTORS:
        dtype = np.uint8 if factor == 1 else np.float16
        mask = np.zeros(
            (
                spec.batch_size,
                1,
                spec.height // factor,
                spec.width // factor,
            ),
            dtype=dtype,
        )
        for row, (width, height) in enumerate(spec.dimensions):
            mask[row, :, : height // factor, : width // factor] = 1
        masks.append(mask)
    return tuple(masks)


def make_level_extents(spec: CallSpec) -> np.ndarray:
    extents = np.zeros((spec.batch_size, len(FACTORS), 2), dtype=np.int32)
    for row, (width, height) in enumerate(spec.dimensions):
        for level, factor in enumerate(FACTORS):
            extents[row, level] = (height // factor, width // factor)
    return extents


class MaskLanes:
    def __init__(self, calls: Sequence[CallSpec], *, device: str) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.host_masks = [make_host_masks(spec) for spec in calls]
        self.level_extents = [make_level_extents(spec) for spec in calls]
        self.coordinates: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for spec in calls:
            for factor in FACTORS:
                shape = (spec.height // factor, spec.width // factor)
                if shape in self.coordinates:
                    continue
                height, width = shape
                rows = torch.arange(
                    height, dtype=torch.int32, device=self.device
                ).view(1, 1, height, 1)
                columns = torch.arange(
                    width, dtype=torch.int32, device=self.device
                ).view(1, 1, 1, width)
                self.coordinates[shape] = (rows, columns)

    def cpu_mask_h2d(self, index: int, _spec: CallSpec) -> tuple[torch.Tensor, ...]:
        masks = self.host_masks[index]
        return (
            torch.from_numpy(masks[0]).to(self.device, dtype=self.dtype),
            *(torch.from_numpy(mask).to(self.device) for mask in masks[1:]),
        )

    def npu_from_extents(self, index: int, spec: CallSpec) -> tuple[torch.Tensor, ...]:
        extents = torch.from_numpy(self.level_extents[index]).to(self.device)
        masks = []
        for level, factor in enumerate(FACTORS):
            rows, columns = self.coordinates[
                (spec.height // factor, spec.width // factor)
            ]
            valid_height = extents[:, level, 0].view(spec.batch_size, 1, 1, 1)
            valid_width = extents[:, level, 1].view(spec.batch_size, 1, 1, 1)
            masks.append(
                ((rows < valid_height) & (columns < valid_width)).to(self.dtype)
            )
        return tuple(masks)


def validate_parity(calls: Sequence[CallSpec], lanes: MaskLanes) -> dict[str, Any]:
    selected = []
    seen_buckets = set()
    for index, spec in enumerate(calls):
        if spec.bucket in seen_buckets:
            continue
        seen_buckets.add(spec.bucket)
        selected.append(index)
    mismatches = []
    for index in selected:
        spec = calls[index]
        control = lanes.cpu_mask_h2d(index, spec)
        candidate = lanes.npu_from_extents(index, spec)
        for factor, expected, actual in zip(FACTORS, control, candidate):
            if not torch.equal(expected, actual):
                mismatches.append({"bucket": spec.bucket, "factor": factor})
    torch.npu.synchronize(lanes.device)
    return {
        "checked_buckets": len(selected),
        "checked_tensors": len(selected) * len(FACTORS),
        "mismatches": mismatches,
        "exact": not mismatches,
    }


def run_sequence(
    calls: Sequence[CallSpec],
    operation: Callable[[int, CallSpec], tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
) -> dict[str, float]:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    last = None
    for index, spec in enumerate(calls):
        last = operation(index, spec)
    end.record()
    host_submit_s = time.perf_counter() - wall_started
    end.synchronize()
    wall_s = time.perf_counter() - wall_started
    if last is None:
        raise AssertionError("empty replay")
    return {
        "device_s": float(start.elapsed_time(end)) / 1000.0,
        "host_submit_s": host_submit_s,
        "wall_s": wall_s,
    }


def benchmark_lane(
    calls: Sequence[CallSpec],
    operation: Callable[[int, CallSpec], tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    warmup_replays: int,
    measured_replays: int,
) -> dict[str, Any]:
    for _ in range(warmup_replays):
        run_sequence(calls, operation, device=device)
    samples = [
        run_sequence(calls, operation, device=device)
        for _ in range(measured_replays)
    ]
    return {
        name: distribution([sample[name] for sample in samples])
        for name in ("device_s", "host_submit_s", "wall_s")
    }


def main() -> None:
    args = parse_args()
    if args.warmup_replays < 1 or args.measured_replays < 1:
        raise ValueError("warmup and measured replay counts must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(args.device)
    calls = read_calls(args.iterations)
    lanes = MaskLanes(calls, device=args.device)
    parity = validate_parity(calls, lanes)
    if not parity["exact"]:
        raise AssertionError(f"NPU-generated masks failed parity: {parity}")

    current_bytes = sum(
        sum(mask.nbytes for mask in masks) for masks in lanes.host_masks
    )
    extent_bytes = sum(extents.nbytes for extents in lanes.level_extents)
    control = benchmark_lane(
        calls,
        lanes.cpu_mask_h2d,
        device=lanes.device,
        warmup_replays=args.warmup_replays,
        measured_replays=args.measured_replays,
    )
    candidate = benchmark_lane(
        calls,
        lanes.npu_from_extents,
        device=lanes.device,
        warmup_replays=args.warmup_replays,
        measured_replays=args.measured_replays,
    )
    output = {
        "schema": SCHEMA,
        "iterations": str(args.iterations),
        "device": str(args.device),
        "call_count": len(calls),
        "bucket_count": len({spec.bucket for spec in calls}),
        "warmup_replays": args.warmup_replays,
        "measured_replays": args.measured_replays,
        "parity": parity,
        "payload_bytes": {
            "current_six_masks": current_bytes,
            "npu_level_extents": extent_bytes,
            "removed": current_bytes - extent_bytes,
            "reduction_ratio": 1.0 - extent_bytes / current_bytes,
        },
        "lanes": {
            "cpu_mask_h2d": control,
            "npu_from_extents": candidate,
        },
        "candidate_speedup": {
            name: float(control[name]["p50"]) / float(candidate[name]["p50"])
            for name in ("device_s", "host_submit_s", "wall_s")
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "UNIREC_VISION_MASK_H2D_LAB PASS "
        f"calls={len(calls)} buckets={output['bucket_count']} "
        f"parity_exact={parity['exact']} "
        f"payload_reduction={output['payload_bytes']['reduction_ratio']:.6f}"
    )
    for lane, result in output["lanes"].items():
        print(
            "UNIREC_VISION_MASK_H2D_LANE "
            f"name={lane} device_p50_s={result['device_s']['p50']:.6f} "
            f"host_submit_p50_s={result['host_submit_s']['p50']:.6f} "
            f"wall_p50_s={result['wall_s']['p50']:.6f}"
        )
    print(
        "UNIREC_VISION_MASK_H2D_SPEEDUP "
        f"device={output['candidate_speedup']['device_s']:.6f} "
        f"host_submit={output['candidate_speedup']['host_submit_s']:.6f} "
        f"wall={output['candidate_speedup']['wall_s']:.6f}"
    )
    print(f"UNIREC_VISION_MASK_H2D_JSON={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
