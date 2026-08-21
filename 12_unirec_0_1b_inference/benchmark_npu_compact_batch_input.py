#!/usr/bin/env python3
"""Compare current and compact UniRec bucket-input preparation on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu


BUCKET_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)_b(?P<batch>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    physical_device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not physical_device:
        parser.error("source npu-setup before running this benchmark")
    if physical_device == "5":
        parser.error("physical NPU 5 is excluded")
    return args


def find_aggregate_vision_batching(value: Any) -> dict[str, Any]:
    candidates = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            batching = node.get("vision_batching")
            if isinstance(batching, dict) and "bucket_calls" in batching:
                candidates.append(batching)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if not candidates:
        raise ValueError("summary contains no vision_batching bucket statistics")
    return max(candidates, key=lambda candidate: int(candidate.get("crops", 0)))


def parse_bucket_calls(
    batching: dict[str, Any],
) -> list[tuple[str, int, int, int, int]]:
    result = []
    for key, calls_value in batching["bucket_calls"].items():
        match = BUCKET_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected bucket key: {key}")
        result.append(
            (
                key,
                int(match.group("batch")),
                int(match.group("height")),
                int(match.group("width")),
                int(calls_value),
            )
        )
    return sorted(result)


def read_fallback_calls(
    crops_jsonl: Path,
) -> list[tuple[str, int, int, int, int]]:
    counts: dict[tuple[int, int], int] = {}
    with crops_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prefill = record["prefill"]
            if prefill.get("vision_bucket") not in {None, "fallback_eager"}:
                continue
            width, height = prefill["prep"]["processed_image_size"]
            shape = (int(width), int(height))
            counts[shape] = counts.get(shape, 0) + 1
    return [
        (f"fallback_{width}x{height}", 1, height, width, count)
        for (width, height), count in sorted(counts.items())
    ]


def normalize_to_fp16(device_pixels: torch.Tensor) -> torch.Tensor:
    output = device_pixels.to(torch.float32)
    output.mul_(np.float32(2.0 / 255.0))
    output.sub_(np.float32(1.0))
    return output.to(torch.float16)


def main() -> None:
    args = parse_args()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    summary = json.loads(args.summary.expanduser().resolve().read_text())
    batching = find_aggregate_vision_batching(summary)
    bucket_calls = parse_bucket_calls(batching)
    crops_jsonl = args.summary.expanduser().resolve().parent / "crops.jsonl"
    fallback_calls = read_fallback_calls(crops_jsonl)
    calls = [*bucket_calls, *fallback_calls]
    expected_fallback_rows = int(batching.get("fallback_rows", 0))
    actual_fallback_rows = sum(count for *_shape, count in fallback_calls)
    if actual_fallback_rows != expected_fallback_rows:
        raise ValueError(
            f"fallback crop mismatch: {actual_fallback_rows} != "
            f"{expected_fallback_rows}"
        )
    device = torch.device(args.device)

    float32_chw = {
        key: np.zeros((batch, 3, height, width), dtype=np.float32)
        for key, batch, height, width, _count in calls
    }
    uint8_chw = {
        key: np.zeros((batch, 3, height, width), dtype=np.uint8)
        for key, batch, height, width, _count in calls
    }
    uint8_hwc = {
        key: np.zeros((batch, height, width, 3), dtype=np.uint8)
        for key, batch, height, width, _count in calls
    }

    def current_float32_chw(key: str) -> torch.Tensor:
        return torch.from_numpy(float32_chw[key]).to(device, dtype=torch.float16)

    def compact_uint8_chw(key: str) -> torch.Tensor:
        pixels = torch.from_numpy(uint8_chw[key]).to(device)
        return normalize_to_fp16(pixels)

    def compact_uint8_hwc(key: str) -> torch.Tensor:
        pixels = torch.from_numpy(uint8_hwc[key]).to(device)
        pixels = pixels.permute(0, 3, 1, 2)
        return normalize_to_fp16(pixels).contiguous()

    modes: dict[str, tuple[Callable[[str], torch.Tensor], int]] = {
        "current_float32_chw": (current_float32_chw, 4),
        "compact_uint8_chw": (compact_uint8_chw, 1),
        "compact_uint8_hwc": (compact_uint8_hwc, 1),
    }
    results = {}
    total_elements = sum(
        batch * 3 * height * width * count
        for _key, batch, height, width, count in calls
    )
    for name, (operation, source_itemsize) in modes.items():
        with torch.inference_mode():
            for key, _batch, _height, _width, _count in calls:
                warmup = operation(key)
                del warmup
            torch.npu.synchronize()
            round_wall_s = []
            for _round_index in range(args.rounds):
                torch.npu.synchronize()
                started = time.perf_counter()
                for key, _batch, _height, _width, count in calls:
                    for _call_index in range(count):
                        output = operation(key)
                        del output
                torch.npu.synchronize()
                round_wall_s.append(time.perf_counter() - started)
        median_s = statistics.median(round_wall_s)
        results[name] = {
            "round_wall_s": round_wall_s,
            "median_s": median_s,
            "source_bytes": total_elements * source_itemsize,
            "source_gb_per_s": total_elements * source_itemsize / 1e9 / median_s,
        }
        torch.npu.empty_cache()

    source = np.arange(256, dtype=np.uint8)
    reference = source.astype(np.float32)
    reference *= np.float32(2.0 / 255.0)
    reference -= np.float32(1.0)
    reference = reference.astype(np.float16)
    actual = normalize_to_fp16(torch.from_numpy(source).to(device)).cpu().numpy()
    value_parity = {
        "all_exact": bool(np.array_equal(actual, reference)),
        "different_values": int(np.count_nonzero(actual != reference)),
        "total_values": 256,
    }
    report = {
        "status": "ok",
        "physical_npu": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "npu_jit_compile": False,
        "device": str(device),
        "summary": str(args.summary),
        "bucket_calls": {key: count for key, *_shape, count in bucket_calls},
        "bucket_call_count": sum(count for *_shape, count in bucket_calls),
        "bucket_real_rows": batching.get("bucket_real_rows"),
        "bucket_physical_rows": batching.get("bucket_physical_rows"),
        "fallback_rows_included": actual_fallback_rows,
        "fallback_shape_count": len(fallback_calls),
        "total_input_call_count": sum(count for *_shape, count in calls),
        "padded_source_elements": total_elements,
        "normalization_fp16_parity": value_parity,
        "timing_scope": (
            "pageable host transfer, device layout conversion where applicable, "
            "float32 normalization, fp16 cast, final device synchronization"
        ),
        "results": results,
    }
    print("UNIREC_NPU_COMPACT_BATCH_INPUT " + json.dumps(report, sort_keys=True))
    if not value_parity["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
