#!/usr/bin/env python3
"""Benchmark CPU page bytes to PP-DocLayoutV2 input tensors.

The two lanes consume the same files sequentially and stop after producing the
CPU ``[1, 3, 800, 800]`` float32 tensor.  They do not copy to NPU or execute
the layout model.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from kornia_rs.image import Image as KorniaImage
from torchvision.io import ImageReadMode, decode_image
from transformers import AutoImageProcessor


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-pages", type=int, default=4)
    parser.add_argument("--parity-pages", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.rounds < 1:
        parser.error("--offset must be non-negative; --limit/--rounds positive")
    if args.warmup_pages < 0 or args.parity_pages < 0:
        parser.error("--warmup-pages/--parity-pages must be non-negative")
    return args


def read_bytes(path: Path) -> tuple[bytes, float]:
    started = time.perf_counter()
    encoded = path.read_bytes()
    return encoded, time.perf_counter() - started


def regular_prepare(
    processor: Any,
    encoded: bytes,
) -> tuple[torch.Tensor, dict[str, float]]:
    started = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    decode_s = time.perf_counter() - started
    if bgr is None:
        raise RuntimeError("OpenCV failed to decode image")

    started = time.perf_counter()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    color_s = time.perf_counter() - started

    started = time.perf_counter()
    pixel_values = processor(images=[rgb], return_tensors="pt")["pixel_values"]
    processor_s = time.perf_counter() - started
    return pixel_values, {
        "decode_s": decode_s,
        "color_conversion_s": color_s,
        "processor_s": processor_s,
    }


def optimized_prepare(
    processor: Any,
    encoded: bytes,
) -> tuple[torch.Tensor, dict[str, float]]:
    started = time.perf_counter()
    if encoded.startswith(PNG_SIGNATURE):
        rgb_hwc = KorniaImage.decode(encoded, "RGB").data
        rgb_chw = torch.from_numpy(rgb_hwc).permute(2, 0, 1)
    else:
        encoded_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        rgb_chw = decode_image(encoded_tensor, mode=ImageReadMode.RGB)
    decode_s = time.perf_counter() - started

    started = time.perf_counter()
    pixel_values = processor(images=[rgb_chw], return_tensors="pt")["pixel_values"]
    processor_s = time.perf_counter() - started
    return pixel_values, {
        "decode_s": decode_s,
        "color_conversion_s": 0.0,
        "processor_s": processor_s,
    }


LANES = {
    "regular_opencv": regular_prepare,
    "optimized_direct_rgb": optimized_prepare,
}


def validate_contract(pixel_values: torch.Tensor) -> None:
    if pixel_values.shape != (1, 3, 800, 800):
        raise RuntimeError(f"unexpected layout input shape: {pixel_values.shape}")
    if pixel_values.dtype != torch.float32 or pixel_values.device.type != "cpu":
        raise RuntimeError(
            "layout input must be a CPU float32 tensor, got "
            f"{pixel_values.dtype} on {pixel_values.device}"
        )
    minimum = float(pixel_values.min().item())
    maximum = float(pixel_values.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"unexpected pixel range: [{minimum}, {maximum}]")


def run_lane(
    name: str,
    processor: Any,
    paths: list[Path],
    round_index: int,
) -> dict[str, Any]:
    prepare = LANES[name]
    stage_s: dict[str, float] = defaultdict(float)
    formats: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    gc.collect()
    gc.disable()
    try:
        for index, path in enumerate(paths):
            encoded, file_read_s = read_bytes(path)
            pixel_values, timing = prepare(processor, encoded)
            validate_contract(pixel_values)
            stage_s["file_read_s"] += file_read_s
            for stage, seconds in timing.items():
                stage_s[stage] += seconds
            formats["png" if encoded.startswith(PNG_SIGNATURE) else "jpeg"] += 1
            if (index + 1) % 32 == 0 or index + 1 == len(paths):
                print(
                    f"LAYOUT_INPUT_AB round={round_index + 1} lane={name} "
                    f"pages={index + 1}/{len(paths)}",
                    flush=True,
                )
    finally:
        gc.enable()
    wall_s = time.perf_counter() - started
    accounted_s = sum(stage_s.values())
    return {
        "lane": name,
        "round": round_index,
        "pages": len(paths),
        "formats": dict(formats),
        "wall_s": wall_s,
        "pages_per_s": len(paths) / wall_s,
        "stage_s": dict(stage_s),
        "unaccounted_s": wall_s - accounted_s,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in LANES:
        rows = [row for row in records if row["lane"] == name]
        stage_names = sorted({key for row in rows for key in row["stage_s"]})
        wall_values = [float(row["wall_s"]) for row in rows]
        output[name] = {
            "wall_s_rounds": wall_values,
            "wall_s_median": statistics.median(wall_values),
            "stage_s_median": {
                stage: statistics.median(
                    float(row["stage_s"].get(stage, 0.0)) for row in rows
                )
                for stage in stage_names
            },
        }
        output[name]["pages_per_s_from_median"] = (
            rows[0]["pages"] / output[name]["wall_s_median"]
        )
    baseline = output["regular_opencv"]["wall_s_median"]
    optimized = output["optimized_direct_rgb"]["wall_s_median"]
    output["comparison"] = {
        "wall_speedup_x": baseline / optimized,
        "wall_saved_s": baseline - optimized,
    }
    return output


def compare_tensors(
    processor: Any,
    paths: list[Path],
) -> dict[str, Any]:
    pages = []
    for index, path in enumerate(paths):
        encoded = path.read_bytes()
        reference, _ = regular_prepare(processor, encoded)
        candidate, _ = optimized_prepare(processor, encoded)
        difference = (candidate - reference).abs()
        pages.append(
            {
                "page": path.name,
                "format": "png" if encoded.startswith(PNG_SIGNATURE) else "jpeg",
                "exact": bool(torch.equal(reference, candidate)),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
            }
        )
        print(
            f"LAYOUT_INPUT_AB parity={index + 1}/{len(paths)} "
            f"exact={pages[-1]['exact']} max_abs={pages[-1]['max_abs']:.6g}",
            flush=True,
        )
    return {
        "page_count": len(pages),
        "all_exact": all(page["exact"] for page in pages),
        "max_abs": max((page["max_abs"] for page in pages), default=0.0),
        "mean_abs_max": max((page["mean_abs"] for page in pages), default=0.0),
        "pages": pages,
    }


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
        f"LAYOUT_INPUT_AB setup pages={len(paths)} rounds={args.rounds} "
        f"warmup_pages={args.warmup_pages}",
        flush=True,
    )
    warmup_paths = paths[: args.warmup_pages]
    for name, prepare in LANES.items():
        for path in warmup_paths:
            encoded = path.read_bytes()
            pixel_values, _ = prepare(processor, encoded)
            validate_contract(pixel_values)
        print(f"LAYOUT_INPUT_AB warmup lane={name} done", flush=True)

    records = []
    names = list(LANES)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            record = run_lane(name, processor, paths, round_index)
            records.append(record)
            print("LAYOUT_INPUT_AB result " + json.dumps(record), flush=True)

    parity = compare_tensors(processor, paths[: args.parity_pages])
    report = {
        "config": {
            "layout_model": str(args.layout_model.expanduser().resolve()),
            "input": str(args.input.expanduser().resolve()),
            "offset": args.offset,
            "limit": args.limit,
            "rounds": args.rounds,
            "warmup_pages": args.warmup_pages,
            "parity_pages": args.parity_pages,
            "device_work": False,
            "output_contract": [1, 3, 800, 800],
            "dtype": "torch.float32",
        },
        "summary": summarize(records),
        "parity": parity,
        "rounds": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("LAYOUT_INPUT_AB summary " + json.dumps(report["summary"]), flush=True)
    print(f"LAYOUT_INPUT_AB done output={output}", flush=True)


if __name__ == "__main__":
    main()
