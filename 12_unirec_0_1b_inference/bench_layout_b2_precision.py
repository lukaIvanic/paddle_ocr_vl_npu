#!/usr/bin/env python3
"""Measure faithful PP-DocLayoutV2 B2 model-forward latency by precision lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layout_page_input import decode_page_rgb, materialize_layout_rgb  # noqa: E402
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fp32-cache-dir", type=Path, required=True)
    parser.add_argument("--fp16-cache-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--torch-cpu-threads", type=int, default=1)
    parser.add_argument(
        "--lane",
        choices=("eager_fp32", "compiled_fp32", "compiled_fp16_body_fp32_ro"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.warmup < 2:
        parser.error("--warmup must be at least 2 to expose first-call loading")
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")
    if args.torch_cpu_threads < 1:
        parser.error("--torch-cpu-threads must be positive")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def image_paths(input_path: Path, *, offset: int) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            path.resolve()
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    selected = candidates[offset : offset + 2]
    if len(selected) != 2:
        raise ValueError(
            f"B2 probe needs two images at offset {offset}, found {len(selected)}"
        )
    return selected


def digest_result(result: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def om_inventory(root: Path) -> dict[str, dict[str, int]]:
    root = root.expanduser().resolve()
    return {
        str(path.relative_to(root)): {
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in sorted(root.rglob("*.om"))
    }


def delta_ms(
    after: dict[str, float], before: dict[str, float], name: str
) -> float:
    return 1000.0 * (float(after.get(name, 0.0)) - float(before.get(name, 0.0)))


def run_lane(
    *,
    name: str,
    images: list[np.ndarray],
    model_path: Path,
    device: str,
    execution: str,
    dtype: str,
    weight_format: str,
    depthwise_rewrite: str,
    preformat_frozen_bn_buffers: bool,
    cache_dir: Path,
    threshold: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    inventory_before = om_inventory(cache_dir)
    print(
        "UNIREC_LAYOUT_B2_LANE_SETUP_BEGIN "
        f"lane={name} execution={execution} dtype={dtype} "
        f"cache_oms={len(inventory_before)}",
        flush=True,
    )
    detector = PPDocLayoutV2NpuAdapter(
        model_path=model_path,
        device=device,
        dtype=dtype,
        reading_order_dtype="float32",
        threshold=threshold,
        profile_stages=True,
        execution=execution,
        compile_cache_dir=cache_dir,
        batch_size=2,
        weight_format=weight_format,
        depthwise_rewrite=depthwise_rewrite,
        preformat_frozen_bn_buffers=preformat_frozen_bn_buffers,
        msda_implementation="decomposed",
        input_color_order="rgb",
    )
    print(
        f"UNIREC_LAYOUT_B2_LANE_SETUP_END lane={name} setup_s={detector.setup_s:.6f}",
        flush=True,
    )

    warmup_wall_ms: list[float] = []
    for index in range(warmup):
        print(
            f"UNIREC_LAYOUT_B2_WARMUP_BEGIN lane={name} call={index + 1}/{warmup}",
            flush=True,
        )
        started = time.perf_counter()
        results = detector(images, threshold=threshold)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        warmup_wall_ms.append(elapsed_ms)
        print(
            f"UNIREC_LAYOUT_B2_WARMUP_END lane={name} call={index + 1}/{warmup} "
            f"wall_ms={elapsed_ms:.6f} boxes={[len(row['boxes']) for row in results]}",
            flush=True,
        )
    detector.reset_timing()

    forward_ms: list[float] = []
    processor_ms: list[float] = []
    h2d_ms: list[float] = []
    postprocess_ms: list[float] = []
    call_wall_ms: list[float] = []
    measured_results: list[dict[str, Any]] | None = None
    for index in range(repeats):
        before = dict(detector.stage_s)
        started = time.perf_counter()
        results = detector(images, threshold=threshold)
        call_wall_ms.append(1000.0 * (time.perf_counter() - started))
        after = dict(detector.stage_s)
        forward_ms.append(delta_ms(after, before, "model_forward_s"))
        processor_ms.append(delta_ms(after, before, "processor_preprocess_s"))
        h2d_ms.append(delta_ms(after, before, "inputs_h2d_s"))
        postprocess_ms.append(delta_ms(after, before, "postprocess_s"))
        if measured_results is None:
            measured_results = results
        if index in {0, repeats - 1}:
            print(
                f"UNIREC_LAYOUT_B2_MEASURE lane={name} call={index + 1}/{repeats} "
                f"forward_ms={forward_ms[-1]:.6f} wall_ms={call_wall_ms[-1]:.6f}",
                flush=True,
            )

    if measured_results is None:
        raise AssertionError("no measured results")
    inventory_after = om_inventory(cache_dir)
    added = sorted(set(inventory_after) - set(inventory_before))
    removed = sorted(set(inventory_before) - set(inventory_after))
    changed = sorted(
        key
        for key in set(inventory_before) & set(inventory_after)
        if inventory_before[key] != inventory_after[key]
    )
    lane = {
        "name": name,
        "execution": execution,
        "dtype": dtype,
        "reading_order_dtype": "float32",
        "batch_size": 2,
        "weight_format": weight_format,
        "depthwise_rewrite": depthwise_rewrite,
        "preformat_frozen_bn_buffers": preformat_frozen_bn_buffers,
        "msda_implementation": "decomposed",
        "setup_s": detector.setup_s,
        "warmup_wall_ms": warmup_wall_ms,
        "forward": distribution(forward_ms),
        "processor_preprocess": distribution(processor_ms),
        "inputs_h2d": distribution(h2d_ms),
        "postprocess": distribution(postprocess_ms),
        "call_wall": distribution(call_wall_ms),
        "box_counts": [len(row["boxes"]) for row in measured_results],
        "result_digests": [digest_result(row) for row in measured_results],
        "cache": {
            "root": str(cache_dir.expanduser().resolve()),
            "om_count_before": len(inventory_before),
            "om_count_after": len(inventory_after),
            "added": added,
            "removed": removed,
            "changed": changed,
        },
    }
    print(
        "UNIREC_LAYOUT_B2_LANE_RESULT "
        f"lane={name} forward_mean_ms={lane['forward']['mean_ms']:.6f} "
        f"forward_median_ms={lane['forward']['median_ms']:.6f} "
        f"wall_mean_ms={lane['call_wall']['mean_ms']:.6f} "
        f"new_oms={len(added)} boxes={lane['box_counts']}",
        flush=True,
    )
    del detector
    torch.npu.synchronize()
    torch.npu.empty_cache()
    return lane


def lane_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "name": "eager_fp32",
            "execution": "eager",
            "dtype": "float32",
            "weight_format": "native",
            "depthwise_rewrite": "native",
            "preformat_frozen_bn_buffers": False,
            "cache_dir": args.fp32_cache_dir,
        },
        {
            "name": "compiled_fp32",
            "execution": "torchair",
            "dtype": "float32",
            "weight_format": "native",
            "depthwise_rewrite": "native",
            "preformat_frozen_bn_buffers": False,
            "cache_dir": args.fp32_cache_dir,
        },
        {
            "name": "compiled_fp16_body_fp32_ro",
            "execution": "torchair",
            "dtype": "float16",
            "weight_format": "torchair_internal",
            "depthwise_rewrite": "constant_grouped",
            "preformat_frozen_bn_buffers": True,
            "cache_dir": args.fp16_cache_dir,
        },
    ]


def load_images(args: argparse.Namespace) -> tuple[list[Path], list[np.ndarray]]:
    selected = image_paths(args.input, offset=args.offset)
    images = []
    for path in selected:
        rgb, _ = decode_page_rgb(path)
        images.append(materialize_layout_rgb(rgb))
    print(
        "UNIREC_LAYOUT_B2_INPUT "
        f"images={[str(path) for path in selected]} "
        f"shapes={[list(image.shape) for image in images]}",
        flush=True,
    )
    return selected, images


def run_lane_process(args: argparse.Namespace) -> None:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(args.torch_cpu_threads)
    torch.set_num_interop_threads(args.torch_cpu_threads)
    _selected, images = load_images(args)
    spec = next(spec for spec in lane_specs(args) if spec["name"] == args.lane)
    lane = run_lane(
        images=images,
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        threshold=args.threshold,
        warmup=args.warmup,
        repeats=args.repeats,
        **spec,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lane, indent=2) + "\n", encoding="utf-8")
    print(f"UNIREC_LAYOUT_B2_LANE_OUTPUT={args.output.resolve()}", flush=True)


def run_controller(args: argparse.Namespace) -> None:
    selected = image_paths(args.input, offset=args.offset)
    lane_paths: list[Path] = []
    for spec in lane_specs(args):
        lane_output = args.output.parent / f"{args.output.stem}.{spec['name']}.json"
        lane_paths.append(lane_output)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-path",
            str(args.model_path),
            "--input",
            str(args.input),
            "--output",
            str(lane_output),
            "--device",
            args.device,
            "--fp32-cache-dir",
            str(args.fp32_cache_dir),
            "--fp16-cache-dir",
            str(args.fp16_cache_dir),
            "--offset",
            str(args.offset),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--threshold",
            str(args.threshold),
            "--torch-cpu-threads",
            str(args.torch_cpu_threads),
            "--lane",
            spec["name"],
        ]
        print(
            f"UNIREC_LAYOUT_B2_SUBPROCESS_BEGIN lane={spec['name']} ",
            flush=True,
        )
        subprocess.run(command, check=True)
        print(
            f"UNIREC_LAYOUT_B2_SUBPROCESS_END lane={spec['name']}",
            flush=True,
        )

    lanes = [json.loads(path.read_text(encoding="utf-8")) for path in lane_paths]
    by_name = {lane["name"]: lane for lane in lanes}
    eager = by_name["eager_fp32"]
    compiled_fp32 = by_name["compiled_fp32"]
    compiled_fp16 = by_name["compiled_fp16_body_fp32_ro"]
    summary = {
        "schema": "unirec_layout_b2_precision_v1",
        "device": args.device,
        "input_images": [str(path) for path in selected],
        "threshold": args.threshold,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch_cpu_threads": args.torch_cpu_threads,
        "lanes": by_name,
        "comparisons": {
            "compiled_fp32_speedup_vs_eager_fp32": (
                eager["forward"]["mean_ms"]
                / compiled_fp32["forward"]["mean_ms"]
            ),
            "compiled_fp16_speedup_vs_eager_fp32": (
                eager["forward"]["mean_ms"]
                / compiled_fp16["forward"]["mean_ms"]
            ),
            "compiled_fp16_speedup_vs_compiled_fp32": (
                compiled_fp32["forward"]["mean_ms"]
                / compiled_fp16["forward"]["mean_ms"]
            ),
            "compiled_fp32_exact_result_digests": (
                eager["result_digests"] == compiled_fp32["result_digests"]
            ),
            "compiled_fp16_exact_result_digests": (
                eager["result_digests"] == compiled_fp16["result_digests"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_LAYOUT_B2_PRECISION: PASS", flush=True)
    print(
        "UNIREC_LAYOUT_B2_COMPARISON "
        + json.dumps(summary["comparisons"], sort_keys=True),
        flush=True,
    )
    print(f"UNIREC_LAYOUT_B2_OUTPUT={args.output.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.lane is None:
        run_controller(args)
    else:
        run_lane_process(args)


if __name__ == "__main__":
    main()
