#!/usr/bin/env python3
"""Profile the production PP-DocLayoutV2 NPU detector one page at a time.

The lab uses the same adapter, processor, model, threshold, BGR page contract,
and OpenDoc image ordering as ``run_opendoc_batched_unirec.py``.  Recognition,
crop construction, and output assembly are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--warmup-pages",
        type=int,
        default=1,
        help="Warmup calls on the first selected page; excluded from results",
    )
    return parser.parse_args()


def decode_page_bgr(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    read_started = time.perf_counter()
    encoded = path.read_bytes()
    read_s = time.perf_counter() - read_started

    decode_started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    decode_s = time.perf_counter() - decode_started
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image, {"page_file_read_s": read_s, "page_image_decode_s": decode_s}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(records: list[dict[str, Any]], setup_s: float) -> dict[str, Any]:
    stage_names = sorted(
        {
            name
            for record in records
            for name in record["stage_s"]
        }
    )
    stages: dict[str, Any] = {}
    for name in stage_names:
        values = [float(record["stage_s"].get(name, 0.0)) for record in records]
        stages[name] = {
            "total_s": sum(values),
            "mean_ms": statistics.fmean(values) * 1000.0,
            "median_ms": statistics.median(values) * 1000.0,
            "p90_ms": percentile(values, 0.90) * 1000.0,
            "min_ms": min(values) * 1000.0,
            "max_ms": max(values) * 1000.0,
        }

    page_wall = [float(record["page_wall_s"]) for record in records]
    measured_wall_s = sum(page_wall)
    detector_total_s = stages.get("detector_total_s", {}).get("total_s", 0.0)
    for name, stage in stages.items():
        stage["page_wall_share_pct"] = (
            100.0 * float(stage["total_s"]) / measured_wall_s
            if measured_wall_s
            else 0.0
        )
        if name not in {"page_file_read_s", "page_image_decode_s"}:
            stage["detector_share_pct"] = (
                100.0 * float(stage["total_s"]) / detector_total_s
                if detector_total_s
                else 0.0
            )

    return {
        "setup_s": setup_s,
        "page_count": len(records),
        "measured_page_wall_s": measured_wall_s,
        "pages_per_s": len(records) / measured_wall_s if measured_wall_s else 0.0,
        "page_wall_mean_ms": statistics.fmean(page_wall) * 1000.0,
        "page_wall_median_ms": statistics.median(page_wall) * 1000.0,
        "page_wall_p90_ms": percentile(page_wall, 0.90) * 1000.0,
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.warmup_pages < 0:
        raise ValueError("--warmup-pages must be >= 0")

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)

    openocr_root = args.openocr_root.expanduser().resolve()
    sys.path.insert(0, str(openocr_root))
    from tools.utils.utility import get_image_file_list

    input_path = args.input.expanduser().resolve()
    image_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(input_path)))
    ][args.offset : args.offset + args.limit]
    if not image_paths:
        raise ValueError(f"No images found under {input_path}")

    print(
        f"LAYOUT_LAB setup pages={len(image_paths)} dtype={args.dtype} "
        f"device={args.device}",
        flush=True,
    )
    detector = PPDocLayoutV2NpuAdapter(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        threshold=args.threshold,
        profile_stages=True,
    )

    warmup_image, _ = decode_page_bgr(image_paths[0])
    for index in range(args.warmup_pages):
        detector([warmup_image], threshold=args.threshold)
        print(f"LAYOUT_LAB warmup {index + 1}/{args.warmup_pages}", flush=True)
    detector.reset_timing()

    records: list[dict[str, Any]] = []
    for page_index, image_path in enumerate(image_paths):
        page_started = time.perf_counter()
        image, decode_timing = decode_page_bgr(image_path)
        before = dict(detector.stage_s)
        result = detector([image], threshold=args.threshold)[0]
        stage_s = {
            name: float(seconds) - float(before.get(name, 0.0))
            for name, seconds in detector.stage_s.items()
        }
        stage_s.update(decode_timing)
        record = {
            "page_index": page_index + args.offset,
            "image": str(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "box_count": len(result["boxes"]),
            "page_wall_s": time.perf_counter() - page_started,
            "stage_s": stage_s,
        }
        records.append(record)
        print(
            f"LAYOUT_LAB page={page_index + 1}/{len(image_paths)} "
            f"wall_ms={record['page_wall_s'] * 1000.0:.1f} "
            f"forward_ms={stage_s.get('model_forward_s', 0.0) * 1000.0:.1f} "
            f"boxes={record['box_count']}",
            flush=True,
        )

    report = {
        "config": {
            "openocr_root": str(openocr_root),
            "model_path": str(args.model_path.expanduser().resolve()),
            "input": str(input_path),
            "device": args.device,
            "dtype": args.dtype,
            "threshold": args.threshold,
            "offset": args.offset,
            "limit": args.limit,
            "warmup_pages": args.warmup_pages,
            "execution": "sequential_b1",
        },
        "summary": summarize(records, detector.setup_s),
        "pages": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("LAYOUT_LAB summary", flush=True)
    for name, stage in sorted(
        report["summary"]["stages"].items(),
        key=lambda item: item[1]["total_s"],
        reverse=True,
    ):
        if name == "detector_total_s":
            continue
        print(
            f"  {name}: total={stage['total_s']:.3f}s "
            f"mean={stage['mean_ms']:.2f}ms p90={stage['p90_ms']:.2f}ms",
            flush=True,
        )
    print(
        f"LAYOUT_LAB done pages_per_s={report['summary']['pages_per_s']:.3f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
