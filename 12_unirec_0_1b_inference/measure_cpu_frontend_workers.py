#!/usr/bin/env python3
"""Measure four CPU-only page and crop preparation workers."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retained-crops", type=int, default=4)
    return parser.parse_args()


def worker_main(
    index: int,
    image_path: str,
    openocr_root: str,
    retained_crops: int,
    ready_queue: Any,
    release_event: Any,
) -> None:
    import cv2  # noqa: F401
    import numpy as np
    from PIL import Image
    from kornia_rs.image import Image as KorniaImage

    sys.path.insert(0, openocr_root)
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: F401
        tokenize_figure_of_table,
    )

    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("CPU frontend worker imported Torch")
    encoded = Path(image_path).read_bytes()
    rgb = KorniaImage.decode(encoded, "RGB").data
    page = Image.fromarray(rgb, mode="RGB")
    width, height = page.size
    boxes = (
        (0, 0, width, max(1, height // 4)),
        (0, max(0, height // 4), width, max(1, height // 2)),
        (0, max(0, height // 2), width, max(1, 3 * height // 4)),
        (0, max(0, 3 * height // 4), width, height),
    )
    prepared = []
    target_shapes = ((960, 64), (512, 64), (960, 256), (448, 384))
    for crop_index in range(retained_crops):
        crop = page.crop(boxes[crop_index % len(boxes)])
        target = target_shapes[crop_index % len(target_shapes)]
        resized = crop.resize(target, resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(resized)
        if pixels.dtype != np.uint8 or not pixels.flags.c_contiguous:
            pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        prepared.append(pixels)

    from host_memory_diagnostics import process_snapshot

    ready_queue.put(
        {
            "worker": index,
            "pid": __import__("os").getpid(),
            "snapshot": process_snapshot(),
            "page_bytes": int(rgb.nbytes),
            "crop_bytes": sum(int(value.nbytes) for value in prepared),
            "torch_imported": "torch" in sys.modules,
            "torch_npu_imported": "torch_npu" in sys.modules,
        }
    )
    release_event.wait()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retained_crops < 0:
        raise ValueError("worker and retained crop counts must be valid")
    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    release_event = context.Event()
    workers = [
        context.Process(
            target=worker_main,
            args=(
                index,
                str(args.image),
                str(args.openocr_root),
                args.retained_crops,
                ready_queue,
                release_event,
            ),
        )
        for index in range(args.workers)
    ]
    for process in workers:
        process.start()
    reports = [ready_queue.get(timeout=120.0) for _ in workers]
    from host_memory_diagnostics import process_snapshot

    parent = process_snapshot()
    worker_pss = sum(
        int(item["snapshot"]["proc_bytes"]["pss"]) for item in reports
    )
    parent_pss = int(parent["proc_bytes"]["pss"])
    report = {
        "status": "pass",
        "worker_count": len(workers),
        "retained_crops_per_worker": args.retained_crops,
        "worker_pss_bytes": worker_pss,
        "parent_pss_bytes": parent_pss,
        "total_pss_bytes": worker_pss + parent_pss,
        "workers": sorted(reports, key=lambda item: int(item["worker"])),
    }
    print("UNIREC_CPU_FRONTEND_MEMORY " + __import__("json").dumps(report, sort_keys=True))
    release_event.set()
    for process in workers:
        process.join(timeout=30.0)
        if process.exitcode != 0:
            raise RuntimeError(
                f"CPU frontend worker {process.pid} exited {process.exitcode}"
            )


if __name__ == "__main__":
    main()
