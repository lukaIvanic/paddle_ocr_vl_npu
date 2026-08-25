#!/usr/bin/env python3
"""Run the exact short-lived layout owner over a full image set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_PURGE_HOST_AFTER_WARMUP", "1")

from low_memory_frontend_pool import SharedLayoutProcess


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--limit", type=int, default=1651)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    paths = sorted(
        path.resolve()
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.limit]
    if len(paths) != args.limit:
        raise RuntimeError(f"expected {args.limit} pages, found {len(paths)}")
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    layout = SharedLayoutProcess(
        paths=paths,
        model_path=args.layout_model.resolve(),
        cache_dir=args.layout_cache.resolve(),
        device=args.device,
        lanes=args.lanes,
        batch_size=args.batch_size,
        threshold=args.threshold,
        page_spool_root=args.spool_dir.resolve(),
    )
    page_count = sum(1 for _ in layout.iter_pages())
    layout.close()
    wall_s = time.perf_counter() - started
    report = {
        "schema": "unirec_low_memory_layout_process_v1",
        "status": "pass",
        "page_count": page_count,
        "wall_s": wall_s,
        "pages_per_s": page_count / wall_s,
        "malloc_conf": layout.malloc_conf,
        "ready": layout.ready,
        "summary": layout.summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("UNIREC_LAYOUT_PROCESS_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
