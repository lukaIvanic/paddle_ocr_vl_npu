#!/usr/bin/env python3
"""Replay saved layout results through the exact W4/T8 crop-preparation pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("UNIREC_PURGE_HOST_AFTER_WARMUP", "1")

from low_memory_frontend_pool import CpuCropPreparePool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recognition-threads", type=int, default=8)
    parser.add_argument("--resize-chunk-size", type=int, default=0)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spool_dir.exists() and any(args.spool_dir.iterdir()):
        raise RuntimeError(f"spool directory is not empty: {args.spool_dir}")
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.layout_manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    pool = CpuCropPreparePool(
        workers=args.workers,
        recognition_threads=args.recognition_threads,
        resize_chunk_size=args.resize_chunk_size,
        openocr_root=args.openocr_root.resolve(),
        spool_root=args.spool_dir.resolve(),
        cross_cache_length=args.cross_cache_length,
    )
    for row in rows:
        pool.submit(
            page_index=int(row["page_index"]),
            path=Path(row["image_path"]),
            rgb=None,
            rgb_descriptor=None,
            layout_result=row["layout_results"],
            started_at=started,
        )
    payloads, summary = pool.finish()
    pool.close()
    wall_s = time.perf_counter() - started
    report = {
        "schema": "unirec_low_memory_crop_pool_v1",
        "status": "pass",
        "page_count": len(payloads),
        "crop_count": summary.crop_count,
        "wall_s": wall_s,
        "pages_per_s": len(payloads) / wall_s,
        "settings": {
            "workers": args.workers,
            "recognition_threads": args.recognition_threads,
            "resize_chunk_size": args.resize_chunk_size,
            "cross_cache_length": args.cross_cache_length,
            "crop_worker_malloc_conf": pool.malloc_conf,
        },
        "summary": asdict(summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("UNIREC_CROP_POOL_BENCHMARK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
