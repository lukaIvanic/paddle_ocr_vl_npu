#!/usr/bin/env python3
"""Audit OmniDocBench GT crop counts for experiment 6.

This intentionally reuses the experiment-6 page loader and GT-layout filtering
rules so the count printed here matches the benchmark's recognizer crop count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bench_page_pipeline_e2e import (
    build_omnidocbench_gt_layout_pages,
    clean_json,
    load_pages_result,
    page_load_summary,
    resolve_dataset_dir,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, required=True)
    parser.add_argument("--include-ignored-gt", action="store_true")
    parser.add_argument("--include-empty-gt", action="store_true")
    parser.add_argument("--expect-count", type=int, default=-1)
    parser.add_argument("--print-count-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    json_path = dataset_dir / "OmniDocBench.json"
    page_load = load_pages_result(
        dataset_dir,
        page_start=int(args.page_start),
        num_pages=int(args.num_pages),
    )
    layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(
        page_load.pages,
        include_ignored=bool(args.include_ignored_gt),
        include_empty_gt=bool(args.include_empty_gt),
    )
    label_counts: Counter[str] = Counter()
    per_page_counts: list[dict[str, Any]] = []
    for page in layout_pages:
        boxes = list(page.get("boxes") or [])
        label_counts.update(str(box.get("label", "unknown")) for box in boxes)
        per_page_counts.append(
            {
                "selected_page_idx": int(page.get("selected_page_idx", 0)),
                "dataset_index": int(page.get("dataset_index", 0)),
                "image_rel": str(page.get("image_rel", "")),
                "recognizer_crop_count": int(len(boxes)),
            }
        )

    recognizer_crop_count = int(sum(row["recognizer_crop_count"] for row in per_page_counts))
    payload = {
        "dataset_dir": str(dataset_dir),
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "page_load": page_load_summary(page_load),
        "include_ignored_gt": bool(args.include_ignored_gt),
        "include_empty_gt": bool(args.include_empty_gt),
        "recognizer_crop_count": recognizer_crop_count,
        "raw_layout_det_count": int(layout_timing.get("gt_layout_raw_box_count", 0)),
        "skipped_ignored_count": int(layout_timing.get("gt_layout_skipped_ignored_count", 0)),
        "skipped_empty_gt_count": int(layout_timing.get("gt_layout_skipped_empty_gt_count", 0)),
        "label_counts": dict(sorted(label_counts.items())),
        "first_page": per_page_counts[0] if per_page_counts else None,
        "last_page": per_page_counts[-1] if per_page_counts else None,
        "per_page_counts_sample": per_page_counts[:8],
        "count_contract": {
            "expected": None if int(args.expect_count) < 0 else int(args.expect_count),
            "actual": recognizer_crop_count,
            "passed": bool(int(args.expect_count) < 0 or recognizer_crop_count == int(args.expect_count)),
        },
    }
    if int(args.expect_count) >= 0 and recognizer_crop_count != int(args.expect_count):
        print(json.dumps(clean_json(payload), ensure_ascii=False, sort_keys=True))
        raise SystemExit(
            "GT crop count mismatch: "
            f"expected {int(args.expect_count)} but got {recognizer_crop_count}"
        )
    if args.print_count_only:
        print(recognizer_crop_count)
    else:
        print(json.dumps(clean_json(payload), ensure_ascii=False, sort_keys=bool(args.json)))


if __name__ == "__main__":
    main()
