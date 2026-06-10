#!/usr/bin/env python3
"""Download a small OmniDocBench page slice directly on the run machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="opendatalab/OmniDocBench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/data/OmniDocBench"))
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=64)
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.write_bytes(src.read_bytes())


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    json_cache = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename="OmniDocBench.json",
        )
    )
    dataset: list[dict[str, Any]] = json.loads(json_cache.read_text(encoding="utf-8"))
    start = int(args.page_start)
    end = min(len(dataset), start + int(args.num_pages))
    if start < 0 or start >= len(dataset) or end <= start:
        raise ValueError(f"invalid page slice start={start} num_pages={args.num_pages} dataset_len={len(dataset)}")

    selected = dataset[start:end]
    copy_file(json_cache, out_dir / "OmniDocBench.json")
    downloaded = []
    for idx, row in enumerate(selected, start=start):
        rel = str(row.get("page_info", {}).get("image_path", ""))
        if not rel:
            raise ValueError(f"dataset row {idx} has no page_info.image_path")
        filename = f"images/{rel}"
        image_cache = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                filename=filename,
            )
        )
        copy_file(image_cache, out_dir / filename)
        downloaded.append(filename)

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "out_dir": str(out_dir),
        "page_start": int(start),
        "num_pages": int(len(selected)),
        "downloaded_images": int(len(downloaded)),
        "first_images": downloaded[:8],
    }
    (out_dir / "first_pages_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
