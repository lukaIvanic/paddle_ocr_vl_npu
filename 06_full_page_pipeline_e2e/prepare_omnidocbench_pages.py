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


def u_escape_path_component(value: str, *, uppercase_hex: bool = False) -> str:
    escaped = []
    for char in str(value):
        code = ord(char)
        if code < 128:
            escaped.append(char)
            continue
        fmt = "04X" if uppercase_hex else "04x"
        escaped.append(f"#U{code:{fmt}}")
    return "".join(escaped)


def u_escape_relative_path(rel: str, *, uppercase_hex: bool = False) -> Path:
    path = Path(str(rel))
    return Path(*(u_escape_path_component(part, uppercase_hex=uppercase_hex) for part in path.parts))


def image_filename_candidates(rel: str) -> list[str]:
    candidates = [f"images/{rel}"]
    escaped_lower = Path("images") / u_escape_relative_path(rel, uppercase_hex=False)
    escaped_upper = Path("images") / u_escape_relative_path(rel, uppercase_hex=True)
    for candidate in (str(escaped_lower), str(escaped_upper)):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def hf_download_first_available(*, repo_id: str, repo_type: str, revision: str, filenames: list[str]) -> tuple[Path, str]:
    errors: list[str] = []
    for filename in filenames:
        try:
            return (
                Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        revision=revision,
                        filename=filename,
                    )
                ),
                filename,
            )
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError("none of the OmniDocBench image filename candidates downloaded: " + " | ".join(errors))


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
    end = start + int(args.num_pages)
    if start < 0 or start >= len(dataset) or end <= start or end > len(dataset):
        raise ValueError(f"invalid page slice start={start} num_pages={args.num_pages} dataset_len={len(dataset)}")

    selected = dataset[start:end]
    copy_file(json_cache, out_dir / "OmniDocBench.json")
    downloaded = []
    for idx, row in enumerate(selected, start=start):
        rel = str(row.get("page_info", {}).get("image_path", ""))
        if not rel:
            raise ValueError(f"dataset row {idx} has no page_info.image_path")
        image_cache, filename = hf_download_first_available(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filenames=image_filename_candidates(rel),
        )
        copy_file(image_cache, out_dir / filename)
        downloaded.append({"dataset_index": int(idx), "json_image_path": rel, "downloaded_filename": filename})

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "out_dir": str(out_dir),
        "page_start": int(start),
        "num_pages": int(len(selected)),
        "downloaded_images": int(len(downloaded)),
        "missing_image_policy": "fatal",
        "first_images": downloaded[:8],
    }
    (out_dir / "first_pages_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
