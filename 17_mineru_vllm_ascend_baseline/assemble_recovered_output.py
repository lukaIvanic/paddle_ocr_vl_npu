#!/usr/bin/env python3
"""Assemble one accuracy-only output from a failed prefix and recovery run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--recovery-output", type=Path, required=True)
    parser.add_argument("--recovery-offset", type=int, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_names(manifest: dict[str, Any]) -> list[str]:
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise RuntimeError("input manifest has no pages list")
    names = [str(page["image"]) for page in pages]
    if manifest.get("count") != len(names):
        raise RuntimeError("input manifest count does not match pages")
    if len(names) != len(set(names)):
        raise RuntimeError("input manifest contains duplicate image names")
    stems = [Path(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise RuntimeError("input manifest contains duplicate output stems")
    return names


def _require_output_pair(output: Path, image_name: str) -> tuple[Path, Path]:
    stem = Path(image_name).stem
    markdown = output / "predictions" / f"{stem}.md"
    content = output / "content_lists" / f"{stem}.json"
    for path in (markdown, content):
        if not path.is_file():
            raise RuntimeError(f"missing recovery component: {path}")
    return markdown.resolve(), content.resolve()


def assemble_recovered_output(
    *,
    primary_output: Path,
    recovery_output: Path,
    recovery_offset: int,
    combined_output: Path,
) -> dict[str, Any]:
    if combined_output.exists():
        raise FileExistsError(combined_output)

    primary_manifest_path = primary_output / "input_manifest.json"
    primary_failure_path = primary_output / "failure.json"
    recovery_manifest_path = recovery_output / "input_manifest.json"
    recovery_summary_path = recovery_output / "run_summary.json"
    primary_manifest = load_json(primary_manifest_path)
    primary_failure = load_json(primary_failure_path)
    recovery_manifest = load_json(recovery_manifest_path)
    recovery_summary = load_json(recovery_summary_path)

    primary_names = _manifest_names(primary_manifest)
    recovery_names = _manifest_names(recovery_manifest)
    total_pages = len(primary_names)
    if not 0 < recovery_offset < total_pages:
        raise ValueError("recovery_offset must split the primary manifest")
    expected_recovery_names = primary_names[recovery_offset:]
    if recovery_names != expected_recovery_names:
        raise RuntimeError("recovery manifest is not the exact primary suffix")
    recovery_pages = total_pages - recovery_offset
    if (
        recovery_summary.get("offset") != recovery_offset
        or recovery_summary.get("selected_pages") != recovery_pages
        or recovery_summary.get("completed") != recovery_pages
        or recovery_summary.get("failed") != 0
    ):
        raise RuntimeError("recovery run summary does not prove a complete suffix")
    if primary_failure.get("selected_pages") != total_pages:
        raise RuntimeError("primary failure does not cover the full manifest")

    predictions = combined_output / "predictions"
    content_lists = combined_output / "content_lists"
    predictions.mkdir(parents=True)
    content_lists.mkdir()
    for index, image_name in enumerate(primary_names):
        source_output = primary_output if index < recovery_offset else recovery_output
        markdown, content = _require_output_pair(source_output, image_name)
        stem = Path(image_name).stem
        os.symlink(markdown, predictions / f"{stem}.md")
        os.symlink(content, content_lists / f"{stem}.json")

    write_json(combined_output / "input_manifest.json", primary_manifest)
    summary = {
        "experiment": primary_failure.get("experiment"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "compiled_async_recovered_stitch",
        "git_commit": recovery_summary.get("git_commit"),
        "input_source": primary_failure.get("input_source"),
        "selected_pages": total_pages,
        "offset": 0,
        "limit": total_pages,
        "completed": total_pages,
        "failed": 0,
        "accuracy_only": True,
        "throughput_comparable": False,
        "recovery_offset": recovery_offset,
        "component_runs": [
            {
                "role": "primary_prefix_after_completed_inference",
                "output": str(primary_output),
                "saved_pages": recovery_offset,
                "failure_sha256": sha256_file(primary_failure_path),
            },
            {
                "role": "recovery_suffix",
                "output": str(recovery_output),
                "saved_pages": recovery_pages,
                "run_summary_sha256": sha256_file(recovery_summary_path),
            },
        ],
    }
    write_json(combined_output / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = assemble_recovered_output(
        primary_output=args.primary_output.expanduser().resolve(),
        recovery_output=args.recovery_output.expanduser().resolve(),
        recovery_offset=args.recovery_offset,
        combined_output=args.combined_output.expanduser().resolve(),
    )
    print("EXPERIMENT17_RECOVERY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
