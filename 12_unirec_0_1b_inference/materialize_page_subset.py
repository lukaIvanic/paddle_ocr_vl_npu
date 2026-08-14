#!/usr/bin/env python3
"""Materialize a UniRec page-manifest as a symlink-only input directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "unirec_representative_pages_v1":
        raise ValueError("unsupported page-manifest schema")
    filenames = [str(page["filename"]) for page in manifest["pages"]]
    if len(filenames) != int(manifest["selection"]["count"]):
        raise ValueError("manifest count mismatch")
    if len(set(filenames)) != len(filenames):
        raise ValueError("manifest contains duplicate filenames")

    images_dir = args.images_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [filename for filename in filenames if not (images_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest images are absent; first={missing[0]!r}"
        )
    for filename in filenames:
        os.symlink(images_dir / filename, output_dir / filename)
    materialization = {
        "schema": manifest["schema"],
        "manifest": str(args.manifest.expanduser().resolve()),
        "selection_sha256": manifest["selection"]["selection_sha256"],
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "page_count": len(filenames),
    }
    metadata_path = output_dir.with_name(f"{output_dir.name}_materialization.json")
    metadata_path.write_text(
        json.dumps(materialization, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "UNIREC_PAGE_SUBSET_MATERIALIZED PASS "
        f"pages={len(filenames)} output_dir={output_dir} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
