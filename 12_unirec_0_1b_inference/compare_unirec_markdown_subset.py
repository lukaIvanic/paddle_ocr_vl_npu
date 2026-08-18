#!/usr/bin/env python3
"""Compare final Markdown for a manifest-selected UniRec page subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def index_markdown(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in root.expanduser().resolve().rglob("*.md"):
        if path.stem in indexed:
            raise RuntimeError(
                f"duplicate Markdown stem {path.stem}: {indexed[path.stem]}, {path}"
            )
        indexed[path.stem] = path
    return indexed


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "unirec_representative_pages_v1":
        raise ValueError("unsupported page-manifest schema")
    stems = [Path(str(row["filename"])).stem for row in manifest["pages"]]
    if len(stems) != 128 or len(set(stems)) != 128:
        raise ValueError("expected 128 unique representative page stems")

    baseline = index_markdown(args.baseline_output)
    candidate = index_markdown(args.candidate_output)
    missing_baseline = sorted(set(stems) - set(baseline))
    missing_candidate = sorted(set(stems) - set(candidate))
    if missing_baseline or missing_candidate:
        raise RuntimeError(
            f"missing baseline={missing_baseline[:5]} candidate={missing_candidate[:5]}"
        )

    rows = []
    for stem in stems:
        baseline_text = canonical_newlines(baseline[stem].read_text(encoding="utf-8"))
        candidate_text = canonical_newlines(candidate[stem].read_text(encoding="utf-8"))
        baseline_stripped = IMAGE_TAG_RE.sub("", baseline_text)
        candidate_stripped = IMAGE_TAG_RE.sub("", candidate_text)
        rows.append(
            {
                "stem": stem,
                "raw_exact": baseline_text == candidate_text,
                "image_tag_stripped_exact": baseline_stripped == candidate_stripped,
                "baseline_sha256": digest(baseline_text),
                "candidate_sha256": digest(candidate_text),
                "baseline_bytes": len(baseline_text.encode("utf-8")),
                "candidate_bytes": len(candidate_text.encode("utf-8")),
            }
        )

    raw_exact = sum(row["raw_exact"] for row in rows)
    stripped_exact = sum(row["image_tag_stripped_exact"] for row in rows)
    report = {
        "schema": "unirec_markdown_manifest_comparison_v1",
        "page_count": len(rows),
        "selection_sha256": manifest["selection"]["selection_sha256"],
        "raw_exact_count": raw_exact,
        "raw_exact_fraction": raw_exact / len(rows),
        "image_tag_stripped_exact_count": stripped_exact,
        "image_tag_stripped_exact_fraction": stripped_exact / len(rows),
        "differing_stems": [
            row["stem"] for row in rows if not row["image_tag_stripped_exact"]
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_MARKDOWN_SUBSET_COMPARISON: PASS "
        f"pages={len(rows)} raw_exact={raw_exact} stripped_exact={stripped_exact} "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
