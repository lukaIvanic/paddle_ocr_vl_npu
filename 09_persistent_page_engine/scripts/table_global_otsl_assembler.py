#!/usr/bin/env python3
"""Assemble row-crop OTSL before converting the complete table to HTML."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.layout_output import (  # noqa: E402
    _parse_otsl_rows,
    convert_otsl_to_html,
    truncate_repetitive_content,
)


CELL_TAGS = ("fcel", "ecel", "lcel", "ucel", "xcel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("raw_concat", "robust_rows"),
        default="raw_concat",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_raw(value: str) -> str:
    return truncate_repetitive_content(value, min_count=5000).strip()


def join_raw_parts(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    inserted_newlines = 0
    rejected = 0
    for row in sorted(rows, key=lambda item: int(item["row_index"])):
        raw = clean_raw(str(row.get("raw_text") or row.get("text") or ""))
        if not _parse_otsl_rows(raw):
            rejected += 1
            continue
        if parts and not parts[-1].rstrip().endswith("<nl>"):
            parts.append("<nl>")
            inserted_newlines += 1
        parts.append(raw)
    return "".join(parts), {
        "source_lanes": len(rows),
        "accepted_lanes": len(rows) - rejected,
        "rejected_non_otsl_lanes": rejected,
        "inserted_lane_newlines": inserted_newlines,
    }


def serialize_rows(rows: list[list[tuple[str, str]]]) -> str:
    return "".join(
        "".join(f"<{token}>{text}" for token, text in row) + "<nl>"
        for row in rows
    )


def robust_rows(raw: str) -> tuple[str, dict[str, Any]]:
    parsed = _parse_otsl_rows(raw)
    widths = [len(row) for row in parsed if row]
    if not widths:
        return raw, {"logical_rows": 0, "modal_width": 0}
    width_counts = Counter(widths)
    modal_width, modal_support = max(
        width_counts.items(), key=lambda item: (item[1], item[0])
    )
    kept: list[list[tuple[str, str]]] = []
    discarded = 0
    padded = 0
    for row in parsed:
        width = len(row)
        if width > max(modal_width + 2, modal_width * 2):
            discarded += 1
            continue
        if 0 < modal_width - width <= 2:
            row = row + [("ecel", "")] * (modal_width - width)
            padded += 1
        kept.append(row)
    return serialize_rows(kept), {
        "logical_rows": len(parsed),
        "kept_rows": len(kept),
        "discarded_width_outliers": discarded,
        "padded_rows": padded,
        "modal_width": modal_width,
        "modal_width_support": modal_support,
        "width_histogram": dict(sorted(width_counts.items())),
    }


def assemble(record: dict[str, Any], mode: str) -> dict[str, Any]:
    raw, diagnostics = join_raw_parts(record.get("rows") or [])
    if mode == "robust_rows":
        raw, robust = robust_rows(raw)
        diagnostics.update(robust)
    prediction = convert_otsl_to_html(raw)
    result = dict(record)
    result["pred_html"] = prediction
    result["global_otsl_raw"] = raw
    result["global_otsl_assembly"] = {
        "mode": mode,
        **diagnostics,
        "output_html_chars": len(prediction),
    }
    return result


def main() -> None:
    args = parse_args()
    records = [assemble(record, args.mode) for record in read_jsonl(args.input)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "tables.jsonl"
    output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    summary = {
        "input": str(args.input),
        "mode": args.mode,
        "tables": len(records),
        "output": str(output),
    }
    (args.output_dir / "assembly_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"assembled tables={len(records)} mode={args.mode} output={output}")


if __name__ == "__main__":
    main()
