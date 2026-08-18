#!/usr/bin/env python3
"""Build one B128 probe cohort around known decode-output mismatches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch-report", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--artifact-crops", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=128)
    parser.add_argument("--control-max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.cohort_size < 1:
        parser.error("--cohort-size must be positive")

    report = json.loads(args.mismatch_report.read_text(encoding="utf-8"))
    mismatches = report.get("first_mismatches") or []
    if not mismatches:
        raise ValueError("mismatch report has no first_mismatches rows")
    mismatch_ids = [str(row["request_id"]) for row in mismatches]
    artifact_ids = {
        str(row["request_id"])
        for row in read_jsonl(args.artifact_crops.expanduser().resolve())
    }
    reference_rows = read_jsonl(args.reference_trace.expanduser().resolve())
    reference = {str(row["request_id"]): row for row in reference_rows}
    missing = [
        request_id
        for request_id in mismatch_ids
        if request_id not in artifact_ids or request_id not in reference
    ]
    if missing:
        raise ValueError(f"mismatch IDs missing from artifact/reference: {missing}")

    selected = list(dict.fromkeys(mismatch_ids))
    controls: list[str] = []
    for row in reference_rows:
        request_id = str(row["request_id"])
        if request_id in selected or request_id not in artifact_ids:
            continue
        generated = int(row.get("generated_token_count", 2**31 - 1))
        if generated > args.control_max_tokens:
            continue
        controls.append(request_id)
        selected.append(request_id)
        if len(selected) == args.cohort_size:
            break
    if len(selected) != args.cohort_size:
        raise ValueError(
            f"could select only {len(selected)}/{args.cohort_size} probe rows"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{value}\n" for value in selected), encoding="utf-8")
    summary = {
        "schema": "unirec_decode_probe_selection_v1",
        "status": "ok",
        "cohort_size": len(selected),
        "mismatch_count": len(mismatch_ids),
        "mismatch_ids": mismatch_ids,
        "control_count": len(controls),
        "control_max_tokens": args.control_max_tokens,
        "output": str(args.output.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_DECODE_PROBE_SELECTION: PASS "
        f"cohort={len(selected)} mismatches={len(mismatch_ids)} "
        f"controls={len(controls)} output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
