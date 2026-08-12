#!/usr/bin/env python3
"""Build a decode vocabulary from saved native generation token IDs.

This tool never decodes or encodes text.  It reads ``rows[*].token_ids`` from
the detailed table-generation JSONL artifact, ranks IDs by observed frequency,
then fills unused capacity with unseen checkpoint IDs in ascending order.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-size", type=int, default=16384)
    parser.add_argument("--full-vocab-size", type=int, default=103424)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.selected_size <= args.full_vocab_size:
        raise ValueError("selected size must be within the full vocabulary")

    counts: Counter[int] = Counter()
    tables = 0
    with args.input.expanduser().resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tables += 1
            for row in record.get("rows") or ():
                counts.update(int(value) for value in row.get("token_ids") or ())
    if not counts:
        raise ValueError("input contains no rows[*].token_ids")
    invalid = [
        token_id
        for token_id in counts
        if not 0 <= token_id < args.full_vocab_size
    ]
    if invalid:
        raise ValueError(f"native IDs outside the checkpoint vocabulary: {invalid[:16]}")

    selected = [
        token_id
        for token_id, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    selected.extend(
        token_id
        for token_id in range(args.full_vocab_size)
        if token_id not in counts
    )
    selected = selected[: args.selected_size]
    digest_payload = json.dumps(selected, separators=(",", ":")).encode("utf-8")
    payload = {
        "format": "paddleocr_vl_decode_vocab_v1",
        "source": str(args.input.expanduser().resolve()),
        "source_tables": tables,
        "native_generated_token_occurrences": sum(counts.values()),
        "native_unique_token_ids": len(counts),
        "full_vocab_size": args.full_vocab_size,
        "selected_vocab_size": len(selected),
        "selection": (
            "native output frequency descending, token ID ascending tie break; "
            "then unseen token IDs ascending"
        ),
        "covers_all_source_token_ids": set(counts).issubset(selected),
        "token_ids_sha256": hashlib.sha256(digest_payload).hexdigest(),
        "token_ids": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "token_ids"}))


if __name__ == "__main__":
    main()
