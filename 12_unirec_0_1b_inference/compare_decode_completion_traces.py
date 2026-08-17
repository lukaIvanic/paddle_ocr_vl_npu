#!/usr/bin/env python3
"""Compare a UniRec completion trace with a token-only reference trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "sum": int(array.sum()),
        "min": int(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        count += 1
    return count


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def max_token_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    maximum = current = 1
    for previous, value in zip(tokens, tokens[1:]):
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
    return maximum


def max_ngram_occurrences(tokens: list[int], width: int = 4) -> int:
    if len(tokens) < width:
        return 0
    return max(
        Counter(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)).values()
    )


def main() -> None:
    args = parse_args()
    candidate_rows = read_jsonl(args.candidate.expanduser().resolve())
    reference_rows = read_jsonl(args.reference.expanduser().resolve())
    if not candidate_rows:
        raise ValueError("candidate completion trace is empty")
    if not reference_rows:
        raise ValueError("reference token trace is empty")
    reference = {str(row["request_id"]): row for row in reference_rows}
    if len(reference) != len(reference_rows):
        raise ValueError("reference token trace contains duplicate request IDs")
    candidate_ids = [str(row["request_id"]) for row in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate completion trace contains duplicate request IDs")
    lengths: list[int] = []
    reference_lengths: list[int] = []
    prefix_lengths: list[int] = []
    terminations: Counter[str] = Counter()
    exact = 0
    length_exact = 0
    missing = 0
    max_runs: list[int] = []
    max_4grams: list[int] = []
    mismatches: list[dict[str, Any]] = []
    for row in candidate_rows:
        tokens = [int(token) for token in row["token_ids"]]
        generated_length = int(row.get("generated_token_count", max(0, len(tokens) - 1)))
        lengths.append(generated_length)
        terminations[str(row.get("termination", "unknown"))] += 1
        max_runs.append(max_token_run(tokens))
        max_4grams.append(max_ngram_occurrences(tokens))
        expected = reference.get(str(row["request_id"]))
        if expected is None:
            missing += 1
            continue
        expected_tokens_raw = expected.get("token_ids")
        expected_tokens = (
            [int(token) for token in expected_tokens_raw]
            if expected_tokens_raw is not None
            else None
        )
        expected_length = (
            max(0, len(expected_tokens) - 1)
            if expected_tokens is not None
            else int(expected["generated_token_count"])
        )
        expected_digest = (
            token_digest(expected_tokens)
            if expected_tokens is not None
            else str(expected["token_sha256"])
        )
        reference_lengths.append(expected_length)
        prefix = (
            common_prefix(tokens, expected_tokens)
            if expected_tokens is not None
            else None
        )
        if prefix is not None:
            prefix_lengths.append(prefix)
        if generated_length == expected_length:
            length_exact += 1
        if token_digest(tokens) == expected_digest:
            exact += 1
        elif len(mismatches) < 20:
            mismatch = {
                "request_id": row["request_id"],
                "label": row.get("label"),
                "termination": row.get("termination"),
                "candidate_length": generated_length,
                "reference_length": expected_length,
                "candidate_token_sha256": token_digest(tokens),
                "reference_token_sha256": expected_digest,
                "candidate_head": tokens[:16],
                "candidate_tail": tokens[-16:],
                "max_same_token_run": max_runs[-1],
                "max_4gram_occurrences": max_4grams[-1],
            }
            if prefix is not None and expected_tokens is not None:
                mismatch["common_prefix_tokens"] = prefix
                mismatch["candidate_at_divergence"] = tokens[
                    prefix : prefix + 16
                ]
                mismatch["reference_at_divergence"] = expected_tokens[
                    prefix : prefix + 16
                ]
            mismatches.append(mismatch)
    compared = len(candidate_rows) - missing
    report = {
        "schema": "unirec_decode_output_parity_v1",
        "status": "ok",
        "candidate_count": len(candidate_rows),
        "reference_count": len(reference_rows),
        "compared_count": compared,
        "missing_reference_count": missing,
        "candidate_generated_length": distribution(lengths),
        "reference_generated_length_for_compared_rows": distribution(reference_lengths),
        "termination_counts": dict(sorted(terminations.items())),
        "long_output_counts": {
            "ge_256": sum(value >= 256 for value in lengths),
            "ge_512": sum(value >= 512 for value in lengths),
            "ge_1024": sum(value >= 1024 for value in lengths),
            "ge_2047": sum(value >= 2047 for value in lengths),
        },
        "token_exact_count": exact,
        "token_exact_fraction": exact / compared if compared else None,
        "length_exact_count": length_exact,
        "length_exact_fraction": length_exact / compared if compared else None,
        "common_prefix_tokens": {
            "available": bool(prefix_lengths),
            "distribution": distribution(prefix_lengths),
        },
        "repetition_indicators": {
            "max_same_token_run": distribution(max_runs),
            "max_4gram_occurrences": distribution(max_4grams),
            "same_token_run_ge_16_count": sum(value >= 16 for value in max_runs),
            "fourgram_occurrences_ge_16_count": sum(value >= 16 for value in max_4grams),
        },
        "first_mismatches": mismatches,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(partial, output)
    print(
        "UNIREC_DECODE_OUTPUT_PARITY: PASS "
        f"compared={compared} exact={exact} length_exact={length_exact} "
        f"eos={terminations.get('eos', 0)} "
        f"length_cap={terminations.get('length_cap', 0)} "
        f"candidate_mean={report['candidate_generated_length'].get('mean')} "
        f"reference_mean={report['reference_generated_length_for_compared_rows'].get('mean')} "
        f"output={output}",
        flush=True,
    )
    print("DECODE_OUTPUT_PARITY_REPORT")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
