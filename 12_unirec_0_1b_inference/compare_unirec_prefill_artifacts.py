#!/usr/bin/env python3
"""Compare matching real-length cross-KV rows from two UniRec artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("request-ID file must be non-empty and unique")
    return values


def load_manifest(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(directory / "crops.jsonl")
    by_id = {str(row["request_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate request IDs in {directory}")
    return summary, by_id


def array_view(storage: np.memmap, row: dict[str, Any]) -> np.ndarray:
    spec = row["cross_kv"]
    return np.ndarray(
        shape=tuple(int(value) for value in spec["shape"]),
        dtype=np.dtype(spec["dtype"]),
        buffer=storage,
        offset=int(spec["offset"]),
        order="C",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--request-ids-file", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_dir = args.reference.expanduser().resolve()
    candidate_dir = args.candidate.expanduser().resolve()
    reference_summary, reference = load_manifest(reference_dir)
    candidate_summary, candidate = load_manifest(candidate_dir)
    ids = read_ids(args.request_ids_file)
    if ids is None:
        ids = list(reference)
    missing = [value for value in ids if value not in reference or value not in candidate]
    if missing:
        raise ValueError(f"request IDs missing from one artifact: {missing[:20]}")

    reference_files = {str(reference[value]["cross_kv"]["file"]) for value in ids}
    candidate_files = {str(candidate[value]["cross_kv"]["file"]) for value in ids}
    if len(reference_files) != 1 or len(candidate_files) != 1:
        raise ValueError("each artifact must use one cross-KV data file")
    reference_storage = np.memmap(
        reference_dir / next(iter(reference_files)), dtype=np.uint8, mode="r"
    )
    candidate_storage = np.memmap(
        candidate_dir / next(iter(candidate_files)), dtype=np.uint8, mode="r"
    )

    total_elements = 0
    total_abs = 0.0
    total_squared = 0.0
    exact_rows = 0
    manifest_mismatches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for request_id in ids:
        left_row = reference[request_id]
        right_row = candidate[request_id]
        left_spec = left_row["cross_kv"]
        right_spec = right_row["cross_kv"]
        manifest_fields = {
            "label": (left_row.get("label"), right_row.get("label")),
            "shape": (left_spec["shape"], right_spec["shape"]),
            "source_length": (
                left_spec["source_length"],
                right_spec["source_length"],
            ),
        }
        unequal_fields = {
            name: values for name, values in manifest_fields.items() if values[0] != values[1]
        }
        if unequal_fields:
            manifest_mismatches.append(
                {"request_id": request_id, "fields": unequal_fields}
            )
            continue
        left = array_view(reference_storage, left_row)
        right = array_view(candidate_storage, right_row)
        exact = bool(np.array_equal(left, right))
        exact_rows += int(exact)
        delta = left.astype(np.float32) - right.astype(np.float32)
        absolute = np.abs(delta)
        count = int(delta.size)
        abs_sum = float(absolute.sum(dtype=np.float64))
        squared_sum = float(np.square(delta).sum(dtype=np.float64))
        reference_rms = math.sqrt(
            float(np.square(left.astype(np.float32)).sum(dtype=np.float64)) / count
        )
        row = {
            "request_id": request_id,
            "label": left_row.get("label"),
            "source_length": int(left_spec["source_length"]),
            "elements": count,
            "exact": exact,
            "max_abs": float(absolute.max()),
            "mean_abs": abs_sum / count,
            "rmse": math.sqrt(squared_sum / count),
            "relative_rmse": (
                math.sqrt(squared_sum / count) / reference_rms
                if reference_rms
                else None
            ),
        }
        rows.append(row)
        total_elements += count
        total_abs += abs_sum
        total_squared += squared_sum

    compared_rows = len(rows)
    report = {
        "schema": "unirec_prefill_artifact_comparison_v1",
        "status": "ok",
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "reference_prefill_wall_s": reference_summary.get("producer_wall_s"),
        "candidate_prefill_wall_s": candidate_summary.get("producer_wall_s"),
        "reference_prefill_pages_per_s": reference_summary.get(
            "throughput", {}
        ).get("pages_per_s"),
        "candidate_prefill_pages_per_s": candidate_summary.get(
            "throughput", {}
        ).get("pages_per_s"),
        "selected_rows": len(ids),
        "compared_rows": compared_rows,
        "manifest_mismatch_count": len(manifest_mismatches),
        "first_manifest_mismatches": manifest_mismatches[: args.top],
        "exact_rows": exact_rows,
        "exact_fraction": exact_rows / compared_rows if compared_rows else None,
        "elements": total_elements,
        "weighted_mean_abs": total_abs / total_elements if total_elements else None,
        "weighted_rmse": (
            math.sqrt(total_squared / total_elements) if total_elements else None
        ),
        "max_abs": max((row["max_abs"] for row in rows), default=None),
        "top_rows_by_rmse": sorted(
            rows, key=lambda row: row["rmse"], reverse=True
        )[: args.top],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    print(
        "UNIREC_PREFILL_ARTIFACT_COMPARISON: PASS "
        f"rows={compared_rows} exact={exact_rows} "
        f"mean_abs={report['weighted_mean_abs']:.8g} "
        f"rmse={report['weighted_rmse']:.8g} max_abs={report['max_abs']:.8g} "
        f"output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
