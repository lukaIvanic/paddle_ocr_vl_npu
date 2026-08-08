#!/usr/bin/env python3
"""Simulate hierarchical row-draft speculation from saved OCR tokens.

Each finer row corpus drafts the next coarser row corpus.  Coarse rows are
independent target sequences, so their target-call cost is the maximum row
call count for that table, not the sum.  The last stage drafts the saved
whole-table target.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
from typing import Any

from table_multicandidate_simulator import read_jsonl, simulate_one, target_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--row-corpus",
        action="append",
        required=True,
        metavar="ROWS=PATH",
        help="Fine-to-coarse saved row OCR corpus, for example 8=u8.jsonl.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draft-lengths", default="16,64,128")
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--maximum-anchor", type=int, default=64)
    parser.add_argument("--column-weight", type=float, default=0.25)
    parser.add_argument(
        "--source-mode",
        choices=("previous", "all"),
        default="previous",
        help="Use only the preceding level or all already-generated levels.",
    )
    return parser.parse_args()


def parse_corpora(values: list[str]) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for value in values:
        rows_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"row corpus must be ROWS=PATH, got {value!r}")
        rows = int(rows_text)
        if rows <= 0:
            raise ValueError(f"row count must be positive, got {rows}")
        result.append((rows, Path(path_text)))
    if len(result) < 1:
        raise ValueError("at least one row corpus is required")
    if any(fine <= coarse for (fine, _), (coarse, _) in zip(result, result[1:])):
        raise ValueError("row corpora must be ordered from fine to coarse")
    return result


def index_records(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["request_id"]): record for record in read_jsonl(path)}


def one_row_target(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    result = copy.copy(record)
    result["rows"] = [row]
    return result


def combined_draft(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.copy(records[-1])
    rows: list[dict[str, Any]] = []
    for level, record in enumerate(records):
        for row in sorted(record.get("rows") or [], key=lambda item: item["row_index"]):
            cloned = copy.copy(row)
            cloned["row_index"] = len(rows)
            cloned["draft_level"] = level
            rows.append(cloned)
    result["rows"] = rows
    return result


def summarize(values: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "sum": sum(values),
        "maximum": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    corpora = parse_corpora(args.row_corpus)
    indexed_levels = [(rows, index_records(path)) for rows, path in corpora]
    targets = index_records(args.targets)
    draft_lengths = [int(value) for value in args.draft_lengths.split(",") if value]
    common = set(targets)
    for _rows, records in indexed_levels:
        common &= set(records)
    request_ids = sorted(common)
    if not request_ids:
        raise ValueError("target and row corpora have no common request IDs")

    all_results: dict[str, Any] = {}
    for draft_length in draft_lengths:
        per_table: list[dict[str, Any]] = []
        for table_index, request_id in enumerate(request_ids, start=1):
            generated_sources: list[dict[str, Any]] = []
            stages: list[dict[str, Any]] = []
            for level_index, (row_count, records) in enumerate(indexed_levels):
                current = records[request_id]
                if level_index == 0:
                    generated_sources.append(current)
                    continue
                source_records = (
                    generated_sources if args.source_mode == "all" else generated_sources[-1:]
                )
                draft_record = combined_draft(source_records)
                row_results: list[dict[str, Any]] = []
                for row in sorted(current.get("rows") or [], key=lambda item: item["row_index"]):
                    target_record = one_row_target(current, row)
                    simulation = simulate_one(
                        target_tokens(target_record),
                        draft_record,
                        tokenizer,
                        candidate_count=args.candidate_count,
                        draft_length=draft_length,
                        maximum_anchor=args.maximum_anchor,
                        column_weight=args.column_weight,
                    )
                    row_results.append(
                        {
                            "row_index": int(row["row_index"]),
                            "row_y": row.get("row_y"),
                            "stop_reason": row.get("stop_reason"),
                            "simulation": simulation,
                        }
                    )
                calls = [row["simulation"]["target_calls"] for row in row_results]
                stages.append(
                    {
                        "stage": f"rows_{indexed_levels[level_index - 1][0]}_to_rows_{row_count}",
                        "concurrent_target_sequences": len(row_results),
                        "target_calls": summarize(calls),
                        "rows": row_results,
                    }
                )
                generated_sources.append(current)

            source_records = (
                generated_sources if args.source_mode == "all" else generated_sources[-1:]
            )
            final_simulation = simulate_one(
                target_tokens(targets[request_id]),
                combined_draft(source_records),
                tokenizer,
                candidate_count=args.candidate_count,
                draft_length=draft_length,
                maximum_anchor=args.maximum_anchor,
                column_weight=args.column_weight,
            )
            stages.append(
                {
                    "stage": f"rows_{indexed_levels[-1][0]}_to_full",
                    "concurrent_target_sequences": 1,
                    "target_calls": summarize([final_simulation["target_calls"]]),
                    "simulation": final_simulation,
                }
            )
            per_table.append({"request_id": request_id, "stages": stages})
            if table_index == 1 or table_index % 10 == 0 or table_index == len(request_ids):
                print(
                    f"D={draft_length} progress={table_index}/{len(request_ids)}",
                    flush=True,
                )

        stage_names = [stage["stage"] for stage in per_table[0]["stages"]]
        aggregate: dict[str, Any] = {}
        for stage_name in stage_names:
            table_calls = [
                next(stage for stage in table["stages"] if stage["stage"] == stage_name)[
                    "target_calls"
                ]["maximum"]
                for table in per_table
            ]
            aggregate[stage_name] = {"per_table_concurrent_calls": summarize(table_calls)}
        all_results[str(draft_length)] = {
            "aggregate": aggregate,
            "tables": per_table,
        }

    result = {
        "configuration": {
            "targets": str(args.targets),
            "row_corpora": [{"rows": rows, "path": str(path)} for rows, path in corpora],
            "draft_lengths": draft_lengths,
            "candidate_count": args.candidate_count,
            "maximum_anchor": args.maximum_anchor,
            "column_weight": args.column_weight,
            "source_mode": args.source_mode,
            "tables": len(request_ids),
        },
        "results": all_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote={output}", flush=True)
    for draft_length in draft_lengths:
        summary = all_results[str(draft_length)]["aggregate"]
        print(
            f"D={draft_length} "
            + " ".join(
                f"{name}={metrics['per_table_concurrent_calls']['sum']}"
                for name, metrics in summary.items()
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
