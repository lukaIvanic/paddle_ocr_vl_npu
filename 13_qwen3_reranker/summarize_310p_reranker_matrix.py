#!/usr/bin/env python3
"""Summarize the 310P dense and W8A8 reranker throughput cross-matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASE_ORDER = {
    ("06b", "dense"): 0,
    ("4b", "dense"): 1,
    ("06b", "w8a8"): 2,
    ("4b", "w8a8"): 3,
    ("06b", "separate_qkv_ffn_w8a8"): 4,
    ("4b", "separate_qkv_ffn_w8a8"): 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--comparison-mode", default="w8a8")
    parser.add_argument("--expected-cells", type=int)
    return parser.parse_args()


def load_rows(output_root: Path) -> list[dict]:
    rows = []
    for result_path in output_root.glob("*_*/b*/result.json"):
        phase_key = result_path.parents[1].name
        if phase_key.endswith("_full_total"):
            phase_key = phase_key.removesuffix("_full_total")
            path_kind = "full_total"
            expected_lane = "full_promptfa_compiled"
        else:
            path_kind = "prefix"
            expected_lane = "prefix_promptfa_compiled"
        try:
            model, mode = phase_key.split("_", maxsplit=1)
        except ValueError:
            continue
        if (model, mode) not in PHASE_ORDER:
            continue
        payload = json.loads(result_path.read_text())
        environment = payload["environment"]
        configuration = payload["configuration"]
        for result in payload["results"]:
            if result["lane"] != expected_lane:
                continue
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "path": path_kind,
                    "batch_size": result["batch_size"],
                    "continuation_length": result["continuation_length"],
                    "full_physical_length": result["full_physical_length"],
                    "requested_sequence_length": result.get(
                        "requested_sequence_length"
                    ),
                    "median_ms": result["median_s"] * 1000.0,
                    "pairs_s": result["pairs_s"],
                    "served_input_tok_s": result["served_input_tok_s"],
                    "executed_model_tok_s": result["executed_model_tok_s"],
                    "physical_attention_q_tok_s": result[
                        "physical_attention_q_tok_s"
                    ],
                    "first_call_s": result["first_call_s"],
                    "cache_was_warm": result["cache_was_warm"],
                    "same_yes_no_choice_vs_prefix_baseline": result.get(
                        "same_yes_no_choice_vs_prefix_baseline"
                    ),
                    "device": environment["device"],
                    "git_commit": environment.get("git_commit"),
                    "source_hash": configuration["source_hash"],
                    "result_path": str(result_path),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            PHASE_ORDER[(row["model"], row["mode"])],
            row["batch_size"],
            row["continuation_length"],
        ),
    )


def comparisons(rows: list[dict], *, quantized_mode: str) -> list[dict]:
    indexed = {
        (
            row["path"],
            row["model"],
            row["mode"],
            row["batch_size"],
            row["continuation_length"],
        ): row
        for row in rows
    }
    values = []
    for key, dense in indexed.items():
        path_kind, model, mode, batch_size, continuation_length = key
        if mode != "dense":
            continue
        quantized = indexed.get(
            (path_kind, model, quantized_mode, batch_size, continuation_length)
        )
        if quantized is None:
            continue
        comparison = {
                "model": model,
                "path": path_kind,
                "batch_size": batch_size,
                "continuation_length": continuation_length,
                "quantized_mode": quantized_mode,
                "dense_median_ms": dense["median_ms"],
                "quantized_median_ms": quantized["median_ms"],
                "latency_speedup": dense["median_ms"] / quantized["median_ms"],
                "dense_executed_model_tok_s": dense["executed_model_tok_s"],
                "quantized_executed_model_tok_s": quantized["executed_model_tok_s"],
                "executed_model_tok_s_speedup": (
                    quantized["executed_model_tok_s"]
                    / dense["executed_model_tok_s"]
                ),
            }
        if quantized_mode == "w8a8":
            comparison.update(
                w8a8_median_ms=quantized["median_ms"],
                w8a8_executed_model_tok_s=quantized["executed_model_tok_s"],
            )
        values.append(comparison)
    return sorted(
        values,
        key=lambda row: (
            0 if row["model"] == "06b" else 1,
            row["batch_size"],
            row["continuation_length"],
        ),
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.output_root)
    paired = comparisons(rows, quantized_mode=args.comparison_mode)
    expected_cells = args.expected_cells
    if expected_cells is None:
        expected_cells = 64 if args.comparison_mode == "w8a8" else 32
    summary = {
        "output_root": str(args.output_root),
        "completed_cells": len(rows),
        "expected_cells": expected_cells,
        "comparison_mode": args.comparison_mode,
        "rows": rows,
        "comparisons": paired,
    }
    print(
        f"MATRIX_SUMMARY completed_cells={len(rows)} expected_cells={expected_cells} "
        f"paired_comparisons={len(paired)}"
    )
    for row in rows:
        print("MATRIX_SUMMARY_ROW " + json.dumps(row, sort_keys=True))
    for comparison in paired:
        print("MATRIX_COMPARISON " + json.dumps(comparison, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
