#!/usr/bin/env python3
"""Summarize Experiment07 small-vision-encoder case JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def case_key(record: dict[str, Any], *, include_attention: bool = True) -> tuple[Any, ...]:
    values: list[Any] = []
    if include_attention:
        values.append(record["attention"])
    values.extend(
        [
            record["preprocessor_min_pixels"],
            record["bucket_min_exclusive"],
            record["fixed_physical_seq_len"],
            record["batch_size"],
            record["ln_impl"],
            record["ln_linear_mode"],
            record["promptfa_pad_head_dim_to"],
        ]
    )
    return tuple(values)


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("case_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "small_visual_encoder_case":
            continue
        case = payload["case"]
        timing = payload["timing"]
        selection = payload["selection"]
        records.append(
            {
                "json": str(path),
                "attention": case["attention"],
                "backend": case["compile_backend"],
                "compile_api": case.get("compile_api"),
                "preprocessor_min_pixels": int(case["preprocessor_min_pixels"]),
                "bucket_min_exclusive": int(case["bucket_min_exclusive"]),
                "fixed_physical_seq_len": int(case["fixed_physical_seq_len"]),
                "batch_size": int(case["batch_size"]),
                "ln_impl": case["ln_impl"],
                "ln_linear_mode": case["ln_linear_mode"],
                "promptfa_pad_head_dim_to": int(case["promptfa_pad_head_dim_to"]),
                "effective_tokens": int(selection["selected_effective_tokens"]),
                "physical_tokens": int(selection["selected_physical_tokens"]),
                "useful_token_fraction": float(selection["selected_useful_token_fraction"]),
                "mean_forward_s": float(timing["mean_forward_s"]),
                "effective_tokens_per_s": float(timing["effective_tokens_per_s"]),
                "physical_tokens_per_s": float(timing["physical_tokens_per_s"]),
                "first_call_s": float(payload["compile"]["first_call_s"]),
                "correctness_passed": bool(payload["correctness"]["passed"]),
                "nonfinite_count": int(payload["correctness"]["final_output_nonfinite_count"]),
            }
        )
    return records


def make_backend_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_backend = {(case_key(record), record["backend"]): record for record in records}
    pairs: list[dict[str, Any]] = []
    for key in sorted({case_key(record) for record in records}, key=str):
        eager = by_key_backend.get((key, "none"))
        compiled = by_key_backend.get((key, "torchair"))
        if eager is None or compiled is None:
            continue
        pairs.append(
            {
                "attention": eager["attention"],
                "preprocessor_min_pixels": eager["preprocessor_min_pixels"],
                "bucket": [eager["bucket_min_exclusive"], eager["fixed_physical_seq_len"]],
                "batch_size": eager["batch_size"],
                "promptfa_pad_head_dim_to": eager["promptfa_pad_head_dim_to"],
                "eager_mean_forward_s": eager["mean_forward_s"],
                "compiled_mean_forward_s": compiled["mean_forward_s"],
                "latency_speedup_compiled_over_eager": safe_ratio(
                    eager["mean_forward_s"], compiled["mean_forward_s"]
                ),
                "eager_physical_tokens_per_s": eager["physical_tokens_per_s"],
                "compiled_physical_tokens_per_s": compiled["physical_tokens_per_s"],
                "physical_throughput_speedup_compiled_over_eager": safe_ratio(
                    compiled["physical_tokens_per_s"], eager["physical_tokens_per_s"]
                ),
                "eager_effective_tokens_per_s": eager["effective_tokens_per_s"],
                "compiled_effective_tokens_per_s": compiled["effective_tokens_per_s"],
                "effective_throughput_speedup_compiled_over_eager": safe_ratio(
                    compiled["effective_tokens_per_s"], eager["effective_tokens_per_s"]
                ),
                "eager_correctness_passed": eager["correctness_passed"],
                "compiled_correctness_passed": compiled["correctness_passed"],
            }
        )
    return pairs


def make_attention_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_attention_backend = {
        (case_key(record, include_attention=False), record["attention"], record["backend"]): record
        for record in records
    }
    pairs: list[dict[str, Any]] = []
    keys = {case_key(record, include_attention=False) for record in records}
    for key in sorted(keys, key=str):
        for backend in ("none", "torchair"):
            manual = by_key_attention_backend.get((key, "manual", backend))
            promptfa = by_key_attention_backend.get((key, "prompt_flash_attention", backend))
            if manual is None or promptfa is None:
                continue
            pairs.append(
                {
                    "backend": backend,
                    "preprocessor_min_pixels": manual["preprocessor_min_pixels"],
                    "bucket": [manual["bucket_min_exclusive"], manual["fixed_physical_seq_len"]],
                    "batch_size": manual["batch_size"],
                    "manual_mean_forward_s": manual["mean_forward_s"],
                    "promptfa_mean_forward_s": promptfa["mean_forward_s"],
                    "latency_speedup_promptfa_over_manual": safe_ratio(
                        manual["mean_forward_s"], promptfa["mean_forward_s"]
                    ),
                    "manual_physical_tokens_per_s": manual["physical_tokens_per_s"],
                    "promptfa_physical_tokens_per_s": promptfa["physical_tokens_per_s"],
                    "physical_throughput_speedup_promptfa_over_manual": safe_ratio(
                        promptfa["physical_tokens_per_s"], manual["physical_tokens_per_s"]
                    ),
                    "manual_correctness_passed": manual["correctness_passed"],
                    "promptfa_correctness_passed": promptfa["correctness_passed"],
                }
            )
    return pairs


def write_tsv(root: Path, records: list[dict[str, Any]], backend_pairs: list[dict[str, Any]]) -> None:
    columns = [
        "attention",
        "backend",
        "preprocessor_min_pixels",
        "bucket_min_exclusive",
        "fixed_physical_seq_len",
        "batch_size",
        "effective_tokens",
        "physical_tokens",
        "useful_token_fraction",
        "mean_forward_s",
        "effective_tokens_per_s",
        "physical_tokens_per_s",
        "first_call_s",
        "correctness_passed",
        "json",
    ]
    lines = ["\t".join(columns)]
    lines.extend("\t".join(str(record.get(column, "")) for column in columns) for record in records)
    (root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pair_columns = [
        "attention",
        "preprocessor_min_pixels",
        "bucket",
        "batch_size",
        "eager_mean_forward_s",
        "compiled_mean_forward_s",
        "latency_speedup_compiled_over_eager",
        "eager_physical_tokens_per_s",
        "compiled_physical_tokens_per_s",
        "physical_throughput_speedup_compiled_over_eager",
        "eager_correctness_passed",
        "compiled_correctness_passed",
    ]
    pair_lines = ["\t".join(pair_columns)]
    pair_lines.extend(
        "\t".join(
            json.dumps(pair.get(column)) if column == "bucket" else str(pair.get(column, ""))
            for column in pair_columns
        )
        for pair in backend_pairs
    )
    (root / "compiled_vs_eager.tsv").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    records = load_records(root)
    if not records:
        raise ValueError(f"no small visual encoder case JSON files found under {root}")
    backend_pairs = make_backend_pairs(records)
    attention_pairs = make_attention_pairs(records)
    output = {
        "schema_version": 1,
        "kind": "small_visual_encoder_matrix_summary",
        "root": str(root),
        "case_count": int(len(records)),
        "all_correctness_passed": bool(all(record["correctness_passed"] for record in records)),
        "records": records,
        "compiled_vs_eager": backend_pairs,
        "promptfa_vs_manual": attention_pairs,
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_tsv(root, records, backend_pairs)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
