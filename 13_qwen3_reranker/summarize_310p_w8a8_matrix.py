#!/usr/bin/env python3
"""Summarize the self-contained 310P dense/W8A8 experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    summary: dict[str, object] = {"run_root": str(args.run_root), "operator_probes": {}, "models": {}}
    for model_key in ("06b", "4b"):
        probe = load_json(args.run_root / f"op_{model_key}" / "result.json")
        if probe is not None:
            summary["operator_probes"][model_key] = {
                "device": probe["environment"]["device"],
                "shape": probe["shape"],
                "weight_quantization_s": probe["weight_quantization_s"],
                "dense_median_ms": probe["dense"]["median_s"] * 1000.0,
                "w8a8_median_ms": probe["w8a8"]["median_s"] * 1000.0,
                "speedup": probe["speedup"],
                "output_diff": probe["output_diff"],
                "quant_weight_format": probe["quant_weight_format"],
            }

        dense = load_json(args.run_root / f"model_{model_key}_dense" / "result.json")
        quant = load_json(args.run_root / f"model_{model_key}_w8a8" / "result.json")
        if dense is None or quant is None:
            summary["models"][model_key] = {
                "complete": False,
                "dense_result_present": dense is not None,
                "w8a8_result_present": quant is not None,
            }
            continue
        dense_result = dense["results"][0]
        quant_result = quant["results"][0]
        summary["models"][model_key] = {
            "complete": True,
            "device": quant["environment"]["device"],
            "batch_size": quant_result["batch_size"],
            "continuation_length": quant_result["continuation_length"],
            "dense_median_ms": dense_result["median_s"] * 1000.0,
            "w8a8_median_ms": quant_result["median_s"] * 1000.0,
            "speedup": dense_result["median_s"] / quant_result["median_s"],
            "dense_executed_tok_s": dense_result["executed_model_tok_s"],
            "w8a8_executed_tok_s": quant_result["executed_model_tok_s"],
            "executed_tok_s_gain_pct": (
                quant_result["executed_model_tok_s"] / dense_result["executed_model_tok_s"] - 1.0
            )
            * 100.0,
            "same_yes_no_choices": dense_result["yes_no_choices"] == quant_result["yes_no_choices"],
            "dense_yes_scores": dense_result["yes_scores"],
            "w8a8_yes_scores": quant_result["yes_scores"],
            "weight_quantization_s": quant["configuration"].get("weight_quantization_s"),
            "dense_cache_was_warm": dense_result["cache_was_warm"],
            "w8a8_cache_was_warm": quant_result["cache_was_warm"],
        }

    output_path = args.run_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("W8A8_310P_MATRIX_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"SUMMARY_JSON {output_path}", flush=True)


if __name__ == "__main__":
    main()
