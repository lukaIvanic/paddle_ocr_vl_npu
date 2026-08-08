#!/usr/bin/env python3
"""Score saved full-sequence Jacobi candidates with OmniDocBench TEDS."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    parser.add_argument("--teds-tolerance", type=float, default=0.005)
    parser.add_argument("--latency-limit-s", type=float, default=2.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def html_document(value: str) -> str:
    text = str(value)
    if "<html" in text.lower():
        return text
    return f"<html><body>{text}</body></html>"


class Scorer:
    def __init__(self, evaluator_root: Path):
        sys.path.insert(0, str(evaluator_root.resolve()))
        from src.metrics.table_metric import TEDS

        self.content = TEDS(structure_only=False)
        self.structure = TEDS(structure_only=True)

    def __call__(self, prediction: str, target: str) -> dict[str, float]:
        prediction_doc = html_document(prediction)
        target_doc = html_document(target)
        started = time.perf_counter()
        return {
            "teds": float(self.content.evaluate(prediction_doc, target_doc)),
            "structure_teds": float(
                self.structure.evaluate(prediction_doc, target_doc)
            ),
            "scoring_wall_s": time.perf_counter() - started,
        }


def score_candidate(
    scorer: Scorer,
    candidate: dict[str, Any],
    *,
    gt_html: str,
    baseline_html: str,
    baseline_teds: float,
    tolerance: float,
    latency_limit_s: float,
) -> None:
    candidate.pop("teds_pending", None)
    candidate["versus_gt"] = scorer(candidate["pred_html"], gt_html)
    candidate["versus_live_baseline"] = scorer(
        candidate["pred_html"], baseline_html
    )
    candidate["teds_delta_from_baseline"] = (
        candidate["versus_gt"]["teds"] - baseline_teds
    )
    candidate["quality_gate_pass"] = (
        candidate["teds_delta_from_baseline"] >= -float(tolerance)
    )
    if "composed_pipeline_wall_s" in candidate:
        candidate["latency_gate_pass"] = (
            candidate["composed_pipeline_wall_s"] < float(latency_limit_s)
        )
        candidate["combined_gate_pass"] = (
            candidate["quality_gate_pass"] and candidate["latency_gate_pass"]
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args.evaluator_root)
    records = copy.deepcopy(read_jsonl(args.input))
    configuration: dict[str, dict[str, int]] = {}
    control_passes = 0

    output_path = args.output_dir / "tables_scored.jsonl"
    output_path.write_text("")
    for index, record in enumerate(records, start=1):
        gt_html = str(record["gt_html"])
        baseline = record["baseline"]
        baseline["scores"] = scorer(baseline["pred_html"], gt_html)
        baseline_teds = float(baseline["scores"]["teds"])
        control = record["self_projection_control"]
        score_candidate(
            scorer,
            control,
            gt_html=gt_html,
            baseline_html=baseline["pred_html"],
            baseline_teds=baseline_teds,
            tolerance=args.teds_tolerance,
            latency_limit_s=args.latency_limit_s,
        )
        control_passes += int(control["quality_gate_pass"])

        for seed_name, sweeps in record["seeds"].items():
            for sweep in sweeps:
                score_candidate(
                    scorer,
                    sweep,
                    gt_html=gt_html,
                    baseline_html=baseline["pred_html"],
                    baseline_teds=baseline_teds,
                    tolerance=args.teds_tolerance,
                    latency_limit_s=args.latency_limit_s,
                )
                key = f"{seed_name}_K{sweep['sweep']}"
                counters = configuration.setdefault(
                    key,
                    {"tables": 0, "quality": 0, "latency": 0, "combined": 0},
                )
                counters["tables"] += 1
                counters["quality"] += int(sweep["quality_gate_pass"])
                counters["latency"] += int(sweep["latency_gate_pass"])
                counters["combined"] += int(sweep["combined_gate_pass"])

        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        flat_final = record["seeds"]["flat_prefix"][-1]
        balanced_final = record["seeds"]["balanced_lane"][-1]
        print(
            f"FULL_JACOBI_SCORE table={index}/{len(records)} "
            f"id={record['request_id']} baseline={baseline_teds:.6f} "
            f"control={control['teds_delta_from_baseline']:+.6f} "
            f"flat_K{flat_final['sweep']}={flat_final['versus_gt']['teds']:.6f}/"
            f"{flat_final['composed_pipeline_wall_s']:.3f}s "
            f"balanced_K{balanced_final['sweep']}="
            f"{balanced_final['versus_gt']['teds']:.6f}/"
            f"{balanced_final['composed_pipeline_wall_s']:.3f}s",
            flush=True,
        )

    summary = {
        "tables": len(records),
        "teds_tolerance": args.teds_tolerance,
        "latency_limit_s": args.latency_limit_s,
        "control_quality_passes": control_passes,
        "control_valid": control_passes == len(records),
        "configuration": configuration,
        "winning_general_configurations": [
            key
            for key, counters in configuration.items()
            if counters["combined"] == counters["tables"] == len(records)
        ],
        "records": str(output_path),
    }
    write_json(args.output_dir / "score_summary.json", summary)
    print(
        f"FULL_JACOBI_SCORE_COMPLETE controls={control_passes}/{len(records)} "
        f"winning={summary['winning_general_configurations']} "
        f"output={args.output_dir}",
        flush=True,
    )
    if not summary["control_valid"]:
        raise RuntimeError("self-projection control exceeded the TEDS gate")


if __name__ == "__main__":
    main()
