#!/usr/bin/env python3
"""Run CDM directly from a saved OmniDocBench display-formula match result."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--save-name", default="predictions_quick_match_cdm")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")

    input_path = args.input.resolve()
    evaluator_root = args.evaluator_root.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (evaluator_root / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {evaluator_root}")

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError(f"expected a list in {input_path}")
    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CDM_SAVE_VIS"] = "1" if args.save_vis else "0"
    os.chdir(output_dir)
    sys.path.insert(0, str(evaluator_root))

    from src.metrics.cal_metric import call_CDM

    print(
        f"[cdm-direct] samples={len(samples)} workers={args.workers} "
        f"save_vis={args.save_vis} output={output_dir}",
        flush=True,
    )
    started = time.monotonic()
    metric = call_CDM(samples, {"cdm_workers": args.workers})
    _evaluated_samples, scores = metric.evaluate(
        save_name=args.save_name,
        max_workers=args.workers,
    )
    wall_s = time.monotonic() - started

    result_root = output_dir / "result"
    summary = {
        "input": str(input_path),
        "evaluator_root": str(evaluator_root),
        "sample_count": len(samples),
        "workers": args.workers,
        "save_vis": args.save_vis,
        "wall_s": wall_s,
        "samples_per_s": len(samples) / wall_s if wall_s else None,
        "scores": scores,
        "debug": metric.debug_info,
        "per_sample_scores": str(
            result_root / f"{args.save_name}_per_sample_CDM.json"
        ),
        "evaluated_samples": str(result_root / f"{args.save_name}_result.json"),
    }
    summary_path = output_dir / "cdm_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[cdm-direct] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
