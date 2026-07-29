#!/usr/bin/env python3
"""Run the bounded S512/S2048, native/NZ, 4304/4352 MatMul matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_matmul_lab"
)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_matmul_lab"
)
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_vision_matmul_profiles"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument(
        "--execution",
        choices=("raw_eager", "torchair"),
        default="torchair",
    )
    parser.add_argument("--allow-compile-if-missing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    return parser.parse_args(argv)


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    measurements = summary.get("measurements", {})
    parsed = summary.get("parsed_profile", {})
    return {
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "shape": summary.get("shape"),
        "requested": summary.get("requested"),
        "weight_format": summary.get("weight_format"),
        "device_median_ms": measurements.get(
            "device_event_per_call_ms", {}
        ).get("median"),
        "physical_tokens_per_s": measurements.get(
            "physical_tokens_per_s_device_median"
        ),
        "linear_tflop_per_s": measurements.get(
            "linear_tflop_per_s_device_median"
        ),
        "dispatch": parsed.get("dispatch"),
        "transdata": parsed.get("transdata"),
        "weighted_cube_utilization_pct": parsed.get(
            "weighted_cube_utilization_pct"
        ),
        "numerics": summary.get("numerics"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = (args.output_root / args.name).expanduser().resolve()
    profile_root = (args.profile_root / args.name).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"matrix output directory already exists and is non-empty: "
            f"{output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for sequence_length in (512, 2048):
        for intermediate_size in (4304, 4352):
            for weight_format in ("native", "fractal_nz"):
                case_name = (
                    f"s{sequence_length}_i{intermediate_size}_{weight_format}"
                )
                case_output = output_root / case_name
                case_profile = profile_root / case_name
                command = [
                    sys.executable,
                    str(HERE / "vision_matmul_lab.py"),
                    "--sequence-length",
                    str(sequence_length),
                    "--intermediate-size",
                    str(intermediate_size),
                    "--weight-format",
                    weight_format,
                    "--execution",
                    args.execution,
                    "--model",
                    str(args.model),
                    "--cache-dir",
                    str(args.cache_dir),
                    "--output-dir",
                    str(case_output),
                    "--profile-dir",
                    str(case_profile),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--calls-per-sample",
                    str(args.calls_per_sample),
                ]
                if args.allow_compile_if_missing:
                    command.append("--allow-compile-if-missing")
                if args.profile:
                    command.append("--profile")
                print(
                    json.dumps(
                        {"case": case_name, "command": command},
                        indent=2,
                    ),
                    flush=True,
                )
                completed = subprocess.run(command, check=False)
                summary_path = case_output / "run_summary.json"
                case: dict[str, Any] = {
                    "case": case_name,
                    "command": command,
                    "exit_code": int(completed.returncode),
                    "summary_path": str(summary_path),
                }
                if summary_path.exists():
                    summary = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                    case["result"] = _compact(summary)
                cases.append(case)
                if completed.returncode != 0:
                    matrix_path = output_root / "matrix_summary.json"
                    matrix_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "name": args.name,
                                "status": "failed",
                                "failed_case": case_name,
                                "cases": cases,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    raise SystemExit(completed.returncode)
    matrix = {
        "schema_version": 1,
        "name": args.name,
        "status": "completed",
        "execution": args.execution,
        "profiled": bool(args.profile),
        "cases": cases,
    }
    matrix_path = output_root / "matrix_summary.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "matrix_summary": str(matrix_path),
                "cases": [
                    {
                        "case": case["case"],
                        **case.get("result", {}),
                    }
                    for case in cases
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
