#!/usr/bin/env python3
"""Capture a focused multi-metric profile of the optimized vision stack.

This runner intentionally fixes the Phase-14 configuration: one B1xS2048
production VisionPrefillStage replay, the zero-extended 4352-wide MLP,
FRACTAL_NZ Linear weights, runtime D72->D80 PromptFA padding, separate manual
RoPE, and a warm TorchAir cache.  Each AI Core metric family is collected in a
separate process because CANN exposes one PMU family per normal capture.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
LAB = HERE / "vision_matmul_lab.py"
ANALYZER = HERE / "analyze_vision_matmul_profile.py"
DEFAULT_MODEL = Path(
    "/workspace/models/PaddleOCR-VL-1.6"
)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_matmul_lab"
)
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_vision_matmul_profiles"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_matmul_profile_suite"
)
METRICS = ("pipe", "memory", "l2", "memory_access")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=["pipe", "memory", "l2"],
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
    )
    parser.add_argument("--allow-compile-if-missing", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=3)
    return parser.parse_args(argv)


def _command_text(command: list[str]) -> str:
    return shlex.join(command) + "\n"


def _disk_usage(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _run_logged(
    command: list[str],
    *,
    log_path: Path,
) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_s = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; "
            f"see {log_path}"
        )
    return wall_s


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = (args.output_root / args.name).expanduser().resolve()
    profile_root = (args.profile_root / args.name).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"suite output already exists and is non-empty: {output_root}"
        )
    if profile_root.exists() and any(profile_root.iterdir()):
        raise RuntimeError(
            f"raw profile output already exists and is non-empty: "
            f"{profile_root}"
        )
    output_root.mkdir(parents=True)
    profile_root.mkdir(parents=True)

    suite: dict[str, object] = {
        "schema_version": 1,
        "name": args.name,
        "purpose": (
            "focused multi-metric profile of the Phase-14 optimized "
            "B1xS2048 full 27-layer vision stack"
        ),
        "metrics": list(args.metrics),
        "raw_profile_disk_before": _disk_usage(profile_root),
        "lanes": {},
    }
    lane_profiles: list[tuple[str, Path]] = []
    contract_path: Path | None = None
    for metric in args.metrics:
        lane_output = output_root / metric
        lane_result = lane_output / "result"
        lane_profile = profile_root / metric
        lane_output.mkdir()
        command = [
            sys.executable,
            str(LAB),
            "--batch-size",
            "1",
            "--sequence-length",
            "2048",
            "--intermediate-size",
            "4352",
            "--weight-format",
            "fractal_nz",
            "--attention-head-padding",
            "runtime",
            "--rotary-implementation",
            "separate_manual",
            "--execution",
            "torchair",
            "--model",
            str(args.model.expanduser().resolve()),
            "--cache-dir",
            str(args.cache_dir.expanduser().resolve()),
            "--output-dir",
            str(lane_result),
            "--profile-dir",
            str(lane_profile),
            "--warmup",
            str(args.warmup),
            "--samples",
            str(args.samples),
            "--calls-per-sample",
            str(args.calls_per_sample),
            "--profile",
            "--profile-metric",
            metric,
            "--profile-warmup-steps",
            str(args.profile_warmup_steps),
            "--profile-steps",
            str(args.profile_steps),
            "--parser-topn",
            "200",
        ]
        if args.allow_compile_if_missing:
            command.append("--allow-compile-if-missing")
        (lane_output / "command.txt").write_text(
            _command_text(command),
            encoding="utf-8",
        )
        wall_s = _run_logged(
            command,
            log_path=lane_output / "run.log",
        )
        summary_path = lane_result / "run_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"lab did not write {summary_path}")
        lane_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if lane_summary.get("status") != "completed":
            raise RuntimeError(
                f"{metric} lane status is {lane_summary.get('status')!r}"
            )
        this_contract = lane_result / "profile_contract.json"
        if contract_path is None:
            contract_path = this_contract
        lane_profiles.append((metric, lane_profile))
        suite["lanes"][metric] = {
            "wall_s": wall_s,
            "command": command,
            "output_dir": str(lane_output),
            "result_dir": str(lane_result),
            "profile_dir": str(lane_profile),
            "run_summary": str(summary_path),
            "device_median_ms": lane_summary["measurements"][
                "device_event_per_call_ms"
            ]["median"],
            "physical_tokens_per_s": lane_summary["measurements"][
                "physical_tokens_per_s_device_median"
            ],
        }
        (output_root / "suite_summary.json").write_text(
            json.dumps(suite, indent=2) + "\n",
            encoding="utf-8",
        )

    if contract_path is None:
        raise AssertionError("no profile lanes were selected")
    combined_cache = profile_root / "combined_analysis"
    combined_output = output_root / "combined_profile"
    analyze_command = [
        sys.executable,
        str(ANALYZER),
        "--output-dir",
        str(combined_cache),
        "--contract",
        str(contract_path),
    ]
    for metric, path in lane_profiles:
        analyze_command.extend(["--lane", f"{metric}={path}"])
    (output_root / "analyze_command.txt").write_text(
        _command_text(analyze_command),
        encoding="utf-8",
    )
    suite["combined_analysis_wall_s"] = _run_logged(
        analyze_command,
        log_path=output_root / "analyze.log",
    )
    combined_output.mkdir()
    for name in (
        "profile_manifest.json",
        "profile_analysis.json",
        "vision_layer_summary.csv",
        "profile_report.md",
    ):
        shutil.copyfile(combined_cache / name, combined_output / name)
    suite["combined_analysis_cache"] = str(combined_cache)
    suite["combined_analysis"] = str(
        combined_output / "profile_analysis.json"
    )
    suite["combined_report"] = str(
        combined_output / "profile_report.md"
    )
    suite["raw_profile_disk_after"] = _disk_usage(profile_root)
    (output_root / "suite_summary.json").write_text(
        json.dumps(suite, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(suite, indent=2), flush=True)


if __name__ == "__main__":
    main()
