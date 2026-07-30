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
import datetime as dt
import json
import os
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
METRICS = (
    "pipe",
    "arithmetic",
    "memory",
    "memory_l0",
    "memory_ub",
    "resource_conflict",
    "l2",
    "memory_access",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=["pipe", "memory"],
        help=(
            "AI Core metric lanes. pipe+memory is the cross-product portable "
            "base. Add l2/memory_access only when the runtime capability "
            "query reports them as supported."
        ),
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
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=15.0,
        help="Seconds between flushed subprocess heartbeat messages.",
    )
    args = parser.parse_args(argv)
    if args.progress_interval_s <= 0:
        parser.error("--progress-interval-s must be positive")
    return args


def _command_text(command: list[str]) -> str:
    return shlex.join(command) + "\n"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _progress(message: str) -> None:
    print(f"[vision-profile {_utc_now()}] {message}", flush=True)


def _disk_usage(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _profile_metric_support(metrics: Sequence[str]) -> dict[str, object]:
    """Fail before model load when this product rejects a requested PMU lane."""
    import torch_npu.profiler as npu_prof

    metric_values = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "arithmetic": npu_prof.AiCMetrics.ArithmeticUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "memory_l0": npu_prof.AiCMetrics.MemoryL0,
        "memory_ub": npu_prof.AiCMetrics.MemoryUB,
        "resource_conflict": npu_prof.AiCMetrics.ResourceConflictRatio,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    query = getattr(npu_prof, "supported_ai_core_metrics", None)
    if query is None:
        return {
            "checked": False,
            "reason": "torch_npu does not expose supported_ai_core_metrics",
            "requested": [str(metric_values[name]) for name in metrics],
            "available": None,
        }
    try:
        available_raw = query()
    except Exception as exc:
        return {
            "checked": False,
            "reason": f"runtime capability query failed: {exc!r}",
            "requested": [str(metric_values[name]) for name in metrics],
            "available": None,
        }
    available = (
        list(available_raw)
        if isinstance(available_raw, (set, frozenset, list, tuple))
        else [available_raw]
    )
    unsupported = [
        name for name in metrics if metric_values[name] not in available
    ]
    result: dict[str, object] = {
        "checked": True,
        "reason": "runtime capability query",
        "requested": [str(metric_values[name]) for name in metrics],
        "available": sorted(str(item) for item in available),
        "unsupported": unsupported,
    }
    if unsupported:
        raise RuntimeError(
            "requested profiler metric lanes are unsupported by this "
            f"NPU/runtime: {', '.join(unsupported)}; supported metrics: "
            f"{', '.join(result['available']) or '<none>'}"
        )
    return result


def _run_logged(
    command: list[str],
    *,
    log_path: Path,
    label: str,
    progress_interval_s: float,
) -> float:
    started = time.perf_counter()
    _progress(f"{label}: starting; subprocess log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=progress_interval_s)
                    break
                except subprocess.TimeoutExpired:
                    elapsed_s = time.perf_counter() - started
                    log.flush()
                    log_bytes = log_path.stat().st_size
                    _progress(
                        f"{label}: running; elapsed_s={elapsed_s:.1f}; "
                        f"subprocess_log_bytes={log_bytes}"
                    )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    wall_s = time.perf_counter() - started
    if returncode:
        _progress(
            f"{label}: failed; exit={returncode}; elapsed_s={wall_s:.1f}; "
            f"see {log_path}"
        )
        raise RuntimeError(
            f"command failed with exit {returncode}; "
            f"see {log_path}"
        )
    _progress(f"{label}: completed; elapsed_s={wall_s:.1f}")
    return wall_s


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _progress(
        "preflight: checking requested profiler metrics "
        f"{', '.join(args.metrics)}"
    )
    metric_support = _profile_metric_support(args.metrics)
    _progress("preflight: requested profiler metrics are supported")
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
    _progress(
        f"suite initialized: name={args.name}; "
        f"output_root={output_root}; raw_root={profile_root}"
    )

    suite: dict[str, object] = {
        "schema_version": 1,
        "name": args.name,
        "purpose": (
            "focused multi-metric profile of the Phase-14 optimized "
            "B1xS2048 full 27-layer vision stack"
        ),
        "metrics": list(args.metrics),
        "metric_support": metric_support,
        "raw_profile_disk_before": _disk_usage(profile_root),
        "lanes": {},
    }
    summary_path = output_root / "suite_summary.json"
    lane_profiles: list[tuple[str, Path]] = []
    contract_path: Path | None = None
    for lane_index, metric in enumerate(args.metrics, start=1):
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
        lane_log = lane_output / "run.log"
        lane_label = f"lane {lane_index}/{len(args.metrics)} {metric}"
        wall_s = _run_logged(
            command,
            log_path=lane_log,
            label=lane_label,
            progress_interval_s=args.progress_interval_s,
        )
        lane_summary_path = lane_result / "run_summary.json"
        if not lane_summary_path.exists():
            raise RuntimeError(f"lab did not write {lane_summary_path}")
        lane_summary = json.loads(
            lane_summary_path.read_text(encoding="utf-8")
        )
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
            "subprocess_log": str(lane_log),
            "run_summary": str(lane_summary_path),
            "device_median_ms": lane_summary["measurements"][
                "device_event_per_call_ms"
            ]["median"],
            "physical_tokens_per_s": lane_summary["measurements"][
                "physical_tokens_per_s_device_median"
            ],
        }
        summary_path.write_text(
            json.dumps(suite, indent=2) + "\n",
            encoding="utf-8",
        )
        _progress(
            f"{lane_label}: checkpoint saved to {summary_path}; "
            f"device_median_ms="
            f"{suite['lanes'][metric]['device_median_ms']:.6f}; "
            f"physical_tokens_per_s="
            f"{suite['lanes'][metric]['physical_tokens_per_s']:.3f}"
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
    analyze_log = output_root / "analyze.log"
    suite["combined_analysis_wall_s"] = _run_logged(
        analyze_command,
        log_path=analyze_log,
        label="combined analysis",
        progress_interval_s=args.progress_interval_s,
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
    summary_path.write_text(
        json.dumps(suite, indent=2) + "\n",
        encoding="utf-8",
    )
    _progress(
        f"suite completed: summary={summary_path}; "
        f"report={suite['combined_report']}"
    )
    print(json.dumps(suite, indent=2), flush=True)


if __name__ == "__main__":
    main()
