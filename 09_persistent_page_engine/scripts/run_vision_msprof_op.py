#!/usr/bin/env python3
"""Run one targeted ``msprof op`` capture of the Phase-14 vision lane.

The application performs its ordinary unprofiled timing first, then emits one
warm compiled full-stack replay inside the MSTX range
``vision_msprof_target``.  msprof filters that range to MatMulV2 and replays
the selected kernel(s) for detailed Occupancy or MemoryDetail analysis.
Small provenance and logs live under ``tmp/``; heavyweight raw profiler
artifacts live under ``.runtime_cache/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
LAB_SCRIPT = HERE / "vision_matmul_lab.py"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_matmul_lab"
)
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_msprof_op"
)
DEFAULT_RAW_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_msprof_op"
)
TARGET_NAME = "vision_msprof_target"
METRICS = ("Occupancy", "MemoryDetail")
FIXED_LANE = {
    "batch_size": 1,
    "sequence_length": 2048,
    "intermediate_size": 4352,
    "weight_format": "fractal_nz",
    "attention_head_padding": "runtime",
    "rotary_implementation": "separate_manual",
    "execution": "torchair",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--metric", choices=METRICS, default="Occupancy")
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Matched MatMulV2 kernels to collect (msprof permits 1..100).",
    )
    parser.add_argument(
        "--msprof-warm-up",
        type=int,
        default=5,
        help="Kernel replay warm-up count passed to msprof.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--allow-compile-if-missing",
        action="store_true",
        help="Permit TorchAir to create the exact fixed-lane graph cache.",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name must contain only A-Z, a-z, 0-9, _, ., or -")
    if not 1 <= args.launch_count <= 100:
        parser.error("--launch-count must be in [1, 100]")
    if not 0 <= args.msprof_warm_up <= 500:
        parser.error("--msprof-warm-up must be in [0, 500]")
    return args


def _empty_or_create(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"{label} already exists and is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _resolve_executable(value: str | Path, label: str) -> str:
    text = str(value)
    if "/" in text:
        path = Path(text).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"{label} is not executable: {path}")
        return str(path)
    resolved = shutil.which(text)
    if resolved is None:
        raise RuntimeError(f"{label} was not found on PATH: {text}")
    return resolved


def _relevant_environment() -> dict[str, str | None]:
    names = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "LD_LIBRARY_PATH",
    )
    return {name: os.environ.get(name) for name in names}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence_dir = (args.evidence_root / args.run_name).expanduser().resolve()
    raw_dir = (args.raw_root / args.run_name).expanduser().resolve()
    if evidence_dir == raw_dir:
        raise RuntimeError("evidence and raw output directories must differ")
    _empty_or_create(evidence_dir, "evidence directory")
    _empty_or_create(raw_dir, "raw profiler directory")
    # msprof requires the output directory and parent to be owned by the
    # current user and not group/other-writable.
    for path in (raw_dir.parent, raw_dir):
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o022:
            raise RuntimeError(
                "msprof output path is group/other-writable "
                f"(mode {mode:o}): {path}"
            )

    python = _resolve_executable(args.python, "Python interpreter")
    msprof = _resolve_executable(args.msprof, "msprof")
    application_dir = evidence_dir / "application"
    command = [
        msprof,
        "op",
        f"--output={raw_dir}",
        f"--aic-metrics={args.metric}",
        "--replay-mode=kernel",
        "--mstx=on",
        f"--mstx-include={TARGET_NAME}",
        "--kernel-name=MatMulV2",
        f"--launch-count={args.launch_count}",
        f"--warm-up={args.msprof_warm_up}",
        python,
        str(LAB_SCRIPT),
        "--batch-size",
        str(FIXED_LANE["batch_size"]),
        "--sequence-length",
        str(FIXED_LANE["sequence_length"]),
        "--intermediate-size",
        str(FIXED_LANE["intermediate_size"]),
        "--weight-format",
        FIXED_LANE["weight_format"],
        "--attention-head-padding",
        FIXED_LANE["attention_head_padding"],
        "--rotary-implementation",
        FIXED_LANE["rotary_implementation"],
        "--execution",
        FIXED_LANE["execution"],
        "--cache-dir",
        str(args.cache_dir.expanduser().resolve()),
        "--output-dir",
        str(application_dir),
        "--emit-msprof-target",
    ]
    if args.model is not None:
        command.extend(["--model", str(args.model.expanduser().resolve())])
    if args.allow_compile_if_missing:
        command.append("--allow-compile-if-missing")

    command_record = {
        "argv": command,
        "shell_escaped": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "fixed_lane": FIXED_LANE,
        "target": {
            "mstx_range": TARGET_NAME,
            "kernel_name_prefix": "MatMulV2",
            "replay_mode": "kernel",
            "launch_count": args.launch_count,
            "metric": args.metric,
        },
    }
    _write_json(evidence_dir / "command.json", command_record)
    started = dt.datetime.now(dt.timezone.utc)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": (
            "targeted msprof-op replay of MatMulV2 inside one warm compiled "
            "27-layer PaddleOCR-VL vision replay"
        ),
        "started_at_utc": started.isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": python,
            "msprof": msprof,
        },
        "environment": _relevant_environment(),
        "fixed_lane": FIXED_LANE,
        "command": command_record,
        "paths": {
            "evidence_dir": str(evidence_dir),
            "raw_dir": str(raw_dir),
            "application_output_dir": str(application_dir),
        },
    }
    _write_json(evidence_dir / "manifest.json", manifest)

    wall_started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall_s = time.perf_counter() - wall_started
    (evidence_dir / "msprof.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (evidence_dir / "msprof.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    raw_files = _file_inventory(raw_dir)
    summary_path = application_dir / "run_summary.json"
    application_summary: dict[str, Any] | None = None
    if summary_path.is_file():
        application_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wall_s": wall_s,
            "returncode": completed.returncode,
            "raw_artifacts": raw_files,
            "application_summary": str(summary_path)
            if summary_path.is_file() else None,
            "application_status": (
                application_summary.get("status")
                if application_summary is not None else None
            ),
        }
    )
    errors = []
    if completed.returncode != 0:
        errors.append(f"msprof exited with status {completed.returncode}")
    if not raw_files:
        errors.append(f"msprof produced no raw artifacts under {raw_dir}")
    if application_summary is None:
        errors.append("vision lab did not produce run_summary.json")
    elif application_summary.get("status") != "completed":
        errors.append(
            "vision lab status is not completed: "
            f"{application_summary.get('status')!r}"
        )
    target = (
        application_summary.get("msprof_target")
        if application_summary is not None else None
    )
    if not isinstance(target, dict) or target.get("name") != TARGET_NAME:
        errors.append("vision lab did not confirm the requested MSTX target")
    manifest["status"] = "failed" if errors else "completed"
    manifest["errors"] = errors
    _write_json(evidence_dir / "manifest.json", manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    print(
        json.dumps(
            {
                "status": "completed",
                "metric": args.metric,
                "evidence_dir": str(evidence_dir),
                "raw_dir": str(raw_dir),
                "raw_file_count": len(raw_files),
                "wall_s": wall_s,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
