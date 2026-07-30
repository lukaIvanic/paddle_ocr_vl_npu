#!/usr/bin/env python3
"""Deep-profile the observable production-equivalent square Linear with msprof.

The full compiled 27-layer graph is measured separately by
``run_vision_matmul_profile_suite.py``. This second-tier runner exposes the
square Q/K/V/output projection as one directly observable MatMulV2 with the
same FP16 shape, ND activation, FRACTAL_NZ weight, bias, and Block Dim as the
optimized graph. FC1/FC2 remain in the full-graph profiler because eager
dispatch differs and ``msprof op`` cannot see inside the cached TorchAir graph.
The analyzer must still match this target back to the full-graph reference.
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
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
TARGET_SCRIPT = HERE / "vision_msprof_linear_target.py"
ROLES = ("square",)
# PipeUtilization and Memory are the portable base tier. Huawei documents
# Occupancy and MemoryDetail only for Atlas A2/A3 products, so those deeper
# families are optional extensions rather than assumptions of this runner.
METRICS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "ResourceConflictRatio",
    "Occupancy",
    "MemoryDetail",
)
METRIC_SUPPORT = {
    "PipeUtilization": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "ArithmeticUtilization": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "Memory": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "MemoryL0": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "MemoryUB": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "ResourceConflictRatio": {
        "tier": "portable_base",
        "documented_products": "Atlas inference, training, A2, and A3 products",
    },
    "Occupancy": {
        "tier": "atlas_a2_a3_extension",
        "documented_products": "Atlas A2 and Atlas A3 training/inference products",
    },
    "MemoryDetail": {
        "tier": "atlas_a2_a3_extension",
        "documented_products": "Atlas A2 and Atlas A3 training/inference products",
    },
}
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_msprof_op"
)
DEFAULT_RAW_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_msprof_op"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--metric", choices=METRICS, default="PipeUtilization")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=ROLES,
        default=list(ROLES),
        help="Production-equivalent direct Linear shape to capture.",
    )
    parser.add_argument(
        "--msprof-warm-up",
        type=int,
        default=5,
        help="Kernel-replay warm-up count passed to msprof.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name must contain only A-Z, a-z, 0-9, _, ., or -")
    if not 0 <= args.msprof_warm_up <= 500:
        parser.error("--msprof-warm-up must be in [0, 500]")
    args.roles = list(dict.fromkeys(args.roles))
    return args


def _empty_or_create(
    path: Path,
    label: str,
    *,
    private: bool = False,
) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"{label} already exists and is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        # Do not depend on the server's umask. msprof requires its output path
        # to be owned by the caller and not writable by group/other.
        path.chmod(0o700)


def _safe_msprof_output(path: Path) -> None:
    for candidate in (path.parent, path):
        mode = stat.S_IMODE(candidate.stat().st_mode)
        if mode & 0o022:
            raise RuntimeError(
                "msprof output path is group/other-writable "
                f"(mode {mode:o}): {candidate}"
            )
        if candidate.stat().st_uid != os.geteuid():
            raise RuntimeError(
                "msprof output path is not owned by the current user: "
                f"{candidate}"
            )


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


def _artifact_summary(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    suffixes = Counter(path.suffix.lower() or "<none>" for path in files)
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "suffix_counts": dict(sorted(suffixes.items())),
        "has_visualize_data": any(
            path.name == "visualize_data.bin" for path in files
        ),
        "csv_names": sorted(
            {path.name for path in files if path.suffix.lower() == ".csv"}
        ),
    }


def _has_csv(summary: dict[str, Any], stem: str) -> bool:
    prefix = stem.casefold()
    return any(
        name.casefold().startswith(prefix)
        for name in summary["csv_names"]
    )


def _artifact_contract_errors(
    metric: str,
    summary: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not _has_csv(summary, "OpBasicInfo"):
        errors.append("msprof output has no OpBasicInfo CSV")
    if metric in {
        "PipeUtilization",
        "ArithmeticUtilization",
        "Memory",
        "MemoryL0",
        "MemoryUB",
        "ResourceConflictRatio",
    } and not _has_csv(summary, metric):
        errors.append(f"{metric} capture has no {metric} CSV")
    elif metric == "Occupancy" and not summary["has_visualize_data"]:
        errors.append("Occupancy capture has no visualize_data.bin")
    elif metric == "MemoryDetail":
        for stem in ("PipeUtilization", "Memory", "L2Cache"):
            if not _has_csv(summary, stem):
                errors.append(
                    f"MemoryDetail capture has no {stem} CSV"
                )
        if not summary["has_visualize_data"]:
            errors.append("MemoryDetail capture has no visualize_data.bin")
    return errors


def _environment() -> dict[str, str | None]:
    names = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
    )
    return {name: os.environ.get(name) for name in names}


def _run_role(
    *,
    role: str,
    args: argparse.Namespace,
    python: str,
    msprof: str,
    evidence_dir: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    role_evidence = evidence_dir / role
    role_raw = raw_dir / role
    _empty_or_create(role_evidence, f"{role} evidence directory")
    _empty_or_create(
        role_raw,
        f"{role} raw profiler directory",
        private=True,
    )
    _safe_msprof_output(role_raw)
    application_dir = role_evidence / "application"
    command = [
        msprof,
        "op",
        f"--output={role_raw}",
        f"--aic-metrics={args.metric}",
        "--replay-mode=kernel",
        "--kernel-name=MatMulV2",
        "--launch-count=1",
        f"--warm-up={args.msprof_warm_up}",
        python,
        str(TARGET_SCRIPT),
        "--role",
        role,
        "--output-dir",
        str(application_dir),
    ]
    command_record = {
        "argv": command,
        "shell_escaped": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "selection": {
            "role": role,
            "kernel_name_prefix": "MatMulV2",
            "launch_count": 1,
            "replay_mode": "kernel",
            "metric": args.metric,
            "metric_support": METRIC_SUPPORT[args.metric],
            "mstx": False,
        },
    }
    _write_json(role_evidence / "command.json", command_record)
    started = dt.datetime.now(dt.timezone.utc)
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
    (role_evidence / "msprof.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (role_evidence / "msprof.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    artifact_summary = _artifact_summary(role_raw)
    target_summary_path = application_dir / "target_summary.json"
    target_summary: dict[str, Any] | None = None
    if target_summary_path.is_file():
        target_summary = json.loads(
            target_summary_path.read_text(encoding="utf-8")
        )
    errors = []
    if completed.returncode != 0:
        errors.append(f"msprof exited with status {completed.returncode}")
    if artifact_summary["file_count"] == 0:
        errors.append(f"msprof produced no raw artifacts under {role_raw}")
    else:
        errors.extend(_artifact_contract_errors(args.metric, artifact_summary))
    if target_summary is None:
        errors.append("target did not produce target_summary.json")
    elif target_summary.get("status") != "completed":
        errors.append(
            "target status is not completed: "
            f"{target_summary.get('status')!r}"
        )
    result = {
        "schema_version": 1,
        "status": "failed" if errors else "captured_unvalidated",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_s": wall_s,
        "returncode": completed.returncode,
        "command": command_record,
        "paths": {
            "evidence": str(role_evidence),
            "raw": str(role_raw),
            "target_summary": (
                str(target_summary_path)
                if target_summary_path.is_file()
                else None
            ),
        },
        "raw_artifacts": artifact_summary,
        "target": target_summary,
        "errors": errors,
        "validation_state": (
            "capture_only; dispatch/shape/format/block/duration matching "
            "must be performed by analyze_vision_msprof_op.py"
        ),
    }
    _write_json(role_evidence / "capture_manifest.json", result)
    if errors:
        raise RuntimeError(f"{role}: {'; '.join(errors)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence_dir = (args.evidence_root / args.run_name).expanduser().resolve()
    raw_dir = (args.raw_root / args.run_name).expanduser().resolve()
    if evidence_dir == raw_dir:
        raise RuntimeError("evidence and raw output directories must differ")
    _empty_or_create(evidence_dir, "suite evidence directory")
    _empty_or_create(
        raw_dir,
        "suite raw profiler directory",
        private=True,
    )
    _safe_msprof_output(raw_dir)
    python = _resolve_executable(args.python, "Python interpreter")
    msprof = _resolve_executable(args.msprof, "msprof")
    suite = {
        "schema_version": 1,
        "status": "running",
        "purpose": (
            "portable deep msprof-op capture of the unique production-shaped "
            "PaddleOCR-VL vision MatMulV2 kernels"
        ),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
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
        "environment": _environment(),
        "metric": args.metric,
        "metric_support": METRIC_SUPPORT[args.metric],
        "roles": args.roles,
        "paths": {
            "evidence": str(evidence_dir),
            "raw": str(raw_dir),
        },
        "captures": {},
        "errors": [],
    }
    _write_json(evidence_dir / "suite_manifest.json", suite)
    try:
        for role in args.roles:
            suite["captures"][role] = _run_role(
                role=role,
                args=args,
                python=python,
                msprof=msprof,
                evidence_dir=evidence_dir,
                raw_dir=raw_dir,
            )
            _write_json(evidence_dir / "suite_manifest.json", suite)
    except BaseException as exc:
        suite["status"] = "failed"
        suite["errors"].append(repr(exc))
        suite["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(evidence_dir / "suite_manifest.json", suite)
        raise
    suite["status"] = "captured_unvalidated"
    suite["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    suite["next_step"] = (
        "run analyze_vision_msprof_op.py with the validated full-graph "
        "reference before interpreting per-core or memory metrics"
    )
    _write_json(evidence_dir / "suite_manifest.json", suite)
    print(
        json.dumps(
            {
                "status": suite["status"],
                "metric": args.metric,
                "roles": args.roles,
                "evidence_dir": str(evidence_dir),
                "raw_dir": str(raw_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
