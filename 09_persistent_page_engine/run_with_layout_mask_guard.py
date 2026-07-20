#!/usr/bin/env python3
"""Run another Python script with the PP-DocLayout empty-mask guard installed."""

from __future__ import annotations

import argparse
import atexit
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from layout_mask_guard import install_layout_mask_guard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-report", type=Path, required=True)
    parser.add_argument("runner", type=Path)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def run_python_script(runner: Path, runner_args: Sequence[str]) -> dict[str, object]:
    """Execute ``runner`` with the import behavior of ``python runner.py``."""

    previous_argv = sys.argv
    sys.argv = [str(runner), *runner_args]
    sys.path.insert(0, str(runner.parent))
    try:
        return runpy.run_path(str(runner), run_name="__main__")
    finally:
        sys.path.pop(0)
        sys.argv = previous_argv


def main() -> None:
    args = parse_args()
    runner = args.runner.expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"runner not found: {runner}")
    report = args.guard_report.expanduser().resolve()
    state = install_layout_mask_guard()
    atexit.register(state.write_snapshot, report)
    run_python_script(runner, args.runner_args)


if __name__ == "__main__":
    main()
