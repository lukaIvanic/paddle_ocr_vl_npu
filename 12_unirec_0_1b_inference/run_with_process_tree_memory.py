#!/usr/bin/env python3
"""Run a command and sample aggregate Linux PSS/RSS for its process tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


def children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def process_tree(root: int) -> list[int]:
    pending = [root]
    found = []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if Path(f"/proc/{pid}").exists():
            found.append(pid)
            pending.extend(children(pid))
    return found


def memory(pid: int) -> tuple[int, int] | None:
    path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        rows = path.read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    values = {}
    for row in rows:
        if row.startswith(("Pss:", "Rss:")):
            name, value, _unit = row.split()
            values[name[:-1]] = int(value) * 1024
    if "Pss" not in values or "Rss" not in values:
        return None
    return values["Pss"], values["Rss"]


def command_line(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def sample(root: int) -> dict[str, object]:
    rows = []
    for pid in process_tree(root):
        values = memory(pid)
        if values is None:
            continue
        pss, rss = values
        rows.append(
            {
                "pid": pid,
                "pss_bytes": pss,
                "rss_bytes": rss,
                "command": command_line(pid),
            }
        )
    return {
        "process_count": len(rows),
        "total_pss_bytes": sum(int(row["pss_bytes"]) for row in rows),
        "total_rss_bytes": sum(int(row["rss_bytes"]) for row in rows),
        "processes": rows,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.Popen(args.command)
    samples = 0
    peak: dict[str, object] = {
        "process_count": 0,
        "total_pss_bytes": 0,
        "total_rss_bytes": 0,
        "processes": [],
        "elapsed_s": 0.0,
    }
    while process.poll() is None:
        current = sample(process.pid)
        samples += 1
        if int(current["total_pss_bytes"]) > int(peak["total_pss_bytes"]):
            peak = {**current, "elapsed_s": time.time() - started}
        time.sleep(args.interval_ms / 1000.0)
    current = sample(process.pid)
    if int(current["total_pss_bytes"]) > int(peak["total_pss_bytes"]):
        peak = {**current, "elapsed_s": time.time() - started}
    report = {
        "schema": "linux_process_tree_memory_v1",
        "command": args.command,
        "root_pid": process.pid,
        "exit_code": process.returncode,
        "wall_s": time.time() - started,
        "interval_ms": args.interval_ms,
        "sample_count": samples,
        "peak": peak,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PROCESS_TREE_MEMORY " + json.dumps(report), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()

