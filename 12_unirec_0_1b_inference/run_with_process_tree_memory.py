#!/usr/bin/env python3
"""Run a command and sample its Linux process-tree memory and NPU HBM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument(
        "--npu-id",
        type=int,
        help="physical NPU ID to sample with npu-smi",
    )
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--npu-interval-ms", type=float, default=2000.0)
    parser.add_argument("--npu-query-timeout-s", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.npu_interval_ms <= 0:
        parser.error("--npu-interval-ms must be positive")
    if args.npu_query_timeout_s <= 0:
        parser.error("--npu-query-timeout-s must be positive")
    return args


_HBM_USAGE_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")


def parse_npu_hbm_usage(output: str, physical_npu: int) -> tuple[int, int]:
    """Return used and total HBM MiB for one physical NPU."""

    awaiting_usage_row = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if awaiting_usage_row:
            if cells:
                match = _HBM_USAGE_RE.search(cells[-1])
                if match is not None:
                    return int(match.group(1)), int(match.group(2))
            awaiting_usage_row = False
        if not cells:
            continue
        device_fields = cells[0].split()
        if (
            len(device_fields) >= 2
            and device_fields[0].isdigit()
            and int(device_fields[0]) == physical_npu
            and not device_fields[1].isdigit()
        ):
            awaiting_usage_row = True
    raise ValueError(f"HBM usage for physical NPU {physical_npu} not found")


def query_npu_hbm(
    npu_smi: str,
    physical_npu: int,
    timeout_s: float,
) -> dict[str, object]:
    completed = subprocess.run(
        [npu_smi, "info"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{npu_smi} info exited {completed.returncode}: {completed.stdout[-1000:]}"
        )
    used_mb, total_mb = parse_npu_hbm_usage(completed.stdout, physical_npu)
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "raw_npu_smi": completed.stdout,
    }


class NpuHbmSampler:
    """Poll exact physical-device HBM without joining the measured process tree."""

    def __init__(
        self,
        *,
        npu_smi: str,
        physical_npu: int,
        interval_ms: float,
        query_timeout_s: float,
    ) -> None:
        self.npu_smi = npu_smi
        self.physical_npu = physical_npu
        self.interval_s = interval_ms / 1000.0
        self.query_timeout_s = query_timeout_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = 0.0
        self.baseline: dict[str, object] | None = None
        self.samples: list[dict[str, object]] = []
        self.peak: dict[str, object] | None = None
        self.errors: list[str] = []

    def collect_baseline(self) -> None:
        self.baseline = query_npu_hbm(
            self.npu_smi,
            self.physical_npu,
            self.query_timeout_s,
        )

    def start(self, started: float) -> None:
        self.started = started
        self.thread = threading.Thread(
            target=self._run,
            name="npu-hbm-sampler",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            query_started = time.monotonic()
            try:
                result = query_npu_hbm(
                    self.npu_smi,
                    self.physical_npu,
                    self.query_timeout_s,
                )
                row = {
                    "elapsed_s": time.time() - self.started,
                    "used_mb": result["used_mb"],
                    "total_mb": result["total_mb"],
                }
                self.samples.append(row)
                if self.peak is None or int(row["used_mb"]) > int(self.peak["used_mb"]):
                    self.peak = {**row, "raw_npu_smi": result["raw_npu_smi"]}
            except Exception as exc:  # keep measuring the target if npu-smi hiccups
                self.errors.append(f"{type(exc).__name__}: {exc}")
            query_elapsed = time.monotonic() - query_started
            self.stop_event.wait(max(0.0, self.interval_s - query_elapsed))

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.query_timeout_s + 1.0)

    def report(self) -> dict[str, object]:
        baseline_used = (
            int(self.baseline["used_mb"]) if self.baseline is not None else None
        )
        peak_used = int(self.peak["used_mb"]) if self.peak is not None else None
        peak_increase = (
            max(0, peak_used - baseline_used)
            if peak_used is not None and baseline_used is not None
            else None
        )
        return {
            "physical_npu": self.physical_npu,
            "query_command": [self.npu_smi, "info"],
            "interval_ms": self.interval_s * 1000.0,
            "query_timeout_s": self.query_timeout_s,
            "baseline": self.baseline,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "peak": self.peak,
            "peak_increase_from_baseline_mb": peak_increase,
            "errors": self.errors,
        }


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
    hbm_sampler = None
    if args.npu_id is not None:
        hbm_sampler = NpuHbmSampler(
            npu_smi=args.npu_smi,
            physical_npu=args.npu_id,
            interval_ms=args.npu_interval_ms,
            query_timeout_s=args.npu_query_timeout_s,
        )
        hbm_sampler.collect_baseline()
    started = time.time()
    process = subprocess.Popen(args.command)
    if hbm_sampler is not None:
        hbm_sampler.start(started)
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
    if hbm_sampler is not None:
        hbm_sampler.stop()
    report = {
        "schema": "linux_process_tree_and_npu_memory_v2",
        "command": args.command,
        "root_pid": process.pid,
        "exit_code": process.returncode,
        "wall_s": time.time() - started,
        "interval_ms": args.interval_ms,
        "sample_count": samples,
        "peak": peak,
        "npu_hbm": hbm_sampler.report() if hbm_sampler is not None else None,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PROCESS_TREE_MEMORY " + json.dumps(report), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
