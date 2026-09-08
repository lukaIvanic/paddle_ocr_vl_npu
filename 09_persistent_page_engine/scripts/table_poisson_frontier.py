#!/usr/bin/env python3
"""Finite, sequential Blue-Zone open-loop screening; run on the bare-metal host.

Uses the existing crop server and client, without changing inference. One server
per B, one client per rate, identical global-shuffle seed/count at every point.
The completed-throughput/P95 frontier is provisional until 10k confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time


MATRIX = {
    1: [1, 1.5],
    2: [1, 2, 2.5, 3],
    3: [2, 3, 3.5],
    4: [2, 3, 4, 4.5],
    5: [3, 4, 5],
    6: [4, 5, 5.5],
    7: [5, 6],
    8: [3, 4, 5, 6, 6.5],
    16: [4, 6, 7, 8],
}
CONTAINER = "research_vllm_ascend_021_external_workspace"
CONTAINER_REPO = "/workspace/repos/paddle_ocr_vl_npu"
PYTHON = "/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python"
SCRIPTS = "09_persistent_page_engine/scripts/"
API = "http://127.0.0.1:8767"
VLLM_CONTAINER = "research_vllm_ascend_023_paddleocr"


def frontier(rows, *, offered=False):
    axis = "target_qps" if offered else "completed_qps"
    valid = [r for r in rows if r["valid"]]
    return [r for r in valid if not any(
        s[axis] >= r[axis] and s["p95_s"] <= r["p95_s"]
        and (s[axis] > r[axis] or s["p95_s"] < r["p95_s"])
        for s in valid)]


def output(cmd, timeout=30):
    return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL)


def docker(*cmd):
    return ["docker", "exec", "-w", CONTAINER_REPO, CONTAINER, *cmd]


def fingerprint(repo):
    paths = list((repo / "09_persistent_page_engine/paddleocr_vl").rglob("*.py"))
    paths += list((repo / "09_persistent_page_engine/presets/table_compact_vocab").glob("*.json"))
    paths += [repo / SCRIPTS / name for name in (
        "serve_crop_ocr_api.py", "table_request_load_simulator.py", "table_closed_loop_api_client.py",
        "serve_vllm_table_reference.sh", "table_poisson_frontier.py")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Sweep:
    def __init__(self, args):
        self.args = args
        self.vllm = args.api_kind == "vllm"
        self.matrix = {n: list(range(1, 9)) for n in (4096, 8192, 16384)} if self.vllm else MATRIX
        self.api = "http://127.0.0.1:18081" if self.vllm else API
        self.repo = Path(__file__).resolve().parents[2]
        self.root = self.repo / args.output_dir
        self.root.mkdir(parents=True, exist_ok=False)
        self.initial_fingerprint = fingerprint(self.repo)
        self.marker = None
        self.server = None
        self.server_pid = None
        self.server_start = None
        self.rows = []
        self.ownership = (self.root / "ownership.jsonl").open("w")
        self.write("plan.json", dict(matrix=self.matrix, api_kind=args.api_kind, count=args.count, seed=1,
            npu=args.npu, source_fingerprint=self.initial_fingerprint,
            git_commit=output(["git", "-C", str(self.repo), "rev-parse", "HEAD"]).strip(),
            note="Open loop, no client concurrency cap. All work and queueing count in latency."))

    def write(self, name, value):
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)

    def log(self, message):
        print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)

    def owned(self):
        # Match only this invocation's unique service-output argument. Never
        # consider all processes in our container to be ours.
        if not self.marker:
            return set(), None
        container = VLLM_CONTAINER if self.vllm else CONTAINER
        lines = output(["docker", "top", container, "-eo", "pid,ppid,args"]).splitlines()[1:]
        entries = [line.split(None, 2) for line in lines if line.strip()]
        candidates = {int(pid) for pid, ppid, cmd in entries
                      if ("vllm serve" if self.vllm else "serve_crop_ocr_api.py") in cmd and self.marker in cmd}
        parent_of = {int(pid): int(ppid) for pid, ppid, _ in entries}
        def has_candidate_ancestor(pid):
            seen = set()
            pid = parent_of.get(pid)
            while pid is not None and pid not in seen:
                if pid in candidates:
                    return True
                seen.add(pid)
                pid = parent_of.get(pid)
            return False
        # bash setup subshells inherit the command line before exec. They are
        # descendants of ONE launch, not independent owners of the device.
        parents = [pid for pid in candidates if not has_candidate_ancestor(pid)]
        if len(parents) > 1:
            raise RuntimeError("Ambiguous owned server PID")
        if not parents:
            return set(), None
        result = set(parents)
        while True:
            expanded = result | {int(pid) for pid, ppid, _ in entries if int(ppid) in result}
            if expanded == result:
                return result, parents[0]
            result = expanded

    def check_device(self):
        raw = output(["npu-smi", "info", "-t", "proc-mem", "-i", str(self.args.npu)])
        pids = set(map(int, re.findall(r"Process id:(\d+)", raw)))
        if not pids and "No process in device" not in raw:
            raise RuntimeError("Cannot establish NPU ownership from npu-smi")
        owned, parent = self.owned()
        record = dict(epoch=time.time(), device=self.args.npu, pids=sorted(pids),
                      owned=sorted(owned), server_parent=parent, raw=raw)
        self.ownership.write(json.dumps(record) + "\n")
        self.ownership.flush()
        if pids - owned:
            raise RuntimeError(f"NPU contamination or occupied device: {sorted(pids - owned)}")
        if parent is not None and self.server_pid is None:
            self.server_pid = parent
            self.server_start = self.start_identity(parent)

    @staticmethod
    def start_identity(pid):
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]

    def stop(self):
        if self.server is None:
            return
        _, parent = self.owned()
        if parent is not None:
            # PID reuse / command mismatch must never signal a different job.
            if self.server_pid is None:
                self.server_pid, self.server_start = parent, self.start_identity(parent)
            if parent != self.server_pid or self.start_identity(parent) != self.server_start:
                raise RuntimeError("Owned server identity changed; refusing signal")
            self.log(f"STOP owned server pid={parent}")
            os.kill(parent, signal.SIGTERM)
        self.server.wait(timeout=90)
        self.server = None
        self.marker = None
        self.server_pid = self.server_start = None
        time.sleep(3)
        self.check_device()

    def run_client(self, cmd, folder, timeout):
        folder.mkdir(parents=True)
        (folder / "command.txt").write_text(shlex.join(cmd) + "\n")
        with (folder / "client.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            start = last_progress = time.monotonic()
            try:
                while process.poll() is None:
                    self.check_device()
                    if time.monotonic() - start > timeout:
                        raise RuntimeError(f"Client deadline exceeded: {folder}")
                    if time.monotonic() - last_progress >= 45:
                        lines = (folder / "client.log").read_text().splitlines()
                        self.log(f"PROGRESS {folder.name}: " + (lines[-1] if lines else "starting"))
                        last_progress = time.monotonic()
                    time.sleep(3)
            finally:
                if process.poll() is None:
                    # Stopping the docker CLI alone leaves its container child
                    # alive. Match only this client's unique output argument.
                    marker = cmd[cmd.index("--output-dir") + 1]
                    script = next(x for x in cmd if x.endswith("_client.py") or x.endswith("_simulator.py"))
                    entries = output(["docker", "top", CONTAINER, "-eo", "pid,ppid,args"]).splitlines()[1:]
                    for entry in entries:
                        pid, _, command = entry.split(None, 2)
                        if script in command and marker in command:
                            pid = int(pid)
                            identity = self.start_identity(pid)
                            current = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                            if script in current and marker in current and self.start_identity(pid) == identity:
                                self.log(f"STOP owned client pid={pid}")
                                os.kill(pid, signal.SIGTERM)
                    process.wait(timeout=30)
            (folder / "exit_code.txt").write_text(str(process.returncode) + "\n")
            self.check_device()
            if process.returncode:
                raise RuntimeError(f"Client failed ({process.returncode}): {folder}")

    def start(self, batch):
        if self.vllm:
            return self.start_vllm(batch)
        if fingerprint(self.repo) != self.initial_fingerprint:
            raise RuntimeError("Relevant source changed during sweep; refusing mixed configuration")
        self.check_device()  # Must be free BEFORE starting any model process.
        folder = self.root / f"b{batch}"
        folder.mkdir()
        relative = folder.relative_to(self.repo)
        self.marker = str(relative / "service.json")
        server = [PYTHON, "-u", SCRIPTS + "serve_crop_ocr_api.py", "--host", "127.0.0.1",
            "--port", "8767", "--request-timeout-s", "3600", "--queue-capacity", "64",
            "--decode-batch-size", str(batch), "--min-pixels", "28224", "--max-pixels", "802816",
            "--service-summary-output", self.marker]
        script = f"source npu-setup >/dev/null; export ASCEND_RT_VISIBLE_DEVICES={self.args.npu}; exec " + shlex.join(server)
        cmd = docker("bash", "-c", script)
        (folder / "server_command.txt").write_text(shlex.join(cmd) + "\n")
        self.log(f"START B{batch}; cached production setup, then one full request warmup")
        with (folder / "server.log").open("w") as log:
            self.server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            self.check_device()
            if self.server.poll() is not None:
                raise RuntimeError(f"Server exited: {folder / 'server.log'}")
            try:
                ready = json.loads(output(docker("curl", "-fsS", "--max-time", "2", API + "/ready"), 5))
                if ready.get("ready"):
                    self.write(str(relative.relative_to(self.args.output_dir) / "ready.json"), ready)
                    break
            except (subprocess.SubprocessError, json.JSONDecodeError):
                pass
            time.sleep(3)
        else:
            raise RuntimeError("Server setup exceeded 20 minutes; no blind retries")
        self.run_client(docker(PYTHON, "-u", SCRIPTS + "table_closed_loop_api_client.py",
            "--api-url", API + "/v1/ocr", "--set", "warm", "--count", "1", "--max-in-flight", "1",
            "--output-dir", str(relative / "warm/results")), folder / "warm", 1200)
        return folder, relative

    def start_vllm(self, budget):
        if fingerprint(self.repo) != self.initial_fingerprint:
            raise RuntimeError("Relevant source changed during sweep")
        self.check_device()
        existing = output(["docker", "top", VLLM_CONTAINER, "-eo", "pid,args"])
        if "vllm serve" in existing:
            raise RuntimeError("An existing vLLM server must not be adopted or stopped")
        folder = self.root / f"budget{budget}"
        folder.mkdir()
        relative = folder.relative_to(self.repo)
        self.marker = "--port 18081"
        cmd = ["docker", "exec", "-e", f"ASCEND_RT_VISIBLE_DEVICES={self.args.npu}",
               "-e", "TABLE_VLLM_MAX_SEQS=16", "-e", f"TABLE_VLLM_TOKEN_BUDGET={budget}",
               VLLM_CONTAINER, "bash", "/workspace/serve_vllm_table_reference.sh"]
        (folder / "server_command.txt").write_text(shlex.join(cmd) + "\n")
        self.log(f"START vLLM budget={budget}, max_seqs=16, per-request context=4096")
        with (folder / "server.log").open("w") as log:
            self.server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        progress = time.monotonic()
        while time.monotonic() < deadline:
            self.check_device()
            if self.server.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            try:
                output(["curl", "-fsS", "--max-time", "2", self.api + "/health"], 5)
                break
            except subprocess.SubprocessError:
                pass
            if time.monotonic() - progress > 45:
                lines = (folder / "server.log").read_text().splitlines()
                self.log("SETUP " + (lines[-1] if lines else "starting"))
                progress = time.monotonic()
            time.sleep(3)
        else:
            raise RuntimeError("vLLM startup exceeded 20 minutes")
        for warm in (1, 2):
            self.run_client(docker(PYTHON, "-u", SCRIPTS + "table_closed_loop_api_client.py",
                "--api-kind", "vllm", "--api-url", self.api + "/v1/chat/completions",
                "--set", "warm", "--count", "1", "--max-in-flight", "1",
                "--output-dir", str(relative / f"warm{warm}/results")), folder / f"warm{warm}", 1200)
        return folder, relative

    def publish(self):
        self.write("results.json", self.rows)
        with (self.root / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.write("frontier_candidates.json", dict(
            note="Screening only. Queue growth is recorded, not inferred away; 10k confirmation required.",
            completed_throughput_p95=frontier(self.rows), offered_qps_p95=frontier(self.rows, offered=True)))

    def run(self):
        try:
            for batch, rates in self.matrix.items():
                folder, relative = self.start(batch)
                for qps in rates:
                    name = f"qps{qps:g}"
                    run = folder / name
                    target = relative / name / "measured"
                    self.log(f"MEASURE {'budget' if self.vllm else 'B'}{batch} target={qps} QPS count={self.args.count}")
                    extra = []
                    if self.vllm:
                        extra = ["--api-kind", "vllm", "--schedule-jsonl", str(self.args.schedule_jsonl),
                                 "--schedule-source-qps", "1"]
                        (folder / f"{name}_metrics_before.txt").write_text(output(["curl", "-fsS", self.api + "/metrics"]))
                    self.run_client(docker(PYTHON, "-u", SCRIPTS + "table_request_load_simulator.py",
                        "--api-url", self.api + ("/v1/chat/completions" if self.vllm else "/v1/ocr"), "--cohort", "all", "--qps", str(qps),
                        "--max-requests", str(self.args.count), "--seed", "1", "--shuffle-all",
                        *extra, "--output-dir", str(target)), run, self.args.count / qps * 3 + 1200)
                    if self.vllm:
                        (folder / f"{name}_metrics_after.txt").write_text(output(["curl", "-fsS", self.api + "/metrics"]))
                    s = json.loads((self.repo / target / "summary.json").read_text())
                    stops = {}
                    sequence = []
                    dispatches = []
                    with (self.repo / target / "results.jsonl").open() as f:
                        for line in f:
                            r = json.loads(line)
                            dispatches.append(r["dispatch_offset_s"])
                            stop = (r.get("service_result") or {}).get("stop_reason", "error")
                            stops[stop] = stops.get(stop, 0) + 1
                    with (self.repo / target / "schedule.jsonl").open() as f:
                        sequence = [json.loads(line)["request_id"] for line in f]
                    seq_hash = hashlib.sha256(json.dumps(sequence).encode()).hexdigest()
                    if self.rows and seq_hash != self.rows[0]["sequence_sha256"]:
                        raise RuntimeError("Request sample/order changed")
                    latency = s["request_latency_s"]
                    dispatches.sort()
                    left = peak = 0
                    for right, timestamp in enumerate(dispatches):
                        while timestamp - dispatches[left] >= 1:
                            left += 1
                        peak = max(peak, right - left + 1)
                    row = dict(batch=16 if self.vllm else batch, token_budget=batch if self.vllm else None, target_qps=qps,
                        actual_arrival_qps=len(sequence)/dispatches[-1], peak_1s_arrivals=peak,
                        completed_qps=s["completed_request_count"]/s["run_wall_s"],
                        count=s["completed_request_count"], errors=s["failed_request_count"],
                        kv_caps=stops.get("kv_cache_full", 0) + stops.get("length", 0), mean_s=latency["mean"],
                        p50_s=latency["p50"], p90_s=latency["p90"], p95_s=latency["p95"],
                        p99_s=latency["p99"], max_s=latency["max"],
                        scheduled_p95_s=s["scheduled_latency_s"]["p95"],
                        drain_s=s["drain_after_last_arrival_s"], max_outstanding=s["max_active_requests"],
                        valid=s["failed_request_count"] == 0 and s["completed_request_count"] == self.args.count,
                        sequence_sha256=seq_hash, artifact=str(target))
                    self.rows.append(row)
                    self.publish()
                    self.log("RESULT " + json.dumps(row))
                    if s["failed_request_count"]:
                        raise RuntimeError("Request failures: keep evidence and reassess before further runs")
                self.stop()
            self.write("status.json", dict(status="screening_complete", points=len(self.rows)))
            self.log("SCREENING COMPLETE; NPU released; inspect frontier before 10k confirmation")
        except BaseException as exc:
            self.write("status.json", dict(status="stopped", error=repr(exc), completed_points=len(self.rows)))
            self.log(f"STOPPED: {exc!r}")
            raise
        finally:
            self.stop()
            self.ownership.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu", type=int, choices=range(8), required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--api-kind", choices=("crop", "vllm"), default="crop")
    parser.add_argument("--schedule-jsonl", type=Path,
        default=Path("tmp/09_persistent_page_engine/table_vllm_poisson100_qps1_e4b4a49e_20260908/measured/schedule.jsonl"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.is_absolute() or ".." in args.output_dir.parts or args.count < (1 if args.api_kind == "vllm" else 665):
        parser.error("Use a new relative artifact directory and a valid positive request count")
    if args.api_kind == "vllm" and (args.schedule_jsonl.is_absolute() or ".." in args.schedule_jsonl.parts):
        parser.error("Saved schedule must be repository relative")
    if args.plan_only:
        matrix = {n: list(range(1, 9)) for n in (4096, 8192, 16384)} if args.api_kind == "vllm" else MATRIX
        print(json.dumps(dict(matrix=matrix, points=sum(map(len, matrix.values())),
            expected_arrival_hours=sum(args.count/q for qs in matrix.values() for q in qs)/3600), indent=2))
        return
    Sweep(args).run()


if __name__ == "__main__":
    main()
