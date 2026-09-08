#!/usr/bin/env python3
"""Verify a saved benchmark lock or PRINT its commands. Never runs inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load():
    return json.loads((HERE / "lock.json").read_text())


def check(lock):
    errors = []
    for relative, expected in lock["artifact_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Missing or modified evidence: {relative}")
    for lane in ("optimized", "vllm"):
        spec = lock[lane]
        lines = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-r", spec["commit"]], text=True).splitlines()
        blobs = {line.split("\t", 1)[1]: line.split()[2] for line in lines}
        for path, expected in spec["source_blobs"].items():
            if blobs.get(path) != expected:
                errors.append(f"Historical source mismatch: {lane}: {path}")
    schedule = [json.loads(line) for line in (ROOT / lock["schedule"]).read_text().splitlines()]
    ids = [r["request_id"] for r in schedule]
    if (len(ids) != lock["request_count"] or len(set(ids)) != lock["unique_tables"] or
            hashlib.sha256(json.dumps(ids).encode()).hexdigest() != lock["sequence_sha256"]):
        errors.append("Frozen request sequence mismatch")
    chart = json.loads((ROOT / lock["chart"]).read_text())
    all_custom = json.loads((ROOT / lock["optimized"]["artifact_root"] / "results.json").read_text())
    for point in chart["custom"]:
        qps = point["target_qps"]
        best = min((r for r in all_custom if r["target_qps"] == qps), key=lambda r: (r["p95_s"], r["batch"]))
        if any(best[key] != point[key] for key in ("batch", "mean_s", "p95_s")):
            errors.append(f"Chart is not paired to the lowest-P95 run at QPS{qps}")
        ready = json.loads((ROOT / lock["optimized"]["artifact_root"] /
                            f'b{point["batch"]}/ready.json').read_text())["configuration"]
        expected = dict(batch_size=point["batch"], cache_length=4096, max_new_tokens=4096,
                        decode_attention="increfa", decode_device_timing=False,
                        compact_decode_control=False, max_prefill_interruptions=None,
                        open_prefill_admission="free_decode_slots_only_cpu_lookahead")
        if any(ready.get(key) != value for key, value in expected.items()):
            errors.append(f"Resolved serving contract changed at QPS{qps}")
        if ready["token_selection"]["rule"] != "ordinary_argmax":
            errors.append(f"Non-greedy token selection at QPS{qps}")
    return errors


def worktree_differences(lock, lane):
    result = []
    for relative, expected in lock[lane]["source_blobs"].items():
        path = ROOT / relative
        if not path.is_file():
            result.append(relative)
            continue
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != expected:
            result.append(relative)
    return result


def commands(lock, lane, qps, npu, output):
    path = Path(output)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "tmp":
        raise ValueError("Use a NEW repository-relative directory below tmp/")
    if (ROOT / path).exists():
        raise ValueError("Output already exists; never overwrite original evidence")
    spec = lock[lane]
    if lane == "optimized":
        batch = spec["chart_batch_by_qps"].get(f"{qps:g}")
        if batch is None:
            raise ValueError("Optimized chart anchors are integer QPS1 through QPS6")
    elif qps not in spec["qps"]:
        raise ValueError("This QPS was not in the locked vLLM sweep")
    docker = ["docker", "exec", "-w", lock["container_repo"], lock["client_container"]]
    python = [lock["client_python"], "-u"]
    scripts = "09_persistent_page_engine/scripts/"
    if lane == "optimized":
        argv = [x.format(batch=batch, output=output) for x in spec["expanded_server_argv"]]
        server = docker + ["bash", "-c", f"source npu-setup >/dev/null; export ASCEND_RT_VISIBLE_DEVICES={npu}; exec " + shlex.join(argv)]
        endpoint = "http://127.0.0.1:8767/v1/ocr"
        kind = []  # Historical custom client predates --api-kind / replay flags.
    else:
        server = ["docker", "exec", "-e", f"ASCEND_RT_VISIBLE_DEVICES={npu}"]
        for key, value in spec["environment_overrides"].items():
            server += ["-e", f"{key}={value}"]
        server += [spec["container"], "bash", spec["container_launcher"]]
        endpoint = "http://127.0.0.1:18081/v1/chat/completions"
        kind = ["--api-kind", "vllm"]
    result = [("SERVER (separate terminal; await readiness)", server)]
    for index in range(1, spec["warmup_requests"]+1):
        result.append((f"WARMUP {index} (wait for response before continuing)", docker + python +
                       [scripts+"table_closed_loop_api_client.py", *kind, "--api-url", endpoint,
                        "--set", "warm", "--count", "1", "--max-in-flight", "1",
                        "--output-dir", f"{output}/warm{index}"]))
    client = docker + python + [scripts+"table_request_load_simulator.py", "--api-url", endpoint,
              "--source-jsonl", lock["source_jsonl"], "--images-dir", lock["images_dir"],
              "--cohort", "all", "--qps", f"{qps:g}", "--max-requests", "1000", "--seed", "1",
              "--shuffle-all", "--request-timeout-s", "900", *kind]
    if lane == "vllm":
        client += ["--vllm-model", "PaddleOCR-VL-1.6", "--schedule-jsonl", lock["schedule"],
                   "--schedule-source-qps", "1"]
    client += ["--output-dir", f"{output}/measured"]
    result.append(("MEASURE (no client in-flight cap; wait for all responses)", client))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", choices=("optimized", "vllm"), help="Also reject source drift in this checkout")
    parser.add_argument("--commands", choices=("optimized", "vllm"), help="Print commands ONLY; use pinned checkout/environment")
    parser.add_argument("--qps", type=float)
    parser.add_argument("--npu", type=int, choices=range(8))
    parser.add_argument("--output-dir")
    parser.add_argument("--model-dir", type=Path, help="Also hash local model/tokenizer files; no model import")
    args = parser.parse_args()
    lock = load()
    errors = check(lock)
    if args.worktree:
        errors += [f"Working-tree source differs: {p}" for p in worktree_differences(lock,args.worktree)]
    if args.model_dir:
        hashes = json.loads((HERE/"environment.json").read_text())["model_sha256"]
        for name, expected in hashes.items():
            path = args.model_dir/name
            if not path.is_file():
                errors.append(f"Missing model file: {name}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                errors.append(f"Model file hash differs: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("PASS: evidence hashes, historical source IDs, frozen 1000/665 sequence, paired chart metrics.")
    if args.commands:
        if args.qps is None or args.npu is None or not args.output_dir:
            parser.error("--commands requires --qps, --npu and --output-dir")
        print(f"# Requires {args.commands} source commit {lock[args.commands]['commit']} and the recorded environment.")
        print("# PRINT ONLY: no commands executed. Verify NPU ownership manually and monitor during measurement.")
        for title, argv in commands(lock,args.commands,args.qps,args.npu,args.output_dir):
            print(f"\n# {title}\n{shlex.join(argv)}")


if __name__ == "__main__":
    main()
