#!/usr/bin/env python3
"""Isolate UniRec fixed-bucket graph compilation/replay in fresh processes.

The parent process never imports torch or initializes an NPU. It launches each
test case in a child, writes the child's combined stdout/stderr directly to a
log, and records its return code. A SIGKILL/SIGABRT/segfault therefore remains
observable even when the NPU child cannot raise or serialize a Python error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
BUCKET_KEYS = (
    "960x64_b16",
    "512x256_b16",
    "960x256_b4",
    "512x512_b8",
    "960x512_b4",
)


def _parse_bucket_keys(value: str) -> tuple[str, ...]:
    keys = tuple(item.strip() for item in value.split(",") if item.strip())
    if not keys:
        raise argparse.ArgumentTypeError("at least one bucket is required")
    unknown = [key for key in keys if key not in BUCKET_KEYS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown buckets {unknown}; expected a subset of {BUCKET_KEYS}"
        )
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("bucket list contains duplicates")
    return keys


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--buckets",
        type=_parse_bucket_keys,
        default=BUCKET_KEYS,
        help="Comma-separated bucket keys in the desired cumulative order.",
    )
    parser.add_argument(
        "--suite",
        choices=("isolated", "cumulative", "both"),
        default="both",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=2,
        help="Synchronized calls per graph. Two exposes first-call versus replay.",
    )
    parser.add_argument(
        "--jit-compile",
        choices=("off", "on"),
        default="off",
        help="Production is off. On exists only to reproduce the prior lab bug.",
    )
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-buckets",
        type=_parse_bucket_keys,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.calls < 1:
        parser.error("--calls must be positive")
    if args.case_timeout_s <= 0:
        parser.error("--case-timeout-s must be positive")
    if args.child and not args.child_buckets:
        parser.error("child mode requires --child-buckets")
    return args


def _emit(event: str, **fields: Any) -> None:
    print(
        "UNIREC_VISION_CRASH_PROBE "
        + json.dumps(
            {
                "event": event,
                "monotonic_s": time.monotonic(),
                "pid": os.getpid(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(
            "source npu-setup and expose exactly one NPU before running the probe; "
            f"got ASCEND_RT_VISIBLE_DEVICES={raw!r}"
        )
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _child_main(args: argparse.Namespace) -> None:
    physical_devices = _physical_devices()
    _emit(
        "child_begin",
        physical_devices=physical_devices,
        buckets=list(args.child_buckets),
        calls=args.calls,
        jit_compile=args.jit_compile,
    )

    import torch
    import torch_npu

    sys.path.insert(0, str(HERE))
    from modeling_optimized_unirec import (  # noqa: PLC0415
        OptimizedUniRecRunner,
    )
    from vision_full_batch import (  # noqa: PLC0415
        DEFAULT_VISION_BUCKETS,
        BucketedFullVisionRuntime,
    )

    torch_npu.npu.set_compile_mode(jit_compile=args.jit_compile == "on")
    specs_by_key = {spec.key: spec for spec in DEFAULT_VISION_BUCKETS}
    specs = tuple(specs_by_key[key] for key in args.child_buckets)

    _emit("model_load_begin", buckets=list(args.child_buckets))
    model_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    _emit(
        "model_load_end",
        buckets=list(args.child_buckets),
        wall_s=time.perf_counter() - model_started,
        memory_allocated_bytes=int(torch.npu.memory_allocated("npu:0")),
        memory_reserved_bytes=int(torch.npu.memory_reserved("npu:0")),
    )

    _emit("runtime_init_begin", buckets=list(args.child_buckets))
    runtime_started = time.perf_counter()
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=specs,
        diagnostic_graph_log=True,
    )
    _emit(
        "runtime_init_end",
        buckets=list(args.child_buckets),
        wall_s=time.perf_counter() - runtime_started,
    )

    _emit(
        "synchronized_graph_calls_begin",
        buckets=list(args.child_buckets),
        calls=args.calls,
    )
    report = runtime.warmup_all(passes=args.calls)
    _emit(
        "synchronized_graph_calls_end",
        buckets=list(args.child_buckets),
        calls=args.calls,
        report=report,
        memory_allocated_bytes=int(torch.npu.memory_allocated("npu:0")),
        memory_reserved_bytes=int(torch.npu.memory_reserved("npu:0")),
        max_memory_allocated_bytes=int(torch.npu.max_memory_allocated("npu:0")),
        max_memory_reserved_bytes=int(torch.npu.max_memory_reserved("npu:0")),
    )
    _emit("child_end", status="ok", buckets=list(args.child_buckets))


def _npu_smi_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30.0,
        )
        return {"returncode": result.returncode, "output": result.stdout}
    except BaseException as exception:
        return {"error": repr(exception)}


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_command(
    args: argparse.Namespace,
    buckets: tuple[str, ...],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-path",
        str(args.model_path.expanduser().resolve()),
        "--cache-dir",
        str(args.cache_dir.expanduser().resolve()),
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
        "--calls",
        str(args.calls),
        "--jit-compile",
        args.jit_compile,
        "--case-timeout-s",
        str(args.case_timeout_s),
        "--child",
        "--child-buckets",
        ",".join(buckets),
    ]


def _run_case(
    args: argparse.Namespace,
    *,
    case_name: str,
    buckets: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / f"{case_name}.log"
    command = _case_command(args, buckets)
    started = time.perf_counter()
    before = _npu_smi_snapshot()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "UNIREC_VISION_CRASH_PROBE_PARENT "
            + json.dumps(
                {
                    "event": "case_begin",
                    "case": case_name,
                    "buckets": list(buckets),
                    "command": command,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            returncode = process.wait(timeout=args.case_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=15.0)
        log_file.write(
            "UNIREC_VISION_CRASH_PROBE_PARENT "
            + json.dumps(
                {
                    "event": "case_end",
                    "case": case_name,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "wall_s": time.perf_counter() - started,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log_file.flush()
    return {
        "case": case_name,
        "buckets": list(buckets),
        "command": command,
        "returncode": returncode,
        "signal": -returncode if returncode < 0 else None,
        "timed_out": timed_out,
        "wall_s": time.perf_counter() - started,
        "log": str(log_path),
        "npu_smi_before": before,
        "npu_smi_after": _npu_smi_snapshot(),
    }


def _parent_main(args: argparse.Namespace) -> None:
    physical_devices = _physical_devices()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = tuple(args.buckets)
    cases: list[tuple[str, tuple[str, ...]]] = []
    if args.suite in {"isolated", "both"}:
        cases.extend((f"isolated_{key}", (key,)) for key in selected)
    if args.suite in {"cumulative", "both"}:
        cases.append(("cumulative_forward", selected))
        if len(selected) > 1:
            cases.append(("cumulative_reverse", tuple(reversed(selected))))

    summary: dict[str, Any] = {
        "status": "running",
        "physical_devices": physical_devices,
        "suite": args.suite,
        "buckets": list(selected),
        "calls_per_graph": args.calls,
        "jit_compile": args.jit_compile,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "cases": [],
    }
    summary_path = output_dir / "summary.json"
    _write_summary(summary_path, summary)
    for case_name, buckets in cases:
        print(
            f"UNIREC_VISION_CRASH_PROBE_CASE_BEGIN case={case_name} "
            f"buckets={','.join(buckets)}",
            flush=True,
        )
        result = _run_case(
            args,
            case_name=case_name,
            buckets=buckets,
            output_dir=output_dir,
        )
        summary["cases"].append(result)
        _write_summary(summary_path, summary)
        print(
            f"UNIREC_VISION_CRASH_PROBE_CASE_END case={case_name} "
            f"returncode={result['returncode']} signal={result['signal']} "
            f"wall_s={result['wall_s']:.3f} log={result['log']}",
            flush=True,
        )
    summary["status"] = (
        "ok" if all(case["returncode"] == 0 for case in summary["cases"])
        else "failed"
    )
    _write_summary(summary_path, summary)
    print(f"OUTPUT_JSON={summary_path}", flush=True)
    if summary["status"] != "ok":
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.child:
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
