#!/usr/bin/env python3
"""Send a fixed table set with a bounded number of outstanding OCR requests."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

import table_request_load_simulator as load


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
IN_FLIGHT_LIMITS = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set", choices=("a", "b", "p90", "warm"), required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument(
        "--max-in-flight", type=int, choices=IN_FLIGHT_LIMITS, default=1,
        help="Client-side outstanding-request limit. Refill after any response.",
    )
    parser.add_argument("--client-label", default="client")
    parser.add_argument("--source-jsonl", type=Path, default=load.DEFAULT_SOURCE)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--start-at-epoch-s",
        type=float,
        help="Wait until this wall-clock epoch after all payloads are ready.",
    )
    return parser.parse_args()


def select_tables(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load.read_jsonl(args.source_jsonl)
    cohort = load.freeze_tail_cohort(records, "p90", exclude_first_record=True)
    if len(cohort) < 65:
        raise ValueError(f"P90 cohort has only {len(cohort)} tables")
    if args.set == "a":
        selected = cohort[:64:2]
    elif args.set == "b":
        selected = cohort[1:64:2]
    elif args.set == "p90":
        selected = cohort[:64]
    else:
        selected = cohort[64:]
    if args.count <= 0:
        raise ValueError("--count must be positive")
    selected = selected[: args.count]
    if len(selected) != args.count:
        raise ValueError(
            f"set {args.set!r} has {len(selected)} tables, requested {args.count}"
        )
    return selected


async def run_closed_loop(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    payloads: dict[str, bytes],
    results_path: Path,
) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    if args.max_in_flight not in IN_FLIGHT_LIMITS:
        raise ValueError(f"--max-in-flight must be one of {IN_FLIGHT_LIMITS}")
    if args.start_at_epoch_s is not None:
        remaining = args.start_at_epoch_s - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)
    actual_start_epoch_s = time.time()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    pending = iter(enumerate(selected, start=1))
    active = 0
    peak_active = 0
    stop_sending = False
    with results_path.open("w", encoding="utf-8") as handle:
        async def worker() -> None:
            while not stop_sending:
                item = next(pending, None)
                if item is None:
                    return
                index, table = item
                await send_one(index, table)

        async def send_one(index: int, table: dict[str, Any]) -> None:
            nonlocal active, peak_active, stop_sending
            request_id = str(table["request_id"])
            request_started = time.perf_counter()
            active += 1
            active_at_send = active
            peak_active = max(peak_active, active)
            print(
                f"SEND {args.client_label} {index:02d}/{len(selected)} "
                f"table={request_id} active={active}", flush=True,
            )
            error: str | None = None
            service_result: dict[str, Any] = {}
            try:
                service_result = await load.post_table_ocr(
                    args.api_url,
                    f"{args.client_label}-{index:03d}-{request_id}",
                    payloads[request_id],
                    args.request_timeout_s,
                    source_request_id=request_id,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                # A timed-out HTTP request may still be executing on the server.
                # Do not replace it and exceed the intended server workload.
                stop_sending = True
            completed = time.perf_counter()
            active -= 1
            record = {
                "sequence": index,
                "client_label": args.client_label,
                "set": args.set,
                "request_id": request_id,
                "tail_rank": table["tail_rank"],
                "baseline_b1_latency_s": table["baseline_b1_latency_s"],
                "has_latex_markup": table["has_latex_markup"],
                "source_output_tokens": table.get("output_tokens"),
                "source_real_vision_tokens": table.get("real_vision_tokens"),
                "latency_s": completed - request_started,
                "dispatch_offset_s": request_started - started,
                "completion_offset_s": completed - started,
                "active_at_send": active_at_send,
                "active_after_response": active,
                "status": "ok" if error is None else "error",
                "error": error,
                "service_result": service_result,
            }
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            print(
                f"RECV {args.client_label} {index:02d}/{len(selected)} "
                f"rank={table['tail_rank']:02d} id={request_id} "
                f"latency={record['latency_s']:.4f}s status={record['status']} "
                f"active={active}",
                flush=True,
            )
        await asyncio.gather(*(worker() for _ in range(args.max_in_flight)))
    stats = {
        "max_in_flight": args.max_in_flight,
        "observed_max_in_flight": peak_active,
        "stopped_sending_after_error": stop_sending,
        "unsent_request_count": len(selected) - len(results),
    }
    return results, time.perf_counter() - started, actual_start_epoch_s, stats


def main() -> None:
    args = parse_args()
    if args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    selected = select_tables(args)
    load.write_jsonl(output_dir / "tables.jsonl", selected)
    ready = load.check_api_ready(args.api_url, min(args.request_timeout_s, 10.0))
    print(
        f"{args.client_label} API ready worker_pid={ready.get('worker_pid')}; "
        f"preloading {len(selected)} crops",
        flush=True,
    )
    payloads = load.prepare_http_payloads(selected, args.images_dir)
    print(
        f"{args.client_label} payloads ready bytes={sum(map(len, payloads.values()))}",
        flush=True,
    )

    results, run_wall_s, actual_start_epoch_s, client_stats = asyncio.run(
        run_closed_loop(
            args,
            selected,
            payloads,
            output_dir / "results.jsonl",
        )
    )
    latencies = [float(row["latency_s"]) for row in results]
    failures = [row for row in results if row["status"] != "ok"]
    summary = {
        "format": "table_closed_loop_api_client_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "client_label": args.client_label,
        "set": args.set,
        "requested_request_count": len(selected),
        "request_count": len(results),
        "failed_request_count": len(failures),
        **client_stats,
        "run_wall_s": run_wall_s,
        "completion_qps": len(results) / run_wall_s if run_wall_s else None,
        "successful_completion_qps": (
            (len(results) - len(failures)) / run_wall_s if run_wall_s else None
        ),
        "requested_start_epoch_s": args.start_at_epoch_s,
        "actual_start_epoch_s": actual_start_epoch_s,
        "start_lag_s": (
            actual_start_epoch_s - args.start_at_epoch_s
            if args.start_at_epoch_s is not None
            else None
        ),
        "latency_s": {
            **load.distribution(latencies),
            "sum": sum(latencies),
            "median": statistics.median(latencies),
        },
        "api_configuration": ready.get("configuration"),
        "scheduling_metrics_request_count": sum(
            bool(row["service_result"].get("response", {}).get("scheduling_metrics"))
            for row in results
        ),
        "request_ids": [str(row["request_id"]) for row in results],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE {args.client_label} requests={len(results)} failures={len(failures)} "
        f"max_in_flight={client_stats['observed_max_in_flight']} "
        f"wall={run_wall_s:.4f}s qps={summary['completion_qps']:.4f} "
        f"p50={summary['latency_s']['p50']:.4f}s "
        f"p95={summary['latency_s']['p95']:.4f}s "
        f"max={summary['latency_s']['max']:.4f}s",
        flush=True,
    )
    if failures or client_stats["unsent_request_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
