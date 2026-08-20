#!/usr/bin/env python3
"""Replay open-loop table requests against a simple simulated OCR service."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable, TextIO


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_SOURCE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl"
)

LATEX_MARKUP = re.compile(
    r"(?:\\\(|\\\)|\\\[|\\\]|\$\$|\$[^$\n]+\$|"
    r"\\(?:frac|pm|times|mathrm|mathbf|mathit|text|sqrt|sum|alpha|beta|gamma)\b|"
    r"[\^_]\{)"
)


@dataclass(frozen=True)
class ScheduledRequest:
    sequence: int
    scheduled_offset_s: float
    table: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Saved per-table B1 records used to freeze the difficult cohort.",
    )
    parser.add_argument(
        "--cohort",
        choices=("p90", "p95"),
        default="p90",
        help="Use tables at or above this B1 latency percentile.",
    )
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument(
        "--ocr-time-s",
        type=float,
        default=0.5,
        help="Asynchronous sleep used as the simulated OCR service time.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a timestamped directory under tmp/09_persistent_page_engine.",
    )
    parser.add_argument(
        "--include-first-record",
        action="store_true",
        help="Keep the first saved B1 request, which is normally a cold-start artifact.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def has_latex_markup(row: dict[str, Any]) -> bool:
    text = str(row.get("ground_truth") or row.get("gt_html") or "")
    return LATEX_MARKUP.search(text) is not None


def freeze_tail_cohort(
    records: list[dict[str, Any]],
    cohort: str,
    *,
    exclude_first_record: bool = True,
) -> list[dict[str, Any]]:
    if cohort not in {"p90", "p95"}:
        raise ValueError(f"unsupported cohort: {cohort}")
    candidates = records[1:] if exclude_first_record else records
    candidates = [
        row
        for row in candidates
        if row.get("request_id") is not None and row.get("worker_wall_s") is not None
    ]
    if not candidates:
        raise ValueError("source has no records with request_id and worker_wall_s")

    tail_fraction = 0.10 if cohort == "p90" else 0.05
    tail_count = math.ceil(len(candidates) * tail_fraction)
    selected = sorted(
        candidates,
        key=lambda row: (-float(row["worker_wall_s"]), str(row["request_id"])),
    )[:tail_count]

    frozen: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        vision = row.get("vision") if isinstance(row.get("vision"), dict) else {}
        frozen.append(
            {
                "request_id": str(row["request_id"]),
                "tail_rank": rank,
                "baseline_b1_latency_s": float(row["worker_wall_s"]),
                "has_latex_markup": has_latex_markup(row),
                "output_tokens": row.get("output_tokens"),
                "real_vision_tokens": vision.get("real_vision_tokens"),
            }
        )
    return frozen


def make_schedule(
    cohort: list[dict[str, Any]],
    qps: float,
    duration_s: float,
    seed: int,
) -> list[ScheduledRequest]:
    if not cohort:
        raise ValueError("cohort must not be empty")
    if qps <= 0:
        raise ValueError("qps must be greater than zero")
    if duration_s <= 0:
        raise ValueError("duration-s must be greater than zero")

    arrival_rng = random.Random(seed)
    table_rng = random.Random(seed + 1)
    table_cycle: list[dict[str, Any]] = []
    schedule: list[ScheduledRequest] = []
    scheduled_offset_s = 0.0

    while True:
        scheduled_offset_s += arrival_rng.expovariate(qps)
        if scheduled_offset_s >= duration_s:
            break
        if not table_cycle:
            table_cycle = list(cohort)
            table_rng.shuffle(table_cycle)
        table = table_cycle.pop()
        schedule.append(
            ScheduledRequest(
                sequence=len(schedule) + 1,
                scheduled_offset_s=scheduled_offset_s,
                table=table,
            )
        )
    return schedule


def schedule_rows(schedule: list[ScheduledRequest]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": item.sequence,
            "scheduled_offset_s": item.scheduled_offset_s,
            **item.table,
        }
        for item in schedule
    ]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def format_seconds(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


async def run_schedule(
    schedule: list[ScheduledRequest],
    ocr_time_s: float,
    result_handle: TextIO,
    *,
    print_events: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ocr_time_s < 0:
        raise ValueError("ocr-time-s must not be negative")

    loop = asyncio.get_running_loop()
    start = loop.time()
    active = 0
    max_active = 0
    results: list[dict[str, Any]] = []

    async def simulate_ocr(item: ScheduledRequest) -> None:
        nonlocal active, max_active
        dispatch_time = loop.time()
        active += 1
        active_at_dispatch = active
        max_active = max(max_active, active)
        dispatch_offset_s = dispatch_time - start
        dispatch_lag_s = dispatch_offset_s - item.scheduled_offset_s
        request_id = str(item.table["request_id"])

        if print_events:
            print(
                f"[{dispatch_offset_s:8.3f}s] SEND #{item.sequence:05d} "
                f"table={request_id} lag={dispatch_lag_s * 1000:6.1f}ms "
                f"active={active}",
                flush=True,
            )

        await asyncio.sleep(ocr_time_s)

        completion_time = loop.time()
        active -= 1
        completion_offset_s = completion_time - start
        latency_s = completion_time - (start + item.scheduled_offset_s)
        result = {
            "sequence": item.sequence,
            "request_id": request_id,
            "scheduled_offset_s": item.scheduled_offset_s,
            "dispatch_offset_s": dispatch_offset_s,
            "dispatch_lag_s": dispatch_lag_s,
            "completion_offset_s": completion_offset_s,
            "latency_s": latency_s,
            "simulated_ocr_s": ocr_time_s,
            "active_at_dispatch": active_at_dispatch,
            "active_after_completion": active,
            "status": "ok",
            **{
                key: item.table.get(key)
                for key in (
                    "tail_rank",
                    "baseline_b1_latency_s",
                    "has_latex_markup",
                    "output_tokens",
                    "real_vision_tokens",
                )
            },
        }
        results.append(result)
        result_handle.write(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        result_handle.flush()

        if print_events:
            print(
                f"[{completion_offset_s:8.3f}s] RECV #{item.sequence:05d} "
                f"table={request_id} latency={latency_s:6.3f}s active={active}",
                flush=True,
            )

    tasks: list[asyncio.Task[None]] = []
    for item in schedule:
        target = start + item.scheduled_offset_s
        await asyncio.sleep(max(0.0, target - loop.time()))
        tasks.append(asyncio.create_task(simulate_ocr(item)))
    if tasks:
        await asyncio.gather(*tasks)

    run_wall_s = loop.time() - start
    results.sort(key=lambda row: int(row["sequence"]))
    return results, {"run_wall_s": run_wall_s, "max_active": max_active}


def make_summary(
    *,
    args: argparse.Namespace,
    cohort: list[dict[str, Any]],
    schedule: list[ScheduledRequest],
    results: list[dict[str, Any]],
    run_stats: dict[str, Any],
) -> dict[str, Any]:
    latencies = [float(row["latency_s"]) for row in results]
    dispatch_lags = [float(row["dispatch_lag_s"]) for row in results]
    latex_latencies = [
        float(row["latency_s"]) for row in results if row["has_latex_markup"]
    ]
    non_latex_latencies = [
        float(row["latency_s"]) for row in results if not row["has_latex_markup"]
    ]
    return {
        "format": "table_request_load_simulator_v1",
        "mode": "async_sleep",
        "source_jsonl": str(args.source_jsonl.resolve()),
        "cohort": args.cohort,
        "cohort_table_count": len(cohort),
        "cohort_latex_table_count": sum(
            bool(row["has_latex_markup"]) for row in cohort
        ),
        "target_qps": args.qps,
        "arrival_process": "poisson",
        "duration_s": args.duration_s,
        "ocr_time_s": args.ocr_time_s,
        "seed": args.seed,
        "scheduled_request_count": len(schedule),
        "completed_request_count": len(results),
        "run_wall_s": run_stats["run_wall_s"],
        "max_active_requests": run_stats["max_active"],
        "latency_s": distribution(latencies),
        "dispatch_lag_s": distribution(dispatch_lags),
        "latex_latency_s": distribution(latex_latencies),
        "non_latex_latency_s": distribution(non_latex_latencies),
    }


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f"tmp/09_persistent_page_engine/table_request_load_{stamp}"


def validate_args(args: argparse.Namespace) -> None:
    if args.qps <= 0:
        raise ValueError("--qps must be greater than zero")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be greater than zero")
    if args.ocr_time_s < 0:
        raise ValueError("--ocr-time-s must not be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)

    records = read_jsonl(args.source_jsonl)
    cohort = freeze_tail_cohort(
        records,
        args.cohort,
        exclude_first_record=not args.include_first_record,
    )
    schedule = make_schedule(cohort, args.qps, args.duration_s, args.seed)

    cohort_path = output_dir / "cohort.jsonl"
    schedule_path = output_dir / "schedule.jsonl"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(cohort_path, cohort)
    write_jsonl(schedule_path, schedule_rows(schedule))

    latex_count = sum(bool(row["has_latex_markup"]) for row in cohort)
    print(
        f"Frozen {args.cohort.upper()} cohort: {len(cohort)} tables "
        f"({latex_count} LaTeX, {len(cohort) - latex_count} non-LaTeX)",
        flush=True,
    )
    print(
        f"Scheduled {len(schedule)} Poisson arrivals at {args.qps:g} QPS over "
        f"{args.duration_s:g}s; simulated OCR time={args.ocr_time_s:g}s",
        flush=True,
    )
    print(f"Writing each completion to {results_path}", flush=True)

    wall_start = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as result_handle:
        results, run_stats = asyncio.run(
            run_schedule(schedule, args.ocr_time_s, result_handle)
        )
    process_wall_s = time.perf_counter() - wall_start
    run_stats["process_wall_s"] = process_wall_s
    summary = make_summary(
        args=args,
        cohort=cohort,
        schedule=schedule,
        results=results,
        run_stats=run_stats,
    )
    summary["process_wall_s"] = process_wall_s
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    latency = summary["latency_s"]
    print(
        "DONE "
        f"completed={summary['completed_request_count']} "
        f"max_active={summary['max_active_requests']} "
        f"p50={format_seconds(latency['p50'])} "
        f"p95={format_seconds(latency['p95'])} "
        f"max={format_seconds(latency['max'])}",
        flush=True,
    )
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
