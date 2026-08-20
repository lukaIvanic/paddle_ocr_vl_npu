#!/usr/bin/env python3
"""Replay open-loop table requests against simulated or HTTP OCR."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Awaitable, Callable, Iterable, TextIO
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlopen


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
        help="Asynchronous sleep used when --api-url is not set.",
    )
    parser.add_argument(
        "--api-url",
        help="Send real table crops to this OCR endpoint instead of sleeping.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
        help="OmniDocBench images used to prepare HTTP request bodies.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
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
                "page_name": row.get("page_name"),
                "bbox_xyxy": row.get("bbox_xyxy"),
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


def prepare_http_payloads(
    cohort: list[dict[str, Any]],
    images_dir: Path,
) -> dict[str, bytes]:
    from PIL import Image

    payloads: dict[str, bytes] = {}
    for table in cohort:
        request_id = str(table["request_id"])
        page_name = table.get("page_name")
        bbox = table.get("bbox_xyxy")
        if not page_name or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{request_id}: missing page_name or bbox_xyxy")
        image_path = images_dir / str(page_name)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as page:
            crop = page.crop(tuple(int(value) for value in bbox)).convert("RGB")
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        payloads[request_id] = buffer.getvalue()
    return payloads


def check_api_ready(api_url: str, timeout_s: float) -> dict[str, Any]:
    parsed = urlparse(api_url)
    ready_url = urlunparse(parsed._replace(path="/ready", query=""))
    with urlopen(ready_url, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not payload.get("ready"):
        raise RuntimeError(f"OCR API is not ready: {payload}")
    return payload


async def post_table_ocr(
    api_url: str,
    request_id: str,
    image_bytes: bytes,
    timeout_s: float,
) -> dict[str, Any]:
    parsed = urlparse(api_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--api-url must be an http:// URL")
    port = parsed.port or 80
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({"crop_type": "table", "request_id": request_id})
    target = urlunparse(("", "", parsed.path or "/v1/ocr", "", urlencode(query), ""))
    host_header = parsed.hostname if port == 80 else f"{parsed.hostname}:{port}"
    header = (
        f"POST {target} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Content-Type: image/png\r\n"
        f"Content-Length: {len(image_bytes)}\r\n"
        "Connection: close\r\n\r\n"
    )

    async def request() -> dict[str, Any]:
        reader, writer = await asyncio.open_connection(parsed.hostname, port)
        try:
            writer.write(bytes(header, "ascii"))
            writer.write(image_bytes)
            await writer.drain()

            status_line = (await reader.readline()).decode("iso-8859-1").strip()
            parts = status_line.split(" ", 2)
            if len(parts) < 2 or not parts[1].isdigit():
                raise RuntimeError(f"invalid HTTP status line: {status_line!r}")
            status = int(parts[1])
            headers: dict[str, str] = {}
            while True:
                raw_line = await reader.readline()
                if raw_line in {b"", b"\r\n", b"\n"}:
                    break
                name, value = raw_line.decode("iso-8859-1").split(":", 1)
                headers[name.strip().lower()] = value.strip()
            content_length = int(headers.get("content-length", "0"))
            body = (
                await reader.readexactly(content_length)
                if content_length
                else await reader.read()
            )
            payload = json.loads(body.decode("utf-8")) if body else {}
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {payload}")
            if not isinstance(payload, dict):
                raise RuntimeError("OCR response must be a JSON object")
            return {
                "http_status": status,
                "worker_wall_s": payload.get("worker_wall_s"),
                "server_http_wall_s": payload.get("http_wall_s"),
                "output_tokens": payload.get("output_tokens"),
                "stop_reason": payload.get("stop_reason"),
                "response": payload,
            }
        finally:
            writer.close()
            await writer.wait_closed()

    return await asyncio.wait_for(request(), timeout=timeout_s)


RequestFunction = Callable[[ScheduledRequest], Awaitable[dict[str, Any]]]


async def run_schedule(
    schedule: list[ScheduledRequest],
    ocr_time_s: float,
    result_handle: TextIO,
    *,
    print_events: bool = True,
    request_function: RequestFunction | None = None,
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
            line = (
                f"[{dispatch_offset_s:8.3f}s] SEND #{item.sequence:05d} "
                f"table={request_id} lag={dispatch_lag_s * 1000:6.1f}ms "
                f"active={active}"
            )
            print(line, flush=True)

        service_result: dict[str, Any] = {}
        error: str | None = None
        try:
            if request_function is None:
                await asyncio.sleep(ocr_time_s)
                service_result = {"simulated_ocr_s": ocr_time_s}
            else:
                service_result = await request_function(item)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

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
            "active_at_dispatch": active_at_dispatch,
            "active_after_completion": active,
            "status": "ok" if error is None else "error",
            "error": error,
            "service_result": service_result,
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
            line = (
                f"[{completion_offset_s:8.3f}s] RECV #{item.sequence:05d} "
                f"table={request_id} latency={latency_s:6.3f}s "
                f"status={result['status']} active={active}"
            )
            print(line, flush=True)

    tasks: list[asyncio.Task[None]] = []
    for item in schedule:
        target = start + item.scheduled_offset_s
        await asyncio.sleep(max(0.0, target - loop.time()))
        tasks.append(asyncio.create_task(simulate_ocr(item)))
    if tasks:
        await asyncio.gather(*tasks)

    run_wall_s = loop.time() - start
    results.sort(key=lambda row: int(row["sequence"]))
    return results, {
        "run_wall_s": run_wall_s,
        "max_active": max_active,
        "error_count": sum(row["status"] != "ok" for row in results),
    }


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
    worker_latencies = [
        float(row["service_result"]["worker_wall_s"])
        for row in results
        if row["service_result"].get("worker_wall_s") is not None
    ]
    return {
        "format": "table_request_load_simulator_v1",
        "mode": "http" if args.api_url else "async_sleep",
        "api_url": args.api_url,
        "source_jsonl": str(args.source_jsonl.resolve()),
        "cohort": args.cohort,
        "cohort_table_count": len(cohort),
        "cohort_latex_table_count": sum(
            bool(row["has_latex_markup"]) for row in cohort
        ),
        "target_qps": args.qps,
        "arrival_process": "poisson",
        "duration_s": args.duration_s,
        "ocr_time_s": None if args.api_url else args.ocr_time_s,
        "seed": args.seed,
        "scheduled_request_count": len(schedule),
        "completed_request_count": len(results),
        "failed_request_count": run_stats["error_count"],
        "run_wall_s": run_stats["run_wall_s"],
        "max_active_requests": run_stats["max_active"],
        "latency_s": distribution(latencies),
        "dispatch_lag_s": distribution(dispatch_lags),
        "latex_latency_s": distribution(latex_latencies),
        "non_latex_latency_s": distribution(non_latex_latencies),
        "server_worker_latency_s": distribution(worker_latencies),
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
    if args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be greater than zero")


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

    request_function: RequestFunction | None = None
    if args.api_url:
        ready = check_api_ready(args.api_url, min(args.request_timeout_s, 10.0))
        print(
            f"OCR API ready worker_pid={ready.get('worker_pid')}; "
            "preparing table crops in RAM",
            flush=True,
        )
        payloads = prepare_http_payloads(cohort, args.images_dir)

        async def send_http_request(item: ScheduledRequest) -> dict[str, Any]:
            request_id = str(item.table["request_id"])
            return await post_table_ocr(
                args.api_url,
                f"load-{item.sequence:05d}-{request_id}",
                payloads[request_id],
                args.request_timeout_s,
            )

        request_function = send_http_request

    latex_count = sum(bool(row["has_latex_markup"]) for row in cohort)
    print(
        f"Frozen {args.cohort.upper()} cohort: {len(cohort)} tables "
        f"({latex_count} LaTeX, {len(cohort) - latex_count} non-LaTeX)",
        flush=True,
    )
    mode_description = (
        f"HTTP endpoint={args.api_url}"
        if args.api_url
        else f"simulated OCR time={args.ocr_time_s:g}s"
    )
    print(
        f"Scheduled {len(schedule)} Poisson arrivals at {args.qps:g} QPS over "
        f"{args.duration_s:g}s; {mode_description}",
        flush=True,
    )
    print(f"Writing each completion to {results_path}", flush=True)

    wall_start = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as result_handle:
        results, run_stats = asyncio.run(
            run_schedule(
                schedule,
                args.ocr_time_s,
                result_handle,
                request_function=request_function,
            )
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
        f"failed={summary['failed_request_count']} "
        f"max_active={summary['max_active_requests']} "
        f"p50={format_seconds(latency['p50'])} "
        f"p95={format_seconds(latency['p95'])} "
        f"max={format_seconds(latency['max'])}",
        flush=True,
    )
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
