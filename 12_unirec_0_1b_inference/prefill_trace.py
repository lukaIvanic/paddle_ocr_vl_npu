"""Raw production-prefill traces and distribution summaries."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRACE_SCHEMA = "unirec_production_prefill_trace_v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(row), ensure_ascii=False))
            handle.write("\n")
    os.replace(partial, path)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize_samples(values: Iterable[float]) -> dict[str, float | int]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {"count": 0}
    total = sum(samples)
    return {
        "count": len(samples),
        "sum_s": total,
        "mean_ms": total * 1000.0 / len(samples),
        "min_ms": samples[0] * 1000.0,
        "p50_ms": _percentile(samples, 0.50) * 1000.0,
        "p75_ms": _percentile(samples, 0.75) * 1000.0,
        "p90_ms": _percentile(samples, 0.90) * 1000.0,
        "p95_ms": _percentile(samples, 0.95) * 1000.0,
        "p99_ms": _percentile(samples, 0.99) * 1000.0,
        "max_ms": samples[-1] * 1000.0,
    }


def _shape_key(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def summarize_crop_cpu_execution(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        event
        for event in events
        if event.get("event") == "recognition_crop_preprocess"
        and event.get("native_thread_id") is not None
    ]
    if not rows:
        return {"available": False, "task_count": 0}

    intervals = sorted(
        (
            int(row["monotonic_start_ns"]),
            int(row["monotonic_end_ns"]),
        )
        for row in rows
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    wall_union_s = sum(end - start for start, end in merged) / 1e9

    sweep = []
    for start, end in intervals:
        sweep.append((start, 1))
        sweep.append((end, -1))
    active = max_active = 0
    for _, delta in sorted(sweep, key=lambda item: (item[0], item[1])):
        active += delta
        max_active = max(max_active, active)

    by_thread: dict[str, dict[str, Any]] = {}
    for row in rows:
        interval_wall_s = (
            int(row["monotonic_end_ns"]) - int(row["monotonic_start_ns"])
        ) / 1e9
        key = str(int(row["native_thread_id"]))
        item = by_thread.setdefault(
            key,
            {
                "thread_name": str(row.get("thread_name", "")),
                "task_count": 0,
                "thread_cpu_s": 0.0,
                "task_wall_s": 0.0,
                "cpu_ids": set(),
            },
        )
        item["task_count"] += 1
        item["thread_cpu_s"] += float(row["thread_cpu_s"])
        item["task_wall_s"] += interval_wall_s
        for name in ("cpu_start", "cpu_end"):
            if row.get(name) is not None:
                item["cpu_ids"].add(int(row[name]))

    task_wall_s = sum(
        (
            int(row["monotonic_end_ns"]) - int(row["monotonic_start_ns"])
        ) / 1e9
        for row in rows
    )
    preprocess_service_wall_s = sum(float(row["wall_s"]) for row in rows)
    thread_cpu_s = sum(float(row["thread_cpu_s"]) for row in rows)
    cpu_ids = sorted(
        {
            int(cpu)
            for item in by_thread.values()
            for cpu in item["cpu_ids"]
        }
    )
    serializable_threads = {
        key: {
            **item,
            "cpu_ids": sorted(item["cpu_ids"]),
        }
        for key, item in sorted(by_thread.items(), key=lambda pair: int(pair[0]))
    }
    return {
        "available": True,
        "task_count": len(rows),
        "native_thread_count": len(by_thread),
        "native_threads": serializable_threads,
        "cpu_ids_observed": cpu_ids,
        "cpu_id_count_observed": len(cpu_ids),
        "max_concurrent_tasks": max_active,
        "task_wall_sum_s": task_wall_s,
        "preprocess_service_wall_sum_s": preprocess_service_wall_s,
        "thread_cpu_sum_s": thread_cpu_s,
        "active_window_union_s": wall_union_s,
        "average_task_concurrency": (
            task_wall_s / wall_union_s if wall_union_s else 0.0
        ),
        "average_cpu_cores_during_active_windows": (
            thread_cpu_s / wall_union_s if wall_union_s else 0.0
        ),
        "thread_cpu_over_task_wall": (
            thread_cpu_s / task_wall_s if task_wall_s else 0.0
        ),
    }


def summarize_trace(
    events: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    samples: dict[str, list[float]] = defaultdict(list)
    event_counts: Counter[str] = Counter()
    shapes: dict[str, Counter[str]] = defaultdict(Counter)

    for event in events:
        event_name = str(event["event"])
        event_counts[event_name] += 1
        if event.get("wall_s") is not None:
            samples[f"{event_name}.wall_s"].append(float(event["wall_s"]))
        for namespace in ("stage_s", "host_stage_s", "device_stage_s"):
            for name, seconds in (event.get(namespace) or {}).items():
                samples[f"{event_name}.{namespace}.{name}"].append(
                    float(seconds)
                )

        if event_name == "layout_batch_call":
            shapes["layout_physical_input"][
                _shape_key(event["physical_input_shape"])
            ] += 1
            shapes["layout_real_rows"][str(int(event["real_rows"]))] += 1
        elif event_name == "recognition_crop_preprocess":
            source = _shape_key(event["source_image_size"])
            processed = _shape_key(event["processed_image_size"])
            shapes["crop_source"][source] += 1
            shapes["crop_processed"][processed] += 1
            shapes["crop_transform"][f"{source}->{processed}"] += 1
        elif event_name in {"vision_bucket_call", "vision_fallback_call"}:
            shapes["vision_bucket"][str(event["bucket"])] += 1
            shapes["vision_physical_input"][
                _shape_key(event["physical_input_shape"])
            ] += 1
            shapes["vision_real_rows"][str(int(event["real_rows"]))] += 1
            for member in event["members"]:
                shapes["vision_member_processed"][
                    _shape_key(member["processed_image_size"])
                ] += 1
        elif event_name == "text_prefill_pack":
            shapes["text_pack_members"][str(int(event["member_count"]))] += 1
            shapes["text_pack_real_tokens"][
                str(int(event["real_source_tokens"]))
            ] += 1
            for member in event["members"]:
                shapes["text_member_source_tokens"][
                    str(int(member["source_tokens"]))
                ] += 1
        elif event_name == "cross_kv_d2h":
            for length in event["source_lengths"]:
                shapes["cross_kv_source_length"][str(int(length))] += 1

    for page in pages:
        for name, seconds in page["frontend_timing_s"].items():
            samples[f"page_frontend.{name}"].append(float(seconds))

    return {
        "schema": TRACE_SCHEMA,
        "config": _json_value(config),
        "page_count": len(pages),
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "stage_distributions": {
            name: summarize_samples(values)
            for name, values in sorted(samples.items())
        },
        "shape_histograms": {
            name: dict(
                sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            )
            for name, counter in sorted(shapes.items())
        },
        "cpu_execution": {
            "recognition_crop_preprocess": summarize_crop_cpu_execution(events)
        },
        "timing_semantics": {
            "npu": (
                "device_stage_s uses NPU events resolved after the enclosing "
                "production group boundary"
            ),
            "cpu": (
                "crop-preprocess events are sequential and additive"
                if int(config.get("recognition_preprocess_threads", 1)) == 1
                else (
                    "crop-preprocess events are per-thread service times and "
                    "must not be summed as wall time"
                )
            ),
            "vision": (
                "vision timings remain attached to real graph calls and are "
                "never divided among member crops"
            ),
        },
    }


def write_prefill_trace(
    output_dir: Path,
    payloads: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = []
    pages = []
    for payload in sorted(payloads, key=lambda item: int(item["page_index"])):
        pages.append(
            {
                "schema": TRACE_SCHEMA,
                "page_index": int(payload["page_index"]),
                "image_path": str(payload["image_path"]),
                "width": int(payload["width"]),
                "height": int(payload["height"]),
                "crop_count": len(payload["crops"]),
                "rejected_crop_count": int(
                    payload.get("cross_capacity_rejected_crops", 0)
                ),
                "frontend_timing_s": payload["frontend_timing_s"],
                "worker_prefill_stats": payload.get("worker_prefill_stats", {}),
            }
        )
        for event in payload.get("prefill_trace_events", []):
            events.append(
                {
                    "schema": TRACE_SCHEMA,
                    "trace_index": len(events),
                    **event,
                }
            )

    event_path = output_dir / "prefill_iterations.jsonl"
    page_path = output_dir / "prefill_pages.jsonl"
    summary_path = output_dir / "prefill_distributions.json"
    _atomic_jsonl(event_path, events)
    _atomic_jsonl(page_path, pages)
    summary = summarize_trace(events, pages, config=config)
    summary["artifacts"] = {
        "iterations": str(event_path),
        "pages": str(page_path),
        "distributions": str(summary_path),
    }
    _atomic_json(summary_path, summary)
    return summary
