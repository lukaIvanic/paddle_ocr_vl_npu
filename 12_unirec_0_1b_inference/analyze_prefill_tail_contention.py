#!/usr/bin/env python3
"""Analyze per-call UniRec prefill tails and multi-worker NPU contention.

This consumes the existing production ``--prefill-trace`` artifacts.  It does
not add synchronization or change the measured inference path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "unirec_prefill_tail_contention_v1"
PAGE_RE = re.compile(
    r"UNIREC_LAYOUT_PROCESS_PAGE label=(?P<phase>\S+) "
    r"pages=(?P<ordinal>\d+)/(?P<total>\d+) "
    r"page_index=(?P<page_index>\d+) worker=(?P<worker>\d+) "
    r"worker_page_s=(?P<worker_page_s>[0-9.]+) "
    r"elapsed_s=(?P<elapsed_s>[0-9.]+) crops=(?P<crops>\d+) "
    r"rejected=(?P<rejected>\d+)"
)
NONWRITEABLE_WARNING = "The given NumPy array is not writable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--iterations", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0}
    total = sum(samples)
    return {
        "count": len(samples),
        "sum_s": total,
        "mean_ms": total * 1000.0 / len(samples),
        "min_ms": min(samples) * 1000.0,
        "p50_ms": percentile(samples, 0.50) * 1000.0,
        "p90_ms": percentile(samples, 0.90) * 1000.0,
        "p95_ms": percentile(samples, 0.95) * 1000.0,
        "p99_ms": percentile(samples, 0.99) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def flatten_duration_samples(
    events: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    samples: dict[str, list[float]] = defaultdict(list)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        event_name = str(event["event"])
        if event.get("wall_s") is not None:
            key = f"{event_name}.wall_s"
            samples[key].append(float(event["wall_s"]))
            rows[key].append(event)
        for namespace in ("stage_s", "host_stage_s", "device_stage_s"):
            for name, seconds in (event.get(namespace) or {}).items():
                key = f"{event_name}.{namespace}.{name}"
                samples[key].append(float(seconds))
                rows[key].append(event)
    return samples, rows


def event_identity(event: dict[str, Any]) -> dict[str, Any]:
    output = {
        "trace_index": event.get("trace_index"),
        "event": event.get("event"),
    }
    for name in (
        "page_index",
        "page_image",
        "page_indices",
        "page_images",
        "crop_index",
        "label",
        "bucket",
        "real_rows",
        "physical_rows",
        "physical_input_shape",
        "member_count",
        "real_source_tokens",
    ):
        if name in event:
            output[name] = event[name]
    if event.get("members"):
        output["member_request_ids"] = [
            member.get("request_id")
            for member in event["members"]
            if member.get("request_id") is not None
        ]
    return output


def top_events(
    samples: dict[str, list[float]],
    rows: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for key, values in samples.items():
        ranked = sorted(
            zip(values, rows[key]), key=lambda pair: pair[0], reverse=True
        )[:limit]
        output[key] = [
            {"duration_ms": seconds * 1000.0, **event_identity(event)}
            for seconds, event in ranked
        ]
    return output


def page_indices_for_event(event: dict[str, Any]) -> list[int]:
    if event.get("page_index") is not None:
        return [int(event["page_index"])]
    if event.get("page_indices") is not None:
        return [int(value) for value in event["page_indices"]]
    indices = set()
    for member in event.get("members") or []:
        request_id = member.get("request_id")
        if not request_id:
            continue
        match = re.match(r"page_(\d+)_crop_\d+", str(request_id))
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def per_page_analysis(
    events: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    page_rows = {
        int(row["page_index"]): {
            "page_index": int(row["page_index"]),
            "page_image": Path(row["image_path"]).name,
            "crop_count": int(row["crop_count"]),
            "layout_wall_s": 0.0,
            "layout_model_forward_s": 0.0,
            "crop_preprocess_service_s": 0.0,
            "vision_bucket_graph_s": 0.0,
            "vision_fallback_graph_s": 0.0,
            "text_prefill_device_s": 0.0,
            "cross_kv_d2h_s": 0.0,
            "shared_pack_s": 0.0,
            "worker_group_wall_s": 0.0,
        }
        for row in pages
    }
    for event in events:
        indices = page_indices_for_event(event)
        if not indices:
            continue
        share = 1.0 / len(indices)
        name = event["event"]
        for page_index in indices:
            row = page_rows.get(page_index)
            if row is None:
                continue
            if name == "layout_batch_call":
                row["layout_wall_s"] += float(event.get("wall_s", 0.0)) * share
                row["layout_model_forward_s"] += float(
                    event.get("stage_s", {}).get("model_forward_s", 0.0)
                ) * share
            elif name == "recognition_crop_preprocess":
                row["crop_preprocess_service_s"] += float(
                    event.get("wall_s", 0.0)
                )
            elif name == "vision_bucket_call":
                row["vision_bucket_graph_s"] += float(
                    event.get("device_stage_s", {}).get("graph_s", 0.0)
                ) * share
            elif name == "vision_fallback_call":
                row["vision_fallback_graph_s"] += float(
                    event.get("device_stage_s", {}).get("graph_s", 0.0)
                ) * share
            elif name == "text_prefill_pack":
                row["text_prefill_device_s"] += sum(
                    float(value)
                    for key, value in event.get("device_stage_s", {}).items()
                    if key.startswith("compiled_packed_text_prefill_s")
                )
            elif name == "cross_kv_d2h":
                row["cross_kv_d2h_s"] += float(event.get("wall_s", 0.0))
            elif name == "page_shared_pack":
                row["shared_pack_s"] += float(event.get("wall_s", 0.0))
            elif name == "worker_page_group":
                row["worker_group_wall_s"] += float(
                    event.get("wall_s", 0.0)
                ) * share
    rows = []
    for row in page_rows.values():
        row["npu_service_s"] = (
            row["layout_model_forward_s"]
            + row["vision_bucket_graph_s"]
            + row["vision_fallback_graph_s"]
            + row["text_prefill_device_s"]
        )
        rows.append(row)
    metric_distributions = {
        name: distribution(float(row[name]) for row in rows)
        for name in (
            "layout_wall_s",
            "layout_model_forward_s",
            "crop_preprocess_service_s",
            "vision_bucket_graph_s",
            "vision_fallback_graph_s",
            "text_prefill_device_s",
            "cross_kv_d2h_s",
            "shared_pack_s",
            "worker_group_wall_s",
            "npu_service_s",
        )
    }
    top = {}
    for metric in (
        "worker_group_wall_s",
        "npu_service_s",
        "crop_preprocess_service_s",
        "vision_fallback_graph_s",
    ):
        top[metric] = sorted(
            rows, key=lambda row: float(row[metric]), reverse=True
        )[:limit]
    return {"distributions": metric_distributions, "top_pages": top}


def npu_service_analysis(
    events: list[dict[str, Any]], trace_summary: dict[str, Any]
) -> dict[str, Any]:
    samples: dict[str, list[float]] = defaultdict(list)
    for event in events:
        name = event["event"]
        if name == "layout_batch_call":
            samples["layout_model_forward_s"].append(
                float(event.get("stage_s", {}).get("model_forward_s", 0.0))
            )
        elif name == "vision_bucket_call":
            samples["vision_bucket_graph_s"].append(
                float(event.get("device_stage_s", {}).get("graph_s", 0.0))
            )
        elif name == "vision_fallback_call":
            samples["vision_fallback_graph_s"].append(
                float(event.get("device_stage_s", {}).get("graph_s", 0.0))
            )
        elif name == "text_prefill_pack":
            samples["text_prefill_device_s"].append(
                sum(
                    float(value)
                    for key, value in event.get("device_stage_s", {}).items()
                    if key.startswith("compiled_packed_text_prefill_s")
                )
            )
    distributions = {
        name: distribution(values) for name, values in sorted(samples.items())
    }
    service_sum_s = sum(
        float(row.get("sum_s", 0.0)) for row in distributions.values()
    )
    trace_wall_s = float(trace_summary["timing_s"]["prefill_phase"])
    return {
        "components": distributions,
        "aggregate_service_sum_s": service_sum_s,
        "trace_prefill_wall_s": trace_wall_s,
        "aggregate_service_sum_over_trace_wall": (
            service_sum_s / trace_wall_s if trace_wall_s else None
        ),
        "interpretation": (
            "The sum is device service time across all workers. A ratio above "
            "one proves cross-worker overlap, but is not a device-utilization "
            "percentage because CPU work overlaps too."
        ),
    }


def bucket_analysis(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["event"] == "vision_bucket_call":
            grouped[str(event["bucket"])].append(event)
    output = {}
    for bucket, rows in sorted(grouped.items()):
        values = [
            float(row.get("device_stage_s", {}).get("graph_s", 0.0))
            for row in rows
        ]
        output[bucket] = {
            **distribution(values),
            "real_rows": sum(int(row["real_rows"]) for row in rows),
            "physical_rows": sum(int(row["physical_rows"]) for row in rows),
            "slot_efficiency": (
                sum(int(row["real_rows"]) for row in rows)
                / sum(int(row["physical_rows"]) for row in rows)
            ),
        }
    return output


def clean_progress_analysis(path: Path, *, limit: int) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pages = []
    warning_lines = []
    for line_number, line in enumerate(lines, 1):
        match = PAGE_RE.search(line)
        if match and match.group("phase") == "two_phase_measured_prefill":
            row: dict[str, Any] = {
                name: int(value)
                for name, value in match.groupdict().items()
                if name
                in {"ordinal", "total", "page_index", "worker", "crops", "rejected"}
            }
            row.update(
                {
                    "worker_page_s": float(match.group("worker_page_s")),
                    "elapsed_s": float(match.group("elapsed_s")),
                    "line_number": line_number,
                }
            )
            pages.append(row)
        if NONWRITEABLE_WARNING in line:
            warning_lines.append(line_number)
    for index, row in enumerate(pages):
        previous_elapsed = pages[index - 1]["elapsed_s"] if index else 0.0
        row["completion_gap_s"] = row["elapsed_s"] - previous_elapsed
        previous_line = pages[index - 1]["line_number"] if index else 0
        row["nonwriteable_warning_between_completions"] = any(
            previous_line < warning_line < row["line_number"]
            for warning_line in warning_lines
        )
    warning_intervals = [
        row for row in pages if row["nonwriteable_warning_between_completions"]
    ]
    top_gaps = sorted(
        pages, key=lambda row: row["completion_gap_s"], reverse=True
    )[:limit]
    return {
        "page_count": len(pages),
        "final_elapsed_s": pages[-1]["elapsed_s"] if pages else None,
        "completion_gap": distribution(row["completion_gap_s"] for row in pages),
        "worker_page": distribution(row["worker_page_s"] for row in pages),
        "top_completion_gaps": top_gaps,
        "nonwriteable_warning_count": len(warning_lines),
        "warning_adjacent_intervals": warning_intervals,
        "warning_adjacent_gap": distribution(
            row["completion_gap_s"] for row in warning_intervals
        ),
        "non_warning_gap": distribution(
            row["completion_gap_s"]
            for row in pages
            if not row["nonwriteable_warning_between_completions"]
        ),
    }


def ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator == 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def compare_reference(
    report: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "reference_label": reference.get("label"),
        "npu_components": {},
        "vision_buckets": {},
    }
    current_components = report["npu_service"]["components"]
    reference_components = reference["npu_service"]["components"]
    for name in sorted(set(current_components) & set(reference_components)):
        output["npu_components"][name] = {
            "mean_ratio": ratio_or_none(
                current_components[name].get("mean_ms"),
                reference_components[name].get("mean_ms"),
            ),
            "p99_ratio": ratio_or_none(
                current_components[name].get("p99_ms"),
                reference_components[name].get("p99_ms"),
            ),
            "sum_ratio": ratio_or_none(
                current_components[name].get("sum_s"),
                reference_components[name].get("sum_s"),
            ),
        }
    for name in sorted(
        set(report["vision_buckets"]) & set(reference["vision_buckets"])
    ):
        current = report["vision_buckets"][name]
        control = reference["vision_buckets"][name]
        output["vision_buckets"][name] = {
            "mean_ratio": ratio_or_none(current.get("mean_ms"), control.get("mean_ms")),
            "p99_ratio": ratio_or_none(current.get("p99_ms"), control.get("p99_ms")),
            "max_ratio": ratio_or_none(current.get("max_ms"), control.get("max_ms")),
        }
    return output


def print_report(report: dict[str, Any]) -> None:
    clean = report["clean_progress"]
    npu = report["npu_service"]
    gap = clean["completion_gap"]
    print(
        "UNIREC_PREFILL_TAIL_CONTENTION PASS "
        f"label={report['label']} pages={clean['page_count']} "
        f"trace_wall_s={npu['trace_prefill_wall_s']:.6f} "
        f"npu_service_sum_s={npu['aggregate_service_sum_s']:.6f} "
        "npu_service_over_wall="
        f"{npu['aggregate_service_sum_over_trace_wall']:.6f} "
        f"clean_final_elapsed_s={clean['final_elapsed_s']:.6f}"
    )
    print(
        "UNIREC_PREFILL_COMPLETION_GAPS "
        f"p50_ms={gap['p50_ms']:.3f} p95_ms={gap['p95_ms']:.3f} "
        f"p99_ms={gap['p99_ms']:.3f} max_ms={gap['max_ms']:.3f} "
        f"warnings={clean['nonwriteable_warning_count']}"
    )
    for name, row in npu["components"].items():
        print(
            "UNIREC_PREFILL_NPU_COMPONENT "
            f"name={name} count={row['count']} sum_s={row['sum_s']:.6f} "
            f"mean_ms={row['mean_ms']:.3f} p95_ms={row['p95_ms']:.3f} "
            f"p99_ms={row['p99_ms']:.3f} max_ms={row['max_ms']:.3f}"
        )
    for bucket, row in report["vision_buckets"].items():
        print(
            "UNIREC_PREFILL_VISION_BUCKET "
            f"bucket={bucket} count={row['count']} sum_s={row['sum_s']:.6f} "
            f"mean_ms={row['mean_ms']:.3f} p95_ms={row['p95_ms']:.3f} "
            f"p99_ms={row['p99_ms']:.3f} max_ms={row['max_ms']:.3f} "
            f"slot_eff={row['slot_efficiency']:.6f}"
        )
    for row in clean["top_completion_gaps"]:
        print(
            "UNIREC_PREFILL_TOP_GAP "
            f"ordinal={row['ordinal']} page_index={row['page_index']} "
            f"worker={row['worker']} gap_ms={row['completion_gap_s'] * 1000:.3f} "
            f"worker_page_ms={row['worker_page_s'] * 1000:.3f} "
            f"crops={row['crops']} warning_between="
            f"{str(row['nonwriteable_warning_between_completions']).lower()}"
        )
    if report.get("comparison"):
        for name, row in report["comparison"]["npu_components"].items():
            print(
                "UNIREC_PREFILL_REFERENCE_RATIO "
                f"name={name} mean_ratio={row['mean_ratio']} "
                f"p99_ratio={row['p99_ratio']} sum_ratio={row['sum_ratio']}"
            )
    print(f"UNIREC_PREFILL_TAIL_JSON={report['output_json']}")


def main() -> None:
    args = parse_args()
    trace_summary = json.loads(args.trace_summary.read_text(encoding="utf-8"))
    events = read_jsonl(args.iterations)
    pages = read_jsonl(args.pages)
    samples, sample_rows = flatten_duration_samples(events)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "label": args.label,
        "inputs": {
            "trace_summary": str(args.trace_summary.resolve()),
            "iterations": str(args.iterations.resolve()),
            "pages": str(args.pages.resolve()),
            "clean_log": str(args.clean_log.resolve()),
        },
        "trace_config": {
            name: trace_summary.get(name)
            for name in (
                "workers",
                "recognition_preprocess_threads",
                "layout_cpu_threads",
                "layout_batch_size",
                "vision_page_lookahead",
                "vision_bucket_preset",
            )
        },
        "event_distributions": {
            name: distribution(values) for name, values in sorted(samples.items())
        },
        "top_events": top_events(samples, sample_rows, limit=args.top),
        "per_page": per_page_analysis(events, pages, limit=args.top),
        "npu_service": npu_service_analysis(events, trace_summary),
        "vision_buckets": bucket_analysis(events),
        "clean_progress": clean_progress_analysis(args.clean_log, limit=args.top),
    }
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        if reference.get("schema") != SCHEMA:
            raise ValueError("reference JSON has an unsupported schema")
        report["comparison"] = compare_reference(report, reference)
    report["output_json"] = str(args.output_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(report)


if __name__ == "__main__":
    main()
