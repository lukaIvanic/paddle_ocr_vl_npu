#!/usr/bin/env python3
"""Analyze the granular UniRec vision input/H2D production trace."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "unirec_vision_h2d_breakdown_v1"
DEVICE_COMPONENTS = (
    "pixels_h2d_uint8_s",
    "pixels_h2d_cast_s",
    "pixels_normalize_layout_s",
    "pixel_mask_h2d_cast_s",
    "pixel_mask_apply_s",
    "mask2_h2d_s",
    "mask4_h2d_s",
    "mask8_h2d_s",
    "mask16_h2d_s",
    "mask32_h2d_s",
)
HOST_COMPONENTS = tuple(
    f"{name.removesuffix('_s')}_submit_s" for name in DEVICE_COMPONENTS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
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
        "p50_ms": percentile(samples, 0.50) * 1000.0,
        "p90_ms": percentile(samples, 0.90) * 1000.0,
        "p95_ms": percentile(samples, 0.95) * 1000.0,
        "p99_ms": percentile(samples, 0.99) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def event_identity(event: dict[str, Any]) -> dict[str, Any]:
    output = {
        "trace_index": event.get("trace_index"),
        "event": event.get("event"),
        "bucket": event.get("bucket"),
        "physical_input_shape": event.get("physical_input_shape"),
        "real_rows": event.get("real_rows"),
        "h2d_bytes": event.get("h2d_bytes"),
    }
    request_ids = [
        member.get("request_id")
        for member in event.get("members", [])
        if member.get("request_id") is not None
    ]
    if request_ids:
        output["request_ids"] = request_ids
    return output


def summarize_group(
    events: list[dict[str, Any]], *, top: int
) -> dict[str, Any]:
    device_samples: dict[str, list[float]] = defaultdict(list)
    host_samples: dict[str, list[float]] = defaultdict(list)
    outer_device = []
    outer_host = []
    device_residual = []
    host_residual = []
    ranked = []
    total_bytes = 0
    total_submissions = 0
    byte_histogram: dict[str, int] = defaultdict(int)

    for event in events:
        device = event.get("device_stage_s", {})
        host = event.get("host_stage_s", {})
        umbrella_device = float(device["input_h2d_normalize_s"])
        umbrella_host = float(host["input_h2d_submit_s"])
        outer_device.append(umbrella_device)
        outer_host.append(umbrella_host)
        component_device_sum = 0.0
        component_host_sum = 0.0
        for name in DEVICE_COMPONENTS:
            if name in device:
                value = float(device[name])
                device_samples[name].append(value)
                component_device_sum += value
        for name in HOST_COMPONENTS:
            if name in host:
                value = float(host[name])
                host_samples[name].append(value)
                component_host_sum += value
        device_residual.append(umbrella_device - component_device_sum)
        host_residual.append(umbrella_host - component_host_sum)
        event_bytes = event.get("h2d_bytes", {})
        total_bytes += int(event_bytes.get("total", 0))
        for name, value in event_bytes.items():
            if name != "total":
                byte_histogram[name] += int(value)
        total_submissions += int(event.get("h2d_tensor_submissions", 0))
        ranked.append((umbrella_device, event))

    device_components = {
        name: distribution(values)
        for name, values in sorted(device_samples.items())
    }
    host_components = {
        name: distribution(values)
        for name, values in sorted(host_samples.items())
    }
    umbrella_device = distribution(outer_device)
    umbrella_host = distribution(outer_host)
    return {
        "call_count": len(events),
        "h2d_tensor_submissions": total_submissions,
        "h2d_bytes": {
            **dict(sorted(byte_histogram.items())),
            "total": total_bytes,
        },
        "umbrella_device": umbrella_device,
        "umbrella_host": umbrella_host,
        "device_components": device_components,
        "host_components": host_components,
        "device_accounting_residual": distribution(device_residual),
        "host_accounting_residual": distribution(host_residual),
        "effective_payload_gib_s": (
            total_bytes / (1024**3) / float(umbrella_device["sum_s"])
            if umbrella_device.get("sum_s")
            else None
        ),
        "top_umbrella_device": [
            {"duration_ms": seconds * 1000.0, **event_identity(event)}
            for seconds, event in sorted(ranked, reverse=True, key=lambda pair: pair[0])[
                :top
            ]
        ],
    }


def compare_group(
    current: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "umbrella_device_sum_ratio": (
            float(current["umbrella_device"]["sum_s"])
            / float(reference["umbrella_device"]["sum_s"])
        ),
        "umbrella_device_mean_ratio": (
            float(current["umbrella_device"]["mean_ms"])
            / float(reference["umbrella_device"]["mean_ms"])
        ),
        "components": {},
    }
    for name in sorted(
        set(current["device_components"])
        & set(reference["device_components"])
    ):
        current_row = current["device_components"][name]
        reference_row = reference["device_components"][name]
        output["components"][name] = {
            "sum_ratio": float(current_row["sum_s"])
            / float(reference_row["sum_s"]),
            "mean_ratio": float(current_row["mean_ms"])
            / float(reference_row["mean_ms"]),
            "p99_ratio": float(current_row["p99_ms"])
            / float(reference_row["p99_ms"]),
        }
    return output


def print_group(prefix: str, group: dict[str, Any]) -> None:
    device = group["umbrella_device"]
    host = group["umbrella_host"]
    print(
        "UNIREC_VISION_H2D_GROUP "
        f"name={prefix} calls={group['call_count']} "
        f"submissions={group['h2d_tensor_submissions']} "
        f"bytes={group['h2d_bytes']['total']} "
        f"device_sum_s={device['sum_s']:.6f} "
        f"device_mean_ms={device['mean_ms']:.3f} "
        f"device_p99_ms={device['p99_ms']:.3f} "
        f"host_sum_s={host['sum_s']:.6f} "
        f"host_mean_ms={host['mean_ms']:.3f} "
        f"payload_gib_s={group['effective_payload_gib_s']:.6f}"
    )
    for name, row in group["device_components"].items():
        print(
            "UNIREC_VISION_H2D_DEVICE_COMPONENT "
            f"group={prefix} name={name} count={row['count']} "
            f"sum_s={row['sum_s']:.6f} mean_ms={row['mean_ms']:.3f} "
            f"p95_ms={row['p95_ms']:.3f} p99_ms={row['p99_ms']:.3f} "
            f"max_ms={row['max_ms']:.3f}"
        )


def main() -> None:
    args = parse_args()
    events = [
        event
        for event in read_jsonl(args.iterations)
        if event.get("event")
        in {"vision_bucket_call", "vision_fallback_call"}
    ]
    bucket_events = [
        event for event in events if event["event"] == "vision_bucket_call"
    ]
    fallback_events = [
        event for event in events if event["event"] == "vision_fallback_call"
    ]
    if not bucket_events:
        raise ValueError("trace contains no vision bucket calls")
    for event in bucket_events:
        if int(event.get("h2d_tensor_submissions", 0)) != 7:
            raise ValueError(
                "vision bucket trace is missing the seven-transfer breakdown"
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "label": args.label,
        "input": str(args.iterations.resolve()),
        "compiled": summarize_group(bucket_events, top=args.top),
        "fallback": summarize_group(fallback_events, top=args.top),
        "by_bucket": {
            bucket: summarize_group(
                [event for event in bucket_events if event["bucket"] == bucket],
                top=args.top,
            )
            for bucket in sorted({str(event["bucket"]) for event in bucket_events})
        },
    }
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        if reference.get("schema") != SCHEMA:
            raise ValueError("reference JSON has an unsupported schema")
        report["comparison"] = {
            "reference_label": reference.get("label"),
            "compiled": compare_group(report["compiled"], reference["compiled"]),
            "fallback": compare_group(report["fallback"], reference["fallback"]),
        }
    report["output_json"] = str(args.output_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "UNIREC_VISION_H2D_BREAKDOWN PASS "
        f"label={args.label} compiled_calls={len(bucket_events)} "
        f"fallback_calls={len(fallback_events)}"
    )
    print_group("compiled", report["compiled"])
    print_group("fallback", report["fallback"])
    for bucket, group in report["by_bucket"].items():
        print_group(bucket, group)
    if report.get("comparison"):
        comparison = report["comparison"]["compiled"]
        print(
            "UNIREC_VISION_H2D_REFERENCE "
            f"label={report['comparison']['reference_label']} "
            "device_sum_ratio="
            f"{comparison['umbrella_device_sum_ratio']:.6f} "
            "device_mean_ratio="
            f"{comparison['umbrella_device_mean_ratio']:.6f}"
        )
    print(f"UNIREC_VISION_H2D_JSON={report['output_json']}")


if __name__ == "__main__":
    main()
