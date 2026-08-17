#!/usr/bin/env python3
"""Compare two production UniRec vision plans from their exact trace artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VISION_EVENTS = {"vision_bucket_call", "vision_fallback_call"}
CONTRACT_KEYS = (
    "page_count",
    "workers",
    "recognition_preprocess_threads",
    "layout_batch_size",
    "layout_cpu_threads",
    "layout_execution",
    "layout_dtype",
    "layout_reading_order_dtype",
    "layout_threshold",
    "vision_focal_depthwise_rewrite",
    "vision_weight_format",
    "cross_cache_length",
    "self_cache_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-buckets", type=int, default=5)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def stage_sum(distributions: dict[str, Any], name: str) -> float:
    row = distributions["stage_distributions"].get(name)
    return 0.0 if row is None else float(row["sum_s"])


def page_identity(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "prefill_pages.jsonl"
    if not path.is_file():
        return {"available": False}
    rows = read_jsonl(path)
    identity = []
    workload = []
    for row in rows:
        identity.append(
            {
                "page_index": int(row["page_index"]),
                "image": Path(row["image_path"]).name,
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )
        workload.append(
            {
                **identity[-1],
                "crop_count": int(row["crop_count"]),
                "rejected_crop_count": int(row["rejected_crop_count"]),
            }
        )

    def digest(values: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "available": True,
        "count": len(rows),
        "identity_digest": digest(identity),
        "workload_digest": digest(workload),
    }


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    call_count = 0
    real_rows = 0
    physical_rows = 0
    effective_pixels = 0
    physical_pixels = 0
    graph_s = 0.0
    input_device_s = 0.0
    output_device_s = 0.0
    h2d_bytes = 0
    graph_samples = []
    for event in events:
        call_count += 1
        real_rows += int(event["real_rows"])
        physical_rows += int(event["physical_rows"])
        shape = [int(value) for value in event["physical_input_shape"]]
        event_physical_pixels = shape[0] * shape[2] * shape[3]
        event_effective_pixels = sum(
            int(member["processed_image_size"][0])
            * int(member["processed_image_size"][1])
            for member in event["members"]
        )
        physical_pixels += event_physical_pixels
        effective_pixels += event_effective_pixels
        device = event.get("device_stage_s", {})
        event_graph_s = float(device.get("graph_s", 0.0))
        graph_s += event_graph_s
        graph_samples.append(event_graph_s)
        input_device_s += float(device.get("input_h2d_normalize_s", 0.0))
        output_device_s += float(device.get("output_compact_s", 0.0))
        h2d_bytes += int(event.get("h2d_bytes", {}).get("total", 0))
    effective_mpix = effective_pixels / 1_000_000.0
    physical_mpix = physical_pixels / 1_000_000.0
    device_total_s = graph_s + input_device_s + output_device_s
    return {
        "calls": call_count,
        "real_rows": real_rows,
        "physical_rows": physical_rows,
        "slot_efficiency": ratio(real_rows, physical_rows),
        "effective_pixels": effective_pixels,
        "physical_pixels": physical_pixels,
        "effective_mpix": effective_mpix,
        "physical_mpix": physical_mpix,
        "pixel_efficiency": ratio(effective_pixels, physical_pixels),
        "graph_s": graph_s,
        "graph_mean_ms": ratio(graph_s * 1000.0, call_count),
        "input_device_s": input_device_s,
        "output_device_s": output_device_s,
        "device_total_s": device_total_s,
        "effective_mpix_per_graph_s": ratio(effective_mpix, graph_s),
        "physical_mpix_per_graph_s": ratio(physical_mpix, graph_s),
        "graph_ms_per_effective_mpix": ratio(graph_s * 1000.0, effective_mpix),
        "h2d_bytes": h2d_bytes,
    }


def load_lane(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    run = read_json(output_dir / "run_summary.json")
    distributions = read_json(output_dir / "prefill_distributions.json")
    events = [
        row
        for row in read_jsonl(output_dir / "prefill_iterations.jsonl")
        if row.get("event") in VISION_EVENTS
    ]
    if run.get("status") != "ok" or not run.get("prefill_trace_enabled"):
        raise ValueError(f"not a completed trace run: {output_dir}")
    if distributions.get("schema") != "unirec_production_prefill_trace_v1":
        raise ValueError(f"unsupported trace schema: {output_dir}")
    if not events:
        raise ValueError(f"no vision events: {output_dir}")
    compiled = [row for row in events if row["event"] == "vision_bucket_call"]
    fallback = [row for row in events if row["event"] == "vision_fallback_call"]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_bucket[str(event["bucket"])].append(event)
    total = summarize_events(events)
    vision_wall_s = stage_sum(
        distributions, "page_frontend.recognition_full_vision_s"
    )
    total["vision_wall_s"] = vision_wall_s
    total["vision_wall_residual_s"] = vision_wall_s - total["device_total_s"]
    return {
        "output_dir": str(output_dir),
        "config": {
            key: run[key]
            for key in (
                *CONTRACT_KEYS,
                "vision_page_lookahead",
                "vision_bucket_preset",
            )
        },
        "prefill": {
            "wall_s": float(run["timing_s"]["prefill_phase"]),
            "pages_per_s": float(run["throughput"]["prefill_pages_per_s"]),
            "crop_count": int(run["retained_bank"]["crop_count"]),
            "rejected_crop_count": int(
                run["retained_bank"]["rejected_crop_count"]
            ),
        },
        "page_identity": page_identity(output_dir),
        "vision": total,
        "compiled": summarize_events(compiled),
        "fallback": summarize_events(fallback),
        "by_bucket": {
            key: summarize_events(rows) for key, rows in sorted(by_bucket.items())
        },
    }


def comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    contract_differences = {
        key: {"baseline": baseline["config"][key], "candidate": candidate["config"][key]}
        for key in CONTRACT_KEYS
        if baseline["config"][key] != candidate["config"][key]
    }
    baseline_pages = baseline["page_identity"]
    candidate_pages = candidate["page_identity"]
    page_identity_exact = bool(
        baseline_pages.get("available")
        and candidate_pages.get("available")
        and baseline_pages["identity_digest"]
        == candidate_pages["identity_digest"]
    )
    page_workload_exact = bool(
        page_identity_exact
        and baseline_pages["workload_digest"]
        == candidate_pages["workload_digest"]
    )
    metric_names = (
        "vision_wall_s",
        "graph_s",
        "input_device_s",
        "output_device_s",
        "device_total_s",
        "vision_wall_residual_s",
        "calls",
        "real_rows",
        "physical_rows",
        "effective_pixels",
        "physical_pixels",
        "slot_efficiency",
        "pixel_efficiency",
        "graph_mean_ms",
        "effective_mpix_per_graph_s",
        "physical_mpix_per_graph_s",
        "graph_ms_per_effective_mpix",
    )
    metrics = {}
    for name in metric_names:
        before = float(baseline["vision"][name])
        after = float(candidate["vision"][name])
        metrics[name] = {
            "baseline": before,
            "candidate": after,
            "candidate_minus_baseline": after - before,
            "candidate_over_baseline": ratio(after, before),
        }
    prefill_before = float(baseline["prefill"]["wall_s"])
    prefill_after = float(candidate["prefill"]["wall_s"])
    graph_delta = metrics["graph_s"]["candidate_minus_baseline"]
    input_delta = metrics["input_device_s"]["candidate_minus_baseline"]
    output_delta = metrics["output_device_s"]["candidate_minus_baseline"]
    residual_delta = metrics["vision_wall_residual_s"]["candidate_minus_baseline"]
    wall_delta = metrics["vision_wall_s"]["candidate_minus_baseline"]
    return {
        "contract_differences": contract_differences,
        "page_identity_exact": page_identity_exact,
        "page_workload_exact": page_workload_exact,
        "metrics": metrics,
        "prefill": {
            "baseline_wall_s": prefill_before,
            "candidate_wall_s": prefill_after,
            "candidate_minus_baseline_s": prefill_after - prefill_before,
            "speedup": ratio(prefill_before, prefill_after),
        },
        "vision_wall_decomposition_s": {
            "graph_delta": graph_delta,
            "input_device_delta": input_delta,
            "output_device_delta": output_delta,
            "wall_residual_delta": residual_delta,
            "reconstructed_wall_delta": (
                graph_delta + input_delta + output_delta + residual_delta
            ),
            "observed_wall_delta": wall_delta,
        },
    }


def render_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}x"


def render_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def print_bucket_top(label: str, rows: dict[str, Any], count: int) -> None:
    ranked = sorted(rows.items(), key=lambda item: item[1]["graph_s"], reverse=True)
    for bucket, row in ranked[:count]:
        print(
            "UNIREC_VISION_PLAN_BUCKET "
            f"lane={label} bucket={bucket} calls={row['calls']} "
            f"graph_s={row['graph_s']:.6f} mean_ms={row['graph_mean_ms']:.3f} "
            f"physical_mpix={row['physical_mpix']:.3f} "
            f"pixel_eff={render_percent(row['pixel_efficiency'])} "
            f"physical_mpix_s={row['physical_mpix_per_graph_s']:.3f}"
        )


def print_report(report: dict[str, Any], top_buckets: int) -> None:
    baseline = report["baseline"]
    candidate = report["candidate"]
    comp = report["comparison"]
    b = baseline["vision"]
    c = candidate["vision"]
    print(
        "UNIREC_VISION_PLAN_AB: PASS "
        f"page_identity_exact={str(comp['page_identity_exact']).lower()} "
        f"page_workload_exact={str(comp['page_workload_exact']).lower()} "
        f"contract_differences={json.dumps(comp['contract_differences'], sort_keys=True)}"
    )
    print(
        "UNIREC_VISION_PLAN_HEADLINE "
        f"baseline_prefill_s={baseline['prefill']['wall_s']:.6f} "
        f"candidate_prefill_s={candidate['prefill']['wall_s']:.6f} "
        f"prefill_speedup={render_ratio(comp['prefill']['speedup'])} "
        f"baseline_vision_s={b['vision_wall_s']:.6f} "
        f"candidate_vision_s={c['vision_wall_s']:.6f} "
        f"vision_saved_s={b['vision_wall_s'] - c['vision_wall_s']:.6f}"
    )
    print(
        "UNIREC_VISION_PLAN_WORK "
        f"baseline_crops={baseline['prefill']['crop_count']} "
        f"candidate_crops={candidate['prefill']['crop_count']} "
        f"baseline_calls={b['calls']} candidate_calls={c['calls']} "
        f"baseline_fallback_calls={baseline['fallback']['calls']} "
        f"candidate_fallback_calls={candidate['fallback']['calls']} "
        f"baseline_slot_eff={render_percent(b['slot_efficiency'])} "
        f"candidate_slot_eff={render_percent(c['slot_efficiency'])} "
        f"baseline_pixel_eff={render_percent(b['pixel_efficiency'])} "
        f"candidate_pixel_eff={render_percent(c['pixel_efficiency'])} "
        f"baseline_physical_mpix={b['physical_mpix']:.3f} "
        f"candidate_physical_mpix={c['physical_mpix']:.3f}"
    )
    print(
        "UNIREC_VISION_PLAN_DEVICE "
        f"baseline_graph_s={b['graph_s']:.6f} "
        f"candidate_graph_s={c['graph_s']:.6f} "
        f"graph_saved_s={b['graph_s'] - c['graph_s']:.6f} "
        f"baseline_compiled_graph_s={baseline['compiled']['graph_s']:.6f} "
        f"baseline_fallback_graph_s={baseline['fallback']['graph_s']:.6f} "
        f"candidate_compiled_graph_s={candidate['compiled']['graph_s']:.6f} "
        f"candidate_fallback_graph_s={candidate['fallback']['graph_s']:.6f} "
        f"baseline_input_s={b['input_device_s']:.6f} "
        f"candidate_input_s={c['input_device_s']:.6f} "
        f"baseline_graph_mean_ms={b['graph_mean_ms']:.3f} "
        f"candidate_graph_mean_ms={c['graph_mean_ms']:.3f} "
        f"baseline_effective_mpix_s={b['effective_mpix_per_graph_s']:.3f} "
        f"candidate_effective_mpix_s={c['effective_mpix_per_graph_s']:.3f} "
        f"baseline_physical_mpix_s={b['physical_mpix_per_graph_s']:.3f} "
        f"candidate_physical_mpix_s={c['physical_mpix_per_graph_s']:.3f}"
    )
    decomposition = comp["vision_wall_decomposition_s"]
    print(
        "UNIREC_VISION_PLAN_RECONCILIATION "
        f"graph_delta_s={decomposition['graph_delta']:.6f} "
        f"input_delta_s={decomposition['input_device_delta']:.6f} "
        f"output_delta_s={decomposition['output_device_delta']:.6f} "
        f"residual_delta_s={decomposition['wall_residual_delta']:.6f} "
        f"observed_vision_delta_s={decomposition['observed_wall_delta']:.6f}"
    )
    print_bucket_top("baseline", baseline["by_bucket"], top_buckets)
    print_bucket_top("candidate", candidate["by_bucket"], top_buckets)
    print(f"OUTPUT_JSON={report['output']}")


def main() -> None:
    args = parse_args()
    if args.top_buckets < 1:
        raise ValueError("top-buckets must be positive")
    baseline = load_lane(args.baseline_dir)
    candidate = load_lane(args.candidate_dir)
    report = {
        "schema": "unirec_vision_plan_ab_v1",
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison(baseline, candidate),
        "output": str(args.output.expanduser().resolve()),
    }
    if report["comparison"]["contract_differences"]:
        raise ValueError(
            "non-vision-plan contract mismatch: "
            + json.dumps(report["comparison"]["contract_differences"], sort_keys=True)
        )
    if not report["comparison"]["page_identity_exact"]:
        raise ValueError("baseline and candidate page identities differ")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report, args.top_buckets)


if __name__ == "__main__":
    main()
