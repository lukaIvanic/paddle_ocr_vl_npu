#!/usr/bin/env python3
"""Print a compact terminal report from a UniRec prefill trace summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--top-shapes", type=int, default=12)
    return parser.parse_args()


def _timing_line(name: str, row: dict[str, Any]) -> str:
    if not row or not row.get("count"):
        return f"UNIREC_PREFILL_TRACE_STAGE name={name} count=0"
    return (
        f"UNIREC_PREFILL_TRACE_STAGE name={name} "
        f"count={row['count']} sum_s={row['sum_s']:.6f} "
        f"mean_ms={row['mean_ms']:.3f} p50_ms={row['p50_ms']:.3f} "
        f"p90_ms={row['p90_ms']:.3f} p95_ms={row['p95_ms']:.3f} "
        f"p99_ms={row['p99_ms']:.3f} max_ms={row['max_ms']:.3f}"
    )


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    if report.get("schema") != "unirec_production_prefill_trace_v1":
        raise ValueError("unsupported prefill trace schema")
    print(
        "UNIREC_PREFILL_TRACE PASS "
        f"pages={report['page_count']} events={report['event_count']} "
        f"workers={report['config']['workers']} "
        "threads="
        f"{report['config']['recognition_preprocess_threads']}"
    )
    stages = report["stage_distributions"]
    preferred = (
        "layout_batch_call.wall_s",
        "layout_batch_call.stage_s.processor_preprocess_s",
        "layout_batch_call.stage_s.inputs_h2d_s",
        "layout_batch_call.stage_s.model_forward_s",
        "layout_batch_call.stage_s.outputs_d2h_s",
        "layout_batch_call.stage_s.box_decode_s",
        "recognition_crop_preprocess.wall_s",
        "recognition_crop_preprocess.stage_s.recognition_processor_resize_s",
        "vision_bucket_call.host_stage_s.host_canvas_pack_s",
        "vision_bucket_call.device_stage_s.input_h2d_normalize_s",
        "vision_bucket_call.device_stage_s.graph_s",
        "vision_bucket_call.device_stage_s.output_compact_s",
        "vision_fallback_call.device_stage_s.graph_s",
        "text_prefill_pack.wall_s",
        "cross_kv_d2h.wall_s",
        "page_shared_pack.wall_s",
        "coordinator_ipc_delivery.wall_s",
        "worker_page_group.wall_s",
    )
    for name in preferred:
        if name in stages:
            print(_timing_line(name, stages[name]))
    for name in sorted(stages):
        if name.startswith("text_prefill_pack.device_stage_s.compiled_packed_"):
            print(_timing_line(name, stages[name]))
    for name, histogram in report["shape_histograms"].items():
        top = list(histogram.items())[: args.top_shapes]
        rendered = ",".join(f"{shape}:{count}" for shape, count in top)
        print(
            "UNIREC_PREFILL_TRACE_SHAPES "
            f"name={name} distinct={len(histogram)} top={rendered}"
        )
    print(
        "UNIREC_PREFILL_TRACE_ARTIFACTS "
        + " ".join(
            f"{name}={path}" for name, path in report["artifacts"].items()
        )
    )


if __name__ == "__main__":
    main()
