#!/usr/bin/env python3
"""Validate and print the optimized-only 310P full-prefill result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--historical-prefill-s", type=float, default=350.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["validation"]["passed"] is True
    assert (summary["offset"], summary["limit"], summary["workers"]) == (
        0,
        1651,
        8,
    )
    assert summary["artifact_storage"] == "discard"
    assert summary["artifact"]["page_count"] == 1651
    assert summary["vision_focal_depthwise_rewrite"] == "constant_grouped"
    assert summary["vision_weight_format"] == "native"

    stages = summary["worker_summary"]["stage_s"]
    wall = float(summary["producer_wall_s"])
    print(
        "UNIREC_310P_GROUPED_FZ_FULL1651_OPTIMIZED "
        f"status=PASS producer_wall_s={wall:.3f} "
        f"pages_per_s={summary['throughput']['pages_per_s']:.3f} "
        f"crops_per_s={summary['throughput']['crops_per_s']:.3f} "
        f"pages={summary['artifact']['page_count']} "
        f"crops={summary['artifact']['crop_count']} "
        f"rejected={summary['artifact']['rejected_crop_count']} "
        f"tokens={summary['artifact']['real_source_tokens']} "
        f"fallback={summary['worker_summary']['vision_batching']['fallback_rows']}"
    )
    print(
        "UNIREC_310P_GROUPED_FZ_FULL1651_STAGES "
        f"layout_sum_s={stages['worker_detector_call_sum_s']:.3f} "
        f"cpu_prepare_sum_s="
        f"{stages['worker_recognition_input_prepare_sum_s']:.3f} "
        f"prefill_sum_s={stages['worker_recognition_prefill_sum_s']:.3f} "
        f"d2h_sum_s="
        f"{stages['worker_recognition_prefill_cache_d2h_sum_s']:.3f} "
        f"worker_max_s="
        f"{max(summary['worker_summary']['worker_busy_s']):.3f} "
        f"setup_s={summary['setup_s']:.3f} "
        f"warmup_s={summary['warmup']['wall_s']:.3f}"
    )
    print(
        "UNIREC_310P_GROUPED_FZ_FULL1651_HISTORICAL_CONTEXT "
        f"historical_approx_s={args.historical_prefill_s:.3f} "
        f"optimized_s={wall:.3f} "
        f"approx_ratio={args.historical_prefill_s / wall:.3f}x "
        "controlled_ab=false"
    )
    print(f"SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
