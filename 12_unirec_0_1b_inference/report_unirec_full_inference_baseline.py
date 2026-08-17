#!/usr/bin/env python3
"""Validate and report the full canonical UniRec inference baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--process-wall", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(args.run_summary.read_text())
    history = json.loads(args.history.read_text())
    process_wall_s = float(args.process_wall.read_text())
    assert run["status"] == "ok"
    assert (run["page_count"], run["workers"]) == (1651, 4)
    assert run["recognition_preprocess_threads"] == 8
    assert run["vision_bucket_preset"] == "production_v1"
    assert run["vision_focal_depthwise_rewrite"] == "native"
    assert run["vision_weight_format"] == "native"
    assert run["layout_execution"] == "eager"
    assert run["layout_dtype"] == "float32"
    assert run["layout_batch_size"] == 2
    assert run["decode_batch_size"] == 128
    assert run["cross_cache_length"] == 1320
    assert run["self_cache_length"] == 2048
    assert len(history["runs"]) == 1
    timing = run["timing_s"]
    throughput = run["throughput"]
    decode = run["decode"]
    slot_efficiency = (
        decode["effective_decode_tokens"] / decode["raw_decode_token_slots"]
    )
    report: dict[str, Any] = {
        "schema": "unirec_310p_full1651_inference_baseline_v1",
        "status": "ok",
        "page_count": run["page_count"],
        "crop_count": run["crop_count"],
        "process_wall_s": process_wall_s,
        "process_pages_per_s": run["page_count"] / process_wall_s,
        "timing_s": timing,
        "throughput": throughput,
        "decode_iterations": decode["decode_iterations"],
        "slot_efficiency": slot_efficiency,
        "recognition_trace": history["runs"][0].get("recognition_trace"),
        "run_summary": str(args.run_summary.resolve()),
    }
    output = args.output.expanduser().resolve()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        "UNIREC_310P_FULL1651_BASELINE: PASS "
        f"pages=1651 crops={run['crop_count']} "
        f"process_wall_s={process_wall_s:.6f} "
        f"process_pg_s={run['page_count'] / process_wall_s:.6f} "
        f"lifecycle_s={timing['lifecycle']:.6f} "
        f"prefill_s={timing['prefill_phase']:.6f} "
        f"decode_ingress_s={timing['decode_inference_including_ingress']:.6f} "
        f"decode_graph_s={timing['decode_graph']:.6f} "
        f"sequential_core_s={timing['sequential_core_prefill_plus_decode']:.6f} "
        f"pipeline_pg_s={throughput['sequential_core_pages_per_s']:.6f} "
        f"iterations={decode['decode_iterations']} "
        f"raw_tok_s={throughput['decode_raw_token_slots_per_s']:.3f} "
        f"effective_tok_s={throughput['decode_effective_tokens_per_s']:.3f} "
        f"slot_eff={slot_efficiency:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
