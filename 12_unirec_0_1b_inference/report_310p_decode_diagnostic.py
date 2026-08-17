#!/usr/bin/env python3
"""Summarize clean and traced production decode replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def lane(payload: dict[str, Any]) -> dict[str, Any]:
    decode = payload["decode"]
    warmup = decode["production_graph_warmup"]
    return {
        "selected_crops": payload["workload"]["selected_crops"],
        "decode_iterations": decode["decode_iterations"],
        "decode_graph_s": decode["decode_s"],
        "decode_wall_excluding_warmup_s": payload[
            "decode_wall_excluding_warmup_s"
        ],
        "raw_tok_s": decode["raw_decode_tokens_per_s"],
        "effective_tok_s": decode["effective_decode_tokens_per_s"],
        "slot_efficiency": payload["slot_efficiency"],
        "warmup_pass_s": warmup["pass_wall_s"],
        "warmup_warnings": warmup["warnings"],
        "generated_length": payload["workload"]["generated_length"],
        "timing_detail": decode["timing_detail"],
    }


def main() -> None:
    args = parse_args()
    clean = json.loads(args.clean.read_text())
    trace = json.loads(args.trace.read_text())
    clean_lane = lane(clean)
    trace_lane = lane(trace)
    report = {
        "schema": "unirec_310p_production_decode_diagnostic_v1",
        "status": "ok",
        "clean": clean_lane,
        "trace": trace_lane,
        "trace_overhead": {
            "raw_tok_s_ratio_trace_over_clean": (
                trace_lane["raw_tok_s"] / clean_lane["raw_tok_s"]
            ),
            "decode_graph_s_ratio_trace_over_clean": (
                trace_lane["decode_graph_s"] / clean_lane["decode_graph_s"]
            ),
        },
        "step_trace": trace["step_trace"],
        "artifact": {
            "directory": clean["config"]["artifact_dir"],
            "source_length": clean["workload"]["source_length"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "UNIREC_310P_DECODE_DIAGNOSTIC: PASS "
        f"crops={clean_lane['selected_crops']} "
        f"iterations={clean_lane['decode_iterations']} "
        f"clean_graph_s={clean_lane['decode_graph_s']:.6f} "
        f"clean_raw_tok_s={clean_lane['raw_tok_s']:.3f} "
        f"clean_effective_tok_s={clean_lane['effective_tok_s']:.3f} "
        f"clean_slot_eff={clean_lane['slot_efficiency']:.6f} "
        f"trace_raw_tok_s={trace_lane['raw_tok_s']:.3f} "
        f"trace_overhead_ratio={report['trace_overhead']['raw_tok_s_ratio_trace_over_clean']:.6f} "
        f"output={args.output.resolve()}",
        flush=True,
    )
    print("CACHE_POSITION_SPLIT")
    print(
        json.dumps(
            report["step_trace"].get("by_cache_position_max", {}),
            separators=(",", ":"),
        )
    )
    print("ACTIVE_COUNT_SPLIT")
    print(
        json.dumps(
            report["step_trace"].get("by_active_count", {}),
            separators=(",", ":"),
        )
    )
    print("SLOWEST_DECODE_STEPS")
    print(
        json.dumps(
            report["step_trace"].get("slowest_decode_steps", []),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
