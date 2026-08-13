#!/usr/bin/env python3
"""Compare native and grouped-FZ compiled UniRec vision bucket suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _vision_lanes(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for lane in report["lanes"]:
        if not lane["name"].startswith("vision_"):
            continue
        contract = lane["input_contract"]
        shape = contract["pixel_values"]
        key = f"{shape[3]}x{shape[2]}_b{shape[0]}"
        result[key] = lane
    return result


def main() -> None:
    args = parse_args()
    native = json.loads(args.native.expanduser().resolve().read_text())
    optimized = json.loads(args.optimized.expanduser().resolve().read_text())
    native_lanes = _vision_lanes(native)
    optimized_lanes = _vision_lanes(optimized)
    if native_lanes.keys() != optimized_lanes.keys():
        raise RuntimeError(
            f"bucket mismatch: {native_lanes.keys()} != {optimized_lanes.keys()}"
        )

    rows = []
    for key in native_lanes:
        baseline = native_lanes[key]
        candidate = optimized_lanes[key]
        exact = candidate["compiled_reference_validation"]
        if not exact["exact"]:
            raise RuntimeError(f"compiled parity failed for {key}: {exact}")
        baseline_ms = float(baseline["steady_device_event_mean_ms"])
        optimized_ms = float(candidate["steady_device_event_mean_ms"])
        rows.append(
            {
                "bucket": key,
                "first128_calls": int(baseline["first128_calls"]),
                "baseline_ms": baseline_ms,
                "optimized_ms": optimized_ms,
                "saved_ms": baseline_ms - optimized_ms,
                "speedup": baseline_ms / optimized_ms,
                "compiled_exact": True,
                "max_abs": float(exact["max_abs"]),
                "mean_abs": float(exact["mean_abs"]),
            }
        )

    baseline_weighted_s = sum(
        row["baseline_ms"] * row["first128_calls"] / 1000.0 for row in rows
    )
    optimized_weighted_s = sum(
        row["optimized_ms"] * row["first128_calls"] / 1000.0 for row in rows
    )
    report = {
        "status": "ok",
        "baseline": "native_compiled_production_buckets",
        "optimized": "constant_grouped_22_compiled_production_buckets",
        "rows": rows,
        "first128_weighted": {
            "baseline_s": baseline_weighted_s,
            "optimized_s": optimized_weighted_s,
            "saved_s": baseline_weighted_s - optimized_weighted_s,
            "speedup": baseline_weighted_s / optimized_weighted_s,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            "UNIREC_VISION_BUCKET_COMPILED_AB "
            f"bucket={row['bucket']} calls={row['first128_calls']} "
            f"baseline_ms={row['baseline_ms']:.6f} "
            f"optimized_ms={row['optimized_ms']:.6f} "
            f"saved_ms={row['saved_ms']:.6f} "
            f"speedup={row['speedup']:.3f}x exact={row['compiled_exact']}",
            flush=True,
        )
    weighted = report["first128_weighted"]
    print(
        "UNIREC_VISION_BUCKET_COMPILED_AB_WEIGHTED "
        f"baseline_s={weighted['baseline_s']:.6f} "
        f"optimized_s={weighted['optimized_s']:.6f} "
        f"saved_s={weighted['saved_s']:.6f} "
        f"speedup={weighted['speedup']:.3f}x",
        flush=True,
    )
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
