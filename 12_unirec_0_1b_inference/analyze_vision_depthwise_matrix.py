#!/usr/bin/env python3
"""Summarize native and exact-rewrite UniRec vision graph profiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BUCKET = "960x64_b16"
TARGET_INPUT = re.compile(r'^"(?:25|48|49|64),(?:24|48),16,16"$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=JSON",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _lane(summary: dict[str, Any]) -> dict[str, Any]:
    matches = [
        lane
        for lane in summary["lanes"]
        if lane["name"].startswith(f"vision_{BUCKET}_fp16")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {BUCKET} lane, got {len(matches)}")
    return matches[0]


def _kernel_details(lane: dict[str, Any]) -> dict[str, Any]:
    runs = lane["parsed_profile"]["summary"]["runs"]
    if len(runs) != 1:
        raise ValueError(f"expected one profiler run, got {len(runs)}")
    return runs[0]["kernel_details"]


def _summarize(name: str, path: Path) -> dict[str, Any]:
    lane = _lane(_load(path))
    kernels = _kernel_details(lane)
    types = {row["name"]: row for row in kernels["top_kernel_types"]}
    target_rows = [
        row
        for row in kernels["top_transdata_shape_signatures"]
        if row.get("input_format_samples") == ["FRACTAL_Z"]
        and any(TARGET_INPUT.fullmatch(shape) for shape in row["input_shape_samples"])
        and any(
            output.startswith("FRACTAL_Z:")
            for output in row["output_format_samples"]
        )
    ]
    validation = lane.get("rewrite_validation")
    return {
        "name": name,
        "path": str(path.expanduser().resolve()),
        "device_ms": float(lane["steady_device_event_mean_ms"]),
        "kernel_rows": int(kernels["row_count"]),
        "profile_total_ms": float(kernels["total_duration_us"]) / 1000.0,
        "transdata_ms": float(types.get("TransData", {}).get("duration_us", 0.0))
        / 1000.0,
        "transdata_count": int(types.get("TransData", {}).get("count", 0)),
        "conv2d_ms": float(types.get("Conv2D", {}).get("duration_us", 0.0))
        / 1000.0,
        "pad_ms": float(types.get("PadV3", {}).get("duration_us", 0.0))
        / 1000.0,
        "target_repack_ms": sum(
            float(row["duration_us"]) for row in target_rows
        )
        / 1000.0,
        "target_repack_count": sum(int(row["count"]) for row in target_rows),
        "target_repack_signatures": [
            {
                "name": row["name"],
                "count": int(row["count"]),
                "duration_ms": float(row["duration_us"]) / 1000.0,
            }
            for row in target_rows
        ],
        "validation": validation,
        "rewrite_summary": lane.get("focal_depthwise_rewrite_summary"),
        "weight_format_summary": lane.get("weight_format_summary"),
    }


def main() -> None:
    args = parse_args()
    rows = [_summarize("native", args.native)]
    for item in args.variant:
        if "=" not in item:
            raise ValueError(f"--variant must be NAME=JSON, got {item!r}")
        name, path = item.split("=", 1)
        rows.append(_summarize(name, Path(path)))
    native_ms = rows[0]["device_ms"]
    for row in rows:
        row["speedup_vs_native"] = native_ms / row["device_ms"]
    report = {
        "format": "unirec_vision_depthwise_matrix_v1",
        "bucket": BUCKET,
        "rows": rows,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for row in rows:
        validation = row["validation"]
        parity = "native"
        max_abs = 0.0
        mean_abs = 0.0
        if validation is not None:
            parity = str(validation["allclose_atol_5e_2_rtol_5e_2"]).lower()
            max_abs = float(validation["max_abs"])
            mean_abs = float(validation["mean_abs"])
        print(
            "UNIREC_VISION_DEPTHWISE "
            f"lane={row['name']} device_ms={row['device_ms']:.6f} "
            f"speedup={row['speedup_vs_native']:.4f}x "
            f"parity={parity} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
            f"TransData={row['transdata_ms']:.3f}ms/{row['transdata_count']} "
            f"target_repack={row['target_repack_ms']:.3f}ms/"
            f"{row['target_repack_count']} "
            f"Conv2D={row['conv2d_ms']:.3f}ms PadV3={row['pad_ms']:.3f}ms "
            f"kernels={row['kernel_rows']}",
            flush=True,
        )
    if args.output is not None:
        print(f"UNIREC_VISION_DEPTHWISE_OUTPUT {output}", flush=True)


if __name__ == "__main__":
    main()
