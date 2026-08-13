#!/usr/bin/env python3
"""Audit focal-depthwise weight TransData rows in one vision profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CHANNELS = (96, 192, 384, 768)
KERNELS = (3, 5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--bucket", default="960x64_b16")
    parser.add_argument("--require-all-45", action="store_true")
    parser.add_argument("--require-exact-reference", action="store_true")
    return parser.parse_args()


def _shapes(raw: str) -> set[str]:
    return {
        item.strip().strip('"')
        for item in raw.split(";")
        if item.strip().strip('"')
    }


def _is_focal_weight_repack(row: dict[str, str]) -> bool:
    input_shapes = _shapes(row["Input Shapes"])
    output_shapes = _shapes(row["Output Shapes"])
    input_formats = set(row["Input Formats"].split(";"))
    output_formats = set(row["Output Formats"].split(";"))
    for channels in CHANNELS:
        for kernel in KERNELS:
            area = kernel * kernel
            native = f"{channels},1,{kernel},{kernel}"
            base_fz = f"{area},{channels // 16},16,16"
            grouped_fz = f"{channels * area // 16},1,16,16"
            native_to_fz = (
                native in input_shapes
                and base_fz in output_shapes
                and bool(input_formats & {"NCHW", "ND"})
                and "FRACTAL_Z" in output_formats
            )
            fz_to_grouped = (
                base_fz in input_shapes
                and grouped_fz in output_shapes
                and "FRACTAL_Z" in input_formats
                and any(value.startswith("FRACTAL_Z") for value in output_formats)
            )
            if native_to_fz or fz_to_grouped:
                return True
    return False


def _lane(summary: dict[str, Any], bucket: str) -> dict[str, Any]:
    matches = [
        lane
        for lane in summary["lanes"]
        if lane["name"].startswith(f"vision_{bucket}_fp16")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {bucket} lane, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    lane = _lane(summary, args.bucket)
    runs = lane["parsed_profile"]["summary"]["runs"]
    if len(runs) != 1:
        raise RuntimeError(f"expected one parsed profile, found {len(runs)}")
    kernel_csv = Path(runs[0]["files"]["kernel_details"])
    with kernel_csv.open(newline="", encoding="utf-8-sig") as handle:
        transdata = [
            row for row in csv.DictReader(handle) if row["Type"] == "TransData"
        ]
    focal_rows = [row for row in transdata if _is_focal_weight_repack(row)]

    rewrite = lane.get("focal_depthwise_rewrite_summary") or {}
    target_count = int(rewrite.get("target_count", rewrite.get("rewritten_count", 0)))
    exact = lane.get("compiled_reference_validation")
    if args.require_all_45 and target_count != 45:
        raise RuntimeError(f"expected 45 rewritten weights, found {target_count}")
    if args.require_exact_reference and (
        exact is None
        or exact.get("exact") is not True
        or float(exact.get("max_abs", 1.0)) != 0.0
        or float(exact.get("mean_abs", 1.0)) != 0.0
    ):
        raise RuntimeError(f"compiled reference is not bit-exact: {exact}")
    if focal_rows:
        for row in focal_rows:
            print(
                "UNIREC_FOCAL_WEIGHT_TRANSDATA_ROW "
                f"name={row['Name']} input={row['Input Shapes']} "
                f"output={row['Output Shapes']} duration_us={row['Duration(us)']}"
            )
        raise RuntimeError(
            f"found {len(focal_rows)} focal-weight TransData operations"
        )

    total_us = sum(float(row["Duration(us)"]) for row in transdata)
    median_ms = float(lane["control_after"]["device_event"]["median_ms"])
    rewrite_name = str(summary["config"]["vision_depthwise_rewrite"])
    print(
        "UNIREC_FOCAL_WEIGHT_TRANSDATA_AUDIT: PASS "
        f"bucket={args.bucket} rewrite={rewrite_name} "
        f"rewritten={target_count} focal_weight_transdata=0 "
        f"total_transdata={len(transdata)} total_transdata_ms={total_us / 1000:.6f} "
        f"device_median_ms={median_ms:.6f} "
        f"exact_reference={str(bool(exact and exact.get('exact'))).lower()}"
    )


if __name__ == "__main__":
    main()
