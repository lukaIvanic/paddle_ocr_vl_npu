#!/usr/bin/env python3
"""Combine per-canvas UniRec vision sweeps and compare an optional chip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SHAPES = ((960, 64), (512, 256), (960, 256), (512, 512), (960, 512))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload["schema"] != "unirec_vision_compiled_shape_batch_sweep_v1":
        raise ValueError(f"unexpected sweep schema in {path}")
    if payload["correctness_policy"] != "warning_only":
        raise ValueError(f"unexpected correctness policy in {path}")
    return payload


def _rows_by_bucket(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["rows"]:
            bucket = str(row["bucket"])
            if bucket in result:
                raise ValueError(f"duplicate bucket: {bucket}")
            result[bucket] = row
    return result


def main() -> None:
    args = parse_args()
    payloads = [_load(path) for path in args.input]
    shapes = tuple(tuple(int(v) for v in payload["shape"]) for payload in payloads)
    if set(shapes) != set(EXPECTED_SHAPES):
        raise ValueError(f"shape coverage mismatch: {sorted(shapes)}")
    devices = {str(payload["device_name"]) for payload in payloads}
    if len(devices) != 1:
        raise ValueError(f"multiple target devices: {sorted(devices)}")
    for payload in payloads:
        if payload["focal_depthwise_rewrite"] != "constant_grouped_all":
            raise ValueError("sweep did not use all-45 focal prepacking")
        if payload["weight_format"] != "torchair_internal":
            raise ValueError("sweep did not use TorchAir internal weights")
    rows = _rows_by_bucket(payloads)
    ordered_rows = sorted(
        rows.values(),
        key=lambda row: (int(row["height"]), int(row["width"]), int(row["batch_size"])),
    )
    b1_shape_curve = [row for row in ordered_rows if int(row["batch_size"]) == 1]
    aspect_left = rows["960x256_b1"]
    aspect_right = rows["512x512_b1"]
    aspect_comparison = {
        "left": aspect_left["bucket"],
        "right": aspect_right["bucket"],
        "pixel_ratio_right_over_left": (
            int(aspect_right["physical_pixels"])
            / int(aspect_left["physical_pixels"])
        ),
        "latency_ratio_right_over_left": (
            float(aspect_right["timing"]["median_ms"])
            / float(aspect_left["timing"]["median_ms"])
        ),
        "mpix_ratio_right_over_left": (
            float(aspect_right["timing"]["mpix_per_s"])
            / float(aspect_left["timing"]["mpix_per_s"])
        ),
    }

    cross_chip = None
    if args.reference is not None:
        reference = json.loads(
            args.reference.expanduser().resolve().read_text(encoding="utf-8")
        )
        reference_rows = {
            str(row["bucket"]): row for row in reference["rows"]
        }
        if set(reference_rows) != set(rows):
            raise ValueError("reference bucket coverage differs from target")
        cross_chip = []
        for row in ordered_rows:
            bucket = str(row["bucket"])
            reference_row = reference_rows[bucket]
            target_ms = float(row["timing"]["median_ms"])
            reference_ms = float(reference_row["timing"]["median_ms"])
            cross_chip.append(
                {
                    "bucket": bucket,
                    "target_ms": target_ms,
                    "reference_ms": reference_ms,
                    "target_slowdown": target_ms / reference_ms,
                    "target_mpix_per_s": float(row["timing"]["mpix_per_s"]),
                    "reference_mpix_per_s": float(
                        reference_row["timing"]["mpix_per_s"]
                    ),
                }
            )

    report = {
        "schema": "unirec_vision_shape_batch_sweep_combined_v1",
        "status": "ok",
        "device_name": next(iter(devices)),
        "correctness_policy": "warning_only",
        "warning_count": sum(int(payload["warning_count"]) for payload in payloads),
        "measurement_scope": payloads[0]["measurement_scope"],
        "rows": ordered_rows,
        "b1_shape_curve": b1_shape_curve,
        "aspect_comparison": aspect_comparison,
        "cross_chip": cross_chip,
        "inputs": [str(path.expanduser().resolve()) for path in args.input],
        "reference": (
            str(args.reference.expanduser().resolve())
            if args.reference is not None
            else None
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in ordered_rows:
        print(
            "UNIREC_VISION_SWEEP_COMBINED "
            f"bucket={row['bucket']} median_ms={row['timing']['median_ms']:.6f} "
            f"crops_s={row['timing']['crops_per_s']:.6f} "
            f"mpix_s={row['timing']['mpix_per_s']:.6f} "
            f"batch_efficiency={row['scaling_vs_b1']['batch_efficiency']:.6f}"
        )
    print(
        "UNIREC_VISION_SWEEP_ASPECT "
        f"left={aspect_comparison['left']} right={aspect_comparison['right']} "
        f"pixel_ratio={aspect_comparison['pixel_ratio_right_over_left']:.6f} "
        f"latency_ratio={aspect_comparison['latency_ratio_right_over_left']:.6f} "
        f"mpix_ratio={aspect_comparison['mpix_ratio_right_over_left']:.6f}"
    )
    if cross_chip is not None:
        for row in cross_chip:
            print(
                "UNIREC_VISION_SWEEP_CROSS_CHIP "
                f"bucket={row['bucket']} target_ms={row['target_ms']:.6f} "
                f"reference_ms={row['reference_ms']:.6f} "
                f"target_slowdown={row['target_slowdown']:.6f}x"
            )
    print(f"UNIREC_VISION_SWEEP_OUTPUT={output}")


if __name__ == "__main__":
    main()
