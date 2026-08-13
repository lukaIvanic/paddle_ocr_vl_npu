#!/usr/bin/env python3
"""Compare matched 310P and 910B2 stock-eager vision B1 profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu310", type=Path, required=True)
    parser.add_argument("--npu910", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topn", type=int, default=20)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _kernels(report: dict[str, Any]) -> dict[str, Any]:
    runs = report["parsed_profile"]["summary"]["runs"]
    if len(runs) != 1 or "kernel_details" not in runs[0]:
        raise ValueError("expected exactly one parsed kernel profile")
    return runs[0]["kernel_details"]


def _mapping(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in rows}


def _compare_rows(
    rows310: list[dict[str, Any]],
    rows910: list[dict[str, Any]],
    *,
    topn: int,
) -> list[dict[str, Any]]:
    left = _mapping(rows310)
    right = _mapping(rows910)
    compared = []
    for name in left.keys() | right.keys():
        row310 = left.get(name, {})
        row910 = right.get(name, {})
        time310 = float(row310.get("duration_us", 0.0))
        time910 = float(row910.get("duration_us", 0.0))
        count310 = int(row310.get("count", 0))
        count910 = int(row910.get("count", 0))
        compared.append(
            {
                "name": name,
                "npu310_count": count310,
                "npu910_count": count910,
                "npu310_duration_us": time310,
                "npu910_duration_us": time910,
                "added_310_us": time310 - time910,
                "duration_ratio_310_over_910": (
                    time310 / time910 if time910 > 0.0 else None
                ),
                "per_call_ratio_310_over_910": (
                    (time310 / count310) / (time910 / count910)
                    if count310 > 0 and count910 > 0 and time910 > 0.0
                    else None
                ),
            }
        )
    return sorted(
        compared,
        key=lambda row: float(row["added_310_us"]),
        reverse=True,
    )[:topn]


def main() -> None:
    args = parse_args()
    npu310 = _load(args.npu310)
    npu910 = _load(args.npu910)
    for report, chip in ((npu310, "310P"), (npu910, "910B2")):
        if report.get("status") != "ok":
            raise ValueError(f"{chip} profile status is not ok")
        if report.get("input_shape") != [1, 3, 64, 960]:
            raise ValueError(f"{chip} input shape is not matched")
        if report.get("execution") != "stock_eager_model_forward_encoder":
            raise ValueError(f"{chip} execution is not stock eager")

    kernels310 = _kernels(npu310)
    kernels910 = _kernels(npu910)
    control310 = float(npu310["control_before"]["device_event"]["median_ms"])
    control910 = float(npu910["control_before"]["device_event"]["median_ms"])
    report = {
        "status": "ok",
        "npu310_device": npu310["device_name"],
        "npu910_device": npu910["device_name"],
        "control_before_median_ms": {
            "npu310": control310,
            "npu910": control910,
            "ratio_310_over_910": control310 / control910,
        },
        "profiler_overhead": {
            "npu310": npu310["profiler_overhead"],
            "npu910": npu910["profiler_overhead"],
        },
        "kernel_totals": {
            "npu310_count": int(kernels310["row_count"]),
            "npu910_count": int(kernels910["row_count"]),
            "npu310_duration_us": float(kernels310["total_duration_us"]),
            "npu910_duration_us": float(kernels910["total_duration_us"]),
            "duration_ratio_310_over_910": (
                float(kernels310["total_duration_us"])
                / float(kernels910["total_duration_us"])
                if float(kernels910["total_duration_us"]) > 0.0
                else None
            ),
            "npu310_cube_pct": float(
                kernels310["weighted_cube_utilization_pct"]
            ),
            "npu910_cube_pct": float(
                kernels910["weighted_cube_utilization_pct"]
            ),
        },
        "kernel_type_gaps": _compare_rows(
            kernels310["top_kernel_types"],
            kernels910["top_kernel_types"],
            topn=args.topn,
        ),
        "exact_shape_gaps": _compare_rows(
            kernels310["top_shape_signatures"],
            kernels910["top_shape_signatures"],
            topn=args.topn,
        ),
        "matmul_shape_gaps": _compare_rows(
            kernels310["top_matmul_shape_signatures"],
            kernels910["top_matmul_shape_signatures"],
            topn=args.topn,
        ),
        "transdata_shape_gaps": _compare_rows(
            kernels310["top_transdata_shape_signatures"],
            kernels910["top_transdata_shape_signatures"],
            topn=args.topn,
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_STOCK_EAGER_VISION_PROFILE_COMPARISON "
        f"control_ratio={report['control_before_median_ms']['ratio_310_over_910']:.3f} "
        f"kernel_ratio={report['kernel_totals']['duration_ratio_310_over_910']:.3f} "
        f"kernel_counts={report['kernel_totals']['npu310_count']}/{report['kernel_totals']['npu910_count']} "
        f"cube_pct={report['kernel_totals']['npu310_cube_pct']:.2f}/{report['kernel_totals']['npu910_cube_pct']:.2f}",
        flush=True,
    )
    for row in report["kernel_type_gaps"][:15]:
        ratio = row["duration_ratio_310_over_910"]
        ratio_text = f"{ratio:.3f}" if ratio is not None else "nan"
        print(
            "UNIREC_STOCK_EAGER_VISION_KERNEL_GAP "
            f"name={json.dumps(row['name'])} "
            f"counts={row['npu310_count']}/{row['npu910_count']} "
            f"times_ms={row['npu310_duration_us'] / 1000.0:.3f}/{row['npu910_duration_us'] / 1000.0:.3f} "
            f"ratio={ratio_text} added_ms={row['added_310_us'] / 1000.0:.3f}",
            flush=True,
        )
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
