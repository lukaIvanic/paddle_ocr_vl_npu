#!/usr/bin/env python3
"""Compare compiled layout internal-weight and constant-grouped profiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LOGICAL_TARGET_SHAPES = {
    '"128,1,3,3"',
    '"512,1,3,3"',
    '"1024,1,3,3"',
    '"192,1,5,5"',
    '"384,1,5,5"',
}

GROUPED_TARGET_SIGNATURES = {
    ('"9,8,16,16"', "FRACTAL_Z", '"72,1,16,16"', "FRACTAL_Z:128"),
    ('"9,32,16,16"', "FRACTAL_Z", '"288,1,16,16"', "FRACTAL_Z:512"),
    ('"9,64,16,16"', "FRACTAL_Z", '"576,1,16,16"', "FRACTAL_Z:1024"),
    ('"25,12,16,16"', "FRACTAL_Z", '"300,1,16,16"', "FRACTAL_Z:192"),
    ('"25,24,16,16"', "FRACTAL_Z", '"600,1,16,16"', "FRACTAL_Z:384"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-forward", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--candidate-forward", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--baseline-kernel-csv", type=Path)
    parser.add_argument("--candidate-kernel-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _kernel_csv_path(profile: dict[str, Any]) -> Path:
    lane = profile["lanes"][0]
    run = lane["parsed_profile"]["summary"]["runs"][0]
    return Path(run["files"]["kernel_details"])


def _load_kernel_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _duration_us(row: dict[str, str]) -> float:
    return float((row.get("Duration(us)") or "0").strip() or 0.0)


def _signature(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["Input Shapes"],
        row["Input Formats"],
        row["Output Shapes"],
        row["Output Formats"],
    )


def _profile_fields(
    profile: dict[str, Any],
    *,
    kernel_csv: Path | None,
) -> dict[str, Any]:
    lane = profile["lanes"][0]
    path = _kernel_csv_path(profile) if kernel_csv is None else kernel_csv
    rows = _load_kernel_rows(path)
    transdata = [row for row in rows if row["Type"] == "TransData"]
    logical_targets = [
        row
        for row in transdata
        if row["Input Formats"] == "NCHW"
        and row["Output Formats"] == "FRACTAL_Z"
        and row["Input Shapes"] in LOGICAL_TARGET_SHAPES
    ]
    grouped_targets = [
        row for row in transdata if _signature(row) in GROUPED_TARGET_SIGNATURES
    ]
    signature_counts = Counter(_signature(row) for row in transdata)
    top_transdata = sorted(
        (
            {
                "count": count,
                "duration_ms": sum(
                    _duration_us(row)
                    for row in transdata
                    if _signature(row) == signature
                )
                / 1000.0,
                "input_shapes": signature[0],
                "input_formats": signature[1],
                "output_shapes": signature[2],
                "output_formats": signature[3],
            }
            for signature, count in signature_counts.items()
        ),
        key=lambda row: row["duration_ms"],
        reverse=True,
    )[:15]
    return {
        "lane_name": lane["name"],
        "clean_mean_ms": float(
            lane["control_before"]["device_event"]["mean_ms"]
        ),
        "clean_median_ms": float(
            lane["control_before"]["device_event"]["median_ms"]
        ),
        "kernel_count": len(rows),
        "transdata_count": len(transdata),
        "transdata_ms": sum(_duration_us(row) for row in transdata) / 1000.0,
        "logical_target_count": len(logical_targets),
        "logical_target_ms": (
            sum(_duration_us(row) for row in logical_targets) / 1000.0
        ),
        "grouped_target_count": len(grouped_targets),
        "grouped_target_ms": (
            sum(_duration_us(row) for row in grouped_targets) / 1000.0
        ),
        "kernel_csv": str(path),
        "top_transdata": top_transdata,
    }


def _paired_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = first["coordinate"]
    bx1, by1, bx2, by2 = second["coordinate"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 1.0


def _output_fields(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_page = baseline["pages"][0]
    candidate_page = candidate["pages"][0]
    baseline_boxes = baseline_page["result"]["boxes"]
    candidate_boxes = candidate_page["result"]["boxes"]
    same_count = len(baseline_boxes) == len(candidate_boxes)
    pairs = list(zip(baseline_boxes, candidate_boxes)) if same_count else []
    ious = [_paired_iou(first, second) for first, second in pairs]
    return {
        "baseline_box_count": len(baseline_boxes),
        "candidate_box_count": len(candidate_boxes),
        "same_box_count": same_count,
        "class_label_sequence_match": same_count
        and all(
            (first["cls_id"], first["label"])
            == (second["cls_id"], second["label"])
            for first, second in pairs
        ),
        "reading_order_changed_count": sum(
            first["custom_value"] != second["custom_value"]
            for first, second in pairs
        ),
        "mean_paired_iou": sum(ious) / len(ious) if ious else None,
        "minimum_paired_iou": min(ious) if ious else None,
        "coordinate_max_abs_px": max(
            (
                abs(first_value - second_value)
                for first, second in pairs
                for first_value, second_value in zip(
                    first["coordinate"], second["coordinate"]
                )
            ),
            default=None,
        ),
        "score_max_abs": max(
            (
                abs(first["score"] - second["score"])
                for first, second in pairs
            ),
            default=None,
        ),
        "digest_match": (
            baseline_page["result_digest"] == candidate_page["result_digest"]
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_forward = _load_json(args.baseline_forward)
    candidate_forward = _load_json(args.candidate_forward)
    baseline = _profile_fields(
        _load_json(args.baseline_profile),
        kernel_csv=args.baseline_kernel_csv,
    )
    candidate = _profile_fields(
        _load_json(args.candidate_profile),
        kernel_csv=args.candidate_kernel_csv,
    )
    output = _output_fields(baseline_forward, candidate_forward)
    rewrite = candidate_forward["config"]["depthwise_rewrite_summary"]
    inventory_ok = (
        rewrite["requested"] == "constant_grouped"
        and rewrite["target_count"] == 27
        and rewrite["rewritten_count"] == 27
        and all(
            row["weight_binding"] == "frozen_prepacked_fractal_z_grouped"
            for row in rewrite["modules"]
        )
    )
    gates = {
        "baseline_has_27_logical_targets": baseline["logical_target_count"] == 27,
        "baseline_has_27_grouped_targets": baseline["grouped_target_count"] == 27,
        "candidate_has_zero_logical_targets": candidate["logical_target_count"] == 0,
        "candidate_has_zero_grouped_targets": candidate["grouped_target_count"] == 0,
        "candidate_rewrite_inventory_exact": inventory_ok,
        "same_box_count": output["same_box_count"],
        "class_label_sequence_match": output["class_label_sequence_match"],
        "mean_paired_iou_at_least_0_999": (
            output["mean_paired_iou"] is not None
            and output["mean_paired_iou"] >= 0.999
        ),
        "reading_order_changed_at_most_1": (
            output["reading_order_changed_count"] <= 1
        ),
    }
    passed = all(gates.values())
    summary = {
        "format": "unirec_layout_constant_grouped_profile_comparison_v1",
        "passed": passed,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": baseline["clean_mean_ms"] / candidate["clean_mean_ms"],
        "output_gate": output,
        "rewrite_inventory": rewrite,
        "gates": gates,
        "artifacts": {
            "baseline_forward": str(args.baseline_forward),
            "baseline_profile": str(args.baseline_profile),
            "candidate_forward": str(args.candidate_forward),
            "candidate_profile": str(args.candidate_profile),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if passed else "FAIL"
    print(
        f"UNIREC_LAYOUT_CONSTANT_GROUPED: {status} "
        f"baseline_ms={baseline['clean_mean_ms']:.6f} "
        f"candidate_ms={candidate['clean_mean_ms']:.6f} "
        f"speedup={summary['speedup']:.6f} "
        f"baseline_td={baseline['transdata_count']}/{baseline['transdata_ms']:.6f}ms "
        f"candidate_td={candidate['transdata_count']}/{candidate['transdata_ms']:.6f}ms "
        f"logical_targets={baseline['logical_target_count']}->{candidate['logical_target_count']} "
        f"grouped_targets={baseline['grouped_target_count']}->{candidate['grouped_target_count']} "
        f"boxes={output['baseline_box_count']}/{output['candidate_box_count']} "
        f"class_label_match={str(output['class_label_sequence_match']).lower()} "
        f"order_changed={output['reading_order_changed_count']} "
        f"mean_iou={output['mean_paired_iou']} "
        f"coord_max={output['coordinate_max_abs_px']} "
        f"score_max={output['score_max_abs']} "
        f"digest_match={str(output['digest_match']).lower()}"
    )
    for rank, row in enumerate(candidate["top_transdata"], start=1):
        print(
            f"UNIREC_LAYOUT_CONSTANT_GROUPED_TD rank={rank} "
            f"count={row['count']} duration_ms={row['duration_ms']:.6f} "
            f"shape={row['input_shapes']}->{row['output_shapes']} "
            f"format={row['input_formats']}->{row['output_formats']}"
        )
    print(f"UNIREC_LAYOUT_CONSTANT_GROUPED_OUTPUT {args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
