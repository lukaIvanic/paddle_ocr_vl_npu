#!/usr/bin/env python3
"""Compare constant-grouped layout profiles with and without FrozenBN buffers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
    nchw_nc1 = [
        row
        for row in transdata
        if row["Input Formats"] == "NCHW"
        and row["Output Formats"] == "NC1HWC0"
    ]
    count_by_shape = Counter(row["Input Shapes"] for row in nchw_nc1)
    duration_by_shape: dict[str, float] = defaultdict(float)
    for row in nchw_nc1:
        duration_by_shape[row["Input Shapes"]] += _duration_us(row)
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
        "clean_mean_ms": float(lane["control_before"]["device_event"]["mean_ms"]),
        "clean_median_ms": float(
            lane["control_before"]["device_event"]["median_ms"]
        ),
        "kernel_count": len(rows),
        "transdata_count": len(transdata),
        "transdata_ms": sum(_duration_us(row) for row in transdata) / 1000.0,
        "nchw_nc1hwc0_count": len(nchw_nc1),
        "nchw_nc1hwc0_ms": sum(_duration_us(row) for row in nchw_nc1) / 1000.0,
        "nchw_nc1hwc0_count_by_input_shape": dict(sorted(count_by_shape.items())),
        "nchw_nc1hwc0_ms_by_input_shape": {
            shape: duration_by_shape[shape] / 1000.0
            for shape in sorted(duration_by_shape)
        },
        "kernel_csv": str(path),
        "top_transdata": top_transdata,
    }


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
            (abs(first["score"] - second["score"]) for first, second in pairs),
            default=None,
        ),
        "digest_match": (
            baseline_page["result_digest"] == candidate_page["result_digest"]
        ),
    }


def _buffer_inventory(forward: dict[str, Any]) -> dict[str, Any]:
    summary = forward["config"]["frozen_bn_buffer_format_summary"]
    modules = summary["modules"]
    expected_reduction = Counter()
    formats_are_nc1hwc0 = True
    for module in modules:
        shape = f'"1,{int(module["channels"])},1,1"'
        expected_reduction[shape] += 4
        formats_are_nc1hwc0 = formats_are_nc1hwc0 and all(
            int(value) == 3 for value in module["formats"].values()
        )
    return {
        "converted_count": int(summary["converted_count"]),
        "expected_buffer_count": sum(expected_reduction.values()),
        "expected_nchw_nc1hwc0_reduction_by_input_shape": dict(
            sorted(expected_reduction.items())
        ),
        "all_buffer_formats_are_nc1hwc0": formats_are_nc1hwc0,
    }


def main() -> None:
    args = parse_args()
    baseline_forward = _load_json(args.baseline_forward)
    candidate_forward = _load_json(args.candidate_forward)
    baseline = _profile_fields(
        _load_json(args.baseline_profile), kernel_csv=args.baseline_kernel_csv
    )
    candidate = _profile_fields(
        _load_json(args.candidate_profile), kernel_csv=args.candidate_kernel_csv
    )
    output = _output_fields(baseline_forward, candidate_forward)
    baseline_inventory = _buffer_inventory(baseline_forward)
    candidate_inventory = _buffer_inventory(candidate_forward)

    baseline_shapes = Counter(baseline["nchw_nc1hwc0_count_by_input_shape"])
    candidate_shapes = Counter(candidate["nchw_nc1hwc0_count_by_input_shape"])
    actual_reduction = baseline_shapes - candidate_shapes
    unexpected_increase = candidate_shapes - baseline_shapes
    expected_reduction = Counter(
        candidate_inventory["expected_nchw_nc1hwc0_reduction_by_input_shape"]
    )
    total_td_reduction = baseline["transdata_count"] - candidate["transdata_count"]
    total_td_saved_ms = baseline["transdata_ms"] - candidate["transdata_ms"]
    gates = {
        "baseline_has_no_formatted_frozenbn_buffers": (
            baseline_inventory["converted_count"] == 0
        ),
        "candidate_has_80_formatted_frozenbn_modules": (
            candidate_inventory["converted_count"] == 80
        ),
        "candidate_has_320_formatted_frozenbn_buffers": (
            candidate_inventory["expected_buffer_count"] == 320
        ),
        "candidate_buffers_are_nc1hwc0": (
            candidate_inventory["all_buffer_formats_are_nc1hwc0"]
        ),
        "nchw_nc1hwc0_shape_reduction_matches_inventory": (
            actual_reduction == expected_reduction and not unexpected_increase
        ),
        "total_transdata_reduction_is_320": total_td_reduction == 320,
        "transdata_time_improved": candidate["transdata_ms"] < baseline["transdata_ms"],
        "clean_forward_not_slower": candidate["clean_mean_ms"] <= baseline["clean_mean_ms"],
        "same_box_count": output["same_box_count"],
        "class_label_sequence_match": output["class_label_sequence_match"],
        "coordinates_exact": output["coordinate_max_abs_px"] == 0,
        "scores_exact": output["score_max_abs"] == 0,
        "reading_order_exact": output["reading_order_changed_count"] == 0,
        "digest_match": output["digest_match"],
    }
    passed = all(gates.values())
    summary = {
        "format": "unirec_layout_frozenbn_profile_comparison_v1",
        "passed": passed,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": baseline["clean_mean_ms"] / candidate["clean_mean_ms"],
        "transdata_count_reduction": total_td_reduction,
        "transdata_saved_ms": total_td_saved_ms,
        "transdata_saved_fraction": total_td_saved_ms / baseline["transdata_ms"],
        "baseline_inventory": baseline_inventory,
        "candidate_inventory": candidate_inventory,
        "actual_nchw_nc1hwc0_reduction_by_input_shape": dict(
            sorted(actual_reduction.items())
        ),
        "unexpected_nchw_nc1hwc0_increase_by_input_shape": dict(
            sorted(unexpected_increase.items())
        ),
        "output_gate": output,
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

    print(
        f"UNIREC_LAYOUT_FROZENBN: {'PASS' if passed else 'FAIL'} "
        f"baseline_ms={baseline['clean_mean_ms']:.6f} "
        f"candidate_ms={candidate['clean_mean_ms']:.6f} "
        f"speedup={summary['speedup']:.6f} "
        f"baseline_td={baseline['transdata_count']}/{baseline['transdata_ms']:.6f}ms "
        f"candidate_td={candidate['transdata_count']}/{candidate['transdata_ms']:.6f}ms "
        f"td_saved={total_td_reduction}/{total_td_saved_ms:.6f}ms "
        f"nchw_nc1={baseline['nchw_nc1hwc0_count']}->{candidate['nchw_nc1hwc0_count']} "
        f"frozenbn_modules={candidate_inventory['converted_count']} "
        f"frozenbn_buffers={candidate_inventory['expected_buffer_count']} "
        f"boxes={output['baseline_box_count']}/{output['candidate_box_count']} "
        f"coord_max={output['coordinate_max_abs_px']} "
        f"score_max={output['score_max_abs']} "
        f"order_changed={output['reading_order_changed_count']} "
        f"digest_match={str(output['digest_match']).lower()}"
    )
    for rank, row in enumerate(candidate["top_transdata"], start=1):
        print(
            f"UNIREC_LAYOUT_FROZENBN_TD rank={rank} "
            f"count={row['count']} duration_ms={row['duration_ms']:.6f} "
            f"shape={row['input_shapes']}->{row['output_shapes']} "
            f"format={row['input_formats']}->{row['output_formats']}"
        )
    print(f"UNIREC_LAYOUT_FROZENBN_OUTPUT {args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
