#!/usr/bin/env python3
"""Compare stabilized and direct-Softmax compiled layout profiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COGVIEW_ARGMAX_INPUT_SHAPE = '"1,8,302,302"'


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


def _profile_fields(
    profile: dict[str, Any],
    *,
    kernel_csv: Path | None,
) -> dict[str, Any]:
    lane = profile["lanes"][0]
    path = _kernel_csv_path(profile) if kernel_csv is None else kernel_csv
    rows = _load_kernel_rows(path)
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["Type"]].append(_duration_us(row))
    argmax = [row for row in rows if row["Type"] == "ArgMaxWithValue"]
    cogview_argmax = [
        row for row in argmax if row["Input Shapes"] == COGVIEW_ARGMAX_INPUT_SHAPE
    ]
    type_summary = sorted(
        (
            {
                "type": kind,
                "count": len(durations),
                "duration_ms": sum(durations) / 1000.0,
            }
            for kind, durations in by_type.items()
        ),
        key=lambda item: item["duration_ms"],
        reverse=True,
    )
    argmax_shapes = Counter(row["Input Shapes"] for row in argmax)
    return {
        "lane_name": lane["name"],
        "clean_mean_ms": float(lane["control_before"]["device_event"]["mean_ms"]),
        "clean_median_ms": float(
            lane["control_before"]["device_event"]["median_ms"]
        ),
        "kernel_count": len(rows),
        "kernel_sum_ms": sum(_duration_us(row) for row in rows) / 1000.0,
        "argmax_count": len(argmax),
        "argmax_ms": sum(_duration_us(row) for row in argmax) / 1000.0,
        "argmax_count_by_input_shape": dict(sorted(argmax_shapes.items())),
        "cogview_argmax_count": len(cogview_argmax),
        "cogview_argmax_ms": (
            sum(_duration_us(row) for row in cogview_argmax) / 1000.0
        ),
        "top_kernel_types": type_summary[:15],
        "kernel_csv": str(path),
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


def _attention_impl(forward: dict[str, Any]) -> str:
    # Artifacts created before this diagnostic selector used stabilized math.
    return str(forward["config"].get("cogview_attention_impl", "stabilized"))


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
    baseline_impl = _attention_impl(baseline_forward)
    candidate_impl = _attention_impl(candidate_forward)
    gates = {
        "baseline_is_stabilized": baseline_impl == "stabilized",
        "candidate_is_direct_softmax": candidate_impl == "direct_softmax",
        "baseline_has_six_cogview_argmax": baseline["cogview_argmax_count"] == 6,
        "candidate_has_zero_cogview_argmax": candidate["cogview_argmax_count"] == 0,
        "total_argmax_count_reduced_by_six": (
            baseline["argmax_count"] - candidate["argmax_count"] == 6
        ),
        "argmax_time_improved": candidate["argmax_ms"] < baseline["argmax_ms"],
        "clean_forward_not_slower": candidate["clean_mean_ms"] <= baseline["clean_mean_ms"],
        "same_box_count": output["same_box_count"],
        "class_label_sequence_match": output["class_label_sequence_match"],
        "coordinate_max_abs_at_most_1px": (
            output["coordinate_max_abs_px"] is not None
            and output["coordinate_max_abs_px"] <= 1.0
        ),
        "score_max_abs_at_most_5e_3": (
            output["score_max_abs"] is not None
            and output["score_max_abs"] <= 5e-3
        ),
        "reading_order_changed_at_most_1": (
            output["reading_order_changed_count"] <= 1
        ),
    }
    passed = all(gates.values())
    summary = {
        "format": "unirec_layout_cogview_softmax_profile_comparison_v1",
        "passed": passed,
        "baseline_impl": baseline_impl,
        "candidate_impl": candidate_impl,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": baseline["clean_mean_ms"] / candidate["clean_mean_ms"],
        "argmax_saved_ms": baseline["argmax_ms"] - candidate["argmax_ms"],
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
        f"UNIREC_LAYOUT_COGVIEW_SOFTMAX: {'PASS' if passed else 'FAIL'} "
        f"baseline_ms={baseline['clean_mean_ms']:.6f} "
        f"candidate_ms={candidate['clean_mean_ms']:.6f} "
        f"speedup={summary['speedup']:.6f} "
        f"argmax={baseline['argmax_count']}/{baseline['argmax_ms']:.6f}ms"
        f"->{candidate['argmax_count']}/{candidate['argmax_ms']:.6f}ms "
        f"cogview_argmax={baseline['cogview_argmax_count']}"
        f"->{candidate['cogview_argmax_count']} "
        f"boxes={output['baseline_box_count']}/{output['candidate_box_count']} "
        f"coord_max={output['coordinate_max_abs_px']} "
        f"score_max={output['score_max_abs']} "
        f"order_changed={output['reading_order_changed_count']} "
        f"digest_match={str(output['digest_match']).lower()}"
    )
    for rank, row in enumerate(candidate["top_kernel_types"], start=1):
        print(
            f"UNIREC_LAYOUT_COGVIEW_SOFTMAX_KERNEL rank={rank} "
            f"type={row['type']} count={row['count']} "
            f"duration_ms={row['duration_ms']:.6f}"
        )
    print(f"UNIREC_LAYOUT_COGVIEW_SOFTMAX_OUTPUT {args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
