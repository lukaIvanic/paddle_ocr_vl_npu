#!/usr/bin/env python3
"""Compare real-page and profiler outputs for layout native MSDA."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROFILE_OPS = (
    "GridSample",
    "Transpose",
    "Cast",
    "MultiScaleDeformableAttnFunction",
    "ReduceProdD",
    "Cumsum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-forward", type=Path, required=True)
    parser.add_argument("--candidate-forward", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path)
    parser.add_argument("--candidate-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def paired_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = first["coordinate"]
    bx1, by1, bx2, by2 = second["coordinate"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 1.0


def compare_outputs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_pages = baseline["pages"]
    candidate_pages = candidate["pages"]
    if len(baseline_pages) != len(candidate_pages):
        raise ValueError(
            f"page-count mismatch: {len(baseline_pages)} != {len(candidate_pages)}"
        )
    all_ious: list[float] = []
    coordinate_diffs: list[float] = []
    score_diffs: list[float] = []
    label_mismatches: list[dict[str, Any]] = []
    order_mismatches: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    for baseline_page, candidate_page in zip(
        baseline_pages, candidate_pages
    ):
        if baseline_page["image"] != candidate_page["image"]:
            raise ValueError(
                "page identity mismatch: "
                f"{baseline_page['image']} != {candidate_page['image']}"
            )
        baseline_boxes = baseline_page["result"]["boxes"]
        candidate_boxes = candidate_page["result"]["boxes"]
        same_count = len(baseline_boxes) == len(candidate_boxes)
        pairs = list(zip(baseline_boxes, candidate_boxes)) if same_count else []
        page_ious = [paired_iou(first, second) for first, second in pairs]
        page_coordinate_diffs = [
            abs(first_value - second_value)
            for first, second in pairs
            for first_value, second_value in zip(
                first["coordinate"], second["coordinate"]
            )
        ]
        page_score_diffs = [
            abs(first["score"] - second["score"])
            for first, second in pairs
        ]
        for box_index, (first, second) in enumerate(pairs):
            if (first["cls_id"], first["label"]) != (
                second["cls_id"], second["label"]
            ):
                label_mismatches.append(
                    {
                        "image": baseline_page["image"],
                        "box_index": box_index,
                        "baseline": first,
                        "candidate": second,
                    }
                )
            if first["custom_value"] != second["custom_value"]:
                order_mismatches.append(
                    {
                        "image": baseline_page["image"],
                        "box_index": box_index,
                        "baseline": first,
                        "candidate": second,
                    }
                )
        all_ious.extend(page_ious)
        coordinate_diffs.extend(page_coordinate_diffs)
        score_diffs.extend(page_score_diffs)
        page_rows.append(
            {
                "image": baseline_page["image"],
                "baseline_box_count": len(baseline_boxes),
                "candidate_box_count": len(candidate_boxes),
                "minimum_iou": min(page_ious) if page_ious else None,
                "coordinate_max_abs_px": (
                    max(page_coordinate_diffs)
                    if page_coordinate_diffs
                    else None
                ),
                "score_max_abs": (
                    max(page_score_diffs) if page_score_diffs else None
                ),
                "digest_match": (
                    baseline_page["result_digest"]
                    == candidate_page["result_digest"]
                ),
            }
        )
    baseline_forward_ms = baseline["summary"]["stages"]["model_forward_s"][
        "mean_ms"
    ]
    candidate_forward_ms = candidate["summary"]["stages"]["model_forward_s"][
        "mean_ms"
    ]
    return {
        "page_count": len(page_rows),
        "total_baseline_boxes": sum(
            row["baseline_box_count"] for row in page_rows
        ),
        "same_box_count_pages": sum(
            row["baseline_box_count"] == row["candidate_box_count"]
            for row in page_rows
        ),
        "digest_match_pages": sum(row["digest_match"] for row in page_rows),
        "label_mismatch_count": len(label_mismatches),
        "order_mismatch_count": len(order_mismatches),
        "label_mismatches": label_mismatches,
        "order_mismatches": order_mismatches,
        "mean_paired_iou": statistics.fmean(all_ious),
        "minimum_paired_iou": min(all_ious),
        "coordinate_max_abs_px": max(coordinate_diffs),
        "coordinate_mean_abs_px": statistics.fmean(coordinate_diffs),
        "score_max_abs": max(score_diffs),
        "score_mean_abs": statistics.fmean(score_diffs),
        "baseline_model_forward_mean_ms": baseline_forward_ms,
        "candidate_model_forward_mean_ms": candidate_forward_ms,
        "model_forward_speedup": baseline_forward_ms / candidate_forward_ms,
        "model_forward_saved_ms": baseline_forward_ms - candidate_forward_ms,
        "baseline_pages_per_s": baseline["summary"]["pages_per_s"],
        "candidate_pages_per_s": candidate["summary"]["pages_per_s"],
        "layout_section_speedup": (
            candidate["summary"]["pages_per_s"]
            / baseline["summary"]["pages_per_s"]
        ),
        "worst_iou_pages": sorted(
            page_rows,
            key=lambda row: (
                1.0 if row["minimum_iou"] is None else row["minimum_iou"]
            ),
        )[:10],
    }


def profile_fields(profile: dict[str, Any]) -> dict[str, Any]:
    lane = profile["lanes"][0]
    op_rows = lane["parsed_profile"]["summary"]["runs"][0]["op_statistic"][
        "top_op_types"
    ]
    by_name = {row["op_type"]: row for row in op_rows}
    return {
        "device_event_mean_ms": lane["steady_device_event_mean_ms"],
        "kernel_count": sum(int(row["count"]) for row in op_rows),
        "compute_ms": sum(float(row["total_time_us"]) for row in op_rows)
        / 1000.0,
        "selected_ops": {
            name: {
                "count": int(by_name.get(name, {}).get("count", 0)),
                "total_time_ms": float(
                    by_name.get(name, {}).get("total_time_us", 0.0)
                )
                / 1000.0,
            }
            for name in PROFILE_OPS
        },
    }


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_forward)
    candidate = load_json(args.candidate_forward)
    output = compare_outputs(baseline, candidate)
    rewrite = candidate["config"]["msda_rewrite_summary"]
    output["candidate_rewrite"] = rewrite
    profiles = None
    if (args.baseline_profile is None) != (args.candidate_profile is None):
        raise ValueError("provide both profile reports or neither")
    structural_profile_gate = None
    if args.baseline_profile is not None:
        baseline_profile = profile_fields(load_json(args.baseline_profile))
        candidate_profile = profile_fields(load_json(args.candidate_profile))
        profiles = {
            "baseline": baseline_profile,
            "candidate": candidate_profile,
            "speedup": (
                baseline_profile["device_event_mean_ms"]
                / candidate_profile["device_event_mean_ms"]
            ),
        }
        structural_profile_gate = (
            baseline_profile["selected_ops"]["GridSample"]["count"] == 18
            and candidate_profile["selected_ops"]["GridSample"]["count"] == 0
            and candidate_profile["selected_ops"][
                "MultiScaleDeformableAttnFunction"
            ]["count"]
            == 6
        )
    gates = {
        "candidate_rewrote_all_six_msda_modules": (
            rewrite["implementation"] == "aclnn"
            and rewrite["target_count"] == 6
            and rewrite["rewritten_count"] == 6
        ),
        "all_pages_keep_box_count": (
            output["same_box_count_pages"] == output["page_count"]
        ),
        "mean_paired_iou_at_least_0_99": output["mean_paired_iou"] >= 0.99,
    }
    if structural_profile_gate is not None:
        gates["profile_has_18_to_0_gridsample_and_six_msda"] = (
            structural_profile_gate
        )
    report = {
        "format": "unirec_layout_msda_real_ab_v1",
        "passed_structural_geometry_gates": all(gates.values()),
        "quality_review_required": (
            output["label_mismatch_count"] > 0
            or output["order_mismatch_count"] > 0
        ),
        "gates": gates,
        "output_comparison": output,
        "profiles": profiles,
        "artifacts": {
            "baseline_forward": str(args.baseline_forward.resolve()),
            "candidate_forward": str(args.candidate_forward.resolve()),
            "baseline_profile": (
                None
                if args.baseline_profile is None
                else str(args.baseline_profile.resolve())
            ),
            "candidate_profile": (
                None
                if args.candidate_profile is None
                else str(args.candidate_profile.resolve())
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    status = "PASS" if report["passed_structural_geometry_gates"] else "FAIL"
    print(
        f"UNIREC_LAYOUT_MSDA_REAL_AB: {status} "
        f"pages={output['page_count']} "
        f"boxes={output['total_baseline_boxes']} "
        f"same_box_pages={output['same_box_count_pages']} "
        f"label_mismatches={output['label_mismatch_count']} "
        f"order_mismatches={output['order_mismatch_count']} "
        f"mean_iou={output['mean_paired_iou']:.9f} "
        f"minimum_iou={output['minimum_paired_iou']:.9f} "
        f"forward_ms={output['baseline_model_forward_mean_ms']:.6f}->"
        f"{output['candidate_model_forward_mean_ms']:.6f} "
        f"forward_speedup={output['model_forward_speedup']:.6f} "
        f"quality_review_required="
        f"{str(report['quality_review_required']).lower()}"
    )
    if profiles is not None:
        print(
            "UNIREC_LAYOUT_MSDA_REAL_PROFILE "
            f"device_ms={profiles['baseline']['device_event_mean_ms']:.6f}->"
            f"{profiles['candidate']['device_event_mean_ms']:.6f} "
            f"speedup={profiles['speedup']:.6f} "
            f"kernels={profiles['baseline']['kernel_count']}->"
            f"{profiles['candidate']['kernel_count']} "
            f"gridsample="
            f"{profiles['baseline']['selected_ops']['GridSample']['count']}->"
            f"{profiles['candidate']['selected_ops']['GridSample']['count']} "
            f"native_msda="
            f"{profiles['candidate']['selected_ops']['MultiScaleDeformableAttnFunction']['count']}"
        )
    print(f"UNIREC_LAYOUT_MSDA_REAL_OUTPUT {args.output.resolve()}")
    if not report["passed_structural_geometry_gates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
