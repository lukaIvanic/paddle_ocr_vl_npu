#!/usr/bin/env python3
"""Analyze paired Experiment 09 OCR-resolution ablations.

For max-pixels candidates, a crop is affected when its resized vision grid
changes.  For text-crop-scale candidates, a text crop is also affected when its
pixel input changes even if alignment produces the same final vision grid.  The
report separates those crops from unaffected crops whose execution route may
still change because production packing was rerouted around them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


TRACE_NAME = "recognition_trace.jsonl"
SUMMARY_NAME = "run_summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-limit", type=int, default=50)
    args = parser.parse_args(argv)
    if args.worst_limit <= 0:
        parser.error("--worst-limit must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_trace(root: Path) -> list[dict[str, Any]]:
    path = root / TRACE_NAME
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    return {
        "count": len(materialized),
        "mean": sum(materialized) / len(materialized) if materialized else None,
        "p50": percentile(materialized, 0.50),
        "p90": percentile(materialized, 0.90),
        "p95": percentile(materialized, 0.95),
        "p99": percentile(materialized, 0.99),
        "max": max(materialized) if materialized else None,
    }


def levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Exact Myers bit-vector distance for arbitrary hashable sequences."""
    if len(left) > len(right):
        left, right = right, left
    width = len(left)
    if width == 0:
        return len(right)
    equality_masks: dict[Any, int] = {}
    for index, value in enumerate(left):
        equality_masks[value] = equality_masks.get(value, 0) | (1 << index)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    positive = mask
    negative = 0
    distance = width
    for value in right:
        equal = equality_masks.get(value, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        elif negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = (positive_horizontal & vertical) & mask
    return distance


def normalized_distance(left: Sequence[Any], right: Sequence[Any]) -> tuple[int, float]:
    if left == right:
        return 0, 0.0
    distance = levenshtein(left, right)
    return distance, distance / max(1, len(left), len(right))


def compact_text(value: str) -> str:
    return "".join(value.split())


def lexical_units(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)


def repetition_fraction(tokens: list[int], width: int = 4) -> float:
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return min(len(left), len(right))


def route_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    vision = row.get("vision") or {}
    return (
        vision.get("execution"),
        vision.get("physical_vision_tokens"),
        vision.get("bucket"),
        tuple(vision.get("pack_row_sizes") or ()),
    )


def crop_size(row: dict[str, Any]) -> tuple[int, int] | None:
    value = row.get("crop_size")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def compare_row(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_tokens = [int(value) for value in reference.get("token_ids", ())]
    candidate_tokens = [int(value) for value in candidate.get("token_ids", ())]
    reference_text = str(reference.get("text", ""))
    candidate_text = str(candidate.get("text", ""))
    reference_compact = compact_text(reference_text)
    candidate_compact = compact_text(candidate_text)
    character_edits, character_distance = normalized_distance(
        reference_compact,
        candidate_compact,
    )
    word_edits, word_distance = normalized_distance(
        lexical_units(reference_text),
        lexical_units(candidate_text),
    )
    token_edits, token_distance = normalized_distance(reference_tokens, candidate_tokens)
    reference_vision = int(reference["vision"]["real_vision_tokens"])
    candidate_vision = int(candidate["vision"]["real_vision_tokens"])
    reference_physical_vision = int(
        reference["vision"]["physical_vision_tokens"]
    )
    candidate_physical_vision = int(
        candidate["vision"]["physical_vision_tokens"]
    )
    label = str(reference["label"])
    reference_crop_size = crop_size(reference)
    candidate_crop_size = crop_size(candidate)
    crop_pixel_input_changed = reference_crop_size != candidate_crop_size
    vision_grid_changed = reference_vision != candidate_vision
    affected_reasons: list[str] = []
    if vision_grid_changed:
        affected_reasons.append("vision_grid_changed")
    if label == "text" and crop_pixel_input_changed:
        affected_reasons.append("text_crop_pixel_input_changed")
    reference_repetition = repetition_fraction(reference_tokens)
    candidate_repetition = repetition_fraction(candidate_tokens)
    runaway = (
        len(candidate_tokens) >= 256
        and len(candidate_tokens) >= max(3 * max(1, len(reference_tokens)), len(reference_tokens) + 128)
    )
    collapsed = (
        len(reference_tokens) >= 256
        and len(reference_tokens) >= max(3 * max(1, len(candidate_tokens)), len(candidate_tokens) + 128)
    )
    repetition_regression = (
        len(candidate_tokens) >= 128
        and candidate_repetition - reference_repetition >= 0.25
    )
    candidate_length_stop = (
        reference.get("stop_reason") == "eos"
        and candidate.get("stop_reason") != "eos"
    )
    flags = []
    if runaway:
        flags.append("candidate_runaway_length")
    if collapsed:
        flags.append("candidate_collapsed_length")
    if repetition_regression:
        flags.append("candidate_repetition_regression")
    if candidate_length_stop:
        flags.append("candidate_lost_eos")
    if character_distance >= 0.50:
        flags.append("large_compact_text_edit")
    return {
        "request_id": str(reference["request_id"]),
        "page_input_index": int(reference["page_input_index"]),
        "source_image_name": str(reference["source_image_name"]),
        "block_index": int(reference["block_index"]),
        "label": label,
        "crop_size": reference.get("crop_size"),
        "reference_crop_size": reference_crop_size,
        "candidate_crop_size": candidate_crop_size,
        "crop_pixel_input_changed": crop_pixel_input_changed,
        "vision_grid_changed": vision_grid_changed,
        "affected_by_ablation": bool(affected_reasons),
        "affected_reasons": affected_reasons,
        "reference_real_vision_tokens": reference_vision,
        "candidate_real_vision_tokens": candidate_vision,
        "vision_tokens_saved": reference_vision - candidate_vision,
        "vision_reduction_fraction": (
            (reference_vision - candidate_vision) / reference_vision
            if reference_vision
            else 0.0
        ),
        "reference_physical_vision_tokens": reference_physical_vision,
        "candidate_physical_vision_tokens": candidate_physical_vision,
        "physical_vision_tokens_saved": (
            reference_physical_vision - candidate_physical_vision
        ),
        "vision_route_exact": route_signature(reference) == route_signature(candidate),
        "token_ids_exact": reference_tokens == candidate_tokens,
        "text_exact": reference_text == candidate_text,
        "compact_text_exact": reference_compact == candidate_compact,
        "common_prefix_tokens": common_prefix(reference_tokens, candidate_tokens),
        "reference_output_tokens": len(reference_tokens),
        "candidate_output_tokens": len(candidate_tokens),
        "output_token_delta": len(candidate_tokens) - len(reference_tokens),
        "token_edit_distance": token_edits,
        "normalized_token_edit_distance": token_distance,
        "compact_character_edit_distance": character_edits,
        "normalized_compact_character_edit_distance": character_distance,
        "word_edit_distance": word_edits,
        "normalized_word_edit_distance": word_distance,
        "reference_repetition_fraction": reference_repetition,
        "candidate_repetition_fraction": candidate_repetition,
        "reference_stop_reason": reference.get("stop_reason"),
        "candidate_stop_reason": candidate.get("stop_reason"),
        "manual_review_flags": flags,
        "reference_text": reference_text,
        "candidate_text": candidate_text,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    for label in sorted({str(row["label"]) for row in rows}):
        selected = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "crops": len(selected),
            "reference_real_vision_tokens": sum(
                row["reference_real_vision_tokens"] for row in selected
            ),
            "candidate_real_vision_tokens": sum(
                row["candidate_real_vision_tokens"] for row in selected
            ),
            "reference_physical_vision_tokens": sum(
                row["reference_physical_vision_tokens"] for row in selected
            ),
            "candidate_physical_vision_tokens": sum(
                row["candidate_physical_vision_tokens"] for row in selected
            ),
            "token_exact": sum(row["token_ids_exact"] for row in selected),
            "compact_text_exact": sum(row["compact_text_exact"] for row in selected),
            "flagged": sum(bool(row["manual_review_flags"]) for row in selected),
            "normalized_compact_character_edit_distance": distribution(
                row["normalized_compact_character_edit_distance"] for row in selected
            ),
        }
    flags = Counter(
        flag
        for row in rows
        for flag in row["manual_review_flags"]
    )
    return {
        "crops": len(rows),
        "pages": len({row["page_input_index"] for row in rows}),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "reference_real_vision_tokens": sum(row["reference_real_vision_tokens"] for row in rows),
        "candidate_real_vision_tokens": sum(row["candidate_real_vision_tokens"] for row in rows),
        "vision_tokens_saved": sum(row["vision_tokens_saved"] for row in rows),
        "reference_physical_vision_tokens": sum(
            row["reference_physical_vision_tokens"] for row in rows
        ),
        "candidate_physical_vision_tokens": sum(
            row["candidate_physical_vision_tokens"] for row in rows
        ),
        "physical_vision_tokens_saved": sum(
            row["physical_vision_tokens_saved"] for row in rows
        ),
        "token_exact": sum(row["token_ids_exact"] for row in rows),
        "text_exact": sum(row["text_exact"] for row in rows),
        "compact_text_exact": sum(row["compact_text_exact"] for row in rows),
        "route_exact": sum(row["vision_route_exact"] for row in rows),
        "flagged": sum(bool(row["manual_review_flags"]) for row in rows),
        "flag_counts": dict(sorted(flags.items())),
        "normalized_token_edit_distance": distribution(
            row["normalized_token_edit_distance"] for row in rows
        ),
        "normalized_compact_character_edit_distance": distribution(
            row["normalized_compact_character_edit_distance"] for row in rows
        ),
        "normalized_word_edit_distance": distribution(
            row["normalized_word_edit_distance"] for row in rows
        ),
        "output_token_delta": distribution(row["output_token_delta"] for row in rows),
        "by_label": by_label,
    }


def render_markdown(report: dict[str, Any]) -> str:
    affected = report["affected"]
    all_crops = report["all_crops"]
    spillover = report["unaffected_with_generation_difference"]
    timing = report["timing"]
    lines = [
        "# OCR resolution ablation",
        "",
        f"- Reference: `{report['reference_output']}`",
        f"- Candidate: `{report['candidate_output']}`",
        f"- Shared crops: **{report['shared_crops']}**",
        f"- Affected crops: **{affected['crops']} across {affected['pages']} pages**",
        f"- Affected because the vision grid changed: **{report['affected_reason_counts'].get('vision_grid_changed', 0)}**",
        f"- Affected because the text-crop pixel input changed: **{report['affected_reason_counts'].get('text_crop_pixel_input_changed', 0)}**",
        f"- Real vision tokens: **{affected['reference_real_vision_tokens']:,} -> {affected['candidate_real_vision_tokens']:,}** ({affected['vision_tokens_saved']:,} saved)",
        f"- Physical vision tokens: **{affected['reference_physical_vision_tokens']:,} -> {affected['candidate_physical_vision_tokens']:,}** ({affected['physical_vision_tokens_saved']:,} saved)",
        f"- Token streams exact among affected crops: **{affected['token_exact']}/{affected['crops']}**",
        f"- Whitespace-insensitive text exact: **{affected['compact_text_exact']}/{affected['crops']}**",
        f"- Automatically flagged for manual review: **{affected['flagged']}**",
        f"- Unaffected crops whose generation changed: **{spillover['crops']}**",
        "",
        "## Runtime",
        "",
        f"- Reference pipeline: **{timing['reference_pipeline_e2e_s']:.3f}s**",
        f"- Candidate pipeline: **{timing['candidate_pipeline_e2e_s']:.3f}s**",
        f"- Delta: **{timing['candidate_minus_reference_s']:+.3f}s ({timing['candidate_vs_reference_percent']:+.2f}%)**",
        "",
        "## Affected crops by recognizer route",
        "",
        "| Route | Crops | Real vision tokens | Physical vision tokens | Token exact | Compact-text exact | Flagged | Mean normalized character edit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in affected["by_label"].items():
        lines.append(
            f"| {label} | {row['crops']} | "
            f"{row['reference_real_vision_tokens']:,} -> {row['candidate_real_vision_tokens']:,} | "
            f"{row['reference_physical_vision_tokens']:,} -> {row['candidate_physical_vision_tokens']:,} | "
            f"{row['token_exact']} | "
            f"{row['compact_text_exact']} | {row['flagged']} | "
            f"{row['normalized_compact_character_edit_distance']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## All crops by recognizer route",
            "",
            "| Route | Crops | Real vision tokens | Physical vision tokens |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in all_crops["by_label"].items():
        lines.append(
            f"| {label} | {row['crops']} | "
            f"{row['reference_real_vision_tokens']:,} -> {row['candidate_real_vision_tokens']:,} | "
            f"{row['reference_physical_vision_tokens']:,} -> {row['candidate_physical_vision_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "## Automatic review flags",
            "",
        ]
    )
    if affected["flag_counts"]:
        lines.extend(
            f"- `{name}`: {count}"
            for name, count in affected["flag_counts"].items()
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Worst affected-crop differences",
            "",
            "| Request | Label | Vision tokens | Output tokens | Character edit | Flags |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["worst_affected"]:
        lines.append(
            f"| `{row['request_id']}` | {row['label']} | "
            f"{row['reference_real_vision_tokens']} -> {row['candidate_real_vision_tokens']} | "
            f"{row['reference_output_tokens']} -> {row['candidate_output_tokens']} | "
            f"{row['normalized_compact_character_edit_distance']:.4f} | "
            f"{', '.join(row['manual_review_flags']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.",
            "Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference_root = args.reference_output.expanduser().resolve()
    candidate_root = args.candidate_output.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_rows = read_trace(reference_root)
    candidate_rows = read_trace(candidate_root)
    reference = {str(row["request_id"]): row for row in reference_rows}
    candidate = {str(row["request_id"]): row for row in candidate_rows}
    if reference.keys() != candidate.keys():
        raise ValueError(
            "paired runs do not contain identical request IDs: "
            f"reference_only={len(reference.keys() - candidate.keys())} "
            f"candidate_only={len(candidate.keys() - reference.keys())}"
        )
    rows = [compare_row(reference[str(row["request_id"])], candidate[str(row["request_id"])]) for row in reference_rows]
    affected_rows = [row for row in rows if row["affected_by_ablation"]]
    unaffected_different = [
        row
        for row in rows
        if not row["affected_by_ablation"] and not row["token_ids_exact"]
    ]
    reference_summary = read_json(reference_root / SUMMARY_NAME)
    candidate_summary = read_json(candidate_root / SUMMARY_NAME)
    reference_e2e = float(reference_summary["pipeline_e2e_s"])
    candidate_e2e = float(candidate_summary["pipeline_e2e_s"])
    worst = sorted(
        affected_rows,
        key=lambda row: (
            bool(row["manual_review_flags"]),
            row["normalized_compact_character_edit_distance"],
            abs(row["output_token_delta"]),
        ),
        reverse=True,
    )[: args.worst_limit]
    report = {
        "schema_version": 1,
        "kind": "max_pixels_ocr_ablation",
        "reference_output": str(reference_root),
        "candidate_output": str(candidate_root),
        "shared_crops": len(rows),
        "affected_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in affected_rows
                    for reason in row["affected_reasons"]
                ).items()
            )
        ),
        "all_crops": summarize(rows),
        "affected": summarize(affected_rows),
        "unaffected": summarize(
            [row for row in rows if not row["affected_by_ablation"]]
        ),
        "unaffected_with_generation_difference": summarize(unaffected_different),
        "timing": {
            "reference_pipeline_e2e_s": reference_e2e,
            "candidate_pipeline_e2e_s": candidate_e2e,
            "candidate_minus_reference_s": candidate_e2e - reference_e2e,
            "candidate_vs_reference_percent": (candidate_e2e / reference_e2e - 1.0) * 100.0,
        },
        "worst_affected": worst,
    }
    with (output_dir / "per_crop.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "request_id", "source_image_name", "label", "reference_crop_size",
            "candidate_crop_size", "affected_reasons",
            "reference_real_vision_tokens", "candidate_real_vision_tokens",
            "reference_physical_vision_tokens", "candidate_physical_vision_tokens",
            "reference_output_tokens", "candidate_output_tokens",
            "normalized_compact_character_edit_distance",
            "normalized_word_edit_distance", "manual_review_flags",
            "reference_text", "candidate_text",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in worst:
            serialized = dict(row)
            serialized["manual_review_flags"] = ";".join(row["manual_review_flags"])
            serialized["affected_reasons"] = ";".join(row["affected_reasons"])
            writer.writerow(serialized)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
