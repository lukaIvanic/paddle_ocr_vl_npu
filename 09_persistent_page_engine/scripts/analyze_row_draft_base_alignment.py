#!/usr/bin/env python3
"""Align each row-crop generation to its closest span in the whole-table output."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any
import unicodedata


CELL_PATTERN = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel)>")
STRUCTURE_PATTERN = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>")
DISPLAY_TOKEN_PATTERN = re.compile(
    r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>|"
    r"\\\([^)]*\\\)|\\\[[^]]*\\\]|\S+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-latency-s", type=float, default=0.0)
    parser.add_argument("--samples-per-tier", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return " ".join(value.split())


def content_text(text: str) -> str:
    return normalize_text(STRUCTURE_PATTERN.sub(" ", text))


def structure_tokens(text: str) -> list[str]:
    return STRUCTURE_PATTERN.findall(text)


def logical_rows(text: str) -> list[str]:
    return [row for row in text.split("<nl>") if row]


def ratio(left: Any, right: Any) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def display_tokens(text: str) -> list[str]:
    return DISPLAY_TOKEN_PATTERN.findall(text)


def diff_segments(base: str, draft: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base_tokens = display_tokens(base)
    draft_tokens = display_tokens(draft)
    matcher = SequenceMatcher(None, base_tokens, draft_tokens, autojunk=False)
    base_segments: list[dict[str, str]] = []
    draft_segments: list[dict[str, str]] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode in ("equal", "delete", "replace") and i1 != i2:
            base_segments.append(
                {
                    "kind": "equal" if opcode == "equal" else "delete",
                    "text": " ".join(base_tokens[i1:i2]),
                }
            )
        if opcode in ("equal", "insert", "replace") and j1 != j2:
            draft_segments.append(
                {
                    "kind": "equal" if opcode == "equal" else "insert",
                    "text": " ".join(draft_tokens[j1:j2]),
                }
            )
    return base_segments, draft_segments


def best_base_span(
    base_rows: list[str],
    draft_text: str,
    expected_fraction: float,
) -> dict[str, Any]:
    draft_rows = logical_rows(draft_text)
    draft_row_count = max(1, len(draft_rows))
    span_lengths = range(
        max(1, draft_row_count - 2),
        min(len(base_rows), draft_row_count + 2) + 1,
    )
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for span_length in span_lengths:
        for start in range(0, len(base_rows) - span_length + 1):
            end = start + span_length
            base_text = "<nl>".join(base_rows[start:end]) + "<nl>"
            raw_similarity = ratio(normalize_text(base_text), normalize_text(draft_text))
            content_similarity = ratio(content_text(base_text), content_text(draft_text))
            structure_similarity = ratio(
                structure_tokens(base_text), structure_tokens(draft_text)
            )
            center_fraction = (start + 0.5 * span_length) / max(1, len(base_rows))
            position_similarity = 1.0 - min(1.0, abs(center_fraction - expected_fraction))
            combined = (
                0.55 * content_similarity
                + 0.25 * raw_similarity
                + 0.15 * structure_similarity
                + 0.05 * position_similarity
            )
            score = (
                combined,
                content_similarity,
                structure_similarity,
                -abs(center_fraction - expected_fraction),
            )
            result = {
                "base_row_start": start,
                "base_row_end": end,
                "base_text": base_text,
                "raw_similarity": raw_similarity,
                "content_similarity": content_similarity,
                "structure_similarity": structure_similarity,
                "position_similarity": position_similarity,
                "combined_similarity": combined,
            }
            if best is None or score > best[0]:
                best = (score, result)
    if best is None:
        return {
            "base_row_start": 0,
            "base_row_end": 0,
            "base_text": "",
            "raw_similarity": 0.0,
            "content_similarity": 0.0,
            "structure_similarity": 0.0,
            "position_similarity": 0.0,
            "combined_similarity": 0.0,
        }
    return best[1]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_examples(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["mean_combined_similarity"])
    if not ordered:
        return []
    tiers = {
        "poor": ordered[:count],
        "mixed": sorted(
            ordered,
            key=lambda row: abs(row["mean_combined_similarity"] - statistics.median(
                item["mean_combined_similarity"] for item in ordered
            )),
        )[:count],
        "strong": list(reversed(ordered[-count:])),
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tier, tier_rows in tiers.items():
        for row in tier_rows:
            if row["request_id"] in seen:
                continue
            seen.add(row["request_id"])
            selected.append({**row, "selection_tier": tier})
    return selected


def main() -> None:
    args = parse_args()
    targets = {row["request_id"]: row for row in read_jsonl(args.targets)}
    drafts = {row["request_id"]: row for row in read_jsonl(args.drafts)}
    latencies = {
        row["request_id"]: float(row["worker_wall_s"])
        for row in read_jsonl(args.baseline_records)
    }
    analyzed: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for request_id in sorted(set(targets) & set(drafts)):
        baseline_latency = latencies.get(request_id, 0.0)
        if baseline_latency < args.minimum_latency_s:
            continue
        target_record = targets[request_id]
        draft_record = drafts[request_id]
        base_text = target_record["rows"][0]["text"]
        base_rows = logical_rows(base_text)
        table_height = max(1, int(draft_record["crop_size"][1]))
        boundaries = [int(value) for value in draft_record["boundaries"]]
        lanes: list[dict[str, Any]] = []
        for row_index, row in enumerate(draft_record["rows"]):
            top = boundaries[min(row_index, len(boundaries) - 1)]
            bottom = boundaries[min(row_index + 1, len(boundaries) - 1)]
            expected_fraction = (top + bottom) / (2.0 * table_height)
            alignment = best_base_span(base_rows, row["text"], expected_fraction)
            base_diff, draft_diff = diff_segments(alignment["base_text"], row["text"])
            if alignment["content_similarity"] >= 0.98 and alignment["structure_similarity"] >= 0.98:
                category = "near_exact"
            elif alignment["structure_similarity"] < 0.85:
                category = "structure"
            elif alignment["content_similarity"] < 0.75:
                category = "content"
            else:
                category = "mixed"
            category_counts[category] += 1
            lanes.append(
                {
                    "row_index": row_index,
                    "crop_size": row.get("crop_size"),
                    "boundary_y": [top, bottom],
                    "expected_fraction": expected_fraction,
                    "draft_text": row["text"],
                    "draft_token_count": len(row.get("token_ids") or ()),
                    "category": category,
                    "base_diff": base_diff,
                    "draft_diff": draft_diff,
                    **alignment,
                }
            )
        weights = [max(1, lane["draft_token_count"]) for lane in lanes]
        total_weight = sum(weights)
        mean_combined = sum(
            lane["combined_similarity"] * weight
            for lane, weight in zip(lanes, weights)
        ) / total_weight
        mean_content = sum(
            lane["content_similarity"] * weight
            for lane, weight in zip(lanes, weights)
        ) / total_weight
        mean_structure = sum(
            lane["structure_similarity"] * weight
            for lane, weight in zip(lanes, weights)
        ) / total_weight
        analyzed.append(
            {
                "request_id": request_id,
                "page_name": draft_record.get("page_name"),
                "annotation_index": draft_record.get("annotation_index"),
                "crop_size": draft_record.get("crop_size"),
                "rotation_cw": draft_record.get("row_draft_rotation_cw", 0),
                "baseline_latency_s": baseline_latency,
                "base_row_count": len(base_rows),
                "lane_count": len(lanes),
                "mean_combined_similarity": mean_combined,
                "mean_content_similarity": mean_content,
                "mean_structure_similarity": mean_structure,
                "base_text": base_text,
                "base_html": target_record.get("whole_table_prediction") or "",
                "draft_html": draft_record.get("pred_html") or "",
                "gt_html": draft_record.get("gt_html") or "",
                "lanes": lanes,
            }
        )
    similarities = [row["mean_combined_similarity"] for row in analyzed]
    report = {
        "inputs": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "baseline_records": str(args.baseline_records),
            "minimum_latency_s": args.minimum_latency_s,
        },
        "summary": {
            "tables": len(analyzed),
            "lanes": sum(len(row["lanes"]) for row in analyzed),
            "mean_similarity": statistics.mean(similarities) if similarities else None,
            "p10_similarity": percentile(similarities, 0.10),
            "p50_similarity": percentile(similarities, 0.50),
            "p90_similarity": percentile(similarities, 0.90),
            "lane_categories": dict(category_counts),
        },
        "selected": select_examples(analyzed, args.samples_per_tier),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"complete tables={len(analyzed)} lanes={report['summary']['lanes']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
