#!/usr/bin/env python3
"""Localize the Phase-57 quality gap from an existing authority audit.

No inference or evaluation is performed.  This joins the exact reference and
candidate crop traces with the already-computed page metric contributions.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any


INPUT_FIELDS = (
    "prompt",
    "input_tokens",
    "projected_image_tokens",
    "crop_size",
    "min_pixels",
    "max_pixels",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _zip_jsonl(path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_image_name"]), int(row["block_index"])


def _find_audit_root() -> Path:
    roots = [
        path
        for path in Path("tmp/09_persistent_page_engine").glob(
            "310p_phase57_cap4096_b64_pse_*/authority_audit_*"
        )
        if (path / "authority_audit.json").is_file()
        and (path / "atlas/page_metric_records.jsonl").is_file()
    ]
    if not roots:
        raise FileNotFoundError("no completed Phase-57 authority audit found")
    return max(roots, key=lambda path: path.stat().st_mtime)


def _minimum_pages_to_recover(losses: list[dict[str, Any]], needed: float) -> dict[str, Any]:
    cumulative = 0.0
    for index, row in enumerate(losses, 1):
        cumulative += max(0.0, float(row["candidate_loss_contribution"]))
        if cumulative >= needed:
            return {
                "pages": index,
                "component_recovery": cumulative,
                "overall_recovery_percentage_points": 100.0 * cumulative / 3.0,
                "last_page": row["image_name"],
            }
    return {"pages": None, "component_recovery": cumulative}


def main() -> None:
    args = _args()
    audit_root = (args.audit_root or _find_audit_root()).resolve()
    audit = _json(audit_root / "authority_audit.json")
    atlas = _json(audit_root / "atlas/report.json")
    generation_records = _jsonl(audit_root / "atlas/generation_records.jsonl")
    page_metrics = _jsonl(audit_root / "atlas/page_metric_records.jsonl")
    reference_bundle = Path(audit["reference_bundle"])
    reference_trace = _zip_jsonl(reference_bundle, "recognition_trace.jsonl")

    full = audit_root.parent / "full"
    candidate_trace = _jsonl(full / "output/recognition_trace.jsonl")
    reference = {_key(row): row for row in reference_trace}
    candidate = {_key(row): row for row in candidate_trace}
    generation = {
        (str(row["source_image_name"]), int(row["block_index"])): row
        for row in generation_records
    }

    field_counts: collections.Counter[str] = collections.Counter()
    combination_counts: collections.Counter[str] = collections.Counter()
    cross: collections.Counter[str] = collections.Counter()
    page_crop_stats: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    crop_records = []
    for key in sorted(reference.keys() & candidate.keys()):
        left, right = reference[key], candidate[key]
        fields = tuple(field for field in INPUT_FIELDS if left.get(field) != right.get(field))
        for field in fields:
            field_counts[field] += 1
        combination_counts["+".join(fields) if fields else "exact"] += 1
        left_text, right_text = str(left.get("text", "")), str(right.get("text", ""))
        left_tokens, right_tokens = left.get("token_ids") or [], right.get("token_ids") or []
        if left_tokens == right_tokens and left_text == right_text:
            difference = "exact"
        elif _normalize(left_text) == _normalize(right_text):
            difference = "whitespace_nfkc_only"
        else:
            difference = "content_difference"
        input_status = "different" if fields else "exact"
        label = str(right.get("label") or left.get("label"))
        cross[f"{label}::{input_status}::{difference}"] += 1
        page = key[0]
        page_crop_stats[page][f"{label}_crops"] += 1
        page_crop_stats[page][f"{label}_{input_status}_input"] += 1
        page_crop_stats[page][f"{label}_{difference}"] += 1
        record = generation.get(key) or {}
        crop_records.append(
            {
                "image_name": page,
                "block_index": key[1],
                "label": label,
                "input_status": input_status,
                "mismatched_fields": fields,
                "difference": difference,
                "reference_tokens": len(left_tokens),
                "candidate_tokens": len(right_tokens),
                "token_delta": len(right_tokens) - len(left_tokens),
                "reference_stop": left.get("stop_reason"),
                "candidate_stop": right.get("stop_reason"),
                "triage_flags": record.get("triage_flags") or [],
                "reference_preview": left_text[:180],
                "candidate_preview": right_text[:180],
            }
        )

    metric_pages: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in page_metrics:
        metric_pages[str(row["metric"])].append(row)
    for rows in metric_pages.values():
        rows.sort(key=lambda row: float(row["candidate_loss_contribution"]), reverse=True)

    text_rows = metric_pages["text_block.Edit_dist"]
    positive_text = [row for row in text_rows if float(row["candidate_loss_contribution"]) > 0]
    text_loss_by_input: collections.Counter[str] = collections.Counter()
    enriched_text_pages = []
    for row in positive_text:
        page = str(row["image_name"])
        stats = page_crop_stats[page]
        category = "has_mismatched_text_input" if stats["text_different_input"] else "all_text_inputs_equal"
        text_loss_by_input[category] += float(row["candidate_loss_contribution"])
        enriched_text_pages.append({**row, "crop_stats": dict(stats)})

    table_rows = metric_pages["table.TEDS.page"]
    positive_table = [row for row in table_rows if float(row["candidate_loss_contribution"]) > 0]
    pathological_page = "book_zh_DLT10902008_extracted_page_8.png"
    pathological_table = next((row for row in table_rows if row["image_name"] == pathological_page), None)
    pathological_crops = [row for row in crop_records if row["image_name"] == pathological_page]

    scores = audit["scores"]
    current_overall = float(scores["candidate_310p"]["official_overall"])
    target = 0.95
    needed_overall = max(0.0, target - current_overall)
    needed_component = 3.0 * needed_overall

    longer = [
        row for row in crop_records
        if row["difference"] == "content_difference" and row["token_delta"] > 0
    ]
    longer.sort(key=lambda row: (row["token_delta"], row["candidate_tokens"]), reverse=True)
    flagged = [row for row in longer if row["triage_flags"]]

    report = {
        "classification": "PHASE57_GAP_LOCALIZED",
        "score_target": {
            "current_overall": current_overall,
            "target": target,
            "overall_deficit": needed_overall,
            "overall_deficit_percentage_points": 100.0 * needed_overall,
            "required_single_component_recovery": needed_component,
            "minimum_text_pages_if_fully_restored": _minimum_pages_to_recover(positive_text, needed_component),
            "minimum_table_pages_if_fully_restored": _minimum_pages_to_recover(positive_table, needed_component),
        },
        "input_mismatches": {
            "field_counts": dict(field_counts),
            "combination_counts": dict(combination_counts),
            "label_input_generation_cross_tab": dict(cross),
        },
        "text_loss": {
            "gross_positive_loss": sum(float(row["candidate_loss_contribution"]) for row in positive_text),
            "loss_by_page_input_status": dict(text_loss_by_input),
            "top_30_pages": enriched_text_pages[:30],
        },
        "longer_generation_differences": {
            "count": len(longer),
            "flagged_count": len(flagged),
            "top_40": longer[:40],
        },
        "pathological_table_page": {
            "page_metric": pathological_table,
            "crops": pathological_crops,
        },
        "table_loss": {
            "gross_positive_loss": sum(float(row["candidate_loss_contribution"]) for row in positive_table),
            "top_20_pages": positive_table[:20],
        },
        "atlas_generation_summary": atlas["generation"],
    }
    output_json = audit_root / "quality_gap_localization.json"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    target_info = report["score_target"]
    text_min = target_info["minimum_text_pages_if_fully_restored"]
    table_min = target_info["minimum_table_pages_if_fully_restored"]
    lines = [
        "# Phase 57 quality-gap localization",
        "",
        f"Current Overall: **{100*current_overall:.4f}**; target: **95.0000**; deficit: **{100*needed_overall:.4f} percentage points**.",
        "",
        f"Input mismatch fields: `{dict(field_counts)}`",
        f"Input mismatch combinations: `{dict(combination_counts)}`",
        f"Text gross loss by input status: `{dict(text_loss_by_input)}`",
        f"Longer differing crops: **{len(longer)}**, heuristic degeneration flags: **{len(flagged)}**.",
        f"Minimum fully restored harmful text pages to cross target: **{text_min.get('pages')}**.",
        f"Minimum fully restored harmful table pages to cross target: **{table_min.get('pages')}**.",
        "",
        "## Top harmful text pages",
    ]
    for row in enriched_text_pages[:20]:
        stats = row["crop_stats"]
        lines.append(
            f"- `{row['image_name']}` loss={row['candidate_loss_contribution']:.8f}; "
            f"text_inputs_different={stats.get('text_different_input', 0)}; "
            f"text_content_differences={stats.get('text_content_difference', 0)}"
        )
    lines.extend(["", "## Longest candidate expansions", ""])
    for row in longer[:20]:
        lines.append(
            f"- `{row['image_name']}` block={row['block_index']} label={row['label']} "
            f"tokens={row['reference_tokens']}->{row['candidate_tokens']} "
            f"input={row['input_status']} fields={list(row['mismatched_fields'])} "
            f"flags={row['triage_flags']} stop={row['candidate_stop']}"
        )
    lines.extend(["", "## Pathological table page", "", f"`{json.dumps(report['pathological_table_page'], ensure_ascii=False)}`"])
    output_md = audit_root / "quality_gap_localization.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "PHASE57_GAP_LOCALIZATION PASS "
        f"overall={100*current_overall:.4f} deficit_pp={100*needed_overall:.4f} "
        f"mismatched_inputs={sum(combination_counts.values()) - combination_counts['exact']} "
        f"longer_differences={len(longer)} flagged={len(flagged)} "
        f"text_pages_to_target={text_min.get('pages')} table_pages_to_target={table_min.get('pages')} "
        f"report={output_md}"
    )


if __name__ == "__main__":
    main()
