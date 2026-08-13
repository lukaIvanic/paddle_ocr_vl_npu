#!/usr/bin/env python3
"""Summarize guarded OmniDocBench metrics and direct CDM into Overall."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-result", type=Path, required=True)
    parser.add_argument("--stage-execution", type=Path, required=True)
    parser.add_argument("--cdm-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    return parser.parse_args()


def page_edit(metric: dict[str, Any], category: str) -> float:
    section = metric[category]
    if "page" in section:
        return float(section["page"]["Edit_dist"]["ALL"])
    return float(section["all"]["Edit_dist"]["ALL_page_avg"])


def main() -> None:
    args = parse_args()
    metric = json.loads(args.metric_result.read_text(encoding="utf-8"))
    stage = json.loads(args.stage_execution.read_text(encoding="utf-8"))
    cdm = json.loads(args.cdm_summary.read_text(encoding="utf-8"))
    evaluated = json.loads(Path(cdm["evaluated_samples"]).read_text(encoding="utf-8"))
    by_page: dict[str, list[float]] = defaultdict(list)
    for sample in evaluated:
        by_page[str(sample["img_id"])].append(float(sample["metric"]["CDM"]))
    sample_cdm = sum(value for values in by_page.values() for value in values) / len(evaluated)
    page_cdm = sum(sum(values) / len(values) for values in by_page.values()) / len(by_page)
    reported_sample_cdm = float(cdm["scores"]["CDM"]["all"])
    if abs(sample_cdm - reported_sample_cdm) >= 1e-12:
        raise RuntimeError("direct CDM summary does not match evaluated samples")

    text_edit = page_edit(metric, "text_block")
    formula_edit = page_edit(metric, "display_formula")
    reading_edit = page_edit(metric, "reading_order")
    page_teds = float(metric["table"]["page"]["TEDS"]["ALL"])
    structure_teds = float(
        metric["table"]["page"]["TEDS_structure_only"]["ALL"]
    )
    sample_teds = float(metric["table"]["all"]["TEDS"]["all"])
    overall = ((1.0 - text_edit) + page_cdm + page_teds) / 3.0
    teds_debug = stage["metrics"]["table"]["TEDS"]
    result = {
        "lane": args.lane,
        "text_block_page_edit": text_edit,
        "display_formula_page_edit": formula_edit,
        "display_formula_sample_cdm": sample_cdm,
        "display_formula_page_cdm": page_cdm,
        "table_sample_teds": sample_teds,
        "table_page_teds": page_teds,
        "table_page_teds_structure_only": structure_teds,
        "reading_order_page_edit": reading_edit,
        "official_overall": overall,
        "formula_samples": len(evaluated),
        "formula_pages": len(by_page),
        "page_match": stage["page_match"],
        "table_teds_execution": teds_debug,
        "cdm_debug": cdm["debug"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "UNIREC_FULL_EVAL PASS "
        f"lane={args.lane} text_edit={text_edit:.6f} "
        f"formula_edit={formula_edit:.6f} page_cdm={page_cdm:.6f} "
        f"page_teds={page_teds:.6f} structure_teds={structure_teds:.6f} "
        f"reading_edit={reading_edit:.6f} overall={100 * overall:.4f}"
    )


if __name__ == "__main__":
    main()
