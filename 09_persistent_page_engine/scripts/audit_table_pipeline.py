#!/usr/bin/env python3
"""Localize table losses across frozen Experiment-09/evaluator artifacts."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace every OmniDocBench GT table through final page regions, "
            "recognition routing/output, and evaluator matching."
        )
    )
    parser.add_argument("--e2e-output", required=True, type=Path)
    parser.add_argument("--table-result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--expected-tables", type=int)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _counter(counter: collections.Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    }


def _page_name(img_id: str) -> str:
    if img_id.endswith((".jpg", ".png")):
        return img_id
    return "_".join(img_id.split("_")[:-1])


def _poly_bbox(poly: list[float]) -> tuple[float, float, float, float]:
    xs = poly[0::2]
    ys = poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0, right[3] - right[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _iou_band(value: float) -> str:
    if value < 0.1:
        return "none_lt_0.1"
    if value < 0.5:
        return "low_0.1_to_0.5"
    return "good_ge_0.5"


def _format(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return "empty"
    if "<fcel" in normalized:
        return "fcel"
    if "<table" in normalized:
        return "html_table"
    return "other"


def _best_block(
    gt_bbox: tuple[float, float, float, float],
    blocks: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best = None
    for block in blocks:
        raw_bbox = block.get("block_bbox")
        if not raw_bbox or len(raw_bbox) != 4:
            continue
        score = _iou(gt_bbox, tuple(float(value) for value in raw_bbox))
        if best is None or score > best_score:
            best_score = score
            best = block
    return best_score, best


def _failure_stage(record: dict[str, Any]) -> str:
    if record["best_iou"] < 0.5:
        return "layout_localization"
    if record["best_block_label"] != "table":
        return "layout_label"
    if not record["recognition_request_present"]:
        return "recognition_request_missing"
    if record["recognition_label"] != "table":
        return "recognition_route"
    if not record["recognition_text_nonempty"]:
        return "recognition_empty"
    if not record["evaluator_pred_nonempty"]:
        return "assembly_or_match"
    return "matched"


def main() -> None:
    args = _parse_args()
    output_root = args.e2e_output.resolve()
    dataset_path = output_root / "OmniDocBench_subset.json"
    page_regions_path = output_root / "page_regions.jsonl"
    recognition_trace_path = output_root / "recognition_trace.jsonl"
    for path in (
        dataset_path,
        page_regions_path,
        recognition_trace_path,
        args.table_result,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    page_region_rows = _read_jsonl(page_regions_path)
    recognition_rows = _read_jsonl(recognition_trace_path)
    table_results = json.loads(args.table_result.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not isinstance(table_results, list):
        raise TypeError("dataset and table result must both be JSON lists")
    if args.expected_pages is not None and len(dataset) != args.expected_pages:
        raise ValueError(f"expected {args.expected_pages} pages, got {len(dataset)}")

    page_regions = {row["image_name"]: row for row in page_region_rows}
    if len(page_regions) != len(page_region_rows):
        raise ValueError("duplicate image names in page_regions.jsonl")
    traces = {
        (row["source_image_name"], int(row["block_index"])): row
        for row in recognition_rows
    }
    if len(traces) != len(recognition_rows):
        raise ValueError("duplicate page/block keys in recognition_trace.jsonl")

    evaluator_by_gt: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in table_results:
        indices = sample.get("gt_idx") or []
        for index in indices:
            if index in (None, ""):
                continue
            key = (_page_name(str(sample["img_id"])), int(index))
            if key in evaluator_by_gt:
                raise ValueError(f"duplicate evaluator GT table key: {key}")
            evaluator_by_gt[key] = sample

    all_layout_labels: collections.Counter[str] = collections.Counter()
    for page in page_region_rows:
        all_layout_labels.update(
            str(block.get("block_label", ""))
            for block in page.get("parsing_res_list", [])
        )
    all_recognition_labels = collections.Counter(
        str(row.get("label", "")) for row in recognition_rows
    )
    all_recognition_prompts = collections.Counter(
        str(row.get("prompt", "")) for row in recognition_rows
    )

    records = []
    for page in dataset:
        image_name = os.path.basename(page["page_info"]["image_path"])
        if image_name not in page_regions:
            raise KeyError(f"missing page-regions row for {image_name}")
        blocks = page_regions[image_name].get("parsing_res_list", [])
        gt_tables = [
            annotation
            for annotation in page.get("layout_dets", [])
            if annotation.get("category_type") == "table"
            and not annotation.get("ignore", False)
        ]
        for table_ordinal, annotation in enumerate(gt_tables):
            evaluator = evaluator_by_gt.get((image_name, table_ordinal))
            if evaluator is None:
                raise KeyError(
                    f"missing evaluator sample for {image_name} table {table_ordinal}"
                )
            gt_bbox = _poly_bbox(annotation["poly"])
            best_iou, block = _best_block(gt_bbox, blocks)
            block = block or {}
            block_id = block.get("block_id")
            trace = (
                traces.get((image_name, int(block_id)))
                if block_id is not None
                else None
            )
            trace = trace or {}
            evaluator_pred = evaluator.get("norm_pred") or evaluator.get("pred") or ""
            record = {
                "image_name": image_name,
                "table_ordinal": table_ordinal,
                "gt_bbox": list(gt_bbox),
                "best_iou": best_iou,
                "iou_band": _iou_band(best_iou),
                "best_block_id": block_id,
                "best_block_bbox": block.get("block_bbox"),
                "best_block_label": block.get("block_label", ""),
                "final_block_content_nonempty": bool(block.get("block_content")),
                "final_block_content_format": _format(block.get("block_content")),
                "recognition_request_present": bool(trace),
                "recognition_label": trace.get("label", ""),
                "recognition_prompt": trace.get("prompt", ""),
                "recognition_text_nonempty": bool(trace.get("text")),
                "recognition_text_format": _format(trace.get("text")),
                "recognition_stop_reason": trace.get("stop_reason", ""),
                "recognition_generated_tokens": trace.get(
                    "generated_tokens_including_eos"
                ),
                "evaluator_pred_nonempty": bool(evaluator_pred),
                "evaluator_pred_format": _format(evaluator_pred),
                "evaluator_pred_idx": evaluator.get("pred_idx"),
                "evaluator_edit_dist": evaluator.get("metric", {}).get(
                    "Edit_dist"
                ),
            }
            record["failure_stage"] = _failure_stage(record)
            records.append(record)

    if args.expected_tables is not None and len(records) != args.expected_tables:
        raise ValueError(f"expected {args.expected_tables} tables, got {len(records)}")
    if len(evaluator_by_gt) != len(records):
        raise ValueError(
            f"evaluator GT keys ({len(evaluator_by_gt)}) != GT tables ({len(records)})"
        )

    def count(field: str) -> dict[str, int]:
        return _counter(collections.Counter(record[field] for record in records))

    empty_records = [record for record in records if not record["evaluator_pred_nonempty"]]
    stage_paths = collections.Counter(
        (
            record["best_block_label"],
            record["recognition_label"],
            str(record["recognition_text_nonempty"]),
            str(record["evaluator_pred_nonempty"]),
        )
        for record in records
    )
    report = {
        "inputs": {
            "e2e_output": str(output_root),
            "table_result": str(args.table_result.resolve()),
        },
        "contracts": {
            "page_count": len(dataset),
            "page_regions_count": len(page_region_rows),
            "recognition_request_count": len(recognition_rows),
            "gt_table_count": len(records),
            "table_result_sample_count": len(table_results),
            "table_result_gt_key_count": len(evaluator_by_gt),
        },
        "whole_run": {
            "layout_label_histogram": _counter(all_layout_labels),
            "recognition_label_histogram": _counter(all_recognition_labels),
            "recognition_prompt_histogram": _counter(all_recognition_prompts),
        },
        "gt_table_path": {
            "iou_band": count("iou_band"),
            "best_block_label": count("best_block_label"),
            "recognition_label": count("recognition_label"),
            "recognition_prompt": count("recognition_prompt"),
            "recognition_text_nonempty": count("recognition_text_nonempty"),
            "recognition_text_format": count("recognition_text_format"),
            "final_block_content_format": count("final_block_content_format"),
            "evaluator_pred_nonempty": count("evaluator_pred_nonempty"),
            "evaluator_pred_format": count("evaluator_pred_format"),
            "failure_stage": count("failure_stage"),
            "stage_paths": [
                {
                    "count": count_value,
                    "best_block_label": path[0],
                    "recognition_label": path[1],
                    "recognition_text_nonempty": path[2] == "True",
                    "evaluator_pred_nonempty": path[3] == "True",
                }
                for path, count_value in sorted(
                    stage_paths.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
        "empty_evaluator_predictions": {
            "count": len(empty_records),
            "failure_stage": _counter(
                collections.Counter(record["failure_stage"] for record in empty_records)
            ),
            "best_block_label": _counter(
                collections.Counter(
                    record["best_block_label"] for record in empty_records
                )
            ),
            "recognition_label": _counter(
                collections.Counter(
                    record["recognition_label"] for record in empty_records
                )
            ),
            "recognition_text_format": _counter(
                collections.Counter(
                    record["recognition_text_format"] for record in empty_records
                )
            ),
        },
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, ensure_ascii=False))
    print(f"[table-pipeline-audit] saved to {args.report.resolve()}")


if __name__ == "__main__":
    main()
