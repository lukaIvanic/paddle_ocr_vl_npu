#!/usr/bin/env python3
"""OCR table-row proposals, stitch them, and record latency/token evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.model.text_prefill import parse_text_buckets
from paddleocr_vl.model.vision_prefill import parse_vision_buckets
from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.types import RecognitionRequest
from pipeline.layout_output import normalize_recognition_text
from table_row_split_lab import (
    SplitProposal,
    analyze,
    load_crop,
    read_jsonl,
    trim_blank_margin,
)


DEFAULT_REQUEST_IDS = (
    "page_000647_table_4",          # one row
    "page_000673_table_1",          # simple ruled
    "page_000635_table_0",          # borderless
    "page_000630_table_2",          # dense borderless
    "page_001595_table_box_id_1",   # dense colored
    "page_000290_table_box_id_1",   # complex multi-line
)
DEFAULT_STRATEGIES = ("ruled", "whitespace", "row_edge", "hybrid", "selected")
SUPPORTED_STRATEGIES = DEFAULT_STRATEGIES + ("whole",)
TR_PATTERN = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
TD_PATTERN = re.compile(r"<td\b([^>]*)>", re.IGNORECASE)
COLSPAN_PATTERN = re.compile(r"\bcolspan\s*=\s*['\"]?(\d+)", re.IGNORECASE)
MARKDOWN_RULE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-records",
        type=Path,
        default=Path(
            "tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/"
            "client/tables.jsonl"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Process every record in --table-records instead of the representative set.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--row-overlap-px", type=int, default=3)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair",
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _extract_stitch_fragment(value: str) -> tuple[str, str]:
    normalized = normalize_recognition_text("table", value)
    rows = TR_PATTERN.findall(normalized)
    if rows:
        return "html_rows", "".join(rows)
    markdown_lines = [
        line.strip()
        for line in normalized.splitlines()
        if "|" in line and not MARKDOWN_RULE.match(line)
    ]
    if markdown_lines:
        rows = []
        for line in markdown_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
                + "</tr>"
            )
        return "markdown_rows", "".join(rows)
    stripped = normalized.strip()
    if stripped:
        return "plain_text", f"<tr><td>{html.escape(stripped)}</td></tr>"
    return "empty", "<tr><td></td></tr>"


def stitch_rows(row_results: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    fragments: list[str] = []
    kinds: Counter[str] = Counter()
    for row in sorted(row_results, key=lambda item: int(item["row_index"])):
        kind, fragment = _extract_stitch_fragment(row["raw_text"])
        kinds[kind] += 1
        fragments.append(fragment)
    html_rows = [row for fragment in fragments for row in TR_PATTERN.findall(fragment)]
    widths = []
    for row in html_rows:
        width = 0
        for attributes in TD_PATTERN.findall(row):
            colspan = COLSPAN_PATTERN.search(attributes)
            width += int(colspan.group(1)) if colspan else 1
        widths.append(width)
    table_width = max(widths, default=0)
    distinct_widths = set(widths)
    dominant_width_rows = sum(width == table_width for width in widths)
    coherent_rectangular_width = bool(table_width) and (
        len(distinct_widths) <= 2
        and dominant_width_rows * 2 >= len(widths)
    )
    if coherent_rectangular_width:
        padded_rows = []
        for row, width in zip(html_rows, widths):
            missing = table_width - width
            if missing:
                row = re.sub(
                    r"</tr\s*>\s*$",
                    "<td></td>" * missing + "</tr>",
                    row,
                    flags=re.IGNORECASE,
                )
            padded_rows.append(row)
        fragments = padded_rows
        kinds["rows_padded_to_table_width"] = sum(
            width < table_width for width in widths
        )
    else:
        kinds["rows_padded_to_table_width"] = 0
    return "<table>" + "".join(fragments) + "</table>", dict(kinds)


def build_recognizer(args: argparse.Namespace) -> ContinuousRecognizer:
    return ContinuousRecognizer(
        model=str(args.model),
        dtype="fp16",
        decode_backend="torchair",
        decode_optimization="combined_apply_pse_sentinel",
        batch_size=args.decode_batch_size,
        cache_length=args.cache_length,
        max_new_tokens=args.cache_length,
        torchair_cache_dir=args.decode_cache_dir.resolve(),
        vision_backend="torchair",
        vision_attention="prompt_flash_attention",
        vision_promptfa_align_128=True,
        vision_mlp_intermediate_size=4352,
        vision_linear_weight_format="fractal_nz",
        vision_buckets=parse_vision_buckets(
            "256,384,512,640,768,1408,1920,2048,2944,4096"
        ),
        vision_torchair_cache_dir=args.vision_cache_dir.resolve(),
        vision_padding="bucket",
        vision_packing="greedy",
        vision_pack_target=768,
        vision_router_lookahead=32,
        text_backend="torchair",
        text_buckets=parse_text_buckets("1152"),
        text_torchair_cache_dir=args.text_cache_dir.resolve(),
        text_padding="bucket",
        text_packing="production_group",
        text_pack_buckets=(128, 256, 384, 512, 768, 1024),
        text_pack_max_members=32,
        text_packed_cache_dir=args.text_packed_cache_dir.resolve(),
        preprocessor_min_pixels=args.min_pixels,
        preprocessor_max_pixels=args.max_pixels,
    )


def crop_rows(
    image: Any,
    boundaries: tuple[int, ...],
    overlap: int,
) -> list[tuple[int, int, Any]]:
    rows = []
    for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:])):
        crop_top = max(0, top - overlap)
        crop_bottom = min(image.height, bottom + overlap)
        rows.append((crop_top, crop_bottom, image.crop((0, crop_top, image.width, crop_bottom))))
    return rows


def aggregate_result_tokens(results: list[dict[str, Any]]) -> dict[str, Any]:
    real_vision = sum(int(item["vision"].get("real_vision_tokens", 0)) for item in results)
    physical_vision = sum(int(item["vision"].get("physical_vision_tokens", 0)) for item in results)
    real_text = sum(int(item["text_prefill"].get("real_text_tokens", 0)) for item in results)
    physical_text = sum(int(item["text_prefill"].get("physical_text_tokens", 0)) for item in results)
    output = sum(int(item["generated_tokens_including_eos"]) for item in results)
    stage_s: defaultdict[str, float] = defaultdict(float)
    for item in results:
        for name, seconds in item["device_stage_s"].items():
            stage_s[name] += float(seconds)
    return {
        "rows": len(results),
        "real_vision_tokens": real_vision,
        "physical_vision_tokens": physical_vision,
        "vision_useful_fraction": real_vision / physical_vision if physical_vision else None,
        "real_text_prefill_tokens": real_text,
        "physical_text_prefill_tokens": physical_text,
        "text_useful_fraction": real_text / physical_text if physical_text else None,
        "output_tokens_including_eos": output,
        "device_stage_s": dict(sorted(stage_s.items())),
        "stop_reasons": dict(sorted(Counter(item["stop_reason"] for item in results).items())),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.decode_batch_size <= 0 or args.decode_batch_size & (args.decode_batch_size - 1):
        raise ValueError("--decode-batch-size must be a positive power of two")
    strategies = tuple(item.strip() for item in args.strategies.split(",") if item.strip())
    unknown = set(strategies) - set(SUPPORTED_STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.table_records)
    by_id = {record["request_id"]: record for record in records}
    if args.all_tables and args.request_id:
        raise ValueError("--all-tables and --request-id are mutually exclusive")
    request_ids = (
        tuple(record["request_id"] for record in records)
        if args.all_tables
        else tuple(args.request_id or DEFAULT_REQUEST_IDS)
    )
    selected = [by_id[request_id] for request_id in request_ids]

    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    if not torch.npu.is_available():
        raise RuntimeError("table row OCR lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)

    setup_started = time.perf_counter()
    recognizer = build_recognizer(args)
    setup_s = time.perf_counter() - setup_started
    output_records: list[dict[str, Any]] = []
    records_path = args.output_dir / "row_ocr_records.jsonl"
    records_path.write_text("", encoding="utf-8")

    for table_index, source in enumerate(selected, start=1):
        raw_image = load_crop(source, args.images_dir)
        image, trim_box = trim_blank_margin(raw_image)
        split_started = time.perf_counter()
        proposals = (
            {proposal.name: proposal for proposal in analyze(image)}
            if any(strategy != "whole" for strategy in strategies)
            else {}
        )
        if "whole" in strategies:
            proposals["whole"] = SplitProposal(
                name="whole",
                boundaries=(0, image.height),
                diagnostics={"source": "whole_table_crop"},
            )
        split_s = time.perf_counter() - split_started

        for strategy in strategies:
            proposal = proposals[strategy]
            row_crop_started = time.perf_counter()
            rows = crop_rows(image, proposal.boundaries, args.row_overlap_px)
            row_crop_s = time.perf_counter() - row_crop_started
            results: list[dict[str, Any]] = []
            requests = [
                RecognitionRequest(
                    request_id=f"{source['request_id']}:{strategy}:row_{row_index:04d}",
                    crop=row_image,
                    prompt="Table Recognition:",
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    source_crop_size=row_image.size,
                )
                for row_index, (_top, _bottom, row_image) in enumerate(rows)
            ]

            recognition_started = time.perf_counter()

            def emit(result: Any) -> None:
                payload = asdict(result)
                payload["raw_text"] = payload["text"]
                payload["row_index"] = int(result.request_id.rsplit("_", 1)[-1])
                top, bottom, _ = rows[payload["row_index"]]
                payload["row_y"] = [top, bottom]
                results.append(payload)

            schedule = recognizer.run(
                requests,
                schedule_id=f"row-ocr:{source['request_id']}:{strategy}",
                emit_result=emit,
            )
            recognition_wall_s = time.perf_counter() - recognition_started
            stitch_started = time.perf_counter()
            stitched, fragment_kinds = stitch_rows(results)
            stitch_s = time.perf_counter() - stitch_started
            metrics = aggregate_result_tokens(results)
            table_e2e_s = split_s + row_crop_s + recognition_wall_s + stitch_s
            record = {
                "request_id": source["request_id"],
                "strategy": strategy,
                "page_name": source["page_name"],
                "annotation_index": source["annotation_index"],
                "bbox_xyxy": source["bbox_xyxy"],
                "raw_crop_size": list(raw_image.size),
                "trim_box_in_raw_crop": list(trim_box),
                "crop_size": list(image.size),
                "boundaries": list(proposal.boundaries),
                "gt_html": source["gt_html"],
                "whole_table_prediction": source["pred_html"],
                "pred_html": stitched,
                "fragment_kinds": fragment_kinds,
                "split_diagnostics": proposal.diagnostics,
                "timing_s": {
                    "split_cpu": split_s,
                    "row_crop_cpu": row_crop_s,
                    "row_recognition_wall": recognition_wall_s,
                    "stitch_cpu": stitch_s,
                    "table_row_ocr_e2e": table_e2e_s,
                },
                "metrics": metrics,
                "decode_schedule": asdict(schedule),
                "rows": sorted(results, key=lambda item: item["row_index"]),
            }
            output_records.append(record)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
            print(
                f"table={table_index}/{len(selected)} strategy={strategy} "
                f"rows={metrics['rows']} e2e_s={table_e2e_s:.3f} "
                f"vision={metrics['real_vision_tokens']}/{metrics['physical_vision_tokens']} "
                f"output={metrics['output_tokens_including_eos']} "
                f"fragments={fragment_kinds}",
                flush=True,
            )

    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        items = [item for item in output_records if item["strategy"] == strategy]
        by_strategy[strategy] = {
            "tables": len(items),
            "rows": sum(item["metrics"]["rows"] for item in items),
            "table_row_ocr_e2e_s": sum(
                item["timing_s"]["table_row_ocr_e2e"] for item in items
            ),
            "real_vision_tokens": sum(item["metrics"]["real_vision_tokens"] for item in items),
            "physical_vision_tokens": sum(item["metrics"]["physical_vision_tokens"] for item in items),
            "real_text_prefill_tokens": sum(item["metrics"]["real_text_prefill_tokens"] for item in items),
            "physical_text_prefill_tokens": sum(item["metrics"]["physical_text_prefill_tokens"] for item in items),
            "output_tokens_including_eos": sum(item["metrics"]["output_tokens_including_eos"] for item in items),
        }

    summary = {
        "configuration": {
            "decode_batch_size": args.decode_batch_size,
            "row_overlap_px": args.row_overlap_px,
            "strategies": list(strategies),
            "request_ids": list(request_ids),
            "recognizer": recognizer.configuration(),
        },
        "setup_s": setup_s,
        "by_strategy": by_strategy,
        "records": str(records_path),
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    for strategy in strategies:
        mode_dir = args.output_dir / strategy
        mode_dir.mkdir(exist_ok=True)
        mode_records = [item for item in output_records if item["strategy"] == strategy]
        (mode_dir / "tables.jsonl").write_text(
            "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in mode_records),
            encoding="utf-8",
        )
    print(f"wrote={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
