#!/usr/bin/env python3
"""Send OmniDocBench ground-truth table crops to the crop OCR HTTP API."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=Path("/workspace/datasets/OmniDocBench/OmniDocBench.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("/workspace/datasets/OmniDocBench/images"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8765/v1/ocr")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=0,
        help=(
            "Pixels added around each GT box. The official OmniDocBench "
            "component-recognition contract uses zero."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit-pages", type=int)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--teds-timeout-s", type=float, default=30.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _bbox(poly: list[float], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    xs = poly[0::2]
    ys = poly[1::2]
    return (
        max(0, math.floor(min(xs)) - padding),
        max(0, math.floor(min(ys)) - padding),
        min(width, math.ceil(max(xs)) + padding),
        min(height, math.ceil(max(ys)) + padding),
    )


def _post_crop(api_url: str, request_id: str, crop: Image.Image, timeout_s: float) -> dict[str, Any]:
    encoded = io.BytesIO()
    crop.save(encoded, format="PNG", optimize=False)
    query = urllib.parse.urlencode({"crop_type": "table", "request_id": request_id})
    request = urllib.request.Request(
        f"{api_url}?{query}",
        data=encoded.getvalue(),
        method="POST",
        headers={"Content-Type": "image/png"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc


def _read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            completed[record["request_id"]] = record
    return completed


def _score(
    output: Path,
    score_output: Path,
    evaluator_root: Path,
    *,
    workers: int,
    timeout_s: float,
) -> None:
    sys.path.insert(0, str(evaluator_root.resolve()))
    from src.core.preprocess import normalized_table
    from src.metrics.cal_metric import _evaluate_teds_pair_with_timeout

    records = list(_read_completed(output).values())
    page_scores: dict[str, list[float]] = {}
    page_structure_scores: dict[str, list[float]] = {}
    scored: list[dict[str, Any]] = []
    started = time.perf_counter()

    def document(table_html: str) -> str:
        table_html = normalized_table(table_html, "html")
        if "<html" in table_html.lower():
            return table_html
        return f"<html><body>{table_html}</body></html>"

    pairs: list[tuple[int, dict[str, Any], str, str]] = []
    for index, record in enumerate(records):
        pred_html = document(record["pred_html"])
        gt_html = document(record["gt_html"])
        pairs.append((index, record, pred_html, gt_html))

    results: dict[int, tuple[float, float, str | None]] = {}

    def evaluate(pair: tuple[int, dict[str, Any], str, str]) -> tuple[int, float, float, str | None]:
        index, _record, pred_html, gt_html = pair
        score, structure, error = _evaluate_teds_pair_with_timeout(
            pred_html,
            gt_html,
            timeout_s,
        )
        return index, float(score), float(structure), error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(evaluate, pair) for pair in pairs]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, score, structure, error = future.result()
            results[index] = (score, structure, error)
            if completed % 25 == 0 or completed == len(records):
                print(
                    f"scored={completed}/{len(records)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    timeout_count = 0
    error_count = 0
    for index, record in enumerate(records):
        score, structure, error = results[index]
        if error:
            if str(error).startswith("timeout:"):
                timeout_count += 1
            else:
                error_count += 1
        page_scores.setdefault(record["page_name"], []).append(score)
        page_structure_scores.setdefault(record["page_name"], []).append(structure)
        scored.append(
            {
                "request_id": record["request_id"],
                "page_name": record["page_name"],
                "TEDS": score,
                "TEDS_structure_only": structure,
                "error": error,
            }
        )
    sample_scores = [item["TEDS"] for item in scored]
    sample_structure = [item["TEDS_structure_only"] for item in scored]
    page_means = {
        page: sum(values) / len(values) for page, values in page_scores.items()
    }
    page_structure_means = {
        page: sum(values) / len(values)
        for page, values in page_structure_scores.items()
    }
    result = {
        "table_count": len(scored),
        "table_page_count": len(page_means),
        "sample_TEDS": sum(sample_scores) / len(sample_scores),
        "sample_TEDS_structure_only": sum(sample_structure) / len(sample_structure),
        "page_TEDS": sum(page_means.values()) / len(page_means),
        "page_TEDS_structure_only": sum(page_structure_means.values())
        / len(page_structure_means),
        "teds_timeout_s": timeout_s,
        "teds_timeout_count": timeout_count,
        "teds_error_count": error_count,
        "per_page_TEDS": page_means,
        "per_table": scored,
    }
    score_output.parent.mkdir(parents=True, exist_ok=True)
    score_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"SCORED page_TEDS={result['page_TEDS']:.6f} "
        f"sample_TEDS={result['sample_TEDS']:.6f} output={score_output}",
        flush=True,
    )


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _write_summary(
    output: Path,
    summary_output: Path,
    *,
    generation_wall_s: float,
    score_output: Path | None,
) -> None:
    records = list(_read_completed(output).values())
    http = [float(record["http_wall_s"]) for record in records]
    worker = [float(record["worker_wall_s"]) for record in records]
    overhead = [max(0.0, h - w) for h, w in zip(http, worker)]
    metrics = None
    if score_output is not None:
        metrics = json.loads(score_output.read_text(encoding="utf-8"))
    summary = {
        "tables": len(records),
        "unique_requests": len({record["request_id"] for record in records}),
        "generation_wall_s": generation_wall_s,
        "tables_per_s": len(records) / generation_wall_s,
        "http_latency_s": _distribution(http),
        "worker_latency_s": _distribution(worker),
        "http_wrapper_overhead_s": {
            "sum": sum(overhead),
            **_distribution(overhead),
        },
        "input_tokens": sum(int(record["input_tokens"]) for record in records),
        "output_tokens_including_eos": sum(
            int(record["output_tokens"]) for record in records
        ),
        "stop_reasons": dict(Counter(record["stop_reason"] for record in records)),
        "metrics": metrics,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SUMMARY output={summary_output}", flush=True)


def main() -> None:
    args = parse_args()
    pages = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    selected = pages[args.offset :]
    if args.limit_pages is not None:
        selected = selected[: args.limit_pages]
    jobs: list[tuple[int, dict[str, Any], int, dict[str, Any]]] = []
    for page_index, page in enumerate(selected, start=args.offset):
        for annotation_index, annotation in enumerate(page.get("layout_dets") or []):
            if annotation.get("ignore") or annotation.get("category_type") != "table":
                continue
            jobs.append((page_index, page, annotation_index, annotation))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed(args.output) if args.resume else {}
    mode = "a" if args.resume else "w"
    started = time.perf_counter()
    done = 0
    with args.output.open(mode, encoding="utf-8") as output:
        for page_index, page, annotation_index, annotation in jobs:
            page_name = Path(page["page_info"]["image_path"]).name
            annotation_id = str(annotation.get("anno_id", annotation_index))
            request_id = f"page_{page_index:06d}_table_{annotation_id}"
            if request_id in completed:
                done += 1
                continue
            image_path = args.images_dir / page_name
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            bbox = _bbox(annotation["poly"], image.width, image.height, args.crop_padding)
            crop = image.crop(bbox)
            response = _post_crop(args.api_url, request_id, crop, args.timeout_s)
            record = {
                "request_id": request_id,
                "page_index": page_index,
                "page_name": page_name,
                "annotation_index": annotation_index,
                "anno_id": annotation.get("anno_id"),
                "bbox_xyxy": list(bbox),
                "crop_size": list(crop.size),
                "gt_html": annotation.get("html") or annotation.get("text") or "",
                "pred_html": response["text"],
                "stop_reason": response["stop_reason"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["generated_tokens_including_eos"],
                "worker_wall_s": response["worker_wall_s"],
                "http_wall_s": response["http_wall_s"],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            done += 1
            elapsed = time.perf_counter() - started
            print(
                f"completed={done}/{len(jobs)} page={page_index} "
                f"elapsed_s={elapsed:.1f} tables_per_s={done / elapsed:.3f}",
                flush=True,
            )
    print(f"DONE tables={len(jobs)} output={args.output}", flush=True)
    generation_wall_s = time.perf_counter() - started
    if args.score_output is not None:
        _score(
            args.output,
            args.score_output,
            args.evaluator_root,
            workers=args.teds_workers,
            timeout_s=args.teds_timeout_s,
        )
    if args.summary_output is not None:
        _write_summary(
            args.output,
            args.summary_output,
            generation_wall_s=generation_wall_s,
            score_output=args.score_output,
        )


if __name__ == "__main__":
    main()
