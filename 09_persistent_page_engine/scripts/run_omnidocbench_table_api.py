#!/usr/bin/env python3
"""Send OmniDocBench ground-truth table crops to the crop OCR HTTP API."""

from __future__ import annotations

import argparse
import io
import json
import math
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
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit-pages", type=int)
    parser.add_argument("--timeout-s", type=float, default=900.0)
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


if __name__ == "__main__":
    main()
