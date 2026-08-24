#!/usr/bin/env python3
"""Materialize UniRec Markdown/JSON from deferred low-memory artifacts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


# The shared page-assembly module imports torch, but this short-lived writer
# performs no accelerator work. Do not make it depend on a sourced CANN shell.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--frontend-payloads", type=Path, required=True)
    parser.add_argument("--recognition-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import cv2
    import numpy as np

    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools import infer_doc_onnx

    from run_low_memory_unirec import _payload_to_pages
    from run_opendoc_batched_unirec import assemble_page

    payloads: list[dict[str, Any]] = json.loads(
        args.frontend_payloads.read_text(encoding="utf-8")
    )
    pages, records = _payload_to_pages(payloads)
    crops = {str(record["request_id"]): record["crop"] for record in records}
    seen = set()
    with args.recognition_trace.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id not in crops:
                raise RuntimeError(f"unknown request_id in trace: {request_id}")
            if request_id in seen:
                raise RuntimeError(f"duplicate request_id in trace: {request_id}")
            seen.add(request_id)
            crops[request_id].result = row
    missing = sorted(set(crops) - seen)
    if missing:
        raise RuntimeError(f"recognition trace is missing {len(missing)} crops")

    pipeline = infer_doc_onnx.OpenDocONNX.__new__(infer_doc_onnx.OpenDocONNX)
    pipeline.use_layout_detection = False
    pipeline.use_chart_recognition = True
    pipeline.markdown_ignore_labels = [
        "number",
        "footnote",
        "header",
        "footer",
        "aside_text",
        "footer_image",
        "header_image",
        "chart",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Path(os.devnull).open("w", encoding="utf-8") as devnull:
        for index, page in enumerate(pages, start=1):
            with redirect_stdout(devnull):
                result = assemble_page(
                    page=page,
                    pipeline=pipeline,
                    infer_doc_onnx=infer_doc_onnx,
                )
                if any(
                    bool(row.get("is_image", False))
                    for row in result["recognition_results"]
                ):
                    page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
                    if page_image is None:
                        raise RuntimeError(
                            f"failed to reload page {result['input_path']}"
                        )
                    result["_page_image"] = page_image
                else:
                    result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
                pipeline.save_to_json(result, str(args.output_dir))
                pipeline.save_to_markdown(result, str(args.output_dir))
            for crop in page.crops:
                crop.result = None
            if index % args.progress_every == 0 or index == len(pages):
                print(
                    "UNIREC_DEFERRED_WRITE_PROGRESS "
                    f"pages={index}/{len(pages)}",
                    flush=True,
                )
    summary = {
        "schema": "unirec_deferred_output_write_v1",
        "status": "pass",
        "page_count": len(pages),
        "crop_count": len(records),
        "wall_s": time.perf_counter() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    summary_path = args.output_dir / "deferred_write_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("UNIREC_DEFERRED_WRITE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
