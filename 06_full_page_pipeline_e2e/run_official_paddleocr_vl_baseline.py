#!/usr/bin/env python3
"""Run the official PaddleOCR-VL pipeline as an experiment-6 baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_page_pipeline_e2e import clean_json, load_pages, tok_per_s  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/data/OmniDocBench"))
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--layout-detection-model-name", default="PP-DocLayoutV3")
    parser.add_argument("--layout-detection-model-dir", type=Path, default=None)
    parser.add_argument("--vl-rec-model-dir", type=Path, default=None)
    parser.add_argument("--vl-rec-backend", default=None)
    parser.add_argument("--vl-rec-server-url", default=None)
    parser.add_argument("--vl-rec-api-model-name", default=None)
    parser.add_argument("--vl-rec-api-key", default=None)
    parser.add_argument("--vl-rec-max-concurrency", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--use-queues", default="true", choices=["true", "false", "none"])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def bool_or_none(value: str) -> bool | None:
    if value == "none":
        return None
    return value == "true"


def result_json(result: Any) -> dict[str, Any]:
    if hasattr(result, "json"):
        attr = getattr(result, "json")
        raw = attr() if callable(attr) else attr
        return clean_json(raw)
    if isinstance(result, dict):
        return clean_json(result)
    raise TypeError(f"official PaddleOCR-VL result has no .json attribute: {type(result).__name__}")


def main() -> None:
    args = parse_args()
    pages = load_pages(args.dataset_dir, page_start=int(args.page_start), num_pages=int(args.num_pages))

    try:
        from paddleocr import PaddleOCRVL
    except Exception as exc:
        raise RuntimeError(
            "Could not import official PaddleOCRVL. This baseline requires a runnable "
            "PaddleOCR/PaddleX/PaddlePaddle environment."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "pipeline_version": "v1.6",
        "layout_detection_model_name": args.layout_detection_model_name,
        "layout_detection_model_dir": str(args.layout_detection_model_dir) if args.layout_detection_model_dir else None,
        "vl_rec_model_dir": str(args.vl_rec_model_dir) if args.vl_rec_model_dir else None,
        "vl_rec_backend": args.vl_rec_backend,
        "vl_rec_server_url": args.vl_rec_server_url,
        "vl_rec_api_model_name": args.vl_rec_api_model_name,
        "vl_rec_api_key": args.vl_rec_api_key,
        "vl_rec_max_concurrency": args.vl_rec_max_concurrency,
        "use_queues": bool_or_none(args.use_queues),
    }
    if args.device is not None:
        init_kwargs["device"] = args.device
    if args.engine is not None:
        init_kwargs["engine"] = args.engine
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

    start = time.perf_counter()
    pipeline = PaddleOCRVL(**init_kwargs)
    init_s = time.perf_counter() - start

    image_paths = [str(page.image_path) for page in pages]
    start = time.perf_counter()
    results = list(
        pipeline.predict(
            image_paths,
            use_layout_detection=True,
            max_new_tokens=int(args.max_new_tokens),
            use_queues=bool_or_none(args.use_queues),
        )
    )
    predict_s = time.perf_counter() - start
    if hasattr(pipeline, "close"):
        pipeline.close()

    json_results = [result_json(result) for result in results]
    parsing_counts = []
    label_counts: dict[str, int] = {}
    for result in json_results:
        root = result.get("res", result)
        parsing = root.get("parsing_res_list", []) if isinstance(root, dict) else []
        parsing_counts.append(len(parsing) if isinstance(parsing, list) else 0)
        if isinstance(parsing, list):
            for block in parsing:
                label = str(block.get("block_label", "unknown"))
                label_counts[label] = label_counts.get(label, 0) + 1

    output = {
        "experiment": "06_official_paddleocr_vl_baseline",
        "scope": "official PaddleOCRVL pipeline, not our optimized hot-swap recognizer",
        "page_start": int(args.page_start),
        "page_count": int(len(pages)),
        "init_kwargs": clean_json(init_kwargs),
        "timing_s": {
            "pipeline_init": float(init_s),
            "predict": float(predict_s),
        },
        "throughput": {
            "pages_per_s_predict": tok_per_s(len(pages), predict_s),
            "seconds_per_page_predict": predict_s / float(len(pages)) if pages else None,
        },
        "result_summary": {
            "result_count": int(len(json_results)),
            "parsing_blocks_total": int(sum(parsing_counts)),
            "parsing_blocks_per_page": parsing_counts,
            "label_counts": dict(sorted(label_counts.items())),
        },
        "results_sample": json_results[: min(2, len(json_results))],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
