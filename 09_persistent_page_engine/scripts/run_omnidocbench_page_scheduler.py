#!/usr/bin/env python3
"""Benchmark the unified page scheduler directly, without HTTP or IPC."""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from serve_page_ocr_api import (  # noqa: E402
    _worker_main,
    parse_args as parse_server_args,
    worker_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/OmniDocBench.json"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _metric_summary(service: dict[str, Any]) -> dict[str, Any]:
    modes = service["device_modes"]
    schedule = service["schedule"]
    stages = modes["device_stage_s"]
    tokens = modes["prefill_tokens"]
    decode_wall = float(schedule["timing_s"]["continuous_decode_wall"])
    vision_wall = float(stages["vision_prefill"])
    text_wall = float(stages["text_prefill"])
    raw_slots = int(schedule["raw_decode_token_slots"])
    active_slots = int(schedule["active_decode_token_slots"])
    return {
        "layout_mode_s_per_page": (
            float(modes["mode_wall_s"].get("layout", 0.0))
            / int(modes["pages_completed"])
        ),
        "layout_page_total_s": modes["layout_page_total_s"],
        "vision_prefill": {
            "device_s": vision_wall,
            "useful_tokens": int(tokens["useful_vision"]),
            "physical_tokens": int(tokens["physical_vision"]),
            "useful_tok_per_s": int(tokens["useful_vision"]) / vision_wall,
            "physical_tok_per_s": int(tokens["physical_vision"]) / vision_wall,
        },
        "text_prefill": {
            "device_s": text_wall,
            "useful_tokens": int(tokens["useful_text"]),
            "physical_tokens": int(tokens["physical_text"]),
            "useful_tok_per_s": int(tokens["useful_text"]) / text_wall,
            "physical_tok_per_s": int(tokens["physical_text"]) / text_wall,
        },
        "decode": {
            "wall_s": decode_wall,
            "graph_calls": int(schedule["graph_calls"]),
            "raw_token_slots": raw_slots,
            "active_token_slots": active_slots,
            "effective_tokens": int(schedule["effective_decode_tokens"]),
            "raw_tok_per_s": raw_slots / decode_wall,
            "active_tok_per_s": active_slots / decode_wall,
            "effective_tok_per_s": (
                int(schedule["effective_decode_tokens"]) / decode_wall
            ),
            "active_slot_fraction": active_slots / raw_slots,
        },
    }


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.dataset_json.expanduser().resolve().read_text())
    subset = annotations[args.offset : args.offset + args.limit]
    if len(subset) != args.limit:
        raise ValueError(f"requested {args.limit} pages, got {len(subset)}")
    images_dir = args.images_dir.expanduser().resolve()
    image_paths = [
        images_dir / Path(item["page_info"]["image_path"]).name
        for item in subset
    ]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")

    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    responses_dir = output_dir / "responses"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue[Any] = queue.Queue()
    results: queue.Queue[dict[str, Any]] = queue.Queue()
    for path in image_paths:
        jobs.put(
            {
                "request_id": path.name,
                "image_path": str(path),
                "submitted_monotonic_s": None,
            }
        )
    jobs.put(None)

    server_args = parse_server_args([])
    _worker_main(jobs, results, worker_config(server_args))

    configuration: dict[str, Any] | None = None
    service: dict[str, Any] | None = None
    page_results: dict[str, dict[str, Any]] = {}
    while not results.empty():
        message = results.get_nowait()
        kind = message["kind"]
        if kind == "ready":
            configuration = message["configuration"]
        elif kind == "result":
            if not message["ok"]:
                raise RuntimeError(message["error"])
            payload = message["payload"]
            page_results[message["request_id"]] = payload
        elif kind == "service_summary":
            service = message["payload"]
        elif kind == "service_error":
            raise RuntimeError(message["traceback"])

    if configuration is None or service is None:
        raise RuntimeError("direct scheduler did not produce a complete summary")
    if len(page_results) != len(image_paths):
        raise RuntimeError(
            f"expected {len(image_paths)} page results, got {len(page_results)}"
        )
    for path in image_paths:
        payload = page_results[path.name]
        (predictions_dir / f"{path.stem}.md").write_text(
            payload["markdown"], encoding="utf-8"
        )
        (responses_dir / f"{path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "mode": "direct_unified_page_scheduler",
        "offset": args.offset,
        "count": len(image_paths),
        "configuration": configuration,
        "service": service,
        "metrics": _metric_summary(service),
        "predictions_dir": str(predictions_dir),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
