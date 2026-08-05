#!/usr/bin/env python3
"""Send OmniDocBench pages to the persistent full-page HTTP API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8766/v1/pages")
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
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--http-workers", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _post(url: str, image_path: Path, timeout_s: float) -> dict[str, Any]:
    request_id = image_path.name
    query = urllib.parse.urlencode(
        {"request_id": request_id, "filename": image_path.name}
    )
    request = urllib.request.Request(
        f"{url}?{query}",
        data=image_path.read_bytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.dataset_json.expanduser().resolve().read_text())
    subset = annotations[args.offset : args.offset + args.limit]
    if len(subset) != args.limit:
        raise ValueError(f"requested {args.limit} pages, got {len(subset)}")
    images_dir = args.images_dir.expanduser().resolve()
    paths = [
        images_dir / Path(item["page_info"]["image_path"]).name
        for item in subset
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")
    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    responses_dir = output_dir / "responses"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    completed = 0
    response_s: list[float] = []
    with ThreadPoolExecutor(max_workers=args.http_workers) as executor:
        futures = {
            executor.submit(_post, args.api_url, path, args.timeout_s): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            payload = future.result()
            (predictions_dir / f"{path.stem}.md").write_text(
                payload["markdown"],
                encoding="utf-8",
            )
            (responses_dir / f"{path.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed += 1
            response_s.append(float(payload["http_wall_s"]))
            elapsed = time.perf_counter() - started
            print(
                f"PAGE_API_PROGRESS completed={completed}/{len(paths)} "
                f"elapsed_s={elapsed:.3f} pages_per_s={completed / elapsed:.3f}",
                flush=True,
            )
    wall_s = time.perf_counter() - started
    summary: dict[str, Any] = {
        "offset": args.offset,
        "count": len(paths),
        "wall_s": wall_s,
        "pages_per_s": len(paths) / wall_s,
        "mean_response_s": sum(response_s) / len(response_s),
        "max_response_s": max(response_s),
        "predictions_dir": str(predictions_dir),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PAGE_API_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
