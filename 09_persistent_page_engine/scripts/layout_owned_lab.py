#!/usr/bin/env python3
"""Run the self-contained layout frontend and stop at RecognitionRequest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.types import RecognitionRequest
from pipeline.layout_frontend import (
    LAYOUT_CONV_WEIGHT_FORMATS,
    OwnedLayoutFrontend,
)
from pipeline.layout_mask_guard import install_layout_mask_guard
from pipeline.omnidocbench_defaults import (
    OMNIDOCBENCH_PAGE_COUNT,
)
from utils.timeline import TimelineRecorder


DEFAULT_DATASET_JSON = Path(
    "/workspace/datasets/OmniDocBench/OmniDocBench.json"
)
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
_REQUEST_ID = re.compile(r"^page_(\d+)_block_(\d+)$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
    )
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=DEFAULT_LAYOUT_MODEL,
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--preprocessor-min-pixels", type=int)
    parser.add_argument("--reference-requests", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--conv-weight-format",
        choices=LAYOUT_CONV_WEIGHT_FORMATS,
        default="native",
    )
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be non-negative and --limit positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pages(
    dataset_json: Path,
    images_dir: Path,
    offset: int,
    limit: int,
) -> list[Path]:
    annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
    if len(annotations) != OMNIDOCBENCH_PAGE_COUNT:
        raise ValueError(
            f"expected {OMNIDOCBENCH_PAGE_COUNT} pages, got {len(annotations)}"
        )
    selected = annotations[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"requested {limit} pages at {offset}, got {len(selected)}"
        )
    paths = [
        images_dir / Path(page["page_info"]["image_path"]).name
        for page in selected
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]}")
    return paths


def _request_record(
    request: RecognitionRequest,
    request_index: int,
    image_paths: list[Path],
) -> dict[str, Any]:
    match = _REQUEST_ID.fullmatch(request.request_id)
    if match is None:
        raise ValueError(f"unexpected request id: {request.request_id!r}")
    page_index = int(match.group(1))
    crop_bytes = request.crop.tobytes()
    crop_hash = hashlib.sha256(
        (
            f"{request.crop.mode}:{request.crop.width}:{request.crop.height}:"
        ).encode()
        + crop_bytes
    ).hexdigest()
    return {
        "request_index": request_index,
        "request_id": request.request_id,
        "page_index": page_index,
        "page_image": image_paths[page_index].name,
        "block_index": int(match.group(2)),
        "prompt": request.prompt,
        "skip_special_tokens": bool(request.skip_special_tokens),
        "min_pixels": request.min_pixels,
        "max_pixels": request.max_pixels,
        "crop_mode": request.crop.mode,
        "crop_size": [request.crop.width, request.crop.height],
        "crop_nbytes": len(crop_bytes),
        "crop_sha256": crop_hash,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    model_dir = args.layout_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _load_pages(
        dataset_json,
        images_dir,
        args.offset,
        args.limit,
    )

    install_layout_mask_guard()
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("owned layout lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    timeline = TimelineRecorder(enabled=args.timeline)
    timeline.reset(
        {
            "kind": "owned_layout_frontend_lab",
            "pages": len(image_paths),
            "paddlex_imported": False,
            "ocr_requests_executed": 0,
            "conv_weight_format": args.conv_weight_format,
        }
    )

    setup_started = time.perf_counter()
    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        timeline=timeline,
        graph_capture=True,
        device_stage_timing=True,
        conv_weight_format=args.conv_weight_format,
    )
    setup_s = time.perf_counter() - setup_started

    requests: list[RecognitionRequest] = []
    stage_totals: defaultdict[str, float] = defaultdict(float)
    page_statistics: list[dict[str, Any]] = []
    frontend_started = time.perf_counter()
    for ordinal, path in enumerate(image_paths):
        page = frontend.prepare_page(
            path,
            ordinal,
            min_pixels=args.preprocessor_min_pixels,
        )
        requests.extend(page.requests)
        for name, seconds in page.timing_s.items():
            stage_totals[name] += float(seconds)
        page_statistics.append(
            {
                "page": path.name,
                **page.statistics,
                "timing_s": page.timing_s,
            }
        )
    frontend_wall_s = time.perf_counter() - frontend_started

    records = [
        _request_record(request, index, image_paths)
        for index, request in enumerate(requests)
    ]
    requests_path = output_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    timeline_json = output_dir / "timeline_trace.json"
    timeline_html = output_dir / "timeline.html"
    if args.timeline:
        timeline.write_json(timeline_json)
        timeline.write_html(timeline_html)

    reference = None
    if args.reference_requests is not None:
        reference_path = args.reference_requests.expanduser().resolve()
        reference = {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "exact": reference_path.read_bytes() == requests_path.read_bytes(),
        }

    page_counts = Counter(record["page_image"] for record in records)
    counts = [page_counts.get(path.name, 0) for path in image_paths]
    summary = {
        "implementation": "owned_pp_doclayout_v3_frontend",
        "paddlex_dependency": False,
        "dataset_json": str(dataset_json),
        "dataset_sha256": _sha256(dataset_json),
        "images_dir": str(images_dir),
        "offset": args.offset,
        "count": len(image_paths),
        "images": [path.name for path in image_paths],
        "layout_model": str(model_dir),
        "layout_model_backend": "transformers_npugraph",
        "conv_weight_format": args.conv_weight_format,
        "fractal_z_conv_weight_count": (
            frontend.fractal_z_conv_weight_count
        ),
        "setup_s": setup_s,
        "frontend_wall_s": frontend_wall_s,
        "pages_per_s": len(image_paths) / frontend_wall_s,
        "s_per_page": frontend_wall_s / len(image_paths),
        "stage_totals_s": dict(sorted(stage_totals.items())),
        "requests": len(records),
        "requests_per_page": {
            "min": min(counts, default=0),
            "mean": sum(counts) / len(counts),
            "max": max(counts, default=0),
        },
        "prompt_counts": dict(
            sorted(Counter(record["prompt"] for record in records).items())
        ),
        "crop_pixels": sum(
            int(record["crop_size"][0]) * int(record["crop_size"][1])
            for record in records
        ),
        "crop_bytes": sum(
            int(record["crop_nbytes"]) for record in records
        ),
        "request_manifest": str(requests_path),
        "request_manifest_sha256": _sha256(requests_path),
        "reference_comparison": reference,
        "layout_mask_rectangle_fast_path": (
            frontend.mask_fast_path.snapshot()
        ),
        "page_statistics": page_statistics,
        "timeline": {
            "enabled": bool(args.timeline),
            "event_count": (
                len(timeline.events()) if args.timeline else 0
            ),
            "trace_json": str(timeline_json) if args.timeline else None,
            "html": str(timeline_html) if args.timeline else None,
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"summary={summary_path}", flush=True)
    if reference is not None and not reference["exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
