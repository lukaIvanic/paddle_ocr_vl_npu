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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.types import RecognitionRequest
from pipeline.layout_frontend import OwnedLayoutFrontend
from pipeline.layout_mask_guard import install_layout_mask_guard
from pipeline.omnidocbench_defaults import (
    OMNIDOCBENCH_PAGE_COUNT,
)
from utils.timeline import TimelineRecorder
from utils.input_fingerprints import fingerprint_pil_image


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
    parser.add_argument(
        "--device",
        default="npu",
        choices=("npu", "cpu"),
        help=(
            "Run layout on NPU or CPU. CPU disables NPU graph capture and "
            "device-event timing."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--preprocessor-min-pixels", type=int)
    parser.add_argument("--reference-requests", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the fixed detector core when layout runs on NPU.",
    )
    parser.add_argument(
        "--layout-indexput-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build PP-DocLayoutV3 spatial-shape metadata without NPU "
            "scalar IndexPut writes."
        ),
    )
    parser.add_argument(
        "--model-backend",
        choices=("transformers", "owned"),
        default="transformers",
        help=(
            "Use the Transformers oracle or the project-owned eager "
            "PP-DocLayoutV3 model. The owned backend requires "
            "--no-graph-capture."
        ),
    )
    parser.add_argument(
        "--model-dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Layout-model inference dtype; FP16 is an NPU experiment.",
    )
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be non-negative and --limit positive")
    if args.model_backend == "owned" and args.graph_capture:
        parser.error(
            "--model-backend owned requires --no-graph-capture"
        )
    if args.model_dtype == "fp16" and args.device != "npu":
        parser.error("--model-dtype fp16 currently requires --device npu")
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
    crop = fingerprint_pil_image(request.crop)
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
        "crop_nbytes": crop["nbytes"],
        "crop_sha256": crop["sha256"],
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
    device = torch.device("cpu" if args.device == "cpu" else "npu:0")
    if device.type == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("owned layout lab requires an NPU")
        torch.npu.set_compile_mode(jit_compile=False)
    timeline = TimelineRecorder(enabled=args.timeline)
    timeline.reset(
        {
            "kind": "owned_layout_frontend_lab",
            "pages": len(image_paths),
            "workers": args.workers,
            "paddlex_imported": False,
            "ocr_requests_executed": 0,
            "model_backend": args.model_backend,
        }
    )

    memory_before_setup = (
        int(torch.npu.memory_allocated(device))
        if device.type == "npu"
        else 0
    )
    setup_started = time.perf_counter()
    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        timeline=timeline,
        graph_capture=device.type == "npu" and args.graph_capture,
        device_stage_timing=device.type == "npu",
        npu_indexput_compat=args.layout_indexput_compat,
        model_backend=args.model_backend,
        model_dtype=(
            torch.float16
            if args.model_dtype == "fp16"
            else torch.float32
        ),
    )
    setup_s = time.perf_counter() - setup_started
    memory_after_setup = (
        int(torch.npu.memory_allocated(device))
        if device.type == "npu"
        else 0
    )

    def page_record(page: Any) -> tuple[
        int,
        list[RecognitionRequest],
        dict[str, float],
        dict[str, Any],
    ]:
        return (
            page.ordinal,
            page.requests,
            page.timing_s,
            page.statistics,
        )

    def prepare_serial() -> list[Any]:
        return [
            page_record(
                frontend.prepare_page(
                    path,
                    ordinal,
                    min_pixels=args.preprocessor_min_pixels,
                )
            )
            for ordinal, path in enumerate(image_paths)
        ]

    def prepare_with_decode_prefetch() -> list[Any]:
        prepared = []
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="layout-input",
        ) as executor:
            pending = executor.submit(
                frontend.decode_page,
                image_paths[0],
                0,
            )
            for ordinal in range(len(image_paths)):
                decoded = pending.result()
                if ordinal + 1 < len(image_paths):
                    pending = executor.submit(
                        frontend.decode_page,
                        image_paths[ordinal + 1],
                        ordinal + 1,
                    )
                prepared.append(
                    page_record(
                        frontend.prepare_decoded_page(
                            decoded,
                            min_pixels=args.preprocessor_min_pixels,
                        )
                    )
                )
        return prepared

    frontend_started = time.perf_counter()
    prepared_pages = (
        prepare_serial()
        if args.workers == 1
        else prepare_with_decode_prefetch()
    )
    frontend_wall_s = time.perf_counter() - frontend_started

    requests: list[RecognitionRequest] = []
    stage_totals: defaultdict[str, float] = defaultdict(float)
    page_statistics: list[dict[str, Any]] = []
    for ordinal, page_requests, page_timing, page_summary in prepared_pages:
        path = image_paths[ordinal]
        requests.extend(page_requests)
        for name, seconds in page_timing.items():
            stage_totals[name] += float(seconds)
        page_statistics.append(
            {
                "page": path.name,
                **page_summary,
                "timing_s": page_timing,
            }
        )

    mask_fast_path = frontend.mask_fast_path.snapshot()

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
        "layout_model_backend": (
            f"{args.model_backend}_"
            f"{'npugraph' if frontend.graph_capture else 'eager'}_"
            f"{device.type}"
        ),
        "layout_model_dtype": str(frontend.model_dtype),
        "device": str(device),
        "graph_capture": bool(frontend.graph_capture),
        "npu_indexput_compat": bool(frontend.npu_indexput_compat),
        "workers": args.workers,
        "worker_strategy": (
            "serial" if args.workers == 1 else "one_page_decode_prefetch"
        ),
        "setup_s": setup_s,
        "setup_by_worker_s": [frontend.setup_s],
        "npu_memory_allocated_bytes": {
            "before_setup": memory_before_setup,
            "after_setup": memory_after_setup,
            "setup_delta": memory_after_setup - memory_before_setup,
        },
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
        "layout_mask_rectangle_fast_path": mask_fast_path,
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
