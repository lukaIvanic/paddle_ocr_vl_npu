#!/usr/bin/env python3
"""Run the faithful PaddleX frontend and stop at RecognitionRequest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.types import RecognitionRequest
from pipeline.layout_mask_guard import install_layout_mask_guard
from pipeline.layout_runtime_optimization import (
    install_layout_runtime_optimizations,
)
from pipeline.omnidocbench_defaults import (
    OMNIDOCBENCH_DECODE_BATCH_SIZE,
    OMNIDOCBENCH_MAX_NEW_TOKENS,
    OMNIDOCBENCH_PAGE_COUNT,
)
from pipeline.paddlex_page_bridge import PaddleXPageBridge
from utils.timeline import TimelineRecorder


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_PADDLEOCR_SOURCE = Path("/workspace/repos/vllm_paddle_ocr/PaddleOCR")
_REQUEST_ID = re.compile(r"^page_(\d+)_block_(\d+)$")


class _CaptureComplete(Exception):
    pass


class _LocalOpenCVImageReader:
    """Exact local equivalent of PaddleX's BGR OpenCV path, with split timing."""

    def __init__(self, timeline: TimelineRecorder) -> None:
        self.format = "BGR"
        self.timeline = timeline
        self.calls = 0
        self.path_inputs = 0
        self.array_inputs = 0
        self.compressed_bytes = 0
        self.file_read_ns = 0
        self.byte_view_ns = 0
        self.decode_ns = 0
        self.wall_ns = 0

    def __call__(self, images: Iterable[str | np.ndarray]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for image in images:
            call_index = self.calls
            self.calls += 1
            flow_id = f"page:{call_index}"
            item_started_ns = time.perf_counter_ns()
            if isinstance(image, np.ndarray):
                self.array_inputs += 1
                outputs.append(image)
                self.wall_ns += time.perf_counter_ns() - item_started_ns
                continue
            if not isinstance(image, str):
                raise TypeError(
                    "local OpenCV reader requires a path string or NumPy array, "
                    f"got {type(image).__name__}"
                )

            self.path_inputs += 1
            read_started_ns = time.perf_counter_ns()
            with open(image, "rb") as handle:
                compressed = handle.read()
            read_finished_ns = time.perf_counter_ns()
            self.file_read_ns += read_finished_ns - read_started_ns
            self.compressed_bytes += len(compressed)
            self.timeline.record_span(
                "Page input",
                "Read compressed page bytes",
                read_started_ns,
                read_finished_ns,
                flow_id=flow_id,
                event_type="io",
                args={"bytes": len(compressed)},
            )

            view_started_ns = time.perf_counter_ns()
            encoded = np.frombuffer(compressed, np.uint8)
            view_finished_ns = time.perf_counter_ns()
            self.byte_view_ns += view_finished_ns - view_started_ns

            decode_started_ns = time.perf_counter_ns()
            decoded = cv2.imdecode(encoded, flags=cv2.IMREAD_COLOR)
            decode_finished_ns = time.perf_counter_ns()
            self.decode_ns += decode_finished_ns - decode_started_ns
            self.timeline.record_span(
                "Page input",
                "Decode compressed page image",
                decode_started_ns,
                decode_finished_ns,
                flow_id=flow_id,
                event_type="work",
                args={
                    "encoded_bytes": len(compressed),
                    "width": int(decoded.shape[1]) if decoded is not None else None,
                    "height": int(decoded.shape[0]) if decoded is not None else None,
                },
            )
            if decoded is None:
                raise RuntimeError(f"OpenCV failed to decode page image: {image}")
            outputs.append(decoded)
            self.wall_ns += time.perf_counter_ns() - item_started_ns
        return outputs

    def summary(self) -> dict[str, Any]:
        seconds = lambda value: value / 1_000_000_000
        return {
            "implementation": "local_open_read_frombuffer_cv2_imdecode_bgr",
            "calls": self.calls,
            "path_inputs": self.path_inputs,
            "array_inputs": self.array_inputs,
            "compressed_bytes": self.compressed_bytes,
            "file_read_s": seconds(self.file_read_ns),
            "byte_view_s": seconds(self.byte_view_ns),
            "decode_s": seconds(self.decode_ns),
            "reader_wall_s": seconds(self.wall_ns),
        }


class _RequestCollector:
    """Recognizer-shaped sink which stops before recognizer execution."""

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self.requests: list[RecognitionRequest] = []
        self.consume_wall_s = 0.0
        self.complete = _CaptureComplete()

    def run(
        self,
        requests: Iterable[RecognitionRequest],
        *,
        schedule_id: str,
        emit_result: Any,
    ) -> NoReturn:
        del schedule_id, emit_result
        started = time.perf_counter()
        self.requests = list(requests)
        self.consume_wall_s = time.perf_counter() - started
        raise self.complete


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--paddleocr-source", type=Path, default=DEFAULT_PADDLEOCR_SOURCE)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--preprocessor-min-pixels", type=int)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=OMNIDOCBENCH_MAX_NEW_TOKENS,
        help="PaddleX request metadata only; no local OCR model is loaded.",
    )
    parser.add_argument(
        "--reference-requests",
        type=Path,
        help="Require byte-exact parity with a previous requests.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--image-reader",
        choices=("paddlex", "local-opencv"),
        default="paddlex",
        help=(
            "Use PaddleX's installed image reader, or an exact local "
            "open/read/frombuffer/cv2.imdecode equivalent with split timings."
        ),
    )
    parser.add_argument(
        "--layout-backend",
        choices=("eager", "npugraph"),
        default="npugraph",
        help="Layout model execution backend (default: exact static NPU graph).",
    )
    parser.add_argument(
        "--layout-polygons",
        choices=("mask", "rect"),
        default="mask",
        help=(
            "Use exact mask-derived crop geometry, or the faster non-exact "
            "box-only research path."
        ),
    )
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be non-negative and --limit positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.preprocessor_min_pixels is not None and args.preprocessor_min_pixels <= 0:
        parser.error("--preprocessor-min-pixels must be positive")
    return args


def _load_pages(
    dataset_json: Path,
    images_dir: Path,
    offset: int,
    limit: int,
) -> list[Path]:
    annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
    if len(annotations) != OMNIDOCBENCH_PAGE_COUNT:
        raise ValueError(
            f"expected {OMNIDOCBENCH_PAGE_COUNT} OmniDocBench pages, "
            f"got {len(annotations)}"
        )
    subset = annotations[offset : offset + limit]
    if len(subset) != limit:
        raise ValueError(f"requested {limit} pages at offset {offset}, got {len(subset)}")
    paths = [
        images_dir / Path(page["page_info"]["image_path"]).name
        for page in subset
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("selected pages contain duplicate image names")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_record(
    request: RecognitionRequest,
    *,
    request_index: int,
    image_paths: list[Path],
) -> dict[str, Any]:
    match = _REQUEST_ID.fullmatch(request.request_id)
    if match is None:
        raise ValueError(f"unexpected request id: {request.request_id!r}")
    page_index = int(match.group(1))
    if page_index >= len(image_paths):
        raise ValueError(f"{request.request_id} references absent page {page_index}")
    crop_bytes = request.crop.tobytes()
    crop_digest = hashlib.sha256(
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
        "crop_sha256": crop_digest,
    }


def _write_requests(
    path: Path,
    requests: list[RecognitionRequest],
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    records = [
        _request_record(request, request_index=index, image_paths=image_paths)
        for index, request in enumerate(requests)
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    paddleocr_source = args.paddleocr_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not layout_model.is_dir():
        raise FileNotFoundError(f"layout model not found: {layout_model}")
    if not paddleocr_source.is_dir():
        raise FileNotFoundError(f"PaddleOCR source not found: {paddleocr_source}")
    image_paths = _load_pages(dataset_json, images_dir, args.offset, args.limit)

    manifest = {
        "dataset_json": str(dataset_json),
        "dataset_sha256": _sha256(dataset_json),
        "images_dir": str(images_dir),
        "offset": args.offset,
        "count": len(image_paths),
        "images": [path.name for path in image_paths],
        "page_preprocessing_mode": "all_before_recognition",
        "frontend_boundary": "materialized_RecognitionRequest",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    layout_mask_guard = install_layout_mask_guard()
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("layout lab requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    sys.path.insert(0, str(paddleocr_source))
    from paddleocr import PaddleOCRVL

    setup_started = time.perf_counter()
    official_pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(layout_model),
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:9/v1",
        vl_rec_api_model_name="unused-layout-lab-adapter",
        vl_rec_max_concurrency=OMNIDOCBENCH_DECODE_BATCH_SIZE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_queues=True,
        device="npu:0",
        engine="transformers",
    )
    auto_parallel = official_pipeline.paddlex_pipeline
    if auto_parallel.multi_device_inference:
        raise ValueError("layout lab requires one selected NPU")
    layout_optimizations = install_layout_runtime_optimizations(
        auto_parallel._pipeline,
        backend=args.layout_backend,
        polygon_mode=args.layout_polygons,
    )
    setup_s = time.perf_counter() - setup_started

    timeline = TimelineRecorder(enabled=args.timeline)
    timeline.reset(
        {
            "kind": "layout_frontend_lab",
            "pages": len(image_paths),
            "local_ocr_model_loaded": False,
            "ocr_requests_executed": 0,
        }
    )
    local_image_reader = None
    if args.image_reader == "local-opencv":
        installed_reader = auto_parallel._pipeline.img_reader
        installed_format = getattr(installed_reader, "format", None)
        if installed_format != "BGR":
            raise RuntimeError(
                "local reader only mirrors PaddleX's BGR path, but the installed "
                f"reader reports format={installed_format!r}"
            )
        local_image_reader = _LocalOpenCVImageReader(timeline)
        auto_parallel._pipeline.img_reader = local_image_reader
    collector = _RequestCollector(args.max_new_tokens)
    empty_trace = output_dir / ".empty_recognition_trace.jsonl"
    bridge = PaddleXPageBridge(
        auto_parallel._pipeline,
        collector,  # type: ignore[arg-type]
        trace_path=empty_trace,
        min_pixels=args.preprocessor_min_pixels,
        timeline=timeline,
    )
    empty_pages: list[str] = []

    frontend_started = time.perf_counter()
    try:
        with timeline.span(
            "Pipeline",
            "Prepare every page through the recognizer boundary",
            event_type="scope",
            args={"pages": len(image_paths)},
        ):
            bridge.run(
                [str(path) for path in image_paths],
                emit_page=lambda result: empty_pages.append(
                    Path(str(result["input_path"])).name
                ),
                schedule_id="layout_lab_capture",
                preprocess_all_pages_first=True,
            )
    except _CaptureComplete as exc:
        if exc is not collector.complete:
            raise
    else:
        raise AssertionError("request collector did not reach the recognizer boundary")
    finally:
        frontend_wall_s = time.perf_counter() - frontend_started
        official_pipeline.close()
        if empty_trace.exists():
            if empty_trace.stat().st_size:
                raise RuntimeError(f"unexpected recognizer output: {empty_trace}")
            empty_trace.unlink()

    artifact_started = time.perf_counter()
    requests_path = output_dir / "requests.jsonl"
    records = _write_requests(requests_path, collector.requests, image_paths)
    requests_sha256 = _sha256(requests_path)
    timeline_json = output_dir / "timeline_trace.json"
    timeline_html = output_dir / "timeline.html"
    if args.timeline:
        timeline.write_json(timeline_json)
        timeline.write_html(timeline_html)
    layout_mask_guard_path = output_dir / "layout_mask_guard.json"
    layout_mask_guard.write_snapshot(layout_mask_guard_path)
    mask_rectangle_fast_path = getattr(
        auto_parallel._pipeline.layout_det_model,
        "_layout_mask_rectangle_fast_path",
        None,
    )
    mask_rectangle_fast_path_summary = (
        mask_rectangle_fast_path.snapshot()
        if mask_rectangle_fast_path is not None
        else None
    )
    artifact_write_s = time.perf_counter() - artifact_started

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
        **manifest,
        "configuration": {
            "layout_model": str(layout_model),
            "paddleocr_source": str(paddleocr_source),
            "device": "npu:0",
            "jit_compile": False,
            "layout_optimizations": layout_optimizations,
            "image_reader": args.image_reader,
            "preprocessor_min_pixels": args.preprocessor_min_pixels,
            "max_new_tokens_passthrough": args.max_new_tokens,
            "recognizer_execution": "replaced_by_request_collector",
            "local_ocr_model_loaded": False,
            "ocr_requests_executed": 0,
        },
        "setup_s": setup_s,
        "frontend_wall_s": frontend_wall_s,
        "pages_per_s": len(image_paths) / frontend_wall_s,
        "s_per_page": frontend_wall_s / len(image_paths),
        "request_materialization_s": collector.consume_wall_s,
        "image_reader_timing": (
            local_image_reader.summary() if local_image_reader is not None else None
        ),
        "requests": len(records),
        "requests_per_s": len(records) / frontend_wall_s,
        "requests_per_page": {
            "min": min(counts, default=0),
            "mean": sum(counts) / len(counts),
            "max": max(counts, default=0),
        },
        "empty_request_pages": empty_pages,
        "prompt_counts": dict(
            sorted(Counter(record["prompt"] for record in records).items())
        ),
        "crop_pixels": sum(
            int(record["crop_size"][0]) * int(record["crop_size"][1])
            for record in records
        ),
        "crop_bytes": sum(int(record["crop_nbytes"]) for record in records),
        "request_manifest": str(requests_path),
        "request_manifest_sha256": requests_sha256,
        "reference_comparison": reference,
        "layout_mask_guard": layout_mask_guard.snapshot(),
        "layout_mask_rectangle_fast_path": (
            mask_rectangle_fast_path_summary
        ),
        "timeline": {
            "enabled": bool(args.timeline),
            "event_count": len(timeline.events()) if args.timeline else 0,
            "trace_json": str(timeline_json) if args.timeline else None,
            "html": str(timeline_html) if args.timeline else None,
            "additional_device_synchronization": False,
        },
        "artifact_write_s": artifact_write_s,
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
