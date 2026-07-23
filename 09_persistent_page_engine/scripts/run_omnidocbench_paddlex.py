#!/usr/bin/env python3
"""Run official PaddleX v1.6 page assembly with Experiment 09 recognition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.runtime_defaults import (
    DEFAULT_DECODE_OPTIMIZATION,
    DEFAULT_TEXT_PACK_BUCKETS,
    DEFAULT_TEXT_PACK_MAX_MEMBERS,
    DEFAULT_TEXT_PACKING,
    DEFAULT_VISION_PACKING,
    DEFAULT_VISION_PACK_TARGET,
    DEFAULT_VISION_ROUTER_LOOKAHEAD,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
    PADDLEOCR_DEFAULT_MIN_PIXELS,
    TEXT_PACKING_CHOICES,
    VISION_PACKING_CHOICES,
)
from paddleocr_vl.model.text_decode import decode_optimization_names
from paddleocr_vl.model.text_prefill import TEXT_PADDING_CHOICES, parse_text_buckets
from paddleocr_vl.model.vision_prefill import (
    VISION_ATTENTION_CHOICES,
    VISION_BACKEND_CHOICES,
    VISION_PADDING_CHOICES,
    parse_vision_buckets,
)
from pipeline.layout_mask_guard import install_layout_mask_guard
from pipeline.omnidocbench_defaults import (
    OMNIDOCBENCH_CACHE_LENGTH,
    OMNIDOCBENCH_DECODE_BATCH_SIZE,
    OMNIDOCBENCH_MAX_NEW_TOKENS,
    OMNIDOCBENCH_PAGE_COUNT,
)
from pipeline.paddlex_page_bridge import PaddleXPageBridge
from utils.timeline import TimelineRecorder


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_RECOGNIZER_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_PADDLEOCR_SOURCE = Path("/workspace/repos/vllm_paddle_ocr/PaddleOCR")
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
DEFAULT_VISION_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
DEFAULT_TEXT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair"
DEFAULT_PACKED_TEXT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
)
DEFAULT_BATCHED_VISION_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_vision_router_batched"
PAGE_ARTIFACT_MAX_PENDING = 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--recognizer-model", type=Path, default=DEFAULT_RECOGNIZER_MODEL)
    parser.add_argument("--paddleocr-source", type=Path, default=DEFAULT_PADDLEOCR_SOURCE)
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=("fp16", "float16", "bf16", "bfloat16"),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=OMNIDOCBENCH_PAGE_COUNT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=OMNIDOCBENCH_DECODE_BATCH_SIZE,
    )
    parser.add_argument(
        "--cache-length",
        type=int,
        default=OMNIDOCBENCH_CACHE_LENGTH,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=OMNIDOCBENCH_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--decode-optimization",
        default=DEFAULT_DECODE_OPTIMIZATION,
        choices=decode_optimization_names(),
    )
    parser.add_argument(
        "--preprocessor-min-pixels",
        type=int,
        default=None,
        help="Override PaddleX's global recognition min_pixels; omit for the v1.6 default.",
    )
    parser.add_argument(
        "--vision-backend",
        default="torchair",
        choices=VISION_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--vision-attention",
        default="manual",
        choices=VISION_ATTENTION_CHOICES,
    )
    parser.add_argument(
        "--vision-buckets",
        default=",".join(str(bucket) for bucket in OPTIMIZED_VISION_BUCKETS),
    )
    parser.add_argument(
        "--vision-padding",
        default="auto",
        choices=VISION_PADDING_CHOICES,
    )
    parser.add_argument(
        "--vision-packing",
        default=DEFAULT_VISION_PACKING,
        choices=VISION_PACKING_CHOICES,
    )
    parser.add_argument(
        "--vision-pack-target",
        type=int,
        default=DEFAULT_VISION_PACK_TARGET,
    )
    parser.add_argument(
        "--vision-router-lookahead",
        type=int,
        default=DEFAULT_VISION_ROUTER_LOOKAHEAD,
    )
    parser.add_argument(
        "--vision-batched-cache-dir",
        type=Path,
        default=DEFAULT_BATCHED_VISION_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-buckets",
        default="auto",
        help=(
            "Comma-separated text buckets. 'auto' omits low-resolution-only "
            "buckets for default min_pixels and retains them for smaller overrides."
        ),
    )
    parser.add_argument(
        "--text-padding",
        default="auto",
        choices=TEXT_PADDING_CHOICES,
    )
    parser.add_argument(
        "--text-packing",
        default=DEFAULT_TEXT_PACKING,
        choices=TEXT_PACKING_CHOICES,
    )
    parser.add_argument(
        "--text-pack-buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_TEXT_PACK_BUCKETS),
    )
    parser.add_argument(
        "--text-pack-max-members",
        type=int,
        default=DEFAULT_TEXT_PACK_MAX_MEMBERS,
    )
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--vision-torchair-cache-dir",
        type=Path,
        default=DEFAULT_VISION_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-torchair-cache-dir",
        type=Path,
        default=DEFAULT_TEXT_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=DEFAULT_PACKED_TEXT_CACHE_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record a synchronization-neutral execution trace and self-contained "
            "HTML timeline (enabled by default)."
        ),
    )
    return parser.parse_args(argv)


def selected_text_buckets(args: argparse.Namespace) -> tuple[int, ...]:
    if args.text_buckets != "auto":
        return parse_text_buckets(args.text_buckets)
    if (
        args.preprocessor_min_pixels is None
        or int(args.preprocessor_min_pixels) >= PADDLEOCR_DEFAULT_MIN_PIXELS
    ):
        return tuple(bucket for bucket in OPTIMIZED_TEXT_BUCKETS if bucket >= 160)
    return OPTIMIZED_TEXT_BUCKETS


def load_page_paths(
    dataset_json: Path,
    images_dir: Path,
    *,
    offset: int,
    limit: int,
) -> tuple[list[dict], list[Path]]:
    annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
    if len(annotations) != OMNIDOCBENCH_PAGE_COUNT:
        raise ValueError(
            "expected OmniDocBench v1.6 to contain "
            f"{OMNIDOCBENCH_PAGE_COUNT} pages, got {len(annotations)}"
        )
    subset = annotations[offset : offset + limit]
    if len(subset) != limit:
        raise ValueError(
            f"requested {limit} pages at offset {offset}, got {len(subset)}"
        )
    paths = [
        images_dir / Path(page["page_info"]["image_path"]).name
        for page in subset
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("selected OmniDocBench pages contain duplicate image names")
    return subset, paths


def append_compact_page_result(handle: Any, result: Any) -> None:
    payload = result.json["res"]
    record = {
        "input_path": payload["input_path"],
        "image_name": Path(payload["input_path"]).name,
        "page_index": payload["page_index"],
        "page_count": payload["page_count"],
        "width": payload["width"],
        "height": payload["height"],
        "parsing_res_list": payload["parsing_res_list"],
    }
    handle.write(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    handle.flush()


def validate_args(args: argparse.Namespace) -> None:
    if args.offset < 0 or args.limit <= 0:
        raise ValueError("--offset must be non-negative and --limit must be positive")
    if args.batch_size <= 0 or args.batch_size & (args.batch_size - 1):
        raise ValueError("--batch-size must be a positive power of two")
    if args.cache_length <= args.max_new_tokens:
        raise ValueError("--cache-length must leave room for the input prompt")
    if args.preprocessor_min_pixels is not None and args.preprocessor_min_pixels <= 0:
        raise ValueError("--preprocessor-min-pixels must be positive")
    if args.vision_pack_target <= 0:
        raise ValueError("--vision-pack-target must be positive")
    if args.vision_router_lookahead <= 0:
        raise ValueError("--vision-router-lookahead must be positive")
    if args.text_pack_max_members <= 0:
        raise ValueError("--text-pack-max-members must be positive")


def run_predictions(
    official_pipeline: Any,
    bridge: PaddleXPageBridge,
    image_paths: list[Path],
    *,
    predictions_dir: Path,
    compact_path: Path,
    timeline: TimelineRecorder,
) -> tuple[int, list[float], float, Any, dict[str, Any]]:
    completion_s: list[float] = []
    result_count = 0
    write_wall_s: list[float] = []
    queue_wait_s: list[float] = []
    pending: deque[Future[None]] = deque()
    max_pending_observed = 0
    producer_backpressure_s = 0.0
    final_drain_s = 0.0
    page_ordinals = {path.name: ordinal for ordinal, path in enumerate(image_paths)}
    predict_started = time.perf_counter()

    def write_result(
        result: Any,
        *,
        completion_index: int,
        completion_elapsed_s: float,
        submitted_ns: int,
        flow_id: str,
        image_name: str,
    ) -> None:
        started_ns = time.perf_counter_ns()
        queue_wait_s.append((started_ns - submitted_ns) / 1_000_000_000)
        timeline.record_span(
            "Artifacts / tracing",
            "Page artifact waiting for writer",
            submitted_ns,
            started_ns,
            flow_id=flow_id,
            event_type="wait",
            track="queue",
            lane="page-artifacts",
            args={"image_name": image_name},
        )
        try:
            result.save_to_markdown(save_path=str(predictions_dir))
            append_compact_page_result(compact_handle, result)
            print(
                f"completed={completion_index}/{len(image_paths)} "
                f"elapsed_s={completion_elapsed_s:.3f}",
                flush=True,
            )
        finally:
            finished_ns = time.perf_counter_ns()
            write_wall_s.append((finished_ns - started_ns) / 1_000_000_000)
            timeline.record_span(
                "Artifacts / tracing",
                "Write page artifacts",
                started_ns,
                finished_ns,
                flow_id=flow_id,
                event_type="io",
                args={"image_name": image_name},
            )

    def wait_for_oldest(*, final_drain: bool) -> None:
        nonlocal producer_backpressure_s, final_drain_s
        future = pending.popleft()
        started_ns = time.perf_counter_ns()
        future.result()
        finished_ns = time.perf_counter_ns()
        waited_s = (finished_ns - started_ns) / 1_000_000_000
        if final_drain:
            final_drain_s += waited_s
            name = "Drain page artifact writer"
        else:
            producer_backpressure_s += waited_s
            name = "Wait for page artifact writer capacity"
        timeline.record_span(
            "Artifacts / tracing",
            name,
            started_ns,
            finished_ns,
            event_type="wait",
        )

    def save_result(result: Any) -> None:
        nonlocal result_count
        nonlocal max_pending_observed
        while pending and pending[0].done():
            pending.popleft().result()
        if len(pending) >= PAGE_ARTIFACT_MAX_PENDING:
            wait_for_oldest(final_drain=False)

        image_name = Path(str(result["input_path"])).name
        ordinal = page_ordinals[image_name]
        result_count += 1
        completion_elapsed_s = time.perf_counter() - predict_started
        completion_s.append(completion_elapsed_s)
        submitted_ns = time.perf_counter_ns()
        pending.append(
            writer.submit(
                write_result,
                result,
                completion_index=result_count,
                completion_elapsed_s=completion_elapsed_s,
                submitted_ns=submitted_ns,
                flow_id=f"page:{ordinal}",
                image_name=image_name,
            )
        )
        max_pending_observed = max(max_pending_observed, len(pending))

    try:
        with (
            compact_path.open("w", encoding="utf-8") as compact_handle,
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="page-artifact-writer",
            ) as writer,
        ):
            try:
                page_run = bridge.run(
                    [str(path) for path in image_paths],
                    emit_page=save_result,
                )
            finally:
                while pending:
                    wait_for_oldest(final_drain=True)
    finally:
        official_pipeline.close()
    writer_summary = {
        "execution": "background_ordered_single_worker",
        "max_pending": PAGE_ARTIFACT_MAX_PENDING,
        "max_pending_observed": max_pending_observed,
        "write_wall_sum_s": sum(write_wall_s),
        "queue_wait_sum_s": sum(queue_wait_s),
        "producer_backpressure_s": producer_backpressure_s,
        "final_drain_s": final_drain_s,
    }
    return (
        result_count,
        completion_s,
        time.perf_counter() - predict_started,
        page_run,
        writer_summary,
    )


def validate_predictions(predictions_dir: Path, image_paths: list[Path]) -> list[str]:
    prediction_files = sorted(path.name for path in predictions_dir.glob("*.md"))
    expected_files = sorted(f"{path.stem}.md" for path in image_paths)
    if prediction_files != expected_files:
        missing = sorted(set(expected_files) - set(prediction_files))
        extra = sorted(set(prediction_files) - set(expected_files))
        raise RuntimeError(
            f"prediction filename mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return prediction_files


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    recognizer_model = args.recognizer_model.expanduser().resolve()
    paddleocr_source = args.paddleocr_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if not paddleocr_source.is_dir():
        raise FileNotFoundError(f"PaddleOCR source not found: {paddleocr_source}")

    subset, image_paths = load_page_paths(
        dataset_json,
        images_dir,
        offset=args.offset,
        limit=args.limit,
    )
    (output_dir / "OmniDocBench_subset.json").write_text(
        json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    vision_buckets = parse_vision_buckets(args.vision_buckets)
    text_buckets = selected_text_buckets(args)
    text_pack_buckets = parse_text_buckets(args.text_pack_buckets)
    manifest = {
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "offset": args.offset,
        "count": args.limit,
        "images": [path.name for path in image_paths],
        "pipeline": "official_paddlex_v1.6_with_cross_page_recognition_bridge",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    layout_mask_guard = install_layout_mask_guard()
    layout_mask_guard_path = output_dir / "layout_mask_guard.json"

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("Experiment 09 requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)

    sys.path.insert(0, str(paddleocr_source))
    from paddleocr import PaddleOCRVL

    setup_started = time.perf_counter()
    official_pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(layout_model),
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:9/v1",
        vl_rec_api_model_name="unused-local-adapter",
        vl_rec_max_concurrency=args.batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_queues=True,
        device="npu:0",
        engine="transformers",
    )
    auto_parallel_pipeline = official_pipeline.paddlex_pipeline
    if auto_parallel_pipeline.multi_device_inference:
        raise ValueError(
            "the cross-page bridge requires one selected NPU; multi-device PaddleX "
            "dispatch would create multiple independent recognition runtimes"
        )
    # The bridge calls the concrete pipeline's page helpers directly. Attribute
    # access on the auto-parallel wrapper only forwards reads.
    paddlex_pipeline = auto_parallel_pipeline._pipeline
    timeline = TimelineRecorder(enabled=args.timeline)

    recognizer = ContinuousRecognizer(
        model=str(recognizer_model),
        dtype=args.dtype,
        decode_backend="torchair",
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        torchair_cache_dir=args.torchair_cache_dir.expanduser().resolve(),
        decode_optimization=args.decode_optimization,
        vision_backend=args.vision_backend,
        vision_attention=args.vision_attention,
        vision_buckets=vision_buckets,
        vision_torchair_cache_dir=(
            args.vision_torchair_cache_dir.expanduser().resolve()
        ),
        vision_padding=args.vision_padding,
        vision_packing=args.vision_packing,
        vision_pack_target=args.vision_pack_target,
        vision_router_lookahead=args.vision_router_lookahead,
        vision_batched_cache_dir=(
            args.vision_batched_cache_dir.expanduser().resolve()
        ),
        text_backend="torchair",
        text_buckets=text_buckets,
        text_torchair_cache_dir=args.text_torchair_cache_dir.expanduser().resolve(),
        text_padding=args.text_padding,
        text_packing=args.text_packing,
        text_pack_buckets=text_pack_buckets,
        text_pack_max_members=args.text_pack_max_members,
        text_packed_cache_dir=args.text_packed_cache_dir.expanduser().resolve(),
        preprocessor_min_pixels=args.preprocessor_min_pixels,
        timeline=timeline,
    )
    bridge = PaddleXPageBridge(
        paddlex_pipeline,
        recognizer,
        trace_path=output_dir / "recognition_trace.jsonl",
        min_pixels=args.preprocessor_min_pixels,
        timeline=timeline,
    )
    setup_s = time.perf_counter() - setup_started

    compact_path = output_dir / "page_regions.jsonl"
    timeline_json_path = output_dir / "timeline_trace.json"
    timeline_html_path = output_dir / "timeline.html"
    timeline.reset(
        {
            "pipeline": "official_paddlex_v1.6_with_cross_page_recognition_bridge",
            "pages": len(image_paths),
            "decode_batch_size": args.batch_size,
            "decode_optimization": args.decode_optimization,
            "vision_backend": args.vision_backend,
            "vision_attention": args.vision_attention,
            "vision_packing": args.vision_packing,
            "vision_pack_target": args.vision_pack_target,
            "vision_router_lookahead": args.vision_router_lookahead,
            "text_packing": args.text_packing,
            "text_pack_buckets": list(text_pack_buckets),
        }
    )
    try:
        with timeline.span(
            "Pipeline",
            "Full PaddleOCR-VL page run",
            event_type="scope",
            args={"pages": len(image_paths), "decode_batch_size": args.batch_size},
        ):
            (
                result_count,
                completion_s,
                pipeline_e2e_s,
                page_run,
                page_artifact_writer,
            ) = run_predictions(
                official_pipeline,
                bridge,
                image_paths,
                predictions_dir=predictions_dir,
                compact_path=compact_path,
                timeline=timeline,
            )
    finally:
        layout_mask_guard.write_snapshot(layout_mask_guard_path)
        if args.timeline:
            timeline.write_json(timeline_json_path)
            timeline.write_html(timeline_html_path)

    prediction_files = validate_predictions(predictions_dir, image_paths)
    if result_count != len(image_paths):
        raise RuntimeError(
            f"expected {len(image_paths)} page results, got {result_count}"
        )

    summary = {
        **manifest,
        "configuration": {
            "layout_model": str(layout_model),
            "recognizer_model": str(recognizer_model),
            "device": "npu:0",
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "decode_optimization": args.decode_optimization,
            "preprocessor_min_pixels": args.preprocessor_min_pixels,
            "effective_global_min_pixels": (
                args.preprocessor_min_pixels
                if args.preprocessor_min_pixels is not None
                else PADDLEOCR_DEFAULT_MIN_PIXELS
            ),
            "vision_backend": args.vision_backend,
            "vision_attention": args.vision_attention,
            "vision_padding": args.vision_padding,
            "vision_packing": args.vision_packing,
            "vision_pack_target": args.vision_pack_target,
            "vision_router_lookahead": args.vision_router_lookahead,
            "vision_batched_cache_dir": str(
                args.vision_batched_cache_dir.expanduser().resolve()
            ),
            "vision_buckets": list(vision_buckets),
            "text_buckets": list(text_buckets),
            "text_packing": args.text_packing,
            "text_pack_buckets": list(text_pack_buckets),
            "text_pack_max_members": args.text_pack_max_members,
            "text_packed_cache_dir": str(
                args.text_packed_cache_dir.expanduser().resolve()
            ),
            "cpu_preprocessing": recognizer.configuration()["cpu_preprocessing"],
        },
        "setup_s": setup_s,
        "recognizer_setup_timing_s": recognizer.setup_timing_s,
        "pipeline_e2e_s": pipeline_e2e_s,
        "pages_per_s": len(image_paths) / pipeline_e2e_s,
        "s_per_page": pipeline_e2e_s / len(image_paths),
        "result_count": result_count,
        "prediction_count": len(prediction_files),
        "completion_s": completion_s,
        "page_artifact_writer": page_artifact_writer,
        "recognition": page_run.summary,
        "page_completion_order": page_run.completion_order,
        "layout_mask_guard": layout_mask_guard.snapshot(),
        "layout_mask_guard_path": str(layout_mask_guard_path),
        "predictions_dir": str(predictions_dir),
        "page_regions_jsonl": str(compact_path),
        "timeline": {
            "enabled": bool(args.timeline),
            "event_count": len(timeline.events()) if args.timeline else 0,
            "trace_json": str(timeline_json_path) if args.timeline else None,
            "html": str(timeline_html_path) if args.timeline else None,
            "additional_device_synchronization": False,
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
