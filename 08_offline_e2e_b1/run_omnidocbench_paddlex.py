#!/usr/bin/env python3
"""Run official PaddleX v1.6 page assembly with Experiment 08 recognition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from engine import ContinuousRecognizer
from paddlex_adapter import PaddleXContinuousRecognizerAdapter
from runtime_defaults import (
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
)
from text_compile import parse_text_buckets
from vision_compile import parse_vision_buckets


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_RECOGNIZER_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_PADDLEOCR_SOURCE = Path("/workspace/repos/vllm_paddle_ocr/PaddleOCR")
DEFAULT_CACHE_ROOT = Path(".runtime_cache/08_offline_e2e_b1_torchair")
DEFAULT_VISION_CACHE_ROOT = Path(".runtime_cache/08_offline_e2e_b1_vision_torchair")
DEFAULT_TEXT_CACHE_ROOT = Path(".runtime_cache/08_offline_e2e_b1_text_torchair")
DEFAULT_MIN_PIXELS = 112896


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--recognizer-model", type=Path, default=DEFAULT_RECOGNIZER_MODEL)
    parser.add_argument("--paddleocr-source", type=Path, default=DEFAULT_PADDLEOCR_SOURCE)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1651)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--preprocessor-min-pixels",
        type=int,
        default=None,
        help="Override PaddleX's global recognition min_pixels; omit for the v1.6 default.",
    )
    parser.add_argument(
        "--vision-compile-buckets",
        default=",".join(str(bucket) for bucket in OPTIMIZED_VISION_BUCKETS),
    )
    parser.add_argument(
        "--text-compile-buckets",
        default="auto",
        help=(
            "Comma-separated text buckets. 'auto' omits low-resolution-only "
            "buckets for default min_pixels and retains them for smaller overrides."
        ),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def selected_text_buckets(args: argparse.Namespace) -> tuple[int, ...]:
    if args.text_compile_buckets != "auto":
        return parse_text_buckets(args.text_compile_buckets)
    if (
        args.preprocessor_min_pixels is None
        or int(args.preprocessor_min_pixels) >= DEFAULT_MIN_PIXELS
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
    if len(annotations) != 1651:
        raise ValueError(
            f"expected OmniDocBench v1.6 to contain 1651 pages, got {len(annotations)}"
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


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit <= 0:
        raise ValueError("--offset must be non-negative and --limit must be positive")
    if args.batch_size <= 0 or args.batch_size & (args.batch_size - 1):
        raise ValueError("--batch-size must be a positive power of two")
    if args.cache_length <= args.max_new_tokens:
        raise ValueError("--cache-length must leave room for the input prompt")
    if args.preprocessor_min_pixels is not None and args.preprocessor_min_pixels <= 0:
        raise ValueError("--preprocessor-min-pixels must be positive")

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
    vision_buckets = parse_vision_buckets(args.vision_compile_buckets)
    text_buckets = selected_text_buckets(args)
    manifest = {
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "offset": args.offset,
        "count": args.limit,
        "images": [path.name for path in image_paths],
        "pipeline": "official_paddlex_v1.6_with_experiment08_vl_rec_model_adapter",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
        device=args.device,
        engine="transformers",
    )
    paddlex_pipeline = official_pipeline.paddlex_pipeline
    original_vl_rec_model = paddlex_pipeline.vl_rec_model
    original_vl_rec_model.close()

    recognizer = ContinuousRecognizer(
        model=str(recognizer_model),
        device=args.device,
        dtype=args.dtype,
        decode_backend="torchair",
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        torchair_cache_dir=args.torchair_cache_dir.expanduser().resolve(),
        vision_backend="torchair",
        vision_buckets=vision_buckets,
        vision_torchair_cache_dir=(
            args.vision_torchair_cache_dir.expanduser().resolve()
        ),
        text_backend="torchair",
        text_buckets=text_buckets,
        text_torchair_cache_dir=args.text_torchair_cache_dir.expanduser().resolve(),
        npu_jit_compile="off",
        preprocessor_min_pixels=args.preprocessor_min_pixels,
    )
    adapter = PaddleXContinuousRecognizerAdapter(
        recognizer,
        batch_size=args.batch_size,
        trace_path=output_dir / "recognition_trace.jsonl",
    )
    paddlex_pipeline.vl_rec_model = adapter
    setup_s = time.perf_counter() - setup_started

    compact_path = output_dir / "page_regions.jsonl"
    completion_s = []
    result_count = 0
    predict_started = time.perf_counter()
    try:
        with compact_path.open("w", encoding="utf-8") as compact_handle:
            for result in official_pipeline.predict_iter(
                [str(path) for path in image_paths],
                use_queues=True,
                min_pixels=args.preprocessor_min_pixels,
                max_new_tokens=args.max_new_tokens,
            ):
                result.save_to_markdown(save_path=str(predictions_dir))
                append_compact_page_result(compact_handle, result)
                result_count += 1
                completion_s.append(time.perf_counter() - predict_started)
                print(
                    f"completed={result_count}/{len(image_paths)} "
                    f"elapsed_s={completion_s[-1]:.3f}",
                    flush=True,
                )
    finally:
        official_pipeline.close()
    pipeline_e2e_s = time.perf_counter() - predict_started

    prediction_files = sorted(path.name for path in predictions_dir.glob("*.md"))
    expected_files = sorted(f"{path.stem}.md" for path in image_paths)
    if prediction_files != expected_files:
        missing = sorted(set(expected_files) - set(prediction_files))
        extra = sorted(set(prediction_files) - set(expected_files))
        raise RuntimeError(
            f"prediction filename mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    if result_count != len(image_paths):
        raise RuntimeError(
            f"expected {len(image_paths)} page results, got {result_count}"
        )

    summary = {
        **manifest,
        "configuration": {
            "layout_model": str(layout_model),
            "recognizer_model": str(recognizer_model),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "preprocessor_min_pixels": args.preprocessor_min_pixels,
            "effective_global_min_pixels": (
                args.preprocessor_min_pixels
                if args.preprocessor_min_pixels is not None
                else DEFAULT_MIN_PIXELS
            ),
            "vision_buckets": list(vision_buckets),
            "text_buckets": list(text_buckets),
        },
        "setup_s": setup_s,
        "recognizer_setup_timing_s": recognizer.setup_timing_s,
        "pipeline_e2e_s": pipeline_e2e_s,
        "pages_per_s": len(image_paths) / pipeline_e2e_s,
        "s_per_page": pipeline_e2e_s / len(image_paths),
        "result_count": result_count,
        "prediction_count": len(prediction_files),
        "completion_s": completion_s,
        "adapter": adapter.summary(),
        "predictions_dir": str(predictions_dir),
        "page_regions_jsonl": str(compact_path),
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
