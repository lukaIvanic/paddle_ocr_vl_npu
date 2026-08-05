#!/usr/bin/env python3
"""Run OpenDoc with exact B1 prefills and cross-page decode scheduling.

OpenDoc/OpenOCR remains an unmodified dependency.  This runner reuses its
layout detector, crop transforms, result assembly helpers, and writers while
owning the crop queue and UniRec decode scheduling locally.  It supports both
fixed cohorts and a fixed-arena continuous decoder with per-slot hot swapping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from continuous_unirec import (
    ContinuousCompletedItem,
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
)
from modeling_optimized_unirec import OptimizedUniRecRunner
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


@dataclass
class CropRequest:
    page_index: int
    crop_index: int
    page_name: str
    image: Image.Image
    label: str
    figure_token_map: dict[str, Any]
    result: dict[str, Any] | None = None

    @property
    def request_id(self) -> str:
        return f"page_{self.page_index:06d}_crop_{self.crop_index:04d}"


@dataclass
class PageRequest:
    page_index: int
    image_path: Path
    width: int
    height: int
    layout_results: dict[str, Any]
    blocks: list[dict[str, Any]]
    vlm_block_ids: list[int]
    crops: list[CropRequest]
    drop_figures_set: set[str]
    started_at: float
    layout_s: float
    prepare_page_total_s: float

    def is_ready(self) -> bool:
        return all(crop.result is not None for crop in self.crops)


@dataclass
class RunMetrics:
    cohort_records: list[dict[str, Any]] = field(default_factory=list)
    crop_records: list[dict[str, Any]] = field(default_factory=list)
    page_records: list[dict[str, Any]] = field(default_factory=list)
    layout_s: float = 0.0
    page_prepare_total_s: float = 0.0
    prepare_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    output_assembly_s: float = 0.0
    output_write_s: float = 0.0
    raw_decode_token_slots: int = 0
    effective_decode_tokens: int = 0
    padding_decode_token_slots: int = 0
    idle_decode_token_slots: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("/root/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx"),
    )
    parser.add_argument(
        "--layout-backend",
        choices=("onnx_cpu", "transformers_npu"),
        default="transformers_npu",
    )
    parser.add_argument(
        "--layout-transformers-model",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
    parser.add_argument(
        "--layout-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--stock-encoder", type=Path, required=True)
    parser.add_argument("--stock-decoder", type=Path, required=True)
    parser.add_argument("--stock-tokenizer-mapping", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--decode-mode",
        choices=("eager", "compiled", "compiled_ifa"),
        default="compiled",
    )
    parser.add_argument("--compile-backend", choices=("torchair",), default="torchair")
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/12_unirec_0_1b_inference/opendoc_model_pth_decode"),
    )
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument(
        "--decode-scheduling",
        choices=("fixed", "continuous"),
        default="fixed",
        help=(
            "fixed waits for the longest request in each cohort; continuous "
            "hot-swaps a new B1-prefilled request into each finished slot"
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--layout-threshold", type=float, default=0.4)
    return parser.parse_args()


def _base_label(label: str) -> str:
    parts = label.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else label


def _postprocess_recognizer_text(markdown_converter: Any, text: str, label: str) -> str:
    if "table" in label:
        return markdown_converter._handle_table(text)
    if "formula" in label and label != "formula_number":
        return markdown_converter._handle_formula(text)
    return markdown_converter._handle_text(text)


def prepare_page(
    *,
    pipeline: Any,
    infer_doc_onnx: Any,
    image_path: Path,
    page_index: int,
    layout_threshold: float,
) -> PageRequest:
    started_at = time.perf_counter()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]

    layout_started = time.perf_counter()
    layout_results = pipeline.layout_detector(
        [image],
        threshold=layout_threshold,
    )[0]
    layout_s = time.perf_counter() - layout_started
    image_labels = (
        infer_doc_onnx.IMAGE_LABELS
        if pipeline.use_chart_recognition
        else infer_doc_onnx.IMAGE_LABELS + ["chart"]
    )

    blocks = []
    for box in layout_results["boxes"]:
        x1, y1, x2, y2 = map(int, box["coordinate"])
        cropped = image[y1:y2, x1:x2]
        blocks.append(
            {
                "img": None if cropped.size == 0 else cropped,
                "box": box["coordinate"],
                "label": box["label"],
                "score": box.get("score", 1.0),
            }
        )
    blocks = infer_doc_onnx.merge_blocks(
        blocks,
        non_merge_labels=image_labels + ["table"],
    )

    imgs_in_doc = []
    for block in blocks:
        label = block["label"]
        if _base_label(label) in image_labels and block["img"] is not None:
            x1, y1, x2, y2 = map(int, block["box"])
            imgs_in_doc.append(
                {
                    "coordinate": block["box"],
                    "path": f"imgs/img_in_{_base_label(label)}_box_{x1}_{y1}_{x2}_{y2}.jpg",
                }
            )

    crops: list[CropRequest] = []
    vlm_block_ids: list[int] = []
    drop_figures_set: set[str] = set()
    for block_index, block in enumerate(blocks):
        block_img = block["img"]
        label = block["label"]
        if _base_label(label) in image_labels or block_img is None:
            continue
        figure_token_map: dict[str, Any] = {}
        drop_figures: list[str] = []
        if "table" in label:
            block_img, figure_token_map, drop_figures = (
                infer_doc_onnx.tokenize_figure_of_table(
                    block_img,
                    block["box"],
                    imgs_in_doc,
                )
            )
        elif "formula" in label and label != "formula_number":
            block_img = infer_doc_onnx.crop_margin(block_img)
        rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
        crops.append(
            CropRequest(
                page_index=page_index,
                crop_index=len(crops),
                page_name=image_path.name,
                image=Image.fromarray(rgb),
                label=label,
                figure_token_map=figure_token_map,
            )
        )
        vlm_block_ids.append(block_index)
        drop_figures_set.update(drop_figures)

    prepare_page_total_s = time.perf_counter() - started_at
    return PageRequest(
        page_index=page_index,
        image_path=image_path,
        width=width,
        height=height,
        layout_results=layout_results,
        blocks=blocks,
        vlm_block_ids=vlm_block_ids,
        crops=crops,
        drop_figures_set=drop_figures_set,
        started_at=started_at,
        layout_s=layout_s,
        prepare_page_total_s=prepare_page_total_s,
    )


def assemble_page(
    *,
    page: PageRequest,
    pipeline: Any,
    infer_doc_onnx: Any,
) -> dict[str, Any]:
    recognition_results = []
    current_crop = 0
    image_labels = (
        infer_doc_onnx.IMAGE_LABELS
        if pipeline.use_chart_recognition
        else infer_doc_onnx.IMAGE_LABELS + ["chart"]
    )
    for block_index, block in enumerate(page.blocks):
        block_img = block["img"]
        bbox = block["box"]
        label = block["label"]
        content = ""
        if (
            current_crop < len(page.vlm_block_ids)
            and page.vlm_block_ids[current_crop] == block_index
        ):
            crop = page.crops[current_crop]
            if crop.result is None:
                raise RuntimeError(f"Crop {crop.request_id} was not recognized")
            content = _postprocess_recognizer_text(
                infer_doc_onnx.markdown_converter,
                crop.result["text"],
                label,
            )
            content = infer_doc_onnx.truncate_repetitive_content(content)
            has_paren = "\\(" in content and "\\)" in content
            has_bracket = "\\[" in content and "\\]" in content
            if has_paren or has_bracket:
                content = content.replace("$", "")
                content = (
                    content.replace("\\(", " $ ")
                    .replace("\\)", " $ ")
                    .replace("\\[", " $$ ")
                    .replace("\\]", " $$ ")
                )
                if label == "formula_number":
                    content = content.replace("$", "")
            if "table" in label:
                html = infer_doc_onnx.convert_otsl_to_html(content)
                if html:
                    content = html
                content = infer_doc_onnx.untokenize_figure_of_table(
                    content,
                    crop.figure_token_map,
                )
            current_crop += 1

        base_label = _base_label(label)
        if base_label in image_labels and block_img is not None:
            x1, y1, x2, y2 = map(int, bbox)
            image_output_path = (
                f"imgs/img_in_{base_label}_box_{x1}_{y1}_{x2}_{y2}.jpg"
            )
            recognition_results.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "score": block.get("score", 1.0),
                    "text": "",
                    "text_unirec": "",
                    "is_image": True,
                    "img_path": image_output_path,
                    "is_merged_continuation": False,
                    "in_table": image_output_path in page.drop_figures_set,
                }
            )
        else:
            recognition_results.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "score": block.get("score", 1.0),
                    "text": content,
                    "text_unirec": content,
                    "is_image": False,
                    "is_merged_continuation": block_img is None,
                }
            )

    return {
        "input_path": str(page.image_path),
        "width": page.width,
        "height": page.height,
        "layout_results": page.layout_results,
        "recognition_results": recognition_results,
        "blocks": page.blocks,
        "timing": {"total": time.perf_counter() - page.started_at},
    }


def recognize_cohort(
    *,
    cohort: list[CropRequest],
    target_batch_size: int,
    runner: OptimizedUniRecRunner,
    args: argparse.Namespace,
    metrics: RunMetrics,
) -> None:
    cohort_index = len(metrics.cohort_records)
    print(
        f"UNIREC_BATCH_BEGIN index={cohort_index} real={len(cohort)} "
        f"physical={target_batch_size}",
        flush=True,
    )
    prefilled = []
    for crop in cohort:
        item = runner.prefill_image_for_cohort(
            crop.image,
            image_source=crop.request_id,
        )
        prefilled.append(item)
        metrics.prepare_s += float(item.prep["prepare_total_s"])
        metrics.prefill_s += item.prefill_s
    decoded = runner.generate_prefilled_cohort(
        prefilled,
        max_length=args.max_length,
        decode_mode=args.decode_mode,
        compile_backend=args.compile_backend,
        pad_to_batch_size=target_batch_size,
    )
    metrics.decode_s += float(decoded["decode_s"])
    metrics.raw_decode_token_slots += int(decoded["raw_decode_token_slots"])
    metrics.effective_decode_tokens += int(decoded["effective_decode_tokens"])
    metrics.padding_decode_token_slots += int(decoded["padding_decode_token_slots"])
    cohort_record = {
        key: value
        for key, value in decoded.items()
        if key not in {"items", "compile"}
    }
    cohort_record["cohort_index"] = cohort_index
    cohort_record["request_ids"] = [crop.request_id for crop in cohort]
    metrics.cohort_records.append(cohort_record)
    for crop, result in zip(cohort, decoded["items"]):
        crop.result = result
        metrics.crop_records.append(
            {
                "request_id": crop.request_id,
                "page": crop.page_name,
                "page_index": crop.page_index,
                "crop_index": crop.crop_index,
                "label": crop.label,
                "crop_size": [crop.image.width, crop.image.height],
                "processed_image_size": result["prep"]["processed_image_size"],
                "encoder_seq_len_hint": result["prep"]["encoder_seq_len_hint"],
                "token_ids": result["generated_ids"],
                "text": result["text"],
                "token_count": result["generated_token_count"],
                "decode_token_count": result["decode_generated_token_count"],
                "prefill_s": result["ttft_s"],
                "cohort_index": cohort_index,
            }
        )
    print(
        f"UNIREC_BATCH_END index={cohort_index} decode_s={decoded['decode_s']:.3f} "
        f"raw_tps={decoded['raw_decode_tokens_per_s']:.1f} "
        f"effective_tps={decoded['effective_decode_tokens_per_s']:.1f} "
        f"padding_slots={decoded['padding_decode_token_slots']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "once",
        message=(
            r"Skip cache as LocalUniRecCachedDecodeStepModule\.forward.*recompiled.*"
        ),
        category=UserWarning,
    )
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be >= 1")
    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.max_length > 256:
        raise ValueError("The current UniRec static self-KV cache supports max-length <= 256")

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)

    sys.path.insert(0, str(openocr_root))
    from tools import infer_doc_onnx
    from tools.utils.utility import get_image_file_list

    image_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(input_path)))
    ][args.offset :]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No input images found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_started = time.perf_counter()
    use_onnx_layout = args.layout_backend == "onnx_cpu"
    pipeline = infer_doc_onnx.OpenDocONNX(
        layout_model_path=str(args.layout_model.expanduser().resolve()),
        unirec_encoder_path=str(args.stock_encoder.expanduser().resolve()),
        unirec_decoder_path=str(args.stock_decoder.expanduser().resolve()),
        tokenizer_mapping_path=str(args.stock_tokenizer_mapping.expanduser().resolve()),
        use_gpu=False,
        layout_threshold=args.layout_threshold,
        use_layout_detection=use_onnx_layout,
        auto_download=False,
        max_parallel_blocks=1,
    )
    if not use_onnx_layout:
        pipeline.layout_detector = PPDocLayoutV2NpuAdapter(
            model_path=args.layout_transformers_model,
            device=args.device,
            dtype=args.layout_dtype,
            threshold=args.layout_threshold,
        )
        pipeline.use_layout_detection = True
    runner = OptimizedUniRecRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=(
            args.compile_cache_dir.expanduser().resolve()
            if args.decode_mode.startswith("compiled")
            else None
        ),
    )
    setup_s = time.perf_counter() - setup_started
    print(
        f"OPENDOC_BATCHED_SETUP_END setup_s={setup_s:.3f} pages={len(image_paths)} "
        f"decode_batch_size={args.decode_batch_size} "
        f"decode_scheduling={args.decode_scheduling}",
        flush=True,
    )

    metrics = RunMetrics()
    pending_crops: deque[CropRequest] = deque()
    pending_pages: deque[PageRequest] = deque()
    pipeline_started = time.perf_counter()
    written_pages = 0
    continuous_decode: dict[str, Any] | None = None

    def flush_ready_pages() -> None:
        nonlocal written_pages
        while pending_pages and pending_pages[0].is_ready():
            page = pending_pages.popleft()
            assembly_started = time.perf_counter()
            result = assemble_page(
                page=page,
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
            )
            metrics.output_assembly_s += time.perf_counter() - assembly_started
            write_started = time.perf_counter()
            pipeline.save_to_json(result, str(output_dir))
            pipeline.save_to_markdown(result, str(output_dir))
            metrics.output_write_s += time.perf_counter() - write_started
            written_pages += 1
            page_s = time.perf_counter() - page.started_at
            metrics.page_records.append(
                {
                    "page_index": page.page_index,
                    "image": str(page.image_path),
                    "crop_count": len(page.crops),
                    "layout_s": page.layout_s,
                    "wall_s": page_s,
                }
            )
            print(
                f"OPENDOC_BATCHED_PAGE_END index={written_pages}/{len(image_paths)} "
                f"image={page.image_path.name} crops={len(page.crops)} wall_s={page_s:.3f}",
                flush=True,
            )

    if args.decode_scheduling == "continuous":
        def ready_source():
            for page_index, image_path in enumerate(image_paths):
                page = prepare_page(
                    pipeline=pipeline,
                    infer_doc_onnx=infer_doc_onnx,
                    image_path=image_path,
                    page_index=page_index,
                    layout_threshold=args.layout_threshold,
                )
                metrics.layout_s += page.layout_s
                metrics.page_prepare_total_s += page.prepare_page_total_s
                pending_pages.append(page)
                print(
                    f"OPENDOC_CONTINUOUS_PAGE_READY "
                    f"index={page_index + 1}/{len(image_paths)} "
                    f"image={image_path.name} crops={len(page.crops)}",
                    flush=True,
                )
                flush_ready_pages()
                for crop in page.crops:
                    item = runner.prefill_image_for_cohort(
                        crop.image,
                        image_source=crop.request_id,
                    )
                    metrics.prepare_s += float(item.prep["prepare_total_s"])
                    metrics.prefill_s += item.prefill_s
                    yield ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=item,
                    )

        def complete_crop(completed_item: ContinuousCompletedItem) -> None:
            crop = completed_item.payload
            if not isinstance(crop, CropRequest):
                raise TypeError(
                    "Continuous scheduler returned an unexpected crop payload: "
                    f"{type(crop)!r}"
                )
            result = completed_item.result
            crop.result = result
            metrics.crop_records.append(
                {
                    "request_id": crop.request_id,
                    "page": crop.page_name,
                    "page_index": crop.page_index,
                    "crop_index": crop.crop_index,
                    "label": crop.label,
                    "crop_size": [crop.image.width, crop.image.height],
                    "processed_image_size": result["prep"]["processed_image_size"],
                    "encoder_seq_len_hint": result["prep"]["encoder_seq_len_hint"],
                    "token_ids": result["generated_ids"],
                    "text": result["text"],
                    "token_count": result["generated_token_count"],
                    "decode_token_count": result["decode_generated_token_count"],
                    "prefill_s": result["ttft_s"],
                    "decode_slot": completed_item.slot,
                    "admission_index": completed_item.admission_index,
                    "completion_index": completed_item.completion_index,
                }
            )
            print(
                f"UNIREC_CONTINUOUS_CROP_END request_id={crop.request_id} "
                f"slot={completed_item.slot} "
                f"admission={completed_item.admission_index} "
                f"completion={completed_item.completion_index} "
                f"tokens={result['generated_token_count']}",
                flush=True,
            )
            flush_ready_pages()

        continuous_runner = ContinuousUniRecDecoder(
            runner=runner,
            batch_size=args.decode_batch_size,
            max_length=args.max_length,
            decode_mode=args.decode_mode,
            compile_backend=args.compile_backend,
        )
        continuous_decode = continuous_runner.run(
            ready_source(),
            on_complete=complete_crop,
        )
        metrics.decode_s = float(continuous_decode["decode_s"])
        metrics.raw_decode_token_slots = int(
            continuous_decode["raw_decode_token_slots"]
        )
        metrics.effective_decode_tokens = int(
            continuous_decode["effective_decode_tokens"]
        )
        metrics.idle_decode_token_slots = int(
            continuous_decode["idle_decode_token_slots"]
        )
        metrics.padding_decode_token_slots = (
            metrics.raw_decode_token_slots - metrics.effective_decode_tokens
        )
        print(
            "UNIREC_CONTINUOUS_END "
            + json.dumps(continuous_decode, ensure_ascii=False),
            flush=True,
        )
    else:
        for page_index, image_path in enumerate(image_paths):
            page = prepare_page(
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
                image_path=image_path,
                page_index=page_index,
                layout_threshold=args.layout_threshold,
            )
            metrics.layout_s += page.layout_s
            metrics.page_prepare_total_s += page.prepare_page_total_s
            pending_pages.append(page)
            pending_crops.extend(page.crops)
            print(
                f"OPENDOC_BATCHED_PAGE_READY index={page_index + 1}/{len(image_paths)} "
                f"image={image_path.name} crops={len(page.crops)} "
                f"queued={len(pending_crops)}",
                flush=True,
            )
            while len(pending_crops) >= args.decode_batch_size:
                cohort = [
                    pending_crops.popleft() for _ in range(args.decode_batch_size)
                ]
                recognize_cohort(
                    cohort=cohort,
                    target_batch_size=args.decode_batch_size,
                    runner=runner,
                    args=args,
                    metrics=metrics,
                )
                flush_ready_pages()
            flush_ready_pages()

        if pending_crops:
            final_cohort = list(pending_crops)
            pending_crops.clear()
            recognize_cohort(
                cohort=final_cohort,
                target_batch_size=args.decode_batch_size,
                runner=runner,
                args=args,
                metrics=metrics,
            )
    flush_ready_pages()
    if pending_pages:
        raise RuntimeError(f"Unfinished pages remain after final cohort: {len(pending_pages)}")

    pipeline_wall_s = time.perf_counter() - pipeline_started
    trace_path = output_dir / "recognition_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in sorted(
            metrics.crop_records,
            key=lambda item: (item["page_index"], item["crop_index"]),
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "openocr_root": str(openocr_root),
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "decode_mode": args.decode_mode,
        "decode_scheduling": args.decode_scheduling,
        "decode_batch_size": args.decode_batch_size,
        "max_length": args.max_length,
        "layout_backend": args.layout_backend,
        "layout_dtype": args.layout_dtype if not use_onnx_layout else None,
        "setup_s": setup_s,
        "pipeline_wall_s": pipeline_wall_s,
        "pages_per_s": len(image_paths) / pipeline_wall_s,
        "page_count": len(image_paths),
        "crop_count": len(metrics.crop_records),
        "cohort_count": len(metrics.cohort_records),
        "layout_s": metrics.layout_s,
        "page_prepare_total_s": metrics.page_prepare_total_s,
        "page_frontend_other_s": metrics.page_prepare_total_s - metrics.layout_s,
        "prepare_s": metrics.prepare_s,
        "prefill_s": metrics.prefill_s,
        "decode_s": metrics.decode_s,
        "output_assembly_s": metrics.output_assembly_s,
        "output_write_s": metrics.output_write_s,
        "raw_decode_token_slots": metrics.raw_decode_token_slots,
        "effective_decode_tokens": metrics.effective_decode_tokens,
        "padding_decode_token_slots": metrics.padding_decode_token_slots,
        "idle_decode_token_slots": metrics.idle_decode_token_slots,
        "raw_decode_tokens_per_s": (
            metrics.raw_decode_token_slots / metrics.decode_s
            if metrics.decode_s > 0
            else None
        ),
        "effective_decode_tokens_per_s": (
            metrics.effective_decode_tokens / metrics.decode_s
            if metrics.decode_s > 0
            else None
        ),
        "cohorts": metrics.cohort_records,
        "continuous_decode": continuous_decode,
        "pages": metrics.page_records,
        "trace_path": str(trace_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("OPENDOC_BATCHED_RUN_END " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
