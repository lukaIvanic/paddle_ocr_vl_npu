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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from continuous_unirec import (
    ContinuousCompletedItem,
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
)
from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    OptimizedUniRecRunner,
    synchronize_device,
)
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter
from text_packed_prefill import PACKED_TEXT_PREFILL_BUCKET
from vision_atlas import (
    ATLAS_CHANNELS,
    ATLAS_HEIGHT,
    ATLAS_MAX_MEMBERS,
    ATLAS_WIDTH,
    UniRecVisionAtlasRuntime,
)


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
    image: np.ndarray
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
    frontend_timing_s: dict[str, float]

    def is_ready(self) -> bool:
        return all(crop.result is not None for crop in self.crops)


@dataclass
class RunMetrics:
    cohort_records: list[dict[str, Any]] = field(default_factory=list)
    crop_records: list[dict[str, Any]] = field(default_factory=list)
    page_records: list[dict[str, Any]] = field(default_factory=list)
    layout_s: float = 0.0
    page_prepare_total_s: float = 0.0
    frontend_timing_s: dict[str, float] = field(default_factory=dict)
    prepare_s: float = 0.0
    prefill_s: float = 0.0
    prefill_device_stage_s: dict[str, float] = field(default_factory=dict)
    text_prefill_real_source_tokens: int = 0
    text_prefill_physical_source_tokens: int = 0
    decode_s: float = 0.0
    output_assembly_s: float = 0.0
    output_write_s: float = 0.0
    output_write_backpressure_s: float = 0.0
    output_write_final_drain_s: float = 0.0
    output_write_max_pending: int = 0
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
    parser.add_argument(
        "--layout-execution",
        choices=("eager", "torchair"),
        default="eager",
    )
    parser.add_argument(
        "--layout-compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_detector_torchair"
        ),
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
    parser.add_argument(
        "--page-decode-workers",
        type=int,
        default=4,
        help=(
            "Bounded OpenCV page-decode workers. Decoding stays exact BGR and "
            "runs ahead of the serialized layout and recognition consumers."
        ),
    )
    parser.add_argument(
        "--text-prefill-mode",
        choices=("eager", "compiled_s512", "compiled_packed_s1024"),
        default="eager",
        help=(
            "Recognition text-prefill execution; compiled_s512 pads each crop "
            "to 512, while compiled_packed_s1024 greedily combines crops into "
            "one B1 source sequence padded to 1024"
        ),
    )
    parser.add_argument(
        "--vision-prefill-mode",
        choices=("eager", "compiled_atlas_stage2"),
        default="eager",
        help=(
            "Vision execution. compiled_atlas_stage2 packs crop-local stage-2 "
            "feature maps into the validated guarded 64x192 atlas graph"
        ),
    )
    parser.add_argument(
        "--prefill-device-timing",
        action="store_true",
        help="Record NPU event timing for each recognition-prefill stage",
    )
    return parser.parse_args()


def accumulate_stage_seconds(
    destination: dict[str, float],
    source: dict[str, float] | None,
) -> None:
    if source is None:
        return
    for name, seconds in source.items():
        destination[name] = destination.get(name, 0.0) + float(seconds)


@dataclass(frozen=True)
class DecodedPage:
    image_path: Path
    image: np.ndarray
    started_at: float
    timing_s: dict[str, float]


def decode_page_bgr(image_path: Path) -> DecodedPage:
    """Read and decode one page with the existing exact OpenCV BGR contract."""
    started_at = time.perf_counter()
    read_started = time.perf_counter()
    encoded = image_path.read_bytes()
    read_s = time.perf_counter() - read_started
    decode_started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    decode_s = time.perf_counter() - decode_started
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return DecodedPage(
        image_path=image_path,
        image=image,
        started_at=started_at,
        timing_s={
            "page_file_read_s": read_s,
            "page_image_decode_s": decode_s,
        },
    )


def iter_decoded_pages(
    image_paths: list[Path],
    *,
    workers: int,
) -> Iterable[DecodedPage]:
    """Decode pages concurrently but yield them in exact input order."""
    if workers < 1:
        raise ValueError("page decode workers must be >= 1")
    max_pending = max(1, workers * 2)
    next_index = 0
    pending: deque[Future[DecodedPage]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="unirec-page-decode",
    ) as executor:
        while next_index < len(image_paths) and len(pending) < max_pending:
            pending.append(executor.submit(decode_page_bgr, image_paths[next_index]))
            next_index += 1
        while pending:
            yield pending.popleft().result()
            if next_index < len(image_paths):
                pending.append(executor.submit(decode_page_bgr, image_paths[next_index]))
                next_index += 1


def record_prefill_metrics(metrics: RunMetrics, item: Any) -> None:
    metrics.prepare_s += float(item.prep["prepare_total_s"])
    metrics.prefill_s += float(item.prefill_s)
    metrics.text_prefill_real_source_tokens += int(
        item.text_prefill_real_source_tokens or 0
    )
    metrics.text_prefill_physical_source_tokens += int(
        item.text_prefill_physical_source_tokens or 0
    )
    accumulate_stage_seconds(
        metrics.prefill_device_stage_s,
        item.prefill_device_stage_s,
    )


def warmup_configured_graphs(
    *,
    args: argparse.Namespace,
    runner: OptimizedUniRecRunner,
    vision_atlas_runtime: UniRecVisionAtlasRuntime | None,
    passes: int = 2,
) -> dict[str, Any]:
    """Load and replay every configured graph before pipeline timing starts."""
    if passes < 1:
        raise ValueError("graph warmup passes must be >= 1")
    device = torch.device(runner.device)
    report: dict[str, Any] = {"passes": passes, "graphs": {}}
    warmup_started = time.perf_counter()
    print(f"UNIREC_GRAPH_WARMUP_BEGIN passes={passes}", flush=True)

    with torch.inference_mode():
        if vision_atlas_runtime is not None:
            cells = ATLAS_HEIGHT * ATLAS_WIDTH
            identity = torch.arange(cells, dtype=torch.long, device=device)
            valid_mask = torch.ones(
                (1, 1, ATLAS_HEIGHT, ATLAS_WIDTH),
                dtype=runner.dtype,
                device=device,
            )
            membership = torch.zeros(
                (ATLAS_MAX_MEMBERS, cells),
                dtype=runner.dtype,
                device=device,
            )
            membership[0].fill_(1)
            normalized_membership = membership / float(cells)
            atlas_inputs = (
                torch.zeros(
                    (1, cells, ATLAS_CHANNELS),
                    dtype=runner.dtype,
                    device=device,
                ),
                identity,
                identity,
                valid_mask,
                membership,
                normalized_membership,
            )
            pass_times = []
            for pass_index in range(passes):
                started = time.perf_counter()
                _ = vision_atlas_runtime.compiled(*atlas_inputs)
                synchronize_device(runner.device)
                elapsed = time.perf_counter() - started
                pass_times.append(elapsed)
                print(
                    "UNIREC_GRAPH_WARMUP_PASS "
                    f"graph=vision_atlas_stage2 pass={pass_index + 1}/{passes} "
                    f"wall_s={elapsed:.3f}",
                    flush=True,
                )
            vision_atlas_runtime.first_call = False
            report["graphs"]["vision_atlas_stage2"] = {
                "pass_wall_s": pass_times,
                "cache_dir": str(vision_atlas_runtime.cache_dir),
            }

        if args.text_prefill_mode == "compiled_packed_s1024":
            text_runtime = runner._get_compiled_packed_text_prefill_runtime()
            text_input = torch.zeros(
                (1, text_runtime.bucket, runner.config.d_model),
                dtype=runner.dtype,
                device=device,
            )
            pass_times = []
            for pass_index in range(passes):
                started = time.perf_counter()
                _ = text_runtime.compiled(text_input)
                synchronize_device(runner.device)
                elapsed = time.perf_counter() - started
                pass_times.append(elapsed)
                print(
                    "UNIREC_GRAPH_WARMUP_PASS "
                    f"graph=text_prefill_packed_s1024 "
                    f"pass={pass_index + 1}/{passes} wall_s={elapsed:.3f}",
                    flush=True,
                )
            text_runtime._first_call = False
            report["graphs"]["text_prefill_packed_s1024"] = {
                "pass_wall_s": pass_times,
                "cache_dir": str(text_runtime.cache_dir),
            }

        if args.decode_mode.startswith("compiled"):
            shape_started = time.perf_counter()
            cross_cache_len = runner._get_static_cross_cache_len()
            shape_discovery_s = time.perf_counter() - shape_started
            self_attention_backend = (
                "increfa" if args.decode_mode == "compiled_ifa" else "eager"
            )
            decode_module, decode_metadata = runner._compile_decode_module(
                backend=args.compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=False,
                cross_cache_len=cross_cache_len,
                batch_size=args.decode_batch_size,
            )
            batch_size = args.decode_batch_size
            heads = runner.config.decoder_attention_heads
            head_dim = runner.config.d_model // heads
            layer_count = runner.config.decoder_layers
            self_keys = tuple(
                torch.zeros(
                    (batch_size, heads, LOCAL_UNIREC_STATIC_CACHE_LEN, head_dim),
                    dtype=runner.dtype,
                    device=device,
                )
                for _ in range(layer_count)
            )
            self_values = tuple(torch.zeros_like(tensor) for tensor in self_keys)
            cross_keys = tuple(
                torch.zeros(
                    (batch_size, heads, cross_cache_len, head_dim),
                    dtype=runner.dtype,
                    device=device,
                )
                for _ in range(layer_count)
            )
            cross_values = tuple(torch.zeros_like(tensor) for tensor in cross_keys)
            cross_mask = torch.zeros(
                (batch_size, 1, 1, cross_cache_len),
                dtype=torch.float32,
                device=device,
            )
            decode_inputs = (
                torch.full(
                    (batch_size, 1),
                    int(runner.config.decoder_start_token_id),
                    dtype=torch.long,
                    device=device,
                ),
                torch.ones((batch_size,), dtype=torch.int64, device=device),
                1 if self_attention_backend == "increfa" else 0,
                self_keys,
                self_values,
                cross_keys,
                cross_values,
                cross_mask,
            )
            pass_times = []
            for pass_index in range(passes):
                started = time.perf_counter()
                _ = decode_module(*decode_inputs)
                synchronize_device(runner.device)
                elapsed = time.perf_counter() - started
                pass_times.append(elapsed)
                print(
                    "UNIREC_GRAPH_WARMUP_PASS "
                    f"graph=decode_b{batch_size} pass={pass_index + 1}/{passes} "
                    f"wall_s={elapsed:.3f}",
                    flush=True,
                )
            report["graphs"][f"decode_b{batch_size}"] = {
                "pass_wall_s": pass_times,
                "shape_discovery_s": shape_discovery_s,
                "cross_cache_len": cross_cache_len,
                "cache_dir": decode_metadata.get("torchair_cache_dir"),
            }

    report["wall_s"] = time.perf_counter() - warmup_started
    print(
        "UNIREC_GRAPH_WARMUP_END " + json.dumps(report, ensure_ascii=False),
        flush=True,
    )
    return report


def iter_greedy_text_packs(
    crops: Any,
    *,
    runner: OptimizedUniRecRunner,
) -> Any:
    """FIFO greedy packs; an over-capacity crop is an eager singleton."""
    current: list[CropRequest] = []
    current_tokens = 0
    for crop in crops:
        tokens = int(
            runner.processor.estimate_encoder_token_count_for_image_size(
                crop.image.width,
                crop.image.height,
            )
        )
        if tokens > PACKED_TEXT_PREFILL_BUCKET:
            if current:
                yield True, current
                current = []
                current_tokens = 0
            yield False, [crop]
            continue
        if current and current_tokens + tokens > PACKED_TEXT_PREFILL_BUCKET:
            yield True, current
            current = []
            current_tokens = 0
        current.append(crop)
        current_tokens += tokens
    if current:
        yield True, current


def prefill_crop_group(
    *,
    crops: list[CropRequest],
    use_packed_graph: bool,
    runner: OptimizedUniRecRunner,
    vision_atlas_runtime: UniRecVisionAtlasRuntime | None,
    args: argparse.Namespace,
) -> list[Any]:
    if use_packed_graph:
        if vision_atlas_runtime is not None:
            return vision_atlas_runtime.prefill_images_packed_for_cohort(
                [(crop.image, crop.request_id) for crop in crops],
                profile_device_stages=args.prefill_device_timing,
            )
        return runner.prefill_images_packed_for_cohort(
            [(crop.image, crop.request_id) for crop in crops],
            profile_device_stages=args.prefill_device_timing,
        )
    if args.text_prefill_mode == "compiled_packed_s1024":
        if len(crops) != 1:
            raise AssertionError("packed text fallback must be a singleton")
        runner.record_packed_text_prefill_fallback()
        mode = "eager"
    else:
        mode = args.text_prefill_mode
    return [
        runner.prefill_image_for_cohort(
            crop.image,
            image_source=crop.request_id,
            profile_device_stages=args.prefill_device_timing,
            text_prefill_mode=mode,
        )
        for crop in crops
    ]


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
    decoded: DecodedPage,
    page_index: int,
    layout_threshold: float,
) -> PageRequest:
    started_at = decoded.started_at
    image_path = decoded.image_path
    image = decoded.image
    frontend_timing_s = dict(decoded.timing_s)
    height, width = image.shape[:2]

    layout_started = time.perf_counter()
    layout_results = pipeline.layout_detector(
        [image],
        threshold=layout_threshold,
    )[0]
    layout_s = time.perf_counter() - layout_started
    frontend_timing_s["layout_s"] = layout_s
    image_labels = (
        infer_doc_onnx.IMAGE_LABELS
        if pipeline.use_chart_recognition
        else infer_doc_onnx.IMAGE_LABELS + ["chart"]
    )

    crop_boxes_started = time.perf_counter()
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
    frontend_timing_s["layout_crop_views_s"] = (
        time.perf_counter() - crop_boxes_started
    )

    image_index_started = time.perf_counter()
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
    frontend_timing_s["document_image_index_s"] = (
        time.perf_counter() - image_index_started
    )

    recognition_crops_started = time.perf_counter()
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
    frontend_timing_s["recognition_crop_build_s"] = (
        time.perf_counter() - recognition_crops_started
    )

    prepare_page_total_s = sum(frontend_timing_s.values())
    return PageRequest(
        page_index=page_index,
        image_path=image_path,
        image=image,
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
        frontend_timing_s=frontend_timing_s,
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
        "_page_image": page.image,
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
    vision_atlas_runtime: UniRecVisionAtlasRuntime | None,
    args: argparse.Namespace,
    metrics: RunMetrics,
) -> None:
    cohort_index = len(metrics.cohort_records)
    print(
        f"UNIREC_BATCH_BEGIN index={cohort_index} real={len(cohort)} "
        f"physical={target_batch_size}",
        flush=True,
    )
    prefilled_by_request: dict[str, Any] = {}
    if args.text_prefill_mode == "compiled_packed_s1024":
        groups = iter_greedy_text_packs(cohort, runner=runner)
    else:
        groups = [(False, cohort)]
    for use_packed_graph, crops in groups:
        items = prefill_crop_group(
            crops=crops,
            use_packed_graph=use_packed_graph,
            runner=runner,
            vision_atlas_runtime=vision_atlas_runtime,
            args=args,
        )
        for crop, item in zip(crops, items):
            prefilled_by_request[crop.request_id] = item
            record_prefill_metrics(metrics, item)
    prefilled = [prefilled_by_request[crop.request_id] for crop in cohort]
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
                "prefill_device_stage_s": result.get("prefill_device_stage_s"),
                "text_prefill_execution": result.get("text_prefill_execution"),
                "text_prefill_real_source_tokens": result.get(
                    "text_prefill_real_source_tokens"
                ),
                "text_prefill_physical_source_tokens": result.get(
                    "text_prefill_physical_source_tokens"
                ),
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
    if args.page_decode_workers < 1:
        raise ValueError("--page-decode-workers must be >= 1")
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
            execution=args.layout_execution,
            compile_cache_dir=args.layout_compile_cache_dir,
        )
        pipeline.use_layout_detection = True
    runner = OptimizedUniRecRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=(
            args.compile_cache_dir.expanduser().resolve()
            if (
                args.decode_mode.startswith("compiled")
                or args.text_prefill_mode
                in {"compiled_s512", "compiled_packed_s1024"}
                or args.vision_prefill_mode == "compiled_atlas_stage2"
            )
            else None
        ),
    )
    if (
        args.vision_prefill_mode == "compiled_atlas_stage2"
        and args.text_prefill_mode != "compiled_packed_s1024"
    ):
        raise ValueError(
            "compiled_atlas_stage2 currently requires "
            "--text-prefill-mode compiled_packed_s1024"
        )
    vision_atlas_runtime = (
        UniRecVisionAtlasRuntime(runner)
        if args.vision_prefill_mode == "compiled_atlas_stage2"
        else None
    )
    graph_warmup = warmup_configured_graphs(
        args=args,
        runner=runner,
        vision_atlas_runtime=vision_atlas_runtime,
    )
    if not use_onnx_layout:
        pipeline.layout_detector.warmup_graph()
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
    pending_writes: deque[
        tuple[PageRequest, Future[tuple[float, float]]]
    ] = deque()
    write_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="unirec-page-writer",
    )
    max_pending_writes = 8
    pipeline_started = time.perf_counter()
    written_pages = 0
    continuous_decode: dict[str, Any] | None = None

    def write_page(result: dict[str, Any]) -> tuple[float, float]:
        started = time.perf_counter()
        pipeline.save_to_json(result, str(output_dir))
        pipeline.save_to_markdown(result, str(output_dir))
        completed_at = time.perf_counter()
        return completed_at - started, completed_at

    def record_completed_write(
        page: PageRequest,
        future: Future[tuple[float, float]],
    ) -> None:
        nonlocal written_pages
        write_s, completed_at = future.result()
        metrics.output_write_s += write_s
        written_pages += 1
        page_s = completed_at - page.started_at
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

    def drain_completed_writes(*, wait: bool, count: int | None = None) -> None:
        drained = 0
        while pending_writes and (count is None or drained < count):
            page, future = pending_writes[0]
            if not wait and not future.done():
                break
            pending_writes.popleft()
            record_completed_write(page, future)
            drained += 1

    def submit_page_write(page: PageRequest, result: dict[str, Any]) -> None:
        drain_completed_writes(wait=False)
        if len(pending_writes) >= max_pending_writes:
            wait_started = time.perf_counter()
            drain_completed_writes(wait=True, count=1)
            metrics.output_write_backpressure_s += (
                time.perf_counter() - wait_started
            )
        pending_writes.append((page, write_executor.submit(write_page, result)))
        metrics.output_write_max_pending = max(
            metrics.output_write_max_pending,
            len(pending_writes),
        )

    def flush_ready_pages() -> None:
        drain_completed_writes(wait=False)
        while pending_pages and pending_pages[0].is_ready():
            page = pending_pages.popleft()
            assembly_started = time.perf_counter()
            result = assemble_page(
                page=page,
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
            )
            metrics.output_assembly_s += time.perf_counter() - assembly_started
            submit_page_write(page, result)

    if args.decode_scheduling == "continuous":
        def crop_source():
            decoded_pages = iter_decoded_pages(
                image_paths,
                workers=args.page_decode_workers,
            )
            for page_index, decoded in enumerate(decoded_pages):
                page = prepare_page(
                    pipeline=pipeline,
                    infer_doc_onnx=infer_doc_onnx,
                    decoded=decoded,
                    page_index=page_index,
                    layout_threshold=args.layout_threshold,
                )
                metrics.layout_s += page.layout_s
                metrics.page_prepare_total_s += page.prepare_page_total_s
                accumulate_stage_seconds(
                    metrics.frontend_timing_s,
                    page.frontend_timing_s,
                )
                pending_pages.append(page)
                print(
                    f"OPENDOC_CONTINUOUS_PAGE_READY "
                    f"index={page_index + 1}/{len(image_paths)} "
                    f"image={decoded.image_path.name} crops={len(page.crops)}",
                    flush=True,
                )
                flush_ready_pages()
                for crop in page.crops:
                    yield crop

        def ready_source():
            crops = crop_source()
            if args.text_prefill_mode == "compiled_packed_s1024":
                groups = iter_greedy_text_packs(crops, runner=runner)
            else:
                groups = ((False, [crop]) for crop in crops)
            for use_packed_graph, crop_group in groups:
                items = prefill_crop_group(
                    crops=crop_group,
                    use_packed_graph=use_packed_graph,
                    runner=runner,
                    vision_atlas_runtime=vision_atlas_runtime,
                    args=args,
                )
                for crop, item in zip(crop_group, items):
                    record_prefill_metrics(metrics, item)
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
                    "prefill_device_stage_s": result.get("prefill_device_stage_s"),
                    "text_prefill_execution": result.get("text_prefill_execution"),
                    "text_prefill_real_source_tokens": result.get(
                        "text_prefill_real_source_tokens"
                    ),
                    "text_prefill_physical_source_tokens": result.get(
                        "text_prefill_physical_source_tokens"
                    ),
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
        decoded_pages = iter_decoded_pages(
            image_paths,
            workers=args.page_decode_workers,
        )
        for page_index, decoded in enumerate(decoded_pages):
            page = prepare_page(
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
                decoded=decoded,
                page_index=page_index,
                layout_threshold=args.layout_threshold,
            )
            metrics.layout_s += page.layout_s
            metrics.page_prepare_total_s += page.prepare_page_total_s
            accumulate_stage_seconds(
                metrics.frontend_timing_s,
                page.frontend_timing_s,
            )
            pending_pages.append(page)
            pending_crops.extend(page.crops)
            print(
                f"OPENDOC_BATCHED_PAGE_READY index={page_index + 1}/{len(image_paths)} "
                f"image={decoded.image_path.name} crops={len(page.crops)} "
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
                    vision_atlas_runtime=vision_atlas_runtime,
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
                vision_atlas_runtime=vision_atlas_runtime,
                args=args,
                metrics=metrics,
            )
    flush_ready_pages()
    if pending_pages:
        raise RuntimeError(f"Unfinished pages remain after final cohort: {len(pending_pages)}")
    final_drain_started = time.perf_counter()
    drain_completed_writes(wait=True)
    metrics.output_write_final_drain_s = time.perf_counter() - final_drain_started
    write_executor.shutdown(wait=True)
    if written_pages != len(image_paths):
        raise RuntimeError(
            f"Written page count mismatch: {written_pages} != {len(image_paths)}"
        )

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
        "text_prefill_mode": args.text_prefill_mode,
        "vision_prefill_mode": args.vision_prefill_mode,
        "max_length": args.max_length,
        "layout_backend": args.layout_backend,
        "layout_dtype": args.layout_dtype if not use_onnx_layout else None,
        "layout_execution": args.layout_execution if not use_onnx_layout else None,
        "layout_graph_warmup": (
            pipeline.layout_detector.graph_warmup if not use_onnx_layout else None
        ),
        "page_decode_workers": args.page_decode_workers,
        "setup_s": setup_s,
        "graph_warmup": graph_warmup,
        "pipeline_wall_s": pipeline_wall_s,
        "pages_per_s": len(image_paths) / pipeline_wall_s,
        "page_count": len(image_paths),
        "crop_count": len(metrics.crop_records),
        "cohort_count": len(metrics.cohort_records),
        "layout_s": metrics.layout_s,
        "page_prepare_total_s": metrics.page_prepare_total_s,
        "page_frontend_other_s": metrics.page_prepare_total_s - metrics.layout_s,
        "page_frontend_stage_s": metrics.frontend_timing_s,
        "prepare_s": metrics.prepare_s,
        "prefill_s": metrics.prefill_s,
        "prefill_device_stage_s": metrics.prefill_device_stage_s,
        "text_prefill_real_source_tokens": (
            metrics.text_prefill_real_source_tokens
        ),
        "text_prefill_physical_source_tokens": (
            metrics.text_prefill_physical_source_tokens
        ),
        "text_prefill_useful_token_fraction": (
            metrics.text_prefill_real_source_tokens
            / metrics.text_prefill_physical_source_tokens
            if metrics.text_prefill_physical_source_tokens > 0
            else None
        ),
        "text_prefill_packing": runner.packed_text_prefill_summary(),
        "vision_prefill": (
            vision_atlas_runtime.summary()
            if vision_atlas_runtime is not None
            else {"execution": "eager_per_crop"}
        ),
        "decode_s": metrics.decode_s,
        "output_assembly_s": metrics.output_assembly_s,
        "output_write_s": metrics.output_write_s,
        "output_write_backpressure_s": metrics.output_write_backpressure_s,
        "output_write_final_drain_s": metrics.output_write_final_drain_s,
        "output_write_max_pending": metrics.output_write_max_pending,
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
