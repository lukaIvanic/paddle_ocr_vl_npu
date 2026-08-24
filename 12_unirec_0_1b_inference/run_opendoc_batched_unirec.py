#!/usr/bin/env python3
"""Run OpenDoc with exact B1 prefills and cross-page decode scheduling.

OpenDoc/OpenOCR remains an unmodified dependency.  This runner reuses its
layout detector, crop transforms, result assembly helpers, and writers while
owning the crop queue and UniRec decode scheduling locally.  It supports both
fixed cohorts and a fixed-arena continuous decoder with per-slot hot swapping.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

# Cached serving graphs do not need CANN's default eight TBE compiler workers or
# its separate eight-worker knowledge-bank service.  Each spawned UniRec
# runtime otherwise retains both process pools for the full inference window.
# Preserve explicit overrides for cold-cache compilation experiments.
os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")

import cv2
import numpy as np
import torch
from PIL import Image

from continuous_unirec import (
    ContinuousCompletedItem,
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
    ContinuousWorkerPrefilledItem,
)
from host_memory_diagnostics import emit as emit_host_memory
from tbe_compiler_lifecycle import deinitialize_after_warmup
from layout_process_pool import DynamicLayoutProcessPool, SharedPageLease
from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    OptimizedUniRecRunner,
    UniRecPrefilledItem,
    synchronize_device,
)
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter
from opendoc_layout_npu import (
    DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
    DEFAULT_LAYOUT_WEIGHT_FORMAT,
    LAYOUT_DEPTHWISE_REWRITE_CHOICES,
    LAYOUT_WEIGHT_FORMAT_CHOICES,
)
from shared_byte_budget import SharedByteBudget
from text_packed_prefill import PACKED_TEXT_PREFILL_BUCKET
from vision_atlas import (
    ATLAS_CHANNELS,
    ATLAS_HEIGHT,
    ATLAS_MAX_MEMBERS,
    ATLAS_WIDTH,
    UniRecVisionAtlasRuntime,
)
from vision_static_shape import StaticShapeUniRecVisionRuntime
from vision_bucket_presets import VISION_BUCKET_PRESET_CHOICES


def parse_spatial_shape(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected WIDTHxHEIGHT, got {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return width, height


@dataclass
class CropRequest:
    page_index: int
    crop_index: int
    page_name: str
    image: Image.Image | None
    label: str
    figure_token_map: dict[str, Any]
    source_image_size: tuple[int, int] | None = None
    prepared_pixel_values: np.ndarray | None = None
    worker_cross_kv: np.ndarray | None = None
    worker_prefill_metadata: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    @property
    def request_id(self) -> str:
        return f"page_{self.page_index:06d}_crop_{self.crop_index:04d}"

    @property
    def image_size(self) -> tuple[int, int]:
        if self.source_image_size is not None:
            return self.source_image_size
        if self.image is None:
            raise RuntimeError(f"Crop {self.request_id} has no source image size")
        return int(self.image.width), int(self.image.height)

    def require_image(self) -> Image.Image:
        if self.image is None:
            raise RuntimeError(f"Crop {self.request_id} has no retained image")
        return self.image


@dataclass
class PageRequest:
    page_index: int
    image_path: Path
    image: np.ndarray | None
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
    frontend_storage_lease: SharedPageLease | None = None

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
    rejected_crops: int = 0


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
        "--layout-reading-order-dtype",
        choices=("float16", "float32"),
        default=None,
    )
    parser.add_argument(
        "--layout-execution",
        choices=("eager", "torchair"),
        default="eager",
    )
    parser.add_argument(
        "--layout-weight-format",
        choices=LAYOUT_WEIGHT_FORMAT_CHOICES,
        default=DEFAULT_LAYOUT_WEIGHT_FORMAT,
    )
    parser.add_argument(
        "--layout-depthwise-rewrite",
        choices=LAYOUT_DEPTHWISE_REWRITE_CHOICES,
        default=DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
    )
    parser.add_argument(
        "--layout-preformat-frozen-bn-buffers",
        action="store_true",
    )
    parser.add_argument("--layout-cpu-threads", type=int, default=1)
    parser.add_argument(
        "--layout-batch-size",
        type=int,
        default=1,
        help=(
            "Static layout-model batch size. Page preprocessing and "
            "postprocessing remain independent; only the 800x800 model "
            "forward is batched."
        ),
    )
    parser.add_argument(
        "--layout-compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_detector_torchair"
        ),
    )
    parser.add_argument(
        "--layout-process-workers",
        type=int,
        default=0,
        help=(
            "Precompute B1 layout with this many persistent isolated NPU "
            "processes using one dynamic filepath queue. Requires "
            "--preprocess-all-pages-first and --layout-batch-size 1."
        ),
    )
    parser.add_argument(
        "--prefill-in-layout-workers",
        action="store_true",
        help=(
            "Extend each persistent layout/page worker through recognition "
            "vision and text prefill. Workers return compact real cross-K/V "
            "through shared memory; the coordinator owns continuous decode."
        ),
    )
    parser.add_argument(
        "--shared-cross-kv-budget-gib",
        type=float,
        default=3.5,
        help=(
            "Maximum live page-scoped CPU shared memory while worker prefill "
            "streams into decode. The default leaves headroom below 5 GB."
        ),
    )
    parser.add_argument(
        "--worker-empty-cache-after-page",
        action="store_true",
        help=(
            "Experimental HBM control: after a layout worker finishes one "
            "page's recognition prefill and all temporary NPU tensors leave "
            "scope, return unused cached allocator blocks to the device."
        ),
    )
    parser.add_argument(
        "--vision-bucket-preset",
        choices=VISION_BUCKET_PRESET_CHOICES,
        default="production_v1",
    )
    parser.add_argument(
        "--vision-focal-depthwise-rewrite",
        choices=(
            "native",
            "constant",
            "constant_grouped",
            "constant_grouped_all",
            "aligned_spatial",
        ),
        default="native",
    )
    parser.add_argument(
        "--vision-weight-format",
        choices=("native", "focal_prepack", "torchair_internal"),
        default="native",
    )
    parser.add_argument("--recognition-preprocess-threads", type=int, default=8)
    parser.add_argument(
        "--recognition-input-contract",
        choices=("legacy_float_chw", "compact_uint8_hwc"),
        default="compact_uint8_hwc",
    )
    parser.add_argument("--stock-encoder", type=Path)
    parser.add_argument("--stock-decoder", type=Path)
    parser.add_argument("--stock-tokenizer-mapping", type=Path)
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
    parser.add_argument("--self-cache-length", type=int, default=1024)
    parser.add_argument("--cross-cache-length", type=int, default=512)
    parser.add_argument(
        "--decode-weight-format",
        choices=("native", "nz"),
        default="native",
    )
    parser.add_argument("--decode-lm-head-rows", type=int, default=0)
    parser.add_argument(
        "--decode-admission-prefetch-depth",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decode-live-arena-warmup-passes",
        type=int,
        default=2,
    )
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
        "--page-image-decoder",
        choices=("opencv", "kornia_torchvision"),
        default="opencv",
        help=(
            "Decode pages with the historical OpenCV BGR path or with the "
            "Kornia-RS PNG / TorchVision JPEG dispatch used by experiment 09. "
            "The fast path converts back to contiguous BGR before OpenDoc."
        ),
    )
    parser.add_argument(
        "--preprocess-all-pages-first",
        action="store_true",
        help=(
            "Finish page decode, layout, and crop construction for the full "
            "selected page set before recognition starts."
        ),
    )
    parser.add_argument(
        "--page-prepare-workers",
        type=int,
        default=1,
        help=(
            "Ordered post-layout page/crop construction workers. Values above "
            "one require --preprocess-all-pages-first."
        ),
    )
    parser.add_argument(
        "--pipeline-warmup-pages",
        type=int,
        default=2,
        help=(
            "Run the first N pages through the complete configured pipeline, "
            "discard their measurements, then restart the measured page set "
            "from its first page. Use 0 to retain synthetic graph warmup."
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
        choices=(
            "eager",
            "compiled_atlas_stage2",
            "compiled_full_buckets",
        ),
        default="eager",
        help=(
            "Vision execution. compiled_atlas_stage2 packs crop-local stage-2 "
            "feature maps into the validated guarded 64x192 atlas graph; "
            "compiled_full_buckets runs five masked fixed-canvas full-encoder "
            "graphs inside layout/prefill workers"
        ),
    )
    parser.add_argument(
        "--vision-page-lookahead",
        type=int,
        default=4,
        help=(
            "Maximum pages one layout/prefill worker may combine into its "
            "worker-local full-vision batches. Used only by "
            "compiled_full_buckets."
        ),
    )
    parser.add_argument(
        "--vision-spatial-execution",
        choices=("eager", "compiled_static"),
        default="eager",
        help=(
            "Execution for the crop-local vision prefix and suffix around the "
            "stage-2 atlas. compiled_static requires --recognition-shape-filter."
        ),
    )
    parser.add_argument(
        "--recognition-shape-filter",
        type=parse_spatial_shape,
        metavar="WIDTHxHEIGHT",
        help=(
            "Keep only crops whose UniRec processed image size exactly matches "
            "WIDTHxHEIGHT. Intended for controlled vision experiments."
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


def filter_page_recognition_shapes(
    page: PageRequest,
    *,
    runner: OptimizedUniRecRunner,
    target: tuple[int, int] | None,
) -> int:
    if target is None:
        return 0
    kept_crops: list[CropRequest] = []
    kept_block_ids: list[int] = []
    for crop, block_id in zip(page.crops, page.vlm_block_ids):
        crop_width, crop_height = crop.image_size
        processed = runner.processor.get_processed_size(
            crop_width,
            crop_height,
        )
        if processed == target:
            kept_crops.append(crop)
            kept_block_ids.append(block_id)
    rejected = len(page.crops) - len(kept_crops)
    page.crops = kept_crops
    page.vlm_block_ids = kept_block_ids
    return rejected


@dataclass(frozen=True)
class DecodedPage:
    image_path: Path
    image: np.ndarray
    started_at: float
    timing_s: dict[str, float]


def decode_page_bgr(
    image_path: Path,
    *,
    decoder: str = "opencv",
) -> DecodedPage:
    """Read one page and return the existing contiguous OpenCV BGR contract."""
    started_at = time.perf_counter()
    read_started = time.perf_counter()
    encoded = image_path.read_bytes()
    read_s = time.perf_counter() - read_started
    decode_started = time.perf_counter()
    if decoder == "opencv":
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    elif decoder == "kornia_torchvision":
        from kornia_rs.image import Image as KorniaImage
        from torchvision.io import ImageReadMode, decode_image

        if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
            image_rgb = KorniaImage.decode(encoded, "RGB").data
        else:
            encoded_tensor = torch.frombuffer(
                bytearray(encoded),
                dtype=torch.uint8,
            )
            image_rgb = (
                decode_image(encoded_tensor, mode=ImageReadMode.RGB)
                .permute(1, 2, 0)
                .numpy()
            )
        image = np.ascontiguousarray(image_rgb[..., ::-1])
    else:
        raise ValueError(f"unsupported page image decoder: {decoder}")
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
    decoder: str = "opencv",
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
            pending.append(
                executor.submit(
                    decode_page_bgr,
                    image_paths[next_index],
                    decoder=decoder,
                )
            )
            next_index += 1
        while pending:
            yield pending.popleft().result()
            if next_index < len(image_paths):
                pending.append(
                    executor.submit(
                        decode_page_bgr,
                        image_paths[next_index],
                        decoder=decoder,
                    )
                )
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


def record_direct_arena_admission_metrics(
    metrics: RunMetrics,
    decode_summary: dict[str, Any],
) -> None:
    timing = decode_summary.get("timing_detail") or {}
    admission_s = float(
        timing.get("initial_arena_admission_enqueue_s", 0.0)
    ) + float(timing.get("cache_refill_direct_admission_enqueue_s", 0.0))
    if admission_s <= 0:
        return
    metrics.prefill_s += admission_s
    metrics.prefill_device_stage_s[
        "coordinator_direct_arena_admission_enqueue"
    ] = (
        metrics.prefill_device_stage_s.get(
            "coordinator_direct_arena_admission_enqueue",
            0.0,
        )
        + admission_s
    )


def warmup_configured_graphs(
    *,
    args: argparse.Namespace,
    runner: OptimizedUniRecRunner,
    vision_atlas_runtime: UniRecVisionAtlasRuntime | None,
    passes: int = 2,
    warmup_decode: bool = True,
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
            atlas_to_source = torch.arange(
                cells,
                dtype=torch.long,
                device=device,
            )
            source_to_atlas = atlas_to_source.clone()
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
                atlas_to_source,
                source_to_atlas,
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
            if isinstance(vision_atlas_runtime, StaticShapeUniRecVisionRuntime):
                report["graphs"].update(
                    vision_atlas_runtime.warmup_static_graphs(passes=passes)
                )

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

        if args.decode_mode.startswith("compiled") and warmup_decode:
            shape_started = time.perf_counter()
            cross_cache_len = runner._get_static_cross_cache_len()
            shape_discovery_s = time.perf_counter() - shape_started
            self_attention_backend = (
                "increfa_all" if args.decode_mode == "compiled_ifa" else "eager"
            )
            decode_module, decode_metadata = runner._compile_decode_module(
                backend=args.compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=False,
                cross_cache_len=cross_cache_len,
                batch_size=args.decode_batch_size,
            )
            batch_size = args.decode_batch_size
            # Use the production allocator itself. This keeps every Dynamo
            # tensor guard identical across warmup and the later continuous
            # decoder: inference-tensor state, shape, stride, storage aliasing,
            # NPU format, and the packed cross-K/V view relationships.
            warmup_decoder = ContinuousUniRecDecoder(
                runner=runner,
                batch_size=batch_size,
                max_length=LOCAL_UNIREC_STATIC_CACHE_LEN,
                decode_mode=args.decode_mode,
                compile_backend=args.compile_backend,
                compile_dynamic=False,
            )
            warmup_cache = warmup_decoder._allocate_empty_arena()
            if warmup_cache.cross_attention_mask is None:
                raise RuntimeError("decode warmup has no cross-attention mask")
            # An empty arena starts fully masked. Real production admits rows
            # before its first graph call, so never exercise the 310P attention
            # kernel with an artificial all-masked cross-attention input.
            warmup_cache.cross_attention_mask.zero_()
            decoder_input_ids, cache_position = (
                ContinuousUniRecDecoder._allocate_decode_device_inputs(
                    batch_size,
                    device,
                )
            )
            decoder_input_ids.fill_(
                int(runner.config.decoder_start_token_id)
            )
            cache_position.fill_(1)
            decode_inputs = (
                decoder_input_ids,
                cache_position,
                0,
                warmup_cache.key_cache,
                warmup_cache.value_cache,
                warmup_cache.cross_key_cache,
                warmup_cache.cross_value_cache,
                warmup_cache.cross_attention_mask,
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
                "production_arena_allocator": True,
                "device_inputs_are_inference_tensors": bool(
                    decoder_input_ids.is_inference()
                    and cache_position.is_inference()
                ),
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
        crop_width, crop_height = crop.image_size
        tokens = int(
            runner.processor.estimate_encoder_token_count_for_image_size(
                crop_width,
                crop_height,
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
    prepared = [
        (
            runner.prepare_preprocessed_pixels(
                crop.prepared_pixel_values,
                original_image_size=crop.image_size,
                image_source=crop.request_id,
            )
            if crop.prepared_pixel_values is not None
            else runner.prepare_pil_image(
                crop.require_image(),
                image_source=crop.request_id,
            )
        )
        for crop in crops
    ]
    if use_packed_graph:
        if vision_atlas_runtime is not None:
            return vision_atlas_runtime.prefill_prepared_packed_for_cohort(
                prepared,
                profile_device_stages=args.prefill_device_timing,
            )
        return runner.prefill_prepared_images_packed_for_cohort(
            prepared,
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
        runner.prefill_prepared_for_cohort(
            prepared_item,
            profile_device_stages=args.prefill_device_timing,
            text_prefill_mode=mode,
        )
        for prepared_item in prepared
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
    precomputed_layout: dict[str, Any] | None = None,
    measured_layout_s: float | None = None,
) -> PageRequest:
    started_at = decoded.started_at
    image_path = decoded.image_path
    image = decoded.image
    frontend_timing_s = dict(decoded.timing_s)
    height, width = image.shape[:2]

    if precomputed_layout is None:
        layout_started = time.perf_counter()
        layout_results = pipeline.layout_detector(
            [image],
            threshold=layout_threshold,
        )[0]
        layout_s = time.perf_counter() - layout_started
    else:
        if measured_layout_s is None:
            raise ValueError("precomputed layout requires measured layout time")
        layout_results = precomputed_layout
        layout_s = float(measured_layout_s)
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


def page_request_from_process_payload(
    payload: dict[str, Any],
    *,
    measured_layout_s: float,
    shared_byte_budget: SharedByteBudget | None = None,
) -> PageRequest:
    """Materialize a process-owned frontend result without decoding again."""
    started = time.perf_counter()
    page_index = int(payload["page_index"])
    image_path = Path(payload["image_path"])
    shared = payload.get("shared_memory")
    lease = (
        SharedPageLease(
            str(shared["name"]),
            byte_budget=shared_byte_budget,
            reserved_nbytes=int(shared.get("budget_nbytes", 0)),
        )
        if isinstance(shared, dict)
        else None
    )
    if lease is None and (
        "image_bgr_descriptor" in payload
        or any(
            any(name.endswith("_descriptor") for name in crop)
            for crop in payload["crops"]
        )
    ):
        raise RuntimeError("process frontend descriptors have no shared arena")
    image_bgr = (
        lease.array(payload["image_bgr_descriptor"])
        if lease is not None and "image_bgr_descriptor" in payload
        else None
    )
    crops = []
    for crop in payload["crops"]:
        image_rgb = (
            Image.fromarray(lease.array(crop["image_rgb_descriptor"]))
            if lease is not None and "image_rgb_descriptor" in crop
            else None
        )
        source_image_size = crop.get("source_image_size")
        if source_image_size is None:
            if image_rgb is None:
                raise RuntimeError("process crop has no image or source size")
            source_image_size = image_rgb.size
        crops.append(
            CropRequest(
                page_index=page_index,
                crop_index=int(crop["crop_index"]),
                page_name=image_path.name,
                image=image_rgb,
                label=crop["label"],
                figure_token_map=crop["figure_token_map"],
                source_image_size=tuple(
                    int(value) for value in source_image_size
                ),
                prepared_pixel_values=(
                    lease.array(crop["processed_pixel_values_descriptor"])
                    if lease is not None
                    and "processed_pixel_values_descriptor" in crop
                    else None
                ),
                worker_cross_kv=(
                    lease.array(crop["worker_cross_kv_descriptor"])
                    if lease is not None
                    and "worker_cross_kv_descriptor" in crop
                    else None
                ),
                worker_prefill_metadata=crop.get("worker_prefill_metadata"),
            )
        )
    frontend_timing_s = dict(payload["frontend_timing_s"])
    frontend_timing_s["layout_s"] = measured_layout_s
    frontend_timing_s["process_payload_materialize_s"] = (
        time.perf_counter() - started
    )
    return PageRequest(
        page_index=page_index,
        image_path=image_path,
        image=image_bgr,
        width=int(payload["width"]),
        height=int(payload["height"]),
        layout_results=payload["layout_results"],
        blocks=payload["blocks"],
        vlm_block_ids=[int(index) for index in payload["vlm_block_ids"]],
        crops=crops,
        drop_figures_set=set(payload["drop_figures_set"]),
        started_at=float(payload["started_at"]),
        layout_s=measured_layout_s,
        prepare_page_total_s=sum(
            float(value)
            for name, value in frontend_timing_s.items()
            if name.endswith("_s")
        ),
        frontend_timing_s=frontend_timing_s,
        frontend_storage_lease=lease,
    )


def build_worker_prefilled_item(
    crop: CropRequest,
) -> ContinuousWorkerPrefilledItem:
    """Wrap worker-produced CPU cross-K/V for direct decode-arena admission."""
    packed_host = crop.worker_cross_kv
    metadata = crop.worker_prefill_metadata
    if packed_host is None or metadata is None:
        raise RuntimeError(
            f"crop {crop.request_id} has no worker-prefill payload"
        )
    if packed_host.ndim != 5 or int(packed_host.shape[0]) % 2:
        raise RuntimeError(
            f"unexpected worker cross-K/V shape: {packed_host.shape}"
        )
    if packed_host.dtype != np.float16 or not packed_host.flags.c_contiguous:
        raise RuntimeError(
            "worker cross-K/V must be contiguous float16, got "
            f"dtype={packed_host.dtype} contiguous={packed_host.flags.c_contiguous}"
        )
    source_len = int(packed_host.shape[-2])
    if source_len != int(metadata["actual_cross_attention_length"]):
        raise RuntimeError(
            "worker cross-K/V length mismatch: "
            f"tensor={source_len} metadata={metadata['actual_cross_attention_length']}"
        )

    stages = dict(metadata.get("prefill_device_stage_s") or {})
    return ContinuousWorkerPrefilledItem(
        packed_cross_kv=packed_host,
        prep=dict(metadata["prep"]),
        prefill_s=(
            float(metadata["prefill_s"])
            + float(metadata.get("cache_d2h_s", 0.0))
        ),
        actual_cross_attention_length=source_len,
        prefill_device_stage_s=stages,
        text_prefill_execution=str(metadata["text_prefill_execution"]),
        text_prefill_real_source_tokens=int(
            metadata["text_prefill_real_source_tokens"]
        ),
        text_prefill_physical_source_tokens=int(
            metadata["text_prefill_physical_source_tokens"]
        ),
    )


def release_page_frontend_storage(page: PageRequest) -> None:
    """Release a page arena after output writing no longer reads its images."""
    lease = page.frontend_storage_lease
    if lease is None:
        return
    page.image = None
    for crop in page.crops:
        if crop.image is not None:
            crop.image.close()
        crop.image = None
        crop.prepared_pixel_values = None
        crop.worker_cross_kv = None
    page.frontend_storage_lease = None
    lease.close()


@dataclass
class PageCrossKvAdmissionTracker:
    """Release one image-free page arena after its last crop enters NPU HBM."""

    page: PageRequest
    admitted_crops: int = 0
    released_cross_kv_bytes: int = 0

    def __post_init__(self) -> None:
        if self.page.frontend_storage_lease is None:
            raise ValueError("page admission tracking requires a shared arena")
        if self.page.image is not None:
            raise ValueError("early arena release requires image-free payloads")
        if any(crop.image is not None for crop in self.page.crops):
            raise ValueError("early arena release requires image-free crops")

    def release_crop(
        self,
        crop: CropRequest,
        item: ContinuousWorkerPrefilledItem,
    ) -> None:
        packed_cross_kv = item.packed_cross_kv
        if packed_cross_kv is None or crop.worker_cross_kv is None:
            raise RuntimeError(
                f"crop {crop.request_id} cross-K/V was already released"
            )
        if packed_cross_kv is not crop.worker_cross_kv:
            raise RuntimeError(
                f"crop {crop.request_id} admission item does not own its page view"
            )
        self.released_cross_kv_bytes += int(packed_cross_kv.nbytes)
        item.packed_cross_kv = None
        crop.worker_cross_kv = None
        self.admitted_crops += 1
        if self.admitted_crops > len(self.page.crops):
            raise RuntimeError("page admitted more crops than it owns")
        if self.admitted_crops == len(self.page.crops):
            release_page_frontend_storage(self.page)


def iter_prepared_pages(
    *,
    pipeline: Any,
    infer_doc_onnx: Any,
    decoded_pages: Iterable[DecodedPage],
    layout_threshold: float,
    layout_batch_size: int,
    page_prepare_workers: int = 1,
    precomputed_layouts: list[dict[str, Any]] | None = None,
    precomputed_layout_page_s: float = 0.0,
) -> Iterable[PageRequest]:
    """Batch only layout inference, then restore exact page order."""
    if page_prepare_workers < 1:
        raise ValueError("page prepare workers must be >= 1")
    source = iter(decoded_pages)
    page_index = 0
    executor = (
        ThreadPoolExecutor(
            max_workers=page_prepare_workers,
            thread_name_prefix="unirec-page-prepare",
        )
        if page_prepare_workers > 1
        else None
    )
    try:
        while True:
            batch = list(islice(source, layout_batch_size))
            if not batch:
                return
            if precomputed_layouts is None:
                layout_started = time.perf_counter()
                layout_results = pipeline.layout_detector(
                    [decoded.image for decoded in batch],
                    threshold=layout_threshold,
                )
                layout_batch_s = time.perf_counter() - layout_started
                layout_page_s = layout_batch_s / len(batch)
            else:
                layout_results = precomputed_layouts[
                    page_index : page_index + len(batch)
                ]
                layout_page_s = precomputed_layout_page_s
            if len(layout_results) != len(batch):
                raise RuntimeError(
                    "Layout result count mismatch: "
                    f"{len(layout_results)} != {len(batch)}"
                )
            page_indices = range(page_index, page_index + len(batch))
            if executor is None:
                prepared_batch = [
                    prepare_page(
                        pipeline=pipeline,
                        infer_doc_onnx=infer_doc_onnx,
                        decoded=decoded,
                        page_index=current_page_index,
                        layout_threshold=layout_threshold,
                        precomputed_layout=layout_result,
                        measured_layout_s=layout_page_s,
                    )
                    for decoded, layout_result, current_page_index in zip(
                        batch,
                        layout_results,
                        page_indices,
                    )
                ]
            else:
                futures = [
                    executor.submit(
                        prepare_page,
                        pipeline=pipeline,
                        infer_doc_onnx=infer_doc_onnx,
                        decoded=decoded,
                        page_index=current_page_index,
                        layout_threshold=layout_threshold,
                        precomputed_layout=layout_result,
                        measured_layout_s=layout_page_s,
                    )
                    for decoded, layout_result, current_page_index in zip(
                        batch,
                        layout_results,
                        page_indices,
                    )
                ]
                prepared_batch = [future.result() for future in futures]
            yield from prepared_batch
            page_index += len(batch)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


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

    result = {
        "input_path": str(page.image_path),
        "width": page.width,
        "height": page.height,
        "layout_results": page.layout_results,
        "recognition_results": recognition_results,
        "blocks": page.blocks,
        "timing": {"total": time.perf_counter() - page.started_at},
    }
    if page.image is not None:
        result["_page_image"] = page.image
    return result


def warmup_full_pipeline(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    infer_doc_onnx: Any,
    runner: OptimizedUniRecRunner,
    vision_atlas_runtime: UniRecVisionAtlasRuntime | None,
    image_paths: list[Path],
    output_dir: Path,
    precomputed_pages: list[PageRequest] | None = None,
    precomputed_layouts: list[dict[str, Any]] | None = None,
    precomputed_layout_page_s: float = 0.0,
) -> dict[str, Any]:
    """Run real pages through the selected end-to-end path before measurement."""
    warmup_paths = image_paths[: args.pipeline_warmup_pages]
    warmup_dir = output_dir / "_pipeline_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        "UNIREC_PIPELINE_WARMUP_BEGIN "
        f"pages={len(warmup_paths)} source_pages="
        + ",".join(path.name for path in warmup_paths),
        flush=True,
    )

    if precomputed_pages is not None:
        pages = precomputed_pages
    else:
        decoded_pages = iter_decoded_pages(
            warmup_paths,
            workers=min(args.page_decode_workers, len(warmup_paths)),
            decoder=args.page_image_decoder,
        )
        pages = list(
            iter_prepared_pages(
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
                decoded_pages=decoded_pages,
                layout_threshold=args.layout_threshold,
                layout_batch_size=args.layout_batch_size,
                page_prepare_workers=min(
                    args.page_prepare_workers,
                    len(warmup_paths),
                ),
                precomputed_layouts=precomputed_layouts,
                precomputed_layout_page_s=precomputed_layout_page_s,
            )
        )
    rejected_crops = 0
    for page in pages:
        rejected_crops += filter_page_recognition_shapes(
            page,
            runner=runner,
            target=args.recognition_shape_filter,
        )

    warmup_metrics = RunMetrics()
    crops = [crop for page in pages for crop in page.crops]
    decode_summary: dict[str, Any] | None = None
    if args.decode_scheduling == "continuous":
        def ready_source() -> Iterable[ContinuousReadyItem]:
            if args.prefill_in_layout_workers:
                for crop in crops:
                    item = build_worker_prefilled_item(crop)
                    record_prefill_metrics(warmup_metrics, item)
                    yield ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=item,
                    )
                return
            if args.text_prefill_mode == "compiled_packed_s1024":
                groups = iter_greedy_text_packs(iter(crops), runner=runner)
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
                    record_prefill_metrics(warmup_metrics, item)
                    yield ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=item,
                    )

        def complete_crop(completed_item: ContinuousCompletedItem) -> None:
            crop = completed_item.payload
            if not isinstance(crop, CropRequest):
                raise TypeError(
                    "Pipeline warmup received an unexpected crop payload: "
                    f"{type(crop)!r}"
                )
            crop.result = completed_item.result

        decode_summary = ContinuousUniRecDecoder(
            runner=runner,
            batch_size=args.decode_batch_size,
            max_length=args.max_length,
            decode_mode=args.decode_mode,
            compile_backend=args.compile_backend,
            admission_prefetch_depth=args.decode_admission_prefetch_depth,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
        ).run(ready_source(), on_complete=complete_crop)
        record_direct_arena_admission_metrics(warmup_metrics, decode_summary)
    else:
        pending = deque(crops)
        while pending:
            cohort = [
                pending.popleft()
                for _ in range(min(args.decode_batch_size, len(pending)))
            ]
            recognize_cohort(
                cohort=cohort,
                target_batch_size=args.decode_batch_size,
                runner=runner,
                vision_atlas_runtime=vision_atlas_runtime,
                args=args,
                metrics=warmup_metrics,
            )

    for page in pages:
        if not page.is_ready():
            raise RuntimeError(
                f"Pipeline warmup left page unfinished: {page.image_path.name}"
            )
        result = assemble_page(
            page=page,
            pipeline=pipeline,
            infer_doc_onnx=infer_doc_onnx,
        )
        if page.image is None:
            if any(
                bool(record.get("is_image", False))
                for record in result["recognition_results"]
            ):
                page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
                if page_image is None:
                    raise RuntimeError(
                        f"failed to reload warmup page: {result['input_path']}"
                    )
                result["_page_image"] = page_image
            else:
                result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
        pipeline.save_to_json(result, str(warmup_dir))
        pipeline.save_to_markdown(result, str(warmup_dir))
        release_page_frontend_storage(page)
    synchronize_device(runner.device)

    report = {
        "mode": "full_pipeline",
        "page_count": len(pages),
        "accepted_crop_count": len(crops),
        "rejected_crop_count": rejected_crops,
        "source_pages": [path.name for path in warmup_paths],
        "output_dir": str(warmup_dir),
        "wall_s": time.perf_counter() - started,
        "decode": decode_summary,
    }
    print(
        "UNIREC_PIPELINE_WARMUP_END "
        + json.dumps(report, ensure_ascii=False),
        flush=True,
    )
    return report


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
                "crop_size": list(crop.image_size),
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
    emit_host_memory("main_enter")
    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)
    os.environ["UNIREC_VISION_BUCKET_PRESET"] = args.vision_bucket_preset
    os.environ["UNIREC_RECOGNITION_PREPROCESS_THREADS"] = str(
        args.recognition_preprocess_threads
    )
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = (
        args.recognition_input_contract
    )
    if args.device.startswith("npu"):
        visible_devices = [
            int(value)
            for value in os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES", ""
            ).split(",")
            if value.strip()
        ]
        if 5 in visible_devices or 6 in visible_devices:
            raise RuntimeError(
                "physical NPU 5 and NPU 6 are excluded from UniRec experiments"
            )
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
    if args.page_prepare_workers < 1:
        raise ValueError("--page-prepare-workers must be >= 1")
    if args.page_prepare_workers > 1 and not args.preprocess_all_pages_first:
        raise ValueError(
            "--page-prepare-workers > 1 requires "
            "--preprocess-all-pages-first"
        )
    if args.pipeline_warmup_pages < 0:
        raise ValueError("--pipeline-warmup-pages must be >= 0")
    if args.layout_batch_size < 1:
        raise ValueError("--layout-batch-size must be >= 1")
    if args.layout_cpu_threads < 1:
        raise ValueError("--layout-cpu-threads must be >= 1")
    if args.recognition_preprocess_threads < 1:
        raise ValueError("--recognition-preprocess-threads must be >= 1")
    if args.layout_process_workers < 0:
        raise ValueError("--layout-process-workers must be >= 0")
    if args.shared_cross_kv_budget_gib <= 0:
        raise ValueError("--shared-cross-kv-budget-gib must be positive")
    if args.self_cache_length < 1 or args.cross_cache_length < 1:
        raise ValueError("decode cache lengths must be positive")
    if args.max_length > args.self_cache_length:
        raise ValueError("--max-length cannot exceed --self-cache-length")
    if args.decode_lm_head_rows < 0:
        raise ValueError("--decode-lm-head-rows must be non-negative")
    if args.decode_admission_prefetch_depth < 0:
        raise ValueError("--decode-admission-prefetch-depth must be non-negative")
    if args.decode_live_arena_warmup_passes < 0:
        raise ValueError("decode live-arena warmup passes must be non-negative")
    if args.vision_page_lookahead < 1:
        raise ValueError("--vision-page-lookahead must be >= 1")
    if args.layout_batch_size > args.vision_page_lookahead:
        raise ValueError(
            "--layout-batch-size cannot exceed --vision-page-lookahead"
        )
    if args.layout_process_workers:
        if args.layout_backend != "transformers_npu":
            raise ValueError(
                "--layout-process-workers requires --layout-backend transformers_npu"
            )
        if args.layout_batch_size != 1 and not args.prefill_in_layout_workers:
            raise ValueError(
                "layout-only process workers require --layout-batch-size 1"
            )
        if (
            not args.preprocess_all_pages_first
            and not args.prefill_in_layout_workers
        ):
            raise ValueError(
                "--layout-process-workers requires --preprocess-all-pages-first "
                "unless --prefill-in-layout-workers enables streaming"
            )
    if args.prefill_in_layout_workers:
        if not args.layout_process_workers:
            raise ValueError(
                "--prefill-in-layout-workers requires --layout-process-workers"
            )
        if args.preprocess_all_pages_first:
            raise ValueError(
                "--prefill-in-layout-workers is a streaming path; remove "
                "--preprocess-all-pages-first"
            )
        if args.decode_scheduling != "continuous":
            raise ValueError(
                "--prefill-in-layout-workers requires --decode-scheduling continuous"
            )
        if args.vision_prefill_mode not in {
            "compiled_atlas_stage2",
            "compiled_full_buckets",
        }:
            raise ValueError(
                "--prefill-in-layout-workers requires "
                "--vision-prefill-mode compiled_atlas_stage2 or "
                "compiled_full_buckets"
            )
        if args.text_prefill_mode != "compiled_packed_s1024":
            raise ValueError(
                "--prefill-in-layout-workers requires "
                "--text-prefill-mode compiled_packed_s1024"
            )
        if args.decode_admission_prefetch_depth:
            raise ValueError(
                "bounded streaming currently requires direct admission; "
                "set --decode-admission-prefetch-depth 0"
            )
    elif args.worker_empty_cache_after_page:
        raise ValueError(
            "--worker-empty-cache-after-page requires "
            "--prefill-in-layout-workers"
        )
    if (
        args.vision_prefill_mode == "compiled_full_buckets"
        and not args.prefill_in_layout_workers
    ):
        raise ValueError(
            "compiled_full_buckets requires --prefill-in-layout-workers"
        )
    if args.layout_batch_size > 1 and args.layout_backend != "transformers_npu":
        raise ValueError(
            "--layout-batch-size > 1 requires --layout-backend transformers_npu"
        )
    if args.vision_spatial_execution == "compiled_static":
        if args.vision_prefill_mode != "compiled_atlas_stage2":
            raise ValueError(
                "compiled_static vision execution requires "
                "--vision-prefill-mode compiled_atlas_stage2"
            )
        if args.recognition_shape_filter is None:
            raise ValueError(
                "compiled_static vision execution requires "
                "--recognition-shape-filter WIDTHxHEIGHT"
            )
    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
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
    use_layout_processes = args.layout_process_workers > 0
    if args.prefill_in_layout_workers:
        pipeline = infer_doc_onnx.OpenDocONNX.__new__(infer_doc_onnx.OpenDocONNX)
        pipeline.use_layout_detection = False
        pipeline.use_chart_recognition = True
        pipeline.markdown_ignore_labels = [
            "number",
            "footnote",
            "header",
            "footer",
            "aside_text",
            "footer_image",
            "header_image",
            "chart",
        ]
    else:
        stock_assets = (
            args.stock_encoder,
            args.stock_decoder,
            args.stock_tokenizer_mapping,
        )
        if any(path is None for path in stock_assets):
            raise ValueError(
                "stock OpenDoc assets are required unless worker prefill owns "
                "the complete inference path"
            )
        pipeline = infer_doc_onnx.OpenDocONNX(
            layout_model_path=str(args.layout_model.expanduser().resolve()),
            unirec_encoder_path=str(args.stock_encoder.expanduser().resolve()),
            unirec_decoder_path=str(args.stock_decoder.expanduser().resolve()),
            tokenizer_mapping_path=str(
                args.stock_tokenizer_mapping.expanduser().resolve()
            ),
            use_gpu=False,
            layout_threshold=args.layout_threshold,
            use_layout_detection=use_onnx_layout,
            auto_download=False,
            max_parallel_blocks=1,
        )
    if not use_onnx_layout and not use_layout_processes:
        pipeline.layout_detector = PPDocLayoutV2NpuAdapter(
            model_path=args.layout_transformers_model,
            device=args.device,
            dtype=args.layout_dtype,
            threshold=args.layout_threshold,
            execution=args.layout_execution,
            compile_cache_dir=args.layout_compile_cache_dir,
            batch_size=args.layout_batch_size,
        )
        pipeline.use_layout_detection = True
    elif use_layout_processes:
        pipeline.use_layout_detection = True
    emit_host_memory("main_after_pipeline_frontend")
    runner_compile_cache = args.compile_cache_dir.expanduser().resolve()
    decode_model_optimizations = None
    if args.prefill_in_layout_workers:
        from continuous_unirec import production_decode_cache_parent
        from decode_model_optimizations import (
            apply_decode_model_optimizations,
            decode_cache_variant_root,
        )

        runner_compile_cache = decode_cache_variant_root(
            production_decode_cache_parent(runner_compile_cache),
            weight_format=args.decode_weight_format,
            lm_head_rows=args.decode_lm_head_rows,
        )
    runner = OptimizedUniRecRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=(
            runner_compile_cache
            if (
                args.decode_mode.startswith("compiled")
                or args.text_prefill_mode
                in {"compiled_s512", "compiled_packed_s1024"}
                or args.vision_prefill_mode
                in {"compiled_atlas_stage2", "compiled_full_buckets"}
            )
            else None
        ),
    )
    emit_host_memory(
        "main_after_unirec_runner",
        modules={"unirec": runner.model},
    )
    if args.prefill_in_layout_workers:
        decode_model_optimizations = apply_decode_model_optimizations(
            runner,
            weight_format=args.decode_weight_format,
            lm_head_rows=args.decode_lm_head_rows,
        )
        emit_host_memory(
            "main_after_decode_weight_optimizations",
            modules={"unirec": runner.model},
            extra=decode_model_optimizations,
        )
    static_cross_cache_len = args.cross_cache_length
    if static_cross_cache_len > 0:
        processor_shape = tuple(int(value) for value in runner.processor.max_side)
        runner._static_cross_cache_len_by_processor_max_side[
            processor_shape
        ] = static_cross_cache_len
    if (
        args.vision_prefill_mode
        in {"compiled_atlas_stage2", "compiled_full_buckets"}
        and args.text_prefill_mode != "compiled_packed_s1024"
    ):
        raise ValueError(
            "compiled vision prefill currently requires "
            "--text-prefill-mode compiled_packed_s1024"
        )
    if args.vision_prefill_mode == "compiled_atlas_stage2":
        if args.vision_spatial_execution == "compiled_static":
            static_width, static_height = args.recognition_shape_filter
            vision_atlas_runtime = StaticShapeUniRecVisionRuntime(
                runner,
                input_width=static_width,
                input_height=static_height,
            )
        else:
            vision_atlas_runtime = UniRecVisionAtlasRuntime(runner)
    else:
        vision_atlas_runtime = None
    layout_process_pool: DynamicLayoutProcessPool | None = None
    layout_process_warmup: dict[str, Any] | None = None
    layout_process_setup_s: float | None = None
    if use_layout_processes:
        layout_process_pool = DynamicLayoutProcessPool(
            worker_count=args.layout_process_workers,
            model_path=args.layout_transformers_model.expanduser().resolve(),
            cache_dir=args.layout_compile_cache_dir.expanduser().resolve(),
            threshold=args.layout_threshold,
            execution=args.layout_execution,
            warmup_paths=image_paths[: args.layout_process_workers],
            layout_dtype=args.layout_dtype,
            layout_reading_order_dtype=args.layout_reading_order_dtype,
            layout_weight_format=args.layout_weight_format,
            layout_depthwise_rewrite=args.layout_depthwise_rewrite,
            layout_preformat_frozen_bn_buffers=(
                args.layout_preformat_frozen_bn_buffers
            ),
            layout_batch_size=args.layout_batch_size,
            layout_cpu_threads=args.layout_cpu_threads,
            openocr_root=openocr_root,
            prepare_pages=True,
            use_chart_recognition=pipeline.use_chart_recognition,
            prefill_recognition=args.prefill_in_layout_workers,
            recognition_model_path=model_path,
            recognition_dtype=args.dtype,
            recognition_cache_dir=args.compile_cache_dir.expanduser().resolve(),
            recognition_full_vision_buckets=(
                args.vision_prefill_mode == "compiled_full_buckets"
            ),
            recognition_vision_focal_depthwise_rewrite=(
                args.vision_focal_depthwise_rewrite
            ),
            recognition_vision_weight_format=args.vision_weight_format,
            recognition_page_lookahead=args.vision_page_lookahead,
            empty_cache_after_page=args.worker_empty_cache_after_page,
            profile_prefill_device_stages=args.prefill_device_timing,
            retain_shared_images=not args.prefill_in_layout_workers,
            shared_payload_budget_bytes=(
                int(args.shared_cross_kv_budget_gib * (1024**3))
                if args.prefill_in_layout_workers
                else 0
            ),
        )
        atexit.register(layout_process_pool.close)
        layout_process_setup_s = layout_process_pool.setup_wall_s
        emit_host_memory("main_after_prefill_worker_ready")
    if args.pipeline_warmup_pages:
        warmup_layouts = None
        warmup_pages = None
        warmup_layout_page_s = 0.0
        if layout_process_pool is not None:
            warmup_payloads, layout_process_warmup = layout_process_pool.map(
                image_paths[: args.pipeline_warmup_pages],
                label="pipeline_warmup",
            )
            warmup_layout_page_s = (
                float(layout_process_warmup["wall_s"]) / len(warmup_payloads)
            )
            warmup_pages = [
                page_request_from_process_payload(
                    payload,
                    measured_layout_s=warmup_layout_page_s,
                    shared_byte_budget=layout_process_pool.shared_byte_budget,
                )
                for payload in warmup_payloads
            ]
            del warmup_payloads
        try:
            graph_warmup = warmup_full_pipeline(
                args=args,
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
                runner=runner,
                vision_atlas_runtime=vision_atlas_runtime,
                image_paths=image_paths,
                output_dir=output_dir,
                precomputed_pages=warmup_pages,
                precomputed_layouts=warmup_layouts,
                precomputed_layout_page_s=warmup_layout_page_s,
            )
        except BaseException:
            if layout_process_pool is not None:
                layout_process_pool.close()
            raise
        runner.reset_packed_text_prefill_stats()
        if vision_atlas_runtime is not None:
            vision_atlas_runtime.reset_stats()
        if not use_onnx_layout and not use_layout_processes:
            pipeline.layout_detector.reset_timing()
    else:
        graph_warmup = warmup_configured_graphs(
            args=args,
            runner=runner,
            vision_atlas_runtime=vision_atlas_runtime,
        )
        if not use_onnx_layout and not use_layout_processes:
            layout_warmup_page = decode_page_bgr(
                image_paths[0],
                decoder=args.page_image_decoder,
            )
            pipeline.layout_detector.warmup_graph(layout_warmup_page.image)
    main_tbe_deinit_report = deinitialize_after_warmup("main_setup_complete")
    emit_host_memory(
        "main_after_tbe_deinit",
        modules={"unirec": runner.model},
        extra={"tbe_deinit": main_tbe_deinit_report},
    )
    setup_s = time.perf_counter() - setup_started
    emit_host_memory(
        "main_setup_end",
        modules={"unirec": runner.model},
    )
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
    precomputed_layouts: list[dict[str, Any]] | None = None
    precomputed_pages: list[PageRequest] | None = None
    precomputed_layout_page_s = 0.0
    layout_process_summary: dict[str, Any] | None = None
    layout_process_shutdown_timing: dict[str, float] = {}
    shared_cross_kv_budget_summary: dict[str, Any] | None = None
    if layout_process_pool is not None and not args.prefill_in_layout_workers:
        try:
            page_payloads, layout_process_summary = layout_process_pool.map(
                image_paths,
                label="measured",
            )
        except BaseException:
            layout_process_pool.close()
            layout_process_pool = None
            raise
        precomputed_layout_page_s = (
            float(layout_process_summary["wall_s"]) / len(page_payloads)
        )
        precomputed_pages = [
            page_request_from_process_payload(
                payload,
                measured_layout_s=precomputed_layout_page_s,
                shared_byte_budget=layout_process_pool.shared_byte_budget,
            )
            for payload in page_payloads
        ]
        del page_payloads
    written_pages = 0
    continuous_decode: dict[str, Any] | None = None

    def write_page(result: dict[str, Any]) -> tuple[float, float]:
        if any(
            bool(record.get("is_image", False))
            for record in result["recognition_results"]
        ):
            page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
            if page_image is None:
                raise RuntimeError(
                    f"failed to reload output page: {result['input_path']}"
                )
            result["_page_image"] = page_image
        else:
            result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
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
        release_page_frontend_storage(page)

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
        page_admission_trackers: dict[int, PageCrossKvAdmissionTracker] = {}

        def crop_source():
            nonlocal layout_process_summary
            if args.prefill_in_layout_workers:
                if layout_process_pool is None:
                    raise RuntimeError("worker-prefill stream has no process pool")

                def iter_worker_pages() -> Iterable[PageRequest]:
                    nonlocal layout_process_summary
                    for payload in layout_process_pool.iter_map(
                        image_paths,
                        label="measured_worker_prefill",
                    ):
                        yield page_request_from_process_payload(
                            payload,
                            measured_layout_s=float(
                                payload["frontend_timing_s"]["layout_s"]
                            ),
                            shared_byte_budget=(
                                layout_process_pool.shared_byte_budget
                            ),
                        )
                    layout_process_summary = (
                        layout_process_pool.last_stream_summary
                    )

                prepared_pages: Iterable[PageRequest] = iter_worker_pages()
            elif precomputed_pages is not None:
                prepared_pages: Iterable[PageRequest] = precomputed_pages
            else:
                decoded_pages = iter_decoded_pages(
                    image_paths,
                    workers=args.page_decode_workers,
                    decoder=args.page_image_decoder,
                )
                prepared_pages = iter_prepared_pages(
                    pipeline=pipeline,
                    infer_doc_onnx=infer_doc_onnx,
                    decoded_pages=decoded_pages,
                    layout_threshold=args.layout_threshold,
                    layout_batch_size=args.layout_batch_size,
                    page_prepare_workers=args.page_prepare_workers,
                    precomputed_layouts=precomputed_layouts,
                    precomputed_layout_page_s=precomputed_layout_page_s,
                )
            if args.preprocess_all_pages_first:
                print(
                    "UNIREC_PAGE_FRONTEND_DRAIN_BEGIN "
                    f"pages={len(image_paths)}",
                    flush=True,
                )
                prepared_pages = list(prepared_pages)
                print(
                    "UNIREC_PAGE_FRONTEND_DRAIN_END "
                    f"pages={len(prepared_pages)}",
                    flush=True,
                )
            for page in prepared_pages:
                page_index = page.page_index
                rejected = filter_page_recognition_shapes(
                    page,
                    runner=runner,
                    target=args.recognition_shape_filter,
                )
                metrics.rejected_crops += rejected
                metrics.layout_s += page.layout_s
                metrics.page_prepare_total_s += page.prepare_page_total_s
                accumulate_stage_seconds(
                    metrics.frontend_timing_s,
                    page.frontend_timing_s,
                )
                pending_pages.append(page)
                if args.prefill_in_layout_workers and page.crops:
                    page_admission_trackers[page.page_index] = (
                        PageCrossKvAdmissionTracker(page)
                    )
                print(
                    f"OPENDOC_CONTINUOUS_PAGE_READY "
                    f"index={page_index + 1}/{len(image_paths)} "
                    f"image={page.image_path.name} crops={len(page.crops)} "
                    f"rejected_crops={rejected}",
                    flush=True,
                )
                flush_ready_pages()
                for crop in page.crops:
                    yield crop

        def ready_source():
            crops = crop_source()
            if args.prefill_in_layout_workers:
                for crop in crops:
                    item = build_worker_prefilled_item(crop)
                    record_prefill_metrics(metrics, item)
                    tracker = page_admission_trackers[crop.page_index]

                    def release_admitted_crop(
                        *,
                        crop: CropRequest = crop,
                        item: ContinuousWorkerPrefilledItem = item,
                        tracker: PageCrossKvAdmissionTracker = tracker,
                    ) -> None:
                        tracker.release_crop(crop, item)
                        if tracker.admitted_crops == len(tracker.page.crops):
                            page_admission_trackers.pop(
                                tracker.page.page_index,
                                None,
                            )

                    yield ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=item,
                        on_admitted=release_admitted_crop,
                    )
                return
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
                    "crop_size": list(crop.image_size),
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
            admission_prefetch_depth=args.decode_admission_prefetch_depth,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
        )
        continuous_decode = continuous_runner.run(
            ready_source(),
            on_complete=complete_crop,
            graph_warmup_passes=args.decode_live_arena_warmup_passes,
        )
        if page_admission_trackers:
            raise RuntimeError(
                "decode ended with unreleased page arenas: "
                + ",".join(
                    str(index) for index in sorted(page_admission_trackers)
                )
            )
        record_direct_arena_admission_metrics(metrics, continuous_decode)
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
        if precomputed_pages is not None:
            prepared_pages = precomputed_pages
        else:
            decoded_pages = iter_decoded_pages(
                image_paths,
                workers=args.page_decode_workers,
                decoder=args.page_image_decoder,
            )
            prepared_pages = iter_prepared_pages(
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
                decoded_pages=decoded_pages,
                layout_threshold=args.layout_threshold,
                layout_batch_size=args.layout_batch_size,
                page_prepare_workers=args.page_prepare_workers,
                precomputed_layouts=precomputed_layouts,
                precomputed_layout_page_s=precomputed_layout_page_s,
            )
        if args.preprocess_all_pages_first:
            print(
                "UNIREC_PAGE_FRONTEND_DRAIN_BEGIN "
                f"pages={len(image_paths)}",
                flush=True,
            )
            prepared_pages = list(prepared_pages)
            print(
                "UNIREC_PAGE_FRONTEND_DRAIN_END "
                f"pages={len(prepared_pages)}",
                flush=True,
            )
        for page in prepared_pages:
            page_index = page.page_index
            rejected = filter_page_recognition_shapes(
                page,
                runner=runner,
                target=args.recognition_shape_filter,
            )
            metrics.rejected_crops += rejected
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
                f"image={page.image_path.name} crops={len(page.crops)} "
                f"rejected_crops={rejected} queued={len(pending_crops)}",
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
    if layout_process_pool is not None:
        if layout_process_pool.shared_byte_budget is not None:
            shared_cross_kv_budget_summary = (
                layout_process_pool.shared_byte_budget.snapshot()
            )
        shutdown_started = time.perf_counter()
        layout_process_pool.close()
        layout_process_shutdown_timing["wall_s"] = (
            time.perf_counter() - shutdown_started
        )
        layout_process_pool = None
    accepted_crop_count = len(metrics.crop_records)
    vision_spatial_device_s = sum(
        metrics.prefill_device_stage_s.get(name, 0.0)
        for name in (
            "vision_crop_prefix_stages_0_1",
            "vision_crop_suffix_stage3_projection",
        )
    )
    trace_path = output_dir / "recognition_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in sorted(
            metrics.crop_records,
            key=lambda item: (item["page_index"], item["crop_index"]),
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "host_runtime_process_controls": {
            "te_parallel_compiler": os.environ.get("TE_PARALLEL_COMPILER"),
            "cann_knowledge_bank_process_num": os.environ.get(
                "CANN_KNOWLEDGE_BANK_PROCESS_NUM"
            ),
            "deinit_tbe_after_warmup": os.environ.get(
                "UNIREC_DEINIT_TBE_AFTER_WARMUP"
            ),
        },
        "openocr_root": str(openocr_root),
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "decode_mode": args.decode_mode,
        "decode_scheduling": args.decode_scheduling,
        "decode_batch_size": args.decode_batch_size,
        "self_cache_length": args.self_cache_length,
        "cross_cache_length": args.cross_cache_length,
        "decode_weight_format": args.decode_weight_format,
        "decode_lm_head_rows": (
            args.decode_lm_head_rows
            or int(runner.config.vocab_size)
        ),
        "decode_model_optimizations": decode_model_optimizations,
        "decode_admission_prefetch_depth": (
            args.decode_admission_prefetch_depth
        ),
        "decode_live_arena_warmup_passes": (
            args.decode_live_arena_warmup_passes
        ),
        "text_prefill_mode": args.text_prefill_mode,
        "vision_prefill_mode": args.vision_prefill_mode,
        "vision_page_lookahead": args.vision_page_lookahead,
        "vision_spatial_execution": args.vision_spatial_execution,
        "recognition_shape_filter": args.recognition_shape_filter,
        "max_length": args.max_length,
        "layout_backend": args.layout_backend,
        "layout_dtype": args.layout_dtype if not use_onnx_layout else None,
        "layout_execution": args.layout_execution if not use_onnx_layout else None,
        "layout_reading_order_dtype": (
            args.layout_reading_order_dtype or args.layout_dtype
            if not use_onnx_layout
            else None
        ),
        "layout_weight_format": (
            args.layout_weight_format if not use_onnx_layout else None
        ),
        "layout_depthwise_rewrite": (
            args.layout_depthwise_rewrite if not use_onnx_layout else None
        ),
        "layout_preformat_frozen_bn_buffers": (
            args.layout_preformat_frozen_bn_buffers
            if not use_onnx_layout
            else None
        ),
        "layout_cpu_threads": args.layout_cpu_threads,
        "layout_batch_size": args.layout_batch_size,
        "layout_process_workers": args.layout_process_workers,
        "prefill_in_layout_workers": args.prefill_in_layout_workers,
        "vision_bucket_preset": args.vision_bucket_preset,
        "vision_focal_depthwise_rewrite": (
            args.vision_focal_depthwise_rewrite
        ),
        "vision_weight_format": args.vision_weight_format,
        "recognition_preprocess_threads": args.recognition_preprocess_threads,
        "recognition_input_contract": args.recognition_input_contract,
        "shared_cross_kv_budget_gib": args.shared_cross_kv_budget_gib,
        "shared_cross_kv_budget": shared_cross_kv_budget_summary,
        "worker_empty_cache_after_page": args.worker_empty_cache_after_page,
        "layout_process_setup_s": layout_process_setup_s,
        "layout_process_warmup": layout_process_warmup,
        "layout_process": layout_process_summary,
        "layout_process_shutdown_s": layout_process_shutdown_timing.get("wall_s"),
        "layout_process_shutdown_join_wait_s": 0.0,
        "layout_graph_warmup": (
            pipeline.layout_detector.graph_warmup
            if not use_onnx_layout and not use_layout_processes
            else None
        ),
        "page_decode_workers": args.page_decode_workers,
        "page_image_decoder": args.page_image_decoder,
        "preprocess_all_pages_first": args.preprocess_all_pages_first,
        "page_prepare_workers": args.page_prepare_workers,
        "setup_s": setup_s,
        "graph_warmup": graph_warmup,
        "pipeline_wall_s": pipeline_wall_s,
        "pages_per_s": len(image_paths) / pipeline_wall_s,
        "page_count": len(image_paths),
        "crop_count": accepted_crop_count,
        "rejected_crop_count": metrics.rejected_crops,
        "cohort_count": len(metrics.cohort_records),
        "layout_s": metrics.layout_s,
        "page_prepare_total_s": metrics.page_prepare_total_s,
        "page_frontend_other_s": metrics.page_prepare_total_s - metrics.layout_s,
        "page_frontend_stage_s": metrics.frontend_timing_s,
        "prepare_s": metrics.prepare_s,
        "prefill_s": metrics.prefill_s,
        "prefill_device_stage_s": metrics.prefill_device_stage_s,
        "vision_spatial_device_s": vision_spatial_device_s,
        "vision_spatial_ms_per_accepted_crop": (
            vision_spatial_device_s * 1000.0 / accepted_crop_count
            if accepted_crop_count > 0
            else None
        ),
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
            {
                "execution": "worker_compiled_full_buckets",
                "preset": args.vision_bucket_preset,
                "batching": (
                    layout_process_summary.get("vision_batching")
                    if layout_process_summary is not None
                    else None
                ),
            }
            if args.prefill_in_layout_workers
            else (
                vision_atlas_runtime.summary()
                if vision_atlas_runtime is not None
                else {"execution": "eager_per_crop"}
            )
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
