"""Persistent dynamic PP-DocLayoutV2 process pool.

The coordinator sends only page indices and file paths.  Every spawned process
owns one complete layout model/runtime.  Workers draw from one shared queue, so
no worker is tied to a slow static shard. Full-vision workers can combine their
bounded page lookahead into real static layout batches.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image

from layout_page_input import (
    decode_page_rgb as _decode_rgb,
    materialize_layout_bgr,
    materialize_layout_rgb,
)
from opendoc_layout_npu import (
    DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
    DEFAULT_LAYOUT_WEIGHT_FORMAT,
    PPDocLayoutV2NpuAdapter,
)


FULL_VISION_PAGE_COLLECT_TIMEOUT_S = 0.02
RECOGNITION_CPU_DETAIL_TIMING_FIELDS = (
    "recognition_capacity_filter_s",
    "recognition_pil_fromarray_s",
    "recognition_processor_image_convert_rgb_s",
    "recognition_processor_target_size_s",
    "recognition_processor_resize_s",
    "recognition_processor_pil_to_uint8_hwc_s",
    "recognition_processor_pil_to_float_array_s",
    "recognition_processor_rescale_s",
    "recognition_processor_normalize_s",
    "recognition_processor_transpose_s",
    "recognition_processor_tensor_view_s",
    "recognition_tensor_numpy_view_s",
    "recognition_contiguous_chw_copy_s",
)


def _current_cpu() -> int | None:
    """Return the calling native thread's current CPU on Linux."""
    try:
        stat = Path(f"/proc/self/task/{threading.get_native_id()}/stat").read_text()
        # Fields 1 and 2 are pid and a parenthesized comm. The first item after
        # the closing parenthesis is field 3, so field 39 (processor) is 36.
        return int(stat.rsplit(")", 1)[1].split()[36])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _cpu_affinity() -> list[int]:
    try:
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


def _resize_recognition_compact_hwc_with_timing(
    image: Image.Image,
    *,
    processor: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resize one RGB crop but leave normalization/layout work for the NPU."""
    timing_s: dict[str, float] = {}
    started = time.perf_counter()
    if image.mode != "RGB":
        raise ValueError(
            "compact UniRec input requires an RGB crop, "
            f"got Pillow mode {image.mode!r}"
        )
    timing_s["recognition_processor_image_convert_rgb_s"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    target_size = processor.get_processed_size(*image.size)
    timing_s["recognition_processor_target_size_s"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    resized = image.resize(target_size, resample=processor.resample)
    timing_s["recognition_processor_resize_s"] = time.perf_counter() - started

    started = time.perf_counter()
    pixels = np.asarray(resized)
    timing_s["recognition_processor_pil_to_uint8_hwc_s"] = (
        time.perf_counter() - started
    )
    expected = (target_size[1], target_size[0], 3)
    if pixels.dtype != np.uint8 or tuple(pixels.shape) != expected:
        raise ValueError(
            "compact UniRec resize produced an invalid array: "
            f"{pixels.dtype} {pixels.shape}, expected uint8 {expected}"
        )
    return pixels, timing_s


class SharedPageLease:
    """Own one page's shared frontend arena in the coordinator process."""

    def __init__(self, name: str) -> None:
        self.storage = SharedMemory(name=name)
        # Remove the public POSIX name immediately.  The mapping remains valid
        # until this coordinator closes its descriptor, but a later crash
        # cannot leak a named shared-memory object.
        self.storage.unlink()
        self.closed = False

    def array(self, descriptor: dict[str, Any]) -> np.ndarray:
        if self.closed:
            raise RuntimeError("shared page lease is already closed")
        return np.ndarray(
            tuple(int(value) for value in descriptor["shape"]),
            dtype=np.dtype(descriptor.get("dtype", "uint8")),
            buffer=self.storage.buf,
            offset=int(descriptor["offset"]),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.storage.close()


def _pack_frontend_payload_shared(
    result: dict[str, Any],
    *,
    retain_images: bool = True,
) -> tuple[dict[str, Any], float, int]:
    """Move selected frontend arrays into one aligned shared arena."""
    entries: list[tuple[str, dict[str, Any] | None, np.ndarray]] = []
    if retain_images:
        page_bgr = result.get("image_bgr")
        if page_bgr is None:
            page_rgb = result.get("page_rgb")
            if page_rgb is None:
                raise RuntimeError("retained frontend payload has no page image")
            # Output assembly still consumes the historical BGR page contract.
            # The optimized no-retain inference path never enters this branch.
            page_bgr = materialize_layout_bgr(page_rgb)
        entries.append(("image_bgr_descriptor", None, page_bgr))
    for crop in result["crops"]:
        height, width = crop["image_rgb"].shape[:2]
        crop["source_image_size"] = [int(width), int(height)]
        if retain_images:
            entries.append(("image_rgb_descriptor", crop, crop["image_rgb"]))
        if "worker_cross_kv" in crop:
            entries.append(
                ("worker_cross_kv_descriptor", crop, crop["worker_cross_kv"])
            )
        else:
            entries.append(
                (
                    "processed_pixel_values_descriptor",
                    crop,
                    crop["processed_pixel_values"],
                )
            )
    arrays = [entry[2] for entry in entries]
    if not arrays:
        if result["crops"]:
            raise RuntimeError(
                "shared frontend payload has crops but no retained arrays; "
                "image-free crops require worker cross-K/V or processed pixels"
            )
        started = time.perf_counter()
        result["shared_memory"] = None
        result["image_bgr"] = None
        result["page_rgb"] = None
        return result, time.perf_counter() - started, 0
    offsets = []
    total_bytes = 0
    for array in arrays:
        total_bytes = (total_bytes + 63) // 64 * 64
        offsets.append(total_bytes)
        total_bytes += int(array.nbytes)
    if total_bytes <= 0:
        raise RuntimeError("full frontend payload has no image bytes")
    started = time.perf_counter()
    storage = SharedMemory(create=True, size=total_bytes)
    ownership_transferred = False
    try:
        descriptors = []
        for array, offset in zip(arrays, offsets):
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            descriptor = {
                "offset": offset,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "nbytes": int(array.nbytes),
            }
            target = np.ndarray(
                array.shape,
                dtype=array.dtype,
                buffer=storage.buf,
                offset=offset,
            )
            np.copyto(target, array, casting="no")
            descriptors.append(descriptor)
        result["shared_memory"] = {
            "name": storage.name,
            "nbytes": total_bytes,
        }
        for (descriptor_name, owner, _array), descriptor in zip(
            entries, descriptors
        ):
            target = result if owner is None else owner
            target[descriptor_name] = descriptor
        result["image_bgr"] = None
        result["page_rgb"] = None
        for crop in result["crops"]:
            crop["image_rgb"] = None
            crop.pop("processed_pixel_values", None)
            crop.pop("worker_cross_kv", None)
        ownership_transferred = True
        pack_s = time.perf_counter() - started
        return result, pack_s, total_bytes
    finally:
        storage.close()
        if ownership_transferred:
            # Python <=3.12 has no SharedMemory(track=False).  Ownership moves
            # to the coordinator, which attaches and unlinks the name.  Stop
            # this worker's resource tracker from unlinking it at process exit.
            tracked_name = storage._name  # noqa: SLF001
            resource_tracker.unregister(tracked_name, "shared_memory")
        else:
            storage.unlink()


def _base_label(label: str) -> str:
    parts = label.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else label


def _crop_margin_rgb(image: np.ndarray) -> np.ndarray:
    """OpenOCR crop_margin with RGB-aware grayscale and identical geometry."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    maximum = gray.max()
    minimum = gray.min()
    if maximum == minimum:
        return image
    data = ((gray - minimum) / (maximum - minimum) * 255).astype(np.uint8)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coordinates = cv2.findNonZero(binary)
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    return image[y : y + height, x : x + width]


def _prepare_frontend_payload(
    *,
    page_index: int,
    path: Path,
    rgb: np.ndarray,
    layout_result: dict[str, Any],
    use_chart_recognition: bool,
    tokenize_figure_of_table: Any,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build the complete CPU page frontend result inside its owner process."""
    image_labels = ["image", "header_image", "footer_image", "seal"]
    if not use_chart_recognition:
        image_labels.append("chart")

    started = time.perf_counter()
    blocks: list[dict[str, Any]] = []
    block_images: list[np.ndarray | None] = []
    for box in layout_result["boxes"]:
        x1, y1, x2, y2 = map(int, box["coordinate"])
        cropped = rgb[y1:y2, x1:x2]
        block_image = None if cropped.size == 0 else cropped
        block_images.append(block_image)
        blocks.append(
            {
                # Downstream assembly only needs the empty/non-empty contract.
                # The full page remains available for output image extraction.
                "img": None if block_image is None else True,
                "box": box["coordinate"],
                "label": box["label"],
                "score": box.get("score", 1.0),
            }
        )
    crop_views_s = time.perf_counter() - started

    started = time.perf_counter()
    imgs_in_doc = []
    for block, block_image in zip(blocks, block_images):
        label = block["label"]
        if _base_label(label) in image_labels and block_image is not None:
            x1, y1, x2, y2 = map(int, block["box"])
            imgs_in_doc.append(
                {
                    "coordinate": block["box"],
                    "path": (
                        f"imgs/img_in_{_base_label(label)}_box_"
                        f"{x1}_{y1}_{x2}_{y2}.jpg"
                    ),
                }
            )
    image_index_s = time.perf_counter() - started

    started = time.perf_counter()
    crops: list[dict[str, Any]] = []
    vlm_block_ids: list[int] = []
    drop_figures_set: set[str] = set()
    for block_index, (block, block_image) in enumerate(zip(blocks, block_images)):
        label = block["label"]
        if _base_label(label) in image_labels or block_image is None:
            continue
        figure_token_map: dict[str, Any] = {}
        drop_figures: list[str] = []
        if "table" in label:
            block_image, figure_token_map, drop_figures = tokenize_figure_of_table(
                block_image,
                block["box"],
                imgs_in_doc,
            )
        elif "formula" in label and label != "formula_number":
            block_image = _crop_margin_rgb(block_image)
        crop_rgb = np.ascontiguousarray(block_image)
        crops.append(
            {
                "crop_index": len(crops),
                "image_rgb": crop_rgb,
                "label": label,
                "figure_token_map": figure_token_map,
            }
        )
        vlm_block_ids.append(block_index)
        drop_figures_set.update(drop_figures)
    crop_build_s = time.perf_counter() - started
    height, width = rgb.shape[:2]
    return (
        {
            "page_index": page_index,
            "image_path": str(path),
            "image_bgr": None,
            "page_rgb": rgb,
            "width": width,
            "height": height,
            "layout_results": layout_result,
            "blocks": blocks,
            "vlm_block_ids": vlm_block_ids,
            "crops": crops,
            "drop_figures_set": sorted(drop_figures_set),
        },
        {
            "layout_crop_views_s": crop_views_s,
            "document_image_index_s": image_index_s,
            "recognition_crop_build_s": crop_build_s,
        },
    )


def _iter_worker_prefill_groups(
    crops: list[dict[str, Any]],
    *,
    runner: Any,
    bucket: int,
) -> Any:
    """Page-local FIFO text packs, matching the coordinator's pack rule."""
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for crop in crops:
        height, width = crop["image_rgb"].shape[:2]
        tokens = int(
            runner.processor.estimate_encoder_token_count_for_image_size(
                width, height
            )
        )
        if tokens > bucket:
            if current:
                yield True, current
                current = []
                current_tokens = 0
            yield False, [crop]
            continue
        if current and current_tokens + tokens > bucket:
            yield True, current
            current = []
            current_tokens = 0
        current.append(crop)
        current_tokens += tokens
    if current:
        yield True, current


def _copy_cross_kv_group_to_host(
    items: list[Any],
) -> tuple[list[tuple[np.ndarray, int, float]], float]:
    """Copy exact real-length cross-KV, optionally once per packed cohort."""
    mode = os.environ.get("UNIREC_CROSS_KV_D2H_MODE", "per_crop")
    if mode not in {"per_crop", "packed_cohort"}:
        raise RuntimeError(f"invalid UNIREC_CROSS_KV_D2H_MODE: {mode!r}")
    lengths = [
        int(item.kv_cache.actual_cross_attention_length or 0) for item in items
    ]
    if any(length <= 0 for length in lengths):
        raise RuntimeError("worker prefill produced an empty cross cache")

    started = time.perf_counter()
    if mode == "per_crop" or len(items) == 1:
        outputs = []
        for item, actual_length in zip(items, lengths):
            item_started = time.perf_counter()
            cache = item.kv_cache
            packed_cache = torch.stack(
                tuple(
                    tensor[:, :, :actual_length, :]
                    for tensor in (
                        *cache.cross_key_cache,
                        *cache.cross_value_cache,
                    )
                ),
                dim=0,
            ).contiguous()
            packed_host = packed_cache.cpu().numpy()
            outputs.append(
                (
                    packed_host,
                    actual_length,
                    time.perf_counter() - item_started,
                )
            )
        return outputs, time.perf_counter() - started

    flat_segments = []
    shapes = []
    numels = []
    for item, actual_length in zip(items, lengths):
        cache = item.kv_cache
        tensors = (*cache.cross_key_cache, *cache.cross_value_cache)
        slices = tuple(
            tensor[:, :, :actual_length, :] for tensor in tensors
        )
        shape = (len(slices), *slices[0].shape)
        shapes.append(shape)
        numels.append(int(np.prod(shape, dtype=np.int64)))
        flat_segments.extend(tensor.reshape(-1) for tensor in slices)
    packed_host_flat = torch.cat(tuple(flat_segments), dim=0).cpu().numpy()
    elapsed = time.perf_counter() - started
    total_numel = sum(numels)
    outputs = []
    offset = 0
    accounted_s = 0.0
    for index, (actual_length, shape, numel) in enumerate(
        zip(lengths, shapes, numels)
    ):
        packed_host = packed_host_flat[offset : offset + numel].reshape(shape)
        if index == len(numels) - 1:
            item_s = elapsed - accounted_s
        else:
            item_s = elapsed * numel / total_numel
            accounted_s += item_s
        outputs.append((packed_host, actual_length, item_s))
        offset += numel
    return outputs, elapsed


def _prefill_worker_page(
    result: dict[str, Any],
    *,
    runner: Any,
    vision_atlas_runtime: Any,
    profile_device_stages: bool = False,
) -> dict[str, float]:
    """Run page-local recognition prefill and retain only real cross K/V."""
    from text_packed_prefill import PACKED_TEXT_PREFILL_BUCKET

    started = time.perf_counter()
    cache_d2h_s = 0.0
    real_tokens = 0
    physical_tokens = 0
    pack_count = 0
    fallback_count = 0
    for use_packed_graph, group in _iter_worker_prefill_groups(
        result["crops"],
        runner=runner,
        bucket=PACKED_TEXT_PREFILL_BUCKET,
    ):
        prepared = []
        for crop in group:
            height, width = crop["image_rgb"].shape[:2]
            request_id = (
                f"page_{int(result['page_index']):06d}_"
                f"crop_{int(crop['crop_index']):04d}"
            )
            prepared.append(
                runner.prepare_preprocessed_pixels(
                    crop["processed_pixel_values"],
                    original_image_size=(width, height),
                    image_source=request_id,
                )
            )
        if use_packed_graph:
            items = vision_atlas_runtime.prefill_prepared_packed_for_cohort(
                prepared,
                profile_device_stages=profile_device_stages,
                decode_ready=False,
            )
            pack_count += 1
        else:
            items = [
                runner.prefill_prepared_for_cohort(
                    prepared[0],
                    profile_device_stages=profile_device_stages,
                    text_prefill_mode="eager",
                    decode_ready=False,
                )
            ]
            fallback_count += 1
        host_exports, group_d2h_s = _copy_cross_kv_group_to_host(items)
        cache_d2h_s += group_d2h_s
        for crop, item, (packed_host, actual_length, item_d2h_s) in zip(
            group, items, host_exports
        ):
            member_real = int(
                item.text_prefill_real_source_tokens or actual_length
            )
            member_physical = int(
                item.text_prefill_physical_source_tokens or member_real
            )
            real_tokens += member_real
            physical_tokens += member_physical
            crop["worker_cross_kv"] = packed_host
            crop["worker_prefill_metadata"] = {
                "prep": item.prep,
                "prefill_s": float(item.prefill_s),
                "cache_d2h_s": item_d2h_s,
                "prefill_device_stage_s": item.prefill_device_stage_s,
                "text_prefill_execution": item.text_prefill_execution,
                "text_prefill_real_source_tokens": member_real,
                "text_prefill_physical_source_tokens": member_physical,
                "actual_cross_attention_length": actual_length,
            }
    return {
        "recognition_prefill_worker_s": time.perf_counter() - started,
        "recognition_prefill_cache_d2h_s": cache_d2h_s,
        "recognition_prefill_real_source_tokens": float(real_tokens),
        "recognition_prefill_physical_source_tokens": float(physical_tokens),
        "recognition_prefill_pack_count": float(pack_count),
        "recognition_prefill_fallback_count": float(fallback_count),
    }


def _prefill_worker_pages_bucketed(
    results: list[dict[str, Any]],
    *,
    runner: Any,
    vision_runtime: Any,
    profile_device_stages: bool = False,
    trace_prefill_iterations: bool = False,
) -> list[dict[str, Any]]:
    """Batch full vision across pages, then keep text packs page-local."""
    from text_packed_prefill import PACKED_TEXT_PREFILL_BUCKET
    from vision_full_batch import PreprocessedVisionInput

    if not results:
        raise ValueError("bucketed worker prefill requires at least one page")
    flat_inputs = []
    crop_source_indices: dict[int, int] = {}
    for result in results:
        for crop in result["crops"]:
            source_index = len(flat_inputs)
            height, width = crop["image_rgb"].shape[:2]
            request_id = (
                f"page_{int(result['page_index']):06d}_"
                f"crop_{int(crop['crop_index']):04d}"
            )
            flat_inputs.append(
                PreprocessedVisionInput(
                    source_index=source_index,
                    pixel_values=crop["processed_pixel_values"],
                    original_image_size=(width, height),
                    image_source=request_id,
                )
            )
            crop_source_indices[id(crop)] = source_index
    if not flat_inputs:
        return [
            {
                "recognition_prefill_worker_s": 0.0,
                "recognition_prefill_cache_d2h_s": 0.0,
                "recognition_prefill_real_source_tokens": 0.0,
                "recognition_prefill_physical_source_tokens": 0.0,
                "recognition_prefill_pack_count": 0.0,
                "recognition_prefill_fallback_count": 0.0,
                "recognition_vision_bucket_rows": {},
                "recognition_vision_group_owner": index == 0,
                "recognition_vision_group_bucket_calls": {},
                "recognition_vision_group_bucket_real_rows": {},
                "recognition_vision_group_bucket_physical_rows": {},
                "recognition_vision_group_fallback_rows": 0,
            }
            for index, _result in enumerate(results)
        ]

    bucket_calls_before = dict(vision_runtime.stats["bucket_calls"])
    bucket_real_rows_before = dict(vision_runtime.stats["bucket_real_rows"])
    fallback_rows_before = int(vision_runtime.stats["fallback_rows"])
    synchronize_started = time.perf_counter()
    torch.npu.synchronize()
    vision_started = time.perf_counter()
    if trace_prefill_iterations:
        vision_runtime.drain_trace_events()
    encoded = vision_runtime.encode(flat_inputs)
    torch.npu.synchronize()
    vision_s = time.perf_counter() - vision_started
    synchronization_s = vision_started - synchronize_started
    if trace_prefill_iterations:
        results[0].setdefault("prefill_trace_events", []).extend(
            vision_runtime.drain_trace_events()
        )
        results[0]["prefill_trace_events"].append(
            {
                "event": "vision_page_group",
                "page_indices": [int(result["page_index"]) for result in results],
                "page_images": [Path(result["image_path"]).name for result in results],
                "crop_count": len(flat_inputs),
                "initial_sync_s": synchronization_s,
                "wall_s": vision_s,
            }
        )
    group_bucket_calls = {
        key: int(value) - int(bucket_calls_before.get(key, 0))
        for key, value in vision_runtime.stats["bucket_calls"].items()
        if int(value) - int(bucket_calls_before.get(key, 0))
    }
    group_bucket_real_rows = {
        key: int(value) - int(bucket_real_rows_before.get(key, 0))
        for key, value in vision_runtime.stats["bucket_real_rows"].items()
        if int(value) - int(bucket_real_rows_before.get(key, 0))
    }
    bucket_batch_sizes = {
        spec.key: int(spec.batch_size) for spec in vision_runtime.specs
    }
    group_bucket_physical_rows = {
        key: calls * bucket_batch_sizes[key]
        for key, calls in group_bucket_calls.items()
    }
    group_fallback_rows = (
        int(vision_runtime.stats["fallback_rows"]) - fallback_rows_before
    )
    encoded_by_source = {item.source_index: item for item in encoded}
    vision_bucket_by_source = {
        item.source_index: item.bucket_key for item in encoded
    }
    encoded.clear()
    vision_share_per_crop = vision_s / len(flat_inputs)
    page_timings = []
    for result_index, result in enumerate(results):
        page_started = time.perf_counter()
        cache_d2h_s = 0.0
        real_tokens = 0
        physical_tokens = 0
        pack_count = 0
        fallback_count = 0
        packed_prefill_rejected_crop_ids: set[int] = set()
        bucket_rows: dict[str, int] = {}
        for crop in result["crops"]:
            item = encoded_by_source[crop_source_indices[id(crop)]]
            bucket_name = item.bucket_key or "fallback_eager"
            bucket_rows[bucket_name] = bucket_rows.get(bucket_name, 0) + 1
        for use_packed_graph, group in _iter_worker_prefill_groups(
            result["crops"],
            runner=runner,
            bucket=PACKED_TEXT_PREFILL_BUCKET,
        ):
            if not use_packed_graph:
                for crop in group:
                    packed_prefill_rejected_crop_ids.add(id(crop))
                    encoded_by_source.pop(crop_source_indices[id(crop)])
                fallback_count += len(group)
                continue
            encoded_group = []
            for crop in group:
                source_index = crop_source_indices[id(crop)]
                vision_item = encoded_by_source.pop(source_index)
                encoded_group.append(
                    (vision_item.hidden_states, vision_item.prep)
                )
            items = runner.prefill_encoder_hidden_states_packed_for_cohort(
                encoded_group,
                profile_device_stages=profile_device_stages,
                decode_ready=False,
            )
            pack_count += 1
            if trace_prefill_iterations:
                device_stage_names = sorted(
                    {
                        name
                        for item in items
                        for name in (item.prefill_device_stage_s or {})
                    }
                )
                result.setdefault("prefill_trace_events", []).append(
                    {
                        "event": "text_prefill_pack",
                        "page_index": int(result["page_index"]),
                        "page_image": Path(result["image_path"]).name,
                        "member_count": len(group),
                        "members": [
                            {
                                "crop_index": int(crop["crop_index"]),
                                "label": str(crop["label"]),
                                "source_image_size": [
                                    int(crop["image_rgb"].shape[1]),
                                    int(crop["image_rgb"].shape[0]),
                                ],
                                "processed_tensor_shape": list(
                                    crop["processed_pixel_values"].shape
                                ),
                                "source_tokens": int(
                                    item.text_prefill_real_source_tokens
                                    or item.kv_cache.actual_cross_attention_length
                                    or 0
                                ),
                            }
                            for crop, item in zip(group, items)
                        ],
                        "real_source_tokens": sum(
                            int(item.text_prefill_real_source_tokens or 0)
                            for item in items
                        ),
                        "physical_source_tokens": sum(
                            int(item.text_prefill_physical_source_tokens or 0)
                            for item in items
                        ),
                        "source_bucket": PACKED_TEXT_PREFILL_BUCKET,
                        "slot_efficiency": (
                            sum(
                                int(item.text_prefill_real_source_tokens or 0)
                                for item in items
                            )
                            / PACKED_TEXT_PREFILL_BUCKET
                        ),
                        "wall_s": sum(float(item.prefill_s) for item in items),
                        "device_stage_s": {
                            name: sum(
                                float(
                                    (item.prefill_device_stage_s or {}).get(
                                        name, 0.0
                                    )
                                )
                                for item in items
                            )
                            for name in device_stage_names
                        },
                    }
                )
            host_exports, group_d2h_s = _copy_cross_kv_group_to_host(items)
            cache_d2h_s += group_d2h_s
            if trace_prefill_iterations:
                result["prefill_trace_events"].append(
                    {
                        "event": "cross_kv_d2h",
                        "page_index": int(result["page_index"]),
                        "page_image": Path(result["image_path"]).name,
                        "member_count": len(group),
                        "source_lengths": [
                            int(actual_length)
                            for _array, actual_length, _item_s in host_exports
                        ],
                        "output_shapes": [
                            list(array.shape)
                            for array, _actual_length, _item_s in host_exports
                        ],
                        "output_bytes": sum(
                            int(array.nbytes)
                            for array, _actual_length, _item_s in host_exports
                        ),
                        "transfer_mode": os.environ.get(
                            "UNIREC_CROSS_KV_D2H_MODE", "per_crop"
                        ),
                        "wall_s": group_d2h_s,
                    }
                )
            for crop, item, (packed_host, actual_length, item_d2h_s) in zip(
                group, items, host_exports
            ):
                member_real = int(
                    item.text_prefill_real_source_tokens or actual_length
                )
                member_physical = int(
                    item.text_prefill_physical_source_tokens or member_real
                )
                real_tokens += member_real
                physical_tokens += member_physical
                crop["worker_cross_kv"] = packed_host
                stages = (
                    dict(item.prefill_device_stage_s)
                    if item.prefill_device_stage_s is not None
                    else None
                )
                if stages is not None:
                    stages["compiled_full_vision_buckets"] = vision_share_per_crop
                crop["worker_prefill_metadata"] = {
                    "prep": item.prep,
                    "prefill_s": float(item.prefill_s + vision_share_per_crop),
                    "cache_d2h_s": item_d2h_s,
                    "prefill_device_stage_s": stages,
                    "text_prefill_execution": item.text_prefill_execution,
                    "text_prefill_real_source_tokens": member_real,
                    "text_prefill_physical_source_tokens": member_physical,
                    "actual_cross_attention_length": actual_length,
                    "vision_bucket": vision_bucket_by_source[
                        crop_source_indices[id(crop)]
                    ],
                }
        if packed_prefill_rejected_crop_ids:
            retained_pairs = [
                (crop, block_id)
                for crop, block_id in zip(
                    result["crops"], result["vlm_block_ids"]
                )
                if id(crop) not in packed_prefill_rejected_crop_ids
            ]
            result["crops"] = [crop for crop, _block_id in retained_pairs]
            result["vlm_block_ids"] = [
                block_id for _crop, block_id in retained_pairs
            ]
            result["cross_capacity_rejected_crops"] = int(
                result.get("cross_capacity_rejected_crops", 0)
            ) + len(packed_prefill_rejected_crop_ids)
        page_crop_count = len(result["crops"])
        page_vision_s = vision_share_per_crop * page_crop_count
        page_timings.append(
            {
                "recognition_prefill_worker_s": (
                    time.perf_counter() - page_started + page_vision_s
                ),
                "recognition_prefill_cache_d2h_s": cache_d2h_s,
                "recognition_prefill_real_source_tokens": float(real_tokens),
                "recognition_prefill_physical_source_tokens": float(
                    physical_tokens
                ),
                "recognition_prefill_pack_count": float(pack_count),
                "recognition_prefill_fallback_count": float(fallback_count),
                "recognition_full_vision_s": page_vision_s,
                "recognition_full_vision_initial_sync_s": (
                    synchronization_s if result is results[0] else 0.0
                ),
                "recognition_vision_bucket_rows": bucket_rows,
                "recognition_vision_group_owner": result_index == 0,
                "recognition_vision_group_bucket_calls": (
                    group_bucket_calls if result_index == 0 else {}
                ),
                "recognition_vision_group_bucket_real_rows": (
                    group_bucket_real_rows if result_index == 0 else {}
                ),
                "recognition_vision_group_bucket_physical_rows": (
                    group_bucket_physical_rows if result_index == 0 else {}
                ),
                "recognition_vision_group_fallback_rows": (
                    group_fallback_rows if result_index == 0 else 0
                ),
            }
        )
    if encoded_by_source:
        raise RuntimeError(
            f"bucketed worker left {len(encoded_by_source)} vision rows unconsumed"
        )
    return page_timings


def _collect_full_vision_worker_tasks(
    first_task: tuple[int, int, str],
    *,
    task_queue: Any,
    page_lookahead: int,
) -> tuple[list[tuple[int, int, str]], bool, tuple[int, int, str] | None]:
    """Collect one bounded same-run page group without delaying a partial tail."""
    tasks = [first_task]
    saw_shutdown = False
    deferred_task = None
    first_run_id = int(first_task[0])
    while len(tasks) < page_lookahead:
        try:
            task = task_queue.get(timeout=FULL_VISION_PAGE_COLLECT_TIMEOUT_S)
        except queue.Empty:
            break
        if task is None:
            saw_shutdown = True
            break
        if int(task[0]) != first_run_id:
            deferred_task = task
            break
        tasks.append(task)
    return tasks, saw_shutdown, deferred_task


def _prepare_full_vision_worker_page(
    task: tuple[int, int, str],
    *,
    runtime: Any,
    threshold: float,
    use_chart_recognition: bool,
    tokenize_figure_of_table: Any,
    recognition_processor: Any,
    recognition_preprocess_executor: ThreadPoolExecutor | None,
    static_cross_cache_len: int,
    predecoded: tuple[float, np.ndarray, dict[str, float]] | None = None,
    layout_result: dict[str, Any] | None = None,
    detector_s: float | None = None,
    trace_prefill_iterations: bool = False,
) -> dict[str, Any]:
    """Run the CPU/layout half of one page before cross-page vision batching."""
    run_id, page_index, path_string = task
    path = Path(path_string)
    if predecoded is None:
        started = time.perf_counter()
        rgb, decode_timing = _decode_rgb(path)
        materialize_started = time.perf_counter()
        rgb = materialize_layout_rgb(rgb)
        decode_timing["rgb_materialize_s"] = (
            time.perf_counter() - materialize_started
        )
        decode_timing["rgb_to_bgr_s"] = 0.0
    else:
        started, rgb, decode_timing = predecoded
    if layout_result is None:
        detector_started = time.perf_counter()
        layout_result = runtime([rgb], threshold=threshold)[0]
        detector_s = time.perf_counter() - detector_started
    elif detector_s is None:
        raise ValueError("precomputed layout result requires detector timing")
    result, frontend_timing = _prepare_frontend_payload(
        page_index=page_index,
        path=path,
        rgb=rgb,
        layout_result=layout_result,
        use_chart_recognition=use_chart_recognition,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )
    result["started_at"] = started
    recognition_prepare_started = time.perf_counter()
    recognition_detail_s = {
        name: 0.0 for name in RECOGNITION_CPU_DETAIL_TIMING_FIELDS
    }
    rejected_crops = 0
    capacity_filter_started = time.perf_counter()
    if static_cross_cache_len > 0:
        kept_crops = []
        kept_block_ids = []
        for crop, block_id in zip(result["crops"], result["vlm_block_ids"]):
            height, width = crop["image_rgb"].shape[:2]
            tokens = recognition_processor.estimate_encoder_token_count_for_image_size(
                width,
                height,
            )
            if tokens > static_cross_cache_len:
                rejected_crops += 1
                continue
            kept_crops.append(crop)
            kept_block_ids.append(block_id)
        result["crops"] = kept_crops
        result["vlm_block_ids"] = kept_block_ids
    recognition_detail_s["recognition_capacity_filter_s"] = (
        time.perf_counter() - capacity_filter_started
    )
    result["cross_capacity_rejected_crops"] = rejected_crops
    if trace_prefill_iterations:
        result["prefill_trace_events"] = []
    recognition_input_contract = os.environ.get(
        "UNIREC_RECOGNITION_INPUT_CONTRACT",
        "compact_uint8_hwc",
    )
    if recognition_input_contract not in {
        "compact_uint8_hwc",
        "legacy_float32_bchw",
    }:
        raise ValueError(
            "invalid UNIREC_RECOGNITION_INPUT_CONTRACT: "
            f"{recognition_input_contract!r}"
        )
    recognition_resize_backend = os.environ.get(
        "UNIREC_RECOGNITION_RESIZE_BACKEND",
        "pillow",
    )
    if recognition_resize_backend not in {"pillow", "kornia_rs"}:
        raise ValueError(
            "invalid UNIREC_RECOGNITION_RESIZE_BACKEND: "
            f"{recognition_resize_backend!r}"
        )
    kornia_image_type = None
    if recognition_resize_backend == "kornia_rs":
        from kornia_rs.image import Image as KorniaImage

        kornia_image_type = KorniaImage

    def prepare_crop(
        crop: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, float], float, dict[str, Any] | None]:
        cpu_evidence = None
        if trace_prefill_iterations:
            monotonic_start_ns = time.perf_counter_ns()
            thread_cpu_start_ns = time.thread_time_ns()
            native_thread_id = threading.get_native_id()
            thread_name = threading.current_thread().name
            cpu_start = _current_cpu()
        prepare_started = time.perf_counter()
        detail_s: dict[str, float] = {}
        if recognition_input_contract == "compact_uint8_hwc":
            if recognition_resize_backend == "kornia_rs":
                assert kornia_image_type is not None
                image_rgb = crop["image_rgb"]
                detail_s["recognition_pil_fromarray_s"] = 0.0
                operation_started = time.perf_counter()
                target_size = recognition_processor.get_processed_size(
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                )
                target_size_s = time.perf_counter() - operation_started
                operation_started = time.perf_counter()
                pixel_values = kornia_image_type.fromarray(image_rgb).resize(
                    target_size[0],
                    target_size[1],
                    "bicubic",
                ).data
                resize_s = time.perf_counter() - operation_started
                processor_timing_s = {
                    "recognition_processor_image_convert_rgb_s": 0.0,
                    "recognition_processor_target_size_s": target_size_s,
                    "recognition_processor_resize_s": resize_s,
                    "recognition_processor_pil_to_uint8_hwc_s": 0.0,
                }
                expected = (target_size[1], target_size[0], 3)
                if (
                    pixel_values.dtype != np.uint8
                    or tuple(pixel_values.shape) != expected
                ):
                    raise ValueError(
                        "kornia-rs UniRec resize produced an invalid array: "
                        f"{pixel_values.dtype} {pixel_values.shape}, "
                        f"expected uint8 {expected}"
                    )
            else:
                operation_started = time.perf_counter()
                image = Image.fromarray(crop["image_rgb"])
                detail_s["recognition_pil_fromarray_s"] = (
                    time.perf_counter() - operation_started
                )
                pixel_values, processor_timing_s = (
                    _resize_recognition_compact_hwc_with_timing(
                        image,
                        processor=recognition_processor,
                    )
                )
        else:
            operation_started = time.perf_counter()
            image = Image.fromarray(crop["image_rgb"])
            detail_s["recognition_pil_fromarray_s"] = (
                time.perf_counter() - operation_started
            )
            inputs, processor_timing_s = (
                recognition_processor.process_with_timing(image)
            )
            operation_started = time.perf_counter()
            pixel_values_numpy = inputs["pixel_values"].numpy()
            detail_s["recognition_tensor_numpy_view_s"] = (
                time.perf_counter() - operation_started
            )
            operation_started = time.perf_counter()
            pixel_values = np.ascontiguousarray(
                pixel_values_numpy,
                dtype=np.float32,
            )
            detail_s["recognition_contiguous_chw_copy_s"] = (
                time.perf_counter() - operation_started
            )
        detail_s.update(
            {name: float(value) for name, value in processor_timing_s.items()}
        )
        prepare_wall_s = time.perf_counter() - prepare_started
        if trace_prefill_iterations:
            cpu_end = _current_cpu()
            thread_cpu_end_ns = time.thread_time_ns()
            monotonic_end_ns = time.perf_counter_ns()
            cpu_evidence = {
                "native_thread_id": int(native_thread_id),
                "thread_name": str(thread_name),
                "cpu_start": cpu_start,
                "cpu_end": cpu_end,
                "thread_cpu_s": (
                    thread_cpu_end_ns - thread_cpu_start_ns
                ) / 1_000_000_000.0,
                "monotonic_start_ns": int(monotonic_start_ns),
                "monotonic_end_ns": int(monotonic_end_ns),
            }
        return pixel_values, detail_s, prepare_wall_s, cpu_evidence

    if recognition_preprocess_executor is None:
        prepared_crops = map(prepare_crop, result["crops"])
    else:
        prepared_crops = recognition_preprocess_executor.map(
            prepare_crop,
            result["crops"],
        )
    for crop, (
        pixel_values,
        detail_s,
        crop_prepare_wall_s,
        crop_cpu_evidence,
    ) in zip(
        result["crops"],
        prepared_crops,
    ):
        for name, value in detail_s.items():
            recognition_detail_s[name] += float(value)
        crop["processed_pixel_values"] = pixel_values
        if trace_prefill_iterations:
            source_height, source_width = crop["image_rgb"].shape[:2]
            if pixel_values.ndim == 3:
                processed_height, processed_width = pixel_values.shape[:2]
            elif pixel_values.ndim == 4:
                processed_height, processed_width = pixel_values.shape[-2:]
            else:
                raise ValueError(
                    "unexpected traced recognition input shape: "
                    f"{pixel_values.shape}"
                )
            event = {
                "event": "recognition_crop_preprocess",
                "page_index": int(page_index),
                "page_image": path.name,
                "crop_index": int(crop["crop_index"]),
                "label": str(crop["label"]),
                "source_image_size": [source_width, source_height],
                "processed_image_size": [
                    int(processed_width),
                    int(processed_height),
                ],
                "processed_tensor_shape": list(pixel_values.shape),
                "encoder_tokens": int(
                    recognition_processor.estimate_encoder_token_count_from_processed_size(
                        processed_width=int(processed_width),
                        processed_height=int(processed_height),
                    )
                ),
                "stage_s": {
                    name: float(value) for name, value in detail_s.items()
                },
                "wall_s": crop_prepare_wall_s,
            }
            if crop_cpu_evidence is None:
                raise RuntimeError("traced crop has no CPU execution evidence")
            event.update(crop_cpu_evidence)
            result["prefill_trace_events"].append(event)
    recognition_prepare_s = time.perf_counter() - recognition_prepare_started
    result["frontend_timing_s"] = {
        "page_file_read_s": decode_timing["file_read_s"],
        "page_image_decode_s": decode_timing["direct_rgb_decode_s"],
        "page_rgb_materialize_s": decode_timing["rgb_materialize_s"],
        "page_rgb_to_bgr_s": decode_timing["rgb_to_bgr_s"],
        "layout_s": detector_s,
        **frontend_timing,
        "recognition_input_prepare_worker_s": recognition_prepare_s,
        **recognition_detail_s,
    }
    return {
        "run_id": int(run_id),
        "page_index": int(page_index),
        "path": path_string,
        "started": started,
        "decode_timing": decode_timing,
        "detector_s": detector_s,
        "frontend_timing": frontend_timing,
        "recognition_prepare_s": recognition_prepare_s,
        "result": result,
    }


def _unlink_worker_shared_payload(result: dict[str, Any]) -> None:
    """Best-effort cleanup if a grouped page fails after shared packing."""
    shared = result.get("shared_memory")
    if shared is None:
        return
    try:
        storage = SharedMemory(name=str(shared["name"]))
    except FileNotFoundError:
        return
    try:
        storage.unlink()
    finally:
        storage.close()


def _run_full_vision_worker_group(
    tasks: list[tuple[int, int, str]],
    *,
    group_started: float,
    collect_s: float,
    worker_index: int,
    runtime: Any,
    threshold: float,
    use_chart_recognition: bool,
    tokenize_figure_of_table: Any,
    recognition_processor: Any,
    recognition_preprocess_executor: ThreadPoolExecutor | None,
    static_cross_cache_len: int,
    recognition_runner: Any,
    full_vision_runtime: Any,
    page_lookahead: int,
    empty_cache_after_page: bool,
    profile_prefill_device_stages: bool,
    trace_prefill_iterations: bool,
    retain_shared_images: bool,
    result_queue: Any,
) -> None:
    """Prepare, batch-prefill, pack, and publish one worker-local page group."""
    layout_batch_size = int(getattr(runtime, "batch_size", 1))
    contexts = []
    for batch_start in range(0, len(tasks), layout_batch_size):
        batch_tasks = tasks[batch_start : batch_start + layout_batch_size]
        decoded_batch = []
        for task in batch_tasks:
            path = Path(task[2])
            page_started = time.perf_counter()
            rgb, decode_timing = _decode_rgb(path)
            materialize_started = time.perf_counter()
            rgb = materialize_layout_rgb(rgb)
            decode_timing["rgb_materialize_s"] = (
                time.perf_counter() - materialize_started
            )
            decode_timing["rgb_to_bgr_s"] = 0.0
            decoded_batch.append(
                (
                    page_started,
                    rgb,
                    decode_timing,
                )
            )
        detector_started = time.perf_counter()
        layout_stage_before = (
            dict(runtime.stage_s) if trace_prefill_iterations else {}
        )
        layout_results = runtime(
            [decoded[1] for decoded in decoded_batch],
            threshold=threshold,
        )
        detector_batch_s = time.perf_counter() - detector_started
        layout_stage_s = (
            {
                name: float(seconds)
                - float(layout_stage_before.get(name, 0.0))
                for name, seconds in runtime.stage_s.items()
            }
            if trace_prefill_iterations
            else {}
        )
        if len(layout_results) != len(batch_tasks):
            raise RuntimeError(
                "layout batch result count mismatch: "
                f"{len(layout_results)} != {len(batch_tasks)}"
            )
        detector_page_s = detector_batch_s / len(batch_tasks)
        batch_contexts = []
        for task, predecoded, layout_result in zip(
            batch_tasks,
            decoded_batch,
            layout_results,
        ):
            context = _prepare_full_vision_worker_page(
                task,
                runtime=runtime,
                threshold=threshold,
                use_chart_recognition=use_chart_recognition,
                tokenize_figure_of_table=tokenize_figure_of_table,
                recognition_processor=recognition_processor,
                recognition_preprocess_executor=recognition_preprocess_executor,
                static_cross_cache_len=static_cross_cache_len,
                predecoded=predecoded,
                layout_result=layout_result,
                detector_s=detector_page_s,
                trace_prefill_iterations=trace_prefill_iterations,
            )
            context["layout_batch_real_size"] = len(batch_tasks)
            context["layout_batch_physical_size"] = layout_batch_size
            contexts.append(context)
            batch_contexts.append(context)
        if trace_prefill_iterations and batch_contexts:
            batch_contexts[0]["result"]["prefill_trace_events"].append(
                {
                    "event": "layout_batch_call",
                    "page_indices": [int(task[1]) for task in batch_tasks],
                    "page_images": [Path(task[2]).name for task in batch_tasks],
                    "source_image_shapes": [
                        [
                            int(decoded[1].shape[0]),
                            int(decoded[1].shape[1]),
                            int(decoded[1].shape[2]),
                        ]
                        for decoded in decoded_batch
                    ],
                    "real_rows": len(batch_tasks),
                    "physical_rows": layout_batch_size,
                    "slot_efficiency": len(batch_tasks) / layout_batch_size,
                    "physical_input_shape": [
                        layout_batch_size,
                        3,
                        800,
                        800,
                    ],
                    "wall_s": detector_batch_s,
                    "stage_s": layout_stage_s,
                }
            )
    results = [context["result"] for context in contexts]
    memory_device = torch.device(recognition_runner.device)
    torch.npu.reset_peak_memory_stats(memory_device)
    npu_memory_before_bytes = int(torch.npu.memory_allocated(memory_device))
    prefill_timings = _prefill_worker_pages_bucketed(
        results,
        runner=recognition_runner,
        vision_runtime=full_vision_runtime,
        profile_device_stages=profile_prefill_device_stages,
        trace_prefill_iterations=trace_prefill_iterations,
    )
    npu_memory_after_bytes = int(torch.npu.memory_allocated(memory_device))
    npu_peak_memory_bytes = int(torch.npu.max_memory_allocated(memory_device))
    group_size = len(contexts)
    group_crop_count = sum(len(result["crops"]) for result in results)
    collect_share_s = collect_s / group_size
    empty_cache_share_s = 0.0
    if empty_cache_after_page:
        empty_cache_started = time.perf_counter()
        torch.npu.empty_cache()
        empty_cache_share_s = (
            time.perf_counter() - empty_cache_started
        ) / group_size

    packed_contexts = []
    try:
        for group_index, (context, prefill_timing) in enumerate(
            zip(contexts, prefill_timings)
        ):
            result = context["result"]
            result["frontend_timing_s"].update(
                {
                    name: value
                    for name, value in prefill_timing.items()
                    if name.endswith("_s")
                }
            )
            result["frontend_timing_s"][
                "recognition_page_lookahead_collect_s"
            ] = collect_share_s
            if empty_cache_after_page:
                result["frontend_timing_s"][
                    "recognition_worker_empty_cache_s"
                ] = empty_cache_share_s
            result["worker_prefill_stats"] = {
                name: value
                for name, value in prefill_timing.items()
                if not name.endswith("_s")
            }
            result["worker_prefill_stats"].update(
                {
                    "recognition_full_vision_page_group_size": group_size,
                    "recognition_full_vision_page_group_crop_count": (
                        group_crop_count
                    ),
                    "recognition_full_vision_page_group_index": group_index,
                    "recognition_full_vision_page_lookahead": page_lookahead,
                    "layout_batch_real_size": context[
                        "layout_batch_real_size"
                    ],
                    "layout_batch_physical_size": context[
                        "layout_batch_physical_size"
                    ],
                }
            )
            if group_index == 0:
                result["worker_prefill_stats"].update(
                    {
                        "recognition_group_npu_memory_before_bytes": (
                            npu_memory_before_bytes
                        ),
                        "recognition_group_npu_memory_after_bytes": (
                            npu_memory_after_bytes
                        ),
                        "recognition_group_npu_peak_memory_bytes": (
                            npu_peak_memory_bytes
                        ),
                        "recognition_group_npu_peak_increment_bytes": max(
                            0,
                            npu_peak_memory_bytes - npu_memory_before_bytes,
                        ),
                    }
                )
            result, shared_pack_s, shared_payload_bytes = (
                _pack_frontend_payload_shared(
                    result,
                    retain_images=retain_shared_images,
                )
            )
            result["frontend_timing_s"]["process_shared_pack_s"] = shared_pack_s
            if trace_prefill_iterations:
                result["prefill_trace_events"].append(
                    {
                        "event": "page_shared_pack",
                        "page_index": int(result["page_index"]),
                        "page_image": Path(result["image_path"]).name,
                        "crop_count": len(result["crops"]),
                        "output_bytes": int(shared_payload_bytes),
                        "wall_s": shared_pack_s,
                    }
                )
            context["result"] = result
            context["prefill_timing"] = prefill_timing
            context["shared_pack_s"] = shared_pack_s
            context["shared_payload_bytes"] = shared_payload_bytes
            packed_contexts.append(context)
    except BaseException:
        for context in packed_contexts:
            _unlink_worker_shared_payload(context["result"])
        raise

    group_wall_s = time.perf_counter() - group_started
    if trace_prefill_iterations and packed_contexts:
        packed_contexts[0]["result"]["prefill_trace_events"].append(
            {
                "event": "worker_page_group",
                "page_indices": [
                    int(context["page_index"]) for context in packed_contexts
                ],
                "page_images": [
                    Path(context["path"]).name for context in packed_contexts
                ],
                "page_count": len(packed_contexts),
                "crop_count": group_crop_count,
                "lookahead_collect_s": collect_s,
                "wall_s": group_wall_s,
            }
        )
    work_estimates = []
    for context in packed_contexts:
        prefill_timing = context["prefill_timing"]
        work_estimates.append(
            sum(float(value) for value in context["decode_timing"].values())
            + float(context["detector_s"])
            + sum(float(value) for value in context["frontend_timing"].values())
            + float(context["recognition_prepare_s"])
            + float(prefill_timing["recognition_prefill_worker_s"])
            + float(context["shared_pack_s"])
        )
    estimate_total = sum(work_estimates)
    if estimate_total <= 0:
        worker_page_shares = [group_wall_s / group_size] * group_size
    else:
        worker_page_shares = [
            group_wall_s * estimate / estimate_total for estimate in work_estimates
        ]

    for context, worker_page_s in zip(packed_contexts, worker_page_shares):
        ready_at = time.perf_counter()
        result_queue.put(
            {
                "status": "ok",
                "worker": worker_index,
                "run_id": context["run_id"],
                "page_index": context["page_index"],
                "path": context["path"],
                "result": context["result"],
                "ready_at": ready_at,
                "timing": {
                    **context["decode_timing"],
                    "detector_call_s": context["detector_s"],
                    **context["frontend_timing"],
                    "shared_pack_s": context["shared_pack_s"],
                    "shared_payload_bytes": context["shared_payload_bytes"],
                    "recognition_input_prepare_s": context[
                        "recognition_prepare_s"
                    ],
                    "recognition_page_lookahead_collect_s": collect_share_s,
                    "layout_batch_call_share": 1.0
                    / float(context["layout_batch_real_size"]),
                    "layout_batch_physical_row_share": float(
                        context["layout_batch_physical_size"]
                    )
                    / float(context["layout_batch_real_size"]),
                    "worker_page_s": worker_page_s,
                },
                "prefix_diagnostics": {
                    "call_count": 0,
                    "new_first_calls": {},
                },
            }
        )


def _new_worker_vision_batch_summary() -> dict[str, Any]:
    return {
        "page_groups": 0,
        "pages": 0,
        "crops": 0,
        "page_group_size_histogram": {},
        "bucket_real_rows": {},
        "bucket_calls": {},
        "bucket_physical_rows": {},
        "fallback_rows": 0,
        "max_npu_memory_before_bytes": 0,
        "max_npu_memory_after_bytes": 0,
        "max_npu_peak_memory_bytes": 0,
        "max_npu_peak_increment_bytes": 0,
    }


def _accumulate_worker_vision_batch_summary(
    summary: dict[str, Any],
    result: dict[str, Any],
) -> None:
    stats = result.get("worker_prefill_stats", {})
    if not stats:
        return
    summary["pages"] += 1
    for key, rows in stats.get("recognition_vision_bucket_rows", {}).items():
        summary["bucket_real_rows"][key] = (
            summary["bucket_real_rows"].get(key, 0) + int(rows)
        )
    if not stats.get("recognition_vision_group_owner", False):
        return
    summary["page_groups"] += 1
    group_size = int(stats["recognition_full_vision_page_group_size"])
    group_crops = int(stats["recognition_full_vision_page_group_crop_count"])
    summary["crops"] += group_crops
    histogram = summary["page_group_size_histogram"]
    histogram[str(group_size)] = histogram.get(str(group_size), 0) + 1
    for destination, source in (
        ("bucket_calls", "recognition_vision_group_bucket_calls"),
        (
            "bucket_physical_rows",
            "recognition_vision_group_bucket_physical_rows",
        ),
    ):
        for key, value in stats.get(source, {}).items():
            summary[destination][key] = (
                summary[destination].get(key, 0) + int(value)
            )
    summary["fallback_rows"] += int(
        stats.get("recognition_vision_group_fallback_rows", 0)
    )
    for destination, source in (
        ("max_npu_memory_before_bytes", "recognition_group_npu_memory_before_bytes"),
        ("max_npu_memory_after_bytes", "recognition_group_npu_memory_after_bytes"),
        ("max_npu_peak_memory_bytes", "recognition_group_npu_peak_memory_bytes"),
        (
            "max_npu_peak_increment_bytes",
            "recognition_group_npu_peak_increment_bytes",
        ),
    ):
        summary[destination] = max(
            int(summary[destination]),
            int(stats.get(source, 0)),
        )


def _finish_worker_vision_batch_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    compiled_real_rows = sum(int(value) for value in summary["bucket_real_rows"].values())
    compiled_physical_rows = sum(
        int(value) for value in summary["bucket_physical_rows"].values()
    )
    return {
        **summary,
        "compiled_real_rows": compiled_real_rows,
        "compiled_physical_rows": compiled_physical_rows,
        "compiled_padding_rows": compiled_physical_rows - compiled_real_rows,
        "compiled_slot_efficiency": (
            compiled_real_rows / compiled_physical_rows
            if compiled_physical_rows
            else None
        ),
    }


def _worker_main(
    worker_index: int,
    model_path: str,
    cache_dir: str,
    threshold: float,
    execution: str,
    layout_dtype: str,
    layout_reading_order_dtype: str | None,
    layout_weight_format: str,
    layout_depthwise_rewrite: str,
    layout_preformat_frozen_bn_buffers: bool,
    layout_batch_size: int,
    warmup_path: str,
    openocr_root: str | None,
    prepare_pages: bool,
    use_chart_recognition: bool,
    prefill_recognition: bool,
    recognition_model_path: str | None,
    recognition_dtype: str,
    recognition_cache_dir: str | None,
    recognition_prefix_shapes_manifest: str | None,
    recognition_full_vision_buckets: bool,
    recognition_vision_focal_depthwise_rewrite: str,
    recognition_vision_weight_format: str,
    recognition_page_lookahead: int,
    empty_cache_after_page: bool,
    profile_prefill_device_stages: bool,
    trace_prefill_iterations: bool,
    retain_shared_images: bool,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        runtime = PPDocLayoutV2NpuAdapter(
            model_path=model_path,
            device="npu:0",
            dtype=layout_dtype,
            reading_order_dtype=layout_reading_order_dtype,
            threshold=threshold,
            profile_stages=trace_prefill_iterations,
            execution=execution,
            compile_cache_dir=cache_dir,
            batch_size=layout_batch_size,
            weight_format=layout_weight_format,
            depthwise_rewrite=layout_depthwise_rewrite,
            preformat_frozen_bn_buffers=(
                layout_preformat_frozen_bn_buffers
            ),
            input_color_order="rgb",
        )
        warmup_rgb, _ = _decode_rgb(Path(warmup_path))
        runtime([materialize_layout_rgb(warmup_rgb)], threshold=threshold)
        runtime.reset_timing()
        tokenize_figure_of_table = None
        if prepare_pages:
            if openocr_root is None:
                raise RuntimeError("full frontend workers require OpenOCR root")
            sys.path.insert(0, openocr_root)
            from tools.utils.opendoc_onnx_utils.utils import (
                tokenize_figure_of_table as openocr_tokenize_figure_of_table,
            )

            tokenize_figure_of_table = openocr_tokenize_figure_of_table
            from modeling_optimized_unirec import UniRecImageProcessor

            recognition_processor = UniRecImageProcessor()
            recognition_preprocess_threads = int(
                os.environ.get("UNIREC_RECOGNITION_PREPROCESS_THREADS", "1")
            )
            if recognition_preprocess_threads < 1:
                raise ValueError(
                    "UNIREC_RECOGNITION_PREPROCESS_THREADS must be positive"
                )
            recognition_preprocess_executor = (
                ThreadPoolExecutor(
                    max_workers=recognition_preprocess_threads,
                    thread_name_prefix=f"unirec-crop-{worker_index}",
                )
                if recognition_preprocess_threads > 1
                else None
            )
            static_cross_cache_len = int(
                os.environ.get("UNIREC_STATIC_CROSS_CACHE_LEN", "0")
            )
        else:
            recognition_processor = None
            recognition_preprocess_threads = 1
            recognition_preprocess_executor = None
        if prefill_recognition:
            if not prepare_pages:
                raise RuntimeError("worker recognition prefill requires page preparation")
            if recognition_model_path is None or recognition_cache_dir is None:
                raise RuntimeError("worker recognition prefill has no model/cache path")
            from modeling_optimized_unirec import OptimizedUniRecRunner
            from vision_atlas import UniRecVisionAtlasRuntime

            recognition_runner = OptimizedUniRecRunner(
                model_path=recognition_model_path,
                device="npu:0",
                dtype=recognition_dtype,
                compile_cache_dir=recognition_cache_dir,
            )
            if static_cross_cache_len > 0:
                processor_shape = tuple(
                    int(value) for value in recognition_runner.processor.max_side
                )
                recognition_runner._static_cross_cache_len_by_processor_max_side[
                    processor_shape
                ] = static_cross_cache_len
            if recognition_full_vision_buckets:
                if recognition_prefix_shapes_manifest is not None:
                    raise RuntimeError(
                        "full-vision buckets cannot be combined with a "
                        "per-shape prefix manifest"
                    )
                from vision_bucket_presets import resolve_vision_bucket_specs
                from vision_full_batch import BucketedFullVisionRuntime

                vision_bucket_preset = os.environ.get(
                    "UNIREC_VISION_BUCKET_PRESET", "production_v1"
                )

                full_vision_runtime = BucketedFullVisionRuntime(
                    recognition_runner,
                    specs=resolve_vision_bucket_specs(vision_bucket_preset),
                    diagnostic_graph_log=(
                        os.environ.get(
                            "UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG", "0"
                        )
                        == "1"
                    ),
                    trace_iterations=trace_prefill_iterations,
                    focal_depthwise_rewrite=(
                        recognition_vision_focal_depthwise_rewrite
                    ),
                    weight_format=recognition_vision_weight_format,
                    preset_name=vision_bucket_preset,
                )
                vision_atlas_runtime = None
                warmup_started = time.perf_counter()
                warmup_report = full_vision_runtime.warmup_all(passes=1)
                fallback_warmup_report = (
                    full_vision_runtime.warmup_eager_fallback(passes=2)
                )
                warmup_wall_s = time.perf_counter() - warmup_started
                prefix_graph_warmup = {
                    "execution": "compiled_masked_full_encoder_buckets",
                    "shape_count": len(warmup_report),
                    "wall_s": warmup_wall_s,
                    "graphs": warmup_report,
                    "fallback_eager": fallback_warmup_report,
                }
            elif recognition_prefix_shapes_manifest is None:
                full_vision_runtime = None
                vision_atlas_runtime = UniRecVisionAtlasRuntime(recognition_runner)
                prefix_graph_warmup = None
            else:
                full_vision_runtime = None
                from vision_static_shape import (
                    PerShapeCompiledPrefixUniRecVisionRuntime,
                    load_static_vision_shapes,
                )

                vision_atlas_runtime = PerShapeCompiledPrefixUniRecVisionRuntime(
                    recognition_runner,
                    shapes=load_static_vision_shapes(
                        Path(recognition_prefix_shapes_manifest)
                    ),
                )
                prefix_graph_warmup_started = time.perf_counter()
                prefix_graph_warmup_report = (
                    vision_atlas_runtime.warmup_all_prefix_graphs(passes=1)
                )
                prefix_graph_warmup_wall_s = (
                    time.perf_counter() - prefix_graph_warmup_started
                )
                prefix_graph_call_wall_s = {
                    str(shape): float(values["pass_wall_s"][0])
                    for shape, values in prefix_graph_warmup_report.items()
                }
                # The explicit graph sweep is the first call for every shape
                # in this worker. Mark those shapes warm so page diagnostics
                # only report unexpected post-warmup first calls.
                vision_atlas_runtime.prefix_first_call_wall_s.update(
                    prefix_graph_call_wall_s
                )
                prefix_graph_warmup = {
                    "shape_count": len(prefix_graph_call_wall_s),
                    "wall_s": prefix_graph_warmup_wall_s,
                    "graph_call_wall_sum_s": sum(
                        prefix_graph_call_wall_s.values()
                    ),
                    "graph_call_wall_min_s": min(
                        prefix_graph_call_wall_s.values()
                    ),
                    "graph_call_wall_max_s": max(
                        prefix_graph_call_wall_s.values()
                    ),
                }
        else:
            recognition_runner = None
            vision_atlas_runtime = None
            full_vision_runtime = None
            prefix_graph_warmup = None
        result_queue.put(
            {
                "status": "ready",
                "worker": worker_index,
                "cpu_affinity": _cpu_affinity(),
                "prefix_graph_warmup": prefix_graph_warmup,
                "recognition_preprocess_threads": (
                    recognition_preprocess_threads
                ),
                "layout_batch_size": layout_batch_size,
                "layout_dtype": layout_dtype,
                "layout_reading_order_dtype": (
                    layout_reading_order_dtype or layout_dtype
                ),
                "layout_weight_format": layout_weight_format,
                "layout_depthwise_rewrite": layout_depthwise_rewrite,
                "layout_preformat_frozen_bn_buffers": (
                    layout_preformat_frozen_bn_buffers
                ),
                "layout_input_color_order": runtime.input_color_order,
                "layout_depthwise_rewrite_summary": (
                    runtime.depthwise_rewrite_summary
                ),
                "layout_weight_format_summary": runtime.weight_format_summary,
                "layout_frozen_bn_buffer_format_summary": (
                    runtime.frozen_bn_buffer_format_summary
                ),
                "vision_focal_depthwise_rewrite": (
                    recognition_vision_focal_depthwise_rewrite
                ),
                "vision_weight_format": recognition_vision_weight_format,
                "vision_bucket_preset": (
                    full_vision_runtime.preset_name
                    if full_vision_runtime is not None
                    else None
                ),
                "vision_focal_depthwise_rewrite_summary": (
                    full_vision_runtime.focal_depthwise_rewrite_summary
                    if full_vision_runtime is not None
                    else None
                ),
                "vision_weight_format_summary": (
                    full_vision_runtime.weight_format_summary
                    if full_vision_runtime is not None
                    else None
                ),
            }
        )
        if full_vision_runtime is not None:
            deferred_task = None
            while True:
                if deferred_task is None:
                    first_task = task_queue.get()
                else:
                    first_task = deferred_task
                    deferred_task = None
                if first_task is None:
                    return
                group_started = time.perf_counter()
                tasks, saw_shutdown, deferred_task = (
                    _collect_full_vision_worker_tasks(
                        first_task,
                        task_queue=task_queue,
                        page_lookahead=recognition_page_lookahead,
                    )
                )
                collect_s = time.perf_counter() - group_started
                _run_full_vision_worker_group(
                    tasks,
                    group_started=group_started,
                    collect_s=collect_s,
                    worker_index=worker_index,
                    runtime=runtime,
                    threshold=threshold,
                    use_chart_recognition=use_chart_recognition,
                    tokenize_figure_of_table=tokenize_figure_of_table,
                    recognition_processor=recognition_processor,
                    recognition_preprocess_executor=(
                        recognition_preprocess_executor
                    ),
                    static_cross_cache_len=static_cross_cache_len,
                    recognition_runner=recognition_runner,
                    full_vision_runtime=full_vision_runtime,
                    page_lookahead=recognition_page_lookahead,
                    empty_cache_after_page=empty_cache_after_page,
                    profile_prefill_device_stages=(
                        profile_prefill_device_stages
                    ),
                    trace_prefill_iterations=trace_prefill_iterations,
                    retain_shared_images=retain_shared_images,
                    result_queue=result_queue,
                )
                if saw_shutdown:
                    return
        while True:
            task = task_queue.get()
            if task is None:
                return
            run_id, page_index, path_string = task
            path = Path(path_string)
            started = time.perf_counter()
            rgb, decode_timing = _decode_rgb(path)
            materialize_started = time.perf_counter()
            rgb = materialize_layout_rgb(rgb)
            decode_timing["rgb_materialize_s"] = (
                time.perf_counter() - materialize_started
            )
            decode_timing["rgb_to_bgr_s"] = 0.0
            detector_started = time.perf_counter()
            layout_result = runtime([rgb], threshold=threshold)[0]
            detector_s = time.perf_counter() - detector_started
            frontend_timing: dict[str, float] = {}
            shared_pack_s = 0.0
            shared_payload_bytes = 0
            prefix_new_first_calls: dict[str, float] = {}
            prefix_call_count_delta = 0
            if prepare_pages:
                result, frontend_timing = _prepare_frontend_payload(
                    page_index=page_index,
                    path=path,
                    rgb=rgb,
                    layout_result=layout_result,
                    use_chart_recognition=use_chart_recognition,
                    tokenize_figure_of_table=tokenize_figure_of_table,
                )
                result["started_at"] = started
                recognition_prepare_started = time.perf_counter()
                rejected_crops = 0
                if static_cross_cache_len > 0:
                    kept_crops = []
                    kept_block_ids = []
                    for crop, block_id in zip(
                        result["crops"], result["vlm_block_ids"]
                    ):
                        height, width = crop["image_rgb"].shape[:2]
                        tokens = recognition_processor.estimate_encoder_token_count_for_image_size(
                            width, height
                        )
                        if tokens > static_cross_cache_len:
                            rejected_crops += 1
                            continue
                        kept_crops.append(crop)
                        kept_block_ids.append(block_id)
                    result["crops"] = kept_crops
                    result["vlm_block_ids"] = kept_block_ids
                result["cross_capacity_rejected_crops"] = rejected_crops
                for crop in result["crops"]:
                    inputs = recognition_processor(Image.fromarray(crop["image_rgb"]))
                    crop["processed_pixel_values"] = np.ascontiguousarray(
                        inputs["pixel_values"].numpy(),
                        dtype=np.float32,
                    )
                recognition_prepare_s = (
                    time.perf_counter() - recognition_prepare_started
                )
                result["frontend_timing_s"] = {
                    "page_file_read_s": decode_timing["file_read_s"],
                    "page_image_decode_s": decode_timing["direct_rgb_decode_s"],
                    "page_rgb_materialize_s": decode_timing["rgb_materialize_s"],
                    "page_rgb_to_bgr_s": decode_timing["rgb_to_bgr_s"],
                    "layout_s": detector_s,
                    **frontend_timing,
                    "recognition_input_prepare_worker_s": recognition_prepare_s,
                }
                if prefill_recognition:
                    prefix_first_calls_before = set(
                        getattr(
                            vision_atlas_runtime,
                            "prefix_first_call_wall_s",
                            {},
                        )
                    )
                    prefix_call_count_before = int(
                        getattr(vision_atlas_runtime, "prefix_call_count", 0)
                    )
                    prefill_timing = _prefill_worker_page(
                        result,
                        runner=recognition_runner,
                        vision_atlas_runtime=vision_atlas_runtime,
                        profile_device_stages=profile_prefill_device_stages,
                    )
                    prefix_first_calls_after = getattr(
                        vision_atlas_runtime,
                        "prefix_first_call_wall_s",
                        {},
                    )
                    prefix_new_first_calls = {
                        str(name): float(wall_s)
                        for name, wall_s in prefix_first_calls_after.items()
                        if name not in prefix_first_calls_before
                    }
                    prefix_call_count_delta = int(
                        getattr(vision_atlas_runtime, "prefix_call_count", 0)
                    ) - prefix_call_count_before
                    result["frontend_timing_s"].update(
                        {
                            name: value
                            for name, value in prefill_timing.items()
                            if name.endswith("_s")
                        }
                    )
                    result["worker_prefill_stats"] = {
                        name: value
                        for name, value in prefill_timing.items()
                        if not name.endswith("_s")
                    }
                    if empty_cache_after_page:
                        empty_cache_started = time.perf_counter()
                        torch.npu.empty_cache()
                        result["frontend_timing_s"][
                            "recognition_worker_empty_cache_s"
                        ] = time.perf_counter() - empty_cache_started
                result, shared_pack_s, shared_payload_bytes = (
                    _pack_frontend_payload_shared(
                        result,
                        retain_images=retain_shared_images,
                    )
                )
                result["frontend_timing_s"]["process_shared_pack_s"] = (
                    shared_pack_s
                )
            else:
                result = layout_result
            ready_at = time.perf_counter()
            result_queue.put(
                {
                    "status": "ok",
                    "worker": worker_index,
                    "run_id": run_id,
                    "page_index": page_index,
                    "path": path_string,
                    "result": result,
                    "ready_at": ready_at,
                    "timing": {
                        **decode_timing,
                        "detector_call_s": detector_s,
                        **frontend_timing,
                        "shared_pack_s": shared_pack_s,
                        "shared_payload_bytes": shared_payload_bytes,
                        "recognition_input_prepare_s": (
                            recognition_prepare_s if prepare_pages else 0.0
                        ),
                        "worker_page_s": ready_at - started,
                    },
                    "prefix_diagnostics": {
                        "call_count": prefix_call_count_delta,
                        "new_first_calls": prefix_new_first_calls,
                    },
                }
            )
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "worker": worker_index,
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


class DynamicLayoutProcessPool:
    """Keep isolated layout runtimes ready and schedule file paths dynamically."""

    def __init__(
        self,
        *,
        worker_count: int,
        model_path: Path,
        cache_dir: Path,
        threshold: float,
        execution: str,
        warmup_paths: list[Path],
        layout_dtype: str = "float32",
        layout_reading_order_dtype: str | None = None,
        layout_weight_format: str = DEFAULT_LAYOUT_WEIGHT_FORMAT,
        layout_depthwise_rewrite: str = DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
        layout_preformat_frozen_bn_buffers: bool = False,
        layout_batch_size: int = 1,
        openocr_root: Path | None = None,
        prepare_pages: bool = False,
        use_chart_recognition: bool = False,
        prefill_recognition: bool = False,
        recognition_model_path: Path | None = None,
        recognition_dtype: str = "float16",
        recognition_cache_dir: Path | None = None,
        recognition_prefix_shapes_manifest: Path | None = None,
        recognition_full_vision_buckets: bool = False,
        recognition_vision_focal_depthwise_rewrite: str = "native",
        recognition_vision_weight_format: str = "native",
        recognition_page_lookahead: int = 1,
        empty_cache_after_page: bool = False,
        profile_prefill_device_stages: bool = False,
        trace_prefill_iterations: bool = False,
        retain_shared_images: bool = True,
        timeout_s: float = 1800.0,
        progress_every_pages: int = 0,
        progress_heartbeat_s: float = 0.0,
    ) -> None:
        if worker_count < 1:
            raise ValueError("layout process worker count must be positive")
        if not warmup_paths:
            raise ValueError("layout process pool requires at least one warmup page")
        if recognition_page_lookahead < 1:
            raise ValueError("recognition page lookahead must be positive")
        if layout_batch_size < 1:
            raise ValueError("layout batch size must be positive")
        if layout_batch_size > 1 and not recognition_full_vision_buckets:
            raise ValueError(
                "process-worker layout batching requires full-vision grouping"
            )
        if layout_batch_size > recognition_page_lookahead:
            raise ValueError(
                "layout batch size cannot exceed recognition page lookahead"
            )
        if progress_every_pages < 0:
            raise ValueError("progress page interval must be non-negative")
        if progress_heartbeat_s < 0:
            raise ValueError("progress heartbeat interval must be non-negative")
        self.worker_count = worker_count
        self.layout_batch_size = int(layout_batch_size)
        self.layout_dtype = str(layout_dtype)
        self.layout_reading_order_dtype = (
            None
            if layout_reading_order_dtype is None
            else str(layout_reading_order_dtype)
        )
        self.layout_weight_format = str(layout_weight_format)
        self.layout_depthwise_rewrite = str(layout_depthwise_rewrite)
        self.layout_preformat_frozen_bn_buffers = bool(
            layout_preformat_frozen_bn_buffers
        )
        self.prepare_pages = prepare_pages
        self.recognition_full_vision_buckets = recognition_full_vision_buckets
        self.recognition_vision_focal_depthwise_rewrite = str(
            recognition_vision_focal_depthwise_rewrite
        )
        self.recognition_vision_weight_format = str(
            recognition_vision_weight_format
        )
        self.recognition_page_lookahead = recognition_page_lookahead
        self.trace_prefill_iterations = bool(trace_prefill_iterations)
        self.retain_shared_images = bool(retain_shared_images)
        self.timeout_s = timeout_s
        self.progress_every_pages = int(progress_every_pages)
        self.progress_heartbeat_s = float(progress_heartbeat_s)
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue(maxsize=max(2, worker_count * 2))
        self.processes = [
            self.context.Process(
                target=_worker_main,
                args=(
                    worker_index,
                    str(model_path),
                    str(cache_dir),
                    threshold,
                    execution,
                    self.layout_dtype,
                    self.layout_reading_order_dtype,
                    self.layout_weight_format,
                    self.layout_depthwise_rewrite,
                    self.layout_preformat_frozen_bn_buffers,
                    self.layout_batch_size,
                    str(warmup_paths[worker_index % len(warmup_paths)]),
                    str(openocr_root) if openocr_root is not None else None,
                    prepare_pages,
                    use_chart_recognition,
                    prefill_recognition,
                    (
                        str(recognition_model_path)
                        if recognition_model_path is not None
                        else None
                    ),
                    recognition_dtype,
                    (
                        str(recognition_cache_dir)
                        if recognition_cache_dir is not None
                        else None
                    ),
                    (
                        str(recognition_prefix_shapes_manifest)
                        if recognition_prefix_shapes_manifest is not None
                        else None
                    ),
                    recognition_full_vision_buckets,
                    self.recognition_vision_focal_depthwise_rewrite,
                    self.recognition_vision_weight_format,
                    recognition_page_lookahead,
                    empty_cache_after_page,
                    profile_prefill_device_stages,
                    self.trace_prefill_iterations,
                    self.retain_shared_images,
                    self.task_queue,
                    self.result_queue,
                ),
                name=f"unirec-layout-process-{worker_index}",
            )
            for worker_index in range(worker_count)
        ]
        setup_started = time.perf_counter()
        for process in self.processes:
            process.start()
        ready = []
        last_ready_at = setup_started
        while len(ready) < len(self.processes):
            message = self._receive_stream_message(
                label="worker_setup",
                completed=len(ready),
                total=len(self.processes),
                stream_started=setup_started,
                last_result_at=last_ready_at,
            )
            last_ready_at = time.perf_counter()
            ready.append(message)
            print(
                "UNIREC_LAYOUT_PROCESS_SETUP_WORKER_READY "
                f"workers={len(ready)}/{len(self.processes)} "
                f"worker={int(message.get('worker', -1))} "
                f"status={message.get('status')} "
                f"elapsed_s={last_ready_at - setup_started:.3f}",
                flush=True,
            )
        errors = [message for message in ready if message["status"] != "ready"]
        if errors:
            self.close()
            raise RuntimeError(f"layout process setup failed: {errors}")
        self.worker_setup_diagnostics = [
            {
                "worker": int(message["worker"]),
                "cpu_affinity": [
                    int(cpu) for cpu in message.get("cpu_affinity", [])
                ],
                "cpu_affinity_count": len(message.get("cpu_affinity", [])),
                "prefix_graph_warmup": message.get("prefix_graph_warmup"),
                "recognition_preprocess_threads": int(
                    message.get("recognition_preprocess_threads", 1)
                ),
                "layout_batch_size": int(message.get("layout_batch_size", 1)),
                "layout_dtype": str(message["layout_dtype"]),
                "layout_input_color_order": str(
                    message["layout_input_color_order"]
                ),
                "layout_weight_format": str(message["layout_weight_format"]),
                "layout_depthwise_rewrite": str(
                    message["layout_depthwise_rewrite"]
                ),
                "layout_preformat_frozen_bn_buffers": bool(
                    message["layout_preformat_frozen_bn_buffers"]
                ),
                "layout_depthwise_rewrite_summary": message[
                    "layout_depthwise_rewrite_summary"
                ],
                "layout_weight_format_summary": message[
                    "layout_weight_format_summary"
                ],
                "layout_frozen_bn_buffer_format_summary": message[
                    "layout_frozen_bn_buffer_format_summary"
                ],
                "vision_focal_depthwise_rewrite": str(
                    message["vision_focal_depthwise_rewrite"]
                ),
                "vision_weight_format": str(message["vision_weight_format"]),
                "vision_focal_depthwise_rewrite_summary": message[
                    "vision_focal_depthwise_rewrite_summary"
                ],
                "vision_weight_format_summary": message[
                    "vision_weight_format_summary"
                ],
            }
            for message in sorted(ready, key=lambda value: int(value["worker"]))
        ]
        self.setup_wall_s = time.perf_counter() - setup_started
        self._next_run_id = 0
        self.last_stream_summary: dict[str, Any] | None = None
        self.closed = False

    def _receive(self) -> dict[str, Any]:
        try:
            return self.result_queue.get(timeout=self.timeout_s)
        except queue.Empty as exception:
            alive = [process.is_alive() for process in self.processes]
            raise TimeoutError(
                f"layout process pool was silent for {self.timeout_s}s; alive={alive}"
            ) from exception

    def _receive_stream_message(
        self,
        *,
        label: str,
        completed: int,
        total: int,
        stream_started: float,
        last_result_at: float,
    ) -> dict[str, Any]:
        """Receive one result while making a silent worker pool observable."""
        if self.progress_heartbeat_s <= 0:
            return self._receive()
        while True:
            try:
                return self.result_queue.get(timeout=self.progress_heartbeat_s)
            except queue.Empty as exception:
                now = time.perf_counter()
                alive = [process.is_alive() for process in self.processes]
                exitcodes = [process.exitcode for process in self.processes]
                silence_s = now - last_result_at
                print(
                    f"UNIREC_LAYOUT_PROCESS_HEARTBEAT label={label} "
                    f"pages={completed}/{total} elapsed_s={now - stream_started:.1f} "
                    f"silence_s={silence_s:.1f} alive={alive} "
                    f"exitcodes={exitcodes}",
                    flush=True,
                )
                if not all(alive):
                    raise RuntimeError(
                        "layout worker exited without returning a result: "
                        f"alive={alive} exitcodes={exitcodes}"
                    ) from exception
                if silence_s >= self.timeout_s:
                    raise TimeoutError(
                        f"layout process pool was silent for {silence_s:.1f}s; "
                        f"alive={alive} exitcodes={exitcodes}"
                    ) from exception

    def map(
        self,
        paths: list[Path],
        *,
        label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.closed:
            raise RuntimeError("layout process pool is closed")
        run_id = self._next_run_id
        self._next_run_id += 1
        started = time.perf_counter()
        for page_index, path in enumerate(paths):
            self.task_queue.put((run_id, page_index, str(path)))
        results: list[dict[str, Any] | None] = [None] * len(paths)
        worker_pages = [0] * self.worker_count
        worker_busy_s = [0.0] * self.worker_count
        stage_s = {
            "worker_file_read_sum_s": 0.0,
            "worker_direct_rgb_decode_sum_s": 0.0,
            "worker_rgb_materialize_sum_s": 0.0,
            "worker_rgb_to_bgr_sum_s": 0.0,
            "worker_detector_call_sum_s": 0.0,
            "worker_layout_crop_views_sum_s": 0.0,
            "worker_document_image_index_sum_s": 0.0,
            "worker_recognition_crop_build_sum_s": 0.0,
            "worker_recognition_input_prepare_sum_s": 0.0,
            "worker_recognition_prefill_sum_s": 0.0,
            "worker_recognition_prefill_cache_d2h_sum_s": 0.0,
            "worker_recognition_page_collect_sum_s": 0.0,
            "worker_shared_pack_sum_s": 0.0,
        }
        stage_s.update(
            {
                f"worker_{name.removesuffix('_s')}_sum_s": 0.0
                for name in RECOGNITION_CPU_DETAIL_TIMING_FIELDS
            }
        )
        vision_batch_summary = _new_worker_vision_batch_summary()
        shared_payload_bytes = 0
        layout_batch_call_shares = 0.0
        layout_batch_physical_row_shares = 0.0
        ipc_delivery_sum_s = 0.0
        ipc_delivery_max_s = 0.0
        progress_step = self.progress_every_pages or max(1, len(paths) // 10)
        completed = 0
        last_result_at = started
        while completed < len(paths):
            message = self._receive_stream_message(
                label=label,
                completed=completed,
                total=len(paths),
                stream_started=started,
                last_result_at=last_result_at,
            )
            last_result_at = time.perf_counter()
            if message["status"] != "ok":
                raise RuntimeError(f"layout process execution failed: {message}")
            if int(message["run_id"]) != run_id:
                raise RuntimeError(
                    f"unexpected layout run id: {message['run_id']} != {run_id}"
                )
            page_index = int(message["page_index"])
            if results[page_index] is not None:
                raise RuntimeError(f"duplicate layout result for page {page_index}")
            results[page_index] = message["result"]
            worker_index = int(message["worker"])
            worker_pages[worker_index] += 1
            timing = message["timing"]
            ipc_delivery_s = time.perf_counter() - float(message["ready_at"])
            if self.trace_prefill_iterations:
                message["result"].setdefault("prefill_trace_events", []).append(
                    {
                        "event": "coordinator_ipc_delivery",
                        "page_index": page_index,
                        "page_image": Path(message["path"]).name,
                        "wall_s": ipc_delivery_s,
                    }
                )
            ipc_delivery_sum_s += ipc_delivery_s
            ipc_delivery_max_s = max(ipc_delivery_max_s, ipc_delivery_s)
            worker_busy_s[worker_index] += float(timing["worker_page_s"])
            stage_s["worker_file_read_sum_s"] += float(timing["file_read_s"])
            stage_s["worker_direct_rgb_decode_sum_s"] += float(
                timing["direct_rgb_decode_s"]
            )
            stage_s["worker_rgb_materialize_sum_s"] += float(
                timing.get("rgb_materialize_s", 0.0)
            )
            stage_s["worker_rgb_to_bgr_sum_s"] += float(
                timing.get("rgb_to_bgr_s", 0.0)
            )
            stage_s["worker_detector_call_sum_s"] += float(
                timing["detector_call_s"]
            )
            stage_s["worker_layout_crop_views_sum_s"] += float(
                timing.get("layout_crop_views_s", 0.0)
            )
            stage_s["worker_document_image_index_sum_s"] += float(
                timing.get("document_image_index_s", 0.0)
            )
            stage_s["worker_recognition_crop_build_sum_s"] += float(
                timing.get("recognition_crop_build_s", 0.0)
            )
            stage_s["worker_recognition_input_prepare_sum_s"] += float(
                timing.get("recognition_input_prepare_s", 0.0)
            )
            frontend = message["result"].get("frontend_timing_s", {})
            for name in RECOGNITION_CPU_DETAIL_TIMING_FIELDS:
                stage_s[f"worker_{name.removesuffix('_s')}_sum_s"] += float(
                    frontend.get(name, 0.0)
                )
            stage_s["worker_recognition_prefill_sum_s"] += float(
                frontend.get("recognition_prefill_worker_s", 0.0)
            )
            stage_s["worker_recognition_prefill_cache_d2h_sum_s"] += float(
                frontend.get("recognition_prefill_cache_d2h_s", 0.0)
            )
            stage_s["worker_recognition_page_collect_sum_s"] += float(
                timing.get("recognition_page_lookahead_collect_s", 0.0)
            )
            stage_s["worker_shared_pack_sum_s"] += float(
                timing.get("shared_pack_s", 0.0)
            )
            _accumulate_worker_vision_batch_summary(
                vision_batch_summary,
                message["result"],
            )
            shared_payload_bytes += int(timing.get("shared_payload_bytes", 0))
            layout_batch_call_shares += float(
                timing.get("layout_batch_call_share", 1.0)
            )
            layout_batch_physical_row_shares += float(
                timing.get("layout_batch_physical_row_share", 1.0)
            )
            completed += 1
            if completed % progress_step == 0 or completed == len(paths):
                print(
                    f"UNIREC_LAYOUT_PROCESS_PAGE label={label} "
                    f"pages={completed}/{len(paths)} "
                    f"page_index={int(message['page_index'])} "
                    f"worker={worker_index} "
                    f"worker_page_s={float(timing['worker_page_s']):.3f} "
                    f"elapsed_s={time.perf_counter() - started:.3f} "
                    f"crops={len(message['result'].get('crops', []))} "
                    f"rejected={int(message['result'].get('cross_capacity_rejected_crops', 0))}",
                    flush=True,
                )
        wall_s = time.perf_counter() - started
        summary = {
            "label": label,
            "worker_count": self.worker_count,
            "page_count": len(paths),
            "wall_s": wall_s,
            "pages_per_s": len(paths) / wall_s if wall_s else None,
            "worker_page_counts": worker_pages,
            "worker_busy_s": worker_busy_s,
            "stage_s": stage_s,
            "ipc_delivery_sum_s": ipc_delivery_sum_s,
            "ipc_delivery_mean_s": ipc_delivery_sum_s / len(paths),
            "ipc_delivery_max_s": ipc_delivery_max_s,
            "shared_payload_bytes": shared_payload_bytes,
            "scheduling": "dynamic_shared_filepath_queue",
            "layout_batch_size": self.layout_batch_size,
            "layout_batching": {
                "calls": int(round(layout_batch_call_shares)),
                "real_rows": len(paths),
                "physical_rows": int(round(layout_batch_physical_row_shares)),
                "slot_efficiency": (
                    len(paths) / layout_batch_physical_row_shares
                    if layout_batch_physical_row_shares
                    else None
                ),
            },
            "full_page_frontend": self.prepare_pages,
            "retained_shared_images": self.retain_shared_images,
            "recognition_full_vision_buckets": (
                self.recognition_full_vision_buckets
            ),
            "recognition_page_lookahead": self.recognition_page_lookahead,
            "trace_prefill_iterations": self.trace_prefill_iterations,
            "vision_batching": _finish_worker_vision_batch_summary(
                vision_batch_summary
            ),
        }
        if any(result is None for result in results):
            raise RuntimeError("layout process pool returned incomplete results")
        return [result for result in results if result is not None], summary

    def iter_map(
        self,
        paths: list[Path],
        *,
        label: str,
    ) -> Any:
        """Yield completed page payloads with bounded result-queue backpressure."""
        if self.closed:
            raise RuntimeError("layout process pool is closed")
        run_id = self._next_run_id
        self._next_run_id += 1
        started = time.perf_counter()
        for page_index, path in enumerate(paths):
            self.task_queue.put((run_id, page_index, str(path)))
        worker_pages = [0] * self.worker_count
        worker_busy_s = [0.0] * self.worker_count
        worker_page_indices: list[list[int]] = [
            [] for _ in range(self.worker_count)
        ]
        worker_prefix_call_counts = [0] * self.worker_count
        worker_prefix_new_first_calls: list[dict[str, float]] = [
            {} for _ in range(self.worker_count)
        ]
        stage_s = {
            "worker_file_read_sum_s": 0.0,
            "worker_direct_rgb_decode_sum_s": 0.0,
            "worker_rgb_materialize_sum_s": 0.0,
            "worker_rgb_to_bgr_sum_s": 0.0,
            "worker_detector_call_sum_s": 0.0,
            "worker_layout_crop_views_sum_s": 0.0,
            "worker_document_image_index_sum_s": 0.0,
            "worker_recognition_crop_build_sum_s": 0.0,
            "worker_recognition_input_prepare_sum_s": 0.0,
            "worker_recognition_prefill_sum_s": 0.0,
            "worker_recognition_prefill_cache_d2h_sum_s": 0.0,
            "worker_recognition_page_collect_sum_s": 0.0,
            "worker_shared_pack_sum_s": 0.0,
        }
        stage_s.update(
            {
                f"worker_{name.removesuffix('_s')}_sum_s": 0.0
                for name in RECOGNITION_CPU_DETAIL_TIMING_FIELDS
            }
        )
        vision_batch_summary = _new_worker_vision_batch_summary()
        shared_payload_bytes = 0
        layout_batch_call_shares = 0.0
        layout_batch_physical_row_shares = 0.0
        ipc_delivery_sum_s = 0.0
        ipc_delivery_max_s = 0.0
        progress_step = self.progress_every_pages or max(1, len(paths) // 10)
        completed = 0
        last_result_at = started
        while completed < len(paths):
            message = self._receive_stream_message(
                label=label,
                completed=completed,
                total=len(paths),
                stream_started=started,
                last_result_at=last_result_at,
            )
            last_result_at = time.perf_counter()
            if message["status"] != "ok":
                raise RuntimeError(f"layout process execution failed: {message}")
            if int(message["run_id"]) != run_id:
                raise RuntimeError(
                    f"unexpected layout run id: {message['run_id']} != {run_id}"
                )
            worker_index = int(message["worker"])
            timing = message["timing"]
            worker_pages[worker_index] += 1
            worker_page_indices[worker_index].append(int(message["page_index"]))
            worker_busy_s[worker_index] += float(timing["worker_page_s"])
            prefix_diagnostics = message.get("prefix_diagnostics", {})
            worker_prefix_call_counts[worker_index] += int(
                prefix_diagnostics.get("call_count", 0)
            )
            worker_prefix_new_first_calls[worker_index].update(
                {
                    str(name): float(wall_s)
                    for name, wall_s in prefix_diagnostics.get(
                        "new_first_calls", {}
                    ).items()
                }
            )
            ipc_delivery_s = time.perf_counter() - float(message["ready_at"])
            if self.trace_prefill_iterations:
                message["result"].setdefault("prefill_trace_events", []).append(
                    {
                        "event": "coordinator_ipc_delivery",
                        "page_index": int(message["page_index"]),
                        "page_image": Path(message["path"]).name,
                        "wall_s": ipc_delivery_s,
                    }
                )
            ipc_delivery_sum_s += ipc_delivery_s
            ipc_delivery_max_s = max(ipc_delivery_max_s, ipc_delivery_s)
            for destination, source in (
                ("worker_file_read_sum_s", "file_read_s"),
                ("worker_direct_rgb_decode_sum_s", "direct_rgb_decode_s"),
                ("worker_rgb_materialize_sum_s", "rgb_materialize_s"),
                ("worker_rgb_to_bgr_sum_s", "rgb_to_bgr_s"),
                ("worker_detector_call_sum_s", "detector_call_s"),
                ("worker_layout_crop_views_sum_s", "layout_crop_views_s"),
                ("worker_document_image_index_sum_s", "document_image_index_s"),
                ("worker_recognition_crop_build_sum_s", "recognition_crop_build_s"),
                ("worker_recognition_input_prepare_sum_s", "recognition_input_prepare_s"),
                (
                    "worker_recognition_page_collect_sum_s",
                    "recognition_page_lookahead_collect_s",
                ),
                ("worker_shared_pack_sum_s", "shared_pack_s"),
            ):
                stage_s[destination] += float(timing.get(source, 0.0))
            frontend = message["result"].get("frontend_timing_s", {})
            for name in RECOGNITION_CPU_DETAIL_TIMING_FIELDS:
                stage_s[f"worker_{name.removesuffix('_s')}_sum_s"] += float(
                    frontend.get(name, 0.0)
                )
            stage_s["worker_recognition_prefill_sum_s"] += float(
                frontend.get("recognition_prefill_worker_s", 0.0)
            )
            stage_s["worker_recognition_prefill_cache_d2h_sum_s"] += float(
                frontend.get("recognition_prefill_cache_d2h_s", 0.0)
            )
            _accumulate_worker_vision_batch_summary(
                vision_batch_summary,
                message["result"],
            )
            shared_payload_bytes += int(timing.get("shared_payload_bytes", 0))
            layout_batch_call_shares += float(
                timing.get("layout_batch_call_share", 1.0)
            )
            layout_batch_physical_row_shares += float(
                timing.get("layout_batch_physical_row_share", 1.0)
            )
            completed += 1
            if completed % progress_step == 0 or completed == len(paths):
                print(
                    f"UNIREC_LAYOUT_PROCESS_PAGE label={label} "
                    f"pages={completed}/{len(paths)} "
                    f"page_index={int(message['page_index'])} "
                    f"worker={worker_index} "
                    f"worker_page_s={float(timing['worker_page_s']):.3f} "
                    f"elapsed_s={time.perf_counter() - started:.3f} "
                    f"crops={len(message['result'].get('crops', []))} "
                    f"rejected={int(message['result'].get('cross_capacity_rejected_crops', 0))}",
                    flush=True,
                )
            yield message["result"]

        wall_s = time.perf_counter() - started
        self.last_stream_summary = {
            "label": label,
            "worker_count": self.worker_count,
            "page_count": len(paths),
            "wall_s": wall_s,
            "pages_per_s": len(paths) / wall_s if wall_s else None,
            "worker_page_counts": worker_pages,
            "worker_page_indices": worker_page_indices,
            "worker_busy_s": worker_busy_s,
            "prefix_diagnostics": {
                "worker_call_counts": worker_prefix_call_counts,
                "worker_new_first_call_counts": [
                    len(values) for values in worker_prefix_new_first_calls
                ],
                "worker_new_first_call_wall_s": [
                    sum(values.values()) for values in worker_prefix_new_first_calls
                ],
                "worker_new_first_call_shapes": [
                    sorted(values) for values in worker_prefix_new_first_calls
                ],
                "new_first_call_count": sum(
                    len(values) for values in worker_prefix_new_first_calls
                ),
                "new_first_call_wall_sum_s": sum(
                    sum(values.values())
                    for values in worker_prefix_new_first_calls
                ),
            },
            "stage_s": stage_s,
            "ipc_delivery_sum_s": ipc_delivery_sum_s,
            "ipc_delivery_mean_s": ipc_delivery_sum_s / len(paths),
            "ipc_delivery_max_s": ipc_delivery_max_s,
            "shared_payload_bytes": shared_payload_bytes,
            "scheduling": "dynamic_completion_order_stream",
            "layout_batch_size": self.layout_batch_size,
            "layout_batching": {
                "calls": int(round(layout_batch_call_shares)),
                "real_rows": len(paths),
                "physical_rows": int(round(layout_batch_physical_row_shares)),
                "slot_efficiency": (
                    len(paths) / layout_batch_physical_row_shares
                    if layout_batch_physical_row_shares
                    else None
                ),
            },
            "full_page_frontend": self.prepare_pages,
            "retained_shared_images": self.retain_shared_images,
            "recognition_full_vision_buckets": (
                self.recognition_full_vision_buckets
            ),
            "recognition_page_lookahead": self.recognition_page_lookahead,
            "trace_prefill_iterations": self.trace_prefill_iterations,
            "vision_batching": _finish_worker_vision_batch_summary(
                vision_batch_summary
            ),
        }

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=10.0)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def __enter__(self) -> "DynamicLayoutProcessPool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
