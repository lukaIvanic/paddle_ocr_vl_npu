"""Persistent dynamic B1 PP-DocLayoutV2 process pool.

The coordinator sends only page indices and file paths.  Every spawned process
owns one complete layout model/runtime.  Workers draw from one shared queue, so
no worker is tied to a slow static shard.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from kornia_rs.image import Image as KorniaImage
from PIL import Image
from torchvision.io import ImageReadMode, decode_image

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
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
) -> tuple[dict[str, Any], float, int]:
    """Move one full frontend payload into one aligned shared arena."""
    entries: list[tuple[str, dict[str, Any] | None, np.ndarray]] = [
        ("image_bgr_descriptor", None, result["image_bgr"])
    ]
    for crop in result["crops"]:
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


def _decode_rgb(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    encoded = path.read_bytes()
    read_s = time.perf_counter() - started
    started = time.perf_counter()
    if encoded.startswith(PNG_SIGNATURE):
        rgb = KorniaImage.decode(encoded, "RGB").data
    else:
        encoded_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        rgb = (
            decode_image(encoded_tensor, mode=ImageReadMode.RGB)
            .permute(1, 2, 0)
            .numpy()
        )
    decode_s = time.perf_counter() - started
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"unsupported decoded image: {rgb.shape} {rgb.dtype}")
    return rgb, {"file_read_s": read_s, "direct_rgb_decode_s": decode_s}


def _prepare_frontend_payload(
    *,
    page_index: int,
    path: Path,
    bgr: np.ndarray,
    layout_result: dict[str, Any],
    use_chart_recognition: bool,
    crop_margin: Any,
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
        cropped = bgr[y1:y2, x1:x2]
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
            block_image = crop_margin(block_image)
        crop_rgb = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)
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
    height, width = bgr.shape[:2]
    return (
        {
            "page_index": page_index,
            "image_path": str(path),
            "image_bgr": bgr,
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
        for crop, item in zip(group, items):
            cache = item.kv_cache
            actual_length = int(cache.actual_cross_attention_length or 0)
            if actual_length <= 0:
                raise RuntimeError("worker prefill produced an empty cross cache")
            d2h_started = time.perf_counter()
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
            item_d2h_s = time.perf_counter() - d2h_started
            cache_d2h_s += item_d2h_s
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
    encoded = vision_runtime.encode(flat_inputs)
    torch.npu.synchronize()
    vision_s = time.perf_counter() - vision_started
    synchronization_s = vision_started - synchronize_started
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
                raise RuntimeError(
                    "bucketed full-vision worker encountered an encoded crop "
                    "larger than the packed text-prefill bucket"
                )
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
            for crop, item in zip(group, items):
                cache = item.kv_cache
                actual_length = int(cache.actual_cross_attention_length or 0)
                if actual_length <= 0:
                    raise RuntimeError(
                        "bucketed worker prefill produced an empty cross cache"
                    )
                d2h_started = time.perf_counter()
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
                item_d2h_s = time.perf_counter() - d2h_started
                cache_d2h_s += item_d2h_s
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
    crop_margin: Any,
    tokenize_figure_of_table: Any,
    recognition_processor: Any,
    static_cross_cache_len: int,
) -> dict[str, Any]:
    """Run the CPU/layout half of one page before cross-page vision batching."""
    run_id, page_index, path_string = task
    path = Path(path_string)
    started = time.perf_counter()
    rgb, decode_timing = _decode_rgb(path)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    detector_started = time.perf_counter()
    layout_result = runtime([bgr], threshold=threshold)[0]
    detector_s = time.perf_counter() - detector_started
    result, frontend_timing = _prepare_frontend_payload(
        page_index=page_index,
        path=path,
        bgr=bgr,
        layout_result=layout_result,
        use_chart_recognition=use_chart_recognition,
        crop_margin=crop_margin,
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
    for crop in result["crops"]:
        operation_started = time.perf_counter()
        image = Image.fromarray(crop["image_rgb"])
        recognition_detail_s["recognition_pil_fromarray_s"] += (
            time.perf_counter() - operation_started
        )
        pixel_values, processor_timing_s = (
            _resize_recognition_compact_hwc_with_timing(
                image,
                processor=recognition_processor,
            )
        )
        for name, value in processor_timing_s.items():
            recognition_detail_s[name] += float(value)
        crop["processed_pixel_values"] = pixel_values
    recognition_prepare_s = time.perf_counter() - recognition_prepare_started
    result["frontend_timing_s"] = {
        "page_file_read_s": decode_timing["file_read_s"],
        "page_image_decode_s": decode_timing["direct_rgb_decode_s"],
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
    crop_margin: Any,
    tokenize_figure_of_table: Any,
    recognition_processor: Any,
    static_cross_cache_len: int,
    recognition_runner: Any,
    full_vision_runtime: Any,
    page_lookahead: int,
    empty_cache_after_page: bool,
    profile_prefill_device_stages: bool,
    result_queue: Any,
) -> None:
    """Prepare, batch-prefill, pack, and publish one worker-local page group."""
    contexts = [
        _prepare_full_vision_worker_page(
            task,
            runtime=runtime,
            threshold=threshold,
            use_chart_recognition=use_chart_recognition,
            crop_margin=crop_margin,
            tokenize_figure_of_table=tokenize_figure_of_table,
            recognition_processor=recognition_processor,
            static_cross_cache_len=static_cross_cache_len,
        )
        for task in tasks
    ]
    results = [context["result"] for context in contexts]
    memory_device = torch.device(recognition_runner.device)
    torch.npu.reset_peak_memory_stats(memory_device)
    npu_memory_before_bytes = int(torch.npu.memory_allocated(memory_device))
    prefill_timings = _prefill_worker_pages_bucketed(
        results,
        runner=recognition_runner,
        vision_runtime=full_vision_runtime,
        profile_device_stages=profile_prefill_device_stages,
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
                _pack_frontend_payload_shared(result)
            )
            result["frontend_timing_s"]["process_shared_pack_s"] = shared_pack_s
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
    recognition_page_lookahead: int,
    empty_cache_after_page: bool,
    profile_prefill_device_stages: bool,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        runtime = PPDocLayoutV2NpuAdapter(
            model_path=model_path,
            device="npu:0",
            dtype="float32",
            threshold=threshold,
            profile_stages=False,
            execution=execution,
            compile_cache_dir=cache_dir,
            batch_size=1,
        )
        warmup_rgb, _ = _decode_rgb(Path(warmup_path))
        runtime([warmup_rgb[..., ::-1]], threshold=threshold)
        runtime.reset_timing()
        crop_margin = None
        tokenize_figure_of_table = None
        if prepare_pages:
            if openocr_root is None:
                raise RuntimeError("full frontend workers require OpenOCR root")
            sys.path.insert(0, openocr_root)
            from tools.utils.opendoc_onnx_utils.utils import (
                crop_margin as openocr_crop_margin,
                tokenize_figure_of_table as openocr_tokenize_figure_of_table,
            )

            crop_margin = openocr_crop_margin
            tokenize_figure_of_table = openocr_tokenize_figure_of_table
            from modeling_optimized_unirec import UniRecImageProcessor

            recognition_processor = UniRecImageProcessor()
            static_cross_cache_len = int(
                os.environ.get("UNIREC_STATIC_CROSS_CACHE_LEN", "0")
            )
        else:
            recognition_processor = None
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
                from vision_full_batch import BucketedFullVisionRuntime

                full_vision_runtime = BucketedFullVisionRuntime(
                    recognition_runner
                )
                vision_atlas_runtime = None
                warmup_started = time.perf_counter()
                warmup_report = full_vision_runtime.warmup_all(passes=1)
                warmup_wall_s = time.perf_counter() - warmup_started
                prefix_graph_warmup = {
                    "execution": "compiled_masked_full_encoder_buckets",
                    "shape_count": len(warmup_report),
                    "wall_s": warmup_wall_s,
                    "graphs": warmup_report,
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
                "prefix_graph_warmup": prefix_graph_warmup,
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
                    crop_margin=crop_margin,
                    tokenize_figure_of_table=tokenize_figure_of_table,
                    recognition_processor=recognition_processor,
                    static_cross_cache_len=static_cross_cache_len,
                    recognition_runner=recognition_runner,
                    full_vision_runtime=full_vision_runtime,
                    page_lookahead=recognition_page_lookahead,
                    empty_cache_after_page=empty_cache_after_page,
                    profile_prefill_device_stages=(
                        profile_prefill_device_stages
                    ),
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
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            detector_started = time.perf_counter()
            layout_result = runtime([bgr], threshold=threshold)[0]
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
                    bgr=bgr,
                    layout_result=layout_result,
                    use_chart_recognition=use_chart_recognition,
                    crop_margin=crop_margin,
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
                    _pack_frontend_payload_shared(result)
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
        openocr_root: Path | None = None,
        prepare_pages: bool = False,
        use_chart_recognition: bool = False,
        prefill_recognition: bool = False,
        recognition_model_path: Path | None = None,
        recognition_dtype: str = "float16",
        recognition_cache_dir: Path | None = None,
        recognition_prefix_shapes_manifest: Path | None = None,
        recognition_full_vision_buckets: bool = False,
        recognition_page_lookahead: int = 1,
        empty_cache_after_page: bool = False,
        profile_prefill_device_stages: bool = False,
        timeout_s: float = 1800.0,
    ) -> None:
        if worker_count < 1:
            raise ValueError("layout process worker count must be positive")
        if not warmup_paths:
            raise ValueError("layout process pool requires at least one warmup page")
        if recognition_page_lookahead < 1:
            raise ValueError("recognition page lookahead must be positive")
        self.worker_count = worker_count
        self.prepare_pages = prepare_pages
        self.recognition_full_vision_buckets = recognition_full_vision_buckets
        self.recognition_page_lookahead = recognition_page_lookahead
        self.timeout_s = timeout_s
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
                    recognition_page_lookahead,
                    empty_cache_after_page,
                    profile_prefill_device_stages,
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
        ready = [self._receive() for _ in self.processes]
        errors = [message for message in ready if message["status"] != "ready"]
        if errors:
            self.close()
            raise RuntimeError(f"layout process setup failed: {errors}")
        self.worker_setup_diagnostics = [
            {
                "worker": int(message["worker"]),
                "prefix_graph_warmup": message.get("prefix_graph_warmup"),
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
        ipc_delivery_sum_s = 0.0
        ipc_delivery_max_s = 0.0
        progress_step = max(1, len(paths) // 10)
        completed = 0
        while completed < len(paths):
            message = self._receive()
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
            ipc_delivery_sum_s += ipc_delivery_s
            ipc_delivery_max_s = max(ipc_delivery_max_s, ipc_delivery_s)
            worker_busy_s[worker_index] += float(timing["worker_page_s"])
            stage_s["worker_file_read_sum_s"] += float(timing["file_read_s"])
            stage_s["worker_direct_rgb_decode_sum_s"] += float(
                timing["direct_rgb_decode_s"]
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
            completed += 1
            if completed % progress_step == 0 or completed == len(paths):
                print(
                    f"UNIREC_LAYOUT_PROCESS_PROGRESS label={label} "
                    f"pages={completed}/{len(paths)}",
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
            "layout_batch_size": 1,
            "full_page_frontend": self.prepare_pages,
            "recognition_full_vision_buckets": (
                self.recognition_full_vision_buckets
            ),
            "recognition_page_lookahead": self.recognition_page_lookahead,
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
        ipc_delivery_sum_s = 0.0
        ipc_delivery_max_s = 0.0
        progress_step = max(1, len(paths) // 10)
        completed = 0
        while completed < len(paths):
            message = self._receive()
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
            ipc_delivery_sum_s += ipc_delivery_s
            ipc_delivery_max_s = max(ipc_delivery_max_s, ipc_delivery_s)
            for destination, source in (
                ("worker_file_read_sum_s", "file_read_s"),
                ("worker_direct_rgb_decode_sum_s", "direct_rgb_decode_s"),
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
            completed += 1
            if completed % progress_step == 0 or completed == len(paths):
                print(
                    f"UNIREC_LAYOUT_PROCESS_PROGRESS label={label} "
                    f"pages={completed}/{len(paths)}",
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
            "layout_batch_size": 1,
            "full_page_frontend": self.prepare_pages,
            "recognition_full_vision_buckets": (
                self.recognition_full_vision_buckets
            ),
            "recognition_page_lookahead": self.recognition_page_lookahead,
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
