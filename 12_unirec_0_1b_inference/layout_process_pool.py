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
                profile_device_stages=False,
            )
            pack_count += 1
        else:
            items = [
                runner.prefill_prepared_for_cohort(
                    prepared[0],
                    profile_device_stages=False,
                    text_prefill_mode="eager",
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
    empty_cache_after_page: bool,
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
            vision_atlas_runtime = UniRecVisionAtlasRuntime(recognition_runner)
        else:
            recognition_runner = None
            vision_atlas_runtime = None
        result_queue.put({"status": "ready", "worker": worker_index})
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
                    prefill_timing = _prefill_worker_page(
                        result,
                        runner=recognition_runner,
                        vision_atlas_runtime=vision_atlas_runtime,
                    )
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
        empty_cache_after_page: bool = False,
        timeout_s: float = 1800.0,
    ) -> None:
        if worker_count < 1:
            raise ValueError("layout process worker count must be positive")
        if not warmup_paths:
            raise ValueError("layout process pool requires at least one warmup page")
        self.worker_count = worker_count
        self.prepare_pages = prepare_pages
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
                    empty_cache_after_page,
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
            "worker_shared_pack_sum_s": 0.0,
        }
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
            stage_s["worker_recognition_prefill_sum_s"] += float(
                frontend.get("recognition_prefill_worker_s", 0.0)
            )
            stage_s["worker_recognition_prefill_cache_d2h_sum_s"] += float(
                frontend.get("recognition_prefill_cache_d2h_s", 0.0)
            )
            stage_s["worker_shared_pack_sum_s"] += float(
                timing.get("shared_pack_s", 0.0)
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
            "worker_shared_pack_sum_s": 0.0,
        }
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
            worker_busy_s[worker_index] += float(timing["worker_page_s"])
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
                ("worker_shared_pack_sum_s", "shared_pack_s"),
            ):
                stage_s[destination] += float(timing.get(source, 0.0))
            frontend = message["result"].get("frontend_timing_s", {})
            stage_s["worker_recognition_prefill_sum_s"] += float(
                frontend.get("recognition_prefill_worker_s", 0.0)
            )
            stage_s["worker_recognition_prefill_cache_d2h_sum_s"] += float(
                frontend.get("recognition_prefill_cache_d2h_s", 0.0)
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
            "worker_busy_s": worker_busy_s,
            "stage_s": stage_s,
            "ipc_delivery_sum_s": ipc_delivery_sum_s,
            "ipc_delivery_mean_s": ipc_delivery_sum_s / len(paths),
            "ipc_delivery_max_s": ipc_delivery_max_s,
            "shared_payload_bytes": shared_payload_bytes,
            "scheduling": "dynamic_completion_order_stream",
            "layout_batch_size": 1,
            "full_page_frontend": self.prepare_pages,
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
