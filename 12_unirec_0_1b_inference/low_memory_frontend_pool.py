"""Torch-free W4/T8 recognition-crop preparation with disk-backed pixels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image


CPU_CROP_WORKER_MALLOC_CONF = (
    "narenas:2,background_thread:true,"
    "dirty_decay_ms:1000,muzzy_decay_ms:1000"
)
LAYOUT_OWNER_MALLOC_CONF = CPU_CROP_WORKER_MALLOC_CONF


def release_page_pixel_spools(payload: dict[str, Any]) -> int:
    """Delete per-page pixel files after the vision owner consumed them."""
    released = 0
    for value in payload.get("pixel_spool_paths", []):
        path = Path(value)
        if not path.exists():
            continue
        released += path.stat().st_size
        path.unlink()
    payload["pixel_spool_paths"] = []
    return released


def decode_page_rgb_cpu(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Decode RGB without importing Torch in the persistent CPU workers."""
    started = time.perf_counter()
    encoded = path.read_bytes()
    read_s = time.perf_counter() - started
    started = time.perf_counter()
    from kornia_rs.image import Image as KorniaImage

    rgb = KorniaImage.decode(encoded, "RGB").data
    decode_s = time.perf_counter() - started
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"unsupported decoded image: {rgb.shape} {rgb.dtype}")
    return rgb, {"page_file_read_s": read_s, "page_image_decode_s": decode_s}


def _base_label(label: str) -> str:
    parts = label.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else label


def _crop_margin_rgb(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
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


def _processed_size(width: int, height: int) -> tuple[int, int]:
    max_width, max_height = 960, 1408
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if (max_width / max_height) >= aspect_ratio:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
    else:
        new_width, new_height = width, height
    return max(int(new_width // 64 * 64), 64), max(int(new_height // 64 * 64), 64)


def _encoder_tokens(processed_width: int, processed_height: int) -> int:
    def axis(value: int) -> int:
        value = (value + 1) // 2
        value = (value + 1) // 2
        value = value // 2
        value = value // 2
        value = value // 2
        return value

    return axis(processed_width) * axis(processed_height)


def _prepare_frontend_payload(
    *,
    page_index: int,
    path: Path,
    rgb: np.ndarray,
    layout_result: dict[str, Any],
    use_chart_recognition: bool,
    tokenize_figure_of_table: Any,
    copy_source_crops: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
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
        crop_rgb = (
            np.ascontiguousarray(block_image)
            if copy_source_crops
            else block_image
        )
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


def _resize_crop(crop: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int], float]:
    started = time.perf_counter()
    image_rgb = crop["image_rgb"]
    source_height, source_width = image_rgb.shape[:2]
    target_width, target_height = _processed_size(source_width, source_height)
    if not image_rgb.flags.c_contiguous:
        image_rgb = np.ascontiguousarray(image_rgb)
    image = Image.fromarray(image_rgb, mode="RGB")
    resized = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )
    pixels = np.asarray(resized)
    if pixels.dtype != np.uint8 or not pixels.flags.c_contiguous:
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    return pixels, (source_width, source_height), time.perf_counter() - started


def _append_pixels(
    spool: Any,
    *,
    path: Path,
    pixels: np.ndarray,
) -> dict[str, Any]:
    offset = int(spool.tell())
    aligned = (offset + 63) // 64 * 64
    if aligned != offset:
        spool.write(b"\0" * (aligned - offset))
    spool.write(memoryview(pixels).cast("B"))
    return {
        "path": str(path),
        "offset": aligned,
        "shape": [int(value) for value in pixels.shape],
        "dtype": "uint8",
        "nbytes": int(pixels.nbytes),
    }


def _worker_main(
    worker_index: int,
    openocr_root: str,
    spool_root: str,
    recognition_threads: int,
    resize_chunk_size: int,
    cross_cache_length: int,
    use_chart_recognition: bool,
    spool_mode: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        if "torch" in sys.modules or "torch_npu" in sys.modules:
            raise RuntimeError("CPU crop worker inherited Torch")
        sys.path.insert(0, openocr_root)
        from tools.utils.opendoc_onnx_utils.utils import tokenize_figure_of_table

        spool_path = Path(spool_root) / f"worker_{worker_index:02d}.bin"
        spool_path.parent.mkdir(parents=True, exist_ok=True)
        executor = ThreadPoolExecutor(
            max_workers=recognition_threads,
            thread_name_prefix=f"unirec-crop-{worker_index}",
        )
        from host_memory_diagnostics import process_snapshot
        from post_warmup_host_cleanup import purge_host_allocator_pages

        result_queue.put(
            {
                "status": "ready",
                "worker": worker_index,
                "pid": os.getpid(),
                "snapshot": process_snapshot(),
                "torch_imported": "torch" in sys.modules,
                "torch_npu_imported": "torch_npu" in sys.modules,
                "malloc_conf": os.environ.get("MALLOC_CONF", ""),
            }
        )
        with spool_path.open("w+b", buffering=0) as spool:
            completed_tasks = 0
            while True:
                task = task_queue.get()
                if task is None:
                    break
                task_started = time.perf_counter()
                page_index = int(task["page_index"])
                request_id = str(task.get("request_id", page_index))
                active_spool = spool
                active_spool_path = spool_path
                page_spool = None
                if spool_mode == "per_page":
                    active_spool_path = (
                        Path(spool_root)
                        / f"worker_{worker_index:02d}_page_{page_index:012d}.bin"
                    )
                    page_spool = active_spool_path.open("w+b", buffering=0)
                    active_spool = page_spool
                rgb_descriptor = task.get("rgb_descriptor")
                if rgb_descriptor is None:
                    if task["rgb"] is None:
                        rgb, decode_timing = decode_page_rgb_cpu(
                            Path(task["path"])
                        )
                    else:
                        rgb = task["rgb"]
                        decode_timing = {
                            "page_file_read_s": 0.0,
                            "page_image_decode_s": 0.0,
                        }
                    rgb_mapping = None
                else:
                    rgb_mapping = np.memmap(
                        rgb_descriptor["path"],
                        mode="r+",
                        dtype=np.uint8,
                        offset=int(rgb_descriptor["offset"]),
                        shape=tuple(rgb_descriptor["shape"]),
                    )
                    rgb = rgb_mapping
                    decode_timing = {
                        "page_file_read_s": 0.0,
                        "page_image_decode_s": 0.0,
                    }
                payload, timing = _prepare_frontend_payload(
                    page_index=page_index,
                    path=Path(task["path"]),
                    rgb=rgb,
                    layout_result=task["layout_result"],
                    use_chart_recognition=use_chart_recognition,
                    tokenize_figure_of_table=tokenize_figure_of_table,
                    copy_source_crops=resize_chunk_size == 0,
                )
                kept_crops = []
                kept_block_ids = []
                rejected = 0
                for crop, block_id in zip(payload["crops"], payload["vlm_block_ids"]):
                    height, width = crop["image_rgb"].shape[:2]
                    processed_width, processed_height = _processed_size(width, height)
                    if _encoder_tokens(processed_width, processed_height) > cross_cache_length:
                        rejected += 1
                        continue
                    kept_crops.append(crop)
                    kept_block_ids.append(block_id)
                payload["crops"] = kept_crops
                payload["vlm_block_ids"] = kept_block_ids
                resize_started = time.perf_counter()
                resize_service_sum_s = 0.0
                chunk_size = resize_chunk_size or max(1, len(kept_crops))
                for start in range(0, len(kept_crops), chunk_size):
                    crop_chunk = kept_crops[start : start + chunk_size]
                    prepared = list(executor.map(_resize_crop, crop_chunk))
                    for crop, (pixels, source_size, service_s) in zip(
                        crop_chunk,
                        prepared,
                    ):
                        crop["source_image_size"] = [
                            int(value) for value in source_size
                        ]
                        crop["processed_pixel_values_descriptor"] = _append_pixels(
                            active_spool,
                            path=active_spool_path,
                            pixels=pixels,
                        )
                        crop["processed_image_size"] = [
                            int(pixels.shape[1]),
                            int(pixels.shape[0]),
                        ]
                        crop.pop("image_rgb", None)
                        resize_service_sum_s += service_s
                    del prepared
                resize_wall_s = time.perf_counter() - resize_started
                payload["cross_capacity_rejected_crops"] = rejected
                payload["request_id"] = request_id
                payload["pixel_spool_paths"] = sorted(
                    {
                        str(
                            crop["processed_pixel_values_descriptor"]["path"]
                        )
                        for crop in kept_crops
                    }
                )
                payload["started_at"] = float(task["started_at"])
                payload["frontend_timing_s"] = {
                    **decode_timing,
                    **timing,
                    "recognition_processor_resize_wall_s": resize_wall_s,
                    "recognition_processor_resize_service_sum_s": resize_service_sum_s,
                    "recognition_input_prepare_worker_s": resize_wall_s,
                }
                if page_spool is not None:
                    page_spool.close()
                result_queue.put(
                    {
                        "status": "ok",
                        "worker": worker_index,
                        "request_id": request_id,
                        "page_index": page_index,
                        "payload": payload,
                        "worker_page_s": time.perf_counter() - task_started,
                    }
                )
                completed_tasks += 1
                if rgb_mapping is not None:
                    del rgb, rgb_mapping
                    Path(rgb_descriptor["path"]).unlink()
                else:
                    del rgb
                del payload
                if completed_tasks % 16 == 0:
                    gc.collect()
                    purge_host_allocator_pages()
        executor.shutdown(wait=True)
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "worker": worker_index,
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


@dataclass
class CpuCropPrepareSummary:
    page_count: int
    wall_s: float
    worker_page_s: float
    worker_pss_bytes: int
    rejected_crops: int
    crop_count: int
    spool_bytes: int


class CpuCropPreparePool:
    """Bounded process pool that never imports Torch in its workers."""

    def __init__(
        self,
        *,
        workers: int,
        recognition_threads: int,
        resize_chunk_size: int = 0,
        malloc_conf: str = CPU_CROP_WORKER_MALLOC_CONF,
        openocr_root: Path,
        spool_root: Path,
        cross_cache_length: int,
        use_chart_recognition: bool = True,
        spool_mode: str = "append_worker",
    ) -> None:
        if workers < 1 or recognition_threads < 1 or resize_chunk_size < 0:
            raise ValueError("worker and thread counts must be positive")
        if spool_mode not in {"append_worker", "per_page"}:
            raise ValueError(f"unsupported crop spool mode: {spool_mode}")
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue(maxsize=workers * 2)
        # Results contain metadata and spool descriptors only. Keep this queue
        # unbounded so a worker can never deadlock behind a parent that is
        # temporarily blocked submitting the next full-resolution RGB page.
        self.result_queue = self.context.Queue()
        self.processes = [
            self.context.Process(
                target=_worker_main,
                args=(
                    index,
                    str(openocr_root),
                    str(spool_root),
                    recognition_threads,
                    resize_chunk_size,
                    cross_cache_length,
                    use_chart_recognition,
                    spool_mode,
                    self.task_queue,
                    self.result_queue,
                ),
                name=f"unirec-cpu-crop-{index}",
            )
            for index in range(workers)
        ]
        previous_malloc_conf = os.environ.get("MALLOC_CONF")
        try:
            if malloc_conf:
                os.environ["MALLOC_CONF"] = malloc_conf
            else:
                os.environ.pop("MALLOC_CONF", None)
            for process in self.processes:
                process.start()
            ready = [self._get(timeout=120.0) for _ in self.processes]
        finally:
            if previous_malloc_conf is None:
                os.environ.pop("MALLOC_CONF", None)
            else:
                os.environ["MALLOC_CONF"] = previous_malloc_conf
        errors = [item for item in ready if item.get("status") != "ready"]
        if errors:
            self.close()
            raise RuntimeError(f"CPU crop workers failed setup: {errors}")
        if any(item["torch_imported"] or item["torch_npu_imported"] for item in ready):
            self.close()
            raise RuntimeError("CPU crop worker imported Torch")
        worker_malloc_confs = {str(item["malloc_conf"]) for item in ready}
        if worker_malloc_confs != {malloc_conf}:
            self.close()
            raise RuntimeError(
                "CPU crop worker allocator mismatch: "
                f"expected {malloc_conf!r}, got {worker_malloc_confs}"
            )
        self.malloc_conf = malloc_conf
        self.spool_mode = spool_mode
        self.worker_pss_bytes = sum(
            int(item["snapshot"]["proc_bytes"]["pss"]) for item in ready
        )
        self.submitted = 0
        self.completed = 0
        self.closed = False
        self.started_at = time.perf_counter()
        self.worker_page_s = 0.0
        self.rejected_crops = 0
        self.crop_count = 0
        self.spool_bytes = 0

    def _get(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        try:
            item = self.result_queue.get(timeout=timeout)
        except queue.Empty as exception:
            raise TimeoutError("CPU crop workers produced no result") from exception
        if item.get("status") == "error":
            raise RuntimeError(
                f"CPU crop worker failed: {item.get('error')}\n{item.get('traceback')}"
            )
        return item

    def submit(
        self,
        *,
        page_index: int,
        path: Path,
        rgb: np.ndarray | None,
        rgb_descriptor: dict[str, Any] | None = None,
        layout_result: dict[str, Any],
        started_at: float,
        request_id: str | None = None,
    ) -> None:
        self.task_queue.put(
            {
                "request_id": (
                    str(page_index) if request_id is None else str(request_id)
                ),
                "page_index": page_index,
                "path": str(path),
                "rgb": rgb,
                "rgb_descriptor": rgb_descriptor,
                "layout_result": layout_result,
                "started_at": started_at,
            }
        )
        self.submitted += 1

    def receive(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        item = self._get(timeout=timeout)
        self.completed += 1
        self.worker_page_s += float(item["worker_page_s"])
        payload = item["payload"]
        self.rejected_crops += int(payload["cross_capacity_rejected_crops"])
        self.crop_count += len(payload["crops"])
        self.spool_bytes += sum(
            int(crop["processed_pixel_values_descriptor"]["nbytes"])
            for crop in payload["crops"]
        )
        return item

    def finish(self) -> tuple[list[dict[str, Any]], CpuCropPrepareSummary]:
        results = [self.receive() for _ in range(self.submitted - self.completed)]
        ordered = [item["payload"] for item in sorted(results, key=lambda row: row["page_index"])]
        summary = CpuCropPrepareSummary(
            page_count=self.completed,
            wall_s=time.perf_counter() - self.started_at,
            worker_page_s=self.worker_page_s,
            worker_pss_bytes=self.worker_pss_bytes,
            rejected_crops=self.rejected_crops,
            crop_count=self.crop_count,
            spool_bytes=self.spool_bytes,
        )
        return ordered, summary

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self.task_queue.close()
        self.result_queue.close()

    def __enter__(self) -> "CpuCropPreparePool":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _layout_owner_process_main(
    paths: list[str],
    model_path: str,
    cache_dir: str,
    device: str,
    lanes: int,
    batch_size: int,
    threshold: float,
    page_spool_root: str,
    result_queue: Any,
) -> None:
    """Heavy layout process; its exit is the graph-runtime memory barrier."""
    try:
        import torch_npu

        from host_memory_diagnostics import process_snapshot
        from layout_page_input import decode_page_rgb, materialize_layout_rgb
        from post_warmup_host_cleanup import purge_host_allocator_pages
        from shared_layout_owner import SharedLayoutOwner

        torch_npu.npu.set_compile_mode(jit_compile=False)
        owner = SharedLayoutOwner(
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            device=device,
            lanes=lanes,
            batch_size=batch_size,
            threshold=threshold,
        )
        result_queue.put(
            {
                "status": "layout_ready",
                "pid": os.getpid(),
                "snapshot": process_snapshot(),
                "chip": torch_npu.npu.get_device_name(0),
                "malloc_conf": os.environ.get("MALLOC_CONF", ""),
            }
        )
        started = time.perf_counter()
        chunk_size = lanes * batch_size

        def prepare_chunk(start: int) -> dict[str, Any]:
            chunk = paths[start : start + lanes * batch_size]
            page_started = []
            prepared = []
            descriptors = []
            timings = []
            for local_index, path in enumerate(chunk):
                page_started.append(time.perf_counter())
                rgb, timing = decode_page_rgb(Path(path))
                rgb = materialize_layout_rgb(rgb)
                descriptors.append(None)
                prepared.append(owner.prepare(rgb))
                timings.append(timing)
                del rgb
            if (start // max(1, lanes * batch_size) + 1) % 16 == 0:
                gc.collect()
                purge_host_allocator_pages()
            return {
                "start": start,
                "chunk": chunk,
                "page_started": page_started,
                "prepared": prepared,
                "descriptors": descriptors,
                "timings": timings,
            }

        starts = list(range(0, len(paths), chunk_size))
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="unirec-layout-prepare",
        ) as prepare_pool:
            future = prepare_pool.submit(prepare_chunk, starts[0]) if starts else None
            for position, start in enumerate(starts):
                assert future is not None
                chunk_data = future.result()
                future = (
                    prepare_pool.submit(prepare_chunk, starts[position + 1])
                    if position + 1 < len(starts)
                    else None
                )
                layouts = owner.predict_prepared(chunk_data["prepared"])
                for local_index, (path, descriptor, layout, timing) in enumerate(
                    zip(
                        chunk_data["chunk"],
                        chunk_data["descriptors"],
                        layouts,
                        chunk_data["timings"],
                    )
                ):
                    result_queue.put(
                        {
                            "status": "layout_page",
                            "page_index": start + local_index,
                            "path": path,
                            "rgb_descriptor": descriptor,
                            "layout_result": layout,
                            "started_at": chunk_data["page_started"][local_index],
                            "decode_timing_s": timing,
                        }
                    )
                del layouts, chunk_data
        result_queue.put(
            {
                "status": "layout_done",
                "pages": len(paths),
                "wall_s": time.perf_counter() - started,
                "owner_calls": owner.calls,
                "owner_wall_s": owner.wall_s,
                "snapshot": process_snapshot(),
            }
        )
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


def _persistent_layout_owner_process_main(
    model_path: str,
    cache_dir: str,
    device: str,
    lanes: int,
    batch_size: int,
    threshold: float,
    flush_timeout_s: float,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Own one hot layout runtime and accept requests until explicit close."""
    try:
        import torch_npu

        from host_memory_diagnostics import process_snapshot
        from layout_page_input import decode_page_rgb, materialize_layout_rgb
        from post_warmup_host_cleanup import purge_host_allocator_pages
        from shared_layout_owner import SharedLayoutOwner

        torch_npu.npu.set_compile_mode(jit_compile=False)
        owner = SharedLayoutOwner(
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            device=device,
            lanes=lanes,
            batch_size=batch_size,
            threshold=threshold,
        )
        result_queue.put(
            {
                "status": "layout_ready",
                "pid": os.getpid(),
                "snapshot": process_snapshot(),
                "chip": torch_npu.npu.get_device_name(0),
                "malloc_conf": os.environ.get("MALLOC_CONF", ""),
            }
        )
        started = time.perf_counter()
        completed = 0
        batch_calls = 0
        closing = False
        max_group_size = lanes * batch_size
        while not closing:
            first = task_queue.get()
            if first is None:
                break
            tasks = [first]
            deadline = time.perf_counter() + flush_timeout_s
            while len(tasks) < max_group_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    task = task_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if task is None:
                    closing = True
                    break
                tasks.append(task)

            prepared = []
            timings = []
            for task in tasks:
                rgb, timing = decode_page_rgb(Path(task["path"]))
                rgb = materialize_layout_rgb(rgb)
                prepared.append(owner.prepare(rgb))
                timings.append(timing)
                del rgb
            layouts = owner.predict_prepared(prepared)
            batch_calls += (len(tasks) + batch_size - 1) // batch_size
            for task, layout, timing in zip(tasks, layouts, timings):
                result_queue.put(
                    {
                        "status": "layout_page",
                        "request_id": str(task["request_id"]),
                        "page_index": int(task["page_index"]),
                        "path": str(task["path"]),
                        "rgb_descriptor": None,
                        "layout_result": layout,
                        "started_at": float(task["started_at"]),
                        "decode_timing_s": timing,
                        "ready_at": time.perf_counter(),
                    }
                )
                completed += 1
            del layouts, prepared, timings
            if completed % 32 == 0:
                gc.collect()
                purge_host_allocator_pages()
        result_queue.put(
            {
                "status": "layout_closed",
                "pages": completed,
                "batch_calls": batch_calls,
                "wall_s": time.perf_counter() - started,
                "owner_calls": owner.calls,
                "owner_wall_s": owner.wall_s,
                "snapshot": process_snapshot(),
            }
        )
    except BaseException as exception:
        result_queue.put(
            {
                "status": "error",
                "error": repr(exception),
                "traceback": traceback.format_exc(),
            }
        )


class PersistentSharedLayoutProcess:
    """Persistent B2 layout service with bounded request backpressure."""

    def __init__(
        self,
        *,
        model_path: Path,
        cache_dir: Path,
        device: str,
        lanes: int = 1,
        batch_size: int = 2,
        threshold: float = 0.5,
        flush_timeout_s: float = 0.005,
        queue_size: int = 32,
        malloc_conf: str = LAYOUT_OWNER_MALLOC_CONF,
    ) -> None:
        if lanes < 1 or batch_size < 1 or queue_size < 1:
            raise ValueError("layout lanes, batch size, and queue size must be positive")
        if flush_timeout_s < 0:
            raise ValueError("layout flush timeout cannot be negative")
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue(maxsize=queue_size)
        self.result_queue = self.context.Queue(maxsize=queue_size)
        self.submitted = 0
        self.received = 0
        self.close_requested = False
        self.closed = False
        self.summary: dict[str, Any] | None = None
        self.process = self.context.Process(
            target=_persistent_layout_owner_process_main,
            args=(
                str(model_path),
                str(cache_dir),
                device,
                lanes,
                batch_size,
                threshold,
                flush_timeout_s,
                self.task_queue,
                self.result_queue,
            ),
            name="unirec-persistent-layout-owner",
        )
        previous_malloc_conf = os.environ.get("MALLOC_CONF")
        try:
            if malloc_conf:
                os.environ["MALLOC_CONF"] = malloc_conf
            else:
                os.environ.pop("MALLOC_CONF", None)
            self.process.start()
            ready = self._get(timeout=300.0)
        finally:
            if previous_malloc_conf is None:
                os.environ.pop("MALLOC_CONF", None)
            else:
                os.environ["MALLOC_CONF"] = previous_malloc_conf
        if ready.get("status") != "layout_ready":
            self.close()
            raise RuntimeError(f"persistent layout owner was not ready: {ready}")
        if str(ready.get("malloc_conf", "")) != malloc_conf:
            self.close()
            raise RuntimeError(
                "persistent layout allocator mismatch: "
                f"expected {malloc_conf!r}, got {ready.get('malloc_conf')!r}"
            )
        self.ready = ready
        self.malloc_conf = malloc_conf

    def _get(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        try:
            item = self.result_queue.get(timeout=timeout)
        except queue.Empty as exception:
            raise TimeoutError("persistent layout owner produced no result") from exception
        if item.get("status") == "error":
            raise RuntimeError(
                f"persistent layout owner failed: {item.get('error')}\n"
                f"{item.get('traceback')}"
            )
        return item

    def submit(
        self,
        *,
        request_id: str,
        page_index: int,
        path: Path,
        started_at: float,
    ) -> None:
        if self.close_requested:
            raise RuntimeError("cannot submit after persistent layout close")
        self.task_queue.put(
            {
                "request_id": str(request_id),
                "page_index": int(page_index),
                "path": str(path),
                "started_at": float(started_at),
            }
        )
        self.submitted += 1

    def receive(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        item = self.receive_event(timeout=timeout)
        if item.get("status") != "layout_page":
            raise RuntimeError(f"unexpected persistent layout message: {item}")
        return item

    def receive_event(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        item = self._get(timeout=timeout)
        status = item.get("status")
        if status == "layout_page":
            self.received += 1
        elif status == "layout_closed":
            self.summary = item
        else:
            raise RuntimeError(f"unexpected persistent layout message: {item}")
        return item

    def request_close(self) -> None:
        if self.close_requested:
            return
        self.close_requested = True
        self.task_queue.put(None)

    def finish(self) -> dict[str, Any]:
        self.request_close()
        while self.received < self.submitted:
            self.receive()
        summary = self.summary or self._get()
        if summary.get("status") != "layout_closed":
            raise RuntimeError(f"persistent layout owner did not close: {summary}")
        self.summary = summary
        return summary

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        if self.process.is_alive():
            self.request_close()
            self.process.join(timeout=60.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        self.task_queue.close()
        self.result_queue.close()

    def __enter__(self) -> "PersistentSharedLayoutProcess":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class SharedLayoutProcess:
    """One short-lived NPU layout process with four cached executor lanes."""

    def __init__(
        self,
        *,
        paths: list[Path],
        model_path: Path,
        cache_dir: Path,
        device: str,
        lanes: int = 4,
        batch_size: int = 2,
        threshold: float = 0.5,
        page_spool_root: Path,
        malloc_conf: str = LAYOUT_OWNER_MALLOC_CONF,
    ) -> None:
        self.context = mp.get_context("spawn")
        self.result_queue = self.context.Queue(maxsize=max(8, lanes * batch_size * 2))
        self.process = self.context.Process(
            target=_layout_owner_process_main,
            args=(
                [str(path) for path in paths],
                str(model_path),
                str(cache_dir),
                device,
                lanes,
                batch_size,
                threshold,
                str(page_spool_root),
                self.result_queue,
            ),
            name="unirec-shared-layout-owner",
        )
        self.page_count = len(paths)
        previous_malloc_conf = os.environ.get("MALLOC_CONF")
        try:
            if malloc_conf:
                os.environ["MALLOC_CONF"] = malloc_conf
            else:
                os.environ.pop("MALLOC_CONF", None)
            self.process.start()
            ready = self._get(timeout=300.0)
        finally:
            if previous_malloc_conf is None:
                os.environ.pop("MALLOC_CONF", None)
            else:
                os.environ["MALLOC_CONF"] = previous_malloc_conf
        if ready.get("status") != "layout_ready":
            self.close()
            raise RuntimeError(f"layout owner did not become ready: {ready}")
        if str(ready.get("malloc_conf", "")) != malloc_conf:
            self.close()
            raise RuntimeError(
                "layout owner allocator mismatch: "
                f"expected {malloc_conf!r}, got {ready.get('malloc_conf')!r}"
            )
        self.malloc_conf = malloc_conf
        self.ready = ready
        self.summary: dict[str, Any] | None = None
        self.closed = False

    def _get(self, *, timeout: float = 1800.0) -> dict[str, Any]:
        try:
            item = self.result_queue.get(timeout=timeout)
        except queue.Empty as exception:
            raise TimeoutError("shared layout owner produced no result") from exception
        if item.get("status") == "error":
            raise RuntimeError(
                f"shared layout owner failed: {item.get('error')}\n{item.get('traceback')}"
            )
        return item

    def iter_pages(self) -> Any:
        received = 0
        while received < self.page_count:
            item = self._get()
            if item.get("status") != "layout_page":
                raise RuntimeError(f"unexpected layout owner message: {item}")
            received += 1
            yield item
        summary = self._get()
        if summary.get("status") != "layout_done":
            raise RuntimeError(f"layout owner did not finish cleanly: {summary}")
        self.summary = summary

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        self.process.join(timeout=60.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.exitcode not in {0, None}:
            raise RuntimeError(
                f"shared layout owner exited with {self.process.exitcode}"
            )
        self.result_queue.close()

    def __enter__(self) -> "SharedLayoutProcess":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
