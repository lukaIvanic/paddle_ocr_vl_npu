"""Torch-free W4/T8 recognition-crop preparation with disk-backed pixels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    cross_cache_length: int,
    use_chart_recognition: bool,
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

        result_queue.put(
            {
                "status": "ready",
                "worker": worker_index,
                "pid": os.getpid(),
                "snapshot": process_snapshot(),
                "torch_imported": "torch" in sys.modules,
                "torch_npu_imported": "torch_npu" in sys.modules,
            }
        )
        with spool_path.open("w+b", buffering=0) as spool:
            while True:
                task = task_queue.get()
                if task is None:
                    break
                task_started = time.perf_counter()
                page_index = int(task["page_index"])
                payload, timing = _prepare_frontend_payload(
                    page_index=page_index,
                    path=Path(task["path"]),
                    rgb=task["rgb"],
                    layout_result=task["layout_result"],
                    use_chart_recognition=use_chart_recognition,
                    tokenize_figure_of_table=tokenize_figure_of_table,
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
                prepared = list(executor.map(_resize_crop, kept_crops))
                resize_wall_s = time.perf_counter() - resize_started
                resize_service_sum_s = 0.0
                for crop, (pixels, source_size, service_s) in zip(kept_crops, prepared):
                    crop["source_image_size"] = [int(value) for value in source_size]
                    crop["processed_pixel_values_descriptor"] = _append_pixels(
                        spool,
                        path=spool_path,
                        pixels=pixels,
                    )
                    crop["processed_image_size"] = [
                        int(pixels.shape[1]),
                        int(pixels.shape[0]),
                    ]
                    crop.pop("image_rgb", None)
                    resize_service_sum_s += service_s
                payload["cross_capacity_rejected_crops"] = rejected
                payload["started_at"] = float(task["started_at"])
                payload["frontend_timing_s"] = {
                    **timing,
                    "recognition_processor_resize_wall_s": resize_wall_s,
                    "recognition_processor_resize_service_sum_s": resize_service_sum_s,
                    "recognition_input_prepare_worker_s": resize_wall_s,
                }
                result_queue.put(
                    {
                        "status": "ok",
                        "worker": worker_index,
                        "page_index": page_index,
                        "payload": payload,
                        "worker_page_s": time.perf_counter() - task_started,
                    }
                )
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
        openocr_root: Path,
        spool_root: Path,
        cross_cache_length: int,
        use_chart_recognition: bool = True,
    ) -> None:
        if workers < 1 or recognition_threads < 1:
            raise ValueError("worker and thread counts must be positive")
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
                    cross_cache_length,
                    use_chart_recognition,
                    self.task_queue,
                    self.result_queue,
                ),
                name=f"unirec-cpu-crop-{index}",
            )
            for index in range(workers)
        ]
        for process in self.processes:
            process.start()
        ready = [self._get(timeout=120.0) for _ in self.processes]
        errors = [item for item in ready if item.get("status") != "ready"]
        if errors:
            self.close()
            raise RuntimeError(f"CPU crop workers failed setup: {errors}")
        if any(item["torch_imported"] or item["torch_npu_imported"] for item in ready):
            self.close()
            raise RuntimeError("CPU crop worker imported Torch")
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
        rgb: np.ndarray,
        layout_result: dict[str, Any],
        started_at: float,
    ) -> None:
        self.task_queue.put(
            {
                "page_index": page_index,
                "path": str(path),
                "rgb": rgb,
                "layout_result": layout_result,
                "started_at": started_at,
            }
        )
        self.submitted += 1

    def receive(self) -> dict[str, Any]:
        item = self._get()
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
