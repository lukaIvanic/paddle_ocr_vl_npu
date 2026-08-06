"""Persistent dynamic B1 PP-DocLayoutV2 process pool.

The coordinator sends only page indices and file paths.  Every spawned process
owns one complete layout model/runtime.  Workers draw from one shared queue, so
no worker is tied to a slow static shard.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kornia_rs.image import Image as KorniaImage
from torchvision.io import ImageReadMode, decode_image

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _worker_main(
    worker_index: int,
    model_path: str,
    cache_dir: str,
    threshold: float,
    execution: str,
    warmup_path: str,
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
        result_queue.put({"status": "ready", "worker": worker_index})
        while True:
            task = task_queue.get()
            if task is None:
                return
            run_id, page_index, path_string = task
            path = Path(path_string)
            started = time.perf_counter()
            rgb, decode_timing = _decode_rgb(path)
            detector_started = time.perf_counter()
            result = runtime([rgb[..., ::-1]], threshold=threshold)[0]
            detector_s = time.perf_counter() - detector_started
            result_queue.put(
                {
                    "status": "ok",
                    "worker": worker_index,
                    "run_id": run_id,
                    "page_index": page_index,
                    "path": path_string,
                    "result": result,
                    "timing": {
                        **decode_timing,
                        "detector_call_s": detector_s,
                        "worker_page_s": time.perf_counter() - started,
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
        timeout_s: float = 1800.0,
    ) -> None:
        if worker_count < 1:
            raise ValueError("layout process worker count must be positive")
        if not warmup_paths:
            raise ValueError("layout process pool requires at least one warmup page")
        self.worker_count = worker_count
        self.timeout_s = timeout_s
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
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
        }
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
            worker_busy_s[worker_index] += float(timing["worker_page_s"])
            stage_s["worker_file_read_sum_s"] += float(timing["file_read_s"])
            stage_s["worker_direct_rgb_decode_sum_s"] += float(
                timing["direct_rgb_decode_s"]
            )
            stage_s["worker_detector_call_sum_s"] += float(
                timing["detector_call_s"]
            )
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
            "scheduling": "dynamic_shared_filepath_queue",
            "layout_batch_size": 1,
        }
        if any(result is None for result in results):
            raise RuntimeError("layout process pool returned incomplete results")
        return [result for result in results if result is not None], summary

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
