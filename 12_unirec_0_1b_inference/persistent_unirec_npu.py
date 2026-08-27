"""Persistent streamed vision, text-prefill, and decode owner for UniRec."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Condition, Event, Thread
import time
from typing import Any, Callable

import torch
import torch_npu

from continuous_unirec import (
    ContinuousCompletedItem,
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
    ContinuousWorkerPrefilledItem,
)
from low_memory_frontend_pool import release_page_pixel_spools
from persistent_ready_queue import PersistentReadyQueue
from run_low_memory_unirec import _payload_to_pages
from run_opendoc_batched_unirec import iter_greedy_text_packs


class PersistentUniRecNpuPipeline:
    """Keep one UniRec model hot while independent page requests arrive."""

    def __init__(
        self,
        *,
        runner: Any,
        vision_owner: Any,
        decoder: ContinuousUniRecDecoder,
        text_runtime: Any,
        device: str,
        on_page_complete: Callable[[str, dict[str, Any]], None],
        response_builder: Callable[[Any], dict[str, Any]],
        vision_page_lookahead: int = 4,
        vision_flush_timeout_s: float = 0.005,
        vision_queue_size: int = 16,
        text_queue_size: int = 8,
        ready_queue_size: int = 128,
        decode_partial_batch_wait_s: float = 0.005,
        decode_graph_warmup_passes: int = 2,
        on_decode_graph_warmup_complete: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
    ) -> None:
        if vision_page_lookahead < 1:
            raise ValueError("vision page lookahead must be positive")
        if vision_flush_timeout_s < 0 or decode_partial_batch_wait_s < 0:
            raise ValueError("service flush timeouts cannot be negative")
        self.runner = runner
        self.vision_owner = vision_owner
        self.decoder = decoder
        self.text_runtime = text_runtime
        self.device = device
        self.on_page_complete = on_page_complete
        self.response_builder = response_builder
        self.vision_page_lookahead = int(vision_page_lookahead)
        self.vision_flush_timeout_s = float(vision_flush_timeout_s)
        self.decode_partial_batch_wait_s = float(decode_partial_batch_wait_s)
        self.decode_graph_warmup_passes = int(decode_graph_warmup_passes)
        self.on_decode_graph_warmup_complete = (
            on_decode_graph_warmup_complete
        )
        self.vision_queue: Queue[dict[str, Any] | None] = Queue(
            maxsize=vision_queue_size
        )
        self.text_queue: Queue[tuple[Any, dict[str, Any]] | None] = Queue(
            maxsize=text_queue_size
        )
        self.ready_queue: PersistentReadyQueue[ContinuousReadyItem] = (
            PersistentReadyQueue(maxsize=ready_queue_size)
        )
        self._condition = Condition()
        self._decode_idle = Event()
        self._decode_idle.set()
        self._submitted_pages = 0
        self._completed_pages = 0
        self._submitted_crops = 0
        self._completed_crops = 0
        self._close_requested = False
        self._error: BaseException | None = None
        self._page_request_ids: dict[int, str] = {}
        self._pages: dict[int, Any] = {}
        self._remaining_crops: dict[int, int] = {}
        self._metrics = self._new_metrics()

        self.text_stream = torch.npu.Stream(device=torch.device(device))
        text_input = torch.zeros(
            (1, self.text_runtime.bucket, runner.config.d_model),
            dtype=runner.dtype,
            device=torch.device(device),
        )
        with torch.inference_mode(), torch.npu.stream(self.text_stream):
            self.text_runtime.compiled(text_input)
        self.text_stream.synchronize()
        self.text_runtime._first_call = False
        del text_input

        self._vision_thread = Thread(
            target=self._vision_loop,
            name="unirec-serving-vision",
            daemon=True,
        )
        self._text_thread = Thread(
            target=self._text_loop,
            name="unirec-serving-text-prefill",
            daemon=True,
        )
        self._decode_thread = Thread(
            target=self._decode_loop,
            name="unirec-serving-decode",
            daemon=True,
        )
        self._vision_thread.start()
        self._text_thread.start()
        self._decode_thread.start()

    @staticmethod
    def _new_metrics() -> dict[str, Any]:
        return {
            "started_at": time.perf_counter(),
            "vision_windows": 0,
            "vision_pages": 0,
            "vision_crops": 0,
            "vision_wall_s": 0.0,
            "vision_graph_wall_s": 0.0,
            "vision_spool_bytes_released": 0,
            "text_groups": 0,
            "text_crops": 0,
            "text_wall_s": 0.0,
        }

    def _record_error(self, exception: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = exception
            self._condition.notify_all()
        try:
            self.ready_queue.close()
        except BaseException:
            pass

    def submit(self, payload: dict[str, Any]) -> None:
        page_index = int(payload["page_index"])
        request_id = str(payload["request_id"])
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot submit after NPU pipeline close")
            if self._error is not None:
                raise RuntimeError("persistent NPU pipeline failed") from self._error
            if page_index in self._page_request_ids:
                raise RuntimeError(f"duplicate service page index {page_index}")
            self._page_request_ids[page_index] = request_id
            self._submitted_pages += 1
            self._submitted_crops += len(payload["crops"])
        self.vision_queue.put(payload)

    def _next_vision_window(
        self,
        first: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        payloads = [first]
        deadline = time.perf_counter() + self.vision_flush_timeout_s
        closing = False
        while len(payloads) < self.vision_page_lookahead:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                payload = self.vision_queue.get(timeout=remaining)
            except Empty:
                break
            if payload is None:
                closing = True
                break
            payloads.append(payload)
        return payloads, closing

    def _finish_empty_page(self, page: Any) -> None:
        page_index = int(page.page_index)
        response = self.response_builder(page)
        request_id = self._page_request_ids.pop(page_index)
        self._pages.pop(page_index, None)
        self.on_page_complete(request_id, response)
        with self._condition:
            self._completed_pages += 1
            self._condition.notify_all()

    def _vision_loop(self) -> None:
        try:
            torch_npu.npu.set_device(self.device)
            closing = False
            while not closing:
                first = self.vision_queue.get()
                if first is None:
                    break
                payloads, closing = self._next_vision_window(first)
                pages, records = _payload_to_pages(payloads)
                pages_by_index = {
                    int(page.page_index): page for page in pages
                }
                self._pages.update(pages_by_index)
                for page in pages:
                    if not page.crops:
                        self._finish_empty_page(page)
                if records:
                    started = time.perf_counter()
                    encoded, summary = self.vision_owner.encode(
                        records,
                        retain_loaded_graphs=True,
                    )
                    vision_s = time.perf_counter() - started
                    by_page: dict[int, dict[str, Any]] = {
                        int(page.page_index): {}
                        for page in pages
                        if page.crops
                    }
                    for record, item in zip(records, encoded):
                        crop = record["crop"]
                        by_page[int(crop.page_index)][crop.request_id] = item
                    for payload in payloads:
                        self._metrics["vision_spool_bytes_released"] += (
                            release_page_pixel_spools(payload)
                        )
                    for page_index, items in by_page.items():
                        page = pages_by_index[page_index]
                        if len(items) != len(page.crops):
                            raise RuntimeError(
                                "vision page output mismatch: "
                                f"page={page_index} outputs={len(items)} "
                                f"crops={len(page.crops)}"
                            )
                        self.text_queue.put((page, items))
                    self._metrics["vision_windows"] += 1
                    self._metrics["vision_pages"] += len(pages)
                    self._metrics["vision_crops"] += len(records)
                    self._metrics["vision_wall_s"] += vision_s
                    self._metrics["vision_graph_wall_s"] += float(
                        summary["wall_s"]
                    )
                else:
                    for payload in payloads:
                        self._metrics["vision_spool_bytes_released"] += (
                            release_page_pixel_spools(payload)
                        )
            self.text_queue.put(None)
        except BaseException as exception:
            self._record_error(exception)
            self.text_queue.put(None)

    def _prefill_page(self, page: Any, encoded_items: dict[str, Any]) -> None:
        page_index = int(page.page_index)
        self._remaining_crops[page_index] = len(page.crops)
        started = time.perf_counter()
        groups = iter_greedy_text_packs(iter(page.crops), runner=self.runner)
        for use_packed, crop_group in groups:
            if not use_packed:
                raise RuntimeError(
                    "accuracy-safe serving path encountered text prefill "
                    f"fallback: {crop_group[0].request_id}"
                )
            encoded_group = [
                (
                    encoded_items[crop.request_id].hidden_states,
                    encoded_items[crop.request_id].prep,
                )
                for crop in crop_group
            ]
            with torch.inference_mode(), torch.npu.stream(self.text_stream):
                items = self.runner.prefill_encoder_hidden_states_packed_for_cohort(
                    encoded_group,
                    profile_device_stages=False,
                    decode_ready=False,
                )
                exports = []
                for item in items:
                    actual_length = int(
                        item.kv_cache.actual_cross_attention_length or 0
                    )
                    if actual_length <= 0:
                        raise RuntimeError("text prefill produced empty cross-KV")
                    packed = torch.stack(
                        tuple(
                            tensor[:, :, :actual_length, :]
                            for tensor in (
                                *item.kv_cache.cross_key_cache,
                                *item.kv_cache.cross_value_cache,
                            )
                        ),
                        dim=0,
                    ).contiguous()
                    exports.append((packed, actual_length))
            self.text_stream.synchronize()
            for crop, item, (packed, actual_length) in zip(
                crop_group,
                items,
                exports,
            ):
                prefilled = ContinuousWorkerPrefilledItem(
                    packed_cross_kv=packed,
                    prep=dict(item.prep),
                    prefill_s=float(item.prefill_s),
                    actual_cross_attention_length=actual_length,
                    prefill_device_stage_s=item.prefill_device_stage_s,
                    text_prefill_execution=str(item.text_prefill_execution),
                    text_prefill_real_source_tokens=int(
                        item.text_prefill_real_source_tokens or actual_length
                    ),
                    text_prefill_physical_source_tokens=int(
                        item.text_prefill_physical_source_tokens
                        or item.text_prefill_real_source_tokens
                        or actual_length
                    ),
                )

                def release(prefilled: Any = prefilled) -> None:
                    prefilled.packed_cross_kv = None

                self._decode_idle.clear()
                self.ready_queue.put(
                    ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=prefilled,
                        on_admitted=release,
                    )
                )
                self._metrics["text_crops"] += 1
            self._metrics["text_groups"] += 1
            del items, exports, encoded_group
        self._metrics["text_wall_s"] += time.perf_counter() - started

    def _text_loop(self) -> None:
        try:
            torch_npu.npu.set_device(self.device)
            while True:
                value = self.text_queue.get()
                if value is None:
                    break
                page, encoded_items = value
                self._prefill_page(page, encoded_items)
            self.ready_queue.close()
        except BaseException as exception:
            self._record_error(exception)

    def _complete_crop(self, completed: ContinuousCompletedItem) -> None:
        crop = completed.payload
        crop.result = completed.result
        page_index = int(crop.page_index)
        self._remaining_crops[page_index] -= 1
        with self._condition:
            self._completed_crops += 1
        if self._remaining_crops[page_index] != 0:
            return
        del self._remaining_crops[page_index]
        page = self._pages.pop(page_index)
        response = self.response_builder(page)
        request_id = self._page_request_ids.pop(page_index)
        self.on_page_complete(request_id, response)
        with self._condition:
            self._completed_pages += 1
            self._condition.notify_all()

    def _decode_loop(self) -> None:
        try:
            torch_npu.npu.set_device(self.device)
            self.decode_summary = self.decoder.run(
                self.ready_queue,
                on_complete=self._complete_crop,
                graph_warmup_passes=self.decode_graph_warmup_passes,
                partial_batch_wait_s=self.decode_partial_batch_wait_s,
                on_graph_warmup_complete=(
                    self.on_decode_graph_warmup_complete
                ),
                on_idle=self._decode_idle.set,
            )
        except BaseException as exception:
            self._record_error(exception)

    def wait_idle(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._completed_pages < self._submitted_pages:
                if self._error is not None:
                    raise RuntimeError("persistent NPU pipeline failed") from self._error
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "persistent NPU pipeline did not become idle: "
                        f"completed={self._completed_pages} "
                        f"submitted={self._submitted_pages}"
                    )
                self._condition.wait(timeout=remaining)
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._decode_idle.wait(timeout=remaining):
            raise TimeoutError("persistent decoder did not enter its idle wait")

    def reset_metrics(self) -> dict[str, Any]:
        self.wait_idle()
        previous = self.metrics()
        self._metrics = self._new_metrics()
        return previous

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._metrics,
                "wall_s": time.perf_counter() - self._metrics["started_at"],
                "submitted_pages": self._submitted_pages,
                "completed_pages": self._completed_pages,
                "submitted_crops": self._submitted_crops,
                "completed_crops": self._completed_crops,
                "vision_queue_depth": self.vision_queue.qsize(),
                "text_queue_depth": self.text_queue.qsize(),
                "ready_queue_depth": self.ready_queue.qsize(),
                "failed": self._error is not None,
            }

    def request_close(self) -> None:
        with self._condition:
            if self._close_requested:
                return
            self._close_requested = True
        self.vision_queue.put(None)

    def close(self) -> dict[str, Any]:
        self.request_close()
        for thread in (
            self._vision_thread,
            self._text_thread,
            self._decode_thread,
        ):
            thread.join(timeout=1800.0)
            if thread.is_alive():
                raise RuntimeError(f"NPU service thread did not stop: {thread.name}")
        if self._error is not None:
            raise RuntimeError("persistent NPU pipeline failed") from self._error
        self.vision_owner.close()
        return self.decode_summary
