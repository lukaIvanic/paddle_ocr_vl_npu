"""Persistent streamed vision, text-prefill, and decode owner for UniRec."""

from __future__ import annotations

from collections import deque
from queue import Queue
from threading import Condition, Event, Thread
import sys
import time
import traceback
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
from vision_bucket_presets import plan_canvas_bucket_calls


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
        on_error: Callable[[BaseException], None] | None = None,
        vision_record_budget: int = 128,
        vision_max_calls_per_key: int = 64,
        vision_queue_size: int = 16,
        text_queue_size: int = 8,
        ready_queue_size: int = 128,
    ) -> None:
        if vision_record_budget < 1 or vision_max_calls_per_key < 1:
            raise ValueError("vision scheduling budgets must be positive")
        self.runner = runner
        self.vision_owner = vision_owner
        self.decoder = decoder
        self.text_runtime = text_runtime
        self.device = device
        self.on_page_complete = on_page_complete
        self.response_builder = response_builder
        self.on_error = on_error
        self.vision_record_budget = int(vision_record_budget)
        self.vision_max_calls_per_key = int(vision_max_calls_per_key)
        self.vision_queue: PersistentReadyQueue[dict[str, Any]] = (
            PersistentReadyQueue(maxsize=vision_queue_size)
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
        self._registered_request_ids: set[str] = set()
        self._accepted_request_ids: set[str] = set()
        self._upstream_completed_request_ids: set[str] = set()
        self._page_request_ids: dict[int, str] = {}
        self._pages: dict[int, Any] = {}
        self._remaining_crops: dict[int, int] = {}
        self._metrics = self._new_metrics()

        self.text_stream = torch.npu.Stream(device=torch.device(device))

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
            "vision_dispatches": 0,
            "vision_pages": 0,
            "vision_crops": 0,
            "vision_max_pending_records": 0,
            "vision_wall_s": 0.0,
            "vision_graph_wall_s": 0.0,
            "vision_spool_bytes_released": 0,
            "text_groups": 0,
            "text_crops": 0,
            "text_wall_s": 0.0,
            "decode_iterations": 0,
            "decode_raw_token_slots": 0,
            "decode_effective_tokens": 0,
            "decode_wall_s": 0.0,
        }

    def _record_error(self, exception: BaseException) -> None:
        first_error = False
        with self._condition:
            if self._error is None:
                self._error = exception
                first_error = True
            self._condition.notify_all()
        if first_error:
            print(
                "UNIREC_SERVING_NPU_ERROR "
                f"type={type(exception).__name__} error={exception!r}",
                flush=True,
            )
            traceback.print_exception(exception)
            sys.stderr.flush()
            if self.on_error is not None:
                self.on_error(exception)
        try:
            self.ready_queue.close()
        except BaseException:
            pass
        try:
            self.vision_queue.close()
        except BaseException:
            pass

    def register_request(self, request_id: str) -> None:
        """Tell both NPU schedulers that one page can still publish work."""
        identifier = str(request_id)
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot register after NPU pipeline close")
            if identifier in self._registered_request_ids:
                raise RuntimeError(f"duplicate service request id {identifier}")
            self._registered_request_ids.add(identifier)
        try:
            self.vision_queue.register_upstream()
            self.ready_queue.register_upstream()
        except BaseException:
            with self._condition:
                self._registered_request_ids.discard(identifier)
            raise

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request that failed before its frontend payload existed."""
        identifier = str(request_id)
        with self._condition:
            if identifier not in self._registered_request_ids:
                return
            if identifier in self._accepted_request_ids:
                return
            self._registered_request_ids.remove(identifier)
        self.vision_queue.complete_upstream()
        self.ready_queue.complete_upstream()

    def _complete_upstream_request(self, request_id: str) -> None:
        identifier = str(request_id)
        with self._condition:
            if identifier in self._upstream_completed_request_ids:
                raise RuntimeError(
                    f"request upstream completed twice: {identifier}"
                )
            self._upstream_completed_request_ids.add(identifier)
        self.ready_queue.complete_upstream()

    def _retire_request(self, request_id: str) -> None:
        identifier = str(request_id)
        with self._condition:
            self._registered_request_ids.discard(identifier)
            self._accepted_request_ids.discard(identifier)
            self._upstream_completed_request_ids.discard(identifier)

    def submit(self, payload: dict[str, Any]) -> None:
        page_index = int(payload["page_index"])
        request_id = str(payload["request_id"])
        with self._condition:
            registered = request_id in self._registered_request_ids
        if not registered:
            self.register_request(request_id)
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot submit after NPU pipeline close")
            if self._error is not None:
                raise RuntimeError("persistent NPU pipeline failed") from self._error
            if page_index in self._page_request_ids:
                raise RuntimeError(f"duplicate service page index {page_index}")
            if request_id in self._accepted_request_ids:
                raise RuntimeError(f"NPU request accepted twice: {request_id}")
            self._page_request_ids[page_index] = request_id
            self._accepted_request_ids.add(request_id)
            self._submitted_pages += 1
            self._submitted_crops += len(payload["crops"])
        self.vision_queue.put(payload)
        self.vision_queue.complete_upstream()

    def _finish_empty_page(self, page: Any) -> None:
        page_index = int(page.page_index)
        request_id = self._page_request_ids[page_index]
        self._complete_upstream_request(request_id)
        response = self.response_builder(page)
        request_id = self._page_request_ids.pop(page_index)
        self._pages.pop(page_index, None)
        self.on_page_complete(request_id, response)
        self._retire_request(request_id)
        with self._condition:
            self._completed_pages += 1
            self._condition.notify_all()

    def _vision_loop(self) -> None:
        try:
            torch_npu.npu.set_device(self.device)
            runtime = self.vision_owner.runtime
            pending_by_canvas: dict[tuple[int, int], deque[dict[str, Any]]] = {
                canvas: deque() for canvas in runtime.specs_by_canvas
            }
            fallbacks: deque[dict[str, Any]] = deque()
            record_by_source: dict[int, dict[str, Any]] = {}
            page_payloads: dict[int, dict[str, Any]] = {}
            page_encoded: dict[int, dict[str, Any]] = {}
            page_vision_remaining: dict[int, int] = {}
            next_source_index = 0
            previous_dispatch_keys: set[str] = set()
            closing = False

            def pending_count() -> int:
                return len(fallbacks) + sum(
                    len(records) for records in pending_by_canvas.values()
                )

            def ingest(payload: dict[str, Any]) -> None:
                nonlocal next_source_index
                pages, records = _payload_to_pages([payload])
                if len(pages) != 1:
                    raise RuntimeError(
                        f"one service payload produced {len(pages)} pages"
                    )
                page = pages[0]
                page_index = int(page.page_index)
                self._pages[page_index] = page
                self._metrics["vision_pages"] += 1
                self._metrics["vision_crops"] += len(records)
                if not records:
                    self._metrics["vision_spool_bytes_released"] += (
                        release_page_pixel_spools(payload)
                    )
                    self._finish_empty_page(page)
                    return
                page_payloads[page_index] = payload
                page_encoded[page_index] = {}
                page_vision_remaining[page_index] = len(records)
                for record in records:
                    record["source_index"] = next_source_index
                    record_by_source[next_source_index] = record
                    next_source_index += 1
                    width, height = (
                        int(record["processed_image_size"][0]),
                        int(record["processed_image_size"][1]),
                    )
                    canvas = runtime.select_canvas(width, height)
                    if canvas is None:
                        fallbacks.append(record)
                    else:
                        pending_by_canvas[canvas].append(record)
                self._metrics["vision_max_pending_records"] = max(
                    self._metrics["vision_max_pending_records"],
                    pending_count(),
                )

            def drain_ready_payloads() -> None:
                nonlocal closing
                while not closing:
                    pulled = self.vision_queue.pull(wait=False)
                    if pulled.closed:
                        closing = True
                        return
                    if pulled.idle:
                        return
                    assert pulled.item is not None
                    ingest(pulled.item)

            def candidate_calls(
                *,
                flush_partials: bool,
            ) -> list[tuple[str, tuple[int, int], Any, int]]:
                candidates = []
                resident = set(self.vision_owner._resident_lane_by_key)
                for canvas, records in pending_by_canvas.items():
                    if not records:
                        continue
                    specs = runtime.specs_by_canvas[canvas]
                    minimum_batch = min(spec.batch_size for spec in specs)
                    if len(records) < minimum_batch and not flush_partials:
                        continue
                    plan = plan_canvas_bucket_calls(specs, len(records))
                    first_spec = plan[0]
                    call_count = 0
                    rows = 0
                    remaining = len(records)
                    for spec in plan:
                        if spec.key != first_spec.key:
                            break
                        if call_count >= self.vision_max_calls_per_key:
                            break
                        call_count += 1
                        real_rows = min(spec.batch_size, remaining)
                        rows += real_rows
                        remaining -= real_rows
                    oldest = int(records[0]["source_index"])
                    candidates.append(
                        (
                            first_spec.key,
                            canvas,
                            first_spec,
                            rows,
                            first_spec.key in resident,
                            first_spec.key in previous_dispatch_keys,
                            oldest,
                        )
                    )
                candidates.sort(
                    key=lambda item: (
                        item[5],
                        not item[4],
                        item[6],
                        item[0],
                    )
                )
                return [item[:4] for item in candidates[: self.vision_owner.lanes]]

            def complete_encoded_batch(batch: list[Any]) -> None:
                ready_pages = []
                for item in batch:
                    source_index = int(item.source_index)
                    record = record_by_source.pop(source_index)
                    crop = record["crop"]
                    page_index = int(crop.page_index)
                    page_encoded[page_index][crop.request_id] = item
                    page_vision_remaining[page_index] -= 1
                    if page_vision_remaining[page_index] == 0:
                        ready_pages.append(page_index)
                for page_index in sorted(ready_pages):
                    page = self._pages[page_index]
                    items = page_encoded.pop(page_index)
                    del page_vision_remaining[page_index]
                    payload = page_payloads.pop(page_index)
                    self._metrics["vision_spool_bytes_released"] += (
                        release_page_pixel_spools(payload)
                    )
                    self.text_queue.put((page, items))

            while True:
                drain_ready_payloads()
                outstanding_upstream = self.vision_queue.upstream_pending
                total_pending = pending_count()
                flush_partials = closing or outstanding_upstream == 0
                have_budget = total_pending >= self.vision_record_budget
                calls = candidate_calls(flush_partials=flush_partials)

                selected_records: list[dict[str, Any]] = []
                if calls and (have_budget or flush_partials):
                    selected_keys = set()
                    for key, canvas, _spec, rows in calls:
                        selected_keys.add(key)
                        for _ in range(rows):
                            selected_records.append(
                                pending_by_canvas[canvas].popleft()
                            )
                    previous_dispatch_keys = selected_keys
                elif fallbacks and (have_budget or flush_partials):
                    selected_records.append(fallbacks.popleft())
                    previous_dispatch_keys = set()

                if selected_records:
                    started = time.perf_counter()
                    _unused, summary = self.vision_owner.encode(
                        selected_records,
                        on_encoded_batch=complete_encoded_batch,
                        retain_outputs=False,
                        retain_loaded_graphs=True,
                    )
                    vision_s = time.perf_counter() - started
                    self._metrics["vision_dispatches"] += 1
                    self._metrics["vision_windows"] += 1
                    self._metrics["vision_wall_s"] += vision_s
                    self._metrics["vision_graph_wall_s"] += float(
                        summary["wall_s"]
                    )
                    continue

                if closing:
                    if total_pending:
                        raise RuntimeError(
                            "closed vision scheduler cannot dispatch "
                            f"{total_pending} pending records"
                        )
                    break
                pulled = self.vision_queue.pull(wait=True)
                if pulled.closed:
                    closing = True
                else:
                    assert pulled.item is not None
                    ingest(pulled.item)

            if record_by_source or page_vision_remaining:
                raise RuntimeError(
                    "vision scheduler closed with incomplete pages: "
                    f"records={len(record_by_source)} "
                    f"pages={len(page_vision_remaining)}"
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
                ready_event = torch.npu.Event()
                ready_event.record(self.text_stream)
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
                    ready_event=ready_event,
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
        self._complete_upstream_request(self._page_request_ids[page_index])

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
        self._retire_request(request_id)
        with self._condition:
            self._completed_pages += 1
            self._condition.notify_all()

    def _decode_loop(self) -> None:
        try:
            torch_npu.npu.set_device(self.device)

            def record_step(report: dict[str, Any]) -> None:
                with self._condition:
                    self._metrics["decode_iterations"] += 1
                    self._metrics["decode_raw_token_slots"] += (
                        self.decoder.batch_size
                    )
                    self._metrics["decode_effective_tokens"] += int(
                        report["active_count"]
                    )
                    self._metrics["decode_wall_s"] += float(
                        report["decode_step_s"]
                    )

            self.decode_summary = self.decoder.run(
                self.ready_queue,
                on_complete=self._complete_crop,
                graph_warmup_passes=0,
                partial_batch_wait_s=0.0,
                on_step=record_step,
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
                "frontend_requests_pending": self.vision_queue.upstream_pending,
                "decode_upstream_pages_pending": self.ready_queue.upstream_pending,
                "failed": self._error is not None,
            }

    def request_close(self) -> None:
        with self._condition:
            if self._close_requested:
                return
            self._close_requested = True
        self.vision_queue.close()

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
