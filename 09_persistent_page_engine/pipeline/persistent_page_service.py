"""Persistent full-page service with one scheduler-owned NPU execution lane.

CPU page input, detector postprocessing, crop construction, and recognizer
input preparation run in bounded background workers.  The decode scheduler
pulls this source only at its existing safe refill boundaries.  A pull either
returns an already-prefilled crop, executes one layout detector call, executes
one existing packed prefill group, or reports that only partial decode can make
progress.  Layout, recognition prefill, and decode therefore never compete on
independent compute streams.
"""

from __future__ import annotations

import queue
import json
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from paddleocr_vl.serving.continuous_decode import ReadyDecodeRequest
from paddleocr_vl.serving.engine import ContinuousRecognizer, CpuPreparedRecognition
from paddleocr_vl.serving.types import (
    ContinuousDecodeResult,
    RecognitionRequest,
    RecognitionResult,
)

from .layout_frontend import (
    OwnedLayoutFrontend,
    PreparedLayoutPage,
    TransferredLayoutPage,
)
from .layout_output import OwnedPageResult, assemble_page_blocks


@dataclass(frozen=True)
class PageSubmission:
    request_id: str
    image_path: Path
    submitted_at: float


class OpenPageSubmissionSource(Protocol):
    @property
    def closed(self) -> bool: ...

    def pull(self, *, block: bool) -> PageSubmission | None: ...


@dataclass
class _PageState:
    submission: PageSubmission
    prepared: PreparedLayoutPage
    remaining: int
    recognition: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _RequestOwner:
    page: _PageState
    block_index: int


@dataclass(frozen=True)
class _LayoutReady:
    submission: PageSubmission
    transferred: TransferredLayoutPage


class _UnifiedPageReadySource:
    """Produce decode-ready crops while arbitrating layout and prefill work."""

    def __init__(
        self,
        frontend: OwnedLayoutFrontend,
        recognizer: ContinuousRecognizer,
        pages: OpenPageSubmissionSource,
        *,
        min_pixels: int | None,
        max_pixels: int,
        text_max_pixels: int | None,
        table_max_pixels: int | None,
        text_crop_scale: float,
        emit_page: Callable[[str, OwnedPageResult, dict[str, float]], None],
        emit_page_error: Callable[[str, BaseException], None],
        input_workers: int = 4,
        postprocess_workers: int = 8,
        max_inflight_pages: int = 8,
    ) -> None:
        self.frontend = frontend
        self.recognizer = recognizer
        self.pages = pages
        self.min_pixels = min_pixels
        self.max_pixels = int(max_pixels)
        self.text_max_pixels = text_max_pixels
        self.table_max_pixels = table_max_pixels
        self.text_crop_scale = float(text_crop_scale)
        self.emit_page = emit_page
        self.emit_page_error = emit_page_error

        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._intake_done = False
        self._fatal_error: BaseException | None = None
        self._ordinal = 0
        self._input_outstanding = 0
        self._postprocess_outstanding = 0
        self._crop_outstanding = 0
        self._page_slots = threading.Semaphore(max_inflight_pages)
        self._layout_ready: queue.Queue[_LayoutReady] = queue.Queue(
            maxsize=max_inflight_pages
        )
        self._raw_crops: queue.Queue[RecognitionRequest | object] = queue.Queue(
            maxsize=max(256, recognizer.batch_size * 8)
        )
        self._prepared_inbox: queue.Queue[
            tuple[CpuPreparedRecognition, float]
        ] = queue.Queue(maxsize=recognizer.cpu_preprocess_max_pending)
        self._prepared: deque[tuple[CpuPreparedRecognition, float]] = deque()
        self._ready: deque[ReadyDecodeRequest] = deque()
        self._owners: dict[str, _RequestOwner] = {}
        self._owners_lock = threading.Lock()
        self._crop_sentinel = object()
        self._mode_counts: Counter[str] = Counter()
        self._mode_wall_s: defaultdict[str, float] = defaultdict(float)
        self._pages_completed = 0

        self._input_executor = ThreadPoolExecutor(
            max_workers=input_workers,
            thread_name_prefix="page-api-layout-input",
        )
        self._postprocess_executor = ThreadPoolExecutor(
            max_workers=postprocess_workers,
            thread_name_prefix="page-api-layout-postprocess",
        )
        self._intake_thread = threading.Thread(
            target=self._intake_loop,
            name="page-api-intake",
            daemon=True,
        )
        self._crop_thread = threading.Thread(
            target=self._crop_prepare_loop,
            name="page-api-crop-prepare",
            daemon=True,
        )
        self._intake_thread.start()
        self._crop_thread.start()

    @property
    def closed(self) -> bool:
        with self._condition:
            return bool(
                self._intake_done
                and self._input_outstanding == 0
                and self._postprocess_outstanding == 0
                and self._crop_outstanding == 0
                and self._layout_ready.empty()
                and self._raw_crops.empty()
                and self._prepared_inbox.empty()
                and not self._prepared
                and not self._ready
            )

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _set_fatal(self, exc: BaseException) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = exc
            self._condition.notify_all()

    def _input_stage(
        self,
        submission: PageSubmission,
        ordinal: int,
    ) -> _LayoutReady:
        decoded = self.frontend.decode_page(submission.image_path, ordinal)
        preprocessed = self.frontend.preprocess_decoded_page(decoded)
        transferred = self.frontend.transfer_preprocessed_page(preprocessed)
        return _LayoutReady(submission, transferred)

    def _input_done(
        self,
        submission: PageSubmission,
        future: Future[_LayoutReady],
    ) -> None:
        try:
            ready = future.result()
        except BaseException as exc:
            self._page_slots.release()
            self.emit_page_error(submission.request_id, exc)
        else:
            while not self._stop.is_set():
                try:
                    self._layout_ready.put(ready, timeout=0.1)
                    break
                except queue.Full:
                    continue
        finally:
            with self._condition:
                self._input_outstanding -= 1
                self._condition.notify_all()

    def _intake_loop(self) -> None:
        try:
            while not self._stop.is_set():
                submission = self.pages.pull(block=True)
                if submission is None:
                    if self.pages.closed:
                        break
                    continue
                while not self._stop.is_set():
                    if self._page_slots.acquire(timeout=0.1):
                        break
                if self._stop.is_set():
                    break
                with self._condition:
                    ordinal = self._ordinal
                    self._ordinal += 1
                    self._input_outstanding += 1
                future = self._input_executor.submit(
                    self._input_stage,
                    submission,
                    ordinal,
                )
                future.add_done_callback(
                    lambda completed, current=submission: self._input_done(
                        current,
                        completed,
                    )
                )
        except BaseException as exc:
            self._set_fatal(exc)
        finally:
            with self._condition:
                self._intake_done = True
                self._condition.notify_all()

    def _crop_prepare_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = self._raw_crops.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is self._crop_sentinel:
                    break
                if not isinstance(item, RecognitionRequest):
                    raise TypeError(f"unexpected crop request: {type(item)!r}")
                with self._condition:
                    self._crop_outstanding += 1
                submitted_at = time.perf_counter()
                try:
                    prepared = self.recognizer._prepare_cpu(item, submitted_at)
                    while not self._stop.is_set():
                        try:
                            self._prepared_inbox.put((prepared, 0.0), timeout=0.1)
                            self._notify()
                            break
                        except queue.Full:
                            continue
                finally:
                    with self._condition:
                        self._crop_outstanding -= 1
                        self._condition.notify_all()
        except BaseException as exc:
            self._set_fatal(exc)

    def _accept_prepared_page(
        self,
        submission: PageSubmission,
        prepared: PreparedLayoutPage,
    ) -> None:
        state = _PageState(
            submission=submission,
            prepared=prepared,
            remaining=len(prepared.requests),
        )
        if state.remaining == 0:
            self._emit_completed_page(state)
            return
        owned_requests = list(
            zip(
                prepared.requests,
                prepared.request_block_indices,
                strict=True,
            )
        )
        with self._owners_lock:
            for request, block_index in owned_requests:
                if request.request_id in self._owners:
                    raise ValueError(f"duplicate recognition request: {request.request_id}")
                self._owners[request.request_id] = _RequestOwner(state, block_index)
        for request, _block_index in owned_requests:
            while not self._stop.is_set():
                try:
                    self._raw_crops.put(request, timeout=0.1)
                    break
                except queue.Full:
                    continue
        prepared.requests.clear()
        prepared.request_block_indices.clear()

    def _postprocess_done(
        self,
        submission: PageSubmission,
        future: Future[PreparedLayoutPage],
    ) -> None:
        try:
            self._accept_prepared_page(submission, future.result())
        except BaseException as exc:
            self.emit_page_error(submission.request_id, exc)
        finally:
            self._page_slots.release()
            with self._condition:
                self._postprocess_outstanding -= 1
                self._condition.notify_all()

    def _run_layout(self, item: _LayoutReady) -> None:
        started = time.perf_counter()
        detected = self.frontend.detect_transferred_page(item.transferred)
        with self._condition:
            self._postprocess_outstanding += 1
        future = self._postprocess_executor.submit(
            self.frontend.prepare_detected_page,
            detected,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            text_max_pixels=self.text_max_pixels,
            table_max_pixels=self.table_max_pixels,
            text_crop_scale=self.text_crop_scale,
        )
        future.add_done_callback(
            lambda completed, submission=item.submission: self._postprocess_done(
                submission,
                completed,
            )
        )
        self._record_mode("layout", time.perf_counter() - started)

    def _harvest_prepared(self) -> None:
        while True:
            try:
                self._prepared.append(self._prepared_inbox.get_nowait())
            except queue.Empty:
                return

    def _run_prefill(self) -> None:
        if not self._prepared:
            raise RuntimeError("prefill requested without prepared crops")
        members = [self._prepared.popleft()]
        total = int(members[0][0].pixel_values.shape[0])
        if self.recognizer.vision_packing != "off" and total <= self.recognizer.vision_pack_target:
            while self._prepared:
                candidate = self._prepared[0]
                candidate_tokens = int(candidate[0].pixel_values.shape[0])
                if total + candidate_tokens > self.recognizer.vision_pack_target:
                    break
                members.append(self._prepared.popleft())
                total += candidate_tokens
        started = time.perf_counter()
        group = self.recognizer._prepared_group(members)
        staged = self.recognizer._stage_prefill_group(group)
        inflight = self.recognizer._enqueue_staged_prefill_group(staged)
        finalized = self.recognizer._finalize_prefill_group(inflight)
        self._ready.extend(
            self.recognizer._ready_from_prefilled(state) for state in finalized
        )
        self._record_mode(
            "prefill",
            time.perf_counter() - started,
            crops=len(finalized),
        )

    def _record_mode(self, mode: str, wall_s: float, *, crops: int = 0) -> None:
        self._mode_counts[mode] += 1
        self._mode_wall_s[mode] += float(wall_s)
        if crops:
            self._mode_counts["prefill_crops"] += int(crops)
        count = self._mode_counts[mode]
        if count == 1 or count % 100 == 0:
            print(
                "PAGE_SCHEDULER "
                + json.dumps(
                    {
                        "mode": mode,
                        "count": count,
                        "wall_s": self._mode_wall_s[mode],
                        "pages_completed": self._pages_completed,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def _raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    def pull(self, *, block: bool) -> ReadyDecodeRequest | None:
        while True:
            self._raise_if_failed()
            self._harvest_prepared()
            if self._ready:
                return self._ready.popleft()

            active = int(self.recognizer.decode_scheduler.arena.num_active)
            free = int(self.recognizer.batch_size - active)
            prepared = len(self._prepared)

            if free <= 0:
                return None
            if prepared > 0 and (active == 0 or prepared >= free):
                self._run_prefill()
                continue

            try:
                layout_item = self._layout_ready.get_nowait()
            except queue.Empty:
                layout_item = None
            if layout_item is not None:
                self._run_layout(layout_item)
                continue

            self._harvest_prepared()
            if self._prepared:
                self._run_prefill()
                continue
            if self.closed:
                return None
            if active > 0 or not block:
                if active > 0:
                    self._mode_counts["partial_decode_handoffs"] += 1
                return None
            with self._condition:
                self._condition.wait(timeout=0.05)

    def accept_result(self, result: RecognitionResult) -> None:
        with self._owners_lock:
            owner = self._owners.pop(result.request_id)
            page = owner.page
            if owner.block_index in page.recognition:
                raise RuntimeError(f"duplicate recognition result: {result.request_id}")
            page.recognition[owner.block_index] = result.text
            page.remaining -= 1
            complete = page.remaining == 0
        if complete:
            self._emit_completed_page(page)

    def _emit_completed_page(self, page: _PageState) -> None:
        prepared = page.prepared
        width, height = prepared.image_size
        result = OwnedPageResult(
            input_path=prepared.image_path,
            width=width,
            height=height,
            blocks=assemble_page_blocks(
                prepared.blocks,
                page.recognition,
                figure_token_maps=prepared.figure_token_maps,
                dropped_figure_paths=prepared.dropped_figure_paths,
            ),
            document_images=prepared.document_images,
        )
        self.emit_page(
            page.submission.request_id,
            result,
            {
                "page_wall_s": time.perf_counter() - page.submission.submitted_at,
                **prepared.timing_s,
            },
        )
        self._pages_completed += 1

    def summary(self) -> dict[str, Any]:
        return {
            "mode_counts": dict(self._mode_counts),
            "mode_wall_s": dict(self._mode_wall_s),
            "pages_completed": self._pages_completed,
        }

    def close(self) -> None:
        self._stop.set()
        try:
            self._raw_crops.put_nowait(self._crop_sentinel)
        except queue.Full:
            pass
        self._notify()
        self._intake_thread.join(timeout=5.0)
        self._crop_thread.join(timeout=5.0)
        self._input_executor.shutdown(wait=True, cancel_futures=True)
        self._postprocess_executor.shutdown(wait=True, cancel_futures=True)


class PersistentPageEngine:
    """Run one persistent decode schedule over an open stream of full pages."""

    def __init__(
        self,
        frontend: OwnedLayoutFrontend,
        recognizer: ContinuousRecognizer,
        *,
        min_pixels: int | None,
        max_pixels: int,
        text_max_pixels: int | None = None,
        table_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
    ) -> None:
        self.frontend = frontend
        self.recognizer = recognizer
        self.min_pixels = min_pixels
        self.max_pixels = int(max_pixels)
        self.text_max_pixels = text_max_pixels
        self.table_max_pixels = table_max_pixels
        self.text_crop_scale = float(text_crop_scale)
        self.last_mode_summary: dict[str, Any] = {}

    def serve(
        self,
        pages: OpenPageSubmissionSource,
        *,
        emit_page: Callable[[str, OwnedPageResult, dict[str, float]], None],
        emit_page_error: Callable[[str, BaseException], None],
        schedule_id: str = "http:pages",
    ) -> ContinuousDecodeResult:
        source = _UnifiedPageReadySource(
            self.frontend,
            self.recognizer,
            pages,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            text_max_pixels=self.text_max_pixels,
            table_max_pixels=self.table_max_pixels,
            text_crop_scale=self.text_crop_scale,
            emit_page=emit_page,
            emit_page_error=emit_page_error,
        )
        self.recognizer._begin_decode_schedule()
        try:
            result = self.recognizer._decode_ready_source(
                source,
                schedule_id=schedule_id,
                emit_result=source.accept_result,
            )
            self.last_mode_summary = source.summary()
            return result
        finally:
            source.close()
