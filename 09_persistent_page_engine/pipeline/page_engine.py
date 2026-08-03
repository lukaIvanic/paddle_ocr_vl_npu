"""Sequential owned layout frontend feeding continuous crop recognition."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.types import (
    ContinuousDecodeResult,
    RecognitionRequest,
    RecognitionResult,
)
from utils.timeline import TimelineRecorder

from .layout_frontend import OwnedLayoutFrontend, PreparedLayoutPage
from .layout_output import OwnedPageResult, assemble_page_blocks


PROMPT_LABELS = {
    "OCR:": "text",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
    "Chart Recognition:": "chart",
    "Spotting:": "spotting",
    "Seal Recognition:": "seal",
}


@dataclass
class _PageState:
    prepared: PreparedLayoutPage
    remaining: int
    recognition: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _RequestOwner:
    page: _PageState
    block_index: int
    request_index: int
    pixel_profile: tuple[int, int]
    source_crop_size: tuple[int, int]


@dataclass
class _QueuedPage:
    state: _PageState
    put_started_ns: int = 0


@dataclass(frozen=True)
class PageRunResult:
    schedule: ContinuousDecodeResult
    summary: dict[str, Any]
    completion_order: list[str]
    frontend: dict[str, Any]


class OwnedPageEngine:
    """Connect the owned page frontend to the crop-invariant recognizer."""

    def __init__(
        self,
        frontend: OwnedLayoutFrontend,
        recognizer: ContinuousRecognizer,
        *,
        trace_path: Path,
        min_pixels: int | None,
        max_pixels: int | None = None,
        text_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
        timeline: TimelineRecorder | None = None,
    ) -> None:
        self.frontend = frontend
        self.recognizer = recognizer
        self.trace_path = trace_path.expanduser().resolve()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.text_max_pixels = text_max_pixels
        self.text_crop_scale = float(text_crop_scale)
        self.timeline = timeline

    @staticmethod
    def _trace(
        result: RecognitionResult,
        owner: _RequestOwner,
    ) -> dict[str, Any]:
        return {
            "global_request_index": owner.request_index,
            "page_input_index": owner.page.prepared.ordinal,
            "block_index": owner.block_index,
            "request_id": result.request_id,
            "source_image_name": owner.page.prepared.image_path.name,
            "label": PROMPT_LABELS.get(result.prompt, "unknown"),
            "prompt": result.prompt,
            "crop_size": list(result.crop_size),
            "source_crop_size": list(owner.source_crop_size),
            "min_pixels": owner.pixel_profile[0],
            "max_pixels": owner.pixel_profile[1],
            "input_tokens": result.input_tokens,
            "projected_image_tokens": result.projected_image_tokens,
            "token_ids": result.token_ids,
            "text": result.text,
            "stop_reason": result.stop_reason,
            "spotting_group_size": 1,
            "generated_tokens_including_eos": (
                result.generated_tokens_including_eos
            ),
            "decode_tokens_after_prefill_including_eos": (
                result.decode_tokens_after_prefill_including_eos
            ),
            "decode_schedule_id": result.decode_schedule_id,
            "decode_slot_index": result.decode_slot_index,
            "decode_slot_epoch": result.decode_slot_epoch,
            "decode_calls_executed": result.decode_calls_executed,
            "vision": result.vision,
            "text_prefill": result.text_prefill,
            "timing_s": result.timing_s,
            "device_stage_s": result.device_stage_s,
            "input_fingerprints": result.input_fingerprints,
        }

    def _finish_page(self, page: _PageState) -> OwnedPageResult:
        prepared = page.prepared
        width, height = prepared.image_size
        blocks = assemble_page_blocks(
            prepared.blocks,
            page.recognition,
            figure_token_maps=prepared.figure_token_maps,
            dropped_figure_paths=prepared.dropped_figure_paths,
        )
        return OwnedPageResult(
            input_path=prepared.image_path,
            width=width,
            height=height,
            blocks=blocks,
            document_images=prepared.document_images,
        )

    @staticmethod
    def _recognition_summary(
        results: list[RecognitionResult],
        schedule: ContinuousDecodeResult,
        wall_s: float,
        profiles: Counter[str],
        trace_path: Path,
    ) -> dict[str, Any]:
        stage_s: defaultdict[str, float] = defaultdict(float)
        for result in results:
            for name, seconds in result.device_stage_s.items():
                stage_s[name] += float(seconds)
        total = lambda field: sum(
            int(getattr(result, field)) for result in results
        )
        nested = lambda section, field: sum(
            int(getattr(result, section).get(field, 0))
            for result in results
        )
        generated = total("generated_tokens_including_eos")
        real_vision = nested("vision", "real_vision_tokens")
        physical_vision = nested("vision", "physical_vision_tokens")
        real_text = nested("text_prefill", "real_text_tokens")
        physical_text = nested("text_prefill", "physical_text_tokens")
        raw_slots = schedule.raw_decode_token_slots
        return {
            "schedules": 1,
            "requests": len(results),
            "wall_s": wall_s,
            "generated_tokens_including_eos": generated,
            "decode_tokens_after_prefill_including_eos": total(
                "decode_tokens_after_prefill_including_eos"
            ),
            "input_tokens": total("input_tokens"),
            "projected_image_tokens": total("projected_image_tokens"),
            "real_vision_tokens": real_vision,
            "physical_vision_tokens": physical_vision,
            "real_text_tokens": real_text,
            "physical_text_tokens": physical_text,
            "decode_graph_calls": schedule.graph_calls,
            "raw_decode_token_slots": raw_slots,
            "active_decode_token_slots": (
                schedule.active_decode_token_slots
            ),
            "effective_decode_tokens": schedule.effective_decode_tokens,
            "idle_decode_token_slots": schedule.idle_decode_token_slots,
            "lookahead_decode_token_slots": (
                schedule.lookahead_decode_token_slots
            ),
            "decode_wall_s": schedule.timing_s["continuous_decode_wall"],
            "run_scoped_scheduler_wall_s": schedule.timing_s[
                "run_scoped_scheduler_wall"
            ],
            "kv_prefix_bytes_copied": schedule.kv_prefix_bytes_copied,
            "stop_reason_counts": dict(
                sorted(
                    Counter(
                        result.stop_reason for result in results
                    ).items()
                )
            ),
            "pixel_profile_request_counts": dict(sorted(profiles.items())),
            "device_stage_s": dict(sorted(stage_s.items())),
            "vision_packing": dict(schedule.vision_packing),
            "text_packing": dict(schedule.text_packing),
            "run_output_tok_per_s": (
                generated / wall_s if wall_s else None
            ),
            "decode_useful_token_fraction": (
                schedule.effective_decode_tokens / raw_slots
                if raw_slots
                else None
            ),
            "vision_useful_token_fraction": (
                real_vision / physical_vision if physical_vision else None
            ),
            "text_useful_token_fraction": (
                real_text / physical_text if physical_text else None
            ),
            "trace_path": str(trace_path),
        }

    def run(
        self,
        image_paths: Iterable[str],
        *,
        emit_page: Callable[[OwnedPageResult], None],
        schedule_id: str = "owned_cross_page",
        preprocess_all_pages_first: bool = False,
    ) -> PageRunResult:
        paths = [Path(path).expanduser().resolve() for path in image_paths]
        if not paths:
            raise ValueError("OwnedPageEngine.run requires at least one page")

        owners: dict[str, _RequestOwner] = {}
        results: list[RecognitionResult] = []
        completion_order: list[str] = []
        profiles: Counter[str] = Counter()
        frontend_stage_s: defaultdict[str, float] = defaultdict(float)
        frontend_statistics: Counter[str] = Counter()
        completed_pages = 0
        next_request = 0
        prepared_pages: queue.Queue[object] = queue.Queue(
            maxsize=len(paths) + 1 if preprocess_all_pages_first else 1
        )
        sentinel = object()
        stop = threading.Event()
        producer_errors: list[BaseException] = []
        producer_thread: threading.Thread | None = None
        layout_stream = torch.npu.Stream()

        def complete_page(page: _PageState) -> None:
            nonlocal completed_pages
            started_ns = time.perf_counter_ns()
            result = self._finish_page(page)
            if self.timeline is not None:
                self.timeline.record_span(
                    "Result assembly",
                    "Assemble owned page result",
                    started_ns,
                    time.perf_counter_ns(),
                    flow_id=f"page:{page.prepared.ordinal}",
                )
            emit_page(result)
            completion_order.append(page.prepared.image_path.name)
            completed_pages += 1
            if self.timeline is not None:
                self.timeline.instant(
                    "Pipeline",
                    "Page completed",
                    flow_id=f"page:{page.prepared.ordinal}",
                    args={
                        "remaining_pages": (
                            len(paths) - completed_pages
                        )
                    },
                )

        def page_requests(page: _PageState) -> Iterable[RecognitionRequest]:
            nonlocal next_request
            if page.remaining == 0:
                complete_page(page)
                return
            try:
                for request, block_index in zip(
                    page.prepared.requests,
                    page.prepared.request_block_indices,
                    strict=True,
                ):
                    owners[request.request_id] = _RequestOwner(
                        page=page,
                        block_index=block_index,
                        request_index=next_request,
                        pixel_profile=(
                            int(request.min_pixels or 112_896),
                            int(request.max_pixels or 1_003_520),
                        ),
                        source_crop_size=tuple(
                            int(value)
                            for value in (
                                request.source_crop_size or request.crop.size
                            )
                        ),
                    )
                    next_request += 1
                    profiles[
                        f"{request.min_pixels}:{request.max_pixels}"
                    ] += 1
                    yield request
            finally:
                page.prepared.requests.clear()
                page.prepared.request_block_indices.clear()

        def put_page(item: _QueuedPage) -> bool:
            item.put_started_ns = time.perf_counter_ns()
            while not stop.is_set():
                try:
                    prepared_pages.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def put_sentinel() -> None:
            while True:
                try:
                    prepared_pages.put(sentinel, timeout=0.1)
                    return
                except queue.Full:
                    if not stop.is_set():
                        continue
                    try:
                        prepared_pages.get_nowait()
                    except queue.Empty:
                        pass

        def produce_pages() -> None:
            try:
                for ordinal, path in enumerate(paths):
                    if stop.is_set():
                        break
                    started_ns = time.perf_counter_ns()
                    with torch.npu.stream(layout_stream):
                        prepared = self.frontend.prepare_page(
                            path,
                            ordinal,
                            min_pixels=self.min_pixels,
                            max_pixels=(
                                self.max_pixels
                                if self.max_pixels is not None
                                else 1_003_520
                            ),
                            text_max_pixels=self.text_max_pixels,
                            text_crop_scale=self.text_crop_scale,
                        )
                    layout_stream.synchronize()
                    for name, seconds in prepared.timing_s.items():
                        frontend_stage_s[name] += float(seconds)
                    for name, value in prepared.statistics.items():
                        frontend_statistics[name] += int(value)
                    if self.timeline is not None:
                        self.timeline.record_span(
                            "Page input",
                            "Prepare owned page",
                            started_ns,
                            time.perf_counter_ns(),
                            flow_id=f"page:{ordinal}",
                            event_type="scope",
                            args={"input_path": str(path)},
                        )
                    state = _PageState(
                        prepared=prepared,
                        remaining=len(prepared.requests),
                    )
                    if not put_page(_QueuedPage(state)):
                        break
            except BaseException as exception:
                producer_errors.append(exception)
            finally:
                put_sentinel()

        def request_source() -> Iterable[RecognitionRequest]:
            while True:
                get_started_ns = time.perf_counter_ns()
                item = prepared_pages.get()
                get_finished_ns = time.perf_counter_ns()
                if item is sentinel:
                    if producer_errors:
                        raise producer_errors[0]
                    break
                if not isinstance(item, _QueuedPage):
                    raise TypeError(
                        f"unexpected prepared-page item: {type(item)!r}"
                    )
                if self.timeline is not None:
                    flow_id = f"page:{item.state.prepared.ordinal}"
                    self.timeline.record_span(
                        "Page input",
                        "Wait for prepared page",
                        get_started_ns,
                        get_finished_ns,
                        flow_id=flow_id,
                        event_type="wait",
                    )
                    self.timeline.record_span(
                        "Page input",
                        "Prepared page waiting for consumer",
                        item.put_started_ns,
                        get_finished_ns,
                        flow_id=flow_id,
                        event_type="wait",
                        track="queue",
                        lane="page-queue",
                    )
                yield from page_requests(item.state)

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        producer_join_timed_out = False
        try:
            if preprocess_all_pages_first:
                produce_pages()
                if producer_errors:
                    raise producer_errors[0]
                if self.timeline is not None:
                    self.timeline.instant(
                        "Pipeline",
                        "All pages prepared; recognition starting",
                        args={"pages": len(paths)},
                    )
            else:
                producer_thread = threading.Thread(
                    target=produce_pages,
                    name="owned-page-producer",
                    daemon=True,
                )
                producer_thread.start()
            with self.trace_path.open("w", encoding="utf-8") as trace:
                def accept_result(result: RecognitionResult) -> None:
                    owner = owners.pop(result.request_id)
                    if owner.block_index in owner.page.recognition:
                        raise RuntimeError(
                            f"duplicate result {result.request_id}"
                        )
                    owner.page.recognition[owner.block_index] = result.text
                    owner.page.remaining -= 1
                    results.append(result)
                    trace_started_ns = time.perf_counter_ns()
                    trace.write(
                        json.dumps(
                            self._trace(result, owner),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    trace.flush()
                    if self.timeline is not None:
                        self.timeline.record_span(
                            "Artifacts / tracing",
                            "Write and flush crop trace record",
                            trace_started_ns,
                            time.perf_counter_ns(),
                            flow_id=result.request_id,
                            event_type="io",
                        )
                    if owner.page.remaining == 0:
                        complete_page(owner.page)

                schedule = self.recognizer.run(
                    request_source(),
                    schedule_id=schedule_id,
                    emit_result=accept_result,
                )
        finally:
            stop.set()
            if producer_thread is not None:
                producer_thread.join(timeout=30.0)
                producer_join_timed_out = producer_thread.is_alive()

        if producer_join_timed_out:
            raise RuntimeError(
                "owned page producer did not stop within 30 seconds"
            )
        wall_s = time.perf_counter() - started
        if (
            owners
            or completed_pages != len(paths)
            or len(results) != schedule.requests
        ):
            raise AssertionError(
                "cross-page accounting mismatch: "
                f"owners={len(owners)} "
                f"pages={completed_pages}/{len(paths)} "
                f"results={len(results)}/{schedule.requests}"
            )
        return PageRunResult(
            schedule=schedule,
            summary=self._recognition_summary(
                results,
                schedule,
                wall_s,
                profiles,
                self.trace_path,
            ),
            completion_order=completion_order,
            frontend={
                "implementation": "owned_no_paddlex",
                "device": str(self.frontend.device),
                "graph_capture": self.frontend.graph_capture,
                "npu_indexput_compat": (
                    self.frontend.npu_indexput_compat
                ),
                "setup_s": self.frontend.setup_s,
                "stage_s": dict(sorted(frontend_stage_s.items())),
                "statistics": dict(sorted(frontend_statistics.items())),
            },
        )
