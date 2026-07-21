"""Persistent fixed-shape decode arena with iteration-level slot reuse."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from ..model.text_decode import LocalPaddleOCRVLStaticCache
from utils.timing import synchronize
from utils.timeline import TimelineRecorder


@dataclass
class ReadyDecodeRequest:
    request_id: str
    payload: Any
    cache: LocalPaddleOCRVLStaticCache | None
    rope_deltas: torch.Tensor | None
    cache_position: torch.Tensor | None
    first_token_tensor: torch.Tensor | None
    first_token: int
    prompt_length: int

    def release_device_state(self) -> None:
        """Drop the per-request NPU prefix after it enters the decode arena."""

        self.cache = None
        self.rope_deltas = None
        self.cache_position = None
        self.first_token_tensor = None


@dataclass
class DecodeSlotState:
    slot_index: int
    epoch: int
    ready: ReadyDecodeRequest
    token_ids: list[int]
    admitted_at: float
    first_decode_launched_at: float | None = None
    iterations_launched: int = 0


@dataclass
class DecodeCompletion:
    ready: ReadyDecodeRequest
    token_ids: list[int]
    stop_reason: str
    slot_index: int | None
    slot_epoch: int | None
    admitted_at: float | None
    first_decode_launched_at: float | None
    completed_at: float
    iterations_launched: int


@dataclass
class PendingTokenCopy:
    iteration: int
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]
    ring_index: int | None
    done_event: Any | None
    host_tokens: list[int] | None


@dataclass
class DecodeStep:
    sampled: torch.Tensor
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]


@dataclass
class ContinuousDecodeRun:
    completions: list[DecodeCompletion]
    submitted_requests: int
    ready_buffer_capacity: int
    ready_buffer_low_watermark: int
    max_ready_queue_depth: int
    ready_source_refill_count: int
    graph_calls: int
    initial_admissions: int
    hot_swap_admissions: int
    prefill_only_completions: int
    raw_decode_token_slots: int
    active_decode_token_slots: int
    effective_decode_tokens: int
    idle_decode_token_slots: int
    lookahead_decode_token_slots: int
    kv_prefix_bytes_copied: int
    initial_kv_prefix_bytes_copied: int
    hot_swap_kv_prefix_bytes_copied: int
    timing_s: dict[str, float]


@dataclass
class _DeviceSpanRecord:
    start_event: Any | None
    end_event: Any | None
    enqueued_ns: int
    duration_s: float | None
    row: str
    name: str
    lane: str
    flow_id: str | None
    flow_ids: tuple[str, ...]
    event_type: str
    args: dict[str, Any]


class DecodeArena:
    """Own the tensors whose shapes and identities remain stable across steps."""

    def __init__(
        self,
        *,
        cache: LocalPaddleOCRVLStaticCache,
        device: torch.device,
        batch_size: int,
        eos_token_id: int,
        timeline: TimelineRecorder | None = None,
    ) -> None:
        self.cache = cache
        self.device = device
        self.batch_size = int(batch_size)
        self.eos_token_id = int(eos_token_id)
        self.timeline = timeline
        self.next_token = torch.full(
            (self.batch_size, 1),
            self.eos_token_id,
            device=self.device,
            dtype=torch.int64,
        )
        self.cache_position = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.int64,
        )
        self.rope_deltas = torch.zeros(
            (self.batch_size, 1),
            device=self.device,
            dtype=torch.int64,
        )
        self.active_mask = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.bool,
        )
        self.slots: list[DecodeSlotState | None] = [None] * self.batch_size
        self._epochs = [0] * self.batch_size
        self._decode_event_spans: list[_DeviceSpanRecord] = []
        self._admission_event_spans: list[_DeviceSpanRecord] = []
        self.admission_enqueue_wall_s = 0.0
        self.kv_prefix_bytes_copied = 0

    def begin_run(self) -> None:
        if any(slot is not None for slot in self.slots):
            raise RuntimeError("decode arena still contains active slots")
        self._decode_event_spans.clear()
        self._admission_event_spans.clear()
        self.admission_enqueue_wall_s = 0.0
        self.kv_prefix_bytes_copied = 0
        self.next_token.fill_(self.eos_token_id)
        self.cache_position.zero_()
        self.rope_deltas.zero_()
        self.active_mask.zero_()

    @property
    def num_active(self) -> int:
        return sum(slot is not None for slot in self.slots)

    def free_slot_indices(self) -> list[int]:
        return [index for index, state in enumerate(self.slots) if state is None]

    def _event(self) -> Any | None:
        if self.device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        if self.device.type == "npu":
            import torch_npu

            return torch_npu.npu.Event(enable_timing=True)
        return None

    def _measure_enqueue(
        self,
        records: list[_DeviceSpanRecord],
        fn: Callable[[], Any],
        *,
        row: str,
        name: str,
        lane: str = "decode",
        flow_id: str | None = None,
        flow_ids: tuple[str, ...] = (),
        event_type: str = "work",
        args: dict[str, Any] | None = None,
    ) -> Any:
        start_event = self._event()
        end_event = self._event()
        enqueued_ns = time.perf_counter_ns()
        if start_event is not None:
            start_event.record()
        result = fn()
        if end_event is not None:
            end_event.record()
            duration_s = None
        else:
            duration_s = (time.perf_counter_ns() - enqueued_ns) / 1_000_000_000
        records.append(
            _DeviceSpanRecord(
                start_event=start_event,
                end_event=end_event,
                enqueued_ns=enqueued_ns,
                duration_s=duration_s,
                row=row,
                name=name,
                lane=lane,
                flow_id=flow_id,
                flow_ids=tuple(flow_ids),
                event_type=event_type,
                args=dict(args or {}),
            )
        )
        return result

    @staticmethod
    def _resolve_spans(records: list[_DeviceSpanRecord]) -> float:
        total = 0.0
        for record in records:
            if record.duration_s is not None:
                total += record.duration_s
            else:
                assert record.start_event is not None
                assert record.end_event is not None
                total += float(record.start_event.elapsed_time(record.end_event)) / 1000.0
        return total

    def resolve_device_timing(self) -> tuple[float, float]:
        all_records = self._admission_event_spans + self._decode_event_spans
        event_records = [
            record for record in all_records if record.start_event is not None
        ]
        anchor = (
            min(event_records, key=lambda record: record.enqueued_ns)
            if event_records
            else None
        )
        if self.timeline is not None:
            for record in all_records:
                if record.duration_s is not None:
                    start_ns = record.enqueued_ns
                    duration_s = record.duration_s
                    clock = "host_monotonic"
                else:
                    if anchor is None or anchor.start_event is None:
                        raise RuntimeError("decode device timing lost its anchor event")
                    assert record.start_event is not None
                    assert record.end_event is not None
                    offset_s = float(
                        anchor.start_event.elapsed_time(record.start_event)
                    ) / 1000.0
                    duration_s = float(
                        record.start_event.elapsed_time(record.end_event)
                    ) / 1000.0
                    start_ns = anchor.enqueued_ns + int(offset_s * 1_000_000_000)
                    clock = "device_event_reconstructed"
                self.timeline.record_span(
                    record.row,
                    record.name,
                    start_ns,
                    start_ns + int(duration_s * 1_000_000_000),
                    flow_id=record.flow_id,
                    flow_ids=list(record.flow_ids),
                    event_type=record.event_type,
                    clock=clock,
                    track="device",
                    lane=record.lane,
                    args=record.args,
                )
        return (
            self._resolve_spans(self._decode_event_spans),
            self._resolve_spans(self._admission_event_spans),
        )

    def admit(
        self,
        slot_index: int,
        ready: ReadyDecodeRequest,
        *,
        hot_swap: bool,
    ) -> tuple[DecodeSlotState, int]:
        if self.slots[slot_index] is not None:
            raise RuntimeError(f"decode slot {slot_index} is not free")
        source_cache = ready.cache
        source_rope_deltas = ready.rope_deltas
        source_cache_position = ready.cache_position
        source_first_token = ready.first_token_tensor
        if (
            source_cache is None
            or source_rope_deltas is None
            or source_cache_position is None
            or source_first_token is None
        ):
            raise RuntimeError(
                f"request {ready.request_id} no longer owns prefill device state"
            )
        prompt_length = int(ready.prompt_length)
        if prompt_length <= 0 or prompt_length > int(self.cache.cache_length):
            raise ValueError(
                f"request {ready.request_id} has invalid prompt length {prompt_length}"
            )
        if len(source_cache.key_caches) != len(self.cache.key_caches):
            raise ValueError("ready cache and decode arena have different layer counts")

        copied_bytes = 0
        for source in (*source_cache.key_caches, *source_cache.value_caches):
            copied_bytes += int(source[:, :, :prompt_length, :].numel()) * source.element_size()

        def copy_state() -> None:
            for destination, source in zip(
                self.cache.key_caches,
                source_cache.key_caches,
            ):
                destination[slot_index : slot_index + 1, :, :prompt_length, :].copy_(
                    source[:, :, :prompt_length, :]
                )
            for destination, source in zip(
                self.cache.value_caches,
                source_cache.value_caches,
            ):
                destination[slot_index : slot_index + 1, :, :prompt_length, :].copy_(
                    source[:, :, :prompt_length, :]
                )
            self.rope_deltas[slot_index : slot_index + 1].copy_(source_rope_deltas)
            self.cache_position[slot_index : slot_index + 1].copy_(
                source_cache_position.reshape(1)
            )
            self.next_token[slot_index : slot_index + 1].copy_(source_first_token)
            self.active_mask[slot_index].fill_(True)

        started = time.perf_counter()
        self._measure_enqueue(
            self._admission_event_spans,
            copy_state,
            row="Decode admission",
            name="Copy prefetched KV prefix into decode slot",
            flow_id=ready.request_id,
            event_type="io",
            args={
                "slot": slot_index,
                "prompt_tokens": prompt_length,
                "copied_bytes": copied_bytes,
                "hot_swap": hot_swap,
            },
        )
        self.admission_enqueue_wall_s += time.perf_counter() - started
        self.kv_prefix_bytes_copied += copied_bytes
        self._epochs[slot_index] += 1
        state = DecodeSlotState(
            slot_index=slot_index,
            epoch=self._epochs[slot_index],
            ready=ready,
            token_ids=[int(ready.first_token)],
            admitted_at=time.perf_counter(),
        )
        self.slots[slot_index] = state
        # DecodeSlotState and DecodeCompletion intentionally retain request
        # metadata, but the copied prefill cache must not survive admission.
        # Large producer streams may contain hundreds of crops; retaining one
        # cache per completed crop grows HBM until the outer call returns.
        ready.release_device_state()
        return state, copied_bytes

    def release(self, slot_index: int) -> DecodeSlotState:
        state = self.slots[slot_index]
        if state is None:
            raise RuntimeError(f"decode slot {slot_index} is already free")
        self.slots[slot_index] = None
        self.active_mask[slot_index].fill_(False)
        self.next_token[slot_index].fill_(self.eos_token_id)
        self.cache_position[slot_index].zero_()
        self.rope_deltas[slot_index].zero_()
        return state

    def step(
        self,
        decode_fn: Callable[..., torch.Tensor],
        *,
        iteration: int,
    ) -> DecodeStep:
        active_slots = tuple(slot is not None for slot in self.slots)
        slot_epochs = tuple(slot.epoch if slot is not None else None for slot in self.slots)
        launched_at = time.perf_counter()
        for slot in self.slots:
            if slot is not None:
                slot.iterations_launched += 1
                if slot.first_decode_launched_at is None:
                    slot.first_decode_launched_at = launched_at

        def execute() -> torch.Tensor:
            logits = decode_fn(
                self.next_token,
                self.cache_position,
                self.rope_deltas,
                *self.cache.flat_tensors(),
            )
            return torch.argmax(
                logits[:, -1, :].float(),
                dim=-1,
                keepdim=True,
            )

        request_ids = tuple(
            slot.ready.request_id for slot in self.slots if slot is not None
        )
        sampled = self._measure_enqueue(
            self._decode_event_spans,
            execute,
            row="Text decode",
            name="Compiled decode iteration",
            flow_id=f"decode-iteration:{iteration}",
            flow_ids=request_ids,
            args={
                "iteration": iteration,
                "active_slots": sum(active_slots),
                "batch_size": self.batch_size,
                "request_ids": list(request_ids),
            },
        )
        self.next_token = torch.where(
            self.active_mask.view(-1, 1),
            sampled,
            torch.full_like(sampled, self.eos_token_id),
        )
        self.cache_position = torch.where(
            self.active_mask,
            self.cache_position + 1,
            torch.zeros_like(self.cache_position),
        )
        return DecodeStep(
            sampled=sampled,
            active_slots=active_slots,
            slot_epochs=slot_epochs,
        )


class ContinuousDecodeScheduler:
    """Run a ready queue through a persistent decode arena until completion."""

    def __init__(
        self,
        *,
        arena: DecodeArena,
        decode_fn: Callable[..., torch.Tensor],
        max_new_tokens: int,
        timeline: TimelineRecorder | None = None,
    ) -> None:
        self.arena = arena
        self.decode_fn = decode_fn
        self.max_new_tokens = int(max_new_tokens)
        self.device = arena.device
        self.batch_size = arena.batch_size
        self.eos_token_id = arena.eos_token_id
        self.timeline = timeline
        self.copy_stream = None
        self.host_token_ring = None
        if self.device.type == "npu":
            import torch_npu

            self.copy_stream = torch_npu.npu.Stream(device=self.device)
            self.host_token_ring = torch.empty(
                (2, self.batch_size),
                dtype=torch.int64,
                pin_memory=True,
            )

    def _schedule_token_copy(
        self,
        step: DecodeStep,
        iteration: int,
    ) -> PendingTokenCopy:
        if self.device.type == "npu":
            import torch_npu

            assert self.copy_stream is not None
            assert self.host_token_ring is not None
            ring_index = iteration % 2
            ready_event = torch_npu.npu.current_stream().record_event()
            done_event = torch_npu.npu.Event()
            with torch_npu.npu.stream(self.copy_stream):
                self.copy_stream.wait_event(ready_event)
                self.host_token_ring[ring_index].copy_(
                    step.sampled.reshape(-1),
                    non_blocking=True,
                )
                done_event.record(self.copy_stream)
            return PendingTokenCopy(
                iteration=iteration,
                active_slots=step.active_slots,
                slot_epochs=step.slot_epochs,
                ring_index=ring_index,
                done_event=done_event,
                host_tokens=None,
            )
        return PendingTokenCopy(
            iteration=iteration,
            active_slots=step.active_slots,
            slot_epochs=step.slot_epochs,
            ring_index=None,
            done_event=None,
            host_tokens=[int(value) for value in step.sampled.detach().cpu().reshape(-1).tolist()],
        )

    def _wait_tokens(self, pending: PendingTokenCopy) -> tuple[list[int], float]:
        started = time.perf_counter()
        if pending.done_event is not None:
            pending.done_event.synchronize()
            assert self.host_token_ring is not None
            assert pending.ring_index is not None
            tokens = [
                int(value)
                for value in self.host_token_ring[pending.ring_index].tolist()
            ]
        else:
            assert pending.host_tokens is not None
            tokens = pending.host_tokens
        return tokens, time.perf_counter() - started

    def run(self, ready_requests: list[ReadyDecodeRequest]) -> ContinuousDecodeRun:
        return self.run_stream(ready_requests)

    def run_stream(
        self,
        ready_requests: Iterable[ReadyDecodeRequest],
        *,
        on_completion: Callable[[DecodeCompletion], None] | None = None,
        ready_buffer_capacity: int | None = None,
        ready_buffer_low_watermark: int | None = None,
    ) -> ContinuousDecodeRun:
        """Decode a bounded, lazily produced request stream.

        The producer may perform eager prefill before yielding each request.  A
        private ready reservoir keeps at most ``ready_buffer_capacity`` NPU KV
        prefixes waiting behind the fixed decode arena.  This makes page
        boundaries irrelevant without materializing an unbounded document.
        """

        self.arena.begin_run()
        ready_source = iter(ready_requests)
        buffer_capacity = (
            self.batch_size
            if ready_buffer_capacity is None
            else int(ready_buffer_capacity)
        )
        if buffer_capacity <= 0:
            raise ValueError("ready_buffer_capacity must be positive")
        low_watermark = (
            min(self.batch_size, buffer_capacity)
            if ready_buffer_low_watermark is None
            else int(ready_buffer_low_watermark)
        )
        if low_watermark <= 0 or low_watermark > buffer_capacity:
            raise ValueError(
                "ready_buffer_low_watermark must be in [1, ready_buffer_capacity]"
            )
        ready_queue: deque[ReadyDecodeRequest] = deque()
        source_exhausted = False
        submitted_order: list[str] = []
        submitted_ids: set[str] = set()
        max_ready_queue_depth = 0
        ready_source_refill_count = 0
        completions: list[DecodeCompletion] = []
        graph_calls = 0
        initial_admissions = 0
        hot_swap_admissions = 0
        prefill_only_completions = 0
        active_decode_slots = 0
        initial_kv_bytes = 0
        hot_swap_kv_bytes = 0
        d2h_wait_wall_s = 0.0
        retire_and_refill_wall_s = 0.0
        ready_source_wall_s = 0.0
        completion_callback_wall_s = 0.0
        ready_queued_ns: dict[str, int] = {}

        def record_completion(completion: DecodeCompletion) -> None:
            nonlocal completion_callback_wall_s
            completions.append(completion)
            if self.timeline is not None:
                if completion.admitted_at is not None:
                    self.timeline.record_span_seconds(
                        "Decode request residency",
                        "Request resident in decode slot",
                        completion.admitted_at,
                        completion.completed_at,
                        flow_id=completion.ready.request_id,
                        track="slot",
                        lane=completion.slot_index,
                        args={
                            "slot": completion.slot_index,
                            "epoch": completion.slot_epoch,
                            "decode_iterations": completion.iterations_launched,
                            "stop_reason": completion.stop_reason,
                        },
                    )
                self.timeline.instant(
                    "Decode request residency",
                    "Decode request completed",
                    timestamp_ns=int(completion.completed_at * 1_000_000_000),
                    flow_id=completion.ready.request_id,
                    track="slot",
                    lane=completion.slot_index,
                    args={
                        "slot": completion.slot_index,
                        "generated_tokens": len(completion.token_ids),
                        "stop_reason": completion.stop_reason,
                    },
                )
            if on_completion is not None:
                started = time.perf_counter()
                on_completion(completion)
                finished = time.perf_counter()
                completion_callback_wall_s += finished - started
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Result assembly",
                        "Crop completion callback",
                        started,
                        finished,
                        flow_id=completion.ready.request_id,
                    )

        def refill_ready_queue() -> None:
            nonlocal source_exhausted, ready_source_wall_s
            nonlocal max_ready_queue_depth, ready_source_refill_count
            pulled = 0
            while not source_exhausted and len(ready_queue) < buffer_capacity:
                started = time.perf_counter()
                try:
                    ready = next(ready_source)
                except StopIteration:
                    source_exhausted = True
                    finished = time.perf_counter()
                    ready_source_wall_s += finished - started
                    if self.timeline is not None:
                        self.timeline.record_span_seconds(
                            "Decode control / wait",
                            "Drain ready-request source",
                            started,
                            finished,
                            event_type="scope",
                        )
                    break
                finished = time.perf_counter()
                ready_source_wall_s += finished - started
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Decode control / wait",
                        "Produce next prefilled crop",
                        started,
                        finished,
                        flow_id=ready.request_id,
                        event_type="scope",
                    )
                if ready.request_id in submitted_ids:
                    raise ValueError(f"duplicate decode request id: {ready.request_id}")
                submitted_ids.add(ready.request_id)
                submitted_order.append(ready.request_id)
                ready_queue.append(ready)
                ready_queued_ns[ready.request_id] = time.perf_counter_ns()
                pulled += 1
                max_ready_queue_depth = max(max_ready_queue_depth, len(ready_queue))
                if self.timeline is not None:
                    self.timeline.counter(
                        "Decode ready wait",
                        "Ready queue depth",
                        len(ready_queue),
                        lane="ready-queue",
                        args={"request_id": ready.request_id},
                    )
            if pulled:
                ready_source_refill_count += 1

        def fill_free_slots(*, hot_swap: bool) -> None:
            nonlocal initial_admissions, hot_swap_admissions
            nonlocal prefill_only_completions, initial_kv_bytes, hot_swap_kv_bytes
            for slot_index in self.arena.free_slot_indices():
                while True:
                    if not ready_queue and not source_exhausted:
                        refill_ready_queue()
                    if not ready_queue:
                        break
                    ready = ready_queue.popleft()
                    admitted_ns = time.perf_counter_ns()
                    queued_ns = ready_queued_ns.pop(ready.request_id, admitted_ns)
                    if self.timeline is not None:
                        self.timeline.record_span(
                            "Decode ready wait",
                            "Prefilled crop waiting for a decode slot",
                            queued_ns,
                            admitted_ns,
                            flow_id=ready.request_id,
                            event_type="wait",
                            track="queue",
                            lane="ready-queue",
                            args={
                                "slot": slot_index,
                                "ready_queue_after_pop": len(ready_queue),
                            },
                        )
                    if ready.first_token == self.eos_token_id or self.max_new_tokens == 1:
                        ready.release_device_state()
                        record_completion(
                            DecodeCompletion(
                                ready=ready,
                                token_ids=[int(ready.first_token)],
                                stop_reason=(
                                    "eos"
                                    if ready.first_token == self.eos_token_id
                                    else "length"
                                ),
                                slot_index=None,
                                slot_epoch=None,
                                admitted_at=None,
                                first_decode_launched_at=None,
                                completed_at=time.perf_counter(),
                                iterations_launched=0,
                            )
                        )
                        prefill_only_completions += 1
                        continue
                    _state, copied_bytes = self.arena.admit(
                        slot_index,
                        ready,
                        hot_swap=hot_swap,
                    )
                    if hot_swap:
                        hot_swap_admissions += 1
                        hot_swap_kv_bytes += copied_bytes
                    else:
                        initial_admissions += 1
                        initial_kv_bytes += copied_bytes
                    break

        synchronize(self.device)
        scheduler_started = time.perf_counter()
        if len(ready_queue) < low_watermark:
            refill_ready_queue()
        fill_free_slots(hot_swap=False)
        refill_ready_queue()
        pending: PendingTokenCopy | None = None
        iteration = 0

        while self.arena.num_active > 0:
            step = self.arena.step(self.decode_fn, iteration=iteration)
            graph_calls += 1
            active_decode_slots += sum(step.active_slots)
            current = self._schedule_token_copy(step, iteration)

            if pending is not None:
                host_tokens, wait_s = self._wait_tokens(pending)
                d2h_wait_wall_s += wait_s
                wait_finished = time.perf_counter()
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Decode control / wait",
                        "Wait for sampled-token D2H",
                        wait_finished - wait_s,
                        wait_finished,
                        flow_id=f"decode-iteration:{pending.iteration}",
                        event_type="wait",
                        args={"iteration": pending.iteration},
                    )
                started = time.perf_counter()
                for slot_index, was_active in enumerate(pending.active_slots):
                    if not was_active:
                        continue
                    state = self.arena.slots[slot_index]
                    expected_epoch = pending.slot_epochs[slot_index]
                    if state is None or state.epoch != expected_epoch:
                        continue
                    token_id = int(host_tokens[slot_index])
                    state.token_ids.append(token_id)
                    if token_id == self.eos_token_id or len(state.token_ids) >= self.max_new_tokens:
                        released = self.arena.release(slot_index)
                        record_completion(
                            DecodeCompletion(
                                ready=released.ready,
                                token_ids=list(released.token_ids),
                                stop_reason=(
                                    "eos" if token_id == self.eos_token_id else "length"
                                ),
                                slot_index=slot_index,
                                slot_epoch=released.epoch,
                                admitted_at=released.admitted_at,
                                first_decode_launched_at=released.first_decode_launched_at,
                                completed_at=time.perf_counter(),
                                iterations_launched=released.iterations_launched,
                            )
                        )
                fill_free_slots(hot_swap=True)
                if len(ready_queue) < low_watermark:
                    refill_ready_queue()
                finished = time.perf_counter()
                retire_and_refill_wall_s += finished - started
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Decode control / wait",
                        "Retire completed slots and refill",
                        started,
                        finished,
                        flow_id=f"decode-iteration:{pending.iteration}",
                        args={
                            "iteration": pending.iteration,
                            "active_after_refill": self.arena.num_active,
                            "ready_queue_depth": len(ready_queue),
                        },
                    )

            pending = current
            iteration += 1
            if self.arena.num_active == 0:
                _ignored, wait_s = self._wait_tokens(pending)
                d2h_wait_wall_s += wait_s
                wait_finished = time.perf_counter()
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Decode control / wait",
                        "Final sampled-token D2H drain",
                        wait_finished - wait_s,
                        wait_finished,
                        flow_id=f"decode-iteration:{pending.iteration}",
                        event_type="wait",
                    )
                pending = None
                break

        synchronize(self.device)
        scheduler_wall_s = time.perf_counter() - scheduler_started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "Pipeline",
                "Continuous decode scheduler",
                scheduler_started,
                scheduler_started + scheduler_wall_s,
                event_type="scope",
                args={"batch_size": self.batch_size},
            )
        decode_host_exclusive_wall_s = max(
            0.0,
            scheduler_wall_s
            - ready_source_wall_s
            - completion_callback_wall_s,
        )
        decode_device_s, admission_device_s = self.arena.resolve_device_timing()
        continuous_decode_wall_s = max(
            decode_host_exclusive_wall_s,
            decode_device_s + admission_device_s,
        )

        if ready_queue or not source_exhausted:
            raise AssertionError(f"continuous decode stopped with {len(ready_queue)} ready requests")
        if len(completions) != len(submitted_order):
            raise AssertionError(
                f"continuous decode completed {len(completions)} of {len(submitted_order)} requests"
            )

        effective_tokens = sum(max(0, len(item.token_ids) - 1) for item in completions)
        raw_slots = graph_calls * self.batch_size
        idle_slots = raw_slots - active_decode_slots
        lookahead_slots = active_decode_slots - effective_tokens
        if idle_slots < 0 or lookahead_slots < 0:
            raise AssertionError(
                "continuous decode accounting went negative: "
                f"raw={raw_slots} active={active_decode_slots} effective={effective_tokens}"
            )
        if raw_slots != effective_tokens + idle_slots + lookahead_slots:
            raise AssertionError("continuous decode slot accounting does not balance")

        completion_by_id = {item.ready.request_id: item for item in completions}
        ordered_completions = [completion_by_id[request_id] for request_id in submitted_order]
        return ContinuousDecodeRun(
            completions=ordered_completions,
            submitted_requests=len(submitted_order),
            ready_buffer_capacity=buffer_capacity,
            ready_buffer_low_watermark=low_watermark,
            max_ready_queue_depth=max_ready_queue_depth,
            ready_source_refill_count=ready_source_refill_count,
            graph_calls=graph_calls,
            initial_admissions=initial_admissions,
            hot_swap_admissions=hot_swap_admissions,
            prefill_only_completions=prefill_only_completions,
            raw_decode_token_slots=raw_slots,
            active_decode_token_slots=active_decode_slots,
            effective_decode_tokens=effective_tokens,
            idle_decode_token_slots=idle_slots,
            lookahead_decode_token_slots=lookahead_slots,
            kv_prefix_bytes_copied=self.arena.kv_prefix_bytes_copied,
            initial_kv_prefix_bytes_copied=initial_kv_bytes,
            hot_swap_kv_prefix_bytes_copied=hot_swap_kv_bytes,
            timing_s={
                "continuous_decode_wall": float(continuous_decode_wall_s),
                "decode_host_exclusive_wall": float(
                    decode_host_exclusive_wall_s
                ),
                "run_scoped_scheduler_wall": float(scheduler_wall_s),
                "ready_source_wall": float(ready_source_wall_s),
                "completion_callback_wall": float(completion_callback_wall_s),
                "decode_model_and_argmax_device": float(decode_device_s),
                "slot_admission_device": float(admission_device_s),
                "slot_admission_enqueue_wall": float(self.arena.admission_enqueue_wall_s),
                "d2h_wait_wall": float(d2h_wait_wall_s),
                "retire_and_refill_host_wall": float(retire_and_refill_wall_s),
            },
        )
