"""Persistent fixed-shape decode arena with iteration-level slot reuse."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import torch

from ..model.text_decode import LocalPaddleOCRVLStaticCache
from ..model.token_selection import (
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
    TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED,
    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
    select_token_ids,
)
from .repetition import ExactCycleTracker, RepetitionEvidence
from .scheduling_metrics import RequestSchedulingMetrics
from utils.timing import stream_synchronize, synchronize
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
    token_selection_policy_active: bool = False
    cache_release: Callable[[], None] | None = None

    def release_device_state(self) -> None:
        """Drop the per-request NPU prefix after it enters the decode arena."""

        cache_release = self.cache_release
        self.cache_release = None
        self.cache = None
        self.rope_deltas = None
        self.cache_position = None
        self.first_token_tensor = None
        if cache_release is not None:
            cache_release()


@dataclass
class DecodeSlotState:
    slot_index: int
    epoch: int
    ready: ReadyDecodeRequest
    token_ids: list[int]
    admitted_at: float
    first_decode_launched_at: float | None = None
    iterations_launched: int = 0
    # Host-only, request-local count for the optional open-serving prefill cap.
    prefill_interruptions: int = 0
    repetition_tracker: ExactCycleTracker = field(
        default_factory=ExactCycleTracker,
    )
    repetition_evidence: RepetitionEvidence | None = None

    def __post_init__(self) -> None:
        for token_id in self.token_ids:
            self.repetition_tracker.update(token_id)


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
    repetition_evidence: dict[str, int | str | None] | None = None
    scheduling_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingTokenCopy:
    iteration: int
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]
    slot_request_ids: tuple[str | None, ...]
    cache_positions: tuple[int | None, ...]
    generated_token_counts: tuple[int | None, ...]
    ring_index: int | None
    done_event: Any | None
    diagnostic_compute_event: Any | None
    host_tokens: list[int] | None


@dataclass
class DecodeStep:
    sampled: torch.Tensor
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]
    slot_request_ids: tuple[str | None, ...]
    cache_positions: tuple[int | None, ...]
    generated_token_counts: tuple[int | None, ...]


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
    timing_s: dict[str, float | None]


class OpenReadyDecodeSource(Protocol):
    """A request source that can be temporarily empty without being closed."""

    @property
    def closed(self) -> bool: ...

    def pull(self, *, block: bool) -> ReadyDecodeRequest | None: ...


class _IterableReadyDecodeSource:
    """Adapt the existing finite iterable contract to the open-source API."""

    def __init__(self, items: Iterable[ReadyDecodeRequest]):
        self._items = iter(items)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def pull(self, *, block: bool) -> ReadyDecodeRequest | None:
        del block
        if self._closed:
            return None
        try:
            return next(self._items)
        except StopIteration:
            self._closed = True
            return None


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
        token_selection: str = TOKEN_SELECTION_GREEDY,
        preferred_token_id: int | None = None,
        alternate_preferred_token_id: int | None = None,
        cell_start_token_ids: Iterable[int] = (),
        math_close_token_id: int | None = None,
        decode_token_id_map: torch.Tensor | None = None,
        timeline: TimelineRecorder | None = None,
        decode_device_timing: bool = True,
    ) -> None:
        self.cache = cache
        self.device = device
        self.batch_size = int(batch_size)
        self.eos_token_id = int(eos_token_id)
        self.decode_device_timing = bool(decode_device_timing)
        if timeline is not None and not self.decode_device_timing:
            raise ValueError("decode timeline requires decode device timing")
        self.token_selection = str(token_selection)
        self.preferred_token_id = (
            None if preferred_token_id is None else int(preferred_token_id)
        )
        self.alternate_preferred_token_id = (
            None
            if alternate_preferred_token_id is None
            else int(alternate_preferred_token_id)
        )
        self.cell_start_token_ids = tuple(int(value) for value in cell_start_token_ids)
        self.math_close_token_id = (
            None if math_close_token_id is None else int(math_close_token_id)
        )
        if (
            decode_token_id_map is not None
            and self.token_selection != TOKEN_SELECTION_GREEDY
        ):
            raise ValueError(
                "a compact decode vocabulary currently supports ordinary greedy "
                "token selection only"
            )
        self.decode_token_id_map = decode_token_id_map
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
        self.token_selection_policy_mask = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.bool,
        )
        self.token_selection_override_used = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.bool,
        )
        self.token_selection_math_open = torch.zeros(
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
        self.token_selection_policy_mask.zero_()
        self.token_selection_override_used.zero_()
        self.token_selection_math_open.zero_()

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
        if records is self._decode_event_spans and not self.decode_device_timing:
            # Profiling events only. Token-copy dependency/completion events
            # remain mandatory and are owned by the scheduler, not this helper.
            return fn()
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
        if int(source_cache.cache_length) != int(self.cache.cache_length):
            raise ValueError("ready cache and decode arena have different cache lengths")

        source_tensors = source_cache.logical_tensors()
        destination_tensors = tuple(
            destination[slot_index : slot_index + 1]
            for destination in self.cache.logical_tensors()
        )
        source_heads = int(source_tensors[0].shape[1])
        destination_heads = int(destination_tensors[0].shape[1])
        if destination_heads % source_heads != 0:
            raise ValueError(
                "decode arena KV heads must equal or be an integer multiple "
                f"of prefill KV heads: source={source_heads}, "
                f"destination={destination_heads}"
            )
        cache_head_expansion = destination_heads // source_heads
        useful_prefix_bytes = sum(
            int(source[:, :, :prompt_length, :].numel()) * source.element_size()
            for source in source_tensors
        )
        physical_copied_bytes = sum(
            int(destination.numel()) * destination.element_size()
            for destination in destination_tensors
        )

        def copy_state() -> None:
            if cache_head_expansion == 1:
                torch._foreach_copy_(destination_tensors, source_tensors)
            else:
                for destination, source in zip(
                    destination_tensors,
                    source_tensors,
                    strict=True,
                ):
                    batch_size, kv_heads, cache_length, head_dim = source.shape
                    expanded = source[:, :, None, :, :].expand(
                        batch_size,
                        kv_heads,
                        cache_head_expansion,
                        cache_length,
                        head_dim,
                    )
                    destination.view_as(expanded).copy_(expanded)
            self.rope_deltas[slot_index : slot_index + 1].copy_(source_rope_deltas)
            self.cache_position[slot_index : slot_index + 1].copy_(
                source_cache_position.reshape(1)
            )
            self.next_token[slot_index : slot_index + 1].copy_(source_first_token)
            self.active_mask[slot_index].fill_(True)
            self.token_selection_policy_mask[slot_index].fill_(
                bool(ready.token_selection_policy_active)
            )
            self.token_selection_override_used[slot_index].fill_(False)
            self.token_selection_math_open[slot_index].fill_(False)

        started = time.perf_counter()
        self._measure_enqueue(
            self._admission_event_spans,
            copy_state,
            row="Decode admission",
            name="Copy full prefetched KV cache into decode slot",
            flow_id=ready.request_id,
            event_type="io",
            args={
                "slot": slot_index,
                "prompt_tokens": prompt_length,
                "useful_prefix_bytes": useful_prefix_bytes,
                "physical_copied_bytes": physical_copied_bytes,
                "source_kv_heads": source_heads,
                "destination_kv_heads": destination_heads,
                "cache_head_expansion": cache_head_expansion,
                "hot_swap": hot_swap,
            },
        )
        self.admission_enqueue_wall_s += time.perf_counter() - started
        self.kv_prefix_bytes_copied += useful_prefix_bytes
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
        return state, useful_prefix_bytes

    def release(self, slot_index: int) -> DecodeSlotState:
        state = self.slots[slot_index]
        if state is None:
            raise RuntimeError(f"decode slot {slot_index} is already free")
        self.slots[slot_index] = None
        self.active_mask[slot_index].fill_(False)
        self.token_selection_policy_mask[slot_index].fill_(False)
        self.token_selection_override_used[slot_index].fill_(False)
        self.token_selection_math_open[slot_index].fill_(False)
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
        slot_request_ids = tuple(
            slot.ready.request_id if slot is not None else None
            for slot in self.slots
        )
        cache_positions = tuple(
            (
                int(slot.ready.prompt_length) + int(slot.iterations_launched)
                if slot is not None
                else None
            )
            for slot in self.slots
        )
        generated_token_counts = tuple(
            len(slot.token_ids) if slot is not None else None
            for slot in self.slots
        )
        launched_at = time.perf_counter()
        for slot in self.slots:
            if slot is not None:
                slot.iterations_launched += 1
                if slot.first_decode_launched_at is None:
                    slot.first_decode_launched_at = launched_at

        def execute() -> torch.Tensor:
            decode_output = decode_fn(
                self.next_token,
                self.cache_position,
                self.rope_deltas,
                *self.cache.flat_tensors(),
            )
            if self.decode_token_id_map is not None:
                return decode_output.reshape(-1, 1)
            logits = decode_output
            if self.preferred_token_id is not None:
                self.token_selection_math_open.copy_(
                    torch.where(
                        self.next_token[:, 0] == int(self.preferred_token_id),
                        torch.ones_like(self.token_selection_math_open),
                        self.token_selection_math_open,
                    )
                )
            if self.math_close_token_id is not None:
                self.token_selection_math_open.copy_(
                    torch.where(
                        self.next_token[:, 0] == int(self.math_close_token_id),
                        torch.zeros_like(self.token_selection_math_open),
                        self.token_selection_math_open,
                    )
                )
            if (
                self.token_selection
                == TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED
            ):
                policy_mask = (
                    self.token_selection_policy_mask
                    & ~self.token_selection_math_open
                )
            elif (
                self.token_selection
                == TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE
            ):
                policy_mask = (
                    self.token_selection_policy_mask
                    & ~self.token_selection_override_used
                )
            elif (
                self.token_selection
                == TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP
            ):
                policy_mask = self.token_selection_policy_mask
            elif (
                self.token_selection in (
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
                )
            ):
                policy_mask = self.token_selection_policy_mask
            elif (
                self.token_selection in (
                    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
                    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
                )
            ):
                cell_start_mask = torch.zeros_like(
                    self.token_selection_policy_mask,
                    dtype=torch.bool,
                )
                for token_id in self.cell_start_token_ids:
                    cell_start_mask |= self.next_token[:, 0] == int(token_id)
                policy_mask = self.token_selection_policy_mask & cell_start_mask
            else:
                policy_mask = torch.zeros_like(
                    self.token_selection_policy_mask,
                    dtype=torch.bool,
                )
            selected = select_token_ids(
                logits[:, -1, :].float(),
                mode=self.token_selection,
                preferred_token_id=self.preferred_token_id,
                alternate_preferred_token_id=self.alternate_preferred_token_id,
                policy_mask=policy_mask,
                legacy_policy_mask=self.token_selection_policy_mask,
            )
            greedy = torch.argmax(logits[:, -1, :].float(), dim=-1)
            if self.preferred_token_id is None:
                override = torch.zeros_like(policy_mask)
            else:
                override = (
                    policy_mask
                    & (selected == int(self.preferred_token_id))
                    & (greedy != int(self.preferred_token_id))
                )
            self.token_selection_override_used.copy_(
                self.token_selection_override_used | override
            )
            return selected.view(-1, 1)

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
            slot_request_ids=slot_request_ids,
            cache_positions=cache_positions,
            generated_token_counts=generated_token_counts,
        )


class ContinuousDecodeScheduler:
    """Run a ready queue through a persistent decode arena until completion.

    ``completion_policy`` is an optional decode-control seam for deterministic
    workload replay. Production leaves it unset and retains the normal
    EOS/KV-capacity/max-length behavior.
    """

    def __init__(
        self,
        *,
        arena: DecodeArena,
        decode_fn: Callable[..., torch.Tensor],
        max_new_tokens: int,
        timeline: TimelineRecorder | None = None,
        completion_policy: (
            Callable[[DecodeSlotState, int], str | None] | None
        ) = None,
        stop_repetitions: bool = False,
        progress: Callable[..., None] | None = None,
        diagnostic_effective_length: int | None = None,
        diagnostic_request_id: str | None = None,
    ) -> None:
        self.arena = arena
        self.decode_fn = decode_fn
        self.max_new_tokens = int(max_new_tokens)
        self.device = arena.device
        self.batch_size = arena.batch_size
        self.eos_token_id = arena.eos_token_id
        self.timeline = timeline
        self.completion_policy = completion_policy
        self.stop_repetitions = bool(stop_repetitions)
        self.progress = progress
        if (
            diagnostic_effective_length is not None
            and int(diagnostic_effective_length) <= 0
        ):
            raise ValueError("diagnostic_effective_length must be positive")
        self.diagnostic_effective_length = (
            None
            if diagnostic_effective_length is None
            else int(diagnostic_effective_length)
        )
        self.diagnostic_request_id = (
            None
            if diagnostic_request_id is None
            else str(diagnostic_request_id)
        )
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

    def _progress(self, event: str, **fields: Any) -> None:
        if self.progress is not None:
            self.progress(event, **fields)

    def _completion_reason(
        self,
        state: DecodeSlotState,
        token_id: int,
    ) -> str | None:
        generated_tokens = len(state.token_ids)
        cache_is_full = (
            int(state.ready.prompt_length) + generated_tokens - 1
            >= int(self.arena.cache.cache_length)
        )
        if self.completion_policy is not None:
            if cache_is_full:
                return "kv_cache_full"
            reason = self.completion_policy(state, token_id)
            if reason is not None and not reason:
                raise ValueError("completion policy returned an empty stop reason")
            return reason
        if token_id == self.eos_token_id:
            return "eos"
        if self.stop_repetitions:
            evidence = state.repetition_tracker.update(token_id)
            if evidence is not None:
                state.repetition_evidence = evidence
                return "repetition"
        if cache_is_full:
            return "kv_cache_full"
        if generated_tokens >= self.max_new_tokens:
            return "length"
        return None

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
            diagnostic_compute_event = None
            if self._diagnostic_slots(step):
                # This is separate from the copy stream's dependency event.
                # Retaining it does not change the lifetime of the production
                # ready_event; it gives the diagnostic path a precise compute
                # completion boundary to synchronize.
                diagnostic_compute_event = (
                    torch_npu.npu.current_stream().record_event()
                )
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
                slot_request_ids=step.slot_request_ids,
                cache_positions=step.cache_positions,
                generated_token_counts=step.generated_token_counts,
                ring_index=ring_index,
                done_event=done_event,
                diagnostic_compute_event=diagnostic_compute_event,
                host_tokens=None,
            )
        return PendingTokenCopy(
            iteration=iteration,
            active_slots=step.active_slots,
            slot_epochs=step.slot_epochs,
            slot_request_ids=step.slot_request_ids,
            cache_positions=step.cache_positions,
            generated_token_counts=step.generated_token_counts,
            ring_index=None,
            done_event=None,
            diagnostic_compute_event=None,
            host_tokens=[int(value) for value in step.sampled.detach().cpu().reshape(-1).tolist()],
        )

    def _diagnostic_slots(
        self,
        step: DecodeStep | PendingTokenCopy,
    ) -> list[dict[str, int | str]]:
        target_length = self.diagnostic_effective_length
        if target_length is None:
            return []
        matches: list[dict[str, int | str]] = []
        for slot, (request_id, cache_position, generated_tokens) in enumerate(
            zip(
                step.slot_request_ids,
                step.cache_positions,
                step.generated_token_counts,
            )
        ):
            if request_id is None or cache_position is None:
                continue
            if (
                self.diagnostic_request_id is not None
                and request_id != self.diagnostic_request_id
            ):
                continue
            effective_length = int(cache_position) + 1
            if effective_length != target_length:
                continue
            matches.append(
                {
                    "slot": slot,
                    "request_id": request_id,
                    "cache_position": int(cache_position),
                    "effective_length": effective_length,
                    "generated_tokens": int(generated_tokens or 0),
                }
            )
        return matches

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
        ready_requests: Iterable[ReadyDecodeRequest] | OpenReadyDecodeSource,
        *,
        on_completion: Callable[[DecodeCompletion], None] | None = None,
        ready_buffer_capacity: int | None = None,
        ready_buffer_low_watermark: int | None = None,
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ) -> ContinuousDecodeRun:
        """Decode a bounded, lazily produced request stream.

        The producer may perform eager prefill before yielding each request.  A
        private ready reservoir keeps at most ``ready_buffer_capacity`` NPU KV
        prefixes waiting behind the fixed decode arena.  This makes page
        boundaries irrelevant without materializing an unbounded document.

        A source with ``pull(block=...)`` and ``closed`` may stay open while it
        is temporarily empty. Decode continues while slots are active. The
        scheduler blocks for another request only when the arena is idle.
        """

        self.arena.begin_run()
        if hasattr(ready_requests, "pull") and hasattr(ready_requests, "closed"):
            ready_source = ready_requests
        else:
            ready_source = _IterableReadyDecodeSource(ready_requests)
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
        source_exhausted = bool(ready_source.closed)
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
        hot_swap_safety_sync_wall_s = 0.0
        ready_source_wall_s = 0.0
        completion_callback_wall_s = 0.0
        ready_queued_ns: dict[str, int] = {}
        refill_sequence = 0

        def progress(event: str, **fields: Any) -> None:
            if self.progress is None:
                return
            active = [
                {
                    "slot": index,
                    "request_id": state.ready.request_id,
                    "tokens": len(state.token_ids),
                    "prompt_length": state.ready.prompt_length,
                }
                for index, state in enumerate(self.arena.slots)
                if state is not None
            ]
            self._progress(
                event,
                active_count=len(active),
                active=active,
                ready_depth=len(ready_queue),
                source_exhausted=source_exhausted,
                submitted=len(submitted_order),
                completed=len(completions),
                **fields,
            )

        def record_completion(completion: DecodeCompletion) -> None:
            nonlocal completion_callback_wall_s
            if scheduling_metrics is not None:
                completion.scheduling_metrics = scheduling_metrics.finish(
                    completion.ready.request_id, completion.completed_at,
                )
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

        def refill_ready_queue(
            *,
            reason: str,
            block_if_idle: bool = False,
        ) -> None:
            nonlocal source_exhausted, ready_source_wall_s
            nonlocal max_ready_queue_depth, ready_source_refill_count
            nonlocal refill_sequence
            refill_sequence += 1
            refill_id = refill_sequence
            pulled = 0
            progress(
                "refill_begin",
                refill_id=refill_id,
                reason=reason,
                target_depth=buffer_capacity,
            )
            while not source_exhausted and len(ready_queue) < buffer_capacity:
                started = time.perf_counter()
                progress(
                    "ready_source_next_begin",
                    refill_id=refill_id,
                    reason=reason,
                    pull_index=pulled,
                )
                should_block = (
                    block_if_idle
                    and pulled == 0
                    and not ready_queue
                    and self.arena.num_active == 0
                )
                try:
                    ready = ready_source.pull(block=should_block)
                except BaseException as exc:
                    progress(
                        "ready_source_next_error",
                        refill_id=refill_id,
                        reason=reason,
                        pull_index=pulled,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                finished = time.perf_counter()
                ready_source_wall_s += finished - started
                if ready is None:
                    source_exhausted = bool(ready_source.closed)
                    progress(
                        (
                            "ready_source_exhausted"
                            if source_exhausted
                            else "ready_source_temporarily_empty"
                        ),
                        refill_id=refill_id,
                        reason=reason,
                        pull_index=pulled,
                        wait_s=finished - started,
                    )
                    if self.timeline is not None:
                        self.timeline.record_span_seconds(
                            "Decode control / wait",
                            (
                                "Drain ready-request source"
                                if source_exhausted
                                else "Wait for an arriving ready request"
                            ),
                            started,
                            finished,
                            event_type="wait" if should_block else "scope",
                        )
                    break
                progress(
                    "ready_source_next_end",
                    refill_id=refill_id,
                    reason=reason,
                    pull_index=pulled,
                    request_id=ready.request_id,
                    wait_s=finished - started,
                )
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
            progress(
                "refill_end",
                refill_id=refill_id,
                reason=reason,
                pulled=pulled,
            )

        def fill_free_slots(*, hot_swap: bool) -> None:
            nonlocal initial_admissions, hot_swap_admissions
            nonlocal prefill_only_completions, initial_kv_bytes, hot_swap_kv_bytes
            for slot_index in self.arena.free_slot_indices():
                while True:
                    if not ready_queue and not source_exhausted:
                        refill_ready_queue(reason="free_slot_empty_queue")
                    if not ready_queue:
                        break
                    ready = ready_queue.popleft()
                    progress(
                        "admission_begin",
                        hot_swap=hot_swap,
                        slot=slot_index,
                        request_id=ready.request_id,
                    )
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
                    prefill_stop_reason = None
                    if ready.first_token == self.eos_token_id:
                        prefill_stop_reason = "eos"
                    elif ready.prompt_length >= int(self.arena.cache.cache_length):
                        prefill_stop_reason = "kv_cache_full"
                    elif self.max_new_tokens == 1:
                        prefill_stop_reason = "length"
                    if prefill_stop_reason is not None:
                        ready.release_device_state()
                        record_completion(
                            DecodeCompletion(
                                ready=ready,
                                token_ids=[int(ready.first_token)],
                                stop_reason=prefill_stop_reason,
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
                    progress(
                        "admission_end",
                        hot_swap=hot_swap,
                        slot=slot_index,
                        request_id=ready.request_id,
                        useful_prefix_bytes=copied_bytes,
                    )
                    if hot_swap:
                        hot_swap_admissions += 1
                        hot_swap_kv_bytes += copied_bytes
                    else:
                        initial_admissions += 1
                        initial_kv_bytes += copied_bytes
                    break

        def retire_pending(
            pending_copy: PendingTokenCopy,
            *,
            iteration: int,
            refill_reason: str,
        ) -> None:
            nonlocal d2h_wait_wall_s, retire_and_refill_wall_s
            nonlocal hot_swap_safety_sync_wall_s
            progress(
                "pending_token_wait_begin",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
            )
            diagnostic_slots = self._diagnostic_slots(pending_copy)
            if diagnostic_slots:
                progress(
                    "diagnostic_pending_state",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
                if pending_copy.diagnostic_compute_event is None:
                    raise RuntimeError(
                        "targeted decode diagnostic lost its compute event"
                    )
                progress(
                    "diagnostic_compute_sync_begin",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
                compute_sync_started = time.perf_counter()
                try:
                    pending_copy.diagnostic_compute_event.synchronize()
                except BaseException as exc:
                    progress(
                        "diagnostic_compute_sync_error",
                        iteration=iteration,
                        pending_iteration=pending_copy.iteration,
                        diagnostic_slots=diagnostic_slots,
                        wait_s=time.perf_counter() - compute_sync_started,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                progress(
                    "diagnostic_compute_sync_end",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                    wait_s=time.perf_counter() - compute_sync_started,
                )
                progress(
                    "diagnostic_d2h_sync_begin",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
            try:
                host_tokens, wait_s = self._wait_tokens(pending_copy)
            except BaseException as exc:
                if diagnostic_slots:
                    progress(
                        "diagnostic_d2h_sync_error",
                        iteration=iteration,
                        pending_iteration=pending_copy.iteration,
                        diagnostic_slots=diagnostic_slots,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                raise
            if diagnostic_slots:
                progress(
                    "diagnostic_d2h_sync_end",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                    wait_s=wait_s,
                )
            progress(
                "pending_token_wait_end",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
                wait_s=wait_s,
            )
            d2h_wait_wall_s += wait_s
            wait_finished = time.perf_counter()
            if self.timeline is not None:
                self.timeline.record_span_seconds(
                    "Decode control / wait",
                    "Wait for sampled-token D2H",
                    wait_finished - wait_s,
                    wait_finished,
                    flow_id=f"decode-iteration:{pending_copy.iteration}",
                    event_type="wait",
                    args={"iteration": pending_copy.iteration},
                )
            started = time.perf_counter()
            completed_before = len(completions)
            if scheduling_metrics is not None:
                scheduling_metrics.consume(
                    state.ready.request_id
                    for slot_index, was_active in enumerate(pending_copy.active_slots)
                    if was_active
                    and (state := self.arena.slots[slot_index]) is not None
                    and state.epoch == pending_copy.slot_epochs[slot_index]
                )
            for slot_index, was_active in enumerate(pending_copy.active_slots):
                if not was_active:
                    continue
                state = self.arena.slots[slot_index]
                expected_epoch = pending_copy.slot_epochs[slot_index]
                if state is None or state.epoch != expected_epoch:
                    continue
                token_id = int(host_tokens[slot_index])
                state.token_ids.append(token_id)
                stop_reason = self._completion_reason(state, token_id)
                if stop_reason is not None:
                    released = self.arena.release(slot_index)
                    completion_tokens = list(released.token_ids)
                    repetition_evidence = released.repetition_evidence
                    if repetition_evidence is not None:
                        completion_tokens = completion_tokens[
                            : repetition_evidence.trim_length
                        ]
                    record_completion(
                        DecodeCompletion(
                            ready=released.ready,
                            token_ids=completion_tokens,
                            stop_reason=stop_reason,
                            slot_index=slot_index,
                            slot_epoch=released.epoch,
                            admitted_at=released.admitted_at,
                            first_decode_launched_at=released.first_decode_launched_at,
                            completed_at=time.perf_counter(),
                            iterations_launched=released.iterations_launched,
                            repetition_evidence=(
                                repetition_evidence.to_dict()
                                if repetition_evidence is not None
                                else None
                            ),
                        )
                    )
            progress(
                "retire_end",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
                newly_completed=len(completions) - completed_before,
            )
            newly_completed = len(completions) - completed_before
            if newly_completed and (ready_queue or not source_exhausted):
                # The next decode graph is submitted before the previous
                # sampled tokens are retired so its D2H can overlap compute.
                # A slot that just completed therefore still participates in
                # that speculative graph.  TorchAir's in-place KV writes are
                # not reliably protected from an immediately following
                # hot-swap copy by enqueue order alone on every Ascend target.
                # Resolve only the compute stream at an actual replacement
                # boundary; iterations without a hot swap remain pipelined.
                progress(
                    "hot_swap_safety_sync_begin",
                    iteration=iteration,
                    newly_completed=newly_completed,
                )
                safety_started = time.perf_counter()
                stream_synchronize(self.device)
                safety_wait_s = time.perf_counter() - safety_started
                hot_swap_safety_sync_wall_s += safety_wait_s
                progress(
                    "hot_swap_safety_sync_end",
                    iteration=iteration,
                    newly_completed=newly_completed,
                    wait_s=safety_wait_s,
                )
            progress("hot_swap_admission_begin", iteration=iteration)
            fill_free_slots(hot_swap=True)
            progress("hot_swap_admission_end", iteration=iteration)
            if len(ready_queue) < low_watermark:
                refill_ready_queue(reason=refill_reason)
            finished = time.perf_counter()
            retire_and_refill_wall_s += finished - started
            if self.timeline is not None:
                self.timeline.record_span_seconds(
                    "Decode control / wait",
                    "Retire completed slots and refill",
                    started,
                    finished,
                    flow_id=f"decode-iteration:{pending_copy.iteration}",
                    args={
                        "iteration": pending_copy.iteration,
                        "active_after_refill": self.arena.num_active,
                        "ready_queue_depth": len(ready_queue),
                    },
                )

        progress("scheduler_device_sync_begin", phase="before_initial_fill")
        synchronize(self.device)
        progress("scheduler_device_sync_end", phase="before_initial_fill")
        scheduler_started = time.perf_counter()
        progress(
            "scheduler_run_begin",
            buffer_capacity=buffer_capacity,
            low_watermark=low_watermark,
        )
        if len(ready_queue) < low_watermark:
            refill_ready_queue(
                reason="initial_low_watermark",
                block_if_idle=True,
            )
        progress("initial_admission_begin")
        fill_free_slots(hot_swap=False)
        progress("initial_admission_end")
        refill_ready_queue(reason="initial_top_up")
        pending: PendingTokenCopy | None = None
        iteration = 0

        while True:
            if self.arena.num_active == 0:
                if not ready_queue and not source_exhausted:
                    refill_ready_queue(
                        reason="idle_wait_for_request",
                        block_if_idle=True,
                    )
                if ready_queue:
                    progress("idle_admission_begin", iteration=iteration)
                    fill_free_slots(hot_swap=graph_calls > 0)
                    progress("idle_admission_end", iteration=iteration)
                    refill_ready_queue(reason="idle_top_up")
                if self.arena.num_active == 0:
                    if source_exhausted:
                        break
                    continue
            progress(
                "iteration_begin",
                iteration=iteration,
                pending_iteration=(
                    None if pending is None else pending.iteration
                ),
            )
            boundary_slots = [
                index
                for index, state in enumerate(self.arena.slots)
                if state is not None
                and int(state.ready.prompt_length) + int(state.iterations_launched)
                >= int(self.arena.cache.cache_length)
            ]
            if pending is not None and boundary_slots:
                progress(
                    "kv_cache_boundary_drain_begin",
                    iteration=iteration,
                    pending_iteration=pending.iteration,
                    slots=boundary_slots,
                )
                retire_pending(
                    pending,
                    iteration=iteration,
                    refill_reason="kv_cache_boundary",
                )
                progress(
                    "kv_cache_boundary_drain_end",
                    iteration=iteration,
                    slots=boundary_slots,
                )
                pending = None
                continue
            progress("decode_step_begin", iteration=iteration)
            if scheduling_metrics is not None:
                scheduling_metrics.step(
                    (state.ready.request_id for state in self.arena.slots if state is not None),
                    time.perf_counter(),
                )
            step = self.arena.step(self.decode_fn, iteration=iteration)
            progress("decode_step_end", iteration=iteration)
            graph_calls += 1
            active_decode_slots += sum(step.active_slots)
            progress("token_copy_schedule_begin", iteration=iteration)
            current = self._schedule_token_copy(step, iteration)
            progress("token_copy_schedule_end", iteration=iteration)

            if pending is not None:
                retire_pending(
                    pending,
                    iteration=iteration,
                    refill_reason="steady_low_watermark",
                )

            pending = current
            iteration += 1
            progress(
                "iteration_end",
                iteration=iteration - 1,
                next_iteration=iteration,
            )
            if self.arena.num_active == 0:
                progress(
                    "final_token_drain_begin",
                    iteration=iteration,
                    pending_iteration=pending.iteration,
                )
                _ignored, wait_s = self._wait_tokens(pending)
                progress(
                    "final_token_drain_end",
                    iteration=iteration,
                    pending_iteration=pending.iteration,
                    wait_s=wait_s,
                )
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
                continue

        progress("scheduler_device_sync_begin", phase="after_decode_loop")
        synchronize(self.device)
        progress("scheduler_device_sync_end", phase="after_decode_loop")
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
        ) if self.arena.decode_device_timing else None

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
                "continuous_decode_wall": continuous_decode_wall_s,
                "decode_host_exclusive_wall": float(
                    decode_host_exclusive_wall_s
                ),
                "run_scoped_scheduler_wall": float(scheduler_wall_s),
                "ready_source_wall": float(ready_source_wall_s),
                "completion_callback_wall": float(completion_callback_wall_s),
                "decode_model_and_argmax_device": (
                    float(decode_device_s) if self.arena.decode_device_timing else None
                ),
                "slot_admission_device": float(admission_device_s),
                "slot_admission_enqueue_wall": float(self.arena.admission_enqueue_wall_s),
                "d2h_wait_wall": float(d2h_wait_wall_s),
                "retire_and_refill_host_wall": float(retire_and_refill_wall_s),
                "hot_swap_safety_sync_wall": float(
                    hot_swap_safety_sync_wall_s
                ),
            },
        )
