"""Two-shape, bounded-quantum continuous decode for worker-prefilled UniRec."""

from __future__ import annotations

import heapq
import time
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import torch

from continuous_unirec import (
    ContinuousCompletedItem,
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
    ContinuousWorkerPrefilledItem,
)
from dual_lane_decode_policy import (
    DecodeLaneSpec,
    DecodeLaneStatus,
    choose_lane,
    route_lane,
)
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device


@dataclass(frozen=True)
class RankedReadyItem:
    ready: ContinuousReadyItem
    global_rank: int
    promoted_from_a: bool = False
    speculative_a_tokens: int = 0
    first_queued_at: float = 0.0
    promotion_queued_at: float | None = None


class _WorkerDecodeLane:
    """One resumable fixed-shape arena and graph."""

    def __init__(
        self,
        *,
        runner: OptimizedUniRecRunner,
        spec: DecodeLaneSpec,
        decode_mode: str,
        compile_backend: str,
        graph_warmup_passes: int,
        emit_final: Callable[[str, RankedReadyItem, dict[str, Any], int], None],
        promote_overflow: (
            Callable[[RankedReadyItem, int], bool] | None
        ) = None,
    ) -> None:
        self.runner = runner
        self.spec = spec
        self.graph_warmup_passes = int(graph_warmup_passes)
        self.emit_final = emit_final
        self.promote_overflow = promote_overflow
        self.helper = ContinuousUniRecDecoder(
            runner=runner,
            batch_size=spec.batch_size,
            max_length=spec.max_length,
            decode_mode=decode_mode,
            compile_backend=compile_backend,
            self_cache_length=spec.self_cache_length,
            cross_cache_length=spec.cross_cache_length,
        )
        self.decode_mode = decode_mode
        self.compile_backend = compile_backend
        self.queue: list[tuple[int, int, int, RankedReadyItem]] = []
        self.queue_serial = 0
        self.initialized = False
        self.cache = None
        self.decode_module = None
        self.compile_meta: dict[str, Any] | None = None
        self.compile_wrap_s: float | None = None
        self.slots: list[RankedReadyItem | None] = []
        self.token_ids: list[list[int]] = []
        self.last_tokens: list[int] = []
        self.cache_positions: list[int] = []
        self.slot_decode_counts: list[int] = []
        self.slot_active_s: list[float] = []
        self.slot_admission_indices: list[int] = []
        self.slot_admitted_at: list[float] = []
        self.slot_ever_used: list[bool] = []
        self.next_admission_index = 0
        self.next_token_host = None
        self.cache_position_host = None
        self.next_token_host_array = None
        self.cache_position_host_array = None
        self.next_token_tensor = None
        self.cache_position_tensor = None
        self.decode_input_host_pinned = False
        self.mean_step_ms: float | None = None
        self.decode_iterations = 0
        self.raw_decode_token_slots = 0
        self.active_decode_token_slots = 0
        self.committed_decode_tokens = 0
        self.speculative_discarded_tokens = 0
        self.idle_decode_token_slots = 0
        self.decode_s = 0.0
        self.quantum_count = 0
        self.slot_refills = 0
        self.final_completions = 0
        self.promotions = 0
        self.direct_admission_count = 0
        self.direct_cross_kv_bytes = 0
        self.direct_reset_bytes = 0
        self.initial_arena_allocate_s = 0.0
        self.initial_arena_admission_enqueue_s = 0.0
        self.cache_refill_direct_admission_enqueue_s = 0.0
        self.decode_input_build_s = 0.0
        self.result_build_s = 0.0
        self.completion_callback_s = 0.0
        self.production_graph_warmup_s = 0.0
        self.production_graph_warmup_pass_s: list[float] = []
        self.production_graph_warmup_warnings: list[str] = []

    def enqueue(self, item: RankedReadyItem, *, promotion: bool = False) -> None:
        priority = 0 if promotion else 1
        heapq.heappush(
            self.queue,
            (priority, item.global_rank, self.queue_serial, item),
        )
        self.queue_serial += 1

    def _pop(self) -> RankedReadyItem | None:
        if not self.queue:
            return None
        return heapq.heappop(self.queue)[-1]

    def _copy_valid_row(self, source_slot: int, destination_slot: int) -> None:
        if self.cache is None:
            raise RuntimeError("decode lane arena is not allocated")
        for tensor_group in (
            self.cache.key_cache,
            self.cache.value_cache,
            self.cache.cross_key_cache or (),
            self.cache.cross_value_cache or (),
        ):
            for tensor in tensor_group:
                tensor[destination_slot : destination_slot + 1].copy_(
                    tensor[source_slot : source_slot + 1]
                )
        self.cache.cross_attention_mask[
            destination_slot : destination_slot + 1
        ].copy_(self.cache.cross_attention_mask[source_slot : source_slot + 1])

    def _admit(self, slot: int, item: RankedReadyItem) -> None:
        if self.cache is None:
            raise RuntimeError("decode lane arena is not allocated")
        source = item.ready.prefilled
        if not isinstance(source, ContinuousWorkerPrefilledItem):
            raise RuntimeError("dual-lane decode requires worker-prefilled rows")
        enqueue_s, transferred_bytes, reset_bytes = self.helper._admit_worker_row(
            self.cache,
            slot,
            source,
            reset_reused_row=self.slot_ever_used[slot],
        )
        if self.slot_ever_used[slot]:
            self.cache_refill_direct_admission_enqueue_s += enqueue_s
            self.slot_refills += 1
        else:
            self.initial_arena_admission_enqueue_s += enqueue_s
        self.direct_admission_count += 1
        self.direct_cross_kv_bytes += transferred_bytes
        self.direct_reset_bytes += reset_bytes
        self.slots[slot] = item
        row = self.helper._initial_token_ids(
            item.ready,
            decoder_start_token_id=int(self.runner.config.decoder_start_token_id),
        )
        self.token_ids[slot] = row
        self.last_tokens[slot] = row[-1]
        self.cache_positions[slot] = len(row) - 1
        self.slot_decode_counts[slot] = 0
        self.slot_active_s[slot] = 0.0
        self.slot_admission_indices[slot] = self.next_admission_index
        self.slot_admitted_at[slot] = time.perf_counter()
        self.next_admission_index += 1
        self.slot_ever_used[slot] = True

    def _make_inactive(self, slot: int) -> None:
        eos = int(self.runner.config.eos_token_id)
        start = int(self.runner.config.decoder_start_token_id)
        self.slots[slot] = None
        self.token_ids[slot] = [start, eos]
        self.last_tokens[slot] = eos
        self.cache_positions[slot] = 1
        self.slot_decode_counts[slot] = 0
        self.slot_active_s[slot] = 0.0
        self.slot_admission_indices[slot] = -1
        self.slot_admitted_at[slot] = 0.0

    def _fill_available_slots(self) -> None:
        if not self.initialized:
            return
        for slot in range(self.spec.batch_size):
            if self.slots[slot] is not None:
                continue
            item = self._pop()
            if item is None:
                return
            self._admit(slot, item)

    def initialize(self) -> None:
        if self.initialized or not self.queue:
            return
        allocate_started = time.perf_counter()
        self.cache = self.helper._allocate_empty_arena()
        self.initial_arena_allocate_s = time.perf_counter() - allocate_started
        batch = self.spec.batch_size
        self.slots = [None for _ in range(batch)]
        self.token_ids = [[] for _ in range(batch)]
        self.last_tokens = [0 for _ in range(batch)]
        self.cache_positions = [0 for _ in range(batch)]
        self.slot_decode_counts = [0 for _ in range(batch)]
        self.slot_active_s = [0.0 for _ in range(batch)]
        self.slot_admission_indices = [-1 for _ in range(batch)]
        self.slot_admitted_at = [0.0 for _ in range(batch)]
        self.slot_ever_used = [False for _ in range(batch)]
        self.initialized = True
        self._fill_available_slots()
        active_slots = [
            index for index, item in enumerate(self.slots) if item is not None
        ]
        if not active_slots:
            raise RuntimeError("decode lane initialization admitted no rows")
        last_valid = active_slots[-1]
        for slot in range(batch):
            if self.slots[slot] is None:
                self._copy_valid_row(last_valid, slot)
                self._make_inactive(slot)

        compile_started = time.perf_counter()
        self.decode_module, self.compile_meta = self.runner._compile_decode_module(
            backend=self.compile_backend,
            self_attention_backend=(
                "increfa_all" if self.decode_mode == "compiled_ifa" else "eager"
            ),
            compile_dynamic=False,
            cross_cache_len=self.spec.cross_cache_length,
            batch_size=self.spec.batch_size,
            self_cache_len=self.spec.self_cache_length,
        )
        self.compile_wrap_s = time.perf_counter() - compile_started
        try:
            self.next_token_host = torch.empty(
                batch,
                dtype=torch.long,
                pin_memory=True,
            )
            self.cache_position_host = torch.empty(
                batch,
                dtype=torch.int64,
                pin_memory=True,
            )
            self.decode_input_host_pinned = True
        except RuntimeError:
            self.next_token_host = torch.empty(batch, dtype=torch.long)
            self.cache_position_host = torch.empty(batch, dtype=torch.int64)
        self.next_token_host_array = self.next_token_host.numpy()
        self.cache_position_host_array = self.cache_position_host.numpy()
        self.next_token_tensor, self.cache_position_tensor = (
            self.helper._allocate_decode_device_inputs(batch, self.runner.device)
        )
        self._warm_graph()

    def _write_inputs(self) -> None:
        started = time.perf_counter()
        self.next_token_host_array[:] = self.last_tokens
        self.cache_position_host_array[:] = self.cache_positions
        self.next_token_tensor.view(-1).copy_(
            self.next_token_host,
            non_blocking=self.decode_input_host_pinned,
        )
        self.cache_position_tensor.copy_(
            self.cache_position_host,
            non_blocking=self.decode_input_host_pinned,
        )
        self.decode_input_build_s += time.perf_counter() - started

    def _warm_graph(self) -> None:
        if self.graph_warmup_passes <= 0:
            return
        started = time.perf_counter()
        with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._write_inputs()
            for pass_index in range(self.graph_warmup_passes):
                pass_started = time.perf_counter()
                _ = self.decode_module(
                    self.next_token_tensor,
                    self.cache_position_tensor,
                    0,
                    self.cache.key_cache,
                    self.cache.value_cache,
                    self.cache.cross_key_cache,
                    self.cache.cross_value_cache,
                    self.cache.cross_attention_mask,
                )
                synchronize_device(self.runner.device)
                pass_s = time.perf_counter() - pass_started
                self.production_graph_warmup_pass_s.append(pass_s)
                print(
                    "UNIREC_DUAL_DECODE_WARMUP_PASS "
                    f"lane={self.spec.name} "
                    f"pass={pass_index + 1}/{self.graph_warmup_passes} "
                    f"wall_s={pass_s:.6f}",
                    flush=True,
                )
            self.production_graph_warmup_warnings = [
                str(warning.message) for warning in caught
            ]
        self.production_graph_warmup_s = time.perf_counter() - started
        recompile_warnings = [
            message
            for message in self.production_graph_warmup_warnings
            if "recompiled" in message.lower()
        ]
        if recompile_warnings:
            raise RuntimeError(
                f"dual decode lane {self.spec.name} recompiled instead of "
                f"loading its static graph: {recompile_warnings[0]}"
            )

    def prepare(self) -> None:
        if not self.initialized:
            self.initialize()
        self._fill_available_slots()

    @property
    def active_count(self) -> int:
        return sum(item is not None for item in self.slots)

    @property
    def runnable(self) -> bool:
        return bool(self.queue) or self.active_count > 0

    def status(self, skipped_quanta: int) -> DecodeLaneStatus:
        return DecodeLaneStatus(
            name=self.spec.name,
            capacity=self.spec.batch_size,
            active_slots=self.active_count,
            queued_items=len(self.queue),
            mean_step_ms=self.mean_step_ms,
            skipped_quanta=skipped_quanta,
        )

    def _finish_slot(self, slot: int, *, eos: bool) -> None:
        item = self.slots[slot]
        if item is None:
            return
        decode_count = self.slot_decode_counts[slot]
        if not eos and self.promote_overflow is not None:
            if self.promote_overflow(item, decode_count):
                self.promotions += 1
                self.speculative_discarded_tokens += decode_count
                self._make_inactive(slot)
                return
        result_started = time.perf_counter()
        result = self.helper._build_result(
            ready=item.ready,
            token_ids=self.token_ids[slot],
            decode_token_count=decode_count,
            decode_active_s=self.slot_active_s[slot],
            compile_wrap_s=self.compile_wrap_s,
            compile_meta=self.compile_meta,
            cross_cache_len=self.spec.cross_cache_length,
        )
        result.update(
            {
                "decode_lane": self.spec.name,
                "decode_global_rank": item.global_rank,
                "decode_promoted_from_a": item.promoted_from_a,
                "decode_speculative_a_tokens": item.speculative_a_tokens,
                "decode_lane_queue_wait_s": max(
                    0.0,
                    self.slot_admitted_at[slot] - item.first_queued_at,
                ),
                "decode_promotion_wait_s": (
                    None
                    if item.promotion_queued_at is None
                    else max(
                        0.0,
                        self.slot_admitted_at[slot] - item.promotion_queued_at,
                    )
                ),
            }
        )
        self.result_build_s += time.perf_counter() - result_started
        callback_started = time.perf_counter()
        self.emit_final(self.spec.name, item, result, slot)
        self.completion_callback_s += time.perf_counter() - callback_started
        self.committed_decode_tokens += decode_count
        self.final_completions += 1
        self._make_inactive(slot)

    def run_quantum(self, steps: int) -> dict[str, Any]:
        if steps < 1:
            raise ValueError("decode quantum steps must be positive")
        self.prepare()
        if self.active_count == 0:
            return {"steps": 0, "wall_s": 0.0, "active_tokens": 0}
        quantum_started = time.perf_counter()
        completed_before = self.final_completions
        promoted_before = self.promotions
        active_tokens = 0
        actual_steps = 0
        with torch.inference_mode():
            for _ in range(steps):
                self._fill_available_slots()
                active_slots = [item is not None for item in self.slots]
                active_count = sum(active_slots)
                if active_count == 0:
                    break
                self._write_inputs()
                step_started = time.perf_counter()
                logits = self.decode_module(
                    self.next_token_tensor,
                    self.cache_position_tensor,
                    0,
                    self.cache.key_cache,
                    self.cache.value_cache,
                    self.cache.cross_key_cache,
                    self.cache.cross_value_cache,
                    self.cache.cross_attention_mask,
                )
                predicted = self.runner.model.select_next_token(logits)
                predicted_ids = [
                    int(token)
                    for token in predicted.detach().cpu().view(-1).tolist()
                ]
                step_s = time.perf_counter() - step_started
                actual_steps += 1
                active_tokens += active_count
                self.decode_iterations += 1
                self.raw_decode_token_slots += self.spec.batch_size
                self.active_decode_token_slots += active_count
                self.idle_decode_token_slots += self.spec.batch_size - active_count
                self.decode_s += step_s
                step_ms = step_s * 1000.0
                self.mean_step_ms = (
                    step_ms
                    if self.mean_step_ms is None
                    else 0.9 * self.mean_step_ms + 0.1 * step_ms
                )
                completed_slots: list[tuple[int, bool]] = []
                eos_token_id = int(self.runner.config.eos_token_id)
                for slot, is_active in enumerate(active_slots):
                    if not is_active:
                        continue
                    token = predicted_ids[slot]
                    self.token_ids[slot].append(token)
                    self.last_tokens[slot] = token
                    self.cache_positions[slot] += 1
                    self.slot_decode_counts[slot] += 1
                    self.slot_active_s[slot] += step_s
                    eos = token == eos_token_id
                    if eos or len(self.token_ids[slot]) >= self.spec.max_length:
                        completed_slots.append((slot, eos))
                for slot, eos in completed_slots:
                    self._finish_slot(slot, eos=eos)
                self._fill_available_slots()
        self.quantum_count += 1
        return {
            "steps": actual_steps,
            "wall_s": time.perf_counter() - quantum_started,
            "active_tokens": active_tokens,
            "completed": self.final_completions - completed_before,
            "promoted": self.promotions - promoted_before,
            "active_after": self.active_count,
            "queued_after": len(self.queue),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "batch_size": self.spec.batch_size,
            "self_cache_length": self.spec.self_cache_length,
            "cross_cache_length": self.spec.cross_cache_length,
            "max_length": self.spec.max_length,
            "decode_iterations": self.decode_iterations,
            "raw_decode_token_slots": self.raw_decode_token_slots,
            "active_decode_token_slots": self.active_decode_token_slots,
            "committed_decode_tokens": self.committed_decode_tokens,
            "speculative_discarded_tokens": self.speculative_discarded_tokens,
            "idle_decode_token_slots": self.idle_decode_token_slots,
            "decode_s": self.decode_s,
            "mean_step_ms_ema": self.mean_step_ms,
            "quantum_count": self.quantum_count,
            "slot_refills": self.slot_refills,
            "final_completions": self.final_completions,
            "promotions": self.promotions,
            "compile": self.compile_meta,
            "compile_wrap_s": self.compile_wrap_s,
            "production_graph_warmup": {
                "passes": self.graph_warmup_passes,
                "wall_s": self.production_graph_warmup_s,
                "pass_wall_s": self.production_graph_warmup_pass_s,
                "warnings": self.production_graph_warmup_warnings,
                "arena": "actual_admitted_decode_arena",
                "included_in_decode_s": False,
            },
            "timing_detail": {
                "initial_arena_allocate_s": self.initial_arena_allocate_s,
                "initial_arena_admission_enqueue_s": (
                    self.initial_arena_admission_enqueue_s
                ),
                "cache_refill_direct_admission_enqueue_s": (
                    self.cache_refill_direct_admission_enqueue_s
                ),
                "direct_admission_count": self.direct_admission_count,
                "direct_cross_kv_bytes": self.direct_cross_kv_bytes,
                "direct_reset_bytes": self.direct_reset_bytes,
                "decode_input_build_s": self.decode_input_build_s,
                "result_build_s": self.result_build_s,
                "completion_callback_s": self.completion_callback_s,
            },
        }


class DualLaneContinuousUniRecDecoder:
    """Interleave two fixed decode shapes in bounded token-step quanta."""

    def __init__(
        self,
        *,
        runner: OptimizedUniRecRunner,
        a_spec: DecodeLaneSpec,
        b_spec: DecodeLaneSpec,
        quantum_steps: int = 16,
        max_skipped_quanta: int = 8,
        overflow_policy: str = "restart_b",
        decode_mode: str = "compiled_ifa",
        compile_backend: str = "torchair",
    ) -> None:
        if a_spec.name != "a" or b_spec.name != "b":
            raise ValueError("dual decode lane names must be 'a' and 'b'")
        if a_spec.cross_cache_length >= b_spec.cross_cache_length:
            raise ValueError("lane A cross capacity must be smaller than lane B")
        if quantum_steps < 1 or max_skipped_quanta < 1:
            raise ValueError("quantum settings must be positive")
        if overflow_policy not in ("finish_at_cap", "restart_b"):
            raise ValueError(f"unsupported overflow policy: {overflow_policy}")
        self.runner = runner
        self.a_spec = a_spec
        self.b_spec = b_spec
        self.quantum_steps = int(quantum_steps)
        self.max_skipped_quanta = int(max_skipped_quanta)
        self.overflow_policy = overflow_policy
        self.decode_mode = decode_mode
        self.compile_backend = compile_backend

    def run(
        self,
        items: Iterable[RankedReadyItem],
        *,
        on_complete: Callable[[ContinuousCompletedItem], None],
        graph_warmup_passes: int = 0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        completion_index = 0
        promoted_count = 0

        def emit_final(
            lane_name: str,
            item: RankedReadyItem,
            result: dict[str, Any],
            slot: int,
        ) -> None:
            nonlocal completion_index
            on_complete(
                ContinuousCompletedItem(
                    request_id=item.ready.request_id,
                    payload=item.ready.payload,
                    result=result,
                    slot=slot,
                    admission_index=item.global_rank,
                    completion_index=completion_index,
                )
            )
            completion_index += 1

        b_lane: _WorkerDecodeLane

        def promote_from_a(item: RankedReadyItem, decode_tokens: int) -> bool:
            nonlocal promoted_count
            if self.overflow_policy != "restart_b":
                return False
            promoted_count += 1
            b_lane.enqueue(
                replace(
                    item,
                    promoted_from_a=True,
                    speculative_a_tokens=(
                        item.speculative_a_tokens + int(decode_tokens)
                    ),
                    promotion_queued_at=time.perf_counter(),
                ),
                promotion=True,
            )
            return True

        a_lane = _WorkerDecodeLane(
            runner=self.runner,
            spec=self.a_spec,
            decode_mode=self.decode_mode,
            compile_backend=self.compile_backend,
            graph_warmup_passes=graph_warmup_passes,
            emit_final=emit_final,
            promote_overflow=promote_from_a,
        )
        b_lane = _WorkerDecodeLane(
            runner=self.runner,
            spec=self.b_spec,
            decode_mode=self.decode_mode,
            compile_backend=self.compile_backend,
            graph_warmup_passes=graph_warmup_passes,
            emit_final=emit_final,
        )
        submitted = 0
        routed_a = 0
        routed_b = 0
        now = time.perf_counter()
        for item in items:
            if item.first_queued_at == 0.0:
                item = replace(item, first_queued_at=now)
            source_len = int(item.ready.prefilled.actual_cross_attention_length)
            lane_name = route_lane(
                cross_length=source_len,
                a_cross_capacity=self.a_spec.cross_cache_length,
            )
            if lane_name == "a":
                a_lane.enqueue(item)
                routed_a += 1
            else:
                if source_len > self.b_spec.cross_cache_length:
                    raise RuntimeError(
                        "worker cross-KV exceeds both decode lanes: "
                        f"request={item.ready.request_id} source={source_len} "
                        f"b_capacity={self.b_spec.cross_cache_length}"
                    )
                b_lane.enqueue(item)
                routed_b += 1
            submitted += 1
        if submitted == 0:
            raise ValueError("dual-lane decode received no items")

        # Load both exact graphs and warm both real arenas before measured work.
        a_lane.initialize()
        b_lane.initialize()
        warmup_s = (
            a_lane.production_graph_warmup_s
            + b_lane.production_graph_warmup_s
        )
        measured_started = time.perf_counter()
        round_robin_next = "a"
        skipped = {"a": 0, "b": 0}
        last_lane: str | None = None
        graph_switches = 0
        scheduler_quanta = 0
        lane_quantum_steps = {"a": 0, "b": 0}
        while True:
            a_lane.prepare()
            b_lane.prepare()
            if not a_lane.runnable and not b_lane.runnable:
                break
            lane_name = choose_lane(
                a_lane.status(skipped["a"]),
                b_lane.status(skipped["b"]),
                round_robin_next=round_robin_next,
                max_skipped_quanta=self.max_skipped_quanta,
            )
            lane = a_lane if lane_name == "a" else b_lane
            other_name = "b" if lane_name == "a" else "a"
            if last_lane is not None and last_lane != lane_name:
                graph_switches += 1
            quantum = lane.run_quantum(self.quantum_steps)
            if quantum["steps"] <= 0:
                raise RuntimeError(f"runnable lane {lane_name} made no progress")
            scheduler_quanta += 1
            lane_quantum_steps[lane_name] += int(quantum["steps"])
            skipped[lane_name] = 0
            other = b_lane if lane_name == "a" else a_lane
            if other.runnable:
                skipped[other_name] += 1
            else:
                skipped[other_name] = 0
            if a_lane.active_count == a_lane.spec.batch_size and (
                b_lane.active_count == b_lane.spec.batch_size
            ):
                round_robin_next = other_name
            last_lane = lane_name
        measured_wall_s = time.perf_counter() - measured_started
        a_summary = a_lane.summary()
        b_summary = b_lane.summary()
        decode_s = a_lane.decode_s + b_lane.decode_s
        raw_slots = (
            a_lane.raw_decode_token_slots + b_lane.raw_decode_token_slots
        )
        active_slots = (
            a_lane.active_decode_token_slots + b_lane.active_decode_token_slots
        )
        committed_tokens = (
            a_lane.committed_decode_tokens + b_lane.committed_decode_tokens
        )
        speculative_tokens = a_lane.speculative_discarded_tokens
        idle_slots = a_lane.idle_decode_token_slots + b_lane.idle_decode_token_slots
        admission_timing = {
            key: sum(
                float(summary["timing_detail"].get(key, 0.0))
                for summary in (a_summary, b_summary)
            )
            for key in (
                "initial_arena_allocate_s",
                "initial_arena_admission_enqueue_s",
                "cache_refill_direct_admission_enqueue_s",
                "direct_admission_count",
                "direct_cross_kv_bytes",
                "direct_reset_bytes",
                "decode_input_build_s",
                "result_build_s",
                "completion_callback_s",
            )
        }
        return {
            "scheduler": "dual_lane_bounded_quantum",
            "submitted": submitted,
            "completed": completion_index,
            "routed_a": routed_a,
            "routed_b": routed_b,
            "promoted_a_to_b": promoted_count,
            "overflow_policy": self.overflow_policy,
            "quantum_steps": self.quantum_steps,
            "max_skipped_quanta": self.max_skipped_quanta,
            "scheduler_quanta": scheduler_quanta,
            "graph_switches": graph_switches,
            "lane_quantum_steps": lane_quantum_steps,
            "decode_iterations": (
                a_lane.decode_iterations + b_lane.decode_iterations
            ),
            "raw_decode_token_slots": raw_slots,
            "active_decode_token_slots": active_slots,
            "effective_decode_tokens": committed_tokens,
            "speculative_discarded_tokens": speculative_tokens,
            "idle_decode_token_slots": idle_slots,
            "decode_s": decode_s,
            "decode_wall_s": measured_wall_s,
            "raw_decode_tokens_per_s": (
                raw_slots / decode_s if decode_s > 0 else None
            ),
            "effective_decode_tokens_per_s": (
                committed_tokens / decode_s if decode_s > 0 else None
            ),
            "wall_effective_decode_tokens_per_s": (
                committed_tokens / measured_wall_s
                if measured_wall_s > 0
                else None
            ),
            "lanes": {"a": a_summary, "b": b_summary},
            "production_graph_warmup": {
                "passes": graph_warmup_passes,
                "wall_s": warmup_s,
                "included_in_decode_s": False,
                "lanes": {
                    "a": a_summary["production_graph_warmup"],
                    "b": b_summary["production_graph_warmup"],
                },
            },
            "timing_detail": {
                "run_wall_s": time.perf_counter() - started,
                **admission_timing,
            },
        }
