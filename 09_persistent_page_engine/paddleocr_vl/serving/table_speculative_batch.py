"""Cross-request batched table speculative verification.

Each adaptive-K lane owns one static batched KV arena. Requests migrate between
lanes when their K policy changes. All generation state remains native token
IDs and NPU KV tensors; generated text is never encoded back into tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Iterable, Sequence

import torch

from ..model.text_decode import TextDecodeRuntime
from ..model.text_spec_verify import TextSpecVerifyRuntime
from .repetition import ExactCycleTracker
from .table_speculative import (
    DraftProposal,
    TableDraftMatcher,
    TableSpecDecodeResult,
)


@dataclass
class BatchedAdaptiveKTableSpecDecodeResult(TableSpecDecodeResult):
    adaptive_k: dict[str, Any] = field(default_factory=dict)
    batching: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RequestState:
    request_id: str
    matcher: TableDraftMatcher
    input_tokens: int
    skip_special_tokens: bool
    token_ids: list[int]
    position: int
    policy_k: int
    max_new_tokens: int
    admitted_at: float
    tracker: ExactCycleTracker = field(default_factory=ExactCycleTracker)
    stop_reason: str | None = None
    repetition: dict[str, Any] = field(default_factory=dict)
    target_calls: int = 0
    speculative_calls: int = 0
    fully_accepted_speculative_calls: int = 0
    rejected_speculative_calls: int = 0
    fallback_calls: int = 0
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    verifier_wall_s: float = 0.0
    fallback_wall_s: float = 0.0
    completed_at: float | None = None
    lane_k: int | None = None
    lane_slot: int | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    transitions: Counter[str] = field(default_factory=Counter)
    per_k: dict[int, dict[str, int | float]] = field(default_factory=dict)


class _VerifierPool:
    def __init__(
        self,
        runtime: TextSpecVerifyRuntime,
        *,
        batch_size: int,
        eos_token_id: int,
        device: torch.device,
    ) -> None:
        self.runtime = runtime
        self.batch_size = int(batch_size)
        self.query_length = int(runtime.query_length)
        self.eos_token_id = int(eos_token_id)
        self.cache = runtime.warm_cache
        self.slots: list[_RequestState | None] = [None] * self.batch_size
        self.host_input = torch.empty(
            (self.batch_size, self.query_length),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.host_input_numpy = self.host_input.numpy()
        self.device_input = torch.empty(
            (self.batch_size, self.query_length),
            device=device,
            dtype=torch.int64,
        )
        self.host_targets = torch.empty_like(self.host_input)
        self.cache_position = torch.zeros(
            (self.batch_size,), device=device, dtype=torch.int64
        )
        self.host_cache_position = torch.zeros(
            (self.batch_size,), dtype=torch.int64, pin_memory=True
        )
        self.host_cache_position_numpy = self.host_cache_position.numpy()
        self.rope_deltas = torch.zeros(
            (self.batch_size, 1), device=device, dtype=torch.int64
        )

    def active_slots(self) -> list[int]:
        return [index for index, state in enumerate(self.slots) if state is not None]

    def free_slot(self) -> int:
        for index, state in enumerate(self.slots):
            if state is None:
                return index
        raise RuntimeError("batched verifier lane has no free slot")


class BatchedAdaptiveKTableSpeculativeDecodeRuntime:
    """Verify independent table drafts in shared fixed-B graph calls."""

    def __init__(
        self,
        recognizer: Any,
        *,
        batch_size: int,
        k_values: tuple[int, ...],
        initial_k: int,
        cache_roots: dict[int, Any],
        verifier_optimization: str,
        decode_cache_root: Any,
    ) -> None:
        if int(batch_size) <= 1:
            raise ValueError("batched speculative verification requires batch_size > 1")
        if int(recognizer.batch_size) not in (1, int(batch_size)):
            raise ValueError(
                "target recognizer batch size must be one or the verifier batch size"
            )
        normalized_k = tuple(sorted({int(value) for value in k_values}))
        if not normalized_k or any(value <= 0 for value in normalized_k):
            raise ValueError("k_values must contain positive integers")
        if int(initial_k) not in normalized_k:
            raise ValueError("initial_k must be one of k_values")
        if set(cache_roots) != set(normalized_k):
            raise ValueError("cache_roots must contain exactly one entry per K")
        if str(recognizer.token_selection) != "greedy":
            raise NotImplementedError(
                "batched table speculation currently preserves ordinary greedy only"
            )

        self.recognizer = recognizer
        self.device = recognizer.device
        self.batch_size = int(batch_size)
        self.k_values = normalized_k
        self.initial_k = int(initial_k)
        self.cache_length = int(recognizer.cache_length)
        self.eos_token_id = int(recognizer.model.config.eos_token_id)
        self.stream = torch.npu.current_stream(self.device)
        self.pools: dict[int, _VerifierPool] = {}
        for value in self.k_values:
            runtime = TextSpecVerifyRuntime(
                recognizer.model,
                batch_size=self.batch_size,
                device=self.device,
                cache_root=cache_roots[value],
                draft_length=value,
                cache_length=self.cache_length,
                dtype=recognizer.dtype,
                model_dir=recognizer.model_dir,
                linear_weight_format=str(
                    recognizer.weight_format["effective_mode"]
                ),
                optimization=verifier_optimization,
                token_selection=recognizer.token_selection,
                preferred_token_id=recognizer.math_open_token_id,
                alternate_preferred_token_id=recognizer.math_slash_token_id,
                cell_start_token_ids=recognizer.table_cell_token_ids,
            )
            self.pools[value] = _VerifierPool(
                runtime,
                batch_size=self.batch_size,
                eos_token_id=self.eos_token_id,
                device=self.device,
            )

        self.decode = TextDecodeRuntime(
            recognizer.model,
            backend="torchair",
            device=self.device,
            cache_root=decode_cache_root,
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            dtype=recognizer.dtype,
            model_dir=recognizer.model_dir,
            linear_weight_format=str(recognizer.weight_format["effective_mode"]),
            optimization=recognizer.decode_optimization,
        )
        self.decode_device_input = torch.empty(
            (self.batch_size, 1), device=self.device, dtype=torch.int64
        )
        self.decode_host_input = torch.empty(
            (self.batch_size, 1), dtype=torch.int64, pin_memory=True
        )
        self.decode_host_input_numpy = self.decode_host_input.numpy()
        self.decode_host_targets = torch.empty_like(self.decode_host_input)
        self.last_summary: dict[str, Any] = {}

    @staticmethod
    def _next_k(
        current: int,
        *,
        fully_accepted: bool,
        k_values: tuple[int, ...],
    ) -> int:
        index = k_values.index(int(current))
        if fully_accepted:
            return k_values[min(len(k_values) - 1, index + 1)]
        return k_values[max(0, index - 1)]

    def _copy_cache_row(
        self,
        source_cache: Any,
        source_slot: int,
        destination_cache: Any,
        destination_slot: int,
        *,
        prefix_length: int,
    ) -> int:
        prefix_length = int(prefix_length)
        if not 0 < prefix_length <= self.cache_length:
            raise ValueError(
                f"invalid KV prefix length {prefix_length} for "
                f"cache length {self.cache_length}"
            )
        source_tensors = tuple(
            tensor[source_slot : source_slot + 1, :, :prefix_length, :]
            for tensor in source_cache.logical_tensors()
        )
        destination_tensors = tuple(
            tensor[destination_slot : destination_slot + 1, :, :prefix_length, :]
            for tensor in destination_cache.logical_tensors()
        )
        if len(source_tensors) != len(destination_tensors):
            raise ValueError("source and destination caches have different layers")
        if any(
            source.shape != destination.shape
            for source, destination in zip(
                source_tensors, destination_tensors, strict=True
            )
        ):
            raise ValueError("source and destination cache rows have different shapes")
        torch._foreach_copy_(destination_tensors, source_tensors)
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in destination_tensors
        )

    def _admit(
        self,
        prefilled: Any,
        matcher: TableDraftMatcher,
        *,
        max_new_tokens: int,
        admitted_at: float,
    ) -> tuple[_RequestState, int]:
        input_tokens = int(prefilled.input_tokens)
        skip_special_tokens = bool(prefilled.skip_special_tokens)
        first_token = int(prefilled.first_token)
        request_id = str(prefilled.request_id)
        cache, rope_deltas, cache_position, _next_token, cache_release = (
            prefilled.take_device_state()
        )
        position = int(cache_position.detach().cpu().reshape(-1)[0].item())
        pool = self.pools[self.initial_k]
        slot = pool.free_slot()
        copied_bytes = self._copy_cache_row(
            cache,
            0,
            pool.cache,
            slot,
            prefix_length=position,
        )
        pool.rope_deltas[slot : slot + 1].copy_(rope_deltas)
        pool.cache_position[slot : slot + 1].copy_(cache_position.reshape(1))
        if cache_release is not None:
            cache_release()

        matcher.start(first_token)
        state = _RequestState(
            request_id=request_id,
            matcher=matcher,
            input_tokens=input_tokens,
            skip_special_tokens=skip_special_tokens,
            token_ids=[first_token],
            position=position,
            policy_k=self.initial_k,
            max_new_tokens=int(max_new_tokens),
            admitted_at=float(admitted_at),
            lane_k=self.initial_k,
            lane_slot=slot,
            per_k={
                value: {
                    "calls": 0,
                    "fully_accepted_calls": 0,
                    "rejected_calls": 0,
                    "proposed_tokens": 0,
                    "accepted_tokens": 0,
                    "call_wall_s": 0.0,
                }
                for value in self.k_values
            },
        )
        state.tracker.update(first_token)
        if first_token == self.eos_token_id:
            state.stop_reason = "eos"
        pool.slots[slot] = state
        return state, copied_bytes

    def _migrate(self, state: _RequestState, destination_k: int) -> int:
        source_k = state.lane_k
        source_slot = state.lane_slot
        if source_k is None or source_slot is None:
            raise RuntimeError("request state is not resident in a verifier lane")
        if int(source_k) == int(destination_k):
            return 0
        source = self.pools[int(source_k)]
        destination = self.pools[int(destination_k)]
        destination_slot = destination.free_slot()
        copied_bytes = self._copy_cache_row(
            source.cache,
            int(source_slot),
            destination.cache,
            destination_slot,
            prefix_length=state.position,
        )
        destination.rope_deltas[destination_slot : destination_slot + 1].copy_(
            source.rope_deltas[int(source_slot) : int(source_slot) + 1]
        )
        destination.cache_position[
            destination_slot : destination_slot + 1
        ].copy_(
            source.cache_position[int(source_slot) : int(source_slot) + 1]
        )
        source.slots[int(source_slot)] = None
        destination.slots[destination_slot] = state
        state.lane_k = int(destination_k)
        state.lane_slot = int(destination_slot)
        return copied_bytes

    def _append_tokens(self, state: _RequestState, values: Iterable[int]) -> None:
        for value in values:
            token = int(value)
            state.token_ids.append(token)
            if token == self.eos_token_id:
                state.stop_reason = "eos"
                return
            evidence = state.tracker.update(token)
            if evidence is not None:
                state.repetition = evidence.to_dict()
                del state.token_ids[evidence.trim_length :]
                state.stop_reason = "repetition"
                return
            if state.input_tokens + len(state.token_ids) - 1 >= self.cache_length:
                state.stop_reason = "kv_cache_full"
                return
            if len(state.token_ids) >= state.max_new_tokens:
                state.stop_reason = "length"
                return

    def _run_decode_pool(
        self,
        pool: _VerifierPool,
        active_slots: list[int],
    ) -> float:
        self.decode_host_input_numpy.fill(self.eos_token_id)
        pool.host_cache_position_numpy.fill(0)
        for slot in active_slots:
            state = pool.slots[slot]
            assert state is not None
            self.decode_host_input_numpy[slot, 0] = int(state.token_ids[-1])
            pool.host_cache_position_numpy[slot] = int(state.position)
        self.decode_device_input.copy_(self.decode_host_input, non_blocking=True)
        pool.cache_position.copy_(pool.host_cache_position, non_blocking=True)
        started = time.perf_counter()
        output = self.decode.fn(
            self.decode_device_input,
            pool.cache_position,
            pool.rope_deltas,
            *pool.cache.flat_tensors(),
        )
        if hasattr(self.recognizer.model, "decode_token_id_map"):
            targets = output.reshape(self.batch_size, 1)
        else:
            targets = torch.argmax(output[:, -1, :], dim=-1).view(-1, 1)
        self.decode_host_targets.copy_(targets, non_blocking=True)
        self.stream.synchronize()
        call_wall_s = time.perf_counter() - started
        for slot in active_slots:
            state = pool.slots[slot]
            assert state is not None
            next_token = int(self.decode_host_targets[slot, 0])
            state.target_calls += 1
            state.fallback_calls += 1
            state.fallback_wall_s += call_wall_s
            state.matcher.commit(
                None,
                accepted_draft_tokens=0,
                emitted_tokens=(next_token,),
            )
            self._append_tokens(state, (next_token,))
            state.position += 1
        return call_wall_s

    def _run_verify_pool(
        self,
        k: int,
        pool: _VerifierPool,
        active_slots: list[int],
    ) -> tuple[float, list[tuple[_RequestState, int]]]:
        pool.host_input_numpy.fill(self.eos_token_id)
        pool.host_cache_position_numpy.fill(0)
        proposals: dict[int, DraftProposal | None] = {}
        for slot in active_slots:
            state = pool.slots[slot]
            assert state is not None
            state.matcher.block_size = int(k)
            proposal = state.matcher.propose(state.token_ids)
            proposals[slot] = proposal
            pool.host_input_numpy[slot, 0] = int(state.token_ids[-1])
            if proposal is not None and proposal.tokens:
                pool.host_input_numpy[
                    slot, 1 : len(proposal.tokens) + 1
                ] = proposal.tokens
            pool.host_cache_position_numpy[slot] = int(state.position)

        pool.device_input.copy_(pool.host_input, non_blocking=True)
        pool.cache_position.copy_(pool.host_cache_position, non_blocking=True)
        started = time.perf_counter()
        targets = pool.runtime.fn(
            pool.device_input,
            pool.cache_position,
            pool.rope_deltas,
            *pool.cache.flat_tensors(),
        )
        pool.host_targets.copy_(targets, non_blocking=True)
        self.stream.synchronize()
        call_wall_s = time.perf_counter() - started
        migrations: list[tuple[_RequestState, int]] = []

        for slot in active_slots:
            state = pool.slots[slot]
            assert state is not None
            proposal = proposals[slot]
            row_targets = pool.host_targets[slot]
            state.target_calls += 1
            state.verifier_wall_s += call_wall_s
            if proposal is None or not proposal.tokens:
                next_token = int(row_targets[0])
                state.fallback_calls += 1
                state.matcher.commit(
                    None,
                    accepted_draft_tokens=0,
                    emitted_tokens=(next_token,),
                )
                self._append_tokens(state, (next_token,))
                state.position += 1
                continue

            proposal_tokens = proposal.tokens
            state.speculative_calls += 1
            state.proposed_draft_tokens += len(proposal_tokens)
            accepted_here = 0
            for draft_token, target_token in zip(proposal_tokens, row_targets):
                if int(draft_token) != int(target_token):
                    break
                accepted_here += 1
            state.accepted_draft_tokens += accepted_here
            fully_accepted = accepted_here == len(proposal_tokens)
            if fully_accepted:
                state.fully_accepted_speculative_calls += 1
            else:
                state.rejected_speculative_calls += 1
            emitted = list(proposal_tokens[:accepted_here])
            emitted.append(int(row_targets[accepted_here]))
            state.matcher.commit(
                proposal,
                accepted_draft_tokens=accepted_here,
                emitted_tokens=emitted,
            )
            self._append_tokens(state, emitted)
            state.position += accepted_here + 1

            stats = state.per_k[int(k)]
            stats["calls"] = int(stats["calls"]) + 1
            stats["fully_accepted_calls"] = int(
                stats["fully_accepted_calls"]
            ) + int(fully_accepted)
            stats["rejected_calls"] = int(stats["rejected_calls"]) + int(
                not fully_accepted
            )
            stats["proposed_tokens"] = int(stats["proposed_tokens"]) + len(
                proposal_tokens
            )
            stats["accepted_tokens"] = int(stats["accepted_tokens"]) + accepted_here
            stats["call_wall_s"] = float(stats["call_wall_s"]) + call_wall_s
            updated_k = self._next_k(
                int(k),
                fully_accepted=fully_accepted,
                k_values=self.k_values,
            )
            state.transitions[f"{int(k)}->{updated_k}"] += 1
            state.trace.append(
                {
                    "position": int(state.position - accepted_here - 1),
                    "k": int(k),
                    "proposed": len(proposal_tokens),
                    "accepted": accepted_here,
                    "fully_accepted": fully_accepted,
                    "next_k": updated_k,
                }
            )
            state.policy_k = updated_k
            if state.stop_reason is None and updated_k != int(k):
                migrations.append((state, updated_k))
        return call_wall_s, migrations

    @torch.inference_mode()
    def decode_many(
        self,
        requests: Sequence[tuple[Any, TableDraftMatcher]],
        *,
        max_new_tokens: int | None = None,
    ) -> list[BatchedAdaptiveKTableSpecDecodeResult]:
        if not requests or len(requests) > self.batch_size:
            raise ValueError(
                f"expected 1..{self.batch_size} requests, got {len(requests)}"
            )
        if any(pool.active_slots() for pool in self.pools.values()):
            raise RuntimeError("batched verifier still owns active request state")

        started = time.perf_counter()
        copied_bytes = 0
        migration_count = 0
        call_counts: Counter[str] = Counter()
        call_wall_s: Counter[str] = Counter()
        physical_token_slots = 0
        active_token_slots = 0
        states: list[_RequestState] = []
        limit = int(max_new_tokens or self.cache_length)
        for prefilled, matcher in requests:
            state, admission_bytes = self._admit(
                prefilled,
                matcher,
                max_new_tokens=limit,
                admitted_at=started,
            )
            copied_bytes += admission_bytes
            states.append(state)

        self.stream.synchronize()
        for state in states:
            if state.stop_reason is None:
                continue
            state.completed_at = time.perf_counter()
            assert state.lane_k is not None and state.lane_slot is not None
            self.pools[state.lane_k].slots[state.lane_slot] = None
            state.lane_k = None
            state.lane_slot = None
        while any(state.stop_reason is None for state in states):
            pending_migrations: list[tuple[_RequestState, int]] = []
            made_progress = False
            for k in self.k_values:
                pool = self.pools[k]
                active_slots = [
                    slot
                    for slot in pool.active_slots()
                    if pool.slots[slot] is not None
                    and pool.slots[slot].stop_reason is None
                ]
                if not active_slots:
                    continue
                made_progress = True
                needs_single_token = any(
                    int(pool.slots[slot].position) + int(k) + 1
                    > self.cache_length
                    for slot in active_slots
                    if pool.slots[slot] is not None
                )
                if needs_single_token:
                    wall_s = self._run_decode_pool(pool, active_slots)
                    call_counts["decode"] += 1
                    call_wall_s["decode"] += wall_s
                    physical_token_slots += self.batch_size
                    active_token_slots += len(active_slots)
                else:
                    wall_s, migrations = self._run_verify_pool(
                        int(k), pool, active_slots
                    )
                    pending_migrations.extend(migrations)
                    call_counts[f"k{k}"] += 1
                    call_wall_s[f"k{k}"] += wall_s
                    physical_token_slots += self.batch_size * (int(k) + 1)
                    active_token_slots += len(active_slots) * (int(k) + 1)

            for state in states:
                if state.stop_reason is not None and state.completed_at is None:
                    state.completed_at = time.perf_counter()
                    if state.lane_k is not None and state.lane_slot is not None:
                        self.pools[state.lane_k].slots[state.lane_slot] = None
                        state.lane_k = None
                        state.lane_slot = None
            for state, destination_k in pending_migrations:
                if state.stop_reason is not None:
                    continue
                copied_bytes += self._migrate(state, destination_k)
                migration_count += 1
            if not made_progress:
                raise RuntimeError("batched verifier made no progress")

        finished = time.perf_counter()
        if any(pool.active_slots() for pool in self.pools.values()):
            raise RuntimeError("batched verifier leaked active request state")
        results: list[BatchedAdaptiveKTableSpecDecodeResult] = []
        for state in states:
            completed_at = state.completed_at or finished
            text = self.recognizer.tokenizer.decode(
                state.token_ids,
                skip_special_tokens=state.skip_special_tokens,
            )
            results.append(
                BatchedAdaptiveKTableSpecDecodeResult(
                    token_ids=list(state.token_ids),
                    text=text,
                    stop_reason=str(state.stop_reason),
                    target_calls=state.target_calls,
                    speculative_calls=state.speculative_calls,
                    fully_accepted_speculative_calls=(
                        state.fully_accepted_speculative_calls
                    ),
                    rejected_speculative_calls=state.rejected_speculative_calls,
                    fallback_calls=state.fallback_calls,
                    proposed_draft_tokens=state.proposed_draft_tokens,
                    accepted_draft_tokens=state.accepted_draft_tokens,
                    verifier_device_s=0.0,
                    fallback_device_s=0.0,
                    wrapper_rescue_probe_device_s=0.0,
                    wall_s=completed_at - started,
                    wrapper_rescue={
                        "enabled": False,
                        "used": False,
                        "probe_calls": 0,
                        "forced_tokens": 0,
                        "events": [],
                    },
                    repetition=dict(state.repetition),
                    adaptive_k={
                        "values": list(self.k_values),
                        "initial_k": self.initial_k,
                        "per_k": {
                            str(key): value for key, value in state.per_k.items()
                        },
                        "transitions": dict(sorted(state.transitions.items())),
                        "trace": list(state.trace),
                    },
                    batching={
                        "batch_size": self.batch_size,
                        "decode_completed_offset_s": completed_at - started,
                    },
                )
            )

        self.last_summary = {
            "batch_size": self.batch_size,
            "requests": len(states),
            "wall_s": finished - started,
            "call_counts": dict(call_counts),
            "call_wall_s": dict(call_wall_s),
            "migrations": migration_count,
            "cache_bytes_copied": copied_bytes,
            "physical_token_slots": physical_token_slots,
            "active_token_slots": active_token_slots,
            "active_slot_fraction": (
                active_token_slots / physical_token_slots
                if physical_token_slots
                else None
            ),
            "results": [asdict(result) for result in results],
        }
        return results
