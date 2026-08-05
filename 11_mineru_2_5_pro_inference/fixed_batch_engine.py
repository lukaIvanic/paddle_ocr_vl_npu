"""Request-owned KV slots and compiled decode scheduling for MinerU.

This is the first scheduler-shaped inference engine.  Each request is
prefilled independently into a view of its own row in a shared KV arena, so
prompt padding never enters decode.  Full groups use one compiled graph;
incomplete tails deliberately reuse the established B1 path.  The continuous
variant keeps the same compiled batch shape and replaces a completed slot with
the next waiting request.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from local_modeling_mineru import LocalMinerUStaticCache
from prefill_timing import PrefillDeviceTimeline
from run_local_model_two_step_extract import maybe_sync_device


@dataclass
class PreparedGeneration:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    max_new_tokens: int


class FixedBatchDecodeEngine:
    """B1 prefill into request slots followed by lockstep compiled decode."""

    def __init__(
        self,
        model: Any,
        compiled_decoder: Any,
        *,
        batch_size: int,
        cache_length: int,
        eos_token_id: int,
        pad_token_id: int,
        collect_prefill_metrics: bool = False,
        packed_text_prefill_runtime: Any | None = None,
    ) -> None:
        if int(batch_size) <= 1:
            raise ValueError("fixed batch engine requires batch_size > 1")
        self.model = model
        self.compiled_decoder = compiled_decoder
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.collect_prefill_metrics = bool(collect_prefill_metrics)
        self.packed_text_prefill_runtime = packed_text_prefill_runtime
        self._arena: LocalMinerUStaticCache | None = None

    def _arena_for_batch(self) -> LocalMinerUStaticCache:
        if self._arena is None:
            self._arena = self.model.allocate_static_cache(
                batch_size=self.batch_size,
                cache_length=self.cache_length,
                device=self.model.device,
                dtype=self.model.dtype,
                init_mode="empty",
            )
        return self._arena

    @staticmethod
    def _slot_view(arena: LocalMinerUStaticCache, slot: int) -> LocalMinerUStaticCache:
        return LocalMinerUStaticCache(
            key_caches=tuple(cache[slot : slot + 1] for cache in arena.key_caches),
            value_caches=tuple(cache[slot : slot + 1] for cache in arena.value_caches),
            cache_length=arena.cache_length,
        )

    @torch.inference_mode()
    def generate_many(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if not requests:
            return [], {
                "enabled": True,
                "batch_size": 0,
                "decode_calls": 0,
                "raw_decode_token_slots": 0,
                "decode_s": 0.0,
                "prefill_s": 0.0,
            }
        if len(requests) != self.batch_size:
            return self._generate_tail_b1(requests)
        return self._generate_full_batch(requests)

    @torch.inference_mode()
    def _generate_tail_b1(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        outputs: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []
        for request in requests:
            generated, metrics = self.compiled_decoder.generate(
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                pixel_values=request.pixel_values,
                image_grid_thw=request.image_grid_thw,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
            outputs.append(generated)
            records.append(metrics)
        decode_calls = sum(int(item["decode_calls"]) for item in records)
        decode_s = sum(float(item["decode_s"]) for item in records)
        return outputs, {
            "enabled": True,
            "mode": "b1_tail",
            "batch_size": len(requests),
            "decode_calls": decode_calls,
            "raw_decode_token_slots": decode_calls,
            "decode_s": decode_s,
            "prefill_s": sum(float(item["prefill_s"]) for item in records),
            "generation_wall_s": float(time.perf_counter() - started),
            "compile_wrapper_s": sum(
                float(item["compile_wrapper_s"])
                for item in records
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
            "compiled_first_call_s": sum(
                float(item["compiled_first_call_s"])
                for item in records
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
        }

    @torch.inference_mode()
    def _generate_full_batch(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        arena = self._arena_for_batch()
        next_tokens: list[torch.Tensor] = []
        rope_deltas: list[torch.Tensor] = []
        cache_positions: list[torch.Tensor] = []
        generated_steps: list[torch.Tensor] = []

        maybe_sync_device(self.model.device)
        prefill_started = time.perf_counter()
        for slot, request in enumerate(requests):
            if request.input_ids.shape[0] != 1:
                raise ValueError("each fixed-batch request must be prepared at B1")
            if request.input_ids.shape[1] + request.max_new_tokens > self.cache_length:
                raise ValueError(
                    "request exceeds static cache capacity: "
                    f"input={int(request.input_ids.shape[1])} "
                    f"max_new={request.max_new_tokens} cache={self.cache_length}"
                )
            prefill = self.model.forward_static_prefill(
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                pixel_values=request.pixel_values,
                image_grid_thw=request.image_grid_thw,
                cache_length=self.cache_length,
                cache=self._slot_view(arena, slot),
                logits_to_keep=1,
            )
            token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_tokens.append(token)
            rope_deltas.append(prefill.rope_deltas)
            cache_positions.append(prefill.next_cache_position)
        maybe_sync_device(self.model.device)
        prefill_s = time.perf_counter() - prefill_started

        next_token = torch.cat(next_tokens, dim=0)
        generated_steps.append(next_token)
        rope_delta = torch.cat(rope_deltas, dim=0)
        cache_position = torch.cat(cache_positions, dim=0)
        generated_counts = torch.ones(
            self.batch_size,
            device=self.model.device,
            dtype=torch.int64,
        )
        max_new = torch.tensor(
            [request.max_new_tokens for request in requests],
            device=self.model.device,
            dtype=torch.int64,
        )
        finished = (next_token.squeeze(1) == self.eos_token_id) | (generated_counts >= max_new)

        compile_started = time.perf_counter()
        compiled_decode, compile_meta = self.compiled_decoder.compiled_decode_for(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
        )
        compile_wrapper_s = time.perf_counter() - compile_started

        graph_calls = 0
        first_call_s = 0.0
        maybe_sync_device(self.model.device)
        decode_started = time.perf_counter()
        while not bool(finished.all().item()):
            call_started = time.perf_counter()
            logits = compiled_decode(
                next_token,
                cache_position,
                rope_delta,
                *arena.flat_tensors(),
            )
            if graph_calls == 0:
                maybe_sync_device(self.model.device)
                first_call_s = time.perf_counter() - call_started
            graph_calls += 1
            candidate = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            active = ~finished
            emitted = torch.where(
                active.view(-1, 1),
                candidate,
                torch.full_like(candidate, self.pad_token_id),
            )
            generated_steps.append(emitted)
            generated_counts.add_(active.to(dtype=torch.int64))
            finished = finished | (candidate.squeeze(1) == self.eos_token_id) | (generated_counts >= max_new)
            next_token = torch.where(
                finished.view(-1, 1),
                torch.full_like(candidate, self.pad_token_id),
                candidate,
            )
            cache_position.add_(active.to(dtype=torch.int64))
        maybe_sync_device(self.model.device)
        decode_s = time.perf_counter() - decode_started

        generated_lengths = [int(value) for value in generated_counts.cpu().tolist()]
        all_generated = torch.cat(generated_steps, dim=1)
        outputs = [
            all_generated[slot : slot + 1, :length]
            for slot, length in enumerate(generated_lengths)
        ]
        effective_decode_tokens = sum(generated_lengths) - self.batch_size
        return outputs, {
            "enabled": True,
            "mode": "fixed_batch",
            "batch_size": self.batch_size,
            "cache_length": self.cache_length,
            "graph_calls": graph_calls,
            "decode_calls": effective_decode_tokens,
            "raw_decode_token_slots": graph_calls * self.batch_size,
            "decode_s": float(decode_s),
            "prefill_s": float(prefill_s),
            "generation_wall_s": float(time.perf_counter() - started),
            "compile_wrapper_s": float(compile_wrapper_s),
            "compiled_first_call_s": float(first_call_s),
            "compile": dict(compile_meta),
        }


class ContinuousBatchDecodeEngine(FixedBatchDecodeEngine):
    """Continuously refill request-owned slots in one static decode arena.

    This first implementation intentionally retains synchronous token
    completion checks.  It isolates the benefit of slot refill from any future
    stream/event or D2H-pipelining optimization.
    """

    @torch.inference_mode()
    def generate_many(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if len(requests) < self.batch_size:
            return super().generate_many(requests)
        return self._generate_continuous(
            request_count=len(requests),
            prepare_request=requests.__getitem__,
        )

    @torch.inference_mode()
    def generate_lazy(
        self,
        request_count: int,
        prepare_request: Callable[[int], PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Generate in input order while preparing only requests being admitted."""
        if request_count < 0:
            raise ValueError("request_count must be non-negative")
        if request_count < self.batch_size:
            requests = [prepare_request(index) for index in range(request_count)]
            return super().generate_many(requests)
        return self._generate_continuous(
            request_count=request_count,
            prepare_request=prepare_request,
        )

    def _validate_request(self, request: PreparedGeneration) -> None:
        if request.input_ids.shape[0] != 1:
            raise ValueError("each continuous request must be prepared at B1")
        if request.input_ids.shape[1] + request.max_new_tokens > self.cache_length:
            raise ValueError(
                "request exceeds static cache capacity: "
                f"input={int(request.input_ids.shape[1])} "
                f"max_new={request.max_new_tokens} cache={self.cache_length}"
            )

    @torch.inference_mode()
    def _prefill_slot(
        self,
        arena: LocalMinerUStaticCache,
        slot: int,
        request: PreparedGeneration,
    ) -> tuple[dict[str, Any], float]:
        self._validate_request(request)
        started = time.perf_counter()
        prefill = self.model.forward_static_prefill(
            input_ids=request.input_ids,
            attention_mask=request.attention_mask,
            pixel_values=request.pixel_values,
            image_grid_thw=request.image_grid_thw,
            cache_length=self.cache_length,
            cache=self._slot_view(arena, slot),
            logits_to_keep=1,
            collect_prefill_metrics=self.collect_prefill_metrics,
        )
        token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        # The scheduler intentionally makes completion state host-visible in
        # this version.  This synchronization matches the fixed-cohort path's
        # semantics; only slot refill is under test.
        token_id = int(token[0, 0].item())
        return {
            "token": token,
            "token_id": token_id,
            "cache_position": prefill.next_cache_position,
            "rope_delta": prefill.rope_deltas,
            "prefill_metrics": prefill.prefill_metrics,
        }, float(time.perf_counter() - started)

    @torch.inference_mode()
    def _prefill_slots(
        self,
        arena: LocalMinerUStaticCache,
        entries: Sequence[tuple[int, int, PreparedGeneration]],
    ) -> tuple[dict[int, dict[str, Any]], float, dict[str, float | int]]:
        """Prefill free slots, using packed static text graphs when enabled."""
        started = time.perf_counter()
        if self.packed_text_prefill_runtime is None:
            states: dict[int, dict[str, Any]] = {}
            aggregate: dict[str, float | int] = {}
            for slot, request_index, request in entries:
                state, _elapsed = self._prefill_slot(arena, slot, request)
                state["request_index"] = request_index
                states[slot] = state
                for name, value in (state.get("prefill_metrics") or {}).items():
                    aggregate[name] = aggregate.get(name, 0) + value
            return states, float(time.perf_counter() - started), aggregate

        from text_prefill_compile import PreparedTextMember

        runtime = self.packed_text_prefill_runtime
        timeline = PrefillDeviceTimeline(self.model.device)
        members: list[PreparedTextMember] = []
        for _slot, _request_index, request in entries:
            self._validate_request(request)
            inputs_embeds = self.model.build_inputs_embeds(
                request.input_ids,
                request.pixel_values,
                request.image_grid_thw,
                device_timeline=timeline,
            )
            position_ids, rope_delta = timeline.measure(
                "mrope_prepare",
                lambda request=request: self.model.get_rope_index(
                    request.input_ids,
                    request.image_grid_thw,
                    request.attention_mask,
                ),
            )
            raw_vision_tokens = (
                0 if request.pixel_values is None else int(request.pixel_values.shape[0])
            )
            merge_factor = int(self.model.config.vision_config.spatial_merge_size) ** 2
            members.append(
                PreparedTextMember(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    rope_delta=rope_delta,
                    sequence_length=int(inputs_embeds.shape[1]),
                    raw_vision_tokens=raw_vision_tokens,
                    merged_vision_tokens=raw_vision_tokens // merge_factor,
                )
            )

        lengths = [member.sequence_length for member in members]
        packs, overflow = runtime.pack_indices(lengths)
        states: dict[int, dict[str, Any]] = {}
        physical_tokens = 0
        copied_bytes = 0
        for pack_indices in packs:
            pack_members = [members[index] for index in pack_indices]
            prepared = runtime.prepare(pack_members)
            physical_tokens += prepared.physical_tokens
            last_hidden_states = timeline.measure(
                "text_transformer_prefill",
                lambda prepared=prepared: runtime.run_prepared(prepared),
            )
            destinations = [
                self._slot_view(arena, entries[index][0]) for index in pack_indices
            ]
            copied_bytes += timeline.measure(
                "text_kv_redistribute",
                lambda prepared=prepared, destinations=destinations: runtime.redistribute_cache(
                    prepared,
                    destinations,
                ),
            )
            logits = timeline.measure(
                "prefill_lm_head",
                lambda last_hidden_states=last_hidden_states: F.linear(
                    last_hidden_states,
                    self.model.model.embed_tokens.weight,
                ),
            )
            tokens = torch.argmax(logits.float(), dim=-1)
            for member_offset, entry_index in enumerate(pack_indices):
                slot, request_index, _request = entries[entry_index]
                member = members[entry_index]
                token = tokens[:, member_offset : member_offset + 1]
                states[slot] = {
                    "request_index": request_index,
                    "token": token,
                    "token_id": None,
                    "cache_position": torch.full(
                        (1,),
                        member.sequence_length,
                        device=self.model.device,
                        dtype=torch.int64,
                    ),
                    "rope_delta": member.rope_delta,
                    "prefill_metrics": None,
                }

        overflow_metrics: dict[str, float | int] = {}
        for entry_index in overflow:
            slot, request_index, request = entries[entry_index]
            state, _elapsed = self._prefill_slot(arena, slot, request)
            state["request_index"] = request_index
            states[slot] = state
            for name, value in (state.get("prefill_metrics") or {}).items():
                overflow_metrics[name] = overflow_metrics.get(name, 0) + value

        stage_metrics = timeline.resolve()
        for state in states.values():
            if state["token_id"] is None:
                state["token_id"] = int(state["token"][0, 0].item())
        aggregate: dict[str, float | int] = {
            "request_count": len(entries),
            "raw_vision_tokens": sum(member.raw_vision_tokens for member in members),
            "merged_vision_tokens": sum(member.merged_vision_tokens for member in members),
            "text_prefill_tokens": sum(lengths),
            "physical_text_prefill_tokens": physical_tokens,
            "text_prefill_pack_count": len(packs),
            "text_prefill_overflow_count": len(overflow),
            "text_kv_redistribute_bytes": copied_bytes,
            **stage_metrics,
        }
        for name, value in overflow_metrics.items():
            aggregate[name] = aggregate.get(name, 0) + value
        return states, float(time.perf_counter() - started), aggregate

    @torch.inference_mode()
    def _generate_continuous(
        self,
        *,
        request_count: int,
        prepare_request: Callable[[int], PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        arena = self._arena_for_batch()
        generated: list[list[int] | None] = [None] * request_count
        outputs: list[torch.Tensor | None] = [None] * request_count
        request_max_new: list[int | None] = [None] * request_count
        slot_requests: list[int | None] = [None] * self.batch_size
        next_request = 0
        prefill_s = 0.0
        refill_count = 0
        immediate_completion_count = 0
        prefill_metrics: dict[str, float | int] = {}

        def admit_many(slots: Sequence[int]) -> dict[int, dict[str, Any]]:
            nonlocal next_request, prefill_s, refill_count, immediate_completion_count
            available = list(slots)
            admitted: dict[int, dict[str, Any]] = {}
            while available and next_request < request_count:
                entries: list[tuple[int, int, PreparedGeneration]] = []
                for slot in available:
                    if next_request >= request_count:
                        break
                    request_index = next_request
                    next_request += 1
                    request = prepare_request(request_index)
                    request_max_new[request_index] = request.max_new_tokens
                    entries.append((slot, request_index, request))
                states, elapsed_s, group_metrics = self._prefill_slots(
                    arena,
                    entries,
                )
                prefill_s += elapsed_s
                for name, value in group_metrics.items():
                    prefill_metrics[name] = prefill_metrics.get(name, 0) + value
                retry_slots: list[int] = []
                for slot, request_index, request in entries:
                    state = states[slot]
                    token_id = int(state["token_id"])
                    generated[request_index] = [token_id]
                    if token_id == self.eos_token_id or request.max_new_tokens <= 1:
                        outputs[request_index] = torch.tensor([[token_id]], dtype=torch.long)
                        immediate_completion_count += 1
                        retry_slots.append(slot)
                        continue
                    slot_requests[slot] = request_index
                    admitted[slot] = state
                    refill_count += 1
                available = retry_slots
            for slot in available:
                slot_requests[slot] = None
            return admitted

        initial_state_map = admit_many(range(self.batch_size))
        initial_states = [initial_state_map.get(slot) for slot in range(self.batch_size)]
        template = next((state for state in initial_states if state is not None), None)
        if template is None:
            if not all(output is not None for output in outputs):
                raise RuntimeError("continuous scheduler lost an immediate result")
            return [output for output in outputs if output is not None], {
                "enabled": True,
                "mode": "continuous_refill",
                "batch_size": self.batch_size,
                "cache_length": self.cache_length,
                "request_count": request_count,
                "graph_calls": 0,
                "decode_calls": 0,
                "raw_decode_token_slots": 0,
                "active_decode_token_slots": 0,
                "idle_decode_token_slots": 0,
                "refill_count": refill_count,
                "immediate_completion_count": immediate_completion_count,
                "decode_s": 0.0,
                "prefill_s": prefill_s,
                "prefill_metrics": prefill_metrics,
                "generation_wall_s": float(time.perf_counter() - started),
                "compile_wrapper_s": 0.0,
                "compiled_first_call_s": 0.0,
            }

        def state_or_dummy(state: dict[str, Any] | None, key: str) -> torch.Tensor:
            if state is not None:
                return state[key]
            if key == "token":
                return torch.full_like(template["token"], self.pad_token_id)
            return torch.zeros_like(template[key])

        next_token = torch.cat(
            [state_or_dummy(state, "token") for state in initial_states], dim=0
        )
        cache_position = torch.cat(
            [state_or_dummy(state, "cache_position") for state in initial_states],
            dim=0,
        )
        rope_delta = torch.cat(
            [state_or_dummy(state, "rope_delta") for state in initial_states], dim=0
        )

        compile_started = time.perf_counter()
        compiled_decode, compile_meta = self.compiled_decoder.compiled_decode_for(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
        )
        compile_wrapper_s = time.perf_counter() - compile_started

        graph_calls = 0
        first_call_s = 0.0
        decode_s = 0.0
        active_decode_token_slots = 0
        while any(request_index is not None for request_index in slot_requests):
            active_decode_token_slots += sum(
                request_index is not None for request_index in slot_requests
            )
            call_started = time.perf_counter()
            logits = compiled_decode(
                next_token,
                cache_position,
                rope_delta,
                *arena.flat_tensors(),
            )
            candidate = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            candidate_ids = [int(value) for value in candidate[:, 0].cpu().tolist()]
            call_s = time.perf_counter() - call_started
            decode_s += call_s
            if graph_calls == 0:
                first_call_s = call_s
            graph_calls += 1

            finished_slots: list[int] = []
            for slot, request_index in enumerate(tuple(slot_requests)):
                if request_index is None:
                    continue
                request_tokens = generated[request_index]
                if request_tokens is None:
                    raise RuntimeError("active request has no generated-token state")
                token_id = candidate_ids[slot]
                request_tokens.append(token_id)
                max_new_tokens = request_max_new[request_index]
                if max_new_tokens is None:
                    raise RuntimeError("active request has no generation limit")
                is_finished = (
                    token_id == self.eos_token_id
                    or len(request_tokens) >= max_new_tokens
                )
                if not is_finished:
                    next_token[slot : slot + 1].copy_(candidate[slot : slot + 1])
                    cache_position[slot].add_(1)
                    continue

                outputs[request_index] = torch.tensor(
                    [request_tokens], dtype=torch.long
                )
                slot_requests[slot] = None
                finished_slots.append(slot)

            replacements = admit_many(finished_slots)
            for slot in finished_slots:
                replacement = replacements.get(slot)
                if replacement is None:
                    next_token[slot].fill_(self.pad_token_id)
                    cache_position[slot].zero_()
                    rope_delta[slot].zero_()
                else:
                    next_token[slot : slot + 1].copy_(replacement["token"])
                    cache_position[slot : slot + 1].copy_(
                        replacement["cache_position"]
                    )
                    rope_delta[slot : slot + 1].copy_(replacement["rope_delta"])

        maybe_sync_device(self.model.device)
        if not all(output is not None for output in outputs):
            missing = [index for index, output in enumerate(outputs) if output is None]
            raise RuntimeError(f"continuous scheduler lost outputs: {missing}")

        effective_decode_tokens = sum(
            max(0, len(tokens or ()) - 1) for tokens in generated
        )
        raw_decode_token_slots = graph_calls * self.batch_size
        return [output for output in outputs if output is not None], {
            "enabled": True,
            "mode": "continuous_refill",
            "batch_size": self.batch_size,
            "cache_length": self.cache_length,
            "request_count": request_count,
            "graph_calls": graph_calls,
            "decode_calls": effective_decode_tokens,
            "raw_decode_token_slots": raw_decode_token_slots,
            "active_decode_token_slots": active_decode_token_slots,
            "idle_decode_token_slots": (
                raw_decode_token_slots - active_decode_token_slots
            ),
            "refill_count": refill_count,
            "immediate_completion_count": immediate_completion_count,
            "decode_s": float(decode_s),
            "prefill_s": float(prefill_s),
            "prefill_metrics": prefill_metrics,
            "generation_wall_s": float(time.perf_counter() - started),
            "compile_wrapper_s": float(compile_wrapper_s),
            "compiled_first_call_s": float(first_call_s),
            "compile": dict(compile_meta),
        }
