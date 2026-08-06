"""Fixed-arena continuous decode for the local UniRec runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    LocalUniRecStaticCache,
    OptimizedUniRecRunner,
    UniRecPrefilledItem,
    synchronize_device,
)


@dataclass
class ContinuousReadyItem:
    request_id: str
    payload: Any
    prefilled: UniRecPrefilledItem


@dataclass
class ContinuousCompletedItem:
    request_id: str
    payload: Any
    result: dict[str, Any]
    slot: int
    admission_index: int
    completion_index: int


class ContinuousUniRecDecoder:
    """Hot-swap B1-prefilled requests in a fixed physical decode batch."""

    def __init__(
        self,
        *,
        runner: OptimizedUniRecRunner,
        batch_size: int,
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Continuous decode batch_size must be >= 1")
        if max_length > LOCAL_UNIREC_STATIC_CACHE_LEN:
            raise ValueError(
                f"max_length must be <= {LOCAL_UNIREC_STATIC_CACHE_LEN}, got {max_length}"
            )
        if decode_mode not in {"eager", "compiled", "compiled_ifa"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        self.runner = runner
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.decode_mode = decode_mode
        self.compile_backend = compile_backend
        self.compile_dynamic = bool(compile_dynamic)

    @staticmethod
    def _copy_cache_row(
        destination: LocalUniRecStaticCache,
        slot: int,
        source: LocalUniRecStaticCache,
    ) -> None:
        if (
            destination.cross_key_cache is None
            or destination.cross_value_cache is None
            or destination.cross_attention_mask is None
            or source.cross_key_cache is None
            or source.cross_value_cache is None
            or source.cross_attention_mask is None
        ):
            raise RuntimeError("Continuous decode requires static cross-attention caches")
        for layer in range(len(destination.key_cache)):
            destination.key_cache[layer][slot : slot + 1].copy_(
                source.key_cache[layer]
            )
            destination.value_cache[layer][slot : slot + 1].copy_(
                source.value_cache[layer]
            )
            destination.cross_key_cache[layer][slot : slot + 1].copy_(
                source.cross_key_cache[layer]
            )
            destination.cross_value_cache[layer][slot : slot + 1].copy_(
                source.cross_value_cache[layer]
            )
        destination.cross_attention_mask[slot : slot + 1].copy_(
            source.cross_attention_mask
        )

    def _build_result(
        self,
        *,
        ready: ContinuousReadyItem,
        token_ids: list[int],
        decode_token_count: int,
        decode_active_s: float,
        compile_wrap_s: float | None,
        compile_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        text = self.runner._decode_text_batch([token_ids])[0]
        prep = ready.prefilled.prep
        cross_attention_mask = ready.prefilled.kv_cache.cross_attention_mask
        if cross_attention_mask is None:
            raise RuntimeError("Continuous decode requires a static cross-attention mask")
        return {
            "image": str(prep["image"]),
            "text": text,
            "generated_ids": token_ids,
            "generated_token_count": max(0, len(token_ids) - 1),
            "prefill_generated_token_count": 0,
            "decode_generated_token_count": int(decode_token_count),
            "ttft_s": ready.prefilled.prefill_s,
            "prefill_device_stage_s": ready.prefilled.prefill_device_stage_s,
            "text_prefill_execution": ready.prefilled.text_prefill_execution,
            "text_prefill_real_source_tokens": (
                ready.prefilled.text_prefill_real_source_tokens
            ),
            "text_prefill_physical_source_tokens": (
                ready.prefilled.text_prefill_physical_source_tokens
            ),
            "decode_s": float(decode_active_s),
            "total_latency_s": (
                float(prep["prepare_total_s"])
                + ready.prefilled.prefill_s
                + float(decode_active_s)
            ),
            "decode_tokens_per_s": (
                float(decode_token_count) / decode_active_s
                if decode_active_s > 0 and decode_token_count > 0
                else None
            ),
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
            "cross_cache_len": int(cross_attention_mask.shape[-1]),
            "static_self_kv_len": int(LOCAL_UNIREC_STATIC_CACHE_LEN),
            "device": self.runner.device,
            "dtype": self.runner.dtype_name,
            "decode_mode": self.decode_mode,
            "compile_backend": (
                self.compile_backend
                if self.decode_mode.startswith("compiled")
                else None
            ),
            "prep": prep,
        }

    def run(
        self,
        source: Iterable[ContinuousReadyItem],
        *,
        on_complete: Callable[[ContinuousCompletedItem], None],
    ) -> dict[str, Any]:
        run_started = time.perf_counter()
        iterator = iter(source)
        source_exhausted = False
        submitted = 0
        completed = 0
        source_pull_s = 0.0
        initial_cache_stack_s = 0.0
        initial_state_build_s = 0.0
        cache_refill_copy_enqueue_s = 0.0
        cache_refill_row_bytes = 0
        result_build_s = 0.0
        completion_callback_s = 0.0
        decode_input_build_s = 0.0
        pre_decode_sync_s = 0.0

        def next_ready() -> ContinuousReadyItem | None:
            nonlocal source_exhausted, source_pull_s, submitted
            if source_exhausted:
                return None
            pull_started = time.perf_counter()
            try:
                ready = next(iterator)
            except StopIteration:
                source_exhausted = True
                source_pull_s += time.perf_counter() - pull_started
                return None
            source_pull_s += time.perf_counter() - pull_started
            submitted += 1
            return ready

        initial: list[ContinuousReadyItem] = []
        for _ in range(self.batch_size):
            ready = next_ready()
            if ready is None:
                break
            initial.append(ready)
        if not initial:
            return {
                "batch_size": self.batch_size,
                "submitted": 0,
                "completed": 0,
                "decode_iterations": 0,
                "raw_decode_token_slots": 0,
                "effective_decode_tokens": 0,
                "idle_decode_token_slots": 0,
                "decode_s": 0.0,
                "first_decode_step_s": None,
                "steady_decode_s": 0.0,
                "raw_decode_tokens_per_s": None,
                "effective_decode_tokens_per_s": None,
                "steady_raw_decode_tokens_per_s": None,
                "steady_effective_decode_tokens_per_s": None,
                "slot_refills": 0,
                "compile_wrap_s": None,
                "compile": None,
            }

        padded_initial = list(initial)
        while len(padded_initial) < self.batch_size:
            padded_initial.append(initial[-1])
        cache_stack_started = time.perf_counter()
        cache = self.runner._stack_prefilled_caches(
            [item.prefilled for item in padded_initial]
        )
        initial_cache_stack_s = time.perf_counter() - cache_stack_started
        state_build_started = time.perf_counter()
        slots: list[ContinuousReadyItem | None] = [
            initial[index] if index < len(initial) else None
            for index in range(self.batch_size)
        ]
        token_ids: list[list[int]] = [
            (
                [int(token) for token in slots[index].prefilled.generated_ids[0].detach().cpu().tolist()]
                if slots[index] is not None
                else [
                    int(self.runner.config.decoder_start_token_id),
                    int(self.runner.config.eos_token_id),
                ]
            )
            for index in range(self.batch_size)
        ]
        last_tokens = [row[-1] for row in token_ids]
        cache_positions = [
            len(row) - 1 if slots[index] is not None else 1
            for index, row in enumerate(token_ids)
        ]
        slot_decode_counts = [0 for _ in range(self.batch_size)]
        slot_active_decode_s = [0.0 for _ in range(self.batch_size)]
        slot_admission_indices = [
            index if index < len(initial) else -1 for index in range(self.batch_size)
        ]
        next_admission_index = len(initial)
        slot_refills = 0
        eos_token_id = int(self.runner.config.eos_token_id)
        initial_state_build_s = time.perf_counter() - state_build_started

        self_attention_backend = (
            "increfa_all" if self.decode_mode == "compiled_ifa" else "eager"
        )
        decode_module = None
        compile_wrap_s = None
        compile_meta = None
        cross_cache_len = int(cache.cross_attention_mask.shape[-1])
        if self.decode_mode.startswith("compiled"):
            compile_started = time.perf_counter()
            decode_module, compile_meta = self.runner._compile_decode_module(
                backend=self.compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=self.compile_dynamic,
                cross_cache_len=cross_cache_len,
                batch_size=self.batch_size,
            )
            compile_wrap_s = time.perf_counter() - compile_started

        def complete_slot(slot: int) -> None:
            nonlocal completed, completion_callback_s, result_build_s
            ready = slots[slot]
            if ready is None:
                return
            result_started = time.perf_counter()
            result = self._build_result(
                ready=ready,
                token_ids=token_ids[slot],
                decode_token_count=slot_decode_counts[slot],
                decode_active_s=slot_active_decode_s[slot],
                compile_wrap_s=compile_wrap_s,
                compile_meta=compile_meta,
            )
            result_build_s += time.perf_counter() - result_started
            callback_started = time.perf_counter()
            on_complete(
                ContinuousCompletedItem(
                    request_id=ready.request_id,
                    payload=ready.payload,
                    result=result,
                    slot=slot,
                    admission_index=slot_admission_indices[slot],
                    completion_index=completed,
                )
            )
            completion_callback_s += time.perf_counter() - callback_started
            completed += 1
            slots[slot] = None

        def refill_slot(slot: int) -> None:
            nonlocal cache_refill_copy_enqueue_s, cache_refill_row_bytes
            nonlocal next_admission_index, slot_refills
            while slots[slot] is None:
                ready = next_ready()
                if ready is None:
                    token_ids[slot] = [
                        int(self.runner.config.decoder_start_token_id),
                        eos_token_id,
                    ]
                    last_tokens[slot] = eos_token_id
                    cache_positions[slot] = 1
                    return
                if cache_refill_row_bytes == 0:
                    cache_refill_row_bytes = sum(
                        int(tensor[slot : slot + 1].numel() * tensor.element_size())
                        for tensor_group in (
                            cache.key_cache,
                            cache.value_cache,
                            cache.cross_key_cache or (),
                            cache.cross_value_cache or (),
                        )
                        for tensor in tensor_group
                    )
                    if cache.cross_attention_mask is not None:
                        mask_row = cache.cross_attention_mask[slot : slot + 1]
                        cache_refill_row_bytes += int(
                            mask_row.numel() * mask_row.element_size()
                        )
                copy_started = time.perf_counter()
                self._copy_cache_row(cache, slot, ready.prefilled.kv_cache)
                cache_refill_copy_enqueue_s += time.perf_counter() - copy_started
                slots[slot] = ready
                token_ids[slot] = [
                    int(token)
                    for token in ready.prefilled.generated_ids[0].detach().cpu().tolist()
                ]
                last_tokens[slot] = token_ids[slot][-1]
                cache_positions[slot] = len(token_ids[slot]) - 1
                slot_decode_counts[slot] = 0
                slot_active_decode_s[slot] = 0.0
                slot_admission_indices[slot] = next_admission_index
                next_admission_index += 1
                slot_refills += 1
                if last_tokens[slot] != eos_token_id:
                    return
                complete_slot(slot)

        # A request may produce EOS directly from B1 prefill. Complete and
        # refill it without launching a useless decode iteration.
        for slot in range(self.batch_size):
            if slots[slot] is not None and last_tokens[slot] == eos_token_id:
                complete_slot(slot)
                refill_slot(slot)

        decode_iterations = 0
        raw_decode_token_slots = 0
        effective_decode_tokens = 0
        idle_decode_token_slots = 0
        decode_s = 0.0
        first_decode_step_s: float | None = None

        with torch.inference_mode():
            while any(slot is not None for slot in slots):
                active_slots = [slot is not None for slot in slots]
                input_build_started = time.perf_counter()
                next_token_tensor = torch.tensor(
                    last_tokens,
                    dtype=torch.long,
                    device=self.runner.device,
                ).view(self.batch_size, 1)
                cache_position_tensor = torch.tensor(
                    cache_positions,
                    dtype=torch.int64,
                    device=self.runner.device,
                )
                decode_input_build_s += time.perf_counter() - input_build_started
                sync_started = time.perf_counter()
                synchronize_device(self.runner.device)
                pre_decode_sync_s += time.perf_counter() - sync_started
                step_started = time.perf_counter()
                if decode_module is None:
                    logits = self.runner.model.forward_cached_logits(
                        decoder_input_ids=next_token_tensor,
                        cache_position=cache_position_tensor,
                        active_length=0,
                        key_cache=cache.key_cache,
                        value_cache=cache.value_cache,
                        cross_key_cache=cache.cross_key_cache,
                        cross_value_cache=cache.cross_value_cache,
                        cross_attention_mask=cache.cross_attention_mask,
                        self_attention_backend="eager",
                    )
                else:
                    logits = decode_module(
                        next_token_tensor,
                        cache_position_tensor,
                        0,
                        cache.key_cache,
                        cache.value_cache,
                        cache.cross_key_cache,
                        cache.cross_value_cache,
                        cache.cross_attention_mask,
                    )
                predicted = self.runner.model.select_next_token(logits)
                predicted_ids = [
                    int(token) for token in predicted.detach().cpu().view(-1).tolist()
                ]
                step_s = time.perf_counter() - step_started
                decode_s += step_s
                if first_decode_step_s is None:
                    first_decode_step_s = step_s
                decode_iterations += 1
                raw_decode_token_slots += self.batch_size
                active_count = sum(active_slots)
                effective_decode_tokens += active_count
                idle_decode_token_slots += self.batch_size - active_count

                completed_slots = []
                for slot, is_active in enumerate(active_slots):
                    if not is_active:
                        continue
                    token = predicted_ids[slot]
                    token_ids[slot].append(token)
                    last_tokens[slot] = token
                    cache_positions[slot] += 1
                    slot_decode_counts[slot] += 1
                    slot_active_decode_s[slot] += step_s
                    if token == eos_token_id or len(token_ids[slot]) >= self.max_length:
                        completed_slots.append(slot)

                for slot in completed_slots:
                    complete_slot(slot)
                for slot in completed_slots:
                    refill_slot(slot)

        steady_decode_s = decode_s - (first_decode_step_s or 0.0)
        steady_raw_slots = max(0, raw_decode_token_slots - self.batch_size)
        first_active = min(self.batch_size, len(initial))
        steady_effective_tokens = max(0, effective_decode_tokens - first_active)
        run_wall_s = time.perf_counter() - run_started
        directly_accounted_s = sum(
            (
                source_pull_s,
                initial_cache_stack_s,
                initial_state_build_s,
                compile_wrap_s or 0.0,
                cache_refill_copy_enqueue_s,
                result_build_s,
                completion_callback_s,
                decode_input_build_s,
                pre_decode_sync_s,
                decode_s,
            )
        )
        return {
            "batch_size": self.batch_size,
            "submitted": submitted,
            "completed": completed,
            "decode_iterations": decode_iterations,
            "raw_decode_token_slots": raw_decode_token_slots,
            "effective_decode_tokens": effective_decode_tokens,
            "idle_decode_token_slots": idle_decode_token_slots,
            "decode_s": decode_s,
            "first_decode_step_s": first_decode_step_s,
            "steady_decode_s": steady_decode_s,
            "raw_decode_tokens_per_s": (
                raw_decode_token_slots / decode_s if decode_s > 0 else None
            ),
            "effective_decode_tokens_per_s": (
                effective_decode_tokens / decode_s if decode_s > 0 else None
            ),
            "steady_raw_decode_tokens_per_s": (
                steady_raw_slots / steady_decode_s if steady_decode_s > 0 else None
            ),
            "steady_effective_decode_tokens_per_s": (
                steady_effective_tokens / steady_decode_s
                if steady_decode_s > 0
                else None
            ),
            "slot_refills": slot_refills,
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
            "timing_detail": {
                "run_wall_s": run_wall_s,
                "source_pull_s": source_pull_s,
                "initial_cache_stack_s": initial_cache_stack_s,
                "initial_state_build_s": initial_state_build_s,
                "cache_refill_copy_enqueue_s": cache_refill_copy_enqueue_s,
                "cache_refill_row_bytes": cache_refill_row_bytes,
                "cache_refill_total_bytes": cache_refill_row_bytes * slot_refills,
                "result_build_s": result_build_s,
                "completion_callback_s": completion_callback_s,
                "decode_input_build_s": decode_input_build_s,
                "pre_decode_sync_s": pre_decode_sync_s,
                "decode_s": decode_s,
                "scheduler_bookkeeping_residual_s": max(
                    0.0, run_wall_s - directly_accounted_s
                ),
            },
        }
