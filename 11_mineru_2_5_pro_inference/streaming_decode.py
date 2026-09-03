"""Open-source continuous decode using the existing MinerU prefill/graph kernels.

Source contract: pull(block=...) -> (unique integer, PreparedGeneration) or
None; closed distinguishes EOF from temporary starvation; complete receives
unfiltered IDs once. Source completion may create additional requests.
"""
from __future__ import annotations

from collections import Counter, deque
import time


def run_decode_stream(engine, source):
    import torch
    from prefill_timing import PrefillDeviceTimeline
    from run_local_model_two_step_extract import maybe_sync_device

    started = time.perf_counter()
    batch = engine.batch_size
    arena = engine._arena_for_batch()
    slots = [None] * batch
    epochs = [0] * batch
    tokens = {}
    limits = {}
    vision_ready = deque()
    next_token = cache_position = rope_delta = None
    graph_calls = active_slots_total = completed = effective_tokens = 0
    immediate = refill_count = 0
    prefill_s = copy_submit_s = copy_wait_s = safety_s = 0.0
    prefill_metrics = Counter()
    occupancy = Counter()
    decode_by_occupancy = Counter()
    decode_timeline = PrefillDeviceTimeline(engine.model.device)
    pending = None
    drain_started = None
    first_call_s = 0.0
    request_count = 0
    compiled_decode = None
    compile_meta = {}
    compile_wrapper_s = 0.0
    max_live_requests = 0
    idle_rows_with_ready_work = 0
    seen_requests = set()
    initial_inactive_filler_rows = 0
    initial_filler_source_slot = None

    def complete(index):
        nonlocal completed, effective_tokens
        row = tokens.pop(index)
        del limits[index]
        effective_tokens += len(row) - 1
        source.complete(index, row)
        completed += 1

    def fill_vision():
        nonlocal prefill_s, request_count, max_live_requests
        if vision_ready:
            return
        window = []
        while len(window) < engine.vision_lookahead:
            # The page source blocks only on known CPU work. If only its own
            # layouts are in flight it returns None immediately, not false EOF.
            item = source.pull(block=True)
            if item is None:
                break
            index, request = item
            if index in seen_requests:
                raise ValueError(f"duplicate request id: {index}")
            seen_requests.add(index)
            limits[index] = request.max_new_tokens
            window.append((index, request))
            request_count += 1
        if window and engine.packed_text_prefill_runtime is not None:
            elapsed, metrics = engine._prepare_vision_window(window)
            prefill_s += elapsed
            prefill_metrics.update(metrics)
        vision_ready.extend(window)
        max_live_requests = max(max_live_requests, len(limits))

    def admit_free():
        nonlocal prefill_s, immediate, refill_count, safety_s
        nonlocal next_token, cache_position, rope_delta
        nonlocal initial_inactive_filler_rows, initial_filler_source_slot
        available = [i for i, request in enumerate(slots) if request is None]
        replacements = {}
        synchronized = False
        while available:
            fill_vision()
            if not vision_ready:
                break
            if graph_calls and not synchronized:
                begin = time.perf_counter()
                maybe_sync_device(engine.model.device)
                safety_s += time.perf_counter() - begin
                synchronized = True
            entries = []
            while available and vision_ready:
                slot = available.pop(0)
                index, request = vision_ready.popleft()
                entries.append((slot, index, request))
            states, elapsed, metrics = engine._prefill_slots(arena, entries)
            prefill_s += elapsed
            prefill_metrics.update(metrics)
            for slot, index, request in entries:
                state = states[slot]
                token = int(state["token_id"])
                tokens[index] = [token]
                if token == engine.eos_token_id or request.max_new_tokens <= 1:
                    complete(index)
                    immediate += 1
                    available.append(slot)
                else:
                    epochs[slot] += 1
                    slots[slot] = index
                    replacements[slot] = state
                    refill_count += 1
            # Unassigned free slots remain in available across every window.
        if not replacements:
            return
        if next_token is None:
            initial_filler_source_slot = next(iter(replacements))
            template = replacements[initial_filler_source_slot]
            filler_slots = [slot for slot in range(batch) if slot not in replacements]
            engine._duplicate_cache_row_(
                arena,
                source_slot=initial_filler_source_slot,
                destination_slots=filler_slots,
            )
            initial_inactive_filler_rows = len(filler_slots)

            def value(slot, key):
                if slot in replacements:
                    return replacements[slot][key]
                return template[key]

            next_token = torch.cat([value(i, "token") for i in range(batch)], dim=0)
            cache_position = torch.cat([value(i, "cache_position") for i in range(batch)], dim=0)
            rope_delta = torch.cat([value(i, "rope_delta") for i in range(batch)], dim=0)
        else:
            for slot, state in replacements.items():
                next_token[slot:slot + 1].copy_(state["token"])
                cache_position[slot:slot + 1].copy_(state["cache_position"])
                rope_delta[slot:slot + 1].copy_(state["rope_delta"])

    def resolve_timing():
        for key, value in decode_timeline.resolve().items():
            decode_by_occupancy[int(key)] += value
        decode_timeline._events.clear()

    with torch.inference_mode():
        while True:
            admit_free()
            active = sum(index is not None for index in slots)
            if not active:
                if source.closed and not vision_ready:
                    break
                # A synchronous finite page source must either produce CPU
                # work or close here. An external source can wait for arrivals.
                if hasattr(source, "wait_for_work"):
                    source.wait_for_work()
                    continue
                raise RuntimeError("source stalled with no active decode and no ready requests")
            if vision_ready:
                idle_rows_with_ready_work += batch - active
                if active != batch:
                    raise RuntimeError("decode has empty rows despite prepared requests")
            if drain_started is None and source.upstream_exhausted and not vision_ready:
                drain_started = time.perf_counter()
            if compiled_decode is None:
                begin = time.perf_counter()
                compiled_decode, compile_meta = engine.compiled_decoder.compiled_decode_for(
                    batch_size=batch, cache_length=engine.cache_length)
                compile_wrapper_s = time.perf_counter() - begin
            launched_requests, launched_epochs = tuple(slots), tuple(epochs)
            occupancy[active] += 1
            active_slots_total += active
            begin = time.perf_counter()
            candidate = decode_timeline.measure(str(active), lambda: torch.argmax(
                compiled_decode(next_token, cache_position, rope_delta, *arena.flat_tensors())[:, -1, :].float(),
                dim=-1, keepdim=True))
            if graph_calls == 0:
                first_call_s = time.perf_counter() - begin
            begin = time.perf_counter()
            current = engine._schedule_token_copy(candidate, iteration=graph_calls,
                slot_requests=launched_requests, slot_epochs=launched_epochs)
            copy_submit_s += time.perf_counter() - begin
            graph_calls += 1
            for slot, index in enumerate(launched_requests):
                if index is not None:
                    next_token[slot:slot + 1].copy_(candidate[slot:slot + 1])
                    cache_position[slot].add_(1)
            if pending is not None:
                row, elapsed = engine._wait_token_copy(pending)
                copy_wait_s += elapsed
                for slot, index in enumerate(pending.slot_requests):
                    if index is None or slots[slot] != index or epochs[slot] != pending.slot_epochs[slot]:
                        continue
                    tokens[index].append(row[slot])
                    if row[slot] == engine.eos_token_id or len(tokens[index]) >= limits[index]:
                        slots[slot] = None
                        complete(index)
            pending = current
            if graph_calls % 1024 == 0:
                resolve_timing()
        if pending is not None:
            _, elapsed = engine._wait_token_copy(pending)
            copy_wait_s += elapsed
        maybe_sync_device(engine.model.device)
        resolve_timing()
    if tokens or limits or completed != request_count:
        raise RuntimeError("stream ended with incomplete requests")
    decode_s = sum(decode_by_occupancy.values())
    raw_slots = graph_calls * batch
    return {
        "enabled": True, "mode": "open_request_stream", "batch_size": batch,
        "cache_length": engine.cache_length, "request_count": request_count,
        "graph_calls": graph_calls, "decode_calls": effective_tokens,
        "raw_decode_token_slots": raw_slots, "active_decode_token_slots": active_slots_total,
        "idle_decode_token_slots": raw_slots - active_slots_total,
        "active_slot_fraction": active_slots_total / raw_slots if raw_slots else 0,
        "decode_time_weighted_active_fraction": sum(i * t for i, t in decode_by_occupancy.items()) / (batch * decode_s) if decode_s else None,
        "occupancy_histogram": dict(sorted(occupancy.items())),
        "decode_seconds_by_active_rows": dict(sorted(decode_by_occupancy.items())),
        "idle_rows_with_ready_work": idle_rows_with_ready_work,
        "max_live_generation_requests": max_live_requests,
        "inactive_filler_policy": "duplicate_first_real_row_retain_controls",
        "initial_inactive_filler_rows": initial_inactive_filler_rows,
        "initial_filler_source_slot": initial_filler_source_slot,
        "final_drain_wall_s": time.perf_counter() - drain_started if drain_started else 0,
        "refill_count": refill_count, "immediate_completion_count": immediate,
        "decode_s": decode_s, "prefill_s": prefill_s,
        "prefill_metrics": dict(prefill_metrics),
        "sampled_token_copy_submit_s": copy_submit_s,
        "sampled_token_d2h_wait_s": copy_wait_s, "hot_swap_safety_sync_s": safety_s,
        "generation_wall_s": time.perf_counter() - started,
        "compile_wrapper_s": compile_wrapper_s, "compiled_first_call_s": first_call_s,
        "compile": dict(compile_meta),
    }
