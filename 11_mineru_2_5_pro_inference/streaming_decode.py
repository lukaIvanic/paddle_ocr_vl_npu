"""Open-source continuous decode using the existing MinerU prefill/graph kernels.

Source contract: pull(block=...) -> (unique integer, PreparedGeneration) or
None; closed distinguishes EOF from temporary starvation; complete receives
unfiltered IDs once. Source completion may create additional requests.
"""
from __future__ import annotations

from collections import Counter, deque
import time

from phase_logging import log_phase


def decode_mask_preflight(cache_position, cache_length):
    """Make the first decode mask contract host-visible before graph replay."""
    positions = [int(value) for value in cache_position.detach().cpu().reshape(-1).tolist()]
    invalid = [value for value in positions if value < 0 or value >= int(cache_length)]
    if invalid:
        raise ValueError(
            "decode cache positions would create invalid attention rows: "
            f"positions={positions} cache_length={int(cache_length)}"
        )
    valid_key_counts = [value + 1 for value in positions]
    if not valid_key_counts or min(valid_key_counts) <= 0:
        raise ValueError("decode attention contains a fully masked query row")
    return {
        "cache_positions": positions,
        "valid_key_counts": valid_key_counts,
        "min_valid_keys": min(valid_key_counts),
        "max_valid_keys": max(valid_key_counts),
        "all_rows_have_finite_attention": True,
    }


def decode_step_state(cache_position, *, cache_length, boundary_period):
    """Return the exact per-row state used by one production decode call."""
    positions = [int(value) for value in cache_position.detach().cpu().reshape(-1).tolist()]
    if not positions:
        raise ValueError("decode step has no cache positions")
    if any(value < 0 or value >= int(cache_length) for value in positions):
        raise ValueError(
            "decode step has an invalid cache position: "
            f"positions={positions} cache_length={int(cache_length)}"
        )
    period = int(boundary_period)
    if period <= 0:
        raise ValueError("decode diagnostic boundary period must be positive")
    effective_lengths = [value + 1 for value in positions]
    position_histogram = dict(sorted(Counter(positions).items()))
    effective_length_histogram = dict(sorted(Counter(effective_lengths).items()))
    return {
        "cache_positions": positions,
        "position_histogram": position_histogram,
        "effective_lengths": effective_lengths,
        "effective_length_histogram": effective_length_histogram,
        "boundary_period": period,
        "effective_length_residues": [value % period for value in effective_lengths],
        "boundary_rows": [
            index for index, value in enumerate(effective_lengths) if value % period == 0
        ],
    }


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
    diagnostic_steps = max(0, int(getattr(engine, "decode_diagnostic_steps", 0)))
    diagnostic_sync = bool(getattr(engine, "decode_diagnostic_sync", False))
    diagnostic_boundary_period = int(
        getattr(engine, "decode_diagnostic_boundary_period", 1280)
    )
    filler_control = str(getattr(engine, "decode_filler_control", "retain"))
    if filler_control not in ("retain", "advance"):
        raise ValueError(f"unsupported decode filler control: {filler_control!r}")

    def log_step(step):
        return int(step) < diagnostic_steps

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
            first_graph_call = graph_calls == 0
            detailed_step = log_step(graph_calls)
            if detailed_step:
                state = decode_step_state(
                    cache_position,
                    cache_length=engine.cache_length,
                    boundary_period=diagnostic_boundary_period,
                )
                log_phase(
                    "decode_step_state",
                    "finish",
                    step=int(graph_calls),
                    active_rows=int(active),
                    inactive_rows=int(batch - active),
                    active_slots=[
                        slot for slot, request in enumerate(launched_requests)
                        if request is not None
                    ],
                    filler_control=filler_control,
                    **state,
                )
            if first_graph_call:
                preflight = decode_mask_preflight(cache_position, engine.cache_length)
                log_phase(
                    "decode_mask_preflight",
                    "finish",
                    batch_size=int(batch),
                    active_rows=int(active),
                    inactive_filler_rows=int(batch - active),
                    filler_source_slot=initial_filler_source_slot,
                    cache_length=int(engine.cache_length),
                    **preflight,
                )
                log_phase(
                    "decode_graph_first_call",
                    "start",
                    batch_size=int(batch),
                    active_rows=int(active),
                    inactive_filler_rows=int(batch - active),
                    cache_length=int(engine.cache_length),
                    cache_dir=compile_meta.get("torchair_cache_dir"),
                    cache_was_warm=compile_meta.get("cache_was_warm"),
                    attention=compile_meta.get("decode_attention"),
                )
            if detailed_step:
                log_phase(
                    "decode_step_graph",
                    "start",
                    step=int(graph_calls),
                    active_rows=int(active),
                    diagnostic_sync=diagnostic_sync,
                )
            begin = time.perf_counter()
            candidate = decode_timeline.measure(str(active), lambda: torch.argmax(
                compiled_decode(next_token, cache_position, rope_delta, *arena.flat_tensors())[:, -1, :].float(),
                dim=-1, keepdim=True))
            if detailed_step and diagnostic_sync:
                maybe_sync_device(engine.model.device)
            if detailed_step:
                log_phase(
                    "decode_step_graph",
                    "finish",
                    step=int(graph_calls),
                    active_rows=int(active),
                    diagnostic_sync=diagnostic_sync,
                    elapsed_s=float(time.perf_counter() - begin),
                )
            if first_graph_call:
                if not (detailed_step and diagnostic_sync):
                    maybe_sync_device(engine.model.device)
                first_call_s = time.perf_counter() - begin
                log_phase(
                    "decode_graph_first_call",
                    "finish",
                    batch_size=int(batch),
                    active_rows=int(active),
                    cache_length=int(engine.cache_length),
                    cache_dir=compile_meta.get("torchair_cache_dir"),
                    elapsed_s=float(first_call_s),
                )
            begin = time.perf_counter()
            if detailed_step:
                log_phase(
                    "decode_step_token_copy_submit",
                    "start",
                    step=int(graph_calls),
                )
            current = engine._schedule_token_copy(candidate, iteration=graph_calls,
                slot_requests=launched_requests, slot_epochs=launched_epochs)
            copy_submit_s += time.perf_counter() - begin
            if detailed_step:
                log_phase(
                    "decode_step_token_copy_submit",
                    "finish",
                    step=int(graph_calls),
                    elapsed_s=float(time.perf_counter() - begin),
                )
            graph_calls += 1
            if filler_control == "advance":
                next_token.copy_(candidate)
                cache_position.add_(1)
            else:
                for slot, index in enumerate(launched_requests):
                    if index is not None:
                        next_token[slot:slot + 1].copy_(candidate[slot:slot + 1])
                        cache_position[slot].add_(1)
            if detailed_step:
                log_phase(
                    "decode_step_control_update",
                    "finish",
                    step=int(graph_calls - 1),
                    filler_control=filler_control,
                )
            if pending is not None:
                if detailed_step:
                    log_phase(
                        "decode_step_previous_token_copy_wait",
                        "start",
                        step=int(graph_calls - 1),
                        pending_step=int(pending.iteration),
                    )
                row, elapsed = engine._wait_token_copy(pending)
                copy_wait_s += elapsed
                if detailed_step:
                    log_phase(
                        "decode_step_previous_token_copy_wait",
                        "finish",
                        step=int(graph_calls - 1),
                        pending_step=int(pending.iteration),
                        elapsed_s=float(elapsed),
                    )
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
        "inactive_filler_policy": f"duplicate_first_real_row_{filler_control}_controls",
        "initial_inactive_filler_rows": initial_inactive_filler_rows,
        "initial_filler_source_slot": initial_filler_source_slot,
        "filler_control": filler_control,
        "decode_diagnostic_steps": diagnostic_steps,
        "decode_diagnostic_sync": diagnostic_sync,
        "decode_diagnostic_boundary_period": diagnostic_boundary_period,
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
