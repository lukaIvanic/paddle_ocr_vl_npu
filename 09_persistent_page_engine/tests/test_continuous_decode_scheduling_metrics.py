"""Exercise real scheduler control on CPU with deterministic synthetic tokens."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model.text_decode import LocalPaddleOCRVLStaticCache
from paddleocr_vl.serving.continuous_decode import (
    ContinuousDecodeScheduler, DecodeArena, ReadyDecodeRequest,
)
from paddleocr_vl.serving.scheduling_metrics import RequestSchedulingMetrics
from paddleocr_vl.serving.engine import _OpenPrefillSource


def cache(batch_size, length=32):
    return LocalPaddleOCRVLStaticCache(
        key_caches=(torch.zeros((batch_size, 1, length, 1)),),
        value_caches=(torch.zeros((batch_size, 1, length, 1)),),
        cache_length=length,
    )


def run_case(batch_size, collect, device_timing=True, compact=False):
    recorder = RequestSchedulingMetrics(batch_size) if collect else None
    arena = DecodeArena(
        cache=cache(batch_size), device=torch.device("cpu"),
        batch_size=batch_size, eos_token_id=9,
        decode_token_id_map=torch.arange(10),
        decode_device_timing=device_timing,
        compact_step_control=compact,
    )
    candidates = deque([("a", 1), ("b", 4), ("c", 3)])
    if batch_size >= 3:
        candidates.extend((f"extra-{i}", 3) for i in range(batch_size - 2))
    submitted = time.perf_counter()
    available = deque((candidates.popleft(), submitted) for _ in range(batch_size))
    emitted = []
    shapes = []

    class Source:
        @property
        def closed(self):
            return not available and not candidates

        def pull(self, *, block):
            if not available:
                return None
            (request_id, first_token), arrived = available.popleft()
            started = time.perf_counter()
            if recorder:
                recorder.register(request_id, arrived)
            result = ReadyDecodeRequest(
                request_id=request_id, payload=None, cache=cache(1),
                rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
                cache_position=torch.ones(1, dtype=torch.int64),
                first_token_tensor=torch.tensor([[first_token]]),
                first_token=first_token, prompt_length=1,
            )
            if recorder:
                recorder.record_prefill(request_id, started, time.perf_counter())
            return result

    def decode(tokens, *unused):
        shapes.append(tuple(tokens.shape))
        return torch.where(tokens >= 5, torch.full_like(tokens, 9), tokens + 1)

    def complete(completion):
        emitted.append(completion)
        if candidates:
            available.append((candidates.popleft(), time.perf_counter()))

    scheduler = ContinuousDecodeScheduler(arena=arena, decode_fn=decode, max_new_tokens=20)
    result = scheduler.run_stream(Source(), on_completion=complete, scheduling_metrics=recorder)
    return result, emitted, shapes


class ContinuousDecodeMetricsTest(unittest.TestCase):
    def test_open_prefill_reserves_only_free_slots_under_backlog(self):
        for batch in (1, 2, 8):
            for late_arrivals in (False, True):
                with self.subTest(batch=batch, late_arrivals=late_arrivals):
                    arena = DecodeArena(cache=cache(batch), device=torch.device("cpu"),
                                        batch_size=batch, eos_token_id=9,
                                        decode_token_id_map=torch.arange(10))
                    names = [str(i) for i in range(batch * 3 + 1)]
                    incoming = deque(names[:batch] if late_arrivals else names)
                    arrived = not late_arrivals
                    live, emitted, prepared = set(), [], []
                    prepared_while_full = []
                    calls = 0

                    class Requests:
                        @property
                        def closed(self):
                            return arrived and not incoming

                        def pull(self, *, block):
                            return (SimpleNamespace(request_id=incoming.popleft(), submitted_at=None)
                                    if incoming else None)

                    class Executor:
                        def submit(self, fn, *args):
                            f = Future()
                            try:
                                f.set_result(fn(*args))
                            except BaseException as exc:
                                f.set_exception(exc)
                            return f

                        def shutdown(self, **kwargs):
                            pass

                    def prepare(request, submitted):
                        prepared.append(request.request_id)
                        if arena.num_active == batch:
                            prepared_while_full.append(request.request_id)
                        return request

                    def prefill(request):
                        # Includes active AND returned-but-not-yet-admitted KV.
                        self.assertLess(len(live), batch, "NPU prefilled ahead of free slots")
                        live.add(request.request_id)
                        token = 9 if request.request_id == names[-1] else 1
                        return ReadyDecodeRequest(
                            request_id=request.request_id, payload=None, cache=cache(1),
                            rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
                            cache_position=torch.ones(1, dtype=torch.int64),
                            first_token_tensor=torch.tensor([[token]]), first_token=token,
                            prompt_length=1)

                    recognizer = SimpleNamespace(
                        cpu_preprocess_max_pending=max(2, batch), _prepare_cpu=prepare,
                        _prepared_group=lambda members: members[0][0],
                        _stage_prefill_group=prefill,
                        _enqueue_staged_prefill_group=lambda x:x,
                        _finalize_prefill_group=lambda x:[x], _ready_from_prefilled=lambda x:x,
                    )
                    with patch("paddleocr_vl.serving.engine.ThreadPoolExecutor", return_value=Executor()):
                        source = _OpenPrefillSource(recognizer, Requests(),
                                                    on_request_error=lambda *args:self.fail(str(args)))

                    def decode(tokens, *args):
                        nonlocal arrived, calls
                        calls += 1
                        self.assertEqual(tuple(tokens.shape), (batch, 1))
                        if not arrived:
                            self.assertEqual(arena.num_active, batch)
                            incoming.extend(names[batch:])
                            arrived = True
                        return torch.where(tokens >= 5, torch.full_like(tokens, 9), tokens + 1)

                    def complete(result):
                        live.remove(result.ready.request_id)
                        emitted.append(result)

                    scheduler = ContinuousDecodeScheduler(arena=arena, decode_fn=decode,
                                                          max_new_tokens=20)
                    try:
                        result = scheduler.run_stream(source, on_completion=complete)
                        self.assertEqual(result.submitted_requests, len(names))
                        self.assertEqual(len(emitted), len(names))
                        self.assertFalse(live)
                        self.assertTrue(source.closed)
                        self.assertGreater(calls, 0)
                        self.assertCountEqual(prepared, names)
                        if late_arrivals:
                            self.assertTrue(prepared_while_full, "CPU lookahead stopped with full decode")
                        for item in emitted:
                            self.assertEqual(item.token_ids, [9] if item.ready.request_id == names[-1]
                                             else [1, 2, 3, 4, 5, 9])
                    finally:
                        source.close()

    def test_compact_control_matches_full_scheduler_across_refills(self):
        for batch in (1, 2, 3, 4, 8):
            control, old, old_shapes = run_case(batch, True)
            candidate, new, new_shapes = run_case(batch, True, compact=True)
            self.assertEqual([x.token_ids for x in old], [x.token_ids for x in new])
            self.assertEqual([x.stop_reason for x in old], [x.stop_reason for x in new])
            self.assertEqual(old_shapes, new_shapes)
            self.assertEqual(control.graph_calls, candidate.graph_calls)
            self.assertEqual(control.hot_swap_admissions, candidate.hot_swap_admissions)

    def test_compact_control_keeps_output_storage_separate_and_positions_stable(self):
        arena = DecodeArena(cache=cache(2), device=torch.device("cpu"), batch_size=2,
                            eos_token_id=9, decode_token_id_map=torch.arange(10),
                            compact_step_control=True)
        ready = ReadyDecodeRequest("a", None, cache(1), torch.zeros((1, 1), dtype=torch.long),
                                   torch.ones(1, dtype=torch.long), torch.tensor([[2]]), 2, 1)
        arena.admit(0, ready, hot_swap=False)
        token_ptr, position_ptr = arena.next_token.data_ptr(), arena.cache_position.data_ptr()
        step = arena.step(lambda tokens, *args: tokens + 1, iteration=0)
        saved = step.sampled.clone()
        self.assertNotEqual(step.sampled.data_ptr(), arena.next_token.data_ptr())
        self.assertEqual(arena.next_token.data_ptr(), token_ptr)
        self.assertEqual(arena.cache_position.data_ptr(), position_ptr)
        self.assertEqual(arena.cache_position.tolist(), [2, 0])
        arena.release(0)
        self.assertTrue(torch.equal(step.sampled, saved))
        self.assertEqual(arena.active_increment.tolist(), [0, 0])
        arena.step(lambda tokens, *args: tokens + 1, iteration=1)
        self.assertEqual(arena.cache_position.tolist(), [0, 0])

    def test_disabling_profiler_events_preserves_outputs_and_reports_unavailable(self):
        from utils.metrics import per_second
        for batch in (1, 2, 4):
            control, old_outputs, old_shapes = run_case(batch, True)
            fast, outputs, shapes = run_case(batch, True, device_timing=False)
            self.assertEqual([x.token_ids for x in old_outputs], [x.token_ids for x in outputs])
            self.assertEqual(old_shapes, shapes)
            self.assertEqual(control.graph_calls, fast.graph_calls)
            self.assertIsNone(fast.timing_s["decode_model_and_argmax_device"])
            self.assertIsNone(fast.timing_s["continuous_decode_wall"])
            self.assertIsNone(per_second(fast.effective_decode_tokens, None))
            self.assertGreater(fast.timing_s["run_scoped_scheduler_wall"], 0)

    def run_capped_case(self, limit, *, collect=True, length=32, fail=False):
        recorder = RequestSchedulingMetrics(2) if collect else None
        arena = DecodeArena(
            cache=cache(2, length), device=torch.device("cpu"), batch_size=2,
            eos_token_id=15, decode_token_id_map=torch.arange(16),
        )
        candidates = deque([
            ("long", 1), ("short0", 14), ("short1", 14), ("short2", 14),
            ("long2", 1), ("short3", 14), ("empty", 15),
            ("short4", 14), ("short5", 14), ("short6", 14),
        ])
        available = deque(candidates.popleft() for _ in range(2))
        emitted, events, shapes, errors = [], [], [], []

        class Requests:
            @property
            def closed(self):
                return not available and not candidates

            def pull(self, *, block):
                if not available:
                    return None
                name, token = available.popleft()
                events.append(("pull", name))
                return SimpleNamespace(
                    request_id=name, first_token=token,
                    submitted_at=time.perf_counter(),
                )

        class Recognizer:
            cpu_preprocess_max_pending = 2
            decode_scheduler = SimpleNamespace(arena=arena)

            def _prepare_cpu(self, request, submitted_at):
                if fail and request.request_id in {"short1", "short2"}:
                    raise ValueError("test preparation failure")
                return ReadyDecodeRequest(
                    request_id=request.request_id, payload=None,
                    cache=cache(1, length),
                    rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
                    cache_position=torch.ones(1, dtype=torch.int64),
                    first_token_tensor=torch.tensor([[request.first_token]]),
                    first_token=request.first_token, prompt_length=1,
                )

            def _prepared_group(self, members):
                return members[0][0]

            def _stage_prefill_group(self, group):
                return group

            def _enqueue_staged_prefill_group(self, group):
                events.append(("prefill", group.request_id))
                return group

            def _finalize_prefill_group(self, group):
                return [group]

            def _ready_from_prefilled(self, state):
                return state

        def refill_client():
            if candidates:
                available.append(candidates.popleft())

        def complete(completion):
            emitted.append(completion)
            events.append(("complete", completion.ready.request_id))
            refill_client()

        def error(request_id, exc):
            errors.append(request_id)
            refill_client()

        def decode(tokens, *unused):
            shapes.append(tuple(tokens.shape))
            return torch.where(tokens >= 14, torch.full_like(tokens, 15), tokens + 1)

        class ImmediateExecutor:
            # These tests isolate interruption-cap policy, not OS CPU-worker
            # scheduling. Delayed CPU preparation has its own tests below.
            def __init__(self, **kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                try:
                    future.set_result(fn(*args))
                except BaseException as exc:
                    future.set_exception(exc)
                return future

            def shutdown(self, **kwargs):
                pass

        with patch("paddleocr_vl.serving.engine.ThreadPoolExecutor", ImmediateExecutor):
            source = _OpenPrefillSource(
                Recognizer(), Requests(), on_request_error=error,
                scheduling_metrics=recorder, max_prefill_interruptions=limit,
            )
        scheduler = ContinuousDecodeScheduler(arena=arena, decode_fn=decode, max_new_tokens=30)
        try:
            result = scheduler.run_stream(
                source, on_completion=complete, scheduling_metrics=recorder,
            )
            self.assertTrue(source.closed)
        finally:
            source.close()
        self.assertTrue(all(shape == (2, 1) for shape in shapes))
        self.assertEqual(result.submitted_requests, len(emitted))
        self.assertEqual(len(emitted) + len(errors), 10)
        return result, emitted, events

    def test_two_interruptions_then_decode_to_completion(self):
        plain, before, _ = self.run_capped_case(None)
        capped, after, events = self.run_capped_case(2)
        outputs = lambda rows: {row.ready.request_id: row.token_ids for row in rows}
        self.assertEqual(outputs(before), outputs(after))
        self.assertGreater(capped.idle_decode_token_slots, plain.idle_decode_token_slots)
        metrics = {r.ready.request_id: r.scheduling_metrics for r in after}
        self.assertEqual(metrics["long"]["decode_other_prefill_count"], 2)
        self.assertEqual(metrics["long2"]["decode_other_prefill_count"], 2)
        self.assertTrue(all(m["decode_other_prefill_count"] <= 2 for m in metrics.values()))
        self.assertLess(events.index(("complete", "long")), events.index(("pull", "long2")))

    def test_cap_does_not_depend_on_logging(self):
        _, measured, _ = self.run_capped_case(2)
        _, plain, _ = self.run_capped_case(2, collect=False)
        self.assertEqual(
            [(r.ready.request_id, r.token_ids) for r in measured],
            [(r.ready.request_id, r.token_ids) for r in plain],
        )

    def test_zero_cap_and_cache_boundary_still_drain(self):
        _, emitted, _ = self.run_capped_case(0, length=8)
        self.assertIn("kv_cache_full", [r.stop_reason for r in emitted])
        self.assertTrue(all(r.scheduling_metrics["decode_other_prefill_count"] == 0 for r in emitted))

    def test_failed_preparation_counts_and_rechecks_cap(self):
        _, emitted, events = self.run_capped_case(2, fail=True)
        long = next(r for r in emitted if r.ready.request_id == "long")
        self.assertEqual(long.scheduling_metrics["decode_other_prefill_count"], 2)
        self.assertLess(events.index(("complete", "long")), events.index(("pull", "long2")))

    def test_open_source_tracks_preparation_but_not_empty_polls(self):
        recorder = RequestSchedulingMetrics(2)
        arrived = time.perf_counter()
        incoming = deque([
            SimpleNamespace(request_id="a", submitted_at=arrived),
            SimpleNamespace(request_id="b", submitted_at=arrived),
        ])

        class Requests:
            closed = False

            def pull(self, *, block):
                return incoming.popleft() if incoming else None

        class Recognizer:
            cpu_preprocess_max_pending = 2

            def _prepare_cpu(self, request, submitted_at):
                return request

            def _prepared_group(self, members):
                return members[0][0]

            def _stage_prefill_group(self, group):
                return group

            def _enqueue_staged_prefill_group(self, group):
                return group

            def _finalize_prefill_group(self, group):
                return [group]

            def _ready_from_prefilled(self, state):
                return state

        def fail(request_id, exc):
            self.fail(f"unexpected failure: {request_id}: {exc}")

        source = _OpenPrefillSource(
            Recognizer(), Requests(), on_request_error=fail,
            scheduling_metrics=recorder,
        )
        try:
            self.assertEqual(source.pull(block=True).request_id, "a")
            self.assertEqual(source.pull(block=True).request_id, "b")
            for _ in range(5):
                self.assertIsNone(source.pull(block=False))
            self.assertEqual(len(recorder.prefills), 2)
            recorder.step(["a", "b"], time.perf_counter())
            result = recorder.finish("b", time.perf_counter())
            self.assertEqual(result["before_first_decode_other_prefill_count"], 1)
        finally:
            source.close()

    def test_unfinished_cpu_preparation_does_not_block_or_disappear(self):
        prepared = SimpleNamespace(request_id="arriving")
        future = Future()
        prefilled = []
        active = SimpleNamespace(first_decode_launched_at=1.0, prefill_interruptions=0)
        recognizer = SimpleNamespace(
            cpu_preprocess_max_pending=1,
            decode_scheduler=SimpleNamespace(arena=SimpleNamespace(slots=[active, None])),
            _prepared_group=lambda members: members[0][0],
            _stage_prefill_group=lambda group: group,
            _enqueue_staged_prefill_group=lambda group: prefilled.append(group) or group,
            _finalize_prefill_group=lambda group: [group],
            _ready_from_prefilled=lambda item: item,
        )
        requests = SimpleNamespace(closed=True, pull=lambda **kwargs: None)
        recorder = RequestSchedulingMetrics(2)
        recorder.register("arriving", time.perf_counter())
        source = _OpenPrefillSource(recognizer, requests,
                                    on_request_error=lambda *args: self.fail(str(args)),
                                    scheduling_metrics=recorder, max_prefill_interruptions=2)
        source.pending.append(("arriving", future))
        try:
            for _ in range(3):
                self.assertIsNone(source.pull(block=False))
                self.assertFalse(source.closed)
                self.assertEqual(len(source.pending), 1)
                self.assertEqual(active.prefill_interruptions, 0)
                self.assertEqual(len(recorder.prefills), 0)
                self.assertEqual(prefilled, [])
            future.set_result(prepared)
            self.assertIs(source.pull(block=False), prepared)
            self.assertTrue(source.closed)
            self.assertEqual(prefilled, [prepared])
            self.assertEqual(active.prefill_interruptions, 1)
            self.assertEqual(len(recorder.prefills), 1)
        finally:
            source.close()

    def test_idle_pull_waits_for_cpu_future(self):
        class FinishOnWait(Future):
            def result(self, timeout=None):
                self.set_exception(ValueError("prepared failure"))
                return super().result(timeout)

        errors = []
        source = _OpenPrefillSource(
            SimpleNamespace(cpu_preprocess_max_pending=1),
            SimpleNamespace(closed=True, pull=lambda **kwargs: None),
            on_request_error=lambda key, exc: errors.append((key, str(exc))),
        )
        source.pending.append(("arriving", FinishOnWait()))
        try:
            self.assertIsNone(source.pull(block=True))
            self.assertEqual(errors, [("arriving", "prepared failure")])
            self.assertTrue(source.closed)
        finally:
            source.close()

    def test_scheduler_decodes_while_second_request_is_preparing(self):
        arena = DecodeArena(cache=cache(2), device=torch.device("cpu"), batch_size=2,
                            eos_token_id=9, decode_token_id_map=torch.arange(10))
        incoming = deque(["first", "second"])
        emitted, calls = [], []

        def ready(name):
            return ReadyDecodeRequest(
                request_id=name, payload=None, cache=cache(1),
                rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
                cache_position=torch.ones(1, dtype=torch.int64),
                first_token_tensor=torch.tensor([[1]]), first_token=1, prompt_length=1,
            )

        class UnfinishedFuture(Future):
            def result(self, timeout=None):
                # An accidental wait must fail this test, not hang forever.
                if not self.done():
                    raise AssertionError("decoder waited for CPU before making progress")
                return super().result(timeout)

        second = UnfinishedFuture()

        class Executor:
            def submit(self, fn, request, submitted):
                if request.request_id == "second":
                    return second
                done = Future()
                done.set_result(ready(request.request_id))
                return done

            def shutdown(self, **kwargs):
                pass

        class Requests:
            @property
            def closed(self):
                return not incoming

            def pull(self, **kwargs):
                return (SimpleNamespace(request_id=incoming.popleft(), submitted_at=None)
                        if incoming else None)
        recognizer = SimpleNamespace(
            cpu_preprocess_max_pending=2, _prepare_cpu=None,
            _prepared_group=lambda members: members[0][0],
            _stage_prefill_group=lambda group: group,
            _enqueue_staged_prefill_group=lambda group: group,
            _finalize_prefill_group=lambda group: [group],
            _ready_from_prefilled=lambda item: item,
        )
        with patch("paddleocr_vl.serving.engine.ThreadPoolExecutor", return_value=Executor()):
            source = _OpenPrefillSource(recognizer, Requests(),
                                        on_request_error=lambda *args: self.fail(str(args)))

        def decode(tokens, *unused):
            calls.append(tuple(tokens.shape))
            if len(calls) == 3:
                self.assertEqual([s.ready.request_id for s in arena.slots if s is not None], ["first"])
                self.assertEqual(len(source.pending), 1)
                second.set_result(ready("second"))
            return torch.where(tokens >= 5, torch.full_like(tokens, 9), tokens + 1)

        scheduler = ContinuousDecodeScheduler(arena=arena, decode_fn=decode, max_new_tokens=20)
        try:
            result = scheduler.run_stream(source, on_completion=emitted.append)
            self.assertEqual(result.submitted_requests, 2)
            self.assertEqual({r.ready.request_id: r.token_ids for r in emitted},
                             {name: [1, 2, 3, 4, 5, 9] for name in ("first", "second")})
            self.assertTrue(all(shape == (2, 1) for shape in calls))
            self.assertTrue(source.closed)
        finally:
            source.close()

    def test_logging_preserves_tokens_and_fixed_batch(self):
        for batch_size in (1, 2, 3, 4, 8, 16):
            with self.subTest(batch_size=batch_size):
                before, plain, _ = run_case(batch_size, False)
                after, measured, shapes = run_case(batch_size, True)
                outputs = lambda rows: {row.ready.request_id: row.token_ids for row in rows}
                self.assertEqual(outputs(plain), outputs(measured))
                self.assertEqual(before.graph_calls, after.graph_calls)
                self.assertTrue(all(shape == (batch_size, 1) for shape in shapes))
                expected = {
                    "a": [1, 2, 3, 4, 5, 9], "b": [4, 5, 9], "c": [3, 4, 5, 9],
                }
                if batch_size >= 3:
                    expected.update({f"extra-{i}": [3, 4, 5, 9] for i in range(batch_size - 2)})
                self.assertEqual(outputs(measured), expected)
                for item in measured:
                    metrics = item.scheduling_metrics
                    self.assertEqual(
                        sum(metrics["launched_decode_iterations_by_active_slots"].values()),
                        item.iterations_launched,
                    )
                    self.assertEqual(
                        sum(metrics["consumed_decode_iterations_by_useful_slots"].values()),
                        len(item.token_ids) - 1,
                    )
                    if batch_size == 1:
                        self.assertEqual(metrics["other_prefill_spans"], [])
                if batch_size == 2:
                    by_id = {item.ready.request_id: item.scheduling_metrics for item in measured}
                    self.assertEqual(measured[0].ready.request_id, "b")
                    self.assertEqual(by_id["a"]["before_first_decode_other_prefill_count"], 1)
                    self.assertEqual(by_id["a"]["decode_other_prefill_count"], 1)
                    self.assertGreater(after.idle_decode_token_slots, 0)
                    self.assertGreater(after.lookahead_decode_token_slots, 0)


if __name__ == "__main__":
    unittest.main()
