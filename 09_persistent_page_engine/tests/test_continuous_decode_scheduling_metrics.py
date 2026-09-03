"""Exercise real scheduler control on CPU with deterministic synthetic tokens."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model.text_decode import LocalPaddleOCRVLStaticCache
from paddleocr_vl.serving.continuous_decode import (
    ContinuousDecodeScheduler, DecodeArena, ReadyDecodeRequest,
)
from paddleocr_vl.serving.scheduling_metrics import RequestSchedulingMetrics
from paddleocr_vl.serving.engine import _OpenPrefillSource


def cache(batch_size):
    return LocalPaddleOCRVLStaticCache(
        key_caches=(torch.zeros((batch_size, 1, 32, 1)),),
        value_caches=(torch.zeros((batch_size, 1, 32, 1)),),
        cache_length=32,
    )


def run_case(batch_size, collect):
    recorder = RequestSchedulingMetrics(batch_size) if collect else None
    arena = DecodeArena(
        cache=cache(batch_size), device=torch.device("cpu"),
        batch_size=batch_size, eos_token_id=9,
        decode_token_id_map=torch.arange(10),
    )
    candidates = deque([("a", 1), ("b", 4), ("c", 3)])
    if batch_size >= 4:
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
            self.assertEqual(source.pull(block=False).request_id, "a")
            self.assertEqual(source.pull(block=False).request_id, "b")
            for _ in range(5):
                self.assertIsNone(source.pull(block=False))
            self.assertEqual(len(recorder.prefills), 2)
            recorder.step(["a", "b"], time.perf_counter())
            result = recorder.finish("b", time.perf_counter())
            self.assertEqual(result["before_first_decode_other_prefill_count"], 1)
        finally:
            source.close()

    def test_logging_preserves_tokens_and_fixed_batch(self):
        for batch_size in (1, 2, 4, 8, 16):
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
                if batch_size >= 4:
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
