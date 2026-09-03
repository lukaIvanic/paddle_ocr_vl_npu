"""Deterministic timing tests; no accelerator or tokenizer is involved."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "request_scheduling_metrics", ROOT / "paddleocr_vl/serving/scheduling_metrics.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Recorder = MODULE.RequestSchedulingMetrics


class SchedulingMetricsTest(unittest.TestCase):
    def test_initial_fill_is_attributed_before_first_decode(self) -> None:
        metrics = Recorder(2)
        metrics.register("a", 0)
        metrics.register("b", 0)
        metrics.record_prefill("a", 0, .2)
        metrics.record_prefill("b", .2, .4)
        metrics.step(["a", "b"], .4)
        metrics.consume(["a", "b"])
        a = metrics.finish("a", 1)
        b = metrics.finish("b", 1)
        self.assertAlmostEqual(a["own_prefill_ready_to_first_decode_s"], .2)
        for result in (a, b):
            self.assertEqual(result["before_first_decode_other_prefill_count"], 1)
            self.assertAlmostEqual(result["before_first_decode_other_prefill_host_s"], .2)
            self.assertEqual(result["decode_other_prefill_count"], 0)
            self.assertEqual(result["launched_decode_iterations_by_active_slots"], {2: 1})

    def test_multiple_refills_and_lookahead_are_separate(self) -> None:
        metrics = Recorder(2)
        metrics.register("long", 0)
        metrics.record_prefill("long", 0, .2)
        metrics.step(["long"], .2)
        metrics.consume(["long"])
        for i in range(2):
            request_id = f"short{i}"
            start = 1 + i
            metrics.register(request_id, start)
            metrics.record_prefill(request_id, start, start + .2)
            metrics.step(["long", request_id], start + .2)
            metrics.consume(["long", request_id])
            # A look-ahead is launched before short's EOS is retired.
            metrics.step(["long", request_id], start + .3)
            short = metrics.finish(request_id, start + .4)
            self.assertEqual(short["launched_decode_iterations_by_active_slots"], {2: 2})
            self.assertEqual(short["consumed_decode_iterations_by_useful_slots"], {2: 1})
            metrics.consume(["long"])
        result = metrics.finish("long", 4)
        self.assertEqual(result["decode_other_prefill_count"], 2)
        self.assertAlmostEqual(result["decode_other_prefill_host_s"], .4)
        self.assertEqual(result["launched_decode_iterations_by_active_slots"], {1: 1, 2: 4})
        self.assertEqual(result["consumed_decode_iterations_by_useful_slots"], {1: 3, 2: 2})

    def test_late_registration_clips_prior_prefill_at_arrival(self) -> None:
        metrics = Recorder(2)
        metrics.register("a", 0)
        metrics.record_prefill("a", 0, .2)
        metrics.step(["a"], .2)
        metrics.register("b", .1)
        metrics.record_prefill("b", .25, .45)
        metrics.step(["a", "b"], .5)
        result = metrics.finish("b", 1)
        self.assertAlmostEqual(result["before_first_decode_other_prefill_host_s"], .1)
        self.assertEqual(result["other_prefill_spans"][0]["start_offset_s"], 0)

    def test_idle_time_is_not_an_interruption(self) -> None:
        metrics = Recorder(1)
        metrics.register("a", 10)
        metrics.record_prefill("a", 0, 10.2)
        metrics.step(["a"], 10.2)
        result = metrics.finish("a", 11)
        self.assertEqual(result["other_prefill_spans"], [])
        self.assertAlmostEqual(result["own_prefill_ready_offset_s"], .2)

    def test_prefill_only_completion_has_no_decode_iterations(self) -> None:
        metrics = Recorder(2)
        metrics.register("a", 0)
        metrics.record_prefill("a", 0, .2)
        result = metrics.finish("a", .2)
        self.assertIsNone(result["first_decode_offset_s"])
        self.assertEqual(result["launched_decode_iterations_by_active_slots"], {})

    def test_failed_cpu_preparation_still_delays_other_request(self) -> None:
        metrics = Recorder(2)
        metrics.register("a", 0)
        metrics.record_prefill("a", 0, .2)
        metrics.step(["a"], .2)
        metrics.register("bad", .3)
        metrics.record_prefill("bad", .3, .4, status="error")
        result = metrics.finish("a", .5)
        self.assertAlmostEqual(result["decode_other_prefill_host_s"], .1)
        self.assertEqual(result["other_prefill_spans"][0]["other_request_status"], "error")
        self.assertNotIn("bad", metrics.requests)


if __name__ == "__main__":
    unittest.main()
