#!/usr/bin/env python3
"""Tests for per-call prefill tail and contention analysis."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "analyze_prefill_tail_contention",
    HERE / "analyze_prefill_tail_contention.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzePrefillTailContentionTest(unittest.TestCase):
    def test_clean_progress_excludes_warmup_and_marks_warning_gap(self) -> None:
        log = "\n".join(
            (
                "UNIREC_LAYOUT_PROCESS_PAGE label=two_phase_warmup "
                "pages=1/8 page_index=0 worker=0 worker_page_s=0.9 "
                "elapsed_s=0.9 crops=1 rejected=0",
                "UNIREC_LAYOUT_PROCESS_PAGE label=two_phase_measured_prefill "
                "pages=1/2 page_index=0 worker=0 worker_page_s=0.1 "
                "elapsed_s=0.1 crops=2 rejected=0",
                "UserWarning: The given NumPy array is not writable",
                "UNIREC_LAYOUT_PROCESS_PAGE label=two_phase_measured_prefill "
                "pages=2/2 page_index=1 worker=1 worker_page_s=0.2 "
                "elapsed_s=0.4 crops=3 rejected=0",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log + "\n", encoding="utf-8")
            report = MODULE.clean_progress_analysis(path, limit=5)
        self.assertEqual(report["page_count"], 2)
        self.assertEqual(report["nonwriteable_warning_count"], 1)
        self.assertEqual(len(report["warning_adjacent_intervals"]), 1)
        self.assertAlmostEqual(
            report["warning_adjacent_intervals"][0]["completion_gap_s"],
            0.3,
        )

    def test_npu_service_sum_keeps_real_call_samples(self) -> None:
        events = [
            {
                "event": "layout_batch_call",
                "stage_s": {
                    "inputs_h2d_s": 0.005,
                    "model_forward_s": 0.01,
                    "outputs_d2h_s": 0.005,
                },
            },
            {
                "event": "vision_bucket_call",
                "device_stage_s": {
                    "input_h2d_normalize_s": 0.01,
                    "graph_s": 0.02,
                    "output_compact_s": 0.005,
                },
            },
            {
                "event": "vision_fallback_call",
                "device_stage_s": {
                    "input_h2d_normalize_s": 0.01,
                    "graph_s": 0.03,
                },
            },
            {
                "event": "text_prefill_pack",
                "device_stage_s": {
                    "compiled_packed_text_prefill_s1024": 0.04,
                    "static_cache_build_and_padding": 0.02,
                },
            },
            {"event": "cross_kv_d2h", "wall_s": 0.01},
        ]
        report = MODULE.npu_service_analysis(
            events,
            {"timing_s": {"prefill_phase": 0.05}},
        )
        self.assertAlmostEqual(report["aggregate_service_sum_s"], 0.165)
        self.assertAlmostEqual(
            report["aggregate_service_sum_over_trace_wall"], 3.3
        )
        self.assertEqual(
            report["components"]["vision_bucket_graph_s"]["count"], 1
        )

    def test_backpressure_splits_consecutive_vision_submissions(self) -> None:
        events = [
            {
                "trace_index": 1,
                "event": "vision_bucket_call",
                "host_stage_s": {"pixels_h2d_uint8_submit_s": 0.001},
            },
            {
                "trace_index": 2,
                "event": "vision_bucket_call",
                "host_stage_s": {"pixels_h2d_uint8_submit_s": 0.004},
            },
            {
                "trace_index": 8,
                "event": "vision_bucket_call",
                "host_stage_s": {"pixels_h2d_uint8_submit_s": 0.0004},
            },
        ]
        report = MODULE.vision_backpressure_analysis(events)
        self.assertEqual(report["consecutive_after_vision"]["count"], 1)
        self.assertEqual(report["nonconsecutive"]["count"], 1)
        self.assertAlmostEqual(report["mean_ratio"], 10.0)
        self.assertFalse(report["host_submit_is_device_service"])


if __name__ == "__main__":
    unittest.main()
