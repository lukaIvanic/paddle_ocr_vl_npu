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
                "stage_s": {"model_forward_s": 0.01},
            },
            {
                "event": "vision_bucket_call",
                "device_stage_s": {"graph_s": 0.02},
            },
            {
                "event": "vision_fallback_call",
                "device_stage_s": {"graph_s": 0.03},
            },
            {
                "event": "text_prefill_pack",
                "device_stage_s": {
                    "compiled_packed_text_prefill_s1024": 0.04,
                    "static_cache_build_and_padding": 1.0,
                },
            },
        ]
        report = MODULE.npu_service_analysis(
            events,
            {"timing_s": {"prefill_phase": 0.05}},
        )
        self.assertAlmostEqual(report["aggregate_service_sum_s"], 0.1)
        self.assertAlmostEqual(
            report["aggregate_service_sum_over_trace_wall"], 2.0
        )
        self.assertEqual(
            report["components"]["vision_bucket_graph_s"]["count"], 1
        )


if __name__ == "__main__":
    unittest.main()
