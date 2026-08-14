#!/usr/bin/env python3
"""Tests for production-prefill trace distribution accounting."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "prefill_trace", HERE / "prefill_trace.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrefillTraceTest(unittest.TestCase):
    def test_distribution_and_shape_histograms(self) -> None:
        events = [
            {
                "event": "recognition_crop_preprocess",
                "source_image_size": [320, 80],
                "processed_image_size": [320, 64],
                "stage_s": {"resize_s": 0.002},
                "wall_s": 0.003,
            },
            {
                "event": "vision_bucket_call",
                "bucket": "960x64_b16",
                "real_rows": 1,
                "physical_rows": 16,
                "physical_input_shape": [16, 3, 64, 960],
                "members": [{"processed_image_size": [320, 64]}],
                "host_stage_s": {"graph_submit_s": 0.001},
                "device_stage_s": {"graph_s": 0.010},
            },
            {
                "event": "text_prefill_pack",
                "member_count": 1,
                "real_source_tokens": 320,
                "members": [{"source_tokens": 320}],
                "wall_s": 0.004,
            },
            {
                "event": "cross_kv_d2h",
                "source_lengths": [320],
                "wall_s": 0.005,
            },
        ]
        pages = [{"frontend_timing_s": {"layout_s": 0.020}}]
        report = MODULE.summarize_trace(
            events,
            pages,
            config={"workers": 1, "recognition_preprocess_threads": 1},
        )
        self.assertEqual(report["event_count"], 4)
        self.assertEqual(
            report["stage_distributions"][
                "vision_bucket_call.device_stage_s.graph_s"
            ]["p50_ms"],
            10.0,
        )
        self.assertEqual(
            report["shape_histograms"]["crop_transform"]["320x80->320x64"],
            1,
        )
        self.assertEqual(
            report["shape_histograms"]["cross_kv_source_length"]["320"],
            1,
        )

    def test_write_prefill_trace(self) -> None:
        payload = {
            "page_index": 0,
            "image_path": "/dataset/page.jpg",
            "width": 100,
            "height": 200,
            "crops": [{}],
            "cross_capacity_rejected_crops": 0,
            "frontend_timing_s": {"layout_s": 0.01},
            "worker_prefill_stats": {},
            "prefill_trace_events": [
                {"event": "coordinator_ipc_delivery", "wall_s": 0.001}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.write_prefill_trace(
                Path(directory),
                [payload],
                config={"workers": 1, "recognition_preprocess_threads": 1},
            )
            self.assertEqual(report["page_count"], 1)
            self.assertTrue(Path(report["artifacts"]["iterations"]).is_file())
            self.assertTrue(Path(report["artifacts"]["pages"]).is_file())
            self.assertTrue(Path(report["artifacts"]["distributions"]).is_file())


if __name__ == "__main__":
    unittest.main()
