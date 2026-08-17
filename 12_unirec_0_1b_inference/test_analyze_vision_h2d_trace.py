#!/usr/bin/env python3
"""Tests for granular UniRec vision H2D trace accounting."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "analyze_vision_h2d_trace",
    HERE / "analyze_vision_h2d_trace.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzeVisionH2DTraceTest(unittest.TestCase):
    def test_group_breakdown_preserves_submissions_bytes_and_residual(self) -> None:
        event = {
            "event": "vision_bucket_call",
            "bucket": "448x64_b4",
            "h2d_tensor_submissions": 7,
            "h2d_bytes": {
                "pixels": 300,
                "pixel_mask": 100,
                "mask2": 50,
                "mask4": 25,
                "mask8": 12,
                "mask16": 6,
                "mask32": 3,
                "total": 496,
            },
            "device_stage_s": {
                "input_h2d_normalize_s": 0.010,
                "pixels_h2d_uint8_s": 0.002,
                "pixels_normalize_layout_s": 0.001,
                "pixel_mask_h2d_cast_s": 0.001,
                "pixel_mask_apply_s": 0.001,
                "mask2_h2d_s": 0.001,
                "mask4_h2d_s": 0.001,
                "mask8_h2d_s": 0.001,
                "mask16_h2d_s": 0.001,
                "mask32_h2d_s": 0.0005,
            },
            "host_stage_s": {
                "input_h2d_submit_s": 0.012,
                "pixels_h2d_uint8_submit_s": 0.002,
                "pixels_normalize_layout_submit_s": 0.001,
                "pixel_mask_h2d_cast_submit_s": 0.001,
                "pixel_mask_apply_submit_s": 0.001,
                "mask2_h2d_submit_s": 0.001,
                "mask4_h2d_submit_s": 0.001,
                "mask8_h2d_submit_s": 0.001,
                "mask16_h2d_submit_s": 0.001,
                "mask32_h2d_submit_s": 0.0005,
            },
        }
        report = MODULE.summarize_group([event], top=3)
        self.assertEqual(report["call_count"], 1)
        self.assertEqual(report["h2d_tensor_submissions"], 7)
        self.assertEqual(report["h2d_bytes"]["total"], 496)
        self.assertAlmostEqual(
            report["device_accounting_residual"]["sum_s"], 0.0005
        )
        self.assertAlmostEqual(
            report["host_accounting_residual"]["sum_s"], 0.0025
        )


if __name__ == "__main__":
    unittest.main()
