"""Tests for durable crop API service summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/serve_crop_ocr_api.py"
)
SPEC = importlib.util.spec_from_file_location("serve_crop_ocr_api", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CropApiServiceSummaryTest(unittest.TestCase):
    def test_setup_gc_freeze_is_opt_in(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            self.assertFalse(MODULE.parse_args().freeze_setup_gc)
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py", "--freeze-setup-gc"]):
            self.assertTrue(MODULE.parse_args().freeze_setup_gc)

    def test_new_request_cycles_remain_collectable(self) -> None:
        # GC freezing is interpreter-global; isolate this test from the runner.
        code = f"""
import gc, importlib.util, weakref
spec = importlib.util.spec_from_file_location('crop_api_gc_test', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class Request:
    pass
old = Request()
state = module._freeze_setup_gc()
assert state['gc_remains_enabled'] and state['frozen_objects'] > 0
new = Request()
new.cycle = new
reference = weakref.ref(new)
del new
gc.collect()
assert reference() is None
gc.unfreeze()
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=15)

    def test_prefill_interruption_limit_is_opt_in(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            self.assertIsNone(MODULE.parse_args().max_prefill_interruptions)
        with patch.object(sys, "argv", [
            "serve_crop_ocr_api.py", "--max-prefill-interruptions", "2",
        ]):
            self.assertEqual(MODULE.parse_args().max_prefill_interruptions, 2)

    def test_scheduling_metrics_are_opt_in(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            self.assertFalse(MODULE.parse_args().request_scheduling_metrics)
        with patch.object(sys, "argv", [
            "serve_crop_ocr_api.py", "--decode-batch-size", "2",
            "--request-scheduling-metrics",
        ]):
            args = MODULE.parse_args()
            self.assertTrue(args.request_scheduling_metrics)
            self.assertEqual(args.decode_batch_size, 2)

    def test_writer_preserves_configuration_and_scheduler_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "service_summary.json"
            MODULE._write_service_summary(
                output,
                configuration={"batch_size": 8},
                worker_pid=123,
                summary={
                    "graph_calls": 10,
                    "raw_decode_token_slots": 80,
                    "active_decode_token_slots": 70,
                },
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["configuration"]["batch_size"], 8)
        self.assertEqual(payload["summary"]["graph_calls"], 10)
        self.assertEqual(payload["summary"]["active_decode_token_slots"], 70)


if __name__ == "__main__":
    unittest.main()
