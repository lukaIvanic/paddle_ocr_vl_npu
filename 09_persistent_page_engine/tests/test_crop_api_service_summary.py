"""Tests for durable crop API service summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


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
