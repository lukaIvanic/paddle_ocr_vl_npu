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
from types import SimpleNamespace
from unittest.mock import Mock


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
    def test_setup_gc_freeze_is_default_with_explicit_opt_out(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            self.assertTrue(MODULE.parse_args().freeze_setup_gc)
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py", "--no-freeze-setup-gc"]):
            self.assertFalse(MODULE.parse_args().freeze_setup_gc)

    def test_defaults_match_validated_b2_inference_contract(self) -> None:
        root = SCRIPT.parents[2]
        saved = json.loads((root / MODULE.TABLE_CONCURRENCY_ANCHORS[2][2]).read_text())["api_configuration"]
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            args = MODULE.parse_args()
        for arg, field in (("decode_backend", "decode_backend"),
                           ("decode_optimization", "decode_optimization"),
                           ("decode_batch_size", "batch_size"),
                           ("cache_length", "cache_length"), ("max_new_tokens", "max_new_tokens"),
                           ("compact_decode_control", "compact_decode_control"),
                           ("vision_attention_weight_padding", "vision_attention_weight_padding"),
                           ("vision_linear_patch_projection", "vision_linear_patch_projection"),
                           ("request_scheduling_metrics", "request_scheduling_metrics"),
                           ("max_prefill_interruptions", "max_prefill_interruptions")):
            self.assertEqual(getattr(args, arg), saved[field], arg)
        self.assertEqual(str(args.model), saved["recognizer_model"])
        self.assertEqual(args.dtype, "fp16")
        self.assertEqual(args.token_selection, saved["token_selection"]["mode"])
        self.assertEqual(not args.no_decode_device_timing, saved["decode_device_timing"])
        self.assertEqual(args.freeze_setup_gc, saved["setup_gc"]["enabled"])
        for stage in ("vision", "text"):
            self.assertEqual(list(map(int, getattr(args, stage + "_buckets").split(","))),
                             saved[stage + "_prefill"]["buckets"])
        for bound in ("min", "max"):
            self.assertEqual(getattr(args, bound + "_pixels"), saved["preprocessor"]["effective_" + bound + "_pixels"])
        vocabulary = json.loads(args.decode_vocab_token_ids.read_text())
        self.assertEqual(vocabulary["token_ids_sha256"], saved["decode_vocab"]["token_ids_sha256"])
        self.assertEqual(vocabulary["token_ids_sha256"], MODULE.TABLE_VOCAB_SHA256)
        self.assertEqual(vocabulary["selected_vocab_size"], 16384)

    def test_explicit_controls_can_restore_full_vocab_and_instrumentation(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py", "--full-vocab",
                "--decode-device-timing", "--no-freeze-setup-gc",
                "--no-vision-attention-weight-padding", "--no-vision-linear-patch-projection",
                "--no-request-scheduling-metrics", "--decode-batch-size", "1"]):
            args = MODULE.parse_args()
        self.assertIsNone(args.decode_vocab_token_ids)
        for field in ("no_decode_device_timing", "freeze_setup_gc", "vision_attention_weight_padding",
                      "vision_linear_patch_projection", "request_scheduling_metrics"):
            self.assertFalse(getattr(args, field), field)
        self.assertEqual(args.decode_batch_size, 1)

    def test_concurrency_hints_point_to_real_1000_request_anchors(self) -> None:
        root = SCRIPT.parents[2]
        self.assertEqual(set(MODULE.TABLE_CONCURRENCY_ANCHORS), {1, 2, 3, 4, 5})
        for concurrency, (batch, status, relative) in MODULE.TABLE_CONCURRENCY_ANCHORS.items():
            summary = json.loads((root / relative).read_text())
            self.assertEqual(summary["request_count"], 1000)
            self.assertEqual(summary["api_configuration"]["batch_size"], batch)
            self.assertEqual(summary["max_in_flight"], concurrency)
            self.assertEqual(status, "current" if concurrency in (2, 5) else "historical")
        self.assertIn("PRE-DATE", MODULE.TABLE_SERVING_GUIDANCE)

    def test_table_vocabulary_cannot_silently_restrict_other_tasks(self) -> None:
        handler = object.__new__(MODULE._Handler)
        handler.path = "/v1/ocr?crop_type=text"
        handler.server = SimpleNamespace(state=SimpleNamespace(configuration={
            "decode_vocab": {"token_ids_sha256": MODULE.TABLE_VOCAB_SHA256}}))
        handler._json = Mock()
        handler.do_POST()
        status, body = handler._json.call_args.args
        self.assertEqual(status, 400)
        self.assertIn("--full-vocab", body["error"])
        handler.server.state.configuration = {"decode_vocab": {"enabled": False}}
        handler.headers = {"Content-Length": "0"}
        handler.do_POST()
        self.assertEqual(handler._json.call_args.args[0], 413)  # Passed vocabulary check; empty image rejected.

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

    def test_scheduling_metrics_default_matches_benchmark(self) -> None:
        with patch.object(sys, "argv", ["serve_crop_ocr_api.py"]):
            self.assertTrue(MODULE.parse_args().request_scheduling_metrics)
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
