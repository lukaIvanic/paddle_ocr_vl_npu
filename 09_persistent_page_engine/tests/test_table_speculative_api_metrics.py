"""Tests for speculative API metric aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/serve_table_speculative_api.py"
)
SPEC = importlib.util.spec_from_file_location(
    "serve_table_speculative_api", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TableSpeculativeApiMetricsTest(unittest.TestCase):
    def test_summary_merges_draft_slots_and_verifier_shapes(self) -> None:
        schedule = {
            "batch_size": 8,
            "requests": 8,
            "graph_calls": 100,
            "raw_decode_token_slots": 800,
            "active_decode_token_slots": 600,
            "effective_decode_tokens": 590,
            "idle_decode_token_slots": 200,
            "lookahead_decode_token_slots": 10,
            "timing_s": {
                "continuous_decode_wall": 0.2,
                "decode_model_and_argmax_device": 0.19,
                "run_scoped_scheduler_wall": 0.25,
            },
        }
        metrics = {
            "draft": {"rows": [], "schedule": schedule},
            "target_prefill": {},
            "verifier": {
                "target_calls": 3,
                "speculative_calls": 2,
                "fully_accepted_speculative_calls": 1,
                "rejected_speculative_calls": 1,
                "fallback_calls": 1,
                "proposed_draft_tokens": 22,
                "accepted_draft_tokens": 18,
                "output_tokens_after_prefill": 21,
                "verifier_device_s": 0.0,
                "fallback_device_s": 0.0,
                "wall_s": 0.01,
                "per_k": {
                    "7": {"calls": 1},
                    "15": {"calls": 1},
                },
            },
        }
        summary = MODULE._summarize_spec_service([metrics], [])
        self.assertEqual(
            summary["draft_decode"]["rates"]["active_slot_fraction"],
            0.75,
        )
        self.assertEqual(summary["verifier"]["physical_verifier_tokens"], 25)
        self.assertEqual(
            summary["verifier"]["physical_verifier_tok_per_verifier_wall_s"],
            2500.0,
        )
        self.assertEqual(
            summary["verifier"]["accepted_fraction_of_proposed"],
            18 / 22,
        )


if __name__ == "__main__":
    unittest.main()
