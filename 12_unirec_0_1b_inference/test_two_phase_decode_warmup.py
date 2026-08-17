#!/usr/bin/env python3
"""CPU-only regression tests for two-phase decode warmup setup."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from run_two_phase_batched_unirec import prepare_decode_warmup_report


class FakeBase:
    def __init__(self) -> None:
        self.calls = []

    def warmup_configured_graphs(self, **kwargs):
        self.calls.append(kwargs)
        return {"passes": kwargs["passes"], "graphs": {}, "wall_s": 1.0}


class TwoPhaseDecodeWarmupTest(unittest.TestCase):
    def test_zero_passes_skips_generic_positive_only_warmup(self) -> None:
        base = FakeBase()
        report = prepare_decode_warmup_report(
            base=base,
            runner=object(),
            batch_size=128,
            passes=0,
        )
        self.assertEqual(base.calls, [])
        self.assertEqual(report["passes"], 0)
        self.assertEqual(report["wall_s"], 0.0)
        self.assertEqual(
            report["decode"],
            {"execution": "disabled_live_arena_warmup", "passes": 0},
        )

    def test_positive_passes_retain_deferred_warmup_contract(self) -> None:
        base = FakeBase()
        runner = object()
        report = prepare_decode_warmup_report(
            base=base,
            runner=runner,
            batch_size=64,
            passes=2,
        )
        self.assertEqual(len(base.calls), 1)
        call = base.calls[0]
        self.assertIs(call["runner"], runner)
        self.assertEqual(call["passes"], 2)
        self.assertFalse(call["warmup_decode"])
        self.assertIsNone(call["vision_atlas_runtime"])
        self.assertEqual(call["args"].decode_batch_size, 64)
        self.assertEqual(
            report["decode"],
            {"execution": "deferred_to_actual_admitted_arena", "passes": 2},
        )

    def test_negative_passes_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prepare_decode_warmup_report(
                base=SimpleNamespace(),
                runner=object(),
                batch_size=128,
                passes=-1,
            )


if __name__ == "__main__":
    unittest.main()
