#!/usr/bin/env python3

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_small_visual_encoder import select_bucket_batch
from summarize_small_visual_encoder_matrix import load_failures, make_attention_pairs, make_backend_pairs


def fake_input(item_id: str, tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        entry={"id": item_id, "layout_label": "text", "crop_size": [10, 10]},
        image_grid_thw=torch.tensor([[1, 1, int(tokens)]], dtype=torch.long),
    )


def record(*, attention: str, backend: str, physical_rate: float, latency: float) -> dict:
    return {
        "attention": attention,
        "backend": backend,
        "preprocessor_min_pixels": 112896,
        "bucket_min_exclusive": 0,
        "fixed_physical_seq_len": 640,
        "batch_size": 1,
        "ln_impl": "module",
        "ln_linear_mode": "normal",
        "promptfa_pad_head_dim_to": 0,
        "effective_tokens_per_s": physical_rate * 0.9,
        "physical_tokens_per_s": physical_rate,
        "mean_forward_s": latency,
        "correctness_passed": True,
    }


class BucketSelectionTest(unittest.TestCase):
    def test_selects_highest_fill_inside_bucket(self) -> None:
        selected, summary = select_bucket_batch(
            [fake_input("low", 300), fake_input("high", 620), fake_input("middle", 500)],
            lower_exclusive=384,
            upper_inclusive=640,
            batch_size=2,
        )
        self.assertEqual([item.entry["id"] for item in selected], ["high", "middle"])
        self.assertEqual(summary["eligible_count"], 2)
        self.assertEqual(summary["selected_effective_tokens"], 1120)
        self.assertAlmostEqual(summary["selected_useful_token_fraction"], 1120 / 1280)

    def test_lower_edge_is_exclusive(self) -> None:
        selected, _summary = select_bucket_batch(
            [fake_input("edge", 384), fake_input("inside", 385)],
            lower_exclusive=384,
            upper_inclusive=512,
            batch_size=1,
        )
        self.assertEqual(selected[0].entry["id"], "inside")


class MatrixSummaryTest(unittest.TestCase):
    def test_compiled_speedup_pair(self) -> None:
        eager = record(attention="manual", backend="none", physical_rate=1000.0, latency=0.64)
        compiled = record(attention="manual", backend="torchair", physical_rate=2000.0, latency=0.32)
        pairs = make_backend_pairs([eager, compiled])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["physical_throughput_speedup_compiled_over_eager"], 2.0)
        self.assertEqual(pairs[0]["latency_speedup_compiled_over_eager"], 2.0)

    def test_promptfa_speedup_pair(self) -> None:
        manual = record(attention="manual", backend="none", physical_rate=1000.0, latency=0.64)
        promptfa = record(
            attention="prompt_flash_attention",
            backend="none",
            physical_rate=1600.0,
            latency=0.4,
        )
        pairs = make_attention_pairs([manual, promptfa])
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["physical_throughput_speedup_promptfa_over_manual"], 1.6)
        self.assertAlmostEqual(pairs[0]["latency_speedup_promptfa_over_manual"], 1.6)

    def test_failed_case_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "failed_cases.tsv").write_text(
                "case\texit_status\tlog\ncompiled_case\t1\tcase.log\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_failures(root),
                [{"case": "compiled_case", "exit_status": "1", "log": "case.log"}],
            )


if __name__ == "__main__":
    unittest.main()
