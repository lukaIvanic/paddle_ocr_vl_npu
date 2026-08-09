"""Unit tests for direct-logit token selection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.token_selection import select_token_ids


class TokenSelectionTest(unittest.TestCase):
    def test_greedy_is_unchanged(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 4.0]])
        selected = select_token_ids(logits, mode="greedy")
        self.assertEqual(selected.tolist(), [1])

    def test_preferred_rank_one_is_selected(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 4.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_top2_first_override",
            preferred_token_id=1,
        )
        self.assertEqual(selected.tolist(), [1])

    def test_preferred_rank_two_replaces_greedy(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 4.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_top2_first_override",
            preferred_token_id=2,
        )
        self.assertEqual(selected.tolist(), [2])

    def test_preferred_rank_three_does_not_replace_greedy(self) -> None:
        logits = torch.tensor([[3.0, 5.0, 4.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_top2_first_override",
            preferred_token_id=0,
        )
        self.assertEqual(selected.tolist(), [1])

    def test_policy_mask_preserves_non_table_rows(self) -> None:
        logits = torch.tensor(
            [
                [1.0, 5.0, 4.0],
                [1.0, 5.0, 4.0],
            ]
        )
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_top2_first_override",
            preferred_token_id=2,
            policy_mask=torch.tensor([True, False]),
        )
        self.assertEqual(selected.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
