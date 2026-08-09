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

    def test_suppressed_math_open_uses_best_other_token(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 4.0]])
        selected = select_token_ids(
            logits,
            mode="suppress_math_open_greedy",
            preferred_token_id=1,
            policy_mask=torch.tensor([True]),
        )
        self.assertEqual(selected.tolist(), [2])

    def test_suppressed_math_open_keeps_ordinary_greedy_otherwise(self) -> None:
        logits = torch.tensor([[5.0, 4.0, 3.0]])
        selected = select_token_ids(
            logits,
            mode="suppress_math_open_greedy",
            preferred_token_id=1,
            policy_mask=torch.tensor([True]),
        )
        self.assertEqual(selected.tolist(), [0])

    def test_suppression_policy_mask_preserves_non_table_row(self) -> None:
        logits = torch.tensor(
            [
                [1.0, 5.0, 4.0],
                [1.0, 5.0, 4.0],
            ]
        )
        selected = select_token_ids(
            logits,
            mode="suppress_math_open_greedy",
            preferred_token_id=1,
            policy_mask=torch.tensor([True, False]),
        )
        self.assertEqual(selected.tolist(), [2, 1])

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

    def test_probability_policy_selects_close_preferred_token(self) -> None:
        logits = torch.tensor([[2.0, 1.4, -10.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_probability_near_top",
            preferred_token_id=1,
        )
        self.assertEqual(selected.tolist(), [1])

    def test_probability_policy_rejects_low_relative_probability(self) -> None:
        logits = torch.tensor([[2.0, 0.7, -10.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_probability_near_top",
            preferred_token_id=1,
        )
        self.assertEqual(selected.tolist(), [0])

    def test_probability_policy_rejects_low_global_probability(self) -> None:
        logits = torch.zeros((1, 20))
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_probability_near_top",
            preferred_token_id=19,
        )
        self.assertEqual(selected.tolist(), [0])

    def test_probability_policy_can_select_qualified_rank_three(self) -> None:
        logits = torch.tensor([[0.0, -0.2, -0.6, -10.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_probability_near_top",
            preferred_token_id=2,
        )
        self.assertEqual(selected.tolist(), [2])

    def test_variant_policy_selects_rank_two_slash_above_ten_percent(self) -> None:
        logits = torch.tensor([[2.0, 0.375, -2.5]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_variants_top2_p10",
            preferred_token_id=2,
            alternate_preferred_token_id=1,
        )
        self.assertEqual(selected.tolist(), [1])

    def test_variant_policy_rejects_rank_two_slash_below_ten_percent(self) -> None:
        logits = torch.tensor([[3.0, 0.7, -2.5]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_variants_top2_p10",
            preferred_token_id=2,
            alternate_preferred_token_id=1,
        )
        self.assertEqual(selected.tolist(), [0])

    def test_combined_policy_keeps_legacy_qualified_rank_three_math_open(self) -> None:
        logits = torch.tensor([[2.0, 1.8, 1.6, -10.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_adjusters_combined",
            preferred_token_id=2,
            alternate_preferred_token_id=3,
        )
        self.assertEqual(selected.tolist(), [2])

    def test_combined_policy_applies_legacy_outside_variant_mask(self) -> None:
        logits = torch.tensor([[2.0, 1.8, 1.6, -10.0]])
        selected = select_token_ids(
            logits,
            mode="prefer_math_open_adjusters_combined",
            preferred_token_id=2,
            alternate_preferred_token_id=3,
            policy_mask=torch.tensor([False]),
            legacy_policy_mask=torch.tensor([True]),
        )
        self.assertEqual(selected.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
