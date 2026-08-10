"""Unit tests for the narrow cell-boundary draft-trust experiment."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from table_spec_adaptive_k_lab import (  # noqa: E402
    cell_boundary_math_open_draft_token,
)
from paddleocr_vl.serving.table_speculative import DraftProposal  # noqa: E402


FCEL = 101309
MATH_OPEN = 47536
OTHER = 123


class CellBoundaryDraftTrustTest(unittest.TestCase):
    def decide(
        self,
        proposal: DraftProposal,
        *,
        accepted: int,
        base_next: int,
    ) -> int | None:
        return cell_boundary_math_open_draft_token(
            [FCEL],
            proposal,
            accepted_before_rejection=accepted,
            base_next_token=base_next,
            cell_token_ids={FCEL},
            math_open_token_id=MATH_OPEN,
            minimum_match=5,
        )

    def test_follows_draft_when_draft_starts_math_after_long_match(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 6)
        self.assertEqual(self.decide(proposal, accepted=0, base_next=OTHER), MATH_OPEN)

    def test_follows_non_math_draft_when_base_starts_math(self) -> None:
        proposal = DraftProposal(10, (OTHER,), 6)
        self.assertEqual(self.decide(proposal, accepted=0, base_next=MATH_OPEN), OTHER)

    def test_requires_match_strictly_greater_than_five(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 5)
        self.assertIsNone(self.decide(proposal, accepted=0, base_next=OTHER))

    def test_requires_cell_boundary(self) -> None:
        proposal = DraftProposal(10, (OTHER, MATH_OPEN), 6)
        self.assertIsNone(self.decide(proposal, accepted=1, base_next=OTHER))


if __name__ == "__main__":
    unittest.main()
