"""Unit tests for the narrow cell-boundary draft-trust experiment."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from table_spec_adaptive_k_lab import (  # noqa: E402
    cell_boundary_math_open_draft_token,
    in_cell_draft_script_open_token,
)
from paddleocr_vl.serving.table_speculative import (  # noqa: E402
    DraftPosition,
    DraftProposal,
)


FCEL = 101309
MATH_OPEN = 47536
OTHER = 123
SLASH = 93980
SCRIPT_OPEN = 1305
NEWLINE = 101313


class CellBoundaryDraftTrustTest(unittest.TestCase):
    @staticmethod
    def matcher(*, column: int = 0, row_width: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            cell_tokens={FCEL},
            newline_token=NEWLINE,
            metadata=[DraftPosition(0, 0, column, row_width)] * 32,
        )

    def decide(
        self,
        proposal: DraftProposal,
        *,
        accepted: int,
        base_next: int,
        trust_slash: bool = False,
    ) -> int | None:
        return cell_boundary_math_open_draft_token(
            [FCEL],
            proposal,
            accepted_before_rejection=accepted,
            base_next_token=base_next,
            matcher=self.matcher(),
            math_open_token_id=MATH_OPEN,
            additional_trigger_token_ids={SLASH} if trust_slash else set(),
        )

    def test_follows_draft_when_draft_starts_math_in_aligned_cell(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 6)
        self.assertEqual(self.decide(proposal, accepted=0, base_next=OTHER), MATH_OPEN)

    def test_follows_non_math_draft_when_base_starts_math(self) -> None:
        proposal = DraftProposal(10, (OTHER,), 6)
        self.assertEqual(self.decide(proposal, accepted=0, base_next=MATH_OPEN), OTHER)

    def test_does_not_require_long_exact_suffix(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 1)
        self.assertEqual(self.decide(proposal, accepted=0, base_next=OTHER), MATH_OPEN)

    def test_requires_matching_column(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 1)
        selected = cell_boundary_math_open_draft_token(
            [FCEL],
            proposal,
            accepted_before_rejection=0,
            base_next_token=OTHER,
            matcher=self.matcher(column=1),
            math_open_token_id=MATH_OPEN,
        )
        self.assertIsNone(selected)

    def test_requires_compatible_completed_row_width(self) -> None:
        proposal = DraftProposal(10, (MATH_OPEN,), 1)
        selected = cell_boundary_math_open_draft_token(
            [FCEL, OTHER, NEWLINE, FCEL],
            proposal,
            accepted_before_rejection=0,
            base_next_token=OTHER,
            matcher=self.matcher(row_width=2),
            math_open_token_id=MATH_OPEN,
        )
        self.assertIsNone(selected)

    def test_requires_cell_boundary(self) -> None:
        proposal = DraftProposal(10, (OTHER, MATH_OPEN), 6)
        self.assertIsNone(self.decide(proposal, accepted=1, base_next=OTHER))

    def test_follows_draft_when_draft_starts_standalone_slash(self) -> None:
        proposal = DraftProposal(10, (SLASH,), 6)
        self.assertEqual(
            self.decide(
                proposal,
                accepted=0,
                base_next=OTHER,
                trust_slash=True,
            ),
            SLASH,
        )

    def test_slash_rule_is_independent(self) -> None:
        proposal = DraftProposal(10, (SLASH,), 6)
        self.assertIsNone(self.decide(proposal, accepted=0, base_next=OTHER))

    def test_follows_draft_script_open_inside_cell(self) -> None:
        proposal = DraftProposal(10, (OTHER, SCRIPT_OPEN), 6)
        selected = in_cell_draft_script_open_token(
            [FCEL],
            proposal,
            accepted_before_rejection=1,
            matcher=self.matcher(),
            script_open_token_id=SCRIPT_OPEN,
        )
        self.assertEqual(selected, SCRIPT_OPEN)

    def test_script_open_requires_content_inside_cell(self) -> None:
        proposal = DraftProposal(10, (SCRIPT_OPEN,), 6)
        selected = in_cell_draft_script_open_token(
            [FCEL],
            proposal,
            accepted_before_rejection=0,
            matcher=self.matcher(),
            script_open_token_id=SCRIPT_OPEN,
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
