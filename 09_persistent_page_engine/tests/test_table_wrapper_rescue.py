"""Unit tests for the one-token table wrapper rescue guard."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.table_speculative import (
    DraftProposal,
    TableDraftMatcher,
    wrapper_rescue_candidate,
)


FCEL = 101309
ECEL = 101308
LCEL = 101311
UCEL = 101312
XCEL = 101310
NL = 101313
EOS = 2
LOG = 100
SIZE = 101
OTHER = 102
SLASH = 93980
PAREN_MINUS = 1070
MATH_OPEN = 47536
MATH_CLOSE = 61124
ZERO = 3


class FakeTokenizer:
    special = {
        "<fcel>": FCEL,
        "<ecel>": ECEL,
        "<lcel>": LCEL,
        "<ucel>": UCEL,
        "<xcel>": XCEL,
        "<nl>": NL,
    }
    decoded = {
        FCEL: "<fcel>",
        ECEL: "<ecel>",
        LCEL: "<lcel>",
        UCEL: "<ucel>",
        XCEL: "<xcel>",
        NL: "<nl>",
        LOG: "Log",
        SIZE: " size",
        OTHER: "Other",
        SLASH: "\\",
        PAREN_MINUS: "(-",
        MATH_OPEN: "\\(",
        MATH_CLOSE: "\\)",
        ZERO: "0",
        EOS: "<eos>",
    }

    def token_to_id(self, token: str) -> int:
        return self.special[token]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "".join(self.decoded[int(token)] for token in token_ids)


def matcher_for(tokenizer: FakeTokenizer) -> TableDraftMatcher:
    return TableDraftMatcher(
        {
            "rows": [
                {
                    "row_index": 0,
                    "token_ids": [
                        FCEL,
                        LOG,
                        SIZE,
                        FCEL,
                        SLASH,
                        PAREN_MINUS,
                        ZERO,
                        NL,
                        EOS,
                    ],
                }
            ]
        },
        tokenizer,
        eos_token_id=EOS,
        block_size=16,
    )


class WrapperRescueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.matcher = matcher_for(self.tokenizer)

    def test_exact_previous_cell_accepts_actual_two_token_wrapper_path(self) -> None:
        proposal = DraftProposal(1, tuple(self.matcher.draft[1:]), 1)
        candidate = wrapper_rescue_candidate(
            [FCEL],
            self.matcher,
            proposal,
            accepted_before_rejection=3,
            tokenizer=self.tokenizer,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.draft_token, SLASH)
        self.assertEqual(candidate.draft_prefix_text, r"\(-")
        self.assertEqual(candidate.previous_cell_match, "exact_ids")
        self.assertEqual(candidate.target_column, 1)
        self.assertEqual(candidate.draft_column, 1)

    def test_outer_math_wrapper_only_previous_cell_is_allowed(self) -> None:
        proposal = DraftProposal(4, tuple(self.matcher.draft[4:]), 1)
        candidate = wrapper_rescue_candidate(
            [FCEL, MATH_OPEN, LOG, SIZE, MATH_CLOSE, FCEL],
            self.matcher,
            proposal,
            accepted_before_rejection=0,
            tokenizer=self.tokenizer,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.previous_cell_match, "outer_math_wrapper_only")

    def test_content_difference_is_rejected(self) -> None:
        proposal = DraftProposal(4, tuple(self.matcher.draft[4:]), 1)
        candidate = wrapper_rescue_candidate(
            [FCEL, OTHER, FCEL],
            self.matcher,
            proposal,
            accepted_before_rejection=0,
            tokenizer=self.tokenizer,
        )
        self.assertIsNone(candidate)

    def test_first_column_is_rejected(self) -> None:
        matcher = TableDraftMatcher(
            {"rows": [{"row_index": 0, "token_ids": [FCEL, SLASH, PAREN_MINUS, EOS]}]},
            self.tokenizer,
            eos_token_id=EOS,
            block_size=16,
        )
        proposal = DraftProposal(1, tuple(matcher.draft[1:]), 1)
        candidate = wrapper_rescue_candidate(
            [FCEL],
            matcher,
            proposal,
            accepted_before_rejection=0,
            tokenizer=self.tokenizer,
        )
        self.assertIsNone(candidate)

    def test_formula_previous_mode_accepts_wrapper_only_content_match(self) -> None:
        matcher = TableDraftMatcher(
            {
                "rows": [
                    {
                        "row_index": 0,
                        "token_ids": [
                            FCEL,
                            MATH_OPEN,
                            LOG,
                            SIZE,
                            MATH_CLOSE,
                            FCEL,
                            SLASH,
                            PAREN_MINUS,
                            ZERO,
                            EOS,
                        ],
                    }
                ]
            },
            self.tokenizer,
            eos_token_id=EOS,
            block_size=16,
        )
        proposal = DraftProposal(6, tuple(matcher.draft[6:]), 1)
        candidate = wrapper_rescue_candidate(
            [FCEL, LOG, SIZE, FCEL],
            matcher,
            proposal,
            accepted_before_rejection=0,
            tokenizer=self.tokenizer,
            formula_previous_only=True,
        )
        self.assertIsNotNone(candidate)

        candidate = wrapper_rescue_candidate(
            [FCEL, MATH_OPEN, LOG, SIZE, MATH_CLOSE, FCEL],
            matcher,
            proposal,
            accepted_before_rejection=0,
            tokenizer=self.tokenizer,
            formula_previous_only=True,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.previous_cell_match, "formula_content")


if __name__ == "__main__":
    unittest.main()
