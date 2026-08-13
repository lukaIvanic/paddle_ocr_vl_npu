from __future__ import annotations

import random
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.repetition import ExactCycleTracker, RepetitionEvidence


class ChronologicalExactCycleTracker:
    """Reference form used before the newest-first history optimization."""

    def __init__(
        self,
        *,
        min_repeat_copies: int = 6,
        min_repeated_span: int = 128,
        max_period: int = 32,
    ) -> None:
        self.min_repeat_copies = min_repeat_copies
        self.min_repeated_span = min_repeated_span
        self.max_period = max_period
        self.tail: list[int] = []
        self.matching_runs = [0] * (max_period + 1)
        self.length = 0

    def update(self, token_id: int) -> RepetitionEvidence | None:
        token_id = int(token_id)
        best: RepetitionEvidence | None = None
        available_periods = min(self.max_period, len(self.tail))
        for period in range(1, available_periods + 1):
            if token_id == self.tail[-period]:
                self.matching_runs[period] += 1
            else:
                self.matching_runs[period] = 0
            repeated_span = self.matching_runs[period] + period
            if (
                repeated_span < self.min_repeated_span
                or repeated_span < period * self.min_repeat_copies
            ):
                continue
            evidence = RepetitionEvidence(
                rule=(
                    f"exact_cycle_{self.min_repeat_copies}copies_"
                    f"{self.min_repeated_span}tokens_p{self.max_period}"
                ),
                trigger_length=self.length + 1,
                trim_length=self.length + 1 - repeated_span + period,
                period=period,
                repeated_positions=repeated_span - period,
                repeated_span=repeated_span,
                repeat_copies=repeated_span // period,
            )
            if best is None or (
                int(evidence.repeated_span or 0),
                -int(evidence.period or 0),
            ) > (
                int(best.repeated_span or 0),
                -int(best.period or 0),
            ):
                best = evidence
        self.tail.append(token_id)
        if len(self.tail) > self.max_period:
            del self.tail[0]
        self.length += 1
        return best


def evidence_dict(value: RepetitionEvidence | None) -> dict | None:
    return value.to_dict() if value is not None else None


def assert_same_evidence(sequence: list[int], **kwargs: int) -> None:
    reference = ChronologicalExactCycleTracker(**kwargs)
    optimized = ExactCycleTracker(**kwargs)
    for token in sequence:
        assert evidence_dict(optimized.update(token)) == evidence_dict(
            reference.update(token)
        )


def test_newest_first_tracker_matches_reference() -> None:
    generator = random.Random(20260813)
    assert_same_evidence([generator.randrange(256) for _ in range(2048)])
    assert_same_evidence([11, 12, 13, 14] * 80)
    assert_same_evidence(
        [generator.randrange(16) for _ in range(80)] + [5, 7, 9] * 60,
        min_repeat_copies=4,
        min_repeated_span=60,
        max_period=16,
    )
