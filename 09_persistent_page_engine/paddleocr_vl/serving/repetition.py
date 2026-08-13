"""Token-only repetition evidence for OCR decode control.

The rules in this module deliberately know nothing about request IDs, crop
labels, languages, decoded text, or reference generations.  They operate only
on one request's generated token history so the same implementation can be
used by frozen-trace audits and, after validation, the live scheduler.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RepetitionEvidence:
    rule: str
    trigger_length: int
    trim_length: int
    window_size: int | None = None
    period: int | None = None
    repeated_positions: int | None = None
    repeated_span: int | None = None
    repeat_copies: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


@dataclass
class ExactCycleTracker:
    """Incrementally detect an exactly periodic generated-token tail.

    The tracker retains only ``max_period`` prior token IDs and one match-run
    counter per candidate period.  Updating it is O(max_period) per generated
    token; it never rescans the request history on the decode hot path.
    """

    min_repeat_copies: int = 6
    min_repeated_span: int = 128
    max_period: int = 32
    _tail: list[int] = field(default_factory=list, init=False, repr=False)
    _matching_runs: list[int] = field(default_factory=list, init=False, repr=False)
    _length: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_repeat_copies < 2:
            raise ValueError("min_repeat_copies must be at least 2")
        if self.min_repeated_span <= 0:
            raise ValueError("min_repeated_span must be positive")
        if self.max_period <= 0:
            raise ValueError("max_period must be positive")
        self._matching_runs = [0] * (self.max_period + 1)

    def update(self, token_id: int) -> RepetitionEvidence | None:
        """Observe one newly generated token and return the first stop proof."""

        token_id = int(token_id)
        best: RepetitionEvidence | None = None
        tail = self._tail
        matching_runs = self._matching_runs
        min_repeated_span = self.min_repeated_span
        min_repeat_copies = self.min_repeat_copies
        for period, previous_token in enumerate(tail, start=1):
            if token_id == previous_token:
                matching_run = matching_runs[period] + 1
                matching_runs[period] = matching_run
            else:
                matching_run = 0
                matching_runs[period] = 0
            repeated_span = matching_run + period
            if (
                repeated_span < min_repeated_span
                or repeated_span < period * min_repeat_copies
            ):
                continue
            evidence = RepetitionEvidence(
                rule=(
                    f"exact_cycle_{min_repeat_copies}copies_"
                    f"{min_repeated_span}tokens_p{self.max_period}"
                ),
                trigger_length=self._length + 1,
                trim_length=self._length + 1 - repeated_span + period,
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

        tail.insert(0, token_id)
        if len(tail) > self.max_period:
            tail.pop()
        self._length += 1
        return best


def _validate_window(window_size: int, repeated_threshold: int) -> None:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 < repeated_threshold <= window_size:
        raise ValueError("repeated_threshold must be in [1, window_size]")


def dominant_token_window(
    tokens: Sequence[int],
    *,
    window_size: int = 30,
    repeated_threshold: int = 20,
) -> RepetitionEvidence | None:
    """Stop when one token occupies at least N positions of the last window."""

    _validate_window(window_size, repeated_threshold)
    for end in range(window_size, len(tokens) + 1):
        window = tokens[end - window_size : end]
        count = Counter(window).most_common(1)[0][1]
        if count >= repeated_threshold:
            return RepetitionEvidence(
                rule=f"dominant_token_{repeated_threshold}_of_{window_size}",
                trigger_length=end,
                trim_length=end,
                window_size=window_size,
                repeated_positions=count,
            )
    return None


def duplicate_excess_window(
    tokens: Sequence[int],
    *,
    window_size: int = 30,
    repeated_threshold: int = 20,
) -> RepetitionEvidence | None:
    """Stop when N window positions repeat a token already present in it.

    ``len(window) - len(set(window))`` counts occurrences beyond the first
    instance of each distinct token.  This is the broadest literal reading of
    "20 of the last 30 are repeated" and is kept separate because it may be
    much less selective than a repeated cycle.
    """

    _validate_window(window_size, repeated_threshold)
    for end in range(window_size, len(tokens) + 1):
        window = tokens[end - window_size : end]
        repeated = len(window) - len(set(window))
        if repeated >= repeated_threshold:
            return RepetitionEvidence(
                rule=f"duplicate_excess_{repeated_threshold}_of_{window_size}",
                trigger_length=end,
                trim_length=end,
                window_size=window_size,
                repeated_positions=repeated,
            )
    return None


def periodic_matches_window(
    tokens: Sequence[int],
    *,
    window_size: int = 30,
    repeated_threshold: int = 20,
) -> RepetitionEvidence | None:
    """Stop when a short period explains N positions in the last window.

    For each possible period that can still supply ``repeated_threshold``
    comparisons, the rule counts positions equal to the token one period
    earlier.  It permits occasional mutations and is therefore an audit rule
    until full-corpus false-positive review says otherwise.
    """

    _validate_window(window_size, repeated_threshold)
    max_period = window_size - repeated_threshold
    if max_period <= 0:
        raise ValueError("periodic matching needs threshold < window size")
    for end in range(window_size, len(tokens) + 1):
        window = tokens[end - window_size : end]
        best_period = None
        best_matches = -1
        for period in range(1, max_period + 1):
            matches = sum(
                window[index] == window[index - period]
                for index in range(period, window_size)
            )
            if matches > best_matches:
                best_matches = matches
                best_period = period
        if best_matches >= repeated_threshold:
            return RepetitionEvidence(
                rule=f"periodic_matches_{repeated_threshold}_of_{window_size}",
                trigger_length=end,
                trim_length=end,
                window_size=window_size,
                period=best_period,
                repeated_positions=best_matches,
            )
    return None


def exact_repeating_suffix(
    tokens: Sequence[int],
    *,
    min_repeat_copies: int = 4,
    min_repeated_span: int = 60,
    max_period: int = 32,
) -> RepetitionEvidence | None:
    """Find the earliest long, exactly periodic generated suffix.

    Once a trigger is found, ``trim_length`` preserves the prefix and one copy
    of the primitive observed cycle.  The rule does not inspect what any token
    means.
    """

    tracker = ExactCycleTracker(
        min_repeat_copies=min_repeat_copies,
        min_repeated_span=min_repeated_span,
        max_period=max_period,
    )
    for token_id in tokens:
        evidence = tracker.update(int(token_id))
        if evidence is not None:
            return evidence
    return None


def run_self_checks() -> None:
    natural = list(range(100))
    assert dominant_token_window(natural) is None
    assert duplicate_excess_window(natural) is None
    assert periodic_matches_window(natural) is None
    assert exact_repeating_suffix(natural) is None

    dominant = list(range(10)) + [7] * 20
    assert dominant_token_window(dominant) is not None

    duplicates = list(range(10)) * 3
    assert duplicate_excess_window(duplicates) is not None
    assert periodic_matches_window(duplicates) is not None

    exact = [101, 102, 103] + [4, 5, 6, 7, 8] * 12
    hit = exact_repeating_suffix(exact)
    assert hit is not None
    assert hit.period == 5
    assert exact[: hit.trim_length] == [101, 102, 103, 4, 5, 6, 7, 8]

    persistent = ExactCycleTracker(min_repeated_span=128)
    assert all(persistent.update(value) is None for value in range(127))
    # A fresh tracker catches a persistent cycle at exactly the configured
    # evidence horizon and trims it back to one observed cycle.
    persistent = ExactCycleTracker(min_repeated_span=128)
    persistent_hit = None
    sequence = [101, 102, 103] + [4, 5, 6, 7] * 32
    for value in sequence:
        persistent_hit = persistent.update(value) or persistent_hit
    assert persistent_hit is not None
    assert persistent_hit.trigger_length == len(sequence)
    assert persistent_hit.trim_length == 7
