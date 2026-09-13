"""Stop a decode that has fallen into an exactly repeating token cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
