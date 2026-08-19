"""Pure scheduling policy for two-shape UniRec continuous decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeLaneSpec:
    name: str
    batch_size: int
    self_cache_length: int
    cross_cache_length: int
    max_length: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("decode lane name must not be empty")
        for field_name in (
            "batch_size",
            "self_cache_length",
            "cross_cache_length",
            "max_length",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.max_length > self.self_cache_length:
            raise ValueError("max_length cannot exceed self_cache_length")


@dataclass(frozen=True)
class DecodeLaneStatus:
    name: str
    capacity: int
    active_slots: int
    queued_items: int
    mean_step_ms: float | None
    skipped_quanta: int = 0

    @property
    def runnable(self) -> bool:
        return self.active_slots > 0 or self.queued_items > 0

    @property
    def full(self) -> bool:
        return self.active_slots == self.capacity

    @property
    def useful_tokens_per_ms(self) -> float:
        if self.active_slots <= 0:
            return 0.0
        if self.mean_step_ms is None or self.mean_step_ms <= 0:
            return float(self.active_slots) / float(self.capacity)
        return float(self.active_slots) / self.mean_step_ms


def route_lane(*, cross_length: int, a_cross_capacity: int) -> str:
    if cross_length < 1:
        raise ValueError("cross_length must be positive")
    if a_cross_capacity < 1:
        raise ValueError("a_cross_capacity must be positive")
    return "a" if cross_length <= a_cross_capacity else "b"


def full_lane_schedule(*, a_quanta: int, b_quanta: int) -> tuple[str, ...]:
    """Return the deterministic weighted schedule used while both lanes are full."""
    if a_quanta < 1 or b_quanta < 1:
        raise ValueError("full-lane quantum weights must be positive")
    return ("a",) * int(a_quanta) + ("b",) * int(b_quanta)


def choose_lane(
    a: DecodeLaneStatus,
    b: DecodeLaneStatus,
    *,
    round_robin_next: str,
    max_skipped_quanta: int,
) -> str:
    """Choose the next bounded quantum without page-age preemption.

    Full lanes use strict round-robin. Partial lanes are selected by measured
    useful tokens per millisecond, with only a quantum-count starvation bound.
    """
    if round_robin_next not in ("a", "b"):
        raise ValueError("round_robin_next must be 'a' or 'b'")
    if max_skipped_quanta < 1:
        raise ValueError("max_skipped_quanta must be positive")
    if not a.runnable and not b.runnable:
        raise ValueError("no runnable decode lane")
    if not a.runnable:
        return "b"
    if not b.runnable:
        return "a"
    if a.full and b.full:
        return round_robin_next
    if a.skipped_quanta >= max_skipped_quanta:
        return "a"
    if b.skipped_quanta >= max_skipped_quanta:
        return "b"
    a_rate = a.useful_tokens_per_ms
    b_rate = b.useful_tokens_per_ms
    if a_rate == b_rate:
        return round_robin_next
    return "a" if a_rate > b_rate else "b"
