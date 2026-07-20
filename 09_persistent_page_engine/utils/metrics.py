"""Small metric helpers shared across runtime layers."""

from __future__ import annotations


def per_second(count: int | float, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return float(count) / float(seconds)
