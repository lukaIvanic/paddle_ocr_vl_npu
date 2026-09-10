"""Small metric helpers shared across runtime layers."""

from __future__ import annotations


def per_second(count: int | float, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return float(count) / float(seconds)
