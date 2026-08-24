"""Process-shared byte credits for bounded page payloads."""

from __future__ import annotations

import time
from typing import Any


class SharedByteBudget:
    """Bound variable-sized allocations across spawned producer processes."""

    def __init__(self, context: Any, limit_bytes: int) -> None:
        if limit_bytes < 1:
            raise ValueError("shared byte budget must be positive")
        self.limit_bytes = int(limit_bytes)
        self.condition = context.Condition(context.RLock())
        self.used_bytes = context.Value("q", 0, lock=False)
        self.peak_bytes = context.Value("q", 0, lock=False)
        self.reservation_count = context.Value("q", 0, lock=False)
        self.release_count = context.Value("q", 0, lock=False)
        self.wait_count = context.Value("q", 0, lock=False)
        self.wait_ns = context.Value("q", 0, lock=False)

    def reserve(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("cannot reserve a negative byte count")
        if nbytes == 0:
            return
        if nbytes > self.limit_bytes:
            raise RuntimeError(
                "one shared page payload exceeds the complete byte budget: "
                f"payload={nbytes} budget={self.limit_bytes}"
            )
        started = time.perf_counter_ns()
        waited = False
        with self.condition:
            while int(self.used_bytes.value) + nbytes > self.limit_bytes:
                waited = True
                self.condition.wait()
            elapsed_ns = time.perf_counter_ns() - started
            self.used_bytes.value = int(self.used_bytes.value) + nbytes
            self.peak_bytes.value = max(
                int(self.peak_bytes.value),
                int(self.used_bytes.value),
            )
            self.reservation_count.value = int(self.reservation_count.value) + 1
            if waited:
                self.wait_count.value = int(self.wait_count.value) + 1
                self.wait_ns.value = int(self.wait_ns.value) + elapsed_ns

    def release(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("cannot release a negative byte count")
        if nbytes == 0:
            return
        with self.condition:
            used = int(self.used_bytes.value)
            if nbytes > used:
                raise RuntimeError(
                    "shared byte budget release exceeds live bytes: "
                    f"release={nbytes} live={used}"
                )
            self.used_bytes.value = used - nbytes
            self.release_count.value = int(self.release_count.value) + 1
            self.condition.notify_all()

    def snapshot(self) -> dict[str, int | float]:
        with self.condition:
            return {
                "limit_bytes": self.limit_bytes,
                "live_bytes": int(self.used_bytes.value),
                "peak_bytes": int(self.peak_bytes.value),
                "reservation_count": int(self.reservation_count.value),
                "release_count": int(self.release_count.value),
                "wait_count": int(self.wait_count.value),
                "wait_s": int(self.wait_ns.value) / 1_000_000_000.0,
            }
